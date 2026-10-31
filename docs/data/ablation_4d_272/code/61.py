import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _smooth_gd(x, steps, a0, a1, s0, s1, mu=0.85, deadline=None):
    x = _norm_rows(x.copy())
    for it in range(steps):
        if deadline is not None and (it % 60 == 0) and time.time() > deadline:
            break
        frac = it / max(steps - 1, 1)
        # geometric interpolation
        alpha = a0 * (a1 / a0) ** frac if a0 != a1 else float(a0)
        smax = s0 * (s1 / s0) ** frac if s0 != s1 else float(s0)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(np.clip(alpha * (G - m), -50.0, 0.0))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum())
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (W @ x) / (0.5 * S + 1e-300)
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        # momentum
        if it == 0:
            vel = gt
        else:
            vel = mu * vel + gt
        nrm = float(np.linalg.norm(vel, axis=1).max())
        if not np.isfinite(nrm) or nrm < 1e-14:
            continue
        x = x - (smax / max(nrm, 1e-300)) * vel
        x = _norm_rows(x)
    return x

def _lp_polish(x, deadline, maxpairs=5000):
    x = _norm_rows(x.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x, 2.0
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline - 0.05 and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > maxpairs:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], maxpairs - 1)[:maxpairs]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[order]] = True
            sel = keep
        c = c_all[sel]
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _refine_hole(Y, c0, beta=70.0, steps=200, s0=0.05, s1=0.004):
    c = c0 / (np.linalg.norm(c0) + 1e-300)
    vel = np.zeros(4)
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        s = s0 * (s1 / s0) ** frac
        dots = Y @ c
        m = float(dots.max())
        w = np.exp(np.clip(beta * (dots - m), -50.0, 0.0))
        sw = float(w.sum())
        if sw <= 0 or not np.isfinite(sw):
            break
        w = w / sw
        g = w @ Y
        gt = g - float(np.dot(g, c)) * c
        nn = float(np.linalg.norm(gt))
        if nn < 1e-14:
            break
        gt = gt / nn
        vel = 0.5 * vel + gt
        vn = float(np.linalg.norm(vel))
        if vn < 1e-14:
            break
        c = c - (s / vn) * vel
        n = float(np.linalg.norm(c))
        if n == 0 or not np.isfinite(n):
            break
        c = c / n
    return c / np.linalg.norm(c)

def _largest_hole(Y, rng, n_samples=40000, beta=70.0):
    d = Y.shape[1]
    best_s = 2.0
    best_c = None
    bs = 8000
    for a in range(0, n_samples, bs):
        nb = min(bs, n_samples - a)
        C = rng.standard_normal((nb, d))
        C = _norm_rows(C)
        S = C @ Y.T
        sm = S.max(axis=1)
        k = int(np.argmin(sm))
        if float(sm[k]) < best_s:
            best_s = float(sm[k])
            best_c = C[k].copy()
    if best_c is None:
        best_c = rng.standard_normal(4)
        best_c /= np.linalg.norm(best_c)
    best_c = _refine_hole(Y, best_c, beta=beta, steps=200)
    return best_c

def _joint_cavity_opt(Y, k_last, steps=100, alpha=60.0, smax=0.03, mu=0.5):
    Y = _norm_rows(Y.copy())
    n = Y.shape[0]
    idx = np.arange(n - k_last, n)
    vel = np.zeros((k_last, 4))
    F = Y.copy()
    for it in range(steps):
        Xa = Y[idx]
        G = Xa @ Y.T
        # mask self
        for p, g in enumerate(idx):
            G[p, int(g)] = -2.0
        m = float(G.max())
        W = np.exp(np.clip(alpha * (G - m), -50.0, 0.0))
        grad = W @ Y
        dot = np.sum(grad * Xa, axis=1, keepdims=True)
        gt = grad - dot * Xa
        vel = mu * vel + gt
        nrm = float(np.linalg.norm(vel, axis=1).max())
        if nrm < 1e-14:
            break
        s = smax * (0.997 ** it)
        Xa2 = Xa - (s / nrm) * vel
        Xa2 = _norm_rows(Xa2)
        Y[idx] = Xa2
    return _norm_rows(Y)

def solve():
    t0 = time.time()
    BUDGET = 172.0
    T_END = t0 + BUDGET
    rng_master = np.random.default_rng(20261031)

    X_best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (272, 4):
                a = _norm_rows(a)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    X_best = a.copy()
        except Exception:
            pass
    if X_best is None:
        X_best = _norm_rows(rng_master.standard_normal((272, 4)))
        best_val = _max_cos(X_best)
    X0 = X_best.copy()

    # precompute degree orderings for diversity
    def degrees(X, window=0.002):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        deg = np.count_nonzero(G > m - window, axis=1)
        rowmax = G.max(axis=1)
        return deg, rowmax, m, G

    deg0, rowmax0, m0, G0 = degrees(X0)
    order_deg = np.argsort(-deg0, kind="stable")
    order_row = np.argsort(-rowmax0, kind="stable")
    seed_c = int(np.argmax(deg0))
    ds = G0[seed_c].copy()
    order_near = np.argsort(-ds, kind="stable")

    rng_r = np.random.default_rng(551)
    rand_a = rng_r.choice(272, size=36, replace=False)
    rng_r2 = np.random.default_rng(552)
    rand_b = rng_r2.choice(272, size=24, replace=False)
    rng_r3 = np.random.default_rng(553)
    rand_c = rng_r3.choice(272, size=28, replace=False)
    # low-degree (rattler side) deletion
    order_low = np.argsort(deg0, kind="stable")

    trials = []
    trials.append(("deg28", order_deg[:28].copy()))
    trials.append(("cluster28", np.concatenate([[seed_c], order_near[:27]]).copy() if len(order_near) >= 27 else order_deg[:28].copy()))
    trials.append(("worst24", order_row[:24].copy()))
    trials.append(("rand36", rand_a.copy()))
    trials.append(("deg16", order_deg[:16].copy()))
    trials.append(("low20", order_low[:20].copy()))
    trials.append(("rand24", rand_b.copy()))
    trials.append(("deg36", order_deg[:36].copy()))
    trials.append(("rand28", rand_c.copy()))
    trials.append(("worst32", order_row[:32].copy()))

    for ti, (name, del_idx) in enumerate(trials):
        if time.time() > T_END - 26.0:
            break
        time_left = T_END - time.time() - 20.0
        rem = len(trials) - ti
        if time_left < 4.0:
            break
        trial_budget = time_left / rem
        trial_budget = min(max(trial_budget, 7.0), 24.0)
        T_TRIAL_END = time.time() + trial_budget
        try:
            D = int(len(del_idx))
            # alternate base between X0 and current best for diversity
            base = X_best if (ti % 2 == 1) else X0
            # recompute deletion on base if degree-based
            if name.startswith("deg") or name.startswith("worst") or name.startswith("low") or name.startswith("cluster"):
                degb, rowb, mb, Gb = degrees(base)
                if name.startswith("deg"):
                    odb = np.argsort(-degb, kind="stable")
                    del2 = odb[:D].copy()
                elif name.startswith("worst"):
                    orb = np.argsort(-rowb, kind="stable")
                    del2 = orb[:D].copy()
                elif name.startswith("low"):
                    olb = np.argsort(degb, kind="stable")
                    del2 = olb[:D].copy()
                else:
                    s2 = int(np.argmax(degb))
                    ds2 = (base @ base[s2]).copy()
                    on2 = np.argsort(-ds2, kind="stable")
                    on2 = on2[on2 != s2]
                    del2 = np.concatenate([[s2], on2[:D-1]]).copy()
                mask = np.ones(272, dtype=bool)
                mask[np.asarray(del2, dtype=int)] = False
                Y = base[mask].copy()
            else:
                mask = np.ones(272, dtype=bool)
                mask[np.asarray(del_idx, dtype=int)] = False
                Y = base[mask].copy()

            rng = np.random.default_rng(9000 + ti * 977)
            Y = Y + 0.0025 * rng.standard_normal(Y.shape)
            Y = _norm_rows(Y)
            # thin anneal
            Y = _smooth_gd(Y, steps=550, a0=12.0, a1=420.0, s0=0.030, s1=0.003, mu=0.85, deadline=T_TRIAL_END - 6.0)
            # regrow
            for k in range(D):
                if time.time() > T_TRIAL_END - 7.0:
                    break
                c = _largest_hole(Y, rng, n_samples=40000, beta=70.0)
                Y = np.vstack([Y, c[None, :]])
                # joint optimize last min(k+1,6) points briefly
                kl = min(k + 1, 6)
                Y = _joint_cavity_opt(Y, kl, steps=60, alpha=60.0, smax=0.03, mu=0.5)
                if ((k + 1) % 6 == 0) or ((k + 1) == D):
                    if (k + 1) == D:
                        Y = _smooth_gd(Y, steps=450, a0=14.0, a1=550.0, s0=0.020, s1=0.002, mu=0.86, deadline=T_TRIAL_END - 5.0)
                    else:
                        Y = _smooth_gd(Y, steps=300, a0=14.0, a1=350.0, s0=0.020, s1=0.003, mu=0.85, deadline=T_TRIAL_END - 5.0)
            if Y.shape[0] != 272:
                continue
            if time.time() < T_TRIAL_END - 4.0:
                Y = _smooth_gd(Y, steps=400, a0=10.0, a1=800.0, s0=0.014, s1=0.0015, mu=0.88, deadline=T_TRIAL_END - 3.0)
            v = _max_cos(Y)
            lp_dl = min(T_TRIAL_END, time.time() + 3.5)
            if lp_dl > time.time() + 0.7 and v < best_val + 0.004:
                Y2, v2 = _lp_polish(Y, lp_dl, maxpairs=5000)
                if v2 < v:
                    Y, v = Y2, v2
            if v < best_val:
                best_val = v
                X_best = Y.copy()
        except Exception:
            continue

    # final polish chain
    try:
        if time.time() < T_END - 3.0:
            X_best = _smooth_gd(X_best, steps=500, a0=12.0, a1=900.0, s0=0.010, s1=0.001, mu=0.88, deadline=T_END - 8.0)
            v = _max_cos(X_best)
            if v < best_val:
                best_val = v
        if time.time() < T_END - 1.5:
            X_best, v = _lp_polish(X_best, T_END - 0.3, maxpairs=6000)
            best_val = min(best_val, v)
    except Exception:
        pass
    X_best = _norm_rows(X_best)
    return X_best
