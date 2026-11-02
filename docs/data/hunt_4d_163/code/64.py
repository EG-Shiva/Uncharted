import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

def _lp_polish(x, deadline, r0=0.003, cap=1000):
    x = _norm(np.array(x, dtype=float, copy=True))
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x, 1.0
    radius = float(r0)
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            G = x @ x.T
            c = G[iu0, iu1]
            m = float(c.max())
            sel = c > m - max(20.0 * radius, 1e-9)
            if int(sel.sum()) == 0:
                sel[int(np.argmax(c))] = True
            if int(sel.sum()) > cap:
                idx = np.argsort(-c)[:cap]
                s2 = np.zeros_like(sel, dtype=bool)
                s2[idx] = True
                sel = s2
            I = iu0[sel]; J = iu1[sel]
            csel = c[sel]
            p = len(I)
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            tmpc = np.empty((p, 2 * d + 1), dtype=int)
            ar = np.arange(d)
            tmpc[:, :d] = I[:, None] * d + ar
            tmpc[:, d:2 * d] = J[:, None] * d + ar
            tmpc[:, 2 * d] = n * d
            cols = tmpc.ravel()
            cc = np.empty((p, 2 * d + 1), dtype=float)
            cc[:, :d] = x[J]
            cc[:, d:2 * d] = x[I]
            cc[:, 2 * d] = -1.0
            vals = cc.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y = _norm(y)
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        except Exception:
            radius *= 0.5
            continue
    return x, best

def _softmax_lbfgs(x_init, deadline, taus=(0.002, 0.0007, 0.0003), iters=(50, 50, 60)):
    x = _norm(np.array(x_init, dtype=float, copy=True))
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Np = len(iu0)
    y = x.ravel().copy()
    for tau, maxiter in zip(taus, iters):
        if time.time() > deadline - 0.3:
            break
        def fg(yflat, _tau=float(tau)):
            Y = yflat.reshape(n, d)
            rn = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
            X = Y / rn
            g = (X @ X.T)[iu0, iu1]
            m = float(g.max())
            e = np.exp((g - m) / _tau)
            s = float(e.sum())
            if s <= 0 or not np.isfinite(s):
                return float(m), np.zeros_like(yflat)
            L = float(m + _tau * np.log(s / float(Np)))
            if not np.isfinite(L):
                return 1e6, np.zeros_like(yflat)
            w = e / s
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            GY = (GX - (GX * X).sum(axis=1, keepdims=True) * X) / rn
            GY = np.nan_to_num(GY, nan=0.0, posinf=0.0, neginf=0.0)
            return float(L), GY.ravel()
        try:
            res = minimize(fg, y, method="L-BFGS-B", jac=True,
                           options={"maxiter": int(maxiter), "maxfun": int(maxiter * 4),
                                    "ftol": 1e-14, "gtol": 1e-9})
            y = np.array(res.x, dtype=float)
        except Exception:
            break
    Yf = _norm(y.reshape(n, d))
    return Yf, _max_cos(Yf)

def _anneal(x0, steps, T0, T1, lr, mu, rng, noise0=0.0, deadline=None):
    x = _norm(np.array(x0, dtype=float, copy=True))
    n, d = x.shape
    vel = np.zeros_like(x)
    try:
        bv = _max_cos(x)
    except Exception:
        bv = 1.0
    bx = x.copy()
    if T0 <= 0:
        T0 = 1e-6
    if T1 <= 0:
        T1 = 5e-5
    lratio = float(np.log(T1 / T0))
    for k in range(int(steps)):
        if deadline is not None and (k & 63) == 0 and time.time() > deadline:
            break
        frac = k / max(1, int(steps) - 1)
        T = float(T0 * np.exp(lratio * frac))
        if T < 5e-5:
            T = 5e-5
        lrk = float(lr * (1.0 - 0.5 * frac))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S < 1e-300:
            S = 1e-300
        Grad = (E @ x) / S
        Grad -= (Grad * x).sum(axis=1, keepdims=True) * x
        vel = float(mu) * vel + (1.0 - float(mu)) * Grad
        x = x - lrk * vel
        if noise0 > 0:
            amp = float(noise0 * (1.0 - frac))
            if amp > 0:
                nz = rng.randn(n, d) * amp
                nz -= (nz * x).sum(axis=1, keepdims=True) * x
                x = x + nz
        x = _norm(x)
        if (k & 7) == 0:
            try:
                v = _max_cos(x)
            except Exception:
                v = 1.0
            if v < bv:
                bv = v
                bx = x.copy()
    return bx, bv

def _kick_big(x, rng, frac, ang):
    y = _norm(np.array(x, dtype=float, copy=True))
    n, d = y.shape
    k = max(1, int(round(n * float(frac))))
    idx = rng.choice(n, size=k, replace=False)
    for i in idx:
        xi = y[i].copy()
        r = rng.randn(d)
        r -= float(r @ xi) * xi
        rn = float(np.linalg.norm(r))
        if rn < 1e-12:
            continue
        r /= rn
        th = float(rng.uniform(float(ang) * 0.4, float(ang)))
        y[i] = np.cos(th) * xi + np.sin(th) * r
    return _norm(y)

def _rand_rot(rng, eps):
    M = np.eye(4) + rng.randn(4, 4) * float(eps)
    Q, R = np.linalg.qr(M)
    dd = np.sign(np.diag(R) + 1e-12)
    Q = Q * dd
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

def _cap_rotate(x, rng):
    y = np.array(x, dtype=float, copy=True)
    n = y.shape[0]
    c = rng.randn(4)
    c /= max(np.linalg.norm(c), 1e-12)
    thr = float(rng.uniform(0.80, 0.93))
    dots = y @ c
    mask = dots > thr
    cnt = int(mask.sum())
    if cnt < 2 or cnt > n - 5:
        # random small global rotation instead
        Q = _rand_rot(rng, 0.05)
        return _norm(y @ Q.T)
    Q = _rand_rot(rng, float(rng.uniform(0.06, 0.18)))
    cen = y[mask].mean(axis=0)
    y[mask] = (y[mask] - cen) @ Q.T + cen
    # also jitter center slightly
    return _norm(y)

def _refine_hole(h0, Xf, steps=120, T0=0.015, T1=0.002, lr0=0.1):
    h = np.array(h0, dtype=float).ravel()
    h /= max(np.linalg.norm(h), 1e-12)
    if T0 <= 0:
        T0 = 1e-6
    if T1 <= 0:
        T1 = 5e-5
    lratio = float(np.log(T1 / T0))
    for k in range(int(steps)):
        frac = k / max(1, int(steps) - 1)
        T = max(float(T0 * np.exp(lratio * frac)), 1e-12)
        lr = float(lr0 * (1.0 - 0.5 * frac))
        dots = Xf @ h
        m = float(dots.max())
        e = np.exp((dots - m) / T)
        s = float(e.sum())
        if s <= 0 or not np.isfinite(s):
            break
        w = e / s
        grad = w @ Xf
        grad = grad - float(grad @ h) * h
        h = h - lr * grad
        nh = float(np.linalg.norm(h))
        if nh < 1e-12:
            break
        h /= nh
    return h

def _find_holes(Xf, rng, n_samples=12000, n_holes=4):
    Y = rng.randn(int(n_samples), 4)
    Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    D = Y @ Xf.T
    vals = D.max(axis=1)
    order = np.argsort(vals)
    cand = order[:min(len(order), max(24, 4 * int(n_holes)))]
    outs = []
    for ii in cand:
        try:
            h = _refine_hole(Y[ii], Xf)
            v = float((Xf @ h).max())
            outs.append((v, h))
        except Exception:
            continue
    outs.sort(key=lambda t: t[0])
    # deduplicate
    picked = []
    for v, h in outs:
        ok = True
        for _, q in picked:
            if float(abs(h @ q)) > 0.96:
                ok = False
                break
        if ok:
            picked.append((v, h))
        if len(picked) >= n_holes:
            break
    while len(picked) < n_holes and len(outs) > len(picked):
        picked.append(outs[len(picked)])
    return [h for _, h in picked[:n_holes]]

def _transplant(x, rng, nh):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    deg = (G > m - 0.003).sum(axis=1)
    rowm = G.max(axis=1)
    order = np.lexsort((-rowm, -deg))
    hubs = order[:int(nh)]
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[hubs] = False
    Xk = x[keep_mask]
    try:
        holes = _find_holes(Xk, rng, n_samples=12000, n_holes=int(nh))
    except Exception:
        return _kick_big(x, rng, 0.2, 0.4)
    if len(holes) < int(nh):
        return _kick_big(x, rng, 0.2, 0.4)
    H = np.stack(holes, axis=0)
    # small jitter on holes to break symmetry
    for i in range(H.shape[0]):
        r = rng.randn(4)
        r -= float(r @ H[i]) * H[i]
        rn = float(np.linalg.norm(r))
        if rn > 1e-12:
            r /= rn
            th = float(rng.uniform(0.0, 0.02))
            H[i] = np.cos(th) * H[i] + np.sin(th) * r
    H = _norm(H)
    y = np.concatenate([Xk, H], axis=0)
    # shuffle order to avoid index bias
    perm = rng.permutation(n)
    y = y[perm]
    return _norm(y)

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    SEARCH_END = t0 + 148.0
    rng = np.random.RandomState(20261102)
    nT = 163
    bx = None
    try:
        a = np.load("best.npy")
        a = np.asarray(a, dtype=float)
        if a.shape == (nT, 4) and np.all(np.isfinite(a)):
            bx = _norm(a)
    except Exception:
        bx = None
    rec = None
    try:
        b = np.load("record.npy")
        b = np.asarray(b, dtype=float)
        if b.shape == (nT, 4) and np.all(np.isfinite(b)):
            rec = _norm(b)
    except Exception:
        rec = None
    if bx is None and rec is None:
        bx = _norm(rng.randn(nT, 4))
    elif bx is None:
        bx = rec
    best_x = bx.copy()
    try:
        best_val = _max_cos(best_x)
    except Exception:
        best_x = _norm(rng.randn(nT, 4))
        best_val = _max_cos(best_x)
    if rec is not None:
        try:
            rv = _max_cos(rec)
            if rv < best_val:
                best_x = rec.copy()
                best_val = rv
        except Exception:
            pass
    cur_x = best_x.copy()
    cur_val = best_val
    # short initial freeze to tighten baseline if time allows
    try:
        if time.time() < SEARCH_END - 10:
            x0, v0 = _lp_polish(best_x, min(SEARCH_END, time.time() + 4.0), r0=0.002, cap=1000)
            if v0 < best_val:
                best_x, best_val = x0, v0
                cur_x, cur_val = x0.copy(), v0
    except Exception:
        pass
    it = 0
    TEMP = 0.0006
    while time.time() < SEARCH_END - 6.0:
        it += 1
        remain = SEARCH_END - time.time()
        if remain < 7.0:
            break
        # choose base: mostly best, sometimes current for diversity
        if rng.rand() < 0.72:
            base = best_x
        else:
            base = cur_x
        r = rng.rand()
        try:
            if r < 0.50:
                nh = int(rng.randint(3, 8))
                y = _transplant(base, rng, nh)
                # add small global jitter
                y = _kick_big(y, rng, 0.10, 0.12)
            elif r < 0.78:
                frac = float(rng.uniform(0.15, 0.35))
                ang = float(rng.uniform(0.18, 0.55))
                y = _kick_big(base, rng, frac, ang)
                if rng.rand() < 0.4:
                    y = _cap_rotate(y, rng)
            elif r < 0.90:
                y = _cap_rotate(base, rng)
                y = _kick_big(y, rng, 0.12, 0.20)
            else:
                # fresh annealed restart from random subset mixed with best
                if rng.rand() < 0.5:
                    y = _norm(rng.randn(nT, 4))
                else:
                    y = _kick_big(best_x, rng, 0.60, 0.80)
        except Exception:
            continue
        # anneal
        try:
            steps = 700 if remain > 50 else 350
            T0 = float(rng.uniform(0.006, 0.010))
            T1 = float(rng.uniform(0.0004, 0.0008))
            lr = float(rng.uniform(0.05, 0.10))
            mu = float(rng.uniform(0.70, 0.88))
            nz = float(rng.uniform(0.0, 0.004))
            t_ann_end = time.time() + min(1.2, remain * 0.25)
            y2, v2a = _anneal(y, steps, T0, T1, lr, mu, rng, noise0=nz, deadline=t_ann_end)
        except Exception:
            continue
        # lbfgs refine
        try:
            t_lbf_end = min(SEARCH_END - 3.0, time.time() + min(2.5, remain * 0.4))
            # adapt iters to remaining
            if remain < 25:
                taus = (0.001, 0.00035)
                iters = (40, 40)
            else:
                taus = (0.002, 0.0007, 0.0003)
                iters = (45, 45, 55)
            y3, v3 = _softmax_lbfgs(y2, t_lbf_end, taus=taus, iters=iters)
        except Exception:
            y3, v3 = y2, v2a
        # short LP freeze
        try:
            t_lp_end = min(SEARCH_END, time.time() + 2.2)
            y4, v4 = _lp_polish(y3, t_lp_end, r0=0.003, cap=900)
        except Exception:
            y4, v4 = y3, v3
        # pick best among stages
        cand_x, cand_v = y4, v4
        if v3 < cand_v:
            cand_x, cand_v = y3, v3
        if v2a < cand_v:
            cand_x, cand_v = y2, v2a
        # update global best
        if cand_v < best_val - 1e-12:
            best_x = _norm(cand_x.copy())
            best_val = float(cand_v)
            cur_x = best_x.copy()
            cur_val = best_val
            TEMP *= 0.98
        else:
            # Metropolis sideways move to change basin
            try:
                if cand_v < cur_val or rng.rand() < float(np.exp((cur_val - cand_v) / TEMP)):
                    cur_x = _norm(cand_x.copy())
                    cur_val = float(cand_v)
            except Exception:
                pass
            # occasional random restart of walker to best with kick
            if rng.rand() < 0.05:
                try:
                    cur_x = _kick_big(best_x, rng, 0.20, 0.30)
                    cur_val = _max_cos(cur_x)
                except Exception:
                    pass
    # final long polish
    try:
        if time.time() < DEADLINE - 2.0:
            x_f, v_f = _lp_polish(best_x, DEADLINE - 1.0, r0=0.002, cap=1400)
            if v_f < best_val:
                best_x, best_val = x_f, v_f
    except Exception:
        pass
    try:
        best_x = _norm(best_x)
        # guard against NaN
        if not np.all(np.isfinite(best_x)):
            best_x = _norm(bx)
    except Exception:
        best_x = _norm(bx)
    return best_x
