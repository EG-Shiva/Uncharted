import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _personal_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)


def _contacts_cnt(x, tol=1e-7):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = float(g.max())
    cnt = (g > m - tol).sum(axis=1).astype(int)
    return cnt, m


def _smooth_polish(x, steps=500, T0=0.02, T1=0.003, lr0=0.04, lr1=0.005):
    n, d = x.shape
    x = x.copy()
    best = _max_cos(x)
    best_x = x.copy()
    # geometric schedules
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        # max for shift
        # set diag low so it does not dominate
        np.fill_diagonal(G, -1.0)
        m = G.max()
        # weights
        # clip argument to avoid overflow (already <=0)
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-12
        C = (W @ x) / S
        dot = (C * x).sum(axis=1, keepdims=True)
        Ct = C - dot * x
        x = x - lr * Ct
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm[nrm < 1e-12] = 1.0
        x = x / nrm
        if (s & 63) == 63 or s == steps - 1:
            v = _max_cos(x)
            if v < best:
                best = v
                best_x = x.copy()
    return best_x, best


def _best_hole_vec(Xkeep, rng, S=4000):
    nkeep, d = Xkeep.shape
    R = rng.standard_normal((S, d))
    R /= np.linalg.norm(R, axis=1, keepdims=True) + 1e-16
    D = R @ Xkeep.T
    m = D.max(axis=1)
    i = int(np.argmin(m))
    return R[i].copy(), float(m[i])


def _lp_polish(x, deadline, r_init=0.01, r_max=0.05, cap=2000, max_it=60):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = r_init
    iu0 = np.triu_indices(n, 1)
    nv = n * d + 1
    obj = np.zeros(nv)
    obj[-1] = 1.0
    it = 0
    while time.time() + 0.35 < deadline and radius > 1e-13 and it < max_it:
        it += 1
        G = x @ x.T
        c_all = G[iu0]
        m = c_all.max()
        thresh = max(20.0 * radius, 1e-9)
        sel = c_all > m - thresh
        nnz = int(np.count_nonzero(sel))
        if nnz == 0:
            radius *= 0.5
            continue
        if nnz > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        c_sel = c_all[sel]
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        bounds = [(-radius, radius)] * (n * d) + [(None, None)]
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y = y / yn
        v = _max_cos(y)
        if v < best:
            # keep best trajectory
            x = y
            best = v
            best_x = y.copy()
            radius = min(radius * 1.5, r_max)
        else:
            # still move x if equal-ish? no, shrink
            radius *= 0.5
            # if stuck at tiny radius, break
            if radius < 1e-12:
                break
    return best_x, best


def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    rng = np.random.default_rng(0)
    n, d = 78, 15
    cand = []
    try:
        b = np.load("best.npy")
        if b.shape == (n, d):
            cand.append(b.astype(float))
    except Exception:
        pass
    try:
        r = np.load("record.npy")
        if r.shape == (n, d):
            cand.append(r.astype(float))
    except Exception:
        pass
    if len(cand) == 0:
        x0 = rng.standard_normal((n, d))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        cand.append(x0)
    best_v = 1e9
    best_x0 = None
    for c in cand:
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        v = _max_cos(c)
        if v < best_v:
            best_v = v
            best_x0 = c.copy()
    x_best = best_x0.copy()
    v_best = best_v

    # brief initial squeeze
    if time.time() < HARD_END - 30:
        x_best, v_best = _lp_polish(x_best, min(time.time() + 6.0, HARD_END - 26), r_init=0.002, r_max=0.015)

    trial = 0
    while time.time() < HARD_END - 22:
        trial += 1
        remaining = HARD_END - 22 - time.time()
        if remaining < 3:
            break
        mode = trial % 7
        x_try = x_best.copy()
        try:
            pm = _personal_max(x_try)
            cnt, _m = _contacts_cnt(x_try, tol=1e-7)
        except Exception:
            pm = np.zeros(n)
            cnt = np.zeros(n, dtype=int)

        if mode == 0:
            x_try = x_try + 0.008 * rng.standard_normal((n, d))
        elif mode == 1:
            x_try = x_try + 0.025 * rng.standard_normal((n, d))
        elif mode == 2:
            order = np.argsort(-pm)
            k = 8
            worst = order[:k]
            noise = 0.005 * rng.standard_normal((n, d))
            noise[worst] += 0.07 * rng.standard_normal((k, d))
            x_try = x_try + noise
        elif mode == 3:
            order = np.argsort(cnt)
            k = 15
            tail = order[:k]
            noise = 0.004 * rng.standard_normal((n, d))
            noise[tail] += 0.055 * rng.standard_normal((k, d))
            x_try = x_try + noise
        elif mode == 4:
            # hole reinsert 1-2 rattlers
            order = np.argsort(cnt)
            k = 2 if (trial % 3) else 1
            repl = order[:k]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[repl] = False
            Xkeep = x_try[keep_mask]
            for ii, idx in enumerate(repl):
                # include previously inserted for sequential hole search
                if ii > 0:
                    Xkeep2 = x_try[keep_mask].copy()
                    # update with already replaced? approximate with current x_try keep + new
                    # rebuild from x_try excluding not-yet-replaced
                    tmp_mask = np.ones(n, dtype=bool)
                    tmp_mask[repl[ii:]] = False
                    # keep already replaced vectors
                    Xkeep2 = x_try[tmp_mask]
                    Xkeep = Xkeep2
                hv, _ = _best_hole_vec(Xkeep, rng, S=4000)
                x_try[idx] = hv
                # update Xkeep for next ii (will be rebuilt)
            # jitter rest slightly
            others = np.where(keep_mask)[0]
            x_try[others] += 0.003 * rng.standard_normal((len(others), d))
        elif mode == 5:
            # relocate overloaded points to holes
            order = np.argsort(-cnt)  # high contacts first
            k = 1 if (trial % 2 == 0) else 2
            repl = order[:k]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[repl] = False
            Xkeep = x_try[keep_mask]
            for ii, idx in enumerate(repl):
                if ii > 0:
                    tmp_mask = np.ones(n, dtype=bool)
                    tmp_mask[repl[ii:]] = False
                    Xkeep = x_try[tmp_mask]
                hv, _ = _best_hole_vec(Xkeep, rng, S=4000)
                # blend hole with old to keep some memory half the time
                if (trial % 4) == 1:
                    hv = hv + 0.3 * rng.standard_normal(d)
                    hv /= np.linalg.norm(hv) + 1e-16
                x_try[idx] = hv
            others = np.where(keep_mask)[0]
            x_try[others] += 0.004 * rng.standard_normal((len(others), d))
        else:
            # mode 6: random block shake
            k = 10
            sel = rng.choice(n, size=k, replace=False)
            noise = 0.003 * rng.standard_normal((n, d))
            noise[sel] += 0.045 * rng.standard_normal((k, d))
            x_try = x_try + noise
            # occasional full random restart injection
            if trial % 28 == 0:
                xr = rng.standard_normal((n, d))
                xr /= np.linalg.norm(xr, axis=1, keepdims=True)
                # mix: replace worst 20 with random
                order = np.argsort(-pm)
                x_try[order[:20]] = xr[order[:20]]

        # renormalize
        x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)

        # fast smooth screening
        # adapt steps by mode: hole modes get more steps
        if mode in (4, 5):
            s_steps = 600
            sT0, sT1 = 0.018, 0.0025
        else:
            s_steps = 400
            sT0, sT1 = 0.015, 0.003
        try:
            x_s, v_s = _smooth_polish(x_try, steps=s_steps, T0=sT0, T1=sT1, lr0=0.035, lr1=0.004)
        except Exception:
            continue
        # direct accept if smooth alone beats best (rare)
        if v_s < v_best - 1e-12:
            # squeeze with LP immediately
            budget = min(6.0, HARD_END - 22 - time.time())
            if budget > 1.5:
                try:
                    x_p, v_p = _lp_polish(x_s, time.time() + budget, r_init=0.012, r_max=0.04)
                except Exception:
                    x_p, v_p = x_s, v_s
                if v_p < v_best - 1e-13:
                    x_best = x_p
                    v_best = v_p
                else:
                    # keep smooth best if LP failed to improve but smooth did
                    if v_s < v_best - 1e-13:
                        x_best = x_s
                        v_best = v_s
            else:
                x_best = x_s
                v_best = v_s
            continue
        # otherwise only LP-polish if close
        if v_s > v_best + 0.012:
            continue
        budget = min(5.0, HARD_END - 22 - time.time())
        if budget < 1.2:
            continue
        try:
            r_init = 0.014 if mode in (2, 3, 4, 5, 6) else 0.008
            x_p, v_p = _lp_polish(x_s, time.time() + budget, r_init=r_init, r_max=0.05)
        except Exception:
            continue
        if v_p < v_best - 1e-13:
            x_best = x_p
            v_best = v_p

    # final intensive polish
    try:
        if time.time() < HARD_END - 1:
            x_best, v_best = _lp_polish(x_best, HARD_END - 0.6, r_init=0.0025, r_max=0.015)
    except Exception:
        pass
    try:
        if time.time() < HARD_END - 6:
            # fine smooth then LP
            xs, vs = _smooth_polish(x_best, steps=400, T0=0.006, T1=0.001, lr0=0.012, lr1=0.002)
            if vs < v_best:
                x_best, v_best = xs, vs
            if time.time() < HARD_END - 0.5:
                x_best, v_best = _lp_polish(x_best, HARD_END - 0.3, r_init=0.0006, r_max=0.006)
    except Exception:
        pass

    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    if not np.all(np.isfinite(x_best)):
        x_best = best_x0 / np.linalg.norm(best_x0, axis=1, keepdims=True)
    return x_best
