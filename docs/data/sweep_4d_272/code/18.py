import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    x = x / n
    bad = ~np.all(np.isfinite(x), axis=1)
    if np.any(bad):
        x[bad] = np.array([1.0, 0.0, 0.0, 0.0])
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _gd(X0, steps=400, b0=80.0, b1=800.0, lr0=0.012, lr1=0.003, mu=0.88, max_step=0.035):
    X = X0.copy()
    n = X.shape[0]
    vel = np.zeros_like(X)
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = b0 * (b1 / b0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = X @ X.T
        np.fill_diagonal(G, -1.0)
        m = float(np.max(G))
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        rs = np.sum(W, axis=1, keepdims=True) + 1e-12
        grad = W @ X
        d = np.sum(grad * X, axis=1, keepdims=True)
        gt = (grad - d * X) / rs
        vel = mu * vel - lr * gt
        vn = np.sqrt(np.sum(vel * vel, axis=1, keepdims=True)) + 1e-12
        sc = np.minimum(1.0, max_step / vn)
        vel = vel * sc
        X = X + vel
        # renormalize
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return X


def _lp_limit(x, max_iters=40, r0=2e-3, rmax=0.02):
    x = _normalize(x)
    n, d = x.shape
    best = _max_cos(x)
    bx = x.copy()
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    for _ in range(max_iters):
        c_all = (x @ x.T)[iu0, iu1]
        m = float(np.max(c_all))
        thr = m - max(15.0 * radius, 2e-9)
        sel = c_all > thr
        if np.count_nonzero(sel) == 0:
            break
        # cap to tightest 2500 to keep HiGHS fast
        if np.count_nonzero(sel) > 2500:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:2500]
            idx2 = idx[order]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx2] = True
        I, J = iu0[sel], iu1[sel]
        cs = c_all[sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cc = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vv = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vv, (rows, cc)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            if radius < 1e-13:
                break
            continue
        if res.status != 0:
            radius *= 0.5
            if radius < 1e-13:
                break
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _normalize(y)
        v = _max_cos(y)
        if v < best:
            best = v
            bx = y.copy()
            x = y
            radius = min(radius * 1.5, rmax)
        else:
            # accept move of x only if it doesn't worsen too much? keep x
            radius *= 0.5
            if radius < 1e-13:
                break
    return bx, best


def _lp_polish_timed(x, t_end):
    x = _normalize(x)
    n, d = x.shape
    best = _max_cos(x)
    bx = x.copy()
    radius = 1e-3
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < t_end and radius > 1e-14:
        if time.time() + 0.8 > t_end and radius < 4e-4:
            break
        c_all = (x @ x.T)[iu0, iu1]
        m = float(np.max(c_all))
        sel = c_all > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) == 0:
            break
        if np.count_nonzero(sel) > 3000:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:3000]
            idx2 = idx[order]
            sel2 = np.zeros_like(sel, dtype=bool)
            sel2[idx2] = True
            sel = sel2
        I, J = iu0[sel], iu1[sel]
        cs = c_all[sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cc = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vv = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vv, (rows, cc)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _normalize(y)
        v = _max_cos(y)
        if v < best:
            x, best, bx = y, v, y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return bx


def solve():
    t0 = time.time()
    T_HARD = t0 + 175.0
    T_HOP = t0 + 152.0
    rng = np.random.default_rng(12345)
    try:
        best = np.load("best.npy")
    except Exception:
        best = np.load("record.npy")
    best = _normalize(np.asarray(best, dtype=np.float64))
    if best.shape != (272, 4):
        best = _normalize(rng.standard_normal((272, 4)))
    best_val = _max_cos(best)
    # snap starting point once (cheap) to ensure local optimum
    try:
        best, best_val2 = _lp_limit(best, max_iters=25, r0=1e-3)
        best_val = min(best_val, best_val2)
    except Exception:
        pass
    cur = best.copy()
    cur_val = best_val
    n = 272
    hop = 0
    # temperature schedule for annealing acceptance on cur
    T0, T1 = 3e-4, 2e-6

    while time.time() < T_HOP and hop < 500:
        frac = hop / 120.0
        Temp = T0 * (T1 / T0) ** min(1.0, (time.time() - t0) / max(1.0, T_HOP - t0))
        # choose base: mostly cur, every 4th from best
        if hop % 4 == 3:
            base = best
        else:
            base = cur
        mode = hop % 6
        Xp = base.copy()
        if mode == 0:
            Xp = Xp + 0.012 * rng.standard_normal(Xp.shape)
        elif mode == 1:
            Xp = Xp + 0.035 * rng.standard_normal(Xp.shape)
        elif mode == 2:
            Xp = Xp + 0.075 * rng.standard_normal(Xp.shape)
        elif mode == 3:
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            cnt = np.sum(G > m - 5e-4, axis=1)
            idx = np.argsort(-cnt)[:22]
            Xp = Xp + 0.008 * rng.standard_normal(Xp.shape)
            Xp[idx] += 0.14 * rng.standard_normal((len(idx), 4))
        elif mode == 4:
            # void reinsertion of worst points
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            pw = np.max(G, axis=1)
            k = int(rng.choice([2, 3, 4]))
            worst = np.argsort(-pw)[:k]
            # sample holes w.r.t. remaining points
            S = _normalize(rng.standard_normal((7000, 4)))
            # iterative greedy for each worst
            tmp = base.copy()
            for w in worst:
                # dots to all except w
                dots = S @ tmp.T
                # ignore column w by setting -inf
                dots[:, w] = -10.0
                md = np.max(dots, axis=1)
                bi = int(np.argmin(md))
                tmp[w] = S[bi] + 0.003 * rng.standard_normal(4)
                tmp[w] /= max(np.linalg.norm(tmp[w]), 1e-12)
                # refresh S locally around bi? keep simple
            Xp = tmp
            mask = np.ones(n, dtype=bool)
            mask[worst] = False
            Xp[mask] += 0.007 * rng.standard_normal((n - k, 4))
        else:
            k = int(rng.choice([3, 5, 8]))
            idx = rng.choice(n, size=k, replace=False)
            Xp[idx] = rng.standard_normal((k, 4))
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            Xp[mask] += 0.009 * rng.standard_normal((n - k, 4))
        Xp = _normalize(Xp)
        if time.time() + 0.6 > T_HOP:
            break
        # short annealed GD
        if mode in (2, 5):
            steps = 500
        else:
            steps = 350
        Xg = _gd(Xp, steps=steps, b0=60.0, b1=900.0, lr0=0.011, lr1=0.0025, mu=0.88, max_step=0.04)
        # snap with LP (limited iters for speed)
        try:
            Xl, vl = _lp_limit(Xg, max_iters=22, r0=1.5e-3)
        except Exception:
            Xl, vl = Xg, _max_cos(Xg)
        # update best
        if vl < best_val:
            best = Xl.copy()
            best_val = vl
            cur = Xl.copy()
            cur_val = vl
        else:
            # annealing on cur
            if vl < cur_val or rng.random() < np.exp(-(vl - cur_val) / max(Temp, 1e-12)):
                cur = Xl.copy()
                cur_val = vl
        hop += 1

    # final polish of best with remaining time
    if time.time() < T_HARD - 3.0:
        best = _lp_polish_timed(best, T_HARD - 1.0)
        best = _normalize(best)
    return best
