import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _lp_polish(x, seconds, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 5e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if t_deadline is not None and time.time() > t_deadline:
            break
        if radius < 1e-13:
            break
        G = x @ x.T
        c_all = G[iu]
        m = float(np.max(c_all))
        win = max(20.0 * radius, 1e-7)
        sel = c_all > m - win
        if np.count_nonzero(sel) > 900:
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[::-1][:900]
            nsel = np.zeros_like(sel, dtype=bool)
            nsel[idx[order]] = True
            sel = nsel
        I, J = iu[0][sel], iu[1][sel]
        c = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.6, 0.05)
        else:
            radius *= 0.5
    return x


def _smooth(x_init, steps, k0, k1, lr0, lr1, beta=0.85):
    x = x_init.copy()
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    vel = np.zeros_like(x)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        k = k0 * (k1 / k0) ** frac if k0 > 0 else k1
        lr = lr0 * (lr1 / lr0) ** frac if steps > 1 else lr0
        G = x @ x.T
        m = float(G[iu].max())
        # softmax weights, threshold for speed
        # use window to sparsify far pairs
        # keep all for correctness when k small
        E = np.exp(np.clip(k * (G - m), -40, 0))
        np.fill_diagonal(E, 0.0)
        Gr = E @ x
        dot = np.sum(Gr * x, axis=1, keepdims=True)
        Gt = Gr - dot * x
        deg = np.sum(E, axis=1, keepdims=True) + 1.0
        Gt /= deg
        vel = beta * vel + Gt
        x = x - lr * vel
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _refine_single(v0, Y, steps, k0, k1, lr0, lr1):
    v = v0.copy()
    v /= np.linalg.norm(v)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        k = k0 + (k1 - k0) * frac
        lr = lr0 * (lr1 / lr0) ** frac if steps > 1 else lr0
        dots = Y @ v
        m = float(np.max(dots))
        w = np.exp(np.clip(k * (dots - m), -40, 0))
        sw = np.sum(w)
        if sw == 0:
            break
        w /= sw
        grad = w @ Y
        gt = grad - float(grad @ v) * v
        v = v - lr * gt
        nv = np.linalg.norm(v)
        if nv == 0:
            v = v0.copy()
            continue
        v /= nv
    return v


def _optimize_single(Y, rng, n_rand=4000, refine_steps=600):
    d = Y.shape[1]
    cand = rng.randn(n_rand, d)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = (cand @ Y.T).max(axis=1)
    bi = int(np.argmin(scores))
    v = cand[bi]
    # two-stage refine: coarse then fine
    v = _refine_single(v, Y, refine_steps, 80.0, 400.0, 0.06, 0.005)
    v = _refine_single(v, Y, refine_steps // 2, 400.0, 900.0, 0.01, 0.002)
    return v


def solve():
    t0 = time.time()
    BUDGET = 170.0
    rng = np.random.RandomState(12345)
    n, d = 54, 23
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(n, d)
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
    except Exception:
        xr = None
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (n, d):
        tmp = rng.randn(n, d)
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        xb = tmp
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    best = _max_cos(xb)
    x_best = xb.copy()
    if xr is not None and xr.shape == (n, d):
        vr = _max_cos(xr)
        if vr < best:
            best, x_best = vr, xr.copy()

    iu = np.triu_indices(n, 1)
    it = 0
    # reserve final polish
    FINAL = 14.0
    while time.time() - t0 < BUDGET - FINAL:
        it += 1
        remain = BUDGET - FINAL - (time.time() - t0)
        if remain < 2.5:
            break
        # contact ranking on best
        Gb = x_best @ x_best.T
        np.fill_diagonal(Gb, -2.0)
        # loose ranking with wider window for guidance
        wide = Gb > best - 0.01
        cnt = wide.sum(axis=1)
        order_lo = np.argsort(cnt)
        order_hi = np.argsort(-cnt)
        x = x_best.copy()
        r = rng.rand()
        lp_time = min(1.6, max(0.6, remain / 8.0))
        sm_steps = 500
        if r < 0.38:
            # single delete-reinsert, biased to loose points
            if rng.rand() < 0.6:
                pool = order_lo[:12]
                idx = int(pool[rng.randint(len(pool))])
            else:
                idx = int(rng.randint(n))
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            Y = x[mask]
            v_new = _optimize_single(Y, rng, n_rand=4000, refine_steps=500)
            x[idx] = v_new
            x += rng.randn(n, d) * 0.002
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _smooth(x, sm_steps, 60, 700, 0.02, 0.003)
            x = _lp_polish(x, lp_time, t_deadline=t0 + BUDGET - FINAL)
        elif r < 0.58:
            # double reinsert
            if rng.rand() < 0.5:
                pool = order_lo[:16]
                idxs = pool[rng.choice(len(pool), size=2, replace=False)]
            else:
                idxs = rng.choice(n, size=2, replace=False)
            rng.shuffle(idxs)
            fixed_mask = np.ones(n, dtype=bool)
            fixed_mask[idxs] = False
            fixed = x[fixed_mask]
            new_vs = []
            cur = fixed
            for kk, idx in enumerate(idxs):
                if kk == 0:
                    Y = cur
                else:
                    Y = np.vstack([cur, np.vstack(new_vs)])
                v_new = _optimize_single(Y, rng, n_rand=3000, refine_steps=400)
                new_vs.append(v_new)
            for idx, v_new in zip(idxs, new_vs):
                x[idx] = v_new
            x += rng.randn(n, d) * 0.003
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _smooth(x, sm_steps + 200, 50, 600, 0.025, 0.003)
            x = _lp_polish(x, lp_time, t_deadline=t0 + BUDGET - FINAL)
        elif r < 0.78:
            # shake loose + jammed differently
            k_lo = 6
            idx_lo = order_lo[:k_lo]
            idx_hi = order_hi[:4]
            x += rng.randn(n, d) * 0.004
            x[idx_lo] += rng.randn(k_lo, d) * (0.05 + 0.05 * rng.rand())
            x[idx_hi] += rng.randn(4, d) * 0.012
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _smooth(x, 700, 70, 800, 0.02, 0.002)
            x = _lp_polish(x, lp_time, t_deadline=t0 + BUDGET - FINAL)
        else:
            # global medium kick + anneal
            sig = 0.015 + 0.02 * rng.rand()
            x += rng.randn(n, d) * sig
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _smooth(x, 800, 40, 500, 0.03, 0.003, beta=0.9)
            x = _lp_polish(x, lp_time + 0.4, t_deadline=t0 + BUDGET - FINAL)
        v = _max_cos(x)
        if v < best - 1e-12:
            best = v
            x_best = x.copy()
        # occasional triple reinsert if stuck for long (every ~25 iters try bigger jump)
        if it % 25 == 0 and time.time() - t0 < BUDGET - FINAL - 8:
            xt = x_best.copy()
            idxs = rng.choice(n, size=3, replace=False)
            fixed_mask = np.ones(n, dtype=bool)
            fixed_mask[idxs] = False
            cur = xt[fixed_mask]
            new_vs = []
            for kk in range(3):
                if kk == 0:
                    Y = cur
                else:
                    Y = np.vstack([cur] + new_vs)
                v_new = _optimize_single(Y, rng, n_rand=3000, refine_steps=400)
                new_vs.append(v_new.reshape(1, -1))
            for idx, vn in zip(idxs, new_vs):
                xt[idx] = vn.ravel()
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)
            xt = _smooth(xt, 600, 50, 600, 0.025, 0.003)
            xt = _lp_polish(xt, 1.4, t_deadline=t0 + BUDGET - FINAL)
            vt = _max_cos(xt)
            if vt < best - 1e-12:
                best = vt
                x_best = xt.copy()

    # final squeeze
    x_best = _smooth(x_best, 300, 150, 900, 0.008, 0.002)
    x_best = _lp_polish(x_best, 10.0)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
