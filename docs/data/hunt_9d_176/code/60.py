import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def _quench(x, steps, a0=12.0, a1=280.0, lr0=0.02, lr1=0.002, mu=0.85):
    n, d = x.shape
    v = np.zeros_like(x)
    for s in range(steps):
        frac = s / max(1, steps - 1) if steps > 1 else 1.0
        alpha = a0 * ((a1 / a0) ** frac) if a0 > 0 else a1
        lr = lr0 * (1.0 - frac) + lr1 * frac
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        thr = m - 8.0 / alpha
        mask = G > thr
        W = np.zeros_like(G)
        # exp in [-8,0] safe
        W[mask] = np.exp(alpha * (G[mask] - m))
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        tang = grad - dot * x
        tn = np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12
        direction = tang / tn
        v = mu * v - lr * direction
        vn = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        maxv = 3.0 * lr
        sc = np.minimum(1.0, maxv / vn)
        v *= sc
        x = x + v
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _lp_polish(x, seconds, r0=1.2e-3, cap=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    t_end = time.time() + seconds
    radius = float(r0)
    iu0 = np.triu_indices(n, 1)
    while time.time() < t_end and radius > 1e-14:
        G = x @ x.T
        c = G[iu0]
        m = float(np.max(c))
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > cap:
            idx_all = np.where(sel)[0]
            ca = c[idx_all]
            order = np.argpartition(-ca, cap)[:cap]
            keep = np.zeros_like(sel)
            keep[idx_all[order]] = True
            sel = keep
            cnt = cap
        I = iu0[0][sel]
        J = iu0[1][sel]
        cs = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        base_I = I[:, None] * d + np.arange(d)
        base_J = J[:, None] * d + np.arange(d)
        cols = np.concatenate([base_I, base_J, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _worst_order(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    row_max = np.max(G, axis=1)
    contacts = np.sum(G > m - 1e-9, axis=1)
    order = np.lexsort((contacts, row_max))
    return order, m, row_max, contacts, G


def solve():
    np.random.seed(0)
    t0 = time.time()
    EXPLORE_END = t0 + 162.0
    HARD_END = t0 + 176.0
    rng = np.random.default_rng(987654321)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal((176, 9))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (176, 9):
        xb = rng.standard_normal((176, 9))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = 176, 9
    best_val = _max_cos(xb)

    it = 0
    while time.time() < EXPLORE_END:
        it += 1
        if time.time() >= EXPLORE_END:
            break
        xc = xb.copy()
        r = rng.random()
        steps = 500
        if r < 0.28:
            # global jitter, new basin probe
            sig = float(rng.uniform(0.03, 0.15))
            xc = xc + sig * rng.standard_normal(xc.shape)
            xc /= np.linalg.norm(xc, axis=1, keepdims=True)
            steps = 450
        elif r < 0.55:
            # hub-targeted subset shake: change topology locally
            order, m, row_max, contacts, G = _worst_order(xc)
            k = int(rng.integers(8, 25))
            # bias to worst half
            pool = order[n // 2:]
            sel = rng.choice(pool, size=k, replace=False)
            sig = float(rng.uniform(0.20, 0.55))
            xc[sel] = xc[sel] + sig * rng.standard_normal((k, d))
            xc[sel] /= np.linalg.norm(xc[sel], axis=1, keepdims=True)
            # tiny global to break symmetry
            xc = xc + 0.005 * rng.standard_normal(xc.shape)
            xc /= np.linalg.norm(xc, axis=1, keepdims=True)
            steps = 550
        elif r < 0.78:
            # worst-point void reinsertion, cheap screening
            order, m, row_max, contacts, G = _worst_order(xc)
            k = int(rng.integers(2, 5))
            sel = rng.choice(order[-10:], size=k, replace=False)
            mask = np.ones(n, dtype=bool)
            mask[sel] = False
            cur = xc[mask].copy()
            for j in sel:
                T = rng.standard_normal((2048, d))
                T /= np.linalg.norm(T, axis=1, keepdims=True)
                mm = np.max(T @ cur.T, axis=1)
                pick = int(np.argmin(mm))
                xc[j] = T[pick]
                cur = np.vstack([cur, xc[j][None, :]])
            steps = 550
        elif r < 0.90:
            # large escape kick
            sig = float(rng.uniform(0.20, 0.35))
            xc = xc + sig * rng.standard_normal(xc.shape)
            xc /= np.linalg.norm(xc, axis=1, keepdims=True)
            steps = 650
        else:
            # fresh random start for regularity (isotropic basin)
            xc = rng.standard_normal((n, d))
            xc /= np.linalg.norm(xc, axis=1, keepdims=True)
            steps = 800

        # annealed quench, no LP inside loop
        xc = _quench(xc, steps)
        v = _max_cos(xc)
        if v < best_val - 1e-12:
            xb = xc
            best_val = v
        # time guard
        if time.time() >= EXPLORE_END:
            break

    remain = HARD_END - time.time()
    if remain > 5.0:
        sec = min(remain - 1.5, 11.0)
        if sec > 1.0:
            xb = _lp_polish(xb, seconds=sec)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    return xb
