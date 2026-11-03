import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def _lp_polish(x, seconds, r0=1.5e-3, cap=2500):
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
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _fast_polish(x, steps):
    n, d = x.shape
    v = np.zeros_like(x)
    mu = 0.85
    for s in range(steps):
        frac = s / max(1, steps)
        alpha = 15.0 + 85.0 * frac
        lr = 0.02 * (1.0 - frac) + 0.002
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        thr = m - 8.0 / alpha
        mask = G > thr
        W = np.zeros_like(G)
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
        v = v * sc
        x = x + v
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _refine_hole(X, h, steps=150):
    h = h / np.linalg.norm(h)
    for s in range(steps):
        dots = X @ h
        j = int(np.argmax(dots))
        dm = float(dots[j])
        gj = X[j] - dm * h
        lr = 0.1 * (1.0 - s / steps) + 0.01
        h = h - lr * gj
        nh = np.linalg.norm(h)
        if nh == 0 or not np.isfinite(nh):
            break
        h /= nh
    return h


def solve():
    np.random.seed(0)
    t0 = time.time()
    EXPLORE_END = t0 + 158.0
    rng = np.random.default_rng(20261103)
    try:
        x_best = np.load("best.npy").astype(float)
    except Exception:
        try:
            x_best = np.load("record.npy").astype(float)
        except Exception:
            x_best = rng.standard_normal((176, 9))
    x_best = np.asarray(x_best, dtype=float)
    if x_best.shape != (176, 9):
        x_best = rng.standard_normal((176, 9))
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    n, d = x_best.shape
    best_val = _max_cos(x_best)

    def worst_indices(xc, k):
        G = xc @ xc.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        row_max = np.max(G, axis=1)
        contacts = np.sum(G > m - 1e-9, axis=1)
        order = np.lexsort((contacts, row_max))
        return order[-k:], float(m), row_max, contacts, G

    it = 0
    while time.time() < EXPLORE_END:
        it += 1
        if time.time() >= EXPLORE_END:
            break
        mod = it % 8
        xc = x_best.copy()
        steps = 700
        if mod == 1:
            xc = xc + 0.008 * rng.standard_normal(xc.shape)
            xc /= np.linalg.norm(xc, axis=1, keepdims=True)
            steps = 500
        elif mod == 2:
            xc = xc + 0.028 * rng.standard_normal(xc.shape)
            xc /= np.linalg.norm(xc, axis=1, keepdims=True)
            steps = 700
        elif mod == 3:
            xc = xc + 0.075 * rng.standard_normal(xc.shape)
            xc /= np.linalg.norm(xc, axis=1, keepdims=True)
            steps = 1000
        elif mod == 4:
            # single worst random reinsert
            order, _, _, _, _ = worst_indices(xc, 10)
            idx = int(rng.choice(order[-5:]))
            rv = rng.standard_normal(d)
            rv /= np.linalg.norm(rv)
            xc[idx] = rv
            steps = 700
        elif mod == 5:
            # single worst hole reinsert (greedy void)
            order, _, _, _, _ = worst_indices(xc, 10)
            idx = int(rng.choice(order[-5:]))
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            Xr = xc[mask]
            S = rng.standard_normal((3000, d))
            S /= np.linalg.norm(S, axis=1, keepdims=True)
            dots = S @ Xr.T
            rowm = np.max(dots, axis=1)
            bi = int(np.argmin(rowm))
            h = _refine_hole(Xr, S[bi], steps=150)
            xc[idx] = h
            steps = 700
        elif mod == 6:
            # coherent cluster shift around overloaded hub
            order, m, row_max, contacts, G = worst_indices(xc, n)
            # hub: high contacts, tie-break by row_max
            hub = int(order[-1])
            if rng.random() < 0.3:
                hub = int(rng.choice(order[-8:]))
            Gi = G[hub].copy()
            Gi[hub] = -2.0
            k = int(rng.integers(6, 13))
            nbrs = np.argpartition(-Gi, k)[:k]
            cluster = np.concatenate(([hub], nbrs))
            common = 0.05 * rng.standard_normal(d)
            xc[cluster] = xc[cluster] + common + 0.018 * rng.standard_normal((len(cluster), d))
            xc[cluster] /= np.linalg.norm(xc[cluster], axis=1, keepdims=True)
            steps = 800
        else:  # mod 0 or 7: multi void reinsert / subset shake
            if (it // 8) % 2 == 0:
                # 3-point sequential void reinsert of worst points
                order, _, _, _, _ = worst_indices(xc, 12)
                sel = rng.choice(order[-8:], size=3, replace=False)
                mask = np.ones(n, dtype=bool)
                mask[sel] = False
                cur = xc[mask].copy()
                for j in sel:
                    T = rng.standard_normal((1500, d))
                    T /= np.linalg.norm(T, axis=1, keepdims=True)
                    mm = np.max(T @ cur.T, axis=1)
                    pick = int(np.argmin(mm))
                    h = _refine_hole(cur, T[pick], steps=100)
                    xc[j] = h
                    cur = np.vstack([cur, h[None, :]])
                steps = 900
            else:
                k = 12
                sel = rng.choice(n, size=k, replace=False)
                xc[sel] = xc[sel] + 0.05 * rng.standard_normal((k, d))
                xc[sel] /= np.linalg.norm(xc[sel], axis=1, keepdims=True)
                steps = 700

        xc = _fast_polish(xc, steps)
        val_fast = _max_cos(xc)
        if val_fast < best_val + 0.005:
            if time.time() >= EXPLORE_END:
                break
            sec = min(2.0, max(0.4, EXPLORE_END - time.time()))
            if sec < 0.4:
                break
            xr = _lp_polish(xc, seconds=sec, r0=1.5e-3)
            vr = _max_cos(xr)
            if vr < best_val - 1e-12:
                x_best = xr
                best_val = vr

    elapsed = time.time() - t0
    remain = 175.0 - elapsed
    if remain > 4.0:
        sec = min(remain - 1.5, 12.0)
        x_best = _lp_polish(x_best, seconds=sec, r0=1e-3)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
