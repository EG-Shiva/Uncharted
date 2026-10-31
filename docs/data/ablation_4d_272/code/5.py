import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds, r0=3e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = float(r0)
    start = time.time()
    iu = np.triu_indices(n, 1)
    # preallocate row pattern helpers
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel_idx = np.where(c_all > thr)[0]
        if sel_idx.size == 0:
            break
        if sel_idx.size > 3000:
            # keep worst 3000 only
            part = np.argpartition(c_all, -3000)[-3000:]
            sel_idx = part
        I = iu[0][sel_idx]
        J = iu[1][sel_idx]
        c_sel = c_all[sel_idx]
        p = sel_idx.size
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        # columns
        cI = (I[:, None] * d + ar[None, :]).ravel()
        cJ = (J[:, None] * d + ar[None, :]).ravel()
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        cols[0::(2 * d + 1)] = 0  # placeholder, will overwrite below
        # build cols blockwise
        cols = np.concatenate([
            (I[:, None] * d + ar[None, :]).ravel(),
        ])  # dummy to keep shape logic simple; rebuild correctly below
        # Correct construction:
        cols_blocks = np.empty((p, 2 * d + 1), dtype=np.int64)
        cols_blocks[:, :d] = I[:, None] * d + ar[None, :]
        cols_blocks[:, d:2 * d] = J[:, None] * d + ar[None, :]
        cols_blocks[:, 2 * d] = n * d
        cols = cols_blocks.ravel()
        vals_blocks = np.empty((p, 2 * d + 1), dtype=np.float64)
        vals_blocks[:, :d] = x[J]
        vals_blocks[:, d:2 * d] = x[I]
        vals_blocks[:, 2 * d] = -1.0
        vals = vals_blocks.ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best - 1e-15:
            x = y
            best = v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
            if radius < 1e-12:
                break
    return x


def anneal_relax(cur, iters=800, step0=0.02, T0=0.004, T1=0.0004):
    n, d = cur.shape
    cur = cur / np.linalg.norm(cur, axis=1, keepdims=True)
    best = max_cos(cur)
    best_x = cur.copy()
    step = float(step0)
    ratio = (T1 / T0) if T0 > 0 else 1.0
    for it in range(iters):
        frac = it / max(1, iters)
        T = T0 * (ratio ** frac) if T0 > 0 else T1
        if T < 1e-12:
            T = 1e-12
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        cutoff = m - 12.0 * T
        # keep window sane
        if cutoff < m - 0.06:
            cutoff = m - 0.06
        mask = G > cutoff
        # weights
        W = np.zeros_like(G)
        # exp only on masked entries
        Gm = G[mask]
        W[mask] = np.exp((Gm - m) / T)
        sumW = W.sum(axis=1, keepdims=True)
        S = W @ cur
        coeff = (W * G).sum(axis=1, keepdims=True)
        F = -S + cur * coeff
        F /= (sumW + 1e-12)
        cur = cur + step * F
        cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        if (it % 50 == 49) or (it == iters - 1):
            v = max_cos(cur)
            if v < best:
                best = v
                best_x = cur.copy()
        # slight step decay
        step *= 0.9995
    return cur, best_x, best


def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline = t0 + BUDGET
    rng = np.random.default_rng(12345)
    # load start
    try:
        xb = np.load("best.npy").astype(np.float64)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(np.float64)
    except Exception:
        xr = None
    cands = []
    if xb is not None and xb.shape == (272, 4):
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        cands.append(xb)
    if xr is not None and xr.shape == (272, 4):
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        cands.append(xr)
    if len(cands) == 0:
        x0 = rng.standard_normal((272, 4))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        cands.append(x0)
    best_val = 1e9
    x_best = cands[0]
    for c in cands:
        v = max_cos(c)
        if v < best_val:
            best_val = v
            x_best = c.copy()
    n, d = 272, 4
    # quick contact info helper inline in loop
    trial = 0
    # reserve ~28s for final polish
    max_trials = 80
    for trial in range(1, max_trials + 1):
        if time.time() > deadline - 28:
            break
        mode = trial % 9
        y = x_best.copy()
        if mode == 0:
            sig = 0.008
            y = y + sig * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            iters, s0, T0, T1 = 800, 0.02, 0.004, 0.0004
        elif mode == 1:
            sig = 0.016
            y = y + sig * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            iters, s0, T0, T1 = 900, 0.025, 0.005, 0.0004
        elif mode == 2:
            k = 1
            idx = rng.choice(n, size=k, replace=False)
            y[idx] = rng.standard_normal((k, 4))
            y = y + 0.003 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            iters, s0, T0, T1 = 800, 0.02, 0.004, 0.0003
        elif mode == 3:
            k = 2
            idx = rng.choice(n, size=k, replace=False)
            y[idx] = rng.standard_normal((k, 4))
            y = y + 0.004 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            iters, s0, T0, T1 = 800, 0.02, 0.004, 0.0003
        elif mode == 4:
            k = 5
            idx = rng.choice(n, size=k, replace=False)
            y[idx] = rng.standard_normal((k, 4))
            y = y + 0.003 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            iters, s0, T0, T1 = 1000, 0.022, 0.005, 0.0004
        elif mode == 5:
            # rattler / low-degree replacement
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            deg = (G > m - 1e-4).sum(axis=1)
            low = np.where(deg <= 1)[0]
            if low.size == 0:
                low = np.argsort(deg)[:3]
            # replace up to 6 lowest
            take = low[:6] if low.size > 6 else low
            y[take] = rng.standard_normal((take.size, 4))
            y = y + 0.004 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            iters, s0, T0, T1 = 800, 0.02, 0.004, 0.0003
        elif mode == 6:
            # worst-pair push apart
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            # find few worst pairs
            iu = np.triu_indices(n, 1)
            vals = G[iu]
            k = 4
            top = np.argpartition(vals, -k)[-k:]
            alpha = 0.06 + 0.02 * rng.random()
            for t in top:
                a = int(iu[0][t]); b = int(iu[1][t])
                gab = float(G[a, b])
                # tangential away directions
                da = gab * y[a] - y[b]
                db = gab * y[b] - y[a]
                y[a] = y[a] + alpha * da
                y[b] = y[b] + alpha * db
            y = y + 0.005 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            iters, s0, T0, T1 = 700, 0.018, 0.003, 0.0003
        elif mode == 7:
            k = 10
            idx = rng.choice(n, size=k, replace=False)
            y[idx] = rng.standard_normal((k, 4))
            y = y + 0.005 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            iters, s0, T0, T1 = 1100, 0.025, 0.006, 0.0005
        else:
            # high-degree shake
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            deg = (G > m - 2e-4).sum(axis=1)
            hub = int(np.argmax(deg))
            # neighbors of hub
            nbrs = np.where(G[hub] > m - 0.01)[0]
            y[hub] = rng.standard_normal(4)
            if nbrs.size > 0:
                take = nbrs[:8]
                y[take] = y[take] + 0.03 * rng.standard_normal((take.size, 4))
            y = y + 0.004 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            iters, s0, T0, T1 = 900, 0.022, 0.005, 0.0004
        cur, best_path, v_path = anneal_relax(y, iters=iters, step0=s0, T0=T0, T1=T1)
        v_cur = max_cos(cur)
        # screen: if endpoint too bad, use best along path if promising
        polish_start = cur
        if v_cur > best_val + 0.015:
            if v_path < best_val + 0.005:
                polish_start = best_path
            else:
                continue
        else:
            # prefer endpoint for new basin; but if path best much better, polish that too quickly?
            # polish endpoint
            pass
        remain = deadline - 28 - time.time()
        if remain < 1.0:
            break
        secs = 1.6 if (trial % 3 != 0) else 2.5
        secs = min(secs, remain)
        z = lp_polish(polish_start, seconds=secs, r0=3e-3)
        v = max_cos(z)
        if v < best_val - 1e-12:
            x_best = z
            best_val = v
        # also opportunistic: if path best beats best, polish it
        if v_path < best_val - 1e-12:
            z2 = lp_polish(best_path, seconds=min(1.2, remain), r0=2e-3)
            v2 = max_cos(z2)
            if v2 < best_val - 1e-12:
                x_best = z2
                best_val = v2
    # final long polish of best
    rem = deadline - time.time() - 1.0
    if rem > 3:
        # a touch of anneal at low T to settle rattlers then LP
        try:
            cur, best_path, v_path = anneal_relax(x_best, iters=400, step0=0.008, T0=0.001, T1=0.0002)
            # choose better of cur/path/best for final LP start
            vc = max_cos(cur)
            if v_path < min(vc, best_val):
                x_best = best_path
                best_val = v_path
            elif vc < best_val:
                x_best = cur
                best_val = vc
        except Exception:
            pass
        rem2 = deadline - time.time() - 0.5
        if rem2 > 2:
            x_best = lp_polish(x_best, seconds=rem2, r0=4e-3)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # sanitize
    x_best = np.nan_to_num(x_best, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(x_best, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    x_best = x_best / norms
    return x_best
