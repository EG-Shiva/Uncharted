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
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel_idx = np.where(c_all > thr)[0]
        if sel_idx.size == 0:
            break
        if sel_idx.size > 3500:
            part = np.argpartition(c_all, -3500)[-3500:]
            sel_idx = part
        I = iu[0][sel_idx]
        J = iu[1][sel_idx]
        c_sel = c_all[sel_idx]
        p = sel_idx.size
        nv = n * d + 1
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


def smooth_relax(cur, iters=700, step0=0.022, T0=0.004, T1=0.0003, mu=0.82):
    n, d = cur.shape
    cur = cur / np.linalg.norm(cur, axis=1, keepdims=True)
    V = np.zeros_like(cur)
    # initial best from current
    G0 = cur @ cur.T
    np.fill_diagonal(G0, -2.0)
    best = float(G0.max())
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
        if cutoff < m - 0.08:
            cutoff = m - 0.08
        mask = G > cutoff
        W = np.zeros_like(G)
        Gm = G[mask]
        if Gm.size > 0:
            W[mask] = np.exp((Gm - m) / T)
        sumW = W.sum(axis=1, keepdims=True)
        zm = sumW[:, 0] < 1e-12
        if np.any(zm):
            zidx = np.where(zm)[0]
            amax = np.argmax(G[zidx], axis=1)
            W[zidx, amax] = 1.0
            sumW[zidx, 0] = 1.0
        S = W @ cur
        coeff = (W * G).sum(axis=1, keepdims=True)
        F = (-S + cur * coeff) / (sumW + 1e-12)
        V = mu * V + step * F
        nd = (V * cur).sum(axis=1, keepdims=True)
        V = V - nd * cur
        vn = np.sqrt((V * V).sum(axis=1, keepdims=True))
        scale = np.minimum(1.0, 0.025 / (vn + 1e-12))
        V = V * scale
        cur = cur + V
        cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        step *= 0.9997
        # track using m of next iter? use current m (before move) + periodic true eval
        if m < best:
            best = m
            best_x = cur.copy()
        if (it % 50 == 49) or (it == iters - 1):
            # true eval of current position (m corresponds to previous pos)
            Gt = cur @ cur.T
            np.fill_diagonal(Gt, -2.0)
            vt = float(Gt.max())
            if vt < best:
                best = vt
                best_x = cur.copy()
    return cur, best_x, best


def void_point(rem, rng, n_samples=2500):
    cand = rng.standard_normal((n_samples, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    Gc = cand @ rem.T
    worst = Gc.max(axis=1)
    bi = int(np.argmin(worst))
    return cand[bi]


def solve():
    t0 = time.time()
    BUDGET = 171.0
    deadline = t0 + BUDGET
    rng = np.random.default_rng(7)
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
    x_best = cands[0].copy()
    for c in cands:
        v = max_cos(c)
        if v < best_val:
            best_val = v
            x_best = c.copy()
    n, d = 272, 4
    iu = np.triu_indices(n, 1)
    trial = 0
    while True:
        trial += 1
        if time.time() > deadline - 26:
            break
        mode = trial % 7
        y = x_best.copy()
        if mode == 0:
            sig = 0.006 + 0.004 * rng.random()
            y = y + sig * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            iters, s0, T0, T1, mu = 600, 0.020, 0.004, 0.0003, 0.82
        elif mode == 1:
            sig = 0.012 + 0.006 * rng.random()
            y = y + sig * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            iters, s0, T0, T1, mu = 700, 0.024, 0.005, 0.0004, 0.85
        elif mode == 2:
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            rowmax = G.max(axis=1)
            order = np.argsort(rowmax)
            k = 1 if (trial % 2 == 0) else 2
            rep = order[:k]
            rem = np.delete(y, rep, axis=0)
            for r in rep:
                nv = void_point(rem, rng, n_samples=2500)
                rem = np.vstack([rem, nv])
                y[r] = nv
            y = y + 0.003 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            iters, s0, T0, T1, mu = 700, 0.020, 0.004, 0.0003, 0.82
        elif mode == 3:
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            deg = (G > m - 1e-4).sum(axis=1)
            rowmax = G.max(axis=1)
            score = deg * 10.0 + (rowmax - m) * 100.0
            order = np.argsort(score)
            k = int(3 + trial % 3)  # 3..5
            rep = order[:k]
            rem = np.delete(y, rep, axis=0)
            for r in rep:
                nv = void_point(rem, rng, n_samples=2000)
                rem = np.vstack([rem, nv])
                y[r] = nv
            y = y + 0.0035 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            iters, s0, T0, T1, mu = 800, 0.022, 0.005, 0.0003, 0.84
        elif mode == 4:
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            vals = G[iu]
            k = 4
            top = np.argpartition(vals, -k)[-k:]
            alpha = 0.05 + 0.03 * rng.random()
            for t in top:
                a = int(iu[0][t]); b = int(iu[1][t])
                gab = float(G[a, b])
                da = gab * y[a] - y[b]
                db = gab * y[b] - y[a]
                y[a] = y[a] + alpha * da
                y[b] = y[b] + alpha * db
            y = y + 0.004 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            iters, s0, T0, T1, mu = 600, 0.018, 0.0035, 0.0003, 0.80
        elif mode == 5:
            k = int(4 + rng.integers(0, 3))  # 4..6
            idx = rng.choice(n, size=k, replace=False)
            rem = np.delete(y, idx, axis=0)
            for r in idx:
                nv = void_point(rem, rng, n_samples=2000)
                rem = np.vstack([rem, nv])
                y[r] = nv
            y = y + 0.004 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            iters, s0, T0, T1, mu = 900, 0.024, 0.0055, 0.0004, 0.85
        else:
            sig = 0.018 + 0.008 * rng.random()
            y = y + sig * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            iters, s0, T0, T1, mu = 900, 0.026, 0.006, 0.00045, 0.86
        try:
            cur, best_path, v_path = smooth_relax(y, iters=iters, step0=s0, T0=T0, T1=T1, mu=mu)
        except Exception:
            continue
        v_cur = max_cos(cur)
        if v_cur > best_val + 0.012 and v_path > best_val + 0.006:
            continue
        remain = deadline - 26 - time.time()
        if remain < 1.0:
            break
        secs = 1.5 if (trial % 3 != 0) else 2.4
        secs = min(secs, remain)
        polish_start = cur
        if v_path + 1e-9 < v_cur and v_path < best_val + 0.002 and rng.random() < 0.25:
            polish_start = best_path
        try:
            z = lp_polish(polish_start, seconds=secs, r0=3e-3)
        except Exception:
            continue
        v = max_cos(z)
        if v < best_val - 1e-12:
            x_best = z
            best_val = v
        if v_path < best_val - 1e-12:
            try:
                rem2 = deadline - 26 - time.time()
                if rem2 > 1.0:
                    z2 = lp_polish(best_path, seconds=min(1.1, rem2), r0=2e-3)
                    v2 = max_cos(z2)
                    if v2 < best_val - 1e-12:
                        x_best = z2
                        best_val = v2
            except Exception:
                pass
    # final settle
    rem = deadline - time.time() - 0.8
    if rem > 3:
        try:
            cur, best_path, v_path = smooth_relax(x_best, iters=350, step0=0.008, T0=0.001, T1=0.0002, mu=0.80)
            vc = max_cos(cur)
            cand_best = x_best; cand_val = best_val
            if vc < cand_val:
                cand_best = cur; cand_val = vc
            if v_path < cand_val:
                cand_best = best_path; cand_val = v_path
            x_best = cand_best; best_val = cand_val
        except Exception:
            pass
        rem2 = deadline - time.time() - 0.4
        if rem2 > 2:
            try:
                x_best = lp_polish(x_best, seconds=rem2, r0=4e-3)
            except Exception:
                pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    x_best = np.nan_to_num(x_best, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(x_best, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    x_best = x_best / norms
    return x_best
