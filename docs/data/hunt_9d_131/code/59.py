import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _per_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _smooth(x, steps, T0, T1, lr0, lr1, mom=0.85):
    n = x.shape[0]
    v = np.zeros_like(x)
    for k in range(steps):
        f = k / (steps - 1) if steps > 1 else 1.0
        T = T0 * (1 - f) + T1 * f
        lr = lr0 * (1 - f) + lr1 * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        s = float(np.sum(W)) / n + 1e-12
        grad = grad / s
        # per-row adaptive clip to keep rattlers alive
        gn = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-12
        # normalize median to 1 to avoid hub dominance
        med = float(np.median(gn)) + 1e-12
        grad = grad / (med * 8.0 + gn * 0.0)
        # actually use grad/s scaled: keep simple
        # revert scaling: use original grad/s but ensure finite step
        # recompute: use W-based grad normalized by s already; scale lr
        # undo above over-normalization: multiply back
        grad = grad * (med * 8.0)
        # the above two lines net to original grad/s; keep for stability when med tiny
        v = mom * v + (1.0 - mom) * grad
        x = x - lr * v
        x = _norm(x)
    return x

def _find_hole(Xk, rng, n_samples=15000):
    d = Xk.shape[1]
    Z = rng.randn(n_samples, d)
    Z = _norm(Z)
    D = Z @ Xk.T
    worst = D.max(axis=1)
    idx = int(np.argmin(worst))
    return Z[idx].copy(), float(worst[idx])

def _refine_hole(z, Xk, steps=120, T=0.008, lr=0.02):
    for _ in range(steps):
        dots = Xk @ z
        m = float(dots.max())
        W = np.exp((dots - m) / T)
        W = W / (W.sum() + 1e-12)
        g = W @ Xk
        g = g - float(np.dot(g, z)) * z
        z = z - lr * g
        nz = np.linalg.norm(z)
        if nz < 1e-12:
            break
        z = z / nz
    return z

def _greedy_block(F, k, rng):
    # sequentially place k points minimizing max dot to F + placed
    d = F.shape[1]
    placed = []
    cur = F
    for _ in range(k):
        Z = rng.randn(4000, d)
        Z = _norm(Z)
        D = Z @ cur.T
        worst = D.max(axis=1)
        # pick best few and refine best one
        idx = int(np.argmin(worst))
        z = Z[idx].copy()
        # local refine
        z = _refine_hole(z, cur, steps=80, T=0.008, lr=0.02)
        placed.append(z.copy())
        cur = np.vstack([cur, z[None, :]])
    return np.array(placed)

def _lp_polish(x, seconds, t_start):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() - t_start < seconds:
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu]
        m = float(c_all.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 4500:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:4500]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I = iu[0][sel]; J = iu[1][sel]; c = c_all[sel]
        p = len(I); nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
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
    return x

def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.RandomState(7)
    nT, dT = 131, 9
    best_x = None; best_s = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (nT, dT):
                a = _norm(a)
                s = _max_cos(a)
                if s < best_s:
                    best_s = s; best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _norm(rng.randn(nT, dT)); best_s = _max_cos(best_x)
    x_best = best_x.copy(); s_best = float(best_s)
    x_cur = best_x.copy(); s_cur = float(best_s)

    HOP_END = 132.0  # leave time for final anneal + LP
    trial = 0
    # temperature for wandering acceptance
    Tacc0 = 0.004
    while time.time() - t0 < HOP_END:
        trial += 1
        remain = HOP_END - (time.time() - t0)
        if remain < 2:
            break
        frac = 1.0 - remain / HOP_END  # 0->1
        Tacc = Tacc0 * (1 - frac * 0.9)
        mode = trial % 6
        try:
            if mode == 0:
                # global jitter from cur
                sig = 0.03 if (trial % 3) else 0.09
                if trial % 29 == 0:
                    sig = 0.22
                x_try = x_cur + sig * rng.randn(*x_cur.shape)
                x_try = _norm(x_try)
                x_try = _smooth(x_try, steps=220, T0=0.022, T1=0.006, lr0=0.014, lr1=0.0025)
                x_try = _smooth(x_try, steps=180, T0=0.006, T1=0.002, lr0=0.004, lr1=0.001)
            elif mode == 1:
                # cap resampling: random direction cap of size k
                k = int(rng.randint(8, 20))
                u = rng.randn(dT); u = u / (np.linalg.norm(u) + 1e-12)
                dots = x_cur @ u
                S = np.argsort(-dots)[:k]
                F = np.delete(x_cur, S, axis=0)
                block = _greedy_block(F, k, rng)
                x_try = x_cur.copy()
                x_try[S] = block + 0.01 * rng.randn(*block.shape)
                x_try = _norm(x_try)
                x_try = _smooth(x_try, steps=260, T0=0.025, T1=0.005, lr0=0.015, lr1=0.002)
                x_try = _smooth(x_try, steps=180, T0=0.005, T1=0.0018, lr0=0.004, lr1=0.001)
            elif mode == 2:
                # hub neighbourhood reshuffle
                G = x_cur @ x_cur.T
                np.fill_diagonal(G, -2.0)
                m = float(G.max())
                # contact count within 1e-7 of max? use threshold 2e-4 to find jammed
                thr = m - 5e-4
                cnt = np.sum(G > thr, axis=1)
                hub = int(np.argmax(cnt))
                # contacts of hub
                nbrs = np.where(G[hub] > thr)[0]
                if len(nbrs) > 12:
                    nbrs = nbrs[np.argsort(-G[hub][nbrs])[:12]]
                S = np.unique(np.concatenate([[hub], nbrs]))
                F = x_cur[np.array([i for i in range(nT) if i not in set(S.tolist())])]
                block = _greedy_block(F, len(S), rng)
                x_try = x_cur.copy()
                x_try[S] = block
                # jitter rest a bit
                x_try = x_try + 0.005 * rng.randn(*x_try.shape)
                x_try = _norm(x_try)
                x_try = _smooth(x_try, steps=260, T0=0.02, T1=0.005, lr0=0.013, lr1=0.002)
                x_try = _smooth(x_try, steps=180, T0=0.005, T1=0.002, lr0=0.004, lr1=0.001)
            elif mode == 3:
                # multi-hole: replace 3 worst by holes
                G = x_cur @ x_cur.T
                np.fill_diagonal(G, -2.0)
                pm = G.max(axis=1)
                worst = np.argsort(-pm)[:3]
                keep_mask = np.ones(nT, dtype=bool); keep_mask[worst] = False
                F = x_cur[keep_mask]
                newpts = []
                curF = F.copy()
                for _ in range(3):
                    z, _ = _find_hole(curF, rng, n_samples=12000)
                    z = _refine_hole(z, curF, steps=120, T=0.008, lr=0.02)
                    newpts.append(z); curF = np.vstack([curF, z[None, :]])
                x_try = x_cur.copy()
                x_try[worst] = np.array(newpts)
                x_try = x_try + 0.004 * rng.randn(*x_try.shape)
                x_try = _norm(x_try)
                x_try = _smooth(x_try, steps=240, T0=0.016, T1=0.004, lr0=0.012, lr1=0.002)
                x_try = _smooth(x_try, steps=180, T0=0.004, T1=0.0018, lr0=0.0035, lr1=0.001)
            elif mode == 4:
                # random subset shake (12 random points, large kick)
                k = int(rng.randint(10, 22))
                S = rng.choice(nT, size=k, replace=False)
                x_try = x_cur.copy()
                x_try[S] = x_try[S] + 0.35 * rng.randn(k, dT)
                x_try = _norm(x_try)
                x_try = _smooth(x_try, steps=260, T0=0.028, T1=0.006, lr0=0.016, lr1=0.0025)
                x_try = _smooth(x_try, steps=180, T0=0.006, T1=0.002, lr0=0.004, lr1=0.001)
            else:
                # antipodal-mirror cap: pick cap and reflect through random ortho to diversify
                k = int(rng.randint(6, 14))
                u = x_cur[int(rng.randint(nT))] + 0.3 * rng.randn(dT)
                u = u / (np.linalg.norm(u) + 1e-12)
                dots = x_cur @ u
                S = np.argsort(-dots)[:k]
                x_try = x_cur.copy()
                # random rotation of cap points: add orthogonal noise
                x_try[S] = _norm(x_try[S] + 0.18 * rng.randn(k, dT))
                x_try = _norm(x_try)
                x_try = _smooth(x_try, steps=240, T0=0.02, T1=0.005, lr0=0.014, lr1=0.002)
                x_try = _smooth(x_try, steps=160, T0=0.005, T1=0.002, lr0=0.0035, lr1=0.001)

            s = _max_cos(x_try)
            if s < s_best - 1e-12:
                s_best = s; x_best = x_try.copy()
                x_cur = x_try.copy(); s_cur = s
            elif s < s_cur:
                x_cur = x_try.copy(); s_cur = s
            else:
                # wandering acceptance to change basin
                if rng.rand() < np.exp(-(s - s_cur) / max(Tacc, 1e-6)):
                    x_cur = x_try.copy(); s_cur = s
                # occasionally restart from best
                if trial % 25 == 0 and rng.rand() < 0.5:
                    x_cur = x_best.copy(); s_cur = s_best
        except Exception:
            continue

    # final intensive smooth on best then LP
    try:
        x_best = _smooth(x_best, steps=300, T0=0.008, T1=0.0015, lr0=0.005, lr1=0.0008)
        sb = _max_cos(x_best)
        if sb < s_best:
            s_best = sb
    except Exception:
        pass
    remain = BUDGET - (time.time() - t0)
    if remain > 10.0:
        try:
            x_best = _lp_polish(x_best, seconds=remain - 2.0, t_start=time.time())
        except Exception:
            pass
    x_best = _norm(x_best)
    if not np.all(np.isfinite(x_best)):
        x_best = _norm(best_x)
    return x_best
