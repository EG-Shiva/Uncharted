import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def row_info(x, m=None):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    if m is None:
        m = float(G.max())
    rm = G.max(axis=1)
    return rm, G, m

def lp_polish(x, seconds, cap_p=2200):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu]
        m = float(c_all.max())
        sel = c_all > m - max(20.0 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt > cap_p:
            # keep largest cap_p
            thr = np.partition(c_all[sel], cnt - cap_p)[cnt - cap_p]
            sel = c_all >= thr
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
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
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
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def smooth(x, iters, T0, T1, lr0, lr1):
    n, d = x.shape
    for k in range(iters):
        frac = k / max(iters - 1, 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        rs = W.sum(axis=1, keepdims=True)
        step = gt / np.maximum(rs, 1e-12)
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        step = step / np.maximum(sn, 1.0)
        x = x - lr * step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def find_holes(x, k, rng, n_probe=12000):
    P = rng.normal(size=(n_probe, 4))
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    # max cosine to set (closest point)
    S = P @ x.T
    s = S.max(axis=1)
    order = np.argsort(s)  # smallest first = largest hole
    holes = []
    for idx in order:
        cand = P[idx]
        ok = True
        for h in holes:
            if float(cand @ h) > 0.85:
                ok = False
                break
        if ok:
            holes.append(cand)
            if len(holes) >= k:
                break
    if len(holes) < k:
        for idx in order:
            holes.append(P[idx])
            if len(holes) >= k:
                break
    return np.array(holes)

def solve():
    rng = np.random.default_rng(7)
    t0 = time.time()
    LIMIT = 172.0
    # load start
    try:
        xb = np.asarray(np.load("best.npy"), dtype=float)
    except Exception:
        xb = None
    try:
        xr = np.asarray(np.load("record.npy"), dtype=float)
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        if c is not None and c.shape == (163, 4):
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            cands.append(c)
    if not cands:
        z = rng.normal(size=(163, 4))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    best = min(cands, key=max_cos).copy()
    best_val = max_cos(best)
    n, d = 163, 4

    # quick initial LP squeeze of start (short)
    if time.time() - t0 < LIMIT - 20:
        best = lp_polish(best, 4.0)
        best_val = max_cos(best)

    sigmas = [0.008, 0.02, 0.05, 0.10, 0.03, 0.15, 0.012, 0.06]
    trial = 0
    # reserve ~25s for final polish
    while time.time() - t0 < LIMIT - 25 and trial < 80:
        sigma = sigmas[trial % len(sigmas)]
        mode = trial % 5
        y = best.copy()
        rm, G, m = row_info(y)
        # contacts with tight tol for counting
        tight = 1e-4
        contacts = np.count_nonzero(G > m - tight, axis=1)
        if mode == 0:
            # global shake
            y = y + rng.normal(scale=sigma, size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif mode == 1:
            # worst-point shake
            k = 8 if trial % 2 == 0 else 16
            idx = np.argsort(rm)[-k:]
            y[idx] += rng.normal(scale=sigma * 2.5, size=(k, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif mode == 2:
            # hole re-insertion: replace worst points by holes
            k = 3 + (trial % 3)  # 3..5
            idx = np.argsort(rm)[-k:]
            # also include rattlers (few contacts) sometimes
            if trial % 2 == 1:
                idx2 = np.argsort(contacts)[:2]
                idx = np.unique(np.concatenate([idx, idx2]))[:k+2]
                k = len(idx)
            holes = find_holes(y, k, rng, n_probe=12000)
            y[idx[:k]] = holes[:k] + rng.normal(scale=0.005, size=(k, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif mode == 3:
            # cap-localized shake around random direction
            u = rng.normal(size=4); u /= np.linalg.norm(u)
            dots = y @ u
            mask = dots > 0.72
            if np.count_nonzero(mask) < 4:
                idxc = np.argsort(dots)[-12:]
                mask = np.zeros(n, dtype=bool); mask[idxc] = True
            nm = int(np.count_nonzero(mask))
            y[mask] += rng.normal(scale=sigma * 2.0, size=(nm, d))
            # tiny global jitter to break symmetry
            y += rng.normal(scale=0.002, size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        else:
            # overloaded neighbourhood shake + small global
            # overloaded = many neighbours within m-0.005
            nn = np.count_nonzero(G > m - 0.005, axis=1)
            order = np.argsort(nn)
            idx = order[-10:]
            y[idx] += rng.normal(scale=sigma * 2.0, size=(10, d))
            y += rng.normal(scale=sigma * 0.3, size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)

        # fast annealed smoothing (screening)
        y = smooth(y, iters=450, T0=0.015, T1=0.0025, lr0=0.010, lr1=0.002)
        v = max_cos(y)
        # only LP if promising (within 0.01) to save time
        if v < best_val + 0.008 and time.time() - t0 < LIMIT - 25:
            # a bit more smoothing at low T for good candidates
            if v < best_val + 0.002:
                y = smooth(y, iters=250, T0=0.003, T1=0.0008, lr0=0.004, lr1=0.001)
            y2 = lp_polish(y, 2.5 if v > best_val else 4.0)
            v2 = max_cos(y2)
            if v2 < best_val - 1e-12:
                best, best_val = y2, v2
            elif v2 < best_val and time.time() - t0 > 20:
                best, best_val = y2, v2
        trial += 1

    # final intensive polish of best
    remain = LIMIT - (time.time() - t0) - 1.0
    if remain > 3:
        # alternate smooth + LP
        best = smooth(best, iters=600, T0=0.004, T1=0.0007, lr0=0.004, lr1=0.0008)
        best = lp_polish(best, min(remain * 0.8, 20.0))
        if remain > 22:
            best = smooth(best, iters=400, T0=0.0015, T1=0.0004, lr0=0.002, lr1=0.0005)
            best = lp_polish(best, remain - 2.0)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
