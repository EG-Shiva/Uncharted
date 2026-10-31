import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _row_maxs_counts(x, m=None, tol=1e-9):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    if m is None:
        m = float(g.max())
    rm = g.max(axis=1)
    cnt = np.count_nonzero(g >= m - tol, axis=1)
    return rm, cnt, g, m

def _smooth(x, iters=400, T0=0.02, T1=0.002, lr0=0.008, lr1=0.001):
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

_IU = None
def _lp_polish(x, seconds, radius0=2e-3):
    global _IU
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    if _IU is None or _IU[0].shape[0] != n * (n - 1) // 2:
        _IU = np.triu_indices(n, 1)
    iu = _IU
    best = _max_cos(x)
    radius = radius0
    start = time.time()
    nv = n * d + 1
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu]
        m = float(c_all.max())
        sel = c_all > m - max(20.0 * radius, 1e-9)
        # cap constraints
        nnz = int(np.count_nonzero(sel))
        if nnz > 2500:
            # keep top 2500
            idx = np.where(sel)[0]
            vals = c_all[idx]
            k = np.argpartition(vals, len(vals) - 2500)[-2500:]
            nsel = np.zeros_like(sel)
            nsel[idx[k]] = True
            sel = nsel
        I, J = iu[0][sel], iu[1][sel]
        c = c_all[sel]
        p = len(I)
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        valsM = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((valsM, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _void_point(existing, rng, ntry=3000):
    # propose point minimizing max dot to existing set
    cand = rng.normal(size=(ntry, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    D = cand @ existing.T
    np.fill_diagonal(D[:, :0], 0) if False else None
    md = D.max(axis=1)
    return cand[int(np.argmin(md))]

def solve():
    rng = np.random.default_rng(7)
    t0 = time.time()
    n, d = 163, 4
    # load starts
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        z = rng.normal(size=(n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    best = None
    best_val = 1e9
    for c in cands:
        v = _max_cos(c)
        if v < best_val:
            best_val = v
            best = c.copy()
    cur = best.copy()
    cur_val = best_val
    # quick initial LP to ensure local opt (short)
    # reserve time: total budget ~172s
    TIME_BUDGET = 172.0
    FINAL_POLISH = 25.0

    trial = 0
    sigmas = [0.01, 0.03, 0.07, 0.15, 0.02, 0.05, 0.10, 0.008]
    while time.time() - t0 < TIME_BUDGET - FINAL_POLISH and trial < 300:
        sigma = sigmas[trial % len(sigmas)]
        move = trial % 7
        y = cur.copy()
        try:
            rm, cnt, G, m = _row_maxs_counts(y)
        except Exception:
            rm = None
        if move == 0:
            y = y + rng.normal(scale=sigma, size=y.shape)
        elif move == 1:
            u = rng.normal(size=4); u /= np.linalg.norm(u)
            dots = y @ u
            mask = dots > 0.70
            if np.count_nonzero(mask) < 3:
                idx = np.argsort(dots)[-12:]
                mask = np.zeros(n, dtype=bool); mask[idx] = True
            nm = int(np.count_nonzero(mask))
            y[mask] += rng.normal(scale=sigma * 2.0, size=(nm, d))
        elif move == 2:
            k = 5 + int(rng.integers(0, 6))  # 5-10
            order = np.argsort(rm)[-k:]
            for ii in order:
                y[ii] += rng.normal(scale=sigma * 1.5, size=(d,))
            # also shake neighbours of worst pair
            if rng.random() < 0.5:
                # find max pair
                G2 = y @ y.T; np.fill_diagonal(G2, -2.0)
                a, b = np.unravel_index(int(np.argmax(G2)), G2.shape)
                y[a] += rng.normal(scale=sigma, size=(d,))
                y[b] += rng.normal(scale=sigma, size=(d,))
        elif move == 3:
            # rattler / low-contact resample via void
            low = np.where(cnt <= 1)[0]
            if len(low) == 0:
                low = np.argsort(cnt)[:3]
            nr = min(len(low), 1 + int(rng.integers(0, 3)))  # 1-3
            picks = rng.choice(low, size=nr, replace=False) if len(low) >= nr else low
            for ii in picks:
                rest = np.delete(y, ii, axis=0)
                rest /= np.linalg.norm(rest, axis=1, keepdims=True)
                y[ii] = _void_point(rest, rng, ntry=2000)
        elif move == 4:
            # kill worst point, reinsert at void
            w = int(np.argmax(rm))
            rest = np.delete(y, w, axis=0)
            y[w] = _void_point(rest, rng, ntry=3000)
            # small jitter others
            y += rng.normal(scale=0.003, size=y.shape)
        elif move == 5:
            # random subset shake large
            k = 4 + int(rng.integers(0, 7))  # 4-10
            idx = rng.choice(n, size=k, replace=False)
            y[idx] += rng.normal(scale=sigma * 1.2, size=(k, d))
        else:
            # global moderate + one void reinsert
            y = y + rng.normal(scale=max(sigma * 0.5, 0.01), size=y.shape)
            w = int(np.argmax(rm))
            # with prob replace worst by random
            if rng.random() < 0.5:
                v = rng.normal(size=(4,)); v /= np.linalg.norm(v)
                y[w] = v
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        # smooth
        iters = 350 if move in (3, 4) else 300
        y = _smooth(y, iters=iters, T0=0.02, T1=0.0015, lr0=0.008, lr1=0.0008)
        vs = _max_cos(y)
        # screen: skip LP if far worse than cur
        if vs > cur_val + 0.015 and vs > best_val + 0.005:
            trial += 1
            # occasional random restart of cur to best
            if trial % 25 == 0:
                cur = best.copy(); cur_val = best_val
            continue
        # short LP
        remain = TIME_BUDGET - FINAL_POLISH - (time.time() - t0)
        if remain < 0.5:
            break
        lp_sec = 1.2 if vs < best_val + 0.002 else 0.7
        lp_sec = min(lp_sec, remain)
        y = _lp_polish(y, seconds=lp_sec, radius0=2e-3)
        v = _max_cos(y)
        if v < best_val:
            best, best_val = y.copy(), v
            cur, cur_val = y.copy(), v
        else:
            # basin acceptance: allow sideways / slightly worse to explore
            temp = 0.0015
            if v < cur_val or rng.random() < np.exp((cur_val - v) / temp):
                cur, cur_val = y.copy(), v
            # periodically reset to best
            if trial % 30 == 29:
                # return to best with kick next iter; keep cur=best if stuck
                if cur_val > best_val + 0.002:
                    cur, cur_val = best.copy(), best_val
        trial += 1
    # final thorough polish of best
    remain = TIME_BUDGET - (time.time() - t0)
    if remain > 1.0:
        # alternate smooth + LP
        best = _smooth(best, iters=600, T0=0.008, T1=0.0008, lr0=0.004, lr1=0.0005)
        best = _lp_polish(best, seconds=max(remain - 0.5, 1.0), radius0=1e-3)
        v = _max_cos(best)
        if v < best_val:
            best_val = v
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
