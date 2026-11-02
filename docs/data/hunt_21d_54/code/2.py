import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, r0=5e-3):
    x = _norm(np.asarray(x, dtype=float))
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        # cap constraints for speed
        if np.count_nonzero(sel) > 900:
            idx = np.argsort(c)[-900:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        ci = (I[:, None] * d + np.arange(d)).ravel()
        cj = (J[:, None] * d + np.arange(d)).ravel()
        cols = np.empty(p * (2 * d + 1), dtype=int)
        cols[0::(2*d+1)] = 0  # placeholder, will overwrite vectorized below
        # build cols vectorized
        cols = np.concatenate([np.repeat(I * d, d) + np.tile(np.arange(d), p),
                               np.repeat(J * d, d) + np.tile(np.arange(d), p),
                               np.full(p, n * d)])
        # need interleaved per-row ordering to match rows: row k has I-block, J-block, t
        # our rows repeat each p times 2d+1, so build vals in same order
        vals = np.concatenate([x[J].ravel(), x[I].ravel(), -np.ones(p)] if False else [np.empty(0)])
        # rebuild correctly row-major:
        vals = np.empty(p * (2 * d + 1))
        vals[0::(2*d+1)] = 0  # dummy
        # simpler: loop-free via reshape
        Arow = np.empty((p, 2 * d + 1))
        # fill via broadcasting
        # x[J] shape (p,d), x[I] shape (p,d)
        Arow[:, :d] = x[J]
        Arow[:, d:2*d] = x[I]
        Arow[:, 2*d] = -1.0
        vals = Arow.ravel()
        Ccol = np.empty((p, 2 * d + 1), dtype=int)
        Ccol[:, :d] = I[:, None] * d + np.arange(d)
        Ccol[:, d:2*d] = J[:, None] * d + np.arange(d)
        Ccol[:, 2*d] = n * d
        cols = Ccol.ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
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

def _gd_repel(x, steps, T0=0.02, T1=0.002, lr0=0.03, lr1=0.003, rng=None):
    x = _norm(x.copy())
    n, d = x.shape
    for k in range(steps):
        frac = k / max(1, steps - 1)
        T = T0 * (1 - frac) + T1 * frac
        lr = lr0 * (1 - frac) + lr1 * frac
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = G.max()
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        # tangential projection
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        nr = np.linalg.norm(gt, axis=1, keepdims=True) + 1e-12
        gt = gt / nr
        # scale: less move for already good points? uniform
        x = x - lr * gt
        x = _norm(x)
    return x

def solve():
    import time as _t
    t0 = _t.time()
    DEADLINE = 178.0
    rng = np.random.RandomState(0)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.randn(54, 21)
    xb = _norm(np.asarray(xb, dtype=float))
    if xb.shape != (54, 21):
        # fix shape if needed by random
        tmp = rng.randn(54, 21)
        xb = _norm(tmp)
    best_val = _max_cos(xb)
    best = xb.copy()
    # quick contact info to pick victims later
    def contacts(x, tol=1e-7):
        G = x @ x.T
        np.fill_diagonal(G, -2)
        m = G.max()
        C = (G > m - tol).sum(axis=1)
        return C, m
    # trial loop
    sigmas = [0.06, 0.03, 0.015, 0.08, 0.025, 0.01, 0.05, 0.12]
    trial = 0
    # reserve ~25s for final polish
    while _t.time() - t0 < DEADLINE - 25:
        trial += 1
        mode = trial % 3
        sigma = sigmas[(trial - 1) % len(sigmas)]
        cand = best.copy()
        if mode == 0:
            # global shake + gd
            cand = cand + sigma * rng.randn(*cand.shape)
            cand = _norm(cand)
            cand = _gd_repel(cand, steps=900, T0=0.02, T1=0.003, lr0=0.04, lr1=0.004)
        elif mode == 1:
            # single-point big move of most jammed point
            C, _ = contacts(best)
            idx = int(np.argmax(C))
            cand[idx] = cand[idx] + (sigma * 2.0) * rng.randn(21)
            cand = _norm(cand)
            # small shake others
            cand = cand + (sigma * 0.25) * rng.randn(*cand.shape)
            cand = _norm(cand)
            cand = _gd_repel(cand, steps=700, T0=0.015, T1=0.002, lr0=0.03, lr1=0.003)
        else:
            # kill and fill worst point into largest hole
            C, _ = contacts(best)
            idx = int(np.argmax(C))
            # find hole: sample candidates minimizing max dot
            S = _norm(rng.randn(4000, 21))
            keep = np.delete(best, idx, axis=0)
            dots = S @ keep.T
            md = dots.max(axis=1)
            hole = S[int(np.argmin(md))]
            # blend with small noise
            cand[idx] = hole + 0.01 * rng.randn(21)
            cand = _norm(cand)
            # light gd with fewer steps, freeze others slightly? just gd all
            cand = _gd_repel(cand, steps=600, T0=0.01, T1=0.002, lr0=0.02, lr1=0.003)
        # short LP polish
        remain = DEADLINE - 25 - (_t.time() - t0)
        if remain < 1.0:
            break
        budget = min(3.5, remain)
        # larger initial radius for larger sigma
        r0 = min(0.02, max(0.003, sigma * 0.4))
        cand = _lp_polish(cand, seconds=budget, r0=r0)
        v = _max_cos(cand)
        if v < best_val - 1e-12:
            best_val = v
            best = cand.copy()
        # also try pure shaken LP without gd occasionally if stuck: already covered
        if _t.time() - t0 > DEADLINE - 26:
            break
    # final long polish of best
    remain = DEADLINE - (_t.time() - t0)
    if remain > 2:
        best = _lp_polish(best, seconds=max(2.0, remain - 1.0), r0=2e-3)
        best_val = _max_cos(best)
    best = _norm(best)
    return best
