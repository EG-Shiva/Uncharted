import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-20
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(x, seconds, r0=5e-3):
    x = _norm(np.asarray(x, dtype=float))
    n, d = x.shape
    best = _max_cos(x)
    radius = float(r0)
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu0]
        m = float(np.max(c))
        thresh = m - max(20.0 * radius, 1e-12)
        sel = c > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 800:
            idx = np.argsort(c)[-800:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
            cnt = 800
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        Ccol = np.empty((p, 2 * d + 1), dtype=int)
        Ccol[:, :d] = I[:, None] * d + np.arange(d)
        Ccol[:, d:2 * d] = J[:, None] * d + np.arange(d)
        Ccol[:, 2 * d] = n * d
        cols = Ccol.ravel()
        Arow = np.empty((p, 2 * d + 1), dtype=float)
        Arow[:, :d] = x[J]
        Arow[:, d:2 * d] = x[I]
        Arow[:, 2 * d] = -1.0
        vals = Arow.ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
            x = y
            best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return x

def _quench(x, steps, T0, T1, lr0, lr1, noise, rng):
    x = _norm(x)
    n, d = x.shape
    vel = np.zeros_like(x)
    mu = 0.85
    for k in range(steps):
        f = k / max(1, steps - 1)
        T = T0 + (T1 - T0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        # stable softmax
        W = np.exp((G - m) / max(T, 1e-12))
        np.fill_diagonal(W, 0.0)
        rs = np.sum(W, axis=1, keepdims=True)
        mean_rs = float(np.mean(rs)) + 1e-12
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        nrm = np.sqrt(np.sum(gt * gt, axis=1, keepdims=True)) + 1e-12
        w = np.clip(rs / mean_rs, 0.3, 3.0)
        direc = (gt / nrm) * (0.4 + 0.6 * w)
        vel = mu * vel + (1.0 - mu) * direc
        xn = x - lr * vel
        if noise > 0:
            amp = noise * (1.0 - f)
            if amp > 1e-13:
                z = rng.randn(n, d)
                zd = np.sum(z * x, axis=1, keepdims=True)
                z = z - zd * x
                zn = np.sqrt(np.sum(z * z, axis=1, keepdims=True)) + 1e-12
                z = (z / zn) * (amp * rng.randn(n, 1))
                xn = xn + z
        nn = np.sqrt(np.sum(xn * xn, axis=1, keepdims=True)) + 1e-20
        x = xn / nn
    return x

def _worst_idx(x, k, T=0.004):
    G = x @ x.T
    np.fill_diagonal(G, -np.inf)
    m = float(np.max(G))
    E = np.clip((G - m) / T, -20, 0)
    W = np.exp(E)
    np.fill_diagonal(W, 0.0)
    rs = np.sum(W, axis=1)
    idx = np.argsort(rs)[-k:]
    return idx[::-1].copy()

def _find_hole(pts, rng, ns=6000):
    d = pts.shape[1]
    S = rng.randn(ns, d)
    S = _norm(S)
    D = S @ pts.T
    md = np.max(D, axis=1)
    return S[int(np.argmin(md))].copy()

def solve():
    t0 = time.time()
    DEADLINE = 178.0
    RESERVE = 20.0
    rng = np.random.RandomState(0)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        try:
            xb = _norm(np.asarray(xb, dtype=float))
            if xb.shape == (54, 21):
                cands.append(xb)
        except Exception:
            pass
    if xr is not None:
        try:
            xr = _norm(np.asarray(xr, dtype=float))
            if xr.shape == (54, 21):
                cands.append(xr)
        except Exception:
            pass
    if len(cands) == 0:
        cands.append(_norm(rng.randn(54, 21)))
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best = cands[bi].copy()
    best_val = float(vals[bi])
    sigmas = [0.05, 0.12, 0.25, 0.08, 0.15, 0.04, 0.20, 0.06]
    trial = 0
    while True:
        now = time.time()
        if now - t0 > DEADLINE - RESERVE:
            break
        trial += 1
        typ = trial % 4
        cand = best.copy()
        if typ == 0:
            sigma = sigmas[(trial - 1) % len(sigmas)]
            cand = cand + sigma * rng.randn(54, 21)
            cand = _norm(cand)
            T0 = 0.03 if (trial % 2 == 0) else 0.02
            cand = _quench(cand, steps=1200, T0=T0, T1=0.002, lr0=0.04, lr1=0.004, noise=0.008, rng=rng)
            r0 = min(0.015, max(0.004, sigma * 0.3))
        elif typ == 1:
            wi = int(_worst_idx(best, 1)[0])
            keep = np.delete(best, wi, axis=0)
            hole = _find_hole(keep, rng, ns=8000)
            cand[wi] = hole + 0.005 * rng.randn(21)
            cand = _norm(cand)
            cand = cand + 0.01 * rng.randn(54, 21)
            cand = _norm(cand)
            cand = _quench(cand, steps=900, T0=0.015, T1=0.0015, lr0=0.03, lr1=0.003, noise=0.005, rng=rng)
            r0 = 0.008
        elif typ == 2:
            k = 5 if (trial % 8 == 2) else 3
            wis = _worst_idx(best, k)
            wis_set = set([int(v) for v in wis])
            keep_idx = [i for i in range(54) if i not in wis_set]
            cur = best[keep_idx].copy()
            for wi in wis:
                wi = int(wi)
                hole = _find_hole(cur, rng, ns=6000)
                cand[wi] = hole
                cur = np.vstack([cur, hole.reshape(1, -1)])
            cand = cand + 0.008 * rng.randn(54, 21)
            cand = _norm(cand)
            cand = _quench(cand, steps=1100, T0=0.025, T1=0.002, lr0=0.035, lr1=0.003, noise=0.007, rng=rng)
            r0 = 0.010
        else:
            k = 6
            wis = _worst_idx(best, k)
            cand[wis] = cand[wis] + 0.15 * rng.randn(k, 21)
            cand = cand + 0.015 * rng.randn(54, 21)
            cand = _norm(cand)
            cand = _quench(cand, steps=1000, T0=0.02, T1=0.002, lr0=0.03, lr1=0.003, noise=0.006, rng=rng)
            r0 = 0.008
        remain = DEADLINE - RESERVE - (time.time() - t0)
        if remain < 1.0:
            # no time for LP, just evaluate quenched
            v = _max_cos(_norm(cand))
            if v < best_val - 1e-12:
                best_val = v
                best = _norm(cand)
            continue
        budget = min(2.2, remain)
        cand = _lp_polish(cand, seconds=budget, r0=r0)
        v = _max_cos(cand)
        if v < best_val - 1e-12:
            best_val = v
            best = cand.copy()
    remain = DEADLINE - (time.time() - t0)
    if remain > 2.0:
        best = _lp_polish(best, seconds=max(2.0, remain - 1.0), r0=2e-3)
    best = _norm(best)
    return best
