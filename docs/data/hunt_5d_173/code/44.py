import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _hard_stats(x, T):
    g = x @ x.T
    iu = np.triu_indices(x.shape[0], 1)
    c = g[iu]
    m = float(c.max())
    d = c - T
    pos = d > 0
    V = float(d[pos].sum()) if np.any(pos) else 0.0
    C = int(pos.sum())
    return V, C, m

def _adam_block(x, T, alpha, steps, lr0, lr1, b1=0.9, b2=0.999, eps=1e-8, deadline=None):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1p = 1.0
    b2p = 1.0
    # precompute lr schedule
    if steps <= 1:
        lrs = np.array([lr0])
    else:
        # exponential decay lr0 -> lr1
        lrs = lr0 * np.power(lr1 / lr0, np.arange(steps) / float(steps - 1))
    for s in range(steps):
        if deadline is not None and (s & 63) == 0 and time.time() > deadline:
            break
        G = x @ x.T
        Z = alpha * (G - T)
        # stable sigmoid via clipping
        np.clip(Z, -30.0, 30.0, out=Z)
        # sigmoid
        # 1/(1+exp(-Z))
        np.negative(Z, out=Z)
        np.exp(Z, out=Z)
        Z += 1.0
        np.reciprocal(Z, out=Z)
        W = Z
        np.fill_diagonal(W, 0.0)
        Ge = W @ x
        dot = (Ge * x).sum(axis=1, keepdims=True)
        Gr = Ge - dot * x
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1.0 - b1) * Gr
        v *= b2
        v += (1.0 - b2) * (Gr * Gr)
        mhat = m / (1.0 - b1p)
        vhat = v / (1.0 - b2p)
        stepd = mhat / (np.sqrt(vhat) + eps)
        # project to tangent
        nd = (stepd * x).sum(axis=1, keepdims=True)
        stepd -= nd * x
        lr = float(lrs[s])
        x = x - lr * stepd
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    deadline = t0 + 170.0
    search_end = t0 + 128.0
    rng = np.random.RandomState(0)
    n, d = 173, 5
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    base = None
    for c in (xb, xr):
        if c is not None and c.shape == (n, d):
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            if base is None:
                base = c.copy()
    if base is None:
        base = rng.standard_normal((n, d))
        base /= np.linalg.norm(base, axis=1, keepdims=True)
    # normalize candidates, pick best as fallback
    cands0 = []
    if xb is not None and xb.shape == (n, d):
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        cands0.append(xb)
    if xr is not None and xr.shape == (n, d):
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        cands0.append(xr)
    cands0.append(base)
    best = base.copy()
    best_val = _max_cos(best)
    for c in cands0:
        v = _max_cos(c)
        if v < best_val:
            best_val = v
            best = c.copy()
    record_x = best.copy()
    record_val = best_val

    # build starts: record, perturbed copies, random
    starts = []
    starts.append(record_x.copy())
    for sig in (0.12, 0.28, 0.45, 0.70):
        p = record_x + sig * rng.standard_normal((n, d))
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        starts.append(p)
    for _ in range(2):
        q = rng.standard_normal((n, d))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        starts.append(q)

    Ts = [0.757, 0.754, 0.751, 0.748]
    # alpha schedule 200 -> 3000
    alphas = [200.0, 800.0, 2500.0]
    # lr per alpha
    lr_pairs = [(0.020, 0.005), (0.015, 0.002), (0.010, 0.001)]
    steps_per = [380, 380, 420]

    global_best = best.copy()
    global_val = best_val
    # track violation best per T for retention logic
    for si, x0 in enumerate(starts):
        if time.time() > search_end:
            break
        X = x0.copy()
        # vary steps slightly per start for diversity
        for ti, T in enumerate(Ts):
            if time.time() > search_end:
                break
            for ai, alpha in enumerate(alphas):
                if time.time() > search_end:
                    break
                lr0, lr1 = lr_pairs[ai]
                # small deterministic jitter per start
                jit = 1.0 + 0.15 * np.sin(si * 3.7 + ti * 1.3 + ai * 0.7)
                steps = int(steps_per[ai] * (0.85 + 0.15 * ((si + ti + ai) % 3) / 2.0))
                if search_end - time.time() < 4.0:
                    steps = min(steps, 150)
                X = _adam_block(X, T, alpha, steps, lr0 * jit, lr1 * jit, deadline=search_end)
            # evaluate after finishing this T (using last alpha's T)
            V, C, M = _hard_stats(X, T)
            # retain by violation/count is implicit: only polish if promising
            # update global by true max
            if M < global_val - 1e-12:
                global_val = M
                global_best = X.copy()
            # conditional LP polish: only if violation small or count changed substantially
            # threshold: V < 0.15 or C < 450 means near achieving T
            do_polish = (V < 0.15 or C < 450)
            # also polish if this trajectory produced new global best
            if do_polish and (time.time() < search_end - 3.0):
                # quick 1.5s polish to test convertibility
                try:
                    Xp = _lp_polish(X, min(1.5, max(0.5, search_end - time.time() - 1.0)))
                    Mp = _max_cos(Xp)
                    if Mp < global_val - 1e-12:
                        global_val = Mp
                        global_best = Xp.copy()
                    # continue from polished if it reduced violation too
                    # keep the better of X, Xp by (V primary, M secondary) for next T
                    Vp, Cp, _ = _hard_stats(Xp, T)
                    # retain trajectory according to violation/count, not max
                    if Vp < V or (Vp <= V and Mp <= M):
                        X = Xp
                except Exception:
                    pass
            # if violation huge and max much worse than record, abandon this start early
            # allow uphill: tolerate up to record+0.03
            if M > record_val + 0.035 and V > 1.0:
                break
        # end Ts loop
    # final selection: polish global best with remaining time
    remain = deadline - time.time() - 2.0
    if remain < 4.0:
        remain = 4.0
    if remain > 32.0:
        remain = 32.0
    try:
        polished = _lp_polish(global_best, remain)
        vp = _max_cos(polished)
        if vp < global_val:
            global_best = polished
            global_val = vp
        else:
            # keep original if polish failed to improve (avoid drift)
            pass
    except Exception:
        pass
    # safety: if we somehow got worse than record, return record
    if global_val > record_val and record_val < 2.0:
        # return the better one (record) to guarantee no regression
        global_best = record_x.copy()
    global_best = global_best / np.linalg.norm(global_best, axis=1, keepdims=True)
    # final NaN guard
    if not np.all(np.isfinite(global_best)):
        global_best = record_x.copy()
    return global_best
