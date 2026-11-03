import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-300
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _global_smooth(X, steps, a0, a1, lr0, lr1):
    n, d = X.shape
    la0 = np.log(a0); la1 = np.log(a1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for t in range(steps):
        f = t / max(steps - 1, 1)
        alpha = np.exp(la0 + (la1 - la0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-30
        grad = W @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        X = X - lr * (gt / S)
        X = _norm(X)
    return X

def _local_smooth(X, idx, steps, a0, a1, lr0, lr1):
    if len(idx) == 0:
        return X
    idx = np.asarray(idx, dtype=np.int64)
    la0 = np.log(a0); la1 = np.log(a1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for t in range(steps):
        f = t / max(steps - 1, 1)
        alpha = np.exp(la0 + (la1 - la0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-30
        grad = W @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        step = lr * (gt[idx] / S[idx])
        X[idx] = _norm(X[idx] - step)
    return X

def _refine_hole(Xfix, y, steps=220, a0=60.0, a1=320.0, lr0=0.06, lr1=0.005):
    la0 = np.log(a0); la1 = np.log(a1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for t in range(steps):
        f = t / max(steps - 1, 1)
        alpha = np.exp(la0 + (la1 - la0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        s = Xfix @ y
        mx = float(np.max(s))
        w = np.exp(alpha * (s - mx))
        sw = float(np.sum(w)) + 1e-30
        g = w @ Xfix
        gt = g - float(np.dot(g, y)) * y
        y = y - lr * gt / sw
        nrm = np.linalg.norm(y)
        if nrm == 0 or not np.isfinite(nrm):
            break
        y = y / nrm
    return y

def _best_hole(Xcur, rng, n_sample=50000, chunk=10000):
    d = Xcur.shape[1]
    best_v = 2.0
    best_y = None
    done = 0
    while done < n_sample:
        bs = min(chunk, n_sample - done)
        Y = rng.standard_normal(size=(bs, d))
        Y = _norm(Y)
        S = Y @ Xcur.T
        m = S.max(axis=1)
        i = int(np.argmin(m))
        if float(m[i]) < best_v:
            best_v = float(m[i])
            best_y = Y[i].copy()
        done += bs
    return best_y, best_v

def _lp_polish(X, seconds, t0, deadline):
    X = _norm(X.copy())
    n, d = X.shape
    best = _max_cos(X)
    xb = X.copy()
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    start = time.time()
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-13:
            break
        if time.time() > deadline - 1.0:
            break
        G = X @ X.T
        cf = G[iu0]
        m = float(cf.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = cf > thr
        I = iu0[0][sel]; J = iu0[1][sel]; cs = cf[sel]
        p = len(I)
        if p == 0:
            break
        if p > 1600:
            idx = np.argpartition(-cs, 1600)[:1600]
            I, J, cs = I[idx], J[idx], cs[idx]
            p = 1600
        nv = n * d + 1
        rows = np.concatenate([np.repeat(np.arange(p), d), np.repeat(np.arange(p), d), np.arange(p)])
        cols = np.concatenate([np.repeat(I * d, d) + np.tile(ar, p),
                               np.repeat(J * d, d) + np.tile(ar, p),
                               np.full(p, n * d)])
        vals = np.concatenate([X[J].ravel(), X[I].ravel(), -np.ones(p)])
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm(X + res.x[:-1].reshape(n, d))
        v = _max_cos(y)
        if v < best:
            best = v; xb = y.copy(); X = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return xb, best

def solve():
    t0 = time.time()
    DEADLINE = t0 + 168.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = np.load("record.npy")
    xb = np.asarray(xb, dtype=np.float64).reshape(184, 8)
    xb = _norm(xb)
    gval = _max_cos(xb)
    gbest = xb.copy()

    # contact graph analysis
    G = xb @ xb.T
    np.fill_diagonal(G, -2.0)
    m0 = float(np.max(G))
    thr = m0 - 1e-7
    iu = np.triu_indices(184, 1)
    cf = (xb @ xb.T)[iu]
    sel = cf > thr
    I = iu[0][sel]; J = iu[1][sel]
    deg = np.zeros(184, dtype=np.int64)
    np.add.at(deg, I, 1)
    np.add.at(deg, J, 1)
    rattler = int(np.argmin(deg))
    # low-contact block (for info, not directly removed except rattler)
    # high pool >=22, extend to top degrees if too few
    high = list(np.where(deg >= 22)[0])
    order = np.argsort(-deg, kind="stable")
    if len(high) < 12:
        for o in order:
            if int(o) not in high and int(o) != rattler:
                high.append(int(o))
            if len(high) >= 14:
                break
    high = np.asarray(high, dtype=np.int64)
    # adjacency for greedy cover
    pairs_I = I; pairs_J = J; P = len(I)
    # incident list per point (boolean via indices)
    # precompute incident pair indices per high candidate
    inc = {}
    for h in high:
        inc[int(h)] = np.where((pairs_I == h) | (pairs_J == h))[0]

    Ks = [1, 2, 3, 4, 6, 8]
    cands = []
    for k in Ks:
        if time.time() > DEADLINE - 70:
            break
        # greedy selection of k-1 highs to maximize eliminated contacts
        chosen = []
        covered = np.zeros(P, dtype=bool)
        pool = [int(h) for h in high if int(h) != rattler]
        for _ in range(k - 1):
            best_gain = -1
            best_h = None
            for h in pool:
                if h in chosen:
                    continue
                idx = inc[h]
                if len(idx) == 0:
                    gain = 0
                else:
                    gain = int(np.sum(~covered[idx]))
                if gain > best_gain:
                    best_gain = gain
                    best_h = h
            if best_h is None:
                break
            chosen.append(best_h)
            idx = inc[best_h]
            if len(idx) > 0:
                covered[idx] = True
        R = [rattler] + chosen
        # ensure k distinct
        R = sorted(set(R))
        # if dedup shrank, fill with next best degrees
        while len(R) < k:
            for o in order:
                if int(o) not in R:
                    R.append(int(o))
                    break
        R = np.asarray(R[:k], dtype=np.int64)
        mask = np.ones(184, dtype=bool); mask[R] = False
        Xcur = xb[mask].copy()
        # reinsert one at a time
        for tins in range(k):
            if time.time() > DEADLINE - 65:
                break
            y0, _hv = _best_hole(Xcur, rng, n_sample=50000, chunk=10000)
            y0 = _refine_hole(Xcur, y0, steps=220)
            Xcur = np.vstack([Xcur, y0.reshape(1, -1)])
            Xcur = _norm(Xcur)
            # local subset: new point + neighbors with cosine > 0.35
            dots = (Xcur[:-1] @ Xcur[-1])
            neigh = list(np.where(dots > 0.35)[0])
            idx_opt = neigh + [Xcur.shape[0] - 1]
            Xcur = _local_smooth(Xcur, idx_opt, steps=160, a0=40.0, a1=280.0, lr0=0.02, lr1=0.003)
            Xcur = _global_smooth(Xcur, steps=150, a0=30.0, a1=220.0, lr0=0.015, lr1=0.002)
            if time.time() > DEADLINE - 65:
                break
        v = _max_cos(Xcur)
        cands.append((v, Xcur.copy()))
        if v < gval:
            gval = v; gbest = Xcur.copy()

    # reserve LP for best two surgical candidates
    cands.sort(key=lambda t: t[0])
    for ci in range(min(2, len(cands))):
        if time.time() > DEADLINE - 8:
            break
        v0, X0 = cands[ci]
        # only polish if promising (within 0.01 of best to save time)
        if v0 > gval + 0.01 and ci > 0:
            continue
        secs = min(22.0, DEADLINE - time.time() - 3.0)
        if secs < 2.0:
            break
        Xp, vp = _lp_polish(X0, secs, t0, DEADLINE - 2.0)
        if vp < gval:
            gval = vp; gbest = Xp.copy()

    gbest = _norm(gbest)
    # tiny shrink guard not needed; ensure finite
    gbest = np.nan_to_num(gbest, nan=0.0, posinf=0.0, neginf=0.0)
    gbest = _norm(gbest)
    return gbest
