import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-300
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _abs_max(Y):
    G = Y @ Y.T
    np.fill_diagonal(G, 0.0)
    return float(np.max(np.abs(G)))

def _greedy_Y(X):
    # pair each point with most negative partner, representative = normalized difference
    n, d = X.shape
    G = X @ X.T
    unpaired = set(range(n))
    pairs = []
    # order by min value (most antipodal first) for determinism
    minv = np.min(G + 2.0*np.eye(n), axis=1)
    order = np.argsort(minv, kind="stable")
    for i in order:
        if i not in unpaired:
            continue
        unpaired.remove(i)
        if not unpaired:
            # odd leftover, pair with itself
            pairs.append((i, i))
            break
        # find j in unpaired minimizing G[i,j]
        cand = list(unpaired)
        vals = G[i, cand]
        j = cand[int(np.argmin(vals))]
        unpaired.remove(j)
        pairs.append((i, j))
    Y = np.zeros((len(pairs), d))
    for k, (i, j) in enumerate(pairs):
        if i == j:
            Y[k] = X[i]
        else:
            v = X[i] - X[j]
            nv = np.linalg.norm(v)
            if nv < 1e-12 or not np.isfinite(nv):
                v = X[i]
                nv = np.linalg.norm(v)
            Y[k] = v / nv
    return Y

def _smooth_abs(Y, steps, a0, a1, lr0, lr1):
    Y = _norm(Y.copy())
    la0, la1 = np.log(a0), np.log(a1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(steps - 1, 1)
        alpha = np.exp(la0 + (la1 - la0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        G = Y @ Y.T
        Sgn = np.sign(G)
        Ga = np.abs(G)
        np.fill_diagonal(Ga, -np.inf)
        mx = float(np.max(Ga))
        W = np.exp(alpha * (Ga - mx))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-30
        grad = (W * Sgn) @ Y
        dot = np.sum(grad * Y, axis=1, keepdims=True)
        gt = grad - dot * Y
        direction = gt / S
        Y = Y - lr * direction
        Y = _norm(Y)
    return Y

def _smooth_std(X, steps, a0, a1, lr0, lr1):
    X = _norm(X.copy())
    la0, la1 = np.log(a0), np.log(a1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(steps - 1, 1)
        alpha = np.exp(la0 + (la1 - la0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        G = X @ X.T
        np.fill_diagonal(G, -np.inf)
        mx = float(np.max(G))
        W = np.exp(alpha * (G - mx))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-30
        grad = W @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        direction = gt / S
        X = X - lr * direction
        X = _norm(X)
    return X

def _lp_polish(X, seconds, deadline, pcap=2200):
    X = _norm(X.copy())
    n, d = X.shape
    best = _max_cos(X)
    xb = X.copy()
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds:
        if time.time() > deadline:
            break
        if radius < 1e-13:
            break
        G = X @ X.T
        cf = G[iu0]
        m = float(cf.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = cf > thr
        I = iu0[0][sel]; J = iu0[1][sel]; cs = cf[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > pcap:
            idx = np.argpartition(-cs, pcap)[:pcap]
            I, J, cs = I[idx], J[idx], cs[idx]
            p = pcap
        nv = n * d + 1
        rows = np.concatenate([np.repeat(np.arange(p), d), np.repeat(np.arange(p), d), np.arange(p)])
        cols = np.concatenate([np.repeat(I * d, d) + np.tile(ar, p),
                               np.repeat(J * d, d) + np.tile(ar, p),
                               np.full(p, n * d)])
        vals = np.concatenate([X[J].ravel(), X[I].ravel(), -np.ones(p)])
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        Y = _norm(X + res.x[:-1].reshape(n, d))
        v = _max_cos(Y)
        if v < best:
            best = v; xb = Y.copy(); X = Y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return xb

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    n, d, m = 184, 8, 92

    # load references
    refs = []
    for f in ("best.npy", "record.npy"):
        try:
            c = np.asarray(np.load(f), dtype=np.float64)
            if c.shape == (n, d):
                c = _norm(c)
                refs.append(c)
        except Exception:
            pass
    if refs:
        # best first
        refs.sort(key=_max_cos)
        x_best = refs[0].copy()
        best_val = _max_cos(x_best)
    else:
        rng0 = np.random.default_rng(0)
        x_best = _norm(rng0.standard_normal((n, d)))
        best_val = _max_cos(x_best)
    gbest = x_best.copy()
    gval = best_val

    # build Y inits: greedy from refs + random
    Y_inits = []
    for X in refs[:2]:
        try:
            Yg = _greedy_Y(X)
            if Yg.shape == (m, d):
                Y_inits.append(Yg)
        except Exception:
            pass
    rng = np.random.default_rng(12345)
    for s in range(3):
        Yr = rng.standard_normal((m, d))
        Yr = _norm(Yr)
        Y_inits.append(Yr)
    # extra random with different seed for diversity
    rng2 = np.random.default_rng(999)
    Yr = _norm(rng2.standard_normal((m, d)))
    Y_inits.append(Yr)

    # optimize each Y with softmax over |dot|, keep time for later stages
    # budget: leave ~70s for release+LP
    bestY = None
    bestYval = 1e9
    # quick evaluate inits
    for Y in Y_inits:
        v = _abs_max(Y)
        if v < bestYval:
            bestYval = v; bestY = Y.copy()

    # schedules: two rounds - coarse then fine on most promising
    # To save time, give each init moderate steps, then best gets extra
    Y_opts = []
    for idx, Y0 in enumerate(Y_inits):
        if time.time() > DEADLINE - 90:
            break
        # vary schedule slightly per idx for diversity
        if idx == 0:
            Yo = _smooth_abs(Y0, steps=1400, a0=12.0, a1=450.0, lr0=0.07, lr1=0.004)
        elif idx == 1:
            Yo = _smooth_abs(Y0, steps=1400, a0=18.0, a1=550.0, lr0=0.06, lr1=0.003)
        else:
            Yo = _smooth_abs(Y0, steps=1100, a0=15.0, a1=400.0, lr0=0.06, lr1=0.004)
        v = _abs_max(Yo)
        Y_opts.append((v, Yo))
        if v < bestYval:
            bestYval = v; bestY = Yo.copy()
        if time.time() > DEADLINE - 90:
            break

    # extra polish of top-2 Y
    Y_opts.sort(key=lambda t: t[0])
    for k in range(min(2, len(Y_opts))):
        if time.time() > DEADLINE - 80:
            break
        v0, Y0 = Y_opts[k]
        Y1 = _smooth_abs(Y0, steps=700, a0=120.0, a1=900.0, lr0=0.008, lr1=0.0015)
        v1 = _abs_max(Y1)
        Y_opts.append((v1, Y1))
        if v1 < bestYval:
            bestYval = v1; bestY = Y1.copy()

    # duplicate best Y antipodally and release constraint
    # try top 2 Y candidates through release path
    Y_opts.sort(key=lambda t: t[0])
    candidates = []
    for k in range(min(2, len(Y_opts))):
        candidates.append(Y_opts[k][1])
    if bestY is not None and all(not np.allclose(bestY, c) for c in candidates):
        candidates.append(bestY)
    # also ensure at least one
    if not candidates:
        candidates = [Y_inits[0]]

    rel_rng = np.random.default_rng(2026)
    for ci, Yc in enumerate(candidates):
        if time.time() > DEADLINE - 65:
            break
        Xsym = np.vstack([Yc, -Yc])
        Xsym = _norm(Xsym)
        vsym = _max_cos(Xsym)
        # symmetry-breaking noise, small for first, larger for second
        sigma = 0.002 if ci == 0 else 0.006
        Xn = Xsym + sigma * rel_rng.standard_normal(Xsym.shape)
        Xn = _norm(Xn)
        # release refine: standard softmax on 184 points
        Xr = _smooth_std(Xn, steps=600, a0=35.0, a1=380.0, lr0=0.035, lr1=0.004)
        vr = _max_cos(Xr)
        # second pass with hotter schedule
        if time.time() < DEADLINE - 55:
            Xr2 = _smooth_std(Xr, steps=350, a0=120.0, a1=700.0, lr0=0.008, lr1=0.0015)
            vr2 = _max_cos(Xr2)
            if vr2 < vr:
                Xr, vr = Xr2, vr2
        if vr < gval:
            gval = vr; gbest = Xr.copy()

    # LP polish once on current global best (released) — as directed, single polish
    # give it remaining time up to 35s but respect deadline
    remaining = DEADLINE - time.time() - 2.0
    if remaining > 5:
        lp_sec = min(remaining, 35.0)
        try:
            Xp = _lp_polish(gbest, seconds=lp_sec, deadline=DEADLINE - 1.0)
            vp = _max_cos(Xp)
            if vp < gval:
                gval = vp; gbest = Xp.copy()
        except Exception:
            pass

    # fallback safety
    try:
        if _max_cos(gbest) > best_val + 1e-12:
            gbest = x_best.copy()
    except Exception:
        gbest = x_best.copy()
    gbest = _norm(np.asarray(gbest, dtype=np.float64))
    return gbest
