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

def _max_abs_off(Y):
    G = Y @ Y.T
    np.fill_diagonal(G, 0.0)
    return float(np.max(np.abs(G)))

def _greedy_pair_Y(X):
    # X: (184,8) normalized; pair each point with most negative partner
    n = X.shape[0]
    G = X @ X.T
    unpaired = set(range(n))
    Y = []
    while unpaired:
        i = next(iter(unpaired))
        unpaired.remove(i)
        if not unpaired:
            break
        # most negative inner product among remaining
        best_j = None
        best_v = 1e9
        for j in unpaired:
            v = G[i, j]
            if v < best_v:
                best_v = v
                best_j = j
        j = best_j
        unpaired.remove(j)
        v = X[i] - X[j]
        nv = np.linalg.norm(v)
        if nv < 1e-12 or not np.isfinite(nv):
            # fallback: use i
            v = X[i].copy()
            nv = np.linalg.norm(v)
        Y.append(v / nv)
    return np.array(Y, dtype=np.float64)

def _sym_smooth(Y, steps=600, a0=15.0, a1=400.0, lr0=0.06, lr1=0.004):
    Y = _norm(Y.copy())
    n, d = Y.shape
    la0 = np.log(a0); la1 = np.log(a1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for t in range(steps):
        f = t / max(steps - 1, 1)
        alpha = np.exp(la0 + (la1 - la0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        G = Y @ Y.T
        np.fill_diagonal(G, 0.0)
        A = np.abs(G)
        # max off-diagonal (diag is 0, but true max is ~0.5 so fine)
        # ensure diag excluded
        np.fill_diagonal(A, -np.inf)
        mx = np.max(A)
        W = np.exp(alpha * (A - mx))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-30
        Sg = np.sign(G)
        # gradient of |dot| : sign * partner
        grad = (W * Sg) @ Y
        dot = np.sum(grad * Y, axis=1, keepdims=True)
        gt = grad - dot * Y
        direction = gt / S
        Y = Y - lr * direction
        Y = _norm(Y)
    return Y

def _smooth_refine(X, steps=350, a0=45.0, a1=320.0, lr0=0.03, lr1=0.003):
    X = _norm(X.copy())
    la0 = np.log(a0); la1 = np.log(a1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for t in range(steps):
        f = t / max(steps - 1, 1)
        alpha = np.exp(la0 + (la1 - la0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        G = X @ X.T
        np.fill_diagonal(G, -np.inf)
        mx = np.max(G)
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

def _lp_polish_once(X, seconds, deadline):
    X = _norm(X.copy())
    n, d = X.shape
    best = _max_cos(X)
    xb = X.copy()
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() < deadline - 0.5 and time.time() < deadline and radius > 1e-13:
        # time budget check
        if seconds is not None:
            pass
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
        if p > 2200:
            idx = np.argpartition(-cs, 2200)[:2200]
            I, J, cs = I[idx], J[idx], cs[idx]
            p = 2200
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
        if time.time() > deadline - 1.0:
            break
    return xb

def solve():
    t0 = time.time()
    DEADLINE = t0 + 175.0
    rng = np.random.default_rng(0)
    n, d = 184, 8
    # load references
    refs = {}
    for f in ("best.npy", "record.npy"):
        try:
            c = np.asarray(np.load(f), dtype=np.float64)
            if c.shape == (n, d):
                c = _norm(c)
                refs[f] = c
        except Exception:
            pass
    if "best.npy" in refs:
        x_best_ref = refs["best.npy"]
    elif "record.npy" in refs:
        x_best_ref = refs["record.npy"]
    else:
        x_best_ref = _norm(rng.standard_normal(size=(n, d)))
    gval = _max_cos(x_best_ref)
    gbest = x_best_ref.copy()

    # build Y inits
    Y_inits = []
    for key in ("record.npy", "best.npy"):
        if key in refs:
            try:
                Yg = _greedy_pair_Y(refs[key])
                if Yg.shape == (92, 8):
                    Y_inits.append(Yg)
            except Exception:
                pass
    # random Y inits
    rng2 = np.random.default_rng(1)
    for _ in range(4):
        Y_inits.append(_norm(rng2.standard_normal(size=(92, 8))))

    # symmetric optimization: reduced Grassmann packing
    bestY = None
    best_sym = 1e9
    # also keep top-2
    scored_Y = []
    # time guard: leave ~70s for release+LP
    sym_deadline = t0 + 75.0
    trial = 0
    for yi, Y0 in enumerate(Y_inits):
        if time.time() > sym_deadline:
            break
        # two jitter variants per init (original + jittered)
        variants = [Y0]
        if yi < 2:
            for sig in (0.02, 0.08):
                Yn = _norm(Y0 + rng.standard_normal(size=Y0.shape) * sig)
                variants.append(Yn)
        for Yv in variants:
            if time.time() > sym_deadline:
                break
            trial += 1
            # vary schedule slightly
            if trial % 3 == 0:
                Yopt = _sym_smooth(Yv, steps=550, a0=12.0, a1=500.0, lr0=0.07, lr1=0.003)
            elif trial % 3 == 1:
                Yopt = _sym_smooth(Yv, steps=650, a0=20.0, a1=350.0, lr0=0.05, lr1=0.005)
            else:
                Yopt = _sym_smooth(Yv, steps=500, a0=25.0, a1=450.0, lr0=0.06, lr1=0.004)
            v = _max_abs_off(Yopt)
            scored_Y.append((v, Yopt))
            if v < best_sym:
                best_sym = v
                bestY = Yopt.copy()
    scored_Y.sort(key=lambda t: t[0])
    # duplicate antipodally and release constraint
    cands = []
    # take up to 3 best symmetric results
    for k in range(min(3, len(scored_Y))):
        Yk = scored_Y[k][1]
        Xk = np.vstack([Yk, -Yk])
        Xk = _norm(Xk)
        cands.append(Xk)
    # also include reference best as fallback candidate
    cands.append(gbest.copy())

    # refine released configurations (no symmetry constraint)
    refined = []
    # budget refine before LP: leave 35s for LP
    refine_deadline = DEADLINE - 35.0
    for ci, Xc in enumerate(cands):
        if time.time() > refine_deadline:
            break
        if ci < 3:
            # break symmetry slightly then refine
            Xp = Xc + rng.standard_normal(size=Xc.shape) * 0.004
            Xp = _norm(Xp)
            Xo = _smooth_refine(Xp, steps=320, a0=45.0, a1=320.0, lr0=0.030, lr1=0.0025)
            # extra polish with different lr for best one
            if ci == 0 and time.time() < refine_deadline - 5:
                Xo = _smooth_refine(Xo, steps=200, a0=80.0, a1=500.0, lr0=0.012, lr1=0.002)
        else:
            Xo = Xc  # reference, no refine to save time (already polished)
        v = _max_cos(Xo)
        refined.append((v, Xo))
    refined.sort(key=lambda t: t[0])
    if refined and refined[0][0] < gval:
        gval = refined[0][0]
        gbest = refined[0][1].copy()
    # LP-polish once the current global best
    remain = DEADLINE - time.time() - 1.0
    if remain > 6:
        lp_deadline = min(DEADLINE - 0.5, time.time() + min(remain, 32.0))
        try:
            gbest = _lp_polish_once(gbest, remain, lp_deadline)
        except Exception:
            pass
    gbest = _norm(gbest)
    if not np.all(np.isfinite(gbest)):
        gbest = _norm(x_best_ref)
    # safety: if somehow worse than ref and time allows, return ref
    try:
        if _max_cos(gbest) > _max_cos(x_best_ref) + 1e-12:
            # keep the better one (still return gbest? we want lower; return ref)
            gbest = x_best_ref.copy()
    except Exception:
        pass
    return gbest
