import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contacts(X, tol=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    C = (G > m - tol).sum(axis=1)
    return C, m

def lp_polish(X, seconds):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (X @ X.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _anneal(X0, betas, iters_list, lrs, lam_list, t_limit=1e9, t0=None):
    n, d = X0.shape
    iu_r, iu_c = np.triu_indices(n, 1)
    X = X0.copy()
    Xn = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = np.maximum(Xn, 1e-12)
    X = X / Xn
    # Adam state on unconstrained X
    M = np.zeros_like(X)
    V = np.zeros_like(X)
    step = 0
    bestX = X.copy()
    # init best value
    G0 = X @ X.T
    bestV = float(G0[iu_r, iu_c].max())
    b1, b2, eps = 0.9, 0.999, 1e-8
    tgt = n / d
    Eye = np.eye(d)
    for bi, beta in enumerate(betas):
        iters = iters_list[bi] if bi < len(iters_list) else iters_list[-1]
        lr = lrs[bi] if bi < len(lrs) else lrs[-1]
        lam = lam_list[bi] if bi < len(lam_list) else lam_list[-1]
        for _ in range(iters):
            if t0 is not None and time.time() - t0 > t_limit:
                return bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
            step += 1
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = X / nr
            G = Y @ Y.T
            gv = G[iu_r, iu_c]
            mval = float(gv.max())
            if mval < bestV:
                bestV = mval
                bestX = Y.copy()
            e = np.exp(beta * (gv - mval))
            S = float(e.sum())
            if not np.isfinite(S) or S <= 0:
                continue
            w = e / S
            # W @ Y without full matrix via dense small matrix
            # 78x78 dense is cheap
            W = np.zeros((n, n), dtype=np.float64)
            W[iu_r, iu_c] = w
            W = W + W.T
            GradY = W @ Y
            if lam != 0:
                C = Y.T @ Y - tgt * Eye
                GradY = GradY + lam * (Y @ C)
            dot = (GradY * Y).sum(axis=1, keepdims=True)
            GradX = (GradY - dot * Y) / nr
            # clip extreme grads
            # Adam update
            M = b1 * M + (1 - b1) * GradX
            V = b2 * V + (1 - b2) * (GradX * GradX)
            bc1 = 1 - b1 ** step
            bc2 = 1 - b2 ** step
            Mh = M / bc1
            Vh = V / bc2
            X = X - lr * Mh / (np.sqrt(Vh) + eps)
    # final check of last X
    Yn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    Gf = Yn @ Yn.T
    vf = float(Gf[iu_r, iu_c].max())
    if vf < bestV:
        bestX = Yn
    return bestX / np.linalg.norm(bestX, axis=1, keepdims=True)

def _whiten(X):
    n, d = X.shape
    C = X.T @ X
    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, 1e-8)
    tgt = n / d
    s = np.sqrt(tgt / vals)
    A = (vecs * s[None, :]) @ vecs.T
    Y = X @ A
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def _void_reinsert(X, rng, k=8, samples=3000):
    n, d = X.shape
    cnts, m = _contacts(X, tol=1e-9)
    order = np.argsort(cnts, kind="stable")
    Y = X.copy()
    for idx in order[:k]:
        cand = rng.standard_normal((samples, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        mask = np.ones(n, dtype=bool); mask[idx] = False
        others = Y[mask]
        dots = cand @ others.T
        md = dots.max(axis=1)
        best = int(np.argmin(md))
        Y[idx] = cand[best]
    return Y

def solve():
    t0 = time.time()
    LIMIT = 168.0
    rng = np.random.default_rng(7)
    n, d = 78, 15
    base = None
    baseV = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (n, d):
                continue
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            v = _max_cos(a)
            if v < baseV:
                baseV = v
                base = a.copy()
        except Exception:
            continue
    if base is None:
        base = rng.standard_normal((n, d))
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        baseV = _max_cos(base)
    # candidates that change topology
    cands = []
    cands.append(base)
    try:
        cands.append(_whiten(base))
    except Exception:
        pass
    # large global kicks to intermix the two PC clusters
    for sig in (0.22, 0.40, 0.60):
        nz = rng.standard_normal((n, d)) * sig
        c = base + nz
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        cands.append(c)
    # whitened + kick
    try:
        w = _whiten(base)
        c = w + rng.standard_normal((n, d)) * 0.15
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        cands.append(c)
    except Exception:
        pass
    # void reinsertion of rattlers / low-contact tail + small kick
    try:
        vr = _void_reinsert(base, rng, k=8, samples=3000)
        cands.append(vr)
        c2 = vr + rng.standard_normal((n, d)) * 0.12
        c2 /= np.linalg.norm(c2, axis=1, keepdims=True)
        cands.append(c2)
    except Exception:
        pass
    # fresh random
    rr = rng.standard_normal((n, d))
    rr /= np.linalg.norm(rr, axis=1, keepdims=True)
    cands.append(rr)

    # Stage 1: short anneal to rank basins (fast)
    scored = []
    short_betas = [40.0, 130.0]
    short_iters = [250, 250]
    short_lrs = [0.015, 0.008]
    short_lam = [3e-4, 0.0]
    for c in cands:
        if time.time() - t0 > LIMIT - 60:
            break
        try:
            y = _anneal(c, short_betas, short_iters, short_lrs, short_lam,
                        t_limit=LIMIT - 55, t0=t0)
            v = _max_cos(y)
            scored.append((v, y))
        except Exception:
            continue
    if len(scored) == 0:
        return base
    scored.sort(key=lambda t: t[0])
    # keep top 2 for long anneal
    top = scored[:2]
    long_betas = [40.0, 80.0, 160.0, 300.0]
    long_iters = [400, 400, 500, 500]
    long_lrs = [0.015, 0.010, 0.007, 0.004]
    long_lam = [2e-4, 1e-4, 0.0, 0.0]
    refined = []
    for v, y in top:
        if time.time() - t0 > LIMIT - 45:
            refined.append((v, y))
            continue
        try:
            z = _anneal(y, long_betas, long_iters, long_lrs, long_lam,
                        t_limit=LIMIT - 40, t0=t0)
            refined.append((_max_cos(z), z))
        except Exception:
            refined.append((v, y))
    refined.sort(key=lambda t: t[0])
    curV, curX = refined[0]
    # also keep global base as fallback
    if baseV < curV:
        curV, curX = baseV, base
    # Stage 2: LP squeeze remaining time
    remain = LIMIT - (time.time() - t0) - 3.0
    if remain > 4:
        try:
            lp_time = min(remain, 30.0)
            curX = lp_polish(curX, lp_time)
            curV = _max_cos(curX)
        except Exception:
            pass
    if curV > baseV:
        return base
    curX /= np.linalg.norm(curX, axis=1, keepdims=True)
    return curX
