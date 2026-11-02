import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _soft_Gr(X, beta):
    # gradient of logsumexp(beta*G)/beta w.r.t X (unprojected), up to scale
    G = X @ X.T
    np.fill_diagonal(G, -10.0)
    gm = float(G.max())
    E = np.exp(beta * (G - gm))
    np.fill_diagonal(E, 0.0)
    S = float(E.sum())
    if S == 0 or not np.isfinite(S):
        return np.zeros_like(X), G
    Gr = 2.0 * (E @ X) / S
    return Gr, G

def _build_c15(seeds):
    # seeds (5,15) unit rows -> X (76,15)
    sn = seeds / np.linalg.norm(seeds, axis=1, keepdims=True)
    X = np.empty((76, 15), dtype=float)
    for s in range(5):
        v = sn[s]
        for k in range(15):
            X[s * 15 + k] = np.roll(v, k)
    X[75] = 1.0 / np.sqrt(15.0)
    return X, sn

def _build_signed(seeds):
    sn = seeds / np.linalg.norm(seeds, axis=1, keepdims=True)
    X = np.empty((76, 15), dtype=float)
    for s in range(5):
        v = sn[s]
        for k in range(15):
            r = np.roll(v, k)
            if k & 1:
                r = -r
            X[s * 15 + k] = r
    X[75] = 1.0 / np.sqrt(15.0)
    return X, sn

def _build_anti(seeds):
    sn = seeds / np.linalg.norm(seeds, axis=1, keepdims=True)
    X = np.empty((76, 15), dtype=float)
    X[:38] = sn[:38]
    X[38:] = -sn[:38]
    return X, sn

def _opt_c15(rng, steps, beta0, beta1, lr0, lr1, signed=False):
    S = rng.normal(size=(5, 15))
    S = _norm(S)
    m = np.zeros_like(S); vv = np.zeros_like(S)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1t, b2t = 1.0, 1.0
    lb = np.log(beta1 / beta0); ll = np.log(lr1 / lr0)
    for it in range(steps):
        f = it / max(1, steps - 1)
        beta = beta0 * np.exp(lb * f)
        lr = lr0 * np.exp(ll * f)
        if signed:
            X, sn = _build_signed(S)
        else:
            X, sn = _build_c15(S)
        Gr, _ = _soft_Gr(X, beta)
        # backprop to seeds
        Gs = np.zeros_like(S)
        for s in range(5):
            acc = np.zeros(15)
            for k in range(15):
                g = Gr[s * 15 + k]
                acc += np.roll(g, -k)
                if signed and (k & 1):
                    acc -= 2 * np.roll(g, -k)  # careful: sign flip
                    # actually X=-roll => dL/dseed += -roll^{-1}(g); above double counted, fix:
                    # acc currently has +roll^{-1}; need -roll^{-1} => subtract 2x
            # tangent project w.r.t sn
            Gs[s] = acc - float(acc @ sn[s]) * sn[s]
        b1t *= b1; b2t *= b2
        m *= b1; m += (1 - b1) * Gs
        vv *= b2; vv += (1 - b2) * Gs * Gs
        mh = m / (1 - b1t); vh = vv / (1 - b2t)
        S = sn - lr * mh / (np.sqrt(vh) + eps)
        S = _norm(S)
    if signed:
        X, _ = _build_signed(S)
    else:
        X, _ = _build_c15(S)
    return X

def _opt_anti(rng, steps, beta0, beta1, lr0, lr1):
    S = rng.normal(size=(38, 15))
    S = _norm(S)
    m = np.zeros_like(S); vv = np.zeros_like(S)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1t, b2t = 1.0, 1.0
    lb = np.log(beta1 / beta0); ll = np.log(lr1 / lr0)
    for it in range(steps):
        f = it / max(1, steps - 1)
        beta = beta0 * np.exp(lb * f)
        lr = lr0 * np.exp(ll * f)
        X, sn = _build_anti(S)
        Gr, _ = _soft_Gr(X, beta)
        Gs = (Gr[:38] - Gr[38:]).copy()
        # tangent project
        Gs = Gs - (np.sum(Gs * sn, axis=1, keepdims=True)) * sn
        b1t *= b1; b2t *= b2
        m *= b1; m += (1 - b1) * Gs
        vv *= b2; vv += (1 - b2) * Gs * Gs
        mh = m / (1 - b1t); vh = vv / (1 - b2t)
        S = sn - lr * mh / (np.sqrt(vh) + eps)
        S = _norm(S)
    X, _ = _build_anti(S)
    return X

def _anneal_full(X, steps, beta0, beta1, lr0, lr1, deadline):
    n, d = X.shape
    m = np.zeros_like(X); vv = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1t, b2t = 1.0, 1.0
    lb = np.log(beta1 / beta0); ll = np.log(lr1 / lr0)
    for it in range(steps):
        if time.time() > deadline - 12:
            break
        f = it / max(1, steps - 1)
        beta = beta0 * np.exp(lb * f)
        lr = lr0 * np.exp(ll * f)
        Gr, _ = _soft_Gr(X, beta)
        Gt = Gr - (np.sum(Gr * X, axis=1, keepdims=True)) * X
        b1t *= b1; b2t *= b2
        m *= b1; m += (1 - b1) * Gt
        vv *= b2; vv += (1 - b2) * Gt * Gt
        mh = m / (1 - b1t); vh = vv / (1 - b2t)
        X = X - lr * mh / (np.sqrt(vh) + eps)
        X = _norm(X)
    return X

def _lp_polish(X, t_end):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return X
    X = _norm(X)
    n, d = X.shape
    best = _maxcos(X)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < t_end - 0.6 and radius > 1e-14:
        c = (X @ X.T)[iu]
        mcur = float(c.max())
        thr = max(20 * radius, 1e-9)
        sel = c > mcur - thr
        if int(sel.sum()) > 1500:
            idx = np.argpartition(-c, 1500)[:1500]
            mask = np.zeros_like(sel); mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
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
            break
        if res.status != 0:
            radius *= 0.5
            continue
        Y = X + res.x[:-1].reshape(n, d)
        Y = _norm(Y)
        v = _maxcos(Y)
        if v < best:
            X, best, radius = Y, v, min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    # load fallback
    try:
        xb = np.load("best.npy").astype(float)
        xb = _norm(xb)
        if xb.shape != (76, 15):
            xb = None
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
        xr = _norm(xr)
        if xr.shape != (76, 15):
            xr = None
    except Exception:
        xr = None
    if xb is None and xr is None:
        rng0 = np.random.default_rng(0)
        X = rng0.normal(size=(76, 15)); return _norm(X)
    base = xb if xb is not None else xr
    bv = _maxcos(base)
    best = base.copy()

    rng = np.random.default_rng(12345)
    cands = []
    # Phase A: C15 orbits
    # few restarts, short steps to stay fast
    for r in range(10):
        if time.time() > deadline - 60:
            break
        X = _opt_c15(rng, steps=900, beta0=25.0, beta1=380.0, lr0=0.03, lr1=0.003, signed=False)
        cands.append(X)
    for r in range(5):
        if time.time() > deadline - 55:
            break
        X = _opt_c15(rng, steps=700, beta0=25.0, beta1=300.0, lr0=0.03, lr1=0.003, signed=True)
        cands.append(X)
    for r in range(6):
        if time.time() > deadline - 50:
            break
        X = _opt_anti(rng, steps=800, beta0=25.0, beta1=350.0, lr0=0.03, lr1=0.003)
        cands.append(X)
    # pick top 2 symmetric cands
    cands.sort(key=_maxcos)
    sym_best = None
    sym_val = 1e9
    for X in cands[:3]:
        v = _maxcos(X)
        if v < sym_val:
            sym_val = v; sym_best = X.copy()
    # release symmetry: anneal full from sym_best (and second best)
    rel_starts = []
    if sym_best is not None:
        rel_starts.append(sym_best)
        if len(cands) > 1:
            rel_starts.append(cands[1])
    # also add perturbed base to not lose record basin
    if time.time() < deadline - 45:
        rel_starts.append(best.copy())
    rel_best = best.copy(); rel_val = bv
    # also consider raw sym before release
    for X in cands:
        v = _maxcos(X)
        if v < rel_val:
            rel_val = v; rel_best = X.copy()
    for Xs in rel_starts:
        if time.time() > deadline - 35:
            break
        Y = Xs.copy()
        # small shake to break exact symmetry (except base)
        if Y is not best:
            Y = Y + 1e-3 * rng.normal(size=Y.shape)
            Y = _norm(Y)
        Y = _anneal_full(Y, steps=2500, beta0=40.0, beta1=380.0, lr0=0.008, lr1=0.0008, deadline=deadline)
        v = _maxcos(Y)
        if v < rel_val:
            rel_val = v; rel_best = Y.copy()
        # extra low-T refinement on improvement
        if v < bv and time.time() < deadline - 30:
            Y2 = _anneal_full(Y.copy(), steps=1200, beta0=380.0, beta1=900.0, lr0=0.0015, lr1=0.0002, deadline=deadline)
            v2 = _maxcos(Y2)
            if v2 < rel_val:
                rel_val = v2; rel_best = Y2.copy()
    best = rel_best; bv = rel_val
    # final LP polish on best
    if time.time() < deadline - 6:
        t_end = min(deadline - 1.0, time.time() + 24.0)
        try:
            Y = _lp_polish(best, t_end)
            v = _maxcos(Y)
            if v < bv:
                best, bv = Y, v
        except Exception:
            pass
    # safety: if polish hurt or NaN, revert to base
    if not np.all(np.isfinite(best)):
        best = base.copy()
    # ensure at least base quality
    try:
        if _maxcos(best) > _maxcos(base) + 1e-12:
            best = base.copy()
    except Exception:
        best = base.copy()
    best = _norm(best)
    return best
