import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _lp_polish(X, seconds, deadline=None):
    X = _norm(np.asarray(X, dtype=np.float64))
    n, d = X.shape
    best = _max_cos(X)
    bx = X.copy()
    if seconds <= 0:
        return bx, best
    start = time.time()
    radius = 1e-3
    iu0, iu1 = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if deadline is not None and time.time() >= deadline:
            break
        if radius < 1e-14:
            break
        Gf = X @ X.T
        c = Gf[iu0, iu1]
        m = float(c.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        if int(sel.sum()) == 0:
            break
        I = iu0[sel]; J = iu1[sel]
        p = len(I)
        # cap pairs to keep LP small
        if p > 3000:
            # keep most tight
            idx = np.argsort(-c[sel])[:3000]
            I = I[idx]; J = J[idx]
            csel = c[sel][idx]
            p = len(I)
        else:
            csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = np.tile(np.arange(d), p) + np.repeat(I * d, d) if False else None
        # build cols/vals vectorised
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            tl = seconds - (time.time() - start)
            tl = max(0.5, min(tl, 8.0))
            if deadline is not None:
                tl = min(tl, max(0.5, deadline - time.time()))
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": tl})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            X, best, bx = y, v, y.copy()
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return bx, best

def _fingerprints(X, tol=1e-6):
    n = X.shape[0]
    G = X @ X.T
    m = float(np.max(G - 2.0 * np.eye(n)))
    deg = (G > m - tol).sum(axis=1) - 1  # exclude self
    deg = np.asarray(deg, dtype=float)
    sdeg = np.sort(deg)
    try:
        spec = np.linalg.eigvalsh(G)
    except Exception:
        spec = np.zeros(n)
    spec = np.sort(spec)
    # personal max
    G2 = G.copy()
    np.fill_diagonal(G2, -2.0)
    pmax = G2.max(axis=1)
    return sdeg, spec, deg, float(m), pmax, G

def solve():
    t0 = time.time()
    HARD = 160.0
    deadline = t0 + HARD
    rng = np.random.default_rng(0)
    n, d = 78, 15

    # incumbent checkpoint (never destroyed)
    incumbent = None
    inc_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape != (78, 15):
                continue
            a = _norm(a)
            v = _max_cos(a)
            if v < inc_val:
                inc_val = v
                incumbent = a.copy()
        except Exception:
            continue
    if incumbent is None:
        incumbent = _norm(rng.standard_normal((n, d)))
        inc_val = _max_cos(incumbent)
    # record fingerprint for diversity filter
    try:
        rec = np.asarray(np.load("record.npy"), dtype=np.float64)
        rec = _norm(rec)
    except Exception:
        rec = incumbent.copy()
    rec_sdeg, rec_spec, _, _, _, _ = _fingerprints(rec, tol=1e-6)

    R = 12
    STEPS = 2400
    T0, T1 = 0.03, 0.002
    # init independent random codes (different basin, no record start)
    X = rng.standard_normal((R, n, d))
    X = X / np.maximum(np.linalg.norm(X, axis=2, keepdims=True), 1e-12)
    m_ad = np.zeros_like(X)
    v_ad = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lr0 = 0.015
    # anneal
    logT0, logT1 = np.log(T0), np.log(T1)
    for k in range(1, STEPS + 1):
        if time.time() >= deadline - 55.0:
            break  # reserve time for LP
        frac = k / STEPS
        T = float(np.exp(logT0 + (logT1 - logT0) * frac))
        lr = lr0 * (1.0 - 0.5 * frac)
        # Gram batch
        G = np.matmul(X, np.transpose(X, (0, 2, 1)))  # R,N,N
        # per-replica max off-diag
        for r in range(R):
            G[r].flat[::n + 1] = -2.0
        m = G.max(axis=(1, 2)).reshape(R, 1, 1)
        W = np.exp((G - m) / T)
        for r in range(R):
            W[r].flat[::n + 1] = 0.0
        Gr = np.matmul(W, X)  # R,N,D
        dot = (Gr * X).sum(axis=2, keepdims=True)
        Gt = Gr - dot * X
        # Adam
        m_ad = b1 * m_ad + (1 - b1) * Gt
        v_ad = b2 * v_ad + (1 - b2) * (Gt * Gt)
        bc1 = 1 - b1 ** k
        bc2 = 1 - b2 ** k
        mh = m_ad / bc1
        vh = v_ad / bc2
        upd = mh / (np.sqrt(vh) + eps)
        X = X - lr * upd
        X = X / np.maximum(np.linalg.norm(X, axis=2, keepdims=True), 1e-12)

    # rank survivors by score + balance, filter record-like fingerprints
    cands = []
    for r in range(R):
        Xr = _norm(X[r])
        v = _max_cos(Xr)
        sdeg, spec, deg, mcur, pmax, _ = _fingerprints(Xr, tol=1e-6)
        dd = float(np.linalg.norm(sdeg - rec_sdeg))
        sd = float(np.linalg.norm(spec - rec_spec))
        std = float(np.std(deg))
        ratt = int((deg <= 0.5).sum())
        cands.append((v, std, ratt, dd, sd, Xr, sdeg, spec, deg, pmax))
    # distinctness: require fingerprint distance from record
    # record split is 20-39 vs 0-19; distinct means sorted-deg L2 large and spec different
    distinct = []
    similar = []
    for c in cands:
        v, std, ratt, dd, sd, Xr, sdeg, spec, deg, pmax = c
        if dd > 12.0 or (sd > 0.8 and ratt == 0):
            distinct.append(c)
        else:
            # also allow if very balanced even when dd small
            if std < 6.0 and ratt == 0:
                distinct.append(c)
            else:
                similar.append(c)
    if len(distinct) < 2:
        # take most distant from record
        cands_sorted = sorted(cands, key=lambda c: (-c[3], c[0]))
        distinct = cands_sorted[:2]
    # rank distinct: score plus small balance penalty (prefer isotropic, no rattlers)
    def rankkey(c):
        v, std, ratt, dd, sd, Xr, sdeg, spec, deg, pmax = c
        return v + 0.00025 * std + 0.002 * ratt
    distinct = sorted(distinct, key=rankkey)
    top2 = distinct[:2]

    # LP-polish best two genuinely distinct survivors, <=20s each
    polished = []
    for c in top2:
        if time.time() >= deadline - 25.0:
            break
        Xr = c[5]
        v0 = c[0]
        budget = min(18.0, max(2.0, (deadline - 22.0) - time.time()))
        if budget <= 0.5:
            polished.append((v0, Xr))
            continue
        try:
            Xp, vp = _lp_polish(Xr, budget, deadline=deadline - 5.0)
        except Exception:
            Xp, vp = Xr, v0
        polished.append((vp, Xp))
        if vp < inc_val - 1e-12:
            inc_val = vp
            incumbent = Xp.copy()
    if len(polished) == 0:
        return incumbent

    # crossover: 39 lowest-personal-max points from each after Procrustes alignment
    try:
        if len(polished) >= 2 and time.time() < deadline - 8.0:
            # order by value
            polished = sorted(polished, key=lambda t: t[0])
            A = polished[0][1]
            B = polished[1][1]
            # Procrustes: B @ Q ~= A
            M = B.T @ A
            U, _, Vt = np.linalg.svd(M)
            Q = U @ Vt
            Bq = B @ Q
            # personal maxes
            Ga = A @ A.T; np.fill_diagonal(Ga, -2.0); pa = Ga.max(axis=1)
            Gb = Bq @ Bq.T; np.fill_diagonal(Gb, -2.0); pb = Gb.max(axis=1)
            ia = np.argsort(pa)[:39]
            ib = np.argsort(pb)[:39]
            C = np.vstack([A[ia], Bq[ib]])
            C = _norm(C)
            # quick tiny Adam snap on crossover (fast, ~200 steps) then LP
            if time.time() < deadline - 12.0:
                Cc = C.copy()
                m2 = np.zeros_like(Cc); v2 = np.zeros_like(Cc)
                for k in range(1, 201):
                    if time.time() >= deadline - 10.0:
                        break
                    T = 0.008
                    Gc = Cc @ Cc.T
                    np.fill_diagonal(Gc, -2.0)
                    mm = float(Gc.max())
                    Wc = np.exp((Gc - mm) / T)
                    np.fill_diagonal(Wc, 0.0)
                    Ggr = Wc @ Cc
                    dd2 = (Ggr * Cc).sum(axis=1, keepdims=True)
                    Gt2 = Ggr - dd2 * Cc
                    m2 = b1 * m2 + (1 - b1) * Gt2
                    v2 = b2 * v2 + (1 - b2) * (Gt2 * Gt2)
                    mh2 = m2 / (1 - b1 ** k); vh2 = v2 / (1 - b2 ** k)
                    Cc = Cc - 0.01 * mh2 / (np.sqrt(vh2) + eps)
                    Cc = _norm(Cc)
                C = Cc
            budget = min(14.0, max(2.0, (deadline - 2.0) - time.time()))
            if budget > 0.5 and time.time() < deadline - 3.0:
                Cp, vp = _lp_polish(C, budget, deadline=deadline - 1.0)
                if vp < inc_val - 1e-12:
                    inc_val = vp
                    incumbent = Cp.copy()
    except Exception:
        pass

    return incumbent
