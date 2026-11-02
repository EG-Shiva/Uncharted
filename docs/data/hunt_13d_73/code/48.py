import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _build_C12(T):
    # T (6,13) normalized rows -> X (73,13)
    n_orb = 6
    X = np.empty((73, 13), dtype=np.float64)
    idx = 0
    for k in range(n_orb):
        a = T[k, :12]
        b = T[k, 12]
        for s in range(12):
            r = np.roll(a, s)
            X[idx, :12] = r
            X[idx, 12] = b
            idx += 1
    X[72, :] = 0.0
    X[72, 12] = 1.0
    return X

def _build_C13(T5, Y8):
    # T5 (5,13), Y8 (8,13) -> X (73,13): 65 orbit + 8 free
    X = np.empty((73, 13), dtype=np.float64)
    idx = 0
    for k in range(5):
        tk = T5[k]
        for s in range(13):
            X[idx] = np.roll(tk, s)
            idx += 1
    for j in range(8):
        X[idx] = Y8[j]
        idx += 1
    return X

def _adam_C12(T0, steps, s0, s1, lr0, lr1):
    T = T0.copy()
    m = np.zeros_like(T)
    v = np.zeros_like(T)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = b1, b2
    ls0, ls1 = np.log(s0), np.log(s1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for k in range(steps):
        f = k / (steps - 1) if steps > 1 else 1.0
        s = np.exp(ls0 + (ls1 - ls0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        X = _build_C12(T)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cm = G.max()
        E = np.exp(s * (G - cm))
        np.fill_diagonal(E, 0.0)
        S = E.sum()
        if not np.isfinite(S) or S <= 0:
            continue
        W = E / S
        GrX = W @ X  # (73,13)
        Gt = np.zeros_like(T)
        idx = 0
        for kk in range(6):
            gacc12 = np.zeros(12)
            gaccb = 0.0
            # aggregate rolled-back grads
            for sf in range(12):
                g = GrX[idx]
                idx += 1
                ga = g[:12]
                # roll back by -sf
                if sf == 0:
                    gacc12 += ga
                else:
                    gacc12 += np.roll(ga, -sf)
                gaccb += g[12]
            Gt[kk, :12] = gacc12
            Gt[kk, 12] = gaccb
        # tangent projection
        dot = (Gt * T).sum(axis=1, keepdims=True)
        Gt = Gt - dot * T
        m *= b1
        m += (1.0 - b1) * Gt
        v *= b2
        v += (1.0 - b2) * (Gt * Gt)
        mh = m / (1.0 - b1p)
        vh = v / (1.0 - b2p)
        b1p *= b1
        b2p *= b2
        T -= lr * (mh / (np.sqrt(vh) + eps))
        nr = np.sqrt((T * T).sum(axis=1, keepdims=True))
        nr[nr == 0] = 1.0
        T /= nr
    X = _build_C12(T)
    return T, X

def _adam_C13(T50, Y80, steps, s0, s1, lr0, lr1):
    T = T50.copy()
    Y = Y80.copy()
    mT = np.zeros_like(T)
    vT = np.zeros_like(T)
    mY = np.zeros_like(Y)
    vY = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = b1, b2
    ls0, ls1 = np.log(s0), np.log(s1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for k in range(steps):
        f = k / (steps - 1) if steps > 1 else 1.0
        s = np.exp(ls0 + (ls1 - ls0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        X = _build_C13(T, Y)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cm = G.max()
        E = np.exp(s * (G - cm))
        np.fill_diagonal(E, 0.0)
        S = E.sum()
        if not np.isfinite(S) or S <= 0:
            continue
        W = E / S
        GrX = W @ X
        Gt = np.zeros_like(T)
        idx = 0
        for kk in range(5):
            acc = np.zeros(13)
            for sf in range(13):
                g = GrX[idx]
                idx += 1
                if sf == 0:
                    acc += g
                else:
                    acc += np.roll(g, -sf)
            Gt[kk] = acc
        Gy = GrX[idx:idx + 8].copy()
        dotT = (Gt * T).sum(axis=1, keepdims=True)
        Gt = Gt - dotT * T
        dotY = (Gy * Y).sum(axis=1, keepdims=True)
        Gy = Gy - dotY * Y
        # adam update both
        mT *= b1; mT += (1.0 - b1) * Gt
        vT *= b2; vT += (1.0 - b2) * (Gt * Gt)
        mY *= b1; mY += (1.0 - b1) * Gy
        vY *= b2; vY += (1.0 - b2) * (Gy * Gy)
        mhT = mT / (1.0 - b1p); vhT = vT / (1.0 - b2p)
        mhY = mY / (1.0 - b1p); vhY = vY / (1.0 - b2p)
        b1p *= b1; b2p *= b2
        T -= lr * (mhT / (np.sqrt(vhT) + eps))
        Y -= lr * (mhY / (np.sqrt(vhY) + eps))
        nT = np.sqrt((T * T).sum(axis=1, keepdims=True)); nT[nT == 0] = 1.0; T /= nT
        nY = np.sqrt((Y * Y).sum(axis=1, keepdims=True)); nY[nY == 0] = 1.0; Y /= nY
    X = _build_C13(T, Y)
    return T, Y, X

def _adam_full(X0, steps, s0, s1, lr0, lr1):
    X = X0.copy()
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = b1, b2
    ls0, ls1 = np.log(s0), np.log(s1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for k in range(steps):
        f = k / (steps - 1) if steps > 1 else 1.0
        s = np.exp(ls0 + (ls1 - ls0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cm = G.max()
        E = np.exp(s * (G - cm))
        np.fill_diagonal(E, 0.0)
        S = E.sum()
        if not np.isfinite(S) or S <= 0:
            continue
        W = E / S
        Gt = W @ X
        dot = (Gt * X).sum(axis=1, keepdims=True)
        Gt = Gt - dot * X
        m *= b1; m += (1.0 - b1) * Gt
        v *= b2; v += (1.0 - b2) * (Gt * Gt)
        mh = m / (1.0 - b1p); vh = v / (1.0 - b2p)
        b1p *= b1; b2p *= b2
        X -= lr * (mh / (np.sqrt(vh) + eps))
        nr = np.sqrt((X * X).sum(axis=1, keepdims=True))
        nr[nr == 0] = 1.0
        X /= nr
    return X

def _lp_polish(X, seconds, t_end):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end:
        if radius < 1e-14:
            break
        G = X @ X.T
        c = G[iu0]
        m = c.max()
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        if len(I) > 2200:
            idx = np.argsort(-c[sel])[:2200]
            I, J = I[idx], J[idx]
            ssel = c[sel][idx]
        else:
            ssel = c[sel]
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
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-ssel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            X, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    SYM_END = t0 + 65.0
    REL_END = t0 + 148.0
    rng = np.random.RandomState(20241)
    n, d = 73, 13
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.randn(n, d)
    if xb.shape != (n, d):
        tmp = rng.randn(n, d)
        m0 = min(xb.shape[0], n); m1 = min(xb.shape[1], d)
        tmp[:m0, :m1] = xb[:m0, :m1]
        xb = tmp
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    Xbest = xb.copy()

    cands = []

    # ---- C12 symmetry search ----
    n_init12 = 12
    for ii in range(n_init12):
        if time.time() > SYM_END or time.time() > HARD_END - 60:
            break
        T0 = rng.randn(6, 13)
        T0 /= np.linalg.norm(T0, axis=1, keepdims=True)
        # vary annealing per init
        if ii % 3 == 0:
            s0, s1 = 30.0, 250.0
            lr0, lr1 = 0.03, 0.003
            steps = 550
        elif ii % 3 == 1:
            s0, s1 = 80.0, 450.0
            lr0, lr1 = 0.02, 0.002
            steps = 550
        else:
            s0, s1 = 150.0, 700.0
            lr0, lr1 = 0.015, 0.0015
            steps = 500
        try:
            _, Xsym = _adam_C12(T0, steps, s0, s1, lr0, lr1)
        except Exception:
            continue
        try:
            v = _max_cos(Xsym)
        except Exception:
            continue
        if np.isfinite(v):
            cands.append((v, Xsym.copy()))
            if v < best_val:
                best_val = v
                Xbest = Xsym.copy()

    # ---- C13 + 8 free search (if time) ----
    n_init13 = 6
    for ii in range(n_init13):
        if time.time() > SYM_END + 18 or time.time() > HARD_END - 55:
            break
        T0 = rng.randn(5, 13); T0 /= np.linalg.norm(T0, axis=1, keepdims=True)
        Y0 = rng.randn(8, 13); Y0 /= np.linalg.norm(Y0, axis=1, keepdims=True)
        try:
            _, _, Xsym = _adam_C13(T0, Y0, 450, 40.0, 350.0, 0.02, 0.002)
        except Exception:
            continue
        try:
            v = _max_cos(Xsym)
        except Exception:
            continue
        if np.isfinite(v):
            cands.append((v, Xsym.copy()))
            if v < best_val:
                best_val = v
                Xbest = Xsym.copy()

    # sort candidates, keep top few distinct
    cands.sort(key=lambda t: t[0])
    top = []
    for v, Xc in cands:
        top.append((v, Xc))
        if len(top) >= 3:
            break
    # if symmetric search failed to produce, use incumbent
    if len(top) == 0:
        top = [(best_val, Xbest.copy())]

    # ---- release symmetry: perturb + unconstrained Adam ----
    sigmas = [0.0003, 0.001, 0.003, 0.01]
    # quick release runs
    for (v0, Xc0) in list(top):
        for sg in sigmas:
            if time.time() > REL_END or time.time() > HARD_END - 35:
                break
            try:
                Xp = Xc0 + sg * rng.randn(n, d)
                Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
                # two-stage anneal: coarse then sharp
                Xa = _adam_full(Xp, 400, 60.0, 300.0, 0.015, 0.003)
                va = _max_cos(Xa)
                if va < best_val + 0.012:
                    Xb = _adam_full(Xa, 350, 300.0, 650.0, 0.005, 0.001)
                    vb = _max_cos(Xb)
                else:
                    Xb, vb = Xa, va
                if vb < best_val:
                    best_val = vb
                    Xbest = Xb.copy()
            except Exception:
                continue
        if time.time() > REL_END:
            break

    # extra global polish of current best with small kicks (limited)
    for _ in range(3):
        if time.time() > REL_END or time.time() > HARD_END - 30:
            break
        try:
            sg = 0.002
            Xp = Xbest + sg * rng.randn(n, d)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            Xq = _adam_full(Xp, 350, 120.0, 550.0, 0.008, 0.0015)
            vq = _max_cos(Xq)
            if vq < best_val:
                best_val = vq
                Xbest = Xq.copy()
        except Exception:
            continue

    # ---- final single LP polish ----
    try:
        if time.time() < HARD_END - 8:
            remain = min(18.0, (HARD_END - 3.0) - time.time())
            if remain > 4.0:
                Xp = _lp_polish(Xbest, remain, HARD_END - 3.0)
                vp = _max_cos(Xp)
                if vp < best_val and np.isfinite(vp):
                    Xbest, best_val = Xp, vp
    except Exception:
        pass

    Xbest /= np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
