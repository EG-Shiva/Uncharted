import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

# precomputed indices for C15 orbits
_IDX = None
_JDX = None
def _indices():
    global _IDX, _JDX
    if _IDX is None:
        K = np.arange(15)[:, None]
        J = np.arange(15)[None, :]
        _IDX = ((J - K) % 15).astype(int)  # (15,15) X[k,j]=s[IDX[k,j]]
        _JDX = ((J + K) % 15).astype(int)  # for pullback
    return _IDX, _JDX

def _build_A(seeds):
    # seeds (5,15) -> X (76,15)
    IDX, _ = _indices()
    s = seeds / (np.linalg.norm(seeds, axis=1, keepdims=True) + 1e-30)
    blk = s[:, IDX]  # (5,15,15)
    X = np.empty((76, 15), dtype=float)
    X[:75] = blk.reshape(75, 15)
    X[75] = np.ones(15) / np.sqrt(15.0)
    return X

def _build_B(seeds):
    IDX, _ = _indices()
    s = seeds / (np.linalg.norm(seeds, axis=1, keepdims=True) + 1e-30)
    blk = s[:, IDX]
    sgn = ((-1) ** np.arange(15)).astype(float)[None, :, None]
    blk = blk * sgn
    X = np.empty((76, 15), dtype=float)
    X[:75] = blk.reshape(75, 15)
    X[75] = np.ones(15) / np.sqrt(15.0)
    return X

def _build_C(Y):
    yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-30)
    return np.vstack([yn, -yn])

def _sym_anneal_A(seeds, steps, k0, k1, lr0, lr1):
    _, JDX = _indices()
    sgnA = ((-1) ** np.arange(15)).astype(float)
    for it in range(steps):
        f = it / max(1, steps - 1)
        kk = k0 + (k1 - k0) * f
        lr = lr0 + (lr1 - lr0) * f
        X = _build_A(seeds)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        W = np.exp(kk * (G - m))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-30
        Gr = (W @ X) / rs
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        Gto = Gt[:75].reshape(5, 15, 15)
        # g[i,j] = sum_k Gto[i,k,JDX[k,j]] /15
        # vectorize over k with loop 15
        g = np.zeros_like(seeds)
        for k in range(15):
            g += Gto[:, k, JDX[k, :]]
        g /= 15.0
        seeds = seeds - lr * g
        seeds = seeds / (np.linalg.norm(seeds, axis=1, keepdims=True) + 1e-30)
    return seeds

def _sym_anneal_B(seeds, steps, k0, k1, lr0, lr1):
    _, JDX = _indices()
    for it in range(steps):
        f = it / max(1, steps - 1)
        kk = k0 + (k1 - k0) * f
        lr = lr0 + (lr1 - lr0) * f
        X = _build_B(seeds)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        W = np.exp(kk * (G - m))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-30
        Gr = (W @ X) / rs
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        Gto = Gt[:75].reshape(5, 15, 15)
        g = np.zeros_like(seeds)
        for k in range(15):
            sk = -1.0 if (k & 1) else 1.0
            g += sk * Gto[:, k, JDX[k, :]]
        g /= 15.0
        seeds = seeds - lr * g
        seeds = seeds / (np.linalg.norm(seeds, axis=1, keepdims=True) + 1e-30)
    return seeds

def _sym_anneal_C(Y, steps, k0, k1, lr0, lr1):
    for it in range(steps):
        f = it / max(1, steps - 1)
        kk = k0 + (k1 - k0) * f
        lr = lr0 + (lr1 - lr0) * f
        X = _build_C(Y)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        W = np.exp(kk * (G - m))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-30
        Gr = (W @ X) / rs
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        g = (Gt[:38] - Gt[38:]) * 0.5
        Y = Y - lr * g
        Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-30)
    return Y

def _polish(X, steps, lr0, lr1, k0, k1, gap):
    n = X.shape[0]
    for it in range(steps):
        f = it / max(1, steps - 1)
        lr = lr0 + (lr1 - lr0) * f
        kk = k0 + (k1 - k0) * f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        M = G >= (m - gap)
        W = np.exp(kk * (G - m)) * M
        np.fill_diagonal(W, 0.0)
        Gr = W @ X
        d = np.sum(Gr * X, axis=1, keepdims=True)
        Gr -= d * X
        X = X - lr * Gr
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-30)
    return X

def solve():
    np.random.seed(0)
    t0 = time.time()
    try:
        X0 = np.load("best.npy").astype(float)
    except Exception:
        try:
            X0 = np.load("record.npy").astype(float)
        except Exception:
            X0 = np.random.randn(76, 15)
    X0 = _norm(X0)
    if X0.shape != (76, 15):
        Y = np.random.randn(76, 15)
        X0 = _norm(Y)
    bestX = X0.copy()
    bestV = _maxcos(bestX)
    _indices()
    rng = np.random.default_rng(0)
    cands = []

    # A: C15 + fixed, 3 restarts
    for r in range(3):
        if time.time() - t0 > 100:
            break
        seeds = rng.normal(size=(5, 15))
        seeds = _norm(seeds)
        seeds = _sym_anneal_A(seeds, steps=900, k0=60.0, k1=350.0, lr0=0.35, lr1=0.03)
        if time.time() - t0 > 110:
            XA = _build_A(seeds)
            cands.append(XA)
            break
        seeds = _sym_anneal_A(seeds, steps=700, k0=350.0, k1=600.0, lr0=0.08, lr1=0.015)
        XA = _build_A(seeds)
        cands.append(XA)
    # B: signed cyclic, 2 restarts
    for r in range(2):
        if time.time() - t0 > 125:
            break
        seeds = rng.normal(size=(5, 15))
        seeds = _norm(seeds)
        seeds = _sym_anneal_B(seeds, steps=900, k0=60.0, k1=350.0, lr0=0.35, lr1=0.03)
        seeds = _sym_anneal_B(seeds, steps=600, k0=350.0, k1=600.0, lr0=0.08, lr1=0.015)
        cands.append(_build_B(seeds))
    # C: antipodal 38 pairs, 2 restarts (cheaper)
    for r in range(2):
        if time.time() - t0 > 145:
            break
        Y = rng.normal(size=(38, 15))
        Y = _norm(Y)
        Y = _sym_anneal_C(Y, steps=800, k0=60.0, k1=350.0, lr0=0.35, lr1=0.03)
        Y = _sym_anneal_C(Y, steps=500, k0=350.0, k1=600.0, lr0=0.08, lr1=0.015)
        cands.append(_build_C(Y))

    # unconstrained release of symmetric candidates
    for X in cands:
        if time.time() - t0 > 160:
            break
        X = X.copy()
        X = _polish(X, steps=700, lr0=0.02, lr1=0.003, k0=150.0, k1=500.0, gap=0.08)
        if time.time() - t0 > 163:
            v = _maxcos(X)
            if v < bestV:
                bestV, bestX = v, X.copy()
            break
        X = _polish(X, steps=500, lr0=0.008, lr1=0.001, k0=500.0, k1=1000.0, gap=0.04)
        v = _maxcos(X)
        if v < bestV:
            bestV, bestX = v, X.copy()

    # final squeeze of global best
    if time.time() - t0 < 170:
        Z = bestX.copy()
        rem = 170 - (time.time() - t0)
        steps = 800 if rem > 12 else 300
        Z = _polish(Z, steps=steps, lr0=0.006, lr1=0.0008, k0=500.0, k1=1200.0, gap=0.035)
        v = _maxcos(Z)
        if v < bestV:
            bestV, bestX = v, Z.copy()

    bestX = _norm(bestX)
    return bestX
