import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _polish(X, steps, lr0, lr1, k0, k1, gap, t_end=None, t0=None):
    n, d = X.shape
    for it in range(steps):
        if t_end is not None and (it & 63) == 0 and (time.time() - t0) > t_end:
            break
        f = it / max(1, steps - 1)
        lr = lr0 + (lr1 - lr0) * f
        k = k0 + (k1 - k0) * f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        # only near-max pairs drive gradient
        thr = m - gap
        # exp weights, stable because G-m <= 0
        # mask to sparsify
        M = G >= thr
        W = np.exp(k * (G - m)) * M
        # gradient of sum exp(k*dot)
        Gr = W @ X
        # tangential projection
        d = np.sum(Gr * X, axis=1, keepdims=True)
        Gr -= d * X
        # clip step for stability
        # scale by typical magnitude
        X = X - lr * Gr
        # renormalize
        nn = np.linalg.norm(X, axis=1, keepdims=True)
        nn[nn == 0] = 1.0
        X /= nn
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
    n, d = X0.shape
    # ensure shape (76,15)
    if X0.shape != (76, 15):
        Y = np.random.randn(76, 15)
        Y = _norm(Y)
        # copy as much as possible
        m = min(n, 76)
        mm = min(d, 15)
        Y[:m, :mm] = X0[:m, :mm]
        X0 = _norm(Y)
        n, d = 76, 15

    bestX = X0.copy()
    bestV = _max_cos(bestX)

    # quick contact analysis for targeted shakes
    def contacts(X, thr=1e-9):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        C = (G >= m - thr).sum(axis=1)  # includes? diag is -2 so no
        return C, m

    # basin hopping: fixed small number of trials to guarantee time
    # alternate small / medium / large perturbations
    NTRIAL = 22
    for trial in range(NTRIAL):
        if time.time() - t0 > 140:
            break
        if trial == 0:
            Y = bestX.copy()
            Y = _polish(Y, steps=800, lr0=0.02, lr1=0.002, k0=150.0, k1=600.0, gap=0.08)
            v = _max_cos(Y)
            if v < bestV:
                bestV, bestX = v, Y.copy()
            continue
        # perturbation strength
        if trial % 4 == 0:
            sigma = 0.10
        elif trial % 4 == 1:
            sigma = 0.05
        elif trial % 4 == 2:
            sigma = 0.02
        else:
            sigma = 0.008
        Y = bestX + sigma * np.random.randn(n, d)
        Y = _norm(Y)
        # rattler / overload teleport every other trial
        if trial % 2 == 0:
            try:
                C, _ = contacts(bestX)
                ratt = np.where(C == 0)[0]
                over = np.argsort(-C)[:4]
                # teleport 2 rattlers to random points
                if len(ratt) >= 2:
                    idx = np.random.choice(ratt, size=min(3, len(ratt)), replace=False)
                    R = np.random.randn(len(idx), d)
                    R = _norm(R)
                    Y[idx] = R
                # shake overloaded neighbourhood
                sh = over[np.random.randint(len(over))]
                Y[sh] += 0.15 * np.random.randn(d)
                Y = _norm(Y)
            except Exception:
                pass
        # two-stage anneal: coarse then fine
        Y = _polish(Y, steps=900, lr0=0.03, lr1=0.005, k0=120.0, k1=400.0, gap=0.10)
        Y = _polish(Y, steps=600, lr0=0.01, lr1=0.001, k0=400.0, k1=800.0, gap=0.05)
        v = _max_cos(Y)
        if v < bestV:
            bestV, bestX = v, Y.copy()

    # final squeeze on current best with small lr, large k
    if time.time() - t0 < 160:
        Z = bestX.copy()
        Z = _polish(Z, steps=1200, lr0=0.008, lr1=0.0008, k0=500.0, k1=1200.0, gap=0.04)
        v = _max_cos(Z)
        if v < bestV:
            bestV, bestX = v, Z.copy()

    bestX = _norm(bestX)
    return bestX
