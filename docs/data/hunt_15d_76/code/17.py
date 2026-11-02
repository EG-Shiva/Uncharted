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

def _anneal(X, n_iter, T0, T1, lr0, lr1):
    # projected soft-max repulsion: minimise T*logsumexp(G/T)
    n, d = X.shape
    for k in range(n_iter):
        f = k / max(1, n_iter - 1)
        T = T0 * (T1 / T0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        # exp((G-m)/T), stable (<=1)
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = E.sum()
        if S <= 0 or not np.isfinite(S):
            break
        Gr = (E @ X) / S
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        # clip huge steps
        X = X - lr * Gt
        X = _norm(X)
    return X

def _refine_hole(v, Y, n_iter=300, T=0.005, lr=0.05):
    # minimise max_j v.Y_j by soft-max descent on sphere
    v = v / np.linalg.norm(v)
    for k in range(n_iter):
        f = k / max(1, n_iter - 1)
        Tk = T * (0.2 ** f)
        lrk = lr * (0.2 ** f)
        s = Y @ v  # (m,)
        m = s.max()
        w = np.exp((s - m) / Tk)
        w = w / w.sum()
        g = w @ Y  # centroid direction to avoid
        g = g - np.dot(g, v) * v
        ng = np.linalg.norm(g)
        if ng > 1e-12:
            g = g / (ng + 1e-12) * min(1.0, ng * 5.0)
        v = v - lrk * g
        nv = np.linalg.norm(v)
        if nv == 0 or not np.isfinite(nv):
            break
        v = v / nv
    return v

def solve():
    rng = np.random.default_rng(0)
    t0 = time.time()
    BUDGET = 168.0  # leave margin for 180s limit

    try:
        Xb = np.load("best.npy")
    except Exception:
        Xb = np.load("record.npy")
    Xb = np.asarray(Xb, dtype=np.float64)
    if Xb.shape != (76, 15):
        # fallback random
        Xb = rng.normal(size=(76, 15))
    Xb = _norm(Xb)
    best = _maxcos(Xb)
    Xbest = Xb.copy()

    n, d = 76, 15

    # quick contact count to identify rattlers/overloaded (for reseeding)
    def contacts(X, thr=1e-4):
        G = X @ X.T
        np.fill_diagonal(G, -2)
        m = G.max()
        C = (G > m - thr)
        return C.sum(axis=1), m

    # main basin-hopping loop
    trial = 0
    # pre-allocate
    while True:
        if time.time() - t0 > BUDGET - 12:
            break
        trial += 1
        cnt, mcur = contacts(Xbest)
        # choose perturbation strength cycling
        sigmas = [0.004, 0.01, 0.025, 0.06, 0.12]
        sigma = sigmas[(trial - 1) % len(sigmas)]

        Xt = Xbest.copy()
        # pick points to reinsert: mix of random + high-contact + rattlers
        # 1-3 points per trial
        k_re = 2 if (trial % 2 == 0) else 1
        if trial % 3 == 0:
            k_re = 3
        # indices: prefer overloaded then random
        order = np.argsort(-cnt)
        cand_idx = []
        # always include one overloaded
        cand_idx.append(int(order[(trial - 1) % n]))
        for _ in range(k_re - 1):
            cand_idx.append(int(rng.integers(0, n)))
        cand_idx = list(dict.fromkeys(cand_idx))

        # hole-optimized reinsertion for chosen indices (sequential)
        for idx in cand_idx:
            if time.time() - t0 > BUDGET - 12:
                break
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            Y = Xt[mask]
            # sample candidates, pick minimax
            NS = 2500
            C = rng.normal(size=(NS, d))
            C = _norm(C)
            scores = (C @ Y.T).max(axis=1)
            bi = int(np.argmin(scores))
            v0 = C[bi]
            v0 = _refine_hole(v0, Y, n_iter=250, T=0.008, lr=0.08)
            Xt[idx] = v0

        # gaussian shake on the rest
        noise = rng.normal(size=(n, d))
        Xt = Xt + sigma * noise
        Xt = _norm(Xt)

        # anneal: params depend on sigma
        if sigma < 0.02:
            Xt = _anneal(Xt, n_iter=2500, T0=0.012, T1=0.0012, lr0=4.0, lr1=0.15)
        elif sigma < 0.07:
            Xt = _anneal(Xt, n_iter=3000, T0=0.016, T1=0.0010, lr0=5.0, lr1=0.12)
        else:
            Xt = _anneal(Xt, n_iter=3500, T0=0.022, T1=0.0015, lr0=6.0, lr1=0.15)

        if time.time() - t0 > BUDGET - 10:
            # evaluate anyway
            pass
        v = _maxcos(Xt)
        if v < best - 1e-12:
            best = v
            Xbest = Xt.copy()
            # extra greedy polish immediately on improvement
            if time.time() - t0 < BUDGET - 10:
                Xbest = _anneal(Xbest, n_iter=2500, T0=0.0012, T1=0.00025, lr0=0.8, lr1=0.05)
                v2 = _maxcos(Xbest)
                if v2 < best:
                    best = v2
                else:
                    # keep previous if polish hurt (re-evaluate)
                    pass
        # early exit if clearly new record with margin
        # continue searching for even better until budget
        if trial > 60:
            break

    # final fine polish on best
    if time.time() - t0 < BUDGET - 2:
        Xbest = _anneal(Xbest, n_iter=4000, T0=0.0010, T1=0.0002, lr0=0.6, lr1=0.03)
        v = _maxcos(Xbest)
        if v < best:
            best = v
        else:
            # if polish worsened due to numerics, keep earlier? we already overwrote;
            # do a short re-polish from stored? just keep as is if close
            pass

    Xbest = _norm(Xbest)
    # safety: remove NaN
    Xbest = np.nan_to_num(Xbest)
    Xbest = _norm(Xbest)
    return Xbest
