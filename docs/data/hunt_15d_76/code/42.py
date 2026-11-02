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

def _anneal_balanced(X, steps, T0, T1, lr0, lr1, alpha=0.5):
    n, d = X.shape
    for k in range(steps):
        f = k / max(1, steps - 1)
        T = T0 * (T1 / T0) ** f if T1 > 0 else T0
        lr = lr0 * (lr1 / lr0) ** f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        # global softmax weights
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if S <= 0 or not np.isfinite(S):
            break
        Wg = E / S
        # per-row softmax (balanced load)
        mrow = G.max(axis=1, keepdims=True)
        Er = np.exp((G - mrow) / T)
        np.fill_diagonal(Er, 0.0)
        rs = Er.sum(axis=1, keepdims=True)
        rs[rs <= 0] = 1.0
        Wr = Er / rs  # row stochastic
        # symmetrize
        Ws = 0.5 * (Wr + Wr.T)
        Ss = float(Ws.sum())
        if Ss > 0 and np.isfinite(Ss):
            Ws = Ws / Ss * S  # same scale as E? we want convex combo of normalized
            # actually normalize both to sum 1 then blend
            Wg1 = Wg  # sums to 1
            Ws1 = Ws / float(Ws.sum())
            W = (1.0 - alpha) * Wg1 + alpha * Ws1
        else:
            W = Wg
        Gr = (W @ X) / 1.0  # W sums to 1
        # W sums to 1, but gradient scale: d/dX T logsum = ... ; absorb into lr
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        # limit step
        X = X - lr * Gt
        X = _norm(X)
    return X

def _refine_hole(v, Y, steps=200, T0=0.008, T1=0.001, lr0=0.08, lr1=0.005):
    v = v / np.linalg.norm(v)
    for k in range(steps):
        f = k / max(1, steps - 1)
        T = T0 * (T1 / T0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        s = Y @ v
        m = float(s.max())
        w = np.exp((s - m) / T)
        sw = float(w.sum())
        if sw <= 0 or not np.isfinite(sw):
            break
        w /= sw
        g = w @ Y
        gd = float(g @ v)
        gt = g - gd * v
        v = v - lr * gt
        nv = float(np.linalg.norm(v))
        if nv == 0 or not np.isfinite(nv):
            break
        v /= nv
    return v

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.default_rng(0)
    try:
        Xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            Xb = np.load("record.npy").astype(float)
        except Exception:
            Xb = rng.normal(size=(76, 15))
    Xb = np.asarray(Xb, dtype=np.float64)
    if Xb.shape != (76, 15):
        Xb = rng.normal(size=(76, 15))
    Xb = _norm(Xb)
    best_val = _maxcos(Xb)
    Xbest = Xb.copy()
    n, d = 76, 15

    # contact counts for targeting
    def rowmax(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return G.max(axis=1), float(G.max())

    trial = 0
    # fast hopping; each trial cheap (~800 anneal steps)
    while time.time() - t0 < BUDGET - 14:
        trial += 1
        rm, gm = rowmax(Xbest)
        # cycle sigma
        sigmas = [0.015, 0.04, 0.09, 0.18, 0.06, 0.025]
        sigma = sigmas[(trial - 1) % len(sigmas)]
        Xt = Xbest.copy()
        # choose 1-3 indices to hole-reinsert: worst rowmax + random + rattler-like
        k_re = 1 + (trial % 3)  # 1..3 cycle (2,3,1...)
        if trial % 4 == 0:
            k_re = 3
        order = np.argsort(-rm)
        cand = []
        cand.append(int(order[(trial - 1) % n]))
        # also a random one
        cand.append(int(rng.integers(0, n)))
        if k_re >= 3:
            cand.append(int(rng.integers(0, n)))
        # dedup
        cand = list(dict.fromkeys(cand))[:k_re]
        for idx in cand:
            if time.time() - t0 > BUDGET - 14:
                break
            mask = np.ones(n, dtype=bool); mask[idx] = False
            Y = Xt[mask]
            NS = 2000
            C = rng.normal(size=(NS, d))
            C = _norm(C)
            scores = (C @ Y.T).max(axis=1)
            bi = int(np.argmin(scores))
            v0 = C[bi]
            v0 = _refine_hole(v0, Y, steps=180, T0=0.008, T1=0.001, lr0=0.08, lr1=0.005)
            Xt[idx] = v0
        # gaussian shake (subset for small sigma, all for large)
        if sigma < 0.05:
            # shake all lightly
            Xt = Xt + sigma * rng.normal(size=(n, d))
        else:
            # shake all more
            Xt = Xt + sigma * rng.normal(size=(n, d))
        Xt = _norm(Xt)
        # anneal params by sigma
        if sigma < 0.03:
            steps, T0a, T1a, lr0a, lr1a = 800, 0.010, 0.0010, 3.0, 0.12
        elif sigma < 0.10:
            steps, T0a, T1a, lr0a, lr1a = 1000, 0.014, 0.0008, 4.0, 0.10
        else:
            steps, T0a, T1a, lr0a, lr1a = 1200, 0.018, 0.0010, 5.0, 0.12
        # shrink steps if low time
        remain = BUDGET - 14 - (time.time() - t0)
        if remain < 20:
            steps = min(steps, 600)
        Xt = _anneal_balanced(Xt, steps, T0a, T1a, lr0a, lr1a, alpha=0.5)
        v = _maxcos(Xt)
        if v < best_val - 1e-12:
            best_val = v
            Xbest = Xt.copy()
            # extra polish on improvement if time allows
            if time.time() - t0 < BUDGET - 16:
                Xbest = _anneal_balanced(Xbest, 900, 0.0012, 0.00025, 0.7, 0.05, alpha=0.4)
                v2 = _maxcos(Xbest)
                if v2 < best_val:
                    best_val = v2
                # else keep polished anyway if very close? keep best
                if v2 > best_val + 1e-9:
                    # revert polish if hurt (keep previous Xt)
                    Xbest = Xt.copy()
                    best_val = v
        # occasional fresh diverse restart with short anneal to seek new basin
        if trial % 12 == 0 and time.time() - t0 < BUDGET - 30:
            Xr = rng.normal(size=(n, d)); Xr = _norm(Xr)
            Xr = _anneal_balanced(Xr, 900, 0.02, 0.001, 5.0, 0.15, alpha=0.5)
            vr = _maxcos(Xr)
            if vr < best_val - 1e-12:
                best_val = vr; Xbest = Xr.copy()

    # final low-T polish (greedy exact-max aware via small T)
    if time.time() - t0 < BUDGET - 2:
        try:
            Xbest = _anneal_balanced(Xbest, 1200, 0.0009, 0.0002, 0.5, 0.04, alpha=0.3)
            best_val = _maxcos(Xbest)
        except Exception:
            pass
    # tiny shrink guard + renormalize
    Xbest = _norm(Xbest)
    return Xbest
