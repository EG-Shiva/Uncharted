import time
import numpy as np

def _norm(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return A / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _smooth(X0, steps, beta0, beta1, lr0, lr1, mu):
    X = X0.copy()
    n = X.shape[0]
    vel = np.zeros_like(X)
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    best_val = _max_cos(X)
    best_X = X.copy()
    for s in range(steps):
        f = s / float(max(1, steps - 1))
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5)
        if S < 1e-30:
            continue
        Gr = (W @ X) / S
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gr -= X * dot
        vel = mu * vel + (1.0 - mu) * Gr
        X -= lr * vel
        rn = np.linalg.norm(X, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        X /= rn
        if s % 200 == 199 or s == steps - 1:
            v = _max_cos(X)
            if v < best_val:
                best_val = v
                best_X = X.copy()
    return best_X, best_val

def _hard(X0, steps, lr0, lr1, mu, marg0, marg1):
    X = X0.copy()
    vel = np.zeros_like(X)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    lm0 = np.log(marg0); lm1 = np.log(marg1)
    best_val = _max_cos(X)
    best_X = X.copy()
    for s in range(steps):
        f = s / float(max(1, steps - 1))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        marg = float(np.exp(lm0 + (lm1 - lm0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = (G >= (m - marg)).astype(np.float64)
        np.fill_diagonal(W, 0.0)
        deg = W.sum(axis=1, keepdims=True)
        Gr = (W @ X) / np.maximum(deg, 1.0)
        zm = (deg[:, 0] == 0)
        if np.any(zm):
            Gr[zm, :] = 0.0
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gr -= X * dot
        vel = mu * vel + (1.0 - mu) * Gr
        X -= lr * vel
        rn = np.linalg.norm(X, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        X /= rn
        if s % 200 == 199 or s == steps - 1:
            v = _max_cos(X)
            if v < best_val:
                best_val = v
                best_X = X.copy()
    return best_X, best_val

def _hole_fill(Xbest, idx, rng, n_cand=3500):
    n, d = Xbest.shape
    X = Xbest.copy()
    mask = np.ones(n, dtype=bool)
    mask[idx] = False
    # order to fill: random
    order = np.array(idx, copy=True)
    rng.shuffle(order)
    for ii in order:
        # fixed set = all except ii (including not-yet-filled which are stale, exclude them)
        # build fixed matrix of points not in idx plus already filled
        fixed_idx = [j for j in range(n) if j != ii and (j not in set(idx) or np.any(order[:list(order).index(ii)] == j) if False else True)]
        # simpler: fixed = all rows except those in idx not yet filled and ii
        # track filled set
        pass
    # iterative greedy using filled tracking
    remaining = set(idx.tolist() if isinstance(idx, np.ndarray) else list(idx))
    filled = set()
    for ii in order:
        # fixed indices: complement of remaining (already placed) plus filled, excluding ii
        fixed = [j for j in range(n) if j not in remaining or j in filled]
        # actually ii in remaining, so fixed = [j not in remaining] + list(filled)
        # which equals complement of (remaining - filled - {ii}?) let's just compute:
        # fixed = all j != ii with j not in (remaining - filled - {ii}... )
        # Easier: fixed = [j for j in range(n) if j != ii and (j not in remaining or j in filled)]
        fixed = [j for j in range(n) if j != ii and (j not in remaining or j in filled)]
        F = X[fixed] if len(fixed) > 0 else np.zeros((0, d))
        C = rng.standard_normal((n_cand, d))
        C = _norm(C)
        if F.shape[0] > 0:
            D = C @ F.T
            scores = D.max(axis=1)
        else:
            scores = np.zeros(n_cand)
        b = int(np.argmin(scores))
        X[ii] = C[b]
        filled.add(int(ii))
        remaining.discard(int(ii))
    return X

def solve():
    t0 = time.time()
    rng = np.random.default_rng(1)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (59, 22):
                continue
            if not np.all(np.isfinite(a)):
                continue
            a = _norm(a)
            cands.append(a)
        except Exception:
            continue
    if not cands:
        Xbest = _norm(rng.standard_normal((59, 22)))
    else:
        vals = [_max_cos(c) for c in cands]
        Xbest = cands[int(np.argmin(vals))].copy()
    best_val = _max_cos(Xbest)

    # quick initial polish
    try:
        y, v = _smooth(Xbest, steps=2200, beta0=40.0, beta1=400.0, lr0=0.015, lr1=0.0012, mu=0.88)
        if v < best_val:
            best_val = v; Xbest = y.copy()
        y, v = _hard(Xbest, steps=900, lr0=0.004, lr1=0.0003, mu=0.85, marg0=0.005, marg1=0.0007)
        if v < best_val:
            best_val = v; Xbest = y.copy()
    except Exception:
        pass

    ks = [4, 7, 3, 10, 5, 8, 2, 12, 6, 4]
    trial = 0
    # main basin-hopping loop, leave ~20s for final polish
    while True:
        if time.time() - t0 > 138.0:
            break
        k = ks[trial % len(ks)]
        trial += 1
        try:
            G = Xbest @ Xbest.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            rowmax = G.max(axis=1)
            deg = (G >= (m - 0.002)).sum(axis=1)
            order_worst = np.argsort(-rowmax - 0.001 * deg.astype(float))
            n_worst = min(k, 59, max(1, k // 2 + 1))
            S = set(order_worst[:n_worst].tolist())
            while len(S) < k:
                S.add(int(rng.integers(0, 59)))
            Sarr = np.array(sorted(S), dtype=int)
            # occasionally pure random block
            if trial % 9 == 0:
                Sarr = np.sort(rng.choice(59, size=k, replace=False))
            Xt = _hole_fill(Xbest, Sarr, rng, n_cand=3000)
            # small shake on untouched points to break rigidity
            noise = rng.standard_normal((59, 22)) * 0.004
            Xt = Xt + noise
            Xt = _norm(Xt)
            # short anneal
            y, v = _smooth(Xt, steps=1300, beta0=60.0, beta1=500.0, lr0=0.008, lr1=0.001, mu=0.88)
            y2, v2 = _hard(y, steps=600, lr0=0.003, lr1=0.0003, mu=0.85, marg0=0.004, marg1=0.0006)
            if v2 < best_val - 1e-12:
                best_val = v2; Xbest = y2.copy()
            elif v < best_val:
                best_val = v; Xbest = y.copy()
            else:
                # opportunistic second try with global shake if hole fill failed and time allows
                if trial % 4 == 0 and (time.time() - t0 < 120.0):
                    sig = 0.02 if (trial % 8 < 4) else 0.05
                    Xs = _norm(Xbest + sig * rng.standard_normal((59, 22)))
                    ys, vs = _smooth(Xs, steps=900, beta0=80.0, beta1=500.0, lr0=0.006, lr1=0.001, mu=0.88)
                    if vs < best_val:
                        best_val = vs; Xbest = ys.copy()
        except Exception:
            continue

    # final polish of best
    try:
        if time.time() - t0 < 165.0:
            y, v = _smooth(Xbest, steps=2500, beta0=120.0, beta1=900.0, lr0=0.004, lr1=0.0004, mu=0.9)
            if v < best_val:
                best_val = v; Xbest = y.copy()
        if time.time() - t0 < 170.0:
            y, v = _hard(Xbest, steps=800, lr0=0.0015, lr1=0.00015, mu=0.85, marg0=0.002, marg1=0.0004)
            if v < best_val:
                best_val = v; Xbest = y.copy()
    except Exception:
        pass

    Xbest = _norm(Xbest)
    return Xbest
