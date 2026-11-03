import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _full_adam(X0, steps, beta0, beta1, lr0, lr1, t_end):
    X = X0.copy()
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    best = _max_cos(X)
    bestX = X.copy()
    pow1 = b1
    pow2 = b2
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    den = max(steps - 1, 1)
    for t in range(1, steps + 1):
        if (t % 50) == 0:
            if time.time() > t_end:
                break
        frac = (t - 1) / den
        beta = float(np.exp(lb0 + (lb1 - lb0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gm = float(G.max())
        W = np.exp(beta * (G - gm))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-16
        D = (W @ X) / rs
        dot = (D * X).sum(axis=1, keepdims=True)
        Gt = D - dot * X
        m *= b1; m += (1.0 - b1) * Gt
        v *= b2; v += (1.0 - b2) * (Gt * Gt)
        mhat = m / (1.0 - pow1); vhat = v / (1.0 - pow2)
        pow1 *= b1; pow2 *= b2
        step = lr * mhat / (np.sqrt(vhat) + eps)
        rn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        scale = np.minimum(1.0, 0.05 / rn)
        X = X - step * scale
        X = _normalize(X)
        if (t % 50) == 0:
            cur = _max_cos(X)
            if cur < best:
                best = cur
                bestX = X.copy()
    cur = _max_cos(X)
    if cur < best:
        best = cur; bestX = X.copy()
    return bestX, best

def _refine_batch_against_frozen(B0, Base, steps, beta0, beta1, lr0, lr1, t_end):
    # B0: (nb=100, d), Base: (m, d) frozen. Minimise per-row max dot with Base via LSE-Adam.
    B = B0.copy()
    nb, d = B.shape
    m = np.zeros_like(B)
    v = np.zeros_like(B)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    pow1 = b1; pow2 = b2
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    den = max(steps - 1, 1)
    for t in range(1, steps + 1):
        if (t % 50) == 0 and time.time() > t_end:
            break
        frac = (t - 1) / den
        beta = float(np.exp(lb0 + (lb1 - lb0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = B @ Base.T  # nb x m
        gm = G.max(axis=1, keepdims=True)
        W = np.exp(beta * (G - gm))
        rs = W.sum(axis=1, keepdims=True) + 1e-16
        D = (W @ Base) / rs
        dot = (D * B).sum(axis=1, keepdims=True)
        Gt = D - dot * B
        m *= b1; m += (1.0 - b1) * Gt
        v *= b2; v += (1.0 - b2) * (Gt * Gt)
        mhat = m / (1.0 - pow1); vhat = v / (1.0 - pow2)
        pow1 *= b1; pow2 *= b2
        step = lr * mhat / (np.sqrt(vhat) + eps)
        rn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        scale = np.minimum(1.0, 0.05 / rn)
        B = B - step * scale
        B = _normalize(B)
    return _normalize(B)

def solve():
    t0 = time.time()
    BUDGET = 160.0
    t_end = t0 + BUDGET
    rng = np.random.default_rng(0)
    n, d = 59, 22

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                a = _normalize(a)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_normalize(rng.standard_normal((n, d))))
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    gbestX = cands[bi].copy()
    gbest = float(vals[bi])

    # precomputed pool >= 20000 random unit vectors
    POOL_N = 20000
    Pool = _normalize(rng.standard_normal((POOL_N, d)))

    # short initial polish to establish tight baseline (cheap)
    try:
        if time.time() < t_end - 10.0:
            Xp, vp = _full_adam(gbestX, 600, 40.0, 250.0, 0.012, 0.004, t_end - 5.0)
            if vp < gbest:
                gbest, gbestX = vp, Xp.copy()
    except Exception:
        pass

    # helper to compute contact ordering
    def contact_info(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mm = float(G.max())
        tol = 0.006
        cnt = (G > (mm - tol)).sum(axis=1)
        rowmax = G.max(axis=1)
        return G, mm, cnt, rowmax

    G, mm, cnt, rowmax = contact_info(gbestX)
    # hubs sorted by contacts then rowmax
    hub_order = np.argsort(-cnt - 2.0 * rowmax)

    n_trials = 0
    max_trials = 30
    # cycle through several clusters
    trial = 0
    while trial < max_trials:
        if time.time() > t_end - 5.0:
            break
        # refresh contact info from current best periodically
        if (trial % 3) == 0:
            G, mm, cnt, rowmax = contact_info(gbestX)
            hub_order = np.argsort(-cnt - 2.0 * rowmax + 0.001 * rng.standard_normal(n))
        # choose high-contact point cyclically among top 12
        topk = hub_order[:12]
        hub = int(topk[trial % len(topk)])
        # neighbours in contact graph
        thr = mm - 0.006
        neigh = np.where(G[hub] > thr)[0]
        neigh = neigh[neigh != hub]
        if neigh.size < 5:
            # widen threshold
            thr2 = mm - 0.012
            neigh = np.where(G[hub] > thr2)[0]
            neigh = neigh[neigh != hub]
        if neigh.size < 5:
            trial += 1
            continue
        # total cluster size 6..10 -> neighbours 5..9
        # vary size: 6,7,8,9 cyclic
        tot = 6 + (trial % 4)  # 6,7,8,9
        nn = min(tot - 1, int(neigh.size))
        nn = max(nn, 5)
        # pick nn neighbours randomly
        perm = rng.permutation(neigh.size)
        chosen = neigh[perm[:nn]]
        victims = np.array([hub] + chosen.tolist(), dtype=int)
        k = int(victims.size)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[victims] = False
        Frozen0 = gbestX[keep_mask]  # (59-k) x d, 50..53 rows

        # sequential reinsertion, one vacancy at a time
        placed = []
        feasible = True
        for s in range(k):
            if time.time() > t_end - 4.0:
                feasible = False
                break
            if len(placed) == 0:
                Base = Frozen0
            else:
                Base = np.vstack([Frozen0] + placed)
            # one matrix multiplication: score pool by max inner product with frozen set
            # chunk to bound memory
            best_idx = None
            # compute scores in chunks then argpartition globally
            # do single matmul in chunks of 5000
            scores = np.empty(POOL_N, dtype=float)
            cs = 5000
            for a in range(0, POOL_N, cs):
                Pb = Pool[a:a+cs]
                sc = (Pb @ Base.T).max(axis=1)
                scores[a:a+Pb.shape[0]] = sc
            # best 100 candidates
            part = np.argpartition(scores, 100)[:100]
            # order them by score (optional)
            o = np.argsort(scores[part])
            cand_idx = part[o]
            B0 = Pool[cand_idx].copy()  # 100 x d
            # locally optimise only reinserted points against frozen set: 300-500 annealed steps
            # use 350 steps, anneal beta 30->300, lr 0.02->0.002
            Bref = _refine_batch_against_frozen(B0, Base, 350, 30.0, 300.0, 0.02, 0.002, t_end - 3.0)
            # pick best refined candidate by true max
            sc2 = (Bref @ Base.T).max(axis=1)
            bi2 = int(np.argmin(sc2))
            best_vec = Bref[bi2].copy().reshape(1, d)
            placed.append(best_vec)
            # time check every vacancy; overall check every 50 iters is inside refiners/polish
        if not feasible:
            trial += 1
            continue
        # build complete configuration
        Ynew = gbestX.copy()
        for ii, vi in enumerate(victims):
            Ynew[int(vi)] = placed[ii][0]
        Ynew = _normalize(Ynew)
        # unfreeze everything for short 1500-step polish
        if time.time() > t_end - 4.0:
            break
        Xpol, vpol = _full_adam(Ynew, 1500, 40.0, 320.0, 0.014, 0.0025, t_end - 1.0)
        # accept only if fully polished maximum improves; always retain best
        if vpol < gbest:
            gbest, gbestX = vpol, Xpol.copy()
            G, mm, cnt, rowmax = contact_info(gbestX)
            hub_order = np.argsort(-cnt - 2.0 * rowmax + 0.001 * rng.standard_normal(n))
        trial += 1
        n_trials += 1
        # periodic time check (every iteration covers ~50-step inner checks)
        if time.time() > t_end - 3.0:
            break

    # final safety: ensure normalised, retain best
    gbestX = _normalize(gbestX)
    return gbestX
