import time
import numpy as np

def _norm(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return A / n

def _max_cos(Y):
    G = Y @ Y.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _full_polish(Y0, steps, beta0, beta1, lr0, lr1, deadline, check_every=50):
    Y = Y0.copy()
    n, d = Y.shape
    M = np.zeros_like(Y)
    V = np.zeros_like(Y)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    G0 = Y @ Y.T
    np.fill_diagonal(G0, -10.0)
    best_val = float(G0.max())
    best_Y = Y.copy()
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    b1t = 1.0; b2t = 1.0
    for t in range(1, steps + 1):
        if (t % check_every) == 0 and time.time() > deadline:
            break
        frac = (t - 1) / float(max(1, steps - 1))
        beta = float(np.exp(lb0 + (lb1 - lb0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = Y @ Y.T
        np.fill_diagonal(G, -10.0)
        gmax = float(G.max())
        if gmax < best_val:
            best_val = gmax
            best_Y = Y.copy()
        G -= gmax
        G *= beta
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        s = float(G.sum())
        if s <= 0 or not np.isfinite(s):
            continue
        G /= s
        Gr = G @ Y
        dot = np.sum(Gr * Y, axis=1, keepdims=True)
        Gr -= Y * dot
        b1t *= b1; b2t *= b2
        M *= b1; M += (1.0 - b1) * Gr
        V *= b2; V += (1.0 - b2) * (Gr * Gr)
        Mh = M / (1.0 - b1t)
        Vh = V / (1.0 - b2t)
        Y -= lr * Mh / (np.sqrt(Vh) + eps)
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        Y /= rn
    Gf = Y @ Y.T
    np.fill_diagonal(Gf, -10.0)
    vf = float(Gf.max())
    if vf < best_val:
        best_val = vf
        best_Y = Y.copy()
    return best_Y, best_val

def _partial_polish(V0, F, steps, beta0, beta1, lr0, lr1, deadline, check_every=50):
    V = V0.copy()
    mv = V.shape[0]
    M = np.zeros_like(V)
    VV = np.zeros_like(V)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    best_val = 1e9
    best_V = V.copy()
    # init val
    Gc0 = V @ F.T
    Gs0 = V @ V.T
    np.fill_diagonal(Gs0, -10.0)
    m0 = float(max(float(Gc0.max()), float(Gs0.max())))
    best_val = m0
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    b1t = 1.0; b2t = 1.0
    for t in range(1, steps + 1):
        if (t % check_every) == 0 and time.time() > deadline:
            break
        frac = (t - 1) / float(max(1, steps - 1))
        beta = float(np.exp(lb0 + (lb1 - lb0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        Gc = V @ F.T
        Gs = V @ V.T
        np.fill_diagonal(Gs, -10.0)
        m = float(max(float(Gc.max()), float(Gs.max())))
        if m < best_val:
            best_val = m
            best_V = V.copy()
        Wc = np.exp(beta * (Gc - m))
        Ws = np.exp(beta * (Gs - m))
        np.fill_diagonal(Ws, 0.0)
        S = float(Wc.sum() + 0.5 * Ws.sum())
        if S <= 0 or not np.isfinite(S):
            continue
        Gr = (Wc @ F + Ws @ V) / S
        dot = np.sum(Gr * V, axis=1, keepdims=True)
        Gr -= V * dot
        b1t *= b1; b2t *= b2
        M *= b1; M += (1.0 - b1) * Gr
        VV *= b2; VV += (1.0 - b2) * (Gr * Gr)
        Mh = M / (1.0 - b1t)
        Vh = VV / (1.0 - b2t)
        V -= lr * Mh / (np.sqrt(Vh) + eps)
        rn = np.linalg.norm(V, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        V /= rn
    # final eval
    Gc = best_V @ F.T
    Gs = best_V @ best_V.T
    np.fill_diagonal(Gs, -10.0)
    # also evaluate current V
    Gc2 = V @ F.T
    Gs2 = V @ V.T
    np.fill_diagonal(Gs2, -10.0)
    m1 = float(max(float(Gc.max()), float(Gs.max())))
    m2 = float(max(float(Gc2.max()), float(Gs2.max())))
    if m2 < m1:
        return V.copy(), m2
    else:
        return best_V, m1

def solve():
    rng = np.random.default_rng(0)
    t0 = time.time()
    deadline = t0 + 160.0

    X0 = None
    best0 = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (59, 22):
                continue
            if not np.all(np.isfinite(a)):
                continue
            a = _norm(a)
            v = _max_cos(a)
            if v < best0:
                best0 = v
                X0 = a.copy()
        except Exception:
            continue
    if X0 is None:
        X0 = _norm(rng.standard_normal((59, 22)))
        best0 = _max_cos(X0)

    x_best = X0.copy()
    best_val = best0

    # precomputed pool >=20000
    NPOOL = 20000
    pool = _norm(rng.standard_normal((NPOOL, 22)))

    # initial short polish to settle baseline
    if time.time() < deadline - 20.0:
        y, v = _full_polish(x_best, steps=2000, beta0=60.0, beta1=600.0,
                            lr0=0.008, lr1=0.001, deadline=deadline)
        if v < best_val:
            best_val = v
            x_best = y.copy()

    # main cluster excision loop
    attempt = 0
    # degrees recomputed periodically
    while True:
        if time.time() > deadline - 12.0:
            break
        # recompute contact graph of current best every outer round
        G = x_best @ x_best.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # neighbours: tight pairs
        margin = 0.002
        # degree
        deg = np.sum(G > (m - margin), axis=1)
        order = np.argsort(-deg)
        # cycle through several high-contact seeds
        nseeds = min(20, 59)
        improved_round = False
        for si in range(nseeds):
            if time.time() > deadline - 12.0:
                break
            seed = int(order[si])
            # contact neighbours sorted by closeness
            row = G[seed].copy()
            # exclude self already -2
            nbr_order = np.argsort(-row)
            # take top 15 as candidate neighbours (skip self)
            cand_nbrs = [int(c) for c in nbr_order if int(c) != seed][:15]
            k = 6 + (attempt % 4)  # 6..9
            nn = k - 1
            # choose nn neighbours: mix of top + random among 15 for diversity
            if len(cand_nbrs) < nn:
                # fill with random others
                others = [i for i in range(59) if i != seed and i not in cand_nbrs]
                rng.shuffle(others)
                cand_nbrs = cand_nbrs + others[:nn - len(cand_nbrs)]
            # deterministic but varying selection
            if attempt % 3 == 0:
                chosen = cand_nbrs[:nn]
            else:
                perm = rng.permutation(len(cand_nbrs))
                # bias to top: take permuted then sort by row value with noise
                # simple: random sample nn among 15
                chosen = sorted([cand_nbrs[int(p)] for p in perm[:nn]])
            cluster = [seed] + chosen
            frozen_idx = [i for i in range(59) if i not in set(cluster)]
            F = x_best[frozen_idx].copy()  # mf x 22
            # score pool vs frozen: one matrix multiplication
            s0 = (pool @ F.T).max(axis=1)  # (NPOOL,)
            # order pool indices by s0
            idx_sorted = np.argsort(s0)
            # sequential greedy refill, best-100 per vacancy
            Vlist = []
            for ti in range(k):
                if len(Vlist) == 0:
                    scores = s0
                else:
                    Vc = np.stack(Vlist, axis=0)
                    sV = (pool @ Vc.T).max(axis=1)
                    scores = np.maximum(s0, sV)
                # best 100 candidates for this vacancy
                # use pre-sorted trick: need argsort of scores (changes each ti)
                part = np.argsort(scores)[:100]
                if attempt % 2 == 0:
                    pick = int(part[0])
                else:
                    # random among top-100 for diversity
                    pick = int(rng.choice(part))
                Vlist.append(pool[pick].copy())
            Vinit = np.stack(Vlist, axis=0)
            # locally optimise only reinserted points vs frozen: 300-500 annealed steps
            psteps = 400
            Vopt, _ = _partial_polish(Vinit, F, steps=psteps, beta0=50.0, beta1=500.0,
                                      lr0=0.02, lr1=0.002, deadline=deadline)
            # assemble full config
            Xtrial = np.empty_like(x_best)
            Xtrial[frozen_idx] = F
            for ti, ci in enumerate(cluster):
                Xtrial[ci] = Vopt[ti]
            Xtrial = _norm(Xtrial)
            # unfreeze: short 1500-step polish
            y, v = _full_polish(Xtrial, steps=1500, beta0=80.0, beta1=800.0,
                                lr0=0.008, lr1=0.0008, deadline=deadline)
            attempt += 1
            if v < best_val - 1e-12:
                best_val = v
                x_best = y.copy()
                improved_round = True
                # refresh graph immediately to exploit new basin
                break
            # time check every iteration (cheap); spec asks every 50 inner iters (done inside polish)
            if time.time() > deadline - 12.0:
                break
        if time.time() > deadline - 12.0:
            break
        # if we looped all seeds with no improvement, continue (different random choices)
        # stop if too many attempts
        if attempt > 400:
            break
        # small guard: if no improvement for long, keep going until deadline
        continue

    # final polish of best
    if time.time() < deadline - 8.0:
        try:
            y, v = _full_polish(x_best, steps=2500, beta0=150.0, beta1=1200.0,
                                lr0=0.004, lr1=0.0004, deadline=deadline + 8.0)
            if v < best_val:
                best_val = v
                x_best = y.copy()
        except Exception:
            pass

    x_best = _norm(x_best)
    return x_best
