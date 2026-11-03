import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normed(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return A / n

def solve():
    import time
    import numpy as np
    t0 = time.time()
    DEADLINE = 160.0

    def timed_out():
        return (time.time() - t0) > DEADLINE

    rng = np.random.default_rng(0)

    # load best / record
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (59, 22) and np.all(np.isfinite(a)):
                a = _normed(a)
                cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        best_X = _normed(rng.standard_normal((59, 22)))
    else:
        vals = [_max_cos(c) for c in cands]
        best_X = cands[int(np.argmin(vals))].copy()
    best_val = _max_cos(best_X)

    # ---- helpers (Adam LSE) ----
    def full_polish(X0, steps, beta0, beta1, lr0, lr1):
        X = X0.copy()
        M = np.zeros_like(X)
        V = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        b1t, b2t = 1.0, 1.0
        lb0, lb1 = np.log(beta0), np.log(beta1)
        ll0, ll1 = np.log(lr0), np.log(lr1)
        bv = _max_cos(X)
        bX = X.copy()
        for s in range(steps):
            if (s % 50) == 0 and timed_out():
                break
            frac = s / float(max(1, steps - 1))
            beta = float(np.exp(lb0 + (lb1 - lb0) * frac))
            lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            W = np.exp(beta * (G - m))
            np.fill_diagonal(W, 0.0)
            S = float(W.sum() * 0.5)
            if not np.isfinite(S) or S < 1e-30:
                continue
            Gr = (W @ X) / S
            dot = (Gr * X).sum(axis=1, keepdims=True)
            Gr = Gr - dot * X
            b1t *= b1; b2t *= b2
            M *= b1; M += (1.0 - b1) * Gr
            V *= b2; V += (1.0 - b2) * (Gr * Gr)
            Mh = M / (1.0 - b1t); Vh = V / (1.0 - b2t)
            X = X - lr * Mh / (np.sqrt(Vh) + eps)
            X = _normed(X)
            if (s % 100 == 99) or (s == steps - 1):
                v = _max_cos(X)
                if v < bv:
                    bv = v; bX = X.copy()
        return bX, bv

    def partial_opt(Y0, F, steps, beta0, beta1, lr0, lr1):
        Y = Y0.copy()
        K = Y.shape[0]
        M = np.zeros_like(Y)
        V = np.zeros_like(Y)
        b1, b2, eps = 0.9, 0.999, 1e-8
        b1t, b2t = 1.0, 1.0
        lb0, lb1 = np.log(beta0), np.log(beta1)
        ll0, ll1 = np.log(lr0), np.log(lr1)
        for s in range(steps):
            if (s % 50) == 0 and timed_out():
                break
            frac = s / float(max(1, steps - 1))
            beta = float(np.exp(lb0 + (lb1 - lb0) * frac))
            lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
            Gyf = Y @ F.T
            Gyy = Y @ Y.T
            m1 = float(Gyf.max()) if Gyf.size else -2.0
            # off-diag max of Gyy
            tmp = Gyy.copy()
            np.fill_diagonal(tmp, -2.0)
            m2 = float(tmp.max()) if K > 1 else -2.0
            m = m1 if m1 > m2 else m2
            Wyf = np.exp(beta * (Gyf - m))
            Wyy = np.exp(beta * (Gyy - m))
            np.fill_diagonal(Wyy, 0.0)
            S = float(Wyf.sum() + Wyy.sum() * 0.5)
            if not np.isfinite(S) or S < 1e-30:
                continue
            Gr = (Wyf @ F + Wyy @ Y) / S
            dot = (Gr * Y).sum(axis=1, keepdims=True)
            Gr = Gr - dot * Y
            b1t *= b1; b2t *= b2
            M *= b1; M += (1.0 - b1) * Gr
            V *= b2; V += (1.0 - b2) * (Gr * Gr)
            Mh = M / (1.0 - b1t); Vh = V / (1.0 - b2t)
            Y = Y - lr * Mh / (np.sqrt(Vh) + eps)
            Y = _normed(Y)
        return Y

    # precomputed pool >= 20000 random unit vectors
    POOL_N = 20000
    pool = _normed(rng.standard_normal((POOL_N, 22)))

    # short initial polish to lock baseline (cheap)
    if not timed_out():
        try:
            Xp, vp = full_polish(best_X, steps=1200, beta0=40.0, beta1=300.0, lr0=0.008, lr1=0.0008)
            if vp < best_val:
                best_val = vp; best_X = Xp
        except Exception:
            pass

    # contact degrees for center ordering
    def contact_order(X):
        try:
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            C = (G >= (m - 1e-7))
            deg = C.sum(axis=1)
            order = np.argsort(-deg)
            return order, G, m
        except Exception:
            return np.arange(59), None, None

    Ks = [7, 6, 8, 9, 5, 7, 8, 6, 9, 7]
    n_trials = 80
    for trial in range(n_trials):
        if timed_out() or (time.time() - t0) > 155.0:
            break
        # recompute order every trial (cheap) to follow current best
        order, Gfull, mfull = contact_order(best_X)
        # pick center cycling through most-loaded points
        ci = int(order[trial % len(order)])
        K = int(Ks[trial % len(Ks)])
        K = max(5, min(9, K))
        # neighbours: top-15 by inner product with center
        try:
            row = Gfull[ci].copy()
            row[ci] = -2.0
            top15 = np.argsort(-row)[:15]
            # deterministic random subset
            perm = rng.permutation(15)[: (K - 1)]
            neigh = top15[perm]
            cluster = np.concatenate(([ci], neigh)).astype(int)
            # dedupe (should be unique)
            cluster = np.unique(cluster)
            # if dedupe shrank, fill with random indices
            while len(cluster) < K:
                cand = int(rng.integers(0, 59))
                if cand not in cluster:
                    cluster = np.append(cluster, cand)
            cluster = cluster[:K]
        except Exception:
            cluster = rng.choice(59, size=K, replace=False)
        frozen_mask = np.ones(59, dtype=bool)
        frozen_mask[cluster] = False
        frozen_idx = np.where(frozen_mask)[0]
        F = best_X[frozen_idx].copy()  # (M,22), M=50..54
        # greedy reinsert one at a time using one matmul per vacancy
        # score = max inner product with frozen+already placed
        placed = []
        Faug = F.copy()
        # random order of vacancies
        vorder = rng.permutation(K)
        # we will fill dict vacancy_pos -> vector
        new_vecs = [None] * K
        for vi in range(K):
            if timed_out():
                break
            # one matrix multiplication
            S = pool @ Faug.T  # (20000, Faug)
            smax = S.max(axis=1)  # (20000,)
            # best 100 candidates
            idx100 = np.argpartition(smax, 100)[:100]
            # sort those 100 by score
            o = np.argsort(smax[idx100])
            idx100 = idx100[o]
            # pick among top100: bias to best but diversify by trial
            pick_rank = int(rng.integers(0, 100)) if (trial % 3 == 2) else int(rng.integers(0, 10))
            # mostly pick best; occasionally deeper rank
            if rng.random() < 0.6:
                pick_rank = 0
            chosen = pool[idx100[pick_rank]].copy()
            # small jitter to break symmetry
            chosen = chosen + 1e-4 * rng.standard_normal(22)
            chosen = chosen / max(1e-12, np.linalg.norm(chosen))
            # assign to vacancy vorder[vi]
            new_vecs[vorder[vi]] = chosen
            # augment frozen set for next vacancy
            Faug = np.vstack([Faug, chosen[None, :]])
            if (vi % 5 == 0) and timed_out():
                break
        if any(v is None for v in new_vecs):
            continue
        Y0 = np.vstack(new_vecs)  # (K,22) in cluster order? need map: new_vecs indexed 0..K-1 by position in cluster
        # Note: vorder maps fill order; Y0[k] corresponds to cluster[k]
        # local optimisation of only reinserted points vs frozen set: 300-500 annealed steps
        try:
            nsteps = 300 + (trial % 3) * 100  # 300,400,500
            Yopt = partial_opt(Y0, F, steps=nsteps, beta0=30.0, beta1=300.0, lr0=0.02, lr1=0.002)
        except Exception:
            continue
        Xt = best_X.copy()
        for k_i, gi in enumerate(cluster):
            Xt[gi] = Yopt[k_i]
        Xt = _normed(Xt)
        # unfreeze everything for short 1500-step polish
        try:
            Xpol, vpol = full_polish(Xt, steps=1500, beta0=30.0, beta1=280.0, lr0=0.010, lr1=0.0006)
        except Exception:
            continue
        if vpol < best_val - 1e-12:
            best_val = vpol
            best_X = Xpol.copy()
        # time check every trial (and inner checks every 50 iters already)
        if timed_out():
            break

    # final fine polish if time remains
    if not timed_out() and (time.time() - t0) < 158.0:
        try:
            Xf, vf = full_polish(best_X, steps=800, beta0=120.0, beta1=500.0, lr0=0.003, lr1=0.0002)
            if vf < best_val:
                best_val = vf; best_X = Xf
        except Exception:
            pass

    best_X = _normed(best_X)
    return best_X
