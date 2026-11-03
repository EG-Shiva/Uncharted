import time
import numpy as np

def solve():
    t0 = time.time()
    TIME_LIMIT = 170.0
    rng = np.random.RandomState(0)

    def max_cos(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return float(np.max(G))

    def normalize(X):
        n = np.linalg.norm(X, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return X / n

    # load start
    try:
        Xb = np.load("best.npy").astype(float)
    except Exception:
        Xb = None
    try:
        Xr = np.load("record.npy").astype(float)
    except Exception:
        Xr = None
    cands = []
    for C in (Xb, Xr):
        if C is not None and C.shape == (250, 6):
            C = normalize(C)
            # remove nan
            if np.all(np.isfinite(C)):
                cands.append((max_cos(C), C))
    if len(cands) == 0:
        Xbest = normalize(rng.randn(250, 6))
        best = max_cos(Xbest)
    else:
        cands.sort(key=lambda t: t[0])
        best, Xbest = cands[0][0], cands[0][1].copy()
    Xbest = normalize(Xbest)

    def gd_trial(Xstart, steps, beta0, beta1, lr0, lr1):
        nonlocal best, Xbest
        Xc = Xstart.copy()
        # track trial best in minimax sense
        trial_best_val = max_cos(Xc)
        trial_best = Xc.copy()
        if trial_best_val < best:
            best = trial_best_val
            Xbest = trial_best.copy()
        for s in range(steps):
            if time.time() - t0 > TIME_LIMIT - 25:
                break
            frac = s / max(1, steps - 1)
            beta = beta0 * (beta1 / beta0) ** frac
            lr = lr0 * (lr1 / lr0) ** frac
            G = Xc @ Xc.T
            np.fill_diagonal(G, -np.inf)
            m = float(np.max(G))
            cutoff = 12.0 / beta
            if cutoff < 0.003:
                cutoff = 0.003
            # weights for near-max pairs
            # W = exp(beta*(G-m)) where G>m-cutoff
            W = np.zeros_like(G)
            # boolean mask
            # to save time use flat
            mask = G > (m - cutoff)
            # exp on masked entries
            Gm = G[mask] - m
            W[mask] = np.exp(beta * Gm)
            np.fill_diagonal(W, 0.0)
            H = W @ Xc
            dot = np.sum(H * Xc, axis=1, keepdims=True)
            Gt = H - dot * Xc
            # SGD step opposite gradient
            delta = lr * Gt
            # clip per-point step to avoid blow up
            dn = np.sqrt(np.sum(delta * delta, axis=1, keepdims=True)) + 1e-20
            maxstep = 0.03
            sc = np.minimum(1.0, maxstep / dn)
            delta = delta * sc
            Xc = Xc - delta
            # renormalize
            nrm = np.linalg.norm(Xc, axis=1, keepdims=True)
            nrm[nrm == 0] = 1.0
            Xc = Xc / nrm
            if (s % 200 == 199) or (s == steps - 1):
                v = max_cos(Xc)
                if v < trial_best_val:
                    trial_best_val = v
                    trial_best = Xc.copy()
                    if v < best:
                        best = v
                        Xbest = Xc.copy()
                # early escape if time low
                if time.time() - t0 > TIME_LIMIT - 25:
                    break
        return trial_best, trial_best_val

    # ---- GD basin hopping ----
    # trial 0: no shake
    if time.time() - t0 < TIME_LIMIT - 30:
        gd_trial(Xbest, 9000, 30.0, 1200.0, 0.008, 0.0008)
    # further trials with shakes
    sigmas = [0.012, 0.035, 0.008, 0.020]
    betas = [(50.0, 2000.0), (25.0, 800.0), (80.0, 2500.0), (40.0, 1500.0)]
    lrs = [(0.006, 0.0006), (0.008, 0.0005), (0.004, 0.0003), (0.005, 0.0004)]
    for ti in range(len(sigmas)):
        if time.time() - t0 > TIME_LIMIT - 30:
            break
        sigma = sigmas[ti]
        Xs = Xbest + sigma * rng.randn(250, 6)
        Xs = normalize(Xs)
        b0, b1 = betas[ti % len(betas)]
        l0, l1 = lrs[ti % len(lrs)]
        gd_trial(Xs, 7000, b0, b1, l0, l1)

    # ---- greedy minimax hill climbing on worst points ----
    # fast incremental max evaluation
    def hill_climb(Xh, cur, iters, s0, s1, seed):
        r = np.random.RandomState(seed)
        n = Xh.shape[0]
        G = Xh @ Xh.T
        np.fill_diagonal(G, -2.0)
        # active list refresh
        active = np.arange(n)
        # precompute for speed: indices
        for it in range(iters):
            if time.time() - t0 > TIME_LIMIT - 2:
                break
            if it % 300 == 0:
                # recompute active set: points involved in pairs close to max
                thr = cur - 0.004
                # find pairs > thr
                # use upper triangle
                # for speed, find rows with any entry > thr
                mx_row = np.max(G, axis=1)
                active = np.where(mx_row > thr)[0]
                if len(active) == 0:
                    active = np.arange(n)
            frac = it / max(1, iters - 1)
            sigma = s0 * (s1 / s0) ** frac if s0 > 0 and s1 > 0 else s0
            p = active[r.randint(len(active))]
            xp = Xh[p]
            # random tangent direction
            rv = r.randn(6)
            rv = rv - np.dot(rv, xp) * xp
            nn = np.linalg.norm(rv)
            if nn < 1e-12:
                continue
            rv = rv / nn
            # random signed step magnitude ~ |N|*sigma? use gaussian
            step = r.randn() * sigma
            v = xp + step * rv
            vn = np.linalg.norm(v)
            if vn == 0 or not np.isfinite(vn):
                continue
            v = v / vn
            dots = Xh @ v
            dots[p] = -2.0
            nd_new = float(np.max(dots))
            # max excluding p: max of G with row/col p ignored
            # compute quickly: mask row/col p to -2 via copy of max?
            # we can compute as: if argmax pair does not involve p, use cur
            # find current argmax pair (maintain approx): just compute max of G ignoring p
            # O(n^2) max but with view: set temp
            # To avoid copy, compute two candidates: max of rows !=p
            # Simplest: temporarily set
            old_row = G[p, :].copy()
            old_col = G[:, p].copy()
            # max excluding p = max of G with row/col p = -2
            # we can compute without full copy by max of remaining:
            # G without p: use np.max on strided? just compute with masking via max of whole minus p:
            # do quick: G[p,:]=-2; G[:,p]=-2; mex = max; restore later if reject
            G[p, :] = -2.0
            G[:, p] = -2.0
            mex = float(np.max(G))
            new_max = nd_new if nd_new > mex else mex
            if new_max < cur - 1e-13:  # strict improvement (allow tiny tol)
                # accept
                Xh[p] = v
                # update G row/col to new dots
                G[p, :] = dots
                G[:, p] = dots
                G[p, p] = -2.0
                cur = new_max
            else:
                # restore
                G[p, :] = old_row
                G[:, p] = old_col
        return Xh, cur

    if time.time() - t0 < TIME_LIMIT - 5:
        Xbest, best = hill_climb(Xbest, best, 35000, 0.004, 0.00002, 1)
    if time.time() - t0 < TIME_LIMIT - 5:
        Xbest, best = hill_climb(Xbest, best, 25000, 0.001, 0.000005, 2)

    # final safety normalize, tiny shrink not needed (score uses normalized rows)
    Xbest = normalize(Xbest)
    # ensure finite
    if not np.all(np.isfinite(Xbest)):
        Xbest = normalize(np.load("best.npy").astype(float))
    return Xbest
