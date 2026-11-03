import time
import numpy as np

def solve():
    t0 = time.time()
    TLIM = 172.0
    rng = np.random.RandomState(42)

    def normed(X):
        n = np.linalg.norm(X, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return X / n

    def max_cos(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return float(np.max(G))

    def row_max(X, G=None):
        if G is None:
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
        return np.max(G, axis=1), G

    # load
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            C = np.load(fn).astype(float)
            if C.shape == (250, 6) and np.all(np.isfinite(C)):
                C = normed(C)
                cands.append((max_cos(C), C))
        except Exception:
            pass
    if len(cands) == 0:
        Xbest = normed(rng.randn(250, 6))
        best = max_cos(Xbest)
    else:
        cands.sort(key=lambda t: t[0])
        best, Xbest = cands[0][0], cands[0][1].copy()
    Xbest = normed(Xbest)
    n, d = 250, 6

    def gd_smooth(Xs, steps, beta0, beta1, lr0, lr1, maxstep=0.02, check_every=200):
        nonlocal best, Xbest, t0
        Xc = Xs.copy()
        trial_best = Xc.copy()
        # initial minimax
        v0 = max_cos(Xc)
        trial_val = v0
        if v0 < best:
            best = v0
            Xbest = Xc.copy()
        for s in range(steps):
            if time.time() - t0 > TLIM - 30:
                break
            frac = s / max(1, steps - 1)
            beta = beta0 * (beta1 / beta0) ** frac
            lr = lr0 * (lr1 / lr0) ** frac
            G = Xc @ Xc.T
            np.fill_diagonal(G, -np.inf)
            m = float(np.max(G))
            cutoff = 12.0 / beta
            if cutoff < 0.004:
                cutoff = 0.004
            if cutoff > 0.15:
                cutoff = 0.15
            # weights
            # full exp is ok but masked is faster
            mask = G > (m - cutoff)
            W = np.zeros_like(G)
            # exp on masked
            tmp = G[mask] - m
            # clip to avoid overflow (tmp<=0)
            W[mask] = np.exp(beta * tmp)
            np.fill_diagonal(W, 0.0)
            H = W @ Xc
            dot = np.sum(H * Xc, axis=1, keepdims=True)
            Gt = H - dot * Xc
            # normalize gradient scale by sum weights
            # scale lr by 1/beta to keep step stable
            delta = (lr / max(1.0, beta * 0.1)) * Gt
            # per-point clip
            dn = np.sqrt(np.sum(delta * delta, axis=1, keepdims=True)) + 1e-20
            sc = np.minimum(1.0, maxstep / (dn + 1e-20))
            # also scale by number of active
            delta = delta * sc
            Xc = Xc - delta
            Xc = normed(Xc)
            if (s % check_every == check_every - 1) or s == steps - 1:
                v = max_cos(Xc)
                if v < trial_val:
                    trial_val = v
                    trial_best = Xc.copy()
                    if v < best:
                        best = v
                        Xbest = Xc.copy()
        return trial_best, trial_val

    def greedy_single(Xs, iters, sigma0, sigma1, batch=512, seed=0):
        nonlocal best, Xbest, t0
        r = np.random.RandomState(seed)
        Xc = Xs.copy()
        G = Xc @ Xc.T
        np.fill_diagonal(G, -2.0)
        cur = float(np.max(G))
        # trial best
        tbest = Xc.copy()
        tval = cur
        if cur < best:
            best = cur
            Xbest = Xc.copy()
        for it in range(iters):
            if time.time() - t0 > TLIM - 20:
                break
            frac = it / max(1, iters - 1)
            sigma = sigma0 * (sigma1 / sigma0) ** frac if sigma0 > 0 and sigma1 > 0 else sigma0
            # pick point: worst with some randomness
            if it % 10 == 0:
                rm = np.max(G, axis=1)
                # top candidates
                order = np.argsort(-rm)[:25]
                p = order[r.randint(len(order))]
            else:
                rm = np.max(G, axis=1)
                p = int(np.argmax(rm))
            # max without p
            # global without p = max of G excluding row/col p
            # compute quickly: max of G is cur; if p not involved in max pair, max_without = cur
            # find max pair
            # approximate: mask
            Gp = G.copy()
            Gp[p, :] = -2.0
            Gp[:, p] = -2.0
            max_without = float(np.max(Gp))
            # generate batch candidates around Xc[p]
            Z = r.randn(batch, d)
            # make tangent-ish: add noise
            C = Xc[p:p+1] + sigma * Z
            C = normed(C)
            # dots to others (exclude p)
            D = C @ Xc.T  # batch x 250
            # ignore self column p (dot=~1) -> set to -2
            D[:, p] = -2.0
            cmax = np.max(D, axis=1)
            # global if we replace p with candidate
            gcan = np.maximum(cmax, max_without)
            bi = int(np.argmin(gcan))
            if gcan[bi] < cur - 1e-12:
                # accept
                Xc[p] = C[bi]
                # update G row/col p
                row = Xc[p] @ Xc.T
                row[p] = -2.0
                G[p, :] = row
                G[:, p] = row
                cur = float(np.max(G))
                if cur < tval:
                    tval = cur
                    tbest = Xc.copy()
                    if cur < best:
                        best = cur
                        Xbest = Xc.copy()
            elif it % 50 == 0:
                # occasional neutral move to diversify if candidate close
                if gcan[bi] < cur + 1e-9 and sigma > 0.005:
                    pass
            if it % 300 == 299:
                # refresh G fully to avoid drift
                G = Xc @ Xc.T
                np.fill_diagonal(G, -2.0)
                cur = float(np.max(G))
        return tbest, tval

    def hole_reinsert(Xs, n_replace=3, n_probe=4000, seed=0):
        nonlocal best, Xbest, t0
        r = np.random.RandomState(seed)
        Xc = Xs.copy()
        G = Xc @ Xc.T
        np.fill_diagonal(G, -2.0)
        cur = float(np.max(G))
        rm = np.max(G, axis=1)
        worst = np.argsort(-rm)[:max(10, n_replace*3)]
        for k in range(n_replace):
            if time.time() - t0 > TLIM - 20:
                break
            p = worst[r.randint(len(worst))]
            # find hole: random probes minimizing max dot to others except p
            P = normed(r.randn(n_probe, d))
            D = P @ Xc.T  # probe x 250
            # exclude p column
            # set column p to -inf so it doesn't count
            D[:, p] = -2.0
            pm = np.max(D, axis=1)
            bi = int(np.argmin(pm))
            v = P[bi:bi+1].copy()
            # local refine probe: hill climb away from nearest
            sigma = 0.05
            for _ in range(120):
                Z = r.randn(64, d)
                C = v + sigma * Z
                C = normed(C)
                D2 = C @ Xc.T
                D2[:, p] = -2.0
                m2 = np.max(D2, axis=1)
                bj = int(np.argmin(m2))
                if m2[bj] < pm[bi]:
                    v = C[bj:bj+1]
                    pm[bi] = m2[bj]
                sigma *= 0.97
            # evaluate replacement
            Gp = G.copy()
            Gp[p, :] = -2.0
            Gp[:, p] = -2.0
            max_without = float(np.max(Gp))
            new_row_max = float(pm[bi])
            new_global = max(max_without, new_row_max)
            if new_global < cur:
                Xc[p] = v[0] / np.linalg.norm(v[0])
                G = Xc @ Xc.T
                np.fill_diagonal(G, -2.0)
                cur = float(np.max(G))
                rm = np.max(G, axis=1)
                worst = np.argsort(-rm)[:max(10, n_replace*3)]
                if cur < best:
                    best = cur
                    Xbest = Xc.copy()
        return Xc, cur

    # Phase 1: smooth on current best
    if time.time() - t0 < TLIM - 30:
        gd_smooth(Xbest, 6000, 30.0, 1500.0, 0.9, 0.08, maxstep=0.025, check_every=250)
    # Phase 2: cycles
    sigmas = [0.010, 0.030, 0.007, 0.018, 0.045]
    b0s = [40.0, 25.0, 80.0, 35.0, 20.0]
    b1s = [2000.0, 900.0, 2500.0, 1600.0, 800.0]
    for cyc in range(5):
        if time.time() - t0 > TLIM - 28:
            break
        sigma = sigmas[cyc % len(sigmas)]
        Xs = Xbest + sigma * rng.randn(n, d)
        Xs = normed(Xs)
        gd_smooth(Xs, 5000, b0s[cyc % len(b0s)], b1s[cyc % len(b1s)], 0.8, 0.06, maxstep=0.03, check_every=250)
        if time.time() - t0 > TLIM - 25:
            break
        # greedy on current Xbest
        greedy_single(Xbest, 1200, 0.02, 0.0008, batch=512, seed=100 + cyc)
        if cyc % 2 == 0:
            hole_reinsert(Xbest, n_replace=2, n_probe=3000, seed=200 + cyc)

    # Phase 3: intensive greedy
    if time.time() - t0 < TLIM - 18:
        greedy_single(Xbest, 4000, 0.012, 0.0004, batch=768, seed=999)
    # Phase 4: final fine smooth high beta small steps
    if time.time() - t0 < TLIM - 10:
        gd_smooth(Xbest, 3000, 400.0, 4000.0, 0.25, 0.02, maxstep=0.008, check_every=200)

    # Final LP-style micro polish via projected subgradient on top pairs (no scipy, fast)
    # minimize max by moving along averaged violating directions with decaying step
    if time.time() - t0 < TLIM - 3:
        Xc = Xbest.copy()
        step = 2e-4
        for it in range(3000):
            if time.time() - t0 > TLIM - 1.5:
                break
            G = Xc @ Xc.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            # active pairs within 2e-4 of max (cap 4000)
            IU = np.where(G > m - 2e-4)
            # keep upper only
            mask_up = IU[0] < IU[1]
            I = IU[0][mask_up]
            J = IU[1][mask_up]
            if len(I) > 4000:
                # keep tightest 4000
                vals = G[I, J]
                idx = np.argsort(-vals)[:4000]
                I, J = I[idx], J[idx]
            if len(I) == 0:
                break
            # gradient: for each k, sum of partners
            Grad = np.zeros_like(Xc)
            # weight by closeness
            w = np.exp(3000.0 * (G[I, J] - m)) + 0.1
            np.add.at(Grad, I, (w[:, None] * Xc[J]))
            np.add.at(Grad, J, (w[:, None] * Xc[I]))
            # tangential
            Grad -= np.sum(Grad * Xc, axis=1, keepdims=True) * Xc
            gn = np.linalg.norm(Grad, axis=1, keepdims=True) + 1e-20
            # step
            Xc = Xc - step * Grad / np.max(gn)
            Xc = normed(Xc)
            v = max_cos(Xc)
            if v < best:
                best = v
                Xbest = Xc.copy()
            else:
                step *= 0.998
                # revert if worsened a lot?
                if v > best + 5e-6:
                    Xc = Xbest.copy()
                    step *= 0.7
            if it % 500 == 499:
                step *= 0.7

    Xbest = normed(Xbest)
    # guard
    Xbest[~np.isfinite(Xbest)] = 0.0
    # if any zero row, replace
    nr = np.linalg.norm(Xbest, axis=1)
    for i in range(n):
        if nr[i] < 1e-12 or not np.finite(nr[i]):
            Xbest[i] = normed(rng.randn(1, d))[0]
    Xbest = normed(Xbest)
    return Xbest
