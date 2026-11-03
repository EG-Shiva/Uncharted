import numpy as np
import time as _time

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    import time
    t0 = time.time()
    T_END = t0 + 172.0
    FINAL_RESERVE = 22.0
    rng = np.random.default_rng(12345)
    n, d = 249, 6
    I0, J0 = np.triu_indices(n, 1)
    P = len(I0)

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
                if np.all(np.isfinite(an)):
                    cands.append(an)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal(size=(n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals0 = [_max_cos(a) for a in cands]
    bi = int(np.argmin(vals0))
    bestX = cands[bi].copy()
    bestVal = float(vals0[bi])
    curX = bestX.copy()
    curVal = bestVal

    from scipy.optimize import minimize as _min

    def anneal(Xinit, taus, iters, creg=0.05):
        Xf = Xinit.ravel().copy()
        # normalize input
        Xm0 = Xf.reshape(n, d)
        rn0 = np.linalg.norm(Xm0, axis=1, keepdims=True)
        Xf = (Xm0 / np.maximum(rn0, 1e-12)).ravel()
        for tau, mi in zip(taus, iters):
            if time.time() > T_END - FINAL_RESERVE:
                break
            tau = float(tau)
            def fg(xf, tau=tau):
                X = xf.reshape(n, d)
                r = np.sqrt(np.sum(X * X, axis=1))
                r = np.maximum(r, 1e-12)
                Xn = X / r[:, None]
                G = Xn @ Xn.T
                g = G[I0, J0]
                gm = float(np.max(g))
                e = np.exp((g - gm) / tau)
                S = float(np.sum(e))
                Lsoft = gm + tau * np.log(S)
                w = e / S
                W = np.zeros((n, n), dtype=float)
                W[I0, J0] = w
                W = W + W.T
                A = W @ Xn
                proj = np.sum(A * Xn, axis=1)
                G0 = (A - proj[:, None] * Xn) / r[:, None]
                diff = r - 1.0
                L = Lsoft + creg * float(np.sum(diff * diff))
                Gr = G0 + (2.0 * creg * diff)[:, None] * Xn
                return float(L), Gr.ravel()
            try:
                res = _min(fg, Xf, jac=True, method='L-BFGS-B',
                           options={'maxiter': int(mi), 'maxfun': int(mi * 5),
                                    'ftol': 1e-14, 'gtol': 1e-7, 'maxcor': 5})
                Xf = res.x
            except Exception:
                break
            # intermediate renorm to keep stable
            Xm = Xf.reshape(n, d)
            rn = np.linalg.norm(Xm, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            Xf = (Xm / rn).ravel()
        Xo = Xf.reshape(n, d)
        Xo /= np.maximum(np.linalg.norm(Xo, axis=1, keepdims=True), 1e-12)
        return Xo

    def greedy_refill(Xbase, del_idx, pool):
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[del_idx] = False
        Xkeep = Xbase[keep_mask]
        C = pool.shape[0]
        D = pool @ Xkeep.T
        curmax = D.max(axis=1)
        used = np.zeros(C, dtype=bool)
        new_pts = []
        for _ in range(len(del_idx)):
            curmax[used] = np.inf
            b = int(np.argmin(curmax))
            used[b] = True
            pt = pool[b].copy()
            new_pts.append(pt)
            dd = pool @ pt
            # max with new point
            np.maximum(curmax, dd, out=curmax)
            curmax[used] = np.inf
        new_pts = np.stack(new_pts, axis=0) if new_pts else np.zeros((0, d))
        # local hole polish: each new point try tangent samples vs fixed others
        # 2 sweeps
        Xkeep_fixed = Xkeep
        for _ in range(2):
            allX = np.vstack([Xkeep_fixed, new_pts])
            for s in range(len(new_pts)):
                idx_all = Xkeep_fixed.shape[0] + s
                # current value
                dots0 = allX @ new_pts[s]
                dots0[idx_all] = -2.0
                cur_v = float(dots0.max())
                # propose candidates around pt
                # tangent basis via random samples
                cand = rng.standard_normal(size=(400, d)) * 0.06
                # project out radial
                pt = new_pts[s]
                cand = cand - np.outer(np.sum(cand * pt, axis=1), pt)
                cand = cand + pt[None, :]
                cand /= np.maximum(np.linalg.norm(cand, axis=1, keepdims=True), 1e-12)
                # evaluate vs allX (excluding self)
                # remove self row for scoring: use allX without idx
                mask = np.ones(allX.shape[0], dtype=bool)
                mask[idx_all] = False
                rest = allX[mask]
                scores = (cand @ rest.T).max(axis=1)
                bm = int(np.argmin(scores))
                if float(scores[bm]) < cur_v - 1e-6:
                    new_pts[s] = cand[bm]
                    allX[idx_all] = cand[bm]
        Xout = np.zeros((n, d))
        Xout[keep_mask] = Xkeep_fixed
        Xout[~keep_mask] = new_pts
        Xout /= np.maximum(np.linalg.norm(Xout, axis=1, keepdims=True), 1e-12)
        return Xout

    def pick_delete(X, mode, k):
        if mode == 0:  # spherical cap
            u = rng.standard_normal(size=(d,))
            u /= max(np.linalg.norm(u), 1e-12)
            dots = X @ u
            idx = np.argsort(dots)[-k:]
            return idx
        elif mode == 1:  # hub BFS
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            # contact counts with adaptive slack
            thr = m - 5e-4
            cm = np.sum(G > thr, axis=1)
            hub = int(np.argmax(cm + rng.random(n) * 0.5))
            # order by dot to hub
            order = np.argsort(G[hub])[::-1]
            # take hub + nearest, plus second ring if needed
            sel = [hub]
            for c in order:
                if c == hub:
                    continue
                if len(sel) >= k:
                    break
                sel.append(int(c))
            return np.array(sel[:k], dtype=int)
        else:  # random + worst points
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            pmax = G.max(axis=1)
            # bias to worst half
            worst = np.argsort(pmax)[-max(k * 3, k):]
            pick = rng.choice(worst, size=min(k, len(worst)), replace=False)
            if len(pick) < k:
                extra = rng.choice(n, size=k - len(pick), replace=False)
                pick = np.concatenate([pick, extra])
            return np.array(pick, dtype=int)

    # quick initial polish of incumbent with tiny anneal if time allows (1 shot)
    # main avalanche loop
    it = 0
    pool_size = 22000
    while time.time() < T_END - FINAL_RESERVE:
        it += 1
        # schedule k and mode cycling
        cyc = it % 6
        if cyc == 0:
            k = int(rng.integers(24, 36)); mode = 0
        elif cyc == 1:
            k = int(rng.integers(18, 30)); mode = 1
        elif cyc == 2:
            k = int(rng.integers(10, 18)); mode = 2
        elif cyc == 3:
            k = int(rng.integers(20, 32)); mode = 0
        elif cyc == 4:
            k = int(rng.integers(14, 24)); mode = 1
        else:
            k = int(rng.integers(8, 14)); mode = 2
        k = max(6, min(k, 40))
        del_idx = pick_delete(curX, mode, k)
        # fresh pool
        try:
            ps = pool_size
            # shrink pool if late to save time
            if time.time() > T_END - 80:
                ps = 14000
            pool = rng.standard_normal(size=(ps, d))
            pool /= np.maximum(np.linalg.norm(pool, axis=1, keepdims=True), 1e-12)
        except Exception:
            continue
        Xref = greedy_refill(curX, del_idx, pool)
        # small kick to refill points to break symmetry
        # anneal schedule: coarser early, finer later
        if it < 4:
            taus = [0.012, 0.006, 0.0025]
            iters = [45, 55, 70]
        else:
            taus = [0.010, 0.005, 0.002]
            iters = [50, 60, 80]
        # add tiny noise before anneal to new points
        try:
            Xnew = anneal(Xref, taus, iters)
        except Exception:
            continue
        v = _max_cos(Xnew)
        if v < bestVal:
            bestVal = v
            bestX = Xnew.copy()
        # non-monotone walk: accept if better than cur or sometimes worse
        if v < curVal:
            curX = Xnew.copy()
            curVal = v
        else:
            # allow uphill with prob decreasing in gap
            gap = v - curVal
            if gap < 0.004 and rng.random() < 0.35:
                curX = Xnew.copy()
                curVal = v
            elif rng.random() < 0.05:
                # random restart from best with cap delete to diversify
                curX = bestX.copy()
                curVal = bestVal
                # small random tangent kick
                kick = rng.standard_normal(size=(n, d)) * 0.01
                kick -= np.sum(kick * curX, axis=1)[:, None] * curX
                curX = curX + kick
                curX /= np.maximum(np.linalg.norm(curX, axis=1, keepdims=True), 1e-12)
        # occasional return to best
        if it % 8 == 0 and time.time() < T_END - FINAL_RESERVE - 10:
            # exploit best with small cap avalanche
            pass
        if time.time() > T_END - FINAL_RESERVE:
            break

    # final deep polish of best
    try:
        bestX = anneal(bestX, [0.004, 0.002, 0.001, 0.0007], [80, 80, 90, 60], creg=0.03)
        v = _max_cos(bestX)
        if v < bestVal:
            bestVal = v
        else:
            # keep previous best if worse (anneal returns something; take min)
            pass
    except Exception:
        pass
    # rattler relocation attempt: move lowest-contact points to best holes
    try:
        if time.time() < T_END - 5:
            G = bestX @ bestX.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            pmax = G.max(axis=1)
            # rattlers: far below max
            order = np.argsort(pmax)  # lowest first
            nrat = int(np.sum(pmax < m - 1e-5))
            nfix = min(max(nrat, 2), 8)
            fix_idx = order[:nfix]
            pool = rng.standard_normal(size=(30000, d))
            pool /= np.maximum(np.linalg.norm(pool, axis=1, keepdims=True), 1e-12)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[fix_idx] = False
            Xkeep = bestX[keep_mask]
            D = pool @ Xkeep.T
            curmax = D.max(axis=1)
            used = np.zeros(pool.shape[0], dtype=bool)
            repl = []
            for _ in range(nfix):
                curmax[used] = np.inf
                b = int(np.argmin(curmax))
                used[b] = True
                pt = pool[b].copy()
                repl.append(pt)
                dd = pool @ pt
                np.maximum(curmax, dd, out=curmax)
                curmax[used] = np.inf
            repl = np.stack(repl, axis=0)
            Xt = bestX.copy()
            Xt[fix_idx] = repl
            Xt = anneal(Xt, [0.002, 0.001], [60, 60], creg=0.03)
            vt = _max_cos(Xt)
            if vt < bestVal:
                bestX = Xt.copy()
                bestVal = vt
    except Exception:
        pass

    bestX /= np.maximum(np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)
    # guard finite
    bestX = np.where(np.isfinite(bestX), bestX, 0.0)
    rn = np.linalg.norm(bestX, axis=1, keepdims=True)
    rn = np.maximum(rn, 1e-12)
    bestX = bestX / rn
    return bestX
