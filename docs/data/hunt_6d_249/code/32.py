import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    FINAL_RESERVE = 25.0
    n, d = 249, 6
    rng = np.random.default_rng(12345)
    np.random.seed(12345)
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
    initX = bestX.copy()
    initVal = bestVal

    def top_set(X, K=1500):
        G = X @ X.T
        g = G[I0, J0]
        if K >= P:
            idx = np.arange(P)
        else:
            idx = np.argpartition(g, -K)[-K:]
        codes = I0[idx].astype(np.int64) * n + J0[idx].astype(np.int64)
        return set(codes.tolist()), codes

    inc_set, inc_codes = top_set(bestX, 1500)
    inc_mask = np.zeros(P, dtype=bool)
    # map code -> pair index for fast mask: build via dict for top pairs
    # instead compute mask by threshold on values: recompute
    G0 = bestX @ bestX.T
    g0 = G0[I0, J0]
    thr = np.sort(g0)[-1500]
    inc_mask = g0 >= thr

    archive = [(bestVal, bestX.copy())]

    def full_anneal(Xinit, taus, iters, boost=None, creg=0.08):
        Xf = Xinit.ravel().copy()
        for tau, mi in zip(taus, iters):
            if time.time() > T_END - FINAL_RESERVE:
                break
            tau = float(tau)
            b = boost  # may be None
            def fg(xf, tau=tau):
                X = xf.reshape(n, d)
                r = np.sqrt(np.sum(X * X, axis=1))
                r = np.maximum(r, 1e-12)
                Xn = X / r[:, None]
                G = Xn @ Xn.T
                g = G[I0, J0]
                gm = np.max(g)
                e = np.exp((g - gm) / tau)
                if b is not None:
                    e = e * b
                S = np.sum(e)
                Lsoft = gm + tau * np.log(S)
                w = e / S
                W = np.zeros((n, n), dtype=float)
                W[I0, J0] = w
                W = W + W.T
                A = W @ Xn
                proj = np.sum(A * Xn, axis=1)
                G0l = (A - proj[:, None] * Xn) / r[:, None]
                diff = r - 1.0
                L = Lsoft + creg * np.sum(diff * diff)
                Gr = G0l + (2.0 * creg * diff)[:, None] * Xn
                return float(L), Gr.ravel()
            try:
                res = minimize(fg, Xf, jac=True, method='L-BFGS-B',
                               options={'maxiter': int(mi), 'maxfun': int(mi * 4),
                                        'ftol': 1e-14, 'gtol': 1e-7, 'maxcor': 8})
                Xf = res.x
            except Exception:
                break
            Xm = Xf.reshape(n, d)
            rn = np.linalg.norm(Xm, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            Xf = (Xm / rn).ravel()
        Xo = Xf.reshape(n, d)
        Xo /= np.maximum(np.linalg.norm(Xo, axis=1, keepdims=True), 1e-12)
        return Xo

    def optimize_hole(h0, Xkeep, steps=250):
        h = h0.copy()
        hn = h / max(np.linalg.norm(h), 1e-12)
        step = 0.05
        for it in range(steps):
            dots = Xkeep @ hn
            j = int(np.argmax(dots))
            g = Xkeep[j] - dots[j] * hn
            gn = np.linalg.norm(g)
            if gn < 1e-12:
                # random tangent kick
                rr = rng.standard_normal(d)
                rr -= (rr @ hn) * hn
                rr /= max(np.linalg.norm(rr), 1e-12)
                hn = hn + 0.01 * rr
                hn /= np.linalg.norm(hn)
                continue
            g /= gn
            # move away from worst
            hn = hn - step * g
            hn /= max(np.linalg.norm(hn), 1e-12)
            step *= 0.995
            if step < 1e-4:
                step = 1e-4
        return hn

    def greedy_refill(Xbase, del_idx, Cnum=60000):
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[del_idx] = False
        Xkeep = Xbase[keep_mask]
        nk = Xkeep.shape[0]
        k = len(del_idx)
        cand = rng.standard_normal(size=(Cnum, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        # scores
        D = cand @ Xkeep.T
        curmax = D.max(axis=1)
        # pick preliminary best few for hole optimization
        order = np.argpartition(curmax, 12)[:12]
        extra = []
        for oi in order[:6]:
            ho = optimize_hole(cand[oi], Xkeep, steps=180)
            extra.append(ho)
        if extra:
            extra = np.stack(extra, axis=0)
            cand = np.vstack([cand, extra])
            D2 = extra @ Xkeep.T
            curmax2 = D2.max(axis=1)
            curmax = np.concatenate([curmax, curmax2])
        C = cand.shape[0]
        used = np.zeros(C, dtype=bool)
        new_pts = []
        # for fast update keep copy
        for _ in range(k):
            curmax[used] = np.inf
            bii = int(np.argmin(curmax))
            used[bii] = True
            pt = cand[bii].copy()
            new_pts.append(pt)
            dd = cand @ pt
            # max of old curmax and new dot
            # curmax already has inf for used; need max
            np.maximum(curmax, dd, out=curmax)
            curmax[used] = np.inf
        new_pts = np.stack(new_pts, axis=0)
        Xnew = np.empty((n, d))
        Xnew[keep_mask] = Xkeep
        Xnew[del_idx] = new_pts
        # normalize (already)
        Xnew /= np.maximum(np.linalg.norm(Xnew, axis=1, keepdims=True), 1e-12)
        return Xnew

    def contact_adj(X, delta=5e-4):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        Mask = G > (m - delta)
        return Mask, m

    def bfs_delete(X, k):
        Mask, m = contact_adj(X, delta=8e-4)
        deg = Mask.sum(axis=1)
        # hub among top degrees
        top = np.argsort(-deg)[:6]
        hub = int(top[rng.integers(0, len(top))])
        visited = {hub}
        queue = [hub]
        # order neighbours by degree
        while len(visited) < k and queue:
            u = queue.pop(0)
            nbrs = np.where(Mask[u])[0]
            # shuffle-ish sort by deg descending
            nbrs = nbrs[np.argsort(-deg[nbrs])]
            for v in nbrs:
                vv = int(v)
                if vv not in visited:
                    visited.add(vv)
                    queue.append(vv)
                    if len(visited) >= k:
                        break
        # if not enough (graph sparse), fill with closest to hub
        if len(visited) < k:
            dots = X @ X[hub]
            order = np.argsort(-dots)
            for v in order:
                vv = int(v)
                if vv not in visited:
                    visited.add(vv)
                    if len(visited) >= k:
                        break
        return np.array(sorted(visited), dtype=int)

    def cap_delete(X, k):
        r = rng.random()
        if r < 0.5:
            # hub direction
            Mask, m = contact_adj(X, delta=8e-4)
            deg = Mask.sum(axis=1)
            hub = int(np.argmax(deg)) if rng.random() < 0.7 else int(rng.integers(0, n))
            direction = X[hub] + 0.05 * rng.standard_normal(d)
        else:
            direction = rng.standard_normal(d)
        direction /= max(np.linalg.norm(direction), 1e-12)
        dots = X @ direction
        idx = np.argsort(-dots)[:k]
        return np.sort(idx.astype(int))

    def jaccard_to_inc(X):
        s, _ = top_set(X, 1500)
        inter = len(s & inc_set)
        union = len(s | inc_set)
        return inter / max(union, 1)

    # short screening cycles
    ncycle = 0
    taus_early = [0.02, 0.009, 0.004]
    iters_early = [22, 26, 30]
    taus_late = [0.002, 0.001]
    iters_late = [40, 55]
    taboo_pen = 2.0

    while time.time() < T_END - FINAL_RESERVE - 1.0 and ncycle < 60:
        ncycle += 1
        # choose base: mostly best, sometimes diverse archive
        if len(archive) > 1 and rng.random() < 0.35:
            # pick random archive entry (not necessarily best)
            ai = int(rng.integers(0, len(archive)))
            baseX = archive[ai][1]
            baseVal = archive[ai][0]
        else:
            baseX = bestX
            baseVal = bestVal
        k = int(rng.integers(18, 36))  # 18-35
        if rng.random() < 0.55:
            del_idx = bfs_delete(baseX, k)
        else:
            del_idx = cap_delete(baseX, k)
        try:
            Xnew = greedy_refill(baseX, del_idx, Cnum=60000)
        except Exception:
            continue
        # taboo boost against old contacts: use inc_mask (or base top mask)
        # compute base top mask quickly
        Gb = baseX @ baseX.T
        gb = Gb[I0, J0]
        thrb = np.sort(gb)[-1500]
        tmask = gb >= thrb
        boost = np.ones(P, dtype=float)
        boost[tmask] = 1.0 + taboo_pen
        # also randomize pen a bit
        # first re-jam with taboo
        try:
            Xj = full_anneal(Xnew, taus_early, iters_early, boost=boost)
        except Exception:
            continue
        vj = _max_cos(Xj)
        # allow intermediate up to incumbent+0.015 : always continue to second stage
        # second stage without taboo (or smaller)
        try:
            Xj2 = full_anneal(Xj, taus_late, iters_late, boost=None)
        except Exception:
            Xj2 = Xj
        vj2 = _max_cos(Xj2)
        Xc, vc = (Xj2, vj2) if vj2 <= vj else (Xj, vj)
        # evaluate diversity even if worse
        try:
            jac = jaccard_to_inc(Xc)
        except Exception:
            jac = 1.0
        # update global best
        if vc < bestVal - 1e-12:
            bestVal = vc
            bestX = Xc.copy()
            archive.append((vc, Xc.copy()))
            # keep archive sorted, prune to 8
            archive = sorted(archive, key=lambda t: t[0])[:8]
        else:
            # retain if diverse and within 2e-4
            if jac < 0.7 and vc < initVal + 2e-4 + 1e-12:
                # avoid duplicates
                archive.append((vc, Xc.copy()))
                archive = sorted(archive, key=lambda t: t[0])
                # keep at most 8, ensure diversity: drop most similar worst?
                if len(archive) > 8:
                    archive = archive[:8]
            elif vc < bestVal + 0.005 and rng.random() < 0.05:
                # occasional neutral to keep search moving (non-monotone)
                archive.append((vc, Xc.copy()))
                if len(archive) > 8:
                    archive = sorted(archive, key=lambda t: t[0])[:8]
        # early exit if time
        if time.time() > T_END - FINAL_RESERVE:
            break

    # final LP polishing on best archived code
    def lp_polish(Xinit, seconds, radius0=3e-3):
        X = Xinit.copy()
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        best = _max_cos(X)
        bestXl = X.copy()
        radius = float(radius0)
        start = time.time()
        iu = np.triu_indices(n, 1)
        while time.time() - start < seconds and radius > 1e-14:
            if time.time() > T_END - 1.0:
                break
            Gfull = X @ X.T
            c = Gfull[iu]
            m = float(c.max())
            sel = c > m - max(20.0 * radius, 2e-7)
            p0 = int(np.sum(sel))
            if p0 > 6000:
                idx = np.argpartition(c, -6000)[-6000:]
                sel = np.zeros_like(sel, dtype=bool)
                sel[idx] = True
                p0 = 6000
            if p0 < 10:
                sel = c > m - 1e-6
                p0 = int(np.sum(sel))
            I = iu[0][sel]; J = iu[1][sel]; p = p0
            csel = c[sel]
            try:
                rows = np.repeat(np.arange(p), 2 * d + 1)
                colsA = np.concatenate([I[:, None] * d + np.arange(d),
                                        J[:, None] * d + np.arange(d),
                                        np.full((p, 1), n * d)], 1).ravel()
                vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
                A_ub = coo_matrix((vals, (rows, colsA)), shape=(p, n * d + 1)).tocsr()
                erows = np.repeat(np.arange(n), d)
                ecols = np.arange(n * d)
                A_eq = coo_matrix((X.ravel(), (erows, ecols)), shape=(n, n * d + 1)).tocsr()
                obj = np.zeros(n * d + 1); obj[-1] = 1.0
                bounds = [(-radius, radius)] * (n * d) + [(None, None)]
                res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                              b_eq=np.zeros(n), bounds=bounds, method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            dx = res.x[:-1].reshape(n, d)
            y = X + dx
            y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
            v = _max_cos(y)
            if v < best:
                X, best = y, v
                bestXl = y.copy()
                radius = min(radius * 1.5, 0.02)
            else:
                radius *= 0.5
        return bestXl, best

    # choose best archived for final polish
    archive = sorted(archive, key=lambda t: t[0])
    finalX = archive[0][1].copy()
    finalV = archive[0][0]
    # if bestX better, use it (should be same)
    if bestVal < finalV:
        finalX = bestX.copy()
        finalV = bestVal
    remain = (T_END - time.time()) - 0.8
    if remain > 2.0:
        try:
            Xl, vl = lp_polish(finalX, min(remain, 24.0))
            if vl < finalV:
                finalX, finalV = Xl, vl
        except Exception:
            pass
    # also try quick softmax final squeeze if time
    if time.time() < T_END - 1.0:
        try:
            Xs = full_anneal(finalX, [0.002, 0.0008, 0.0004], [60, 80, 60], boost=None)
            vs = _max_cos(Xs)
            if vs < finalV:
                finalX, finalV = Xs, vs
                # one more tiny LP if time
                if time.time() < T_END - 4.0:
                    Xl2, vl2 = lp_polish(finalX, min(3.0, T_END - time.time() - 0.5))
                    if vl2 < finalV:
                        finalX, finalV = Xl2, vl2
        except Exception:
            pass
    finalX /= np.maximum(np.linalg.norm(finalX, axis=1, keepdims=True), 1e-12)
    return finalX
