import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    FINAL_RESERVE = 25.0
    rng = np.random.default_rng(42)
    n, d = 249, 6
    I0, J0 = np.triu_indices(n, 1)
    P = len(I0)
    # load start
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal(size=(n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [_max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    bestX = cands[bi].copy()
    bestVal = float(vals[bi])
    # archive: list of (val, X)
    archive = [(bestVal, bestX.copy())]

    def top_pairs_set(X, K=1500):
        G = X @ X.T
        g = G[I0, J0]
        if K >= P:
            idx = np.arange(P)
        else:
            idx = np.argpartition(g, -K)[-K:]
        codes = I0[idx].astype(np.int64) * n + J0[idx].astype(np.int64)
        return set(codes.tolist())

    inc_set = top_pairs_set(bestX, 1500)

    def full_anneal(Xinit, taus, iters, taboo_mask=None, taboo_pen=0.0, creg=0.08):
        Xf = Xinit.ravel().copy()
        boost = None
        if taboo_mask is not None and taboo_pen > 0:
            boost = np.ones(P, dtype=float)
            boost[taboo_mask] = 1.0 + taboo_pen
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
                gm = np.max(g)
                e = np.exp((g - gm) / tau)
                if boost is not None:
                    e = e * boost
                S = np.sum(e)
                Lsoft = gm + tau * np.log(S)
                w = e / S
                W = np.zeros((n, n), dtype=float)
                W[I0, J0] = w
                W = W + W.T
                A = W @ Xn
                proj = np.sum(A * Xn, axis=1)
                G0 = (A - proj[:, None] * Xn) / r[:, None]
                diff = r - 1.0
                L = Lsoft + creg * np.sum(diff * diff)
                Gr = G0 + (2.0 * creg * diff)[:, None] * Xn
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

    def greedy_refill(Xbase, del_idx, cand_pool):
        # cand_pool (C,6) normalized
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[del_idx] = False
        Xkeep = Xbase[keep_mask]  # (n-k,6)
        # scores vs keep
        D = cand_pool @ Xkeep.T  # C x (n-k)
        curmax = D.max(axis=1)
        C = cand_pool.shape[0]
        used = np.zeros(C, dtype=bool)
        new_pts = []
        for _ in range(len(del_idx)):
            # pick best unused
            curmax[used] = np.inf
            bi_ = int(np.argmin(curmax))
            used[bi_] = True
            pt = cand_pool[bi_].copy()
            new_pts.append(pt)
            # update curmax with this new point
            dd = cand_pool @ pt  # C,
            # also mutual: max with new
            np.maximum(curmax, dd, out=curmax)
            curmax[used] = np.inf
        new_pts = np.stack(new_pts, axis=0)
        # hole local optimization for 2 worst new points vs fixed set
        # fixed = Xkeep + other new pts (leave-one-out approx with Xkeep only for speed + others)
        # do 2 rounds of single-vector softmax polish
        for _ in range(2):
            # compute max dot of each new pt vs rest
            allX = np.vstack([Xkeep, new_pts])
            # find worst new pt
            worst = -1
            worstv = -2
            for s in range(len(new_pts)):
                idx_all = (n - len(del_idx)) + s
                dots = allX @ new_pts[s]
                dots[idx_all] = -2
                mv = float(dots.max())
                if mv > worstv:
                    worstv = mv
                    worst = s
            # optimize new_pts[worst] against others
            others = np.delete(np.vstack([Xkeep, new_pts]), (n - len(del_idx)) + worst, axis=0)
            v0 = new_pts[worst].copy()
            tau_h = 0.004
            def fg1(yf):
                y = yf.reshape(6)
                rn = max(np.sqrt(np.sum(y * y)), 1e-12)
                yn = y / rn
                dots = others @ yn
                gm = float(dots.max())
                e = np.exp((dots - gm) / tau_h)
                S = float(e.sum())
                L = gm + tau_h * np.log(S) + 0.5 * (rn - 1.0) ** 2
                w = e / S
                A = w @ others  # (6,)
                proj = float(A @ yn)
                G0 = (A - proj * yn) / rn
                Gr = G0 + (rn - 1.0) * yn
                return float(L), Gr.reshape(6)
            try:
                res = minimize(fg1, v0.copy(), jac=True, method='L-BFGS-B',
                               options={'maxiter': 60, 'maxfun': 240, 'ftol': 1e-14, 'gtol': 1e-8})
                vo = res.x.reshape(6)
                vo /= max(np.linalg.norm(vo), 1e-12)
                # accept if improves max
                oldm = float(np.max(others @ v0))
                newm = float(np.max(others @ vo))
                if newm < oldm:
                    new_pts[worst] = vo
            except Exception:
                pass
        Xt = Xbase.copy()
        Xt[del_idx] = new_pts
        Xt /= np.maximum(np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)
        return Xt

    def jaccard(a, b):
        if not a and not b:
            return 1.0
        inter = len(a & b)
        union = len(a | b)
        return inter / max(union, 1)

    taus_fast = [0.02, 0.01, 0.005, 0.002, 0.001]
    iters_fast = [40, 40, 55, 65, 80]

    ncycle = 0
    # pre-generate? generate per cycle for diversity but 60k each time
    while time.time() < T_END - FINAL_RESERVE:
        ncycle += 1
        # pick base: 70% best, else random archive member
        if len(archive) > 1 and rng.random() < 0.30:
            ai = rng.integers(len(archive))
            Xbase = archive[ai][1]
            baseVal = archive[ai][0]
        else:
            Xbase = bestX
            baseVal = bestVal
        Gbase = Xbase @ Xbase.T
        np.fill_diagonal(Gbase, -2.0)
        mbase = float(Gbase.max())
        # contact degrees for hub (threshold 5e-4 slack)
        thr_deg = mbase - 5e-4
        adj = Gbase > thr_deg
        deg = adj.sum(axis=1)
        k = int(rng.integers(18, 36))  # 18-35 inclusive
        mode = rng.random()
        if mode < 0.5:
            # hub BFS neighbourhood
            hub = int(np.argmax(deg + rng.random(n) * 0.01))
            # BFS using tighter adjacency (1e-3)
            thr_bfs = mbase - 1e-3
            adjb = Gbase > thr_bfs
            seen = np.zeros(n, dtype=bool)
            seen[hub] = True
            frontier = [hub]
            order = [hub]
            while frontier and len(order) < k:
                nxt = []
                for u in frontier:
                    nbrs = np.where(adjb[u])[0]
                    # randomize order
                    if len(nbrs) > 1:
                        nbrs = nbrs[rng.permutation(len(nbrs))]
                    for w in nbrs:
                        if not seen[w]:
                            seen[w] = True
                            order.append(w)
                            nxt.append(w)
                            if len(order) >= k:
                                break
                    if len(order) >= k:
                        break
                frontier = nxt
                if not frontier:
                    break
            if len(order) < k:
                # fill with closest to hub (largest dots)
                dots_h = (Xbase @ Xbase[hub])
                dots_h[seen] = -2
                extra = np.argsort(dots_h)[- (k - len(order)):]
                for e_ in extra:
                    if not seen[e_]:
                        seen[e_] = True
                        order.append(int(e_))
            del_idx = np.array(order[:k], dtype=int)
        else:
            # spherical cap around random/hub direction
            if rng.random() < 0.5:
                hub = int(np.argmax(deg))
                direction = Xbase[hub] + 0.25 * rng.standard_normal(6)
            else:
                direction = rng.standard_normal(6)
            direction /= max(np.linalg.norm(direction), 1e-12)
            dots = Xbase @ direction
            # add tiny noise to break ties deterministically varied
            dots = dots + 1e-12 * rng.standard_normal(n)
            del_idx = np.argsort(dots)[-k:]
            del_idx = np.asarray(del_idx, dtype=int)
        # taboo mask: old contacts among survivors
        thr_taboo = mbase - 4e-4
        # find old contact pairs
        # to avoid O(P) scan heavy? P=30k fine
        gold = (Xbase @ Xbase.T)[I0, J0]
        is_contact = gold > thr_taboo
        # only survivor-survivor
        del_mask = np.zeros(n, dtype=bool)
        del_mask[del_idx] = True
        surv_pair = (~del_mask[I0]) & (~del_mask[J0])
        taboo_mask = is_contact & surv_pair
        # candidate pool 60k
        cand_pool = rng.standard_normal(size=(60000, 6))
        cand_pool /= np.maximum(np.linalg.norm(cand_pool, axis=1, keepdims=True), 1e-12)
        try:
            Xtrial = greedy_refill(Xbase, del_idx, cand_pool)
        except Exception:
            continue
        # re-jam: first tau with taboo penalty, rest without
        # split: do first stage separately with penalty
        try:
            Xj = full_anneal(Xtrial, [taus_fast[0]], [iters_fast[0]], taboo_mask=taboo_mask, taboo_pen=2.5)
            # remaining without penalty, allow worse intermediate (no rejection)
            Xj = full_anneal(Xj, taus_fast[1:], iters_fast[1:], taboo_mask=None)
        except Exception:
            continue
        vj = _max_cos(Xj)
        # jaccard vs incumbent best set
        try:
            sj = top_pairs_set(Xj, 1500)
            js = jaccard(sj, inc_set)
        except Exception:
            js = 1.0
        # non-monotone acceptance into archive
        # allow up to +0.015 for archive diversity, +2e-4 for LP-polish retention
        if vj < bestVal:
            bestVal = float(vj)
            bestX = Xj.copy()
            inc_set = top_pairs_set(bestX, 1500)
            archive.append((bestVal, bestX.copy()))
            # keep archive small, ranked by val; prefer keeping best 6
            archive = sorted(archive, key=lambda t: t[0])[:6]
        elif vj < bestVal + 2e-4 and js < 0.7:
            archive.append((float(vj), Xj.copy()))
            archive = sorted(archive, key=lambda t: t[0])[:6]
            # also keep diverse worse ones for future bases if very diverse
        elif vj < bestVal + 0.005 and js < 0.55:
            # very novel topology, keep as base for future avalanches
            if len(archive) < 6:
                archive.append((float(vj), Xj.copy()))
                archive = sorted(archive, key=lambda t: t[0])[:6]
            else:
                # replace worst if more diverse? just keep best vals
                pass
        # occasional: if stuck many cycles, perturb best slightly to avoid repeat? handled by randomness
        if time.time() > T_END - FINAL_RESERVE - 1:
            break

    # final LP polish best archived code, reserve ~FINAL_RESERVE
    def lp_polish(X, deadline):
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        best = _max_cos(X)
        bx = X.copy()
        radius = 2e-3
        iuA, iuB = np.triu_indices(n, 1)
        while time.time() < deadline - 0.5 and radius > 1e-14:
            Gf = X @ X.T
            c = Gf[iuA, iuB]
            m = float(c.max())
            sel = c > m - max(20.0 * radius, 2e-7)
            p0 = int(np.count_nonzero(sel))
            if p0 == 0:
                break
            if p0 > 5500:
                idx = np.argpartition(c, -5500)[-5500:]
                tmp = np.zeros_like(sel)
                tmp[idx] = True
                sel = tmp
            I = iuA[sel]; J = iuB[sel]
            p = len(I)
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            valsA = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            try:
                A_ub = coo_matrix((valsA, (rows, cols)), shape=(p, n * d + 1)).tocsr()
                A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
                obj = np.zeros(n * d + 1)
                obj[-1] = 1.0
                tl = max(1.0, deadline - time.time() - 0.3)
                res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                              method="highs", options={"time_limit": tl})
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = X + res.x[:-1].reshape(n, d)
            y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
            v = _max_cos(y)
            if v < best:
                best = v
                bx = y.copy()
                X = y
                radius = min(radius * 1.5, 0.02)
            else:
                radius *= 0.5
        return bx, best

    # polish archive best-first
    archive = sorted(archive, key=lambda t: t[0])
    deadline = T_END - 1.0
    # try best then others if time
    for (av, ax) in list(archive):
        if time.time() > deadline - 2:
            break
        try:
            y, vy = lp_polish(ax.copy(), deadline)
            if vy < bestVal:
                bestVal = vy
                bestX = y.copy()
        except Exception:
            continue
    bestX = bestX / np.maximum(np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)
    return bestX
