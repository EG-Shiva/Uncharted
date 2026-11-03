import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    rng = np.random.default_rng(7)
    n, d = 249, 6
    I0, J0 = np.triu_indices(n, 1)
    P = len(I0)
    Nv = n * 5

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
    vals = [_max_cos(a) for a in cands]
    bestX = cands[int(np.argmin(vals))].copy()
    bestVal = float(np.min(vals))
    bestX /= np.maximum(np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)

    E6 = np.eye(6)

    def tangent_bases(X):
        T = np.empty((n, 6, 5), dtype=float)
        for i in range(n):
            a = X[i]
            s = -1.0 if a[0] >= 0 else 1.0
            v = a.copy()
            v[0] -= s
            nv = float(np.linalg.norm(v))
            if nv < 1e-12:
                T[i] = E6[:, 1:]
            else:
                w = (v / nv).reshape(6, 1)
                H = E6 - 2.0 * (w @ w.T)
                T[i] = H[:, 1:]
        return T

    def lp_jam(Xinit, deadline, radius_init=3e-3, slack=2e-3, max_it=50):
        nonlocal bestX, bestVal
        X = Xinit.copy()
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        mcur = _max_cos(X)
        if mcur < bestVal:
            bestVal = mcur
            bestX = X.copy()
        radius = float(radius_init)
        last_dual = None
        last_pair = None
        it = 0
        cvec = np.zeros(Nv + 1)
        cvec[Nv] = 1.0
        while it < max_it and time.time() < deadline:
            it += 1
            if radius < 1e-12:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            gfull = G[I0, J0]
            sel = np.where(gfull > m - slack)[0]
            if len(sel) < 800:
                K0 = min(P, 800)
                sel = np.argpartition(gfull, -K0)[-K0:]
            elif len(sel) > 6000:
                sel = sel[np.argpartition(gfull[sel], -6000)[-6000:]]
            Ic = I0[sel]; Jc = J0[sel]; cc = gfull[sel]
            K = len(sel)
            T = tangent_bases(X)
            TI = T[Ic]
            TJ = T[Jc]
            XJ = X[Jc]
            XI = X[Ic]
            Ai = np.sum(TI * XJ[:, :, None], axis=1)
            Aj = np.sum(TJ * XI[:, :, None], axis=1)
            r1 = np.repeat(np.arange(K), 5)
            rows = np.concatenate([r1, r1, np.arange(K)])
            cols = np.concatenate([
                (Ic[:, None] * 5 + np.arange(5)).ravel(),
                (Jc[:, None] * 5 + np.arange(5)).ravel(),
                np.full(K, Nv, dtype=np.int64)
            ]).astype(np.int64)
            data = np.concatenate([Ai.ravel(), Aj.ravel(), -np.ones(K)])
            try:
                Aub = coo_matrix((data, (rows, cols)), shape=(K, Nv + 1)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            bub = -cc
            bnds = [(-radius, radius)] * Nv + [(None, float(m))]
            try:
                res = linprog(cvec, A_ub=Aub, b_ub=bub, bounds=bnds, method='highs',
                              options={'presolve': True})
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0 or res.x is None:
                radius *= 0.5
                continue
            tpred = float(res.x[Nv])
            try:
                mg = getattr(getattr(res, 'ineqlin', None), 'marginals', None)
                if mg is not None:
                    last_dual = -np.asarray(mg, dtype=float).ravel()
                    last_pair = (Ic.copy(), Jc.copy())
            except Exception:
                pass
            if not np.isfinite(tpred) or tpred >= m - 1e-13:
                radius *= 0.5
                if radius < 5e-11:
                    break
                continue
            y = res.x[:Nv]
            Y = y.reshape(n, 5)
            V = np.einsum('ijk,ik->ij', T, Y)
            Xt = X + V
            Xt /= np.maximum(np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)
            vnew = _max_cos(Xt)
            if vnew < mcur - 1e-12:
                X = Xt
                mcur = vnew
                if vnew < bestVal:
                    bestVal = vnew
                    bestX = Xt.copy()
                radius = min(radius * 1.35, 8e-3)
                if radius < 5e-11 and abs(tpred - mcur) < 1e-9:
                    break
            else:
                radius *= 0.5
        return X, mcur, last_dual, last_pair

    def contacts(X, tol=1e-9):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        mask = G > m - tol
        cnt = np.sum(mask, axis=1)
        rowm = np.max(G, axis=1)
        return cnt, rowm, m, G

    def greedy_refill(Xbase, del_idx, Cnum=25000):
        k = len(del_idx)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[del_idx] = False
        Xkeep = Xbase[keep_mask]
        Cand = rng.standard_normal(size=(Cnum, 6))
        Cand /= np.maximum(np.linalg.norm(Cand, axis=1, keepdims=True), 1e-12)
        D = Cand @ Xkeep.T
        curmax = D.max(axis=1)
        used = np.zeros(Cnum, dtype=bool)
        new_pts = []
        for _ in range(k):
            curmax[used] = np.inf
            bi = int(np.argmin(curmax))
            used[bi] = True
            pt = Cand[bi].copy()
            new_pts.append(pt)
            dd = Cand @ pt
            np.maximum(curmax, dd, out=curmax)
            curmax[used] = np.inf
        new_pts = np.stack(new_pts, axis=0)
        Xt = Xbase.copy()
        Xt[del_idx] = new_pts
        Xt /= np.maximum(np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)
        return Xt

    def far_reinsert(Xbase, idx_list, Cnum=8000):
        Xt = Xbase.copy()
        Cand = rng.standard_normal(size=(Cnum, 6))
        Cand /= np.maximum(np.linalg.norm(Cand, axis=1, keepdims=True), 1e-12)
        for w in idx_list:
            dots = Cand @ Xt.T
            dots[:, int(w)] = -2.0
            mx = np.max(dots, axis=1)
            Xt[int(w)] = Cand[int(np.argmin(mx))]
        Xt /= np.maximum(np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)
        return Xt

    # stage 1: exact jam incumbent
    try:
        if time.time() < T_END - 40:
            lp_jam(bestX.copy(), T_END - 80, radius_init=3e-3, slack=2e-3, max_it=55)
    except Exception:
        pass

    cyc = 0
    while time.time() < T_END - 14:
        cyc += 1
        if cyc > 60:
            break
        try:
            cnt, rowm, mcur, Gf = contacts(bestX, tol=1e-9)
            ratt = np.where(cnt == 0)[0]
            order_low = np.argsort(cnt + rowm * 1e-6)
            order_high = np.argsort(-cnt)
            order_worst = np.argsort(-rowm)
            mode = cyc % 4
            Xt = None
            if mode == 1:
                # hub BFS neighbourhood deletion
                hub = int(order_high[0]) if len(order_high) > 0 else int(order_worst[0])
                # nearest neighbours by dot
                dots_h = bestX @ bestX[hub]
                dots_h[hub] = -2
                k = 10 + (cyc % 3)  # 10..12
                nbr = np.argpartition(-dots_h, k - 1)[:k - 1]
                del_idx = np.unique(np.concatenate([[hub], nbr]))
                Xt = greedy_refill(bestX, del_idx, Cnum=25000)
            elif mode == 2:
                # random spherical cap deletion
                pole = rng.standard_normal(size=6)
                pole /= np.linalg.norm(pole)
                dots_p = bestX @ pole
                k = 8 + (cyc % 4)  # 8..11
                del_idx = np.argpartition(-dots_p, k - 1)[:k]
                Xt = greedy_refill(bestX, del_idx, Cnum=25000)
            elif mode == 3:
                # rattler + worst far reinsert
                cand_idx = []
                if len(ratt) > 0:
                    # worst rattlers by rowm
                    ro = ratt[np.argsort(-rowm[ratt])]
                    cand_idx.extend(ro[:2].tolist())
                cand_idx.extend(order_worst[:2].tolist())
                cand_idx = list(dict.fromkeys([int(c) for c in cand_idx]))[:3]
                if len(cand_idx) == 0:
                    cand_idx = order_worst[:2].tolist()
                Xt = far_reinsert(bestX, cand_idx, Cnum=12000)
                # plus small tangent jitter of low-contact points
                T = tangent_bases(bestX)
                jit = order_low[:6]
                for ii in jit:
                    ii = int(ii)
                    if ii in cand_idx:
                        continue
                    step = rng.standard_normal(size=5) * 0.015
                    Xt[ii] = Xt[ii] + T[ii] @ step
                Xt /= np.maximum(np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)
            else:
                # jitter low-contact / multiplier-free points
                T = tangent_bases(bestX)
                Xt = bestX.copy()
                nmove = 10
                tgt = order_low[:nmove]
                kick = 0.02 if (cyc % 2 == 0) else 0.035
                for ii in tgt:
                    ii = int(ii)
                    step = rng.standard_normal(size=5) * kick
                    Xt[ii] = Xt[ii] + T[ii] @ step
                Xt /= np.maximum(np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)
            if Xt is None:
                continue
            tlim = min(T_END - 10, time.time() + 13)
            lp_jam(Xt, tlim, radius_init=2e-3, slack=2e-3, max_it=24)
            # occasional polish of best directly
            if cyc % 6 == 0 and time.time() < T_END - 12:
                lp_jam(bestX.copy(), min(T_END - 10, time.time() + 5), radius_init=5e-4, slack=1.5e-3, max_it=14)
        except Exception:
            continue

    try:
        if time.time() < T_END - 3:
            lp_jam(bestX.copy(), T_END - 0.8, radius_init=5e-4, slack=1.5e-3, max_it=25)
    except Exception:
        pass

    bestX /= np.maximum(np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)
    return bestX
