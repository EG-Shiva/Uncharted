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
    rng = np.random.default_rng(0)
    n, d = 249, 6
    I0, J0 = np.triu_indices(n, 1)

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

    def tangent_bases(X):
        T = np.empty((n, 6, 5), dtype=float)
        E = np.eye(6)
        for i in range(n):
            a = X[i]
            s = -1.0 if a[0] >= 0 else 1.0
            v = a.copy()
            v[0] -= s
            nv = float(np.linalg.norm(v))
            if nv < 1e-12:
                T[i] = E[:, 1:]
            else:
                w = (v / nv).reshape(6, 1)
                H = E - 2.0 * (w @ w.T)
                T[i] = H[:, 1:]
        return T

    def lp_jam(Xinit, deadline, radius_init=2e-3, slack=2e-3, max_it=60):
        nonlocal bestX, bestVal
        X = Xinit.copy()
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        mcur = _max_cos(X)
        if mcur < bestVal:
            bestVal = mcur
            bestX = X.copy()
        radius = float(radius_init)
        last_dual = None
        last_sel = None
        it = 0
        fails = 0
        Nv = n * 5
        while it < max_it and time.time() < deadline:
            it += 1
            if radius < 1e-12:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            gfull = G[I0, J0]
            sel = np.where(gfull > m - slack)[0]
            if len(sel) == 0:
                sel = np.argsort(gfull)[-500:]
            if len(sel) > 4000:
                part = np.argpartition(gfull[sel], -4000)[-4000:]
                sel = sel[part]
            Ic = I0[sel]; Jc = J0[sel]; cc = gfull[sel]
            K = len(sel)
            T = tangent_bases(X)
            # vectorized linear coeffs: ai = T_i^T x_j
            TI = T[Ic]  # K,6,5
            TJ = T[Jc]
            XJ = X[Jc]  # K,6
            XI = X[Ic]
            Ai = np.sum(TI * XJ[:, :, None], axis=1)  # K,5
            Aj = np.sum(TJ * XI[:, :, None], axis=1)
            # sparse A_ub: K x (Nv+1)
            rows = np.repeat(np.arange(K), 11)
            ci = np.repeat(Ic * 5, 5) + np.tile(np.arange(5), K)
            cj = np.repeat(Jc * 5, 5) + np.tile(np.arange(5), K)
            cols = np.empty(K * 11, dtype=np.int64)
            cols[0::11] = 0  # placeholder, fill below vectorized
            # build cols array blockwise
            cols = np.concatenate([
                (Ic[:, None] * 5 + np.arange(5)).ravel(),
                (Jc[:, None] * 5 + np.arange(5)).ravel(),
                np.full(K, Nv)
            ])
            # need rows matching cols ordering: first K*5 for Ai, next K*5 for Aj, last K for t
            r1 = np.repeat(np.arange(K), 5)
            rows = np.concatenate([r1, r1, np.arange(K)])
            data = np.concatenate([Ai.ravel(), Aj.ravel(), -np.ones(K)])
            try:
                Aub = coo_matrix((data, (rows, cols)), shape=(K, Nv + 1)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            bub = -cc
            cvec = np.zeros(Nv + 1); cvec[Nv] = 1.0
            bnds = [(-radius, radius)] * Nv + [(None, None)]
            try:
                res = linprog(cvec, A_ub=Aub, b_ub=bub, bounds=bnds, method='highs',
                              options={'presolve': True})
            except Exception:
                radius *= 0.5
                fails += 1
                if fails > 5:
                    break
                continue
            if res.status != 0 or res.x is None:
                radius *= 0.5
                fails += 1
                if fails > 8:
                    break
                continue
            fails = 0
            y = res.x[:Nv]
            # duals for force balance
            try:
                marg = getattr(getattr(res, 'ineqlin', None), 'marginals', None)
                if marg is not None:
                    last_dual = -np.asarray(marg, dtype=float).ravel()
                    last_sel = (Ic.copy(), Jc.copy())
                else:
                    last_dual = None
            except Exception:
                last_dual = None
            Y = y.reshape(n, 5)
            V = np.einsum('ijk,ik->ij', T, Y)
            Xt = X + V
            Xt /= np.maximum(np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)
            vnew = _max_cos(Xt)
            # accept only if exact max decreases
            if vnew < mcur - 1e-12:
                X = Xt
                mcur = vnew
                if vnew < bestVal:
                    bestVal = vnew
                    bestX = Xt.copy()
                radius = min(radius * 1.4, 5e-3)
            elif vnew < mcur + 1e-12 and radius < 1e-9:
                X = Xt
                radius *= 0.5
            else:
                radius *= 0.5
            # certified jam: tiny radius and prediction tight
            try:
                tpred = float(res.x[Nv])
                if radius < 5e-11 and abs(tpred - mcur) < 1e-9:
                    break
            except Exception:
                pass
        return X, mcur, last_dual, last_sel

    def find_rattlers(X, tol=1e-7):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        mask = G > m - tol
        cnt = np.sum(mask, axis=1)
        rowm = np.max(G, axis=1)
        return cnt, rowm, m, mask

    # stage 1: exact jam from incumbent
    try:
        if time.time() < T_END - 30:
            _, _, _, _ = lp_jam(bestX.copy(), T_END - 60, radius_init=2e-3, slack=2e-3, max_it=50)
    except Exception:
        pass

    # stage 2: rattler / multiplier-free targeted moves + re-jam
    kicks = [0.025, 0.05, 0.012, 0.035, 0.008]
    cyc = 0
    while time.time() < T_END - 12:
        cyc += 1
        try:
            cnt, rowm, mcur, _ = find_rattlers(bestX, tol=1e-7)
            ratt = np.where(cnt == 0)[0]
            if len(ratt) == 0:
                ratt = np.argsort(cnt)[:10]
            # worst rattlers by row max
            order = ratt[np.argsort(-rowm[ratt])]
            kick = kicks[(cyc - 1) % len(kicks)]
            nmove = 6 if cyc % 3 else 12
            tomove = order[:min(nmove, len(order))]
            # also add a multiplier-free random point occasionally
            T = tangent_bases(bestX)
            Xt = bestX.copy()
            for idx in tomove:
                step = rng.standard_normal(size=5) * kick
                # clip to trust
                v = T[int(idx)] @ step
                Xt[int(idx)] = Xt[int(idx)] + v
            # occasional single far reinsert for worst point
            if cyc % 4 == 0 and len(order) > 0:
                w = int(order[0])
                cand = rng.standard_normal(size=(2000, 6))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                dots = cand @ Xt.T
                dots[:, w] = -2.0
                # minimize max dot
                mx = np.max(dots, axis=1)
                Xt[w] = cand[int(np.argmin(mx))]
            Xt /= np.maximum(np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)
            # short jam, must improve to keep; lp_jam updates global best on its own
            before = bestVal
            Xj, vj, dual, selpair = lp_jam(Xt, min(T_END - 8, time.time() + 14), radius_init=min(kick * 0.2, 2e-3), slack=2e-3, max_it=22)
            # if no gain, try smaller kick jam from best directly (polish)
            if vj >= before - 1e-12 and cyc % 5 == 0:
                if time.time() < T_END - 10:
                    lp_jam(bestX.copy(), time.time() + 6, radius_init=5e-4, slack=2e-3, max_it=15)
            if cyc > 40:
                break
        except Exception:
            continue

    # final tiny polish jam on best
    try:
        if time.time() < T_END - 4:
            lp_jam(bestX.copy(), T_END - 1.0, radius_init=5e-4, slack=2e-3, max_it=20)
    except Exception:
        pass

    bestX /= np.maximum(np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)
    # strict feasibility: no NaN, no zero rows (already unit)
    return bestX
