import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import csr_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    RESERVE = 18.0
    rng = np.random.default_rng(12345)
    n, d = 249, 6
    I0, J0 = np.triu_indices(n, 1)
    P = len(I0)
    NV = n * 5

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
    bestX = cands[int(np.argmin(vals))].copy()
    bestVal = float(np.min(vals))

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
                w = v / nv
                H = E - 2.0 * np.outer(w, w)
                T[i] = H[:, 1:]
        return T

    def lp_jam(Xinit, max_iter=40, init_radius=0.008, slack=2e-3, tlim=None):
        X = Xinit.copy()
        m = _max_cos(X)
        T = tangent_bases(X)
        radius = float(init_radius)
        c = np.zeros(NV + 1)
        c[-1] = 1.0
        it = 0
        while it < max_iter:
            if time.time() > (tlim if tlim is not None else T_END - 4.0):
                break
            it += 1
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            g = G[I0, J0]
            sel = np.where(g > m - slack)[0]
            if len(sel) < 800:
                # ensure enough constraints for rigidity test
                K0 = min(P, 800)
                sel = np.argpartition(g, -K0)[-K0:]
            elif len(sel) > 8000:
                K0 = 8000
                sel = np.argpartition(g, -K0)[-K0:]
            Ic = I0[sel]; Jc = J0[sel]
            gc = g[sel]
            K = len(sel)
            # build sparse A_ub K x (NV+1)
            # each row has 5+5+1 nonzeros
            nnz = K * 11
            ridx = np.empty(nnz, dtype=np.int64)
            cidx = np.empty(nnz, dtype=np.int64)
            data = np.empty(nnz, dtype=float)
            p = 0
            # vectorized ai, aj
            # T[i] is 6x5, X[j] is 6 -> ai = T[i].T @ X[j]
            for k in range(K):
                i = int(Ic[k]); j = int(Jc[k])
                ai = T[i].T @ X[j]
                aj = T[j].T @ X[i]
                base = k * 11
                for s in range(5):
                    ridx[base+s] = k; cidx[base+s] = i*5+s; data[base+s] = ai[s]
                    ridx[base+5+s] = k; cidx[base+5+s] = j*5+s; data[base+5+s] = aj[s]
                ridx[base+10] = k; cidx[base+10] = NV; data[base+10] = -1.0
            A = csr_matrix((data, (ridx, cidx)), shape=(K, NV+1))
            b = -gc
            lb_u = -radius; ub_u = radius
            bounds = [(lb_u, ub_u)] * NV + [(-1.0, float(m))]
            try:
                res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs',
                              options={'presolve': True})
            except Exception:
                radius *= 0.5
                if radius < 1e-9:
                    break
                continue
            if res.status != 0 or res.x is None:
                radius *= 0.5
                if radius < 5e-9:
                    break
                continue
            u = res.x[:NV]
            t_pred = float(res.x[NV])
            # need predicted gain
            if not np.isfinite(t_pred) or t_pred >= m - 1e-13:
                # no predicted progress: try shrinking slack/radius -> declare jammed
                # if radius still large, halve and retry once to certify
                if radius > 2e-6 and it < max_iter:
                    radius *= 0.5
                    # if radius tiny, break
                    if radius < 2e-7:
                        break
                    continue
                break
            U = u.reshape(n, 5)
            # apply step: X + T*U
            Xn = X + np.einsum('nik,nk->ni', T, U)
            Xn /= np.maximum(np.linalg.norm(Xn, axis=1, keepdims=True), 1e-12)
            m_new = _max_cos(Xn)
            if m_new < m - 1e-12:
                X = Xn
                T = tangent_bases(X)
                radius = min(0.015, radius * 1.25)
            else:
                radius *= 0.5
                if radius < 5e-9:
                    break
        return X, _max_cos(X)

    def softmax_anneal(Xinit, taus, iters, creg=0.08, tlim=None):
        xf = (Xinit / np.maximum(np.linalg.norm(Xinit, axis=1, keepdims=True), 1e-12)).ravel().copy()
        for tau, mi in zip(taus, iters):
            if time.time() > (tlim if tlim is not None else T_END - RESERVE):
                break
            tau = float(tau)
            def fg(xflat, tau=tau):
                X = xflat.reshape(n, d)
                r = np.sqrt(np.sum(X * X, axis=1))
                r = np.maximum(r, 1e-12)
                Xn = X / r[:, None]
                G = Xn @ Xn.T
                g = G[I0, J0]
                gm = np.max(g)
                e = np.exp((g - gm) / tau)
                S = np.sum(e)
                Lsoft = gm + tau * np.log(S)
                w = e / S
                W = np.zeros((n, n))
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
                res = minimize(fg, xf, jac=True, method='L-BFGS-B',
                               options={'maxiter': int(mi), 'maxfun': int(mi*3),
                                        'ftol': 1e-14, 'gtol': 1e-7, 'maxcor': 8})
                xf = res.x
            except Exception:
                break
            try:
                Xm = xf.reshape(n, d)
                rn = np.maximum(np.linalg.norm(Xm, axis=1, keepdims=True), 1e-12)
                xf = (Xm / rn).ravel()
            except Exception:
                break
        Xo = xf.reshape(n, d)
        Xo /= np.maximum(np.linalg.norm(Xo, axis=1, keepdims=True), 1e-12)
        return Xo

    def greedy_refill(Xbase, del_idx, pool):
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[del_idx] = False
        Xkeep = Xbase[keep_mask]
        D = pool @ Xkeep.T
        curmax = D.max(axis=1)
        used = np.zeros(pool.shape[0], dtype=bool)
        new_pts = []
        for _ in range(len(del_idx)):
            curmax[used] = np.inf
            b = int(np.argmin(curmax))
            used[b] = True
            pt = pool[b].copy()
            new_pts.append(pt)
            dd = pool @ pt
            np.maximum(curmax, dd, out=curmax)
            curmax[used] = np.inf
        new_pts = np.stack(new_pts, axis=0)
        Xt = Xbase.copy()
        Xt[del_idx] = new_pts
        Xt /= np.maximum(np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)
        return Xt

    def contacts_info(X, tol=1e-9):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        C = (G > m - tol)
        # count upper only per point
        cnt = (C.sum(axis=1)).astype(int)
        return m, cnt, G

    # Phase 0: exact jam of incumbent (certify / gain)
    try:
        Xj0, vj0 = lp_jam(bestX, max_iter=50, init_radius=0.008, slack=2e-3, tlim=T_END-RESERVE-90)
        if vj0 < bestVal and np.isfinite(vj0):
            bestX, bestVal = Xj0, float(vj0)
    except Exception:
        pass

    taus_fast = [0.01, 0.005, 0.002]
    iters_fast = [25, 30, 40]
    taus_coarse = [0.02, 0.01, 0.005]
    iters_coarse = [20, 20, 25]

    ncyc = 0
    while time.time() < T_END - RESERVE:
        ncyc += 1
        # identify rattlers / multiplier-free points from current best
        try:
            mcur, cnt, Gcur = contacts_info(bestX, tol=1e-9)
            ratt = np.where(cnt == 0)[0]
            # also near-rattlers at 1e-6
            _, cnt6, _ = contacts_info(bestX, tol=2e-6)
            # degrees for hub
            deg = cnt6
        except Exception:
            ratt = np.array([], dtype=int)
            deg = np.zeros(n, dtype=int)
            mcur = bestVal
            Gcur = bestX @ bestX.T
            np.fill_diagonal(Gcur, -2.0)
        op = rng.random()
        Xbase = bestX
        try:
            if op < 0.40 and len(ratt) > 3:
                # targeted rattler moves: delete subset of rattlers with worst rowmax
                k = int(rng.integers(5, 21))
                k = min(k, len(ratt))
                # rowmax for rattlers
                rowmax = np.max(Gcur, axis=1)
                rscore = rowmax[ratt] + 1e-9 * rng.standard_normal(len(ratt))
                del_idx = ratt[np.argsort(rscore)[-k:]]
                pool = rng.standard_normal(size=(20000, 6))
                pool /= np.maximum(np.linalg.norm(pool, axis=1, keepdims=True), 1e-12)
                Xt = greedy_refill(Xbase, del_idx, pool)
                Xc = softmax_anneal(Xt, taus_fast, iters_fast)
                Xj, vj = lp_jam(Xc, max_iter=14, init_radius=0.006, slack=2e-3)
            elif op < 0.70:
                # hub cap avalanche
                k = int(rng.integers(30, 65))
                # hub = max degree
                hub = int(np.argmax(deg)) if deg.sum() > 0 else int(np.argmax(np.max(Gcur, axis=1)))
                direction = Xbase[hub] + 0.25 * rng.standard_normal(6)
                direction /= max(np.linalg.norm(direction), 1e-12)
                dots = Xbase @ direction + 1e-12 * rng.standard_normal(n)
                del_idx = np.argsort(dots)[-k:]
                pool = rng.standard_normal(size=(22000, 6))
                pool /= np.maximum(np.linalg.norm(pool, axis=1, keepdims=True), 1e-12)
                Xt = greedy_refill(Xbase, del_idx, pool)
                Xc = softmax_anneal(Xt, taus_fast, iters_fast)
                Xj, vj = lp_jam(Xc, max_iter=14, init_radius=0.006, slack=2e-3)
            elif op < 0.88:
                # rigid twist of cap + jitter then rejam
                k = int(rng.integers(25, 61))
                direction = rng.standard_normal(6)
                direction /= max(np.linalg.norm(direction), 1e-12)
                dots = Xbase @ direction
                cap = np.argsort(dots)[-k:]
                a = rng.standard_normal(6); b = rng.standard_normal(6)
                a /= max(np.linalg.norm(a), 1e-12)
                b = b - float(b @ a) * a
                bn = np.linalg.norm(b)
                if bn < 1e-9:
                    b = rng.standard_normal(6); b /= max(np.linalg.norm(b), 1e-12)
                else:
                    b /= bn
                th = float(rng.uniform(0.12, 0.38)) * (1.0 if rng.random() < 0.5 else -1.0)
                cth = np.cos(th); sth = np.sin(th)
                Q = np.eye(6) + (cth-1.0)*(np.outer(a,a)+np.outer(b,b)) + sth*(np.outer(b,a)-np.outer(a,b))
                Xt = Xbase.copy()
                Xt[cap] = (Q @ Xt[cap].T).T
                Xt[cap] += 0.02 * rng.standard_normal(size=(k, 6))
                Xt[cap] /= np.maximum(np.linalg.norm(Xt[cap], axis=1, keepdims=True), 1e-12)
                Xc = softmax_anneal(Xt, taus_fast, iters_fast)
                Xj, vj = lp_jam(Xc, max_iter=14, init_radius=0.006, slack=2e-3)
            else:
                # fresh basin coarse then LP
                Xr = rng.standard_normal(size=(n, d))
                Xr /= np.linalg.norm(Xr, axis=1, keepdims=True)
                Xc = softmax_anneal(Xr, taus_coarse, iters_coarse)
                Xj, vj = lp_jam(Xc, max_iter=14, init_radius=0.006, slack=2e-3)
        except Exception:
            continue
        if not np.all(np.isfinite(Xj)):
            continue
        # LP gain test: full jam promising candidates
        if vj < bestVal - 2e-7:
            try:
                Xf, vf = lp_jam(Xj, max_iter=60, init_radius=0.005, slack=2e-3)
                if vf < vj:
                    Xj, vj = Xf, vf
            except Exception:
                pass
        if vj < bestVal and np.isfinite(vj):
            bestVal = float(vj); bestX = Xj.copy()

    # final exact jamming polish of best
    try:
        Xf, vf = lp_jam(bestX, max_iter=80, init_radius=0.006, slack=2e-3, tlim=T_END-2.0)
        if vf < bestVal and np.isfinite(vf):
            bestVal = float(vf); bestX = Xf.copy()
    except Exception:
        pass
    bestX /= np.maximum(np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)
    # strict feasibility: tiny shrink not needed (unit already); ensure finite
    return bestX
