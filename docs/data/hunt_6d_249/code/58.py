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
    n, d = 249, 6
    Nv = n * 5
    I0, J0 = np.triu_indices(n, 1)
    P = len(I0)
    RETAIN_THRESH = 0.7034808169015

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
        rng0 = np.random.default_rng(0)
        a = rng0.standard_normal(size=(n, d))
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

    def lp_jam_short(Xinit, deadline, radius_init=2e-3, slack=1.5e-3, max_it=8):
        X = Xinit.copy()
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        mcur = _max_cos(X)
        radius = float(radius_init)
        cvec = np.zeros(Nv + 1)
        cvec[Nv] = 1.0
        it = 0
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
            elif len(sel) > 4000:
                sel = sel[np.argpartition(gfull[sel], -4000)[-4000:]]
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
                radius = min(radius * 1.35, 8e-3)
            else:
                radius *= 0.5
        return X, mcur

    def edge_set_from(X, mval, slack=1e-4):
        G = X @ X.T
        # mval assumed max off-diag
        g = G[I0, J0]
        idx = np.where(g > mval - slack)[0]
        codes = I0[idx].astype(np.int64) * n + J0[idx].astype(np.int64)
        return set(codes.tolist())

    def null_dirs(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        gfull = G[I0, J0]
        act = np.where(gfull > m - 1e-7)[0]
        if len(act) < 200:
            K0 = min(P, 800)
            act = np.argpartition(gfull, -K0)[-K0:]
        elif len(act) > 5000:
            act = act[np.argpartition(gfull[act], -5000)[-5000:]]
        Ic = I0[act]; Jc = J0[act]
        K = len(act)
        T = tangent_bases(X)
        TI = T[Ic]
        TJ = T[Jc]
        XJ = X[Jc]
        XI = X[Ic]
        Ai = np.sum(TI * XJ[:, :, None], axis=1)  # Kx5
        Aj = np.sum(TJ * XI[:, :, None], axis=1)
        A = np.zeros((K, Nv), dtype=float)
        for k in range(K):
            A[k, int(Ic[k]) * 5:int(Ic[k]) * 5 + 5] = Ai[k]
            A[k, int(Jc[k]) * 5:int(Jc[k]) * 5 + 5] = Aj[k]
        # rotation modes 15
        Rmat = np.zeros((Nv, 15), dtype=float)
        col = 0
        for a in range(6):
            for b in range(a + 1, 6):
                for i in range(n):
                    Ti = T[i]  # 6x5
                    xi = X[i]
                    vec = Ti[a, :] * xi[b] - Ti[b, :] * xi[a]
                    Rmat[i * 5:(i + 1) * 5, col] = vec
                col += 1
        Q, _ = np.linalg.qr(Rmat)
        ATA = A.T @ A
        # symmetrize
        ATA = 0.5 * (ATA + ATA.T)
        QTA = Q.T @ ATA  # 15xNv
        QTATAQ = QTA @ Q  # 15x15
        M = ATA - Q @ QTA - QTA.T @ Q.T + Q @ QTATAQ @ Q.T
        M = 0.5 * (M + M.T)
        w, V = np.linalg.eigh(M)
        # w ascending, first 15 ~0 (rotations). take next 20
        dirs = []
        for k in range(15, min(35, Nv)):
            dirs.append(V[:, k].copy())
        return dirs, T, m

    # incumbent contact graph at slack 1e-4
    try:
        G0 = bestX @ bestX.T
        np.fill_diagonal(G0, -2.0)
        m0tmp = float(np.max(G0))
        set0 = edge_set_from(bestX, m0tmp, slack=1e-4)
    except Exception:
        set0 = set()

    outer = 0
    while time.time() < T_END - 15:
        outer += 1
        if outer > 10:
            break
        try:
            dirs, Tcur, mcur0 = null_dirs(bestX)
        except Exception:
            break
        # refresh incumbent set (best may have changed)
        try:
            G0 = bestX @ bestX.T
            np.fill_diagonal(G0, -2.0)
            m0tmp = float(np.max(G0))
            # keep bestVal consistent (exact)
            if m0tmp < bestVal:
                bestVal = m0tmp
            set0 = edge_set_from(bestX, m0tmp, slack=1e-4)
        except Exception:
            pass
        if len(dirs) == 0:
            break
        improved = False
        for dvec in dirs:
            if time.time() > T_END - 12:
                break
            Y0 = dvec.reshape(n, 5)
            nrm = np.linalg.norm(Y0, axis=1)
            nmax = float(np.max(nrm))
            if not np.isfinite(nmax) or nmax < 1e-12:
                continue
            for sign in (1.0, -1.0):
                if improved or time.time() > T_END - 12:
                    break
                for alpha in (0.02, 0.05, 0.10, 0.20):
                    if time.time() > T_END - 12:
                        break
                    scale = sign * float(alpha) / nmax
                    Y = Y0 * scale
                    V = np.einsum('ijk,ik->ij', Tcur, Y)
                    theta = np.linalg.norm(V, axis=1)
                    c = np.cos(theta)
                    s = np.sin(theta)
                    Xc = bestX.copy()
                    mask = theta > 1e-14
                    # vectorized geodesic
                    Xc[mask] = c[mask, None] * bestX[mask] + (s[mask] / theta[mask])[:, None] * V[mask]
                    Xc /= np.maximum(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-12)
                    mc = _max_cos(Xc)
                    if mc > bestVal + 2e-3 + 1e-12:
                        continue
                    if mc >= bestVal - 1e-12:
                        # need Jaccard < 0.9
                        try:
                            setc = edge_set_from(Xc, mc, slack=1e-4)
                        except Exception:
                            continue
                        inter = len(set0.intersection(setc))
                        union = len(set0) + len(setc) - inter
                        jacc = (inter / union) if union > 0 else 1.0
                        if jacc >= 0.9:
                            continue
                    # surviving: re-jam at most 8
                    deadline = min(T_END - 10, time.time() + 8)
                    try:
                        Xj, vj = lp_jam_short(Xc, deadline, radius_init=2e-3, slack=1.5e-3, max_it=8)
                    except Exception:
                        continue
                    if vj < bestVal - 1e-12 and vj < RETAIN_THRESH + 1e-12:
                        bestX = Xj.copy()
                        bestVal = float(vj)
                        try:
                            set0 = edge_set_from(bestX, bestVal, slack=1e-4)
                        except Exception:
                            pass
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break

    bestX /= np.maximum(np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)
    return bestX
