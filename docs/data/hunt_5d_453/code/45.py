import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _tangent_bases(X):
    n, d = X.shape
    B = np.empty((n, d, 4), dtype=np.float64)
    for i in range(n):
        x = X[i]
        k = int(np.argmin(np.abs(x)))
        T = np.empty((d, 4), dtype=np.float64)
        c = 0
        for j in range(d):
            if j == k:
                continue
            v = -x[j] * x
            v[j] += 1.0
            T[:, c] = v
            c += 1
        Q, _ = np.linalg.qr(T)
        B[i] = Q[:, :4]
    return B

def _max_cos_full(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G)), G

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    EXPL_END = t0 + 122.0
    rng = np.random.default_rng(0)
    n, d = 453, 5
    Nvar = n * 4
    # load incumbent
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            cands.append(np.asarray(a, dtype=np.float64))
        except Exception:
            pass
    bestX = None
    bestV = float("inf")
    for cc in cands:
        try:
            if cc.shape != (453, 5):
                continue
            nn = np.sqrt(np.sum(cc*cc, axis=1, keepdims=True))
            nn[nn == 0] = 1.0
            cc = cc / nn
            if not np.all(np.isfinite(cc)):
                continue
            G = cc @ cc.T
            np.fill_diagonal(G, -2.0)
            vv = float(np.max(G))
            if vv < bestV:
                bestV = vv
                bestX = cc.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        G = bestX @ bestX.T
        np.fill_diagonal(G, -2.0)
        bestV = float(np.max(G))
    iu_r, iu_c = np.triu_indices(n, 1)
    ar4 = np.arange(4)
    c_obj = np.zeros(Nvar + 1, dtype=np.float64)
    c_obj[-1] = 1.0

    def lp_polish(Xstart, cur_best_X, cur_best_V, r_init, max_solves, cutoff):
        X = Xstart.copy()
        bX = cur_best_X.copy()
        bV = float(cur_best_V)
        radius = float(r_init)
        for _ in range(max_solves):
            if time.time() > cutoff or time.time() > T_END - 2.0:
                break
            if radius < 1e-12:
                break
            G = X @ X.T
            c_all = G[iu_r, iu_c]
            m = float(np.max(c_all))
            if m < bV and np.isfinite(m):
                bV = m
                bX = X.copy()
            slack_all = m - c_all
            tight = np.where(slack_all < 2e-5)[0]
            if tight.size == 0:
                tight = np.argsort(slack_all)[:500]
            if tight.size > 2500:
                part = np.argpartition(slack_all[tight], 2500)[:2500]
                tight = tight[part]
            I = iu_r[tight]; J = iu_c[tight]
            p = int(I.size)
            b_ub = slack_all[tight].astype(np.float64)
            B = _tangent_bases(X)
            aI = np.einsum('pdk,pd->pk', B[I], X[J])
            aJ = np.einsum('pdk,pd->pk', B[J], X[I])
            cols2 = np.empty((p, 9), dtype=np.int64)
            cols2[:, 0:4] = I[:, None]*4 + ar4[None, :]
            cols2[:, 4:8] = J[:, None]*4 + ar4[None, :]
            cols2[:, 8] = Nvar
            vals2 = np.empty((p, 9), dtype=np.float64)
            vals2[:, 0:4] = aI
            vals2[:, 4:8] = aJ
            vals2[:, 8] = -1.0
            rows = np.repeat(np.arange(p), 9)
            try:
                A_ub = coo_matrix((vals2.ravel(), (rows, cols2.ravel())), shape=(p, Nvar+1)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            rem = min(cutoff - time.time(), T_END - 2.0 - time.time(), 10.0)
            if rem < 0.5:
                break
            bounds = [(-radius, radius)]*Nvar + [(None, None)]
            try:
                res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs", options={"time_limit": max(0.5, rem)})
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0 or res.x is None:
                radius *= 0.5
                continue
            try:
                U = np.asarray(res.x[:Nvar], dtype=np.float64).reshape(n, 4)
            except Exception:
                radius *= 0.5
                continue
            if not np.all(np.isfinite(U)):
                radius *= 0.5
                continue
            DX = np.einsum('ndk,nk->nd', B, U)
            Xt = X + DX
            nn = np.sqrt(np.sum(Xt*Xt, axis=1, keepdims=True)); nn[nn==0]=1.0
            Xt = Xt / nn
            Gt = Xt @ Xt.T
            np.fill_diagonal(Gt, -2.0)
            mt = float(np.max(Gt))
            if np.isfinite(mt) and mt < m - 1e-13:
                X = Xt
                if mt < bV:
                    bV = mt; bX = Xt.copy()
                radius = min(radius*1.5, 5e-4)
            else:
                radius *= 0.5
        return bX, bV, X

    def null_attempt(Xcur, mcur, deadline, seed_off):
        # build tangent + rigidity for contacts slack<1e-9
        B = _tangent_bases(Xcur)
        G = Xcur @ Xcur.T
        c_all = G[iu_r, iu_c]
        m = float(np.max(c_all))
        slack_all = m - c_all
        order = np.argsort(slack_all)
        # contacts
        cmask = slack_all < 1e-9
        cidx = np.where(cmask)[0]
        if cidx.size > 2200:
            cidx = order[:2200]
        if cidx.size < 600:
            # take tightest 1200 to have a matrix
            cidx = order[:1200]
        p = int(cidx.size)
        if p < 200:
            return None
        I = iu_r[cidx]; J = iu_c[cidx]
        aI = np.einsum('pdk,pd->pk', B[I], Xcur[J])
        aJ = np.einsum('pdk,pd->pk', B[J], Xcur[I])
        # dense R
        try:
            R = np.zeros((p, Nvar), dtype=np.float64)
        except Exception:
            return None
        pr = np.arange(p)
        for k in range(4):
            R[pr, I*4+k] = aI[:, k]
            R[pr, J*4+k] = aJ[:, k]
        # C = R^T R
        try:
            C = R.T @ R
            del R
        except Exception:
            return None
        # symmetrize
        C = 0.5*(C + C.T)
        try:
            w, V = np.linalg.eigh(C)
            del C
        except Exception:
            return None
        K = 40
        if V.shape[1] < K:
            K = V.shape[1]
        Vnull = V[:, :K].copy()
        del V
        # shell pairs 5e-7..1e-4
        smask = (slack_all >= 5e-7) & (slack_all < 1e-4)
        sidx = np.where(smask)[0]
        if sidx.size == 0:
            # take next pairs above contacts
            # order is sorted ascending slack
            # find first index with slack>=5e-7
            sidx = order[p:p+300]
            # filter slack<5e-4
            keep = sidx[slack_all[sidx] < 5e-4]
            if keep.size > 0:
                sidx = keep
            else:
                sidx = order[p:p+200]
        if sidx.size > 80:
            # tightest 80
            part = np.argpartition(slack_all[sidx], 80)[:80]
            sidx = sidx[part]
        ns = int(sidx.size)
        if ns == 0:
            return None
        Is = iu_r[sidx]; Js = iu_c[sidx]
        aIs = np.einsum('pdk,pd->pk', B[Is], Xcur[Js])
        aJs = np.einsum('pdk,pd->pk', B[Js], Xcur[Is])
        # targeted null directions
        r1 = np.random.default_rng(1000+seed_off)
        Ylist = []
        for k in range(K):
            e = np.zeros(K); e[k]=1.0
            Ylist.append(e)
        for s in range(ns):
            ii = int(Is[s]); jj = int(Js[s])
            bI = Vnull[ii*4:ii*4+4, :]
            bJ = Vnull[jj*4:jj*4+4, :]
            g = bI.T @ aIs[s] + bJ.T @ aJs[s]
            nnrm = float(np.linalg.norm(g))
            if nnrm > 1e-14 and np.isfinite(nnrm):
                Ylist.append(g/nnrm)
            if len(Ylist) > 110:
                break
        # random combos
        for _ in range(45):
            v = r1.standard_normal(K)
            nrm = float(np.linalg.norm(v))
            if nrm > 1e-12:
                Ylist.append(v/nrm)
            if len(Ylist) >= 155:
                break
        # precompute D dirs
        Ddirs = []
        for y in Ylist:
            u = Vnull @ y  # Nvar
            Um = u.reshape(n, 4)
            Dd = np.einsum('ndk,nk->nd', B, Um)
            Ddirs.append(Dd)
        # current contact count
        G0 = Xcur @ Xcur.T
        np.fill_diagonal(G0, -2.0)
        m0 = float(np.max(G0))
        cnt0 = int(np.count_nonzero(G0 > m0 - 1e-9)//2)
        s_vals = [0.006, 0.012, 0.025]
        best_m = m0
        best_X = None
        best_cnt = cnt0
        topo_X = None
        topo_cnt = cnt0
        # search
        for Dd in Ddirs:
            if time.time() > deadline:
                break
            for s in s_vals:
                if time.time() > deadline:
                    break
                for sgn in (1.0, -1.0):
                    ss = sgn*s
                    Xt = Xcur + ss*Dd
                    nn = np.sqrt(np.sum(Xt*Xt, axis=1, keepdims=True)); nn[nn==0]=1.0
                    Xt = Xt/nn
                    # quick max
                    Gt = Xt @ Xt.T
                    np.fill_diagonal(Gt, -2.0)
                    mt = float(np.max(Gt))
                    if not np.isfinite(mt):
                        continue
                    if mt < best_m - 1e-12:
                        best_m = mt
                        best_X = Xt.copy()
                        # also update cnt
                        cnt = int(np.count_nonzero(Gt > mt - 1e-9)//2)
                        best_cnt = cnt
                    elif mt <= m0 + 5e-13:
                        cnt = int(np.count_nonzero(Gt > mt - 1e-9)//2)
                        if cnt > topo_cnt:
                            topo_cnt = cnt
                            topo_X = Xt.copy()
        if best_X is not None:
            return (best_X, best_m, True)
        if topo_X is not None and topo_cnt > cnt0:
            # new contact without raising max
            Gt = topo_X @ topo_X.T
            np.fill_diagonal(Gt, -2.0)
            mt = float(np.max(Gt))
            return (topo_X, mt, False)
        return None

    Xcur = bestX.copy()
    mcur = bestV
    # null-space exploration iterations
    for it in range(6):
        if time.time() > EXPL_END or time.time() > T_END - 35.0:
            break
        deadline = min(EXPL_END, T_END - 35.0, time.time()+22.0)
        try:
            res = null_attempt(Xcur, mcur, deadline, seed_off=it)
        except Exception:
            res = None
        if res is None:
            break
        Xn, mn, lowered = res
        # accept: update cur, run short LP polish
        Xcur = Xn
        if mn < bestV and np.isfinite(mn):
            bestV = mn
            bestX = Xn.copy()
        # short polish to settle (2-3 solves)
        try:
            bX2, bV2, Xc2 = lp_polish(Xcur, bestX, bestV, 8e-5, 3, min(deadline+8.0, T_END-30.0))
            bestX, bestV = bX2, bV2
            Xcur = Xc2.copy()
            # update mcur
            Gc = Xcur @ Xcur.T
            np.fill_diagonal(Gc, -2.0)
            mcur = float(np.max(Gc))
            if not lowered:
                # topology changed but not lowered; continue exploring
                pass
            else:
                pass
        except Exception:
            pass
        if time.time() > EXPL_END:
            break

    # final polish with remaining time (reserve 1s for return)
    try:
        if time.time() < T_END - 2.0:
            # polish incumbent starting from best
            bXf, bVf, _ = lp_polish(bestX, bestX, bestV, 1e-4, 12, T_END-1.0)
            bestX, bestV = bXf, bVf
    except Exception:
        pass
    # normalize and return
    try:
        nn = np.sqrt(np.sum(bestX*bestX, axis=1, keepdims=True))
        nn[nn==0]=1.0
        bestX = bestX/nn
        bestX[~np.isfinite(bestX)] = 0.0
        # fix any zero rows
        for i in range(n):
            if float(np.linalg.norm(bestX[i])) < 1e-12:
                v = rng.standard_normal(d); v/=np.linalg.norm(v)
                bestX[i]=v
        nn = np.sqrt(np.sum(bestX*bestX, axis=1, keepdims=True)); nn[nn==0]=1.0
        bestX = bestX/nn
    except Exception:
        pass
    return np.asarray(bestX, dtype=np.float64)
