import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

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

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    LP_CUTOFF = t0 + 150.0
    rng = np.random.default_rng(0)
    n, d = 453, 5
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
            vv = _max_cos(cc)
            if vv < bestV:
                bestV = vv
                bestX = cc.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)

    iu_r, iu_c = np.triu_indices(n, 1)
    Nvar = n * 4
    c_obj = np.zeros(Nvar + 1, dtype=np.float64)
    c_obj[-1] = 1.0
    ar4 = np.arange(4)

    def lp_burst(X, bx, bv, r_init, max_solves):
        X = X.copy()
        bx = bx.copy()
        bv = float(bv)
        radius = float(r_init)
        for _ in range(max_solves):
            if time.time() > LP_CUTOFF or time.time() > T_END - 8.0:
                break
            if radius < 1e-12:
                break
            G = X @ X.T
            c_all = G[iu_r, iu_c]
            m = float(np.max(c_all))
            if m < bv and np.isfinite(m):
                bv = m
                bx = X.copy()
            slack = m - c_all
            tight = np.where(slack < 2e-5)[0]
            if tight.size == 0:
                tight = np.argsort(slack)[:500]
            if tight.size > 2200:
                part = np.argpartition(slack[tight], 2200)[:2200]
                tight = tight[part]
            I = iu_r[tight]
            J = iu_c[tight]
            p = int(I.size)
            b_ub = slack[tight].astype(np.float64)
            B = _tangent_bases(X)
            aI = np.einsum('pdk,pd->pk', B[I], X[J])
            aJ = np.einsum('pdk,pd->pk', B[J], X[I])
            cols2 = np.empty((p, 9), dtype=np.int64)
            cols2[:, 0:4] = I[:, None] * 4 + ar4[None, :]
            cols2[:, 4:8] = J[:, None] * 4 + ar4[None, :]
            cols2[:, 8] = Nvar
            vals2 = np.empty((p, 9), dtype=np.float64)
            vals2[:, 0:4] = aI
            vals2[:, 4:8] = aJ
            vals2[:, 8] = -1.0
            rows = np.repeat(np.arange(p), 9)
            try:
                A_ub = coo_matrix((vals2.ravel(), (rows, cols2.ravel())), shape=(p, Nvar + 1)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            rem = min(LP_CUTOFF - time.time(), T_END - 8.0 - time.time(), 5.0)
            if rem < 0.4:
                break
            bounds = [(-radius, radius)] * Nvar + [(None, None)]
            try:
                res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                              method="highs", options={"time_limit": max(0.4, rem)})
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
            nn = np.sqrt(np.sum(Xt * Xt, axis=1, keepdims=True))
            nn[nn == 0] = 1.0
            Xt = Xt / nn
            if not np.all(np.isfinite(Xt)):
                radius *= 0.5
                continue
            mt = _max_cos(Xt)
            if mt < m - 1e-13 and np.isfinite(mt):
                X = Xt
                if mt < bv:
                    bv = mt
                    bx = Xt.copy()
                radius = min(radius * 1.5, 4e-4)
            else:
                radius *= 0.5
        return bx, bv, X

    # initial polish
    try:
        bestX, bestV, Xcur = lp_burst(bestX, bestX, bestV, 1e-4, 5)
    except Exception:
        Xcur = bestX.copy()

    # iterated surgery + repolish cycles
    NS = 8000
    Spool = rng.standard_normal((NS, d))
    Spool /= np.linalg.norm(Spool, axis=1, keepdims=True) + 1e-30

    for cycle in range(3):
        if time.time() > LP_CUTOFF - 3.0 or time.time() > T_END - 12.0:
            break
        try:
            G = bestX @ bestX.T
            cb = G[iu_r, iu_c]
            mb = float(np.max(cb))
            cmask = (mb - cb) < 1e-9
            deg = np.zeros(n, dtype=int)
            np.add.at(deg, iu_r[cmask], 1)
            np.add.at(deg, iu_c[cmask], 1)
            Gt = G.copy()
            np.fill_diagonal(Gt, -2.0)
            rmax = np.max(Gt, axis=1)
            rmin = np.min(Gt, axis=1)
        except Exception:
            break
        victims = []
        ratt = np.where(deg == 0)[0]
        if ratt.size > 0:
            o = ratt[np.argsort(-rmax[ratt])]
            for v in o[:6]:
                victims.append(int(v))
        # low-degree floppy points
        low = np.where((deg >= 1) & (deg <= 2))[0]
        if low.size > 0:
            o = low[np.argsort(-rmax[low])]
            for v in o[:3]:
                if int(v) not in victims:
                    victims.append(int(v))
        # hubs to relieve over-constraint
        hubs = np.argsort(-deg)[:3]
        for v in hubs:
            if deg[int(v)] >= 12 and int(v) not in victims:
                victims.append(int(v))
            if len(victims) >= 9:
                break
        # deep well-separated points (candidates to donate)
        if len(victims) < 8:
            deep = np.argsort(rmax)[:4]
            for v in deep:
                if int(v) not in victims:
                    victims.append(int(v))
                if len(victims) >= 8:
                    break
        victims = victims[:8]
        if len(victims) == 0:
            break
        Xnew = bestX.copy()
        # relocate each victim into current largest hole sequentially
        for vv in victims:
            if time.time() > LP_CUTOFF - 2.0 or time.time() > T_END - 10.0:
                break
            mask = np.ones(n, dtype=bool)
            mask[vv] = False
            Xe = Xnew[mask]
            D = Spool @ Xe.T
            md = np.max(D, axis=1)
            bi = int(np.argmin(md))
            h = Spool[bi].copy()
            for _ in range(40):
                dots = Xe @ h
                ci = int(np.argmax(dots))
                xc = Xe[ci]
                c_ = float(dots[ci])
                t = xc - c_ * h
                tn = float(np.linalg.norm(t))
                if tn < 1e-12:
                    break
                t /= tn
                h = h - 0.008 * t
                nh = float(np.linalg.norm(h))
                if nh == 0 or not np.isfinite(nh):
                    break
                h /= nh
            Xnew[vv] = h
        vs = _max_cos(Xnew)
        if not np.isfinite(vs) or vs > bestV + 0.004:
            # too destructive, jitter instead with smaller move
            Xnew = bestX.copy()
            for vv in victims[:4]:
                rnd = rng.standard_normal(d)
                rnd -= float(rnd @ Xnew[vv]) * Xnew[vv]
                rn = float(np.linalg.norm(rnd))
                if rn < 1e-12:
                    continue
                rnd /= rn
                h = Xnew[vv] + 0.02 * rnd
                h /= np.linalg.norm(h)
                Xnew[vv] = h
        try:
            bestX, bestV, Xcur = lp_burst(Xnew, bestX, bestV, 8e-5, 5)
        except Exception:
            continue

    # final short greedy worst-pair shave on best
    try:
        Xg = bestX.copy()
        cur = bestV
        t_end_g = min(T_END - 0.7, time.time() + 5.0)
        eps = 2e-4
        it = 0
        while time.time() < t_end_g and it < 400:
            it += 1
            G = Xg @ Xg.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            flat = int(np.argmax(G))
            i = flat // n
            j = flat % n
            xi = Xg[i].copy(); xj = Xg[j].copy()
            cij = float(xi @ xj)
            cij = max(-1.0, min(1.0, cij))
            ti = xj - cij * xi
            ni = float(np.linalg.norm(ti))
            if ni < 1e-12:
                eps *= 0.9
                continue
            ti /= ni
            tj = xi - cij * xj
            nj = float(np.linalg.norm(tj))
            if nj < 1e-12:
                eps *= 0.9
                continue
            tj /= nj
            improved = False
            for scale in (eps, eps * 0.5):
                vi = xi - scale * ti
                vi /= np.linalg.norm(vi)
                vj = xj - scale * tj
                vj /= np.linalg.norm(vj)
                oi = Xg[i].copy(); oj = Xg[j].copy()
                Xg[i] = vi; Xg[j] = vj
                G2 = Xg @ Xg.T
                np.fill_diagonal(G2, -2.0)
                m2 = float(np.max(G2))
                if m2 < m - 1e-12:
                    if m2 < bestV:
                        bestV = m2
                        bestX = Xg.copy()
                    eps = min(eps * 1.05, 5e-4)
                    improved = True
                    break
                else:
                    Xg[i] = oi; Xg[j] = oj
            if not improved:
                eps *= 0.94
                if eps < 1e-7:
                    eps = 1e-7
        # sync
        pass
    except Exception:
        pass

    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
