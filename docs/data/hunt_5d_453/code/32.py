import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X, iu_r=None, iu_c=None):
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

def _get_duals(res):
    try:
        o = res.ineqlin
        try:
            return np.asarray(o.marginals, dtype=np.float64)
        except Exception:
            pass
        try:
            return np.asarray(o['marginals'], dtype=np.float64)
        except Exception:
            pass
        try:
            return np.asarray(res['ineqlin']['marginals'], dtype=np.float64)
        except Exception:
            pass
    except Exception:
        pass
    return None

def solve():
    t0 = time.time()
    T_END = t0 + 170.0
    LP_CUTOFF = t0 + 138.0
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
            nn = np.sqrt(np.sum(cc * cc, axis=1, keepdims=True))
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
    Nvar = n * 4
    c_obj = np.zeros(Nvar + 1, dtype=np.float64)
    c_obj[-1] = 1.0
    ar4 = np.arange(4)

    def lp_round(X, cur_best_X, cur_best_V, r_init, max_solves):
        X = X.copy()
        bestXl = cur_best_X.copy()
        bestVl = float(cur_best_V)
        radius = float(r_init)
        last_duals = None
        last_I = None
        last_J = None
        last_slack = None
        for _ in range(max_solves):
            if time.time() > LP_CUTOFF or time.time() > T_END - 8.0:
                break
            if radius < 1e-12:
                break
            G = X @ X.T
            c_all = G[iu_r, iu_c]
            m = float(np.max(c_all))
            if m < bestVl and np.isfinite(m):
                bestVl = m
                bestXl = X.copy()
            slack_all = m - c_all
            tight = np.where(slack_all < 2e-5)[0]
            if tight.size == 0:
                tight = np.argsort(slack_all)[:500]
            if tight.size > 2500:
                part = np.argpartition(slack_all[tight], 2500)[:2500]
                tight = tight[part]
            I = iu_r[tight]
            J = iu_c[tight]
            p = int(I.size)
            b_ub = slack_all[tight].astype(np.float64)
            B = _tangent_bases(X)
            aI = np.einsum('pdk,pd->pk', B[I], X[J])
            aJ = np.einsum('pdk,pd->pk', B[J], X[I])
            # build sparse A_ub with 8 + 1 nonzeros per row
            rows = np.repeat(np.arange(p, dtype=np.int64), 9)
            cols = np.empty(p * 9, dtype=np.int64)
            vals = np.empty(p * 9, dtype=np.float64)
            # fill in blocks
            # for speed use vectorized
            ci = (I[:, None] * 4 + ar4[None, :]).reshape(-1)
            cj = (J[:, None] * 4 + ar4[None, :]).reshape(-1)
            # interleave per row: need rows order [aI0..3, aJ0..3, t]
            # build via loop-free reshape
            cols2 = np.empty((p, 9), dtype=np.int64)
            cols2[:, 0:4] = I[:, None] * 4 + ar4[None, :]
            cols2[:, 4:8] = J[:, None] * 4 + ar4[None, :]
            cols2[:, 8] = Nvar
            vals2 = np.empty((p, 9), dtype=np.float64)
            vals2[:, 0:4] = aI
            vals2[:, 4:8] = aJ
            vals2[:, 8] = -1.0
            cols = cols2.ravel()
            vals = vals2.ravel()
            try:
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, Nvar + 1)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            rem = min(LP_CUTOFF - time.time(), T_END - 8.0 - time.time(), 9.0)
            if rem < 0.5:
                break
            bounds = [(-radius, radius)] * Nvar + [(None, None)]
            try:
                res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                              method="highs", options={"time_limit": max(0.5, rem)})
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
            Gt = Xt @ Xt.T
            np.fill_diagonal(Gt, -2.0)
            mt = float(np.max(Gt))
            du = _get_duals(res)
            last_duals = du
            last_I = I.copy()
            last_J = J.copy()
            last_slack = b_ub.copy()
            if np.isfinite(mt) and mt < m - 1e-13:
                X = Xt
                if mt < bestVl:
                    bestVl = mt
                    bestXl = Xt.copy()
                radius = min(radius * 1.5, 5e-4)
            else:
                radius *= 0.5
        try:
            Gf = X @ X.T
            np.fill_diagonal(Gf, -2.0)
            mf = float(np.max(Gf))
        except Exception:
            mf = float("inf")
        if np.isfinite(mf) and mf < bestVl:
            bestVl = mf
            bestXl = X.copy()
        return bestXl, bestVl, X, mf, (last_duals, last_I, last_J, last_slack)

    # Round 1
    try:
        bestX, bestV, Xcur, mcur, diag = lp_round(bestX, bestX, bestV, 1e-4, 7)
    except Exception:
        Xcur = bestX.copy()
        mcur = bestV
        diag = (None, None, None, None)

    if time.time() < LP_CUTOFF - 1.0 and time.time() < T_END - 10.0:
        # stress / rattler diagnostics on bestX
        try:
            Gb = bestX @ bestX.T
            cb = Gb[iu_r, iu_c]
            mb = float(np.max(cb))
            contact_mask = (mb - cb) < 1e-9
            deg = np.zeros(n, dtype=int)
            np.add.at(deg, iu_r[contact_mask], 1)
            np.add.at(deg, iu_c[contact_mask], 1)
            Gtmp = Gb.copy()
            np.fill_diagonal(Gtmp, -2.0)
            row_max = np.max(Gtmp, axis=1)
        except Exception:
            deg = np.zeros(n, dtype=int)
            row_max = np.zeros(n)
        stressed = None
        try:
            du, dI, dJ, dsl = diag
            if du is not None and dI is not None and len(du) == len(dI):
                s = -np.asarray(du, dtype=np.float64)
                pos = s > 1e-9
                has_stress = np.zeros(n, dtype=bool)
                for a, b in zip(dI[pos], dJ[pos]):
                    has_stress[int(a)] = True
                    has_stress[int(b)] = True
                stressed = has_stress
        except Exception:
            stressed = None
        victims = []
        try:
            ratt = np.where(deg == 0)[0]
            if ratt.size > 0:
                order = ratt[np.argsort(-row_max[ratt])]
                for v in order[:8]:
                    victims.append(int(v))
            if stressed is not None:
                has_c = deg > 0
                zpts = np.where(has_c & (~stressed))[0]
                if zpts.size > 0:
                    o2 = zpts[np.argsort(-row_max[zpts])]
                    for v in o2[:4]:
                        if int(v) not in victims:
                            victims.append(int(v))
                        if len(victims) >= 10:
                            break
            victims = victims[:10]
            if len(victims) == 0:
                victims = [int(v) for v in np.argsort(row_max)[:4]]
        except Exception:
            victims = [int(v) for v in np.argsort(row_max)[:4]] if 'row_max' in locals() else [0, 1, 2, 3]

        # hole relocation surgery for victims
        try:
            Xnew = bestX.copy()
            NS = 15000
            S = rng.standard_normal((NS, d))
            S /= np.linalg.norm(S, axis=1, keepdims=True) + 1e-30
            for vv in victims:
                if time.time() > LP_CUTOFF - 1.0 or time.time() > T_END - 10.0:
                    break
                mask = np.ones(n, dtype=bool)
                mask[vv] = False
                Xe = Xnew[mask]
                D = S @ Xe.T
                md = np.max(D, axis=1)
                bi = int(np.argmin(md))
                h = S[bi].copy()
                for _ in range(40):
                    dots = Xe @ h
                    ci = int(np.argmax(dots))
                    cc_ = float(dots[ci])
                    xc = Xe[ci]
                    tt = xc - cc_ * h
                    tn = float(np.linalg.norm(tt))
                    if tn < 1e-12:
                        break
                    tt /= tn
                    h = h - 0.01 * tt
                    nh = float(np.linalg.norm(h))
                    if nh == 0 or not np.isfinite(nh):
                        break
                    h /= nh
                Xnew[vv] = h
            Gn = Xnew @ Xnew.T
            np.fill_diagonal(Gn, -2.0)
            vs = float(np.max(Gn))
            if not np.isfinite(vs) or vs > bestV + 0.005:
                Xnew = bestX.copy()
            Xcur2 = Xnew
        except Exception:
            Xcur2 = bestX.copy()

        # Round 2
        try:
            bestX, bestV, Xcur, mcur, diag2 = lp_round(Xcur2, bestX, bestV, 1e-4, 5)
        except Exception:
            pass

    # cheap greedy micro-shave if time remains
    try:
        t_gr_end = min(T_END - 0.5, time.time() + 4.0)
        Xg = bestX.copy()
        cur = bestV
        eps = 6e-7
        it = 0
        while time.time() < t_gr_end and it < 400:
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
                break
            ti /= ni
            tj = xi - cij * xj
            nj = float(np.linalg.norm(tj))
            if nj < 1e-12:
                break
            tj /= nj
            improved = False
            for sc in (eps, eps * 0.5):
                vi = xi - sc * ti
                vi /= np.linalg.norm(vi)
                vj = xj - sc * tj
                vj /= np.linalg.norm(vj)
                oi = Xg[i].copy(); oj = Xg[j].copy()
                Xg[i] = vi; Xg[j] = vj
                # fast max check only rows i,j then full if promising
                d1 = float(np.max(Xg[i] @ Xg.T[np.arange(n) != i].T))
                d2 = float(np.max(Xg[j] @ Xg.T[np.arange(n) != j].T))
                nm = max(d1, d2, m - 1e-12)
                # full eval only if rows ok
                if max(d1, d2) < m:
                    G2 = Xg @ Xg.T
                    np.fill_diagonal(G2, -2.0)
                    m2 = float(np.max(G2))
                else:
                    m2 = m
                    Xg[i] = oi; Xg[j] = oj
                    continue
                if m2 < m - 1e-13:
                    m = m2
                    if m2 < cur:
                        cur = m2
                        bestX = Xg.copy()
                        bestV = cur
                    eps = min(eps * 1.05, 3e-6)
                    improved = True
                    break
                else:
                    Xg[i] = oi; Xg[j] = oj
            if not improved:
                eps *= 0.97
                if eps < 1e-9:
                    eps = 1e-9
    except Exception:
        pass

    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    # strict finite guard
    bestX[~np.isfinite(bestX)] = 0.0
    nn = np.sqrt(np.sum(bestX * bestX, axis=1, keepdims=True))
    nn[nn == 0] = 1.0
    bestX = bestX / nn
    return bestX
