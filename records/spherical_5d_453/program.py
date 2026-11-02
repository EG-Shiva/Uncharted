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
        # 4 vectors e_j - x_j * x, orthogonal to x
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
        B[i] = Q[:, :4] if Q.shape[1] >= 4 else Q
    return B

def _get_duals(res):
    try:
        o = res.ineqlin
        try:
            m = o.marginals
            return np.asarray(m, dtype=np.float64)
        except Exception:
            pass
        try:
            m = o['marginals']
            return np.asarray(m, dtype=np.float64)
        except Exception:
            pass
        try:
            m = res['ineqlin']['marginals']
            return np.asarray(m, dtype=np.float64)
        except Exception:
            pass
    except Exception:
        pass
    return None

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    LP_CUTOFF = t0 + 150.0
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 453, 5

    # load starts
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

    def lp_round(X, cur_best_X, cur_best_V, r_init, max_solves):
        X = X.copy()
        bestXl = cur_best_X.copy()
        bestVl = cur_best_V
        radius = float(r_init)
        last_duals = None
        last_I = None
        last_J = None
        last_slack = None
        Nvar = n * 4
        c_obj = np.zeros(Nvar + 1, dtype=np.float64)
        c_obj[-1] = 1.0
        ar4 = np.arange(4)
        for _ in range(max_solves):
            if time.time() > LP_CUTOFF or time.time() > T_END - 6.0:
                break
            if radius < 1e-12:
                break
            # current max over upper triangle
            G = X @ X.T
            c_all = G[iu_r, iu_c]
            m = float(np.max(c_all))
            if m < bestVl:
                bestVl = m
                bestXl = X.copy()
            slack_all = m - c_all
            # tight set slack < 2e-5, cap 2500 tightest
            tight = np.where(slack_all < 2e-5)[0]
            if tight.size == 0:
                # nothing tight? expand slightly
                tight = np.argsort(slack_all)[:500]
            if tight.size > 2500:
                part = np.argpartition(slack_all[tight], 2500)[:2500]
                tight = tight[part]
            I = iu_r[tight]
            J = iu_c[tight]
            p = int(I.size)
            b_ub = slack_all[tight]
            # tangent bases once per solve
            B = _tangent_bases(X)
            # coefficients a_ij = B_i^T x_j
            # B[I]: p x5x4, X[J]: p x5
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
            cols_f = cols2.ravel()
            vals_f = vals2.ravel()
            try:
                A_ub = coo_matrix((vals_f, (rows, cols_f)), shape=(p, Nvar + 1)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            rem = min(LP_CUTOFF - time.time(), T_END - 6.0 - time.time(), 12.0)
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
            # retract
            DX = np.einsum('ndk,nk->nd', B, U)
            Xt = X + DX
            nn = np.sqrt(np.sum(Xt * Xt, axis=1, keepdims=True))
            nn[nn == 0] = 1.0
            Xt = Xt / nn
            mt = _max_cos(Xt)
            # store duals for diagnostics (last successful)
            du = _get_duals(res)
            last_duals = du
            last_I = I.copy()
            last_J = J.copy()
            last_slack = b_ub.copy()
            if mt < m - 1e-13 and np.isfinite(mt):
                X = Xt
                if mt < bestVl:
                    bestVl = mt
                    bestXl = Xt.copy()
                radius = min(radius * 1.5, 5e-4)
            else:
                radius *= 0.5
        # final sync best
        Gf = X @ X.T
        cf = Gf[iu_r, iu_c]
        mf = float(np.max(cf))
        if mf < bestVl:
            bestVl = mf
            bestXl = X.copy()
        return bestXl, bestVl, X, mf, (last_duals, last_I, last_J, last_slack)

    # Round 1: polish record basin, get stress diagnostic
    try:
        bestX, bestV, Xcur, mcur, diag = lp_round(bestX, bestX, bestV, 1e-4, 6)
    except Exception:
        Xcur = bestX.copy()
        mcur = bestV
        diag = (None, None, None, None)

    if time.time() > LP_CUTOFF - 2.0 or time.time() > T_END - 10.0:
        bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
        return bestX

    # --- stress / rattler analysis on bestX ---
    try:
        Gb = bestX @ bestX.T
        cb = Gb[iu_r, iu_c]
        mb = float(np.max(cb))
        contact_mask = (mb - cb) < 1e-9
        deg = np.zeros(n, dtype=int)
        np.add.at(deg, iu_r[contact_mask], 1)
        np.add.at(deg, iu_c[contact_mask], 1)
        row_max = np.empty(n, dtype=np.float64)
        # row max excluding diag
        Gtmp = Gb.copy()
        np.fill_diagonal(Gtmp, -2.0)
        row_max = np.max(Gtmp, axis=1)
    except Exception:
        deg = np.zeros(n, dtype=int)
        row_max = np.zeros(n)
        mb = bestV

    # dual-based stressed contacts
    stressed = None
    try:
        du, dI, dJ, dsl = diag
        if du is not None and dI is not None and len(du) == len(dI):
            s = -np.asarray(du, dtype=np.float64)
            # positive stress = s > tol
            tol = 1e-9
            is_stressed_pair = s > tol
            stressed_pairs = set(zip([int(a) for a in dI[is_stressed_pair]],
                                     [int(a) for a in dJ[is_stressed_pair]]))
            has_stress = np.zeros(n, dtype=bool)
            for (a, b) in stressed_pairs:
                has_stress[a] = True
                has_stress[b] = True
            stressed = has_stress
    except Exception:
        stressed = None

    # victims: rattlers + zero-stress-only points
    victims = []
    rattlers = np.where(deg == 0)[0]
    if rattlers.size > 0:
        # pick rattlers with largest row_max (closest to bottleneck) first
        order = rattlers[np.argsort(-row_max[rattlers])]
        for v in order[:8]:
            victims.append(int(v))
    if stressed is not None:
        # points with contacts but no stressed contact
        has_contact = deg > 0
        zero_stress_pts = np.where(has_contact & (~stressed))[0]
        if zero_stress_pts.size > 0:
            order2 = zero_stress_pts[np.argsort(-row_max[zero_stress_pts])]
            for v in order2[:4]:
                if int(v) not in victims:
                    victims.append(int(v))
                if len(victims) >= 12:
                    break
    # cap
    victims = victims[:10]
    # if no victims (rigid), still nudge a few highest-degree? fallback
    if len(victims) == 0:
        # pick 4 points with smallest row_max (deepest, likely rattlers at looser tol)
        order = np.argsort(row_max)[:4]
        victims = [int(v) for v in order]

    # --- hole relocation surgery ---
    try:
        Xnew = bestX.copy()
        # shared sample pool for hole search (deterministic)
        NS = 20000
        S = rng.standard_normal((NS, d))
        S /= np.linalg.norm(S, axis=1, keepdims=True) + 1e-30
        for vv in victims:
            if time.time() > LP_CUTOFF - 2.0 or time.time() > T_END - 10.0:
                break
            mask = np.ones(n, dtype=bool)
            mask[vv] = False
            Xe = Xnew[mask]
            D = S @ Xe.T
            md = np.max(D, axis=1)
            bi = int(np.argmin(md))
            h = S[bi].copy()
            # refine: push away from closest point
            for _ in range(60):
                dots = Xe @ h
                ci = int(np.argmax(dots))
                xc = Xe[ci]
                c_ = float(dots[ci])
                t = xc - c_ * h
                tn = float(np.linalg.norm(t))
                if tn < 1e-12:
                    break
                t /= tn
                h = h - 0.01 * t
                nh = float(np.linalg.norm(h))
                if nh == 0 or not np.isfinite(nh):
                    break
                h /= nh
            Xnew[vv] = h
        # keep surgery only if not catastrophically worse (allow small increase)
        vs = _max_cos(Xnew)
        # accept surgery regardless if within 0.005 (topology change), else revert worst?
        # we keep Xnew as restart for round 2 even if slightly worse
        if not np.isfinite(vs) or vs > bestV + 0.005:
            Xnew = bestX.copy()
        Xcur2 = Xnew
    except Exception:
        Xcur2 = bestX.copy()

    # Round 2: re-polish new topology
    try:
        remaining = int(12 - 6)
        if remaining < 4:
            remaining = 4
        bestX, bestV, Xcur, mcur, diag2 = lp_round(Xcur2, bestX, bestV, 1e-4, remaining)
    except Exception:
        pass

    # short greedy micro-shave if time remains (cheap, few iters)
    try:
        t_gr_end = min(T_END - 0.5, time.time() + 6.0)
        Xg = bestX.copy()
        cur = bestV
        eps = 5e-8
        it = 0
        while time.time() < t_gr_end and it < 1500:
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
            # mask rest max
            G[i, :] = -2.0; G[:, i] = -2.0
            G[j, :] = -2.0; G[:, j] = -2.0
            m_rest = float(np.max(G))
            improved = False
            for sc in (eps, eps * 0.5):
                vi = xi - sc * ti; vi /= np.linalg.norm(vi)
                vj = xj - sc * tj; vj /= np.linalg.norm(vj)
                di = Xg @ vi; di[i] = -2.0; di[j] = -2.0
                dj = Xg @ vj; dj[i] = -2.0; dj[j] = -2.0
                mc = m_rest
                a = float(np.max(di))
                if a > mc: mc = a
                b = float(np.max(dj))
                if b > mc: mc = b
                c2 = float(vi @ vj)
                if c2 > mc: mc = c2
                if mc < m - 1e-15:
                    Xg[i] = vi; Xg[j] = vj
                    improved = True
                    if mc < cur:
                        cur = mc
                        bestX = Xg.copy()
                        bestV = cur
                    break
            if improved:
                eps = min(eps * 1.05, 2e-7)
            else:
                eps *= 0.6
                if eps < 1e-13:
                    break
    except Exception:
        pass

    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
