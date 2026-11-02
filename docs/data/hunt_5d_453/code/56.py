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
    T_END = t0 + 172.0
    LP_CUTOFF = t0 + 150.0
    rng = np.random.default_rng(0)
    n, d = 453, 5
    Nvar = n * 4
    ar4 = np.arange(4)
    iu_r, iu_c = np.triu_indices(n, 1)

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

    c_obj = np.zeros(Nvar + 1, dtype=np.float64)
    c_obj[-1] = 1.0

    def lp_round(X, cur_best_X, cur_best_V, r_init, max_solves, t_cap=12.0):
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
            rem = min(LP_CUTOFF - time.time(), T_END - 8.0 - time.time(), t_cap)
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
            mt = _max_cos(Xt)
            du = _get_duals(res)
            last_duals = du; last_I = I.copy(); last_J = J.copy(); last_slack = b_ub.copy()
            if mt < m - 1e-13 and np.isfinite(mt):
                X = Xt
                if mt < bestVl:
                    bestVl = mt; bestXl = Xt.copy()
                radius = min(radius*1.5, 5e-4)
            else:
                radius *= 0.5
        Gf = X @ X.T
        cf = Gf[iu_r, iu_c]
        mf = float(np.max(cf))
        if mf < bestVl:
            bestVl = mf; bestXl = X.copy()
        return bestXl, bestVl, X, mf, (last_duals, last_I, last_J, last_slack)

    # Phase 1: short polish to reach basin bottom + duals
    try:
        bestX, bestV, Xcur, mcur, diag = lp_round(bestX, bestX, bestV, 1e-4, 5, t_cap=9.0)
    except Exception:
        Xcur = bestX.copy(); mcur = bestV; diag = (None,None,None,None)
    if time.time() > LP_CUTOFF - 3.0 or time.time() > T_END - 12.0:
        bestX = bestX/np.linalg.norm(bestX, axis=1, keepdims=True)
        return bestX

    # ---- dual stress backbone ----
    Xbase = bestX.copy()
    mbase = float(bestV)
    backbone_I = None; backbone_J = None
    try:
        du, dI, dJ, dsl = diag
        if du is not None and dI is not None and len(du)==len(dI):
            s = -np.asarray(du, dtype=np.float64)
            s[~np.isfinite(s)] = 0.0
            smax = float(np.max(s)) if s.size>0 else 0.0
            if smax > 0 and np.isfinite(smax):
                thr = smax*1e-6
                msk = s > max(thr, 1e-12)
                if np.sum(msk) > 10:
                    backbone_I = np.asarray(dI[msk], dtype=np.int64)
                    backbone_J = np.asarray(dJ[msk], dtype=np.int64)
    except Exception:
        backbone_I = None
    # fallback: tightest 800 as pseudo-backbone if duals missing
    if backbone_I is None:
        try:
            G = Xbase@Xbase.T
            ca = G[iu_r, iu_c]; sl = mbase - ca
            idx = np.argsort(sl)[:800]
            backbone_I = iu_r[idx]; backbone_J = iu_c[idx]
        except Exception:
            backbone_I = np.empty(0,dtype=np.int64); backbone_J = np.empty(0,dtype=np.int64)

    # ---- build active matrix A for nullspace (slack < 1e-9) ----
    try:
        Bbase = _tangent_bases(Xbase)
        G = Xbase@Xbase.T
        ca = G[iu_r, iu_c]
        sl = mbase - ca
        # adaptive cutoff to get p in [800, 2000]
        for cut in (1e-9, 3e-9, 1e-8, 2e-7, 2e-5):
            t_idx = np.where(sl < cut)[0]
            if t_idx.size >= 600:
                break
        if t_idx.size > 2200:
            part = np.argpartition(sl[t_idx], 2200)[:2200]
            t_idx = t_idx[part]
        TI = iu_r[t_idx]; TJ = iu_c[t_idx]
        p = int(TI.size)
        aI = np.einsum('pdk,pd->pk', Bbase[TI], Xbase[TJ])
        aJ = np.einsum('pdk,pd->pk', Bbase[TJ], Xbase[TI])
        # dense A
        Amat = np.zeros((p, Nvar), dtype=np.float64)
        rr = np.arange(p)
        # fill via loops over k (fast)
        for k in range(4):
            Amat[rr, TI*4+k] = aI[:, k]
            Amat[rr, TJ*4+k] = aJ[:, k]
        # Gram + eigh for smallest modes
        Gram = Amat.T @ Amat
        # symmetrize
        Gram = 0.5*(Gram+Gram.T)
        w, V = np.linalg.eigh(Gram)  # V columns, ascending
        # rotation basis
        R = np.zeros((Nvar, 10), dtype=np.float64)
        ci = 0
        for a in range(5):
            for b in range(a+1, 5):
                Sx = np.zeros_like(Xbase)
                Sx[:, b] = Xbase[:, a]
                Sx[:, a] = -Xbase[:, b]
                # Sx already tangent? project via B
                U = np.einsum('ndk,nd->nk', Bbase, Sx)
                R[:, ci] = U.ravel()
                ci += 1
        # orthonormalize R
        Qr, _ = np.linalg.qr(R)
        # take smallest ~60 vectors
        K = min(60, Nvar)
        Vc = V[:, :K]  # Nvar x K
        # project out rotations
        Vc = Vc - Qr @ (Qr.T @ Vc)
        # norms
        nrm = np.sqrt(np.sum(Vc*Vc, axis=0))
        keep = nrm > 1e-6
        Vc = Vc[:, keep]
        nrm = nrm[keep]
        if Vc.shape[1] == 0:
            raise ValueError("no floppy")
        Vc = Vc / nrm[None, :]
        # orthonormalize remainder via QR to get clean basis
        Qf, _ = np.linalg.qr(Vc)
        # Qf is Nvar x F; columns are trial directions
        F = min(Qf.shape[1], 24)
        # order by eigenvalue? keep first F (they correspond to smallest since Vc from smallest)
        # use randomized mixing later
        dirs = [Qf[:, i].copy() for i in range(F)]
    except Exception:
        dirs = []
        Bbase = _tangent_bases(Xbase)

    # ---- trial exact retracted moves ----
    amps = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
    scored = []  # (score, maxcos, amp, vec)
    try:
        # precompute backbone dots for penalty: backbone max
        bI = backbone_I; bJ = backbone_J
        has_bb = bI is not None and len(bI) > 0
        # single directions
        if len(dirs) > 0:
            for di, vec in enumerate(dirs):
                Vn = vec.reshape(n, 4)
                DX = np.einsum('ndk,nk->nd', Bbase, Vn)
                for amp in amps:
                    if time.time() > LP_CUTOFF - 2.0 or time.time() > T_END - 10.0:
                        break
                    Xt = Xbase + amp*DX
                    nn = np.sqrt(np.sum(Xt*Xt, axis=1, keepdims=True)); nn[nn==0]=1.0
                    Xt = Xt/nn
                    mc = _max_cos(Xt)
                    pen = 0.0
                    if has_bb:
                        # mean slack of backbone beyond 1e-4
                        bd = np.sum(Xt[bI]*Xt[bJ], axis=1)
                        slack = mbase - bd
                        # penalize slack >1e-4 (breaking backbone)
                        excess = np.maximum(0.0, slack - 1e-4)
                        pen = 0.001*float(np.mean(excess))
                    sc = mc + pen
                    scored.append((sc, mc, amp, vec.copy()))
            # random combos (deterministic)
            ncomb = 40
            for ci_ in range(ncomb):
                if time.time() > LP_CUTOFF - 2.0 or time.time() > T_END - 10.0:
                    break
                coeff = rng.standard_normal(len(dirs))
                coeff /= (np.linalg.norm(coeff)+1e-30)
                vec = sum(coeff[i]*dirs[i] for i in range(len(dirs)))
                vn = float(np.linalg.norm(vec))
                if vn < 1e-12:
                    continue
                vec = vec/vn
                Vn = vec.reshape(n, 4)
                DX = np.einsum('ndk,nk->nd', Bbase, Vn)
                for amp in (1e-3, 3e-3, 1e-2, 3e-2):
                    Xt = Xbase + amp*DX
                    nn = np.sqrt(np.sum(Xt*Xt, axis=1, keepdims=True)); nn[nn==0]=1.0
                    Xt = Xt/nn
                    mc = _max_cos(Xt)
                    pen = 0.0
                    if has_bb:
                        bd = np.sum(Xt[bI]*Xt[bJ], axis=1)
                        slack = mbase - bd
                        excess = np.maximum(0.0, slack - 1e-4)
                        pen = 0.001*float(np.mean(excess))
                    sc = mc + pen
                    scored.append((sc, mc, amp, vec.copy()))
        scored.sort(key=lambda t: t[0])
    except Exception:
        scored = []

    # ---- follow promising directions with short LP polish ----
    # keep those not immediately raising max by more than 2e-4 (allow 2nd-order rise)
    cands2 = []
    try:
        for (sc, mc, amp, vec) in scored:
            if mc <= mbase + 2e-4:
                cands2.append((sc, mc, amp, vec))
            if len(cands2) >= 40:
                break
    except Exception:
        cands2 = []
    # deduplicate: keep diverse (limit per direction) - take top 6 spaced
    shortlist = []
    try:
        seen_amps = set()
        for tup in cands2:
            shortlist.append(tup)
            if len(shortlist) >= 6:
                break
    except Exception:
        pass

    # polish shortlist (5s cap each via 3 solves)
    for si, (sc, mc, amp, vec) in enumerate(shortlist):
        if time.time() > LP_CUTOFF or time.time() > T_END - 9.0:
            break
        try:
            Vn = vec.reshape(n, 4)
            DX = np.einsum('ndk,nk->nd', Bbase, Vn)
            Xt = Xbase + amp*DX
            nn = np.sqrt(np.sum(Xt*Xt, axis=1, keepdims=True)); nn[nn==0]=1.0
            Xt = Xt/nn
            # quick check max already computed
            nX, nV, _, _, _ = lp_round(Xt, bestX, bestV, 1e-4, 3, t_cap=5.0)
            if nV < bestV - 1e-12:
                bestX = nX.copy(); bestV = float(nV)
                # rebase for further tries if big win
                if bestV < mbase - 1e-7:
                    Xbase = bestX.copy(); mbase = float(bestV)
        except Exception:
            continue
        # only polish first few fully; time guard
        if time.time() > LP_CUTOFF:
            break

    # ---- final longer polish on best ----
    try:
        if time.time() < LP_CUTOFF and time.time() < T_END - 6.0:
            # remaining solves budget based on time
            rem_t = min(LP_CUTOFF - time.time(), T_END - 6.0 - time.time())
            ns = 8 if rem_t > 40 else (5 if rem_t > 20 else 3)
            if ns > 0 and rem_t > 2.0:
                bestX, bestV, _, _, _ = lp_round(bestX, bestX, bestV, 8e-5, ns, t_cap=9.0)
    except Exception:
        pass

    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
