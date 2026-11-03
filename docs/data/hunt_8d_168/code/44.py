import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import lsmr

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _full_sumsq_max(X, c):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    R = G - c
    np.maximum(R, 0.0, out=R)
    # only upper triangle counted once; G symmetric, diagonal -2-c<0
    # sumsq over all ordered pairs /2
    s = float((R*R).sum() / 2.0)
    m = float(G.max())
    return s, m, G

def _stress_weights(X, I, J, Gvals):
    # LP: min sum s_i s.t. -s_i <= (Mw) <= s_i, sum w=1, w,s>=0
    # M maps w -> tangent imbalance
    try:
        n, d = X.shape
        p = len(I)
        if p == 0 or p > 900:
            return None
        # build M (nd x p) dense
        nd = n*d
        M = np.zeros((nd, p), dtype=np.float64)
        # vectorized per pair loop (p<=900 fine)
        for k in range(p):
            i = int(I[k]); j = int(J[k])
            g = float(Gvals[k])
            ui = X[j] - g*X[i]
            vj = X[i] - g*X[j]
            M[i*d:(i+1)*d, k] = ui
            M[j*d:(j+1)*d, k] = vj
        nvar = p + n
        # A_ub: 2*nd rows
        # row t=i*d+a: Mw - s_i <=0 ; -Mw - s_i <=0
        rows = []
        cols = []
        data = []
        # build sparse
        # for each t, w-part is M[t,:], s-part -1
        # we build with coo lists
        import math
        R = 2*nd
        rids = []
        cids = []
        vals = []
        for t in range(nd):
            i = t // d
            # + row = 2*t? use t and t+nd
            # + row
            # w part
            row_p = t
            row_m = t + nd
            # iterate over k where M[t,k]!=0 : only pairs incident to i
            # find them via sparse structure: instead dense row loop over p (1344*900=1.2M) ok
            pass
        # faster dense -> sparse conversion
        A_dense_w_p = M  # nd x p
        # A_ub = [[M, -E],[ -M, -E]] where E maps t->i
        # Build with coo via numpy
        # rows 0..nd-1 : [M, -1 at s_i]
        # rows nd..2nd-1 : [-M, -1]
        r = np.repeat(np.arange(nd), p)
        c = np.tile(np.arange(p), nd)
        v = M.ravel()
        # filter tiny?
        mask = np.abs(v) > 1e-14
        r1 = r[mask]; c1 = c[mask]; v1 = v[mask]
        # s part for + rows
        rs = np.arange(nd)
        cs = p + (np.arange(nd)//d)
        vs = -np.ones(nd)
        r_all = np.concatenate([r1, rs])
        c_all = np.concatenate([c1, cs])
        v_all = np.concatenate([v1, vs])
        # - rows
        r2 = nd + r[mask]; c2 = c[mask]; v2 = -v[mask]
        rs2 = nd + np.arange(nd); cs2 = cs; vs2 = -np.ones(nd)
        Rall = np.concatenate([r_all, r2, rs2])
        Call = np.concatenate([c_all, c2, cs2])
        Vall = np.concatenate([v_all, v2, vs2])
        A_ub = coo_matrix((Vall, (Rall, Call)), shape=(2*nd, nvar)).tocsr()
        b_ub = np.zeros(2*nd)
        A_eq = coo_matrix((np.ones(p), (np.zeros(p, dtype=int), np.arange(p))), shape=(1, nvar)).tocsr()
        b_eq = np.ones(1)
        obj = np.concatenate([np.zeros(p), np.ones(n)])
        bounds = [(0, None)]*nvar
        res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if res.status != 0:
            return None
        w = np.asarray(res.x[:p], dtype=float)
        w = np.maximum(w, 0.0)
        return w
    except Exception:
        return None

def _lp_polish(x, seconds, r0=1.2e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20*radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 2800:
            idx = np.argsort(c[sel])[-2800:]
            I, J = I[idx], J[idx]
            cc = c[sel][idx]
        else:
            cc = c[sel]
            p = len(I)
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:, None]*d+np.arange(d), J[:, None]*d+np.arange(d), np.full((p, 1), n*d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius*1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def solve():
    t0 = time.time()
    BUDGET = 174.0
    n, d = 168, 8
    best = None; best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape != (n, d):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            va = _max_cos(a)
            if va < best_val:
                best_val = va; best = a.copy()
        except Exception:
            pass
    if best is None:
        rng0 = np.random.RandomState(0)
        best = rng0.randn(n, d); best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    m0 = best_val
    Xglob = best.copy()
    Gglob = best_val
    t_homotopy_end = t0 + 118.0
    targets = [m0-1e-6, m0-2e-6, m0-5e-6]
    N = n*d
    iu0 = np.triu_indices(n, 1)
    iu_I0 = iu0[0]; iu_J0 = iu0[1]
    for tc, c in enumerate(targets):
        if time.time() > t_homotopy_end:
            break
        Xcur = Xglob.copy()
        _, _, _G = _full_sumsq_max(Xcur, c)
        s_cur, m_cur, _ = _full_sumsq_max(Xcur, c)
        best_s = s_cur
        trust = 1e-2
        # stress scaling dict: default 1
        row_scale = None
        wmap = None
        # for mapping pair->weight, keep last stress pairs
        sI = None; sJ = None; sW = None
        patience = 0
        # zero counters not needed explicitly; use sW
        for it in range(90):
            if time.time() > t_homotopy_end:
                break
            if time.time()+1.5 > t0+BUDGET:
                break
            G = Xcur @ Xcur.T
            # active: G > c-3e-4, upper triangle
            # collect
            cvals = G[iu_I0, iu_J0]
            thr = c - 3e-4
            mask = cvals > thr
            if np.count_nonzero(mask) == 0:
                # feasible
                s_cur2, m_cur2, _ = _full_sumsq_max(Xcur, c)
                if m_cur2 < Gglob:
                    Xglob = Xcur.copy(); Gglob = m_cur2
                break
            AI = iu_I0[mask]; AJ = iu_J0[mask]; AV = cvals[mask]
            # cap to tightest 3500
            if len(AI) > 3500:
                o = np.argsort(AV)[-3500:]
                AI, AJ, AV = AI[o], AJ[o], AV[o]
            R = AV - c
            viol_m = R > 0
            m_viol = int(np.count_nonzero(viol_m))
            s_cur_here = float((np.maximum(R, 0.0)**2).sum())
            # max
            m_here = float(AV.max()) if len(AV)>0 else -2.0
            # full max for global tracking
            full_max = _max_cos(Xcur)
            if full_max < Gglob - 1e-12:
                Gglob = full_max; Xglob = Xcur.copy()
            if m_viol == 0:
                # feasible for this active window
                if full_max <= c + 1e-12:
                    Xglob = Xcur.copy(); Gglob = full_max
                    break
                else:
                    # need wider? but active window covers all above c-3e-4, so feasible
                    Xglob = Xcur.copy(); Gglob = full_max
                    break
            VI = AI[viol_m]; VJ = AJ[viol_m]; VR = R[viol_m]
            # build Jacobian sparse for violating rows
            mm = m_viol
            # cap rows for lsmr if huge (shouldn't be)
            if mm > 3500:
                o = np.argsort(VR)[-3500:]
                VI, VJ, VR = VI[o], VJ[o], VR[o]
                mm = 3500
            # row scales from stress
            scl = np.ones(mm)
            if sI is not None and sW is not None and len(sI)>0:
                # map via dict
                # build dict for quick lookup: encode i*n+j
                dmap = {int(sI[k])*n+int(sJ[k]): float(sW[k]) for k in range(len(sI))}
                wmax = max(dmap.values()) if len(dmap)>0 else 1.0
                if wmax <= 0: wmax = 1.0
                for k in range(mm):
                    key = int(VI[k])*n+int(VJ[k])
                    wv = dmap.get(key, 0.0)
                    scl[k] = np.sqrt((wv/max(wmax,1e-18)) + 0.15)
            # Build CSR: each row 2*d entries
            nnz = mm*2*d
            rrow = np.repeat(np.arange(mm), 2*d)
            ccol = np.empty(nnz, dtype=np.int64)
            ddat = np.empty(nnz, dtype=np.float64)
            # fill
            # columns VI*d+a and VJ*d+a
            # use vectorization
            # for efficiency loop over d (8)
            ptr = 0
            # we can build with tile
            # VI block
            for a in range(d):
                ccol[ptr:ptr+mm] = VI*d+a
                ddat[ptr:ptr+mm] = Xcur[VJ, a]*scl
                ptr += mm
            for a in range(d):
                ccol[ptr:ptr+mm] = VJ*d+a
                ddat[ptr:ptr+mm] = Xcur[VI, a]*scl
                ptr += mm
            # need rows ordering to match rrow? rrow currently repeats per row contiguous 2d,
            # but we filled column-major. Fix: build correctly
            # Rebuild with row-major: simpler to build coo then csr
            # redo:
            rrow2 = np.repeat(np.arange(mm), 2*d)
            ccol2 = np.empty(nnz, dtype=np.int64)
            ccol2[0::2*d] = VI*d+0  # placeholder? do loop
            # easiest: construct via loops in numpy
            ccol2 = np.zeros(nnz, dtype=np.int64)
            ddat2 = np.zeros(nnz, dtype=np.float64)
            for k in range(d):
                # positions within each row: k and d+k
                idx1 = np.arange(mm)*(2*d)+k
                idx2 = np.arange(mm)*(2*d)+d+k
                ccol2[idx1] = VI*d+k
                ddat2[idx1] = Xcur[VJ, k]*scl
                ccol2[idx2] = VJ*d+k
                ddat2[idx2] = Xcur[VI, k]*scl
            A = coo_matrix((ddat2, (rrow2, ccol2)), shape=(mm, N)).tocsr()
            b = -VR*scl
            damp = 1e-3/max(trust, 1e-5)
            damp = float(np.clip(damp, 0.05, 20.0))
            try:
                ls = lsmr(A, b, damp=damp, maxiter=min(400, max(100, mm//4)))
                s_vec = np.asarray(ls[0], dtype=float)
            except Exception:
                trust *= 0.5
                if trust < 9e-5:
                    break
                continue
            if not np.all(np.isfinite(s_vec)):
                trust *= 0.5
                continue
            S = s_vec.reshape(n, d)
            # project onto tangent
            dot = (S*Xcur).sum(axis=1, keepdims=True)
            S = S - dot*Xcur
            # trust clip per point
            nr = np.sqrt((S*S).sum(axis=1, keepdims=True)) + 1e-18
            over = nr[:, 0] > trust
            if np.any(over):
                S[over] *= (trust/nr[over, 0])[:, None]
            Xnew = Xcur + S
            Xnew /= np.sqrt((Xnew*Xnew).sum(axis=1, keepdims=True))
            s_new, m_new, _ = _full_sumsq_max(Xnew, c)
            full_new = m_new  # m_new is full max (since G full)
            if s_new < s_cur_here*(1.0-1e-12) or (s_new <= s_cur_here and full_new < m_cur - 1e-14):
                Xcur = Xnew
                s_cur_here = s_new
                if s_new < best_s:
                    best_s = s_new
                trust = min(trust*1.25, 1e-2)
                patience = 0
                if full_new < Gglob:
                    Gglob = full_new; Xglob = Xcur.copy()
                if s_new == 0.0 or full_new <= c:
                    Xglob = Xcur.copy(); Gglob = full_new
                    break
            else:
                trust *= 0.5
                patience += 1
                if trust < 1e-4 and patience > 4:
                    # try smaller step still? break
                    if patience > 12:
                        break
                if patience > 14:
                    break
            # stress testing every 8 iters
            if (it % 8 == 7) and (time.time() < t_homotopy_end - 2.0):
                try:
                    G2 = Xcur @ Xcur.T
                    cv2 = G2[iu_I0, iu_J0]
                    # tightest 800 above c-3e-4 for LP
                    m2 = cv2 > (c-3e-4)
                    II2 = iu_I0[m2]; JJ2 = iu_J0[m2]; VV2 = cv2[m2]
                    if len(II2) > 800:
                        o2 = np.argsort(VV2)[-800:]
                        II2, JJ2, VV2 = II2[o2], JJ2[o2], VV2[o2]
                    if 10 <= len(II2) <= 900:
                        w = _stress_weights(Xcur, II2, JJ2, VV2)
                        if w is not None:
                            sI, sJ, sW = II2, JJ2, w
                            # drop persistently zero: keep only w>tiny for mapping? keep all but scl handles
                            # recruit: nothing extra, active recomputed each iter
                except Exception:
                    pass
            # continue only while feasibility improves: handled by patience
        # end iters for this target; if reached (Gglob<=c) continue to next more aggressive
        # else try next target anyway from Xglob? continue loop
    # finish with trust-region LP polish on incumbent
    remain = (t0+BUDGET) - time.time() - 1.0
    if remain < 3.0:
        remain = 3.0
    # polish best two? just Xglob
    try:
        secs = min(remain, 55.0)
        Xp, vp = _lp_polish(Xglob, secs)
        if vp < Gglob:
            Xglob, Gglob = Xp, vp
        # extra short polish with smaller radius if time left
        remain2 = (t0+BUDGET)-time.time()-0.5
        if remain2 > 4.0:
            Xp2, vp2 = _lp_polish(Xglob, min(remain2-0.5, 20.0), r0=4e-4)
            if vp2 < Gglob:
                Xglob, Gglob = Xp2, vp2
    except Exception:
        pass
    Xglob = Xglob / np.linalg.norm(Xglob, axis=1, keepdims=True)
    return Xglob
