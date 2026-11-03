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
    NV = n * 5
    I0, J0 = np.triu_indices(n, 1)
    P = len(I0)

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
    bestVal = _max_cos(bestX)

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

    def contact_set(X, m, slack=1e-4):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        g = G[I0, J0]
        sel = np.where(g > m - slack)[0]
        codes = (I0[sel].astype(np.int64) * n + J0[sel].astype(np.int64))
        return set(codes.tolist())

    def lp_jam_short(Xinit, deadline, max_it=8, radius_init=0.004, slack=1.5e-3):
        X = Xinit.copy()
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        Nv = n * 5
        radius = float(radius_init)
        fails = 0
        for _ in range(int(max_it)):
            if time.time() > deadline:
                break
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
            elif len(sel) > 3000:
                K0 = 3000
                part = np.argpartition(gfull[sel], -K0)[-K0:]
                sel = sel[part]
            Ic = I0[sel]; Jc = J0[sel]; cc = gfull[sel]
            K = len(sel)
            T = tangent_bases(X)
            TI = T[Ic]
            TJ = T[Jc]
            XJ = X[Jc]
            XI = X[Ic]
            Ai = np.sum(TI * XJ[:, :, None], axis=1)
            Aj = np.sum(TJ * XI[:, :, None], axis=1)
            cols = np.concatenate([(Ic[:, None] * 5 + np.arange(5)).ravel(),
                                   (Jc[:, None] * 5 + np.arange(5)).ravel(),
                                   np.full(K, Nv)])
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
                if fails > 6:
                    break
                continue
            fails = 0
            y = res.x[:Nv]
            t_pred = float(res.x[Nv])
            if not np.isfinite(t_pred) or t_pred >= m - 1e-13:
                radius *= 0.5
                if radius < 1e-7:
                    break
                continue
            U = y.reshape(n, 5)
            Xn = X + np.einsum('nik,nk->ni', T, U)
            Xn /= np.maximum(np.linalg.norm(Xn, axis=1, keepdims=True), 1e-12)
            m_new = _max_cos(Xn)
            if m_new < m - 1e-12:
                X = Xn
                radius = min(0.010, radius * 1.2)
            else:
                radius *= 0.5
                if radius < 5e-9:
                    break
        return X, _max_cos(X)

    STEPS = [0.02, 0.05, 0.10, 0.20]
    TOL_WORSE = 2e-3
    JACC_TH = 0.9

    # outer cycling over nullspaces
    while time.time() < T_END - 12.0:
        cycle_deadline = T_END - 10.0
        # build active Jacobian at slack 1e-7
        try:
            G0 = bestX @ bestX.T
            np.fill_diagonal(G0, -2.0)
            m0 = float(np.max(G0))
            g0 = G0[I0, J0]
            selA = np.where(g0 > m0 - 1e-7)[0]
            if len(selA) < 200:
                K0 = min(P, 800)
                selA = np.argpartition(g0, -K0)[-K0:]
            if len(selA) > 6000:
                K0 = 6000
                part = np.argpartition(g0[selA], -K0)[-K0:]
                selA = selA[part]
            IcA = I0[selA]; JcA = J0[selA]
            KA = len(selA)
            TA = tangent_bases(bestX)
            TIA = TA[IcA]; TJA = TA[JcA]
            XJA = bestX[JcA]; XIA = bestX[IcA]
            AiA = np.sum(TIA * XJA[:, :, None], axis=1)
            AjA = np.sum(TJA * XIA[:, :, None], axis=1)
            Amat = np.zeros((KA, NV), dtype=float)
            # fill rows: use advanced indexing via loops in vectorized chunks
            # build with scatter using numpy (KA~1300, NV 1245 fine)
            ridx = np.repeat(np.arange(KA), 5)
            # Ai part
            # faster: loop over k in python may be ~1300* ok but use vectorized assignment
            # Use flat indexing
            cols_i = (IcA[:, None] * 5 + np.arange(5))
            cols_j = (JcA[:, None] * 5 + np.arange(5))
            # assign
            # Amat[np.repeat(arange,5), cols.ravel()] needs repeat
            Amat[r1 if False else np.repeat(np.arange(KA), 5), cols_i.ravel()] = AiA.ravel()
            Amat[np.repeat(np.arange(KA), 5), cols_j.ravel()] += AjA.ravel()
            # Note second assignment uses same rows but different cols; need separate row index
            # The above second line reused same rows variable but cols differ -> need to redo correctly:
            # Actually first line already filled Ai; second line adds Aj (cols disjoint unless i==j impossible) so += is = .
            # rotation modes NV x 15
            Rmat = np.zeros((NV, 15), dtype=float)
            c = 0
            for p in range(6):
                for q in range(p + 1, 6):
                    # Omega x: o[p]=x_q, o[q]=-x_p
                    Ox = np.zeros((n, 6), dtype=float)
                    Ox[:, p] = bestX[:, q]
                    Ox[:, q] = -bestX[:, p]
                    # coeff = T_i^T o_i
                    Cc = np.sum(TA * Ox[:, :, None], axis=1)  # n x 5
                    Rmat[:, c] = Cc.ravel()
                    c += 1
            Qrot, _ = np.linalg.qr(Rmat)
            M = Amat.T @ Amat
            tr = float(np.trace(M)) / float(NV) if NV > 0 else 1.0
            pen = max(100.0, (tr + 1e-6) * 1000.0)
            Mpen = M + pen * (Qrot @ Qrot.T)
            w, V = np.linalg.eigh(Mpen)
            # V columns ascending; take 20 smallest
            ndir = min(20, V.shape[1])
            dirs = V[:, :ndir]  # NV x ndir
        except Exception:
            break

        try:
            inc_set = contact_set(bestX, bestVal, slack=1e-4)
        except Exception:
            inc_set = set()
        if len(inc_set) == 0:
            inc_set = set([0])

        # generate candidates
        surv = []  # (mnew, Xnew)
        try:
            Tcur = tangent_bases(bestX)
        except Exception:
            break
        m_base = bestVal
        for di in range(dirs.shape[1]):
            if time.time() > cycle_deadline - 8.0:
                break
            v = dirs[:, di]
            # ensure orthogonal to rotations (project)
            # v already penalized, but re-project for safety
            # v = v - Qrot@(Qrot.T@v)
            try:
                coeff = Qrot.T @ v
                v = v - Qrot @ coeff
                nv = float(np.linalg.norm(v))
                if nv < 1e-12:
                    continue
                v = v / nv
            except Exception:
                pass
            Vb = v.reshape(n, 5)
            maxn0 = float(np.max(np.linalg.norm(Vb, axis=1)))
            if maxn0 < 1e-14:
                continue
            for sgn in (1.0, -1.0):
                for st in STEPS:
                    if time.time() > cycle_deadline - 8.0:
                        break
                    scale = float(st) / maxn0
                    U = Vb * (sgn * scale)
                    D = np.einsum('nik,nk->ni', Tcur, U)
                    nrm = np.linalg.norm(U, axis=1)
                    cosn = np.cos(nrm)
                    sinn = np.sin(nrm)
                    fact = np.ones_like(nrm)
                    nz = nrm > 1e-12
                    fact[nz] = sinn[nz] / (nrm[nz] + 1e-20)
                    Xt = cosn[:, None] * bestX + fact[:, None] * D
                    Xt /= np.maximum(np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)
                    Gt = Xt @ Xt.T
                    np.fill_diagonal(Gt, -2.0)
                    mt = float(np.max(Gt))
                    if mt < m_base - 1e-12:
                        surv.append((mt, Xt.copy()))
                    elif mt <= m_base + TOL_WORSE + 1e-12:
                        # Jaccard at slack 1e-4
                        try:
                            gtf = Gt[I0, J0]
                            selT = np.where(gtf > mt - 1e-4)[0]
                            codesT = set(((I0[selT].astype(np.int64) * n + J0[selT].astype(np.int64))).tolist())
                            inter = len(inc_set.intersection(codesT))
                            union = len(inc_set) + len(codesT) - inter
                            jacc = (float(inter) / float(union)) if union > 0 else 1.0
                        except Exception:
                            jacc = 1.0
                        if jacc < JACC_TH:
                            surv.append((mt, Xt.copy()))
                    # else reject
        if not surv:
            # no diverse candidates: break to avoid loop
            # try slightly larger slack for contact set? just exit outer to return best
            break
        # sort by exact max (best first)
        surv.sort(key=lambda z: z[0])
        # keep unique / limit to fit time: at most 6 jams per cycle
        # prioritize: improvements first then diverse; already sorted
        max_jams = 6
        # estimate time per jam: leave reserve
        jammed_any = False
        for si in range(min(len(surv), max_jams)):
            if time.time() > T_END - 10.0:
                break
            mt0, Xt0 = surv[si]
            # skip near-duplicates? simple skip if mt0 much worse than best surv and we already improved
            try:
                Xj, vj = lp_jam_short(Xt0, deadline=T_END - 6.0, max_it=8,
                                      radius_init=0.004, slack=1.5e-3)
            except Exception:
                continue
            if np.isfinite(vj) and vj < bestVal - 1e-12:
                bestX = (Xj / np.maximum(np.linalg.norm(Xj, axis=1, keepdims=True), 1e-12)).copy()
                bestVal = float(vj)
                jammed_any = True
                # rebuild nullspace immediately for new basin
                break
        if not jammed_any:
            # try remaining survivors if time allows (next batch) without rebuilding
            # jam a second batch of next best survivors
            remaining = surv[max_jams:max_jams + 4]
            improved = False
            for mt0, Xt0 in remaining:
                if time.time() > T_END - 10.0:
                    break
                try:
                    Xj, vj = lp_jam_short(Xt0, deadline=T_END - 6.0, max_it=8,
                                          radius_init=0.004, slack=1.5e-3)
                except Exception:
                    continue
                if np.isfinite(vj) and vj < bestVal - 1e-12:
                    bestX = (Xj / np.maximum(np.linalg.norm(Xj, axis=1, keepdims=True), 1e-12)).copy()
                    bestVal = float(vj)
                    improved = True
                    break
            if improved:
                continue
            # no improvement in this cycle -> stop (avoid wasting time cycling same nullspace)
            break

    bestX /= np.maximum(np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)
    return bestX
