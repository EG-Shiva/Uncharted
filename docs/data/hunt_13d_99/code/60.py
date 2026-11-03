import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _norm_rows(V):
    n = np.linalg.norm(V, axis=1, keepdims=True)
    n = np.where(n < 1e-15, 1.0, n)
    return V / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _tangent_bases(X):
    n, d = X.shape
    dm = d - 1
    B = np.empty((n, d, dm))
    am = np.argmax(np.abs(X), axis=1)
    eye = np.eye(dm)
    for i in range(n):
        xi = X[i]
        a = int(am[i])
        mask = np.arange(d) != a
        xj = xi[mask]
        M = -xi[:, None] * xj[None, :]
        M[mask, :] += eye
        Q, _ = np.linalg.qr(M, mode='reduced')
        B[i] = Q
    return B

def _lse_polish(X0, alpha, maxiter, t_end):
    N, d = X0.shape
    if time.time() > t_end:
        return X0.copy(), _max_cos(X0)
    U0 = X0.ravel().astype(float).copy()
    def fun(Uf):
        U = Uf.reshape(N, d)
        nr = np.linalg.norm(U, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = U / nr
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        E = np.exp(alpha * (G - m))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5) + 1e-300
        L = m + np.log(S) / alpha
        GX = (E @ X) / S
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GU = (GX - dot * X) / nr
        return float(L), GU.ravel().astype(float)
    try:
        res = minimize(fun, U0, method='L-BFGS-B', jac=True,
                       options={'maxiter': int(maxiter), 'ftol': 1e-20,
                                'gtol': 1e-9, 'maxfun': int(maxiter)*6, 'maxcor': 8})
        U = res.x.reshape(N, d)
        Xn = _norm_rows(U)
        return Xn, _max_cos(Xn)
    except Exception:
        return X0.copy(), _max_cos(X0)

def solve():
    t0 = time.time()
    HARD = t0 + 150.0
    # load incumbent
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and np.shape(xb) == (99, 13):
        cands.append(_norm_rows(np.asarray(xb, float)))
    if xr is not None and np.shape(xr) == (99, 13):
        cands.append(_norm_rows(np.asarray(xr, float)))
    if len(cands) == 0:
        rng0 = np.random.RandomState(0)
        cands.append(_norm_rows(rng0.standard_normal(size=(99, 13))))
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    Xbest = cands[bi].copy()
    best = float(vals[bi])
    X = Xbest.copy()
    m_cur = best
    n, d = 99, 13
    dm = d - 1
    NVAR = n * dm
    NCOL = NVAR + 1
    iu0, iu1 = np.triu_indices(n, 1)
    ar = np.arange(dm)
    rng = np.random.RandomState(1)

    r = 3e-4
    r_min = 1e-7
    EPS = 2e-6
    stall = 0
    it = 0
    # precompute bounds template will be built per r
    while time.time() < HARD - 4.0:
        if r < r_min:
            break
        it += 1
        if time.time() > HARD - 4.0:
            break
        G = X @ X.T
        c_all = G[iu0, iu1]
        m_cur = float(np.max(c_all))
        sel = (m_cur - c_all) < EPS
        cnt = int(np.sum(sel))
        if cnt == 0:
            r *= 0.5
            continue
        # cap to keep LP tractable: keep worst 1100
        if cnt > 1100:
            idx = np.where(sel)[0]
            cv = c_all[idx]
            keep = np.argpartition(cv, -1100)[-1100:]
            sel2 = np.zeros_like(sel, dtype=bool)
            sel2[idx[keep]] = True
            sel = sel2
            cnt = 1100
        I = iu0[sel].astype(np.int64)
        J = iu1[sel].astype(np.int64)
        c_sel = c_all[sel]
        p = cnt
        B = _tangent_bases(X)
        XI = X[I]; XJ = X[J]
        BI = B[I]; BJ = B[J]
        aI = np.einsum('pi,pij->pj', XJ, BI)
        aJ = np.einsum('pi,pij->pj', XI, BJ)
        cols = np.empty((p, 25), dtype=np.int64)
        cols[:, :12] = I[:, None] * 12 + ar
        cols[:, 12:24] = J[:, None] * 12 + ar
        cols[:, 24] = NVAR
        vmat = np.empty((p, 25), dtype=float)
        vmat[:, :12] = aI
        vmat[:, 12:24] = aJ
        vmat[:, 24] = -1.0
        rows = np.repeat(np.arange(p), 25)
        try:
            A_ub = coo_matrix((vmat.ravel(), (rows, cols.ravel())), shape=(p, NCOL)).tocsr()
        except Exception:
            r *= 0.5
            continue
        b_ub = -c_sel
        obj = np.zeros(NCOL); obj[-1] = 1.0
        bounds = [(-r, r)] * NVAR + [(None, None)]
        A_eq = None
        # build equality xi^T di =0 already encoded via tangent coords, no eq needed
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs",
                          options={"time_limit": 4.0, "presolve": True})
        except Exception:
            r *= 0.5
            continue
        if res.status != 0:
            r *= 0.5
            stall += 1
        else:
            t_pred = float(res.x[-1])
            U = res.x[:-1]
            # only proceed if predicted improvement
            if t_pred < m_cur - 1e-12:
                Umat = U.reshape(n, dm)
                D = np.einsum('ndk,nk->nd', B, Umat)
                Xt = _norm_rows(X + D)
                mt = _max_cos(Xt)
                if mt < m_cur - 1e-12 and mt < best + 1e-12 or mt < best:
                    # accept only true decrease in exact maximum (allow equal to best then improve)
                    if mt < best:
                        Xbest = Xt.copy(); best = mt
                    X = Xt.copy(); m_cur = mt
                    stall = 0
                    # slow geometric decay on success
                    r = max(r * 0.92, r_min * 1.01)
                    # add newly violated pairs automatically next iter
                    continue
                else:
                    # predicted but not realized (curvature) -> shrink
                    r *= 0.5
                    stall += 1
            else:
                stall += 1
                r *= 0.7
        # ---- null-space lateral exploration when stalled ----
        if stall >= 2:
            if time.time() > HARD - 25.0:
                # not enough time for polish, just decay
                r *= 0.6
                stall = 0
                continue
            # Build dense Jacobian for SVD (p x NVAR)
            try:
                # dense J
                Jd = np.zeros((p, NVAR), dtype=float)
                # fill row by row (vectorized via loops, p<=1100 fine)
                for k in range(p):
                    Jd[k, int(I[k])*12:int(I[k])*12+12] = aI[k]
                    Jd[k, int(J[k])*12:int(J[k])*12+12] = aJ[k]
                # SVD of active Jacobian (economy)
                _, Ss, Vt = np.linalg.svd(Jd, full_matrices=False)
            except Exception:
                r *= 0.6
                stall = 0
                continue
            # generate null directions: random vectors projected off row space
            ndir = 4
            dirs = []
            # 1) projected LSE gradient-ish: use sum gradient? use random first then gradient
            # compute tangent LSE gradient for projection
            try:
                alpha_g = 2500.0
                Gg = X @ X.T
                np.fill_diagonal(Gg, -10.0)
                mg = float(np.max(Gg))
                Eg = np.exp(np.clip(alpha_g*(Gg-mg), -60, 0.0))
                np.fill_diagonal(Eg, 0.0)
                Sg = float(Eg.sum()*0.5)+1e-300
                GXg = (Eg @ X)/Sg
                # tangent coords
                Gt = np.einsum('ndk,nd->nk', B, GXg)  # (n,12)
                gv = Gt.ravel()
                # project to null
                coeff = Vt @ gv
                gnull = gv - Vt.T @ coeff
                nn = np.linalg.norm(gnull)
                if nn > 1e-12:
                    dirs.append(gnull/nn)
            except Exception:
                pass
            for _ in range(ndir):
                z = rng.standard_normal(size=NVAR)
                coeff = Vt @ z
                zn = z - Vt.T @ coeff
                nn = np.linalg.norm(zn)
                if nn < 1e-12:
                    continue
                dirs.append(zn/nn)
            if len(dirs) == 0:
                r *= 0.6
                stall = 0
                continue
            # normalize each dir to max per-point norm =1
            steps = [1e-4, 3e-5, 1e-5]
            improved = False
            for di, dv in enumerate(dirs[:5]):
                if time.time() > HARD - 12.0:
                    break
                Dm = dv.reshape(n, dm)
                per = np.sqrt((Dm**2).sum(axis=1)).max()
                if per < 1e-15:
                    continue
                Dm = Dm / per
                Dfull_base = np.einsum('ndk,nk->nd', B, Dm)
                for s in steps:
                    for sgn in (+1.0, -1.0):
                        if time.time() > HARD - 12.0:
                            break
                        Xt0 = _norm_rows(X + sgn*s*Dfull_base)
                        m0 = _max_cos(Xt0)
                        # quick filter: if much worse skip polish
                        if m0 > m_cur + 5e-6:
                            continue
                        # short high-alpha polish
                        Xp, mp = _lse_polish(Xt0, 3500.0, 25, HARD-3.0)
                        if mp < best - 1e-12:
                            Xbest = Xp.copy(); best = mp
                            X = Xp.copy(); m_cur = mp
                            improved = True
                            break
                        elif mp < m_cur - 2e-12:
                            X = Xp.copy(); m_cur = mp
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                stall = 0
                # keep radius, or slightly reduce
                r = max(r*0.95, r_min)
            else:
                # no basin change helped -> decay trust geometrically
                r *= 0.55
                stall = 0
            # loop continues; newly violated pairs added next iter via recomputed active set
    return _norm_rows(Xbest.copy())
