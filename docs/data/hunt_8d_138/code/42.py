import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = X / np.linalg.norm(X, axis=1, keepdims=True)
    # avoid nan
    n[~np.isfinite(n)] = 0.0
    rn = np.linalg.norm(n, axis=1, keepdims=True)
    rn[rn == 0] = 1.0
    return n / rn

def _tangent_bases(X):
    n, d = X.shape
    Bs = np.empty((n, d, d-1))
    for i in range(n):
        xi = X[i]
        M = np.eye(d) - np.outer(xi, xi)
        w, V = np.linalg.eigh(M)
        # smallest eigenvalue 0 -> discard first
        idx = np.argsort(w)
        V = V[:, idx[1:]]  # d x 7
        Bs[i] = V
    return Bs

def _build_J(X, Bs, pairs_I, pairs_J):
    n, d = X.shape
    k = d-1
    N = n*k
    p = len(pairs_I)
    J = np.empty((p, N))
    for r in range(p):
        i = pairs_I[r]; j = pairs_J[r]
        vi = Bs[i].T @ X[j]  # 7
        vj = Bs[j].T @ X[i]
        row = np.zeros(N)
        row[i*k:(i+1)*k] = vi
        row[j*k:(j+1)*k] = vj
        J[r] = row
    return J

def _lse_jam(X0, beta, maxiter):
    from scipy.optimize import minimize
    n, d = X0.shape
    iu = np.triu_indices(n, 1)
    Y = _norm(X0.copy())
    def fun(Yf):
        Ym = Yf.reshape(n, d)
        nr = np.sqrt(np.sum(Ym*Ym, axis=1, keepdims=True)) + 1e-30
        X = Ym / nr
        G = X @ X.T
        c = G[iu]
        mv = c.max()
        E = np.exp(beta*(c-mv))
        S = E.sum() + 1e-300
        f = float(mv + np.log(S)/beta)
        W = np.zeros((n, n))
        W[iu] = E
        W = W + W.T
        GX = (W @ X) / S
        dot = np.sum(GX*X, axis=1, keepdims=True)
        Gt = GX - dot*X
        GY = Gt / nr
        return f, GY.ravel()
    try:
        res = minimize(fun, Y.ravel(), jac=True, method='L-BFGS-B',
                       options={'maxiter': int(maxiter), 'ftol':1e-14, 'gtol':1e-6, 'maxcor':12})
        Y = _norm(res.x.reshape(n, d))
    except Exception:
        pass
    return _norm(Y)

def _lse_jam_drop(X0, beta, maxiter, di, dj, m_thr):
    from scipy.optimize import minimize
    n, d = X0.shape
    iu = np.triu_indices(n, 1)
    # mask out dropped pair from LSE? keep penalty instead
    Y = _norm(X0.copy())
    def fun(Yf):
        Ym = Yf.reshape(n, d)
        nr = np.sqrt(np.sum(Ym*Ym, axis=1, keepdims=True)) + 1e-30
        X = Ym / nr
        G = X @ X.T
        c = G[iu]
        mv_all = c.max()
        # LSE over all except dropped pair to allow it to stay low
        # find index of dropped pair in iu ordering is messy; just compute penalty
        cd = float(X[di] @ X[dj])
        # LSE value over all pairs (for gradient) but add penalty for dropped
        E = np.exp(beta*(c-mv_all))
        S = E.sum() + 1e-300
        f = float(mv_all + np.log(S)/beta)
        # penalty if cd > m_thr
        over = cd - m_thr
        pen = 0.0
        gpen = np.zeros_like(X)
        if over > 0:
            pen = 200.0*over*over + 20.0*over
            # grad w.r.t X: d cd / d Xdi = Xdj etc (projected later)
            coef = 400.0*over + 20.0
            gpen[di] += coef*X[dj]
            gpen[dj] += coef*X[di]
            # project to tangent for consistency then convert to Y grad
            dotp = np.sum(gpen*X, axis=1, keepdims=True)
            gpen = gpen - dotp*X
            f = f + 200.0*over*over + 20.0*over
        W = np.zeros((n, n))
        W[iu] = E
        W = W + W.T
        GX = (W @ X) / S
        dot = np.sum(GX*X, axis=1, keepdims=True)
        Gt = GX - dot*X + gpen
        GY = Gt / nr
        return f, GY.ravel()
    try:
        res = minimize(fun, Y.ravel(), jac=True, method='L-BFGS-B',
                       options={'maxiter': int(maxiter), 'ftol':1e-14, 'gtol':1e-6, 'maxcor':12})
        Y = _norm(res.x.reshape(n, d))
    except Exception:
        pass
    return _norm(Y)

def _lp_polish(X, deadline, radius0=1e-2, active=1e-3, cap=3000):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    radius = radius0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gfull = X @ X.T
        c = Gfull[iu]
        m = c.max()
        thr = m - active
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:, None]*d + np.arange(d), J[:, None]*d + np.arange(d), np.full((p,1), n*d)],1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p,1))],1).ravel()
        A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
        A_eq = coo_matrix((X.ravel(),(np.repeat(np.arange(n),d), np.arange(n*d))), shape=(n,nv)).tocsr()
        obj = np.zeros(nv); obj[-1]=1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method='highs')
        except Exception:
            radius*=0.5
            continue
        if res.status!=0:
            radius*=0.5
            continue
        y = X + res.x[:-1].reshape(n,d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius*1.5, 0.05)
        else:
            radius*=0.5
    return X

def solve():
    import time
    rng = np.random.default_rng(0)
    t0 = time.time()
    deadline = t0 + 170.0
    n, d = 138, 8
    best = None
    best_val = 1e9
    for fn in ("best.npy","record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape==(n,d):
                v = _norm(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val=c; best=v.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n,d)))
        best_val = _max_cos(best)
    incumbent = best.copy()
    inc_val = best_val

    # --- floppy mode analysis at several tolerances ---
    try:
        G0 = best @ best.T
        np.fill_diagonal(G0, -2.0)
        m0 = G0.max()
        iu_all = np.triu_indices(n,1)
        c_all = G0[iu_all]
        order_all = np.argsort(-c_all)
        sorted_c = c_all[order_all]
        # choose p target ~700 for Jacobian
        p_target = 700
        thr_star = m0 - sorted_c[min(p_target, len(sorted_c)-1)]
        # clamp to [1e-7,1e-3]
        thr_star = float(np.clip(thr_star, 1e-7, 1e-3))
        tols = [1e-7, 1e-5, 1e-4, thr_star]
        cand_points = []
        Bs = _tangent_bases(best)
        for tol in tols:
            if time.time() > deadline-40:
                break
            sel_idx = np.where(c_all > m0 - tol)[0]
            # also ensure at least 200 and at most 2500; adjust by rank if needed
            if len(sel_idx) < 150:
                # take top 300
                sel_idx = order_all[:300]
            if len(sel_idx) > 2500:
                sel_idx = order_all[:2500]
            I = iu_all[0][sel_idx]; J = iu_all[1][sel_idx]
            p = len(I)
            Jmat = _build_J(best, Bs, I, J)
            # SVD
            try:
                U,S,Vt = np.linalg.svd(Jmat, full_matrices=False)
            except Exception:
                continue
            # null / low-energy directions = smallest singular values
            ndir = min(20, Vt.shape[0])
            # take bottom ndir
            dirs = Vt[-ndir:]  # (ndir, N)
            k = d-1
            # build tangent direction fields
            for di in range(ndir):
                z = dirs[di].reshape(n,k)
                Uf = np.empty((n,d))
                for i in range(n):
                    Uf[i] = Bs[i] @ z[i]
                # normalize
                nr = np.sqrt(np.sum(Uf*Uf)) + 1e-30
                Uf = Uf / nr
                # exact line search over log spaced steps both signs
                steps = np.array([1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,6e-2,1e-1])
                for sgn in (1.0,-1.0):
                    for st in steps:
                        if time.time() > deadline-40:
                            break
                        t = sgn*float(st)
                        Y = _norm(best + t*Uf)
                        v = _max_cos(Y)
                        # allow continuation until inactive pair hits max:
                        # keep if not worse than m0+2e-4 (topology change candidate)
                        # or strictly better
                        if v < m0 + 2e-4:
                            cand_points.append((v, Y.copy(), tol, di, t))
                    if time.time() > deadline-40:
                        break
        # deduplicate / sort candidates: prefer lowest max
        cand_points.sort(key=lambda x: x[0])
        # keep diverse subset: take up to 6 with spacing in step
        kept = []
        seen = []
        for v,Y,tol,di,t in cand_points:
            if len(kept) >= 6:
                break
            # avoid near-duplicates (same di/tol sign)
            kept.append((v,Y))
        # polish kept with LSE + LP (short)
        for v,Y in kept:
            if time.time() > deadline-30:
                break
            # only polish promising: within 1e-3 of incumbent or better
            if v > inc_val + 1e-3:
                continue
            try:
                Y2 = _lse_jam(Y, beta=220.0, maxiter=220)
                v2 = _max_cos(Y2)
                # second stage higher beta
                Y2 = _lse_jam(Y2, beta=480.0, maxiter=180)
                v2 = _max_cos(Y2)
                Y3 = _lp_polish(Y2, min(deadline-25, time.time()+4.0), radius0=1e-2, active=1e-3)
                v3 = _max_cos(Y3)
                if v3 < inc_val:
                    inc_val = v3; incumbent = Y3.copy()
                elif v2 < inc_val:
                    inc_val = v2; incumbent = Y2.copy()
            except Exception:
                continue
    except Exception:
        pass

    # --- systematic drop-one-contact experiments ---
    try:
        # tight contacts of incumbent
        G = incumbent @ incumbent.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        iu = np.triu_indices(n,1)
        c = G[iu]
        topk = np.argsort(-c)[:12]
        for kk in range(len(topk)):
            if time.time() > deadline-12:
                break
            idx = topk[kk]
            di = int(iu[0][idx]); dj = int(iu[1][idx])
            m_thr = float(m - 1e-3)
            try:
                Y = incumbent.copy()
                # small kick to break the pair
                Y[di] += 0.02*rng.standard_normal(d)
                Y[dj] += 0.02*rng.standard_normal(d)
                Y = _norm(Y)
                Y = _lse_jam_drop(Y, beta=240.0, maxiter=200, di=di, dj=dj, m_thr=m_thr)
                # follow with normal LSE to re-jam without constraint
                Y = _lse_jam(Y, beta=450.0, maxiter=120)
                v = _max_cos(Y)
                if v < inc_val + 5e-4:
                    Yp = _lp_polish(Y, min(deadline-10, time.time()+3.0), radius0=1e-2, active=1e-3)
                    vp = _max_cos(Yp)
                    if vp < inc_val:
                        inc_val = vp; incumbent = Yp.copy()
            except Exception:
                continue
    except Exception:
        pass

    # --- final polish of incumbent with restarted trust radius ---
    try:
        remain = deadline - time.time()
        if remain > 3:
            Yf = _lp_polish(incumbent, deadline-0.5, radius0=1e-2, active=1e-3, cap=3000)
            vf = _max_cos(Yf)
            if vf < inc_val:
                inc_val = vf; incumbent = Yf
            else:
                # keep incumbent (return unchanged if no improvement per spec)
                pass
    except Exception:
        pass

    # safety: if incumbent worse than best loaded, return best
    try:
        if _max_cos(incumbent) > best_val + 1e-12:
            # still return incumbent? spec says return incumbent unchanged if no experiment improves.
            # That means return original best.
            return _norm(best)
    except Exception:
        pass
    return _norm(incumbent)
