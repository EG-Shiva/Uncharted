import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm[~np.isfinite(nrm)] = 1.0
    nrm[nrm == 0] = 1.0
    return X / nrm

def _tangent_bases(X):
    n, d = X.shape
    Bs = np.empty((n, d, d-1))
    for i in range(n):
        xi = X[i]
        M = np.eye(d) - np.outer(xi, xi)
        w, V = np.linalg.eigh(M)
        idx = np.argsort(w)
        V = V[:, idx[1:]]
        Bs[i] = V
    return Bs

def _build_J(X, Bs, I, J):
    n, d = X.shape
    k = d - 1
    N = n * k
    p = len(I)
    Jm = np.empty((p, N))
    for r in range(p):
        i = int(I[r]); j = int(J[r])
        vi = Bs[i].T @ X[j]
        vj = Bs[j].T @ X[i]
        row = np.zeros(N)
        row[i*k:(i+1)*k] = vi
        row[j*k:(j+1)*k] = vj
        Jm[r] = row
    return Jm

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
                       options={'maxiter': int(maxiter), 'ftol': 1e-14, 'gtol': 1e-6, 'maxcor': 12})
        Y = _norm(res.x.reshape(n, d))
    except Exception:
        pass
    return _norm(Y)

def _lse_jam_drop(X0, beta, maxiter, di, dj, m_thr):
    from scipy.optimize import minimize
    n, d = X0.shape
    iu = np.triu_indices(n, 1)
    # index of dropped pair in iu arrays
    iu0 = iu[0]; iu1 = iu[1]
    drop_mask = ((iu0 == di) & (iu1 == dj)) | ((iu0 == dj) & (iu1 == di))
    keep = ~drop_mask
    Y = _norm(X0.copy())
    def fun(Yf):
        Ym = Yf.reshape(n, d)
        nr = np.sqrt(np.sum(Ym*Ym, axis=1, keepdims=True)) + 1e-30
        X = Ym / nr
        G = X @ X.T
        c = G[iu]
        ck = c[keep]
        mv = ck.max()
        E = np.exp(beta*(ck-mv))
        S = E.sum() + 1e-300
        f = float(mv + np.log(S)/beta)
        Wk = np.zeros_like(c)
        Wk[keep] = E
        W = np.zeros((n, n))
        W[iu] = Wk
        W = W + W.T
        GX = (W @ X) / S
        dot = np.sum(GX*X, axis=1, keepdims=True)
        Gt = GX - dot*X
        cd = float(X[di] @ X[dj])
        over = cd - m_thr
        if over > 0:
            coef = 400.0*over + 20.0
            gpen = np.zeros_like(X)
            gpen[di] += coef*X[dj]
            gpen[dj] += coef*X[di]
            dotp = np.sum(gpen*X, axis=1, keepdims=True)
            gpen = gpen - dotp*X
            Gt = Gt + gpen
            f = f + 200.0*over*over + 20.0*over
        GY = Gt / nr
        return f, GY.ravel()
    try:
        res = minimize(fun, Y.ravel(), jac=True, method='L-BFGS-B',
                       options={'maxiter': int(maxiter), 'ftol': 1e-14, 'gtol': 1e-6, 'maxcor': 12})
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
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius*1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    rng = np.random.default_rng(1)
    t0 = time.time()
    deadline = t0 + 170.0
    n, d = 138, 8
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = _norm(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c; best = v.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(best)
    incumbent = best.copy()
    inc_val = best_val

    # Phase 1: floppy modes via SVD at several tolerances
    try:
        G0 = incumbent @ incumbent.T
        np.fill_diagonal(G0, -2.0)
        m0 = float(G0.max())
        iu_all = np.triu_indices(n, 1)
        c_all = G0[iu_all]
        order_all = np.argsort(-c_all)
        sorted_c = c_all[order_all]
        p_target = 800
        thr_star = float(m0 - sorted_c[min(p_target, len(sorted_c)-1)])
        thr_star = float(np.clip(thr_star, 1e-7, 1e-3))
        tols = [1e-7, 1e-5, 1e-4, thr_star]
        Bs = _tangent_bases(incumbent)
        cand = []
        for tol in tols:
            if time.time() > deadline - 60:
                break
            sel_idx = np.where(c_all > m0 - tol)[0]
            if len(sel_idx) < 150:
                sel_idx = order_all[:300]
            if len(sel_idx) > 2500:
                sel_idx = order_all[:2500]
            I = iu_all[0][sel_idx]; J = iu_all[1][sel_idx]
            Jm = _build_J(incumbent, Bs, I, J)
            try:
                U, S, Vt = np.linalg.svd(Jm, full_matrices=False)
            except Exception:
                continue
            ndir = min(12, Vt.shape[0])
            dirs = Vt[-ndir:]
            k = d - 1
            for di in range(ndir):
                z = dirs[di].reshape(n, k)
                Uf = np.empty((n, d))
                for i in range(n):
                    Uf[i] = Bs[i] @ z[i]
                nr = np.sqrt(np.sum(Uf*Uf)) + 1e-30
                Uf = Uf / nr
                steps = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 6e-2, 1e-1)
                for sgn in (1.0, -1.0):
                    for st in steps:
                        if time.time() > deadline - 60:
                            break
                        t = sgn*float(st)
                        Y = _norm(incumbent + t*Uf)
                        v = _max_cos(Y)
                        if v < m0 + 3e-4:
                            cand.append((v, Y.copy()))
        cand.sort(key=lambda x: x[0])
        kept = cand[:8]
        for v, Y in kept:
            if time.time() > deadline - 35:
                break
            if v > inc_val + 1e-3:
                continue
            try:
                Y2 = _lse_jam(Y, beta=250.0, maxiter=200)
                Y2 = _lse_jam(Y2, beta=500.0, maxiter=150)
                v2 = _max_cos(Y2)
                if v2 < inc_val + 5e-4:
                    Y3 = _lp_polish(Y2, min(deadline-25, time.time()+4.0), radius0=1e-2, active=1e-3)
                    v3 = _max_cos(Y3)
                    if v3 < inc_val:
                        inc_val = v3; incumbent = Y3.copy()
                    elif v2 < inc_val:
                        inc_val = v2; incumbent = Y2.copy()
                elif v2 < inc_val:
                    inc_val = v2; incumbent = Y2.copy()
            except Exception:
                continue
    except Exception:
        pass

    # Phase 2: drop-one-contact
    try:
        G = incumbent @ incumbent.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        iu = np.triu_indices(n, 1)
        c = G[iu]
        topk = np.argsort(-c)[:10]
        for kk in range(len(topk)):
            if time.time() > deadline - 14:
                break
            idx = topk[kk]
            di = int(iu[0][idx]); dj = int(iu[1][idx])
            m_thr = float(m - 1e-3)
            try:
                Y = incumbent.copy()
                Y[di] += 0.015*rng.standard_normal(d)
                Y[dj] += 0.015*rng.standard_normal(d)
                Y = _norm(Y)
                Y = _lse_jam_drop(Y, beta=250.0, maxiter=180, di=di, dj=dj, m_thr=m_thr)
                Y = _lse_jam(Y, beta=450.0, maxiter=100)
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

    # Phase 3: random tangential kicks fallback if time remains
    try:
        while time.time() < deadline - 8:
            sc = float(rng.choice([0.005, 0.01, 0.02]))
            Y = incumbent + sc*rng.standard_normal((n, d))
            # project kick to tangent to keep close
            dot = np.sum(Y*incumbent, axis=1, keepdims=True)
            Y = incumbent + (Y - dot*incumbent)
            Y = _norm(Y)
            Y = _lse_jam(Y, beta=300.0, maxiter=150)
            v = _max_cos(Y)
            if v < inc_val + 3e-4:
                Yp = _lp_polish(Y, min(deadline-5, time.time()+2.5), radius0=1e-2, active=1e-3)
                vp = _max_cos(Yp)
                if vp < inc_val:
                    inc_val = vp; incumbent = Yp.copy()
                    continue
                elif v < inc_val:
                    inc_val = v; incumbent = Y.copy()
            if time.time() > deadline - 8:
                break
            # only few tries
            break
    except Exception:
        pass

    # Final polish
    try:
        remain = deadline - time.time()
        if remain > 3:
            Yf = _lp_polish(incumbent, deadline-0.5, radius0=1e-2, active=1e-3, cap=3000)
            vf = _max_cos(Yf)
            if vf < inc_val and np.all(np.isfinite(Yf)):
                incumbent = Yf
    except Exception:
        pass

    incumbent = _norm(incumbent)
    return incumbent
