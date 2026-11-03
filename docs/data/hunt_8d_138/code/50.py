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

def _lse_jam(X0, beta, maxiter):
    from scipy.optimize import minimize
    n, d = X0.shape
    iu = np.triu_indices(n, 1)
    Y0 = _norm(X0.copy())
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
        res = minimize(fun, Y0.ravel(), jac=True, method='L-BFGS-B',
                       options={'maxiter': int(maxiter), 'ftol':1e-14, 'gtol':1e-6, 'maxcor':12})
        Y0 = _norm(res.x.reshape(n, d))
    except Exception:
        pass
    return _norm(Y0)

def _lp_polish(X, deadline, radius0=1e-2, active=1e-3, cap=2500):
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
        sel = c > (m - active)
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
        dx = res.x[:-1].reshape(n, d)
        y = _norm(X + dx)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius*1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _find_void(X, rng, n_sample=20000, refine=80):
    n, d = X.shape
    C = rng.standard_normal((n_sample, d))
    C = C / np.linalg.norm(C, axis=1, keepdims=True)
    # max cosine to existing set
    G = C @ X.T
    mx = G.max(axis=1)
    bi = int(np.argmin(mx))
    v = C[bi].copy()
    # local projected descent on max (smooth max via top-k average)
    step = 0.05
    for _ in range(refine):
        dots = X @ v
        # gradient of top few
        k = 8
        idx = np.argpartition(-dots, k)[:k]
        g = np.zeros(d)
        for j in idx:
            g += (X[j] - dots[j]*v)
        g /= k
        gn = np.linalg.norm(g)
        if gn < 1e-12:
            break
        g /= gn
        # move opposite to increase distance (decrease dots)
        v = v - step*g
        v = v/np.linalg.norm(v)
        step *= 0.97
    return v

def solve():
    rng = np.random.default_rng(0)
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

    # A: squeeze incumbent with LP
    try:
        y = _lp_polish(incumbent, min(deadline-90, time.time()+12.0), radius0=1e-2, active=1e-3)
        v = _max_cos(y)
        if v < inc_val:
            inc_val = v; incumbent = y.copy()
    except Exception:
        pass

    # B: floppy SVD + line search
    try:
        base = incumbent.copy()
        m0 = _max_cos(base)
        iu = np.triu_indices(n, 1)
        G0 = base @ base.T
        np.fill_diagonal(G0, -2.0)
        c_all = G0[iu]
        order = np.argsort(-c_all)
        P = 900
        sel_idx = order[:P]
        I = iu[0][sel_idx]; J = iu[1][sel_idx]
        Bs = _tangent_bases(base)
        k = d-1
        N = n*k
        Jm = np.empty((P, N))
        for r in range(P):
            i = int(I[r]); j = int(J[r])
            vi = Bs[i].T @ base[j]
            vj = Bs[j].T @ base[i]
            row = np.zeros(N)
            row[i*k:(i+1)*k] = vi
            row[j*k:(j+1)*k] = vj
            Jm[r] = row
        try:
            U, S, Vt = np.linalg.svd(Jm, full_matrices=False)
        except Exception:
            Vt = None
        if Vt is not None:
            ndir = min(12, Vt.shape[0])
            dirs = Vt[-ndir:]
            cands = []
            steps = np.array([3e-4,1e-3,3e-3,1e-2,3e-2,6e-2,1e-1])
            for di in range(ndir):
                if time.time() > deadline-70:
                    break
                z = dirs[di].reshape(n, k)
                Uf = np.empty((n, d))
                for i in range(n):
                    Uf[i] = Bs[i] @ z[i]
                nr = np.sqrt(np.sum(Uf*Uf)) + 1e-30
                Uf /= nr
                for sgn in (1.0, -1.0):
                    for st in steps:
                        t = sgn*float(st)
                        Y = _norm(base + t*Uf)
                        v = _max_cos(Y)
                        if v < m0 + 5e-4:
                            cands.append((v, Y.copy()))
            cands.sort(key=lambda x: x[0])
            # keep up to 4 diverse (by value spacing)
            kept = []
            for v, Y in cands:
                if len(kept) >= 4:
                    break
                if v > inc_val + 8e-4:
                    continue
                # diversity: check far from already kept
                ok = True
                for _, Y2 in kept:
                    if np.linalg.norm(Y-Y2) < 1e-3:
                        ok = False; break
                if ok:
                    kept.append((v, Y))
            if len(kept) == 0 and len(cands) > 0:
                kept = [(cands[0][0], cands[0][1])]
            for v, Y in kept:
                if time.time() > deadline-55:
                    break
                try:
                    Y2 = _lse_jam(Y, beta=260.0, maxiter=200)
                    Y2 = _lse_jam(Y2, beta=500.0, maxiter=120)
                    Y3 = _lp_polish(Y2, min(deadline-50, time.time()+4.0), radius0=1e-2, active=1e-3)
                    v3 = _max_cos(Y3)
                    if v3 < inc_val:
                        inc_val = v3; incumbent = Y3.copy()
                    else:
                        v2 = _max_cos(Y2)
                        if v2 < inc_val:
                            inc_val = v2; incumbent = Y2.copy()
                except Exception:
                    continue
    except Exception:
        pass

    # C: void reinsertion for low-degree points
    try:
        G = incumbent @ incumbent.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        # degrees at 1e-4 tolerance
        tol = 1e-4
        iu2 = np.triu_indices(n, 1)
        c = G[iu2]
        sel = c > (m - tol)
        deg = np.zeros(n, dtype=int)
        II, JJ = iu2[0][sel], iu2[1][sel]
        for a, b in zip(II, JJ):
            deg[a] += 1; deg[b] += 1
        order_pts = np.argsort(deg)  # lowest first
        n_try = 4
        for ti in range(n_try):
            if time.time() > deadline-30:
                break
            pi = int(order_pts[ti])
            # find void w.r.t. others
            Xrest = np.delete(incumbent, pi, axis=0)
            vv = _find_void(Xrest, rng, n_sample=12000, refine=60)
            Y = incumbent.copy()
            Y[pi] = vv
            # small jitter neighbours that were in contact
            Y = _norm(Y)
            Y = _lse_jam(Y, beta=280.0, maxiter=180)
            v = _max_cos(Y)
            if v < inc_val + 5e-4:
                Yp = _lp_polish(Y, min(deadline-25, time.time()+3.5), radius0=1e-2, active=1e-3)
                vp = _max_cos(Yp)
                if vp < inc_val:
                    inc_val = vp; incumbent = Yp.copy()
                elif v < inc_val:
                    inc_val = v; incumbent = Y.copy()
    except Exception:
        pass

    # D: drop-one-contact push-apart
    try:
        G = incumbent @ incumbent.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        iu3 = np.triu_indices(n, 1)
        c = G[iu3]
        topk = np.argsort(-c)[:6]
        for kk in range(len(topk)):
            if time.time() > deadline-12:
                break
            idx = int(topk[kk])
            di = int(iu3[0][idx]); dj = int(iu3[1][idx])
            try:
                Y = incumbent.copy()
                # push apart along tangent
                xi = Y[di].copy(); xj = Y[dj].copy()
                Y[di] = _norm((xi - 0.03*xj).reshape(1,-1))[0] + 0.005*rng.standard_normal(d))
                Y[dj] = _norm((xj - 0.03*xi).reshape(1,-1))[0] + 0.005*rng.standard_normal(d))
                Y = _norm(Y)
                Y = _lse_jam(Y, beta=300.0, maxiter=180)
                Y = _lse_jam(Y, beta=550.0, maxiter=100)
                v = _max_cos(Y)
                if v < inc_val + 3e-4:
                    Yp = _lp_polish(Y, min(deadline-8, time.time()+2.5), radius0=1e-2, active=1e-3)
                    vp = _max_cos(Yp)
                    if vp < inc_val:
                        inc_val = vp; incumbent = Yp.copy()
            except Exception:
                continue
    except Exception:
        pass

    # final polish
    try:
        remain = deadline - time.time()
        if remain > 4:
            Yf = _lp_polish(incumbent, deadline-0.5, radius0=1e-2, active=1e-3, cap=2500)
            vf = _max_cos(Yf)
            if vf < inc_val:
                inc_val = vf; incumbent = Yf
    except Exception:
        pass

    return _norm(incumbent)
