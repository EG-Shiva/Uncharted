import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-300)

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _per_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _tangent_perturb(X, scale, rng):
    n, d = X.shape
    R = rng.standard_normal((n, d))
    R = R - np.sum(R * X, axis=1, keepdims=True) * X
    R /= (np.linalg.norm(R, axis=1, keepdims=True) + 1e-300)
    Y = X + scale * R
    return _norm(Y)

def _run_full(X0, beta, maxiter):
    n, d = X0.shape
    iu = np.triu_indices(n, 1)
    def fg(yf):
        U = yf.reshape(n, d)
        r = np.linalg.norm(U, axis=1, keepdims=True)
        r = np.maximum(r, 1e-12)
        X = U / r
        G = X @ X.T
        g = G[iu]
        mx = float(g.max())
        e = np.exp(beta * (g - mx))
        S = float(e.sum()) + 1e-300
        F = mx + np.log(S) / beta
        w = e / S
        P = np.zeros((n, n))
        P[iu] = w
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GU = (GX - dot * X) / r
        return F, GU.ravel()
    y0 = X0.ravel().copy()
    try:
        res = minimize(fg, y0, jac=True, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-7, 'maxfun': maxiter*2, 'maxcor': 20})
        Y = res.x.reshape(n, d)
        return _norm(Y)
    except Exception:
        return X0.copy()

def _run_subset(S0, Rmat, beta, maxiter):
    k, d = S0.shape
    m = Rmat.shape[0]
    iu = np.triu_indices(k, 1) if k > 1 else (np.array([], dtype=int), np.array([], dtype=int))
    p1 = len(iu[0])
    def fg(yf):
        U = yf.reshape(k, d)
        r = np.linalg.norm(U, axis=1, keepdims=True)
        r = np.maximum(r, 1e-12)
        XS = U / r
        if p1 > 0:
            Gss = XS @ XS.T
            g1 = Gss[iu]
        else:
            g1 = np.zeros(0)
        Gsr = XS @ Rmat.T
        g2 = Gsr.ravel()
        if p1 > 0:
            g = np.concatenate([g1, g2])
        else:
            g = g2
        mx = float(g.max())
        e = np.exp(beta * (g - mx))
        S = float(e.sum()) + 1e-300
        F = mx + np.log(S) / beta
        w = e / S
        if p1 > 0:
            w1 = w[:p1]
            w2 = w[p1:].reshape(k, m)
            P = np.zeros((k, k))
            P[iu] = w1
            P = P + P.T
            t1 = P @ XS
        else:
            w2 = w.reshape(k, m)
            t1 = 0.0
        t2 = w2 @ Rmat
        GX = t1 + t2 if p1 > 0 else t2
        dot = np.sum(GX * XS, axis=1, keepdims=True)
        GU = (GX - dot * XS) / r
        return F, GU.ravel()
    y0 = S0.ravel().copy()
    try:
        res = minimize(fg, y0, jac=True, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-7, 'maxfun': maxiter*2, 'maxcor': 20})
        Y = res.x.reshape(k, d)
        return _norm(Y)
    except Exception:
        return S0.copy()

def _opt_single(Xr, y_start, beta, maxiter=80):
    def fg(z):
        r = max(float(np.linalg.norm(z)), 1e-12)
        y = z / r
        s = Xr @ y
        mx = float(s.max())
        e = np.exp(beta * (s - mx))
        S = float(e.sum()) + 1e-300
        F = mx + np.log(S) / beta
        w = e / S
        gy = Xr.T @ w
        gz = (gy - float(gy @ y) * y) / r
        return F, gz
    try:
        res = minimize(fg, y_start.copy(), jac=True, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-9, 'maxfun': maxiter*2})
        z = res.x
        return z / (np.linalg.norm(z) + 1e-300)
    except Exception:
        return y_start / (np.linalg.norm(y_start) + 1e-300)

def _best_hole(Xrest, rng, ncand=25000):
    d = Xrest.shape[1]
    cand = rng.standard_normal((ncand, d))
    cand /= (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-300)
    scores = (cand @ Xrest.T).max(axis=1)
    return cand[int(np.argmin(scores))].copy()

def _lp_polish(X, deadline, r0=6e-4, cap_p=1500):
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline - 0.25 and radius > 1e-14:
        c = (X @ X.T)[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        csel = c[sel]
        if p > cap_p:
            od = np.argsort(-csel)[:cap_p]
            I, J = I[od], J[od]
            csel = csel[od]
            p = cap_p
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        step = res.x[:-1].reshape(n, d)
        improved = False
        for sh in (1.0, 0.3, 0.1):
            y = X + step * sh
            y = _norm(y)
            v = _max_cos(y)
            if v < best - 1e-15:
                X, best = y, v
                radius = min(radius * 1.5, 0.02)
                improved = True
                break
        if not improved:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    T_END = t0 + 168.0
    n, d = 68, 12
    rng = np.random.default_rng(42)
    bestX = None
    best_score = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (n, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    s = _max_cos(a)
                    if s < best_score:
                        best_score = s
                        bestX = a.copy()
        except Exception:
            pass
    if bestX is None:
        bestX = _norm(rng.standard_normal((n, d)))
        best_score = _max_cos(bestX)
    # initial squeeze
    try:
        bestX = _lp_polish(bestX, time.time() + 6.0, r0=8e-4)
        best_score = _max_cos(bestX)
    except Exception:
        pass
    cur = bestX.copy()
    cur_score = best_score
    it = 0
    ks = [4, 6, 3, 8, 5, 2, 7]
    while time.time() < T_END - 15.0:
        it += 1
        k = ks[it % len(ks)]
        pm = _per_max(cur)
        order = np.argsort(-pm)
        pool = order[:22]
        chosen = list(order[:min(2, k)])
        perm = rng.permutation(pool)
        for pp in perm:
            pp = int(pp)
            if pp not in chosen and len(chosen) < k:
                chosen.append(pp)
            if len(chosen) >= k:
                break
        fixed_mask = np.ones(n, dtype=bool)
        fixed_mask[np.array(chosen, dtype=int)] = False
        newX = cur.copy()
        cur_fixed = cur[fixed_mask].copy()
        rorder = rng.permutation(np.array(chosen, dtype=int))
        for idx in rorder:
            idx = int(idx)
            hole = _best_hole(cur_fixed, rng, ncand=22000)
            hole2 = _opt_single(cur_fixed, hole, beta=400.0, maxiter=60)
            m1 = float((cur_fixed @ hole).max())
            m2 = float((cur_fixed @ hole2).max())
            h = hole2 if m2 < m1 else hole
            newX[idx] = h
            cur_fixed = np.vstack([cur_fixed, h[None, :]])
        S0 = newX[np.array(chosen, dtype=int)].copy()
        Rmat = newX[fixed_mask].copy()
        # joint subset optimization with two betas
        try:
            Sopt = _run_subset(S0, Rmat, beta=200.0, maxiter=120)
            Sopt = _run_subset(Sopt, Rmat, beta=500.0, maxiter=120)
        except Exception:
            Sopt = S0
        newX[np.array(chosen, dtype=int)] = Sopt
        try:
            newX = _run_full(newX, beta=180.0, maxiter=110)
            newX = _run_full(newX, beta=600.0, maxiter=110)
        except Exception:
            pass
        sc = _max_cos(newX)
        if sc < best_score + 0.0012 and time.time() < T_END - 16.5:
            try:
                newXp = _lp_polish(newX, time.time() + 1.1, r0=5e-4, cap_p=1300)
                scp = _max_cos(newXp)
                if scp < sc:
                    newX, sc = newXp, scp
            except Exception:
                pass
        if sc < best_score - 1e-12:
            best_score = sc
            bestX = newX.copy()
            cur, cur_score = newX.copy(), sc
        elif sc < cur_score:
            cur, cur_score = newX.copy(), sc
        else:
            Temp = 0.0008
            if rng.random() < np.exp(-(sc - cur_score) / Temp):
                cur, cur_score = newX.copy(), sc
            if it % 6 == 0 and time.time() < T_END - 17.0:
                try:
                    kick = _tangent_perturb(bestX, 0.05, rng)
                    kick = _run_full(kick, beta=350.0, maxiter=100)
                    sk = _max_cos(kick)
                    if sk < best_score - 1e-12:
                        best_score, bestX = sk, kick.copy()
                    if sk < cur_score:
                        cur, cur_score = kick.copy(), sk
                except Exception:
                    pass
    # final thorough polish
    try:
        remain = T_END - time.time()
        if remain > 3.0:
            bestX = _run_full(bestX, beta=700.0, maxiter=200)
            bestX = _lp_polish(bestX, T_END - 0.5, r0=6e-4, cap_p=2000)
    except Exception:
        pass
    return _norm(bestX)
