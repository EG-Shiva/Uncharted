import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, seconds, t_end):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end and radius > 1e-14:
        Cfull = X @ X.T
        c = Cfull[iu0, iu1]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        if np.count_nonzero(sel) > 2500:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:2500]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I, J = iu0[sel], iu1[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = I[:, None] * d + np.arange(d)
        cJ = J[:, None] * d + np.arange(d)
        ct = np.full((p, 1), n * d)
        cols = np.concatenate([cI, cJ, ct], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        nrm = np.linalg.norm(y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        y /= nrm
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X, best

def solve():
    t0 = time.time()
    T_END = t0 + 165.0
    rng = np.random.default_rng(7)
    n, d = 73, 13

    # load incumbent / record
    Xrec = None
    best_file = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=np.float64)
            if v.shape == (n, d):
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                if np.all(np.isfinite(v)):
                    c = _max_cos(v)
                    if c < best_file:
                        best_file = c
                        Xrec = v.copy()
        except Exception:
            pass
    if Xrec is None:
        Xrec = rng.standard_normal(size=(n, d))
        Xrec /= np.linalg.norm(Xrec, axis=1, keepdims=True)
        best_file = _max_cos(Xrec)
    Xbest = Xrec.copy()
    best = float(best_file)
    cur_base = Xrec.copy()

    taus = [0.2065, 0.2055, 0.2045]
    sigmas = [0.01, 0.03, 0.08]

    def make_feas_fun(tau):
        def fun(Yflat):
            Y = Yflat.reshape(n, d)
            nr = np.sqrt(np.sum(Y * Y, axis=1))
            nr[nr < 1e-12] = 1e-12
            X = Y / nr[:, None]
            G = X @ X.T
            # exclude diagonal
            V = G - tau
            np.fill_diagonal(V, -1.0)
            M = np.maximum(V, 0.0)
            F = float(np.sum(np.triu(M * M, 1)))
            W = 2.0 * M
            # W symmetric already (G sym), diag 0
            GradX = W @ X
            s = np.sum(GradX * X, axis=1)
            Gt = (GradX - s[:, None] * X) / nr[:, None]
            return F, Gt.ravel()
        return fun

    lowestF_X = cur_base.copy()
    lowestF = 1e100
    any_feas = False
    feas_X = None

    for ti, tau in enumerate(taus):
        if time.time() > T_END - 25.0:
            break
        bestF_this = 1e100
        bestX_this = None
        success_this = False
        success_X = None
        for sigma in sigmas:
            if time.time() > T_END - 25.0:
                break
            X0 = cur_base + rng.standard_normal(size=(n, d)) * sigma
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            Y0 = X0.ravel().copy()
            fun = make_feas_fun(tau)
            try:
                res = minimize(fun, Y0, method="L-BFGS-B", jac=True,
                               options={"maxiter": 250, "maxfun": 2000, "ftol": 1e-20, "gtol": 1e-10})
                Yopt = res.x.reshape(n, d)
                nr = np.linalg.norm(Yopt, axis=1, keepdims=True)
                nr[nr == 0] = 1.0
                Xopt = Yopt / nr
            except Exception:
                continue
            if not np.all(np.isfinite(Xopt)):
                continue
            # true values
            G = Xopt @ Xopt.T
            np.fill_diagonal(G, -2.0)
            tru = float(G.max())
            V = G - tau
            V[V < 0] = 0.0
            # F from triu
            Fopt = float(np.sum(np.triu(V * V, 1)))
            # track lowest F
            if Fopt < bestF_this:
                bestF_this = Fopt
                bestX_this = Xopt.copy()
            if Fopt < lowestF:
                lowestF = Fopt
                lowestF_X = Xopt.copy()
            if tru < best:
                best = tru
                Xbest = Xopt.copy()
            if Fopt < 1e-10:
                success_this = True
                success_X = Xopt.copy()
                any_feas = True
                feas_X = Xopt.copy()
                break
            # early time check using true max evaluation
            if time.time() > T_END - 25.0:
                break
        if success_this:
            cur_base = success_X.copy()
            # brief LP polish on feasible record if time allows
            if time.time() < T_END - 20.0:
                try:
                    Xp, vp = _lp_polish(cur_base, seconds=2.0, t_end=T_END - 12.0)
                    if vp < best:
                        best = vp
                        Xbest = Xp.copy()
                        cur_base = Xp.copy()
                        feas_X = Xp.copy()
                except Exception:
                    pass
            continue
        else:
            # failed at this tau: keep lowest-F endpoint as stressed seed
            if bestX_this is not None:
                cur_base = bestX_this.copy()
                lowestF_X = bestX_this.copy()
            break

    # if feasible reached, already polished; else one short LogSumExp polish from lowest-F endpoint
    if time.time() < T_END - 14.0:
        # only do LSE if we have time and a seed
        seed = lowestF_X.copy() if lowestF_X is not None else Xbest.copy()
        s_val = 400.0
        def lse_fun(Yflat):
            Y = Yflat.reshape(n, d)
            nr = np.sqrt(np.sum(Y * Y, axis=1))
            nr[nr < 1e-12] = 1e-12
            X = Y / nr[:, None]
            G = X @ X.T
            iu0, iu1 = np.triu_indices(n, 1)
            c = G[iu0, iu1]
            m = float(c.max())
            e = np.exp(s_val * (c - m))
            se = float(e.sum())
            F = m + np.log(se) / s_val - np.log(float(len(c))) / s_val if se > 0 else m
            # weights
            w = e / se if se > 0 else np.ones_like(e) / len(e)
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W[iu1, iu0] = w
            GradX = W @ X
            s = np.sum(GradX * X, axis=1)
            Gt = (GradX - s[:, None] * X) / nr[:, None]
            return float(F), Gt.ravel()
        try:
            Y0 = (seed / np.linalg.norm(seed, axis=1, keepdims=True)).ravel().copy()
            res = minimize(lse_fun, Y0, method="L-BFGS-B", jac=True,
                           options={"maxiter": 120, "maxfun": 1000, "ftol": 1e-20, "gtol": 1e-10})
            Yopt = res.x.reshape(n, d)
            nr = np.linalg.norm(Yopt, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            Xopt = Yopt / nr
            if np.all(np.isfinite(Xopt)):
                v = _max_cos(Xopt)
                if v < best:
                    best = v
                    Xbest = Xopt.copy()
        except Exception:
            pass

    # final brief LP polish of incumbent
    if time.time() < T_END - 3.0:
        try:
            remain = T_END - time.time() - 1.0
            if remain > 1.5:
                Xp, vp = _lp_polish(Xbest, seconds=min(remain, 10.0), t_end=T_END - 0.5)
                if vp < best and np.all(np.isfinite(Xp)):
                    best = vp
                    Xbest = Xp.copy()
        except Exception:
            pass

    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    if not np.all(np.isfinite(Xbest)):
        Xbest = Xrec / np.linalg.norm(Xrec, axis=1, keepdims=True)
    return Xbest
