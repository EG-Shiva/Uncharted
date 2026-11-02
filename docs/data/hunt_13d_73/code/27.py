import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def solve():
    t0 = time.time()
    DEADLINE = t0 + 165.0
    rng = np.random.default_rng(0)
    n, d = 73, 13

    def max_cos(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return float(G.max())

    # load incumbent and record base
    try:
        Xbest = np.load("best.npy").astype(float)
    except Exception:
        Xbest = None
    try:
        Xrec = np.load("record.npy").astype(float)
    except Exception:
        Xrec = None
    if Xrec is None:
        Xrec = Xbest if Xbest is not None else rng.normal(size=(n, d))
    Xrec = np.asarray(Xrec, dtype=float)
    if Xrec.shape != (n, d):
        tmp = rng.normal(size=(n, d))
        try:
            m0 = min(Xrec.shape[0], n); m1 = min(Xrec.shape[1], d)
            tmp[:m0, :m1] = Xrec[:m0, :m1]
        except Exception:
            pass
        Xrec = tmp
    Xrec = Xrec / np.linalg.norm(Xrec, axis=1, keepdims=True)
    if Xbest is None:
        Xbest = Xrec.copy()
    else:
        Xbest = np.asarray(Xbest, dtype=float)
        if Xbest.shape != (n, d):
            Xbest = Xrec.copy()
        else:
            Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    best_val = max_cos(Xbest)
    rec_val = max_cos(Xrec)
    # prefer better of the two as incumbent
    if rec_val < best_val:
        Xbest = Xrec.copy(); best_val = rec_val
    # record base for perturbations: published record
    base_rec = Xrec.copy()

    iu0 = np.triu_indices(n, 1)

    def F_val(X, tau):
        G = X @ X.T
        dots = G[iu0]
        ex = dots - tau
        ex[ex < 0] = 0.0
        return float((ex * ex).sum())

    def feas_opt(X0, tau, maxiter=250):
        Y0 = X0.ravel().copy()
        def fun(Yflat):
            Ym = Yflat.reshape(n, d)
            rn = np.linalg.norm(Ym, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Ym / rn
            G = X @ X.T
            dots = G[iu0]
            ex = dots - tau
            # loss
            pos = ex > 0
            loss = float((ex[pos] * ex[pos]).sum())
            # weights
            w = np.zeros_like(dots)
            w[pos] = 2.0 * ex[pos]
            W = np.zeros((n, n))
            W[iu0] = w
            W = W + W.T
            GX = W @ X
            dg = (GX * X).sum(axis=1, keepdims=True)
            Gt = GX - dg * X
            grad_Y = (Gt / rn).ravel()
            return loss, grad_Y.astype(float)
        try:
            res = minimize(fun, Y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-8, "maxcor": 10})
            Ym = res.x.reshape(n, d)
            Ym /= np.linalg.norm(Ym, axis=1, keepdims=True)
            return Ym, float(res.fun) if hasattr(res, 'fun') else F_val(Ym, tau)
        except Exception:
            return X0, F_val(X0, tau)

    def lp_polish(X, seconds, t_end):
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        best = max_cos(X)
        radius = 1e-3
        start = time.time()
        iu = np.triu_indices(n, 1)
        while time.time() - start < seconds and time.time() < t_end and radius > 1e-14:
            c = (X @ X.T)[iu]
            m = c.max()
            sel = c > m - max(20 * radius, 1e-9)
            I, J = iu[0][sel], iu[1][sel]
            p = len(I)
            if p == 0:
                break
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            try:
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
                obj = np.zeros(nv); obj[-1] = 1.0
                res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = X + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = max_cos(y)
            if v < best:
                X, best, radius = y, v, min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        return X

    def logsumexp_polish(X0, alpha=200.0, maxiter=120):
        Y = X0.ravel().copy()
        def fun(Yflat):
            Ym = Yflat.reshape(n, d)
            rn = np.linalg.norm(Ym, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Ym / rn
            G = X @ X.T
            dots = G[iu0]
            M = dots.max()
            e = np.exp(alpha * (dots - M))
            S = e.sum()
            loss = M + np.log(S) / alpha
            p = e / S
            W = np.zeros((n, n))
            W[iu0] = p
            W = W + W.T
            GX = W @ X
            dg = (GX * X).sum(axis=1, keepdims=True)
            Gt = GX - dg * X
            grad_Y = (Gt / rn).ravel()
            return float(loss), grad_Y.astype(float)
        try:
            res = minimize(fun, Y, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-13, "gtol": 1e-7, "maxcor": 10})
            Y = res.x
        except Exception:
            pass
        Ym = Y.reshape(n, d)
        Ym /= np.linalg.norm(Ym, axis=1, keepdims=True)
        return Ym

    taus = [0.2065, 0.2055, 0.2045]
    sigmas = [0.01, 0.03, 0.08]
    X_feas = None
    # track lowest-F endpoint for final different-stress seed
    seed_lowF = None
    seed_lowF_val = np.inf
    seed_lowF_tau = None

    for ti, tau in enumerate(taus):
        if time.time() > DEADLINE - 12.0:
            break
        starts = []
        if X_feas is not None:
            starts.append(("feas_cont", None))
        for s in sigmas:
            starts.append(("pert", s))
        for kind, s in starts:
            if time.time() > DEADLINE - 10.0:
                break
            try:
                if kind == "feas_cont":
                    X0 = X_feas.copy()
                else:
                    X0 = base_rec + float(s) * rng.normal(size=(n, d))
                    X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            except Exception:
                continue
            # run at most 250 L-BFGS iterations
            try:
                Xopt, Floss = feas_opt(X0, tau, maxiter=250)
            except Exception:
                continue
            # evaluate true maximum after every stage
            try:
                Floss = F_val(Xopt, tau)
                v = max_cos(Xopt)
            except Exception:
                continue
            # track lowest F endpoint
            if Floss < seed_lowF_val:
                seed_lowF_val = float(Floss)
                seed_lowF = Xopt.copy()
                seed_lowF_tau = tau
            if v < best_val - 1e-15:
                Xbest = Xopt.copy()
                best_val = float(v)
            # feasibility reached?
            if Floss < 1e-10:
                # true max at most tau -> new record, polish briefly with LP
                try:
                    if time.time() < DEADLINE - 6.0:
                        Xp = lp_polish(Xopt, seconds=2.5, t_end=DEADLINE - 4.0)
                        vp = max_cos(Xp)
                        if vp < best_val:
                            Xbest = Xp.copy(); best_val = float(vp)
                            Xopt = Xp.copy()
                        # evaluate
                        Floss2 = F_val(Xopt, tau)
                        if Floss2 < seed_lowF_val:
                            seed_lowF = Xopt.copy(); seed_lowF_val = float(Floss2)
                except Exception:
                    pass
                X_feas = Xopt.copy()
                # move to next lower tau
                break
            # early stop if time
            if time.time() > DEADLINE - 8.0:
                break
        # if feasible we continue to next tau automatically
        continue

    # if no feasibility to zero, use lowest-F endpoint as differently stressed seed for one short LogSumExp polish
    if time.time() < DEADLINE - 6.0 and seed_lowF is not None:
        try:
            # only do this if it is promising or as required seed
            Xa = logsumexp_polish(seed_lowF, alpha=180.0, maxiter=120)
            va = max_cos(Xa)
            if va < best_val:
                Xbest = Xa.copy(); best_val = float(va)
            # brief LP squeeze if time allows
            if time.time() < DEADLINE - 6.0:
                try:
                    Xb = lp_polish(Xa if va < max_cos(seed_lowF) else seed_lowF, seconds=2.0, t_end=DEADLINE - 2.0)
                    vb = max_cos(Xb)
                    if vb < best_val:
                        Xbest = Xb.copy(); best_val = float(vb)
                except Exception:
                    pass
        except Exception:
            pass

    # final safety LP on incumbent if time remains
    if time.time() < DEADLINE - 5.0:
        try:
            remain = min(6.0, (DEADLINE - 1.0) - time.time())
            if remain > 1.0:
                Xp = lp_polish(Xbest, seconds=remain, t_end=DEADLINE - 1.0)
                vp = max_cos(Xp)
                if vp < best_val:
                    Xbest = Xp; best_val = vp
        except Exception:
            pass

    Xbest /= np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
