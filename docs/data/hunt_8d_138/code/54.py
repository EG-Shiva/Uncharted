import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-30
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _hinge_opt(X0, tau, maxiter, scale=1e6):
    from scipy.optimize import minimize
    n, d = X0.shape
    iu = np.triu_indices(n, 1)
    iu0, iu1 = iu
    Y = _norm(X0.copy())
    def fun(Yf):
        Ym = Yf.reshape(n, d)
        nr = np.sqrt(np.sum(Ym*Ym, axis=1, keepdims=True)) + 1e-30
        X = Ym / nr
        G = X @ X.T
        c = G[iu0, iu1]
        over = c - tau
        m = over > 0.0
        if np.any(m):
            o = over[m]
            f = float(np.sum(o*o)) * scale
            w = 2.0 * o * scale
            # build GX = W @ X efficiently
            Wrows = iu0[m]
            Wcols = iu1[m]
            GX = np.zeros((n, d))
            # GX[i] += w * X[j], GX[j] += w * X[i]
            np.add.at(GX, Wrows, w[:, None] * X[Wcols])
            np.add.at(GX, Wcols, w[:, None] * X[Wrows])
            dot = np.sum(GX * X, axis=1, keepdims=True)
            Gt = GX - dot * X
            GY = Gt / nr
        else:
            f = 0.0
            GY = np.zeros_like(Ym)
        return f, GY.ravel()
    try:
        res = minimize(fun, Y.ravel(), jac=True, method='L-BFGS-B',
                       options={'maxiter': int(maxiter), 'ftol': 1e-20, 'gtol': 1e-9, 'maxcor': 12})
        Y = _norm(res.x.reshape(n, d))
    except Exception:
        pass
    return _norm(Y)

def _soft_hinge_opt(X0, tau, beta, maxiter, scale=1e6):
    from scipy.optimize import minimize
    n, d = X0.shape
    iu = np.triu_indices(n, 1)
    iu0, iu1 = iu
    Y = _norm(X0.copy())
    def fun(Yf):
        Ym = Yf.reshape(n, d)
        nr = np.sqrt(np.sum(Ym*Ym, axis=1, keepdims=True)) + 1e-30
        X = Ym / nr
        G = X @ X.T
        c = G[iu0, iu1]
        z = beta * (c - tau)
        # softplus(z)/beta, stable
        # sp = log1p(exp(z))/beta, for z>20 ~ z/beta
        sp = np.empty_like(c)
        big = z > 30.0
        sp[big] = (z[big]) / beta
        zb = z[~big]
        sp[~big] = np.log1p(np.exp(zb)) / beta
        f = float(np.sum(sp*sp)) * scale
        # d sp/dc = sigmoid(z)
        sig = np.empty_like(c)
        sig[big] = 1.0
        sig[~big] = 1.0 / (1.0 + np.exp(-zb))
        coeff = 2.0 * sp * sig * scale
        GX = np.zeros((n, d))
        np.add.at(GX, iu0, coeff[:, None] * X[iu1])
        np.add.at(GX, iu1, coeff[:, None] * X[iu0])
        dot = np.sum(GX * X, axis=1, keepdims=True)
        Gt = GX - dot * X
        GY = Gt / nr
        return f, GY.ravel()
    try:
        res = minimize(fun, Y.ravel(), jac=True, method='L-BFGS-B',
                       options={'maxiter': int(maxiter), 'ftol': 1e-20, 'gtol': 1e-9, 'maxcor': 12})
        Y = _norm(res.x.reshape(n, d))
    except Exception:
        pass
    return _norm(Y)

def _lp_polish(X, deadline):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gfull = X @ X.T
        c = Gfull[iu]
        m = c.max()
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 2800:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:2800]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    rng = np.random.default_rng(7)
    t0 = time.time()
    deadline = t0 + 168.0
    n, d = 138, 8
    bestX = None
    bestV = 1e9
    recX = None
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = _norm(v)
                c = _max_cos(v)
                if fn == "record.npy":
                    recX = v.copy()
                if c < bestV:
                    bestV = c
                    bestX = v.copy()
        except Exception:
            pass
    if bestX is None:
        bestX = _norm(rng.standard_normal((n, d)))
        bestV = _max_cos(bestX)
    if recX is None:
        recX = bestX.copy()
    # feasibility ladder
    tau = bestV - 5e-06
    # if best is record ~0.4674149, tau ~0.4674099; will step down toward 0.4668
    step = 1.2e-05
    perts = [0.03, 0.05, 0.08, 0.12, 0.2, 0.035, 0.07, 0.15]
    it = 0
    # keep a chain point for feasible continuation
    chainX = bestX.copy()
    while time.time() < deadline - 28.0 and it < 80:
        it += 1
        kind = it % 3
        if kind == 0:
            # record start (or best) + tangent perturb
            s = perts[(it // 3) % len(perts)]
            base = recX if (it % 6 == 0) else bestX
            Xs = _norm(base + s * rng.standard_normal((n, d)))
        elif kind == 1:
            # continue from chain / best with small shake
            s = 0.02 + 0.03 * rng.random()
            Xs = _norm(chainX + s * rng.standard_normal((n, d)))
        else:
            # fresh random
            Xs = _norm(rng.standard_normal((n, d)))
        # choose hinge vs softplus alternating
        try:
            if it % 4 == 3:
                Xr = _soft_hinge_opt(Xs, tau, beta=1500.0, maxiter=250)
                # follow with hard hinge to squeeze
                Xr = _hinge_opt(Xr, tau, maxiter=200)
            else:
                Xr = _hinge_opt(Xs, tau, maxiter=300)
        except Exception:
            continue
        try:
            v = _max_cos(Xr)
        except Exception:
            continue
        if v < bestV - 1e-12:
            bestV = float(v)
            bestX = Xr.copy()
        if v < tau:
            # feasible: record, lower tau, continue from here
            chainX = Xr.copy()
            # set next tau just below achieved value
            tau = v - (8e-06 if it % 2 == 0 else 1.2e-05)
        else:
            # not feasible but if close, keep chain as best hinge result for diversity
            # occasionally relax tau upward if too aggressive and no progress
            if it % 12 == 0 and tau < bestV - 3e-05:
                tau = bestV - 5e-06
        # also if we improved best a lot, sync tau to stay below best
        if tau > bestV - 2e-06:
            tau = bestV - 5e-06
        # floor to avoid impossible
        if tau < 0.46650:
            tau = 0.46650
        if time.time() >= deadline - 28.0:
            break
    # final polish of best
    try:
        # one more hinge squeeze at best level to balance, then LP
        if time.time() < deadline - 22.0:
            try:
                Xr = _hinge_opt(bestX, bestV - 1e-07, maxiter=200)
                if _max_cos(Xr) < bestV:
                    bestX = Xr
                    bestV = _max_cos(Xr)
            except Exception:
                pass
        bestX = _lp_polish(bestX, deadline - 1.0)
        bestV = _max_cos(bestX)
    except Exception:
        pass
    # guaranteed fallback to published code
    try:
        rv = _max_cos(_norm(recX))
        if not np.isfinite(bestV) or bestV > rv:
            bestX = _norm(recX)
    except Exception:
        pass
    return _norm(bestX)
