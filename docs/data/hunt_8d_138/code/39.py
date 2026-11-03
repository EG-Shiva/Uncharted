import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)

def _hinge_opt(X0, tau, maxiter, seed_jit=0.0, rng=None):
    from scipy.optimize import minimize
    n, d = X0.shape
    iu0 = np.triu_indices(n, 1)
    Y0 = X0.copy()
    if seed_jit > 0 and rng is not None:
        Y0 = Y0 + seed_jit * rng.standard_normal(size=(n, d))
        Y0 = _norm(Y0)
    y0 = Y0.ravel()

    def fun(yf):
        Ym = yf.reshape(n, d)
        r = np.sqrt(np.sum(Ym * Ym, axis=1, keepdims=True)) + 1e-30
        X = Ym / r
        G = X @ X.T
        c = G[iu0]
        v = c - tau
        pos = v > 0
        viol = np.where(pos, v, 0.0)
        f = float(np.sum(viol * viol))
        # radial penalty to keep r~1 (removes scale invariance)
        rn = r.ravel()
        pen = 1e-4 * np.sum((rn - 1.0) ** 2)
        f = f + pen
        # gradient w.r.t X
        # V symmetric with 2*viol on violating pairs
        w = 2.0 * viol  # zero where not pos
        # build Grad_X = V @ X without dense V
        # use scatter: Grad_X[i] += sum_j w_{ij} X_j
        Gx = np.zeros_like(X)
        # vectorized via dense W small (138x138) is fine
        W = np.zeros((n, n))
        W[iu0] = w
        W = W + W.T
        Gx = W @ X
        dot = np.sum(Gx * X, axis=1, keepdims=True)
        Gt = (Gx - dot * X) / r
        # add penalty grad: 2e-4*(r-1)*Ym/r
        Gt = Gt + (2e-4 * (rn - 1.0) / (rn + 1e-30))[:, None] * Ym
        return f, Gt.ravel()

    try:
        res = minimize(fun, y0, jac=True, method='L-BFGS-B',
                       options={'maxiter': int(maxiter), 'ftol': 1e-14,
                                'gtol': 1e-6, 'maxcor': 12})
        Y = res.x.reshape(n, d)
    except Exception:
        Y = y0.reshape(n, d)
    X = _norm(Y)
    return X

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
        if cnt > 2500:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:2500]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
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
    import time
    rng = np.random.default_rng(0)
    t0 = time.time()
    deadline = t0 + 168.0
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
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val

    ladder = [0.46735, 0.46720, 0.46700, 0.46680, 0.46650]
    # drop taus already above best (already feasible)
    # keep them in order; skip if best already below
    sigmas = [0.0, 0.03, 0.05, 0.08, 0.12, 0.20]

    for tau in ladder:
        if time.time() > deadline - 20.0:
            break
        if best_val < tau:
            # already feasible, start next from best
            cur = best.copy()
            cur_val = best_val
            continue
        # attempts for this tau
        for ai in range(6):
            if time.time() > deadline - 20.0:
                break
            if ai == 0:
                X0 = cur.copy()
                mi = 500
            elif ai == 5:
                # fresh random code
                X0 = _norm(rng.standard_normal((n, d)))
                mi = 600
            else:
                sigma = sigmas[ai % len(sigmas)]
                if ai == 1 and cur_val < 1e8:
                    X0 = best.copy()
                else:
                    X0 = best.copy()
                X0 = X0 + sigma * rng.standard_normal(size=(n, d))
                X0 = _norm(X0)
                mi = 400
            # adapt tau if best improved a lot: use min(tau, best_val-2e-05)
            # keep original tau to force push, but if best far above tau
            # use intermediate to stabilise
            eff_tau = tau
            Xo = _hinge_opt(X0, eff_tau, maxiter=mi)
            v = _max_cos(Xo)
            if v < best_val - 1e-12:
                best_val = v
                best = Xo.copy()
            if v < eff_tau:
                cur = Xo.copy()
                cur_val = v
                break
            # even if not feasible, allow cur to follow improved region
            # keep cur as best of this tau attempts for next start
            if v < cur_val:
                cur = Xo.copy()
                cur_val = v
        # small adaptive retry: if we improved but not reached tau,
        # try one more push with tau = (best_val+tau)/2 implicitly by continuing;
        # ladder next is lower so it will keep pushing.
    # extra squeezing phase: tau just below best
    extra_rounds = 0
    while time.time() < deadline - 18.0 and extra_rounds < 6:
        extra_rounds += 1
        tau2 = best_val - 2.5e-05
        # two starts: from best and perturbed
        for ai in range(3):
            if time.time() > deadline - 18.0:
                break
            if ai == 0:
                X0 = best.copy()
            elif ai == 1:
                X0 = _norm(best + 0.04 * rng.standard_normal(size=(n, d)))
            else:
                X0 = _norm(rng.standard_normal((n, d)))
            Xo = _hinge_opt(X0, tau2, maxiter=400 if ai < 2 else 500)
            v = _max_cos(Xo)
            if v < best_val - 1e-12:
                best_val = v
                best = Xo.copy()
                break
        else:
            break
        # if no improvement, stop
        if best_val <= tau2:
            continue
        else:
            # improved but not to tau2 -> reduce step
            if extra_rounds >= 3:
                break

    # finish with trust-region LP polish
    try:
        best = _lp_polish(best, min(deadline, time.time() + 12.0))
        best_val = _max_cos(best)
    except Exception:
        pass
    best = _norm(best)
    return best
