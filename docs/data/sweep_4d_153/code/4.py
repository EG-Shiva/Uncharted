import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while True:
        if time.time() - start > seconds:
            break
        if time.time() > deadline:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) == 0:
            break
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn < 1e-12] = 1e-12
        y = y / yn
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _contacts_per_point(x, thresh=1e-9):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    cnt = np.count_nonzero(G > m - thresh, axis=1)
    return cnt, m

def _smooth_opt(x_init, alpha, maxiter, deadline):
    n, d = x_init.shape
    Y0 = x_init.ravel().copy()
    def fun(Yf):
        if time.time() > deadline:
            # early abort by returning current value with zero grad would confuse;
            # just compute normally (fast) - deadline checked by caller
            pass
        Y = Yf.reshape(n, d)
        norms = np.linalg.norm(Y, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        X = Y / norms
        G = X @ X.T
        iu = np.triu_indices(n, 1)
        cij = G[iu]
        ac = alpha * cij
        mm = ac.max()
        w = np.exp(ac - mm)
        S = w.sum()
        E = (np.log(S) + mm) / alpha
        K = np.zeros((n, n))
        K[iu] = w / S
        K = K + K.T
        dEdX = K @ X
        dot = (dEdX * X).sum(axis=1, keepdims=True)
        dEdY = (dEdX - X * dot) / norms
        # tiny radial regularizer to keep norms ~1
        diff = (norms[:, 0] - 1.0)
        Ereg = 1e-4 * np.mean(diff ** 2)
        Etot = E + Ereg
        # grad of reg: 2e-4*(norm-1)/n * Y/norm
        coeff = (2e-4 / n) * diff / norms[:, 0]
        dEdY = dEdY + coeff[:, None] * Y
        return float(Etot), dEdY.ravel()
    try:
        res = minimize(fun, Y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-9, "maxfun": 10000})
        Y = res.x.reshape(n, d)
    except Exception:
        Y = Y0.reshape(n, d)
    Yn = np.linalg.norm(Y, axis=1, keepdims=True)
    Yn[Yn < 1e-12] = 1e-12
    return Y / Yn

def solve():
    t0 = time.time()
    DEADLINE = t0 + 168.0
    rng = np.random.default_rng(0)
    np.random.seed(0)
    # load start
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        if c is not None:
            try:
                a = np.array(c, dtype=float)
                if a.shape == (153, 4):
                    a = a / np.linalg.norm(a, axis=1, keepdims=True)
                    if np.all(np.isfinite(a)):
                        cands.append(( _max_cos(a), a))
            except Exception:
                pass
    if len(cands) == 0:
        a = rng.standard_normal((153, 4))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append((_max_cos(a), a))
    cands.sort(key=lambda t: t[0])
    best_score, best_x = cands[0]
    # quick polish of start if time allows
    remain = DEADLINE - time.time()
    if remain > 10:
        best_x, best_score = _lp_polish(best_x, min(6.0, remain - 4.0), DEADLINE - 2.0)
    n, d = best_x.shape
    cycle = 0
    # precompute for speed
    while time.time() < DEADLINE - 4.0:
        cycle += 1
        time_left = DEADLINE - time.time()
        if time_left < 5.0:
            break
        # choose kick
        mode = cycle % 6
        pert = best_x.copy()
        cnt, mval = _contacts_per_point(best_x, thresh=1e-9)
        ratt = np.where(cnt == 0)[0]
        if mode == 0:
            sigma = 0.015
            pert = pert + sigma * rng.standard_normal(pert.shape)
        elif mode == 1:
            sigma = 0.06
            pert = pert + sigma * rng.standard_normal(pert.shape)
        elif mode == 2:
            # kick overloaded cluster
            order = np.argsort(-cnt)
            k = 12
            idx = order[:k]
            pert[idx] = pert[idx] + 0.18 * rng.standard_normal((k, d))
        elif mode == 3:
            # random subset strong kick
            k = 15
            idx = rng.choice(n, size=k, replace=False)
            pert[idx] = pert[idx] + 0.30 * rng.standard_normal((k, d))
        elif mode == 4:
            # global medium + rattler randomize
            pert = pert + 0.03 * rng.standard_normal(pert.shape)
            if len(ratt) > 0:
                for r in ratt:
                    v = rng.standard_normal(d)
                    v /= np.linalg.norm(v)
                    pert[r] = v
            else:
                # randomize one point to hole search
                idx = int(rng.integers(0, n))
                # sample holes
                S = 8000
                cand = rng.standard_normal((S, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                dots = cand @ best_x.T
                # exclude self column? approximate: max over all
                md = dots.max(axis=1)
                bi = int(np.argmin(md))
                pert[idx] = cand[bi]
        else:
            # topology shake: random direction global + subset
            pert = pert + 0.10 * rng.standard_normal(pert.shape)
            k = 8
            idx = rng.choice(n, size=k, replace=False)
            pert[idx] = pert[idx] + 0.25 * rng.standard_normal((k, d))
        # always randomize rattlers a bit except mode 4 already did
        if mode != 4 and len(ratt) > 0 and (cycle % 2 == 0):
            for r in ratt:
                pert[r] = pert[r] + 0.5 * rng.standard_normal(d)
        pert /= np.linalg.norm(pert, axis=1, keepdims=True)
        # smooth optimization: two-stage alpha annealing
        # adapt alpha with cycle
        if mode in (2, 3, 5):
            a1, a2 = 50.0, 180.0
        else:
            a1, a2 = 80.0, 250.0
        # budget maxiter based on time left
        x_sm = _smooth_opt(pert, a1, 70, DEADLINE)
        if time.time() > DEADLINE - 4.0:
            v = _max_cos(x_sm)
            if v < best_score:
                best_x, best_score = x_sm, v
            break
        x_sm = _smooth_opt(x_sm, a2, 60, DEADLINE)
        # LP polish short
        lp_budget = 2.5
        if time_left < 12:
            lp_budget = 1.2
        x_pol, v_pol = _lp_polish(x_sm, lp_budget, DEADLINE - 1.0)
        # also try polishing perturbed directly without smoothing sometimes? no
        if v_pol < best_score - 1e-12:
            best_x, best_score = x_pol, v_pol
        elif v_pol < best_score:
            best_x, best_score = x_pol, v_pol
        # opportunistic: if no improvement, try pure LP on perturbed (cheap) every few cycles
        if cycle % 8 == 0 and time.time() < DEADLINE - 6.0:
            x2, v2 = _lp_polish(pert, 1.0, DEADLINE - 1.0)
            if v2 < best_score:
                best_x, best_score = x2, v2
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    # final safety: remove NaN
    best_x = np.where(np.isfinite(best_x), best_x, 0.0)
    nn = np.linalg.norm(best_x, axis=1, keepdims=True)
    nn[nn < 1e-12] = 1.0
    best_x = best_x / nn
    return best_x
