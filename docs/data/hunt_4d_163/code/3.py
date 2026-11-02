import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _normed(x):
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-16
    return x / n


def _lp_polish_deadline(x, deadline, r0=1e-3):
    x = _normed(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu0 = np.triu_indices(n, 1)
    iuI0, iuJ0 = iu0
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iuI0, iuJ0]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) == 0:
            radius *= 0.5
            continue
        I = iuI0[sel]
        J = iuJ0[sel]
        csel = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _normed(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best


def _smooth_polish(x0, deadline, alphas=(50.0, 150.0), iters=(120, 120)):
    n, d = x0.shape
    iu = np.triu_indices(n, 1)
    iuI, iuJ = iu
    y = x0.ravel().copy()

    for alpha, maxiter in zip(alphas, iters):
        if time.time() >= deadline:
            break

        def fun(yflat, alpha=alpha):
            Y = yflat.reshape(n, d)
            rn = np.sqrt(np.sum(Y * Y, axis=1)) + 1e-16
            X = Y / rn[:, None]
            G = X @ X.T
            g = G[iuI, iuJ]
            m = np.max(g)
            e = np.exp(alpha * (g - m))
            S = np.sum(e)
            L = m + np.log(S) / alpha
            w = e / S
            Ws = np.zeros((n, n), dtype=np.float64)
            Ws[iuI, iuJ] = w
            Ws = Ws + Ws.T
            Gx = Ws @ X  # dL/dX
            proj = np.sum(Gx * X, axis=1, keepdims=True)
            GY = (Gx - X * proj) / rn[:, None]
            return float(L), GY.ravel()

        try:
            res = minimize(fun, y, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "maxfun": int(maxiter * 3),
                                    "ftol": 1e-14, "gtol": 1e-9})
            y = res.x
        except Exception:
            break
    X = _normed(y.reshape(n, d))
    return X


def solve():
    import time as _t
    t0 = _t.time()
    HARD_END = t0 + 172.0
    rs = np.random.RandomState(0)
    np.random.seed(0)

    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        cands.append(np.asarray(xb, dtype=np.float64))
    if xr is not None:
        cands.append(np.asarray(xr, dtype=np.float64))
    if len(cands) == 0:
        cands.append(_normed(rs.randn(163, 4)))
    best = None
    best_val = 1e100
    for c in cands:
        c = _normed(c)
        v = _max_cos(c)
        if v < best_val:
            best_val = v
            best = c.copy()

    # initial snap
    best, best_val = _lp_polish_deadline(best, min(HARD_END - 100, t0 + 8))

    n, d = best.shape
    it = 0
    # precompute contacts for biasing (updated when best improves)
    contacts = None
    def _contacts(X, mv):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        C = np.count_nonzero(G > mv - 1e-9, axis=1)
        return C

    contacts = _contacts(best, best_val)

    while _t.time() < HARD_END - 8:
        it += 1
        mode = it % 5
        trial = best.copy()
        if mode == 0:
            sigma = 0.008 + 0.004 * (rs.rand())
            trial = trial + sigma * rs.randn(n, d)
            trial = _normed(trial)
            sm_alphas = (60.0, 200.0)
            sm_iters = (120, 120)
            lp_time = 1.2
        elif mode == 1:
            # subset shake: pick 8-15 points, shake hard + global tiny shake
            k = 8 + int(rs.randint(8))
            idx = rs.choice(n, size=k, replace=False)
            # bias to high-contact points
            if rs.rand() < 0.7:
                order = np.argsort(-contacts)
                pool = order[:40]
                idx[:min(k, 5)] = pool[rs.choice(len(pool), size=min(k, 5), replace=False)]
            trial = trial + 0.002 * rs.randn(n, d)
            trial[idx] += 0.12 * rs.randn(len(idx), d)
            trial = _normed(trial)
            sm_alphas = (50.0, 160.0)
            sm_iters = (100, 100)
            lp_time = 1.2
        elif mode == 2:
            # reinsert 1-2 worst points at random
            G = trial @ trial.T
            np.fill_diagonal(G, -2.0)
            mv = G.max()
            # worst pair
            aj, bj = np.unravel_index(int(np.argmax(G)), G.shape)
            victims = [aj]
            if rs.rand() < 0.6:
                victims.append(bj)
            # sometimes victim = most overloaded point
            if rs.rand() < 0.5:
                victims[0] = int(np.argmax(contacts))
            for v in victims:
                if rs.rand() < 0.5:
                    trial[v] = rs.randn(d)
                else:
                    # place near void: far from others ~ near rattler/min-contact
                    mc = int(np.argmin(contacts))
                    trial[v] = trial[mc] + 0.3 * rs.randn(d)
            trial = _normed(trial)
            sm_alphas = (40.0, 140.0)
            sm_iters = (130, 130)
            lp_time = 1.4
        elif mode == 3:
            # separate worst pair + neighbours
            G = trial @ trial.T
            np.fill_diagonal(G, -2.0)
            mv = G.max()
            aj, bj = np.unravel_index(int(np.argmax(G)), G.shape)
            trial[aj] = trial[aj] + 0.05 * (trial[aj] - trial[bj]) + 0.01 * rs.randn(d)
            trial[bj] = trial[bj] + 0.05 * (trial[bj] - trial[aj]) + 0.01 * rs.randn(d)
            trial = trial + 0.005 * rs.randn(n, d)
            trial = _normed(trial)
            sm_alphas = (80.0, 250.0)
            sm_iters = (100, 100)
            lp_time = 1.2
        else:
            sigma = 0.02 + 0.02 * rs.rand()
            trial = trial + sigma * rs.randn(n, d)
            trial = _normed(trial)
            sm_alphas = (30.0, 100.0)
            sm_iters = (120, 100)
            lp_time = 1.0

        # smooth stage with inner deadline
        inner_dead = min(HARD_END - 8, _t.time() + 4.5)
        try:
            trial_s = _smooth_polish(trial, inner_dead, alphas=sm_alphas, iters=sm_iters)
        except Exception:
            trial_s = trial
        v_s = _max_cos(trial_s)
        # LP snap only if promising (within 0.02 of best) to save time
        if v_s < best_val + 0.02 and _t.time() < HARD_END - 8:
            trial_s, v_s = _lp_polish_deadline(trial_s, min(HARD_END - 8, _t.time() + lp_time))
        else:
            v_s = _max_cos(trial_s)
        if v_s < best_val - 1e-15:
            best = trial_s
            best_val = v_s
            contacts = _contacts(best, best_val)
        # occasional refresh contacts if stuck
        if it % 20 == 0:
            contacts = _contacts(best, best_val)

    # final polish with remaining time
    if _t.time() < HARD_END:
        best, best_val = _lp_polish_deadline(best, HARD_END)
    best = _normed(best)
    return best
