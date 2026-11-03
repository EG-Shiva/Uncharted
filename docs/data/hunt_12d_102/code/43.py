import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _top_pairs(X, K):
    n = X.shape[0]
    G = X @ X.T
    iu = np.triu_indices(n, 1)
    c = G[iu]
    K = min(K, c.shape[0])
    # largest c = most violated
    part = np.argpartition(-c, K - 1)[:K]
    # sort for determinism
    order = part[np.argsort(-c[part])]
    I = iu[0][order]
    J = iu[1][order]
    ck = c[order]
    return I, J, ck, float(np.max(c))

def _feasibility_target(X0, t_target, incumbent, t_deadline, rng, K=320, max_iter=60):
    X = X0.copy()
    n, d = X.shape
    N = n * d
    G0 = X @ X.T
    iu_all = np.triu_indices(n, 1)
    # sum violation baseline
    def sum_viol(G):
        v = G[iu_all] - t_target
        v[v < 0] = 0
        return float(np.sum(v))
    Gcur = G0
    Vold = sum_viol(Gcur)
    maxold = float(np.max(Gcur[iu_all]))
    r = 0.006
    rmax = 0.02
    rmin = 1e-7
    stall = 0
    for it in range(max_iter):
        if time.time() > t_deadline:
            break
        # check actual feasibility below incumbent
        cur_max = _max_cos(X)
        if cur_max < incumbent - 1e-12:
            return X, True, cur_max
        if cur_max <= t_target:
            return X, True, cur_max
        I, J, ck, mx = _top_pairs(X, K)
        p = len(I)
        # Build LP: min sum s s.t. <u_i,x_j>+<x_i,u_j> - s <= t-ck, <x_i,u_i>=0, |u|<=r, s>=0
        b_ub = (t_target - ck).astype(float)
        # sparse A_ub: p x (N+p)
        rows = np.repeat(np.arange(p), 2 * d + 1)
        # cols for u part + slack part
        cI = (I[:, None] * d + np.arange(d)).ravel()
        cJ = (J[:, None] * d + np.arange(d)).ravel()
        cS = (N + np.arange(p)).repeat(1)
        # need interleave per row: [u_I block, u_J block, slack]
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        cols[0::(2 * d + 1)] = 0  # placeholder
        # build row by row vectorized:
        # create cols array in row-major order: for each k: Iblock, Jblock, slack
        base_I = (I[:, None] * d + np.arange(d))
        base_J = (J[:, None] * d + np.arange(d))
        cols = np.concatenate([base_I, base_J, (N + np.arange(p))[:, None]], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        # rows already repeat
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, N + p)).tocsr()
        except Exception:
            r *= 0.5
            continue
        # A_eq: n x (N+p)
        try:
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(N))), shape=(n, N + p)).tocsr()
        except Exception:
            r *= 0.5
            continue
        obj = np.zeros(N + p)
        obj[N:] = 1.0
        bounds = [(-r, r)] * N + [(0.0, None)] * p
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bounds, method="highs")
        except Exception:
            r *= 0.5
            if r < rmin:
                break
            continue
        if res.status != 0:
            r *= 0.5
            stall += 1
            if stall >= 4 and r < 2e-6:
                # small random kick to change topology
                sig = 1.5e-4 + rng.rand() * 5e-4
                X = X + sig * rng.randn(n, d)
                X /= np.linalg.norm(X, axis=1, keepdims=True)
                Vold = sum_viol(X @ X.T)
                r = 0.003
                stall = 0
            if r < rmin:
                break
            continue
        U = res.x[:N].reshape(n, d)
        # predicted slack sum
        pred = float(res.fun) if res.fun is not None else 0.0
        Y = X + U
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        Y /= nr
        GY = Y @ Y.T
        Vnew = sum_viol(GY)
        maxnew = float(np.max(GY[iu_all]))
        # progress if sum violation or max decreased
        if Vnew < Vold * (1.0 - 1e-9) or maxnew < maxold - 1e-12:
            X = Y
            Vold = Vnew
            maxold = maxnew
            # increase radius after progress
            r = min(r * 1.35, rmax)
            stall = 0
            # early exit if actually below incumbent
            if maxnew < incumbent - 1e-12:
                return X, True, maxnew
        else:
            # no progress: shrink
            r *= 0.5
            stall += 1
            if pred < 1e-12:
                # linear says feasible but nonlinear stalls -> need smaller r, keep X
                pass
            else:
                # linear infeasible, shrink; occasional kick
                if stall >= 6:
                    sig = r * 0.5 + 1e-4
                    Xt = X + sig * rng.randn(n, d)
                    Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
                    # accept kick only if not much worse
                    mk = _max_cos(Xt)
                    if mk < maxold + 5e-4:
                        X = Xt
                        Vold = sum_viol(X @ X.T)
                        maxold = _max_cos(X)
                    r = max(r, 1e-4)
                    stall = 0
            if r < rmin:
                break
    cur_max = _max_cos(X)
    return X, (cur_max < incumbent - 1e-12), cur_max

def _ordinary_polish(x, t_deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < t_deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    DEADLINE = t0 + 165.0
    rng = np.random.RandomState(7)
    X0 = None
    R0 = None
    try:
        X0 = np.load("best.npy")
    except Exception:
        X0 = None
    try:
        R0 = np.load("record.npy")
    except Exception:
        R0 = None
    bestX = None
    bestV = np.inf
    for c in (X0, R0):
        if c is None:
            continue
        try:
            c = np.asarray(c, dtype=np.float64)
            if c.shape != (102, 12):
                continue
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            if not np.all(np.isfinite(c)):
                continue
            v = _max_cos(c)
            if v < bestV:
                bestV = v
                bestX = c.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.randn(102, 12)
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)
    incumbent = float(bestV)
    Xcur = bestX.copy()
    # target-feasibility phase: reserve ~25s for final polish
    FEAS_END = t0 + 140.0
    deltas = [1e-6, 3e-6, 1e-5, 3e-5]
    # cycle through deltas; after success restart with smallest delta
    attempt = 0
    while time.time() < min(FEAS_END, DEADLINE - 25.0):
        delta = deltas[attempt % len(deltas)]
        t_target = incumbent - delta
        # start point: incumbent best + tiny jitter for larger deltas to break symmetry
        if delta >= 1e-5:
            sig = 2e-4
            Xs = Xcur + sig * rng.randn(102, 12)
            Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
        else:
            Xs = Xcur
        # K adapts: 200-500
        K = 280 + (attempt % 3) * 80  # 280,360,440
        t_lim = min(FEAS_END, DEADLINE - 25.0)
        try:
            Xn, ok, vmax = _feasibility_target(Xs, t_target, incumbent, t_lim, rng, K=K, max_iter=45)
        except Exception:
            attempt += 1
            continue
        if ok:
            # found below incumbent
            Xcur = Xn
            incumbent = float(vmax)
            bestX = Xn.copy()
            bestV = incumbent
            attempt = 0  # retry smallest delta from new incumbent
            # quick check: if very good, continue
            if time.time() >= min(FEAS_END, DEADLINE - 25.0):
                break
            continue
        else:
            # no success; if Xn slightly better than Xcur keep it
            if vmax < _max_cos(Xcur) - 1e-13:
                Xcur = Xn.copy()
                incumbent = float(vmax)
                bestX = Xn.copy()
                bestV = incumbent
                attempt = 0
                continue
        attempt += 1
        if attempt > 12:
            break
    # final ordinary LP polish under strict deadline
    if time.time() < DEADLINE - 3.0:
        try:
            Xp = _ordinary_polish(bestX, DEADLINE - 2.0)
            vp = _max_cos(Xp)
            if vp < bestV and np.all(np.isfinite(Xp)):
                bestX = Xp
                bestV = vp
        except Exception:
            pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
