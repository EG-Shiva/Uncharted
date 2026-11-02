import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def smooth_opt(X, steps, T0, T1, lr, iu0, iu1):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    # precompute log ratio
    logr = np.log(T1 / T0) if T0 != T1 else 0.0
    W = np.empty((n, n))
    for t in range(steps):
        frac = t / max(steps - 1, 1)
        T = T0 * np.exp(logr * frac)
        G = X @ X.T
        gu = G[iu0, iu1]
        mg = gu.max()
        E = np.exp((gu - mg) / T)
        S = E.sum()
        w = E / S
        W.fill(0.0)
        W[iu0, iu1] = w
        # symmetrize (diag stays 0)
        W = W + W.T  # creates new array; keep simple
        # need W symmetric for grad; after add, reuse variable
        grad = W @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        m = b1 * m + (1.0 - b1) * gt
        v = b2 * v + (1.0 - b2) * gt * gt
        c1 = 1.0 - b1 ** (t + 1)
        c2 = 1.0 - b2 ** (t + 1)
        mhat = m / c1
        vhat = v / c2
        step = lr * mhat / (np.sqrt(vhat) + eps)
        # limit max row step to avoid blowup
        # simple global clip
        # sn = np.linalg.norm(step, axis=1, keepdims=True).max()
        X = X - step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        # need fresh W buffer next iter
        W = np.empty((n, n))
    return X

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        mcur = c.max()
        sel = c > mcur - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
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
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    np.random.seed(0)
    t0 = time.time()
    TIME_LIMIT = 172.0
    # load best
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = np.load("record.npy")
    xb = np.asarray(xb, dtype=float)
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = xb.shape
    assert xb.shape == (76, 15)
    best_val = max_cos(xb)
    x_best = xb.copy()
    iu0, iu1 = np.triu_indices(n, 1)

    # quick evaluation helper
    def eval_x(X):
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        return max_cos(X), X

    # Phase 1: stochastic basin search with fast Adam smoothing
    # budget ~ 135s, leave rest for LP
    search_budget = 135.0
    it = 0
    # first try direct smoothing of record without perturbation (different basin via annealing)
    # we do it inside loop
    while time.time() - t0 < search_budget:
        it += 1
        time_left = search_budget - (time.time() - t0)
        if time_left < 1.0:
            break
        # choose perturbation
        if it == 1:
            Xp = x_best.copy()
            steps, T0, T1, lr = 500, 0.02, 0.002, 0.008
        elif it % 7 == 0:
            # big hop: replace 2 random points uniform
            Xp = x_best.copy()
            k = 2
            idx = np.random.choice(n, size=k, replace=False)
            R = np.random.randn(k, d)
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            Xp[idx] = R
            steps, T0, T1, lr = 600, 0.025, 0.0015, 0.008
        elif it % 5 == 0:
            # shake all lightly
            Xp = x_best + 0.03 * np.random.randn(n, d)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            steps, T0, T1, lr = 400, 0.015, 0.0015, 0.006
        else:
            # subset shake
            k = int(np.random.randint(3, 13))
            sigma = float(np.random.choice([0.05, 0.1, 0.2, 0.35]))
            Xp = x_best.copy()
            idx = np.random.choice(n, size=k, replace=False)
            Xp[idx] += sigma * np.random.randn(k, d)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            if sigma > 0.15:
                steps, T0, T1, lr = 700, 0.03, 0.002, 0.009
            else:
                steps, T0, T1, lr = 450, 0.015, 0.0012, 0.006
        # adapt steps to time left (rough 0.05s per 500 steps)
        if time_left < 5 and steps > 300:
            steps = 300
        try:
            Xo = smooth_opt(Xp, steps, T0, T1, lr, iu0, iu1)
        except Exception:
            continue
        v, Xo = eval_x(Xo)
        if v < best_val - 1e-12:
            best_val = v
            x_best = Xo.copy()
            # intensify around new best immediately with cooler refine
            try:
                Xo2 = smooth_opt(x_best.copy(), 300, 0.008, 0.0008, 0.004, iu0, iu1)
                v2, Xo2 = eval_x(Xo2)
                if v2 < best_val:
                    best_val = v2
                    x_best = Xo2
            except Exception:
                pass
        # early stop if well below record? continue to push further
        if time.time() - t0 > search_budget:
            break

    # Phase 2: short LP polish on best
    remain = TIME_LIMIT - (time.time() - t0) - 3.0
    if remain > 5.0:
        lp_time = min(remain, 28.0)
        try:
            x_pol = lp_polish(x_best.copy(), lp_time)
            vpol = max_cos(x_pol)
            if vpol < best_val:
                best_val = vpol
                x_best = x_pol
        except Exception:
            pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
