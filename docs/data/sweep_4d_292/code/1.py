import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _soft_anneal(x, steps, beta0, beta1, lr0, mu):
    n, d = x.shape
    v = np.zeros_like(x)
    # precompute log ratio
    lb = np.log(beta1 / beta0) if beta1 > beta0 else 0.0
    for t in range(steps):
        frac = t / max(1, steps - 1)
        beta = beta0 * np.exp(lb * frac)
        lr = lr0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        # weights, stable: exp(beta*(G-gmax)) <=1
        # clip very negative to zero implicitly by exp underflow
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-300
        A = (W @ x) / S
        dot = (A * x).sum(axis=1, keepdims=True)
        Gt = A - dot * x
        v = mu * v + Gt
        # keep velocity tangential to avoid radial build-up
        vd = (v * x).sum(axis=1, keepdims=True)
        v = v - vd * x
        x = x - lr * v
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(nrm, 1e-12, out=nrm)
        x /= nrm
    return x


def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    TIME_LIMIT = 172.0
    # load start
    try:
        xb = np.load("best.npy").astype(np.float64)
    except Exception:
        try:
            xb = np.load("record.npy").astype(np.float64)
        except Exception:
            rng0 = np.random.default_rng(0)
            xb = rng0.standard_normal((292, 4))
    if xb.shape != (292, 4):
        rng0 = np.random.default_rng(0)
        xb = rng0.standard_normal((292, 4))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best = _max_cos(xb)
    x_best = xb.copy()

    rng = np.random.default_rng(7)
    n, d = x_best.shape

    # reserve final polish ~32s
    explore_deadline = t0 + TIME_LIMIT - 32.0

    it = 0
    sigmas = [0.008, 0.02, 0.05, 0.12, 0.30]
    while time.time() < explore_deadline:
        it += 1
        # time left check for one anneal
        if time.time() + 0.4 > explore_deadline and it > 5:
            break
        X = x_best.copy()
        mode = it % 7
        if mode == 6 and it > 3:
            # occasional fresh random restart (different basin)
            R = rng.standard_normal((n, d))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            # mix with best to keep some structure half the time
            if rng.random() < 0.5:
                X = R
            else:
                # heavy shake of best
                X = X + 0.6 * rng.standard_normal((n, d))
                X /= np.linalg.norm(X, axis=1, keepdims=True)
        else:
            sigma = sigmas[rng.integers(len(sigmas))]
            X = X + sigma * rng.standard_normal((n, d))
            # kill-reinsert a few points into random positions (topology change)
            if rng.random() < 0.7:
                k = int(rng.integers(1, 9))
                idx = rng.choice(n, size=k, replace=False)
                R = rng.standard_normal((k, d))
                R /= np.linalg.norm(R, axis=1, keepdims=True)
                X[idx] = R
            X /= np.linalg.norm(X, axis=1, keepdims=True)

        # random anneal hyperparams for diversity (deterministic via rng)
        r = rng.random()
        if r < 0.35:
            steps, b0, b1, lr0, mu = 1200, 25.0, 350.0, 0.06, 0.85
        elif r < 0.65:
            steps, b0, b1, lr0, mu = 1800, 15.0, 250.0, 0.10, 0.88
        elif r < 0.85:
            steps, b0, b1, lr0, mu = 2500, 40.0, 500.0, 0.04, 0.80
        else:
            steps, b0, b1, lr0, mu = 900, 10.0, 150.0, 0.15, 0.90

        # shorten steps if little time left
        if time.time() + steps * 0.00025 > explore_deadline:
            steps = max(400, int((explore_deadline - time.time()) / 0.00025))
            if steps <= 100:
                break

        Xo = _soft_anneal(X, steps, b0, b1, lr0, mu)
        val = _max_cos(Xo)
        if val < best and np.all(np.isfinite(Xo)):
            best = val
            x_best = Xo.copy()

    # final polish of the best found
    time_left = TIME_LIMIT - (time.time() - t0) - 1.0
    time_left = max(5.0, min(38.0, time_left))
    try:
        x_best = _lp_polish(x_best, time_left)
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # safety: remove any non-finite
    if not np.all(np.isfinite(x_best)):
        x_best = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    return x_best
