import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def soft_polish(x_init, k0, k1, steps, eta0, eta1, mu):
    x = x_init.copy()
    n, d = x.shape
    V = np.zeros_like(x)
    for s in range(steps):
        frac = s / (steps - 1) if steps > 1 else 0.0
        k = k0 + (k1 - k0) * frac
        eta = eta0 + (eta1 - eta0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        # weights, max == 1
        W = np.exp(k * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True)
        A = (W @ x) / np.maximum(S, 1e-12)
        dot = np.sum(A * x, axis=1, keepdims=True)
        At = A - dot * x
        if s == 0:
            V = At
        else:
            V = mu * V + (1.0 - mu) * At
        x = x - eta * V
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        x /= nrm
    return x


def perturb_global(x, sigma, rng):
    noise = rng.standard_normal(size=x.shape)
    dot = np.sum(noise * x, axis=1, keepdims=True)
    noise -= dot * x
    y = x + sigma * noise
    y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
    return y


def perturb_local(x, sigma, rng):
    n = x.shape[0]
    noise = rng.standard_normal(size=x.shape)
    dot = np.sum(noise * x, axis=1, keepdims=True)
    noise -= dot * x
    center = int(rng.integers(n))
    sims = x @ x[center]
    idx = np.argpartition(-sims, 16)[:16]
    w = np.full(n, 0.2)
    w[idx] = 1.0
    y = x + sigma * noise * w[:, None]
    y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
    return y


def lp_polish_deadline(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() + 1.0 > deadline:
            break
        Gfull = x @ x.T
        c = Gfull[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
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
        y = x + res.x[:-1].reshape(n, d)
        y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    np.random.seed(0)
    rng = np.random.default_rng(0)
    t0 = time.time()
    deadline_final = t0 + 172.0
    deadline_explore = t0 + 118.0

    # load start
    best_x = None
    best_v = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (292, 4):
                a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
                if np.all(np.isfinite(a)):
                    v = max_cos(a)
                    if v < best_v:
                        best_v = v
                        best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = rng.standard_normal(size=(292, 4))
        best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
        best_v = max_cos(best_x)

    cur_x = best_x.copy()
    cur_v = best_v
    T = 0.0006

    sigmas = [0.006, 0.012, 0.022, 0.035, 0.009, 0.018]
    trial = 0
    while time.time() < deadline_explore:
        trial += 1
        sigma = sigmas[trial % len(sigmas)]
        # sometimes bigger shake
        if trial % 11 == 0:
            sigma *= 1.8
        local = (trial % 2 == 0)
        if local:
            y0 = perturb_local(cur_x, sigma, rng)
        else:
            y0 = perturb_global(cur_x, sigma, rng)

        k0 = float(rng.choice([60.0, 80.0, 100.0]))
        k1 = float(rng.choice([250.0, 350.0, 500.0]))
        steps = int(rng.choice([400, 600, 800]))
        eta0 = float(rng.choice([0.08, 0.12, 0.15]))
        eta1 = eta0 * 0.12
        mu = float(rng.choice([0.0, 0.5, 0.7]))

        # guard remaining time: need at least ~0.4s per trial
        if time.time() + 0.5 > deadline_explore:
            break
        try:
            y1 = soft_polish(y0, k0, k1, steps, eta0, eta1, mu)
        except Exception:
            continue
        v = max_cos(y1)
        if v < best_v - 1e-12:
            best_v = v
            best_x = y1.copy()
            cur_x = y1.copy()
            cur_v = v
            T *= 0.99
        elif v < cur_v:
            cur_x = y1.copy()
            cur_v = v
        else:
            # Metropolis move to wander basins
            if rng.random() < np.exp(-(v - cur_v) / T):
                cur_x = y1.copy()
                cur_v = v
        # occasional restart cur to best
        if trial % 25 == 0:
            cur_x = best_x.copy()
            cur_v = best_v

    # final polish of best
    try:
        best_x = lp_polish_deadline(best_x, deadline_final)
    except Exception:
        pass
    best_x = best_x / np.maximum(np.linalg.norm(best_x, axis=1, keepdims=True), 1e-12)
    return best_x
