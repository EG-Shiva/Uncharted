import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def per_point_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)


def lp_polish(x, seconds, r0=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = r0, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
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
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def smooth_lbfgs(x_init, alphas=(60.0, 250.0, 900.0), maxiters=(50, 60, 80)):
    n, d = x_init.shape
    y = x_init.ravel().copy().astype(float)

    for alpha, mi in zip(alphas, maxiters):
        def fun(y_flat, alpha=alpha):
            Y = y_flat.reshape(n, d)
            nr = np.linalg.norm(Y, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            X = Y / nr
            C = X @ X.T
            m = float(np.max(C - np.eye(n) * 10.0))
            E = np.exp(alpha * (C - m))
            np.fill_diagonal(E, 0.0)
            S = float(E.sum() / 2.0)
            if S <= 0:
                S = 1e-300
            f = m + np.log(S) / alpha
            W = E / S
            r = np.sum(W * C, axis=1)
            G = W @ X - r[:, None] * X
            grad = G / nr
            return float(f), grad.ravel()
        try:
            res = minimize(fun, y, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(mi), "ftol": 1e-14, "gtol": 1e-7, "maxfun": int(mi * 10)})
            y = res.x
        except Exception:
            break
    Y = y.reshape(n, d)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y


def tangent_shake(x, sigma, rng, idx=None):
    n, d = x.shape
    Z = rng.standard_normal((n, d))
    Z -= (Z * x).sum(axis=1, keepdims=True) * x
    if idx is not None:
        M = np.zeros(n, dtype=bool)
        M[np.asarray(idx)] = True
        Z[~M] = 0.0
    y = x + sigma * Z
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y


def hole_reinsert(x, k, rng, n_samples=20000):
    n, d = x.shape
    C = x @ x.T
    np.fill_diagonal(C, -2.0)
    pm = C.max(axis=1)
    worst = np.argsort(-pm)[:k]
    worst_set = set(worst.tolist())
    keep_idx = [i for i in range(n) if i not in worst_set]
    X_rem = x[keep_idx]
    X_cur = X_rem.copy()
    new_pts = []
    for _ in range(k):
        R = rng.standard_normal((n_samples, d))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        dots = R @ X_cur.T
        md = dots.max(axis=1)
        bi = int(np.argmin(md))
        bp = R[bi]
        new_pts.append(bp)
        X_cur = np.vstack([X_cur, bp[None, :]])
    return X_cur


def solve():
    t0 = time.time()
    BUDGET = 170.0
    rng = np.random.default_rng(0)
    # load starting points
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (55, 21):
                a /= np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        a = rng.standard_normal((55, 21))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    scores = [max_cos(a) for a in cands]
    bi = int(np.argmin(scores))
    x_best = cands[bi].copy()
    s_best = float(scores[bi])

    # archive of a few elites for diversity
    elites = [(s_best, x_best.copy())]

    trial = 0
    # explore until ~135s, reserve rest for final polish
    while time.time() - t0 < 132.0:
        trial += 1
        remain = BUDGET - (time.time() - t0)
        if remain < 25:
            break
        strat = trial % 6
        try:
            if strat == 1:
                # full small shake
                sigma = float(0.02 + 0.08 * rng.random())
                x_try = tangent_shake(x_best, sigma, rng)
                x_try = smooth_lbfgs(x_try, alphas=(60.0, 250.0, 900.0), maxiters=(40, 50, 70))
                x_try = lp_polish(x_try, seconds=1.2)
            elif strat == 2:
                # subset hard shake
                k = int(rng.integers(2, 9))
                idx = rng.choice(55, size=k, replace=False)
                sigma = float(0.25 + 0.40 * rng.random())
                x_try = tangent_shake(x_best, sigma, rng, idx=idx)
                x_try = smooth_lbfgs(x_try, alphas=(50.0, 200.0, 800.0), maxiters=(40, 50, 70))
                x_try = lp_polish(x_try, seconds=1.2)
            elif strat == 3:
                # hole reinsert worst points
                k = int(rng.integers(1, 4))
                ns = 15000 if (trial % 2 == 0) else 25000
                x_try = hole_reinsert(x_best, k, rng, n_samples=ns)
                x_try = smooth_lbfgs(x_try, alphas=(60.0, 250.0, 900.0), maxiters=(40, 60, 80))
                x_try = lp_polish(x_try, seconds=1.4)
            elif strat == 4:
                # large global shake
                sigma = float(0.30 + 0.30 * rng.random())
                x_try = tangent_shake(x_best, sigma, rng)
                x_try = smooth_lbfgs(x_try, alphas=(30.0, 120.0, 600.0), maxiters=(50, 60, 80))
                x_try = lp_polish(x_try, seconds=1.2)
            elif strat == 5:
                # interpolate best with random far point
                R = rng.standard_normal((55, 21))
                R /= np.linalg.norm(R, axis=1, keepdims=True)
                a = float(0.10 + 0.25 * rng.random())
                x_try = (1 - a) * x_best + a * R
                x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
                x_try = smooth_lbfgs(x_try, alphas=(60.0, 250.0, 1000.0), maxiters=(40, 50, 70))
                x_try = lp_polish(x_try, seconds=1.2)
            else:
                # perturb elite / second best occasionally
                if len(elites) > 1 and (trial % 3 == 0):
                    eb = elites[1][1]
                else:
                    eb = x_best
                k = int(rng.integers(3, 7))
                idx = rng.choice(55, size=k, replace=False)
                sigma = float(0.10 + 0.20 * rng.random())
                x_try = tangent_shake(eb, sigma, rng, idx=idx)
                # also shake one worst point to hole
                if rng.random() < 0.5:
                    C = x_try @ x_try.T
                    np.fill_diagonal(C, -2.0)
                    w = int(np.argmax(C.max(axis=1)))
                    R = rng.standard_normal((8000, 21))
                    R /= np.linalg.norm(R, axis=1, keepdims=True)
                    rest = np.delete(x_try, w, axis=0)
                    dots = R @ rest.T
                    bi2 = int(np.argmin(dots.max(axis=1)))
                    x_try[w] = R[bi2]
                x_try = smooth_lbfgs(x_try, alphas=(60.0, 250.0, 900.0), maxiters=(40, 50, 70))
                x_try = lp_polish(x_try, seconds=1.2)
        except Exception:
            continue
        try:
            s_try = max_cos(x_try)
        except Exception:
            continue
        if not np.isfinite(s_try):
            continue
        if s_try < s_best - 1e-12:
            s_best = s_try
            x_best = x_try / np.linalg.norm(x_try, axis=1, keepdims=True)
            elites.append((s_best, x_best.copy()))
            elites = sorted(elites, key=lambda z: z[0])[:3]
        elif s_try < elites[-1][0] and len(elites) < 3:
            elites.append((s_try, (x_try / np.linalg.norm(x_try, axis=1, keepdims=True)).copy()))
            elites = sorted(elites, key=lambda z: z[0])[:3]
        # time guard
        if time.time() - t0 > 132.0:
            break

    # final intensive polish of the best
    remain = BUDGET - (time.time() - t0) - 2.0
    if remain > 4:
        try:
            x_best = lp_polish(x_best, seconds=min(remain, 30.0), r0=8e-4)
            s_best = max_cos(x_best)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
