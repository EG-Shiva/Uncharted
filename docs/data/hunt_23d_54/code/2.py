import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def smooth_polish(x0, iters, beta0, beta1, lr):
    x = x0.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    for t in range(1, iters + 1):
        frac = t / iters
        beta = beta0 + (beta1 - beta0) * frac
        lrt = lr * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        gmax = np.max(G)
        # weights for log-sum-exp
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = np.sum(W) + 1e-300
        grad = (W @ x) / S
        proj = np.sum(grad * x, axis=1, keepdims=True)
        g = grad - proj * x
        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * (g * g)
        bc1 = 1.0 - (b1 ** t)
        bc2 = 1.0 - (b2 ** t)
        mhat = m / bc1
        vhat = v / bc2
        step = mhat / (np.sqrt(vhat) + eps)
        x = x - lrt * step
        x = _norm_rows(x)
    return x


def joint_polish(x0):
    # coarse + fine
    x = smooth_polish(x0, 600, 50.0, 200.0, 0.01)
    x = smooth_polish(x, 600, 200.0, 800.0, 0.003)
    return x


def lp_polish(x, seconds):
    """Trust-region LP: move points tangentially to lower the largest cosine."""
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        mval = c.max()
        sel = c > mval - max(20 * radius, 1e-9)
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


def random_sphere(rng, n, d):
    a = rng.standard_normal(size=(n, d))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    return a


def solve():
    t0 = time.time()
    rng = np.random.default_rng(0)
    try:
        x_init = np.load("best.npy")
    except Exception:
        try:
            x_init = np.load("record.npy")
        except Exception:
            x_init = random_sphere(rng, 54, 23)
    x_init = np.asarray(x_init, dtype=float)
    # enforce shape (54,23)
    if x_init.shape != (54, 23):
        # try to reshape / truncate / pad deterministically
        flat = x_init.ravel()
        need = 54 * 23
        if flat.size < need:
            extra = random_sphere(rng, 54, 23).ravel()[: need - flat.size]
            flat = np.concatenate([flat, extra])
        x_init = flat[:need].reshape(54, 23)
    x_init = _norm_rows(x_init)
    best = x_init.copy()
    best_val = max_cos(best)

    n, d = 54, 23
    # hopping phase: fixed number of trials, time-guarded
    MAX_TRIALS = 400
    for trial in range(MAX_TRIALS):
        if time.time() - t0 > 125:
            break
        mode = trial % 4
        x_try = best.copy()
        if mode == 0:
            sigma = 0.015
            x_try = x_try + sigma * rng.standard_normal(size=(n, d))
            x_try = _norm_rows(x_try)
        elif mode == 1:
            sigma = 0.05
            x_try = x_try + sigma * rng.standard_normal(size=(n, d))
            # resample 2 random points fully
            idx = rng.choice(n, 2, replace=False)
            x_try[idx] = random_sphere(rng, 2, d)
            x_try = _norm_rows(x_try)
        elif mode == 2:
            s = int(rng.integers(3, 5))  # 3 or 4
            idx = rng.choice(n, s, replace=False)
            x_try[idx] = random_sphere(rng, s, d)
            x_try = x_try + 0.005 * rng.standard_normal(size=(n, d))
            x_try = _norm_rows(x_try)
        else:
            # targeted kick of worst points
            G = x_try @ x_try.T
            np.fill_diagonal(G, -2.0)
            rowmax = np.max(G, axis=1)
            k = 5
            worst = np.argsort(-rowmax)[:k]
            x_try[worst] = x_try[worst] + 0.10 * rng.standard_normal(size=(k, d))
            x_try = x_try + 0.01 * rng.standard_normal(size=(n, d))
            x_try = _norm_rows(x_try)

        x_pol = joint_polish(x_try)
        v = max_cos(x_pol)
        if v < best_val and np.all(np.isfinite(x_pol)):
            best_val = v
            best = x_pol.copy()
        # occasional small refinement of current best without perturbation
        if trial % 50 == 49:
            # single-point greedy reinsertion attempt on worst point
            if time.time() - t0 > 125:
                break
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            rowmax = np.max(G, axis=1)
            wi = int(np.argmax(rowmax))
            # sample candidates for this point
            cand = random_sphere(rng, 2000, d)
            dots = cand @ best.T
            dots[:, wi] = -10.0
            cmax = np.max(dots, axis=1)
            bi = int(np.argmin(cmax))
            x_tmp = best.copy()
            x_tmp[wi] = cand[bi]
            x_tmp = joint_polish(x_tmp)
            v2 = max_cos(x_tmp)
            if v2 < best_val:
                best_val = v2
                best = x_tmp.copy()

    # final LP tightening with remaining budget
    elapsed = time.time() - t0
    remaining = 172.0 - elapsed
    if remaining > 8:
        lp_time = min(remaining - 2.0, 40.0)
        if lp_time > 3:
            try:
                best = lp_polish(best, lp_time)
            except Exception:
                pass
    best = _norm_rows(best)
    # strict feasibility: no zero rows, no NaN already ensured
    return best
