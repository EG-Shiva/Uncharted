import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def quench_fn(x_init, steps):
    """Annealed soft-max tangential descent, returns best seen (never worse)."""
    x = x_init.copy()
    n, d = x.shape
    G0 = x @ x.T
    np.fill_diagonal(G0, -2.0)
    mb = float(np.max(G0))
    xb = x.copy()
    log_t0 = np.log(0.008)
    log_t1 = np.log(0.0007)
    for s in range(steps):
        frac = s / steps
        tau = np.exp(log_t0 + (log_t1 - log_t0) * frac)
        lr = 0.05 - 0.035 * frac  # 0.05 -> 0.015
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        if m < mb:
            mb = m
            xb = x.copy()
        # soft-max weights
        W = np.exp((G - m) / tau)
        # row sums (diag ~0)
        rs = np.sum(W, axis=1)
        F = W @ x
        dot = np.einsum('ij,ij->i', F, x)
        F -= dot[:, None] * x
        F /= (rs[:, None] + 1e-12)
        x -= lr * F
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        x /= nrm
    Gf = x @ x.T
    np.fill_diagonal(Gf, -2.0)
    mf = float(np.max(Gf))
    if mf < mb:
        return x, mf
    return xb, mb


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        # time guard for overall solve is handled by caller via small seconds
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


def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(12345)
    # load start
    x0 = None
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (58, 22):
                x0 = a
                break
        except Exception:
            continue
    if x0 is None:
        x0 = rng.standard_normal((58, 22))
    x0 = x0 / np.linalg.norm(x0, axis=1, keepdims=True)
    best_x = x0.copy()
    best_val = max_cos(best_x)
    n, d = 58, 22

    # short quench of start to get baseline soft-max optimum (cheap)
    try:
        qx, qv = quench_fn(best_x, steps=800)
        if qv < best_val:
            best_val, best_x = qv, qx
    except Exception:
        pass

    sigmas = [0.02, 0.05, 0.10, 0.20]
    NTRIAL = 100
    for trial in range(NTRIAL):
        if time.time() - t0 > 132.0:
            break
        mode = trial % 3
        if mode == 0:
            sigma = sigmas[trial % len(sigmas)]
            noise = rng.standard_normal((n, d))
            x_try = best_x + sigma * noise
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        elif mode == 1:
            G = best_x @ best_x.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            idx = np.argsort(rowmax)[-3:]
            x_try = best_x.copy()
            x_try[idx] += 0.55 * rng.standard_normal((len(idx), d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        else:
            G = best_x @ best_x.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            k = 1 if (trial % 2 == 0) else 2
            idx = np.argsort(rowmax)[-k:]
            x_try = best_x.copy()
            x_try[idx] = rng.standard_normal((len(idx), d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        try:
            xq, vq = quench_fn(x_try, steps=2500)
        except Exception:
            continue
        if vq < best_val - 1e-12:
            best_val = vq
            best_x = xq.copy()

    # final LP polish with remaining time
    remain = BUDGET - (time.time() - t0) - 2.0
    if remain > 5.0:
        secs = min(remain, 30.0)
        try:
            best_x = lp_polish(best_x, secs)
        except Exception:
            pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
