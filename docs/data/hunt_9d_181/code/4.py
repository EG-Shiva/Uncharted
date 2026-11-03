import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
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

def smooth_opt(x, alpha, steps, s0, mu=0.9):
    n, d = x.shape
    m = np.zeros_like(x)
    for k in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        # weights focused on near-max pairs
        W = np.exp(alpha * (G - gmax))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        m = mu * m + (1.0 - mu) * gt
        rms = np.sqrt(np.mean(m * m)) + 1e-12
        step = s0 * (1.0 - 0.5 * k / max(1, steps))
        x = x - (step / rms) * m
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def find_void(x, exclude_idx, rng, n_cand=5000):
    n, d = x.shape
    mask = np.ones(n, dtype=bool)
    mask[exclude_idx] = False
    Xo = x[mask]
    cand = rng.standard_normal((n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ Xo.T
    vals = dots.max(axis=1)
    return cand[int(np.argmin(vals))]

def solve():
    t0 = time.time()
    DEADLINE = 172.0
    rng = np.random.default_rng(0)
    # load starting points
    best_x = None
    best_v = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            y = np.load(fn)
            y = np.asarray(y, dtype=float)
            if y.shape != (181, 9):
                continue
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            v = max_cos(y)
            if v < best_v:
                best_v = v
                best_x = y.copy()
        except Exception:
            continue
    if best_x is None:
        y = rng.standard_normal((181, 9))
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        best_x = y
        best_v = max_cos(y)
    x_best = best_x
    v_best = best_v
    n, d = x_best.shape
    n_lp_mid = 0
    # main basin-hopping loop, reserve ~40s for final polish
    while time.time() - t0 < 128.0:
        if time.time() - t0 > DEADLINE - 45:
            break
        G = x_best @ x_best.T
        np.fill_diagonal(G, -2.0)
        vmax = G.max()
        cnt = np.sum(G > vmax - 0.01, axis=1).astype(float)
        k = int(rng.integers(3, 12))
        probs = cnt + 2.0
        probs /= probs.sum()
        idx = rng.choice(n, size=k, replace=False, p=probs)
        sigma_soft = float(rng.uniform(0.002, 0.010))
        sigma_hard = float(rng.uniform(0.03, 0.12))
        noise = rng.standard_normal((n, d)) * sigma_soft
        noise[idx] += rng.standard_normal((k, d)) * sigma_hard
        x_try = x_best + noise
        x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        # occasional void teleport of a low-contact point
        if rng.random() < 0.40:
            j = int(np.argmin(cnt + rng.random(n) * 0.8))
            try:
                v = find_void(x_try, j, rng, n_cand=4000)
                x_try[j] = v
                x_try[j] /= np.linalg.norm(x_try[j])
            except Exception:
                pass
        alpha1 = float(rng.choice([50.0, 80.0, 120.0, 180.0]))
        x_try = smooth_opt(x_try, alpha1, steps=550, s0=0.006)
        v1 = max_cos(x_try)
        if v1 > v_best + 0.005 and rng.random() > 0.08:
            continue
        alpha2 = float(rng.choice([200.0, 300.0, 400.0]))
        x_try = smooth_opt(x_try, alpha2, steps=750, s0=0.003)
        v2 = max_cos(x_try)
        if v2 < v_best:
            # confirm with short LP if close call is real
            if v2 < v_best - 1e-6:
                x_best = x_try.copy()
                v_best = v2
            else:
                # need LP to resolve tiny gain
                if n_lp_mid < 8 and (time.time() - t0) < 110:
                    try:
                        x_lp = lp_polish(x_try, 2.5)
                        v_lp = max_cos(x_lp)
                        n_lp_mid += 1
                        if v_lp < v_best:
                            x_best = x_lp
                            v_best = v_lp
                        elif v2 < v_best:
                            x_best = x_try.copy()
                            v_best = v2
                    except Exception:
                        x_best = x_try.copy()
                        v_best = v2
                else:
                    x_best = x_try.copy()
                    v_best = v2
        elif v2 < v_best + 0.0006 and n_lp_mid < 8 and rng.random() < 0.15:
            # sideways LP probe to change basin
            if (time.time() - t0) < 110:
                try:
                    x_lp = lp_polish(x_try, 2.0)
                    v_lp = max_cos(x_lp)
                    n_lp_mid += 1
                    if v_lp < v_best:
                        x_best = x_lp
                        v_best = v_lp
                except Exception:
                    pass
    # final convergence: high-alpha smoothing then LP
    try:
        remain = DEADLINE - (time.time() - t0) - 2.0
        if remain > 8:
            x_best = smooth_opt(x_best, 350.0, steps=900, s0=0.002)
            x_best = smooth_opt(x_best, 500.0, steps=600, s0=0.001)
            remain = DEADLINE - (time.time() - t0) - 1.0
            if remain > 5:
                x_best = lp_polish(x_best, min(remain, 35.0))
    except Exception:
        pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
