import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds):
    """Trust-region LP polish, time-bounded."""
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start >= seconds:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def soft_anneal(x, steps, beta0, beta1, lr, rng, noise_amp=0.0, mu=0.8):
    n, d = x.shape
    vel = np.zeros_like(x)
    for k in range(steps):
        frac = k / max(1, steps - 1)
        beta = beta0 + (beta1 - beta0) * frac
        lr_k = lr * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        # stable weights, max weight = 1
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        vel = mu * vel + (1.0 - mu) * gt
        x = x - lr_k * vel
        if noise_amp > 0:
            nz = rng.standard_normal(x.shape)
            nd = np.sum(nz * x, axis=1, keepdims=True)
            nz -= nd * x
            x = x + noise_amp * (1.0 - frac) * nz
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def contact_counts(x, best_val, tol=1e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    cnt = np.sum(G > best_val - tol, axis=1)
    return cnt


def solve():
    t0 = time.time()
    T_LIMIT = 170.0
    rng = np.random.default_rng(0)
    np.random.seed(0)

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (138, 8):
                a /= np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        a = rng.standard_normal((138, 8))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)

    best = None
    best_val = np.inf
    for c in cands:
        v = max_cos(c)
        if v < best_val:
            best_val = v
            best = c.copy()

    # cache contact counts for targeted kicks
    try:
        cc = contact_counts(best, best_val)
    except Exception:
        cc = np.ones(138)

    trial = 0
    sigmas = [0.30, 0.20, 0.12, 0.07, 0.04, 0.02, 0.012]
    # exploration until ~125s
    while time.time() - t0 < 122.0:
        trial += 1
        time_left = T_LIMIT - (time.time() - t0)
        if time_left < 12.0:
            break
        sigma = float(sigmas[(trial - 1) % len(sigmas)]) * (0.7 + 0.6 * rng.random())
        # refresh contact counts occasionally
        if trial % 5 == 1:
            try:
                cc = contact_counts(best, best_val)
            except Exception:
                pass
        x_try = best.copy()
        if trial % 2 == 0:
            # full kick
            x_try = best + sigma * rng.standard_normal(best.shape)
        else:
            # subset kick biased to jammed points
            k = int(10 + rng.integers(0, 16))  # 10..25
            w = (cc.astype(float) + 1.0)
            w /= w.sum()
            idx = rng.choice(138, size=k, replace=False, p=w)
            x_try = best + 0.015 * rng.standard_normal(best.shape)
            x_try[idx] += (sigma * 2.0) * rng.standard_normal((k, 8))
        x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)

        beta0 = 10.0 + rng.random() * 20.0
        beta1 = 80.0 + rng.random() * 120.0
        lr = 0.020 + rng.random() * 0.025
        steps = 2200
        # smaller steps if little time
        if time.time() - t0 > 100:
            steps = 1500
        noise_amp = sigma * 0.05 * rng.random()
        try:
            x_try = soft_anneal(x_try, steps, beta0, beta1, lr, rng,
                                noise_amp=noise_amp, mu=0.8)
        except Exception:
            continue
        try:
            v_try = max_cos(x_try)
        except Exception:
            continue
        # only LP-snap promising candidates
        if v_try < best_val + 0.012:
            polish_sec = 1.6
            if v_try < best_val:
                polish_sec = 2.5
            # reserve time for final polish
            tl = T_LIMIT - (time.time() - t0)
            polish_sec = min(polish_sec, max(0.3, tl - 9.0))
            if polish_sec < 0.25:
                break
            try:
                x_pol = lp_polish(x_try, polish_sec)
                v_pol = max_cos(x_pol)
            except Exception:
                continue
            if v_pol < best_val - 1e-12:
                best = x_pol
                best_val = v_pol
                try:
                    cc = contact_counts(best, best_val)
                except Exception:
                    pass

    # final polish of best
    tl = T_LIMIT - (time.time() - t0)
    final_sec = max(1.0, min(32.0, tl - 1.5))
    if tl > 2.0:
        try:
            best = lp_polish(best, final_sec)
            best_val = max_cos(best)
        except Exception:
            pass
    # tiny extra: try few micro-kicks + snap if time remains
    while time.time() - t0 < T_LIMIT - 1.0:
        try:
            sigma = 0.004
            x_try = best + sigma * rng.standard_normal(best.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_pol = lp_polish(x_try, 0.8)
            v_pol = max_cos(x_pol)
            if v_pol < best_val:
                best, best_val = x_pol, v_pol
            else:
                break
        except Exception:
            break
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
