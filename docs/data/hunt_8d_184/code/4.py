import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def contact_degrees(x, thr):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return (g > thr).sum(axis=1)


def find_holes(x, n_need, rng, m=8000):
    # random candidates, pick those farthest from existing points
    cand = rng.standard_normal((m, x.shape[1]))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ x.T
    md = dots.max(axis=1)
    idx = np.argpartition(md, n_need)[:n_need]
    return cand[idx]


def smooth_refine(x, iters, T0, T1, lr0, lr1, mu, rng=None):
    n, d = x.shape
    v = np.zeros_like(x)
    # exponential schedules in log space
    lt0, lt1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for it in range(iters):
        f = it / max(1, iters - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mrow = G.max(axis=1, keepdims=True)
        # row softmax
        W = np.exp((G - mrow) / T)
        s = W.sum(axis=1, keepdims=True) + 1e-30
        C = (W @ x) / s
        dot = (C * x).sum(axis=1, keepdims=True)
        Gt = C - dot * x
        # clip huge tangents for stability
        nm = np.sqrt((Gt * Gt).sum(axis=1, keepdims=True)) + 1e-30
        # limit per-point norm to 1
        scale = np.minimum(1.0 / nm, np.ones_like(nm))
        # only clip if >1
        Gt = Gt * np.where(nm > 1.0, scale, 1.0)
        v = mu * v + (1.0 - mu) * Gt
        x = x - lr * v
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-30
    return x


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
        vv = max_cos(y)
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
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            rng0 = np.random.default_rng(0)
            xb = rng0.standard_normal((184, 8))
    xb = xb / (np.linalg.norm(xb, axis=1, keepdims=True) + 1e-30)
    if xb.shape != (184, 8):
        rng0 = np.random.default_rng(0)
        xb = rng0.standard_normal((184, 8))
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = max_cos(xb)
    x_best = xb.copy()

    master = np.random.default_rng(0)
    # pre-draw per-trial seeds deterministically
    trial_seeds = master.integers(0, 2**31 - 1, size=150)

    n_trials = 150
    for t in range(n_trials):
        # reserve ~35s for final LP
        if time.time() - t0 > TIME_LIMIT - 38.0:
            break
        rng = np.random.default_rng(int(trial_seeds[t]))
        x = x_best.copy()
        cur_max = max_cos(x)
        thr = cur_max - 0.01
        deg = contact_degrees(x, thr)
        order = np.argsort(deg)
        n_move = int(rng.integers(2, 7))  # 2..6 rattlers to relocate
        victims = order[:n_move]
        try:
            holes = find_holes(x, n_move, rng, m=8000)
        except Exception:
            holes = rng.standard_normal((n_move, 8))
            holes /= np.linalg.norm(holes, axis=1, keepdims=True)
        x[victims] = holes
        # global jitter + extra shake on overloaded points
        sigma_all = float(rng.uniform(0.008, 0.030))
        x = x + rng.standard_normal(x.shape) * sigma_all
        # extra shake for top overloaded
        n_over = int(rng.integers(3, 10))
        over = order[-n_over:]
        x[over] += rng.standard_normal((n_over, x.shape[1])) * 0.02
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-30

        # randomized annealing schedule
        T0 = float(rng.uniform(0.015, 0.035))
        T1 = float(rng.uniform(0.002, 0.006))
        lr0 = float(rng.uniform(0.04, 0.10))
        lr1 = float(rng.uniform(0.001, 0.006))
        mu = float(rng.uniform(0.80, 0.92))
        iters = int(rng.integers(1200, 2500))
        # cap iters if little time
        if time.time() - t0 > TIME_LIMIT - 70.0:
            iters = min(iters, 1200)
        x = smooth_refine(x, iters, T0, T1, lr0, lr1, mu)
        v = max_cos(x)
        if v < best_val:
            best_val = v
            x_best = x.copy()
            continue
        # second-chance squeeze if close
        if v < best_val + 0.004:
            x2 = smooth_refine(x, 700, 0.006, 0.002, 0.015, 0.002, 0.85)
            v2 = max_cos(x2)
            if v2 < best_val:
                best_val = v2
                x_best = x2.copy()

    # final polish with remaining time
    rem = TIME_LIMIT - (time.time() - t0) - 2.0
    if rem > 5.0:
        # short smooth cool-down first (cheap, helps LP)
        try:
            x_best = smooth_refine(x_best, 600, 0.006, 0.002, 0.01, 0.001, 0.85)
            vv = max_cos(x_best)
            # keep track (smooth may slightly worsen before LP, but allow)
            best_val = min(best_val, vv)
        except Exception:
            pass
        rem2 = TIME_LIMIT - (time.time() - t0) - 1.0
        if rem2 > 3.0:
            try:
                x_best = lp_polish(x_best, min(rem2, 40.0))
            except Exception:
                pass

    x_best = x_best / (np.linalg.norm(x_best, axis=1, keepdims=True) + 1e-30)
    # sanitize
    x_best = np.nan_to_num(x_best, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(x_best, axis=1, keepdims=True) + 1e-30
    x_best = x_best / nr
    return x_best
