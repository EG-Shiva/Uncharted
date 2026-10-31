import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _anneal(x, steps, tau0, tau1, lr0, lr1):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for k in range(steps):
        frac = k / (steps - 1) if steps > 1 else 0.0
        tau = tau0 * (tau1 / tau0) ** frac
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gm = G.max()
        # E = exp((G-gm)/tau) in-place
        np.subtract(G, gm, out=G)
        np.divide(G, tau, out=G)
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        S = G.sum()
        if not np.isfinite(S) or S == 0:
            break
        G /= S
        grad = G @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        g = grad - dot * x
        m *= b1
        m += (1 - b1) * g
        v *= b2
        v += (1 - b2) * (g * g)
        bc1 = 1.0 - b1 ** (k + 1)
        bc2 = 1.0 - b2 ** (k + 1)
        mh = m / bc1
        vh = v / bc2
        step = mh / (np.sqrt(vh) + eps)
        # clip extreme steps for stability
        np.clip(step, -2.0, 2.0, out=step)
        x = x - lr * step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _find_void(x_fixed, rng, n_samples=12000):
    cand = rng.standard_normal((n_samples, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    sims = cand @ x_fixed.T
    best = sims.max(axis=1)
    idx = int(best.argmin())
    return cand[idx]

def _lp_polish(x, seconds, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        mcur = c.max()
        sel = c > mcur - max(20 * radius, 1e-9)
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
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    rng = np.random.default_rng(7)
    t0 = time.time()
    deadline = t0 + 172.0
    # load start
    try:
        x0 = np.load("best.npy")
    except Exception:
        x0 = np.load("record.npy")
    x0 = np.asarray(x0, dtype=float)
    if x0.shape != (272, 4):
        # fallback: truncate/pad deterministically
        flat = x0.ravel()
        need = 272 * 4
        if flat.size < need:
            extra = rng.standard_normal((need - flat.size,))
            flat = np.concatenate([flat, extra])
        x0 = flat[:need].reshape(272, 4)
    x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
    # clean NaN
    x0 = np.where(np.isfinite(x0), x0, 0.0)
    xn = np.linalg.norm(x0, axis=1, keepdims=True)
    xn[xn == 0] = 1.0
    x0 = x0 / xn

    x_best = x0.copy()
    best_val = _max_cos(x_best)

    sigmas = [0.03, 0.012, 0.06, 0.02, 0.09, 0.008]
    taus = [(0.015, 0.0015), (0.02, 0.002), (0.008, 0.0008), (0.03, 0.003)]
    lrs = [(0.008, 0.0008), (0.012, 0.001), (0.005, 0.0005)]

    trial = 0
    # Stage 1: basin hopping with anneal, budget ~115s
    while time.time() < t0 + 118 and time.time() < deadline - 45:
        trial += 1
        sigma = sigmas[trial % len(sigmas)]
        tau0, tau1 = taus[trial % len(taus)]
        lr0, lr1 = lrs[trial % len(lrs)]
        x = x_best + sigma * rng.standard_normal((272, 4))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        # topology change: reinsert worst offenders into voids
        if trial % 2 == 0:
            nkill = 2 + (trial % 3)  # 2..4
            for _ in range(nkill):
                G = x @ x.T
                np.fill_diagonal(G, -2.0)
                # victim: endpoint of current max pair
                idx = int(G.argmax())
                ii, jj = divmod(idx, 272)
                victim = ii if rng.random() < 0.5 else jj
                mask = np.ones(272, dtype=bool)
                mask[victim] = False
                vv = _find_void(x[mask], rng, n_samples=10000)
                x[victim] = vv
        elif trial % 5 == 0:
            # relocate random rattler-like point: point with smallest max
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            mx = G.max(axis=1)
            victim = int(mx.argmin())
            mask = np.ones(272, dtype=bool)
            mask[victim] = False
            vv = _find_void(x[mask], rng, n_samples=10000)
            x[victim] = vv
        steps = 600
        # shorter steps when time is short
        x = _anneal(x, steps, tau0, tau1, lr0, lr1)
        v = _max_cos(x)
        if v < best_val:
            best_val = v
            x_best = x.copy()
    # Stage 2: greedy local improvement on best with a few more anneals at low temp
    for _ in range(3):
        if time.time() > deadline - 42:
            break
        x = _anneal(x_best.copy(), 400, 0.006, 0.0007, 0.004, 0.0004)
        v = _max_cos(x)
        if v < best_val:
            best_val = v
            x_best = x.copy()
    # Stage 3: LP snap
    remain = deadline - time.time() - 2.0
    if remain > 5:
        x_pol = _lp_polish(x_best, min(remain, 38.0), deadline - 1.0)
        vpol = _max_cos(x_pol)
        if vpol <= best_val:
            x_best = x_pol
            best_val = vpol
    # fallback safety
    if not np.all(np.isfinite(x_best)):
        x_best = x0.copy()
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
