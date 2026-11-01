import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def per_point_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def loads(x, beta=200.0):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    W = np.exp(beta * (G - m))
    np.fill_diagonal(W, 0.0)
    return W.sum(axis=1), m

def smooth_descent(x, steps, beta0, beta1, lr0, lr1):
    n, d = x.shape
    for s in range(steps):
        t = s / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** t
        lr = lr0 + (lr1 - lr0) * t
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ x
        proj = np.sum(Grad * x, axis=1, keepdims=True)
        Tg = Grad - proj * x
        rms = np.sqrt(np.mean(Tg * Tg)) + 1e-12
        x = x - (lr / rms) * Tg
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def max_subgradient_descent(x, iters, lr0=0.02, margin0=0.01, margin1=1e-4):
    n, d = x.shape
    for s in range(iters):
        t = s / max(iters - 1, 1)
        margin = margin0 * (margin1 / margin0) ** t
        lr = lr0 * (1.0 - 0.9 * t)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        Mask = G > m - margin
        # upper only weight 1
        iu = np.triu_indices(n, 1)
        sel = Mask[iu]
        if not np.any(sel):
            continue
        I = iu[0][sel]; J = iu[1][sel]
        Grad = np.zeros_like(x)
        # each pair pushes both points apart along the other
        # accumulate: grad_i += x_j, grad_j += x_i
        np.add.at(Grad, I, x[J])
        np.add.at(Grad, J, x[I])
        proj = np.sum(Grad * x, axis=1, keepdims=True)
        Tg = Grad - proj * x
        rms = np.sqrt(np.mean(Tg * Tg)) + 1e-12
        x = x - (lr / rms) * Tg
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def refine_points(x, idx_list, steps=250, beta=300.0, lr=0.05):
    n, d = x.shape
    for idx in idx_list:
        for s in range(steps):
            dots = x[idx] @ x.T
            dots[idx] = -2.0
            m = dots.max()
            w = np.exp(beta * (dots - m))
            w[idx] = 0.0
            g = w @ x  # (d,)
            g = g - np.dot(g, x[idx]) * x[idx]
            nrm = np.linalg.norm(g) + 1e-12
            step = lr * (0.3 + 0.7 * (1 - s / steps))
            x[idx] = x[idx] - (step / (nrm + 1e-12) * 10.0 + 1e-12) * g * 0.1
            # simpler stable update:
            # use normalized tangent
            # recompute quickly: already have g; scale
            x[idx] /= np.linalg.norm(x[idx]) + 1e-30
        # extra: small random tangent jitter to avoid exact saddle then re-refine briefly
    return x

def best_hole_candidates(x_rest, rng, ncand=12000):
    d = x_rest.shape[1]
    cand = rng.standard_normal((ncand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = (cand @ x_rest.T).max(axis=1)
    return cand[int(np.argmin(scores))]

def lp_polish(x, seconds, t0=None, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start > seconds - 0.3:
            break
        if deadline is not None and time.time() > deadline - 0.5:
            break
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            radius *= 0.5
            if radius < 1e-14:
                break
            continue
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
            if radius < 1e-14:
                break
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
            if radius < 1e-14:
                break
    return x

def solve():
    t0 = time.time()
    LIMIT = 172.0
    deadline = t0 + LIMIT
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        cands.append(xb)
    if xr is not None:
        cands.append(xr)
    if len(cands) == 0:
        rng0 = np.random.default_rng(0)
        x0 = rng0.standard_normal((68, 12))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        x_best = x0
        best_val = max_cos(x_best)
    else:
        best_val = 1e9
        x_best = None
        for c in cands:
            c = np.asarray(c, dtype=float)
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            v = max_cos(c)
            if v < best_val:
                best_val = v
                x_best = c.copy()
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    n, d = x_best.shape
    rng = np.random.default_rng(20261031)

    explore_end = deadline - 30.0
    trial = 0
    # precompute loads for bias
    while time.time() < explore_end:
        trial += 1
        time_left = explore_end - time.time()
        if time_left < 0.5:
            break
        x = x_best.copy()
        u = rng.random()
        ld, _ = loads(x, beta=200.0)
        order = np.argsort(-ld)  # overloaded first
        if u < 0.28:
            sigma = float(rng.choice(np.array([0.06, 0.12, 0.22, 0.35, 0.5])))
            # bias noise to overloaded points: larger noise there
            noise = rng.standard_normal(x.shape)
            scale = np.ones((n, 1))
            # top-8 overloaded get 2x
            scale[order[:8]] = 2.0
            x = x + sigma * scale * noise
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif u < 0.52:
            # replace k points: mix of worst-load and random
            k = int(rng.integers(2, 9))
            idx = set()
            nw = min(k, int(rng.integers(1, 4)))
            for j in range(nw):
                idx.add(int(order[j]))
            while len(idx) < k:
                idx.add(int(rng.integers(0, n)))
            idx = list(idx)
            for j in idx:
                v = rng.standard_normal(d)
                v /= np.linalg.norm(v)
                # 50% best-hole guided, 50% pure random
                if rng.random() < 0.5:
                    mask = np.ones(n, dtype=bool); mask[j] = False
                    # use current x without j
                    xr_ = np.delete(x, j, axis=0)
                    v = best_hole_candidates(xr_, rng, ncand=8000)
                x[j] = v
            # local refine replaced
            refine_points(x, idx, steps=200, beta=300.0, lr=0.05)
        elif u < 0.72:
            # delete worst k and greedy reinsert via big candidate pool
            k = int(rng.integers(1, 5))
            idx = [int(order[j]) for j in range(k)]
            # keep rest fixed, find holes sequentially
            for j in idx:
                xr_ = np.delete(x, j, axis=0)
                v = best_hole_candidates(xr_, rng, ncand=15000)
                x[j] = v
            refine_points(x, idx, steps=250, beta=350.0, lr=0.06)
        elif u < 0.86:
            # cluster shake: pick random direction, push a cap
            vdir = rng.standard_normal(d); vdir /= np.linalg.norm(vdir)
            dots = x @ vdir
            thresh = np.quantile(dots, 0.75)
            sel = np.where(dots > thresh)[0]
            if len(sel) < 3:
                sel = order[:6]
            sigma = float(rng.uniform(0.15, 0.4))
            x[sel] = x[sel] + sigma * rng.standard_normal((len(sel), d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        else:
            # strong global jitter + partial random restart of a chunk
            sigma = float(rng.uniform(0.03, 0.10))
            x = x + sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            # randomize a few
            k = int(rng.integers(1, 4))
            picks = rng.choice(n, size=k, replace=False)
            for j in picks:
                v = rng.standard_normal(d); v /= np.linalg.norm(v)
                x[j] = v

        # two-phase descent; adapt steps to remaining time
        # phase 1: softmax
        b0 = float(rng.choice(np.array([30.0, 60.0, 100.0])))
        b1 = float(rng.choice(np.array([300.0, 450.0, 600.0])))
        steps1 = int(rng.choice(np.array([400, 700, 1100])))
        lr0 = float(rng.uniform(0.02, 0.06))
        lr1 = float(rng.uniform(0.002, 0.008))
        # reduce steps if little time
        if time_left < 8:
            steps1 = min(steps1, 350)
        x = smooth_descent(x, steps=steps1, beta0=b0, beta1=b1, lr0=lr0, lr1=lr1)
        x = max_subgradient_descent(x, iters=250, lr0=0.015, margin0=0.015, margin1=2e-4)
        # extra short softmax high-beta snap
        x = smooth_descent(x, steps=150, beta0=400.0, beta1=800.0, lr0=0.008, lr1=0.002)
        v = max_cos(x)
        if v < best_val - 1e-12:
            best_val = v
            x_best = x.copy()
            # quick opportunistic LP snap on improvement if time allows (2s)
            if time.time() < explore_end - 4:
                x_lp = lp_polish(x_best, seconds=2.0, deadline=deadline)
                vlp = max_cos(x_lp)
                if vlp < best_val:
                    best_val = vlp
                    x_best = x_lp.copy()
        # occasional sideways acceptance to change basin even without improvement:
        # if within 2e-5 and random, keep as current base for next trial (not global best)
        # we implement by sometimes starting from last x instead of best: handled via u branch? skip.

    # final polish of global best with remaining time
    rem = deadline - time.time() - 1.0
    if rem > 3:
        x_best = lp_polish(x_best, seconds=rem, deadline=deadline)
        best_val = max_cos(x_best)
    # safety: normalize, remove NaN
    x_best = np.asarray(x_best, dtype=float)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    # shrink margin tiny to enforce strict feasibility (no-op for cosine but keep normalized)
    return x_best
