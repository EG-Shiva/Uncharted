import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def smooth_descent(x, steps, beta0, beta1, lr0, lr1):
    n, d = x.shape
    for s in range(steps):
        t = s / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** t if beta0 > 0 else beta1
        lr = lr0 + (lr1 - lr0) * t
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        # weights focus on largest dots
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ x
        proj = np.sum(Grad * x, axis=1, keepdims=True)
        Tg = Grad - proj * x
        rms = np.sqrt(np.mean(Tg * Tg)) + 1e-12
        x = x - (lr / rms) * Tg
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def lp_polish(x, seconds):
    """Trust-region LP: move points tangentially to lower the largest cosine."""
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.2:
            break
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            radius *= 0.5
            continue
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


def worst_index(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    per = G.max(axis=1)
    return int(np.argmax(per))


def best_hole(x_rest, rng, ncand=5000):
    d = x_rest.shape[1]
    cand = rng.standard_normal((ncand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = (cand @ x_rest.T).max(axis=1)
    return cand[int(np.argmin(scores))]


def solve():
    t0 = time.time()
    TIME_LIMIT = 172.0
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
        xb = rng0.standard_normal((68, 12))
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    else:
        # pick best of loaded
        bestv = 1e9
        xb = cands[0]
        for c in cands:
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            v = max_cos(c)
            if v < bestv:
                bestv = v
                xb = c.copy()
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = max_cos(xb)
    x_best = xb.copy()

    rng = np.random.default_rng(12345)
    n, d = x_best.shape

    # exploration deadline: reserve ~28s for final polish
    explore_end = t0 + TIME_LIMIT - 28.0
    trial = 0
    while time.time() < explore_end:
        trial += 1
        x = x_best.copy()
        u = rng.random()
        if u < 0.40:
            # global shake
            choice = rng.integers(0, 3)
            sigma = [0.008, 0.025, 0.07][int(choice)]
            x = x + sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = smooth_descent(x, steps=700, beta0=60.0, beta1=380.0, lr0=0.03, lr1=0.003)
        elif u < 0.72:
            # kick / randomize a few points, bias to high-contact overloaded points
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            order = np.argsort(-per)
            k = int(rng.integers(1, 4))
            idx = []
            # half from worst, half random
            nw = min(k, 2)
            for j in range(nw):
                idx.append(int(order[j]))
            while len(idx) < k:
                q = int(rng.integers(0, n))
                if q not in idx:
                    idx.append(q)
            idx = np.array(idx, dtype=int)
            if rng.random() < 0.5:
                # full re-randomize
                for q in idx:
                    v = rng.standard_normal(d)
                    v /= np.linalg.norm(v)
                    x[q] = v
            else:
                # strong tangent kick
                x[idx] += 0.5 * rng.standard_normal((len(idx), d))
                x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = smooth_descent(x, steps=800, beta0=80.0, beta1=450.0, lr0=0.03, lr1=0.002)
        else:
            # delete worst point, reinsert at best hole
            wi = worst_index(x)
            mask = np.ones(n, dtype=bool)
            mask[wi] = False
            x_rest = x[mask]
            hole = best_hole(x_rest, rng, ncand=8000)
            x[wi] = hole
            # small shake of neighbours to break topology
            G = x @ x.T
            nbrs = np.argsort(-G[wi])[:6]
            x[nbrs] += 0.02 * rng.standard_normal((len(nbrs), d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = smooth_descent(x, steps=800, beta0=80.0, beta1=450.0, lr0=0.025, lr1=0.002)

        # extra high-beta fine tune (cheap)
        x = smooth_descent(x, steps=250, beta0=450.0, beta1=650.0, lr0=0.008, lr1=0.001)
        v = max_cos(x)
        if v < best_val - 1e-12:
            # got improvement without LP
            # quick LP to squeeze
            remain = explore_end - time.time()
            if remain > 3.0 and v < best_val - 2e-7:
                x = lp_polish(x, min(4.0, remain - 0.5))
                v = max_cos(x)
            if v < best_val:
                x_best = x.copy()
                best_val = v
        elif v < best_val + 0.002:
            # promising basin, try short LP with small probability / frequency
            if (trial % 4 == 0) and (time.time() < explore_end - 4.0):
                y = lp_polish(x, 1.6)
                v2 = max_cos(y)
                if v2 < best_val:
                    x_best = y.copy()
                    best_val = v2
        # safety: stop if time low
        if time.time() > explore_end:
            break
        if trial > 400:
            break

    remain = TIME_LIMIT - (time.time() - t0)
    if remain > 3.0:
        x_best = lp_polish(x_best, min(remain - 1.0, 26.0))
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
