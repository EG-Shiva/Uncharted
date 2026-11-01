import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def quench_momentum(x, steps, beta0, beta1, lr0, lr1, mu, rng=None):
    n, d = x.shape
    vel = np.zeros_like(x)
    for s in range(steps):
        t = s / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** t if beta0 > 0 else beta1
        lr = lr0 + (lr1 - lr0) * t
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        # softmax weights on largest dots
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ x
        proj = np.sum(Grad * x, axis=1, keepdims=True)
        Tg = Grad - proj * x
        rms = np.sqrt(np.mean(Tg * Tg)) + 1e-12
        Tg_n = Tg / rms
        vel = mu * vel + Tg_n
        # correct drift to tangent
        vproj = np.sum(vel * x, axis=1, keepdims=True)
        vel_t = vel - vproj * x
        x = x - lr * vel_t
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def quench_frozen(x, steps, beta0, beta1, lr0, lr1, mu, frozen_mask):
    # frozen_mask True = frozen
    n, d = x.shape
    vel = np.zeros_like(x)
    for s in range(steps):
        t = s / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** t if beta0 > 0 else beta1
        lr = lr0 + (lr1 - lr0) * t
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ x
        proj = np.sum(Grad * x, axis=1, keepdims=True)
        Tg = Grad - proj * x
        Tg[frozen_mask] = 0.0
        rms = np.sqrt(np.mean(Tg * Tg)) + 1e-12
        Tg_n = Tg / rms
        vel = mu * vel + Tg_n
        vel[frozen_mask] = 0.0
        vproj = np.sum(vel * x, axis=1, keepdims=True)
        vel_t = vel - vproj * x
        x = x - lr * vel_t
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.25:
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


def best_hole(x_rest, rng, ncand=8000):
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
    n, d = x_best.shape
    rng = np.random.default_rng(20261031)

    explore_end = t0 + TIME_LIMIT - 30.0
    trial = 0
    # precompute for speed
    while time.time() < explore_end:
        trial += 1
        x = x_best.copy()
        u = rng.random()
        if u < 0.30:
            # global shake + momentum quench
            sigma = float(rng.choice([0.006, 0.015, 0.035, 0.08, 0.15]))
            x = x + sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            beta0 = float(rng.uniform(40, 120))
            beta1 = float(rng.uniform(320, 750))
            lr0 = float(rng.uniform(0.02, 0.05))
            lr1 = float(rng.uniform(0.001, 0.005))
            mu = float(rng.uniform(0.0, 0.92))
            steps = int(rng.integers(550, 900))
            x = quench_momentum(x, steps, beta0, beta1, lr0, lr1, mu)
        elif u < 0.62:
            # jammed-cluster reshuffle with two-stage quench
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            # seed: worst or high-contact with prob
            if rng.random() < 0.6:
                seed = int(np.argmax(per))
            else:
                # weighted by per-point max
                w = np.exp(60 * (per - per.max()))
                w = w / w.sum()
                seed = int(rng.choice(n, p=w))
            k = int(rng.integers(4, 10))
            nbrs = np.argsort(-G[seed])[:k]
            cluster = np.unique(np.concatenate([[seed], nbrs]))
            # re-randomize cluster
            if rng.random() < 0.5:
                for q in cluster:
                    v = rng.standard_normal(d)
                    v /= np.linalg.norm(v)
                    x[q] = v
            else:
                x[cluster] += float(rng.uniform(0.3, 0.8)) * rng.standard_normal((len(cluster), d))
                x /= np.linalg.norm(x, axis=1, keepdims=True)
            frozen = np.ones(n, dtype=bool)
            frozen[cluster] = False
            frozen_mask = frozen  # True=frozen
            # stage 1: frozen background
            b0 = float(rng.uniform(50, 120))
            b1 = float(rng.uniform(300, 500))
            x = quench_frozen(x, 250, b0, b1, 0.03, 0.005, 0.5, frozen_mask)
            # stage 2: full
            beta0 = float(rng.uniform(60, 130))
            beta1 = float(rng.uniform(350, 750))
            lr0 = float(rng.uniform(0.02, 0.04))
            lr1 = float(rng.uniform(0.001, 0.004))
            mu = float(rng.uniform(0.3, 0.9))
            x = quench_momentum(x, int(rng.integers(600, 900)), beta0, beta1, lr0, lr1, mu)
        elif u < 0.80:
            # delete worst, reinsert at best hole
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            wi = int(np.argmax(per))
            mask = np.ones(n, dtype=bool)
            mask[wi] = False
            hole = best_hole(x[mask], rng, ncand=8000)
            x[wi] = hole
            G2 = x @ x.T
            nbrs = np.argsort(-G2[wi])[:6]
            x[nbrs] += 0.02 * rng.standard_normal((len(nbrs), d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            beta0 = float(rng.uniform(60, 120))
            beta1 = float(rng.uniform(350, 700))
            x = quench_momentum(x, 750, beta0, beta1, 0.03, 0.002, float(rng.uniform(0.2, 0.85)))
        else:
            # heavy multi-point randomization
            k = int(rng.integers(3, 8))
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            order = np.argsort(-per)
            idx = set()
            for j in range(min(2, k)):
                idx.add(int(order[j]))
            while len(idx) < k:
                idx.add(int(rng.integers(0, n)))
            idx = np.array(list(idx), dtype=int)
            for q in idx:
                v = rng.standard_normal(d)
                v /= np.linalg.norm(v)
                x[q] = v
            beta0 = float(rng.uniform(50, 110))
            beta1 = float(rng.uniform(350, 750))
            x = quench_momentum(x, 850, beta0, beta1, 0.035, 0.002, float(rng.uniform(0.0, 0.9)))

        # high-beta fine tune
        x = quench_momentum(x, 200, 450.0, 650.0, 0.008, 0.001, 0.6)
        v = max_cos(x)
        if v < best_val - 1e-12:
            remain = explore_end - time.time()
            if remain > 3.0 and v < best_val - 2e-7:
                x = lp_polish(x, min(4.0, remain - 0.5))
                v = max_cos(x)
            if v < best_val:
                x_best = x.copy()
                best_val = v
        elif v < best_val + 0.0025:
            if (trial % 3 == 0) and (time.time() < explore_end - 4.0):
                y = lp_polish(x, 1.5)
                v2 = max_cos(y)
                if v2 < best_val:
                    x_best = y.copy()
                    best_val = v2
        if trial > 500:
            break
        if time.time() > explore_end:
            break

    remain = TIME_LIMIT - (time.time() - t0)
    if remain > 3.0:
        x_best = lp_polish(x_best, min(remain - 1.0, 27.0))
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
