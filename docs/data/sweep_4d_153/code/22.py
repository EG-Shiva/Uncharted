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
    bx = x.copy()
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
            x, best, bx = y, v, y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return bx, best

def anneal(x, steps, k0, k1, lr0, lr1, mu, noise, rng):
    n, d = x.shape
    v = np.zeros_like(x)
    for t in range(steps):
        frac = t / max(1, steps - 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        W = np.exp(k * (G - m))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        v = mu * v + gt
        denom = 1.0 - mu ** (t + 1) if mu > 0 else 1.0
        vc = v / denom
        # clip extreme steps
        nm = np.linalg.norm(vc, axis=1, keepdims=True) + 1e-12
        # limit per-point step to 0.05
        scale = np.minimum(1.0, 0.05 / (lr * nm + 1e-12))
        x = x - lr * vc * scale
        if noise > 0 and frac < 0.6:
            amp = noise * (1.0 - frac / 0.6)
            pert = rng.standard_normal(x.shape)
            pert -= np.sum(pert * x, axis=1, keepdims=True) * x
            x = x + amp * pert
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.default_rng(12345)
    n, d = 153, 4
    try:
        init = np.load("best.npy")
    except Exception:
        try:
            init = np.load("record.npy")
        except Exception:
            init = rng.standard_normal((n, d))
    init = np.asarray(init, dtype=float)
    if init.shape != (n, d):
        init = rng.standard_normal((n, d))
    init /= np.linalg.norm(init, axis=1, keepdims=True)
    inc = init.copy()
    inc_val = max_cos(inc)
    # quick record of best polished
    best_x, best_val = inc.copy(), inc_val

    SEARCH_END = t0 + BUDGET - 38.0
    trial = 0
    # precompute triu not needed
    while time.time() < SEARCH_END:
        trial += 1
        remain = SEARCH_END - time.time()
        if remain < 1.0:
            break
        # choose base: mostly best
        rbase = rng.random()
        if rbase < 0.7:
            base = best_x
        elif rbase < 0.9:
            base = inc
        else:
            base = init
        x = base.copy()

        # analyse contacts of base
        G0 = x @ x.T
        np.fill_diagonal(G0, -2.0)
        m0 = G0.max()
        maxper = G0.max(axis=1)
        contacts = (G0 > m0 - 1e-9).sum(axis=1)

        mode = rng.random()
        if mode < 0.30:
            sigma = float(rng.choice(np.array([0.015, 0.03, 0.06, 0.10, 0.18, 0.30])))
            x = x + sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif mode < 0.70:
            # remove worst / overloaded, reinsert into voids
            k = int(rng.integers(2, 11))
            score = maxper + 0.015 * contacts + 0.005 * rng.standard_normal(n)
            # force rattlers (0 contacts) to sometimes be moved too
            # boost rattlers occasionally
            if rng.random() < 0.4:
                ratt = np.where(contacts == 0)[0]
                score[ratt] += 0.05
            idx = np.argsort(-score)[:k]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[idx] = False
            Xkeep = x[keep_mask]
            # greedy farthest insertion
            cur = Xkeep.copy()
            newpts = []
            for _ in range(k):
                nc = 2500
                cand = rng.standard_normal((nc, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                dots = cand @ cur.T
                md = dots.max(axis=1)
                order = np.argsort(md)
                # randomize among top few
                if rng.random() < 0.5:
                    pick = order[0]
                else:
                    pick = order[int(rng.integers(0, min(8, nc)))]
                chosen = cand[pick]
                newpts.append(chosen)
                cur = np.vstack([cur, chosen[None, :]])
            x[keep_mask] = Xkeep
            # place newpts
            x[idx] = np.array(newpts)
        elif mode < 0.87:
            # cap reshuffle: pick random direction, randomize points inside cap
            u = rng.standard_normal(d); u /= np.linalg.norm(u)
            dots = x @ u
            k = int(rng.integers(5, 18))
            # threshold to get ~k points: take top k closest to u
            idx = np.argsort(-dots)[:k]
            # replace with random points in cap around u with some spread
            spread = float(rng.choice(np.array([0.3, 0.6, 1.0, 1.5])))
            for j in idx:
                cand = u + spread * rng.standard_normal(d)
                cand /= np.linalg.norm(cand)
                x[j] = cand
            # also jitter neighbours slightly
            x += 0.01 * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        else:
            # differential twist + jitter: rotate one hemisphere more
            u = rng.standard_normal(d); u /= np.linalg.norm(u)
            proj = x @ u
            # random skew direction orthogonal
            w = rng.standard_normal(d); w -= (w @ u) * u; w /= np.linalg.norm(w) + 1e-12
            ang = 0.15 + 0.35 * rng.random()
            # shift proportional to proj
            x = x + ang * (proj[:, None] * w[None, :]) * (0.5 + 0.5 * rng.random())
            x += 0.02 * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)

        # anneal params
        steps = int(rng.choice(np.array([1200, 1800, 2500, 3500])))
        # adapt steps to remaining time
        if remain < 8 and steps > 1500:
            steps = 1200
        k0 = float(rng.choice(np.array([12.0, 20.0, 35.0])))
        k1 = float(rng.choice(np.array([400.0, 800.0, 1500.0, 2500.0])))
        lr0 = 0.02
        lr1 = 0.002
        mu = 0.88 if rng.random() < 0.5 else 0.92
        noise = 0.004 if rng.random() < 0.6 else 0.0
        if mode < 0.30 and rng.random() < 0.5:
            noise = 0.01

        x = anneal(x, steps, k0, k1, lr0, lr1, mu, noise, rng)
        v = max_cos(x)
        # screen: if promising, short LP polish
        # threshold: within 0.004 of best or better
        if v < best_val + 0.004:
            # short polish 1.2s (or less if time low)
            pt = 1.5 if remain > 20 else 0.8
            xp, vp = lp_polish(x, pt)
            if vp < best_val:
                best_x, best_val = xp.copy(), vp
                # early exit if big win? continue to squeeze
            # also update inc for diversity
            if v < inc_val:
                inc, inc_val = x.copy(), v
        else:
            if v < inc_val - 0.001:
                inc, inc_val = x.copy(), v

        # occasional refresh of inc to best to avoid drift
        if trial % 25 == 0:
            inc = best_x.copy(); inc_val = best_val

    # final long polish of best
    time_left = BUDGET - (time.time() - t0) - 2.0
    if time_left < 5:
        time_left = 5
    if time_left > 36:
        time_left = 36
    bx, bv = lp_polish(best_x, time_left)
    if bv < best_val:
        best_x, best_val = bx, bv
    best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
