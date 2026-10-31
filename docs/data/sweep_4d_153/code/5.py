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
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        cfull = x @ x.T
        c = cfull[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) == 0:
            radius *= 0.5
            continue
        I = iu[0][sel]
        J = iu[1][sel]
        cc = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def adam_refine(x, steps, k0, k1, lr0, lr1):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lrk = (k1 / k0) ** (1.0 / max(steps - 1, 1)) if steps > 1 else 1.0
    lrr = (lr1 / lr0) ** (1.0 / max(steps - 1, 1)) if steps > 1 else 1.0
    k = k0
    lr = lr0
    with np.errstate(under='ignore', over='ignore'):
        for s in range(steps):
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            cmax = float(G.max())
            E = np.exp(k * (G - cmax))
            np.fill_diagonal(E, 0.0)
            S = float(E.sum() * 0.5)
            if S <= 0 or not np.isfinite(S):
                break
            grad = (E @ x) / S
            dot = (grad * x).sum(axis=1, keepdims=True)
            gt = grad - dot * x
            m = b1 * m + (1 - b1) * gt
            v = b2 * v + (1 - b2) * (gt * gt)
            mhat = m / (1 - b1 ** (s + 1))
            vhat = v / (1 - b2 ** (s + 1))
            step = mhat / (np.sqrt(vhat) + eps)
            x = x - lr * step
            x = x / np.linalg.norm(x, axis=1, keepdims=True)
            k *= lrk
            lr *= lrr
    return x


def random_sphere(rng, n, d):
    a = rng.standard_normal((n, d))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    return a


def solve():
    t0 = time.time()
    DEADLINE = t0 + 168.0
    rng = np.random.default_rng(0)
    # load start
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = random_sphere(rng, 153, 4)
    if xb.shape != (153, 4):
        # try to fix / fallback
        xb = random_sphere(rng, 153, 4)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    vb = max_cos(xb)
    try:
        xr = np.load("record.npy").astype(float)
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        vr = max_cos(xr)
        if vr < vb:
            xb, vb = xr, vr
    except Exception:
        pass
    pool = [(vb, xb.copy())]
    it = 0
    # reserve ~22s for final polish
    while time.time() < DEADLINE - 22.0:
        it += 1
        remain = DEADLINE - 22.0 - time.time()
        if remain <= 1.0:
            break
        # pick base: mostly best, sometimes 2nd best
        pool_sorted = sorted(pool, key=lambda t: t[0])
        if len(pool_sorted) > 1 and (it % 4 == 0):
            base = pool_sorted[1][1]
        else:
            base = pool_sorted[0][1]
        n, d = base.shape
        mode = it % 6
        xt = base.copy()
        if mode == 0:
            sig = 0.035
            xt = xt + sig * rng.standard_normal((n, d))
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)
            steps, k0, k1, lr0, lr1 = 700, 15.0, 250.0, 0.02, 0.002
        elif mode == 1:
            sig = 0.11
            xt = xt + sig * rng.standard_normal((n, d))
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)
            steps, k0, k1, lr0, lr1 = 900, 12.0, 300.0, 0.025, 0.002
        elif mode == 2:
            s = 6
            idx = rng.choice(n, size=s, replace=False)
            xt[idx] = random_sphere(rng, s, d)
            # small jitter rest
            xt = xt + 0.01 * rng.standard_normal((n, d))
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)
            steps, k0, k1, lr0, lr1 = 900, 15.0, 280.0, 0.02, 0.002
        elif mode == 3:
            s = 12
            idx = rng.choice(n, size=s, replace=False)
            xt[idx] = random_sphere(rng, s, d)
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)
            steps, k0, k1, lr0, lr1 = 1000, 12.0, 300.0, 0.025, 0.002
        elif mode == 4:
            # targeted shake of overloaded cluster
            G = xt @ xt.T
            np.fill_diagonal(G, -2.0)
            thr = vb - 0.008
            mask = G > thr
            deg = mask.sum(axis=1)
            order = np.argsort(-deg)
            k_top = 10
            top = order[:k_top]
            sig_top = 0.16
            sig_rest = 0.015
            noise = rng.standard_normal((n, d)) * sig_rest
            noise[top] = rng.standard_normal((k_top, d)) * sig_top
            # also shake neighbors of top
            neigh = set()
            for ii in top:
                js = np.where(mask[ii])[0]
                for jj in js:
                    neigh.add(int(jj))
            neigh = list(neigh)[:24]
            for jj in neigh:
                noise[jj] += rng.standard_normal(d) * 0.08
            xt = xt + noise
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)
            steps, k0, k1, lr0, lr1 = 1000, 12.0, 300.0, 0.025, 0.002
        else:
            # rattler reinsertion + global soft move
            G = xt @ xt.T
            np.fill_diagonal(G, -2.0)
            cm = G.max(axis=1)
            # rattlers = smallest cm (largest hole around them)
            ratt = np.argsort(cm)[:3]
            # farthest hole search
            cand = random_sphere(rng, 20000, d)
            # min cosine to existing (excluding rattlers to replace?)
            keep = np.ones(n, dtype=bool)
            keep[ratt[0]] = False
            sims = cand @ xt[keep].T
            best_cos = sims.max(axis=1)
            best_idx = int(np.argmin(best_cos))
            xt[ratt[0]] = cand[best_idx]
            if len(ratt) > 1:
                xt[ratt[1]] += 0.1 * rng.standard_normal((d,))
                xt[ratt[1]] /= np.linalg.norm(xt[ratt[1]])
            xt = xt + 0.02 * rng.standard_normal((n, d))
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)
            steps, k0, k1, lr0, lr1 = 800, 15.0, 260.0, 0.02, 0.002
        # smooth anneal
        xt = adam_refine(xt, steps, k0, k1, lr0, lr1)
        vt = max_cos(xt)
        # maintain pool (keep up to 4 diverse, within 0.015 of best)
        if len(pool) < 4:
            # avoid duplicates
            if all(np.linalg.norm(xt - p[1]) > 1e-3 for p in pool):
                pool.append((vt, xt.copy()))
                pool = sorted(pool, key=lambda t: t[0])[:4]
        else:
            worst = max(p[0] for p in pool)
            if vt < worst - 5e-6:
                # replace worst
                idxw = int(np.argmax([p[0] for p in pool]))
                pool[idxw] = (vt, xt.copy())
        # LP screening if promising
        cur_best = min(p[0] for p in pool)
        if vt < cur_best + 0.004:
            # short polish, but respect deadline
            sec = min(2.2, (DEADLINE - 22.0 - time.time()) - 0.2)
            if sec > 0.5:
                xp = lp_polish(xt, sec)
                vp = max_cos(xp)
                if vp < cur_best - 1e-12:
                    # new global best
                    pool.append((vp, xp.copy()))
                    pool = sorted(pool, key=lambda t: t[0])[:4]
                elif len(pool) < 4 and vp < cur_best + 0.015:
                    pool.append((vp, xp.copy()))
                    pool = sorted(pool, key=lambda t: t[0])[:4]
        # keep loop time-safe
        if time.time() >= DEADLINE - 22.0:
            break
    best = sorted(pool, key=lambda t: t[0])[0][1]
    # final long polish with remaining time
    rem = DEADLINE - time.time() - 2.0
    if rem > 3.0:
        best = lp_polish(best, rem)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
