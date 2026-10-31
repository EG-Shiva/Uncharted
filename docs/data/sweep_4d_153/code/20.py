import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def indiv_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)


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
        sel = c > m - max(20.0 * radius, 1e-9)
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
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
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


def smooth_refine(x, stages, deadline=None):
    n, d = x.shape
    y0 = x.ravel().copy()
    for k, maxit in stages:
        if deadline is not None and time.time() > deadline - 0.5:
            break
        def fun(yf):
            Y = yf.reshape(n, d)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Y / rn
            G = X @ X.T
            iu = np.triu_indices(n, 1)
            gu = G[iu]
            cm = float(gu.max())
            # scaled exp
            E = np.exp(k * (gu - cm))
            S = float(E.sum())
            if not np.isfinite(S) or S <= 0:
                return 1e6, np.zeros_like(yf)
            f = cm + np.log(S + 1e-300) / k
            # weights on upper, scatter to full
            W = np.zeros((n, n))
            W[iu] = E / S
            W = W + W.T
            GX = W @ X  # df/dX
            dot = (GX * X).sum(axis=1, keepdims=True)
            GT = GX - dot * X
            GY = GT / rn
            return f, GY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxit, "ftol": 1e-14, "gtol": 1e-9, "maxfun": 100000})
            y0 = res.x
        except Exception:
            break
    X = y0.reshape(n, d)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X


def random_sphere(rng, n, d):
    a = rng.standard_normal((n, d))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    return a


def largest_hole(x_keep, rng, ncand=20000):
    d = x_keep.shape[1]
    cand = random_sphere(rng, ncand, d)
    sims = cand @ x_keep.T
    bc = sims.max(axis=1)
    return cand[int(np.argmin(bc))]


def solve():
    t0 = time.time()
    DEADLINE = t0 + 168.0
    rng = np.random.default_rng(12345)
    n, d = 153, 4
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = random_sphere(rng, n, d)
    if xb.shape != (n, d):
        xb = random_sphere(rng, n, d)
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
    # quick ensure local opt baseline is polished a bit if time allows
    it = 0
    STAGES_FAST = [(60, 120), (180, 150), (450, 180)]
    STAGES_SHORT = [(60, 80), (200, 100), (450, 120)]
    while time.time() < DEADLINE - 24.0:
        it += 1
        if time.time() > DEADLINE - 24.5:
            break
        pool = sorted(pool, key=lambda t: t[0])
        cur_best = pool[0][0]
        # pick base
        if len(pool) > 1 and (it % 5 == 0):
            base = pool[1][1]
        else:
            # mostly best, sometimes random among pool
            if len(pool) > 2 and (it % 7 == 0):
                base = pool[rng.integers(len(pool))][1]
            else:
                base = pool[0][1]
        xt = base.copy()
        mode = it % 7
        G = xt @ xt.T
        np.fill_diagonal(G, -2.0)
        cm_row = G.max(axis=1)
        if mode == 0:
            sig = 0.03
            xt = xt + sig * rng.standard_normal((n, d))
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)
            stages = STAGES_FAST
        elif mode == 1:
            sig = 0.09
            xt = xt + sig * rng.standard_normal((n, d))
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)
            stages = STAGES_FAST
        elif mode == 2:
            # block-coherent move of overloaded cluster
            thr = float(G.max()) - 0.012
            deg = (G > thr).sum(axis=1)
            seed = int(np.argmax(deg + 0.3 * rng.standard_normal(n)))
            neigh1 = np.where(G[seed] > thr)[0]
            cluster = set([seed] + neigh1.tolist())
            # second hop to reach size 8..20
            for j in list(neigh1):
                if len(cluster) >= 18:
                    break
                n2 = np.where(G[j] > thr)[0]
                for qq in n2:
                    cluster.add(int(qq))
                    if len(cluster) >= 18:
                        break
            cluster = list(cluster)
            common = rng.standard_normal(d) * 0.09
            xt[cluster] += common + 0.02 * rng.standard_normal((len(cluster), d))
            mask = np.ones(n, dtype=bool)
            mask[cluster] = False
            xt[mask] += 0.006 * rng.standard_normal((int(mask.sum()), d))
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)
            stages = STAGES_FAST
        elif mode == 3:
            # block move seeded at random point with medium threshold
            thr = float(G.max()) - 0.02
            seed = int(rng.integers(n))
            neigh1 = np.where(G[seed] > thr)[0]
            cluster = set([seed] + neigh1.tolist())
            # expand by closest cosines if too small
            if len(cluster) < 8:
                order = np.argsort(-G[seed])[:8]
                for qq in order:
                    cluster.add(int(qq))
            cluster = list(cluster)[:22]
            common = rng.standard_normal(d) * 0.12
            xt[cluster] += common + 0.025 * rng.standard_normal((len(cluster), d))
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)
            stages = STAGES_FAST
        elif mode == 4:
            # worst-point hole reinsertion (2-3 points)
            kbad = 3 if (it % 2 == 0) else 2
            worst = np.argsort(-cm_row)[:kbad]
            keep = np.ones(n, dtype=bool)
            keep[worst] = False
            # sequential greedy
            x_keep = xt[keep].copy()
            newpts = []
            for _ in range(kbad):
                h = largest_hole(x_keep, rng, ncand=25000)
                # slight random offset to avoid exact lattice
                h = h + 0.005 * rng.standard_normal(d)
                h /= np.linalg.norm(h)
                newpts.append(h)
                x_keep = np.vstack([x_keep, h[None, :]])
            for w, h in zip(worst, newpts):
                xt[w] = h
            # soft jitter rest
            xt += 0.012 * rng.standard_normal((n, d))
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)
            stages = STAGES_SHORT
        elif mode == 5:
            s = 8 if (it % 2 == 0) else 14
            idx = rng.choice(n, size=s, replace=False)
            xt[idx] = random_sphere(rng, s, d)
            xt += 0.008 * rng.standard_normal((n, d))
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)
            stages = STAGES_FAST
        else:
            # antithetic pair: perturb along weakest eigenvector? use random tangent + hole for rattler
            ratt = np.argsort(cm_row)[:2]
            # move rattlers to holes w.r.t rest
            for r in ratt[:1]:
                keep = np.ones(n, dtype=bool)
                keep[r] = False
                h = largest_hole(xt[keep], rng, ncand=25000)
                xt[r] = h
            xt += 0.025 * rng.standard_normal((n, d))
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)
            stages = STAGES_SHORT
        # smooth anneal with L-BFGS
        xt = smooth_refine(xt, stages, deadline=DEADLINE - 22.0)
        vt = max_cos(xt)
        # pool update
        pool = sorted(pool, key=lambda t: t[0])
        worst = pool[-1][0] if len(pool) >= 4 else 1e9
        if len(pool) < 4:
            # diversity check
            if all(float(np.linalg.norm(xt - p[1])) > 1e-3 for p in pool):
                pool.append((vt, xt.copy()))
        elif vt < worst - 2e-6:
            idxw = int(np.argmax([p[0] for p in pool]))
            pool[idxw] = (vt, xt.copy())
        pool = sorted(pool, key=lambda t: t[0])[:4]
        cur_best = pool[0][0]
        # LP screening if promising
        if vt < cur_best + 0.004:
            sec = min(1.6, (DEADLINE - 24.0 - time.time()) - 0.1)
            if sec > 0.4:
                xp = lp_polish(xt, sec)
                vp = max_cos(xp)
                if vp < pool[0][0] - 1e-12:
                    pool.append((vp, xp.copy()))
                    pool = sorted(pool, key=lambda t: t[0])[:4]
                elif len(pool) < 4 and vp < pool[0][0] + 0.015:
                    pool.append((vp, xp.copy()))
                    pool = sorted(pool, key=lambda t: t[0])[:4]
    pool = sorted(pool, key=lambda t: t[0])
    best = pool[0][1]
    rem = DEADLINE - time.time() - 1.5
    if rem > 3.0:
        best = lp_polish(best, rem)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
