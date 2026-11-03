import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _per_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)


def _balanced_anneal(x, steps, beta0, beta1, lr0, lr1):
    n, d = x.shape
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** f if beta0 > 0 else beta1
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = G.max(axis=1, keepdims=True)
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        S = E.sum(axis=1, keepdims=True) + 1e-300
        P = E / S
        A = 0.5 * (P + P.T)
        Grad = A @ x
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Tan = Grad - dot * x
        x = x - lr * Tan
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _global_anneal(x, steps, beta0, beta1, lr0, lr1):
    n, d = x.shape
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** f if beta0 > 0 else beta1
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ x
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Tan = Grad - dot * x
        # scale to avoid huge norms when beta large
        tn = np.sqrt((Tan * Tan).sum()) + 1e-30
        # normalize step roughly
        x = x - lr * Tan / max(1.0, tn * 0.2)
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _refine_void(v0, Xrem, beta=120.0, steps=150, lr=0.12):
    v = v0 / (np.linalg.norm(v0) + 1e-30)
    for t in range(steps):
        dots = Xrem @ v
        mv = dots.max()
        W = np.exp(beta * (dots - mv))
        S = W.sum() + 1e-300
        g = (W @ Xrem) / S
        gt = g - np.dot(g, v) * v
        st = lr / (1.0 + t * 0.02)
        v = v - st * gt
        nrm = np.linalg.norm(v) + 1e-30
        v = v / nrm
    return v


def _largest_void_vector(Xrem, rng, ncand=20000, beta=120.0):
    nrem, d = Xrem.shape
    cand = rng.standard_normal(size=(ncand, d))
    cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
    smax = (cand @ Xrem.T).max(axis=1)
    # take best few and refine
    k = 4
    idx = np.argpartition(smax, k)[:k]
    best_v = None
    best_s = 1e9
    for ii in idx:
        v = _refine_void(cand[ii], Xrem, beta=beta, steps=120, lr=0.1)
        s = float((Xrem @ v).max())
        if s < best_s:
            best_s = s
            best_v = v
    return best_v


def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        c = Gfull[iu]
        m = c.max()
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 3000:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:3000]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1
        ).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv)
        ).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(
                obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
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
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    rng = np.random.default_rng(0)
    t0 = time.time()
    deadline = t0 + 168.0
    n, d = 138, 8
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal(size=(n, d))
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)

    # Stage 0: direct rebalancing of incumbent (no shake) to a load-sharing basin
    try:
        y0 = best.copy()
        y0 = _balanced_anneal(y0, steps=1200, beta0=25.0, beta1=350.0, lr0=0.07, lr1=0.006)
        y0 = _global_anneal(y0, steps=300, beta0=80.0, beta1=300.0, lr0=0.02, lr1=0.004)
        v0 = _max_cos(y0)
        if v0 < best_val + 0.01 and time.time() < deadline - 8:
            y0p = _lp_polish(y0, min(deadline - 6.0, time.time() + 4.0))
            vp = _max_cos(y0p)
            if vp < best_val - 1e-12:
                best, best_val = y0p, vp
            elif v0 < best_val - 1e-12:
                best, best_val = y0, v0
    except Exception:
        pass

    it = 0
    while time.time() < deadline - 6.0 and it < 200:
        it += 1
        now = time.time()
        time_left = deadline - 6.0 - now
        if time_left < 1.0:
            break
        try:
            if it % 12 == 0:
                # fresh random balanced basin
                y = rng.standard_normal(size=(n, d))
                y = y / np.linalg.norm(y, axis=1, keepdims=True)
                y = _balanced_anneal(y, steps=1400, beta0=20.0, beta1=400.0, lr0=0.08, lr1=0.005)
                y = _global_anneal(y, steps=250, beta0=60.0, beta1=280.0, lr0=0.02, lr1=0.004)
                v = _max_cos(y)
                if v < best_val + 0.015 and time.time() < deadline - 6.0:
                    y = _lp_polish(y, min(deadline - 6.0, time.time() + 2.5))
                    v = _max_cos(y)
                if v < best_val - 1e-12:
                    best, best_val = y.copy(), v
                continue
            # analyze incumbent topology
            per = _per_max(best)
            order_worst = np.argsort(-per)
            order_best = np.argsort(per)  # isolated first
            y = best.copy()
            if it % 3 == 0:
                # heavy destroy: worst cluster + rattler rebirth into voids
                nw = 14
                worst_set = order_worst[:nw]
                # neighbors of the very worst
                Gw = best @ best[order_worst[0]]
                neigh = np.argsort(-Gw)[:14]
                target = np.unique(np.concatenate([worst_set, neigh]))
                sigma = 0.35
                y[target] = y[target] + sigma * rng.standard_normal(size=(target.shape[0], d))
                # pick 2 victims among worst 10 for void reinsertion
                for _ in range(2):
                    pool = order_worst[:10]
                    victim = int(pool[rng.integers(len(pool))])
                    Xrem = np.delete(y, victim, axis=0)
                    vv = _largest_void_vector(Xrem, rng, ncand=15000, beta=130.0)
                    y[victim] = vv
                # also jitter a few rattlers lightly
                for rr in order_best[:5]:
                    if rng.random() < 0.6:
                        y[rr] = y[rr] + 0.15 * rng.standard_normal(size=(d,))
                y = y / np.linalg.norm(y, axis=1, keepdims=True)
                y = _balanced_anneal(y, steps=900, beta0=25.0, beta1=380.0, lr0=0.07, lr1=0.006)
                v = _max_cos(y)
                if v < best_val + 0.004:
                    y = _lp_polish(y, min(deadline - 6.0, time.time() + 1.6))
                    v = _max_cos(y)
                if v < best_val - 1e-12:
                    best, best_val = y.copy(), v
            elif it % 3 == 1:
                # moderate biased shake
                sigma = 0.12
                # bias to worst half
                cand_idx = np.concatenate([order_worst[:60], rng.choice(n, size=20, replace=False)])
                sel = rng.choice(cand_idx, size=28, replace=False)
                y[sel] = y[sel] + sigma * rng.standard_normal(size=(len(sel), d))
                y = y / np.linalg.norm(y, axis=1, keepdims=True)
                y = _balanced_anneal(y, steps=700, beta0=30.0, beta1=320.0, lr0=0.06, lr1=0.006)
                v = _max_cos(y)
                if v < best_val + 0.003:
                    y = _lp_polish(y, min(deadline - 6.0, time.time() + 1.4))
                    v = _max_cos(y)
                if v < best_val - 1e-12:
                    best, best_val = y.copy(), v
            else:
                # pure void repair: replace 1-2 worst by largest voids, short rebalance
                for _ in range(2 if it % 6 == 5 else 1):
                    pool = order_worst[:12]
                    victim = int(pool[rng.integers(len(pool))])
                    Xrem = np.delete(y, victim, axis=0)
                    vv = _largest_void_vector(Xrem, rng, ncand=20000, beta=120.0)
                    y[victim] = vv
                y = _balanced_anneal(y, steps=600, beta0=35.0, beta1=300.0, lr0=0.05, lr1=0.006)
                v = _max_cos(y)
                if v < best_val + 0.0025:
                    y = _lp_polish(y, min(deadline - 6.0, time.time() + 1.4))
                    v = _max_cos(y)
                if v < best_val - 1e-12:
                    best, best_val = y.copy(), v
        except Exception:
            continue

    try:
        if time.time() < deadline - 1.0:
            best = _lp_polish(best, min(deadline, time.time() + 4.5))
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
