import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def per_point_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)


def lp_polish(x, seconds, radius_init=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = float(radius_init)
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.05:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = c > thresh
        if int(sel.sum()) > 1500:
            idx = np.argpartition(c, -1500)[-1500:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
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


def smooth_refine(X, steps, T0, T1, lr0, lr1, mu=0.9):
    n, d = X.shape
    V = np.zeros_like(X)
    lT0 = np.log(T0)
    lT1 = np.log(T1)
    ll0 = np.log(lr0)
    ll1 = np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = float(G.max())
        W = np.exp((G - mx) / T)
        np.fill_diagonal(W, 0.0)
        ssum = W.sum()
        if not np.isfinite(ssum) or ssum == 0:
            break
        W /= ssum
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        gn = float(np.linalg.norm(Gt))
        if not np.isfinite(gn) or gn == 0:
            break
        # normalize to per-point rms 1 so lr ~ angular step
        Gt = Gt / (gn / np.sqrt(float(n)))
        V = mu * V + Gt
        X = X - lr * V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def best_hole(others, rng, n_samples=3000):
    d = others.shape[1]
    cand = rng.standard_normal(size=(n_samples, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    md = dots.max(axis=1)
    j = int(np.argmin(md))
    return cand[j]


def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline = t0 + BUDGET
    EXPL_END = t0 + 132.0
    rng = np.random.default_rng(0)

    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None

    inits = []
    if xb is not None:
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        inits.append(xb)
    if xr is not None:
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        inits.append(xr)
    if not inits:
        r0 = rng.standard_normal((95, 13))
        r0 /= np.linalg.norm(r0, axis=1, keepdims=True)
        inits.append(r0)

    vals = [max_cos(c) for c in inits]
    bi = int(np.argmin(vals))
    x_best = inits[bi].copy()
    best_val = float(vals[bi])
    n, d = x_best.shape

    trial = 0
    sigmas = [0.01, 0.03, 0.06, 0.12, 0.25, 0.45]
    while time.time() < EXPL_END and time.time() < deadline - 35:
        trial += 1
        remain = EXPL_END - time.time()
        if remain < 2.0:
            break
        mode = trial % 7
        Xt = x_best.copy()
        if mode in (0, 1, 2):
            sigma = sigmas[rng.integers(0, len(sigmas))]
            if mode == 0:
                Xt = Xt + sigma * rng.standard_normal(Xt.shape)
                Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
                steps = 900
                T0, T1 = 0.018, 0.0012
                lr0, lr1 = 0.03, 0.003
            elif mode == 1:
                ksub = int(rng.integers(4, 16))
                idx = rng.choice(n, size=ksub, replace=False)
                Xt[idx] += (sigma * 1.5) * rng.standard_normal((ksub, d))
                Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
                steps = 900
                T0, T1 = 0.014, 0.0009
                lr0, lr1 = 0.025, 0.0025
            else:
                Xt = Xt + sigma * rng.standard_normal(Xt.shape)
                Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
                steps = 1200
                T0, T1 = 0.022, 0.001
                lr0, lr1 = 0.035, 0.002
            Xt = smooth_refine(Xt, steps, T0, T1, lr0, lr1, mu=0.9)
        elif mode in (3, 4):
            # greedy hole re-insertion of overloaded / random points
            ppm = per_point_max(Xt)
            order = np.argsort(-ppm)
            if rng.random() < 0.7:
                k = int(rng.integers(1, 4)) if mode == 3 else int(rng.integers(3, 7))
                # mostly worst, sometimes random
                cand_idx = list(order[:8]) + list(rng.choice(n, size=8, replace=False))
                idx = np.array(rng.choice(cand_idx, size=min(k, len(cand_idx)), replace=False))
            else:
                k = int(rng.integers(1, 4))
                idx = rng.choice(n, size=k, replace=False)
            keep = np.ones(n, dtype=bool)
            keep[idx] = False
            others = Xt[keep]
            for j in idx:
                h = best_hole(others, rng, n_samples=4000)
                # local jitter around hole then pick best of few
                c2 = rng.standard_normal((400, d)) * 0.02 + h[None, :]
                c2 /= np.linalg.norm(c2, axis=1, keepdims=True)
                md0 = float((h[None, :] @ others.T).max())
                md2 = (c2 @ others.T).max(axis=1)
                bj = int(np.argmin(md2))
                if float(md2[bj]) < md0:
                    h = c2[bj]
                Xt[j] = h
                others = np.concatenate([others, h[None, :]], axis=0)
            # light shake then anneal
            Xt += 0.008 * rng.standard_normal(Xt.shape)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = smooth_refine(Xt, 900, 0.012, 0.0008, 0.025, 0.002, mu=0.9)
        elif mode == 5:
            # large jump: orthogonal-ish random perturbation
            sigma = 0.5
            Xt = Xt + sigma * rng.standard_normal(Xt.shape)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = smooth_refine(Xt, 1400, 0.025, 0.001, 0.04, 0.0025, mu=0.9)
        else:
            # rattler / low-contact targeted move + polish surroundings
            G = Xt @ Xt.T
            np.fill_diagonal(G, -2.0)
            mc = float(G.max())
            cc = (G > mc - 1e-9).sum(axis=1)
            low = np.where(cc <= 5)[0]
            if len(low) == 0:
                low = np.argsort(cc)[:5]
            k = min(len(low), int(rng.integers(1, 4)))
            idx = rng.choice(low, size=k, replace=False)
            keep = np.ones(n, dtype=bool)
            keep[idx] = False
            others = Xt[keep]
            for j in idx:
                h = best_hole(others, rng, n_samples=5000)
                Xt[j] = h
                others = np.concatenate([others, h[None, :]], axis=0)
            Xt += 0.01 * rng.standard_normal(Xt.shape)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = smooth_refine(Xt, 1000, 0.015, 0.0009, 0.03, 0.002, mu=0.9)

        v_pre = max_cos(Xt)
        # screen: only LP-polish if promising
        if v_pre > best_val + 0.03:
            continue
        # short LP screen
        t_rem = deadline - time.time()
        if t_rem < 6:
            break
        ps = 1.6 if v_pre < best_val + 0.005 else 1.0
        # give more time if very promising
        if v_pre < best_val:
            ps = 2.2
        Xt2 = lp_polish(Xt, seconds=min(ps, max(0.6, t_rem - 32)))
        v2 = max_cos(Xt2)
        if v2 < best_val - 1e-12:
            x_best = Xt2
            best_val = v2

    # final long squeeze on best
    t_rem = deadline - time.time()
    if t_rem > 3:
        x_best = lp_polish(x_best, seconds=max(1.0, t_rem - 1.0), radius_init=2e-3)
        best_val = max_cos(x_best)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
