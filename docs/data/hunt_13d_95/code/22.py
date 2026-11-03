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


def contact_counts(x, mc, tol=0.02):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return (g > mc - tol).sum(axis=1)


def best_hole(others, rng, n_samples=2000):
    d = others.shape[1]
    cand = rng.standard_normal((n_samples, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    md = dots.max(axis=1)
    j = int(np.argmin(md))
    return cand[j]


def gd_refine(X, steps, T0, T1, lr0, lr1, mu=0.9, noise=0.0, rng=None, freeze=None):
    n, d = X.shape
    X = X.copy()
    V = np.zeros_like(X)
    lT0, lT1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    fr = None
    if freeze is not None:
        fr = freeze.astype(bool)
    for s in range(steps):
        f = s / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = float(G.max())
        W = np.exp((G - mx) / T)
        np.fill_diagonal(W, 0.0)
        sw = W.sum()
        if not np.isfinite(sw) or sw == 0:
            break
        W /= sw
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        if fr is not None:
            Gt[fr] = 0.0
        if noise > 0 and rng is not None:
            # tangent noise
            N = rng.standard_normal((n, d))
            N -= np.sum(N * X, axis=1, keepdims=True) * X
            Gt = Gt + noise * N
            if fr is not None:
                Gt[fr] = 0.0
        V = mu * V + Gt
        X = X - lr * V
        if fr is not None:
            pass
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


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
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
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


def solve():
    t0 = time.time()
    BUDGET = 172.0
    EXPL_T = 128.0
    rng = np.random.default_rng(0)

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
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        cands.append(xb)
    if xr is not None:
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        cands.append(xr)
    if not cands:
        x0 = rng.standard_normal((95, 13))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        cands.append(x0)
    vals = [max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    best_val = float(vals[bi])
    cur = x_best.copy()
    cur_val = best_val
    n, d = x_best.shape

    sigmas = [0.01, 0.03, 0.07, 0.15, 0.30, 0.02, 0.05, 0.12]
    trial = 0
    lp_count = 0
    while time.time() - t0 < EXPL_T:
        trial += 1
        remain = EXPL_T - (time.time() - t0)
        if remain < 1.5:
            break
        if time.time() - t0 > BUDGET - 40:
            break
        u = rng.random()
        Xt = cur.copy()
        if u < 0.50:
            # global shake + fast GD
            sigma = sigmas[rng.integers(0, len(sigmas))]
            if rng.random() < 0.30:
                ksub = int(rng.integers(4, 16))
                idx = rng.choice(n, size=ksub, replace=False)
                Xt[idx] += sigma * 2.0 * rng.standard_normal((ksub, d))
            else:
                Xt += sigma * rng.standard_normal((n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            # anneal params depend on sigma
            if sigma >= 0.12:
                Xt = gd_refine(Xt, 800, 0.018, 0.001, 0.12, 0.008, mu=0.9)
            elif sigma >= 0.05:
                Xt = gd_refine(Xt, 700, 0.012, 0.0008, 0.10, 0.006, mu=0.9)
            else:
                Xt = gd_refine(Xt, 600, 0.008, 0.0006, 0.08, 0.005, mu=0.9)
        elif u < 0.75:
            # kill worst / overloaded / rattler re-insertion
            cc = contact_counts(cur, cur_val, tol=0.02)
            ppm = per_point_max(cur)
            r = rng.random()
            if r < 0.45:
                k = int(rng.integers(1, 4)) if rng.random() < 0.7 else int(rng.integers(4, 7))
                idx = np.argsort(-ppm)[:k]
                if k > 1 and rng.random() < 0.4:
                    idx = idx.copy()
                    idx[-1] = int(rng.integers(0, n))
            elif r < 0.65:
                k = int(rng.integers(1, 4))
                idx = np.argsort(-cc)[:k]
            elif r < 0.85:
                k = int(rng.integers(1, 3))
                # rattlers: smallest ppm
                idx = np.argsort(ppm)[:k]
            else:
                k = int(rng.integers(2, 5))
                idx = rng.choice(n, size=k, replace=False)
            use_hole = rng.random() < 0.75
            for j in idx:
                if use_hole:
                    mask = np.ones(n, dtype=bool)
                    mask[j] = False
                    others = Xt[mask]
                    ns = 4000 if len(idx) == 1 else 2000
                    Xt[j] = best_hole(others, rng, n_samples=ns)
                else:
                    v = rng.standard_normal(d)
                    v /= np.linalg.norm(v)
                    Xt[j] = v
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = gd_refine(Xt, 700, 0.014, 0.0007, 0.10, 0.006, mu=0.9)
        else:
            # cap-localized shake: spatial cluster
            uu = rng.standard_normal(d)
            uu /= np.linalg.norm(uu)
            proj = cur @ uu
            k = int(rng.integers(5, 16))
            idx = np.argsort(-proj)[:k]
            if rng.random() < 0.5:
                # reinit cluster randomly
                for j in idx:
                    if rng.random() < 0.7:
                        mask = np.ones(n, dtype=bool)
                        mask[j] = False
                        Xt[j] = best_hole(Xt[mask], rng, n_samples=1500)
                    else:
                        v = rng.standard_normal(d)
                        Xt[j] = v / np.linalg.norm(v)
            else:
                sigma = 0.25 + 0.25 * rng.random()
                Xt[idx] += sigma * rng.standard_normal((k, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            # two-stage: freeze rest briefly then full
            freeze = np.ones(n, dtype=bool)
            freeze[idx] = False
            Xt = gd_refine(Xt, 300, 0.015, 0.002, 0.12, 0.01, mu=0.9, freeze=freeze)
            Xt = gd_refine(Xt, 500, 0.010, 0.0007, 0.10, 0.006, mu=0.9)

        gval = max_cos(Xt)
        # screen: only LP if promising
        thresh_lp = min(cur_val + 0.002, best_val + 0.008)
        # periodically force LP to keep diversity
        force = (trial % 25 == 0)
        if (gval < thresh_lp or force) and (time.time() - t0 < EXPL_T - 1.0):
            psec = 1.0
            if gval < best_val + 0.001:
                psec = 1.4
            try:
                Xp = lp_polish(Xt, seconds=psec, radius_init=0.006)
            except Exception:
                continue
            lp_count += 1
            v = max_cos(Xp)
            if v < best_val - 1e-12:
                x_best = Xp.copy()
                best_val = v
                cur = Xp.copy()
                cur_val = v
            elif v < cur_val:
                cur = Xp.copy()
                cur_val = v
            else:
                if rng.random() < 0.05 and v < cur_val + 0.008:
                    cur = Xp.copy()
                    cur_val = v
                if rng.random() < 0.10:
                    cur = x_best.copy()
                    cur_val = best_val
        else:
            # fast-only acceptance for cur drift (no LP)
            if gval < cur_val:
                # keep GD result as cur to propagate (will be LP'd later if good)
                # only move cur without LP occasionally to save time
                if rng.random() < 0.3:
                    cur = Xt.copy()
                    cur_val = gval
            elif rng.random() < 0.02:
                cur = Xt.copy()
                cur_val = gval
            if rng.random() < 0.05:
                cur = x_best.copy()
                cur_val = best_val

    # final rattler-targeted tries if time allows
    # short extra hops on best before long polish
    while time.time() - t0 < BUDGET - 32:
        # try single-hole move on current best's worst/rattler point
        ppm = per_point_max(x_best)
        # alternate worst and best (rattler)
        if rng.random() < 0.5:
            j = int(np.argmax(ppm))
        else:
            j = int(np.argmin(ppm))
        Xt = x_best.copy()
        mask = np.ones(n, dtype=bool)
        mask[j] = False
        Xt[j] = best_hole(Xt[mask], rng, n_samples=4000)
        Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
        Xt = gd_refine(Xt, 600, 0.010, 0.0006, 0.09, 0.005, mu=0.9)
        gval = max_cos(Xt)
        if gval < best_val + 0.003:
            try:
                Xp = lp_polish(Xt, seconds=1.2, radius_init=0.005)
            except Exception:
                break
            v = max_cos(Xp)
            if v < best_val - 1e-12:
                x_best = Xp.copy()
                best_val = v
        else:
            break
        if time.time() - t0 > BUDGET - 34:
            break

    remain = BUDGET - (time.time() - t0)
    final_sec = float(np.clip(remain - 2.0, 5.0, 30.0))
    if final_sec > 1.0:
        try:
            x_best = lp_polish(x_best, seconds=final_sec * 0.6, radius_init=1e-3)
            rest = BUDGET - (time.time() - t0) - 1.0
            if rest > 2.0:
                x_best = lp_polish(x_best, seconds=min(rest, 15.0), radius_init=5e-4)
        except Exception:
            pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
