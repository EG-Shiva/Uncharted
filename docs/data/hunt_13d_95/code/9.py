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


def lp_polish(x, seconds, radius_init=1e-3, cap=1500):
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
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        if int(sel.sum()) > cap:
            idx = np.argpartition(c, -cap)[-cap:]
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


def gd_anneal(X, steps, T0, T1, lr0, lr1, mu=0.9):
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
        mx = G.max()
        W = np.exp((G - mx) / T)
        np.fill_diagonal(W, 0.0)
        sw = W.sum()
        if not np.isfinite(sw) or sw == 0:
            break
        W /= sw
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        V = mu * V + Gt
        X = X - lr * V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def best_hole_vec(others, rng, n_samples=2500):
    d = others.shape[1]
    cand = rng.normal(size=(n_samples, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    md = dots.max(axis=1)
    j = int(np.argmin(md))
    return cand[j]


def solve():
    t0 = time.time()
    BUDGET = 170.0
    EXPL_T = 126.0
    rng = np.random.default_rng(7)

    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and xb.shape == (95, 13):
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        cands.append(xb)
    if xr is not None and xr.shape == (95, 13):
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        cands.append(xr)
    if len(cands) == 0:
        x0 = rng.normal(size=(95, 13))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        cands.append(x0)
    vals = [max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    best_val = float(vals[bi])
    cur = x_best.copy()
    cur_val = best_val
    n, d = x_best.shape

    it = 0
    # for cluster moves cache iu
    while time.time() - t0 < EXPL_T:
        it += 1
        remain = EXPL_T - (time.time() - t0)
        if remain < 1.0:
            break
        if time.time() - t0 > BUDGET - 40:
            break
        polish_sec = min(1.6, max(0.6, remain * 0.10))
        # reduce polish for fast screening cycles
        mode = rng.random()
        x_try = cur.copy()

        if mode < 0.30:
            # global shake + anneal
            sigma = float(rng.choice([0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.30]))
            x_try = cur + sigma * rng.normal(size=(n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            # anneal params depend on sigma
            if sigma < 0.05:
                x_try = gd_anneal(x_try, 400, 0.010, 0.0008, 0.08, 0.006)
            elif sigma < 0.15:
                x_try = gd_anneal(x_try, 600, 0.014, 0.0008, 0.10, 0.007)
            else:
                x_try = gd_anneal(x_try, 800, 0.018, 0.0010, 0.12, 0.008)
            rinit = 0.004 if sigma < 0.06 else 0.008
            do_lp = True
            # screen: if annealed value far worse, skip LP with prob
            v_pre = max_cos(x_try)
            if v_pre > cur_val + 0.03 and rng.random() < 0.7:
                do_lp = False
                # still maybe accept? skip
                if v_pre > cur_val + 0.05:
                    continue
            if do_lp:
                try:
                    x_pol = lp_polish(x_try, seconds=polish_sec, radius_init=rinit)
                except Exception:
                    continue
            else:
                x_pol = x_try
            v = max_cos(x_pol)
        elif mode < 0.55:
            # coherent cluster shake around overloaded hub
            ppm = per_point_max(cur)
            # pick hub biased to high ppm
            order = np.argsort(-ppm)
            # top-15 hub random
            hub = int(order[rng.integers(0, min(15, n))]) if rng.random() < 0.8 else int(rng.integers(0, n))
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            neigh = np.argsort(-G[hub])[:int(rng.integers(6, 14))]
            # coherent direction + individual jitter
            sigma_c = float(rng.choice([0.02, 0.05, 0.10, 0.20]))
            common = rng.normal(size=(d,))
            common /= np.linalg.norm(common)
            for jj in neigh:
                x_try[jj] = x_try[jj] + sigma_c * common + (sigma_c * 0.5) * rng.normal(size=(d,))
            # also shake hub more
            x_try[hub] += sigma_c * rng.normal(size=(d,))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_try = gd_anneal(x_try, 500, 0.012, 0.0007, 0.09, 0.006)
            try:
                x_pol = lp_polish(x_try, seconds=polish_sec, radius_init=0.006)
            except Exception:
                continue
            v = max_cos(x_pol)
        elif mode < 0.80:
            # kill worst k and reinsert at holes
            ppm = per_point_max(cur)
            order_hi = np.argsort(-ppm)
            r = rng.random()
            if r < 0.35:
                k = 1
            elif r < 0.65:
                k = 2
            elif r < 0.85:
                k = 3
            else:
                k = int(rng.integers(4, 7))
            if rng.random() < 0.70:
                idx = order_hi[:k].copy()
                if k > 1 and rng.random() < 0.4:
                    idx[-1] = int(rng.integers(0, n))
            else:
                # random or rattler kill
                if rng.random() < 0.5:
                    idx = rng.choice(n, size=k, replace=False)
                else:
                    order_lo = np.argsort(ppm)
                    idx = order_lo[:k]
            use_hole = rng.random() < 0.75
            for j in idx:
                if use_hole:
                    mask = np.ones(n, dtype=bool)
                    mask[int(j)] = False
                    others = x_try[mask]
                    ns = 3500 if k == 1 else 1800
                    x_try[int(j)] = best_hole_vec(others, rng, n_samples=ns)
                else:
                    vv = rng.normal(size=(d,))
                    vv /= np.linalg.norm(vv)
                    x_try[int(j)] = vv
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            # light anneal freezing others? just global short anneal
            x_try = gd_anneal(x_try, 350, 0.008, 0.0006, 0.07, 0.005)
            try:
                x_pol = lp_polish(x_try, seconds=polish_sec + 0.3, radius_init=0.010)
            except Exception:
                continue
            v = max_cos(x_pol)
        else:
            # large subspace / tangent jump
            sigma = float(rng.choice([0.10, 0.20, 0.35]))
            # random subset orthogonal-ish jump
            ksub = int(rng.integers(8, 25))
            idx = rng.choice(n, size=ksub, replace=False)
            noise = rng.normal(size=(ksub, d))
            # make tangential to current points for larger effective move
            for a, j in enumerate(idx):
                vv = noise[a]
                vv = vv - np.dot(vv, cur[j]) * cur[j]
                nrm = np.linalg.norm(vv)
                if nrm > 1e-12:
                    vv /= nrm
                x_try[j] = cur[j] + sigma * vv + 0.3 * sigma * rng.normal(size=(d,))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_try = gd_anneal(x_try, 700, 0.016, 0.0009, 0.11, 0.007)
            try:
                x_pol = lp_polish(x_try, seconds=polish_sec, radius_init=0.008)
            except Exception:
                continue
            v = max_cos(x_pol)

        if v < best_val - 1e-12:
            x_best = x_pol.copy()
            best_val = v
            cur = x_pol.copy()
            cur_val = v
        elif v < cur_val:
            cur = x_pol.copy()
            cur_val = v
        else:
            if rng.random() < 0.05 and v < cur_val + 0.008:
                cur = x_pol.copy()
                cur_val = v
            if rng.random() < 0.10:
                cur = x_best.copy()
                cur_val = best_val

    remain = BUDGET - (time.time() - t0)
    final_sec = float(np.clip(remain - 2.0, 5.0, 42.0))
    if final_sec > 1.0 and remain > 3.0:
        try:
            x_best = lp_polish(x_best, seconds=final_sec * 0.6, radius_init=8e-4)
            rest = BUDGET - (time.time() - t0) - 1.0
            if rest > 2.0:
                x_best = lp_polish(x_best, seconds=min(rest, 18.0), radius_init=4e-4)
        except Exception:
            pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
