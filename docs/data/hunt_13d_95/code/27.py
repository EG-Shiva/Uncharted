import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


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


def gd_anneal(X, steps, T0, T1, lr0, lr1, mu=0.9, noise=0.0, rng=None, freeze=None):
    n, d = X.shape
    V = np.zeros_like(X)
    lT0, lT1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = float(G.max())
        W = np.exp((G - mx) / T)
        np.fill_diagonal(W, 0.0)
        sw = float(W.sum())
        if not np.isfinite(sw) or sw <= 0:
            break
        W /= sw
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        if freeze is not None:
            Gt[freeze] = 0.0
        V = mu * V + Gt
        X = X - lr * V
        if noise > 0 and rng is not None:
            X = X + noise * rng.standard_normal(X.shape) * 0.01
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def contact_counts(x, mc, tol=0.015):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return (g > mc - tol).sum(axis=1)


def solve():
    t0 = time.time()
    BUDGET = 168.0
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
    if xb is not None and xb.shape == (95, 13):
        cands.append(xb / np.linalg.norm(xb, axis=1, keepdims=True))
    if xr is not None and xr.shape == (95, 13):
        cands.append(xr / np.linalg.norm(xr, axis=1, keepdims=True))
    if not cands:
        q = rng.standard_normal((95, 13))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        cands.append(q)
    vals = [max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    best_val = float(vals[bi])
    cur = x_best.copy()
    cur_val = best_val
    n, d = 95, 13

    it = 0
    # pre-tuned schedules
    while time.time() - t0 < EXPL_T:
        it += 1
        remain = EXPL_T - (time.time() - t0)
        if remain < 2.0:
            break
        u = rng.random()
        Xt = cur.copy()
        steps, T0, T1, lr0, lr1, mu, noise = 900, 0.014, 0.0012, 0.10, 0.010, 0.88, 0.0
        do_lp = True
        if u < 0.35:
            # global shake + anneal (basin jump)
            sigma = float(rng.choice([0.08, 0.12, 0.20, 0.30, 0.45]))
            Xt = Xt + sigma * rng.standard_normal(Xt.shape)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            steps = int(rng.choice([800, 1000, 1300]))
            T0 = float(rng.choice([0.012, 0.016, 0.020]))
            T1 = float(rng.choice([0.0009, 0.0012, 0.0016]))
            lr0 = float(rng.choice([0.08, 0.12, 0.16]))
            lr1 = lr0 * 0.1
            mu = 0.88
        elif u < 0.70:
            # cap cluster re-jam: pick random center, take k nearest, reinit them
            k = int(rng.integers(7, 17))
            cen = rng.standard_normal((d,))
            cen /= np.linalg.norm(cen)
            dots = Xt @ cen
            # sometimes center on overloaded point or rattler neighbourhood
            r2 = rng.random()
            if r2 < 0.3:
                cc = contact_counts(cur, cur_val, tol=0.015)
                hub = int(np.argmax(cc))
                cen = cur[hub] + 0.3 * rng.standard_normal((d,))
                cen /= np.linalg.norm(cen)
                dots = Xt @ cen
            idx = np.argsort(-dots)[:k]
            # reinit cluster: random uniform (big topology change)
            if rng.random() < 0.5:
                Xt[idx] = rng.standard_normal((k, d))
            else:
                # greedy holes sequentially
                for j in idx:
                    C = rng.standard_normal((2500, d))
                    C /= np.linalg.norm(C, axis=1, keepdims=True)
                    others = np.delete(Xt, j, axis=0)
                    md = (C @ others.T).max(axis=1)
                    Xt[j] = C[int(np.argmin(md))]
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            steps = int(rng.choice([900, 1200, 1500]))
            T0, T1 = 0.014, 0.0010
            lr0, lr1 = 0.12, 0.010
            mu = 0.88
        elif u < 0.85:
            # targeted rebalance: remove overloaded + rattler
            cc = contact_counts(cur, cur_val, tol=0.015)
            order_hi = np.argsort(-cc)
            order_lo = np.argsort(cc)
            k_hi = int(rng.integers(2, 5))
            k_lo = int(rng.integers(1, 3))
            idx = np.concatenate([order_hi[:k_hi], order_lo[:k_lo]])
            idx = np.unique(idx)
            for j in idx:
                C = rng.standard_normal((3000, d))
                C /= np.linalg.norm(C, axis=1, keepdims=True)
                mask = np.ones(n, dtype=bool); mask[j] = False
                md = (C @ Xt[mask].T).max(axis=1)
                Xt[j] = C[int(np.argmin(md))]
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            steps, T0, T1, lr0, lr1 = 1000, 0.012, 0.0010, 0.10, 0.008
        else:
            # from scratch random + anneal (new basin)
            Xt = rng.standard_normal((n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            steps, T0, T1, lr0, lr1 = 1400, 0.020, 0.0012, 0.14, 0.010

        try:
            Xt = gd_anneal(Xt, steps, T0, T1, lr0, lr1, mu=mu, noise=noise, rng=rng)
        except Exception:
            continue
        try:
            v_gd = max_cos(Xt)
        except Exception:
            continue
        # screen: only LP-polish if within band (or scratch and good)
        thresh_lp = best_val + 0.015
        # always polish global best contenders, occasionally polish worse to explore
        if v_gd > thresh_lp and rng.random() > 0.05:
            # accept as basin move without LP if it is diverse? just update cur occasionally
            if v_gd < cur_val + 0.005 and rng.random() < 0.03:
                cur, cur_val = Xt, v_gd
            continue
        lp_sec = 0.9
        if remain < 10:
            lp_sec = 0.6
        try:
            Xp = lp_polish(Xt, seconds=lp_sec, radius_init=0.006)
            v = max_cos(Xp)
        except Exception:
            continue
        if v < best_val - 1e-12:
            x_best, best_val = Xp.copy(), float(v)
            cur, cur_val = Xp.copy(), float(v)
        elif v < cur_val:
            cur, cur_val = Xp.copy(), float(v)
        else:
            # Metropolis-ish sideways to change basin
            if rng.random() < 0.08 and v < cur_val + 0.008:
                cur, cur_val = Xp.copy(), float(v)
            if rng.random() < 0.10:
                cur, cur_val = x_best.copy(), best_val
        if time.time() - t0 > BUDGET - 35:
            break

    remain = BUDGET - (time.time() - t0)
    final_sec = float(np.clip(remain - 2.0, 5.0, 38.0))
    if final_sec > 1.0:
        try:
            x_best = lp_polish(x_best, seconds=final_sec * 0.6, radius_init=1e-3)
            rest = BUDGET - (time.time() - t0) - 1.0
            if rest > 2.0:
                x_best = lp_polish(x_best, seconds=min(rest, 18.0), radius_init=5e-4)
        except Exception:
            pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
