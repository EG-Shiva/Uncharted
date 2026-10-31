import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds, r0=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    if seconds <= 0:
        return xb
    start = time.time()
    radius = r0
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu0]
        m = c.max()
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        if not np.any(sel):
            break
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        ci = I[:, None] * d + np.arange(d)
        cj = J[:, None] * d + np.arange(d)
        cols = np.concatenate([ci, cj, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
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
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            best = v
            xb = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return xb


def _soft_relax(x, steps=600, beta0=30.0, beta1=350.0, lr0=0.04, lr1=0.004, mu=0.85):
    n, d = x.shape
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    vel = np.zeros_like(x)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = beta0 * (beta1 / beta0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        r = W.sum(axis=1, keepdims=True)
        A = (W @ x) / np.maximum(r, 1e-30)
        Adot = np.sum(A * x, axis=1, keepdims=True)
        At = A - Adot * x
        vel = mu * vel + (1.0 - mu) * At
        bias = 1.0 - mu ** (s + 1)
        vh = vel / max(bias, 1e-12)
        x = x - lr * vh
        xn = np.linalg.norm(x, axis=1, keepdims=True)
        xn[xn == 0] = 1.0
        x = x / xn
    return x


def _find_hole(x, exclude_idx=None, n_samples=15000, rng=None):
    n, d = x.shape
    Z = rng.standard_normal((n_samples, d))
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    if exclude_idx is not None:
        mask = np.ones(n, dtype=bool)
        mask[exclude_idx] = False
        X = x[mask]
    else:
        X = x
    D = Z @ X.T
    worst = D.max(axis=1)
    bi = int(np.argmin(worst))
    return Z[bi]


def solve():
    t0 = time.time()
    rng = np.random.default_rng(20261031)
    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            z = rng.standard_normal((292, 4))
            z = z / np.linalg.norm(z, axis=1, keepdims=True)
            x0 = z
    x0 = np.asarray(x0, dtype=float)
    if x0.shape != (292, 4):
        z = rng.standard_normal((292, 4))
        z = z / np.linalg.norm(z, axis=1, keepdims=True)
        x0 = z
    x0 = x0 / np.linalg.norm(x0, axis=1, keepdims=True)
    best = x0.copy()
    best_val = _max_cos(best)
    n, d = best.shape

    it = 0
    # explore phase ~115s
    while time.time() - t0 < 115:
        it += 1
        mod = it % 5
        if mod == 0 or mod == 1:
            sigma = 0.015 if mod == 0 else 0.045
            xt = best + sigma * rng.standard_normal(best.shape)
            xt = xt / np.linalg.norm(xt, axis=1, keepdims=True)
            xt = _soft_relax(xt, steps=550, beta0=25.0, beta1=300.0, lr0=0.035, lr1=0.0035, mu=0.85)
        elif mod == 2:
            xt = best.copy()
            k = 25
            idx = rng.choice(n, size=k, replace=False)
            xt[idx] += 0.22 * rng.standard_normal((k, d))
            xt += 0.008 * rng.standard_normal((n, d))
            xt = xt / np.linalg.norm(xt, axis=1, keepdims=True)
            xt = _soft_relax(xt, steps=650, beta0=20.0, beta1=350.0, lr0=0.045, lr1=0.004, mu=0.85)
        elif mod == 3:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            cnt = (G > m - 1e-9).sum(axis=1)
            order = np.argsort(-cnt)
            cand = order[:6]
            idx_rep = int(cand[rng.integers(0, len(cand))])
            hole = _find_hole(best, exclude_idx=idx_rep, n_samples=15000, rng=rng)
            xt = best.copy()
            xt[idx_rep] = hole + 0.012 * rng.standard_normal(d)
            xt = xt / np.linalg.norm(xt, axis=1, keepdims=True)
            xt = _soft_relax(xt, steps=650, beta0=25.0, beta1=400.0, lr0=0.04, lr1=0.003, mu=0.87)
        else:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            # worst pair
            flat = int(np.argmax(G))
            wi, wj = divmod(flat, n)
            cnt = (G > m - 1e-9).sum(axis=1)
            tgt = wi if cnt[wi] >= cnt[wj] else wj
            xt = best.copy()
            # kick target and its contact neighbours
            nbrs = np.where(G[tgt] > m - 5e-4)[0]
            if len(nbrs) > 12:
                nbrs = nbrs[np.argsort(-G[tgt][nbrs])[:12]]
            xt[tgt] += 0.25 * rng.standard_normal(d)
            if len(nbrs) > 0:
                xt[nbrs] += 0.06 * rng.standard_normal((len(nbrs), d))
            xt += 0.005 * rng.standard_normal((n, d))
            xt = xt / np.linalg.norm(xt, axis=1, keepdims=True)
            xt = _soft_relax(xt, steps=600, beta0=30.0, beta1=350.0, lr0=0.04, lr1=0.0035, mu=0.85)

        vt = _max_cos(xt)
        # promising enough to spend LP time?
        if vt < best_val + 0.004:
            remain_explore = 115 - (time.time() - t0)
            if remain_explore <= 0.3:
                # no time for LP, just check soft value
                if vt < best_val - 1e-12:
                    best = xt / np.linalg.norm(xt, axis=1, keepdims=True)
                    best_val = vt
                continue
            short = 1.4
            if vt < best_val:
                short = 2.2
            short = min(short, max(0.4, remain_explore - 0.1))
            xp = _lp_polish(xt, seconds=short, r0=1e-3)
            vp = _max_cos(xp)
            if vp < best_val - 1e-12:
                best = xp
                best_val = vp
        # safety
        if time.time() - t0 > 160:
            break

    # final polish with remaining budget
    elapsed = time.time() - t0
    remain = min(30.0, max(0.0, 168.0 - elapsed - 1.0))
    if remain > 2.0:
        best = _lp_polish(best, seconds=remain, r0=8e-4)
        best_val = _max_cos(best)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
