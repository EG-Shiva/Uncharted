import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contact_counts(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = g.max()
    cnt = (g > m - 1e-9).sum(axis=1)
    return m, cnt, g

def _soft_relax(x, steps=500, beta0=25.0, beta1=350.0, lr0=0.035, lr1=0.0035, mu=0.85):
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

def _local_relax(x, mobile, steps=350, beta0=30.0, beta1=500.0, lr0=0.04, lr1=0.004, mu=0.86):
    n, d = x.shape
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    vel = np.zeros_like(x)
    mob = np.zeros(n, dtype=bool)
    mob[mobile] = True
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
        step = np.zeros_like(x)
        step[mob] = vh[mob]
        # freeze others
        x = x - lr * step
        xn = np.linalg.norm(x, axis=1, keepdims=True)
        xn[xn == 0] = 1.0
        x = x / xn
    return x

def _find_k_holes(x, k, n_samples=20000, rng=None, exclude_idx=None, sep_cos=0.88):
    n, d = x.shape
    Z = rng.standard_normal((n_samples, d))
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    if exclude_idx is not None:
        mask = np.ones(n, dtype=bool)
        mask[np.asarray(exclude_idx)] = False
        X = x[mask] if np.any(mask) else x
    else:
        X = x
    D = Z @ X.T
    worst = D.max(axis=1)
    order = np.argsort(worst)
    holes = []
    for idx in order:
        z = Z[idx]
        ok = True
        for h in holes:
            if float(h @ z) > sep_cos:
                ok = False
                break
        if ok:
            holes.append(z)
            if len(holes) >= k:
                break
    while len(holes) < k:
        holes.append(Z[order[len(holes)]])
    return np.array(holes)

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
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
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

def solve():
    t0 = time.time()
    rng = np.random.default_rng(987654321)
    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            z = rng.standard_normal((292, 4))
            x0 = z / np.linalg.norm(z, axis=1, keepdims=True)
    x0 = np.asarray(x0, dtype=float)
    if x0.shape != (292, 4) or not np.all(np.isfinite(x0)):
        z = rng.standard_normal((292, 4))
        x0 = z / np.linalg.norm(z, axis=1, keepdims=True)
    x0 = x0 / np.linalg.norm(x0, axis=1, keepdims=True)
    try:
        r0 = np.load("record.npy")
        r0 = np.asarray(r0, dtype=float)
        if r0.shape == (292, 4) and np.all(np.isfinite(r0)):
            r0 = r0 / np.linalg.norm(r0, axis=1, keepdims=True)
            if _max_cos(r0) < _max_cos(x0):
                x0 = r0
    except Exception:
        pass
    best = x0.copy()
    best_val = _max_cos(best)
    n, d = best.shape
    EXPLORE_END = 138.0
    it = 0
    while time.time() - t0 < EXPLORE_END:
        it += 1
        if time.time() - t0 > 160:
            break
        u = rng.random()
        try:
            m_cur, cnt_cur, Gcur = _contact_counts(best)
        except Exception:
            m_cur = best_val
            cnt_cur = np.zeros(n, dtype=int)
        if u < 0.30:
            sigma = float(rng.choice(np.array([0.012, 0.025, 0.05, 0.09])))
            xt = best + sigma * rng.standard_normal(best.shape)
            xt = xt / np.linalg.norm(xt, axis=1, keepdims=True)
            xt = _soft_relax(xt, steps=550, beta0=22.0, beta1=320.0, lr0=0.038, lr1=0.0035, mu=0.85)
        elif u < 0.55:
            order = np.argsort(-cnt_cur)
            top = order[:20]
            center = int(top[rng.integers(0, len(top))])
            k = int(rng.integers(6, 14))
            dots = best @ best[center]
            S = np.argsort(-dots)[:k]
            xt = best.copy()
            xt[S] += 0.35 * rng.standard_normal((k, d))
            xt = xt / np.linalg.norm(xt, axis=1, keepdims=True)
            xt = _local_relax(xt, S, steps=350, beta0=30.0, beta1=500.0, lr0=0.04, lr1=0.004, mu=0.86)
            xt = _soft_relax(xt, steps=450, beta0=25.0, beta1=380.0, lr0=0.035, lr1=0.003, mu=0.86)
        elif u < 0.78:
            order = np.argsort(-cnt_cur)
            ns = int(rng.integers(2, 4))
            pool = order[:12]
            sources = rng.choice(pool, size=ns, replace=False)
            holes = _find_k_holes(best, k=ns, n_samples=20000, rng=rng, exclude_idx=sources, sep_cos=0.86)
            xt = best.copy()
            for s, h in zip(sources, holes):
                xt[int(s)] = h + 0.008 * rng.standard_normal(d)
            xt = xt / np.linalg.norm(xt, axis=1, keepdims=True)
            xt = _soft_relax(xt, steps=650, beta0=22.0, beta1=420.0, lr0=0.04, lr1=0.003, mu=0.87)
        else:
            flat = int(np.argmax(Gcur))
            wi, wj = divmod(flat, n)
            tgt = wi if cnt_cur[wi] >= cnt_cur[wj] else wj
            nbrs = np.where(Gcur[tgt] > m_cur - 5e-4)[0]
            if len(nbrs) > 12:
                nbrs = nbrs[np.argsort(-Gcur[tgt][nbrs])[:12]]
            xt = best.copy()
            xt[tgt] += 0.28 * rng.standard_normal(d)
            if len(nbrs) > 0:
                xt[nbrs] += 0.07 * rng.standard_normal((len(nbrs), d))
            xt += 0.006 * rng.standard_normal((n, d))
            xt = xt / np.linalg.norm(xt, axis=1, keepdims=True)
            xt = _soft_relax(xt, steps=600, beta0=28.0, beta1=380.0, lr0=0.04, lr1=0.0035, mu=0.85)
        # squeeze to near-optimum without LP
        xt = _soft_relax(xt, steps=250, beta0=120.0, beta1=900.0, lr0=0.012, lr1=0.0015, mu=0.88)
        try:
            vt = _max_cos(xt)
        except Exception:
            continue
        if vt < best_val - 1e-12:
            remain = EXPLORE_END - (time.time() - t0)
            if remain > 0.8:
                short = min(2.0, max(0.6, remain - 0.2))
                xp = _lp_polish(xt, seconds=short, r0=9e-4)
                vp = _max_cos(xp)
                if vp < best_val - 1e-12:
                    best = xp
                    best_val = vp
                elif vt < best_val - 1e-12:
                    best = xt / np.linalg.norm(xt, axis=1, keepdims=True)
                    best_val = vt
            else:
                best = xt / np.linalg.norm(xt, axis=1, keepdims=True)
                best_val = vt
        elif vt < best_val + 0.0015 and (it % 9 == 0):
            remain = EXPLORE_END - (time.time() - t0)
            if remain > 2.0:
                xp = _lp_polish(xt, seconds=0.8, r0=8e-4)
                vp = _max_cos(xp)
                if vp < best_val - 1e-12:
                    best = xp
                    best_val = vp
    elapsed = time.time() - t0
    remain = min(30.0, max(0.0, 170.0 - elapsed - 0.5))
    if remain > 1.5:
        best = _lp_polish(best, seconds=remain, r0=7e-4)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
