import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _smooth_refine(x, steps, K0, K1, lr0, lr1, d0, d1):
    n, d = x.shape
    m_ad = np.zeros_like(x)
    v_ad = np.zeros_like(x)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    for s in range(steps):
        frac = s / (steps - 1) if steps > 1 else 0.0
        K = K0 * (K1 / K0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        delta = d0 * (d1 / d0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        # in-place E = exp(K*(G-m)) masked to [m-delta, m]
        G -= m
        G *= K
        thr = -K * delta
        # mask far pairs to -inf -> 0 weight
        # use boolean indexing
        G[G < thr] = -np.inf
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        S = float(np.sum(G))
        if not np.isfinite(S) or S <= 0:
            continue
        G /= S
        grad = G @ x
        proj = np.sum(grad * x, axis=1, keepdims=True)
        grad -= proj * x
        # Adam
        m_ad *= b1
        m_ad += (1.0 - b1) * grad
        v_ad *= b2
        # v_ad += (1-b2)*grad^2 in place
        np.add(v_ad, (1.0 - b2) * (grad * grad), out=v_ad)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        mhat = m_ad / bc1
        vhat = v_ad / bc2
        step = mhat / (np.sqrt(vhat) + eps)
        # clip extreme steps for stability
        # limit per-row norm
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x

def _hole_improve(x, rng, n_cand=3000, n_worst=5):
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    row_max = np.max(G, axis=1)
    glob = float(np.max(G))
    order = np.argsort(-row_max)
    improved = False
    for k in range(min(n_worst, n)):
        idx = int(order[k])
        # global without idx
        mask = np.ones(n, dtype=bool)
        mask[idx] = False
        # max over pairs not involving idx
        G2 = x[mask] @ x[mask].T
        np.fill_diagonal(G2, -2.0)
        glob_excl = float(np.max(G2)) if G2.size else -2.0
        Xo = x[mask]  # (n-1,d)
        C = rng.standard_normal(size=(n_cand, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        D = C @ Xo.T
        cmax = np.max(D, axis=1)
        bi = int(np.argmin(cmax))
        bv = float(cmax[bi])
        new_glob = glob_excl if glob_excl > bv else bv
        if new_glob < glob - 1e-7:
            x = x.copy()
            x[idx] = C[bi]
            glob = new_glob
            # update G/row_max for next worst (recompute)
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            row_max = np.max(G, axis=1)
            order = np.argsort(-row_max)
            improved = True
    return x, improved

def _lp_polish(x, seconds, t_end=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t_end is not None and time.time() > t_end:
            break
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(np.max(c))
        sel = c > m - max(20 * radius, 1e-9)
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
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
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    np.random.seed(0)
    rng = np.random.default_rng(0)
    # load starts
    x_best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape != (249, 6):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            v = _max_cos(a)
            if v < best_val:
                best_val = v
                x_best = a.copy()
        except Exception:
            continue
    if x_best is None:
        a = rng.standard_normal(size=(249, 6))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        x_best = a
        best_val = _max_cos(a)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)

    sigmas = [0.0, 0.008, 0.025, 0.05, 0.015, 0.07, 0.035]
    # reserve ~55s for final LP
    for ti, sigma in enumerate(sigmas):
        if time.time() > HARD_END - 65:
            break
        if ti == 0:
            x0 = x_best.copy()
            steps = 3000
        else:
            noise = rng.standard_normal(size=x_best.shape)
            x0 = x_best + sigma * noise
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            # occasionally fully randomize a few points
            if ti in (3, 5):
                k = 3 if ti == 3 else 5
                idx = rng.choice(249, size=k, replace=False)
                r = rng.standard_normal(size=(k, 6))
                r /= np.linalg.norm(r, axis=1, keepdims=True)
                x0[idx] = r
            steps = 2200
        # main anneal
        x1 = _smooth_refine(x0, steps=steps, K0=25.0, K1=350.0,
                            lr0=0.02, lr1=0.002, d0=0.20, d1=0.03)
        v1 = _max_cos(x1)
        # hole + short re-anneal cycles
        for _ in range(3):
            if time.time() > HARD_END - 62:
                break
            xh, imp = _hole_improve(x1, rng, n_cand=3000, n_worst=5)
            if not imp:
                break
            xh = _smooth_refine(xh, steps=900, K0=150.0, K1=500.0,
                                lr0=0.006, lr1=0.001, d0=0.06, d1=0.02)
            vh = _max_cos(xh)
            x1, v1 = xh, vh
            if time.time() > HARD_END - 62:
                break
        if v1 < best_val - 1e-12:
            x_best = x1
            best_val = v1
        # also try a sharp-only polish of current even if not global best
        # to escape via different basin: short sharp refine on x1
        if time.time() > HARD_END - 62:
            break

    remain = HARD_END - time.time()
    remain = float(min(max(remain, 12.0), 60.0))
    # give most time to LP on best
    x_best = _lp_polish(x_best, seconds=max(remain - 1.0, 8.0))
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # final safety: tiny extra smooth if LP stalled? just return
    return x_best
