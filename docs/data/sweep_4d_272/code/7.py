import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds, radius_init=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = radius_init
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        # upper triangle values
        c = G[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        # if too many constraints, keep tightest subset
        if np.count_nonzero(sel) > 2500:
            thr = np.sort(c[sel])[-2500]
            sel = c >= thr
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
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
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _soft_polish(x, steps=350, beta0=12.0, beta1=160.0, s0=0.02, s1=0.002):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n = x.shape[0]
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** frac
        step = s0 * (1 - frac) + s1 * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gm = G.max()
        W = np.exp(beta * (G - gm))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        norms = np.sqrt(np.sum(gt * gt, axis=1, keepdims=True))
        mn = float(norms.max())
        if not np.isfinite(mn) or mn < 1e-14:
            continue
        x = x - gt * (step / mn)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _refine_single(Y, c0, beta=60.0, steps=200, s0=0.03, s1=0.003):
    c = c0 / np.linalg.norm(c0)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        step = s0 * (1 - frac) + s1 * frac
        dots = Y @ c
        dm = dots.max()
        w = np.exp(beta * (dots - dm))
        w /= w.sum()
        grad = w @ Y
        gt = grad - float(grad @ c) * c
        nrm = float(np.linalg.norm(gt))
        if not np.isfinite(nrm) or nrm < 1e-14:
            break
        c = c - gt * (step / nrm)
        c /= np.linalg.norm(c)
    return c


def _best_insertion(Y, rng, n_cand=20000, beta=60.0):
    cand = rng.standard_normal((n_cand, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ Y.T
    md = dots.max(axis=1)
    K = 5
    idx = np.argpartition(md, K)[:K]
    best_c = None
    best_v = 2.0
    for k in idx:
        c0 = cand[k]
        c = _refine_single(Y, c0, beta=beta, steps=200)
        v = float((Y @ c).max())
        if v < best_v:
            best_v = v
            best_c = c
    return best_c


def _contacts(x, thr=1e-9):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = g.max()
    adj = g > m - thr
    cnt = np.sum(adj, axis=1)
    return m, cnt, g


def solve():
    rng = np.random.default_rng(0)
    t0 = time.time()
    BUDGET = 172.0

    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = np.load("record.npy").astype(float)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    if xb.shape != (272, 4):
        # fallback: tile/truncate deterministically
        tmp = rng.standard_normal((272, 4))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        # copy overlap
        m = min(xb.shape[0], 272)
        tmp[:m] = xb[:m] / np.linalg.norm(xb[:m], axis=1, keepdims=True)
        xb = tmp
    best_val = _max_cos(xb)

    # initial short LP to sit at local optimum
    if BUDGET - (time.time() - t0) > 20:
        xb = _lp_polish(xb, seconds=4.0)
        best_val = _max_cos(xb)

    no_improve = 0
    it = 0
    # main basin-hopping loop, reserve ~10s for final polish
    while time.time() - t0 < BUDGET - 12.0:
        it += 1
        if time.time() - t0 > BUDGET - 12.0:
            break
        mode = it % 5
        xt = xb.copy()
        if mode == 0:
            sigma = 0.030 if no_improve < 8 else 0.060
            xt = xt + sigma * rng.standard_normal(xt.shape)
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)
        elif mode == 1:
            sigma = 0.008
            xt = xt + sigma * rng.standard_normal(xt.shape)
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)
        elif mode == 2:
            # rattler / low-contact re-insertion
            try:
                m, cnt, _ = _contacts(xt)
            except Exception:
                cnt = np.zeros(272)
            order = np.argsort(cnt)
            k_rep = 3 if no_improve < 10 else 5
            targets = order[:k_rep]
            # replace one by one with farthest-point search
            for t_idx in targets:
                mask = np.ones(272, dtype=bool)
                mask[t_idx] = False
                Y = xt[mask]
                c_new = _best_insertion(Y, rng, n_cand=16000, beta=60.0)
                xt[t_idx] = c_new
                if time.time() - t0 > BUDGET - 12.0:
                    break
        elif mode == 3:
            # cluster shake around random center
            center = int(rng.integers(0, 272))
            dots = xt @ xt[center]
            nn = np.argsort(-dots)[:12]
            noise = 0.005 * rng.standard_normal(xt.shape)
            noise[nn] += 0.055 * rng.standard_normal((len(nn), 4))
            xt = xt + noise
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)
        else:
            # random subset shake
            k = 24
            sel = rng.choice(272, size=k, replace=False)
            noise = np.zeros_like(xt)
            noise[sel] = 0.045 * rng.standard_normal((k, 4))
            # tiny global jitter
            noise += 0.002 * rng.standard_normal(xt.shape)
            xt = xt + noise
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)

        # annealed soft repulsion (fast)
        xt = _soft_polish(xt, steps=320, beta0=12.0, beta1=160.0, s0=0.02, s1=0.002)

        # short LP polish
        remain = BUDGET - 12.0 - (time.time() - t0)
        if remain < 1.0:
            # no time for LP, just evaluate
            v = _max_cos(xt)
            if v < best_val - 1e-12:
                xb, best_val = xt, v
            break
        xt = _lp_polish(xt, seconds=min(2.8, remain))
        v = _max_cos(xt)
        if v < best_val - 1e-12:
            xb, best_val = xt, v
            no_improve = 0
        else:
            no_improve += 1
        # safety: stop if time nearly out
        if it > 200:
            break

    # final long polish on best
    remain = BUDGET - (time.time() - t0)
    if remain > 1.0:
        xb = _lp_polish(xb, seconds=min(9.0, remain - 0.5))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    return xb
