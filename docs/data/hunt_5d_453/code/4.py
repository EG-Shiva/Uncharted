import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def smooth_trust(x, cur, beta, lr_init, max_steps, t_end):
    n, d = x.shape
    lr = lr_init
    for _ in range(max_steps):
        if time.time() > t_end:
            break
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(G.max())
        # weights exp(beta*(c-m)), diag -> 0
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Z = W.sum() * 0.5
        if not np.isfinite(Z) or Z == 0:
            break
        grad = (W @ x) / Z
        dot = (grad * x).sum(axis=1, keepdims=True)
        Gt = grad - dot * x
        nrm = float(np.linalg.norm(Gt))
        if not np.isfinite(nrm) or nrm < 1e-16:
            break
        y = x - lr * Gt
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = max_cos(y)
        if v < cur:
            x = y
            cur = v
            lr = min(lr * 1.3, 2e-3)
        else:
            lr *= 0.5
            if lr < 1e-12:
                break
    return x, cur


def lp_polish_small(x, cur, t_end, r_init=3e-4):
    n, d = x.shape
    best = x.copy()
    best_val = cur
    radius = r_init
    iu0 = np.triu_indices(n, 1)
    while time.time() < t_end and radius > 1e-13:
        # need time for one linprog
        if t_end - time.time() < 2.5:
            break
        Gfull = x @ x.T
        c = Gfull[iu0]
        m = float(c.max())
        delta = max(20.0 * radius, 2e-7)
        sel = c > m - delta
        # cap constraints for speed
        if np.count_nonzero(sel) > 4000:
            # keep top 4000
            idx = np.argpartition(c, -4000)[-4000:]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
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
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = max_cos(y)
        if v < best_val:
            x = y
            best = y.copy()
            best_val = v
            radius = min(radius * 1.4, 5e-4)
        else:
            radius *= 0.5
    return best, best_val


def micro_hill(x, cur, t_end, rng):
    n = x.shape[0]
    iu0 = np.triu_indices(n, 1)
    sigmas = [3e-5, 1e-5, 5e-5, 2e-4, 1e-6]
    k = 0
    while time.time() < t_end:
        G = x @ x.T
        # find worst pair via triu
        c = G[iu0]
        a = int(np.argmax(c))
        i = int(iu0[0][a])
        j = int(iu0[1][a])
        sigma = sigmas[k % len(sigmas)]
        k += 1
        y = x.copy()
        # jitter i and j tangentially
        for kk in (i, j):
            nz = rng.normal(scale=sigma, size=5)
            nz = nz - np.dot(nz, y[kk]) * y[kk]
            y[kk] = y[kk] + nz
            nn = np.linalg.norm(y[kk])
            if nn == 0 or not np.isfinite(nn):
                y[kk] = x[kk]
            else:
                y[kk] = y[kk] / nn
        # occasionally jitter one random rattler too
        if (k % 5) == 0:
            r = int(rng.integers(0, n))
            nz = rng.normal(scale=sigma * 2.0, size=5)
            nz = nz - np.dot(nz, y[r]) * y[r]
            y[r] = y[r] + nz
            nn = np.linalg.norm(y[r])
            if nn != 0 and np.isfinite(nn):
                y[r] = y[r] / nn
        v = max_cos(y)
        if v < cur:
            x = y
            cur = v
    return x, cur


def solve():
    t0 = time.time()
    t_end_total = t0 + 170.0
    rng = np.random.default_rng(0)
    try:
        x0 = np.load("best.npy")
    except Exception:
        x0 = np.load("record.npy")
    x0 = np.asarray(x0, dtype=np.float64)
    nrm = np.linalg.norm(x0, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    x0 = x0 / nrm
    cur = max_cos(x0)
    best = x0.copy()
    best_val = cur
    n, d = best.shape

    # Phase 1: adaptive smoothed-max descent, coarse to fine
    t1 = min(t0 + 45.0, t_end_total - 100.0)
    if time.time() < t1:
        x, v = smooth_trust(best, best_val, beta=150.0, lr_init=5e-4,
                            max_steps=2500, t_end=t1)
        if v < best_val:
            best, best_val = x, v
    t1b = min(t0 + 70.0, t_end_total - 80.0)
    if time.time() < t1b:
        x, v = smooth_trust(best, best_val, beta=600.0, lr_init=1e-4,
                            max_steps=2500, t_end=t1b)
        if v < best_val:
            best, best_val = x, v
    t1c = min(t0 + 90.0, t_end_total - 60.0)
    if time.time() < t1c:
        x, v = smooth_trust(best, best_val, beta=2000.0, lr_init=3e-5,
                            max_steps=2500, t_end=t1c)
        if v < best_val:
            best, best_val = x, v

    # Phase 2: basin hopping on rattlers / worst neighbourhoods
    t2 = min(t0 + 115.0, t_end_total - 55.0)
    while time.time() < t2:
        # pick worst pair + random subset to shake
        G = best @ best.T
        iu0 = np.triu_indices(n, 1)
        c = G[iu0]
        a = int(np.argmax(c))
        wi = int(iu0[0][a])
        wj = int(iu0[1][a])
        shake_idx = set([wi, wj])
        # add neighbours of wi,wj (closest dots)
        for kk in (wi, wj):
            row = G[kk].copy()
            row[kk] = -2
            top = np.argpartition(row, -8)[-8:]
            for t_ in top:
                shake_idx.add(int(t_))
        # add random rattlers
        for _ in range(8):
            shake_idx.add(int(rng.integers(0, n)))
        shake_idx = np.array(list(shake_idx), dtype=int)
        sigma = float(rng.choice([0.004, 0.008, 0.015, 0.002]))
        y = best.copy()
        noise = rng.normal(scale=sigma, size=(len(shake_idx), d))
        for k2, idx in enumerate(shake_idx):
            v0 = noise[k2] - np.dot(noise[k2], y[idx]) * y[idx]
            y[idx] = y[idx] + v0
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        # quick relax
        inner_end = min(time.time() + 4.0, t2)
        y2, v2 = smooth_trust(y, max_cos(y), beta=600.0, lr_init=8e-5,
                              max_steps=600, t_end=inner_end)
        if v2 < best_val:
            best, best_val = y2, v2

    # Phase 3: small-radius LP polish
    t3 = t_end_total - 12.0
    if time.time() < t3:
        xlp, vlp = lp_polish_small(best, best_val, t_end=t3, r_init=2.5e-4)
        if vlp < best_val:
            best, best_val = xlp, vlp

    # Phase 4: micro hill climbing to squeeze last 1e-9
    if time.time() < t_end_total:
        xh, vh = micro_hill(best, best_val, t_end_total, rng)
        if vh < best_val:
            best, best_val = xh, vh

    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
