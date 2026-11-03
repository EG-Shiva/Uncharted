import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _smooth_refine(x, steps, k0, k1, lr0, mu, rng, jitter=0.0):
    n, d = x.shape
    v = np.zeros_like(x)
    if jitter > 0:
        x = x + jitter * rng.standard_normal(size=x.shape)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        m = np.max(G - np.eye(n) * 2.0)
        E = np.exp(k * (G - m))
        np.fill_diagonal(E, 0.0)
        S = E.sum()
        if S == 0 or not np.isfinite(S):
            break
        Grad = E @ x / S
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - dot * x
        v = mu * v + Gt
        x = x - lr * v
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        x = x / nrm
    return x

def _rand_plane(d, rng):
    A = rng.standard_normal(size=(d, 2))
    q, _ = np.linalg.qr(A)
    return q[:, 0].copy(), q[:, 1].copy()

def _apply_rotations(X, a, b, thetas):
    # X:(n,d), a,b:(d,), thetas:(n,) rotation angle per point in plane (a,b)
    xa = X @ a
    xb = X @ b
    c = np.cos(thetas) - 1.0
    s = np.sin(thetas)
    ca = xa * c - xb * s
    cb = xa * s + xb * c
    return X + ca[:, None] * a[None, :] + cb[:, None] * b[None, :]

def _rigid_cap_move(X, rng):
    n, d = X.shape
    u = rng.standard_normal(size=d); u /= np.linalg.norm(u)
    dots = X @ u
    k = int(rng.integers(6, 31))  # cap size 6..30
    idx = np.argsort(dots)[-k:]
    a, b = _rand_plane(d, rng)
    theta = float(rng.uniform(0.03, 0.35)) * (1.0 if rng.random() < 0.5 else -1.0)
    thetas = np.zeros(n)
    thetas[idx] = theta
    Y = _apply_rotations(X, a, b, thetas)
    # tiny break of exact rigidity
    Y[idx] += 0.008 * rng.standard_normal(size=(k, d))
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def _shear_move(X, rng):
    n, d = X.shape
    u = rng.standard_normal(size=d); u /= np.linalg.norm(u)
    a, b = _rand_plane(d, rng)
    # make a,b roughly orthogonal to u for a twist-like shear
    proj = X @ u  # (n,)
    alpha = float(rng.uniform(0.4, 1.6)) * (1.0 if rng.random() < 0.5 else -1.0)
    width = float(rng.uniform(0.3, 1.0))
    # smooth profile: scaled projection
    thetas = alpha * proj / max(width, 1e-9)
    # clip to avoid huge rotations
    thetas = np.clip(thetas, -0.6, 0.6)
    Y = _apply_rotations(X, a, b, thetas)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def _multi_reinsert(X, rng, k=None):
    n, d = X.shape
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    mx = G.max()
    worst = G.max(axis=1)
    if k is None:
        k = int(rng.integers(3, 9))  # 3..8 joint reinsert
    order = np.argsort(-worst)
    victims = list(order[:min(k, n)])
    # add some random to diversify
    if rng.random() < 0.5 and k > 3:
        victims[-1] = int(rng.integers(0, n))
        victims = list(dict.fromkeys(victims))
        while len(victims) < k:
            victims.append(int(rng.integers(0, n)))
            victims = list(dict.fromkeys(victims))
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[victims] = False
    others = X[keep_mask]
    # greedy farthest-point selection from random pool
    n_pool = 8000
    cand = rng.standard_normal(size=(n_pool, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    # scores to others
    s0 = (cand @ others.T).max(axis=1)
    picked = []
    # iteratively pick
    cur_max = s0.copy()
    # for diversity penalize closeness among picked
    for _ in range(len(victims)):
        j = int(np.argmin(cur_max + 1e-6 * rng.standard_normal(size=n_pool)))
        picked.append(cand[j].copy())
        # update cur_max with dot to picked j
        dj = cand @ picked[-1]
        cur_max = np.maximum(cur_max, dj)
        # discourage repicking same: set to large
        cur_max[j] = 2.0
    picked = np.array(picked)
    # joint tangential refinement of picked against others+each other, others fixed
    for it in range(120):
        step = 0.12 * (1.0 - 0.5 * it / 120)
        for jj in range(len(picked)):
            c = picked[jj]
            d1 = others @ c
            m1 = d1.max() if d1.size else -2
            # dots to other picked
            if len(picked) > 1:
                rest = np.delete(picked, jj, axis=0)
                d2 = rest @ c
                m2 = d2.max()
                m = m1 if m1 > m2 else m2
                kk = 50.0
                w1 = np.exp(kk * (d1 - m))
                w2 = np.exp(kk * (d2 - m))
                tot = w1.sum() + w2.sum()
                g = (w1 @ others + w2 @ rest) / tot
            else:
                kk = 50.0
                w1 = np.exp(kk * (d1 - m1))
                w1 /= w1.sum()
                g = w1 @ others
            gt = g - (g @ c) * c
            c = c - step * gt
            nc = np.linalg.norm(c)
            if nc == 0 or not np.isfinite(nc):
                continue
            picked[jj] = c / nc
    Y = X.copy()
    Y[victims] = picked
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def _lp_polish(x, seconds, t0):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < seconds and radius > 1e-14:
        cmat = (x @ x.T)[iu]
        m = cmat.max()
        sel = cmat > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cmat[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    T0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(7)
    np.random.seed(7)
    try:
        x_best = np.load("best.npy").astype(float)
    except Exception:
        x_best = np.load("record.npy").astype(float)
    try:
        x_rec = np.load("record.npy").astype(float)
        x_rec /= np.linalg.norm(x_rec, axis=1, keepdims=True)
        xb = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
        if _max_cos(x_rec) < _max_cos(xb):
            x_best = x_rec.copy()
    except Exception:
        pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    best_val = _max_cos(x_best)
    cur_x = x_best.copy()
    cur_val = best_val
    HOP_END = T0 + 118.0
    it = 0
    while time.time() - T0 < HOP_END and time.time() - T0 < BUDGET - 50:
        it += 1
        mode = it % 5
        try:
            if mode == 0:
                y = _rigid_cap_move(cur_x, rng)
                y = _smooth_refine(y, steps=650, k0=25, k1=320, lr0=0.22, mu=0.87, rng=rng)
            elif mode == 1:
                y = _multi_reinsert(cur_x, rng, k=None)
                y = _smooth_refine(y, steps=700, k0=25, k1=350, lr0=0.24, mu=0.88, rng=rng)
            elif mode == 2:
                y = _shear_move(cur_x, rng)
                y = _smooth_refine(y, steps=650, k0=20, k1=320, lr0=0.25, mu=0.88, rng=rng)
            elif mode == 3:
                # rigid cap from global best + multi reinsert combo
                base = x_best if rng.random() < 0.6 else cur_x
                y = _rigid_cap_move(base, rng)
                # occasionally follow with small multi-reinsert on worst point
                if rng.random() < 0.4:
                    y = _multi_reinsert(y, rng, k=int(rng.integers(2, 5)))
                y = _smooth_refine(y, steps=600, k0=30, k1=380, lr0=0.20, mu=0.88, rng=rng)
            else:
                sigma = float(rng.uniform(0.02, 0.10))
                base = x_best if rng.random() < 0.5 else cur_x
                y = base + sigma * rng.standard_normal(size=base.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _smooth_refine(y, steps=550, k0=30, k1=350, lr0=0.20, mu=0.88, rng=rng)
        except Exception:
            continue
        try:
            new_val = _max_cos(y)
        except Exception:
            continue
        if new_val < best_val - 1e-15:
            x_best = y.copy()
            best_val = new_val
            cur_x = y.copy()
            cur_val = new_val
        else:
            elapsed = time.time() - T0
            frac = elapsed / max(1e-9, (HOP_END - T0))
            Temp = 0.0015 * max(0.08, 1.0 - frac)
            if new_val < cur_val or rng.random() < np.exp((cur_val - new_val) / Temp):
                cur_x = y.copy()
                cur_val = new_val
        # periodic restart of walker to best if drifted too high
        if cur_val > best_val + 0.02 and rng.random() < 0.3:
            cur_x = x_best.copy()
            cur_val = best_val
    remain = BUDGET - (time.time() - T0) - 2.0
    if remain > 5:
        try:
            x_best = _smooth_refine(x_best, steps=300, k0=80, k1=600, lr0=0.07, mu=0.9, rng=rng)
            nv = _max_cos(x_best)
            if nv < best_val:
                best_val = nv
        except Exception:
            pass
        try:
            x_pol = _lp_polish(x_best, seconds=remain, t0=time.time())
            pv = _max_cos(x_pol)
            if pv < best_val and np.all(np.isfinite(x_pol)):
                x_best = x_pol
                best_val = pv
        except Exception:
            pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    try:
        x0 = np.load("best.npy").astype(float)
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        if _max_cos(x_best) > _max_cos(x0) + 1e-12:
            # return better of the two to avoid regression
            if _max_cos(x0) < _max_cos(x_best):
                x_best = x0
    except Exception:
        pass
    return x_best
