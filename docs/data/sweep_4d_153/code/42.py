import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if t_deadline is not None and time.time() >= t_deadline:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        iu_c = G[iu]
        m = iu_c.max()
        sel = iu_c > m - max(20.0 * radius, 1e-9)
        if np.count_nonzero(sel) == 0:
            break
        I = iu[0][sel]
        J = iu[1][sel]
        c = iu_c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _repulse(x, iters, k0, k1, lr0, lr1, mu, noise_amp, rng):
    n, d = x.shape
    v = np.zeros_like(x)
    for t in range(iters):
        frac = t / max(1, iters - 1)
        k = k0 * (k1 / k0) ** frac if k0 > 0 and k1 > 0 else k0
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        moff = G.max()
        W = np.exp(k * (G - moff))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        v = mu * v + gt
        denom = 1.0 - mu ** (t + 1) if mu > 0 else 1.0
        vc = v / denom if denom != 0 else v
        x = x - lr * vc
        if noise_amp > 0 and frac < 0.6:
            amp = noise_amp * (1.0 - frac / 0.6)
            pert = rng.standard_normal(x.shape)
            pert -= np.sum(pert * x, axis=1, keepdims=True) * x
            x = x + amp * pert
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _void_reinsert(x, k, n_cand, rng):
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    mx = G.max(axis=1)
    idx = np.argsort(-mx)[:k]
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[idx] = False
    Xkeep = x[keep_mask]
    new_pts = []
    for _ in idx:
        cand = rng.standard_normal((n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        if len(new_pts) > 0:
            XA = np.vstack([Xkeep, np.array(new_pts)])
        else:
            XA = Xkeep
        dots = cand @ XA.T
        worst = dots.max(axis=1)
        pick = int(np.argmin(worst))
        new_pts.append(cand[pick])
        Xkeep = np.vstack([Xkeep, cand[pick:pick+1]])
    y = x.copy()
    for ii, jj in enumerate(idx):
        y[jj] = new_pts[ii]
    return y

def _random_rotation(d, rng):
    A = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(A)
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    Q = Q * s[None, :]
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

def _crossover(a, b, rng):
    n, d = a.shape
    Qa = _random_rotation(d, rng)
    Qb = _random_rotation(d, rng)
    A = (a @ Qa.T)
    B = (b @ Qb.T)
    u = rng.standard_normal(d)
    u /= np.linalg.norm(u)
    pa = A @ u
    pb = B @ u
    # take upper cap from A, lower cap from B
    mask_a = pa > 0
    mask_b = pb <= 0
    pts = []
    pts_a = A[mask_a]
    pts_b = B[mask_b]
    # interleave to keep count
    combined = []
    if len(pts_a) > 0:
        combined.append(pts_a)
    if len(pts_b) > 0:
        combined.append(pts_b)
    if combined:
        C = np.vstack(combined)
    else:
        C = A.copy()
    # adjust to n points
    if len(C) >= n:
        # farthest point sampling to n
        rng_idx = int(rng.integers(len(C)))
        sel = [rng_idx]
        D = C @ C.T
        mind = np.full(len(C), 1e9)
        # iterative greedy max-min
        for _ in range(n - 1):
            last = sel[-1]
            d2 = 2.0 - 2.0 * (C @ C[last])
            mind = np.minimum(mind, d2)
            mind[sel] = -1
            sel.append(int(np.argmax(mind)))
        C = C[np.array(sel)]
    else:
        need = n - len(C)
        extra = rng.standard_normal((need * 3, d))
        extra /= np.linalg.norm(extra, axis=1, keepdims=True)
        # pick farthest from C
        for _ in range(need):
            dots = extra @ C.T
            w = dots.max(axis=1) if len(C) > 0 else np.zeros(len(extra))
            bi = int(np.argmin(w))
            C = np.vstack([C, extra[bi:bi+1]])
            extra = np.delete(extra, bi, axis=0)
            if len(extra) == 0:
                break
    # small jitter to break symmetry
    C = C + 0.01 * rng.standard_normal(C.shape)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    return C

def solve():
    t0 = time.time()
    DEADLINE = t0 + 168.0
    rng = np.random.default_rng(0)
    n, d = 153, 4
    try:
        init = np.load("best.npy")
    except Exception:
        try:
            init = np.load("record.npy")
        except Exception:
            init = rng.standard_normal((n, d))
    init = np.asarray(init, dtype=float)
    if init.shape != (n, d):
        tmp = rng.standard_normal((n, d))
        m = min(n, init.shape[0])
        tmp[:m] = init[:m, :d] if init.shape[1] >= d else 0
        init = tmp
    init = init / np.linalg.norm(init, axis=1, keepdims=True)
    # quick polish of record to lock baseline
    best = _lp_polish(init, 6.0, DEADLINE - 100.0 if DEADLINE - 100.0 > time.time() else None)
    best_val = _max_cos(best)
    pool = [(best_val, best.copy())]
    # build a couple diverse seeds quickly
    for _ in range(2):
        if time.time() >= DEADLINE - 120:
            break
        r = rng.standard_normal((n, d))
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        r = _repulse(r, 400, 8.0, 60.0, 0.08, 0.01, 0.7, 0.02, rng)
        v = _max_cos(r)
        pool.append((v, r.copy()))
        if v < best_val:
            best, best_val = r.copy(), v
    trial = 0
    SEARCH_END = t0 + 138.0
    while time.time() < SEARCH_END and time.time() < DEADLINE - 25:
        trial += 1
        u = rng.random()
        # pick base
        if u < 0.5:
            base = best
        else:
            bi = int(rng.integers(len(pool)))
            base = pool[bi][1]
        x = base.copy()
        mode = rng.random()
        if mode < 0.30:
            # large global kick to leave basin
            sigma = float(rng.choice(np.array([0.25, 0.35, 0.5, 0.7])))
            x = x + sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            it = int(rng.integers(500, 1100))
            k0 = float(rng.choice(np.array([6.0, 10.0, 15.0])))
            k1 = k0 * float(rng.choice(np.array([4.0, 6.0, 8.0])))
            x = _repulse(x, it, k0, k1, 0.10, 0.008, 0.8, 0.015, rng)
        elif mode < 0.60:
            # surgical void reinsertion targeting overloaded + rattler neighbourhood
            k = int(rng.integers(4, 16))
            nc = int(rng.choice(np.array([30, 80, 200])))
            x = _void_reinsert(x, k, nc, rng)
            it = int(rng.integers(500, 1000))
            x = _repulse(x, it, 10.0, 80.0, 0.08, 0.008, 0.75, 0.005, rng)
        elif mode < 0.75:
            # crossover between pool members
            if len(pool) >= 2:
                i1 = int(rng.integers(len(pool)))
                i2 = int(rng.integers(len(pool)))
                if i1 == i2:
                    i2 = (i2 + 1) % len(pool)
                x = _crossover(pool[i1][1], pool[i2][1], rng)
                x = _repulse(x, 700, 8.0, 70.0, 0.09, 0.008, 0.8, 0.01, rng)
            else:
                x = x + 0.3 * rng.standard_normal(x.shape)
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                x = _repulse(x, 700, 10.0, 80.0, 0.09, 0.008, 0.8, 0.01, rng)
        else:
            # fresh random basin
            x = rng.standard_normal((n, d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            it = int(rng.integers(800, 1400))
            x = _repulse(x, it, 6.0, 90.0, 0.10, 0.006, 0.8, 0.0, rng)
        v = _max_cos(x)
        # keep pool diverse (size 6)
        pool.append((v, x.copy()))
        pool = sorted(pool, key=lambda t: t[0])[:6]
        # only LP-polish if promising (within 0.03 of best) to save time
        if v < best_val + 0.03:
            budget = 2.0 if v > best_val else 4.0
            # don't exceed search end too much
            y = _lp_polish(x, budget, SEARCH_END + 10)
            vy = _max_cos(y)
            # update pool with polished
            pool.append((vy, y.copy()))
            pool = sorted(pool, key=lambda t: t[0])[:6]
            if vy < best_val:
                best, best_val = y.copy(), vy
    # final polish of champion with remaining time
    remain = DEADLINE - time.time() - 1.0
    if remain > 1.0:
        best = _lp_polish(best, min(remain, 25.0), DEADLINE - 0.5)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
