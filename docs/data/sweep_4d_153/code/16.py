import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def lp_polish(x, seconds, r0=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        Ct = (x @ x.T)[iu]
        m = Ct.max()
        win = max(20.0 * radius, 1e-9)
        sel = Ct > m - win
        # cap constraints for speed
        if int(sel.sum()) > 4000:
            k = 4000
            top = np.argpartition(-Ct, k)[:k]
            ns = np.zeros_like(sel)
            ns[top] = True
            sel = ns
        I, J = iu[0][sel], iu[1][sel]
        csub = Ct[sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                         shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
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
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def smooth_relax(x, iters, k0, k1, lr0, lr1, mu, noise, rng):
    n, d = x.shape
    v = np.zeros_like(x)
    for t in range(iters):
        frac = t / max(1, iters - 1)
        k = k0 * (k1 / k0) ** frac if k0 > 0 else k1
        lr = lr0 * (lr1 / lr0) ** frac if lr0 > 0 else lr1
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
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
        if noise > 0 and frac < 0.6:
            amp = noise * (1.0 - frac / 0.6)
            pert = rng.standard_normal(x.shape)
            pd = np.sum(pert * x, axis=1, keepdims=True)
            pert -= pd * x
            x = x + amp * pert
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def void_reinsert(x, k, nc, rng):
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    maxper = G.max(axis=1)
    m = G.max()
    idx = list(np.argsort(-maxper)[:k])
    # also move a rattler if exists
    ratt = np.where(maxper < m - 0.002)[0]
    if len(ratt) > 0:
        r = int(ratt[rng.integers(len(ratt))])
        if r not in idx:
            idx[-1] = r
    idx = np.array(idx, dtype=int)
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[idx] = False
    cur = x[keep_mask]
    new_pts = []
    for _ in idx:
        cand = rng.standard_normal((nc, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ cur.T
        vals = dots.max(axis=1)
        pt = cand[int(np.argmin(vals))]
        new_pts.append(pt)
        cur = np.vstack([cur, pt[None, :]])
    x2 = x.copy()
    for ii, j in enumerate(idx):
        x2[j] = new_pts[ii]
    return x2

def solve():
    t0 = time.time()
    DEADLINE = t0 + 168.0
    SEARCH_END = t0 + 118.0
    rng = np.random.default_rng(0)
    n, d = 153, 4
    try:
        base0 = np.load("best.npy")
    except Exception:
        try:
            base0 = np.load("record.npy")
        except Exception:
            base0 = rng.standard_normal((n, d))
    base0 = np.asarray(base0, dtype=float)
    if base0.shape != (n, d):
        base0 = rng.standard_normal((n, d))
    base0 = base0 / np.linalg.norm(base0, axis=1, keepdims=True)
    incumbent = base0.copy()
    inc_val = max_cos(incumbent)
    pool = [(inc_val, incumbent.copy())]
    trial = 0
    # quick pre-polish incumbent briefly if time allows (already polished, skip)
    while time.time() < SEARCH_END:
        trial += 1
        time_left = DEADLINE - time.time()
        if time_left < 12:
            break
        # choose base
        rb = rng.random()
        if rb < 0.70 or len(pool) == 1:
            base = incumbent
        else:
            # pick random from pool weighted to best
            bi = int(rng.integers(len(pool)))
            base = pool[bi][1]
        x = base.copy()
        mode = rng.random()
        if mode < 0.32:
            sigma = float(rng.uniform(0.005, 0.035))
            x = x + sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif mode < 0.58:
            sigma = float(rng.uniform(0.04, 0.13))
            x = x + sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif mode < 0.90:
            k = int(rng.integers(2, 11))
            nc = int(rng.integers(150, 501))
            # sometimes add extra noise to keepers
            if rng.random() < 0.4:
                sig = float(rng.uniform(0.005, 0.03))
                x = x + sig * rng.standard_normal(x.shape)
                x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = void_reinsert(x, k, nc, rng)
        else:
            # fresh random or large shuffle
            if rng.random() < 0.5:
                x = rng.standard_normal((n, d))
                x /= np.linalg.norm(x, axis=1, keepdims=True)
            else:
                sigma = 0.25
                x = x + sigma * rng.standard_normal(x.shape)
                x /= np.linalg.norm(x, axis=1, keepdims=True)
        iters = int(rng.integers(600, 1800))
        k0 = float(rng.uniform(15, 80))
        k1 = float(k0 * rng.uniform(5, 30))
        if k1 > 2500:
            k1 = 2500.0
        lr0 = float(rng.uniform(0.005, 0.022))
        lr1 = float(lr0 * rng.uniform(0.08, 0.5))
        mu = float(rng.uniform(0.5, 0.92))
        noise = float(rng.uniform(0.0, 0.012)) if rng.random() < 0.6 else 0.0
        x = smooth_relax(x, iters, k0, k1, lr0, lr1, mu, noise, rng)
        v = max_cos(x)
        # track pool (keep 6 best relaxed)
        if v < pool[-1][0] + 1e-12 or len(pool) < 6:
            pool.append((v, x.copy()))
            pool = sorted(pool, key=lambda z: z[0])[:6]
        if v < inc_val - 2e-5:
            # clear gain: high-k tighten + short LP
            x2 = smooth_relax(x.copy(), 400, 800.0, 2500.0, 0.004, 0.001, 0.85, 0.0, rng)
            v2 = max_cos(x2)
            if v2 < v:
                x, v = x2, v2
            budget = min(5.0, max(1.5, DEADLINE - time.time() - 12))
            if budget > 1.0:
                xp = lp_polish(x, budget, r0=8e-4)
                vp = max_cos(xp)
            else:
                xp, vp = x, v
            if vp < inc_val:
                incumbent, inc_val = xp, vp
                pool = sorted([(vp, xp.copy())] + pool, key=lambda z: z[0])[:6]
        elif v < inc_val + 0.0025 and (trial % 7 == 0 or v < inc_val):
            # occasional opportunistic LP on near-incumbent
            budget = min(3.0, max(1.0, SEARCH_END - time.time()))
            if budget >= 1.0 and time.time() + budget < DEADLINE - 10:
                xp = lp_polish(x, budget, r0=8e-4)
                vp = max_cos(xp)
                if vp < inc_val:
                    incumbent, inc_val = xp, vp
                    pool = sorted([(vp, xp.copy())] + pool, key=lambda z: z[0])[:6]
    # final polishing of pool with remaining time
    pool = sorted(pool, key=lambda z: z[0])
    # ensure incumbent first
    # re-sort with incumbent
    if max_cos(incumbent) < pool[0][0]:
        pool = [(max_cos(incumbent), incumbent.copy())] + pool
    pool = sorted(pool, key=lambda z: z[0])[:4]
    for i, (pv, px) in enumerate(list(pool)):
        if time.time() > DEADLINE - 6:
            break
        budget = (DEADLINE - 4 - time.time()) / (len(pool) - i)
        budget = float(max(2.0, min(12.0, budget)))
        if budget < 1.0:
            break
        xp = lp_polish(px, budget, r0=6e-4 if pv < inc_val + 1e-6 else 1e-3)
        vp = max_cos(xp)
        if vp < inc_val:
            incumbent, inc_val = xp.copy(), vp
    incumbent = incumbent / np.linalg.norm(incumbent, axis=1, keepdims=True)
    # safety: remove NaN/zero
    incumbent = np.nan_to_num(incumbent, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(incumbent, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    incumbent = incumbent / nr
    return incumbent
