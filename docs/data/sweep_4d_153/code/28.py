import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def smooth_relax(x, iters, k0, k1, lr0, lr1, mu, noise, rng):
    n, d = x.shape
    v = np.zeros_like(x)
    for t in range(iters):
        frac = t / max(1, iters - 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
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
        if mu > 0:
            vc = v / (1.0 - mu ** (t + 1))
        else:
            vc = v
        x = x - lr * vc
        if noise > 0 and frac < 0.6:
            amp = noise * (1.0 - frac / 0.6)
            pert = rng.standard_normal(x.shape)
            pd = np.sum(pert * x, axis=1, keepdims=True)
            pert -= pd * x
            x = x + amp * pert
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
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
        vv = max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def cap_reshuffle_move(x, rng):
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    maxper = G.max(axis=1)
    hub_cand = np.argsort(-maxper)[:10]
    hub_idx = int(hub_cand[rng.integers(len(hub_cand))])
    sparse_idx = int(np.argmin(maxper))
    if rng.random() < 0.35:
        cand = rng.standard_normal((2000, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ x.T
        md = dots.max(axis=1)
        sparse_center = cand[int(np.argmin(md))]
    else:
        sparse_center = x[sparse_idx]
    thr_d = float(rng.uniform(0.70, 0.90))
    thr_s = float(rng.uniform(0.70, 0.90))
    idx_d = np.where(x @ x[hub_idx] > thr_d)[0]
    idx_s = np.where(x @ sparse_center > thr_s)[0]
    idx = np.union1d(idx_d, idx_s)
    max_rem = int(rng.integers(8, 21))
    if len(idx) > max_rem:
        idx = rng.choice(idx, size=max_rem, replace=False)
    if len(idx) < 6:
        need = 6 - len(idx)
        extra = np.argsort(-maxper)
        add = []
        for e in extra:
            if e not in idx:
                add.append(int(e))
                if len(add) >= need:
                    break
        idx = np.union1d(idx, np.array(add, dtype=int)) if len(add) else idx
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[idx] = False
    Xcur = x[keep_mask].copy()
    m = len(idx)
    new_pts = np.empty((m, d))
    ncand = 2000
    for k in range(m):
        cand = rng.standard_normal((ncand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ Xcur.T
        md = dots.max(axis=1)
        order = np.argsort(md)
        if rng.random() < 0.2:
            pick = int(order[int(rng.integers(1, min(5, ncand)))])
        else:
            pick = int(order[0])
        chosen = cand[pick]
        new_pts[k] = chosen
        Xcur = np.vstack([Xcur, chosen[None, :]])
    x_new = x.copy()
    x_new[idx] = new_pts
    sigma = float(rng.uniform(0.0, 0.03))
    if sigma > 0:
        x_new[idx] += sigma * rng.standard_normal((m, d))
        x_new[idx] /= np.linalg.norm(x_new[idx], axis=1, keepdims=True)
    return x_new

def block_rotation_move(x, rng):
    n, d = x.shape
    if rng.random() < 0.5:
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        maxper = G.max(axis=1)
        hub_cand = np.argsort(-maxper)[:12]
        hub_idx = int(hub_cand[rng.integers(len(hub_cand))])
        center = x[hub_idx].copy()
    else:
        center = rng.standard_normal(d)
        center /= np.linalg.norm(center)
    thr = float(rng.uniform(0.75, 0.92))
    dots = x @ center
    idx = np.where(dots > thr)[0]
    if len(idx) < 5:
        idx = np.argsort(-dots)[:int(rng.integers(6, 13))]
    if len(idx) > 25:
        idx = rng.choice(idx, size=25, replace=False)
    A = rng.standard_normal((d, d))
    A = (A - A.T) * 0.5
    ang = float(rng.uniform(0.05, 0.35))
    nrm = float(np.linalg.norm(A))
    if nrm > 1e-12:
        A = A / nrm * ang
    try:
        from scipy.linalg import expm
        R = expm(A)
    except Exception:
        R = np.eye(d) + A
    x_new = x.copy()
    x_new[idx] = (R @ x_new[idx].T).T
    sigma = float(rng.uniform(0.01, 0.06))
    x_new[idx] += sigma * rng.standard_normal((len(idx), d))
    x_new /= np.linalg.norm(x_new, axis=1, keepdims=True)
    return x_new

def point_kick_move(x, rng):
    n, d = x.shape
    x_new = x.copy()
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    maxper = G.max(axis=1)
    k = int(rng.integers(3, 10))
    idx = np.argsort(-maxper)[:k]
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[idx] = False
    Xcur = x[keep_mask].copy()
    for j in idx:
        nc = 2000
        cand = rng.standard_normal((nc, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ Xcur.T
        md = dots.max(axis=1)
        order = np.argsort(md)
        if rng.random() < 0.2:
            pick = int(order[int(rng.integers(1, min(5, nc)))])
        else:
            pick = int(order[0])
        x_new[j] = cand[pick]
        Xcur = np.vstack([Xcur, x_new[j][None, :]])
    return x_new

def solve():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    n, d = 153, 4
    try:
        best = np.load("best.npy")
    except Exception:
        try:
            best = np.load("record.npy")
        except Exception:
            best = rng.standard_normal((n, d))
    best = np.asarray(best, dtype=float)
    if best.shape != (n, d):
        tmp = rng.standard_normal((n, d))
        best = tmp
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    try:
        rec = np.load("record.npy")
        rec = np.asarray(rec, dtype=float)
        if rec.shape == (n, d):
            rec = rec / np.linalg.norm(rec, axis=1, keepdims=True)
            if max_cos(rec) < max_cos(best):
                best = rec.copy()
    except Exception:
        pass
    best_val = max_cos(best)
    incumbent = best.copy()
    inc_val = best_val
    cands = [(inc_val, incumbent.copy())]
    SEARCH_END = t0 + 108
    trial = 0
    while time.time() < SEARCH_END:
        trial += 1
        rbase = rng.random()
        if rbase < 0.60:
            base = incumbent
        elif rbase < 0.85:
            base = best
        else:
            base = cands[int(rng.integers(len(cands)))][1]
        x = base.copy()
        mode = rng.random()
        try:
            if mode < 0.32:
                x = point_kick_move(x, rng)
            elif mode < 0.62:
                x = cap_reshuffle_move(x, rng)
            elif mode < 0.82:
                x = block_rotation_move(x, rng)
            else:
                sigma = float(rng.choice(np.array([0.03, 0.07, 0.12, 0.20])))
                x = x + sigma * rng.standard_normal(x.shape)
                x /= np.linalg.norm(x, axis=1, keepdims=True)
        except Exception:
            continue
        k0 = float(rng.choice(np.array([3.0, 6.0, 10.0, 15.0, 20.0])))
        k1 = float(rng.choice(np.array([120.0, 200.0, 300.0, 450.0])))
        lr0 = float(rng.choice(np.array([0.02, 0.035, 0.05])))
        lr1 = float(rng.choice(np.array([0.001, 0.002, 0.004])))
        mu = float(rng.choice(np.array([0.0, 0.8, 0.9])))
        noise = float(rng.choice(np.array([0.0, 0.0, 0.005, 0.012])))
        iters = int(rng.choice(np.array([500, 800, 1100, 1400])))
        if time.time() > SEARCH_END - 2:
            break
        try:
            xo = smooth_relax(x, iters, k0, k1, lr0, lr1, mu, noise, rng)
        except Exception:
            continue
        v = max_cos(xo)
        if v < inc_val - 1e-12:
            inc_val = v
            incumbent = xo.copy()
            cands.append((v, xo.copy()))
            cands = sorted(cands, key=lambda z: z[0])[:5]
        if trial % 22 == 0:
            xr = rng.standard_normal((n, d))
            xr /= np.linalg.norm(xr, axis=1, keepdims=True)
            try:
                xo2 = smooth_relax(xr, 900, 6.0, 250.0, 0.04, 0.002, 0.85, 0.0, rng)
            except Exception:
                continue
            v2 = max_cos(xo2)
            if v2 < inc_val - 1e-12:
                inc_val = v2
                incumbent = xo2.copy()
                cands.append((v2, xo2.copy()))
                cands = sorted(cands, key=lambda z: z[0])[:5]
        if time.time() > SEARCH_END:
            break
    cands = sorted(cands, key=lambda z: z[0])
    if best_val < cands[0][0]:
        cands = [(best_val, best.copy())] + cands
        cands = cands[:5]
    time_left = 170 - (time.time() - t0)
    if time_left < 12:
        time_left = 12
    per = time_left / min(3, len(cands))
    polished_best = None
    polished_val = 1e9
    for i, (v, xc) in enumerate(cands[:3]):
        budget = per * (1.6 if i == 0 else 1.0)
        budget = min(budget, 60)
        if time.time() - t0 > 172:
            break
        try:
            xp = lp_polish(xc, budget)
        except Exception:
            xp = xc
        vp = max_cos(xp)
        if vp < polished_val:
            polished_val = vp
            polished_best = xp
    if polished_best is None:
        polished_best = incumbent
    polished_best = polished_best / np.linalg.norm(polished_best, axis=1, keepdims=True)
    bv = max_cos(best)
    pv = max_cos(polished_best)
    if bv <= pv:
        return best
    return polished_best
