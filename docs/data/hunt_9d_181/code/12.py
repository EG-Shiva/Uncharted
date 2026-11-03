import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, seconds, init_radius=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = init_radius
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start + 0.25 > seconds and radius < 6e-4:
            break
        G = x @ x.T
        c_all = G[iu]
        m = c_all.max()
        thresh = max(20.0 * radius, 1e-9)
        sel = c_all > m - thresh
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 2500:
            kth = np.partition(c_all, -2500)[-2500]
            sel = c_all >= kth
            if int(sel.sum()) > 2800:
                idx = np.where(sel)[0]
                order = np.argsort(-c_all[idx])[:2500]
                mask = np.zeros_like(c_all, dtype=bool)
                mask[idx[order]] = True
                sel = mask
        I = iu[0][sel]; J = iu[1][sel]; csel = c_all[sel]
        p = len(I); nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def smooth_exp(x, alpha, steps, s0):
    n, d = x.shape
    m = np.zeros_like(x)
    mu = 0.9
    for k in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        W = np.exp(alpha * (G - gmax))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        m = mu * m + (1.0 - mu) * gt
        rms = np.sqrt(np.mean(m * m)) + 1e-12
        step = s0 * (1.0 - 0.5 * k / max(1, steps))
        x = x - (step / rms) * m
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def contact_counts(x, tol=2e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    B = G > m - tol
    return np.sum(B, axis=1), m, G

def find_void(x_keep, rng, n_cand=3000):
    n, d = x_keep.shape
    cand = rng.standard_normal((n_cand, x_keep.shape[1]))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ x_keep.T
    vals = dots.max(axis=1)
    return cand[int(np.argmin(vals))]

def solve():
    t0 = time.time()
    BUDGET = 165.0
    rng = np.random.default_rng(7)
    cand = []
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (181, 9):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cand.append(a)
        except Exception:
            pass
    if not cand:
        a = rng.standard_normal((181, 9))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cand.append(a)
    scores = [max_cos(a) for a in cand]
    xb = cand[int(np.argmin(scores))].copy()
    # quick baseline polish
    rem = BUDGET - (time.time() - t0)
    if rem > 20:
        xb = lp_polish(xb, seconds=min(4.0, rem - 15.0), init_radius=2e-3)
    vb = max_cos(xb)
    x_best, v_best = xb.copy(), vb
    x_cur, v_cur = xb.copy(), vb
    n, d = xb.shape
    it = 0
    T0 = 8e-05
    while True:
        now = time.time() - t0
        rem = BUDGET - now
        if rem < 8.0:
            break
        it += 1
        cnt, mcur, Gcur = contact_counts(x_cur)
        y = x_cur.copy()
        mode = it % 4
        if mode == 0:
            # rattler / loose void fill: move 1-2 lowest-contact points
            k = 1 if rng.random() < 0.7 else 2
            order = np.argsort(cnt + rng.random(n) * 0.5)
            mov = order[:k]
            keep_mask = np.ones(n, dtype=bool); keep_mask[mov] = False
            xk = y[keep_mask]
            for j in mov:
                # recompute keep without j (use current y for others)
                mask2 = np.ones(n, dtype=bool); mask2[j] = False
                v = find_void(y[mask2], rng, n_cand=3000)
                y[j] = v
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            # tiny jitter others
            y = y + rng.standard_normal((n, d)) * 0.002
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            a1 = float(rng.choice([80.0, 120.0, 180.0]))
            y = smooth_exp(y, a1, steps=400, s0=0.005)
        elif mode == 1:
            # overloaded cluster shake
            probs = (cnt.astype(float) + 2.0)
            probs /= probs.sum()
            hub = int(rng.choice(n, p=probs))
            # neighbors in contact
            row = Gcur[hub]
            nbrs = np.where(row > mcur - 0.01)[0]
            if len(nbrs) > 12:
                nbrs = rng.choice(nbrs, size=12, replace=False)
            tgt = np.concatenate([[hub], nbrs])
            tgt = np.unique(tgt)
            sig_soft = float(rng.uniform(0.002, 0.008))
            sig_hard = float(rng.uniform(0.03, 0.10))
            y = y + rng.standard_normal((n, d)) * sig_soft
            y[tgt] += rng.standard_normal((len(tgt), d)) * sig_hard
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            a1 = float(rng.choice([60.0, 100.0, 160.0]))
            y = smooth_exp(y, a1, steps=500, s0=0.006)
        elif mode == 2:
            # random subset hard kick
            k = int(rng.integers(4, 11))
            idx = rng.choice(n, size=k, replace=False)
            sig = float(rng.uniform(0.04, 0.12))
            y[idx] += rng.standard_normal((k, d)) * sig
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            a1 = float(rng.choice([80.0, 150.0]))
            y = smooth_exp(y, a1, steps=500, s0=0.006)
        else:
            # global soft jitter
            sig = float(rng.uniform(0.006, 0.02))
            y = y + rng.standard_normal((n, d)) * sig
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            a1 = float(rng.choice([100.0, 180.0]))
            y = smooth_exp(y, a1, steps=400, s0=0.005)
        v_sm = max_cos(y)
        # gate: skip LP if far worse
        if v_sm > v_best + 0.006 and v_sm > v_cur + 0.003:
            # still allow occasional move without LP
            if rng.random() < 0.05:
                x_cur, v_cur = y, v_sm
            continue
        # short LP eval, time-aware
        now2 = time.time() - t0
        rem2 = BUDGET - now2
        if rem2 < 8.0:
            # no time for LP, just greedy on smoothed
            if v_sm < v_best:
                x_best, v_best = y.copy(), v_sm
                x_cur, v_cur = y.copy(), v_sm
            continue
        lp_time = min(2.2, max(0.8, rem2 - 7.0))
        # reduce time for far-worse candidates
        if v_sm > v_best + 0.002:
            lp_time = min(lp_time, 1.2)
        y2 = lp_polish(y, seconds=lp_time, init_radius=0.006)
        v2 = max_cos(y2)
        # acceptance: greedy for best, SA for walker
        if v2 < v_best - 1e-12:
            x_best, v_best = y2.copy(), v2
            x_cur, v_cur = y2.copy(), v2
        else:
            frac = (time.time() - t0) / BUDGET
            T = T0 * (1.0 - 0.7 * frac)
            if v2 < v_cur + T or rng.random() < 0.02:
                x_cur, v_cur = y2.copy(), v2
            elif v_sm < v_cur - 5e-05 and rng.random() < 0.3:
                x_cur, v_cur = y.copy(), v_sm
        # restart walker if stuck far above best
        if v_cur > v_best + 0.002 and rng.random() < 0.15:
            x_cur, v_cur = x_best.copy(), v_best
    # final polish of best
    rem = BUDGET - (time.time() - t0)
    if rem > 1.5:
        x_best = lp_polish(x_best, seconds=max(1.0, rem - 0.5), init_radius=1e-3)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
