import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def lp_polish(x, end_time, r0=0.006):
    x = norm_rows(x)
    n, d = x.shape
    best = max_cos(x)
    xb = x.copy()
    radius = float(r0)
    iu0 = np.triu_indices(n, 1)
    while time.time() < end_time and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(np.max(c_all))
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        # cap constraints for speed
        cnt = int(np.sum(sel))
        if cnt == 0:
            break
        if cnt > 2500:
            # keep tightest 2500
            idx = np.argsort(-c_all)[:2500]
            mask = np.zeros_like(c_all, dtype=bool)
            mask[idx] = True
            sel = sel & mask
            cnt = 2500
        I = iu0[0][sel]; J = iu0[1][sel]
        csel = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),
                               (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
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
        y = norm_rows(y)
        v = max_cos(y)
        if v < best:
            # improvement: accept, enlarge slightly
            x, best, xb = y, v, y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            # check if y is still close and time remains? shrink
            radius *= 0.5
        if time.time() >= end_time:
            break
    return xb

def anneal_adam(x, steps, t_start, t_end, lr0, lr1):
    n, d = x.shape
    m = np.zeros_like(x)
    vv = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    # log interpolation factors
    lt0 = np.log(t_start); lt1 = np.log(t_end)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for t in range(1, steps + 1):
        frac = t / float(steps)
        temp = float(np.exp(lt0 + (lt1 - lt0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mx = float(np.max(G))
        # softmax weights exp((G-mx)/temp)
        W = np.exp((G - mx) / temp)
        np.fill_diagonal(W, 0.0)
        S = float(np.sum(W))
        if not np.finite(S) or S <= 0:
            break
        Grad = 2.0 * (W @ x) / S
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        m = b1 * m + (1 - b1) * Gt
        vv = b2 * vv + (1 - b2) * (Gt * Gt)
        mhat = m / (1.0 - b1 ** t)
        vhat = vv / (1.0 - b2 ** t)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        clip = 0.05
        scale = np.minimum(1.0, clip / sn)
        step = step * scale
        x = x - step
        x = norm_rows(x)
        if not np.all(np.finite(x)):
            break
    return x

def find_hole(Y, rng, n_cand=3000):
    d = Y.shape[1]
    cand = rng.randn(n_cand, d)
    cand = norm_rows(cand)
    dots = cand @ Y.T
    md = np.max(dots, axis=1)
    return cand[int(np.argmin(md))]

def solve():
    t0 = time.time()
    DEADLINE = 172.0
    rng = np.random.RandomState(0)
    cands = []
    try:
        cands.append(np.load("best.npy"))
    except Exception:
        pass
    try:
        cands.append(np.load("record.npy"))
    except Exception:
        pass
    best_val = 1e9
    x_best = None
    for c in cands:
        try:
            c = np.asarray(c, dtype=float)
            if c.shape != (55, 21):
                continue
            c = norm_rows(c)
            v = max_cos(c)
            if v < best_val:
                best_val = v
                x_best = c.copy()
        except Exception:
            continue
    if x_best is None:
        x_best = norm_rows(rng.randn(55, 21))
        best_val = max_cos(x_best)
    n, d = 55, 21
    # initial quick squeeze
    if time.time() - t0 < DEADLINE - 10:
        x_best = lp_polish(x_best, time.time() + min(2.0, max(0.5, DEADLINE - 10 - (time.time() - t0))), r0=1e-3)
        best_val = max_cos(x_best)
    cur = x_best.copy()
    cur_val = best_val
    it = 0
    fail_streak = 0
    while time.time() - t0 < DEADLINE - 9:
        it += 1
        remain = DEADLINE - 9 - (time.time() - t0)
        if remain < 1.0:
            break
        # periodic reset to best if wandering too long
        if fail_streak >= 12 or (it % 25 == 0):
            cur = x_best.copy()
            cur_val = best_val
            fail_streak = 0
        prop = cur.copy()
        mode = it % 5
        if mode == 0 or mode == 1:
            # cluster shake around random center
            c_idx = int(rng.randint(n))
            dots = prop[c_idx] @ prop.T
            k = 4 + int(rng.randint(6))  # 4-9 neighbours
            neigh = np.argsort(-dots)[:k+1]
            sigma_big = 0.06 + rng.rand() * 0.09
            sigma_small = 0.005 + rng.rand() * 0.01
            shift = rng.randn(d)
            shift /= (np.linalg.norm(shift) + 1e-12)
            shift = shift * sigma_big
            prop[neigh] += shift[None, :] * 0.6 + sigma_small * rng.randn(len(neigh), d)
            # light jitter elsewhere
            mask_rest = np.ones(n, dtype=bool); mask_rest[neigh] = False
            prop[mask_rest] += 0.004 * rng.randn(int(np.sum(mask_rest)), d)
            prop = norm_rows(prop)
        elif mode == 2 or mode == 3:
            # hole reinsertion of overloaded/worst points
            G = prop @ prop.T
            np.fill_diagonal(G, -2.0)
            mx = float(np.max(G))
            cnt = np.sum(G > mx - 5e-4, axis=1)
            k = 1 if rng.rand() < 0.6 else 2
            victims = np.argsort(-cnt)[:k]
            if rng.rand() < 0.25:
                victims = rng.choice(n, size=k, replace=False)
            keep = np.ones(n, dtype=bool); keep[victims] = False
            Y = prop[keep]
            for vi in victims:
                h = find_hole(Y, rng, n_cand=3000)
                prop[vi] = h
                Y = np.vstack([Y, h[None, :]])
            prop = prop + 0.008 * rng.randn(n, d)
            prop = norm_rows(prop)
        else:
            # global jitter
            sigma = 0.008 + rng.rand() * 0.022
            # extra shake on overloaded rows
            G = prop @ prop.T
            np.fill_diagonal(G, -2.0)
            mx = float(np.max(G))
            cnt = np.sum(G > mx - 1e-3, axis=1)
            over = np.argsort(-cnt)[:6]
            prop = prop + sigma * rng.randn(n, d)
            prop[over] += sigma * rng.randn(len(over), d)
            prop = norm_rows(prop)
        # annealed soft-max adam
        steps = 600 + int(rng.rand() * 800)
        ts = 0.008 + rng.rand() * 0.012  # 0.008-0.02
        te = 0.0007 + rng.rand() * 0.0013  # 0.0007-0.002
        lr_s = 0.015 + rng.rand() * 0.015
        lr_e = 0.002 + rng.rand() * 0.003
        prop = anneal_adam(prop, steps, ts, te, lr_s, lr_e)
        # short LP squeeze
        secs = 1.4
        if remain < 3:
            secs = max(0.6, remain * 0.5)
        prop = lp_polish(prop, time.time() + secs, r0=0.005)
        v = max_cos(prop)
        if v < best_val - 1e-12:
            best_val = v
            x_best = prop.copy()
            cur = prop.copy(); cur_val = v
            fail_streak = 0
            # extra tight squeeze if time allows
            r2 = DEADLINE - 9 - (time.time() - t0)
            if r2 > 2.5:
                sq = lp_polish(x_best, time.time() + min(1.5, r2 * 0.3), r0=1e-3)
                vs = max_cos(sq)
                if vs < best_val:
                    best_val = vs; x_best = sq.copy()
                    cur = sq.copy(); cur_val = vs
        elif v < cur_val:
            cur = prop.copy(); cur_val = v
            fail_streak = 0
        else:
            fail_streak += 1
            # SA-like sideways move occasionally
            if rng.rand() < 0.15:
                cur = prop.copy(); cur_val = v
        if it > 200:
            break
    rem = DEADLINE - (time.time() - t0)
    if rem > 1.2:
        x_best = lp_polish(x_best, time.time() + min(rem - 0.5, 10.0), r0=1e-3)
    x_best = norm_rows(x_best)
    x_best[~np.isfinite(x_best)] = 0.01
    x_best = norm_rows(x_best)
    return x_best
