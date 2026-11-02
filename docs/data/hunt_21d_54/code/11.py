import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def proj_unit(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    x = x / n
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

def contact_info(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cnt = (G > m - 1e-9).sum(axis=1)
    per_max = G.max(axis=1)
    return cnt, per_max, m, G

def smooth_anneal(y, steps, T0, T1, lr0, lr1, beta=0.85):
    n, d = y.shape
    v = np.zeros_like(y)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 * (T1 / T0) ** frac if T0 > 0 and T1 > 0 else T0
        lr = lr0 * (lr1 / lr0) ** frac if lr0 > 0 and lr1 > 0 else lr0
        G = y @ y.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ y) / rs
        grad -= (np.sum(grad * y, axis=1, keepdims=True)) * y
        v = beta * v + (1.0 - beta) * grad
        y = y - lr * v
        y = proj_unit(y)
    return y

def refine_single(A, y0, steps=200, lr0=0.1, lr1=0.005, width0=0.03, width1=0.002):
    y = y0 / (np.linalg.norm(y0) + 1e-12)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        lr = lr0 * (lr1 / lr0) ** frac
        width = width0 * (width1 / width0) ** frac
        dots = A @ y
        m = float(dots.max())
        act = dots > m - width
        if not np.any(act):
            act[np.argmax(dots)] = True
        g = A[act].mean(axis=0)
        g -= float(g @ y) * y
        y = y - lr * g
        y = y / (np.linalg.norm(y) + 1e-12)
    return y

def find_hole(A, rng, n_sample=4000):
    d = A.shape[1]
    C = rng.normal(size=(n_sample, d))
    C = proj_unit(C)
    scores = (C @ A.T).max(axis=1)
    idx = np.argpartition(scores, 8)[:8]
    best_y = None
    best_s = 1e9
    for k in idx:
        y = refine_single(A, C[k])
        s = float((A @ y).max())
        if s < best_s:
            best_s = s
            best_y = y
    return best_y, best_s

def lp_polish(x, seconds, t_deadline):
    x = proj_unit(x)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < min(t_deadline, time.time() + 1000) and radius > 1e-14:
        if time.time() >= t_deadline:
            break
        # remaining
        remain = t_deadline - time.time()
        if remain < 0.3:
            break
        if seconds is not None and seconds <= 0:
            break
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 2500:
            order = np.argpartition(-c[sel], 2500)[:2500]
            I, J = I[order], J[order]
            csv = c[sel][order]
            p = 2500
        else:
            csv = c[sel]
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
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": min(4.0, max(0.5, remain - 0.2))})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = proj_unit(y)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    T0 = time.time()
    LIMIT = 176.0
    deadline = T0 + LIMIT
    rng = np.random.default_rng(0)
    try:
        x0 = np.load("best.npy")
    except Exception:
        x0 = np.load("record.npy")
    x0 = proj_unit(np.array(x0, dtype=float))
    if x0.shape != (54, 21):
        # fallback reshape-safe
        x0 = proj_unit(rng.normal(size=(54, 21)))
    best = x0.copy()
    best_val = max_cos(best)
    cur = best.copy()
    cur_val = best_val
    pool_vals = [best_val]
    pool_xs = [best.copy()]
    def push_pool(y, v):
        # keep top 5
        for k in range(len(pool_vals)):
            if v < pool_vals[k] - 1e-12:
                pool_vals.insert(k, v)
                pool_xs.insert(k, y.copy())
                break
        else:
            if len(pool_vals) < 5:
                pool_vals.append(v)
                pool_xs.append(y.copy())
        while len(pool_vals) > 5:
            pool_vals.pop()
            pool_xs.pop()

    phase1_end = T0 + 118.0
    it = 0
    no_imp = 0
    while time.time() < min(phase1_end, deadline - 45.0):
        it += 1
        n, d = cur.shape
        mode = it % 7
        y = cur.copy()
        if mode == 0:
            sig = 0.02 + 0.02 * ((it // 7) % 3)
            noise = rng.normal(size=(n, d)) * sig
            noise -= (np.sum(noise * y, axis=1, keepdims=True)) * y
            y = proj_unit(y + noise)
            y = smooth_anneal(y, steps=350, T0=0.015, T1=0.002, lr0=0.08, lr1=0.01)
        elif mode == 1:
            cnt, pm, m, G = contact_info(cur)
            order = np.argsort(-cnt)
            k = 4 + (it % 3)
            targets = order[:k].tolist()
            # add one random
            targets.append(int(rng.integers(0, n)))
            targets = list(set(targets))
            for i in targets:
                v = rng.normal(size=d)
                v -= float(v @ y[i]) * y[i]
                nv = np.linalg.norm(v) + 1e-12
                v /= nv
                ang = 0.08 + rng.random() * 0.25
                y[i] = np.cos(ang) * y[i] + np.sin(ang) * v
            y = proj_unit(y)
            y = smooth_anneal(y, steps=450, T0=0.02, T1=0.0025, lr0=0.1, lr1=0.015)
        elif mode == 2 or mode == 3:
            cnt, pm, m, G = contact_info(cur)
            # worst by per-point max then counts
            worst = int(np.argmax(pm + 0.001 * cnt))
            if mode == 3:
                # two worst
                order = np.argsort(-(pm + 0.001 * cnt))
                worsts = order[:2].tolist()
            else:
                worsts = [worst]
            for w in worsts:
                A = np.delete(y, w, axis=0)
                hole, _ = find_hole(A, rng, n_sample=3000)
                y[w] = hole
            y = proj_unit(y)
            y = smooth_anneal(y, steps=500, T0=0.012, T1=0.0015, lr0=0.08, lr1=0.01)
        elif mode == 4:
            # global shake + high-T anneal
            sig = 0.05
            noise = rng.normal(size=(n, d)) * sig
            noise -= (np.sum(noise * y, axis=1, keepdims=True)) * y
            y = proj_unit(y + noise)
            y = smooth_anneal(y, steps=600, T0=0.03, T1=0.003, lr0=0.12, lr1=0.02)
        elif mode == 5:
            # single random point reinsert at hole
            w = int(rng.integers(0, n))
            A = np.delete(y, w, axis=0)
            hole, _ = find_hole(A, rng, n_sample=4000)
            y[w] = hole
            y = proj_unit(y)
            y = smooth_anneal(y, steps=400, T0=0.01, T1=0.0015, lr0=0.06, lr1=0.01)
        else:
            # small jitter all + low-T quench
            sig = 0.01
            noise = rng.normal(size=(n, d)) * sig
            noise -= (np.sum(noise * y, axis=1, keepdims=True)) * y
            y = proj_unit(y + noise)
            y = smooth_anneal(y, steps=300, T0=0.008, T1=0.001, lr0=0.05, lr1=0.008)
        v = max_cos(y)
        push_pool(y, v)
        if v < best_val - 1e-12:
            # improved without LP: adopt
            best, best_val = y.copy(), v
            cur, cur_val = y.copy(), v
            no_imp = 0
        elif v < cur_val - 1e-12:
            cur, cur_val = y.copy(), v
            no_imp = 0
        else:
            # sideways acceptance
            if v < cur_val + 0.004 and rng.random() < 0.25:
                cur, cur_val = y.copy(), v
            no_imp += 1
            if no_imp > 60:
                cur, cur_val = best.copy(), best_val
                no_imp = 0
        # quick time check
        if time.time() >= min(phase1_end, deadline - 45.0):
            break

    # Phase 2: LP polish most promising
    # order pool by val
    order = np.argsort(np.array(pool_vals))
    # ensure best is first
    candidates = []
    candidates.append(best.copy())
    for o in order:
        if len(candidates) >= 4:
            break
        y = pool_xs[o]
        # skip duplicates of best
        if np.allclose(y, best):
            continue
        candidates.append(y.copy())
    # allocate time
    remaining = deadline - time.time() - 1.0
    if remaining < 5:
        return proj_unit(best)
    per = remaining / len(candidates)
    per = min(per, 18.0)
    for y in candidates:
        if time.time() >= deadline - 2.0:
            break
        t_d = min(time.time() + per, deadline - 1.0)
        yp = lp_polish(y, seconds=per, t_deadline=t_d)
        vp = max_cos(yp)
        if vp < best_val:
            best, best_val = yp, vp
    return proj_unit(best)
