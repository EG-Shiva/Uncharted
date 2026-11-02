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
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    n2 = np.linalg.norm(x, axis=1, keepdims=True)
    n2[n2 == 0] = 1.0
    return x / n2

def smooth_anneal(x, iters=1500, T0=0.02, T1=0.0012, lr0=0.06, lr1=0.004, mu=0.85):
    n, d = x.shape
    y = x.copy()
    vel = np.zeros_like(y)
    for t in range(iters):
        frac = t / max(1, iters - 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (1 - frac) + lr1 * frac
        G = y @ y.T
        np.fill_diagonal(G, -1e9)
        m = G.max()
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        s = W.sum()
        if s <= 0 or not np.isfinite(s):
            break
        W /= s
        grad = W @ y
        # tangent projection
        grad -= (np.sum(grad * y, axis=1, keepdims=True)) * y
        gn = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-12
        ghat = grad / gn
        # project momentum to tangent
        vel -= (np.sum(vel * y, axis=1, keepdims=True)) * y
        vel = mu * vel + (1.0 - mu) * ghat
        # project again
        vel -= (np.sum(vel * y, axis=1, keepdims=True)) * y
        y = y - lr * vel
        # also direct step to keep moving
        # small direct component helps
        y = y - (lr * 0.5) * ghat
        y = proj_unit(y)
    return y

def lp_polish(x, seconds):
    x = proj_unit(x)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        remain = seconds - (time.time() - start)
        if remain <= 0.05:
            break
        G = x @ x.T
        c = G[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        csv = c[sel]
        if p > 3000:
            idx = np.argpartition(-csv, 3000)[:3000]
            I, J = I[idx], J[idx]
            csv = csv[idx]
            p = 3000
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": min(5.0, max(0.5, remain - 0.05))})
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

def contact_counts(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    C = (G > m - 1e-9).sum(axis=1)
    return C, m, G

def perturb_hole(x, rng, n_reinit):
    n, d = x.shape
    cnts, m, G = contact_counts(x)
    # pick worst overloaded points + random
    order = np.argsort(-cnts)
    k = int(n_reinit)
    victims = list(order[:max(1, k // 2)])
    while len(victims) < k:
        victims.append(int(rng.integers(0, n)))
    victims = victims[:k]
    keep = [i for i in range(n) if i not in set(victims)]
    y = x.copy()
    Xk = x[keep] if len(keep) > 0 else x
    for v in victims:
        # find hole: sample candidates, minimize max dot with keepers
        cand = rng.normal(size=(2500, d))
        cand = proj_unit(cand)
        dots = cand @ Xk.T
        md = dots.max(axis=1)
        # also try to stay away from other victims already placed? ignore
        bi = int(np.argmin(md))
        y[v] = cand[bi]
    return y

def perturb_shake(x, rng, sigma_all, sigma_kick, n_kick):
    n, d = x.shape
    cnts, m, G = contact_counts(x)
    order_low = np.argsort(cnts)
    order_high = np.argsort(-cnts)
    targets = set(order_high[:2].tolist())
    targets.update(order_low[:2].tolist())
    while len(targets) < n_kick:
        targets.add(int(rng.integers(0, n)))
    y = x.copy()
    for i in list(targets):
        v = rng.normal(size=d)
        v -= (v @ y[i]) * y[i]
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            continue
        v /= nv
        ang = float(rng.normal() * sigma_kick)
        y[i] = np.cos(ang) * y[i] + np.sin(ang) * v
    noise = rng.normal(size=(n, d)) * sigma_all
    noise -= (np.sum(noise * y, axis=1, keepdims=True)) * y
    y = y + noise
    return proj_unit(y)

def solve():
    T0 = time.time()
    LIMIT = 168.0
    FINAL_POLISH = 22.0
    rng = np.random.default_rng(0)
    try:
        b = np.load("best.npy")
    except Exception:
        b = np.load("record.npy")
    try:
        r = np.load("record.npy")
    except Exception:
        r = b
    b = proj_unit(np.array(b, dtype=float))
    r = proj_unit(np.array(r, dtype=float))
    best = b if max_cos(b) <= max_cos(r) else r
    best_val = max_cos(best)
    # quick squeeze if stuck exactly at parent (already polished) -> skip long initial LP
    # main exploration loop
    trial = 0
    while time.time() - T0 < LIMIT - FINAL_POLISH - 1.0:
        trial += 1
        remain = LIMIT - FINAL_POLISH - (time.time() - T0)
        if remain < 2.0:
            break
        mode = trial % 8
        if mode in (0, 4):
            k = int(rng.integers(2, 6))
            y = perturb_hole(best, rng, k)
            iters = 1600
            T0a, T1a = 0.015, 0.0008
            lr0a = 0.07
        elif mode in (1, 5):
            y = perturb_shake(best, rng, sigma_all=0.02 + 0.03 * rng.random(), sigma_kick=0.15 + 0.35 * rng.random(), n_kick=int(rng.integers(4, 10)))
            iters = 1400
            T0a, T1a = 0.02, 0.001
            lr0a = 0.06
        elif mode == 2:
            y = perturb_shake(best, rng, sigma_all=0.08, sigma_kick=0.5, n_kick=int(rng.integers(6, 12)))
            iters = 1800
            T0a, T1a = 0.025, 0.001
            lr0a = 0.07
        elif mode == 3:
            # full jitter all points medium
            y = perturb_shake(best, rng, sigma_all=0.05, sigma_kick=0.08, n_kick=54)
            iters = 1200
            T0a, T1a = 0.012, 0.0008
            lr0a = 0.05
        elif mode == 6:
            k = int(rng.integers(1, 4))
            y = perturb_hole(best, rng, k)
            # add small global noise
            noise = rng.normal(size=y.shape) * 0.02
            noise -= (np.sum(noise * y, axis=1, keepdims=True)) * y
            y = proj_unit(y + noise)
            iters = 2000
            T0a, T1a = 0.01, 0.0006
            lr0a = 0.05
        else:
            # random orthogonal-ish twist: random rotation of subset
            y = perturb_shake(best, rng, sigma_all=0.01, sigma_kick=0.25, n_kick=int(rng.integers(2, 5)))
            iters = 1400
            T0a, T1a = 0.018, 0.001
            lr0a = 0.06
        # adapt iters to remaining time
        if remain < 10:
            iters = min(iters, 800)
        y = smooth_anneal(y, iters=iters, T0=T0a, T1=T1a, lr0=lr0a, lr1=0.004, mu=0.85)
        v = max_cos(y)
        if v < best_val - 1e-12:
            # promising: squeeze briefly
            tlp = min(3.0, max(1.0, remain * 0.15))
            # only LP if enough time left
            if remain > tlp + 2.0:
                y2 = lp_polish(y, tlp)
                v2 = max_cos(y2)
                if v2 < best_val:
                    best, best_val = y2, v2
                elif v < best_val:
                    best, best_val = y, v
                else:
                    best, best_val = y2, v2
            else:
                best, best_val = y, v
        elif v < best_val + 0.0004 and (trial % 6 == 0):
            # occasional squeeze of near-best basin (topology may improve after LP)
            tlp = min(2.0, max(0.8, remain * 0.1))
            if remain > tlp + 2.0:
                y2 = lp_polish(y, tlp)
                v2 = max_cos(y2)
                if v2 < best_val:
                    best, best_val = y2, v2
        # else discard
    # final squeeze of champion
    frem = LIMIT - (time.time() - T0) - 0.5
    if frem > 1.0:
        frem = min(frem, FINAL_POLISH + 5.0)
        best = lp_polish(best, frem)
        best_val = max_cos(best)
    best = proj_unit(best)
    return best
