import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _proj(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, cap=1500):
    if seconds <= 0.05:
        return x
    x = _proj(np.asarray(x, dtype=np.float64))
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        I = iu0[0][sel]; J = iu0[1][sel]
        csv = c[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > cap:
            idx = np.argpartition(-csv, cap)[:cap]
            I, J = I[idx], J[idx]
            csv = csv[idx]
            p = cap
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        tlim = max(0.4, min(1.5, seconds))
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": tlim})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _proj(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return x

def _refine_point(p0, others, steps=250, lr0=0.08):
    p = p0.copy()
    m = others.shape[0]
    if m == 0:
        return p / (np.linalg.norm(p) + 1e-12)
    for s in range(steps):
        frac = s / max(1, steps)
        K = 100.0 * (1500.0 / 100.0) ** frac
        dots = others @ p
        dm = float(dots.max())
        e = K * (dots - dm)
        np.clip(e, -20, 0, out=e)
        w = np.exp(e)
        sw = w.sum() + 1e-12
        w /= sw
        g = w @ others
        g = g - float(g @ p) * p
        lr = lr0 * (1.0 - 0.5 * frac)
        p = p - lr * g
        nrm = np.linalg.norm(p)
        if nrm == 0 or not np.isfinite(nrm):
            return p0
        p /= nrm
    return p

def _best_hole(current, rng, nh=20000, batch=5000):
    d = current.shape[1]
    best_c = None
    best_s = 1e9
    nb = nh // batch
    for _ in range(nb):
        C = rng.randn(batch, d)
        C = _proj(C)
        D = C @ current.T
        s = D.max(axis=1)
        k = int(np.argmin(s))
        if float(s[k]) < best_s:
            best_s = float(s[k])
            best_c = C[k].copy()
    return best_c

def solve():
    T0 = time.time()
    LIMIT = 160.0
    rng = np.random.RandomState(7)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.randn(54, 21)
    xb = np.asarray(xb, dtype=np.float64)
    if xb.shape != (54, 21):
        flat = xb.ravel()
        need = 54 * 21
        if flat.size < need:
            extra = rng.randn(need - flat.size)
            flat = np.concatenate([flat, extra])
        xb = flat[:need].reshape(54, 21)
    xb = _proj(xb)
    best = xb.copy()
    try:
        best_val = _max_cos(best)
    except Exception:
        best_val = 1.0

    def left():
        return LIMIT - (time.time() - T0)

    n, d = 54, 21
    # contact analysis on current best
    G = best @ best.T
    np.fill_diagonal(G, -2.0)
    m0 = float(G.max())
    # tight contacts for cluster definition
    cnt = (G > m0 - 1e-7).sum(axis=1)
    order = np.argsort(-cnt)  # descending
    # build cluster choices (sizes 5-10)
    clusters = []
    # 0: top-8 by count
    clusters.append(list(order[:8]))
    # 1: seed max + 7 strongest neighbours
    s0 = int(order[0])
    dots0 = (best @ best[s0]).copy()
    dots0[s0] = -2
    nb0 = np.argsort(-dots0)[:7]
    clusters.append([s0] + list(nb0))
    # 2: second seed + 6 neighbours (size 7)
    s1 = int(order[1])
    dots1 = (best @ best[s1]).copy()
    dots1[s1] = -2
    nb1 = np.argsort(-dots1)[:6]
    clusters.append([s1] + list(nb1))
    # 3: top-6
    clusters.append(list(order[:6]))
    # 4: top-5 + 3 random from top-20
    extra = list(rng.choice(order[:20], size=3, replace=False))
    clusters.append(list(order[:5]) + extra)
    # 5: third seed cluster size 9
    s2 = int(order[2])
    dots2 = (best @ best[s2]).copy()
    dots2[s2] = -2
    nb2 = np.argsort(-dots2)[:8]
    clusters.append([s2] + list(nb2))
    # 6: random 7 among top-20
    clusters.append(list(rng.choice(order[:20], size=7, replace=False)))
    # 7: top-10
    clusters.append(list(order[:10]))

    # dedup / clip
    ucl = []
    for cl in clusters:
        cl = [int(c) for c in cl]
        cl = sorted(set(cl))
        if 5 <= len(cl) <= 10:
            ucl.append(cl)
    clusters = ucl

    for ci, cl in enumerate(clusters):
        if left() < 18:
            break
        trial_start = time.time()
        trial_budget = min(12.0, left() - 8.0)
        if trial_budget < 5.0:
            break
        kdel = len(cl)
        mask = np.ones(n, dtype=bool)
        mask[cl] = False
        keep = best[mask].copy()  # (54-k) x 21
        # LP-polish remainder to gain slack/rattlers
        t_use = min(1.2, trial_budget * 0.12)
        keep = _lp_polish(keep, t_use, cap=1200)
        cur = keep.copy()
        # insertion orders: try hole-greedy; also shuffle order effect via different rng stream
        # reinsert one at a time
        ok = True
        for ins in range(kdel):
            if time.time() - trial_start > trial_budget - 3.0:
                ok = False
                break
            if left() < 10:
                ok = False
                break
            hole = _best_hole(cur, rng, nh=20000, batch=5000)
            if hole is None:
                hole = rng.randn(d); hole /= np.linalg.norm(hole) + 1e-12
            hole = _refine_point(hole, cur, steps=250, lr0=0.08)
            cur = np.vstack([cur, hole[None, :]])
            # brief LP polish after each insertion
            cur = _lp_polish(cur, 0.45, cap=1100)
        if not ok:
            continue
        v = _max_cos(cur)
        # keep only candidates below 0.070 for full polishing
        if v < 0.070:
            rem = trial_budget - (time.time() - trial_start)
            full_t = min(2.5, max(0.6, rem - 0.3))
            if full_t > 0.3 and left() > 8:
                cur = _lp_polish(cur, full_t, cap=1800)
                v = _max_cos(cur)
            if v < best_val:
                best = _proj(cur).copy()
                best_val = v
                # refresh contact order for remaining trials
                G2 = best @ best.T
                np.fill_diagonal(G2, -2.0)
        # enforce cutoff
        if time.time() - T0 > LIMIT - 5:
            break

    # final polish of global best with remaining time
    rem = LIMIT - (time.time() - T0)
    if rem > 2.5:
        best = _lp_polish(best, min(rem - 1.0, 12.0), cap=2000)
        best_val = _max_cos(best)
    best = _proj(best)
    return best
