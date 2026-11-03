import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-30
    return x / n

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def per_point_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return np.max(g, axis=1)

def contact_counts(x, thresh):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = float(np.max(g))
    cnt = np.sum(g > m - thresh, axis=1)
    return cnt, m

def adam_relax(x, schedule, rng=None):
    # schedule: list of (alpha, steps, lr)
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    for alpha, steps, lr in schedule:
        for _ in range(steps):
            t += 1
            G = x @ x.T
            np.fill_diagonal(G, -10.0)
            gmax = float(np.max(G))
            # avoid overflow: exp(alpha*(G-gmax)), G-gmax<=0
            E = np.exp(alpha * (G - gmax))
            np.fill_diagonal(E, 0.0)
            S = float(np.sum(E))
            if not np.finite(S) or S <= 0:
                break
            grad = (E @ x) * (n / S)
            # tangential projection
            nd = np.sum(grad * x, axis=1, keepdims=True)
            gt = grad - nd * x
            m = b1 * m + (1.0 - b1) * gt
            v = b2 * v + (1.0 - b2) * gt * gt
            bc1 = 1.0 - b1 ** t
            bc2 = 1.0 - b2 ** t
            step = lr * (m / bc1) / (np.sqrt(v / bc2) + eps)
            # remove normal component
            dn = np.sum(step * x, axis=1, keepdims=True)
            step = step - dn * x
            # clip per-point
            rn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
            sc = np.minimum(1.0, 0.05 / rn)
            step = step * sc
            x = x - step
            # renormalize
            x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-30)
    return x

def largest_voids(x, fixed_mask, k, n_cand=4000, rng=None):
    # find k directions minimizing max dot to fixed points
    n, d = x.shape
    if rng is None:
        rng = np.random.RandomState(0)
    cand = rng.randn(n_cand, d)
    cand /= (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-30)
    F = x[fixed_mask] if np.any(fixed_mask) else x
    # max dot of each candidate to F
    dots = cand @ F.T
    md = np.max(dots, axis=1)
    order = np.argsort(md)  # smallest max-dot = largest void
    # greedily separate voids
    picked = []
    for idx in order:
        c = cand[idx]
        ok = True
        for p in picked:
            if float(c @ p) > 0.85:
                ok = False
                break
        if ok:
            picked.append(c)
        if len(picked) >= k:
            break
    if len(picked) < k:
        for idx in order:
            if len(picked) >= k:
                break
            picked.append(cand[idx])
    return np.array(picked)

def lp_polish(x, seconds, radius_init=1e-3, cap=5000):
    x = _norm(x)
    n, d = x.shape
    best = max_cos(x)
    radius = radius_init
    start = time.time()
    iu = np.triu_indices(n, 1)
    Gfull = None
    while time.time() - start < seconds and radius > 1e-14:
        c_full = (x @ x.T)[iu]
        m = float(np.max(c_full))
        tol = max(20 * radius, 1e-9)
        sel = c_full > m - tol
        if np.count_nonzero(sel) > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-c_full[idx])[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        c = c_full[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return x

def solve():
    t0 = time.time()
    DEAD = t0 + 176.0
    HOP_END = t0 + 140.0
    rng = np.random.RandomState(12345)
    n, d = 180, 9
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=float)
            if a.shape == (n, d):
                a = _norm(a)
                v = max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.randn(n, d))
        best_val = max_cos(best)
    # fast schedules (cheap: ~300-450 steps total per hop)
    SCH_FAST = [(30.0, 120, 0.008), (120.0, 120, 0.004), (350.0, 100, 0.0015)]
    SCH_MED  = [(25.0, 100, 0.008), (80.0, 120, 0.005), (200.0, 120, 0.0025), (500.0, 80, 0.001)]
    it = 0
    # also try a few isotropic random restarts early to get balanced topology
    # budget: first ~25s for random restarts
    try:
        while time.time() < t0 + 28.0 and time.time() < HOP_END:
            it += 1
            y = _norm(rng.randn(n, d))
            y = adam_relax(y, SCH_FAST)
            v = max_cos(y)
            if v < best_val:
                # quick snap
                rem = min(2.0, max(0.5, HOP_END - time.time() - 1.0))
                if rem > 0.7 and time.time() < HOP_END - 3:
                    y = lp_polish(y, rem, radius_init=0.003)
                    v = max_cos(y)
                if v < best_val:
                    best, best_val = y, v
            if time.time() >= DEAD - 35:
                break
    except Exception:
        pass
    # main basin-hopping with void surgery from best
    sigmas = [0.006, 0.015, 0.035, 0.07, 0.12]
    while time.time() < HOP_END and time.time() < DEAD - 32:
        it += 1
        if time.time() > HOP_END - 1.0:
            break
        sigma = sigmas[it % len(sigmas)]
        y = best.copy()
        mode = it % 5
        try:
            if mode == 0:
                # global small shake
                y = _norm(y + sigma * rng.randn(n, d))
            elif mode == 1 or mode == 2:
                # void reinsertion surgery: target rattlers + overloaded
                cnt, mval = contact_counts(best, 1e-4)
                # rattlers = low counts
                order = np.argsort(cnt)  # ascending
                # overloaded = high counts
                over = np.argsort(-cnt)
                k = int(rng.randint(2, 6))
                # pick: half from lowest, half from highest
                idx = set()
                for j in range((k + 1) // 2):
                    idx.add(int(order[j % n]))
                    # also add random low
                    if rng.rand() < 0.5:
                        idx.add(int(rng.randint(0, n)))
                for j in range(k // 2 + 1):
                    # overloaded with some randomness among top 20
                    pool = over[:25]
                    idx.add(int(pool[rng.randint(0, len(pool))]))
                    if len(idx) >= k:
                        break
                idx = np.array(sorted(list(idx))[:k])
                fixed = np.ones(n, dtype=bool)
                fixed[idx] = False
                voids = largest_voids(best, fixed, len(idx), n_cand=3000, rng=rng)
                y[idx] = voids[:len(idx)] + 0.002 * rng.randn(len(idx), d)
                # tiny jitter elsewhere
                y = _norm(y + 0.002 * rng.randn(n, d))
                y[idx] = _norm(y[idx])
                y = _norm(y)
            elif mode == 3:
                # subset shake biased to high per-point max
                ppm = per_point_max(best)
                order = np.argsort(-ppm)
                k = int(rng.randint(8, 22))
                pool = order[:60]
                idx1 = pool[rng.choice(len(pool), min(k // 2, len(pool)), replace=False)]
                idx2 = rng.choice(n, k - len(idx1), replace=False)
                idx = np.unique(np.concatenate([idx1, idx2]))[:k]
                y[idx] = y[idx] + sigma * 1.6 * rng.randn(len(idx), d)
                y = _norm(y + 0.0015 * rng.randn(n, d))
            else:
                # random orthogonal-ish kick: rotate a random subset coherently
                k = int(rng.randint(10, 30))
                idx = rng.choice(n, k, replace=False)
                Q, _ = np.linalg.qr(rng.randn(d, d))
                # small rotation: mix identity and random orthogonal
                ang = sigma * 3.0
                R = np.eye(d) * np.cos(ang) + Q * np.sin(ang) / max(1.0, np.linalg.norm(Q))
                # re-orthogonalize R roughly
                y[idx] = y[idx] @ R.T
                y = _norm(y + 0.002 * rng.randn(n, d))
            # relax
            sch = SCH_MED if (it % 4 == 0) else SCH_FAST
            y2 = adam_relax(y, sch)
            v2 = max_cos(y2)
            # accept as new best only if better (elitist) to save LP time;
            # but occasionally keep near-best as base? we always hop from best.
            if v2 < best_val:
                # snap it with short LP if time allows
                if time.time() < HOP_END - 4:
                    y3 = lp_polish(y2, 1.5, radius_init=0.003)
                    v3 = max_cos(y3)
                    if v3 < v2:
                        y2, v2 = y3, v3
                if v2 < best_val:
                    best, best_val = y2, v2
            elif v2 < best_val + 0.002 and (it % 9 == 0) and time.time() < HOP_END - 6:
                # occasional lateral move: polish a promising non-record to see if it snaps lower
                y3 = lp_polish(y2, 0.8, radius_init=0.002)
                v3 = max_cos(y3)
                if v3 < best_val:
                    best, best_val = y3, v3
        except Exception:
            continue
    # final polish with remaining time
    try:
        remain = DEAD - time.time() - 1.0
        if remain > 2.0:
            best = lp_polish(best, min(remain, 30.0), radius_init=1e-3)
    except Exception:
        pass
    best = _norm(best)
    return best
