import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def per_point_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)

def contact_counts(x, thresh=1e-9):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = g.max()
    return (g > m - thresh).sum(axis=1), m

def fast_descend(x, steps=700, seed=0):
    # annealed softmax repulsion, momentum tangential GD, very cheap
    n, d = x.shape
    vel = np.zeros_like(x)
    mu = 0.85
    for t in range(steps):
        frac = t / max(1, steps - 1)
        # alpha schedule 20 -> 500, lr schedule
        alpha = 20.0 * (500.0 / 20.0) ** frac
        lr = 0.02 * (1.0 - frac) + 0.002 * frac
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        gm = G.max()
        # avoid overflow: exp(alpha*(G-gm))
        E = np.exp(alpha * (G - gm))
        np.fill_diagonal(E, 0.0)
        S = E.sum()
        if not np.isfinite(S) or S == 0:
            break
        grad = (E @ x) / S  # averaged
        # tangent projection
        nd = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - nd * x
        # normalize gradient scale
        gn = np.linalg.norm(gt) + 1e-12
        gt = gt / (gn / n + 1e-12) * 0.05  # keep O(1) scale
        # clip alpha growth instability
        vel = mu * vel + (1.0 - mu) * gt
        step = lr * vel
        # per-point clip
        rn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        step = step * np.minimum(1.0, 0.03 / rn)
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def void_candidate(X, rng, n_cand=2000):
    d = X.shape[1]
    C = rng.randn(n_cand, d)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    S = C @ X.T  # (cand, n)
    m = S.max(axis=1)
    return C[np.argmin(m)]

def lp_polish(x, seconds, radius_init=1e-3, cap=4000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = radius_init
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c_full = (x @ x.T)[iu]
        m = c_full.max()
        sel = c_full > m - max(20 * radius, 1e-9)
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
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": max(1.0, seconds - (time.time() - start))})
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

def solve():
    np.random.seed(1)
    t0 = time.time()
    deadline = t0 + 176.0
    hop_deadline = t0 + 145.0
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (180, 9):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        r0 = np.random.RandomState(1)
        best = r0.randn(180, 9)
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = max_cos(best)
    n, d = 180, 9
    rng = np.random.RandomState(1)
    # precompute baseline contact info
    it = 0
    lp_calls = 0
    sigmas = [0.006, 0.015, 0.03, 0.06, 0.12]
    while time.time() < hop_deadline and it < 80:
        it += 1
        sigma = sigmas[it % len(sigmas)]
        mode = it % 6
        y = best.copy()
        try:
            cnts, mcur = contact_counts(best)
        except Exception:
            cnts = np.ones(n)
        if mode == 0:
            # global small shake
            y = best + sigma * rng.randn(n, d)
        elif mode == 1:
            # rattler void re-insertion + shake overloaded
            ratt = np.where(cnts == 0)[0]
            if len(ratt) == 0:
                ppm = per_point_max(best)
                ratt = np.argsort(-ppm)[:2]
            # replace rattlers by void candidates computed without them
            keep = np.ones(n, dtype=bool)
            keep[ratt] = False
            Xk = y[keep]
            for r in ratt:
                vc = void_candidate(Xk, rng, n_cand=1500)
                y[r] = vc
                Xk = np.vstack([Xk, vc[None, :]])
            # also jitter overloaded
            over = np.argsort(-cnts)[:8]
            y[over] += sigma * rng.randn(len(over), d)
            y += 0.002 * rng.randn(n, d)
        elif mode == 2:
            # coherent cluster kick: pick worst point + its contacts, rotate together
            ppm = per_point_max(best)
            seed_idx = int(np.argmax(ppm))
            G = best @ best[seed_idx]
            G[seed_idx] = -2
            nbr = np.argsort(-G)[:10]
            grp = np.concatenate([[seed_idx], nbr])
            R = rng.randn(d, d)
            R, _ = np.linalg.qr(R)
            # small rotation: y = (I + eps*A) x approx; use random skew
            A = rng.randn(d, d) * sigma * 2.0
            A = A - A.T
            y[grp] = y[grp] + y[grp] @ A.T
            y += 0.002 * rng.randn(n, d)
        elif mode == 3:
            # subset shake biased to high contacts
            k = int(rng.randint(8, 22))
            order = np.argsort(-cnts)
            pool = order[:60]
            idx1 = pool[rng.choice(len(pool), min(k // 2, len(pool)), replace=False)]
            idx2 = rng.choice(n, k - len(idx1), replace=False)
            idx = np.unique(np.concatenate([idx1, idx2]))
            y[idx] += sigma * 1.5 * rng.randn(len(idx), d)
            y += 0.0015 * rng.randn(n, d)
        elif mode == 4:
            # re-init few worst points randomly
            k = int(rng.randint(2, 5))
            ppm = per_point_max(best)
            order = np.argsort(-ppm)
            idx = order[:k].copy()
            # replace one by void candidate for diversity
            y[idx] = rng.randn(k, d)
            y += 0.003 * rng.randn(n, d)
        else:
            # single void re-insertion (topology change)
            ppm = per_point_max(best)
            worst = int(np.argmax(ppm))
            mask = np.ones(n, dtype=bool)
            mask[worst] = False
            vc = void_candidate(y[mask], rng, n_cand=2500)
            y[worst] = vc
            y += 0.002 * rng.randn(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        # fast anneal
        try:
            steps = 650 if (it % 3 == 0) else 450
            y2 = fast_descend(y, steps=steps)
            v2 = max_cos(y2)
        except Exception:
            continue
        if v2 < best_val:
            # promising: snap with short LP only if clearly better or limited calls
            if lp_calls < 6 and time.time() < hop_deadline - 4.0:
                secs = 1.5
                try:
                    y3 = lp_polish(y2, secs, radius_init=0.004, cap=4000)
                    v3 = max_cos(y3)
                    lp_calls += 1
                    if v3 < best_val:
                        best, best_val = y3, v3
                    else:
                        best, best_val = y2, v2
                except Exception:
                    best, best_val = y2, v2
            else:
                best, best_val = y2, v2
        # occasional keep-near-miss for diversity: no LP, continue hopping from best only
        if time.time() >= hop_deadline:
            break
    remain = deadline - time.time() - 1.0
    if remain > 3.0:
        try:
            best = lp_polish(best, min(remain, 26.0), radius_init=1e-3, cap=5000)
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
