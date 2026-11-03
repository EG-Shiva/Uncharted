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

def contact_counts(x, tol=1e-7):
    n = x.shape[0]
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = g.max()
    cnt = np.zeros(n, dtype=int)
    # count pairs within tol of max
    iu = np.triu_indices(n, 1)
    c = g[iu]
    sel = c > m - tol
    I = iu[0][sel]; J = iu[1][sel]
    for i in I: cnt[i] += 1
    for j in J: cnt[j] += 1
    return cnt, m

def lp_polish(x, seconds, radius_init=1e-3, cap=4000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = radius_init
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_full = G[iu]
        m = c_full.max()
        thresh = m - max(20 * radius, 1e-9)
        sel = c_full > thresh
        nnz = int(np.count_nonzero(sel))
        if nnz == 0:
            break
        if nnz > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-c_full[idx])[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        c = c_full[sel]
        p = len(I)
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def adam_stage(x, alpha, steps, lr, t_end=None):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        if t_end is not None and t % 50 == 0 and time.time() > t_end:
            break
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        gmax = G.max()
        E = np.exp(alpha * (G - gmax))
        np.fill_diagonal(E, 0.0)
        S = E.sum()
        if not np.isfinite(S) or S == 0:
            break
        grad = (E @ x) * (n / S)
        nd = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - nd * x
        m = b1 * m + (1.0 - b1) * gt
        vv = b2 * v + (1.0 - b2) * gt * gt
        v = vv
        mhat = m / (1.0 - b1 ** t)
        vhat = vv / (1.0 - b2 ** t)
        delta = lr * mhat / (np.sqrt(vhat) + eps)
        dn = (delta * x).sum(axis=1, keepdims=True)
        delta = delta - dn * x
        rn = np.linalg.norm(delta, axis=1, keepdims=True) + 1e-12
        scale = np.minimum(1.0, 0.05 / rn)
        delta = delta * scale
        x = x - delta
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def smooth_refine(x, t_end=None):
    x = adam_stage(x, 30.0, 200, 0.006, t_end)
    x = adam_stage(x, 90.0, 250, 0.004, t_end)
    x = adam_stage(x, 220.0, 300, 0.002, t_end)
    x = adam_stage(x, 500.0, 180, 0.0009, t_end)
    return x

def largest_voids(x, k, rng, n_samples=6000):
    n, d = x.shape
    cand = rng.randn(n_samples, d)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    # max dot to x in chunks
    best_dots = np.empty(n_samples)
    BS = 1500
    for s in range(0, n_samples, BS):
        e = s + min(BS, n_samples - s)
        D = cand[s:e] @ x.T
        best_dots[s:e] = D.max(axis=1)
    order = np.argsort(best_dots)  # smallest max-dot = largest hole
    # greedily diversify: pick k well separated
    picked = []
    for idx in order:
        c = cand[idx]
        ok = True
        for p in picked:
            if abs(float(c @ p)) > 0.9:
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

def solve():
    t0 = time.time()
    deadline = t0 + 176.0
    hop_deadline = t0 + 148.0
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
        r0 = np.random.RandomState(0)
        best = r0.randn(180, 9)
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = max_cos(best)
    n, d = 180, 9
    rng = np.random.RandomState(12345)
    sigmas = [0.006, 0.015, 0.03, 0.06, 0.12]
    it = 0
    lp_calls = 0
    # cache contact info for best
    ppm = per_point_max(best)
    try:
        cnt_best, _ = contact_counts(best, tol=1e-7)
    except Exception:
        cnt_best = np.ones(n, dtype=int)
    while time.time() < hop_deadline and it < 250:
        it += 1
        if time.time() > hop_deadline - 1.0:
            break
        sigma = sigmas[it % len(sigmas)]
        mode = it % 8
        y = best.copy()
        if mode == 0:
            y = best + sigma * rng.randn(n, d)
        elif mode == 1:
            # kick overloaded points (high contacts)
            k = int(rng.randint(6, 20))
            order = np.argsort(-cnt_best)
            pool = order[:50]
            idx1 = pool[rng.choice(len(pool), min(k // 2 + 2, len(pool)), replace=False)]
            idx2 = rng.choice(n, k - len(idx1), replace=False)
            idx = np.unique(np.concatenate([idx1, idx2]))[:k]
            y[idx] = y[idx] + sigma * 1.6 * rng.randn(len(idx), d)
            y += 0.0015 * rng.randn(n, d)
        elif mode == 2:
            # void reinsertion: replace rattlers + worst with voids
            try:
                ratt = np.where(cnt_best == 0)[0]
            except Exception:
                ratt = np.array([], dtype=int)
            k = int(rng.randint(2, 6))
            # candidates to replace: rattlers plus worst ppm
            order = np.argsort(-ppm)
            repl = list(ratt[:k])
            for o in order:
                if len(repl) >= k:
                    break
                if o not in repl:
                    repl.append(int(o))
            repl = np.array(repl[:k], dtype=int)
            try:
                voids = largest_voids(best, k, rng, n_samples=6000)
                y[repl] = voids + 0.003 * rng.randn(k, d)
                # small shake elsewhere
                y += 0.001 * rng.randn(n, d)
            except Exception:
                y[repl] = rng.randn(k, d)
        elif mode == 3:
            # break tightest pairs: pick random tight pair, move one endpoint
            try:
                G = best @ best.T
                np.fill_diagonal(G, -2.0)
                m = G.max()
                iu = np.triu_indices(n, 1)
                c = G[iu]
                tight = np.where(c > m - 1e-7)[0]
                if len(tight) > 0:
                    pick = tight[rng.choice(len(tight), min(4, len(tight)), replace=False)]
                    I = iu[0][pick]; J = iu[1][pick]
                    victims = np.unique(np.concatenate([I, J]))
                    # move half of them
                    vv = victims[rng.choice(len(victims), max(1, len(victims)//2), replace=False)]
                    y[vv] = y[vv] + (sigma + 0.02) * rng.randn(len(vv), d)
                else:
                    y += sigma * rng.randn(n, d)
            except Exception:
                y += sigma * rng.randn(n, d)
        elif mode == 4:
            # heavy re-init of few points
            k = int(rng.randint(3, 6))
            order = np.argsort(-ppm)
            idx = list(order[:k // 2 + 1])
            while len(idx) < k:
                c0 = int(rng.randint(0, n))
                if c0 not in idx:
                    idx.append(c0)
            idx = np.array(idx[:k])
            y[idx] = rng.randn(k, d)
            y += 0.004 * rng.randn(n, d)
        elif mode == 5:
            # subset shake biased to jammed (high ppm)
            k = int(rng.randint(8, 26))
            order = np.argsort(-ppm)
            pool = order[:60]
            idx1 = pool[rng.choice(len(pool), min(k // 2, len(pool)), replace=False)]
            idx2 = rng.choice(n, k - len(idx1), replace=False)
            idx = np.unique(np.concatenate([idx1, idx2]))[:k]
            y[idx] = y[idx] + sigma * 1.5 * rng.randn(len(idx), d)
            y += 0.002 * rng.randn(n, d)
        elif mode == 6:
            # mirror / rotate basin: random orthogonal-ish tweak + global shake
            y = best + sigma * 0.8 * rng.randn(n, d)
            # flip sign of random coordinate for subset to jump basin
            q = rng.choice(n, int(rng.randint(4, 12)), replace=False)
            jc = int(rng.randint(0, d))
            y[q, jc] *= -1.0
            y[q] += 0.01 * rng.randn(len(q), d)
        else:
            # directed rattler kick + global small noise
            try:
                ratt = np.where(cnt_best == 0)[0]
                if len(ratt) > 0:
                    y[ratt] = rng.randn(len(ratt), d) * 1.0
                y += sigma * 0.5 * rng.randn(n, d)
            except Exception:
                y += sigma * rng.randn(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        try:
            y2 = smooth_refine(y, t_end=hop_deadline)
        except Exception:
            continue
        try:
            v2 = max_cos(y2)
        except Exception:
            continue
        improved = v2 < best_val
        # LP polish decision
        want_lp = False
        if improved:
            want_lp = True
        elif v2 < best_val + 0.0006 and (it % 4 == 0) and lp_calls < 40:
            want_lp = True
        if want_lp:
            remain = hop_deadline - time.time()
            if remain < 2.5:
                if improved:
                    best = y2; best_val = v2
                    ppm = per_point_max(best)
                    try:
                        cnt_best, _ = contact_counts(best, tol=1e-7)
                    except Exception:
                        pass
                continue
            secs = 2.0 if improved else 1.0
            secs = min(secs, max(0.7, remain - 0.8))
            # tighter cap for non-improved to save time
            cap = 4000 if improved else 2500
            try:
                y3 = lp_polish(y2, secs, radius_init=0.004, cap=cap)
                v3 = max_cos(y3)
                lp_calls += 1
                if v3 < best_val:
                    best = y3; best_val = v3
                    ppm = per_point_max(best)
                    try:
                        cnt_best, _ = contact_counts(best, tol=1e-7)
                    except Exception:
                        pass
                elif improved:
                    best = y2; best_val = v2
                    ppm = per_point_max(best)
                    try:
                        cnt_best, _ = contact_counts(best, tol=1e-7)
                    except Exception:
                        pass
            except Exception:
                if improved:
                    best = y2; best_val = v2
                    ppm = per_point_max(best)
                    try:
                        cnt_best, _ = contact_counts(best, tol=1e-7)
                    except Exception:
                        pass
        else:
            if improved:
                best = y2; best_val = v2
                ppm = per_point_max(best)
                try:
                    cnt_best, _ = contact_counts(best, tol=1e-7)
                except Exception:
                    pass
    remain = deadline - time.time() - 0.8
    if remain > 3.0:
        try:
            best = lp_polish(best, min(remain, 26.0), radius_init=1e-3, cap=5000)
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
