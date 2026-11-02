import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds, r0=1e-3, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = float(r0)
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if t_deadline is not None and time.time() > t_deadline:
            break
        if radius < 1e-14:
            break
        c_all = (x @ x.T)[iu]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        # limit size for speed
        if np.count_nonzero(sel) > 1200:
            # keep tightest 1200
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:1200]
            sel2 = np.zeros_like(sel, dtype=bool)
            sel2[idx[order]] = True
            sel = sel2
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 1.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        ar = np.arange(d)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        b_ub = -c_all[sel]
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def smooth_opt(x, steps, T0, T1, lr, mu):
    n, d = x.shape
    v = np.zeros_like(x)
    l0, l1 = np.log(T0), np.log(T1)
    for k in range(steps):
        f = k / max(steps - 1, 1)
        T = float(np.exp(l0 * (1 - f) + l1 * f))
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(G.max())
        # (G-m)/T <= 0
        E = np.exp((G - m) / T)
        # diag is 0 already (exp(-inf)=0)
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            break
        grad = 2.0 * ((E / S) @ x)
        proj = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - proj * x
        # clip to avoid huge steps at tiny T
        gn = np.linalg.norm(gt, axis=1, keepdims=True)
        cap = 0.2
        # scale down if too large
        scale = np.ones_like(gn)
        mask = gn > cap
        scale[mask] = cap / (gn[mask] + 1e-12)
        gt = gt * scale
        v = mu * v + lr * gt
        x = x - v
        xn = np.linalg.norm(x, axis=1, keepdims=True)
        xn[xn == 0] = 1.0
        x = x / xn
    return x


def contact_counts(x, gap=1e-7):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cnt = np.sum(G > m - gap, axis=1)
    return m, cnt


def hole_candidates(X_keep, n_pick, rng, n_samples=20000):
    d = X_keep.shape[1]
    C = rng.standard_normal(size=(n_samples, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    cur = X_keep.copy()
    picks = []
    for _ in range(n_pick):
        dots = C @ cur.T
        m = dots.max(axis=1)
        # pick among best 20 randomly for diversity
        order = np.argpartition(m, 20)[:20]
        # choose the very best most of time, random among top sometimes
        # deterministic via rng
        j = int(rng.choice(order))
        # refine: take best of order
        # bias to best: if rng random <0.7 take argmin
        if rng.random() < 0.7:
            j = int(order[np.argmin(m[order])])
        picks.append(C[j].copy())
        cur = np.vstack([cur, C[j]])
    return np.array(picks)


def solve():
    import time as _time
    t0 = _time.time()
    DEADLINE = t0 + 175.0
    rng = np.random.default_rng(12345)

    def load_norm(path):
        try:
            a = np.load(path).astype(float)
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            return a
        except Exception:
            return None

    xb = load_norm("best.npy")
    xr = load_norm("record.npy")
    if xb is None and xr is None:
        # fallback random
        a = rng.standard_normal(size=(76, 15))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        xb = a
    elif xb is None:
        xb = xr
    if xr is not None:
        if max_cos(xr) < max_cos(xb):
            xb = xr.copy()

    x_best = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = max_cos(x_best)
    n, d = x_best.shape

    it = 0
    sigmas = [0.03, 0.07, 0.015, 0.10, 0.05]
    # hopping loop, reserve ~14s for final polish
    while _time.time() < DEADLINE - 14.0 and it < 300:
        it += 1
        mode = it % 5
        sigma = sigmas[it % len(sigmas)]
        # compute contacts of best
        _, cnt = contact_counts(x_best, gap=5e-7)
        ratt = np.where(cnt == 0)[0]
        order_hi = np.argsort(-cnt)

        cand = x_best.copy()
        if mode == 0:
            # global shake
            cand = cand + sigma * rng.standard_normal(size=cand.shape)
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        elif mode == 1:
            # subset shake: rattlers + few random + few overloaded
            k = int(rng.integers(4, 10))
            idx = set()
            if len(ratt) > 0:
                rr = rng.choice(ratt, size=min(len(ratt), 3), replace=False).tolist()
                idx.update(rr)
            idx.update(rng.choice(n, size=min(k, n), replace=False).tolist())
            idx.update(order_hi[:2].tolist())
            idx = np.array(sorted(idx), dtype=int)
            amp = sigma * 2.0
            cand[idx] = cand[idx] + amp * rng.standard_normal(size=(len(idx), d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        elif mode == 2:
            # hole reinsertion for rattlers / random points
            if len(ratt) >= 2:
                nr = min(int(rng.integers(2, 6)), len(ratt))
                idx_rep = rng.choice(ratt, size=nr, replace=False)
            else:
                nr = int(rng.integers(2, 5))
                idx_rep = rng.choice(n, size=nr, replace=False)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[idx_rep] = False
            X_keep = cand[keep_mask]
            picks = hole_candidates(X_keep, len(idx_rep), rng, n_samples=16000)
            cand[idx_rep] = picks
            # small jitter to all
            cand = cand + 0.005 * rng.standard_normal(size=cand.shape)
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        elif mode == 3:
            # shake overloaded cluster strongly + tiny global
            nh = min(8, n)
            idx_hi = order_hi[:nh]
            # random subset of them
            k = int(rng.integers(3, 7))
            idx_rep = rng.choice(idx_hi, size=min(k, len(idx_hi)), replace=False)
            cand[idx_rep] = cand[idx_rep] + sigma * 1.5 * rng.standard_normal(size=(len(idx_rep), d))
            cand = cand + 0.008 * rng.standard_normal(size=cand.shape)
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        else:
            # random orthogonal-ish kick to a random subset
            k = int(rng.integers(6, 14))
            idx = rng.choice(n, size=k, replace=False)
            cand[idx] = cand[idx] + sigma * rng.standard_normal(size=(k, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)

        # annealed soft-max descent (cheap, changes basin)
        steps = 500
        T0, T1 = 0.03, 0.004
        # vary slightly by iteration
        if it % 2 == 0:
            steps = 700
        cand = smooth_opt(cand, steps=steps, T0=T0, T1=T1, lr=0.5, mu=0.85)

        # short LP re-polish
        remain_hop = DEADLINE - 14.0 - _time.time()
        if remain_hop < 0.5:
            break
        secs = min(2.2, remain_hop)
        r0 = min(0.02, max(0.004, sigma * 0.5))
        cand = lp_polish(cand, seconds=secs, r0=r0, t_deadline=DEADLINE - 14.0)
        v = max_cos(cand)
        if v < best_val - 1e-12:
            x_best = cand
            best_val = v
        # else discard; continue (no annealing acceptance to save time)
        # occasional forced diversity: if stuck long, increase sigma cycle automatically via mode

    # final long polish on best
    rem = DEADLINE - _time.time()
    if rem > 2.0:
        x_best = lp_polish(x_best, seconds=min(rem - 1.0, 12.0), r0=2e-3, t_deadline=DEADLINE)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
