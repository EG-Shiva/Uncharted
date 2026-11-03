import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def per_point_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return np.max(g, axis=1)


def annealed_smooth(v, a0, a1, steps, lr0, beta=0.9):
    n, d = v.shape
    cur = v.copy()
    best = cur.copy()
    best_s = max_cos(cur)
    mom = np.zeros_like(cur)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        alpha = a0 + (a1 - a0) * frac
        G = cur @ cur.T
        np.fill_diagonal(G, -10.0)
        m = np.max(G)
        # shift for stability
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ cur
        proj = np.sum(Grad * cur, axis=1, keepdims=True)
        Gt = Grad - proj * cur
        rs = np.sum(W, axis=1, keepdims=True) + 1e-12
        Gt = Gt / rs
        mom = beta * mom + (1.0 - beta) * Gt
        bc = 1.0 - beta ** (s + 1)
        step = mom / bc
        # cosine decay
        cur_lr = lr0 * (0.5 + 0.5 * np.cos(np.pi * s / max(steps, 1)))
        cur = cur - cur_lr * step
        cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        if (s % 200 == 199) or (s == steps - 1):
            sc = max_cos(cur)
            if sc < best_s:
                best_s = sc
                best = cur.copy()
    return best, best_s


def far_replace(base, idx, rng, n_cand=3000):
    n, d = base.shape
    x0 = base.copy()
    for k in idx:
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        others = np.delete(x0, k, axis=0)
        dots = cand @ others.T
        md = np.max(dots, axis=1)
        x0[k] = cand[np.argmin(md)]
    return x0


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c_full = (x @ x.T)[iu0]
        m = float(np.max(c_full))
        sel = c_full > m - max(20.0 * radius, 1e-9)
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], axis=1
        ).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv),
        ).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(
                obj, A_ub=A_ub, b_ub=-c_full[sel], A_eq=A_eq,
                b_eq=np.zeros(n),
                bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                method="highs",
            )
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
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        # keep x at best region
        if max_cos(x) > best + 1e-12:
            x = best_x.copy()
    return best_x


def solve():
    t0 = time.time()
    BUDGET = 172.0
    EXPLORE = 132.0
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal(size=(99, 13))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (99, 13):
        xb = rng.standard_normal(size=(99, 13))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = (99, 13)

    best_x = xb.copy()
    best_val = max_cos(best_x)
    cur_x = best_x.copy()
    cur_val = best_val

    sigmas = [0.015, 0.03, 0.06, 0.12, 0.22, 0.35]
    it = 0
    # temperature for basin hopping
    T0 = 0.004

    while time.time() - t0 < EXPLORE and it < 1000:
        it += 1
        frac = (time.time() - t0) / max(EXPLORE, 1e-9)
        Temp = T0 * (1.0 - 0.8 * frac) + 1e-4
        # pick base
        rpick = rng.random()
        if rpick < 0.65:
            base = best_x
        else:
            base = cur_x
        mode = rng.random()
        sigma = sigmas[int(rng.integers(0, len(sigmas)))]
        if mode < 0.35:
            noise = rng.standard_normal(size=(n, d)) * sigma
            x0 = base + noise
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        elif mode < 0.60:
            # subset shake
            k = int(rng.integers(4, 16))
            idx = rng.choice(n, size=k, replace=False)
            x0 = base.copy()
            s2 = sigma * 2.0 + 0.05
            x0[idx] += rng.standard_normal(size=(k, d)) * s2
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        elif mode < 0.85:
            # worst-point surgery: replace 1-4 worst points by far voids
            ppm = per_point_max(base)
            order = np.argsort(-ppm)
            nk = int(rng.integers(1, 5))
            # mostly worst, sometimes random among top 15
            if rng.random() < 0.7:
                idx = order[:nk]
            else:
                pool = order[:15]
                idx = rng.choice(pool, size=nk, replace=False)
            x0 = far_replace(base, idx, rng, n_cand=3000)
            # small shake to rest
            noise = rng.standard_normal(size=(n, d)) * (0.01 + 0.3 * sigma)
            # don't shake freshly inserted much
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            x0[mask] += noise[mask]
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        else:
            # large global restart-ish shake
            noise = rng.standard_normal(size=(n, d)) * (0.3 + 0.2 * rng.random())
            x0 = base + noise
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)

        # hyperparams
        a0 = float(rng.choice([20.0, 30.0, 40.0, 50.0]))
        a1 = float(rng.choice([80.0, 120.0, 160.0, 220.0]))
        steps = int(rng.choice([500, 700, 900, 1100]))
        lr0 = float(rng.choice([0.012, 0.02, 0.03, 0.045]))
        # cap steps near end of explore
        if time.time() - t0 > EXPLORE - 10:
            steps = min(steps, 500)

        refined, rval = annealed_smooth(x0, a0, a1, steps, lr0)

        if rval < best_val:
            best_val = rval
            best_x = refined.copy()
            cur_x = refined.copy()
            cur_val = rval
        else:
            # Metropolis move for current walker
            if rval < cur_val or rng.random() < np.exp((cur_val - rval) / max(Temp, 1e-6)):
                cur_x = refined.copy()
                cur_val = rval

        # occasional LP snap to evaluate true basin depth (cheap, rare)
        if it % 25 == 0 and (time.time() - t0) < EXPLORE - 5:
            snapped = lp_polish(cur_x, 1.5)
            sval = max_cos(snapped)
            if sval < best_val:
                best_val = sval
                best_x = snapped.copy()
            if sval < cur_val + 0.002:
                cur_x = snapped.copy()
                cur_val = sval

    remain = BUDGET - (time.time() - t0)
    remain = max(remain - 2.0, 5.0)
    # final polish of best
    best_x = lp_polish(best_x, min(remain, 35.0))
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    # safety: remove NaN
    best_x = np.nan_to_num(best_x, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(best_x, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    best_x = best_x / nr
    return best_x
