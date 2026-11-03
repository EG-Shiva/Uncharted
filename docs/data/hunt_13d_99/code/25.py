import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def per_point_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)


def lp_polish(X, seconds):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = max_cos(X)
    best_x = X.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        G = X @ X.T
        c = G[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar,
                               J[:, None] * d + ar,
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(),
                               (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            X = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x


def lse_run(X0, steps, a0, a1, lr0, lr1, beta):
    n, d = X0.shape
    cur = X0.copy()
    best = cur.copy()
    best_s = max_cos(cur)
    V = np.zeros_like(cur)
    la0, la1 = np.log(a0), np.log(a1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        alpha = float(np.exp(la0 + (la1 - la0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = cur @ cur.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        # W = exp(alpha*(G-m)), G-m <= 0
        W = np.exp((G - m) * alpha)
        np.fill_diagonal(W, 0.0)
        S = float(np.sum(W)) + 1e-30
        Grad = (W @ cur) * (n / S)
        dot = np.sum(Grad * cur, axis=1, keepdims=True)
        Gt = Grad - dot * cur
        # clip extreme grads
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-30
        cap = 2.0
        scale = np.minimum(1.0, cap / gn)
        Gt = Gt * scale
        V = beta * V + (1.0 - beta) * Gt
        bc = 1.0 - beta ** (s + 1)
        step = V / bc
        cur = cur - lr * step
        cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        if (s % 200 == 199) or (s == steps - 1):
            sc = max_cos(cur)
            if sc < best_s:
                best_s = sc
                best = cur.copy()
    return best, best_s


def far_reinsert(base, idx, rng, n_cand=3000):
    x0 = base.copy()
    n, d = base.shape
    for k in idx:
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        others = np.delete(x0, k, axis=0)
        dots = cand @ others.T
        md = dots.max(axis=1)
        x0[k] = cand[np.argmin(md)]
    return x0


def solve():
    t0 = time.time()
    EXPLORE = 132.0
    rng = np.random.default_rng(12345)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal(size=(99, 13))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (99, 13):
        tmp = rng.standard_normal(size=(99, 13))
        try:
            if xb.ndim == 2:
                m0 = min(xb.shape[0], 99)
                m1 = min(xb.shape[1], 13)
                tmp[:m0, :m1] = xb[:m0, :m1]
        except Exception:
            pass
        xb = tmp
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    # also consider record.npy, keep better
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        if xr.shape == (99, 13) and max_cos(xr) < max_cos(xb):
            xb = xr
    except Exception:
        pass
    n, d = 99, 13
    best_x = xb.copy()
    best_val = max_cos(best_x)
    cur_x = best_x.copy()
    cur_val = best_val

    sigmas_small = [0.015, 0.03, 0.05, 0.08]
    sigmas_big = [0.18, 0.28, 0.40]
    it = 0
    while time.time() - t0 < EXPLORE and it < 500:
        it += 1
        if time.time() - t0 > EXPLORE - 2.0:
            break
        # choose base: mostly best, sometimes wanderer
        rbase = rng.random()
        if rbase < 0.62:
            base = best_x
            base_is_best = True
        else:
            base = cur_x
            base_is_best = False
        mode = rng.random()
        X0 = base.copy()
        if mode < 0.28:
            sigma = float(sigmas_small[int(rng.integers(0, len(sigmas_small)))])
            # occasional subset-only small shake
            if rng.random() < 0.3:
                m = int(rng.integers(5, 26))
                sel = rng.choice(n, m, replace=False)
                X0[sel] += sigma * 1.5 * rng.standard_normal(size=(m, d))
            else:
                X0 += sigma * rng.standard_normal(size=(n, d))
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        elif mode < 0.48:
            sigma = float(sigmas_big[int(rng.integers(0, len(sigmas_big)))])
            X0 += sigma * rng.standard_normal(size=(n, d))
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        elif mode < 0.78:
            # hub-targeted killing: worst by per-point max
            rpm = per_point_max(base)
            k = int(rng.integers(1, 4))
            if rng.random() < 0.5:
                idx = np.argsort(rpm)[-k:]
            else:
                # probabilistic among worst 15
                cand_idx = np.argsort(rpm)[-15:]
                probs = rpm[cand_idx] - rpm[cand_idx].min() + 1e-6
                probs = probs / probs.sum()
                idx = rng.choice(cand_idx, size=k, replace=False, p=probs)
            nc = int(rng.integers(2000, 5001))
            X0 = far_reinsert(base, idx, rng, n_cand=nc)
            X0 += 0.02 * rng.standard_normal(size=(n, d))
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        else:
            # block shake on random subset
            m = int(rng.integers(8, 30))
            sel = rng.choice(n, m, replace=False)
            sigma = float(rng.uniform(0.10, 0.25))
            X0[sel] += sigma * rng.standard_normal(size=(m, d))
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)

        steps = int(rng.integers(600, 1401))
        a0 = float(rng.uniform(12.0, 40.0))
        a1 = float(rng.uniform(120.0, 350.0))
        if a1 < a0:
            a0, a1 = a1, a0
        lr0 = float(rng.uniform(0.05, 0.20))
        lr1 = float(rng.uniform(0.002, 0.009))
        beta = float(rng.uniform(0.80, 0.90))
        X1, v1 = lse_run(X0, steps, a0, a1, lr0, lr1, beta)

        if v1 < best_val:
            # polish immediately
            budget = 2.0 if (time.time() - t0 < EXPLORE - 10) else 1.0
            try:
                Xp = lp_polish(X1, budget)
                vp = max_cos(Xp)
            except Exception:
                Xp, vp = X1, v1
            if vp < best_val:
                best_x, best_val = Xp, vp
                cur_x, cur_val = Xp.copy(), vp
            elif v1 < best_val:
                best_x, best_val = X1, v1
                cur_x, cur_val = X1.copy(), v1
        else:
            # wandering acceptance (allow sideways)
            accept = False
            if v1 < cur_val:
                accept = True
            else:
                Tacc = 0.0025
                if rng.random() < np.exp(-(v1 - cur_val) / Tacc):
                    accept = True
            if accept:
                cur_x, cur_val = X1.copy(), v1
            # near-miss polish from best basin occasionally
            if base_is_best and v1 < best_val + 0.003 and (it % 3 == 0):
                try:
                    Xp = lp_polish(X1, 1.0)
                    vp = max_cos(Xp)
                except Exception:
                    vp = v1 + 1.0
                    Xp = X1
                if vp < best_val:
                    best_x, best_val = Xp, vp
                    cur_x, cur_val = Xp.copy(), vp
                elif vp < cur_val:
                    cur_x, cur_val = Xp.copy(), vp
            # restart wanderer if too bad
            if cur_val > best_val + 0.015:
                cur_x = best_x.copy()
                cur_val = best_val

    remain = 172.0 - (time.time() - t0)
    if remain < 5:
        remain = 5
    if remain > 35:
        remain = 35
    try:
        best_x = lp_polish(best_x, remain)
    except Exception:
        pass
    best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
