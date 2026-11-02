import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def contact_counts(x, tol=2e-3):
    n = x.shape[0]
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = g.max()
    cnt = np.sum(g > m - tol, axis=1)
    return cnt, float(m)


def lp_polish(x, seconds, radius_init=2e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = radius_init
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        Gfull = x @ x.T
        c = Gfull[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        # cap size to keep linprog fast
        if np.sum(sel) > 2500:
            # keep tightest 2500
            idx = np.argpartition(-c, 2500)[:2500]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = sel & mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        ci = (I[:, None] * d + np.arange(d)).ravel()
        cj = (J[:, None] * d + np.arange(d)).ravel()
        cols_I = ci
        cols_J = cj
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
            # early stop if diverging? continue
        else:
            radius *= 0.5
    return x


def soft_quench(x, steps=500, beta0=10.0, beta1=110.0, lr0=0.03, lr1=0.003, mu=0.75):
    n, d = x.shape
    vel = np.zeros_like(x)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        # stable weights in (0,1]
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        # sparsify tiny weights for speed/sharpness late
        # W[W < 1e-9] = 0.0
        scale = W.sum() / n
        if scale < 1e-12:
            scale = 1.0
        grad = W @ x
        # tangential projection
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad -= dot * x
        grad /= scale
        # clip per-row norm to 1 for stability
        rn = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-12
        clip = np.minimum(1.0, 1.0 / rn)
        grad = grad * clip
        vel = mu * vel - lr * grad
        # project velocity tangentially (approx)
        vdot = np.sum(vel * x, axis=1, keepdims=True)
        vel -= vdot * x
        x = x + vel
        xn = np.linalg.norm(x, axis=1, keepdims=True)
        xn[xn == 0] = 1.0
        x = x / xn
    return x


def greedy_candidates(x_fixed, n_cand, rng):
    # sample random sphere points, return array (n_cand, d)
    d = x_fixed.shape[1]
    z = rng.standard_normal((n_cand, d))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    return z


def solve():
    import time as _time
    t0 = _time.time()
    LIMIT = 168.0
    rng = np.random.default_rng(0)
    np.random.seed(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = np.load("record.npy").astype(float)
    n, d = 82, 14
    if xb.shape != (n, d):
        # fallback: tile/truncate or random
        z = rng.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        # copy what we can
        m = min(xb.shape[0], n)
        z[:m, :min(xb.shape[1], d)] = xb[:m, :min(xb.shape[1], d)]
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        xb = z
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = max_cos(xb)
    # short initial polish to get true local value (cheap)
    if _time.time() - t0 < LIMIT - 20:
        xb = lp_polish(xb, seconds=2.0)
        best_val = max_cos(xb)
    X_best = xb.copy()
    it = 0
    # precompute schedule
    while _time.time() - t0 < LIMIT - 6:
        it += 1
        time_left = LIMIT - (_time.time() - t0)
        # allocate ~2.5s polish per iter, less if little time left
        polish_sec = 2.2 if time_left > 30 else 1.2
        cnt, cur_m = contact_counts(X_best, tol=2e-3)
        order_low = np.argsort(cnt, kind="stable")
        order_high = order_low[::-1]
        # choose mode
        mode = it % 6
        X = X_best.copy()
        # number to replace and noise level (deterministic via rng)
        log_sigma = rng.uniform(-2.6, -0.6)  # sigma ~0.0025..0.25
        sigma = 10.0 ** log_sigma
        if mode == 0:
            n_rep = int(rng.integers(5, 9))
            victims = order_low[:n_rep]
            use_greedy = True
        elif mode == 1:
            n_rep = int(rng.integers(2, 5))
            # mix low + random
            victims = np.concatenate([order_low[:n_rep // 2 + 1], rng.choice(n, size=n_rep, replace=False)])
            victims = np.unique(victims)[:n_rep + 1]
            use_greedy = True
        elif mode == 2:
            n_rep = int(rng.integers(1, 4))
            victims = rng.choice(n, size=n_rep, replace=False)
            use_greedy = rng.random() < 0.7
        elif mode == 3:
            # attack overloaded points
            n_rep = int(rng.integers(2, 5))
            victims = order_high[:n_rep]
            use_greedy = False
        elif mode == 4:
            n_rep = 0
            victims = np.array([], dtype=int)
            use_greedy = False
            sigma = float(10.0 ** rng.uniform(-2.0, -0.9))
        else:
            n_rep = int(rng.integers(3, 8))
            victims = rng.choice(n, size=n_rep, replace=False)
            use_greedy = True
        # apply replacement
        if len(victims) > 0:
            # for stability, replace sequentially with greedy best-of-random
            for v in victims:
                if use_greedy:
                    cand = greedy_candidates(X, 2500, rng)
                    # score = max dot to all other points
                    others = np.delete(X, v, axis=0)
                    dots = cand @ others.T
                    s = dots.max(axis=1)
                    # also penalize closeness among victims already replaced? already in others
                    best_c = np.argmin(s)
                    X[v] = cand[best_c]
                else:
                    z = rng.standard_normal(d)
                    z /= np.linalg.norm(z)
                    X[v] = z
            # small jitter on victims
            if sigma > 1e-12:
                noise = rng.standard_normal((len(victims), d)) * (sigma * 0.7)
                X[victims] += noise
                X[victims] /= np.linalg.norm(X[victims], axis=1, keepdims=True)
        # global jitter
        if mode in (4, 5, 2):
            X += rng.standard_normal((n, d)) * (sigma * 0.5 if len(victims) > 0 else sigma)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        else:
            # light global jitter
            X += rng.standard_normal((n, d)) * (sigma * 0.25)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        # soft quench: steps adapt to time
        steps = 450 if time_left > 40 else 250
        # vary beta range slightly per iter
        b0 = float(8 + rng.random() * 6)
        b1 = float(80 + rng.random() * 60)
        lr0 = 0.03
        lr1 = 0.003
        X = soft_quench(X, steps=steps, beta0=b0, beta1=b1, lr0=lr0, lr1=lr1, mu=0.75)
        # quick check before expensive LP: if far worse skip LP?
        qv = max_cos(X)
        # if much worse than best (+0.03) skip polish to save time, add bigger kick next time
        if qv > best_val + 0.04 and (_time.time() - t0) < LIMIT - 20:
            continue
        Xp = lp_polish(X, seconds=polish_sec)
        pv = max_cos(Xp)
        if pv < best_val - 1e-12:
            X_best = Xp
            best_val = pv
        # sideways acceptance for diversity rarely: if within 1e-4 and random, move base?
        # we keep greedy (only improvements) to stay in best basin-hopping; no sideways.
        if _time.time() - t0 >= LIMIT - 6:
            break
    # final polish with remaining time
    rem = LIMIT - (_time.time() - t0) - 1.0
    if rem > 1.5:
        X_best = lp_polish(X_best, seconds=min(rem, 8.0), radius_init=1e-3)
        best_val = max_cos(X_best)
    X_best = X_best / np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best
