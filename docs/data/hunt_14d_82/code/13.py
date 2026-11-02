import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def lp_polish(x, seconds, deadline=None, radius_init=1.5e-3, cap=1800):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = radius_init
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while True:
        now = time.time()
        if now - start >= seconds:
            break
        if deadline is not None and now >= deadline:
            break
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c = Gfull[iu0]
        m = float(np.max(c))
        thr = max(20.0 * radius, 1e-9)
        sel = c > m - thr
        ns = int(np.sum(sel))
        if ns == 0:
            break
        if ns > cap:
            # keep tightest cap
            idx = np.argpartition(-c, cap)[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = sel & mask
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel],
                          A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
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
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x


def quench(x, steps=350, beta0=10.0, beta1=110.0, lr0=0.03, lr1=0.003, mu=0.75):
    n, d = x.shape
    vel = np.zeros_like(x)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        scale = float(np.sum(W)) / n
        if scale < 1e-12 or not np.isfinite(scale):
            scale = 1.0
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad -= dot * x
        grad /= scale
        rn = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-12
        clip = np.minimum(1.0, 1.0 / rn)
        grad = grad * clip
        vel = mu * vel - lr * grad
        vdot = np.sum(vel * x, axis=1, keepdims=True)
        vel -= vdot * x
        x = x + vel
        xn = np.linalg.norm(x, axis=1, keepdims=True)
        xn[xn == 0] = 1.0
        x = x / xn
    return x


def hole_replace(X, victims, rng, n_cand=2000):
    n, d = X.shape
    for v in victims:
        cand = rng.standard_normal((n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        others = np.delete(X, int(v), axis=0)
        dots = cand @ others.T
        s = np.max(dots, axis=1)
        X[int(v)] = cand[int(np.argmin(s))]
    return X


def solve():
    t0 = time.time()
    LIMIT = 170.0
    deadline_final = t0 + LIMIT
    rng = np.random.default_rng(7)
    n, d = 82, 14
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal((n, d))
    if xb.shape != (n, d):
        z = rng.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        xb = z
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    # initial short polish to get true local value
    xb = lp_polish(xb, seconds=3.0, deadline=deadline_final - 5.0)
    best_val = max_cos(xb)
    X_best = xb.copy()

    it = 0
    # basin hopping until ~140s
    while time.time() < t0 + 138.0 and time.time() < deadline_final - 25.0:
        it += 1
        time_left = deadline_final - time.time()
        # contact analysis
        G = X_best @ X_best.T
        np.fill_diagonal(G, -2.0)
        m0 = float(np.max(G))
        cnt = np.sum(G > m0 - 2e-3, axis=1)
        order_low = np.argsort(cnt, kind="stable")
        order_high = order_low[::-1]

        mode = it % 7
        X = X_best.copy()
        log_sigma = rng.uniform(-2.6, -0.6)
        sigma = 10.0 ** log_sigma
        victims = np.array([], dtype=int)
        use_greedy = True

        if mode == 0:
            # re-insert rattlers / loosest into holes
            loose = order_low[:max(3, int(np.sum(cnt <= 1)))]
            # take up to 7 loosest
            n_rep = min(len(loose), 7)
            victims = loose[:n_rep].copy()
            use_greedy = True
            sigma *= 0.5
        elif mode == 1:
            n_rep = int(rng.integers(4, 8))
            victims = order_low[:n_rep].copy()
            use_greedy = True
        elif mode == 2:
            n_rep = int(rng.integers(2, 5))
            victims = order_high[:n_rep].copy()
            use_greedy = True  # move hubs to holes
        elif mode == 3:
            n_rep = int(rng.integers(3, 7))
            victims = rng.choice(n, size=n_rep, replace=False)
            use_greedy = True
        elif mode == 4:
            n_rep = int(rng.integers(1, 4))
            victims = rng.choice(n, size=n_rep, replace=False)
            use_greedy = False
            sigma = float(10.0 ** rng.uniform(-1.8, -0.7))
        elif mode == 5:
            victims = np.array([], dtype=int)
            use_greedy = False
            sigma = float(10.0 ** rng.uniform(-2.2, -0.9))
        else:
            n_rep = int(rng.integers(2, 6))
            victims = np.concatenate([order_low[:2], rng.choice(n, size=n_rep, replace=False)])
            victims = np.unique(victims)[:n_rep + 2]
            use_greedy = rng.random() < 0.8

        if len(victims) > 0:
            if use_greedy:
                X = hole_replace(X, victims, rng, n_cand=2000)
                # small jitter on victims
                nz = rng.standard_normal((len(victims), d)) * (sigma * 0.6)
                X[victims] += nz
                X[victims] /= np.linalg.norm(X[victims], axis=1, keepdims=True)
            else:
                for v in victims:
                    z = rng.standard_normal(d)
                    z /= np.linalg.norm(z)
                    X[int(v)] = z
                nz = rng.standard_normal((len(victims), d)) * (sigma * 0.5)
                X[victims] += nz
                X[victims] /= np.linalg.norm(X[victims], axis=1, keepdims=True)
            # light global jitter
            X += rng.standard_normal((n, d)) * (sigma * 0.25)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        else:
            X += rng.standard_normal((n, d)) * sigma
            X /= np.linalg.norm(X, axis=1, keepdims=True)

        steps = 400 if time_left > 50 else 220
        b0 = float(7.0 + rng.random() * 7.0)
        b1 = float(80.0 + rng.random() * 70.0)
        mu = 0.72 + rng.random() * 0.08
        X = quench(X, steps=steps, beta0=b0, beta1=b1, lr0=0.03, lr1=0.003, mu=mu)
        qv = max_cos(X)
        if qv > best_val + 0.035 and time.time() < t0 + 120.0:
            continue
        # staged LP: short first
        Xp = lp_polish(X, seconds=0.9, deadline=deadline_final - 12.0)
        pv = max_cos(Xp)
        if pv < best_val - 1e-12:
            # extend polish on promising
            Xp2 = lp_polish(Xp, seconds=1.4, deadline=deadline_final - 12.0, radius_init=8e-4)
            pv2 = max_cos(Xp2)
            if pv2 < pv:
                Xp, pv = Xp2, pv2
            X_best = Xp
            best_val = pv
        elif pv < best_val + 0.002:
            # occasional extra squeeze if very close and cheap
            pass
        if time.time() >= deadline_final - 25.0:
            break

    rem = deadline_final - time.time() - 1.0
    if rem > 2.0:
        X_best = lp_polish(X_best, seconds=min(rem, 18.0), deadline=deadline_final - 1.0, radius_init=8e-4)
    X_best = X_best / np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best
