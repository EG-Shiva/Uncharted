import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _anneal(x, steps, beta0, beta1, lr0, mu):
    n, d = x.shape
    v = np.zeros_like(x)
    lb = np.log(beta1 / beta0) if beta1 > beta0 else 0.0
    for t in range(steps):
        frac = t / max(1, steps - 1)
        beta = beta0 * np.exp(lb * frac)
        lr = lr0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-300
        A = (W @ x) / S
        dot = (A * x).sum(axis=1, keepdims=True)
        Gt = A - dot * x
        v = mu * v + Gt
        vd = (v * x).sum(axis=1, keepdims=True)
        v -= vd * x
        x = x - lr * v
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(nrm, 1e-12, out=nrm)
        x /= nrm
    return x


def _cheap_max_descent(x, steps, lr0, beta, tol=0.004):
    # subgradient descent on near-max pairs only, tangential
    n, d = x.shape
    lr = lr0
    for t in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        # active band shrinks
        band = tol * (1.0 - 0.8 * t / max(1, steps - 1))
        Mask = G > m - band
        # weights: uniform over active, symmetric
        # gradient of active cosine sum
        # Gact = active cosines
        W = np.zeros_like(G)
        W[Mask] = 1.0
        np.fill_diagonal(W, 0.0)
        deg = W.sum(axis=1, keepdims=True) + 1e-300
        A = (W @ x) / deg
        # only move points that are active
        has = (deg.ravel() > 0.5)
        dot = (A * x).sum(axis=1, keepdims=True)
        Gt = A - dot * x
        Gt[~has] = 0.0
        # normalize step by max norm
        gn = np.linalg.norm(Gt, axis=1).max() + 1e-300
        step = lr / (1.0 + gn * 20.0) * 20.0
        # simpler: fixed lr scaled
        x = x - lr * Gt
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        lr *= 0.998
    return x


def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    TL = 175.0
    try:
        xb = np.load("best.npy").astype(np.float64)
    except Exception:
        try:
            xb = np.load("record.npy").astype(np.float64)
        except Exception:
            xb = None
    if xb is None or xb.shape != (292, 4):
        rng0 = np.random.default_rng(0)
        xb = rng0.standard_normal((292, 4))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best = _max_cos(xb)
    x_best = xb.copy()

    rng = np.random.default_rng(20261031)
    n, d = 292, 4

    # precompute contact degrees of current best for guided kills
    def contact_degrees(X, thresh):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        B = G > m - thresh
        deg = B.sum(axis=1)
        return deg, m

    explore_end = t0 + TL - 30.0
    it = 0
    sigmas = [0.005, 0.012, 0.03, 0.07, 0.15, 0.32]

    # cache degrees of best, refresh when best improves
    deg_best, _ = contact_degrees(x_best, 1e-7)
    order_best = np.argsort(-deg_best)

    while time.time() < explore_end:
        it += 1
        if time.time() + 0.5 > explore_end and it > 10:
            break
        X = x_best.copy()
        # choose kill count
        r = rng.random()
        if r < 0.3:
            k = int(rng.integers(1, 4))
        elif r < 0.6:
            k = int(rng.integers(4, 9))
        else:
            k = int(rng.integers(9, 18))
        # choose victims: bias to high-degree jammed nodes
        victims = set()
        if rng.random() < 0.75:
            pool_top = order_best[:60]
            # sample with random permutation
            perm = rng.permutation(pool_top)
            for idx in perm:
                victims.add(int(idx))
                if len(victims) >= k:
                    break
            while len(victims) < k:
                victims.add(int(rng.integers(n)))
        else:
            victims = set(rng.choice(n, size=k, replace=False).tolist())
        victims = np.array(sorted(victims), dtype=int)
        # global shake
        sigma = sigmas[int(rng.integers(len(sigmas)))]
        X += sigma * rng.standard_normal((n, d))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        # hole filling: keep remaining points, search holes
        mask = np.ones(n, dtype=bool)
        mask[victims] = False
        Xrem = X[mask]
        # sample candidates
        M = 3000
        R = rng.standard_normal((M, d))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        # score = max dot to remaining
        D = R @ Xrem.T
        s_rem = D.max(axis=1)
        # greedy sequential pick to spread holes
        picked = []
        s_pick = np.full(M, -2.0)
        # order candidates by s_rem
        for _ in range(k):
            # total score = max(s_rem, s_pick)
            tot = np.maximum(s_rem, s_pick)
            # add tiny noise to break ties deterministically via rng
            # pick best (minimum)
            j = int(np.argmin(tot))
            picked.append(R[j].copy())
            # update s_pick with dot to picked
            dp = R @ picked[-1]
            s_pick = np.maximum(s_pick, dp)
            # penalize re-picking same: set to large
            s_pick[j] = 2.0
            s_rem[j] = 2.0
        X[victims] = np.stack(picked, axis=0)
        X /= np.linalg.norm(X, axis=1, keepdims=True)

        # anneal hyperparams diverse
        u = rng.random()
        if u < 0.3:
            steps, b0, b1, lr0, mu = 900, 20.0, 400.0, 0.07, 0.86
        elif u < 0.55:
            steps, b0, b1, lr0, mu = 600, 8.0, 150.0, 0.10, 0.80
        elif u < 0.8:
            steps, b0, b1, lr0, mu = 1200, 30.0, 700.0, 0.05, 0.88
        else:
            steps, b0, b1, lr0, mu = 400, 5.0, 80.0, 0.12, 0.75
        # small chance of very long refine
        if rng.random() < 0.08:
            steps = 1800

        # time guard: shorten steps if little time
        if time.time() + steps * 0.00035 > explore_end:
            steps = max(150, int((explore_end - time.time()) / 0.00035))
            if steps <= 150:
                break

        X = _anneal(X, steps, b0, b1, lr0, mu)
        v = _max_cos(X)
        # cheap max descent if promising or very close
        if v < best + 0.004:
            X = _cheap_max_descent(X, 180, 0.02, 0.0)
            v = _max_cos(X)
        if v < best - 2e-12:
            # quick LP refine 1.5s to confirm
            Xp = _lp_polish(X, 1.5)
            vp = _max_cos(Xp)
            if vp < best:
                x_best = Xp
                best = vp
                deg_best, _ = contact_degrees(x_best, 1e-7)
                order_best = np.argsort(-deg_best)
            elif v < best:
                x_best = X
                best = v
                deg_best, _ = contact_degrees(x_best, 1e-7)
                order_best = np.argsort(-deg_best)
        elif v < best + 0.0005 and (it % 25 == 0):
            # occasional lateral move to different basin with same quality
            # accept if it changes topology (keep diversity) - small chance
            if rng.random() < 0.15:
                Xp = _lp_polish(X, 1.0)
                vp = _max_cos(Xp)
                if vp < best + 1e-7:
                    # keep as new base if contact structure differs? just keep if better or equal + noise
                    if vp < best + 5e-9 and rng.random() < 0.3:
                        pass  # do not replace to avoid drift; keep best
                    else:
                        pass

    # final intensive polish of best
    time_left = TL - (time.time() - t0)
    fp = max(8.0, min(28.0, time_left - 1.0))
    if fp > 2.0:
        x_best = _lp_polish(x_best, fp)
        # extra cheap + lp alternate
        if time.time() - t0 < TL - 4.0:
            x_best = _cheap_max_descent(x_best, 250, 0.008, 0.0)
            x_best = _lp_polish(x_best, max(2.0, TL - (time.time() - t0) - 0.5))
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
