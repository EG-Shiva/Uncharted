import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _soft_momentum(x, steps, beta0, beta1, lr0, mu):
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
        v = v - vd * x
        x = x - lr * v
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(nrm, 1e-12, out=nrm)
        x /= nrm
    return x


def _adam_repulsion(X, steps, a0, a1, lr0, lr1):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2 = 0.9, 0.999
    eps = 1e-8
    p1 = 1.0
    p2 = 1.0
    for t in range(1, steps + 1):
        frac = t / float(steps)
        alpha = a0 + (a1 - a0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        D = X @ X.T
        np.fill_diagonal(D, -10.0)
        dm = D.max()
        W = np.exp(alpha * (D - dm))
        np.fill_diagonal(W, 0.0)
        G = W @ X
        s = (G * X).sum(axis=1, keepdims=True)
        Gt = G - s * X
        m *= b1
        m += (1.0 - b1) * Gt
        v *= b2
        v += (1.0 - b2) * (Gt * Gt)
        p1 *= b1
        p2 *= b2
        mhat = m / (1.0 - p1 + 1e-20)
        vhat = v / (1.0 - p2 + 1e-20)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sn = (step * X).sum(axis=1, keepdims=True)
        step -= sn * X
        X -= step
        nr = np.sqrt((X * X).sum(axis=1, keepdims=True))
        X /= np.maximum(nr, 1e-12)
    return X


def _greedy_reinsert(X_keep, k, rng, n_cand=12000):
    d = X_keep.shape[1]
    cand = rng.standard_normal((n_cand, d))
    cand /= np.maximum(np.linalg.norm(cand, axis=1, keepdims=True), 1e-12)
    # max dot to kept set
    D = cand @ X_keep.T
    curmax = D.max(axis=1)
    chosen = []
    cur = X_keep
    for _ in range(k):
        j = int(np.argmin(curmax))
        pt = cand[j].copy()
        chosen.append(pt)
        # update curmax with new point
        nd = cand @ pt
        np.maximum(curmax, nd, out=curmax)
        # mask chosen so not picked again (set to large)
        curmax[j] = 2.0
    pts = np.stack(chosen, axis=0)
    # small jitter to break symmetry
    pts = pts + 0.003 * rng.standard_normal(pts.shape)
    pts /= np.maximum(np.linalg.norm(pts, axis=1, keepdims=True), 1e-12)
    return np.vstack([X_keep, pts])


def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    TIME_LIMIT = 172.0
    try:
        xb = np.load("best.npy").astype(np.float64)
    except Exception:
        try:
            xb = np.load("record.npy").astype(np.float64)
        except Exception:
            rng0 = np.random.default_rng(0)
            xb = rng0.standard_normal((292, 4))
    if xb.shape != (292, 4):
        rng0 = np.random.default_rng(0)
        xb = rng0.standard_normal((292, 4))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best = _max_cos(xb)
    x_best = xb.copy()

    rng = np.random.default_rng(20261031)
    n, d = 292, 4

    explore_deadline = t0 + TIME_LIMIT - 32.0
    it = 0
    sigmas = [0.008, 0.02, 0.05, 0.12, 0.30]

    while time.time() < explore_deadline:
        it += 1
        if time.time() + 0.5 > explore_deadline and it > 5:
            break
        X = x_best.copy()
        u = rng.random()
        if u < 0.55:
            # targeted jammed-hub excision + farthest-void reinsertion
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            mval = float(G.max())
            rowmax = G.max(axis=1)
            deg = (G > mval - 3e-4).sum(axis=1).astype(np.float64)
            jitter = rng.random(n) * 1e-6
            score = rowmax + 0.0015 * deg + jitter
            k = int(rng.integers(2, 7))
            idx = np.argpartition(-score, k)[:k]
            X_keep = np.delete(X, idx, axis=0)
            try:
                X = _greedy_reinsert(X_keep, k, rng, n_cand=12000)
            except Exception:
                R = rng.standard_normal((k, d))
                R /= np.linalg.norm(R, axis=1, keepdims=True)
                X = np.vstack([X_keep, R])
            X = X + 0.004 * rng.standard_normal(X.shape)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        elif u < 0.85:
            sigma = sigmas[int(rng.integers(len(sigmas)))]
            X = X + sigma * rng.standard_normal(X.shape)
            if rng.random() < 0.6:
                k = int(rng.integers(1, 7))
                idx = rng.choice(n, size=k, replace=False)
                R = rng.standard_normal((k, d))
                R /= np.linalg.norm(R, axis=1, keepdims=True)
                X[idx] = R
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        else:
            # heavy shake / partial fresh restart
            if rng.random() < 0.4:
                R = rng.standard_normal((n, d))
                R /= np.linalg.norm(R, axis=1, keepdims=True)
                if rng.random() < 0.5:
                    X = R
                else:
                    X = X + 0.6 * rng.standard_normal((n, d))
                    X /= np.linalg.norm(X, axis=1, keepdims=True)
            else:
                X = X + 0.25 * rng.standard_normal(X.shape)
                X /= np.linalg.norm(X, axis=1, keepdims=True)

        # choose annealer
        r = rng.random()
        # adjust steps to remaining time
        remain = explore_deadline - time.time()
        if r < 0.55:
            steps, b0, b1, lr0, mu = 1400, 25.0, 380.0, 0.06, 0.85
            if r < 0.2:
                steps, b0, b1, lr0, mu = 2200, 15.0, 260.0, 0.10, 0.88
            use_adam = False
        elif r < 0.8:
            steps, b0, b1, lr0, mu = 2600, 40.0, 550.0, 0.04, 0.80
            use_adam = False
        else:
            # Adam branch
            steps = 1400
            use_adam = True
            a0, a1, alr0, alr1 = 30.0, 400.0, 0.004, 0.0005
            if rng.random() < 0.5:
                a0, a1, alr0, alr1 = 15.0, 250.0, 0.008, 0.001
                steps = 1800

        if time.time() + steps * 0.00028 > explore_deadline:
            steps = max(400, int((explore_deadline - time.time()) / 0.00028))
            if steps <= 120:
                break

        try:
            if use_adam:
                Xo = _adam_repulsion(X, steps, a0, a1, alr0, alr1)
            else:
                Xo = _soft_momentum(X, steps, b0, b1, lr0, mu)
        except Exception:
            continue
        if not np.all(np.isfinite(Xo)):
            continue
        val = _max_cos(Xo)
        # quick snap if promising
        promising = (val < best + 5e-4)
        if promising and (it % 2 == 0 or val < best):
            # short LP snap with small budget
            try:
                Xs = _lp_polish(Xo, 0.9)
                vals = _max_cos(Xs)
            except Exception:
                Xs, vals = Xo, val
            if vals < best and np.all(np.isfinite(Xs)):
                best = vals
                x_best = Xs.copy()
            elif val < best and np.all(np.isfinite(Xo)):
                best = val
                x_best = Xo.copy()
        else:
            if val < best:
                best = val
                x_best = Xo.copy()

    time_left = TIME_LIMIT - (time.time() - t0) - 1.0
    time_left = max(5.0, min(31.0, time_left))
    try:
        x_best = _lp_polish(x_best, time_left)
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    if not np.all(np.isfinite(x_best)):
        x_best = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    return x_best
