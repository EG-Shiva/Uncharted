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

def _micro(x, steps, beta, lr0, lr1):
    # plain tangential GD on sharp softmax, no momentum
    n, d = x.shape
    for t in range(steps):
        frac = t / max(1, steps - 1)
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        # sharp weights
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-300
        A = (W @ x) / S
        dot = (A * x).sum(axis=1, keepdims=True)
        Gt = A - dot * x
        x = x - lr * Gt
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x

def _greedy_max_steps(x, steps, lr):
    # attack absolute max pair directly, 2 points at a time
    n, d = x.shape
    for _ in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        k = int(np.argmax(G))
        j = k % n
        i = k // n
        if i == j:
            continue
        m = float(G[i, j])
        gi = x[j] - m * x[i]
        gj = x[i] - m * x[j]
        x[i] -= lr * gi
        x[j] -= lr * gj
        ni = np.linalg.norm(x[i])
        nj = np.linalg.norm(x[j])
        if ni < 1e-12 or nj < 1e-12:
            x[i] /= max(ni, 1e-12)
            x[j] /= max(nj, 1e-12)
        else:
            x[i] /= ni
            x[j] /= nj
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
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
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
    TL = 174.0
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

    # quick micro-equalize starting point immediately (cheap gain)
    # keep time bounded
    def timed_micro_equalize(X, budget):
        s = time.time()
        # schedule increasing sharpness
        sched = [(2e6, 3e-7, 1.2e-7, 1500), (1e7, 1.5e-7, 5e-8, 2000), (5e7, 6e-8, 2e-8, 2500), (2e8, 2.5e-8, 8e-9, 2500)]
        for beta, l0, l1, st in sched:
            if time.time() - s > budget:
                break
            # trim steps if budget tight (approx 25us/step)
            X = _micro(X, st, beta, l0, l1)
        # ultra-fine greedy on absolute max
        if time.time() - s < budget - 0.3:
            X = _greedy_max_steps(X, 3000, 4e-9)
        return X

    # initial fast squeeze of loaded best (2 sec)
    try:
        X0 = timed_micro_equalize(x_best.copy(), 2.0)
        v0 = _max_cos(X0)
        if v0 < best:
            best = v0
            x_best = X0
    except Exception:
        pass

    # contact degrees for guided kills
    def degrees(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        deg = (G > m - 1e-7).sum(axis=1)
        return deg

    try:
        deg_best = degrees(x_best)
        order_best = np.argsort(-deg_best)
    except Exception:
        order_best = np.arange(n)

    explore_end = t0 + TL - 26.0
    it = 0
    sigmas = [0.004, 0.01, 0.025, 0.06, 0.12, 0.28]
    while time.time() < explore_end:
        it += 1
        remain = explore_end - time.time()
        if remain < 0.6:
            break
        X = x_best.copy()
        r = rng.random()
        if r < 0.35:
            k = int(rng.integers(1, 4))
        elif r < 0.65:
            k = int(rng.integers(4, 9))
        else:
            k = int(rng.integers(9, 17))
        victims = set()
        if rng.random() < 0.7:
            perm = rng.permutation(order_best[:70])
            for idx in perm:
                victims.add(int(idx))
                if len(victims) >= k:
                    break
            while len(victims) < k:
                victims.add(int(rng.integers(n)))
        else:
            victims = set(rng.choice(n, size=k, replace=False).tolist())
        victims = np.array(sorted(victims), dtype=int)
        sigma = sigmas[int(rng.integers(len(sigmas)))]
        X += sigma * rng.standard_normal((n, d))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        # farthest-void reinsertion
        mask = np.ones(n, dtype=bool)
        mask[victims] = False
        Xrem = X[mask]
        M = 4000
        R = rng.standard_normal((M, d))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        s_rem = (R @ Xrem.T).max(axis=1)
        s_pick = np.full(M, -2.0)
        picked = []
        for _ in range(k):
            tot = np.maximum(s_rem, s_pick)
            j = int(np.argmin(tot))
            picked.append(R[j].copy())
            dp = R @ picked[-1]
            s_pick = np.maximum(s_pick, dp)
            s_pick[j] = 2.0
            s_rem[j] = 2.0
        X[victims] = np.stack(picked, axis=0)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        u = rng.random()
        if u < 0.35:
            steps, b0, b1, lr0, mu = 800, 20.0, 400.0, 0.07, 0.86
        elif u < 0.6:
            steps, b0, b1, lr0, mu = 550, 8.0, 150.0, 0.10, 0.80
        elif u < 0.85:
            steps, b0, b1, lr0, mu = 1100, 30.0, 700.0, 0.05, 0.88
        else:
            steps, b0, b1, lr0, mu = 350, 5.0, 80.0, 0.12, 0.75
        # time guard
        if time.time() + steps * 0.00003 + 0.15 > explore_end:
            steps = max(120, int((explore_end - time.time() - 0.15) / 0.00003))
            if steps <= 120:
                break
        X = _anneal(X, steps, b0, b1, lr0, mu)
        v = _max_cos(X)
        if v < best + 0.0015:
            # sharp micro screen (fast)
            X = _micro(X, 1200, 8e6, 1.2e-7, 4e-8)
            v = _max_cos(X)
        if v < best - 1e-12:
            # confirm with a touch more micro + short LP
            try:
                X2 = _micro(X.copy(), 1500, 4e7, 5e-8, 1.5e-8)
                v2 = _max_cos(X2)
                if v2 < v:
                    X, v = X2, v2
            except Exception:
                pass
            if v < best:
                # short LP snap to verify / improve
                try:
                    Xp = _lp_polish(X.copy(), 0.9)
                    vp = _max_cos(Xp)
                    # micro again after LP
                    Xq = _micro(Xp.copy(), 1200, 3e7, 4e-8, 1e-8)
                    vq = _max_cos(Xq)
                    if vq < vp:
                        Xp, vp = Xq, vq
                    if vp < best:
                        x_best, best = Xp, vp
                    elif v < best:
                        x_best, best = X, v
                    else:
                        x_best, best = X, v
                except Exception:
                    x_best, best = X, v
                try:
                    deg_best = degrees(x_best)
                    order_best = np.argsort(-deg_best)
                except Exception:
                    pass
        # occasional lateral diversification: keep topology change without improvement
        elif v < best + 2e-7 and (it % 40 == 0) and rng.random() < 0.2:
            pass

    # final intensive polish
    time_left = TL - (time.time() - t0)
    if time_left > 3.0:
        # alternate LP and micro
        lp_t = min(9.0, max(3.0, time_left * 0.35))
        try:
            x_best = _lp_polish(x_best, lp_t)
            best = _max_cos(x_best)
        except Exception:
            pass
        rem = TL - (time.time() - t0) - 0.5
        if rem > 2.0:
            # long micro schedule
            s_start = time.time()
            sched = [(3e6, 2.5e-7, 1e-7, 2500), (1.2e7, 1.2e-7, 5e-8, 3500), (6e7, 5e-8, 2e-8, 4000), (2.5e8, 2e-8, 6e-9, 4000)]
            for beta, l0, l1, st in sched:
                if time.time() - t0 > TL - 1.0:
                    break
                try:
                    x_best = _micro(x_best, st, beta, l0, l1)
                except Exception:
                    break
                nv = _max_cos(x_best)
                if nv < best:
                    best = nv
            # greedy finish if time remains
            if time.time() - t0 < TL - 1.5:
                try:
                    x_best = _greedy_max_steps(x_best, 5000, 3e-9)
                    best = _max_cos(x_best)
                except Exception:
                    pass
            # final LP touch
            if time.time() - t0 < TL - 1.0:
                try:
                    x_best = _lp_polish(x_best, max(1.0, TL - (time.time() - t0) - 0.4))
                    best = _max_cos(x_best)
                except Exception:
                    pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
