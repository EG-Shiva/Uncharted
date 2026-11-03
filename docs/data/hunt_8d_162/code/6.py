import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _lp_polish(X, seconds, start_time=None, deadline=None):
    X = _normalize(X.copy())
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    if start_time is None:
        start_time = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        now = time.time()
        if now - start_time >= seconds:
            break
        if deadline is not None and now >= deadline:
            break
        if radius < 1e-14:
            break
        Gfull = X @ X.T
        c = Gfull[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        y = X + res.x[:-1].reshape(n, d)
        y = _normalize(y)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _smooth_descend(X, betas, iters_list, lr0=0.15, mu=0.85):
    n, d = X.shape
    X = _normalize(X.copy())
    delta = np.zeros_like(X)
    lr = float(lr0)
    for bi, beta in enumerate(betas):
        iters = iters_list[bi]
        for k in range(iters):
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            W = np.exp(beta * (G - m))
            np.fill_diagonal(W, 0.0)
            S = float(W.sum() * 0.5)
            if S < 1e-300:
                S = 1e-300
            L = m + np.log(max(S, 1e-300)) / beta
            row = W.sum(axis=1, keepdims=True)
            row = np.maximum(row, 1e-30)
            Gavg = (W @ X) / row
            dot = (Gavg * X).sum(axis=1, keepdims=True)
            Gt = Gavg - dot * X
            # momentum proposal
            delta = mu * delta - lr * Gt
            ddot = (delta * X).sum(axis=1, keepdims=True)
            dt = delta - ddot * X
            Xt = X + dt
            Xt = _normalize(Xt)
            Gn = Xt @ Xt.T
            np.fill_diagonal(Gn, -2.0)
            mn = float(Gn.max())
            # sum exp, diag negligible (exp(-large)=0)
            Sn = float(np.exp(beta * (Gn - mn)).sum() * 0.5)
            Ln = mn + np.log(max(Sn, 1e-300)) / beta
            if Ln < L + 1e-12:
                X = Xt
                lr *= 1.03
                if lr > 1.0:
                    lr = 1.0
            else:
                lr *= 0.5
                delta *= 0.2
                if lr < 1e-5:
                    lr = 1e-5
        delta *= 0.5
        if lr < 1e-3:
            lr = 1e-3
        elif lr > 0.5:
            lr = 0.5
    return X

def solve():
    np.random.seed(0)
    rng = np.random.default_rng(0)
    t0 = time.time()
    DEADLINE = t0 + 172.0
    # load starts
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (162, 8):
                a = _normalize(a)
                cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        X0 = rng.standard_normal((162, 8))
        X0 = _normalize(X0)
        cands.append(X0)
    best = None
    best_val = 1e9
    for a in cands:
        v = _max_cos(a)
        if v < best_val:
            best_val = v
            best = a.copy()
    n, d = 162, 8
    # contact counts for targeting
    def contact_info(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        tight = G > m - 1e-7
        # each row count (sym, no diag)
        cnt = tight.sum(axis=1)
        return m, cnt, G
    trial = 0
    # smooth schedules
    betas_fast = [50.0, 120.0, 250.0]
    iters_fast = [220, 220, 220]
    betas_ref = [400.0]
    iters_ref = [150]
    while time.time() < DEADLINE - 14.0 and trial < 300:
        trial += 1
        _, cnt, _ = contact_info(best)
        order_hi = np.argsort(-cnt)
        order_lo = np.argsort(cnt)
        X = best.copy()
        mode = trial % 5
        if mode == 1:
            sig = 0.015 + 0.01 * float(rng.random())
            X = X + sig * rng.standard_normal(X.shape)
        elif mode == 2:
            sig = 0.05 + 0.02 * float(rng.random())
            X = X + sig * rng.standard_normal(X.shape)
        elif mode == 3:
            # targeted shake of overloaded hubs
            k = 6 + int(rng.integers(0, 3))
            victims = order_hi[:k]
            sig = 0.25 + 0.1 * float(rng.random())
            noise = rng.standard_normal((k, d))
            X[victims] = X[victims] + sig * noise
            # small global jitter
            X = X + 0.005 * rng.standard_normal(X.shape)
        elif mode == 4:
            # kill and reinsert: 1 rattler/low + 2 high + 1 random
            nv = 3 + int(rng.integers(0, 2))
            victims = set()
            victims.add(int(order_lo[int(rng.integers(0, 3))]))
            for j in range(2):
                victims.add(int(order_hi[int(rng.integers(0, 10))]))
            while len(victims) < nv:
                victims.add(int(rng.integers(0, n)))
            victims = sorted(victims)
            R = rng.standard_normal((len(victims), d))
            R = _normalize(R)
            X[victims] = R
            # if we killed too many, shake rest a bit
            X = X + 0.005 * rng.standard_normal(X.shape)
        else:
            sig = 0.10 + 0.05 * float(rng.random())
            X = X + sig * rng.standard_normal(X.shape)
        X = _normalize(X)
        # smooth descent
        try:
            Xs = _smooth_descend(X, betas_fast, iters_fast, lr0=0.15, mu=0.85)
        except Exception:
            continue
        v = _max_cos(Xs)
        # promising? refine with higher beta
        if v < best_val + 0.004:
            try:
                Xs2 = _smooth_descend(Xs, betas_ref, iters_ref, lr0=0.08, mu=0.85)
            except Exception:
                Xs2 = Xs
            v2 = _max_cos(Xs2)
            if v2 < v:
                Xs, v = Xs2, v2
        if v < best_val - 1e-12:
            # squeeze to local optimum with short LP
            remain = DEADLINE - time.time()
            if remain > 6.0:
                try:
                    Xs = _lp_polish(Xs, seconds=min(4.0, remain - 2.0))
                    v = _max_cos(Xs)
                except Exception:
                    pass
            best, best_val = Xs, v
            # early exit if huge gain? continue to improve
        # time check
        if time.time() >= DEADLINE - 14.0:
            break
    # final long polish of best
    try:
        remain = DEADLINE - time.time()
        if remain > 2.0:
            best = _lp_polish(best, seconds=min(12.0, max(2.0, remain - 1.0)))
    except Exception:
        pass
    best = _normalize(best)
    return best
