import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _mc(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _anneal(X, steps, T0, T1, lr, rng=None):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    if steps > 1:
        la0 = np.log(max(T0, 1e-12))
        la1 = np.log(max(T1, 1e-12))
    for t in range(1, steps + 1):
        frac = t / steps
        if steps > 1:
            T = float(np.exp(la0 + (la1 - la0) * frac))
        else:
            T = T0
        if T < 5e-4:
            T = 5e-4
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mv = float(G.max())
        E = np.exp((G - mv) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            break
        Grad = (E @ X) / S
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        # adam
        b1p = 1.0 - b1 ** t
        b2p = 1.0 - b2 ** t
        m *= b1
        m += (1.0 - b1) * Gt
        v *= b2
        v += (1.0 - b2) * (Gt * Gt)
        step = (m / b1p) / (np.sqrt(v / b2p) + eps)
        # mild decay of lr
        lrc = lr * (1.0 - 0.5 * frac)
        X = X - lrc * step
        X = _norm(X)
    return X

def _hard(X, iters=200, lr0=0.012, window=0.004):
    X = X.copy()
    for it in range(iters):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(G.max())
        W = (G > M - window).astype(float)
        np.fill_diagonal(W, 0.0)
        # symmetric weight; gradient
        F = W @ X
        dot = (F * X).sum(axis=1, keepdims=True)
        Ft = F - dot * X
        # normalize scale by mean degree
        lr = lr0 * (1.0 - 0.7 * it / max(iters, 1))
        X = X - lr * Ft / (1.0 + float(W.sum() / X.shape[0]))
        X = _norm(X)
    return X

def _lp_polish(x, seconds, deadline=1e18):
    x = _norm(x)
    n, d = x.shape
    best = _mc(x)
    radius = 2e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-14:
            break
        xx = x @ x.T
        c = xx[iu]
        m = float(c.max())
        sel = c > m - max(25.0 * radius, 2e-9)
        # cap to top 3000 to keep linprog fast
        idx = np.where(sel)[0]
        if len(idx) > 3000:
            order = np.argpartition(c[idx], -3000)[-3000:]
            idx = idx[order]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx] = True
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
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
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        vv = _mc(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    rng = np.random.default_rng(42)
    n, d = 154, 10
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d):
                a = _norm(a.astype(float))
                vv = _mc(a)
                if vv < best_val:
                    best_val = float(vv)
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.normal(size=(n, d)))
        best_val = _mc(best)
    base_best = best.copy()
    # exploration with large topology changes, no LP inside
    HOP_END = DEAD - 32.0
    trial = 0
    # keep a small pool for crossover diversity
    pool = [best.copy()]
    pool_vals = [best_val]
    while time.time() < HOP_END:
        trial += 1
        u = rng.random()
        if u < 0.35:
            # fresh random restart
            X = _norm(rng.normal(size=(n, d)))
            steps = 750
            T0 = 0.02 + rng.random() * 0.01
            T1 = 0.0009 + rng.random() * 0.0006
            lr = 0.018 + rng.random() * 0.008
        elif u < 0.65:
            # large Gaussian shake of best
            sigma = 0.25 + rng.random() * 0.35  # 0.25..0.60 far basin
            X = best + sigma * rng.normal(size=(n, d))
            X = _norm(X)
            steps = 650
            T0 = 0.016 + rng.random() * 0.008
            T1 = 0.0008 + rng.random() * 0.0007
            lr = 0.016 + rng.random() * 0.008
        elif u < 0.85:
            # block replacement 10-38 points
            k = int(rng.integers(10, 39))
            # choose source: best or pool member
            src = best if rng.random() < 0.7 else pool[int(rng.integers(len(pool)))]
            X = src.copy()
            victims = rng.choice(n, size=k, replace=False)
            X[victims] = _norm(rng.normal(size=(k, d)))
            # optionally shuffle a bit the rest
            X = _norm(X + rng.normal(scale=0.02, size=(n, d)) * 0.3)
            steps = 700
            T0 = 0.018 + rng.random() * 0.008
            T1 = 0.0008 + rng.random() * 0.0007
            lr = 0.017 + rng.random() * 0.007
        else:
            # crossover of best with random / pool
            if len(pool) > 1 and rng.random() < 0.5:
                a2 = pool[int(rng.integers(len(pool)))]
            else:
                a2 = _norm(rng.normal(size=(n, d)))
            mask = rng.random(size=(n, 1)) < 0.5
            X = np.where(mask, best, a2)
            # random orthogonal-ish jitter: add noise
            X = _norm(X + rng.normal(scale=0.12, size=(n, d)))
            steps = 700
            T0 = 0.018 + rng.random() * 0.008
            T1 = 0.0009 + rng.random() * 0.0006
            lr = 0.017 + rng.random() * 0.007
        # adapt steps to remaining time (avoid overrun)
        remain = HOP_END - time.time()
        if remain < 3:
            break
        # quick budget: reduce steps if little time
        if remain < 20:
            steps = min(steps, 450)
        try:
            X = _anneal(X, steps=steps, T0=T0, T1=T1, lr=lr)
            # hard-max tail to tie contacts
            X = _hard(X, iters=160, lr0=0.010, window=0.005)
            # second short anneal at low T to settle
            X = _anneal(X, steps=180, T0=0.003, T1=0.0006, lr=0.010)
        except Exception:
            continue
        try:
            vv = _mc(X)
        except Exception:
            continue
        if vv < best_val:
            best_val = float(vv)
            best = X.copy()
            pool.append(X.copy())
            pool_vals.append(vv)
            if len(pool) > 5:
                # keep best 5
                o = np.argsort(pool_vals)[:5]
                pool = [pool[i] for i in o]
                pool_vals = [pool_vals[i] for i in o]
        elif rng.random() < 0.05 and len(pool) < 6:
            # keep some diversity
            pool.append(X.copy())
            pool_vals.append(vv)
        # occasional micro-jitter intensification around best
        if trial % 25 == 0 and time.time() < HOP_END - 5:
            try:
                Y = best + rng.normal(scale=0.015, size=best.shape)
                Y = _norm(Y)
                Y = _anneal(Y, steps=300, T0=0.004, T1=0.0006, lr=0.012)
                Y = _hard(Y, iters=120, lr0=0.008, window=0.003)
                vy = _mc(Y)
                if vy < best_val:
                    best_val = float(vy)
                    best = Y.copy()
            except Exception:
                pass
    # final intensification: anneal + hard + LP, plus jittered LP retries
    try:
        # a couple of low-T anneal passes on best
        for _ in range(2):
            if time.time() > DEAD - 28:
                break
            best = _anneal(best, steps=350, T0=0.0035, T1=0.0005, lr=0.010)
            best = _hard(best, iters=150, lr0=0.008, window=0.003)
            best_val = _mc(best)
    except Exception:
        pass
    remain = DEAD - time.time() - 1.0
    if remain < 5:
        remain = 5
    if remain > 30:
        remain = 30
    # LP polish chunk
    try:
        lp_time = remain * 0.6
        best = _lp_polish(best, seconds=lp_time, deadline=DEAD - 6.0)
        best_val = _mc(best)
    except Exception:
        pass
    # jittered LP retries to escape tolerance artifact
    try:
        while time.time() < DEAD - 3.0:
            sig = 3e-4 if rng.random() < 0.5 else 1e-3
            Y = _norm(best + rng.normal(scale=sig, size=best.shape))
            # short hard push then quick LP
            Y = _hard(Y, iters=80, lr0=0.005, window=0.002)
            Y = _lp_polish(Y, seconds=2.0, deadline=DEAD - 1.0)
            vy = _mc(Y)
            if vy < best_val:
                best = Y.copy()
                best_val = float(vy)
            # if stuck and time low, break
            if DEAD - time.time() < 4.0:
                break
    except Exception:
        pass
    # safety: if we somehow got worse than loaded base, return base
    try:
        if _mc(best) > _mc(base_best) + 1e-12:
            # keep the better (base) polished briefly if time remains
            best = base_best
    except Exception:
        pass
    best = _norm(best)
    return best
