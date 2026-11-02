import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _random_sphere(rng, n, d):
    Y = rng.standard_normal(size=(n, d))
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def _hole_for(X_keep, rng, n_cand=2500):
    d = X_keep.shape[1]
    C = rng.standard_normal(size=(n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = C @ X_keep.T
    m = dots.max(axis=1)
    return C[int(np.argmin(m))]

def _refine(X0, steps, s0, s1, lr0, lr1, rng=None, do_kill=True):
    X = X0.copy()
    n, d = X.shape
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = b1, b2
    ls0, ls1 = np.log(s0), np.log(s1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for k in range(steps):
        frac = k / (steps - 1) if steps > 1 else 1.0
        s = np.exp(ls0 + (ls1 - ls0) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cmax = float(G.max())
        E = np.exp(s * (G - cmax))
        np.fill_diagonal(E, 0.0)
        sumE = float(E.sum() * 0.5)
        if sumE == 0.0 or not np.isfinite(sumE):
            Mask = (G >= cmax - 1e-12)
            np.fill_diagonal(Mask, False)
            cnt = float(Mask.sum() * 0.5)
            if cnt <= 0:
                continue
            W = Mask.astype(float) / cnt
        else:
            W = E / sumE
        Gt = W @ X
        dot = (Gt * X).sum(axis=1, keepdims=True)
        gt = Gt - dot * X
        m1 *= b1
        m1 += (1.0 - b1) * gt
        v1 *= b2
        v1 += (1.0 - b2) * (gt * gt)
        mhat = m1 / (1.0 - b1p)
        vhat = v1 / (1.0 - b2p)
        b1p *= b1
        b2p *= b2
        step = mhat / (np.sqrt(vhat) + eps)
        # clip extreme steps
        X -= lr * step
        nr = np.sqrt((X * X).sum(axis=1, keepdims=True))
        nr[nr == 0] = 1.0
        X /= nr
        if do_kill and rng is not None and (k == steps // 2 or k == steps // 4 * 3):
            # replace single worst point by hole
            G2 = X @ X.T
            np.fill_diagonal(G2, -2.0)
            mx = G2.max(axis=1)
            wi = int(np.argmax(mx))
            keep = np.delete(X, wi, axis=0)
            X[wi] = _hole_for(keep, rng, n_cand=2000)
    return X

def _lp_polish(X, seconds, t_end):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end and radius > 1e-14:
        Cfull = X @ X.T
        c = Cfull[iu0]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu0[0][sel], iu0[1][sel]
        p = len(I)
        if p == 0:
            break
        # cap constraints for speed
        if p > 600:
            # keep most tight
            idx = np.argsort(-c[sel])[:600]
            I, J = I[idx], J[idx]
            c_sel = c[sel][idx]
            p = 600
        else:
            c_sel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            X, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    DEADLINE = t0 + 170.0
    SEARCH_END = t0 + 148.0
    LP_END = t0 + 168.0
    rng = np.random.RandomState(12345)
    n, d = 73, 13
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.randn(n, d)
    if xb.shape != (n, d):
        tmp = rng.randn(n, d)
        m0 = min(xb.shape[0], n); m1 = min(xb.shape[1], d)
        tmp[:m0, :m1] = xb[:m0, :m1]
        xb = tmp
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    Xbest = xb.copy()

    trial = 0
    # quick diverse search
    while time.time() < SEARCH_END and time.time() < DEADLINE - 20:
        trial += 1
        remain = SEARCH_END - time.time()
        if remain < 1.0:
            break
        r = rng.rand()
        if trial % 6 == 0:
            # large kick off incumbent + kill worst 2
            sigma = 0.35 if (trial % 12 == 0) else 0.22
            X0 = Xbest + sigma * rng.randn(n, d)
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            # kill 2 worst into holes
            mx = _row_max(X0)
            order = np.argsort(-mx)[:2]
            for wi in order:
                keep = np.delete(X0, int(wi), axis=0)
                X0[int(wi)] = _hole_for(keep, rng, n_cand=2500)
            steps1, s0, s1, lr0, lr1 = 900, 40.0, 350.0, 0.02, 0.003
        elif r < 0.75:
            X0 = _random_sphere(rng, n, d)
            steps1, s0, s1, lr0, lr1 = 1100, 30.0, 300.0, 0.025, 0.004
        else:
            # medium kick
            sigma = 0.12
            base = Xbest
            X0 = base + sigma * rng.randn(n, d)
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            steps1, s0, s1, lr0, lr1 = 700, 50.0, 400.0, 0.015, 0.002
        try:
            X1 = _refine(X0, steps=steps1, s0=s0, s1=s1, lr0=lr0, lr1=lr1, rng=rng, do_kill=True)
        except Exception:
            continue
        # one extra greedy reinsert of worst then short anneal
        try:
            v1 = _max_cos(X1)
            # only do extra work if promising (within 0.03 of best)
            if v1 < best_val + 0.03:
                mx = _row_max(X1)
                wi = int(np.argmax(mx))
                keep = np.delete(X1, wi, axis=0)
                X1[wi] = _hole_for(keep, rng, n_cand=3000)
                X1 = _refine(X1, steps=350, s0=150.0, s1=500.0, lr0=0.008, lr1=0.0015, rng=None, do_kill=False)
                v1 = _max_cos(X1)
            else:
                v1 = _max_cos(X1)
        except Exception:
            continue
        if v1 < best_val and np.all(np.isfinite(X1)):
            Xbest = X1 / np.linalg.norm(X1, axis=1, keepdims=True)
            best_val = float(v1)
            # early exit if clearly new record by margin
            if best_val < 0.207434422294 - 1e-6:
                pass
        if time.time() > SEARCH_END:
            break

    # short final LP squeeze on incumbent if time permits
    try:
        if time.time() < LP_END - 2:
            secs = max(1.0, min(14.0, LP_END - time.time() - 1.0))
            Xp = _lp_polish(Xbest, seconds=secs, t_end=LP_END)
            vp = _max_cos(Xp)
            if vp < best_val and np.all(np.isfinite(Xp)):
                Xbest = Xp
                best_val = vp
    except Exception:
        pass
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
