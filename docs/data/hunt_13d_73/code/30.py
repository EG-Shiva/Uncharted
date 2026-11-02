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
    Y = rng.normal(size=(n, d))
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def _find_hole(X_keep, rng, n_cand=4000):
    d = X_keep.shape[1]
    C = rng.normal(size=(n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = C @ X_keep.T
    m = dots.max(axis=1)
    return C[int(np.argmin(m))]

def _refine(X0, steps, s0, s1, lr0, lr1, t_end=1e30):
    X = X0.copy()
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = b1, b2
    ls0, ls1 = np.log(s0), np.log(s1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for k in range(steps):
        if k % 32 == 0 and time.time() > t_end:
            break
        frac = k / (steps - 1) if steps > 1 else 1.0
        s = np.exp(ls0 + (ls1 - ls0) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cmax = float(G.max())
        # softmax weights, guard overflow (s*(G-cmax)<=0)
        E = np.exp(s * (G - cmax))
        np.fill_diagonal(E, 0.0)
        sumE = E.sum() * 0.5
        if not np.isfinite(sumE) or sumE <= 0:
            continue
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
        # clip extreme steps for stability
        X -= lr * step
        nr = np.sqrt((X * X).sum(axis=1, keepdims=True))
        nr[nr == 0] = 1.0
        X /= nr
    return X

def _lp_polish(X, seconds, t_end=1e30):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end and radius > 1e-14:
        c = (X @ X.T)[iu0]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu0[0][sel], iu0[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
    DEADLINE = t0 + 172.0
    SEARCH_END = t0 + 140.0
    rng = np.random.default_rng(0)
    n, d = 73, 13
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = _random_sphere(rng, n, d)
    if xb.shape != (n, d):
        tmp = _random_sphere(rng, n, d)
        m0 = min(xb.shape[0], n); m1 = min(xb.shape[1], d)
        tmp[:m0, :m1] = xb[:m0, :m1]
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        xb = tmp
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    best = _max_cos(xb)
    Xbest = xb.copy()
    trial = 0
    # precompute nothing heavy; fast loop
    while time.time() < SEARCH_END and time.time() < DEADLINE - 30.0:
        trial += 1
        mode = trial % 7
        try:
            if mode == 0:
                # fresh random basin for isotropy/balance
                Xt = _random_sphere(rng, n, d)
                steps, s0, s1, lr0, lr1 = 700, 12.0, 280.0, 0.03, 0.002
            elif mode == 1:
                # heavy shake to leave jammed graph
                Xt = Xbest + 0.45 * rng.normal(size=(n, d))
                Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
                steps, s0, s1, lr0, lr1 = 600, 15.0, 250.0, 0.02, 0.002
            elif mode == 2:
                # kill 2 most overloaded -> holes (coordinate expansion)
                rm = _row_max(Xbest)
                order = np.argsort(-rm)
                k = 2
                victims = order[:k].copy()
                if rng.random() < 0.4:
                    victims[0] = int(rng.integers(0, n))
                Xt = Xbest.copy()
                keep = np.ones(n, dtype=bool); keep[victims] = False
                cur_keep = Xbest[keep].copy()
                for idx in victims:
                    h = _find_hole(cur_keep, rng, n_cand=4000)
                    Xt[idx] = h
                    cur_keep = np.vstack([cur_keep, h[None, :]])
                Xt += 0.01 * rng.normal(size=Xt.shape)
                Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
                steps, s0, s1, lr0, lr1 = 500, 15.0, 220.0, 0.02, 0.002
            elif mode == 3:
                # kill 1 worst -> hole
                rm = _row_max(Xbest)
                vi = int(np.argmax(rm))
                if rng.random() < 0.3:
                    vi = int(rng.integers(0, n))
                mask = np.ones(n, dtype=bool); mask[vi] = False
                h = _find_hole(Xbest[mask], rng, n_cand=5000)
                Xt = Xbest.copy(); Xt[vi] = h
                Xt += 0.008 * rng.normal(size=Xt.shape)
                Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
                steps, s0, s1, lr0, lr1 = 450, 15.0, 220.0, 0.02, 0.002
            elif mode == 4:
                # medium shake
                Xt = Xbest + 0.09 * rng.normal(size=(n, d))
                Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
                steps, s0, s1, lr0, lr1 = 450, 20.0, 260.0, 0.015, 0.0015
            elif mode == 5:
                # random subset shake: move a coherent cap
                Xt = Xbest.copy()
                c = _random_sphere(rng, 1, d)[0]
                dots = Xt @ c
                cap = np.where(dots > 0.55)[0]
                if len(cap) < 3:
                    cap = np.argsort(-dots)[:6]
                Xt[cap] += 0.25 * rng.normal(size=(len(cap), d))
                Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
                steps, s0, s1, lr0, lr1 = 500, 15.0, 240.0, 0.02, 0.002
            else:
                # small shake
                Xt = Xbest + 0.03 * rng.normal(size=(n, d))
                Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
                steps, s0, s1, lr0, lr1 = 400, 20.0, 260.0, 0.015, 0.0015
            Xo = _refine(Xt, steps, s0, s1, lr0, lr1, t_end=min(SEARCH_END, DEADLINE - 30.0))
            v = _max_cos(Xo)
            # one greedy balance pass if close to best: reinsert worst and short anneal
            if v < best + 0.004 and time.time() < SEARCH_END - 2.0:
                rm = _row_max(Xo)
                vi = int(np.argmax(rm))
                # only if clearly overloaded and rattler exists
                G = Xo @ Xo.T
                np.fill_diagonal(G, -2.0)
                cm = float(G.max())
                cnt = (G >= cm - 1e-3).sum(axis=1)
                if cnt[vi] >= 18 and cnt.min() <= 6:
                    mask = np.ones(n, dtype=bool); mask[vi] = False
                    h = _find_hole(Xo[mask], rng, n_cand=3000)
                    Xg = Xo.copy(); Xg[vi] = h
                    Xg = _refine(Xg, 180, 20.0, 200.0, 0.012, 0.0015, t_end=min(SEARCH_END, DEADLINE - 30.0))
                    vg = _max_cos(Xg)
                    if vg < v:
                        Xo, v = Xg, vg
            if v < best - 1e-12:
                Xbest = Xo.copy()
                best = float(v)
                if best < 0.19:
                    break
        except Exception:
            continue
    # final squeeze of incumbent
    remain = DEADLINE - time.time() - 2.0
    if remain > 5.0:
        try:
            Xp = _lp_polish(Xbest, seconds=min(remain, 28.0), t_end=DEADLINE - 1.0)
            vp = _max_cos(Xp)
            if vp < best:
                Xbest = Xp
                best = float(vp)
        except Exception:
            pass
    Xbest /= np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
