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

def _rand_sphere(rng, n, d):
    Y = rng.standard_normal(size=(n, d))
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def _adam_softmax(X0, steps, s0, s1, lr0, lr1, rng=None, t_end=1e30):
    X = X0.copy()
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = b1, b2
    ls0, ls1 = np.log(s0), np.log(s1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for k in range(steps):
        if k % 200 == 0 and time.time() > t_end:
            break
        frac = k / (steps - 1) if steps > 1 else 1.0
        s = np.exp(ls0 + (ls1 - ls0) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cm = G.max()
        E = np.exp(s * (G - cm))
        np.fill_diagonal(E, 0.0)
        se = E.sum()
        if not np.isfinite(se) or se <= 0:
            continue
        W = E / (se * 0.5)
        Gt = W @ X
        dot = (Gt * X).sum(axis=1, keepdims=True)
        gt = Gt - dot * X
        m *= b1
        m += (1 - b1) * gt
        v *= b2
        v += (1 - b2) * (gt * gt)
        mh = m / (1 - b1p)
        vh = v / (1 - b2p)
        b1p *= b1
        b2p *= b2
        X -= lr * mh / (np.sqrt(vh) + eps)
        nr = np.sqrt((X * X).sum(axis=1, keepdims=True))
        nr[nr == 0] = 1.0
        X /= nr
    return X

def _adam_power(X0, steps, lr0, lr1, p0, p1, margin, t_end=1e30):
    X = X0.copy()
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = b1, b2
    ll0, ll1 = np.log(lr0), np.log(lr1)
    lp0, lp1 = np.log(p0), np.log(p1)
    for k in range(steps):
        if k % 200 == 0 and time.time() > t_end:
            break
        frac = k / (steps - 1) if steps > 1 else 1.0
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        pw = np.exp(lp0 + (lp1 - lp0) * frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cm = G.max()
        th = cm - margin
        A = G - th
        A[A < 0] = 0.0
        np.fill_diagonal(A, 0.0)
        # weight = p * A^{p-1}
        with np.errstate(over='ignore', invalid='ignore'):
            W = np.power(A, pw - 1.0, out=np.zeros_like(A), where=A > 0)
            W *= pw
        np.fill_diagonal(W, 0.0)
        se = W.sum()
        if not np.isfinite(se) or se <= 0:
            continue
        W = W / (se * 0.5 + 1e-30)
        Gt = W @ X
        dot = (Gt * X).sum(axis=1, keepdims=True)
        gt = Gt - dot * X
        m *= b1
        m += (1 - b1) * gt
        v *= b2
        v += (1 - b2) * (gt * gt)
        mh = m / (1 - b1p)
        vh = v / (1 - b2p)
        b1p *= b1
        b2p *= b2
        X -= lr * mh / (np.sqrt(vh) + eps)
        nr = np.sqrt((X * X).sum(axis=1, keepdims=True))
        nr[nr == 0] = 1.0
        X /= nr
    return X

def _hole_replace(Y, idxs, rng, ntry=2500):
    Z = Y.copy()
    d = Y.shape[1]
    for idx in idxs:
        C = rng.standard_normal(size=(ntry, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        others = np.delete(Z, idx, axis=0)
        dots = C @ others.T
        md = dots.max(axis=1)
        bi = int(np.argmin(md))
        Z[idx] = C[bi]
    return Z

def _lp_polish(X, seconds, t_end=1e30):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end and radius > 1e-14:
        c = (X @ X.T)[iu0]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
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
        vv = _max_cos(y)
        if vv < best:
            X, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    SEARCH_END = t0 + 132.0
    rng = np.random.default_rng(12345)
    n, d = 73, 13
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = _rand_sphere(rng, n, d)
    if xb.shape != (n, d):
        tmp = _rand_sphere(rng, n, d)
        m0 = min(xb.shape[0], n); m1 = min(xb.shape[1], d)
        tmp[:m0, :m1] = xb[:m0, :m1]
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        xb = tmp
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    Xbest = xb.copy()
    Xcur = xb.copy()
    cur_val = best_val

    trial = 0
    # quick diverse restarts
    while time.time() < SEARCH_END and time.time() < DEADLINE - 38:
        trial += 1
        mode = trial % 4
        if mode == 0:
            # far random basin
            X0 = _rand_sphere(rng, n, d)
            steps = 1400
            s0, s1 = 25.0, 500.0
            lr0, lr1 = 0.02, 0.002
            X1 = _adam_softmax(X0, steps, s0, s1, lr0, lr1, t_end=DEADLINE - 35)
        elif mode == 1:
            # power-law repulsion from random
            X0 = _rand_sphere(rng, n, d)
            X1 = _adam_power(X0, 1200, 0.03, 0.003, 3.0, 12.0, 0.25, t_end=DEADLINE - 35)
            # sharpen with softmax
            X1 = _adam_softmax(X1, 500, 80.0, 600.0, 0.01, 0.002, t_end=DEADLINE - 35)
        elif mode == 2:
            # medium kick of best
            sig = float(rng.choice([0.02, 0.05, 0.12, 0.25]))
            X0 = Xbest + rng.standard_normal(size=(n, d)) * sig
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            X1 = _adam_softmax(X0, 1100, 40.0, 550.0, 0.015, 0.002, t_end=DEADLINE - 35)
        else:
            # kill worst 1-3 of best/cur and hole reinsert
            base = Xbest if rng.random() < 0.7 else Xcur
            k = int(rng.choice([1, 2, 2, 3]))
            rm = _row_max(base)
            order = np.argsort(-rm)
            victims = order[:k].copy()
            if rng.random() < 0.35:
                victims[0] = int(rng.integers(0, n))
            X0 = _hole_replace(base, victims.tolist(), rng, ntry=3000)
            # small jitter others
            X0 = X0 + rng.standard_normal(size=(n, d)) * 0.008
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            X1 = _adam_softmax(X0, 900, 50.0, 500.0, 0.012, 0.002, t_end=DEADLINE - 35)
            # second greedy pass on worst
            if time.time() < SEARCH_END and rng.random() < 0.6:
                w = np.argsort(-_row_max(X1))[:1].tolist()
                X1 = _hole_replace(X1, w, rng, ntry=2000)
                X1 = _adam_softmax(X1, 400, 120.0, 600.0, 0.008, 0.0015, t_end=DEADLINE - 35)
        v1 = _max_cos(X1)
        if v1 < cur_val:
            Xcur = X1.copy(); cur_val = v1
        if v1 < best_val:
            Xbest = X1.copy(); best_val = v1
        # occasional Metropolis to keep diversity
        if v1 < cur_val + 0.004 and rng.random() < 0.25:
            Xcur = X1.copy(); cur_val = v1

    # final sharpen + LP polish top candidates
    # short sharp anneal of best
    if time.time() < DEADLINE - 32:
        try:
            Xs = _adam_softmax(Xbest, 600, 150.0, 900.0, 0.006, 0.001, t_end=DEADLINE - 30)
            vs = _max_cos(Xs)
            if vs < best_val:
                Xbest = Xs; best_val = vs
        except Exception:
            pass
    if time.time() < DEADLINE - 30:
        try:
            Xs2 = _adam_softmax(Xcur, 600, 150.0, 900.0, 0.006, 0.001, t_end=DEADLINE - 30)
            vs2 = _max_cos(Xs2)
            if vs2 < best_val:
                Xbest = Xs2; best_val = vs2
        except Exception:
            pass
    # LP polish best (and cur if distinct)
    remain = DEADLINE - time.time()
    if remain > 5:
        try:
            Xp = _lp_polish(Xbest, seconds=min(remain - 1, 22), t_end=DEADLINE - 1)
            vp = _max_cos(Xp)
            if vp < best_val:
                Xbest = Xp; best_val = vp
            elif vp <= best_val + 1e-12:
                Xbest = Xp
        except Exception:
            pass
    remain = DEADLINE - time.time()
    if remain > 4:
        # try polishing cur as alternative basin
        try:
            Xp2 = _lp_polish(Xcur, seconds=min(remain - 1, 8), t_end=DEADLINE - 1)
            vp2 = _max_cos(Xp2)
            if vp2 < best_val:
                Xbest = Xp2; best_val = vp2
        except Exception:
            pass
    Xbest /= np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
