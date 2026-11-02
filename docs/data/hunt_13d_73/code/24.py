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

def _adam_smooth(X0, steps, s0, s1, lr0, lr1):
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
        se = E.sum()
        if se <= 0 or not np.isfinite(se):
            continue
        W = E / se
        Gt = W @ X
        dot = (Gt * X).sum(axis=1, keepdims=True)
        gt = Gt - dot * X
        m1 *= b1
        m1 += (1 - b1) * gt
        v1 *= b2
        v1 += (1 - b2) * (gt * gt)
        mhat = m1 / (1 - b1p)
        vhat = v1 / (1 - b2p)
        b1p *= b1
        b2p *= b2
        step = mhat / (np.sqrt(vhat) + eps)
        X -= lr * step
        nr = np.sqrt((X * X).sum(axis=1, keepdims=True))
        nr[nr == 0] = 1.0
        X /= nr
    return X

def _adam_power(X0, steps, p0, p1, lr0, lr1, thresh=0.0):
    X = X0.copy()
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = b1, b2
    lp0, lp1 = np.log(p0), np.log(p1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for k in range(steps):
        frac = k / (steps - 1) if steps > 1 else 1.0
        p = np.exp(lp0 + (lp1 - lp0) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        A = np.maximum(G - thresh, 0.0) ** max(p - 1.0, 0.0)
        np.fill_diagonal(A, 0.0)
        # weight by p * dot^{p-1}
        W = A * p
        s = W.sum()
        if s <= 0 or not np.isfinite(s):
            continue
        W = W / s
        Gt = W @ X
        dot = (Gt * X).sum(axis=1, keepdims=True)
        gt = Gt - dot * X
        m1 *= b1
        m1 += (1 - b1) * gt
        v1 *= b2
        v1 += (1 - b2) * (gt * gt)
        mhat = m1 / (1 - b1p)
        vhat = v1 / (1 - b2p)
        b1p *= b1
        b2p *= b2
        step = mhat / (np.sqrt(vhat) + eps)
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
    it = 0
    while time.time() - start < seconds and time.time() < t_end and radius > 1e-14 and it < 300:
        it += 1
        Cf = X @ X.T
        c = Cf[iu0]
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
        v = _max_cos(y)
        if v < best:
            X, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    SEARCH_END = t0 + 128.0
    rng = np.random.RandomState(7)
    n, d = 73, 13
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and xb.shape == (n, d):
        cands.append(xb / np.linalg.norm(xb, axis=1, keepdims=True))
    if xr is not None and xr.shape == (n, d):
        cands.append(xr / np.linalg.norm(xr, axis=1, keepdims=True))
    if len(cands) == 0:
        tmp = rng.randn(n, d)
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        cands.append(tmp)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    Xbest = cands[bi].copy()
    best = float(vals[bi])
    Xcur = Xbest.copy()
    cur = best

    trial = 0
    while time.time() < SEARCH_END and time.time() < DEADLINE - 35:
        trial += 1
        frac_g = (time.time() - t0) / max(SEARCH_END - t0, 1e-9)
        Temp = 0.0012 * (1 - frac_g) + 0.00004
        use_best = rng.rand() < 0.75
        base = Xbest if use_best else Xcur
        # choose victim count and sigmas
        mode = trial % 12
        Xtry = base.copy()
        try:
            if mode in (0, 4, 8, 10):
                # kill-and-hole: 1-3 victims
                if mode == 10:
                    k = 3
                elif mode == 4:
                    k = 2
                else:
                    k = 1
                    if rng.rand() < 0.25:
                        k = 2
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                cm = G.max()
                rmax = G.max(axis=1)
                cnt = (G >= cm - 2e-3).sum(axis=1)
                order_rmax = np.argsort(-rmax)
                order_cnt = np.argsort(-cnt)
                victims = set()
                for ii in range(k):
                    r = rng.rand()
                    if r < 0.5:
                        # worst rmax among top-8 random pick
                        pool = order_rmax[:8]
                        victims.add(int(pool[rng.randint(len(pool))]))
                    elif r < 0.8:
                        pool = order_cnt[:10]
                        victims.add(int(pool[rng.randint(len(pool))]))
                    else:
                        victims.add(int(rng.randint(n)))
                # ensure size k
                while len(victims) < k:
                    victims.add(int(rng.randint(n)))
                victims = list(victims)
                for idx in victims:
                    ntry = 4000 if time.time() < SEARCH_END - 10 else 1500
                    C = rng.randn(ntry, d)
                    C /= np.linalg.norm(C, axis=1, keepdims=True)
                    others = np.delete(Xtry, idx, axis=0)
                    dots = C @ others.T
                    md = dots.max(axis=1)
                    bii = int(np.argmin(md))
                    Xtry[idx] = C[bii]
                # short single-point tangential refine for inserted points
                # whole-config adam smoothing
                if rng.rand() < 0.5:
                    Xtry = _adam_smooth(Xtry, steps=260, s0=60.0, s1=400.0, lr0=0.004, lr1=0.0004)
                else:
                    Xtry = _adam_smooth(Xtry, steps=180, s0=40.0, s1=250.0, lr0=0.005, lr1=0.0005)
                    Xtry = _adam_power(Xtry, steps=120, p0=8.0, p1=28.0, lr0=0.004, lr1=0.0006)
            elif mode in (1, 5, 9):
                # gaussian kick to subset or all
                sig = [0.008, 0.02, 0.045, 0.09, 0.16][rng.randint(5)]
                if rng.rand() < 0.6:
                    # subset kick: worst 6-12 points stronger
                    rmax = _row_max(base)
                    ksub = int(rng.randint(5, 13))
                    idxs = np.argsort(-rmax)[:ksub]
                    noise = rng.randn(n, d) * (sig * 0.3)
                    noise[idxs] *= 3.0
                    Xtry = base + noise
                else:
                    Xtry = base + rng.randn(n, d) * sig
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
                Xtry = _adam_smooth(Xtry, steps=300, s0=50.0, s1=500.0, lr0=0.003, lr1=0.0003)
            elif mode in (2, 6):
                # coherent cap shake
                u = rng.randn(d)
                u /= np.linalg.norm(u)
                proj = base @ u
                thr = rng.uniform(0.35, 0.75)
                mask = proj > thr
                if mask.sum() < 2 or mask.sum() > 40:
                    # random subset instead
                    m = int(rng.randint(4, 12))
                    idxs = rng.choice(n, size=m, replace=False)
                    mask = np.zeros(n, dtype=bool)
                    mask[idxs] = True
                sig = rng.uniform(0.02, 0.12)
                Xtry = base.copy()
                Xtry[mask] += rng.randn(mask.sum(), d) * sig
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
                Xtry = _adam_smooth(Xtry, steps=280, s0=70.0, s1=450.0, lr0=0.0035, lr1=0.00035)
            else:
                # rattler / random-point greedy reinsert + polish
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                cm = G.max()
                cnt = (G >= cm - 1e-3).sum(axis=1)
                # rattlers = low contacts
                order = np.argsort(cnt)
                k = 1 if rng.rand() < 0.7 else 2
                victims = order[:k].copy()
                if rng.rand() < 0.4:
                    victims[0] = int(rng.randint(n))
                for idx in victims:
                    ntry = 5000
                    C = rng.randn(ntry, d)
                    C /= np.linalg.norm(C, axis=1, keepdims=True)
                    others = np.delete(Xtry, int(idx), axis=0)
                    md = (C @ others.T).max(axis=1)
                    Xtry[int(idx)] = C[int(np.argmin(md))]
                Xtry = _adam_smooth(Xtry, steps=220, s0=80.0, s1=350.0, lr0=0.004, lr1=0.0004)
            v = _max_cos(Xtry)
            # accept
            if v < cur or rng.rand() < np.exp(-(v - cur) / max(Temp, 1e-12)):
                Xcur = Xtry.copy()
                cur = v
            if v < best:
                Xbest = Xtry.copy()
                best = v
        except Exception:
            continue
    # final polish of best with remaining time
    try:
        remain = DEADLINE - time.time() - 2.0
        if remain > 5:
            Xbest = _lp_polish(Xbest, seconds=min(remain, 30.0), t_end=DEADLINE - 1.0)
            best = _max_cos(Xbest)
    except Exception:
        pass
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
