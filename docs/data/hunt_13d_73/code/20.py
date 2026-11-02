import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _rowwise_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _lp_polish(X, seconds, t_end=1e30):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end and radius > 1e-14:
        Cfull = X @ X.T
        c = Cfull[iu]
        m = c.max()
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
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
    SEARCH_END = t0 + 138.0
    rng = np.random.default_rng(0)
    n, d = 73, 13
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.normal(size=(n, d))
    if xb.shape != (n, d):
        tmp = rng.normal(size=(n, d))
        m0 = min(xb.shape[0], n); m1 = min(xb.shape[1], d)
        try:
            tmp[:m0, :m1] = xb[:m0, :m1]
        except Exception:
            pass
        xb = tmp
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    best = _max_cos(xb)
    Xbest = xb.copy()
    Xcur = xb.copy()
    cur = best

    def refine(X0, steps, s0, s1, lr0, lr1):
        X = X0.copy()
        m1 = np.zeros_like(X)
        v1 = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        b1p, b2p = b1, b2
        ls0, ls1 = np.log(s0), np.log(s1)
        ll0, ll1 = np.log(lr0), np.log(lr1)
        for k in range(steps):
            if time.time() > DEADLINE - 3:
                break
            frac = k / (steps - 1) if steps > 1 else 1.0
            s = np.exp(ls0 + (ls1 - ls0) * frac)
            lr = np.exp(ll0 + (ll1 - ll0) * frac)
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            cmax = G.max()
            # softmax weights
            E = np.exp(s * (G - cmax))
            np.fill_diagonal(E, 0.0)
            sumE = E.sum() * 0.5
            if sumE <= 0 or not np.isfinite(sumE):
                continue
            W = E / sumE
            Gt = W @ X
            dot = (Gt * X).sum(axis=1, keepdims=True)
            gt = Gt - dot * X
            m1 *= b1; m1 += (1 - b1) * gt
            v1 *= b2; v1 += (1 - b2) * (gt * gt)
            mhat = m1 / (1 - b1p + 1e-16)
            vhat = v1 / (1 - b2p + 1e-16)
            b1p *= b1; b2p *= b2
            step = mhat / (np.sqrt(vhat) + eps)
            X -= lr * step
            nr = np.sqrt((X * X).sum(axis=1, keepdims=True))
            nr[nr == 0] = 1.0
            X /= nr
        return X

    def deep_hole(others, n_cand=12000):
        C = rng.normal(size=(n_cand, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        dots = C @ others.T
        md = dots.max(axis=1)
        return C[int(np.argmin(md))]

    trial = 0
    sigmas = [0.006, 0.012, 0.025, 0.05, 0.09, 0.16, 0.28]
    while time.time() < SEARCH_END and time.time() < DEADLINE - 10:
        trial += 1
        frac_g = (time.time() - t0) / max(1e-9, (SEARCH_END - t0))
        Temp = 0.0018 * (1 - frac_g) + 0.00012
        base = Xbest if rng.random() < 0.70 else Xcur
        r = rng.random()
        Y = None
        try:
            if r < 0.34:
                sig = sigmas[int(rng.integers(0, len(sigmas)))]
                Y = base + sig * rng.normal(size=base.shape)
                if rng.random() < 0.18:
                    vi = int(rng.integers(0, n))
                    Y[vi] = base[vi] + (0.4 + 0.4 * rng.random()) * rng.normal(size=(d,))
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            elif r < 0.62:
                # kill worst k and reinsert into holes
                k = 1
                u = rng.random()
                if u < 0.15: k = 3
                elif u < 0.45: k = 2
                rm = _rowwise_max(base)
                order = np.argsort(-rm)
                victims = order[:k].copy()
                if rng.random() < 0.25:
                    victims[0] = int(rng.integers(0, n))
                Y = base.copy()
                keep = np.ones(n, dtype=bool); keep[victims] = False
                cur_keep = base[keep].copy()
                for idx in victims:
                    h = deep_hole(cur_keep, n_cand=12000)
                    Y[idx] = h
                    cur_keep = np.vstack([cur_keep, h[None, :]])
                Y += 0.006 * rng.normal(size=Y.shape)
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            elif r < 0.78:
                # rattler / loose point big move
                Y = base.copy()
                rm = _rowwise_max(base)
                # loosest points
                order_loose = np.argsort(rm)
                vi = int(order_loose[int(rng.integers(0, min(n, 8)))]) if rng.random() < 0.7 else int(rng.integers(0, n))
                mask = np.ones(n, dtype=bool); mask[vi] = False
                h = deep_hole(base[mask], n_cand=15000)
                # mix hole with random for diversity
                if rng.random() < 0.3:
                    rr = rng.normal(size=(d,)); rr /= np.linalg.norm(rr)
                    h = h + 0.3 * rr; h /= np.linalg.norm(h)
                Y[vi] = h
                Y += 0.008 * rng.normal(size=Y.shape)
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            else:
                # coherent cap shake
                Y = base.copy()
                uvec = rng.normal(size=(d,)); uvec /= np.linalg.norm(uvec)
                proj = base @ uvec
                thr = 0.35 + 0.35 * rng.random()
                if rng.random() < 0.5:
                    thr = -thr  # other side
                sel = np.where(proj > thr)[0]
                if len(sel) < 2 or len(sel) > 25:
                    sel = np.argsort(-proj)[:max(2, int(rng.integers(3, 9)))]
                sig = 0.03 + 0.08 * rng.random()
                Y[sel] += sig * rng.normal(size=(len(sel), d))
                # small global jitter
                Y += 0.004 * rng.normal(size=Y.shape)
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        except Exception:
            continue
        # choose anneal
        s1c = 280.0 + 220.0 * rng.random()
        s0c = 35.0 + 25.0 * rng.random()
        steps = 550 if r < 0.5 else 700
        if time.time() > SEARCH_END - 6:
            steps = 350
        lr0c = 0.006; lr1c = 0.0004
        if rng.random() < 0.3:
            lr0c = 0.010; lr1c = 0.0006
        try:
            Xo = refine(Y, steps=steps, s0=s0c, s1=s1c, lr0=lr0c, lr1=lr1c)
            v = _max_cos(Xo)
        except Exception:
            continue
        if v < best - 1e-12:
            Xbest = Xo.copy(); best = float(v)
            Xcur = Xo.copy(); cur = float(v)
            if time.time() < SEARCH_END - 10 and time.time() < DEADLINE - 20:
                try:
                    Xp = _lp_polish(Xbest, seconds=1.2, t_end=SEARCH_END)
                    vp = _max_cos(Xp)
                    if vp < best:
                        Xbest = Xp; best = float(vp)
                        Xcur = Xp.copy(); cur = float(vp)
                except Exception:
                    pass
        elif v < cur or rng.random() < np.exp(-(v - cur) / max(1e-9, Temp)):
            Xcur = Xo.copy(); cur = float(v)
        if best < 0.19:
            break
    # sharp final anneal
    try:
        if time.time() < DEADLINE - 8:
            Xs = refine(Xbest, steps=900, s0=60.0, s1=700.0, lr0=0.004, lr1=0.00015)
            vs = _max_cos(Xs)
            if vs < best:
                Xbest = Xs; best = float(vs)
    except Exception:
        pass
    remain = DEADLINE - time.time() - 1.5
    if remain > 3.0:
        try:
            Xbest = _lp_polish(Xbest, seconds=min(remain, 28.0), t_end=DEADLINE - 0.8)
        except Exception:
            pass
    Xbest /= np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
