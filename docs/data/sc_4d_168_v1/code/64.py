import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _mc(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _nm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _lp(x, deadline):
    x = _nm(x.copy())
    n, d = x.shape
    best = _mc(x)
    rad = 2e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and rad > 1e-14:
        c = (x @ x.T)[iu]
        m = float(np.max(c))
        sel = c > m - max(20 * rad, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        Aub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        Aeq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            r = linprog(obj, A_ub=Aub, b_ub=-c[sel], A_eq=Aeq, b_eq=np.zeros(n),
                        bounds=[(-rad, rad)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if r.status != 0:
            rad *= 0.5
            continue
        y = _nm(x + r.x[:-1].reshape(n, d))
        v = _mc(y)
        if v < best - 1e-15:
            x, best, rad = y, v, min(rad * 1.6, 0.05)
        else:
            rad *= 0.5
    return x

def _equi_adam(X, steps, b0=25.0, b1=600.0, lr0=0.02, lr1=0.0015, deadline=None):
    X = _nm(X.copy())
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1a, b2a, eps = 0.9, 0.999, 1e-8
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(1, steps + 1):
        if deadline is not None and (t % 50 == 0) and time.time() > deadline:
            break
        f = t / steps
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        rowmax = np.max(G, axis=1, keepdims=True)
        E = np.exp(np.clip(beta * (G - rowmax), -25.0, 0.0))
        np.fill_diagonal(E, 0.0)
        rs = np.sum(E, axis=1, keepdims=True) + 1e-12
        grad = (E @ X) / rs
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        m = b1a * m + (1 - b1a) * gt
        v = b2a * v + (1 - b2a) * (gt * gt)
        mh = m / (1 - b1a ** t)
        vh = v / (1 - b2a ** t)
        X = _nm(X - lr * mh / (np.sqrt(vh) + eps))
    return X

def _single_opt(s, Xref, steps=100, b0=50.0, b1=300.0, lr0=0.08, lr1=0.005):
    s = s / (np.linalg.norm(s) + 1e-30)
    lb0, lb1 = np.log(b0), np.log(b1)
    for it in range(steps):
        f = it / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = lr0 * (1.0 - f) + lr1 * f
        dots = Xref @ s
        cm = float(np.max(dots))
        w = np.exp(np.clip(beta * (dots - cm), -25.0, 0.0))
        w = w / (np.sum(w) + 1e-30)
        g = w @ Xref
        gt = g - float(np.dot(g, s)) * s
        gn = float(np.linalg.norm(gt)) + 1e-12
        gt = gt / gn
        s = s - lr * gt
        ns = float(np.linalg.norm(s))
        if ns < 1e-12 or not np.isfinite(ns):
            s = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            s = s / ns
    return s

def _void_fill(X_keep, k, rng, nsamp=15000):
    tmp = X_keep.copy()
    outs = []
    for i in range(k):
        C = rng.normal(size=(nsamp, 4))
        C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-30)
        D = C @ tmp.T
        mx = D.max(axis=1)
        bi = int(np.argmin(mx))
        s = C[bi]
        s = _single_opt(s, tmp, steps=90)
        outs.append(s)
        tmp = np.vstack([tmp, s.reshape(1, 4)])
    return np.stack(outs, axis=0)

def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    SEARCH = t0 + 146.0
    rng = np.random.default_rng(20261031)
    cands = []
    for f in ("best.npy", "record.npy"):
        try:
            c = np.load(f).astype(float)
            if c.shape == (168, 4):
                c = _nm(c)
                if np.all(np.isfinite(c)):
                    cands.append(c)
        except Exception:
            pass
    if not cands:
        xb = _nm(rng.normal(size=(168, 4)))
        bv = _mc(xb)
    else:
        vs = [_mc(c) for c in cands]
        xb = cands[int(np.argmin(vs))].copy()
        bv = float(min(vs))
    x_best = xb.copy()
    best_val = bv
    x_cur = xb.copy()
    cur_val = bv
    trial = 0
    # initial equitable polish of incumbent to get equitable baseline
    try:
        if time.time() < SEARCH - 5:
            Xo = _equi_adam(xb, steps=500, b0=25.0, b1=500.0, lr0=0.015, lr1=0.0015, deadline=SEARCH - 2)
            vo = _mc(Xo)
            if vo < best_val:
                best_val = vo
                x_best = Xo.copy()
                x_cur = Xo.copy()
                cur_val = vo
    except Exception:
        pass
    sigmas = [0.08, 0.15, 0.25, 0.35]
    while time.time() < SEARCH - 1.0:
        trial += 1
        if time.time() > SEARCH - 1.0:
            break
        try:
            # choose base: mostly best, sometimes walker
            if trial % 4 == 0 and cur_val < best_val + 0.005:
                base = x_cur
            else:
                base = x_best
            r = float(rng.random())
            if r < 0.55:
                # overload-cluster eject
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                mc = float(np.max(G))
                degs = np.sum(G > mc - 0.004, axis=1)
                order = np.argsort(-degs)
                k = int(rng.integers(8, 15))
                if float(rng.random()) < 0.7:
                    seed_idx = int(order[0])
                else:
                    seed_idx = int(rng.choice(order[:20]))
                dots = base @ base[seed_idx]
                nn = np.argsort(-dots)[:k]
                keep_mask = np.ones(168, dtype=bool)
                keep_mask[nn] = False
                X_keep = base[keep_mask]
                new_pts = _void_fill(X_keep, k, rng, nsamp=14000)
                X0 = np.zeros((168, 4))
                X0[keep_mask] = X_keep
                X0[~keep_mask] = new_pts
                X0 = _nm(X0 + 0.003 * rng.normal(size=(168, 4)))
                steps = 900
            elif r < 0.85:
                sigma = sigmas[int(rng.integers(0, len(sigmas)))]
                X0 = _nm(base + sigma * rng.normal(size=(168, 4)))
                steps = 850
            else:
                sigma = 0.03 + 0.03 * float(rng.random())
                # move worst points preferentially
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                mc = float(np.max(G))
                degs = np.sum(G > mc - 0.003, axis=1)
                order = np.argsort(-degs)
                X0 = base.copy()
                nmove = int(rng.integers(3, 8))
                idxs = order[:nmove]
                X0[idxs] = _nm(X0[idxs] + 0.35 * rng.normal(size=(nmove, 4)))
                X0 = _nm(X0 + sigma * 0.3 * rng.normal(size=(168, 4)))
                steps = 800
            Xo = _equi_adam(X0, steps=steps, b0=25.0, b1=650.0, lr0=0.02, lr1=0.0012, deadline=SEARCH - 0.5)
            v = _mc(Xo)
            # walker update
            if v < cur_val or (v < cur_val + 0.002 and float(rng.random()) < 0.25):
                x_cur = Xo.copy()
                cur_val = v
            if v < best_val - 1e-12:
                best_val = v
                x_best = Xo.copy()
                x_cur = Xo.copy()
                cur_val = v
                # quick LP squeeze on improvement
                if time.time() < SEARCH:
                    try:
                        Xp = _lp(Xo, min(time.time() + 1.2, HARD - 4.0))
                        vp = _mc(Xp)
                        if vp < best_val:
                            best_val = vp
                            x_best = Xp.copy()
                            x_cur = Xp.copy()
                            cur_val = vp
                    except Exception:
                        pass
            elif v < best_val + 0.0012 and (trial % 3 == 0) and time.time() < SEARCH - 2:
                try:
                    Xp = _lp(Xo, min(time.time() + 0.6, HARD - 4.0, SEARCH))
                    vp = _mc(Xp)
                    if vp < best_val:
                        best_val = vp
                        x_best = Xp.copy()
                        x_cur = Xp.copy()
                        cur_val = vp
                except Exception:
                    pass
        except Exception:
            continue
    # final squeeze
    try:
        if time.time() < HARD - 1.0:
            x_best = _lp(x_best, HARD - 0.7)
    except Exception:
        pass
    return _nm(x_best)
