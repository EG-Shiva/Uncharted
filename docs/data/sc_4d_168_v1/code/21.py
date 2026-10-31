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
    rad = 1e-3
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
        obj = np.zeros(nv); obj[-1] = 1.0
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
        if v < best:
            x, best, rad = y, v, min(rad * 1.5, 0.05)
        else:
            rad *= 0.5
    return x

def _adam(X, steps, b0, b1, lr0, lr1):
    n, d = X.shape
    m = np.zeros_like(X); v = np.zeros_like(X)
    b1a, b2a, eps = 0.9, 0.999, 1e-8
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(1, steps + 1):
        f = t / steps
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gm = float(np.max(G))
        W = np.exp(np.clip(beta * (G - gm), -20, 0))
        np.fill_diagonal(W, 0.0)
        Gr = (W @ X) * beta
        Gt = Gr - np.sum(Gr * X, axis=1, keepdims=True) * X
        m = b1a * m + (1 - b1a) * Gt
        v = b2a * v + (1 - b2a) * (Gt * Gt)
        mh = m / (1 - b1a ** t)
        vh = v / (1 - b2a ** t)
        X = _nm(X - lr * mh / (np.sqrt(vh) + eps))
    return X

def _ref1(x0, Xr, steps=120, beta=55.0, lr=0.08):
    x = x0 / np.linalg.norm(x0)
    for _ in range(steps):
        d = Xr @ x
        dm = float(np.max(d))
        w = np.exp(np.clip(beta * (d - dm), -20, 0))
        g = w @ Xr
        g = g - np.dot(g, x) * x
        gn = np.linalg.norm(g)
        if gn > 0:
            g = g / gn
        x = x - lr * g
        x = x / np.linalg.norm(x)
        lr *= 0.995
        beta = min(beta * 1.006, 220.0)
    return x / np.linalg.norm(x)

def _hole(Xk, rng, ns=9000):
    C = rng.normal(size=(ns, 4)); C = _nm(C)
    D = C @ Xk.T
    mc = D.max(axis=1)
    bi = int(np.argmin(mc))
    return _ref1(C[bi], Xk)

def _voids(X, rng, nv=12, ns=15000):
    S = _nm(rng.normal(size=(ns, 4)))
    mx = (S @ X.T).max(axis=1)
    idx = np.argsort(mx)[:nv]
    return S[idx]

def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    SEARCH = t0 + 148.0
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
        xb = _nm(rng.normal(size=(168, 4))); bv = _mc(xb)
    else:
        vs = [_mc(c) for c in cands]
        xb = cands[int(np.argmin(vs))].copy(); bv = float(min(vs))
    n, d = 168, 4
    trial = 0
    while time.time() < SEARCH:
        trial += 1
        if time.time() > SEARCH - 1.0:
            break
        X = xb.copy()
        typ = trial % 10
        try:
            G = X @ X.T; np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            degN = np.sum(G > cmax - 0.004, axis=1)
            degT = np.sum(G > cmax - 1e-9, axis=1)
            order_hi = np.argsort(-degN, kind="stable")
            order_lo = np.argsort(degN, kind="stable")
        except Exception:
            order_hi = np.arange(n); order_lo = np.arange(n)
        if typ == 0:
            X = _nm(X + 0.015 * rng.normal(size=(n, d)))
            st, kw = 600, dict(b0=40., b1=380., lr0=0.02, lr1=0.002)
        elif typ == 1:
            X = _nm(X + 0.050 * rng.normal(size=(n, d)))
            st, kw = 850, dict(b0=30., b1=400., lr0=0.02, lr1=0.002)
        elif typ == 2:
            X = _nm(X + 0.12 * rng.normal(size=(n, d)))
            k = 2 + (trial % 3)
            vict = order_hi[:k].copy()
            V = _voids(X, rng, nv=12, ns=15000)
            for a, vi in enumerate(vict):
                vv = V[a % len(V)] + 0.02 * rng.normal(size=(d,))
                X[vi] = vv / np.linalg.norm(vv)
            X = _nm(X + 0.008 * rng.normal(size=(n, d)))
            st, kw = 1000, dict(b0=25., b1=420., lr0=0.02, lr1=0.0018)
        elif typ == 3:
            k = 4 + (trial % 4)
            vict = order_hi[:k].copy()
            keep = np.ones(n, dtype=bool); keep[vict] = False
            Xk = X[keep]
            newp = []
            for s in range(k):
                Xr = Xk if s == 0 else np.vstack([Xk, np.array(newp)])
                newp.append(_hole(Xr, rng, ns=8000))
            X[vict] = np.array(newp)
            X = _nm(X + 0.012 * rng.normal(size=(n, d)))
            st, kw = 1100, dict(b0=28., b1=450., lr0=0.018, lr1=0.0018)
        elif typ == 4:
            # rattler / low-contact relocation
            k = 3
            vict = order_lo[:k].copy()
            for j in vict:
                mask = np.ones(n, dtype=bool); mask[j] = False
                X[j] = _hole(X[mask], rng, ns=8000)
            hi = order_hi[0]
            X[hi] = _nm((X[hi] + 0.10 * rng.normal(size=(d,))).reshape(1, -1))[0]
            st, kw = 1000, dict(b0=32., b1=400., lr0=0.018, lr1=0.002)
        elif typ == 5:
            # break worst edge directly
            ij = np.unravel_index(int(np.argmax(G)), G.shape)
            i, j = int(ij[0]), int(ij[1])
            # push i away from j along tangent
            dij = X[i] - G[i, j] * X[j]
            nn = np.linalg.norm(dij)
            if nn > 1e-12:
                dij /= nn
            X[i] = _nm((X[i] + 0.08 * dij + 0.03 * rng.normal(size=(d,))).reshape(1, -1))[0]
            X[j] = _nm((X[j] - 0.03 * dij + 0.03 * rng.normal(size=(d,))).reshape(1, -1))[0]
            nbr = np.argsort(-G[i])[:6]
            X[nbr] = _nm(X[nbr] + 0.05 * rng.normal(size=(len(nbr), d)))
            X = _nm(X + 0.01 * rng.normal(size=(n, d)))
            st, kw = 900, dict(b0=35., b1=380., lr0=0.018, lr1=0.002)
        elif typ == 6:
            # cluster kick around random center
            c0 = int(rng.integers(0, n))
            nbr = np.argsort(-G[c0])[:8]
            X[nbr] = _nm(X[nbr] + 0.28 * rng.normal(size=(len(nbr), d)))
            X[c0] = _nm((X[c0] + 0.28 * rng.normal(size=(d,))).reshape(1, -1))[0]
            st, kw = 1100, dict(b0=22., b1=430., lr0=0.022, lr1=0.0018)
        elif typ == 7:
            k = 2
            vict = rng.choice(n, size=k, replace=False)
            for j in vict:
                mask = np.ones(n, dtype=bool); mask[j] = False
                X[j] = _hole(X[mask], rng, ns=9000)
            X = _nm(X + 0.025 * rng.normal(size=(n, d)))
            st, kw = 950, dict(b0=30., b1=400., lr0=0.02, lr1=0.002)
        elif typ == 8:
            X = _nm(rng.normal(size=(n, d)))
            st, kw = 1700, dict(b0=18., b1=480., lr0=0.025, lr1=0.002)
        else:
            X = _nm(X + 0.085 * rng.normal(size=(n, d)))
            st, kw = 1000, dict(b0=24., b1=400., lr0=0.02, lr1=0.002)
            # random orthogonal mix on subset to change basin
            if trial % 3 == 0:
                a = rng.normal(size=(4, 4)); Q, _ = np.linalg.qr(a)
                if np.linalg.det(Q) < 0:
                    Q[:, 0] = -Q[:, 0]
                sel = rng.choice(n, size=20, replace=False)
                X[sel] = _nm(X[sel] @ Q.T + 0.02 * rng.normal(size=(20, d)))
        try:
            Xo = _adam(X, st, **kw)
        except Exception:
            continue
        v = _mc(Xo)
        imp = False
        if v < bv:
            bv = v; xb = Xo.copy(); imp = True
        do = False; bud = 0.0
        if imp:
            do = True; bud = 2.0
        elif v < bv + 0.002 and (trial % 4 == 0):
            do = True; bud = 0.9
        elif v < bv + 0.006 and (trial % 14 == 0):
            do = True; bud = 0.7
        if do and time.time() < SEARCH - 0.4:
            try:
                Xp = _lp(Xo, min(time.time() + bud, SEARCH + 3.0, HARD - 7.0))
                vp = _mc(Xp)
                if vp < bv:
                    bv = vp; xb = Xp.copy()
            except Exception:
                pass
    if time.time() < HARD - 2.0:
        try:
            rem = min(HARD - 1.0, time.time() + max(6.0, HARD - time.time() - 1.0))
            xb = _lp(xb, rem)
        except Exception:
            pass
    return _nm(xb)
