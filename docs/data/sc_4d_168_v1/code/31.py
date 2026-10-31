import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _mc(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

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
        C = (x @ x.T)[iu]
        m = float(np.max(C))
        sel = C > m - max(20.0 * rad, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        cs = C[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        Aub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        Aeq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            r = linprog(obj, A_ub=Aub, b_ub=-cs, A_eq=Aeq, b_eq=np.zeros(n),
                        bounds=[(-rad, rad)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if r.status != 0:
            rad *= 0.5
            continue
        y = _nm(x + r.x[:-1].reshape(n, d))
        v = _mc(y)
        if v < best:
            x, best = y, v
            rad = min(rad * 1.5, 0.05)
        else:
            rad *= 0.5
    return x

def _adam(X, steps, beta0, beta1, lr0, lr1, mu=0.88, deadline=None):
    n, d = X.shape
    vel = np.zeros_like(X)
    lb0, lb1 = np.log(beta0), np.log(beta1)
    for s in range(steps):
        if deadline is not None and s % 200 == 0 and time.time() > deadline:
            break
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = lr0 * (1.0 - f) + lr1 * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cmax = float(np.max(G))
        E = np.exp(np.clip(beta * (G - cmax), -25.0, 0.0))
        np.fill_diagonal(E, 0.0)
        rs = np.sum(E, axis=1, keepdims=True) + 1e-12
        grad = (E @ X) / rs
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        sc = float(np.mean(np.linalg.norm(gt, axis=1))) + 1e-12
        gt = gt / sc
        vel = mu * vel + (1.0 - mu) * gt
        X = _nm(X - lr * vel)
    return X

def _ref_single(x0, Xr, steps=90, b0=45.0, b1=260.0, lr0=0.09, lr1=0.006, rng=None):
    s = x0 / (np.linalg.norm(x0) + 1e-30)
    lb0, lb1 = np.log(b0), np.log(b1)
    for it in range(steps):
        f = it / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = lr0 * (1.0 - f) + lr1 * f
        dots = Xr @ s
        cm = float(np.max(dots))
        w = np.exp(np.clip(beta * (dots - cm), -25.0, 0.0))
        w = w / (np.sum(w) + 1e-30)
        g = w @ Xr
        gt = g - np.dot(g, s) * s
        gn = np.linalg.norm(gt) + 1e-12
        gt = gt / gn
        s = s - lr * gt
        ns = np.linalg.norm(s)
        if ns < 1e-12:
            s = rng.normal(size=(4,)) if rng is not None else np.random.randn(4)
            ns = np.linalg.norm(s)
        s = s / ns
    return s

def _best_void(Xref, rng, ns=12000, refine=True):
    C = rng.normal(size=(ns, 4)); C = _nm(C)
    D = C @ Xref.T
    m = D.max(axis=1)
    bi = int(np.argmin(m))
    c0 = C[bi]
    if refine:
        c0 = _ref_single(c0, Xref, rng=rng)
    return c0

def _rand_Q(rng):
    A = rng.normal(size=(4, 4))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

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
        xb = _nm(rng.normal(size=(168, 4))); bv = _mc(xb)
    else:
        vs = [_mc(c) for c in cands]
        xb = cands[int(np.argmin(vs))].copy(); bv = float(min(vs))
    n, d = 168, 4
    xcur = xb.copy(); cv = bv
    Temp = 0.0025
    trial = 0
    while time.time() < SEARCH:
        trial += 1
        if time.time() > SEARCH - 1.0:
            break
        # choose base: mostly best, sometimes walk from cur
        if trial % 6 == 0 and cv < bv + 0.004:
            base = xcur.copy()
        else:
            base = xb.copy()
        X = base.copy()
        typ = trial % 10
        try:
            G = X @ X.T; np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            deg = np.sum(G > cmax - 0.004, axis=1)
            order_hi = np.argsort(-deg, kind="stable")
        except Exception:
            order_hi = np.arange(n)
        if typ == 0:
            X = _nm(X + 0.012 * rng.normal(size=(n, d)))
            st, kw = 650, dict(beta0=40., beta1=380., lr0=0.016, lr1=0.0015)
        elif typ == 1:
            X = _nm(X + 0.045 * rng.normal(size=(n, d)))
            st, kw = 850, dict(beta0=30., beta1=420., lr0=0.016, lr1=0.0012)
        elif typ == 2:
            # geographic cap repack: delete cap cluster, sequential void reinsert
            u = _nm(rng.normal(size=(1, 4)))[0]
            dots = X @ u
            k = 10 + (trial % 6)  # 10-15 points
            vict = np.argsort(-dots, kind="stable")[:k]
            keep = np.ones(n, dtype=bool); keep[vict] = False
            cur_set = X[keep]
            newp = []
            for s in range(k):
                Xr = cur_set if s == 0 else np.vstack([cur_set, np.array(newp)])
                newp.append(_best_void(Xr, rng, ns=12000, refine=True))
            X[vict] = np.array(newp)
            X = _nm(X + 0.008 * rng.normal(size=(n, d)))
            st, kw = 1050, dict(beta0=25., beta1=450., lr0=0.017, lr1=0.0012)
        elif typ == 3:
            # high-degree scattered ejection into voids
            k = 5 + (trial % 5)
            vict = order_hi[:k].copy()
            keep = np.ones(n, dtype=bool); keep[vict] = False
            cur_set = X[keep]
            newp = []
            for s in range(k):
                Xr = cur_set if s == 0 else np.vstack([cur_set, np.array(newp)])
                newp.append(_best_void(Xr, rng, ns=10000, refine=True))
            X[vict] = np.array(newp)
            X = _nm(X + 0.010 * rng.normal(size=(n, d)))
            st, kw = 1050, dict(beta0=28., beta1=450., lr0=0.016, lr1=0.0012)
        elif typ == 4:
            # cap rotation: collective rigid move of cap
            u = _nm(rng.normal(size=(1, 4)))[0]
            dots = X @ u
            k = 18 + (trial % 8)
            sel = np.argsort(-dots, kind="stable")[:k]
            Q = _rand_Q(rng)
            # small rotation: mix Q with identity
            ang = 0.25 + 0.15 * rng.random()
            # interpolate: apply Q partially via tangent
            X[sel] = _nm(X[sel] @ Q.T + 0.03 * rng.normal(size=(k, d)))
            # also jitter neighbours
            X = _nm(X + 0.008 * rng.normal(size=(n, d)))
            # push selected away a bit along u to break contacts
            st, kw = 1000, dict(beta0=24., beta1=430., lr0=0.018, lr1=0.0012)
            _ = ang
        elif typ == 5:
            # worst-edge surgery + neighbourhood kick
            ij = int(np.argmax(G)); i, j = ij // n, ij % n
            dij = X[i] - G[i, j] * X[j]
            nn = np.linalg.norm(dij)
            if nn > 1e-12:
                dij /= nn
            X[i] = _nm((X[i] + 0.09 * dij + 0.03 * rng.normal(size=(d,))).reshape(1, -1))[0]
            X[j] = _nm((X[j] - 0.04 * dij + 0.03 * rng.normal(size=(d,))).reshape(1, -1))[0]
            nbr = np.argsort(-G[i], kind="stable")[:7]
            X[nbr] = _nm(X[nbr] + 0.06 * rng.normal(size=(len(nbr), d)))
            X = _nm(X + 0.008 * rng.normal(size=(n, d)))
            st, kw = 900, dict(beta0=35., beta1=400., lr0=0.015, lr1=0.0015)
        elif typ == 6:
            # large geographic cluster kick
            c0 = int(rng.integers(0, n))
            nbr = np.argsort(-G[c0], kind="stable")[:9]
            X[nbr] = _nm(X[nbr] + 0.30 * rng.normal(size=(len(nbr), d)))
            X[c0] = _nm((X[c0] + 0.30 * rng.normal(size=(d,))).reshape(1, -1))[0]
            st, kw = 1100, dict(beta0=22., beta1=450., lr0=0.019, lr1=0.0012)
        elif typ == 7:
            # random void transplant of few points
            k = 3
            vict = rng.choice(n, size=k, replace=False)
            for jj in vict:
                mask = np.ones(n, dtype=bool); mask[jj] = False
                X[jj] = _best_void(X[mask], rng, ns=10000, refine=True)
            X = _nm(X + 0.015 * rng.normal(size=(n, d)))
            st, kw = 950, dict(beta0=30., beta1=420., lr0=0.016, lr1=0.0015)
        elif typ == 8:
            X = _nm(rng.normal(size=(n, d)))
            st, kw = 1600, dict(beta0=18., beta1=500., lr0=0.020, lr1=0.0015)
        else:
            X = _nm(X + 0.085 * rng.normal(size=(n, d)))
            if trial % 3 == 0:
                Q = _rand_Q(rng)
                sel = rng.choice(n, size=22, replace=False)
                X[sel] = _nm(X[sel] @ Q.T + 0.02 * rng.normal(size=(22, d)))
            st, kw = 1000, dict(beta0=24., beta1=420., lr0=0.017, lr1=0.0012)
        try:
            Xo = _adam(X, st, deadline=SEARCH, **kw)
        except Exception:
            continue
        v = _mc(Xo)
        # Metropolis walk update
        if v < cv or rng.random() < np.exp(-(v - cv) / Temp):
            xcur, cv = Xo.copy(), float(v)
        imp = False
        if v < bv:
            bv = float(v); xb = Xo.copy(); imp = True
        do = False; bud = 0.0
        if imp:
            do = True; bud = 2.0
        elif v < bv + 0.002 and (trial % 4 == 0):
            do = True; bud = 0.9
        elif v < bv + 0.006 and (trial % 13 == 0):
            do = True; bud = 0.7
        if do and time.time() < SEARCH - 0.4:
            try:
                Xp = _lp(Xo, min(time.time() + bud, SEARCH + 3.0, HARD - 7.0))
                vp = _mc(Xp)
                if vp < cv:
                    xcur, cv = Xp.copy(), float(vp)
                if vp < bv:
                    bv = float(vp); xb = Xp.copy()
            except Exception:
                pass
    if time.time() < HARD - 2.0:
        try:
            rem = min(HARD - 1.0, time.time() + max(6.0, HARD - time.time() - 1.0))
            xb = _lp(xb, rem)
        except Exception:
            pass
    return _nm(xb)
