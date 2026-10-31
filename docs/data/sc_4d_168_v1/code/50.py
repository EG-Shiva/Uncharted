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
    rad = 1.5e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and rad > 1e-14:
        C = (x @ x.T)[iu]
        m = float(np.max(C))
        sel = C > m - max(22.0 * rad, 1e-9)
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
            rad = min(rad * 1.6, 0.05)
        else:
            rad *= 0.5
    return x

def _adam(X, steps, beta0, beta1, lr0, lr1, mu=0.88, deadline=None):
    n, d = X.shape
    vel = np.zeros_like(X)
    lb0, lb1 = np.log(beta0), np.log(beta1)
    for s in range(steps):
        if deadline is not None and (s % 200 == 0) and time.time() > deadline:
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

def _rand_Q(rng):
    A = rng.normal(size=(4, 4))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

def _rand_Q3(rng):
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

def _fib_sphere(m, rng):
    if m <= 0:
        return np.zeros((0, 3))
    if m == 1:
        v = rng.normal(size=(3,))
        return (v / (np.linalg.norm(v) + 1e-30))[None, :]
    g = np.pi * (3.0 - np.sqrt(5.0))
    idx = np.arange(m, dtype=float)
    z = 1.0 - 2.0 * (idx + 0.5) / m
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    th = g * idx + rng.random() * 2 * np.pi
    y = np.stack([r * np.cos(th), r * np.sin(th), z], axis=1)
    Q = _rand_Q3(rng)
    return y @ Q.T

def _make_layered(rng):
    n = 168
    L = int(rng.integers(7, 12))
    h = rng.uniform(-0.96, 0.96, size=L)
    h.sort()
    # enforce min separation a bit
    for _ in range(5):
        dh = np.diff(h)
        j = np.argmin(dh) if len(dh) > 0 else -1
        if j >= 0 and dh[j] < 0.07:
            h[j] -= 0.02; h[j+1] += 0.02
            h = np.clip(h, -0.97, 0.97); h.sort()
        else:
            break
    s = np.sqrt(np.maximum(1e-6, 1 - h * h))
    w = s * s + 0.06
    m = np.floor(168 * w / w.sum()).astype(int)
    m = np.maximum(m, 1)
    while m.sum() > 168:
        m[int(np.argmax(m))] -= 1
    while m.sum() < 168:
        # add to layer with largest area per point
        score = w / np.maximum(1, m)
        m[int(np.argmax(score))] += 1
    pts = []
    for k in range(L):
        y = _fib_sphere(int(m[k]), rng)
        sk, hk = s[k], h[k]
        P = np.concatenate([sk * y, np.full((len(y), 1), hk)], axis=1)
        pts.append(P)
    X = np.vstack(pts)
    # fix count mismatch
    if len(X) > 168:
        X = X[:168]
    while len(X) < 168:
        v = rng.normal(size=(1, 4)); v = _nm(v)
        X = np.vstack([X, v])
    Q = _rand_Q(rng)
    X = X @ Q.T
    X = _nm(X + 0.012 * rng.normal(size=X.shape))
    return X

def _make_hopf(rng):
    n = 168
    T = int(rng.integers(4, 8))
    a = rng.uniform(0.18, 1.39, size=T)
    a.sort()
    w = np.sin(2 * a) + 0.35
    m = np.floor(168 * w / w.sum()).astype(int)
    m = np.maximum(m, 4)
    while m.sum() > 168:
        m[int(np.argmax(m))] -= 1
    while m.sum() < 168:
        m[int(np.argmax(w / np.maximum(1, m)))] += 1
    pts = []
    for t in range(T):
        mt = int(m[t])
        at = float(a[t])
        ca, sa = np.cos(at), np.sin(at)
        cot = ca / max(1e-3, sa)
        n1 = max(2, int(round(np.sqrt(mt * cot))))
        n2 = max(2, int(round(mt / max(1, n1))))
        # adjust to approx mt
        while n1 * n2 < mt:
            if n1 * sa < n2 * ca:
                n1 += 1
            else:
                n2 += 1
        o1 = rng.random() * 2 * np.pi
        o2 = rng.random() * 2 * np.pi
        cnt = 0
        for i in range(n1):
            for j in range(n2):
                if cnt >= mt and len(pts) + cnt >= 0:
                    break
                th = o1 + 2 * np.pi * i / n1
                ph = o2 + 2 * np.pi * j / n2
                pts.append([ca * np.cos(th), ca * np.sin(th), sa * np.cos(ph), sa * np.sin(ph)])
                cnt += 1
                if cnt >= mt:
                    break
            if cnt >= mt:
                break
        # if overshoot due to loops, trim later
    X = np.array(pts, dtype=float)
    if len(X) > 168:
        X = X[:168]
    while len(X) < 168:
        v = rng.normal(size=(1, 4)); v = _nm(v)
        X = np.vstack([X, v])
    Q = _rand_Q(rng)
    X = X @ Q.T
    X = _nm(X + 0.012 * rng.normal(size=X.shape))
    return X

def _best_void(Xref, rng, ns=8000):
    C = _nm(rng.normal(size=(ns, 4)))
    D = C @ Xref.T
    m = D.max(axis=1)
    bi = int(np.argmin(m))
    s = C[bi]
    # few tangent steps to deepen hole
    for it in range(40):
        dots = Xref @ s
        cm = float(np.max(dots))
        beta = 60.0
        w = np.exp(np.clip(beta * (dots - cm), -20.0, 0.0))
        w = w / (np.sum(w) + 1e-30)
        g = w @ Xref
        gt = g - np.dot(g, s) * s
        gn = np.linalg.norm(gt) + 1e-12
        s = s - 0.06 * gt / gn
        s = s / (np.linalg.norm(s) + 1e-30)
    return s

def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    SEARCH = t0 + 143.0
    rng = np.random.default_rng(987654321)
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
    # Stage A: regular pool - quick screen
    pool_best = xb.copy(); pool_val = bv
    npool = 26
    for pi in range(npool):
        if time.time() > SEARCH - 20.0:
            break
        try:
            if pi % 2 == 0:
                X0 = _make_layered(rng)
            else:
                X0 = _make_hopf(rng)
            Xp = _adam(X0, 450, beta0=18.0, beta1=320.0, lr0=0.018, lr1=0.0015, mu=0.88, deadline=SEARCH)
            v = _mc(Xp)
            if v < pool_val:
                pool_val = float(v); pool_best = Xp.copy()
            if v < bv:
                bv = float(v); xb = Xp.copy()
            # keep walk state if promising
            if v < cv + 0.003 and rng.random() < 0.5:
                xcur, cv = Xp.copy(), float(v)
        except Exception:
            continue
    # longer polish of pool best if it is competitive
    if pool_val < bv + 0.01 and time.time() < SEARCH - 10.0:
        try:
            Xp = _adam(pool_best, 900, beta0=22.0, beta1=480.0, lr0=0.016, lr1=0.0010, mu=0.90, deadline=SEARCH)
            v = _mc(Xp)
            if v < cv:
                xcur, cv = Xp.copy(), float(v)
            if v < bv:
                bv, xb = float(v), Xp.copy()
            if time.time() < SEARCH - 5.0:
                Xl = _lp(Xp, min(time.time() + 2.0, SEARCH, HARD - 7.0))
                vl = _mc(Xl)
                if vl < bv:
                    bv, xb = float(vl), Xl.copy()
                if vl < cv:
                    xcur, cv = Xl.copy(), float(vl)
        except Exception:
            pass
    Temp = 0.0022
    trial = 0
    while time.time() < SEARCH:
        trial += 1
        if time.time() > SEARCH - 1.0:
            break
        # base selection
        r = rng.random()
        if r < 0.35:
            base = xb.copy()
        elif r < 0.6:
            base = pool_best.copy()
        else:
            if cv < bv + 0.004:
                base = xcur.copy()
            else:
                base = xb.copy()
        X = base.copy()
        typ = trial % 12
        try:
            G = X @ X.T; np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            deg = np.sum(G > cmax - 0.004, axis=1)
            order_hi = np.argsort(-deg, kind="stable")
        except Exception:
            order_hi = np.arange(n)
        try:
            if typ == 0 or typ == 1:
                # fresh regular basin
                X = _make_layered(rng) if (trial % 2 == 0) else _make_hopf(rng)
                st, kw = 800, dict(beta0=20., beta1=420., lr0=0.017, lr1=0.0012)
            elif typ == 2:
                X = _nm(X + 0.015 * rng.normal(size=(n, d)))
                st, kw = 700, dict(beta0=38., beta1=380., lr0=0.015, lr1=0.0015)
            elif typ == 3:
                # geographic cap delete + sequential void reinsert
                u = _nm(rng.normal(size=(1, 4)))[0]
                dots = X @ u
                k = 9 + int(rng.integers(0, 6))
                vict = np.argsort(-dots, kind="stable")[:k]
                keep = np.ones(n, dtype=bool); keep[vict] = False
                cur_set = X[keep]
                newp = []
                for s in range(k):
                    Xr = cur_set if s == 0 else np.vstack([cur_set, np.array(newp)])
                    newp.append(_best_void(Xr, rng, ns=7000))
                X[vict] = np.array(newp)
                X = _nm(X + 0.008 * rng.normal(size=(n, d)))
                st, kw = 950, dict(beta0=24., beta1=450., lr0=0.017, lr1=0.0011)
            elif typ == 4:
                # hub ejection into voids
                k = 5 + int(rng.integers(0, 4))
                vict = order_hi[:k].copy()
                keep = np.ones(n, dtype=bool); keep[vict] = False
                cur_set = X[keep]
                newp = []
                for s in range(k):
                    Xr = cur_set if s == 0 else np.vstack([cur_set, np.array(newp)])
                    newp.append(_best_void(Xr, rng, ns=7000))
                X[vict] = np.array(newp)
                X = _nm(X + 0.009 * rng.normal(size=(n, d)))
                st, kw = 950, dict(beta0=26., beta1=450., lr0=0.016, lr1=0.0011)
            elif typ == 5:
                # collective cap rotation to break overloaded hubs
                u = _nm(rng.normal(size=(1, 4)))[0]
                dots = X @ u
                k = 20 + int(rng.integers(0, 10))
                sel = np.argsort(-dots, kind="stable")[:k]
                Q = _rand_Q(rng)
                X[sel] = _nm(X[sel] @ Q.T + 0.05 * rng.normal(size=(k, d)))
                X = _nm(X + 0.010 * rng.normal(size=(n, d)))
                st, kw = 900, dict(beta0=22., beta1=430., lr0=0.018, lr1=0.0012)
            elif typ == 6:
                # worst edge surgery
                ij = int(np.argmax(G)); i, j = ij // n, ij % n
                dij = X[i] - G[i, j] * X[j]
                nn = np.linalg.norm(dij)
                if nn > 1e-12:
                    dij /= nn
                X[i] = _nm((X[i] + 0.10 * dij + 0.03 * rng.normal(size=(d,))).reshape(1, -1))[0]
                X[j] = _nm((X[j] - 0.05 * dij + 0.03 * rng.normal(size=(d,))).reshape(1, -1))[0]
                nbr = np.argsort(-G[i], kind="stable")[:6]
                X[nbr] = _nm(X[nbr] + 0.07 * rng.normal(size=(len(nbr), d)))
                X = _nm(X + 0.007 * rng.normal(size=(n, d)))
                st, kw = 800, dict(beta0=32., beta1=400., lr0=0.015, lr1=0.0014)
            elif typ == 7:
                k = 3
                vict = rng.choice(n, size=k, replace=False)
                for jj in vict:
                    mask = np.ones(n, dtype=bool); mask[jj] = False
                    X[jj] = _best_void(X[mask], rng, ns=7000)
                X = _nm(X + 0.014 * rng.normal(size=(n, d)))
                st, kw = 850, dict(beta0=28., beta1=420., lr0=0.016, lr1=0.0014)
            elif typ == 8:
                # strong shake of random cluster
                c0 = int(rng.integers(0, n))
                nbr = np.argsort(-G[c0], kind="stable")[:9]
                X[nbr] = _nm(X[nbr] + 0.28 * rng.normal(size=(len(nbr), d)))
                X = _nm(X + 0.010 * rng.normal(size=(n, d)))
                st, kw = 1000, dict(beta0=20., beta1=450., lr0=0.019, lr1=0.0011)
            elif typ == 9:
                X = _nm(X + 0.05 * rng.normal(size=(n, d)))
                st, kw = 850, dict(beta0=28., beta1=420., lr0=0.016, lr1=0.0012)
            elif typ == 10:
                # antipodal-ish shuffle: rotate random subset
                Q = _rand_Q(rng)
                sel = rng.choice(n, size=24, replace=False)
                X[sel] = _nm(X[sel] @ Q.T + 0.02 * rng.normal(size=(24, d)))
                X = _nm(X + 0.010 * rng.normal(size=(n, d)))
                st, kw = 900, dict(beta0=24., beta1=430., lr0=0.017, lr1=0.0012)
            else:
                X = _nm(rng.normal(size=(n, d)))
                st, kw = 1400, dict(beta0=16., beta1=480., lr0=0.020, lr1=0.0014)
            Xo = _adam(X, st, deadline=SEARCH, **kw)
        except Exception:
            continue
        v = _mc(Xo)
        if v < cv or rng.random() < np.exp(-(v - cv) / Temp):
            xcur, cv = Xo.copy(), float(v)
        imp = False
        if v < bv:
            bv = float(v); xb = Xo.copy(); imp = True
            # update pool best too if regular-like (balanced)
            try:
                G2 = Xo @ Xo.T; np.fill_diagonal(G2, -2.0)
                m2 = float(np.max(G2))
                dg2 = np.sum(G2 > m2 - 1e-9, axis=1)
                if float(np.std(dg2)) < 3.0:
                    pool_best, pool_val = Xo.copy(), float(v)
            except Exception:
                pass
        do = False; bud = 0.0
        if imp:
            do = True; bud = 2.2
        elif v < bv + 0.002 and (trial % 4 == 0):
            do = True; bud = 0.9
        elif v < bv + 0.006 and (trial % 13 == 0):
            do = True; bud = 0.7
        if do and time.time() < SEARCH - 0.4:
            try:
                Xp = _lp(Xo, min(time.time() + bud, SEARCH + 2.0, HARD - 7.0))
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
