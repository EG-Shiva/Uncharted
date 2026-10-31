import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _mc(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _nm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _rand_Q(rng):
    A = rng.normal(size=(4, 4))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

def _adam_bal(X, steps, beta0, beta1, lr0, lr1, mu=0.88, deadline=None, eq=0.6):
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
        # degree-equalized per-point step: hubs move more
        try:
            deg = np.sum(G > cmax - 0.005, axis=1)
            md = float(np.mean(deg)) + 1e-9
            mult = 1.0 + eq * (deg - md) / (md + 1.0)
            mult = np.clip(mult, 0.5, 2.0).reshape(-1, 1)
        except Exception:
            mult = 1.0
        X = _nm(X - lr * mult * vel)
    return X

def _lp_polish(X, deadline):
    X = _nm(X.copy())
    n, d = X.shape
    best = _mc(X)
    rad = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and rad > 1e-14:
        C = (X @ X.T)[iu]
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
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        Aub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        Aeq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            r = linprog(obj, A_ub=Aub, b_ub=-cs, A_eq=Aeq, b_eq=np.zeros(n),
                        bounds=[(-rad, rad)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if r.status != 0:
            rad *= 0.5
            continue
        Y = _nm(X + r.x[:-1].reshape(n, d))
        v = _mc(Y)
        if v < best:
            X, best = Y, v
            rad = min(rad * 1.5, 0.05)
        else:
            rad *= 0.5
    return X

def _ref_single(s0, Xr, steps=80, b0=45.0, b1=260.0, lr0=0.09, lr1=0.006, rng=None):
    s = s0 / (np.linalg.norm(s0) + 1e-30)
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

def _best_void(Xref, rng, ns=10000, refine=True):
    C = _nm(rng.normal(size=(ns, 4)))
    D = C @ Xref.T
    m = D.max(axis=1)
    c0 = C[int(np.argmin(m))]
    if refine:
        c0 = _ref_single(c0, Xref, rng=rng)
    return c0

def _make_layered(rng):
    n = 168
    L = int(rng.integers(6, 13))
    # equal-area theta layers with jitter
    base = (np.arange(L) + 0.5) / L
    jit = (rng.random(L) - 0.5) * 0.6 / L
    u = np.clip(base + jit, 0.005, 0.995)
    thetas = np.arcsin(np.sqrt(u))
    # points per layer proportional to sin2t (area) ~ equal since equal-area, add random integer noise
    w = np.sin(2.0 * thetas) + 0.05
    w = w / w.sum()
    counts = np.floor(w * n).astype(int)
    while counts.sum() < n:
        k = int(rng.integers(0, L))
        counts[k] += 1
    while counts.sum() > n:
        k = int(np.argmax(counts))
        counts[k] -= 1
    pts = []
    for li in range(L):
        th = thetas[li]
        nl = int(counts[li])
        if nl <= 0:
            continue
        ct, st = np.cos(th), np.sin(th)
        if ct < 1e-6:
            ct = 1e-6
        if st < 1e-6:
            st = 1e-6
        # grid aspect to equalize arc spacing
        ratio = ct / st
        a = int(max(1, round(np.sqrt(nl * ratio))))
        b = int(max(1, round(np.sqrt(nl / max(ratio, 1e-9)))))
        # adjust to cover nl
        while a * b < nl:
            if a * st < b * ct:
                a += 1
            else:
                b += 1
        o1 = rng.random() * 2 * np.pi
        o2 = rng.random() * 2 * np.pi
        # generate a*b grid then subsample/jitter
        grid = []
        for ia in range(a):
            for ib in range(b):
                p1 = o1 + 2 * np.pi * ia / max(1, a) + 0.35 * rng.normal() / max(1, a)
                p2 = o2 + 2 * np.pi * ib / max(1, b) + 0.35 * rng.normal() / max(1, b)
                grid.append((p1, p2))
        grid = np.array(grid)
        # pick nl of them randomly
        if len(grid) > nl:
            sel = rng.choice(len(grid), size=nl, replace=False)
            grid = grid[sel]
        for (p1, p2) in grid:
            pts.append([ct * np.cos(p1), ct * np.sin(p1), st * np.cos(p2), st * np.sin(p2)])
    pts = np.array(pts[:n])
    if len(pts) < n:
        extra = _nm(rng.normal(size=(n - len(pts), 4)))
        pts = np.vstack([pts, extra])
    # random global rotation for diversity
    Q = _rand_Q(rng)
    pts = pts @ Q.T
    return _nm(pts + 0.002 * rng.normal(size=(n, 4)))

def _hub_spread(X, rng, thresh=0.004):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    cmax = float(np.max(G))
    deg = np.sum(G > cmax - thresh, axis=1)
    order = np.argsort(-deg, kind="stable")
    # pick top 1-2 hubs
    nh = 1 + int(rng.integers(0, 2))
    Y = X.copy()
    for h in order[:nh]:
        if deg[h] < 7:
            continue
        nbr = np.argsort(-G[h], kind="stable")[:int(deg[h])]
        # push neighbors away from hub tangentially
        for j in nbr:
            d = Y[j] - np.dot(Y[j], Y[h]) * Y[h]
            nd = np.linalg.norm(d)
            if nd < 1e-12:
                d = rng.normal(size=(4,)); d -= np.dot(d, Y[h]) * Y[h]
                nd = np.linalg.norm(d) + 1e-12
            d = d / nd
            step = 0.05 + 0.05 * rng.random()
            Y[j] = Y[j] + step * d + 0.02 * rng.normal(size=(4,))
        # also kick hub slightly
        Y[h] = Y[h] + 0.03 * rng.normal(size=(4,))
    return _nm(Y)

def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    SEARCH = t0 + 144.0
    rng = np.random.default_rng(777123)
    n, d = 168, 4
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
    if cands:
        vs = [_mc(c) for c in cands]
        xb = cands[int(np.argmin(vs))].copy()
        bv = float(min(vs))
    else:
        xb = _nm(rng.normal(size=(n, d)))
        bv = _mc(xb)
    # Phase A: layered pool short screening
    pool_best = []
    try:
        NPOOL = 26
        for pi in range(NPOOL):
            if time.time() > SEARCH - 60.0:
                break
            if pi < 20:
                X0 = _make_layered(rng)
            else:
                X0 = _nm(rng.normal(size=(n, d)))
            Y = _adam_bal(X0, 380, 28.0, 320.0, 0.016, 0.0015, deadline=SEARCH)
            v = _mc(Y)
            pool_best.append((v, Y))
    except Exception:
        pass
    pool_best.sort(key=lambda t: t[0])
    deep = []
    for v, Y in pool_best[:6]:
        deep.append(Y.copy())
    # always keep global best as fallback member
    # Phase B: deep basin hopping focused on balanced basins
    xcur = None
    cv = 1e9
    if deep:
        # init walk from best of pool
        bi = int(np.argmin([t[0] for t in pool_best[:6]])) if len(pool_best) >= 1 else 0
        xcur = deep[bi].copy(); cv = float(pool_best[bi][0])
        for v, Y in pool_best[:6]:
            if v < bv:
                bv = float(v); xb = Y.copy()
    else:
        xcur = xb.copy(); cv = bv
        deep = [xb.copy()]
    trial = 0
    ndeep = max(1, len(deep))
    while time.time() < SEARCH:
        trial += 1
        if time.time() > SEARCH - 1.0:
            break
        base = deep[trial % ndeep] if (trial % 3 != 0) else (xb.copy() if rng.random() < 0.35 else xcur.copy())
        X = base.copy()
        typ = trial % 8
        try:
            G = X @ X.T; np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            deg = np.sum(G > cmax - 0.004, axis=1)
            order_hi = np.argsort(-deg, kind="stable")
            order_lo = np.argsort(deg, kind="stable")
        except Exception:
            order_hi = np.arange(n); order_lo = np.arange(n)
        if typ == 0:
            X = _hub_spread(X, rng, thresh=0.004)
            X = _nm(X + 0.008 * rng.normal(size=(n, d)))
            st, kw = 950, dict(beta0=24., beta1=460., lr0=0.017, lr1=0.0012)
        elif typ == 1:
            # eject worst hub into void
            h = int(order_hi[0])
            mask = np.ones(n, dtype=bool); mask[h] = False
            X[h] = _best_void(X[mask], rng, ns=10000, refine=True)
            nbr = np.argsort(-G[h], kind="stable")[:6]
            X[nbr] = _nm(X[nbr] + 0.05 * rng.normal(size=(len(nbr), d)))
            X = _nm(X + 0.006 * rng.normal(size=(n, d)))
            st, kw = 950, dict(beta0=26., beta1=460., lr0=0.016, lr1=0.0012)
        elif typ == 2:
            # low-degree defect relocation + hub spread
            h = int(order_lo[0])
            # relocate isolated defect to void (it may be stuck)
            mask = np.ones(n, dtype=bool); mask[h] = False
            X[h] = _best_void(X[mask], rng, ns=10000, refine=True)
            X = _hub_spread(X, rng, thresh=0.006)
            X = _nm(X + 0.007 * rng.normal(size=(n, d)))
            st, kw = 1000, dict(beta0=24., beta1=450., lr0=0.017, lr1=0.0012)
        elif typ == 3:
            # geographic cap delete + sequential void reinsert
            u = _nm(rng.normal(size=(1, 4)))[0]
            dots = X @ u
            k = 9 + (trial % 5)
            vict = np.argsort(-dots, kind="stable")[:k]
            keep = np.ones(n, dtype=bool); keep[vict] = False
            cur_set = X[keep]
            newp = []
            for s in range(k):
                Xr = cur_set if s == 0 else np.vstack([cur_set, np.array(newp)])
                newp.append(_best_void(Xr, rng, ns=9000, refine=True))
            X[vict] = np.array(newp)
            X = _nm(X + 0.007 * rng.normal(size=(n, d)))
            st, kw = 1000, dict(beta0=24., beta1=450., lr0=0.017, lr1=0.0012)
        elif typ == 4:
            # fresh layered restart (new basin) occasionally
            if trial % 16 == 0:
                X = _make_layered(rng)
                st, kw = 1400, dict(beta0=20., beta1=500., lr0=0.019, lr1=0.0012)
            else:
                X = _nm(X + 0.045 * rng.normal(size=(n, d)))
                st, kw = 800, dict(beta0=30., beta1=420., lr0=0.016, lr1=0.0012)
        elif typ == 5:
            # worst-edge surgery
            ij = int(np.argmax(G)); i, j = ij // n, ij % n
            dij = X[i] - G[i, j] * X[j]
            nn = np.linalg.norm(dij)
            if nn > 1e-12:
                dij /= nn
            X[i] = _nm((X[i] + 0.08 * dij + 0.025 * rng.normal(size=(d,))).reshape(1, -1))[0]
            X[j] = _nm((X[j] - 0.03 * dij + 0.025 * rng.normal(size=(d,))).reshape(1, -1))[0]
            nbr = np.argsort(-G[i], kind="stable")[:6]
            X[nbr] = _nm(X[nbr] + 0.05 * rng.normal(size=(len(nbr), d)))
            st, kw = 850, dict(beta0=32., beta1=420., lr0=0.015, lr1=0.0014)
        elif typ == 6:
            k = 4 + (trial % 3)
            vict = order_hi[:k].copy()
            keep = np.ones(n, dtype=bool); keep[vict] = False
            cur_set = X[keep]
            newp = []
            for s in range(k):
                Xr = cur_set if s == 0 else np.vstack([cur_set, np.array(newp)])
                newp.append(_best_void(Xr, rng, ns=9000, refine=True))
            X[vict] = np.array(newp)
            X = _nm(X + 0.008 * rng.normal(size=(n, d)))
            st, kw = 950, dict(beta0=26., beta1=450., lr0=0.016, lr1=0.0012)
        else:
            X = _nm(X + 0.014 * rng.normal(size=(n, d)))
            st, kw = 700, dict(beta0=36., beta1=380., lr0=0.015, lr1=0.0015)
        try:
            Xo = _adam_bal(X, st, deadline=SEARCH, **kw)
        except Exception:
            continue
        v = _mc(Xo)
        Temp = 0.0022
        if v < cv or rng.random() < np.exp(-(v - cv) / Temp):
            xcur, cv = Xo.copy(), float(v)
            # refresh deep slot
            try:
                deep[trial % ndeep] = Xo.copy()
            except Exception:
                pass
        if v < bv:
            bv = float(v); xb = Xo.copy()
        do = False; bud = 0.0
        if v < bv + 1e-12:
            do = True; bud = 1.6
        elif v < bv + 0.002 and (trial % 5 == 0):
            do = True; bud = 0.8
        elif v < bv + 0.006 and (trial % 14 == 0):
            do = True; bud = 0.6
        if do and time.time() < SEARCH - 0.4:
            try:
                Xp = _lp_polish(Xo, min(time.time() + bud, SEARCH + 2.0, HARD - 6.0))
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
            xb = _lp_polish(xb, rem)
        except Exception:
            pass
    return _nm(xb)
