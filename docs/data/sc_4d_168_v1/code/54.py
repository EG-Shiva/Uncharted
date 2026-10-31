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

def _lp(X, deadline):
    X = _nm(X.copy())
    n, d = X.shape
    best = _mc(X)
    rad = 1.2e-3
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and rad > 1e-14:
        G = X @ X.T
        c = G[iu0, iu1]
        m = float(np.max(c))
        sel = c > m - max(20.0 * rad, 1e-9)
        I = iu0[sel]; J = iu1[sel]; cs = c[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            Aub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            Aeq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
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
            X, best, rad = Y, v, min(rad * 1.5, 0.05)
        else:
            rad *= 0.5
    return X

def _adam(X, steps, b0, b1, lr0, lr1, balance=False):
    n, d = X.shape
    m = np.zeros_like(X); v = np.zeros_like(X)
    b1a, b2a, eps = 0.9, 0.999, 1e-8
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    bal_w = np.ones(n)
    for t in range(1, steps + 1):
        f = t / steps
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gm = float(np.max(G))
        if balance and (t % 40 == 1 or t == 1):
            deg = np.sum(G > gm - 0.005, axis=1)
            bal_w = 1.0 + 0.6 * (deg - np.mean(deg)) / 6.0
            bal_w = np.clip(bal_w, 0.5, 2.5)
        W = np.exp(np.clip(beta * (G - gm), -20, 0))
        np.fill_diagonal(W, 0.0)
        if balance:
            W = W * bal_w[None, :] * bal_w[:, None]
        Gr = (W @ X) * beta
        Gt = Gr - np.sum(Gr * X, axis=1, keepdims=True) * X
        m = b1a * m + (1 - b1a) * Gt
        v = b2a * v + (1 - b2a) * (Gt * Gt)
        mh = m / (1 - b1a ** t)
        vh = v / (1 - b2a ** t)
        X = _nm(X - lr * mh / (np.sqrt(vh) + eps))
    return X

def _ref1(x0, Xr, steps=80, beta=60.0, lr=0.08):
    x = x0 / (np.linalg.norm(x0) + 1e-30)
    for _ in range(steps):
        dd = Xr @ x
        dm = float(np.max(dd))
        w = np.exp(np.clip(beta * (dd - dm), -20, 0))
        g = w @ Xr
        g = g - np.dot(g, x) * x
        gn = np.linalg.norm(g)
        if gn > 0:
            g = g / gn
        x = x - lr * g
        x = x / (np.linalg.norm(x) + 1e-30)
        lr *= 0.995
        beta = min(beta * 1.006, 220.0)
    return x / (np.linalg.norm(x) + 1e-30)

def _hole(Xk, rng, ns=8000):
    C = _nm(rng.normal(size=(ns, 4)))
    D = C @ Xk.T
    bi = int(np.argmin(D.max(axis=1)))
    return _ref1(C[bi], Xk)

def _fib_s2(m, rng, jit=0.0):
    pts = np.zeros((m, 3))
    ga = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(m):
        z = 1.0 - (2.0 * i + 1.0) / m
        r = np.sqrt(max(0.0, 1.0 - z * z))
        th = ga * i
        pts[i] = [r * np.cos(th), r * np.sin(th), z]
    if jit > 0:
        pts = pts + jit * rng.normal(size=pts.shape)
        pts = pts / (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-30)
    return pts

def _hopf_init(n_total, rng):
    # pick factor n2 * k ~= n_total
    best = None
    opts = [(42, 4), (28, 6), (24, 7), (21, 8), (14, 12), (12, 14), (56, 3), (84, 2)]
    choice = opts[int(rng.integers(0, len(opts)))]
    n2, k = choice[0], choice[1]
    while n2 * k != n_total:
        # adjust k to fit
        k = max(1, n_total // n2)
        rem = n_total - n2 * k
        break
    base = _fib_s2(n2, rng, jit=0.05 * rng.random())
    base = base / (np.linalg.norm(base, axis=1, keepdims=True) + 1e-30)
    pts = []
    for j in range(n2):
        nx, ny, nz = base[j]
        theta = np.arccos(np.clip(nz, -1, 1))
        phi = np.arctan2(ny, nx)
        off = rng.random() * 2 * np.pi
        for s in range(k):
            psi = off + 2 * np.pi * s / max(1, k) + 0.05 * rng.normal()
            z1 = np.cos(theta / 2) * np.exp(1j * (phi + psi) / 2)
            z2 = np.sin(theta / 2) * np.exp(1j * (phi - psi) / 2)
            pts.append([z1.real, z1.imag, z2.real, z2.imag])
    pts = np.array(pts[:n_total], dtype=float)
    if pts.shape[0] < n_total:
        extra = rng.normal(size=(n_total - pts.shape[0], 4))
        pts = np.vstack([pts, extra])
    # random rotation
    A = rng.normal(size=(4, 4))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    pts = pts @ Q.T
    pts = pts + 0.02 * rng.normal(size=pts.shape)
    return _nm(pts)

def _torus_init(n_total, rng):
    opts = [(12, 14), (14, 12), (21, 8), (28, 6), (24, 7), (42, 4)]
    a_n, b_n = opts[int(rng.integers(0, len(opts)))]
    while a_n * b_n != n_total:
        a_n = 12; b_n = 14
        break
    aa = np.linspace(0, 2 * np.pi, a_n, endpoint=False) + rng.random() * 0.2
    bb = np.linspace(0, 2 * np.pi, b_n, endpoint=False) + rng.random() * 0.2
    rng.shuffle(aa); rng.shuffle(bb)
    pts = []
    for a in aa:
        for b in bb:
            pts.append([np.cos(a) / np.sqrt(2), np.sin(a) / np.sqrt(2),
                        np.cos(b) / np.sqrt(2), np.sin(b) / np.sqrt(2)])
    pts = np.array(pts[:n_total], dtype=float)
    A = rng.normal(size=(4, 4))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    pts = pts @ Q.T
    pts = pts + 0.03 * rng.normal(size=pts.shape)
    return _nm(pts)

def _smooth_kick(X, rng, amp=0.12):
    n = X.shape[0]
    nk = int(rng.integers(3, 7))
    C = _nm(rng.normal(size=(nk, 4)))
    Vr = rng.normal(size=(nk, 4))
    Vr = Vr - np.sum(Vr * C, axis=1, keepdims=True) * C
    nn = np.linalg.norm(Vr, axis=1, keepdims=True); nn[nn == 0] = 1.0
    Vr = Vr / nn
    kap = rng.uniform(4.0, 25.0, size=nk)
    cf = rng.normal(size=nk)
    D = X @ C.T
    W = np.exp(kap[None, :] * (D - 1.0)) * cf[None, :]
    DX = W @ Vr
    DX = DX - np.sum(DX * X, axis=1, keepdims=True) * X
    rms = float(np.sqrt(np.mean(np.sum(DX * DX, axis=1)))) + 1e-12
    DX = DX / rms * amp
    return _nm(X + DX)

def solve():
    t0 = time.time()
    HARD = t0 + 171.0
    SEARCH = t0 + 142.0
    rng = np.random.default_rng(777001)
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
    try:
        if time.time() < SEARCH - 5:
            xq = _lp(xb.copy(), min(time.time() + 4.0, SEARCH - 1.0, HARD - 10.0))
            vq = _mc(xq)
            if vq < bv and np.all(np.isfinite(xq)):
                xb, bv = xq, vq
    except Exception:
        pass
    trial = 0
    while time.time() < SEARCH:
        trial += 1
        if time.time() > SEARCH - 1.0:
            break
        typ = trial % 8
        X = None
        st = 800
        kw = dict(b0=30., b1=420., lr0=0.02, lr1=0.002, balance=False)
        try:
            if typ == 0:
                X = _nm(xb + 0.02 * rng.normal(size=(n, d)))
                st, kw = 650, dict(b0=40., b1=380., lr0=0.02, lr1=0.002, balance=False)
            elif typ == 1:
                X = _hopf_init(n, rng)
                X = _nm(X + 0.01 * rng.normal(size=(n, d)))
                st, kw = 1100, dict(b0=20., b1=450., lr0=0.022, lr1=0.0018, balance=True)
            elif typ == 2:
                X = _torus_init(n, rng)
                X = _nm(X + 0.01 * rng.normal(size=(n, d)))
                st, kw = 1100, dict(b0=20., b1=450., lr0=0.022, lr1=0.0018, balance=True)
            elif typ == 3:
                X = _smooth_kick(xb, rng, amp=0.10 + 0.06 * rng.random())
                X = _nm(X + 0.008 * rng.normal(size=(n, d)))
                st, kw = 1000, dict(b0=25., b1=450., lr0=0.02, lr1=0.0018, balance=(trial % 2 == 0))
            elif typ == 4:
                G = xb @ xb.T; np.fill_diagonal(G, -2.0)
                cmax = float(np.max(G))
                deg = np.sum(G > cmax - 0.004, axis=1)
                k = 3 + (trial % 3)
                vict = np.argsort(-deg, kind="stable")[:k].copy()
                keep = np.ones(n, dtype=bool); keep[vict] = False
                Xk = xb[keep]
                newp = []
                for s in range(k):
                    Xr = Xk if s == 0 else np.vstack([Xk, np.array(newp)])
                    newp.append(_hole(Xr, rng, ns=7000))
                X = xb.copy(); X[vict] = np.array(newp)
                X = _nm(X + 0.010 * rng.normal(size=(n, d)))
                st, kw = 1050, dict(b0=28., b1=450., lr0=0.018, lr1=0.0018, balance=True)
            elif typ == 5:
                G = xb @ xb.T; np.fill_diagonal(G, -2.0)
                c0 = int(rng.integers(0, n))
                nbr = np.argsort(-G[c0])[:8]
                X = xb.copy()
                X[nbr] = _nm(X[nbr] + 0.30 * rng.normal(size=(len(nbr), d)))
                X[c0] = _nm((X[c0] + 0.30 * rng.normal(size=(d,))).reshape(1, -1))[0]
                X = _smooth_kick(X, rng, amp=0.05)
                st, kw = 1050, dict(b0=22., b1=430., lr0=0.022, lr1=0.0018, balance=False)
            elif typ == 6:
                X = xb.copy()
                k = 2
                vict = rng.choice(n, size=k, replace=False)
                for j in vict:
                    mask = np.ones(n, dtype=bool); mask[j] = False
                    X[j] = _hole(X[mask], rng, ns=7000)
                X = _nm(X + 0.015 * rng.normal(size=(n, d)))
                st, kw = 900, dict(b0=30., b1=420., lr0=0.02, lr1=0.002, balance=True)
            else:
                X = _nm(xb + 0.09 * rng.normal(size=(n, d)))
                st, kw = 950, dict(b0=24., b1=400., lr0=0.02, lr1=0.002, balance=False)
        except Exception:
            continue
        try:
            Xo = _adam(X, st, **kw)
        except Exception:
            continue
        if time.time() > SEARCH:
            try:
                vv = _mc(Xo)
                if vv < bv:
                    xb, bv = Xo.copy(), vv
            except Exception:
                pass
            break
        v = _mc(Xo)
        imp = False
        if v < bv:
            bv = v; xb = Xo.copy(); imp = True
        do = False; bud = 0.0
        if imp:
            do = True; bud = 2.0
        elif v < bv + 0.002 and (trial % 4 == 0):
            do = True; bud = 0.9
        elif v < bv + 0.007 and (trial % 12 == 0):
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
