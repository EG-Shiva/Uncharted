import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _adam_unc(X, iters, a0, a1, lr0, lr1):
    X = _norm(X.copy())
    n, d = X.shape
    m = np.zeros_like(X); vv = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    for it in range(1, iters + 1):
        f = it / float(iters)
        alpha = a0 + (a1 - a0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cm = float(G.max())
        W = np.exp(alpha * (G - cm))
        np.fill_diagonal(W, 0.0)
        s = float(W.sum())
        if not np.isfinite(s) or s <= 0:
            break
        W /= s
        Grad = W @ X
        pr = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - pr * X
        b1p *= b1; b2p *= b2
        m *= b1; m += (1.0 - b1) * Gt
        vv *= b2; vv += (1.0 - b2) * Gt * Gt
        mh = m / (1.0 - b1p); vh = vv / (1.0 - b2p)
        X = X - lr * mh / (np.sqrt(vh) + eps)
        X = _norm(X)
    return X

def _adam_abs(Y, iters, a0, a1, lr0, lr1, rng=None):
    Y = _norm(Y.copy())
    n, d = Y.shape
    m = np.zeros_like(Y); vv = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    for it in range(1, iters + 1):
        f = it / float(iters)
        alpha = a0 + (a1 - a0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = Y @ Y.T
        np.fill_diagonal(G, 0.0)
        A = np.abs(G)
        np.fill_diagonal(A, -10.0)
        cm = float(A.max())
        W = np.exp(alpha * (A - cm))
        np.fill_diagonal(W, 0.0)
        s = float(W.sum())
        if not np.isfinite(s) or s <= 0:
            break
        W /= s
        S = np.sign(G)
        np.fill_diagonal(S, 0.0)
        WS = W * S
        Grad = WS @ Y
        # symmetrize (W symmetric, S symmetric)
        Grad = Grad + (WS.T * 1.0).T @ Y * 0.0  # already symmetric
        pr = (Grad * Y).sum(axis=1, keepdims=True)
        Gt = Grad - pr * Y
        b1p *= b1; b2p *= b2
        m *= b1; m += (1.0 - b1) * Gt
        vv *= b2; vv += (1.0 - b2) * Gt * Gt
        mh = m / (1.0 - b1p); vh = vv / (1.0 - b2p)
        Y = Y - lr * mh / (np.sqrt(vh) + eps)
        Y = _norm(Y)
    return Y

def _expand_antipodal(Y):
    return _norm(np.vstack([Y, -Y]))

def _expand_orbit(Yg, u):
    # Yg (9,6), u (3,)
    pts = []
    for i in range(Yg.shape[0]):
        y = Yg[i]
        for k in range(6):
            pts.append(np.roll(y, k))
            pts.append(-np.roll(y, k))
    # special 6-orbit from u
    un = u / (np.linalg.norm(u) + 1e-15)
    v = np.concatenate([un, un]) / np.sqrt(2.0)
    v = v / np.linalg.norm(v)
    for k in range(3):
        pts.append(np.roll(v, k))
        pts.append(-np.roll(v, k))
    v0 = np.ones(6) / np.sqrt(6.0)
    pts.append(v0.copy()); pts.append(-v0.copy())
    X = np.stack(pts, axis=0)
    return _norm(X)

def _adam_orbit(Yg, u, iters, a0, a1, lr0, lr1):
    Yg = _norm(Yg.copy())
    u = np.asarray(u, dtype=float).copy()
    u = u / (np.linalg.norm(u) + 1e-15)
    mg = np.zeros_like(Yg); vg = np.zeros_like(Yg)
    mu = np.zeros_like(u); vu = np.zeros_like(u)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    for it in range(1, iters + 1):
        f = it / float(iters)
        alpha = a0 + (a1 - a0) * f
        lr = lr0 + (lr1 - lr0) * f
        X = _expand_orbit(Yg, u)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cm = float(G.max())
        W = np.exp(alpha * (G - cm))
        np.fill_diagonal(W, 0.0)
        s = float(W.sum())
        if not np.isfinite(s) or s <= 0:
            break
        W /= s
        GradF = W @ X
        prF = (GradF * X).sum(axis=1, keepdims=True)
        GtF = GradF - prF * X
        # map back to Yg: indices 0..107 are generic, 12 per rep
        Gg = np.zeros_like(Yg)
        idx = 0
        for i in range(Yg.shape[0]):
            acc = np.zeros(6)
            for k in range(6):
                # point idx = s=+ roll k ; idx+1 = s=- roll k
                gpos = GtF[idx]; idx += 1
                gneg = GtF[idx]; idx += 1
                acc += np.roll(gpos, -k) - np.roll(gneg, -k)
            Gg[i] = acc
        # special u part: next 6 points
        accv = np.zeros(6)
        for k in range(3):
            gpos = GtF[idx]; idx += 1
            gneg = GtF[idx]; idx += 1
            accv += np.roll(gpos, -k) - np.roll(gneg, -k)
        # v = [u,u]/sqrt(2)|u| ; project accv to u
        # dv/du linear part: (accv[0:3]+accv[3:6])/sqrt2
        gu = (accv[0:3] + accv[3:6]) / np.sqrt(2.0)
        # project both
        prg = (Gg * Yg).sum(axis=1, keepdims=True)
        GtG = Gg - prg * Yg
        pru = float(gu @ u)
        GtU = gu - pru * u
        b1p *= b1; b2p *= b2
        mg *= b1; mg += (1.0 - b1) * GtG
        vg *= b2; vg += (1.0 - b2) * GtG * GtG
        mu *= b1; mu += (1.0 - b1) * GtU
        vu *= b2; vu += (1.0 - b2) * GtU * GtU
        mhg = mg / (1.0 - b1p); vhg = vg / (1.0 - b2p)
        mhu = mu / (1.0 - b1p); vhu = vu / (1.0 - b2p)
        Yg = Yg - lr * mhg / (np.sqrt(vhg) + eps)
        u = u - lr * mhu / (np.sqrt(vhu) + eps)
        Yg = _norm(Yg)
        u = u / (np.linalg.norm(u) + 1e-15)
    return Yg, u

def _lp_polish(X, seconds):
    X = _norm(X.copy())
    n, d = X.shape
    best = _maxcos(X)
    bestX = X.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if radius < 1e-14:
            break
        if time.time() - start > seconds - 0.05:
            break
        G = X @ X.T
        c = G[iu]
        m = float(c.max())
        idx = np.where(c > m - max(20.0 * radius, 1e-9))[0]
        if len(idx) > 1500:
            order = np.argsort(c[idx])[-1500:]
            idx = idx[order]
        if len(idx) == 0:
            break
        sel = np.zeros_like(c, dtype=bool); sel[idx] = True
        I = iu[0][sel]; J = iu[1][sel]; csel = c[sel]
        p = len(I); nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _maxcos(y)
        if v < best:
            best = v; bestX = y.copy(); X = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return bestX

def solve():
    t0 = time.time()
    rng = np.random.default_rng(0)
    fam = []  # list of (X, val)
    # Phase A: antipodal 58 reps
    tA_end = t0 + 48.0
    bestA = None; bestAval = 1e9
    cntA = 0
    while time.time() < tA_end:
        cntA += 1
        Y0 = _norm(rng.standard_normal((58, 6)))
        # quick anneal beta 50->1500
        Y1 = _adam_abs(Y0, iters=900, a0=50.0, a1=900.0, lr0=0.02, lr1=0.002)
        if time.time() > tA_end - 1.0:
            # evaluate anyway
            pass
        Y1 = _adam_abs(Y1, iters=500, a0=900.0, a1=1500.0, lr0=0.005, lr1=0.001)
        X = _expand_antipodal(Y1)
        v = _maxcos(X)
        fam.append((X.copy(), v))
        if v < bestAval:
            bestAval = v; bestA = X.copy()
        if time.time() > tA_end:
            break
    # Phase B: orbit construction 9x12+6+2
    tB_end = t0 + 92.0
    bestB = None; bestBval = 1e9
    while time.time() < tB_end:
        Yg0 = _norm(rng.standard_normal((9, 6)))
        u0 = rng.standard_normal(3); u0 /= np.linalg.norm(u0)
        Yg1, u1 = _adam_orbit(Yg0, u0, iters=700, a0=50.0, a1=900.0, lr0=0.02, lr1=0.003)
        Yg1, u1 = _adam_orbit(Yg1, u1, iters=400, a0=900.0, a1=1500.0, lr0=0.005, lr1=0.001)
        X = _expand_orbit(Yg1, u1)
        v = _maxcos(X)
        fam.append((X.copy(), v))
        if v < bestBval:
            bestBval = v; bestB = X.copy()
        if time.time() > tB_end:
            break
    # select best 2-3 constrained candidates for release
    fam_sorted = sorted(fam, key=lambda t: t[1])[:4]
    # add current bestA/B explicitly
    released = []
    for (Xc, vc) in fam_sorted:
        if time.time() > t0 + 150.0:
            break
        for sigma in ([0.003] if len(released) > 3 else [0.003, 0.008]):
            if time.time() > t0 + 150.0:
                break
            Y = Xc + sigma * rng.standard_normal(Xc.shape)
            Y = _norm(Y)
            Y = _adam_unc(Y, iters=600, a0=120.0, a1=1600.0, lr0=0.008, lr1=0.001)
            Y = _adam_unc(Y, iters=300, a0=300.0, a1=2500.0, lr0=0.003, lr1=0.0005)
            v = _maxcos(Y)
            released.append((Y.copy(), v))
    # pool: constrained + released
    pool = fam_sorted + released
    # LP-polish best two-three released candidates (lowest val)
    pool_sorted = sorted(pool, key=lambda t: t[1])
    final_cands = []
    for i, (Xc, vc) in enumerate(pool_sorted[:3]):
        left = (t0 + 172.0) - time.time()
        if left < 4.0:
            final_cands.append((Xc, vc))
            continue
        secs = min(6.0, left / (3 - i + 0.5))
        try:
            Xp = _lp_polish(Xc, seconds=secs)
            vp = _maxcos(Xp)
        except Exception:
            Xp, vp = Xc, vc
        final_cands.append((Xp, vp))
    # also keep unpolished rest
    for (Xc, vc) in pool_sorted[3:6]:
        final_cands.append((Xc, vc))
    if len(final_cands) == 0:
        final_cands.append((_norm(rng.standard_normal((116, 6))), 1.0))
    vals = [v for (_, v) in final_cands]
    bi = int(np.argmin(vals))
    Xbest, bestv = final_cands[bi][0].copy(), float(vals[bi])
    # final comparison with best.npy / record.npy (allowed at end)
    try:
        b = np.asarray(np.load("best.npy"), dtype=float)
        if b.shape == (116, 6) and np.all(np.isfinite(b)):
            b = _norm(b)
            vb = _maxcos(b)
            if vb < bestv:
                Xbest, bestv = b, vb
    except Exception:
        pass
    try:
        r = np.asarray(np.load("record.npy"), dtype=float)
        if r.shape == (116, 6) and np.all(np.isfinite(r)):
            r = _norm(r)
            vr = _maxcos(r)
            if vr < bestv:
                Xbest, bestv = r, vr
    except Exception:
        pass
    Xbest = _norm(Xbest)
    return Xbest
