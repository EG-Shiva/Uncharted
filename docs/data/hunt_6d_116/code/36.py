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
    return float(np.max(G))

def _adam_unconstrained(X0, steps, b0, b1, lr0, lr1, rng=None):
    X = _norm(X0.copy())
    n, d = X.shape
    m = np.zeros_like(X); v = np.zeros_like(X)
    b1p = 1.0; b2p = 1.0
    be1, be2 = 0.9, 0.999
    eps = 1e-8
    for it in range(1, steps+1):
        f = it / float(steps)
        beta = b0 * (b1 / b0) ** f
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cmax = float(np.max(G))
        W = np.exp(beta * (G - cmax))
        np.fill_diagonal(W, 0.0)
        S = float(np.sum(W))
        if not np.isfinite(S) or S <= 0:
            break
        Gr = (W @ X) / S
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        b1p *= be1; b2p *= be2
        m *= be1; m += (1.0-be1)*Gt
        v *= be2; v += (1.0-be2)*Gt*Gt
        mh = m/(1.0-b1p); vh = v/(1.0-b2p)
        step = lr * mh/(np.sqrt(vh)+eps)
        # cap
        X = X - step
        X = _norm(X)
    return X

def _adam_antipodal(Y0, steps, b0, b1, lr0, lr1):
    Y = Y0 / (np.linalg.norm(Y0, axis=1, keepdims=True)+1e-15)
    m = np.zeros_like(Y); v = np.zeros_like(Y)
    b1p = 1.0; b2p = 1.0
    be1, be2 = 0.9, 0.999
    eps = 1e-8
    for it in range(1, steps+1):
        f = it / float(steps)
        beta = b0 * (b1 / b0) ** f
        lr = lr0 + (lr1 - lr0) * f
        X = np.vstack([Y, -Y])
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cmax = float(np.max(G))
        W = np.exp(beta * (G - cmax))
        np.fill_diagonal(W, 0.0)
        S = float(np.sum(W))
        if not np.isfinite(S) or S <= 0:
            break
        Grf = (W @ X) / S
        Gr = Grf[:Y.shape[0]] - Grf[Y.shape[0]:]
        dot = np.sum(Gr * Y, axis=1, keepdims=True)
        Gt = Gr - dot * Y
        b1p *= be1; b2p *= be2
        m *= be1; m += (1.0-be1)*Gt
        v *= be2; v += (1.0-be2)*Gt*Gt
        mh = m/(1.0-b1p); vh = v/(1.0-b2p)
        step = lr * mh/(np.sqrt(vh)+eps)
        Y = Y - step
        Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True)+1e-15)
    return Y

_S3 = np.sqrt(3.0)

def _build_orbit(Rg, th):
    X = np.empty((116, 6))
    for j in range(9):
        y = Rg[j]
        for k in range(6):
            X[j*12+k] = np.roll(y, k)
            X[j*12+6+k] = -np.roll(y, k)
    for s in range(2):
        a = np.cos(th[s])/_S3; b = np.sin(th[s])/_S3
        p = np.array([a,b,a,b,a,b])
        p1 = np.roll(p, 1)
        base = 108 + s*4
        X[base+0] = p; X[base+1] = p1
        X[base+2] = -p; X[base+3] = -p1
    return X

def _adam_orbit(Rg0, th0, steps, b0, b1, lr0, lr1, lrth0=0.05, lrth1=0.005):
    Rg = Rg0 / (np.linalg.norm(Rg0, axis=1, keepdims=True)+1e-15)
    th = th0.copy().astype(float)
    mR = np.zeros_like(Rg); vR = np.zeros_like(Rg)
    mt = np.zeros_like(th); vt = np.zeros_like(th)
    b1p = 1.0; b2p = 1.0
    be1, be2 = 0.9, 0.999
    eps = 1e-8
    for it in range(1, steps+1):
        f = it / float(steps)
        beta = b0 * (b1 / b0) ** f
        lr = lr0 + (lr1 - lr0) * f
        lrth = lrth0 + (lrth1 - lrth0) * f
        X = _build_orbit(Rg, th)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cmax = float(np.max(G))
        W = np.exp(beta * (G - cmax))
        np.fill_diagonal(W, 0.0)
        S = float(np.sum(W))
        if not np.isfinite(S) or S <= 0:
            break
        Grf = (W @ X) / S
        # aggregate generic
        Gr = np.zeros_like(Rg)
        for j in range(9):
            acc = np.zeros(6)
            for k in range(6):
                gp = Grf[j*12+k]
                gm = Grf[j*12+6+k]
                acc += np.roll(gp - gm, -k)
            Gr[j] = acc
        # aggregate theta
        Gt_h = np.zeros_like(th)
        for s in range(2):
            base = 108 + s*4
            g0 = Grf[base+0]; g1 = Grf[base+1]
            g2 = Grf[base+2]; g3 = Grf[base+3]
            c = np.cos(th[s]); sn = np.sin(th[s])
            dp = np.array([-sn, c, -sn, c, -sn, c])/_S3
            dp1 = np.roll(dp, 1)
            Gt_h[s] = float(np.dot(g0-g2, dp) + np.dot(g1-g3, dp1))
        # project Rg
        dot = np.sum(Gr * Rg, axis=1, keepdims=True)
        GtR = Gr - dot * Rg
        b1p *= be1; b2p *= be2
        mR *= be1; mR += (1.0-be1)*GtR
        vR *= be2; vR += (1.0-be2)*GtR*GtR
        mh = mR/(1.0-b1p); vh = vR/(1.0-b2p)
        Rg = Rg - lr * mh/(np.sqrt(vh)+eps)
        Rg = Rg/(np.linalg.norm(Rg, axis=1, keepdims=True)+1e-15)
        # theta adam (no projection)
        mt *= be1; mt += (1.0-be1)*Gt_h
        vt *= be2; vt += (1.0-be2)*Gt_h*Gt_h
        mth = mt/(1.0-b1p); vth = vt/(1.0-b2p)
        th = th - lrth * mth/(np.sqrt(vth)+eps)
    return Rg, th

def _lp_polish(X, deadline):
    X = _norm(X.copy())
    n, d = X.shape
    best = _maxcos(X)
    bx = X.copy()
    radius = 2.0e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() > deadline - 0.4:
            break
        G = X @ X.T
        c = G[iu]
        m = float(c.max())
        thr = m - max(20.0*radius, 1e-9)
        idx = np.where(c > thr)[0]
        if idx.size == 0:
            break
        if idx.size > 1400:
            o = np.argsort(c[idx])[-1400:]
            idx = idx[o]
        sel = np.zeros_like(c, dtype=bool)
        sel[idx] = True
        I = iu[0][sel]; J = iu[1][sel]
        cs = c[sel]
        p = len(I)
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p,1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
            A_eq = coo_matrix((X.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))), shape=(n,nv)).tocsr()
            obj = np.zeros(nv); obj[-1]=1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n,d)
        y = _norm(y)
        vv = _maxcos(y)
        if vv < best:
            best = vv; bx = y.copy(); X = y
            radius = min(radius*1.5, 0.05)
        else:
            if vv <= best + 1e-15:
                X = y
            radius *= 0.5
    return bx, best

def _contact_var(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    C = G > (m - 1e-9)
    deg = np.sum(C, axis=1)
    return float(np.var(deg)), deg

def solve():
    t0 = time.time()
    HARD = t0 + 171.0
    rng = np.random.default_rng(20261031)
    # Phase A: antipodal constrained search (ignore best/record)
    bestA_val = 1e9; bestA_Y = None
    deadlineA = t0 + 38.0
    rA = 0
    while time.time() < deadlineA - 0.5:
        rA += 1
        Y0 = rng.standard_normal((58,6))
        Y0 /= np.linalg.norm(Y0, axis=1, keepdims=True)
        # vary steps/beta slightly per restart for diversity
        steps = 900 + int(rng.integers(0, 500))
        b0 = 50.0
        b1 = 1200.0 + float(rng.integers(0, 600))
        lr0 = 0.03; lr1 = 0.003
        if rA % 3 == 0:
            lr0 = 0.05
        Y = _adam_antipodal(Y0, steps=steps, b0=b0, b1=b1, lr0=lr0, lr1=lr1)
        Xf = np.vstack([Y, -Y])
        Xf = _norm(Xf)
        v = _maxcos(Xf)
        if v < bestA_val:
            bestA_val = v; bestA_Y = Y.copy()
        if time.time() > HARD - 60:
            break
    # Phase B: orbit constrained search
    bestO_val = 1e9; bestO_Rg = None; bestO_th = None
    deadlineB = t0 + 70.0
    rO = 0
    while time.time() < deadlineB - 0.5:
        rO += 1
        Rg0 = rng.standard_normal((9,6))
        Rg0 /= np.linalg.norm(Rg0, axis=1, keepdims=True)
        th0 = rng.uniform(0, 2*np.pi, size=2)
        steps = 800 + int(rng.integers(0, 400))
        b1 = 1200.0 + float(rng.integers(0, 600))
        Rg, th = _adam_orbit(Rg0, th0, steps=steps, b0=50.0, b1=b1, lr0=0.03, lr1=0.003)
        Xf = _build_orbit(Rg, th)
        Xf = _norm(Xf)
        v = _maxcos(Xf)
        if v < bestO_val:
            bestO_val = v; bestO_Rg = Rg.copy(); bestO_th = th.copy()
        if time.time() > HARD - 55:
            break
    # collect constrained optima
    cands = []
    if bestA_Y is not None:
        Xa = _norm(np.vstack([bestA_Y, -bestA_Y]))
        cands.append(("ant", Xa, bestA_val))
    if bestO_Rg is not None:
        Xo = _norm(_build_orbit(bestO_Rg, bestO_th))
        cands.append(("orb", Xo, bestO_val))
    # contact variance info (choose balanced) - sort by variance then value
    scored = []
    for name, X, v in cands:
        var, _ = _contact_var(X)
        scored.append((var, v, name, X))
    scored.sort(key=lambda t: (t[1],))
    # Phase C: release + perturb + unconstrained re-optimise
    released = []
    # take up to 2 constrained families
    for _, _, name, Xc in scored[:2]:
        for sigma in (0.003, 0.008):
            if time.time() > HARD - 45:
                break
            Xp = Xc + rng.normal(scale=sigma, size=(116,6))
            Xp = _norm(Xp)
            # annealed softmax unconstrained
            Xq = _adam_unconstrained(Xp, steps=1400, b0=60.0, b1=1800.0, lr0=0.04, lr1=0.002)
            vq = _maxcos(Xq)
            released.append((vq, Xq))
            if time.time() > HARD - 40:
                break
        if time.time() > HARD - 40:
            break
    # also one direct unconstrained from best constrained without perturb (short quench)
    # pick best released + constrained for LP polish (best 3)
    pool = [(v, X) for _, X, v in cands] + released
    pool.sort(key=lambda t: t[0])
    pool = pool[:3]
    best_val = 1e9; best_X = None
    for v, X in pool:
        best_val_tmp = v; best_X_tmp = X
        if v < best_val:
            best_val = v; best_X = X.copy()
    # LP polish best 2-3 released candidates with remaining time
    # allocate ~10s each
    pool_sorted = sorted(pool, key=lambda t: t[0])[:3]
    final_best_val = best_val
    final_best_X = best_X.copy() if best_X is not None else _norm(rng.standard_normal((116,6)))
    for i, (v, X) in enumerate(pool_sorted):
        remain = HARD - time.time()
        if remain < 6.0:
            break
        budget = min(12.0, remain - 4.0)
        if budget < 1.0:
            break
        Xp, vp = _lp_polish(X, time.time()+budget)
        if vp < final_best_val:
            final_best_val = vp; final_best_X = Xp.copy()
        # also try one more unconstrained+LP cycle if time
        if HARD - time.time() > 12.0 and i == 0:
            # extra perturb of current best
            X2 = final_best_X + rng.normal(scale=0.002, size=(116,6))
            X2 = _norm(X2)
            X2q = _adam_unconstrained(X2, steps=800, b0=120.0, b1=2000.0, lr0=0.02, lr1=0.001)
            v2q = _maxcos(X2q)
            if v2q < final_best_val + 0.005:
                b2 = min(8.0, HARD - time.time() - 3.0)
                if b2 > 1.0:
                    X2p, v2p = _lp_polish(X2q, time.time()+b2)
                    if v2p < final_best_val:
                        final_best_val = v2p; final_best_X = X2p.copy()
    # final comparison with best.npy / record.npy (allowed now)
    def _clean(a):
        try:
            a = np.asarray(a, dtype=float)
        except Exception:
            return None
        if a.shape != (116,6) or not np.all(np.isfinite(a)):
            return None
        n = np.linalg.norm(a, axis=1, keepdims=True)
        if np.any(n < 1e-12):
            return None
        return a / n
    for f in ("best.npy", "record.npy"):
        try:
            c = _clean(np.load(f))
            if c is not None:
                vc = _maxcos(c)
                if vc < final_best_val:
                    final_best_val = vc; final_best_X = c.copy()
        except Exception:
            pass
    final_best_X = _norm(final_best_X)
    return final_best_X
