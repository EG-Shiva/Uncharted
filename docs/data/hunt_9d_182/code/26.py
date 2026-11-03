import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n < 1e-16] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _adam_lse(X0, steps, beta, lr0, lr1, deadline=1e18, track=100):
    X = _norm(X0.copy())
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    b1p = 1.0
    b2p = 1.0
    try:
        bestv = _max_cos(X)
    except Exception:
        bestv = 1e9
    best = X.copy()
    l0 = np.log(lr0)
    l1 = np.log(lr1)
    for it in range(steps):
        if (it & 63) == 0 and time.time() > deadline:
            break
        frac = it / max(1, steps - 1)
        lr = float(np.exp(l0 + (l1 - l0) * frac))
        t += 1
        b1p *= b1
        b2p *= b2
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = float(np.max(G))
        E = np.exp(beta * (G - mx))
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E))
        if not np.isfinite(S) or S <= 0:
            continue
        Grad = (E @ X) / S
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        m1 = b1 * m1 + (1.0 - b1) * Gt
        m2 = b2 * m2 + (1.0 - b2) * (Gt * Gt)
        mh = m1 / (1.0 - b1p)
        vh = m2 / (1.0 - b2p)
        upd = mh / (np.sqrt(vh) + eps)
        dn = np.sum(upd * X, axis=1, keepdims=True)
        upd = upd - dn * X
        X = X - lr * upd
        X = _norm(X)
        if (it % track) == (track - 1):
            v = _max_cos(X)
            if v < bestv:
                bestv = v
                best = X.copy()
    v = _max_cos(X)
    if v < bestv:
        bestv = v
        best = X.copy()
    return best, bestv

def _bulk_uniform(X0, steps, topk, lr, deadline=1e18):
    X = _norm(X0.copy())
    n, d = X.shape
    try:
        bestv = _max_cos(X)
    except Exception:
        return X, 1e9
    best = X.copy()
    iu0 = np.triu_indices(n, 1)
    for it in range(steps):
        if (it & 31) == 0 and time.time() > deadline:
            break
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        c = G[iu0]
        k = min(topk, c.size)
        thr = np.partition(c, -k)[-k]
        sel = c >= thr
        I = iu0[0][sel]
        J = iu0[1][sel]
        # uniform weight adjacency
        W = np.zeros((n, n), dtype=float)
        np.add.at(W, (I, J), 1.0)
        W = W + W.T
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        # normalize step per point to avoid blowup
        nm = np.linalg.norm(Gt, axis=1, keepdims=True)
        Gt = Gt / (nm + 1.0)
        X = X - lr * Gt
        X = _norm(X)
        if (it % 50) == 49:
            v = _max_cos(X)
            if v < bestv:
                bestv = v
                best = X.copy()
    v = _max_cos(X)
    if v < bestv:
        bestv = v
        best = X.copy()
    return best, bestv

def _single_hole(y0, Others, iters=220, beta=120.0, lr=0.03):
    y = np.asarray(y0, dtype=float).copy()
    nn = float(np.linalg.norm(y))
    if not np.finite(nn) or nn < 1e-16:
        y = Others[0].copy()
        nn = float(np.linalg.norm(y))
    y = y / nn
    m1 = np.zeros_like(y)
    m2 = np.zeros_like(y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    dots0 = Others @ y
    best_s = float(np.max(dots0))
    best_y = y.copy()
    for t in range(1, iters + 1):
        dots = Others @ y
        m0 = float(np.max(dots))
        W = np.exp(beta * (dots - m0))
        S = float(np.sum(W))
        if not np.finite(S) or S <= 0:
            break
        grad = (W @ Others) / S
        gt = grad - float(grad @ y) * y
        m1 = b1 * m1 + (1.0 - b1) * gt
        m2 = b2 * m2 + (1.0 - b2) * (gt * gt)
        c1 = 1.0 - b1 ** t
        c2 = 1.0 - b2 ** t
        st = (m1 / c1) / (np.sqrt(m2 / c2) + eps)
        st = st - float(st @ y) * y
        lrt = lr / (1.0 + t / 200.0)
        y = y - lrt * st
        nn = float(np.linalg.norm(y))
        if not np.finite(nn) or nn < 1e-16:
            break
        y = y / nn
        if (t % 40) == 0:
            s = float(np.max(Others @ y))
            if s < best_s:
                best_s = s
                best_y = y.copy()
    s2 = float(np.max(Others @ y))
    if s2 < best_s:
        return y, s2
    return best_y, best_s

def _hole_reinsert(Xbest, bestv, rng):
    n, d = Xbest.shape
    G = Xbest @ Xbest.T
    np.fill_diagonal(G, -2.0)
    rowmax = np.max(G, axis=1)
    order = np.argsort(-rowmax)[:4]
    R = _norm(rng.standard_normal(size=(2500, d)))
    improved = False
    Xcur = Xbest.copy()
    vcur = bestv
    for idx in order:
        idx = int(idx)
        Others = np.delete(Xcur, idx, axis=0)
        sc = R @ Others.T
        sm = np.max(sc, axis=1)
        o = np.argsort(sm)[:3]
        for kk in o:
            y, _ = _single_hole(R[int(kk)], Others, iters=220, beta=130.0, lr=0.05)
            Y = Xcur.copy()
            Y[idx] = y
            v = _max_cos(Y)
            if v < vcur - 1e-12:
                Xcur = Y
                vcur = v
                improved = True
                break
        if improved:
            break
    return Xcur, vcur, improved

def _lp_jam(X0, deadline, r0=2e-3, max_it=60):
    X = _norm(X0.copy())
    n, d = X.shape
    best = _max_cos(X)
    bestX = X.copy()
    iu = np.triu_indices(n, 1)
    radius = r0
    stall = 0
    it = 0
    while time.time() < deadline and it < max_it and radius > 1e-13:
        it += 1
        Gf = X @ X.T
        cf = Gf[iu]
        m = float(np.max(cf))
        thr = m - max(8.0 * radius, 8e-4)
        sel = cf > thr
        p = int(np.count_nonzero(sel))
        if p < 800:
            kk = min(1200, cf.size)
            idxp = np.argpartition(-cf, kk - 1)[:kk]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idxp] = True
            p = kk
        if p > 4500:
            idxw = np.where(sel)[0]
            keep = np.argsort(cf[idxw])[-4500:]
            ns = np.zeros_like(sel, dtype=bool)
            ns[idxw[keep]] = True
            sel = ns
            p = 4500
        I = iu[0][sel]
        J = iu[1][sel]
        cs = cf[sel]
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
            obj = np.zeros(n * d + 1)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            stall += 1
            if stall >= 6:
                rnd = np.random.randn(n, d) * 5e-7
                rnd -= np.sum(rnd * X, axis=1, keepdims=True) * X
                X = _norm(X + rnd)
                stall = 0
            continue
        dx = res.x[:-1].reshape(n, d)
        done = False
        for alpha in (1.0, 0.5, 0.25, 0.125):
            Y = _norm(X + alpha * dx)
            v = _max_cos(Y)
            if v < best - 1e-13:
                best = v
                bestX = Y.copy()
                X = Y
                radius = min(radius * 1.4, 2.5e-3)
                stall = 0
                done = True
                break
        if not done:
            radius *= 0.5
            stall += 1
            if stall >= 7:
                rnd = np.random.randn(n, d) * (radius * 0.4 + 5e-8)
                rnd -= np.sum(rnd * X, axis=1, keepdims=True) * X
                Yt = _norm(X + rnd)
                vt = _max_cos(Yt)
                if vt < best + 2e-7:
                    X = Yt
                    if vt < best:
                        best = vt
                        bestX = Yt.copy()
                stall = 0
    return bestX, best

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.default_rng(0)
    np.random.seed(0)
    Xbest = None
    bestv = 1e9
    for fname in ("best.npy", "record.npy"):
        try:
            Y = np.load(fname)
            Y = np.asarray(Y, dtype=float)
            if Y.shape != (182, 9):
                continue
            Y = _norm(Y)
            if np.any(~np.isfinite(Y)):
                continue
            v = _max_cos(Y)
            if v < bestv:
                bestv = v
                Xbest = Y.copy()
        except Exception:
            continue
    if Xbest is None:
        Xbest = _norm(rng.standard_normal(size=(182, 9)))
        bestv = _max_cos(Xbest)
    # initial broad equalization
    try:
        X1, v1 = _adam_lse(Xbest, steps=700, beta=70.0, lr0=0.005, lr1=0.001, deadline=deadline - 40.0)
        if v1 < bestv:
            Xbest, bestv = X1, v1
        X2, v2 = _bulk_uniform(Xbest, steps=300, topk=1400, lr=0.002, deadline=deadline - 40.0)
        if v2 < bestv:
            Xbest, bestv = X2, v2
        X3, v3 = _adam_lse(Xbest, steps=500, beta=150.0, lr0=0.003, lr1=0.0006, deadline=deadline - 40.0)
        if v3 < bestv:
            Xbest, bestv = X3, v3
    except Exception:
        pass
    sigmas = [0.025, 0.012, 0.04, 0.008, 0.018, 0.03]
    trial = 0
    while time.time() < deadline - 42.0 and trial < 60:
        trial += 1
        if time.time() > deadline - 42.0:
            break
        try:
            if trial % 7 == 0:
                Xh, vh, ok = _hole_reinsert(Xbest, bestv, rng)
                if ok and vh < bestv:
                    Xbest, bestv = Xh, vh
                continue
            sigma = sigmas[(trial - 1) % len(sigmas)]
            pert = sigma * rng.standard_normal(size=Xbest.shape)
            if trial % 3 == 0:
                G = Xbest @ Xbest.T
                np.fill_diagonal(G, -2.0)
                rowmax = np.max(G, axis=1)
                worst = np.argsort(-rowmax)[:12]
                pert[worst] += rng.standard_normal(size=(12, 9)) * (sigma * 1.5 + 0.01)
            X0 = _norm(Xbest + pert)
            X1, v1 = _adam_lse(X0, steps=550, beta=75.0, lr0=0.0045, lr1=0.001, deadline=deadline - 40.0)
            X2, v2 = _bulk_uniform(X1, steps=220, topk=1350, lr=0.0016, deadline=deadline - 40.0)
            X3, v3 = _adam_lse(X2, steps=350, beta=160.0, lr0=0.0022, lr1=0.0005, deadline=deadline - 40.0)
            # pick best among stages
            cand, candv = X3, v3
            if v1 < candv:
                cand, candv = X1, v1
            if v2 < candv:
                cand, candv = X2, v2
            if candv < bestv - 1e-12:
                Xbest, bestv = cand, candv
        except Exception:
            continue
    # jam with LP
    if time.time() < deadline - 6.0:
        try:
            rem = deadline - time.time() - 2.0
            lp_dead = time.time() + min(max(rem * 0.8, 5.0), 38.0)
            Xlp, vlp = _lp_jam(Xbest, deadline=lp_dead, r0=2e-3, max_it=60)
            if vlp < bestv:
                Xbest, bestv = Xlp, vlp
        except Exception:
            pass
    # final micro squeeze if time remains
    if time.time() < deadline - 2.0:
        try:
            Xf, vf = _adam_lse(Xbest, steps=300, beta=220.0, lr0=0.0012, lr1=0.0003, deadline=deadline - 0.5)
            if vf < bestv:
                Xbest, bestv = Xf, vf
        except Exception:
            pass
    Xbest = _norm(Xbest)
    # tiny shrink guard against exact ties due to rounding (no-op, keep unit rows)
    return Xbest
