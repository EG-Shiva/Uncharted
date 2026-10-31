import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(X, seconds):
    X = _norm(X.copy())
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (X @ X.T)[iu]
        m = float(np.max(c))
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        Y = _norm(X + res.x[:-1].reshape(n, d))
        v = _max_cos(Y)
        if v < best:
            X, best, radius = Y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _gd_full(X, steps, b0, b1, lr0, lr1, deadline=1e30):
    n, d = X.shape
    m = np.zeros_like(X); vv = np.zeros_like(X)
    b1a = 0.9; b2a = 0.999; eps = 1e-8
    lb0 = np.log(b0); lb1 = np.log(b1); lr0l = np.log(lr0); lr1l = np.log(lr1)
    for t in range(1, steps + 1):
        if (t & 127) == 0 and time.time() > deadline:
            break
        f = t / steps
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(lr0l + (lr1l - lr0l) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(np.max(G))
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        Grad = (W @ X) * beta
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        m = b1a * m + (1.0 - b1a) * Gt
        vv = b2a * vv + (1.0 - b2a) * (Gt * Gt)
        mhat = m / (1.0 - b1a ** t)
        vhat = vv / (1.0 - b2a ** t)
        X = X - lr * mhat / (np.sqrt(vhat) + eps)
        X = _norm(X)
    return X

def _gd_cavity(Xfull, free, steps, b0, b1, lr0, lr1):
    # optimize only free rows, background frozen
    X = Xfull.copy()
    nf = len(free)
    m = np.zeros((nf, 4)); vv = np.zeros((nf, 4))
    b1a = 0.9; b2a = 0.999; eps = 1e-8
    lb0 = np.log(b0); lb1 = np.log(b1); lr0l = np.log(lr0); lr1l = np.log(lr1)
    for t in range(1, steps + 1):
        f = t / steps
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(lr0l + (lr1l - lr0l) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(np.max(G))
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        Grad = (W @ X) * beta
        Gf = Grad[free]
        Xf = X[free]
        dot = np.sum(Gf * Xf, axis=1, keepdims=True)
        Gt = Gf - dot * Xf
        m = b1a * m + (1.0 - b1a) * Gt
        vv = b2a * vv + (1.0 - b2a) * (Gt * Gt)
        mhat = m / (1.0 - b1a ** t)
        vhat = vv / (1.0 - b2a ** t)
        Xf = Xf - lr * mhat / (np.sqrt(vhat) + eps)
        X[free] = _norm(Xf)
    return X

def solve():
    t0 = time.time()
    HOP_END = 146.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        if c is not None and c.shape == (168, 4) and np.all(np.isfinite(c)):
            cands.append(_norm(c.astype(float)))
    if len(cands) == 0:
        bestX = _norm(rng.normal(size=(168, 4)))
        best = _max_cos(bestX)
    else:
        vals = [_max_cos(c) for c in cands]
        bestX = cands[int(np.argmin(vals))].copy()
        best = float(np.min(vals))
    n, d = 168, 4
    curX = bestX.copy()
    cur = best
    it = 0
    while True:
        now = time.time() - t0
        if now > HOP_END:
            break
        it += 1
        if time.time() - t0 > HOP_END - 1.0:
            break
        typ = it % 8
        deadline = t0 + HOP_END - 0.5
        if typ == 0:
            sigma = 0.02
            Xt = _norm(curX + sigma * rng.normal(size=(n, d)))
            Xo = _gd_full(Xt, 700, 30.0, 350.0, 0.012, 0.0008, deadline)
        elif typ == 1:
            sigma = 0.07
            Xt = _norm(curX + sigma * rng.normal(size=(n, d)))
            Xo = _gd_full(Xt, 1000, 25.0, 400.0, 0.012, 0.0007, deadline)
        elif typ == 2:
            Xt = _norm(curX + 0.22 * rng.normal(size=(n, d)))
            k = 6
            idx = rng.choice(n, size=k, replace=False)
            Xt[idx] = _norm(rng.normal(size=(k, d)))
            Xo = _gd_full(Xt, 1300, 20.0, 450.0, 0.014, 0.0006, deadline)
        elif typ in (3, 4, 6):
            # cavity repack
            base = curX
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            deg = np.sum(G > cmax - 0.008, axis=1)
            if typ == 3:
                # overloaded cluster
                c = int(np.argmax(deg))
                center = base[c]
                k = 10 + (it % 5)  # 10..14
            elif typ == 4:
                # random geographic cap
                center = _norm(rng.normal(size=(1, 4)))[0]
                k = 8 + (it % 5)  # 8..12
            else:
                # rattler / low-contact void region
                order = np.argsort(deg)
                c = int(order[(it // 8) % n])
                # void: direction opposite to dense? use point itself
                center = base[c]
                # if rattler, center is rattler pos
                k = 8 + (it % 4)
            dots = base @ center
            free = np.argsort(-dots)[:k]
            # multi-restart cavity
            best_cav = None; best_cav_v = 1e30
            nrest = 3
            for r in range(nrest):
                Xt = base.copy()
                # init free points scattered in cap around center
                sig = 0.35 + 0.15 * r
                init = center[None, :] + sig * rng.normal(size=(k, d))
                init = _norm(init)
                # for one restart use uniform to diversify
                if r == 2:
                    init = _norm(rng.normal(size=(k, d)))
                    # pull half toward center
                    init[:k // 2] = _norm(center[None, :] + 0.5 * rng.normal(size=(k // 2, d)))
                Xt[free] = init
                Xc = _gd_cavity(Xt, free, 900, 25.0, 350.0, 0.02, 0.001)
                vc = _max_cos(Xc)
                if vc < best_cav_v:
                    best_cav_v = vc
                    best_cav = Xc
            # short full settle
            if typ == 3:
                Xo = _gd_full(best_cav, 700, 30.0, 300.0, 0.010, 0.0008, deadline)
            else:
                Xo = _gd_full(best_cav, 900, 25.0, 350.0, 0.012, 0.0008, deadline)
        elif typ == 5:
            # jammed transplant into voids
            G = curX @ curX.T
            np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            deg = np.sum(G > cmax - 0.003, axis=1)
            order = np.argsort(-deg)
            S = _norm(rng.normal(size=(12000, 4)))
            mx = (S @ curX.T).max(axis=1)
            void_idx = np.argsort(mx)[:12]
            voids = S[void_idx]
            Xt = _norm(curX + 0.02 * rng.normal(size=(n, d)))
            k = 2 + (it % 3)
            victims = order[:k]
            for a, vi in enumerate(victims):
                vv = voids[a % len(voids)] + 0.03 * rng.normal(size=(d,))
                vv = vv / (np.linalg.norm(vv) + 1e-30)
                Xt[vi] = vv
            Xo = _gd_full(Xt, 1100, 25.0, 300.0, 0.015, 0.0012, deadline)
        else:  # typ 7
            G = curX @ curX.T
            np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            deg = np.sum(G > cmax - 0.003, axis=1)
            sig = np.where(deg >= 6, 0.12, np.where(deg <= 1, 0.01, 0.04))
            Xt = _norm(curX + sig[:, None] * rng.normal(size=(n, d)))
            Xo = _gd_full(Xt, 900, 30.0, 250.0, 0.016, 0.0012, deadline)
        v = _max_cos(Xo)
        # LP snap if promising
        do_polish = False
       ptime = 0.0
        if v < best:
            do_polish = True
            ptime = 1.6
        elif v < best + 0.0015 and (it % 5 == 0):
            do_polish = True
            ptime = 0.8
        if do_polish and time.time() - t0 < HOP_END - 2.0:
            try:
                Xp = _lp_polish(Xo, ptime)
                vp = _max_cos(Xp)
                if vp < v:
                    Xo, v = Xp, vp
            except Exception:
                pass
        if v < best - 1e-12:
            best, bestX = v, Xo.copy()
            cur, curX = v, Xo.copy()
        else:
            # random-walk acceptance to leave basin
            tol = 0.004 if typ in (2, 3, 4) else 0.0015
            if v < cur + tol and rng.random() < 0.6:
                cur, curX = v, Xo.copy()
            elif v < best + 0.0005 and rng.random() < 0.15:
                cur, curX = v, Xo.copy()
        # occasional restart walk from best
        if it % 25 == 0 and rng.random() < 0.5:
            cur, curX = best, bestX.copy()
    rem = 178.0 - (time.time() - t0)
    if rem > 2.0:
        rem = min(rem, 28.0)
        try:
            Xf = _lp_polish(bestX, rem)
            vf = _max_cos(Xf)
            if vf < best:
                bestX = Xf
        except Exception:
            pass
    return _norm(bestX)
