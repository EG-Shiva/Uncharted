import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def gd_refine(X, steps, b0, b1, lr0, lr1):
    n, d = X.shape
    m = np.zeros_like(X)
    vv = np.zeros_like(X)
    b1a = 0.9; b2a = 0.999; eps = 1e-8
    lb0 = np.log(b0); lb1 = np.log(b1)
    llr0 = np.log(lr0); llr1 = np.log(lr1)
    for t in range(1, steps + 1):
        f = t / steps
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(llr0 + (llr1 - llr0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        Grad = (W @ X) * beta
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        m = b1a * m + (1.0 - b1a) * Gt
        vv = b2a * vv + (1.0 - b2a) * (Gt * Gt)
        c1 = 1.0 - b1a ** t
        c2 = 1.0 - b2a ** t
        mhat = m / c1
        vhat = vv / c2
        X = X - lr * mhat / (np.sqrt(vhat) + eps)
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X

def refine_single(x0, Xref, steps=150, beta=60.0, lr=0.05):
    x = x0 / np.linalg.norm(x0)
    for t in range(steps):
        d = Xref @ x
        dm = d.max()
        w = np.exp(beta * (d - dm))
        g = w @ Xref
        g = g - np.dot(g, x) * x
        gn = np.linalg.norm(g)
        if gn > 0:
            g = g / gn
        x = x - lr * g
        x = x / np.linalg.norm(x)
        lr *= 0.995
        beta = min(beta * 1.005, 200.0)
    return x / np.linalg.norm(x)

def surgical_proposal(Xcur, rng, k_move, n_sample=8000):
    n, d = Xcur.shape
    G = Xcur @ Xcur.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    deg = np.sum(G > m - 0.003, axis=1)
    order = np.argsort(-deg)
    victims = order[:k_move].copy()
    # also occasionally include the least-contact point instead of one victim
    # keep victims as most crowded
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[victims] = False
    X_keep = Xcur[keep_mask]
    new_pts = []
    for s in range(k_move):
        if len(new_pts) == 0:
            Xref = X_keep
        else:
            Xref = np.vstack([X_keep, np.array(new_pts)])
        C = rng.normal(size=(n_sample, d))
        C = C / np.linalg.norm(C, axis=1, keepdims=True)
        D = C @ Xref.T
        mc = D.max(axis=1)
        bi = int(np.argmin(mc))
        x_new = refine_single(C[bi], Xref, steps=120, beta=50.0, lr=0.08)
        new_pts.append(x_new)
    Xn = Xcur.copy()
    Xn[victims] = np.array(new_pts)
    # tiny shake of keepers to break symmetry
    Xn = Xn + 0.008 * rng.normal(size=Xn.shape)
    return _norm(Xn)

def void_random_proposal(Xcur, rng, k_move, n_sample=8000):
    n, d = Xcur.shape
    victims = rng.choice(n, size=k_move, replace=False)
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[victims] = False
    X_keep = Xcur[keep_mask]
    new_pts = []
    for s in range(k_move):
        Xref = X_keep if len(new_pts) == 0 else np.vstack([X_keep, np.array(new_pts)])
        C = rng.normal(size=(n_sample, d))
        C = C / np.linalg.norm(C, axis=1, keepdims=True)
        D = C @ Xref.T
        mc = D.max(axis=1)
        bi = int(np.argmin(mc))
        x_new = refine_single(C[bi], Xref, steps=100, beta=50.0, lr=0.08)
        new_pts.append(x_new)
    Xn = Xcur.copy()
    Xn[victims] = np.array(new_pts)
    Xn = Xn + 0.01 * rng.normal(size=Xn.shape)
    return _norm(Xn)

def solve():
    t0 = time.time()
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        cands.append(_norm(xb))
    if xr is not None:
        cands.append(_norm(xr))
    if len(cands) == 0:
        bestX = _norm(rng.normal(size=(168, 4)))
        best = max_cos(bestX)
    else:
        vals = [max_cos(c) for c in cands]
        bi = int(np.argmin(vals))
        bestX = cands[bi].copy()
        best = float(vals[bi])
    n, d = 168, 4
    curX = bestX.copy()
    cur_val = best
    HOP_END = 145.0
    it = 0
    while True:
        if time.time() - t0 > HOP_END:
            break
        it += 1
        mode = it % 10
        if mode == 0:
            Xtry = _norm(rng.normal(size=(n, d)))
            steps, b0, b1e, lr0, lr1 = 1500, 20.0, 250.0, 0.02, 0.002
        elif mode == 1 or mode == 6:
            k = 2 + (it % 3)
            Xtry = surgical_proposal(curX, rng, k_move=k, n_sample=6000)
            steps, b0, b1e, lr0, lr1 = 1200, 25.0, 300.0, 0.015, 0.0015
        elif mode == 2 or mode == 7:
            k = 4 + (it % 4)
            Xtry = void_random_proposal(curX, rng, k_move=k, n_sample=6000)
            steps, b0, b1e, lr0, lr1 = 1200, 25.0, 280.0, 0.015, 0.0015
        elif mode == 3:
            sigma = 0.30
            Xtry = _norm(curX + sigma * rng.normal(size=(n, d)))
            k = 8
            idx = rng.choice(n, size=k, replace=False)
            Xtry[idx] = _norm(rng.normal(size=(k, d)))
            steps, b0, b1e, lr0, lr1 = 1400, 20.0, 260.0, 0.018, 0.0018
        else:
            sig = [0.015, 0.05, 0.12][it % 3]
            Xtry = _norm(curX + sig * rng.normal(size=(n, d)))
            if it % 4 == 0:
                k = 2
                idx = rng.choice(n, size=k, replace=False)
                Xtry[idx] = _norm(rng.normal(size=(k, d)))
            if it % 2 == 0:
                steps, b0, b1e, lr0, lr1 = 900, 30.0, 220.0, 0.02, 0.002
            else:
                steps, b0, b1e, lr0, lr1 = 1100, 20.0, 300.0, 0.012, 0.0012
        try:
            Xopt = gd_refine(Xtry, steps, b0, b1e, lr0, lr1)
        except Exception:
            continue
        v = max_cos(Xopt)
        # basin-hopping acceptance for cur
        if v < best - 1e-12:
            # polish immediately
            pt = 1.5
            if time.time() - t0 < HOP_END - 3.0:
                try:
                    Xp = lp_polish(Xopt, pt)
                    vp = max_cos(Xp)
                except Exception:
                    Xp, vp = Xopt, v
                if vp < v:
                    Xopt, v = Xp, vp
            best = v
            bestX = Xopt.copy()
            curX = Xopt.copy()
            cur_val = v
        elif v < cur_val + 0.004 and rng.random() < 0.35:
            curX = Xopt.copy()
            cur_val = v
            # occasional polish of near-best wandering point
            if v < best + 0.0015 and (it % 5 == 0):
                if time.time() - t0 < HOP_END - 3.0:
                    try:
                        Xp = lp_polish(Xopt, 0.9)
                        vp = max_cos(Xp)
                    except Exception:
                        Xp, vp = Xopt, v
                    if vp < best:
                        best = vp
                        bestX = Xp.copy()
                        curX = Xp.copy()
                        cur_val = vp
        else:
            if v < best + 0.002 and (it % 12 == 0):
                if time.time() - t0 < HOP_END - 3.0:
                    try:
                        Xp = lp_polish(Xopt, 0.7)
                        vp = max_cos(Xp)
                    except Exception:
                        vp = v
                    if vp < best:
                        best = vp
                        bestX = Xp.copy() if vp < v else Xopt.copy()
                        curX = bestX.copy()
                        cur_val = best
        if time.time() - t0 > HOP_END:
            break
    rem = 178.0 - (time.time() - t0)
    if rem > 4.0:
        # strong final: smooth refine then LP
        try:
            bestX = gd_refine(bestX, 800, 40.0, 400.0, 0.008, 0.0008)
        except Exception:
            pass
        rem2 = 178.0 - (time.time() - t0)
        if rem2 > 2.0:
            try:
                bestX = lp_polish(bestX, min(rem2 - 0.5, 25.0))
            except Exception:
                pass
    bestX = _norm(bestX)
    return bestX
