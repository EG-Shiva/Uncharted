import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n < 1e-16] = 1.0
    return X / n

def _adam_lse(X0, betas, lrs, steps_list, deadline=1e18):
    X = _norm(np.array(X0, dtype=float, copy=True))
    n = X.shape[0]
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    b1p = 1.0
    b2p = 1.0
    bestv = _max_cos(X)
    best = X.copy()
    for beta, lr, steps in zip(betas, lrs, steps_list):
        for _ in range(steps):
            t += 1
            if (t & 255) == 0 and time.time() > deadline:
                return best, bestv
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
            mhat = m1 / (1.0 - b1p)
            vhat = m2 / (1.0 - b2p)
            upd = mhat / (np.sqrt(vhat) + eps)
            dn = np.sum(upd * X, axis=1, keepdims=True)
            upd = upd - dn * X
            X = X - lr * upd
            X = _norm(X)
            if (t % 200) == 0:
                v = _max_cos(X)
                if v < bestv:
                    bestv = v
                    best = X.copy()
    v = _max_cos(X)
    if v < bestv:
        bestv = v
        best = X.copy()
    return best, bestv

def _single_refine(y0, Others, beta=500.0, iters=220, lr=0.03):
    y = np.asarray(y0, dtype=float).copy()
    nn = float(np.linalg.norm(y))
    if not np.finite(nn) or nn < 1e-16:
        y = Others[0].copy()
        nn = float(np.linalg.norm(y))
    y = y / nn
    m1 = np.zeros_like(y)
    m2 = np.zeros_like(y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best_y = y.copy()
    best_s = float(np.max(Others @ y))
    for t in range(1, iters + 1):
        dots = Others @ y
        m0 = float(np.max(dots))
        W = np.exp(beta * (dots - m0))
        S = float(np.sum(W))
        if not np.isfinite(S) or S <= 0:
            break
        grad = (W @ Others) / S
        gt = grad - float(grad @ y) * y
        m1 = b1 * m1 + (1.0 - b1) * gt
        m2 = b2 * m2 + (1.0 - b2) * (gt * gt)
        c1 = 1.0 - b1 ** t
        c2 = 1.0 - b2 ** t
        st = (m1 / c1) / (np.sqrt(m2 / c2) + eps)
        st = st - float(st @ y) * y
        lrt = lr / (1.0 + t / 250.0)
        y = y - lrt * st
        nn = float(np.linalg.norm(y))
        if not np.finite(nn) or nn < 1e-16:
            break
        y = y / nn
        if (t % 40) == 0 or t == iters:
            s = float(np.max(Others @ y))
            if s < best_s:
                best_s = s
                best_y = y.copy()
    s2 = float(np.max(Others @ y))
    if s2 < best_s:
        return y, s2
    return best_y, best_s

def _hole_once(X, bestv, rng, n_cand=2500):
    n, d = X.shape
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    rowmax = np.max(G, axis=1)
    order = np.argsort(-rowmax)[:6]
    R = _norm(rng.standard_normal(size=(n_cand, d)))
    for idx in order:
        idx = int(idx)
        Others = np.delete(X, idx, axis=0)
        sc = R @ Others.T
        smax = np.max(sc, axis=1)
        o = np.argsort(smax)
        improved = False
        for kk in range(min(2, n_cand)):
            y0 = R[o[kk]]
            y, _ = _single_refine(y0, Others, beta=500.0, iters=220, lr=0.04)
            Y = X.copy()
            Y[idx] = y
            v = _max_cos(Y)
            if v < bestv - 1e-12:
                return Y, v
        y0 = X[idx].copy()
        y, _ = _single_refine(y0, Others, beta=700.0, iters=220, lr=0.02)
        Y = X.copy()
        Y[idx] = y
        v = _max_cos(Y)
        if v < bestv - 1e-12:
            return Y, v
    return None, bestv

def _lp_jam(X0, bestv, seconds, deadline):
    X = _norm(X0.copy())
    best = bestv
    bestX = X.copy()
    n, d = X.shape
    iu = np.triu_indices(n, 1)
    radius = 5e-4
    start = time.time()
    Pcap = 1300
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 5e-15:
            break
        G = X @ X.T
        c = G[iu]
        m = float(np.max(c))
        # take top Pcap pairs
        if c.size > Pcap:
            kth = np.argpartition(-c, Pcap)[:Pcap]
            thr = float(np.min(c[kth]))
            sel = c >= thr
        else:
            sel = np.ones_like(c, dtype=bool)
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        if p < 4:
            radius *= 0.5
            continue
        csel = c[sel]
        rows = np.repeat(np.arange(p), 2 * d + 1)
        ci = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, ci)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(n * d + 1)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        dx = res.x[:-1].reshape(n, d)
        improved = False
        for alpha in (1.0, 0.5, 0.25, 0.125):
            Y = _norm(X + alpha * dx)
            v = _max_cos(Y)
            if v < best - 1e-12:
                best = v
                bestX = Y.copy()
                X = Y
                radius = min(radius * 1.5, 2e-3)
                improved = True
                break
        if not improved:
            radius *= 0.5
    return bestX, best

def solve():
    t0 = time.time()
    deadline = t0 + 168.0
    rng = np.random.default_rng(11)
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
    cfgs = [
        ([60.0, 160.0, 350.0, 650.0], [0.006, 0.003, 0.0015, 0.0008], [320, 320, 380, 380]),
        ([40.0, 130.0, 300.0, 600.0], [0.007, 0.0035, 0.0018, 0.0009], [300, 320, 360, 360]),
        ([100.0, 260.0, 550.0], [0.004, 0.002, 0.001], [320, 380, 420]),
    ]
    sigmas = [0.005, 0.012, 0.025, 0.007, 0.04, 0.010, 0.018, 0.004]
    trial = 0
    # Phase 1: basin hops until ~110s
    while time.time() < t0 + 108.0 and time.time() < deadline - 55.0:
        trial += 1
        if trial > 300:
            break
        try:
            if trial % 9 == 0:
                Y, v = _hole_once(Xbest, bestv, rng, n_cand=2200)
                if Y is not None and v < bestv:
                    Xbest, bestv = Y, v
                continue
            sigma = sigmas[(trial - 1) % len(sigmas)]
            cfg = cfgs[(trial - 1) % len(cfgs)]
            betas, lrs, stepsl = cfg
            if trial % 5 == 0:
                G = Xbest @ Xbest.T
                np.fill_diagonal(G, -2.0)
                rowmax = np.max(G, axis=1)
                worst = np.argsort(-rowmax)[:12]
                pert = rng.standard_normal(size=Xbest.shape) * sigma * 0.5
                pert[worst] += rng.standard_normal(size=(12, 9)) * 0.05
                X0 = _norm(Xbest + pert)
            else:
                X0 = _norm(Xbest + sigma * rng.standard_normal(size=Xbest.shape))
            lim = min(deadline - 55.0, t0 + 108.0)
            X1, v1 = _adam_lse(X0, betas, lrs, stepsl, deadline=lim)
            if v1 < bestv - 1e-12:
                Xbest, bestv = X1, v1
        except Exception:
            continue
    # Phase 2: LP jamming
    if time.time() < deadline - 8.0:
        rem = deadline - time.time() - 6.0
        if rem > 6.0:
            lp_t = min(rem * 0.7, 42.0)
            try:
                Xlp, vlp = _lp_jam(Xbest, bestv, seconds=lp_t, deadline=time.time() + lp_t)
                if vlp < bestv:
                    Xbest, bestv = Xlp, vlp
            except Exception:
                pass
    # Phase 3: single-point sweep on worst points
    if time.time() < deadline - 3.0:
        try:
            for _ in range(2):
                if time.time() > deadline - 2.5:
                    break
                G = Xbest @ Xbest.T
                np.fill_diagonal(G, -2.0)
                rowmax = np.max(G, axis=1)
                order = np.argsort(-rowmax)[:14]
                for idx in order:
                    if time.time() > deadline - 2.5:
                        break
                    idx = int(idx)
                    Others = np.delete(Xbest, idx, axis=0)
                    y0 = Xbest[idx].copy()
                    y, _ = _single_refine(y0, Others, beta=800.0, iters=200, lr=0.015)
                    Y = Xbest.copy()
                    Y[idx] = y
                    v = _max_cos(Y)
                    if v < bestv - 1e-12:
                        Xbest = Y
                        bestv = v
        except Exception:
            pass
    return _norm(Xbest)
