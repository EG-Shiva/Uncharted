import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _rowmax_cnts(X, tol=2e-4):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    rowmax = G.max(axis=1)
    cnts = (G > m - tol).sum(axis=1)
    return G, m, rowmax, cnts

def _best_void(X, exclude, rng, nc=3000):
    n, d = X.shape
    cand = rng.standard_normal((nc, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    Gc = cand @ X.T
    if exclude is not None:
        Gc[:, exclude] = -2.0
    s = Gc.max(axis=1)
    bi = int(np.argmin(s))
    return cand[bi], float(s[bi])

def _adam_lse(X0, betas, iters_list, lr_list, rng=None):
    n, d = X0.shape
    Y = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
    iu = np.triu_indices(n, 1)
    for beta, iters, lr0 in zip(betas, iters_list, lr_list):
        m1 = np.zeros_like(Y)
        v1 = np.zeros_like(Y)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for t in range(1, iters + 1):
            G = Y @ Y.T
            gv = G[iu]
            m = float(gv.max())
            e = np.exp(beta * (gv - m))
            S = float(e.sum())
            w = e / S
            W = np.zeros((n, n), dtype=np.float64)
            W[iu] = w
            W += W.T
            GY = W @ Y
            dot = (GY * Y).sum(axis=1, keepdims=True)
            Gt = GY - dot * Y  # dF/dY tangent
            # Adam update (descent)
            m1 = b1 * m1 + (1 - b1) * Gt
            v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
            mhat = m1 / (1 - b1 ** t)
            vhat = v1 / (1 - b2 ** t)
            step = mhat / (np.sqrt(vhat) + eps)
            # decay
            lr = lr0 / (1.0 + 0.004 * t)
            Y = Y - lr * step
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def _lp_polish(X, deadline):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gfull = X @ X.T
        c = Gfull[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 2500:
            # keep tightest only to keep LP small
            cc_all = c[sel]
            k = 2500
            idx = np.argpartition(-cc_all, k - 1)[:k]
            I = I[idx]; J = J[idx]
            p = k
        cc = (X[I] * X[J]).sum(axis=1)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    HARD = 174.0
    rng = np.random.default_rng(12345)
    n, d = 78, 15
    bestX = None; bestV = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape != (78, 15):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            v = _max_cos(a)
            if v < bestV:
                bestV = v; bestX = a.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)
    # quick snap
    try:
        if time.time() - t0 < HARD - 30:
            bestX = _lp_polish(bestX, min(t0 + 2.5, t0 + HARD - 25))
            bestV = _max_cos(bestX)
    except Exception:
        pass

    it = 0
    stall = 0
    while time.time() - t0 < HARD - 12:
        it += 1
        tl = HARD - 12 - (time.time() - t0)
        if tl < 2.0:
            break
        try:
            G, mcur, rowmax, cnts = _rowmax_cnts(bestX, tol=2e-4)
        except Exception:
            cnts = np.zeros(n); rowmax = np.zeros(n); mcur = bestV
        # order loose first: few contacts, then large gap (small rowmax is looser but keep stable)
        order_loose = np.lexsort((rowmax, cnts))  # small cnts first
        order_hi = np.argsort(-cnts, kind='stable')
        Xtry = bestX.copy()
        mode = it % 6
        if mode == 5:
            # global shake (rare)
            sig = 0.08 + 0.06 * rng.random()
            Xtry = bestX + sig * rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        else:
            if mode == 0: k = 1
            elif mode == 1: k = 1
            elif mode == 2: k = 2
            elif mode == 3: k = 2
            else: k = 3
            if stall > 15:
                k = min(4, k + 1)
            pool = order_loose[:20]
            # prefer points with small cnts; sample k without replacement
            chosen = rng.choice(pool, size=min(k, len(pool)), replace=False)
            # occasionally include the top hub instead of a loose point
            if mode == 4 and rng.random() < 0.4:
                chosen = np.append(chosen[:max(1, k-1)], order_hi[0])
            for idx in chosen:
                idx = int(idx)
                v0, _ = _best_void(Xtry, idx, rng, nc=3000)
                # diversity: sometimes second try
                if rng.random() < 0.2:
                    v1, s1 = _best_void(Xtry, idx, rng, nc=3000)
                    # keep random among them
                    Xtry[idx] = v1 if rng.random() < 0.5 else v0
                else:
                    Xtry[idx] = v0
            # nudge overloaded hub tangentially
            try:
                hub = int(order_hi[0])
                if cnts[hub] > 20 and rng.random() < 0.7:
                    Xtry[hub] = Xtry[hub] + rng.standard_normal(d) * 0.025
                    Xtry[hub] /= np.linalg.norm(Xtry[hub])
                # tiny jitter
                Xtry = Xtry + rng.standard_normal((n, d)) * 0.005
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            except Exception:
                pass
        # fast anneal quench (manual Adam, cheap)
        try:
            # vary schedule slightly per iteration for diversity
            if it % 2 == 0:
                Xsm = _adam_lse(Xtry, betas=[90.0, 280.0, 700.0],
                                iters_list=[90, 90, 70], lr_list=[0.02, 0.012, 0.008])
            else:
                Xsm = _adam_lse(Xtry, betas=[140.0, 450.0, 900.0],
                                iters_list=[70, 70, 60], lr_list=[0.025, 0.014, 0.007])
        except Exception:
            stall += 1
            continue
        try:
            vq = _max_cos(Xsm)
        except Exception:
            stall += 1
            continue
        # LP squeeze only if promising to save time
        Xpol = Xsm
        if vq < bestV + 0.004 and tl > 4.0:
            try:
                bdg = min(0.7, max(0.3, tl * 0.08))
                Xpol = _lp_polish(Xsm, time.time() + bdg)
            except Exception:
                Xpol = Xsm
        try:
            v = _max_cos(Xpol)
        except Exception:
            stall += 1
            continue
        if v < bestV - 1e-12:
            bestX = Xpol; bestV = v; stall = 0
        else:
            stall += 1
            # keep searching; if close, try extra LP-free refinement occasionally
            if stall > 30 and rng.random() < 0.1:
                # random restart from best with bigger kick handled next iter
                stall = 0

    # final intensive polish
    try:
        rem = HARD - (time.time() - t0) - 1.0
        if rem > 1.5:
            bestX = _lp_polish(bestX, time.time() + min(rem * 0.6, 7.0))
            bestV = _max_cos(bestX)
            rem2 = HARD - (time.time() - t0) - 0.5
            if rem2 > 2.0:
                try:
                    Xsm = _adam_lse(bestX, betas=[500.0], iters_list=[80], lr_list=[0.008])
                    Xpol = _lp_polish(Xsm, time.time() + min(2.5, rem2 - 0.5))
                    if _max_cos(Xpol) < bestV:
                        bestX = Xpol
                except Exception:
                    pass
    except Exception:
        pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    bestX = np.where(np.isnan(bestX), 0.0, bestX)
    nr = np.linalg.norm(bestX, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-12)
    bestX = bestX / nr
    return bestX
