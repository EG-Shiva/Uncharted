import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _smooth_descend(X0, iters, beta0, beta1, lr):
    X = X0.copy()
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    lb0 = np.log(beta0)
    lb1 = np.log(beta1)
    b1p = 1.0
    b2p = 1.0
    for t in range(1, iters + 1):
        if iters > 1:
            beta = np.exp(lb0 + (lb1 - lb0) * (t - 1) / (iters - 1))
        else:
            beta = beta0
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        E = np.exp(beta * (G - gmax))
        np.fill_diagonal(E, 0.0)
        S = E.sum()
        if S == 0:
            S = 1.0
        Grad = (E @ X) / S
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        b1p *= b1
        b2p *= b2
        m = b1 * m + (1.0 - b1) * Gt
        v = b2 * v + (1.0 - b2) * (Gt * Gt)
        mhat = m / (1.0 - b1p)
        vhat = v / (1.0 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        X = X - step
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X


def _refine_hole(y0, X, beta=250.0, iters=250, lr=0.01):
    y = y0.copy() / np.linalg.norm(y0)
    d = y.shape[0]
    m = np.zeros(d)
    v = np.zeros(d)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    b1p = 1.0
    b2p = 1.0
    for t in range(1, iters + 1):
        s = X @ y
        smax = s.max()
        E = np.exp(beta * (s - smax))
        S = E.sum()
        if S == 0:
            S = 1.0
        grad = (E @ X) / S
        gt = grad - np.dot(grad, y) * y
        b1p *= b1
        b2p *= b2
        m = b1 * m + (1.0 - b1) * gt
        v = b2 * v + (1.0 - b2) * (gt * gt)
        mhat = m / (1.0 - b1p)
        vhat = v / (1.0 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        y = y - step
        nrm = np.linalg.norm(y)
        if nrm == 0:
            nrm = 1.0
        y = y / nrm
    return y


def _find_hole(X, rng):
    n, d = X.shape
    M = 3000
    Y = rng.standard_normal(size=(M, d))
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    scores = (Y @ X.T).max(axis=1)
    idx = np.argsort(scores)[:3]
    best_y = Y[idx[0]]
    best_s = scores[idx[0]]
    for k in idx:
        y0 = Y[k]
        y = _refine_hole(y0, X, beta=250.0, iters=250, lr=0.01)
        s = float((X @ y).max())
        if s < best_s:
            best_s = s
            best_y = y
    return best_y / np.linalg.norm(best_y)


def _lp_polish(x, seconds, t0, t_end):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            radius *= 0.5
            continue
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    t_end = t0 + 172.0
    rng = np.random.default_rng(7)
    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            x0 = rng.standard_normal(size=(176, 9))
    x0 = np.asarray(x0, dtype=np.float64)
    if x0.shape != (176, 9):
        # fallback: tile/truncate to required shape
        tmp = rng.standard_normal(size=(176, 9))
        r = min(x0.shape[0], 176)
        tmp[:r, :min(x0.shape[1], 9)] = x0[:r, :min(x0.shape[1], 9)]
        x0 = tmp
    x0 = x0 / np.linalg.norm(x0, axis=1, keepdims=True)
    bestX = x0.copy()
    bestV = _max_cos(bestX)
    curX = bestX.copy()
    curV = bestV
    T = 0.0025
    sigmas = [0.09, 0.045, 0.022, 0.011]
    # fixed number of basin cycles, time-guarded
    NCYC = 70
    for cyc in range(NCYC):
        if time.time() > t_end - 55.0:
            break
        sigma = sigmas[cyc % 4]
        noise = rng.standard_normal(size=curX.shape)
        # tangent projection to keep moves on sphere
        pr = np.sum(noise * curX, axis=1, keepdims=True)
        noise = noise - pr * curX
        Xtry = curX + sigma * noise
        Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        # annealed smooth descent, parameters vary slightly per cycle
        if cyc % 4 == 0:
            Xopt = _smooth_descend(Xtry, 550, 25.0, 700.0, 0.012)
        elif cyc % 4 == 1:
            Xopt = _smooth_descend(Xtry, 550, 40.0, 900.0, 0.008)
        elif cyc % 4 == 2:
            Xopt = _smooth_descend(Xtry, 500, 30.0, 600.0, 0.010)
        else:
            Xopt = _smooth_descend(Xtry, 450, 20.0, 500.0, 0.012)
        Vopt = _max_cos(Xopt)
        if Vopt < bestV:
            bestV = Vopt
            bestX = Xopt.copy()
            curX = Xopt.copy()
            curV = Vopt
        elif Vopt < curV or rng.random() < np.exp((curV - Vopt) / max(T, 1e-12)):
            curX = Xopt.copy()
            curV = Vopt
        T *= 0.997
        if T < 0.0004:
            T = 0.0004
        # periodic kill-and-fill of worst point
        if cyc % 10 == 9 and time.time() < t_end - 60.0:
            try:
                G = bestX @ bestX.T
                np.fill_diagonal(G, -2.0)
                rowmax = G.max(axis=1)
                worst = int(np.argmax(rowmax))
                hole = _find_hole(bestX, rng)
                Xtry2 = bestX.copy()
                Xtry2[worst] = hole
                Xopt2 = _smooth_descend(Xtry2, 450, 30.0, 700.0, 0.010)
                V2 = _max_cos(Xopt2)
                if V2 < bestV:
                    bestV = V2
                    bestX = Xopt2.copy()
                    curX = Xopt2.copy()
                    curV = V2
                elif V2 < curV:
                    curX = Xopt2.copy()
                    curV = V2
            except Exception:
                pass
    # final trust-region LP polish with remaining budget
    remain = t_end - time.time()
    if remain > 5.0:
        # leave 2s margin
        bestX = _lp_polish(bestX, remain - 2.0, time.time(), t_end - 1.0)
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
