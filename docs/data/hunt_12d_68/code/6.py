import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _lp_polish(x, seconds, t_start=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    if t_start is None:
        t_start = time.time()
    iu = np.triu_indices(n, 1)
    while (time.time() - t_start < seconds) and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(np.max(c))
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
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
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best


def _softmax_anneal(X0, steps, beta0, beta1, lr0, lr1, mu=0.9):
    X = X0.copy()
    n, d = X.shape
    vel = np.zeros_like(X)
    # track best via G max (no extra matmul)
    G0 = X @ X.T
    np.fill_diagonal(G0, -np.inf)
    best_v = float(np.max(G0))
    best_X = X.copy()
    lb0 = np.log(beta0)
    lb1 = np.log(beta1)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        cur_lr = float(lr0 + (lr1 - lr0) * f)
        G = X @ X.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        # weights
        # E = exp(beta*(G-m)), diag -> 0
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E) * 0.5)
        if not np.isfinite(S) or S < 1e-300:
            S = 1.0
        Grad = (E @ X) / S
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        # clip extreme rows for stability
        # (optional) limit row norm to 5
        rn = np.sqrt(np.sum(Gt * Gt, axis=1, keepdims=True)) + 1e-12
        Gt = Gt / np.maximum(rn / 3.0, 1.0)
        vel = mu * vel + Gt
        X = X - cur_lr * vel
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        X /= nrm
        if (s % 100 == 0) or (s == steps - 1):
            G2 = X @ X.T
            np.fill_diagonal(G2, -np.inf)
            mv = float(np.max(G2))
            if mv < best_v:
                best_v = mv
                best_X = X.copy()
        # early nan guard
        if not np.all(np.isfinite(X)):
            break
    return best_X, best_v


def _hard_polish(X0, steps=400, lr0=0.008, lr1=0.0005):
    X = X0.copy()
    G0 = X @ X.T
    np.fill_diagonal(G0, -np.inf)
    best_v = float(np.max(G0))
    best_X = X.copy()
    for s in range(steps):
        f = s / max(steps - 1, 1)
        cur_lr = float(lr0 + (lr1 - lr0) * f)
        tol = float(0.01 * (1.0 - f) + 1e-4 * f)
        G = X @ X.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        if m < best_v:
            best_v = m
            best_X = X.copy()
        mask = G > (m - tol)
        # uniform weight over active pairs
        W = mask.astype(np.float64)
        np.fill_diagonal(W, 0.0)
        cnt = float(np.sum(W) * 0.5)
        if cnt < 1:
            continue
        Grad = (W @ X) / cnt
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        rn = np.sqrt(np.sum(Gt * Gt, axis=1, keepdims=True)) + 1e-12
        Gt = Gt / np.maximum(rn / 2.0, 1.0)
        X = X - cur_lr * Gt
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    Gf = X @ X.T
    np.fill_diagonal(Gf, -np.inf)
    vf = float(np.max(Gf))
    if vf < best_v:
        return X, vf
    return best_X, best_v


def solve():
    t0 = time.time()
    rng = np.random.default_rng(0)
    # load start point
    Xbest = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            X = np.load(fn)
            X = np.asarray(X, dtype=np.float64)
            if X.shape == (68, 12):
                X = X / np.linalg.norm(X, axis=1, keepdims=True)
                v = _max_cos(X)
                if v < best_val:
                    best_val = v
                    Xbest = X.copy()
        except Exception:
            pass
    if Xbest is None:
        Xbest = rng.standard_normal((68, 12))
        Xbest /= np.linalg.norm(Xbest, axis=1, keepdims=True)
        best_val = _max_cos(Xbest)
    n, d = 68, 12

    basin = 0
    # main basin-hopping loop, reserve ~20s for final LP
    while time.time() - t0 < 148:
        basin += 1
        r = rng.random()
        # contact counts for guided removal (threshold 0.01 below best)
        if r < 0.35:
            # full jitter, log-uniform sigma
            sigma = float(10.0 ** rng.uniform(-2.6, -0.7))
            X0 = Xbest + sigma * rng.standard_normal((n, d))
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        elif r < 0.75:
            # subset re-insertion, target jammed points
            k = int(rng.integers(2, 9))
            try:
                G = Xbest @ Xbest.T
                np.fill_diagonal(G, -2.0)
                thr = best_val - 0.015
                cnt = np.sum(G > thr, axis=1)
                # pick k with prob proportional to cnt
                probs = cnt.astype(np.float64) + 0.5
                probs /= probs.sum()
                idx = rng.choice(n, size=k, replace=False, p=probs)
            except Exception:
                idx = rng.choice(n, size=k, replace=False)
            X0 = Xbest.copy()
            R = rng.standard_normal((k, d))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            X0[idx] = R
        elif r < 0.90:
            # medium shake + random rotation of a subset? just larger jitter
            sigma = float(10.0 ** rng.uniform(-1.0, -0.3))
            X0 = Xbest + sigma * rng.standard_normal((n, d))
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        else:
            # fresh random start with short anneal (diversity)
            X0 = rng.standard_normal((n, d))
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)

        steps = int(rng.integers(800, 1800))
        beta0 = float(10.0 ** rng.uniform(1.2, 1.9))   # ~16-80
        beta1 = float(10.0 ** rng.uniform(2.4, 3.1))   # ~250-1250
        lr0 = float(10.0 ** rng.uniform(-2.1, -1.5))   # ~0.008-0.03
        lr1 = float(lr0 * float(rng.uniform(0.05, 0.3)))

        Xa, va = _softmax_anneal(X0, steps, beta0, beta1, lr0, lr1, mu=0.9)
        Xh, vh = _hard_polish(Xa, steps=350, lr0=0.007, lr1=0.0004)

        if vh < best_val - 1e-12:
            # promising: quick LP polish to get true local optimum (1.2s)
            if vh < best_val + 0.005:
                try:
                    Xp, vp = _lp_polish(Xh, seconds=1.2)
                except Exception:
                    Xp, vp = Xh, vh
            else:
                Xp, vp = Xh, vh
            if vp < best_val:
                Xbest = Xp
                best_val = vp
        # opportunistic early exit check on time
        if time.time() - t0 > 148:
            break
        # safety cap on basins
        if basin > 800:
            break

    # final squeeze of global best with remaining time
    remain = 178.0 - (time.time() - t0)
    if remain > 3:
        try:
            Xbest, best_val = _lp_polish(Xbest, seconds=min(remain, 25.0))
        except Exception:
            pass
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
