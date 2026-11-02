import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds, init_radius=1e-3):
    if seconds <= 0.05:
        return x
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(init_radius)
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        # time guard for linprog itself
        if time.time() - start > seconds - 0.05:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) == 0:
            break
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
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
    return x


def _adam_push(X, target, steps, lr=0.004):
    n, d = X.shape
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    b1p = 1.0
    b2p = 1.0
    for s in range(1, steps + 1):
        G = X @ X.T
        D = G - target
        np.fill_diagonal(D, -1.0)
        np.maximum(D, 0, out=D)
        if D.max() <= 0:
            target -= 0.0005
            # tiny random tangent drift to keep moving when feasible
            # use zero grad this step
            b1p *= b1
            b2p *= b2
            continue
        Gr = (D @ X) * 2.0
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gr -= dot * X
        m1 *= b1
        m1 += (1 - b1) * Gr
        v1 *= b2
        v1 += (1 - b2) * (Gr * Gr)
        b1p *= b1
        b2p *= b2
        mhat = m1 / (1.0 - b1p)
        vhat = v1 / (1.0 - b2p)
        X = X - lr * mhat / (np.sqrt(vhat) + eps)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X, target


def _counts(X, best, thr=0.005):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return np.sum(G > (best - thr), axis=1)


def solve():
    t0 = time.time()
    deadline = t0 + 175.0
    rng = np.random.default_rng(0)
    # load start
    try:
        Xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            Xb = np.load("record.npy").astype(float)
        except Exception:
            Xb = rng.standard_normal((76, 15))
    if Xb.shape != (76, 15):
        # fallback: truncate / pad deterministically
        tmp = rng.standard_normal((76, 15))
        r = min(Xb.shape[0], 76)
        c = min(Xb.shape[1], 15)
        tmp[:r, :c] = Xb[:r, :c]
        Xb = tmp
    Xb = Xb / np.linalg.norm(Xb, axis=1, keepdims=True)
    best = _max_cos(Xb)

    n, d = 76, 15
    # main basin-hopping loop, reserve ~28s for final polish
    for trial in range(80):
        if time.time() > deadline - 30.0:
            break
        mode = trial % 5
        if mode == 0 or mode == 1:
            sigma = 0.10 if mode == 0 else 0.03
            Xt = Xb + sigma * rng.standard_normal((n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            steps = 1600
        elif mode == 2:
            cnt = _counts(Xb, best, thr=0.005)
            k = 4
            idx = np.argsort(cnt)[-k:]
            Xt = Xb.copy()
            Xt[idx] = rng.standard_normal((k, d))
            Xt[idx] /= np.linalg.norm(Xt[idx], axis=1, keepdims=True)
            Xt = Xt + 0.01 * rng.standard_normal((n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            steps = 1500
        elif mode == 3:
            cnt = _counts(Xb, best, thr=0.005)
            k = 5
            idx = np.argsort(cnt)[:k]
            Xt = Xb.copy()
            for ii in idx:
                cand = rng.standard_normal((1500, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                dots = cand @ Xt.T
                dots[:, ii] = -1.0
                scores = dots.max(axis=1)
                Xt[ii] = cand[np.argmin(scores)]
            steps = 1300
        else:
            cnt = _counts(Xb, best, thr=0.008)
            k = 7
            idx = np.argsort(cnt)[:k]
            # mix low-contact and one high-contact
            idx2 = np.argsort(cnt)[-1:]
            Xt = Xb.copy()
            for ii in np.concatenate([idx, idx2]):
                cand = rng.standard_normal((1200, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                dots = cand @ Xt.T
                dots[:, ii] = -1.0
                scores = dots.max(axis=1)
                Xt[ii] = cand[np.argmin(scores)]
            Xt = Xt + 0.005 * rng.standard_normal((n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            steps = 1500

        target = best - 0.002 - 0.002 * float(rng.random())
        Xt, _ = _adam_push(Xt, target, steps, lr=0.004)
        v = _max_cos(Xt)
        if v < best + 0.015:
            rem = (deadline - 30.0) - time.time()
            if rem < 0.6:
                break
            secs = min(1.2, rem)
            Xp = _lp_polish(Xt, secs, init_radius=0.005)
            vp = _max_cos(Xp)
            if vp < best:
                Xb = Xp
                best = vp
        # early exit if we already beat record comfortably
        # continue hopping to push further

    rem = deadline - 2.0 - time.time()
    if rem > 0.5:
        Xb = _lp_polish(Xb, min(rem, 28.0), init_radius=1e-3)
    Xb = Xb / np.linalg.norm(Xb, axis=1, keepdims=True)
    return Xb
