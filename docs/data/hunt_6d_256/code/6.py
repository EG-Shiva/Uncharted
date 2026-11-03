import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def adam_refine(y, Ks=(40.0, 120.0, 400.0), steps=(120, 120, 150), lrs=(0.015, 0.008, 0.004)):
    n, d = y.shape
    y = y.copy()
    yn = np.linalg.norm(y, axis=1, keepdims=True)
    yn = np.maximum(yn, 1e-12)
    y = y / yn
    m = np.zeros_like(y)
    v = np.zeros_like(y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    for K, S, lr in zip(Ks, steps, lrs):
        for _ in range(S):
            t += 1
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            gmax = G.max()
            # softmax weights, G-gmax <= 0 so no overflow
            E = np.exp(K * (G - gmax))
            np.fill_diagonal(E, 0.0)
            s = E.sum()
            if s == 0 or not np.isfinite(s):
                continue
            E /= s
            grad = E @ y
            dot = (grad * y).sum(axis=1, keepdims=True)
            g = grad - dot * y
            m[:] = b1 * m + (1.0 - b1) * g
            v[:] = b2 * v + (1.0 - b2) * g * g
            mhat = m / (1.0 - b1 ** t)
            vhat = v / (1.0 - b2 ** t)
            upd = mhat / (np.sqrt(vhat) + eps)
            udot = (upd * y).sum(axis=1, keepdims=True)
            upd = upd - udot * y
            y = y - lr * upd
            nrm = np.linalg.norm(y, axis=1, keepdims=True)
            nrm = np.maximum(nrm, 1e-12)
            y /= nrm
    return y


def lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = c.max()
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        I = iu[0][sel]
        J = iu[1][sel]
        csel = c[sel]
        p = len(I)
        if p == 0:
            radius *= 1.5
            continue
        if p > 4000:
            # keep tightest 4000
            part = np.argpartition(-csel, 4000)[:4000]
            I = I[part]
            J = J[part]
            csel = csel[part]
            p = 4000
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, n * d + 1)).tocsr()
        except Exception:
            break
        obj = np.zeros(n * d + 1)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nrm = np.linalg.norm(y, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        y = y / nrm
        vv = max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.RandomState(0)
    np.random.seed(0)

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (256, 6) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                nr = np.maximum(nr, 1e-12)
                a = a / nr
                cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        a = rng.randn(256, 6)
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    best = cands[bi].copy()
    best_val = float(vals[bi])
    n, d = best.shape

    trial = 0
    # main basin-hopping loop, reserve ~25s for final polish
    while time.time() < deadline - 25.0 and trial < 150:
        trial += 1
        cand = best.copy()
        mode = trial % 6
        if mode == 0 or mode == 1:
            sigma = 0.015 if mode == 0 else 0.04
            cand = cand + sigma * rng.randn(n, d)
        elif mode == 2:
            k = 25
            idx = rng.choice(n, k, replace=False)
            cand[idx] = cand[idx] + 0.25 * rng.randn(k, d)
        elif mode == 3:
            k = 10
            idx = rng.choice(n, k, replace=False)
            cand[idx] = cand[idx] + 0.50 * rng.randn(k, d)
        elif mode == 4:
            # targeted: reinit rattlers, shake overloaded hubs
            try:
                G = best @ best.T
                np.fill_diagonal(G, -2.0)
                mm = G.max()
                deg = (G > mm - 1e-7).sum(axis=1)
                ratt = np.where(deg == 0)[0]
                over = np.argsort(-deg)[:12]
            except Exception:
                ratt = np.array([], dtype=int)
                over = rng.choice(n, 12, replace=False)
            if len(ratt) > 0:
                cand[ratt] = rng.randn(len(ratt), d)
            cand[over] = cand[over] + 0.20 * rng.randn(len(over), d)
            cand = cand + 0.01 * rng.randn(n, d)
        else:
            idx = rng.choice(n, 4, replace=False)
            cand[idx] = rng.randn(4, d)
            cand = cand + 0.01 * rng.randn(n, d)
        nrm = np.linalg.norm(cand, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        cand /= nrm

        cand = adam_refine(cand)
        v = max_cos(cand)
        if v < best_val:
            # squeeze with short LP if clearly better or marginally better
            if v < best_val - 2e-6:
                if time.time() < deadline - 30.0:
                    cand2 = lp_polish(cand, min(deadline - 20.0, time.time() + 4.0))
                    v2 = max_cos(cand2)
                    if v2 < v:
                        cand, v = cand2, v2
            else:
                if time.time() < deadline - 30.0:
                    cand2 = lp_polish(cand, min(deadline - 20.0, time.time() + 2.0))
                    v2 = max_cos(cand2)
                    if v2 < v:
                        cand, v = cand2, v2
            if v < best_val:
                best, best_val = cand, v
        # else discard

    if time.time() < deadline - 1.5:
        try:
            best = lp_polish(best, deadline - 1.0)
        except Exception:
            pass
    best = best / np.maximum(np.linalg.norm(best, axis=1, keepdims=True), 1e-12)
    return best
