import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def fast_polish(X, steps, lr0, lr1, m0, m1, pw=1.0):
    n, d = X.shape
    for s in range(steps):
        f = s / max(1, steps - 1)
        lr = lr0 * ((lr1 / lr0) ** f) if lr0 > 0 and lr1 > 0 else lr0
        margin = m0 * (1 - f) + m1 * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = np.max(G)
        th = m - margin
        Mask = G >= th
        W = np.zeros_like(G)
        if margin > 1e-12:
            r = (G[Mask] - th) / margin
            r = np.clip(r, 0, 1)
            if pw != 1.0:
                r = r ** pw
            W[Mask] = 0.2 + 0.8 * r
        else:
            W[Mask] = 1.0
        np.fill_diagonal(W, 0.0)
        S = W @ X
        deg = np.sum(W, axis=1, keepdims=True)
        dot = np.sum(S * X, axis=1, keepdims=True)
        F = S - dot * X
        F /= np.maximum(1.0, deg)
        X = X - lr * F
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def refine_hole(Y_fixed, y0, steps=250):
    y = y0.copy()
    lr = 0.05
    for s in range(steps):
        dts = Y_fixed @ y
        m = np.max(dts)
        T = 0.02 * (1 - s / steps) + 0.002
        e = np.exp((dts - m) / T)
        w = e / np.sum(e)
        g = w @ Y_fixed
        g = g - (g @ y) * y
        gn = np.linalg.norm(g)
        if gn > 1e-12:
            g /= gn
        y = y - lr * g
        ny = np.linalg.norm(y)
        if ny < 1e-12:
            y = y0
        else:
            y /= ny
        lr *= 0.995
    return y

def hole_transplant(X, idx, rng, n_sample=10000, n_best=4):
    n, d = X.shape
    Y_fixed = np.delete(X, idx, axis=0)
    S = rng.standard_normal((n_sample, d))
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    worst = (S @ Y_fixed.T).max(axis=1)
    part = np.argpartition(worst, n_best)[:n_best]
    best_y = None
    best_v = 1e9
    for k in part:
        y = refine_hole(Y_fixed, S[k], steps=220)
        v = float(np.max(Y_fixed @ y))
        if v < best_v:
            best_v = v
            best_y = y
    X2 = X.copy()
    X2[idx] = best_y
    return X2

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

def solve():
    t0 = time.time()
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 131, 9
    best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v /= np.linalg.norm(v, axis=1, keepdims=True)
                c = max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal((n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = max_cos(best)

    def current_probs(X, mv):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        C = G >= (mv - 1e-7)
        deg = C.sum(axis=1).astype(float)
        w = (deg + 1.0) ** 1.5
        return w / w.sum(), deg

    probs, degs = current_probs(best, best_val)

    def worst_pair(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        k = int(np.argmax(G))
        return k // n, k % n

    it = 0
    # main hopping, leave ~35s for final LP
    while time.time() - t0 < 138:
        it += 1
        if it % 10 == 1:
            probs, degs = current_probs(best, best_val)
        cycle = it % 8
        Y = best.copy()
        transplant_done = False
        if cycle == 0:
            Y += 0.008 * rng.standard_normal(Y.shape)
        elif cycle == 1:
            Y += 0.022 * rng.standard_normal(Y.shape)
        elif cycle == 2:
            k, sig = 2, 0.10
            idx = rng.choice(n, k, replace=False, p=probs)
            Y[idx] += sig * rng.standard_normal((k, d))
        elif cycle == 3:
            # hub -> largest hole transplant
            hub = int(np.argmax(degs + rng.random(n) * 2.0))
            try:
                Y = hole_transplant(best, hub, rng, n_sample=8000, n_best=4)
            except Exception:
                Y = best.copy()
                Y[hub] += 0.15 * rng.standard_normal(d)
            Y += 0.004 * rng.standard_normal(Y.shape)
            transplant_done = True
        elif cycle == 4:
            # worst-pair point -> hole
            try:
                a, b = worst_pair(best)
                tgt = a if degs[a] >= degs[b] else b
                # sometimes pick random hub instead
                if rng.random() < 0.3:
                    tgt = int(rng.choice(n, p=probs))
                Y = hole_transplant(best, tgt, rng, n_sample=8000, n_best=4)
            except Exception:
                pass
            Y += 0.004 * rng.standard_normal(Y.shape)
            transplant_done = True
        elif cycle == 5:
            k, sig = 1, 0.35
            idx = rng.choice(n, k, replace=False, p=probs)
            Y[idx] += sig * rng.standard_normal((k, d))
        elif cycle == 6:
            # rattler random reinsert + hub kick
            ratt = np.where(degs < 0.5)[0]
            if len(ratt) > 0:
                k = min(len(ratt), int(rng.integers(1, 4)))
                idx = rng.choice(ratt, k, replace=False)
                R = rng.standard_normal((k, d))
                R /= np.linalg.norm(R, axis=1, keepdims=True)
                Y[idx] = R
            hid = rng.choice(n, 1, replace=False, p=probs)
            Y[hid] += 0.12 * rng.standard_normal((1, d))
            Y += 0.008 * rng.standard_normal(Y.shape)
        else:
            Y += 0.012 * rng.standard_normal(Y.shape)
            idx = rng.choice(n, 1, replace=False, p=probs)
            Y[idx] += 0.12 * rng.standard_normal((1, d))
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)

        lr0 = float(rng.uniform(0.012, 0.028))
        lr1 = float(rng.uniform(0.0002, 0.0008))
        m0 = float(rng.uniform(0.02, 0.05))
        m1 = float(rng.uniform(0.0015, 0.004))
        pw = float(rng.choice([1.0, 1.0, 2.0, 0.5]))
        steps = int(rng.integers(1500, 2300))
        if cycle in (5, 3, 4):
            steps = int(steps * 1.25)
        if transplant_done:
            steps = int(steps * 1.1)
        Y = fast_polish(Y, steps, lr0, lr1, m0, m1, pw)
        Y = fast_polish(Y, 350, 0.0006, 0.00015, 0.002, 0.001, 1.0)
        v = max_cos(Y)
        if v < best_val - 1e-12:
            best = Y
            best_val = v
            probs, degs = current_probs(best, best_val)
        if time.time() - t0 > 139:
            break

    remain = 178 - (time.time() - t0)
    if remain > 6:
        best = lp_polish(best, min(remain - 2, 30))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
