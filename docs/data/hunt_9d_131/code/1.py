import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def fast_polish(X, steps, lr0, lr1, m0, m1):
    n, d = X.shape
    for s in range(steps):
        f = s / max(1, steps - 1)
        lr = lr0 * ((lr1 / lr0) ** f) if lr0 > 0 and lr1 > 0 else lr0
        margin = m0 * (1 - f) + m1 * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = np.max(G)
        th = m - margin
        # linear weight: worst pairs weight 1.0, threshold pairs weight 0.2
        Mask = G >= th
        W = np.zeros_like(G)
        # avoid divide by zero
        if margin > 1e-12:
            W[Mask] = 0.2 + 0.8 * (G[Mask] - th) / margin
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

    # contact-degree based sampling probs (favor hubs, ignore rattlers)
    def current_probs(X, mv):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        C = G >= (mv - 1e-7)
        deg = C.sum(axis=1).astype(float)
        w = deg + 1.0  # rattlers get weight 1, hubs ~28
        # boost top hubs a bit more
        w = w ** 1.5
        return w / w.sum(), deg

    probs, _ = current_probs(best, best_val)

    it = 0
    # leave ~35s for final LP
    while time.time() - t0 < 132:
        it += 1
        # refresh probs every 10 iters
        if it % 10 == 1:
            probs, _ = current_probs(best, best_val)
        cycle = it % 7
        Y = best.copy()
        if cycle == 0:
            Y += 0.008 * rng.standard_normal(Y.shape)
        elif cycle == 1:
            Y += 0.022 * rng.standard_normal(Y.shape)
        elif cycle == 2:
            k, sig = 1, 0.10
            idx = rng.choice(n, k, replace=False, p=probs)
            Y[idx] += sig * rng.standard_normal((k, d))
        elif cycle == 3:
            k, sig = 2, 0.10
            idx = rng.choice(n, k, replace=False, p=probs)
            Y[idx] += sig * rng.standard_normal((k, d))
        elif cycle == 4:
            k, sig = 3, 0.16
            idx = rng.choice(n, k, replace=False, p=probs)
            Y[idx] += sig * rng.standard_normal((k, d))
        elif cycle == 5:
            k, sig = 1, 0.35
            idx = rng.choice(n, k, replace=False, p=probs)
            Y[idx] += sig * rng.standard_normal((k, d))
        else:
            # random-subspace global shake + one hub kick
            Y += 0.012 * rng.standard_normal(Y.shape)
            idx = rng.choice(n, 1, replace=False, p=probs)
            Y[idx] += 0.12 * rng.standard_normal((1, d))
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)

        # randomize polish hyperparams slightly for diversity
        lr0 = float(rng.uniform(0.012, 0.028))
        lr1 = float(rng.uniform(0.0002, 0.0008))
        m0 = float(rng.uniform(0.02, 0.05))
        m1 = float(rng.uniform(0.0015, 0.004))
        steps = int(rng.integers(1800, 2600))
        # big jumps get longer polish
        if cycle in (4, 5):
            steps = int(steps * 1.3)
        Y = fast_polish(Y, steps, lr0, lr1, m0, m1)
        # extra fine tail
        Y = fast_polish(Y, 400, 0.0006, 0.00015, 0.002, 0.001)
        v = max_cos(Y)
        if v < best_val - 1e-12:
            best = Y
            best_val = v
            probs, _ = current_probs(best, best_val)
        # safety time break
        if time.time() - t0 > 133:
            break

    remain = 178 - (time.time() - t0)
    if remain > 5:
        # final squeeze, cap to ~30s
        best = lp_polish(best, min(remain - 2, 30))
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
