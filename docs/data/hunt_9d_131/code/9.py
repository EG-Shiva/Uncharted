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
        if lr0 > 0 and lr1 > 0:
            lr = lr0 * ((lr1 / lr0) ** f)
        else:
            lr = lr0
        margin = m0 * (1 - f) + m1 * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = np.max(G)
        th = m - margin
        Mask = G >= th
        W = np.zeros_like(G)
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

def largest_voids(X_keep, k, rng, n_probe=12000):
    d = X_keep.shape[1]
    P = rng.standard_normal((n_probe, d))
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    S = P @ X_keep.T
    b = S.max(axis=1)
    order = np.argsort(b)
    chosen = []
    for idx in order:
        if len(chosen) >= k:
            break
        p = P[idx]
        ok = True
        for q in chosen:
            if float(p @ q) > 0.75:
                ok = False
                break
        if ok:
            chosen.append(p)
    if len(chosen) < k:
        for idx in order:
            p = P[idx]
            dup = False
            for q in chosen:
                if np.allclose(p, q):
                    dup = True
                    break
            if not dup:
                chosen.append(p)
                if len(chosen) >= k:
                    break
    return np.array(chosen)

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

    def degrees(X, thr):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        C = G >= (np.max(G) - thr)
        return C.sum(axis=1), G

    it = 0
    # main hopping, leave time for final LP
    while time.time() - t0 < 132:
        it += 1
        deg, _ = degrees(best, 1e-7)
        # hub sampling weights
        w = (deg.astype(float) + 1.0) ** 1.5
        probs = w / w.sum()
        r = rng.random()
        Y = best.copy()
        if r < 0.22:
            # global small shake
            sig = float(rng.uniform(0.006, 0.022))
            Y += sig * rng.standard_normal(Y.shape)
        elif r < 0.62:
            # surgical hub excision + void reinsert
            if rng.random() < 0.55:
                k = 1
            elif rng.random() < 0.7:
                k = 2
            else:
                k = 3
            # pick k among top pool
            pool = np.argsort(deg)[-12:]
            # weighted within pool by deg
            pw = (deg[pool].astype(float) + 1.0)
            pw = pw / pw.sum()
            idx = rng.choice(pool, size=k, replace=False, p=pw)
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            X_keep = best[mask]
            V = largest_voids(X_keep, k, rng)
            Y[idx] = V + 0.008 * rng.standard_normal((k, d))
            # small global jitter to break symmetry
            Y += 0.004 * rng.standard_normal(Y.shape)
        elif r < 0.78:
            # random cap shake: move a spatially clustered region
            u = rng.standard_normal(d)
            u /= np.linalg.norm(u)
            proj = best @ u
            thr = float(rng.uniform(0.3, 0.8))
            idx = np.where(proj > thr)[0]
            if len(idx) < 2 or len(idx) > 30:
                # fallback: random subset
                k = int(rng.integers(6, 16))
                idx = rng.choice(n, size=k, replace=False)
            sig = float(rng.uniform(0.05, 0.15))
            Y[idx] += sig * rng.standard_normal((len(idx), d))
            Y += 0.004 * rng.standard_normal(Y.shape)
        else:
            # big single/double kick on hubs
            k = int(rng.integers(1, 3))
            idx = rng.choice(n, size=k, replace=False, p=probs)
            sig = float(rng.uniform(0.12, 0.35))
            Y[idx] += sig * rng.standard_normal((k, d))
            Y += 0.008 * rng.standard_normal(Y.shape)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)

        lr0 = float(rng.uniform(0.012, 0.028))
        lr1 = float(rng.uniform(0.0002, 0.0008))
        m0 = float(rng.uniform(0.02, 0.05))
        m1 = float(rng.uniform(0.0015, 0.004))
        # short screening polish
        Ys = fast_polish(Y, 700, lr0, lr1, m0, m1)
        vs = max_cos(Ys)
        # accept / extend criteria
        if vs < best_val - 1e-12:
            # promising record: extend thoroughly
            Ys = fast_polish(Ys, 1800, lr0 * 0.6, 0.0002, 0.008, 0.001)
            Ys = fast_polish(Ys, 400, 0.0006, 0.00015, 0.002, 0.001)
            v = max_cos(Ys)
            if v < best_val - 1e-12:
                best = Ys
                best_val = v
        elif vs < best_val + 0.004:
            # near-best: extend, may cross to new basin
            extra = int(rng.integers(1200, 2000))
            Ys = fast_polish(Ys, extra, lr0 * 0.5, 0.00025, 0.006, 0.0012)
            Ys = fast_polish(Ys, 400, 0.0006, 0.00015, 0.002, 0.001)
            v = max_cos(Ys)
            if v < best_val - 1e-12:
                best = Ys
                best_val = v
        # else discard
        if time.time() - t0 > 133:
            break

    remain = 178 - (time.time() - t0)
    if remain > 5:
        best = lp_polish(best, min(remain - 2, 30))
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
