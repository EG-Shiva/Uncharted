import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _quench(X, steps, beta0, beta1, lr0, lr1):
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    lb0 = np.log(beta0)
    lb1 = np.log(beta1)
    ll0 = np.log(lr0)
    ll1 = np.log(lr1)
    for t in range(1, steps + 1):
        frac = t / steps
        beta = np.exp(lb0 + (lb1 - lb0) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        Z = W.sum()
        if Z == 0:
            continue
        Grad = (W @ X) / Z
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        m *= b1
        m += (1.0 - b1) * Gt
        v *= b2
        v += (1.0 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        mhat = m / bc1
        vhat = v / bc2
        step = mhat / (np.sqrt(vhat) + eps)
        X = X - lr * step
        nrm = np.sqrt((X * X).sum(axis=1, keepdims=True))
        nrm[nrm == 0] = 1.0
        X /= nrm
    return X


def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        # cap constraints for speed
        if int(sel.sum()) > 4000:
            # keep 4000 tightest
            idx = np.argsort(-c)
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[:4000]] = True
            sel = sel & keep
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
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


def _e8_roots():
    pts = []
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = np.zeros(8)
                    v[i] = si
                    v[j] = sj
                    pts.append(v)
    for mask in range(256):
        signs = np.ones(8)
        nm = 0
        for k in range(8):
            if (mask >> k) & 1:
                signs[k] = -1.0
                nm += 1
        if nm % 2 == 0:
            pts.append(0.5 * signs)
    a = np.array(pts, dtype=float)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    return a


def solve():
    t0 = time.time()
    BUDGET = 172.0
    n, d = 168, 8
    rng = np.random.default_rng(0)

    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape != (n, d):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            va = _max_cos(a)
            if va < best_val:
                best_val = va
                best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.normal(size=(n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    best_x = best.copy()

    e8 = _e8_roots()

    def void_centers(R, k, M=20000):
        C = rng.normal(size=(M, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        D = R @ C.T
        cur = D.max(axis=0)
        out = []
        for _ in range(k):
            idx = int(np.argmin(cur))
            out.append(idx)
            v = C[idx]
            dvc = C @ v
            cur = np.maximum(cur, dvc)
            cur[idx] = np.inf
        return C[out]

    # E8 subset starts - different basin
    for _ in range(3):
        if time.time() - t0 > BUDGET - 45:
            break
        idx = rng.choice(240, size=n, replace=False)
        x0 = e8[idx] + rng.normal(size=(n, d)) * 0.02
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        b0 = float(rng.uniform(25, 50))
        b1 = float(rng.uniform(200, 500))
        lr0 = float(rng.uniform(0.003, 0.007))
        xq = _quench(x0, 2200, b0, b1, lr0, lr0 * 0.1)
        v = _max_cos(xq)
        if v < best_val:
            best_val = v
            best_x = xq

    it = 0
    while time.time() - t0 < BUDGET - 40:
        it += 1
        X = best_x.copy()
        r = rng.random()
        if r < 0.35:
            sigma = float(10.0 ** rng.uniform(-2.3, -1.1))
            X = X + rng.normal(size=(n, d)) * sigma
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        elif r < 0.68:
            # void re-insertion of worst points
            k = int(rng.integers(2, 9))
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            ind = G.max(axis=1)
            worst = np.argsort(-ind)[:k]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[worst] = False
            R = X[keep_mask]
            V = void_centers(R, k)
            X[worst] = V + rng.normal(size=(k, d)) * 0.005
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        elif r < 0.88:
            k = int(rng.integers(3, 13))
            sel = rng.choice(n, size=k, replace=False)
            X[sel] = rng.normal(size=(k, d))
            X += rng.normal(size=(n, d)) * 0.004
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        else:
            # shake overloaded points
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            cnt = (G > m - 0.004).sum(axis=1)
            order = np.argsort(-cnt)
            pool = order[:30]
            k = int(rng.integers(4, 11))
            sel = rng.choice(pool, size=min(k, len(pool)), replace=False)
            s_big = float(rng.uniform(0.1, 0.45))
            X[sel] += rng.normal(size=(len(sel), d)) * s_big
            X += rng.normal(size=(n, d)) * 0.006
            X /= np.linalg.norm(X, axis=1, keepdims=True)

        beta0 = float(rng.uniform(20, 60))
        beta1 = float(rng.uniform(200, 700))
        lr0 = float(rng.uniform(0.0025, 0.008))
        lr1 = lr0 * float(rng.uniform(0.05, 0.2))
        steps = int(rng.integers(1200, 2400))
        Xq = _quench(X, steps, beta0, beta1, lr0, lr1)
        v = _max_cos(Xq)
        if v < best_val - 1e-12:
            best_val = v
            best_x = Xq
            # intensify in new basin if time allows
            if time.time() - t0 < BUDGET - 55:
                Xq2 = _quench(best_x.copy(), 900, 200.0, 800.0, 0.0015, 0.00025)
                v2 = _max_cos(Xq2)
                if v2 < best_val:
                    best_val = v2
                    best_x = Xq2
        if time.time() - t0 > BUDGET - 40:
            break

    remain = BUDGET - (time.time() - t0) - 2.0
    if remain > 5:
        best_x = _lp_polish(best_x, min(remain, 38.0))
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
