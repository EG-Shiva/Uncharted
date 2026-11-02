import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def fast_quench(x, steps, b0, b1, lr0, lr1):
    # annealed soft-max Riemannian descent, vectorized
    n, d = x.shape
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = b0 + (b1 - b0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        m = G.max()
        # weights focused on near-max pairs
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / rs
        # tangent projection
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        # limit step
        gn = np.sqrt(np.sum(grad * grad, axis=1, keepdims=True)) + 1e-12
        # clip extreme grads
        scale = np.minimum(1.0, 1.0 / (gn + 1e-12))
        # actually gn~O(1), keep simple
        x = x - lr * grad
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-30
    return x

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
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
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.RandomState(12345)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        if c is not None and c.shape == (58, 22):
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            cands.append((max_cos(c), c))
    if len(cands) == 0:
        x0 = rng.randn(58, 22)
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        cands.append((max_cos(x0), x0))
    cands.sort(key=lambda t: t[0])
    best_val, x_best = cands[0][0], cands[0][1].copy()
    n, d = 58, 22

    # basin hopping with fast quench, keep within ~120s
    QUENCH_BUDGET = 125.0
    it = 0
    while time.time() - t0 < QUENCH_BUDGET:
        it += 1
        y = x_best.copy()
        mode = it % 5
        if mode == 0:
            # global small shake
            y = y + 0.015 * rng.randn(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps, b0, b1, lr0, lr1 = 300, 150.0, 600.0, 0.02, 0.005
        elif mode == 1:
            # kick subset
            k = 3 + (it % 3)
            idx = rng.choice(n, k, replace=False)
            y[idx] += 0.5 * rng.randn(k, d)
            y[idx] /= np.linalg.norm(y[idx], axis=1, keepdims=True)
            steps, b0, b1, lr0, lr1 = 400, 120.0, 700.0, 0.02, 0.004
        elif mode == 2:
            # worst point random reinit
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            worst = int(np.argmax(G.max(axis=1)))
            v = rng.randn(d)
            v /= np.linalg.norm(v)
            y[worst] = v
            steps, b0, b1, lr0, lr1 = 400, 120.0, 700.0, 0.02, 0.004
        elif mode == 3:
            # hole reinsertion: replace worst by best of random holes
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            worst = int(np.argmax(G.max(axis=1)))
            cand = rng.randn(600, d)
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            dots = cand @ y.T
            dots[:, worst] = -1.0
            md = dots.max(axis=1)
            y[worst] = cand[int(np.argmin(md))]
            steps, b0, b1, lr0, lr1 = 350, 150.0, 800.0, 0.015, 0.004
        else:
            # two-hole shuffle
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            mx = G.max(axis=1)
            worst2 = np.argsort(mx)[-2:]
            for w in worst2:
                v = rng.randn(d)
                v /= np.linalg.norm(v)
                y[w] = v
            steps, b0, b1, lr0, lr1 = 450, 100.0, 600.0, 0.025, 0.005
        y = fast_quench(y, steps, b0, b1, lr0, lr1)
        v = max_cos(y)
        if v < best_val and np.all(np.isfinite(y)):
            best_val = v
            x_best = y.copy()
        # time guard for quench loop overhead
        if time.time() - t0 > QUENCH_BUDGET:
            break

    remain = BUDGET - (time.time() - t0) - 2.0
    if remain > 5.0:
        # final trust-region LP polish with limited time
        x_best = lp_polish(x_best, min(remain, 35.0))
        best_val = max_cos(x_best)

    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # strict finite check, replace any bad row
    if not np.all(np.isfinite(x_best)):
        x_best = np.where(np.isfinite(x_best), x_best, 0.01)
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
