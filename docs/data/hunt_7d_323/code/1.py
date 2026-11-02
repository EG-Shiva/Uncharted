import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def contacts_of(x, tol=1e-7):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = g.max()
    cnt = (g > m - tol).sum(axis=1)
    return cnt, m, g


def lp_polish(x, seconds, t_end=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 2e-3
    start = time.time()
    if t_end is None:
        t_end = start + seconds
    else:
        t_end = min(t_end, start + seconds)
    iu = np.triu_indices(n, 1)
    while time.time() < t_end and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        thr = max(20 * radius, 2e-9)
        sel = c > m - thr
        # cap size for speed
        if int(sel.sum()) > 6000:
            # keep tightest 6000
            idx = np.argpartition(-c, 6000)[:6000]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
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
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def smooth_polish(x, steps, beta, lr, delta):
    n, d = x.shape
    for _ in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        # weights focused on near-max pairs
        # avoid overflow: exp(beta*(G-m)) in [0,1]
        sel = G > (m - delta)
        W = np.zeros_like(G)
        # compute only on sel
        W[sel] = np.exp(beta * (G[sel] - m))
        np.fill_diagonal(W, 0.0)
        Wx = W @ x
        s = (W * G).sum(axis=1, keepdims=True)
        rs = W.sum(axis=1, keepdims=True) + 1.0
        grad = (Wx - s * x) / rs
        x = x - lr * grad
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def anneal(x, steps1=300, steps2=400):
    x = smooth_polish(x, steps1, beta=35.0, lr=0.20, delta=0.20)
    x = smooth_polish(x, steps2, beta=110.0, lr=0.08, delta=0.08)
    return x


def best_hole_candidate(x, exclude, rng, n_cand=20000):
    n, d = x.shape
    cand = rng.normal(size=(n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ x.T
    # ignore excluded point
    dots[:, exclude] = -2.0
    md = dots.max(axis=1)
    return cand[int(np.argmin(md))]


def solve():
    t0 = time.time()
    BUDGET = 172.0
    t_end = t0 + BUDGET
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.normal(size=(323, 7))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    # guard against bad shapes
    if xb.shape != (323, 7):
        # resize / random fallback
        tmp = rng.normal(size=(323, 7))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        m = min(xb.shape[0], 323)
        tmp[:m] = xb[:m, :7] if xb.shape[1] >= 7 else xb[:m]
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        xb = tmp
    best_val = max_cos(xb)
    x_best = xb.copy()

    cycle = 0
    # quick anneal+polish of starting point if time allows (already polished)
    # main basin-hopping loop; reserve ~30s for final polish
    while time.time() < t_end - 30.0 and cycle < 60:
        cnt, m, _ = contacts_of(x_best, tol=1e-7)
        order = np.argsort(cnt)  # ascending: rattlers first
        y = x_best.copy()
        mode = cycle % 4
        if mode == 0:
            # relocate rattlers / loneliest points to largest holes
            k = 3
            lonely = order[:k]
            # also include true rattlers
            for idx in lonely:
                hole = best_hole_candidate(x_best, int(idx), rng, n_cand=15000)
                # small blend to keep determinism stable
                y[int(idx)] = hole
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif mode == 1:
            # shake overloaded hubs strongly
            hubs = np.argsort(-cnt)[:8]
            y[hubs] += rng.normal(scale=0.20, size=(len(hubs), 7))
            # also shake their neighbours lightly
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif mode == 2:
            # random subset shake
            idx = rng.choice(323, size=24, replace=False)
            y[idx] += rng.normal(scale=0.12, size=(24, 7))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        else:
            # global small jitter
            y += rng.normal(scale=0.018, size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)

        # smooth repulsion to find new basin
        # adapt steps to remaining time
        y = anneal(y, steps1=250, steps2=350)
        # short LP squeeze
        remain = t_end - 30.0 - time.time()
        if remain < 2.0:
            break
        y = lp_polish(y, seconds=min(6.0, remain), t_end=t_end - 30.0)
        v = max_cos(y)
        if v < best_val - 1e-12:
            best_val = v
            x_best = y.copy()
        cycle += 1
        # early exit if very close to time
        if time.time() > t_end - 32.0:
            break

    # final intensive polish of best
    remain = t_end - time.time() - 1.0
    if remain > 3.0:
        # one more anneal with high beta then LP
        x_best = smooth_polish(x_best, 300, beta=150.0, lr=0.05, delta=0.06)
        x_best = lp_polish(x_best, seconds=remain, t_end=t_end)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
