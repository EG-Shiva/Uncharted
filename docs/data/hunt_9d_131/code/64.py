import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _norm(x):
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x


def fast_polish(X, steps, lr0, lr1, m0, m1, bg=0.03, Tbg=0.18):
    n, d = X.shape
    for s in range(steps):
        f = s / max(1, steps - 1)
        lr = lr0 * ((lr1 / lr0) ** f) if lr0 > 0 and lr1 > 0 else lr0
        margin = m0 * (1 - f) + m1 * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        th = m - margin
        Mask = G >= th
        W = np.zeros_like(G)
        if margin > 1e-12:
            W[Mask] = 0.2 + 0.8 * (G[Mask] - th) / margin
        else:
            W[Mask] = 1.0
        np.fill_diagonal(W, 0.0)
        # uniform background to keep rattlers moving (load uniformity)
        if bg > 0:
            # exp weight for all pairs, cheap
            E = np.exp((G - m) / Tbg)
            np.fill_diagonal(E, 0.0)
            W = W + bg * E
        S = W @ X
        deg = np.sum(W, axis=1, keepdims=True)
        dot = np.sum(S * X, axis=1, keepdims=True)
        F = S - dot * X
        F /= np.maximum(1.0, deg)
        X = X - lr * F
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return X


def contact_deg(X, mv, tol=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    C = G >= (mv - tol)
    return C.sum(axis=1).astype(float)


def find_hole(Xk, rng, n_samples=15000, T=0.01, lr=0.05, steps=120):
    d = Xk.shape[1]
    Z = rng.standard_normal((n_samples, d))
    Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    D = Z @ Xk.T
    worst = D.max(axis=1)
    idx = int(np.argmin(worst))
    z = Z[idx].copy()
    # refine: minimize smooth max dot
    for _ in range(steps):
        dots = Xk @ z
        m = float(dots.max())
        W = np.exp((dots - m) / T)
        W /= (W.sum() + 1e-12)
        g = W @ Xk
        g = g - float(np.dot(g, z)) * z
        z = z - lr * g
        nz = np.linalg.norm(z)
        if nz < 1e-12:
            break
        z = z / nz
    return z


def lp_polish(x, seconds):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        # cap constraints for speed
        if p > 4000:
            order = np.argsort(-c[sel])[:4000]
            I, J = I[order], J[order]
            c = c[sel][order]
            p = len(I)
            csel = c
        else:
            csel = c[sel]
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    rng = np.random.default_rng(0)
    n, d = 131, 9
    best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = _norm(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal((n, d))
        best = _norm(best)
        best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val
    # annealing temps
    T0, T1 = 0.0015, 2e-06
    it = 0
    # time for hopping; leave ~35s for LP
    HOP_DEADLINE = t0 + 132
    while time.time() < HOP_DEADLINE:
        it += 1
        frac = (time.time() - t0) / max(1e-9, (HOP_DEADLINE - t0))
        Temp = T0 * (1 - frac) + T1 * frac
        # reset walker if drifted too far
        if cur_val > best_val + 0.008 and (it % 5 == 0):
            cur = best.copy()
            cur_val = best_val
        r = rng.random()
        Y = cur.copy()
        # degree weights for victim selection
        deg = contact_deg(cur, cur_val)
        w = (deg + 1.0) ** 1.6
        probs = w / w.sum()
        move_kind = None
        if r < 0.28:
            # transplant 1-3 hubs to holes (core relief)
            if rng.random() < 0.5:
                k = 1
            elif rng.random() < 0.7:
                k = 2
            else:
                k = 3
            # pick victims biased to hubs
            try:
                victims = rng.choice(n, size=k, replace=False, p=probs)
            except Exception:
                victims = rng.choice(n, size=k, replace=False)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[victims] = False
            Xk = Y[keep_mask]
            placed = []
            for t in range(k):
                if placed:
                    Xkb = np.vstack([Xk, np.stack(placed)])
                else:
                    Xkb = Xk
                z = find_hole(Xkb, rng)
                placed.append(z)
            for vi, z in zip(victims, placed):
                Y[vi] = z
            # small jitter others to break symmetry
            Y += 0.004 * rng.standard_normal(Y.shape)
            move_kind = 'hole'
        elif r < 0.50:
            # hub kicks
            k = int(rng.integers(1, 4))
            sig = float(rng.uniform(0.10, 0.22))
            idx = rng.choice(n, size=k, replace=False, p=probs)
            Y[idx] += sig * rng.standard_normal((k, d))
            Y += 0.006 * rng.standard_normal(Y.shape)
            move_kind = 'hub'
        elif r < 0.70:
            # global shake
            sig = float(rng.uniform(0.008, 0.025))
            Y += sig * rng.standard_normal(Y.shape)
            move_kind = 'global'
        else:
            # combined: global + one hub big kick
            Y += 0.010 * rng.standard_normal(Y.shape)
            idx = rng.choice(n, size=1, replace=False, p=probs)
            Y[idx] += float(rng.uniform(0.10, 0.30)) * rng.standard_normal((1, d))
            move_kind = 'mix'
        Y = _norm(Y)
        # polish hyperparams
        lr0 = float(rng.uniform(0.012, 0.030))
        lr1 = float(rng.uniform(0.0002, 0.0009))
        m0 = float(rng.uniform(0.02, 0.06))
        m1 = float(rng.uniform(0.0015, 0.005))
        bg = float(rng.uniform(0.01, 0.06))
        if move_kind == 'hole':
            steps = int(rng.integers(1600, 2400))
        elif move_kind == 'hub':
            steps = int(rng.integers(1400, 2200))
        else:
            steps = int(rng.integers(1200, 2000))
        Y = fast_polish(Y, steps, lr0, lr1, m0, m1, bg=bg)
        # short fine tail without bg for convergence
        Y = fast_polish(Y, 250, 0.0006, 0.00015, 0.002, 0.001, bg=0.0)
        v = _max_cos(Y)
        # annealing acceptance on walker
        if v < cur_val or rng.random() < np.exp(-(v - cur_val) / max(Temp, 1e-12)):
            cur = Y
            cur_val = v
        if v < best_val - 1e-12:
            best = Y.copy()
            best_val = v
        # opportunistic reset of walker to best occasionally
        if it % 25 == 0 and rng.random() < 0.5:
            cur = best.copy()
            cur_val = best_val
        if time.time() > HOP_DEADLINE:
            break
    remain = 178 - (time.time() - t0)
    if remain > 6:
        try:
            best = lp_polish(best, min(remain - 2, 30))
        except Exception:
            pass
    best = _norm(best)
    return best
