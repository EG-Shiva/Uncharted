import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def proj_unit(x):
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    x = x / nrm
    x = np.nan_to_num(x)
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return x / nrm

def lp_polish(x, seconds):
    x = proj_unit(x)
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        win = max(20 * radius, 1e-9)
        sel = c > m - win
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        csv = c[sel]
        if p > 2500:
            idx = np.argpartition(-csv, 2500)[:2500]
            I, J = I[idx], J[idx]
            csv = csv[idx]
            p = 2500
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        ar = np.arange(d)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        dx = res.x[:-1].reshape(n, d)
        y = proj_unit(x + dx)
        v = max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            # accept sideways move only if LP predicted improvement but nonlinear failed? no
            radius *= 0.5
            # stay at best_x if too stuck? keep x as is
        if best < m - 1e-12:
            pass
    return proj_unit(best_x)

def contact_info(x, tol=1e-9):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    C = (G > m - tol)
    cnts = C.sum(axis=1)
    return cnts, m, G

def smooth_anneal(x, steps, T0, T1, lr, rng=None):
    y = x.copy()
    n, d = y.shape
    for s in range(steps):
        T = T0 + (T1 - T0) * (s / max(1, steps - 1))
        G = y @ y.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        W = np.exp((G - m) / max(T, 1e-9))
        np.fill_diagonal(W, 0.0)
        # symmetrize (already symmetric)
        grad = W @ y
        # tangent projection
        grad = grad - (np.sum(grad * y, axis=1, keepdims=True)) * y
        gn = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-12
        # use weighted step
        step = lr * (1.0 - 0.4 * s / steps)
        y = y - step * grad / gn
        y = proj_unit(y)
    return y

def refine_single(v, others, steps=200, T0=0.02, T1=0.002, lr=0.05):
    # minimize soft-max dot to others by moving v on sphere
    for s in range(steps):
        T = T0 + (T1 - T0) * (s / max(1, steps - 1))
        dots = others @ v  # (m,)
        m = float(dots.max())
        w = np.exp((dots - m) / max(T, 1e-9))
        w = w / (w.sum() + 1e-12)
        grad = w @ others  # (d,)
        # tangent
        grad = grad - (grad @ v) * v
        gn = np.linalg.norm(grad) + 1e-12
        step = lr * (1.0 - 0.5 * s / steps)
        v = v - step * grad / gn
        v = v / (np.linalg.norm(v) + 1e-12)
    return v

def reinsert_move(x, rng, k, n_cand=3000):
    n, d = x.shape
    cnts, m, G = contact_info(x, tol=1e-7)
    order_low = np.argsort(cnts)
    order_high = np.argsort(-cnts)
    victims = set()
    # one most overloaded, one least loaded (peripheral), rest random
    victims.add(int(order_high[0]))
    victims.add(int(order_low[0]))
    if k > 2:
        victims.add(int(order_high[1 % n]))
    while len(victims) < k:
        victims.add(int(rng.integers(0, n)))
    victims = list(victims)
    # keep others fixed initially
    fixed_idx = [i for i in range(n) if i not in victims]
    y = x.copy()
    for vi in victims:
        others_idx = [i for i in range(n) if i != vi]
        others = y[others_idx]
        cand = rng.normal(size=(n_cand, d))
        cand = proj_unit(cand)
        scores = cand @ others.T
        mx = scores.max(axis=1)
        # take best few and refine
        bidx = np.argpartition(mx, 5)[:5]
        best_v = None
        best_s = 1e9
        for bi in bidx:
            v0 = cand[bi]
            v1 = refine_single(v0, others, steps=150, T0=0.01, T1=0.002, lr=0.08)
            s = float(np.max(others @ v1))
            if s < best_s:
                best_s = s
                best_v = v1
        y[vi] = best_v
    return y

def tangent_shake(x, rng, strength):
    n, d = x.shape
    y = x.copy()
    noise = rng.normal(size=(n, d)) * strength
    noise = noise - (np.sum(noise * y, axis=1, keepdims=True)) * y
    y = proj_unit(y + noise)
    return y

def solve():
    T0 = time.time()
    LIMIT = 170
    rng = np.random.default_rng(7)
    try:
        x0 = np.load("best.npy")
    except Exception:
        x0 = np.load("record.npy")
    x0 = proj_unit(np.nan_to_num(x0))
    best = x0.copy()
    best_val = max_cos(best)
    elapsed = lambda: time.time() - T0

    # initial quick polish to sit at local optimum
    if elapsed() < LIMIT - 30:
        best = lp_polish(best, min(7.0, max(1.0, LIMIT - elapsed() - 60)))
        best_val = max_cos(best)

    it = 0
    # alternate reinsert and shake
    while elapsed() < LIMIT - 22:
        it += 1
        k = int(rng.integers(2, 6))  # number of points to reinsert
        if it % 3 != 0:
            cand = reinsert_move(best, rng, k=k, n_cand=2500)
            # short global smoothing to settle new points
            cand = smooth_anneal(cand, steps=250, T0=0.015, T1=0.003, lr=0.03)
        else:
            # larger collective shake
            s = [0.04, 0.10, 0.20, 0.30][(it // 3) % 4]
            cand = tangent_shake(best, rng, strength=s)
            cand = smooth_anneal(cand, steps=350, T0=0.02, T1=0.002, lr=0.035)
        budget = min(5.0, (LIMIT - 22 - elapsed()) / 2.0)
        if budget < 0.8:
            break
        cand = lp_polish(cand, budget)
        v = max_cos(cand)
        if v < best_val - 1e-12:
            best = cand
            best_val = v
        # opportunistic small polish of best with different tolerance? skip
    if elapsed() < LIMIT - 1:
        best = lp_polish(best, max(1.0, LIMIT - elapsed() - 0.5))
    best = proj_unit(best)
    return best
