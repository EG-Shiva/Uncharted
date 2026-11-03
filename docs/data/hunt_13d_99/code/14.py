import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def refine_softmax(v, alpha0, alpha1, steps, lr0, lr1, beta=0.9):
    n, d = v.shape
    cur = v.copy()
    best = cur.copy()
    best_s = max_cos(cur)
    mom = np.zeros_like(cur)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        alpha = alpha0 + (alpha1 - alpha0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = cur @ cur.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        # softmax weights, shift for stability
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ cur
        # tangent projection
        proj = np.sum(Grad * cur, axis=1, keepdims=True)
        Gt = Grad - proj * cur
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        Gt = Gt / rs
        # momentum with bias correction
        mom = beta * mom + (1.0 - beta) * Gt
        bc = 1.0 - beta ** (s + 1)
        step = mom / bc
        # clip large steps
        nr = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        # normalize step scale to ~1
        step = step / np.maximum(nr, 1e-12) * np.minimum(nr, 1.0)
        cur = cur - lr * step
        cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        if (s % 200 == 199) or (s == steps - 1):
            sc = max_cos(cur)
            if sc < best_s:
                best_s = sc
                best = cur.copy()
    return best, best_s


def far_replace(base, idx, rng, n_cand=2000):
    x0 = base.copy()
    n, d = base.shape
    for k in idx:
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        others = np.delete(x0, k, axis=0)
        dots = cand @ others.T
        md = dots.max(axis=1)
        x0[k] = cand[np.argmin(md)]
    return x0


def worst_points(x, w):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    pm = G.max(axis=1)
    return np.argsort(-pm)[:w]


def contact_counts(x, thresh=1e-4):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    cnt = (G > m - thresh).sum(axis=1)
    return cnt, m


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
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
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x


def solve():
    t0 = time.time()
    BUDGET = 172.0
    EXPLORE_UNTIL = 140.0
    rng = np.random.default_rng(20261103)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal(size=(99, 13))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = 99, 13
    if xb.shape != (99, 13):
        xb = rng.standard_normal(size=(99, 13))
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = max_cos(xb)
    best_x = xb.copy()
    cur_x = xb.copy()
    cur_val = best_val

    it = 0
    sigmas = [0.01, 0.03, 0.07, 0.12, 0.20, 0.32, 0.50]
    while time.time() - t0 < EXPLORE_UNTIL and it < 1000:
        it += 1
        elapsed = time.time() - t0
        # temperature for basin-hopping acceptance, annealed
        Temp = 0.002 * (1.0 - elapsed / (EXPLORE_UNTIL + 1e-9)) + 1e-5
        # choose base: mostly current, sometimes best
        if rng.random() < 0.7:
            base = cur_x
        else:
            base = best_x
        u = rng.random()
        if u < 0.30:
            sigma = sigmas[rng.integers(0, len(sigmas))]
            x0 = base + rng.standard_normal(size=(n, d)) * sigma
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        elif u < 0.50:
            # subset re-init: kill k random points
            k = int(rng.integers(2, 25))
            idx = rng.choice(n, size=k, replace=False)
            x0 = base.copy()
            rnd = rng.standard_normal(size=(k, d))
            rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
            x0[idx] = rnd
        elif u < 0.72:
            # worst-point far replacement then small shake
            w = int(rng.integers(1, 7))
            wp = worst_points(base, w)
            # optionally add overloaded hubs
            if rng.random() < 0.5:
                cnt, _ = contact_counts(base, thresh=2e-4)
                hub = np.argsort(-cnt)[:w]
                wp = np.unique(np.concatenate([wp, hub]))[:w]
            x0 = far_replace(base, wp, rng, n_cand=2000)
            sigma = float(rng.choice([0.005, 0.01, 0.02]))
            x0 = x0 + rng.standard_normal(size=(n, d)) * sigma
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        elif u < 0.87:
            # shake only worst points strongly
            w = int(rng.integers(2, 12))
            wp = worst_points(base, w)
            x0 = base.copy()
            sigma = float(rng.choice([0.1, 0.2, 0.35, 0.5]))
            x0[wp] = x0[wp] + rng.standard_normal(size=(w, d)) * sigma
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        else:
            # partial random restart blended
            x0 = rng.standard_normal(size=(n, d))
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            # blend with base to keep some structure
            a = float(rng.choice([0.0, 0.0, 0.3, 0.5]))
            if a > 0:
                x0 = a * base + (1 - a) * x0
                x0 /= np.linalg.norm(x0, axis=1, keepdims=True)

        # random anneal schedule
        a0 = float(rng.choice([8.0, 15.0, 25.0, 35.0]))
        a1 = float(rng.choice([50.0, 80.0, 120.0, 160.0]))
        if a1 < a0:
            a0, a1 = a1, a0
        steps = int(rng.choice([600, 900, 1300, 1800]))
        # shrink steps if time low
        remain = EXPLORE_UNTIL - (time.time() - t0)
        if remain < 15 and steps > 600:
            steps = 600
        lr0 = float(rng.choice([0.05, 0.10, 0.20]))
        lr1 = float(rng.choice([0.005, 0.01, 0.02]))
        beta = float(rng.choice([0.85, 0.9, 0.92]))

        x1, v1 = refine_softmax(x0, a0, a1, steps, lr0, lr1, beta=beta)

        if v1 < best_val - 1e-12:
            best_val = v1
            best_x = x1.copy()
            cur_x, cur_val = x1.copy(), v1
        else:
            # basin-hopping acceptance
            if v1 < cur_val or rng.random() < np.exp(-(v1 - cur_val) / max(Temp, 1e-9)):
                cur_x, cur_val = x1.copy(), v1
            # occasional restart of walker to best
            if rng.random() < 0.05:
                cur_x, cur_val = best_x.copy(), best_val

    # final polish of best with remaining time
    remain = BUDGET - (time.time() - t0)
    if remain > 5:
        # use most of remain but leave 2s margin
        best_x = lp_polish(best_x, max(5.0, remain - 2.0))
        best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
