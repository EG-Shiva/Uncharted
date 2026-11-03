import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def smooth_refine(v, alpha, steps, lr, beta=0.9):
    n, d = v.shape
    cur = v.copy()
    best = cur.copy()
    best_s = max_cos(cur)
    mom = np.zeros_like(cur)
    for s in range(steps):
        G = cur @ cur.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ cur
        proj = np.sum(Grad * cur, axis=1, keepdims=True)
        Gt = Grad - proj * cur
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        Gt = Gt / rs
        mom = beta * mom + (1.0 - beta) * Gt
        # bias correction
        bc = 1.0 - beta ** (s + 1)
        step = mom / bc
        cur_lr = lr * (1.0 - 0.5 * s / max(steps, 1))
        cur = cur - cur_lr * step
        cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        if (s % 200 == 199) or (s == steps - 1):
            sc = max_cos(cur)
            if sc < best_s:
                best_s = sc
                best = cur.copy()
    return best, best_s


def far_replacement(base, idx, rng, n_cand=2000):
    n, d = base.shape
    x0 = base.copy()
    for k in idx:
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        others = np.delete(x0, k, axis=0)
        dots = cand @ others.T
        md = dots.max(axis=1)
        best_c = np.argmin(md)
        x0[k] = cand[best_c]
    return x0


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
            # keep best but shrink
            # if stuck, try moving x to y even if slightly worse to allow LP to continue? no, shrink
            radius *= 0.5
        # track global best
        if best < max_cos(x) - 1e-12:
            x = best_x.copy()
    return best_x


def solve():
    t0 = time.time()
    TIME_BUDGET = 172.0
    EXPLORE_UNTIL = 128.0  # seconds for exploration, rest for LP
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal(size=(99, 13))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = xb.shape
    if (n, d) != (99, 13):
        # fix shape if needed by random init
        xb = rng.standard_normal(size=(99, 13))
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = max_cos(xb)
    best_x = xb.copy()
    # small population for diversity
    pop = [(best_val, best_x.copy())]

    it = 0
    # precomputed schedule pools
    sigmas = [0.02, 0.05, 0.10, 0.18, 0.30, 0.45]
    alphas = [25.0, 40.0, 60.0, 90.0, 130.0]
    while time.time() - t0 < EXPLORE_UNTIL and it < 500:
        it += 1
        # pick base: mostly best, sometimes 2nd best
        rpick = rng.random()
        if rpick < 0.75 or len(pop) == 1:
            base = best_x
        else:
            # pick random from pop
            k = rng.integers(0, len(pop))
            base = pop[k][1]
        mode = rng.random()
        sigma = sigmas[rng.integers(0, len(sigmas))]
        if mode < 0.45:
            # full gaussian shake
            noise = rng.standard_normal(size=(n, d)) * sigma
            x0 = base + noise
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        elif mode < 0.80:
            # kick worst k points
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            kk = int(rng.integers(2, 9))
            idx = np.argsort(rowmax)[-kk:]
            x0 = base.copy()
            noise = rng.standard_normal(size=(kk, d)) * (sigma * 1.5 + 0.02)
            x0[idx] = x0[idx] + noise
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        else:
            # remove worst and re-insert far
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            kk = int(rng.integers(1, 6))
            idx = np.argsort(rowmax)[-kk:]
            x0 = far_replacement(base, idx, rng, n_cand=2500)
        alpha = alphas[rng.integers(0, len(alphas))]
        steps = int(rng.integers(800, 2600))
        lr = float(rng.uniform(0.015, 0.06))
        x1, s1 = smooth_refine(x0, alpha, steps, lr)
        # quick extra anneal at higher alpha for minimax focus
        if rng.random() < 0.5:
            x1, s1b = smooth_refine(x1, alpha * 1.6 + 20.0, 600, lr * 0.5)
            s1 = min(s1, s1b) if isinstance(s1b, float) else s1
            # s1 already is best of second run; need correct val
            s1 = max_cos(x1) if max_cos(x1) < s1 else s1
            # recompute to be safe
            s1 = max_cos(x1)
        else:
            s1 = max_cos(x1)
        if s1 < best_val - 1e-12:
            best_val = s1
            best_x = x1.copy()
            pop.append((s1, x1.copy()))
            pop = sorted(pop, key=lambda z: z[0])[:5]
        elif s1 < pop[-1][0] and len(pop) < 5:
            pop.append((s1, x1.copy()))
            pop = sorted(pop, key=lambda z: z[0])[:5]
        elif rng.random() < 0.02:
            # occasional diversity injection
            if len(pop) < 5:
                pop.append((s1, x1.copy()))
    # final polish with remaining time
    remain = TIME_BUDGET - (time.time() - t0) - 1.0
    if remain > 5:
        # polish best; if time allows polish runner-up and pick best
        best_x = lp_polish(best_x, remain * 0.85)
        best_val = max_cos(best_x)
        # if still time, try polishing second candidate briefly
        remain2 = TIME_BUDGET - (time.time() - t0) - 1.0
        if remain2 > 8 and len(pop) > 1:
            for _, px in list(pop[1:2]):
                qx = lp_polish(px, remain2 * 0.4)
                qv = max_cos(qx)
                if qv < best_val:
                    best_val = qv
                    best_x = qx.copy()
                break
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
