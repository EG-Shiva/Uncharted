import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def lp_polish(x, seconds, r0=2e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = r0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-15:
        if time.time() - start > seconds - 0.3:
            break
        G = x @ x.T
        cfull = G[iu0]
        m = float(cfull.max())
        sel = cfull > m - max(25.0 * radius, 2e-9)
        I, J = iu0[0][sel], iu0[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cc = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vv = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vv, (rows, cc)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cfull[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
        if v < best - 1e-15:
            x, best = y, v
            radius = min(radius * 1.4, 0.02)
        else:
            # try smaller step along same direction
            improved = False
            for shrink in (0.3, 0.1):
                yy = x + (res.x[:-1].reshape(n, d)) * shrink
                yy /= np.linalg.norm(yy, axis=1, keepdims=True)
                vv2 = max_cos(yy)
                if vv2 < best:
                    x, best = yy, vv2
                    improved = True
                    break
            if not improved:
                radius *= 0.5
    return x

def make_fg(n, d, IU0, beta):
    def fg(yf):
        Y = yf.reshape(n, d)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = Y / nr
        g = (X @ X.T)[IU0]
        m = float(g.max())
        e = np.exp(beta * (g - m))
        S = float(e.sum()) + 1e-300
        F = m + np.log(S) / beta
        w = e / S
        P = np.zeros((n, n))
        P[IU0] = w
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return F, GY.ravel()
    return fg

def run_lbfgs(x0, beta, maxiter):
    n, d = x0.shape
    IU0 = np.triu_indices(n, 1)
    fg = make_fg(n, d, IU0, beta)
    y0 = (x0 / np.linalg.norm(x0, axis=1, keepdims=True)).ravel().copy()
    try:
        r = minimize(fg, y0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-8, 'maxfun': maxiter * 3})
        Y = r.x.reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    except Exception:
        return x0

def micro_hill(x, seconds, rng):
    # greedy tangent micro steps on worst pairs
    start = time.time()
    n, d = x.shape
    best = max_cos(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    while time.time() - start < seconds:
        # find worst pair
        ij = int(np.argmax(G.ravel()))
        a, b = divmod(ij, n)
        # try moving a or b
        improved = False
        for q in (a, b):
            # tangent basis: random tangent vectors
            for _ in range(40):
                if time.time() - start >= seconds:
                    break
                step = float(10 ** rng.uniform(-6.5, -3.5))
                v = rng.standard_normal(d)
                v -= (v @ x[q]) * x[q]
                nv = np.linalg.norm(v)
                if nv < 1e-12:
                    continue
                v /= nv
                y = x.copy()
                y[q] = y[q] + step * v
                y[q] /= np.linalg.norm(y[q])
                v2 = max_cos(y)
                if v2 < best:
                    x, best = y, v2
                    G = x @ x.T
                    np.fill_diagonal(G, -2.0)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            # shake a random member of top-20 pairs slightly
            flat = G.ravel()
            # get top pairs indices
            top = np.argpartition(-flat, 30)[:30]
            pick = top[int(rng.integers(0, len(top)))]
            a2, b2 = divmod(int(pick), n)
            q = a2 if rng.random() < 0.5 else b2
            v = rng.standard_normal(d)
            v -= (v @ x[q]) * x[q]
            v /= (np.linalg.norm(v) + 1e-12)
            y = x.copy()
            y[q] = y[q] + 10 ** rng.uniform(-6, -4) * v
            y[q] /= np.linalg.norm(y[q])
            v2 = max_cos(y)
            if v2 < best:
                x, best = y, v2
                G = x @ x.T
                np.fill_diagonal(G, -2.0)
            else:
                # small random global jitter to escape micro plateau
                if rng.random() < 0.1:
                    y2 = x + 1e-6 * rng.standard_normal(x.shape)
                    y2 /= np.linalg.norm(y2, axis=1, keepdims=True)
                    # accept sideways to explore cluster
                    x = y2
                    G = x @ x.T
                    np.fill_diagonal(G, -2.0)
                    best = float(G.max())
                else:
                    break
    return x

def solve():
    t0 = time.time()
    TLIM = 172.0
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        cands.append(xb)
    if xr is not None:
        cands.append(xr)
    if len(cands) == 0:
        rng0 = np.random.default_rng(0)
        x_init = rng0.standard_normal((68, 12))
        x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
    else:
        bv = 1e9
        x_init = cands[0]
        for c in cands:
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            v = max_cos(c)
            if v < bv:
                bv = v
                x_init = c.copy()
    x_init = x_init / np.linalg.norm(x_init, axis=1, keepdims=True)
    x_best = x_init.copy()
    best_val = max_cos(x_best)
    n, d = x_best.shape
    rng = np.random.default_rng(1234567)

    # Stage 1: quick deep LP on incoming best (it is loose)
    remain = TLIM - (time.time() - t0)
    # spend ~12s initial squeeze
    if remain > 20:
        x_best = lp_polish(x_best, min(12.0, remain - 10.0), r0=5e-4)
        best_val = max_cos(x_best)
        # sharp smooth tune
        for beta, it in [(800.0, 120), (2500.0, 120), (8000.0, 100)]:
            if TLIM - (time.time() - t0) < 12:
                break
            y = run_lbfgs(x_best, beta, it)
            v = max_cos(y)
            if v < best_val:
                x_best, best_val = y, v
        if TLIM - (time.time() - t0) > 12:
            x_best = lp_polish(x_best, 4.0, r0=3e-4)
            best_val = max_cos(x_best)

    explore_end = t0 + TLIM - 22.0
    trial = 0
    # small-perturbation rebalancing loop staying in good basin
    while time.time() < explore_end and trial < 400:
        trial += 1
        # perturb: mostly tiny to rebalance cluster, occasionally medium
        u = rng.random()
        x = x_best.copy()
        if u < 0.55:
            sigma = float(10 ** rng.uniform(-3.5, -2.0))  # 0.0003..0.01
            x = x + sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif u < 0.75:
            # move one of worst 10 points a bit
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            order = np.argsort(-per)
            k = int(rng.integers(1, 4))
            idx = order[:k]
            amp = float(10 ** rng.uniform(-2.5, -1.0))
            x[idx] += amp * rng.standard_normal((len(idx), d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        else:
            # reinsert single worst point into hole + tiny shake
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            wi = int(np.argmax(G.max(axis=1)))
            # candidate holes
            d_ = d
            cand = rng.standard_normal((6000, d_))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            mask = np.ones(n, dtype=bool); mask[wi] = False
            scores = (cand @ x[mask].T).max(axis=1)
            hole = cand[int(np.argmin(scores))]
            # blend hole with current to keep determinism small
            alpha = float(rng.uniform(0.7, 1.0))
            if rng.random() < 0.7:
                x[wi] = hole
            else:
                x[wi] = alpha * hole + (1 - alpha) * x[wi]
                x[wi] /= np.linalg.norm(x[wi])
            x += 0.002 * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)

        b1 = float(rng.uniform(300, 900))
        b2 = float(rng.uniform(2000, 9000))
        m1 = int(rng.integers(60, 130))
        m2 = int(rng.integers(60, 130))
        x = run_lbfgs(x, b1, m1)
        if time.time() >= explore_end:
            v = max_cos(x)
            if v < best_val:
                x_best, best_val = x.copy(), v
            break
        x = run_lbfgs(x, b2, m2)
        v = max_cos(x)
        if v < best_val - 2e-7:
            # promising: squeeze now
            budget = min(3.0, max(0.5, explore_end - time.time() - 0.5))
            if budget > 0.4:
                x = lp_polish(x, budget, r0=3e-4)
                v = max_cos(x)
            if v < best_val:
                x_best, best_val = x.copy(), v
        elif v < best_val:
            x_best, best_val = x.copy(), v
        elif v < best_val + 0.001 and (trial % 7 == 0) and (time.time() < explore_end - 3):
            y = lp_polish(x, 1.0, r0=3e-4)
            v2 = max_cos(y)
            if v2 < best_val:
                x_best, best_val = y.copy(), v2
        if time.time() >= explore_end:
            break

    # final intensive squeeze
    remain = TLIM - (time.time() - t0)
    if remain > 4:
        # escalating sharpness
        try:
            y = run_lbfgs(x_best, 3000.0, 150)
            if max_cos(y) < best_val:
                x_best = y; best_val = max_cos(y)
        except Exception:
            pass
    remain = TLIM - (time.time() - t0)
    if remain > 4:
        try:
            y = run_lbfgs(x_best, 10000.0, 120)
            if max_cos(y) < best_val:
                x_best = y; best_val = max_cos(y)
        except Exception:
            pass
    remain = TLIM - (time.time() - t0)
    if remain > 3:
        x_best = lp_polish(x_best, remain - 1.5, r0=2e-4)
        best_val = max_cos(x_best)
    remain = TLIM - (time.time() - t0)
    if remain > 2.5:
        x_best = micro_hill(x_best, remain - 1.0, rng)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
