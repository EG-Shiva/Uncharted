import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds):
    """Trust-region LP: move points tangentially to lower the largest cosine."""
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


def refine_momentum(X_init, iters, beta, delta, lr0, lr1, mu):
    X = X_init.copy()
    n, d = X.shape
    V = np.zeros_like(X)
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    cm = float(G.max())
    best_val = cm
    best_X = X.copy()
    lr = float(lr0)
    decay = float((lr1 / lr0) ** (1.0 / max(iters - 1, 1))) if iters > 1 else 1.0
    for _ in range(iters):
        # exponential weights on near-max pairs
        # W = exp(beta*(G-cm)) where G > cm-delta else 0
        W = np.exp(beta * (G - cm))
        # threshold
        W[G < cm - delta] = 0.0
        np.fill_diagonal(W, 0.0)
        GX = W @ X
        dot = np.einsum('ij,ij->i', GX, X)
        Gr = GX - dot[:, None] * X
        V = mu * V - lr * Gr
        X += V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cm = float(G.max())
        if cm < best_val:
            best_val = cm
            best_X = X.copy()
        lr *= decay
        if not np.isfinite(cm):
            break
        if cm > 0.95:
            break
    return best_X, best_val


def solve():
    t0 = time.time()
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 168, 8

    cands = []
    try:
        b = np.load("best.npy")
        b = b / np.linalg.norm(b, axis=1, keepdims=True)
        cands.append(b)
    except Exception:
        pass
    try:
        r = np.load("record.npy")
        r = r / np.linalg.norm(r, axis=1, keepdims=True)
        cands.append(r)
    except Exception:
        pass
    if len(cands) == 0:
        X0 = rng.standard_normal((n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands.append(X0)
    vals = [max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    gbest = cands[bi].copy()
    gval = float(vals[bi])

    # contact counts of gbest for targeted moves
    def contact_counts(X, thr):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cm = G.max()
        C = (G > cm - thr).sum(axis=1)
        return C, cm

    SEARCH_BUDGET = 120.0
    trial = 0
    # pre-allocate? loop
    while time.time() - t0 < SEARCH_BUDGET and trial < 300:
        trial += 1
        # --- propose start ---
        u = rng.random()
        if u < 0.55:
            sigma = 10.0 ** rng.uniform(-2.5, -0.7)
            X_init = gbest + sigma * rng.standard_normal((n, d))
            X_init /= np.linalg.norm(X_init, axis=1, keepdims=True)
        elif u < 0.80:
            # subset reinsert random
            X_init = gbest.copy()
            k = int(rng.integers(4, 24))
            idx = rng.choice(n, size=k, replace=False)
            if rng.random() < 0.5:
                R = rng.standard_normal((k, d))
                R /= np.linalg.norm(R, axis=1, keepdims=True)
                X_init[idx] = R
            else:
                sigma2 = 10.0 ** rng.uniform(-1.5, -0.5)
                X_init[idx] += sigma2 * rng.standard_normal((k, d))
                X_init[idx] /= np.linalg.norm(X_init[idx], axis=1, keepdims=True)
        else:
            # targeted: perturb most-contacted points strongly
            try:
                cc, _ = contact_counts(gbest, 0.02)
                prob = (cc.astype(float) + 1.0)
                prob /= prob.sum()
                k = int(rng.integers(5, 20))
                idx = rng.choice(n, size=k, replace=False, p=prob)
            except Exception:
                idx = rng.choice(n, size=10, replace=False)
                k = len(idx)
            X_init = gbest.copy()
            sigma3 = 10.0 ** rng.uniform(-2.0, -0.7)
            X_init[idx] += sigma3 * rng.standard_normal((k, d))
            X_init /= np.linalg.norm(X_init, axis=1, keepdims=True)

        # --- sample hyperparams ---
        iters = int(rng.integers(700, 2400))
        beta = float(rng.choice([25.0, 45.0, 70.0, 100.0, 140.0, 190.0]) + rng.uniform(-8, 8))
        beta = max(15.0, beta)
        delta = float(rng.uniform(0.02, 0.08))
        lr0 = float(10.0 ** rng.uniform(-3.5, -2.0))
        lr1 = float(lr0 * (10.0 ** rng.uniform(-2.0, -0.5)))
        mu = float(rng.uniform(0.70, 0.95))

        # shrink iters if little time left
        remain = SEARCH_BUDGET - (time.time() - t0)
        if remain < 8 and iters > 800:
            iters = 800

        try:
            Xo, vo = refine_momentum(X_init, iters, beta, delta, lr0, lr1, mu)
        except Exception:
            continue
        if vo < gval and np.isfinite(vo):
            gbest = Xo
            gval = vo
        # opportunistic fine polish when close: small-lr short refine from new point
        if vo < gval + 0.002 and (time.time() - t0) < SEARCH_BUDGET:
            # quick greedy cool-down with large beta small lr (cheap, few hundred steps)
            try:
                Xo2, vo2 = refine_momentum(Xo, 400, 180.0, 0.03, 3e-4, 5e-5, 0.85)
                if vo2 < gval:
                    gbest = Xo2
                    gval = vo2
            except Exception:
                pass

    # final squeeze: short momentum cool + LP
    try:
        if time.time() - t0 < 165:
            Xo, vo = refine_momentum(gbest, 1200, 150.0, 0.04, 5e-4, 3e-5, 0.9)
            if vo < gval:
                gbest, gval = Xo, vo
    except Exception:
        pass

    remain = 178.0 - (time.time() - t0)
    if remain > 6:
        lp_sec = min(32.0, remain - 2.0)
        try:
            polished = lp_polish(gbest, lp_sec)
            pv = max_cos(polished)
            if pv < gval and np.isfinite(pv):
                gbest, gval = polished, pv
        except Exception:
            pass

    gbest = gbest / np.linalg.norm(gbest, axis=1, keepdims=True)
    # safety: replace any bad row
    bad = ~np.all(np.isfinite(gbest), axis=1)
    if np.any(bad):
        R = rng.standard_normal((int(bad.sum()), d))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        gbest[bad] = R
    norms = np.linalg.norm(gbest, axis=1, keepdims=True)
    norms[~np.isfinite(norms)] = 1.0
    norms[norms < 1e-12] = 1.0
    gbest = gbest / norms
    return gbest
