import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _degrees(x, eps=1e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    cnt = (G > m - eps).sum(axis=1)
    return cnt, m


def _anneal(x, steps, b0, b1, lr0, mu):
    n, d = x.shape
    v = np.zeros_like(x)
    lb = np.log(b1 / b0) if b1 > b0 else 0.0
    for t in range(steps):
        frac = t / max(1, steps - 1)
        beta = b0 * np.exp(lb * frac)
        lr = lr0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-300
        A = (W @ x) / S
        dot = (A * x).sum(axis=1, keepdims=True)
        Gt = A - dot * x
        v = mu * v + Gt
        vd = (v * x).sum(axis=1, keepdims=True)
        v = v - vd * x
        x = x - lr * v
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(nrm, 1e-12, out=nrm)
        x /= nrm
    return x


def _refine_void(v, Y, steps=200, beta=200.0, lr=0.05):
    # minimize max dot to fixed set Y
    for _ in range(steps):
        g = Y @ v  # (m,)
        gm = g.max()
        w = np.exp(beta * (g - gm))
        s = w.sum() + 1e-300
        a = (w @ Y) / s  # (4,)
        # move away from weighted neighbour, tangential
        grad = a - (a @ v) * v
        v = v - lr * grad
        n = np.linalg.norm(v)
        if n < 1e-12:
            break
        v = v / n
    return v


def _find_void(Y, rng, n_cand=30000):
    # random candidates on S^3, pick minimal max-dot
    C = rng.standard_normal((n_cand, 4))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    # chunk to save memory
    best_i = -1
    best_s = 1e9
    chunk = 5000
    scores = np.empty(n_cand)
    for s in range(0, n_cand, chunk):
        e = min(s + chunk, n_cand)
        D = C[s:e] @ Y.T
        scores[s:e] = D.max(axis=1)
    # pick among top few for diversity
    order = np.argpartition(scores, 5)[:5]
    # choose best of top-5 randomly to diversify (rng driven)
    pick = order[rng.integers(len(order))]
    v = C[pick].copy()
    v = _refine_void(v, Y)
    return v


def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 2e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        Gfull = x @ x.T
        iu_c = Gfull[iu]
        m = iu_c.max()
        thr = max(20 * radius, 1e-9)
        sel = iu_c > m - thr
        if np.count_nonzero(sel) == 0:
            radius *= 0.5
            continue
        I = iu[0][sel]
        J = iu[1][sel]
        c = iu_c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = (I[:, None] * d + np.arange(d)).ravel()
        cJ = (J[:, None] * d + np.arange(d)).ravel()
        # interleave: for each pair, I-block, J-block, t-col
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        cols[0::(2 * d + 1)] = cI[0::d]
        # easier: build via concatenate per block
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    TIME_BUDGET = 172.0
    FINAL_POLISH = 30.0

    rng = np.random.default_rng(20261031)
    # load start
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (292, 4):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        a = rng.standard_normal((292, 4))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    # pick best of loaded
    vals = [_max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    best_val = float(vals[bi])

    n, d = 292, 4
    explore_end = t0 + TIME_BUDGET - FINAL_POLISH

    it = 0
    # quick initial polish to lock local optimum
    # (short, to leave time for exploration)
    if time.time() < explore_end:
        x_best = _lp_polish(x_best, min(4.0, max(1.0, explore_end - time.time())))
        best_val = _max_cos(x_best)

    sigmas = [0.005, 0.015, 0.04, 0.10, 0.25]

    while time.time() < explore_end and it < 300:
        it += 1
        time_left = explore_end - time.time()
        if time_left < 2.5:
            break
        X = x_best.copy()
        mode = it % 4
        if mode == 0 or mode == 2:
            # targeted void-guided reinsert of jammed + rattlers
            degs, _ = _degrees(X, eps=1e-7)
            # rattlers = zero contacts with tight eps
            _, _m = _degrees(X, eps=1e-9)
            # compute tight degrees for rattler detect
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            tight = (G > G.max() - 1e-9).sum(axis=1)
            ratt = np.where(tight == 0)[0]
            order = np.argsort(-degs)
            # k reinsert
            k = int(rng.integers(2, 7))
            # include rattlers first
            chosen = list(ratt[:min(len(ratt), k)])
            for o in order:
                if len(chosen) >= k:
                    break
                if o not in chosen:
                    # diversify: sometimes skip top to avoid repeat
                    if rng.random() < 0.15 and len(chosen) > 0:
                        continue
                    chosen.append(int(o))
            # small shake of rest to allow relaxation
            sigma = float(sigmas[rng.integers(len(sigmas))] * 0.3)
            X = X + sigma * rng.standard_normal((n, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            # sequentially place voids
            # build fixed set incrementally
            for idx in chosen:
                mask = np.ones(n, dtype=bool)
                mask[idx] = False
                # for sequential, use current X without idx
                Y = np.delete(X, idx, axis=0)
                # fewer candidates when little time
                nc = 20000 if time_left > 20 else 8000
                v = _find_void(Y, rng, n_cand=nc)
                X[idx] = v
            # anneal
            r = rng.random()
            if r < 0.5:
                steps, b0, b1, lr0, mu = 800, 30.0, 400.0, 0.06, 0.85
            else:
                steps, b0, b1, lr0, mu = 400, 60.0, 800.0, 0.04, 0.90
            X = _anneal(X, steps, b0, b1, lr0, mu)
            # quick LP to evaluate
            qs = 2.2 if time_left > 15 else 1.0
            Xp = _lp_polish(X, min(qs, time_left - 0.2))
            vp = _max_cos(Xp)
            if vp < best_val - 1e-12:
                x_best, best_val = Xp, vp
            elif vp < best_val:
                x_best, best_val = Xp, vp
        else:
            # random shake + softmax anneal (different basin)
            sigma = float(sigmas[rng.integers(len(sigmas))])
            X = X + sigma * rng.standard_normal((n, d))
            # random kill-reinsert 0-4 points to random pos
            if rng.random() < 0.6:
                k = int(rng.integers(1, 5))
                idx = rng.choice(n, size=k, replace=False)
                R = rng.standard_normal((k, d))
                R /= np.linalg.norm(R, axis=1, keepdims=True)
                X[idx] = R
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            r = rng.random()
            if r < 0.4:
                steps, b0, b1, lr0, mu = 1000, 25.0, 350.0, 0.06, 0.85
            elif r < 0.7:
                steps, b0, b1, lr0, mu = 500, 50.0, 600.0, 0.05, 0.88
            else:
                steps, b0, b1, lr0, mu = 250, 100.0, 1000.0, 0.03, 0.90
            X = _anneal(X, steps, b0, b1, lr0, mu)
            # evaluate raw first; only LP if promising to save time
            vr = _max_cos(X)
            # threshold: within 0.005 of best -> worth polishing
            if vr < best_val + 0.005:
                qs = 2.0 if time_left > 15 else 1.0
                Xp = _lp_polish(X, min(qs, time_left - 0.2))
                vp = _max_cos(Xp)
                if vp < best_val:
                    x_best, best_val = Xp, vp

    # final long polish
    remain = t0 + TIME_BUDGET - time.time()
    fp = min(FINAL_POLISH, max(5.0, remain))
    x_best = _lp_polish(x_best, fp)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
