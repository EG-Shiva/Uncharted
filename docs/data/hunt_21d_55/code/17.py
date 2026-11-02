import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _relax(X0, steps, a0, a1, lr0, lr1, t_deadline):
    X = X0.copy()
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        if t % 100 == 0 and time.time() > t_deadline:
            break
        frac = t / steps
        alpha = a0 * (a1 / a0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = np.max(G)
        E = np.exp(alpha * (G - M))
        np.fill_diagonal(E, 0.0)
        S = np.sum(E)
        if S <= 0 or not np.isfinite(S):
            continue
        grad = (E @ X) / S
        proj = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - proj * X
        m = b1 * m + (1.0 - b1) * gt
        v = b2 * v + (1.0 - b2) * (gt * gt)
        mhat = m / (1.0 - b1 ** t)
        vhat = v / (1.0 - b2 ** t)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        X = X - step
        X = _norm(X)
    return X

def _hole(Y, rng, n_cand=3000):
    d = Y.shape[1]
    cand = rng.standard_normal(size=(n_cand, d))
    cand = _norm(cand)
    dots = cand @ Y.T
    md = np.max(dots, axis=1)
    return cand[np.argmin(md)]

def _lp_polish(x, time_budget, t_deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < t_deadline and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        m = float(cfull.max())
        sel = cfull > m - max(20.0 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 900:
            idx = np.where(sel)[0]
            order = np.argsort(-cfull[idx])[:900]
            nsel = np.zeros_like(sel)
            nsel[idx[order]] = True
            sel = nsel
        csel = cfull[sel]
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
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
        y = _norm(y)
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time_budget is not None and time.time() > t_deadline:
            break
    return x, best

def solve():
    t0 = time.time()
    HARD_END = t0 + 170.0
    SEARCH_END = t0 + 152.0
    rng = np.random.default_rng(7)
    best = None
    best_score = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (55, 21):
                a = _norm(a)
                s = _max_cos(a)
                if s < best_score:
                    best_score = s
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal(size=(55, 21)))
        best_score = _max_cos(best)
    x_best = best.copy()
    cur = best.copy()
    cur_score = best_score
    n, d = 55, 21
    trial = 0
    while time.time() < SEARCH_END:
        trial += 1
        remain = HARD_END - time.time()
        if remain < 8:
            break
        if trial % 30 == 0:
            cur = x_best.copy()
            cur_score = best_score
        prop = cur.copy()
        mode = trial % 6
        try:
            if mode in (0, 1, 3):
                G = prop @ prop.T
                np.fill_diagonal(G, -2.0)
                mx = float(np.max(G))
                cnt = np.sum(G > mx - 0.002, axis=1)
                worst = np.max(G, axis=1)
                k = 1 if mode in (0, 3) else 2
                if rng.random() < 0.7:
                    victims = np.argsort(-cnt - 0.1 * (worst - worst.min()))[:k]
                    # tie-break jitter sometimes
                    if rng.random() < 0.25:
                        victims = rng.choice(n, size=k, replace=False)
                else:
                    victims = rng.choice(n, size=k, replace=False)
                keep = np.ones(n, dtype=bool)
                keep[victims] = False
                Y = prop[keep]
                for vi in victims:
                    h = _hole(Y, rng, n_cand=3000)
                    prop[vi] = h
                    Y = np.vstack([Y, h[None, :]])
                prop = prop + (0.008 + 0.012 * rng.random()) * rng.standard_normal(size=prop.shape)
                prop = _norm(prop)
                prop = _relax(prop, steps=700, a0=30.0, a1=600.0, lr0=0.02, lr1=0.002, t_deadline=SEARCH_END)
            elif mode == 2:
                sigma = 0.03 + 0.05 * rng.random()
                prop = prop + sigma * rng.standard_normal(size=prop.shape)
                prop = _norm(prop)
                prop = _relax(prop, steps=500, a0=60.0, a1=800.0, lr0=0.015, lr1=0.002, t_deadline=SEARCH_END)
            elif mode == 4:
                k = 5 + int(rng.integers(0, 4))
                idx = rng.choice(n, size=k, replace=False)
                prop[idx] = prop[idx] + 0.15 * rng.standard_normal(size=(k, d))
                prop = _norm(prop)
                prop = _relax(prop, steps=600, a0=40.0, a1=700.0, lr0=0.02, lr1=0.003, t_deadline=SEARCH_END)
            else:  # mode 5: small global jitter
                prop = prop + 0.012 * rng.standard_normal(size=prop.shape)
                prop = _norm(prop)
                prop = _relax(prop, steps=400, a0=100.0, a1=900.0, lr0=0.01, lr1=0.002, t_deadline=SEARCH_END)
        except Exception:
            continue
        try:
            v = _max_cos(prop)
        except Exception:
            continue
        # accept move for next basin (annealing)
        if v < cur_score or rng.random() < 0.08:
            cur = prop.copy()
            cur_score = v
        # polish only promising
        if v < best_score + 0.002 and time.time() < SEARCH_END:
            try:
                pol_end = min(time.time() + 2.5, SEARCH_END, HARD_END - 6)
                if pol_end - time.time() > 0.4:
                    xp, sp = _lp_polish(prop, 2.5, pol_end)
                    if sp < best_score and np.all(np.isfinite(xp)):
                        x_best = _norm(xp)
                        best_score = sp
                        cur = x_best.copy()
                        cur_score = best_score
            except Exception:
                pass
        # direct update without polish if big win
        if v < best_score and time.time() >= SEARCH_END:
            if np.all(np.isfinite(prop)):
                x_best = _norm(prop)
                best_score = v
    # final squeeze of global best
    try:
        remain = HARD_END - time.time() - 1.0
        if remain > 1.0 and np.all(np.isfinite(x_best)):
            xp, sp = _lp_polish(x_best, remain, HARD_END - 0.5)
            if sp < best_score and np.all(np.isfinite(xp)):
                x_best = _norm(xp)
                best_score = sp
    except Exception:
        pass
    x_best = _norm(x_best)
    return x_best
