import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def anneal_relax(X0, steps, a0, a1, lr0, lr1):
    X = X0.copy()
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
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
        m = b1 * m + (1 - b1) * gt
        v = b2 * v + (1 - b2) * (gt * gt)
        mhat = m / (1 - b1 ** t)
        vhat = v / (1 - b2 ** t)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        X = X - step
        X = norm_rows(X)
    return X

def find_hole(Y, rng, n_cand=3000):
    d = Y.shape[1]
    cand = rng.standard_normal(size=(n_cand, d))
    cand = norm_rows(cand)
    dots = cand @ Y.T
    md = np.max(dots, axis=1)
    return cand[np.argmin(md)]

def lp_polish(x, deadline):
    # monotonic trust-region LP until deadline (time.time value)
    x = norm_rows(x)
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        thr = max(20 * radius, 1e-9)
        sel = c > m - thr
        # cap constraints for speed
        if np.sum(sel) > 2500:
            # keep largest 2500
            idx = np.argsort(-c)
            sel = np.zeros_like(sel)
            sel[idx[:2500]] = True
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
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
        y = norm_rows(y)
        v = max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            # stay at best_x if x drifted? keep x as best_x for stability when stuck
            # if no progress, shrink around best
            x = best_x.copy()
            radius *= 0.5
        # early stop if time nearly up
        if time.time() >= deadline:
            break
    return norm_rows(best_x), best

def solve():
    t_start = time.time()
    HARD_END = t_start + 172.0
    SEARCH_END = t_start + 145.0
    rng = np.random.default_rng(0)
    n, d = 55, 21
    best = None
    best_score = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (55, 21):
                a = norm_rows(a)
                s = max_cos(a)
                if s < best_score:
                    best_score = s
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = norm_rows(rng.standard_normal(size=(n, d)))
        best_score = max_cos(best)
    cur = best.copy()
    cur_score = best_score
    T = 0.0012
    trial = 0
    # quick LP pre-polish of starting point for 3 sec to jam it (monotonic)
    try:
        if HARD_END - time.time() > 10:
            best, best_score_new = lp_polish(best, min(time.time() + 3.0, SEARCH_END))
            best_score = float(best_score_new)
            cur = best.copy(); cur_score = best_score
    except Exception:
        pass
    while time.time() < SEARCH_END:
        trial += 1
        if trial % 30 == 0:
            cur = best.copy()
            cur_score = best_score
        prop = cur.copy()
        mode = trial % 6
        if mode == 0 or mode == 1:
            G = prop @ prop.T
            np.fill_diagonal(G, -2.0)
            mx = float(np.max(G))
            cnt = np.sum(G > mx - 2e-3, axis=1)
            k = 1 if mode == 0 else 2
            victims = np.argsort(-cnt)[:k]
            if rng.random() < 0.3:
                victims = rng.choice(n, size=k, replace=False)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[victims] = False
            Y = prop[keep_mask]
            for vi in victims:
                h = find_hole(Y, rng, n_cand=2500)
                prop[vi] = h
                Y = np.vstack([Y, h[None, :]])
            prop = prop + 0.008 * rng.standard_normal(size=prop.shape)
            prop = norm_rows(prop)
            steps, a0, a1, lr0, lr1 = 1400, 25.0, 450.0, 0.02, 0.002
        elif mode == 2:
            prop = prop + 0.05 * rng.standard_normal(size=prop.shape)
            prop = norm_rows(prop)
            steps, a0, a1, lr0, lr1 = 1000, 20.0, 350.0, 0.015, 0.002
        elif mode == 3:
            prop = prop + 0.18 * rng.standard_normal(size=prop.shape)
            prop = norm_rows(prop)
            steps, a0, a1, lr0, lr1 = 1300, 15.0, 350.0, 0.02, 0.002
        elif mode == 4:
            k = int(rng.integers(1, 4))
            idx = rng.choice(n, size=k, replace=False)
            for ii in idx:
                vv = rng.standard_normal(size=(d,))
                vv /= np.linalg.norm(vv)
                prop[ii] = vv
            prop = prop + 0.015 * rng.standard_normal(size=prop.shape)
            prop = norm_rows(prop)
            steps, a0, a1, lr0, lr1 = 1200, 25.0, 450.0, 0.015, 0.002
        else:
            # rattler-targeted: random tangent move of low-contact points
            G = prop @ prop.T
            np.fill_diagonal(G, -2.0)
            mx = float(np.max(G))
            cnt = np.sum(G > mx - 1e-3, axis=1)
            ratt = np.where(cnt == 0)[0]
            if len(ratt) > 0:
                k = min(len(ratt), int(rng.integers(1, 4)))
                idx = rng.choice(ratt, size=k, replace=False)
                for ii in idx:
                    vv = rng.standard_normal(size=(d,))
                    vv /= np.linalg.norm(vv)
                    prop[ii] = norm_rows((prop[ii] + 0.5 * vv)[None, :])[0]
            else:
                prop = prop + 0.02 * rng.standard_normal(size=prop.shape)
            prop = norm_rows(prop)
            steps, a0, a1, lr0, lr1 = 1100, 20.0, 400.0, 0.015, 0.002
        if time.time() > SEARCH_END - 8:
            steps = min(steps, 500)
        new = anneal_relax(prop, steps, a0, a1, lr0, lr1)
        s = max_cos(new)
        if s < best_score - 1e-12:
            # refine new best with tiny extra high-alpha polish (monotonic check)
            extra = anneal_relax(new, 300, 100.0, 600.0, 0.006, 0.001)
            se = max_cos(extra)
            if se < s:
                new, s = extra, se
            # quick LP touch (0.6s) if promising and time allows
            if time.time() < SEARCH_END - 5:
                try:
                    new2, s2 = lp_polish(new, min(time.time() + 0.6, SEARCH_END))
                    if s2 < s:
                        new, s = new2, s2
                except Exception:
                    pass
            best, best_score = new.copy(), s
            cur, cur_score = new.copy(), s
        elif s < cur_score:
            cur, cur_score = new.copy(), s
        else:
            if rng.random() < np.exp(-(s - cur_score) / T):
                cur, cur_score = new.copy(), s
        if trial > 150 and best_score < 0.0666:
            T = 0.0006
    # final monotonic polish of global best
    remain_end = HARD_END - 1.0
    if time.time() < remain_end:
        # short high-alpha smooth improvement only if it helps
        try:
            cand = anneal_relax(best, 600, 80.0, 700.0, 0.005, 0.0008)
            sc = max_cos(cand)
            if sc < best_score:
                best, best_score = cand, sc
        except Exception:
            pass
    if time.time() < remain_end:
        try:
            best, best_score = lp_polish(best, remain_end)
        except Exception:
            pass
    best = norm_rows(best)
    return best
