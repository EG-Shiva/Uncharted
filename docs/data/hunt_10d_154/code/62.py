import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _mc(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _stats(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = float(np.max(g))
    c3 = int(np.sum(g > m - 1e-3))
    c4 = int(np.sum(g > m - 1e-4))
    return m, c3, c4

def _adam(x, steps, T0, T1, lr, deadline=1e18):
    x = _norm(x)
    m_ad = np.zeros_like(x)
    v_ad = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p = 1.0
    b2p = 1.0
    if steps > 1:
        logr = np.log(max(T1, 1e-12) / max(T0, 1e-12)) / (steps - 1)
    else:
        logr = 0.0
    for t in range(1, steps + 1):
        if t % 250 == 0 and time.time() > deadline:
            break
        T = T0 * np.exp(logr * (t - 1))
        if T < 5e-4:
            T = 5e-4
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        mv = float(G.max())
        E = np.exp((G - mv) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            break
        Grad = (E @ x) / S
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - dot * x
        b1p *= b1
        b2p *= b2
        m_ad = b1 * m_ad + (1.0 - b1) * Gt
        v_ad = b2 * v_ad + (1.0 - b2) * (Gt * Gt)
        mhat = m_ad / (1.0 - b1p)
        vhat = v_ad / (1.0 - b2p)
        step = mhat / (np.sqrt(vhat) + eps)
        frac = t / steps
        lrt = lr * (1.0 - 0.6 * frac)
        x = x - lrt * step
        x = _norm(x)
    return x

def _push(X, iters=60, eta0=0.02, margin=0.01):
    X = X.copy()
    for it in range(iters):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(G.max())
        W = (G > M - margin).astype(float)
        F = W @ X
        dot = (F * X).sum(axis=1, keepdims=True)
        Ft = F - dot * X
        eta = eta0 * (1.0 - 0.8 * it / max(iters, 1))
        X = X - eta * Ft
        X = _norm(X)
    return X

def _hole_regrow(X, rng, K, C):
    n, d = X.shape
    X = X.copy()
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    rowmax = G.max(axis=1)
    order = np.argsort(-rowmax)
    n_worst = min(K, max(2, K // 2 + 1))
    idx = list(order[:n_worst])
    if K > n_worst:
        pool = np.arange(n)
        extra = rng.choice(pool, size=K - n_worst, replace=False)
        for e in extra:
            if int(e) not in idx:
                idx.append(int(e))
            else:
                # pick a random not already in
                cand = int(rng.integers(0, n))
                if cand not in idx:
                    idx.append(cand)
    idx = idx[:K]
    rng.shuffle(idx)
    for vv in idx:
        vv = int(vv)
        mask = np.ones(n, dtype=bool)
        mask[vv] = False
        Xt = X[mask]
        cand = _norm(rng.standard_normal((C, d)))
        # max dot to remaining set
        md = (cand @ Xt.T).max(axis=1)
        bi = int(np.argmin(md))
        best = cand[bi]
        # local refine best candidate vs its nearest neighbours (few subgradient steps)
        y = best.copy()
        for _ in range(25):
            dots = Xt @ y
            # top few neighbours
            k = min(6, Xt.shape[0])
            nn = np.argsort(-dots)[:k]
            F = Xt[nn].sum(axis=0)
            # project to tangent at y
            Ft = F - np.dot(F, y) * y
            nm = np.linalg.norm(Ft)
            if nm < 1e-12:
                break
            y = y - 0.05 * Ft / nm
            y = y / np.linalg.norm(y)
        # check if refined is better
        m0 = float((Xt @ best).max())
        m1 = float((Xt @ y).max())
        X[vv] = y if m1 < m0 else best
    return X

def _lp_polish(x, seconds, deadline=1e18):
    x = _norm(x)
    n, d = x.shape
    best = _mc(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    itn = 0
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-14 or itn >= 40:
            break
        itn += 1
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 2800:
            o = np.argsort(-c[sel])[:2800]
            I, J = I[o], J[o]
            cv = c[sel][o]
        else:
            cv = c[sel]
            p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _mc(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    rng = np.random.default_rng(0)
    n, d = 154, 10
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d):
                a = _norm(a.astype(float))
                v = _mc(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_val = _mc(best)
    Xbest = best.copy()
    Xcur = best.copy()
    cur_val = best_val
    _, cur_c3, cur_c4 = _stats(Xcur)
    best_c3 = cur_c3

    HOP_END = t0 + 128.0
    trial = 0
    while time.time() < HOP_END:
        trial += 1
        # annealing temperature for wandering acceptance
        frac = (time.time() - t0) / max(HOP_END - t0, 1.0)
        wander_T = 0.004 * (1.0 - frac) + 0.0005
        r = rng.random()
        if r < 0.55:
            K = int(rng.integers(6, 13))
            C = 9000 if trial % 3 else 14000
        elif r < 0.8:
            K = int(rng.integers(12, 20))
            C = 7000
        else:
            K = int(rng.integers(3, 7))
            C = 12000
        Xtry = _hole_regrow(Xcur, rng, K, C)
        # quench at moderate temperature (sees basin, not 1e-8 ties)
        if K >= 12:
            steps, T0, T1, lr = 700, 0.05, 0.006, 0.018
        else:
            steps, T0, T1, lr = 550, 0.04, 0.004, 0.018
        Xtry = _adam(Xtry, steps, T0, T1, lr, deadline=HOP_END)
        Xtry = _push(Xtry, iters=50, eta0=0.015, margin=0.01)
        m, c3, c4 = _stats(Xtry)
        # acceptance: lexicographic with annealing
        improved_best = m < best_val - 2e-9
        # also accept as best if equal max but clearly fewer near contacts
        if not improved_best and abs(m - best_val) < 5e-9 and c3 < best_c3 - 30:
            # polish check: quick lp to see true max
            improved_best = True
        if improved_best:
            # verify with short LP snap (cheap, 1s) to get true local max
            if time.time() < HOP_END - 4:
                try:
                    Xp = _lp_polish(Xtry, 1.0, deadline=HOP_END)
                    mp = _mc(Xp)
                    if mp < m:
                        Xtry, m = Xp, mp
                        _, c3, c4 = _stats(Xtry)
                except Exception:
                    pass
            if m < best_val:
                Xbest = Xtry.copy()
                best_val = m
                best_c3 = c3
                Xcur = Xtry.copy()
                cur_val, cur_c3 = m, c3
                continue
        # wandering move acceptance
        if m < cur_val - 1e-9:
            Xcur, cur_val, cur_c3 = Xtry.copy(), m, c3
        elif m < cur_val + 0.012:
            # lexicographic bonus for fewer near-max pairs
            bonus = -0.00015 * max(0, cur_c3 - c3) / 100.0
            delta = (m - cur_val) + bonus
            if delta < 0 or rng.random() < np.exp(-delta / wander_T):
                Xcur, cur_val, cur_c3 = Xtry.copy(), m, c3
        # occasional restart of walker to best
        if trial % 25 == 0 and rng.random() < 0.5:
            Xcur, cur_val = Xbest.copy(), best_val
            _, cur_c3, _ = _stats(Xcur)
        # occasional global jitter of walker to avoid stagnation
        if trial % 12 == 0:
            Gc = Xcur @ Xcur.T
            np.fill_diagonal(Gc, -2.0)
            rm = Gc.max(axis=1)
            worst = np.argsort(-rm)[:4]
            Xcur[worst] += 0.05 * rng.standard_normal((len(worst), d))
            Xcur = _norm(Xcur)
            cur_val = _mc(Xcur)
            _, cur_c3, _ = _stats(Xcur)

    # final polish of best
    remaining = DEAD - time.time() - 1.5
    if remaining > 5:
        try:
            Xbest = _lp_polish(Xbest, min(remaining - 2, 14.0), deadline=DEAD - 2.0)
            best_val = _mc(Xbest)
        except Exception:
            pass
    # micro shake squeeze
    t_left = DEAD - time.time() - 1.0
    if t_left > 5:
        for _ in range(3):
            if time.time() > DEAD - 2.5:
                break
            try:
                Xt = Xbest + 3e-4 * rng.standard_normal((n, d))
                Xt = _norm(Xt)
                Xt = _lp_polish(Xt, 1.2, deadline=DEAD - 1.0)
                vt = _mc(Xt)
                if vt < best_val:
                    Xbest, best_val = Xt, vt
            except Exception:
                break
    Xbest = _norm(Xbest)
    return Xbest
