import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, seconds, deadline=None):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    try:
        best = _max_cos(X)
    except Exception:
        return X
    radius = 2e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        G = X @ X.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        cs = c[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[~np.isfinite(yn)] = 1.0
        yn[yn == 0] = 1.0
        y /= yn
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _quench(X_init, steps, T0, T1, lr0, lr1, rng=None, noise=0.0):
    X = X_init.copy()
    n, d = X.shape
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lT0, lT1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best_m = float(G0.max())
    best_X = X.copy()
    for s in range(1, steps + 1):
        frac = (s - 1) / max(steps - 1, 1)
        T = float(np.exp(lT0 * (1 - frac) + lT1 * frac))
        lr = float(np.exp(ll0 * (1 - frac) + ll1 * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < best_m:
            best_m = m
            best_X = X.copy()
        D = (G - m) / T
        E = np.exp(D)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            S = 1.0
        Gr = (E @ X) / S
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        if rng is not None and noise > 0:
            N = rng.standard_normal((n, d)) * noise
            N -= X * np.sum(N * X, axis=1, keepdims=True)
            Gt = Gt + N
        if not np.all(np.isfinite(Gt)):
            break
        m1 = b1 * m1 + (1 - b1) * Gt
        v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** s
        c2 = 1.0 - b2 ** s
        step = (m1 / c1) / (np.sqrt(v1 / c2) + eps)
        sn = float(np.linalg.norm(step))
        mxn = 8.0 * np.sqrt(n * d)
        if sn > mxn:
            step *= (mxn / sn)
        X = X - lr * step
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[~np.isfinite(nr)] = 1.0
        nr[nr == 0] = 1.0
        X /= nr
    Gf = X @ X.T
    np.fill_diagonal(Gf, -2.0)
    mf = float(Gf.max())
    if mf < best_m:
        best_m = mf
        best_X = X.copy()
    return best_X

def _per_point_stats(X, m):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    cnt = (G > m - 2e-4).sum(axis=1)
    mx = G.max(axis=1)
    return G, cnt, mx

def _greedy_candidate(Y, rng, n_try=2000):
    d = Y.shape[1]
    C = rng.standard_normal((n_try, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    D = C @ Y.T
    w = D.max(axis=1)
    return C[np.argmin(w)]

def _single_point_try(X, cur_max, rng):
    n, d = X.shape
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    # order points by worst max
    pmx = G.max(axis=1)
    order = np.argsort(-pmx)
    improved = False
    for ii in range(min(n, 6)):
        i = int(order[ii])
        mask = np.ones(n, dtype=bool); mask[i] = False
        Y = X[mask]
        Go = Y @ Y.T
        np.fill_diagonal(Go, -2.0)
        others_max = float(Go.max()) if Go.size else -2.0
        if others_max >= cur_max - 1e-12:
            continue
        # sample
        C = rng.standard_normal((3000, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        D = C @ Y.T
        w = D.max(axis=1)
        bi = int(np.argmin(w))
        bv = float(w[bi])
        if bv >= cur_max:
            continue
        cand = C[bi:bi+1].copy()
        # refine cand with smooth max descent (fixed Y)
        cc = cand.copy()
        for s in range(200):
            T = 0.004 * (0.995 ** s) + 0.0003
            lr = 0.02 * (0.99 ** s) + 0.002
            dots = (cc @ Y.T).ravel()
            mm = float(dots.max())
            E = np.exp((dots - mm) / T)
            E /= E.sum()
            g = (E @ Y)
            g = g - cc * float((g * cc).sum())
            cc = cc - lr * g
            cc /= np.linalg.norm(cc)
        dots = (cc @ Y.T).ravel()
        bv2 = float(dots.max())
        if bv2 < cur_max - 2e-7:
            Xn = X.copy()
            Xn[i] = cc.ravel() / np.linalg.norm(cc.ravel())
            nm = _max_cos(Xn)
            if nm < cur_max:
                X = Xn
                cur_max = nm
                improved = True
    return X, cur_max, improved

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    FINAL_POLISH = 22.0
    HOP_END = HARD_END - FINAL_POLISH - 1.0
    rng = np.random.default_rng(20261101)
    best_X = None
    try:
        best_X = np.load("best.npy").astype(float)
    except Exception:
        pass
    if best_X is None or best_X.shape != (71, 12) or not np.all(np.isfinite(best_X)):
        try:
            best_X = np.load("record.npy").astype(float)
        except Exception:
            best_X = rng.standard_normal((71, 12))
    best_X = np.asarray(best_X, dtype=float)
    best_X /= np.linalg.norm(best_X, axis=1, keepdims=True)
    best_score = _max_cos(best_X)
    n, d = 71, 12
    it = 0
    sigmas = [0.015, 0.04, 0.09, 0.20, 0.32, 0.06, 0.12]
    while time.time() < HOP_END:
        it += 1
        if time.time() > HOP_END - 2.0:
            break
        r = rng.random()
        X0 = best_X.copy()
        Gc, cntc, mxc = _per_point_stats(X0, best_score)
        order_worst = np.argsort(-mxc - 0.0005 * cntc)
        if r < 0.42:
            sigma = sigmas[it % len(sigmas)]
            noise = rng.standard_normal((n, d))
            # amplify overloaded
            w = np.ones(n)
            k = 10
            w[order_worst[:k]] *= 2.2
            noise *= (sigma * w[:, None])
            # tangentialize
            noise -= X0 * np.sum(noise * X0, axis=1, keepdims=True)
            X0 = X0 + noise
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            steps = 900 if sigma < 0.1 else 1200
            T0 = 0.012 if sigma < 0.1 else 0.018
            T1 = 0.00035
            lr0, lr1 = 0.025, 0.0025
            Xq = _quench(X0, steps, T0, T1, lr0, lr1, rng if sigma > 0.05 else None, noise=0.0005 if sigma > 0.1 else 0.0)
        elif r < 0.72:
            # kill and greedy refill
            k = int(rng.integers(2, 6))
            kill = order_worst[:k].copy()
            # occasionally random kill
            if rng.random() < 0.25:
                kill = rng.choice(n, size=k, replace=False)
            X0[kill] = rng.standard_normal((k, d))
            X0[kill] /= np.linalg.norm(X0[kill], axis=1, keepdims=True)
            # greedy farthest sequential refine for killed
            for idx in kill:
                mask = np.ones(n, dtype=bool); mask[idx] = False
                Y = X0[mask]
                X0[idx] = _greedy_candidate(Y, rng, n_try=3000)
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            Xq = _quench(X0, 1100, 0.016, 0.0003, 0.03, 0.002, rng, noise=0.0008)
        elif r < 0.88:
            # large subset shake
            m = int(rng.integers(8, 20))
            idx = rng.choice(n, size=m, replace=False)
            X0[idx] += rng.standard_normal((m, d)) * 0.35
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            Xq = _quench(X0, 1300, 0.020, 0.0004, 0.035, 0.003, rng, noise=0.001)
        else:
            # fresh random with quench (diversity)
            Xr = rng.standard_normal((n, d))
            Xr /= np.linalg.norm(Xr, axis=1, keepdims=True)
            # mix with best: keep 60 of best, 11 random farthest
            keep = rng.choice(n, size=60, replace=False)
            mask_keep = np.zeros(n, dtype=bool); mask_keep[keep] = True
            X0 = best_X.copy()
            # replace complement greedily
            for idx in np.where(~mask_keep)[0]:
                mask = np.ones(n, dtype=bool); mask[idx] = False
                Y = X0[mask]
                # use random candidate pool
                X0[idx] = _greedy_candidate(Y, rng, n_try=2000)
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            Xq = _quench(X0, 1400, 0.022, 0.0004, 0.04, 0.003, rng, noise=0.0012)
        try:
            qscore = _max_cos(Xq)
        except Exception:
            continue
        # single-point relocation if promising
        if qscore < best_score + 0.004 and time.time() < HOP_END - 3.0:
            try:
                Xq2, qscore2, _ = _single_point_try(Xq, qscore, rng)
                Xq, qscore = Xq2, qscore2
            except Exception:
                pass
        # short LP touch if very close (cheap topology snap) - only occasionally
        if qscore < best_score + 0.0005 and (it % 5 == 0):
            try:
                Xq = _lp_polish(Xq, 1.2, deadline=HOP_END)
                qscore = _max_cos(Xq)
            except Exception:
                pass
        if qscore < best_score - 1e-12:
            best_X, best_score = Xq, qscore
        # opportunistic fine quench around new best at low T
        if qscore < best_score + 1e-9 and rng.random() < 0.15 and time.time() < HOP_END - 5:
            try:
                Xf = _quench(best_X, 500, 0.002, 0.00015, 0.008, 0.001)
                sf = _max_cos(Xf)
                if sf < best_score:
                    best_X, best_score = Xf, sf
            except Exception:
                pass
    # final polish: low-T quench + LP
    try:
        if time.time() < HARD_END - 2:
            Xf = _quench(best_X, 700, 0.003, 0.00012, 0.010, 0.0012)
            sf = _max_cos(Xf)
            if sf < best_score:
                best_X, best_score = Xf, sf
    except Exception:
        pass
    try:
        remain = HARD_END - time.time() - 0.5
        if remain > 3:
            Xp = _lp_polish(best_X, remain, deadline=HARD_END - 0.3)
            sp = _max_cos(Xp)
            if sp < best_score:
                best_X = Xp
    except Exception:
        pass
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    return best_X
