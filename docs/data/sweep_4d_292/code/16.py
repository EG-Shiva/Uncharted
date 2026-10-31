import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contacts(x, best=None, tol=1e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    if best is None:
        best = float(G.max())
    C = G > (best - tol)
    cnt = C.sum(axis=1)
    return G, best, cnt

def _smooth_adam(X, steps, a0, a1, lr0, lr1):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2 = 0.9, 0.999
    eps = 1e-8
    p1 = 1.0
    p2 = 1.0
    for t in range(1, steps + 1):
        frac = t / float(steps)
        alpha = a0 + (a1 - a0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        D = X @ X.T
        np.fill_diagonal(D, -10.0)
        dm = float(D.max())
        # exp weights, focused on large dots
        W = np.exp(alpha * (D - dm))
        np.fill_diagonal(W, 0.0)
        G = W @ X
        s = (G * X).sum(axis=1, keepdims=True)
        Gt = G - s * X
        # normalize scale to keep lr stable
        # Gt scale grows with sum weights; use mean normalization
        m *= b1
        m += (1.0 - b1) * Gt
        v *= b2
        v += (1.0 - b2) * (Gt * Gt)
        p1 *= b1
        p2 *= b2
        mhat = m / (1.0 - p1 + 1e-20)
        vhat = v / (1.0 - p2 + 1e-20)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        # project to tangent
        sn = (step * X).sum(axis=1, keepdims=True)
        step -= sn * X
        X -= step
        nr = np.sqrt((X * X).sum(axis=1, keepdims=True))
        X /= np.maximum(nr, 1e-12)
    return X

def _greedy_reinsert(X_rem, k, rng, n_cand=15000):
    cand = rng.standard_normal((n_cand, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    Xa = X_rem
    for _ in range(k):
        # max dot of each candidate to current set
        # chunk to save memory
        best_score = 2.0
        best_pt = None
        # process in chunks
        for a in range(0, n_cand, 3000):
            cc = cand[a:a+3000]
            dd = cc @ Xa.T
            mm = dd.max(axis=1)
            j = int(np.argmin(mm))
            if float(mm[j]) < best_score:
                best_score = float(mm[j])
                best_pt = cc[j]
        jp = best_pt + 0.008 * rng.standard_normal(4)
        nrm = float(np.linalg.norm(jp))
        if nrm < 1e-12 or not np.isfinite(nrm):
            jp = best_pt.copy()
        else:
            jp = jp / nrm
        Xa = np.vstack([Xa, jp.reshape(1, 4)])
    return Xa

def _lp_polish(x, deadline):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return x
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        mval = float(c.max())
        sel = c > mval - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if getattr(res, "status", 1) != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() > deadline:
            break
    return x

def solve():
    t0 = time.time()
    TIME_LIMIT = 172.0
    explore_end = t0 + 138.0
    final_deadline = t0 + TIME_LIMIT
    rng = np.random.default_rng(20261031)
    n_target, d_target = 292, 4
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n_target, d_target) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if np.min(nr) > 1e-9:
                    cands.append(a / nr)
        except Exception:
            pass
    if len(cands) == 0:
        X_best = rng.standard_normal((n_target, d_target))
        X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
    else:
        vals = [_max_cos(c) for c in cands]
        X_best = cands[int(np.argmin(vals))].copy()
    best_val = _max_cos(X_best)
    n, d = n_target, d_target
    it = 0
    # quick contact info
    while time.time() < explore_end:
        it += 1
        # contact counts of best
        try:
            _, bv, cnt = _contacts(X_best, best_val, tol=1e-7)
            best_val = float(bv)
        except Exception:
            cnt = np.zeros(n, dtype=int)
        order = np.argsort(-cnt)
        op = int(rng.integers(0, 5))
        X = None
        try:
            if op == 0:
                # gaussian shake + anneal
                sigmas = [0.005, 0.015, 0.04, 0.10, 0.25]
                sigma = sigmas[int(rng.integers(len(sigmas)))]
                X = X_best + sigma * rng.standard_normal((n, d))
                X /= np.linalg.norm(X, axis=1, keepdims=True)
                r = rng.random()
                if r < 0.4:
                    steps, a0, a1, lr0, lr1 = 900, 30.0, 400.0, 0.03, 0.003
                elif r < 0.7:
                    steps, a0, a1, lr0, lr1 = 500, 80.0, 800.0, 0.02, 0.002
                else:
                    steps, a0, a1, lr0, lr1 = 1400, 20.0, 250.0, 0.04, 0.005
                X = _smooth_adam(X, steps, a0, a1, lr0, lr1)
            elif op == 1:
                # random kill-reinsert
                k = int(rng.integers(1, 8))
                idx = rng.choice(n, size=k, replace=False)
                X = X_best.copy()
                R = rng.standard_normal((k, d))
                R /= np.linalg.norm(R, axis=1, keepdims=True)
                X[idx] = R
                X /= np.linalg.norm(X, axis=1, keepdims=True)
                X = _smooth_adam(X, 700, 30.0, 500.0, 0.03, 0.003)
            elif op == 2 or op == 3:
                # guided excision of jammed cluster -> holes
                # number to move
                if op == 2:
                    k = int(rng.integers(2, 7))
                else:
                    k = int(rng.integers(4, 10))
                # pick seed among top-30 overloaded
                top = order[:30]
                seed = top[int(rng.integers(len(top)))]
                # nearest neighbors of seed in best
                Grow = X_best @ X_best[seed]
                Grow[seed] = -2.0
                nn = np.argsort(-Grow)
                # take k-1 neighbors + seed (or random high points)
                if rng.random() < 0.6:
                    rem = np.concatenate([[seed], nn[:k-1]])
                else:
                    rem = np.concatenate([[seed], rng.choice(top, size=k-1, replace=False)])
                rem = np.unique(rem)
                k = len(rem)
                mask = np.ones(n, dtype=bool)
                mask[rem] = False
                X_rem = X_best[mask]
                Xa = _greedy_reinsert(X_rem, k, rng, n_cand=15000)
                # small shake then smooth
                Xa = Xa + 0.005 * rng.standard_normal(Xa.shape)
                Xa /= np.linalg.norm(Xa, axis=1, keepdims=True)
                r = rng.random()
                if r < 0.5:
                    Xa = _smooth_adam(Xa, 900, 40.0, 600.0, 0.025, 0.002)
                else:
                    Xa = _smooth_adam(Xa, 500, 100.0, 900.0, 0.015, 0.002)
                X = Xa
            else:
                # heavy shake + long anneal (different basin)
                X = X_best + 0.35 * rng.standard_normal((n, d))
                X /= np.linalg.norm(X, axis=1, keepdims=True)
                # also reinsert 2 random
                if rng.random() < 0.5:
                    idx = rng.choice(n, size=3, replace=False)
                    R = rng.standard_normal((3, d))
                    R /= np.linalg.norm(R, axis=1, keepdims=True)
                    X[idx] = R
                    X /= np.linalg.norm(X, axis=1, keepdims=True)
                X = _smooth_adam(X, 1100, 25.0, 350.0, 0.035, 0.004)
        except Exception:
            continue
        if X is None or X.shape != (n, d):
            continue
        # quick check
        try:
            v = _max_cos(X)
        except Exception:
            continue
        # only LP-polish if promising (within margin) to save time
        # margin shrinks as we improve: allow up to +0.003
        if v < best_val + 0.003:
            # short LP budget 1.2s (or less if near deadline)
            lp_dead = min(time.time() + 1.4, explore_end)
            if lp_dead - time.time() < 0.3:
                # no time for LP, just hill-check
                if v < best_val:
                    X_best, best_val = X, v
                continue
            try:
                Xp = _lp_polish(X, lp_dead)
                vp = _max_cos(Xp)
            except Exception:
                continue
            if vp < best_val - 1e-12:
                X_best, best_val = Xp, vp
                # save? not needed
            elif v < best_val - 1e-12:
                # LP made it worse? keep raw if better
                X_best, best_val = X, v
        # occasional time check
        if time.time() >= explore_end:
            break
    # final polish of best with remaining time
    try:
        X_best = _lp_polish(X_best, final_deadline - 0.5)
        best_val = _max_cos(X_best)
    except Exception:
        pass
    X_best = X_best / np.linalg.norm(X_best, axis=1, keepdims=True)
    # safety: remove NaN
    if not np.all(np.isfinite(X_best)):
        X_best = np.load("best.npy").astype(np.float64)
        X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best
