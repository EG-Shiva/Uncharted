import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def iso_err(X):
    n = X.shape[0]
    d = X.shape[1]
    S = X.T @ X
    tgt = (n / d) * np.eye(d)
    return float(np.linalg.norm(S - tgt, 'fro'))

def degree_stats(X, tol=0.005):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    deg = np.sum(G > m - tol, axis=1)
    var = float(np.var(deg))
    ratt = int(np.sum(deg == 0))
    return m, var, ratt, deg

def adam_lse(X0, betas, steps_list, lrs, deadline=None):
    X = X0.copy()
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    best = X.copy()
    bv = max_cos(X)
    for bi, beta in enumerate(betas):
        S = steps_list[bi]
        lr = lrs[bi]
        for s in range(S):
            if deadline is not None and (s % 50 == 0) and time.time() > deadline - 0.5:
                return best, bv
            t += 1
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            # gmax for stability
            gmax = float(np.max(G))
            # exp weights
            E = np.exp(beta * (G - gmax))
            np.fill_diagonal(E, 0.0)
            grad = E @ X
            dot = np.sum(grad * X, axis=1, keepdims=True)
            gtan = grad - dot * X
            m1 = b1 * m1 + (1.0 - b1) * gtan
            m2 = b2 * m2 + (1.0 - b2) * (gtan * gtan)
            c1 = 1.0 - b1 ** t
            c2 = 1.0 - b2 ** t
            m1h = m1 / c1
            m2h = m2 / c2
            upd = m1h / (np.sqrt(m2h) + eps)
            # tangential projection of update
            ud = np.sum(upd * X, axis=1, keepdims=True)
            upd = upd - ud * X
            X = X - lr * upd
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            if (s % 50 == 49) or (s == S - 1):
                v = max_cos(X)
                if v < bv:
                    bv = v
                    best = X.copy()
    return best, bv

def greedy_void_insert(X, k, rng, pool=8000):
    n, d = X.shape
    C = rng.randn(pool, d)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    # current max dot to X
    D = C @ X.T
    cur = np.max(D, axis=1)
    new_pts = []
    # avoid re-picking same candidate
    used = np.zeros(pool, dtype=bool)
    for _ in range(k):
        # mask used as +inf
        curm = cur.copy()
        curm[used] = np.inf
        bi = int(np.argmin(curm))
        used[bi] = True
        y = C[bi]
        new_pts.append(y.copy())
        # update cur with new point
        nd = C @ y
        cur = np.maximum(cur, nd)
    if len(new_pts) == 0:
        return X
    return np.vstack([X, np.array(new_pts)])

def lp_polish(X, seconds, deadline=None):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = max_cos(X)
    radius = 5e-4
    tstart = time.time()
    if deadline is None:
        deadline = tstart + seconds
    end = min(tstart + seconds, deadline)
    iu0 = np.triu_indices(n, 1)
    MAXP = 2500
    while time.time() < end and radius > 1e-14:
        Gfull = X @ X.T
        cc = Gfull[iu0]
        m = float(np.max(cc))
        sel = cc > m - max(20 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        csub = cc[sel]
        p = len(I)
        if p == 0:
            break
        if p > MAXP:
            # keep largest
            idx = np.argpartition(csub, -MAXP)[-MAXP:]
            I = I[idx]; J = J[idx]; csub = csub[idx]
            p = MAXP
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
        if time.time() >= end - 0.2:
            break
    return X

def solve():
    t0 = time.time()
    deadline = t0 + 175.0
    # incumbent fallback (never used as start, only for return)
    inc = None
    inc_val = np.inf
    for f in ("best.npy", "record.npy"):
        try:
            c = np.asarray(np.load(f), dtype=float)
            if c.shape == (180, 9):
                c = c / np.linalg.norm(c, axis=1, keepdims=True)
                if np.all(np.isfinite(c)):
                    v = max_cos(c)
                    if v < inc_val:
                        inc_val = v
                        inc = c.copy()
        except Exception:
            pass
    betas_all = [25.0, 70.0, 180.0, 450.0]
    lrs_all = [0.02, 0.015, 0.01, 0.006]
    steps_mid = [120, 120, 150, 150]
    steps_final = [250, 250, 300, 400]
    steps_init = [150, 150, 200, 200]
    sched = [120, 135, 150, 165, 180]
    cands = []
    NRESTART = 5
    for r in range(NRESTART):
        # reserve last ~25s for polishing
        if time.time() > deadline - 28.0:
            break
        rng = np.random.RandomState(1000 + r)
        n0 = sched[0]
        X = rng.randn(n0, 9)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        try:
            Xb, vb = adam_lse(X, betas_all, steps_init, lrs_all, deadline=deadline - 26.0)
        except Exception:
            continue
        X = Xb
        ok = True
        for si in range(1, len(sched)):
            if time.time() > deadline - 27.0:
                ok = False
                break
            k = sched[si] - sched[si-1]
            try:
                X = greedy_void_insert(X, k, rng, pool=7000)
            except Exception:
                ok = False
                break
            if sched[si] < 180:
                st = steps_mid
            else:
                st = steps_final
            try:
                Xb, vb = adam_lse(X, betas_all, st, lrs_all, deadline=deadline - 26.0)
            except Exception:
                ok = False
                break
            X = Xb
        if not ok:
            continue
        try:
            m, var, ratt, _ = degree_stats(X, tol=0.005)
            ie = iso_err(X)
        except Exception:
            continue
        cands.append({"X": X.copy(), "max": m, "var": var, "ratt": ratt, "iso": ie})
        if time.time() > deadline - 27.0:
            break
    # if nothing built, return incumbent
    if len(cands) == 0:
        if inc is not None:
            return inc / np.linalg.norm(inc, axis=1, keepdims=True)
        rng = np.random.RandomState(0)
        X = rng.randn(180, 9)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X
    # rank: filter near best max then balance
    best_max = min(c["max"] for c in cands)
    # balanced pool within 0.02 of best
    pool = [c for c in cands if c["max"] <= best_max + 0.02]
    if len(pool) == 0:
        pool = cands
    def bscore(c):
        return c["var"] + 4.0 * c["ratt"] + 0.1 * c["iso"]
    pool_sorted = sorted(pool, key=bscore)
    # short LP polishes for most balanced few (max 2)
    n_polish = min(2, len(pool_sorted))
    polished = []
    for i in range(n_polish):
        if time.time() > deadline - 22.0:
            break
        c = pool_sorted[i]
        secs = 4.0
        # don't eat final reserve
        if deadline - time.time() < 24.0:
            secs = max(1.0, (deadline - time.time() - 20.0))
        try:
            Y = lp_polish(c["X"], seconds=secs, deadline=deadline - 20.0)
            v = max_cos(Y)
            m2, var2, ratt2, _ = degree_stats(Y, tol=1e-7)
            ie2 = iso_err(Y)
            polished.append({"X": Y, "max": v, "var": var2, "ratt": ratt2, "iso": ie2})
        except Exception:
            pass
    # overall best candidate (by max) among cands+polished
    allc = cands + polished
    # include incumbent as option for final choice only
    allc_sorted = sorted(allc, key=lambda c: c["max"])
    best_cand = allc_sorted[0]
    Xbest = best_cand["X"].copy()
    Vbest = best_cand["max"]
    # if polished improved, use it; else use best max
    # reserve last 20-25s for polishing overall best
    rem = deadline - time.time()
    if rem > 3.0:
        # shortlist: overall best by max gets final polish
        try:
            secs = min(rem - 0.5, 20.0)
            Y = lp_polish(Xbest, seconds=secs, deadline=deadline - 0.3)
            v = max_cos(Y)
            if v < Vbest:
                Xbest = Y
                Vbest = v
        except Exception:
            pass
    # final return: incumbent if better (ensures no regression) else our best
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    if inc is not None and inc_val < Vbest:
        return inc / np.linalg.norm(inc, axis=1, keepdims=True)
    # safety: finite check
    if not np.all(np.isfinite(Xbest)):
        if inc is not None:
            return inc / np.linalg.norm(inc, axis=1, keepdims=True)
    return Xbest
