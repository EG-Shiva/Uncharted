import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def _fast_polish(X, steps, lr0=0.06, T0=0.015, T1=0.0015):
    X = _norm(X.copy())
    n, d = X.shape
    # log schedule for T
    lt0, lt1 = np.log(T0), np.log(T1)
    for s in range(steps):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        frac = s / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        # weights
        # (G-m)<=0 so safe
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        dot = np.einsum('ij,ij->i', Grad, X)[:, None]
        Gt = Grad - dot * X
        rs = W.sum(axis=1, keepdims=True) + 1.0
        lr = lr0 * (1.0 - 0.7 * frac)
        X = X - (lr / rs) * Gt
        X = _norm(X)
    return X


def _lp_polish(X, seconds, radius0=2e-3):
    X = _norm(X.copy())
    n, d = X.shape
    best = _max_cos(X)
    radius = radius0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = X @ X.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        if int(sel.sum()) > 3000:
            idx = np.where(sel)[0]
            # keep tightest 3000
            part = np.argpartition(c_all[idx], len(idx) - 3000)[-3000:]
            mask = np.zeros_like(sel)
            mask[idx[part]] = True
            sel = mask
        I = iu0[0][sel]
        J = iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        cols = np.concatenate([(I[:, None] * d + np.arange(d)),
                               (J[:, None] * d + np.arange(d)),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
            # early stop if clearly stuck? continue
        else:
            radius *= 0.5
    return X


def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline = t0 + BUDGET
    rng = np.random.default_rng(12345)
    n, d = 173, 5

    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (n, d):
                an = _norm(a)
                v = _max_cos(an)
                if v < best_val:
                    best_val = v
                    best = an.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.normal(size=(n, d)))
        best_val = _max_cos(best)

    # contact info helper
    def contact_counts(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        mask = G >= m - 1e-9
        cnt = mask.sum(axis=1)
        return m, cnt, G

    # pool of candidates for final LP
    pool = [(best_val, best.copy())]

    trial = 0
    # exploration phase until deadline-28s
    while time.time() < deadline - 28.0:
        trial += 1
        rem = deadline - 28.0 - time.time()
        if rem <= 0:
            break
        y = best.copy()
        mode = trial % 8
        if mode == 0:
            y = y + 0.010 * rng.normal(size=(n, d))
        elif mode == 1:
            y = y + 0.030 * rng.normal(size=(n, d))
        elif mode == 2:
            k = 20
            idx = rng.choice(n, k, replace=False)
            y[idx] = y[idx] + 0.12 * rng.normal(size=(k, d))
        elif mode == 3:
            # kick jammed hubs
            _, cnt, _ = contact_counts(best)
            order = np.argsort(-cnt)
            k = 5
            victims = order[:k]
            y[victims] = y[victims] + 0.20 * rng.normal(size=(k, d))
            # also nudge random 3
            r = rng.choice(n, 3, replace=False)
            y[r] = y[r] + 0.05 * rng.normal(size=(3, d))
        elif mode == 4:
            # farthest re-insertion of worst / rattler
            _, cnt, G = contact_counts(best)
            # prefer rattler if exists else worst
            ratt = np.where(cnt == 0)[0]
            if len(ratt) > 0 and (trial % 2 == 0):
                victim = int(ratt[rng.integers(len(ratt))])
            else:
                # random among top-10 degree
                order = np.argsort(-cnt)
                victim = int(order[rng.integers(min(10, n))])
            others = np.delete(np.arange(n), victim)
            Xo = y[others]
            # sample candidates, pick minimizing max dot
            Nc = 2000
            C = _norm(rng.normal(size=(Nc, d)))
            dots = C @ Xo.T
            md = dots.max(axis=1)
            bi = int(np.argmin(md))
            y[victim] = C[bi]
            # small shake of neighbours
            y[others] = y[others] + 0.005 * rng.normal(size=((n - 1), d))
        elif mode == 5:
            # shake low-contact points
            _, cnt, _ = contact_counts(best)
            order = np.argsort(cnt)  # ascending
            k = 8
            victims = order[:k]
            y[victims] = y[victims] + 0.10 * rng.normal(size=(k, d))
        elif mode == 6:
            y = y + 0.004 * rng.normal(size=(n, d))
        else:  # mode 7 large sparse kick
            k = 3
            idx = rng.choice(n, k, replace=False)
            y[idx] = rng.normal(size=(k, d))  # full reinit
            y = y + 0.01 * rng.normal(size=(n, d))
        y = _norm(y)
        # fast smooth polish, steps depend on mode
        if mode in (4, 7):
            steps = 500
        elif mode in (0, 6):
            steps = 250
        else:
            steps = 350
        # vary temperature slightly by trial
        T0 = 0.015 if (trial % 3) else 0.03
        yf = _fast_polish(y, steps=steps, lr0=0.06, T0=T0, T1=0.0015)
        v = _max_cos(yf)
        if v < best_val:
            best_val = v
            best = yf.copy()
            pool.append((v, yf.copy()))
            if len(pool) > 5:
                pool.sort(key=lambda z: z[0])
                pool = pool[:5]
            # quick LP squeeze if we improved and have time
            if time.time() < deadline - 60:
                yl = _lp_polish(yf, seconds=1.2)
                vl = _max_cos(yl)
                if vl < best_val:
                    best_val = vl
                    best = yl.copy()
                    pool.append((vl, yl.copy()))
        elif v < best_val + 0.004 and (trial % 12 == 0) and time.time() < deadline - 60:
            # opportunistic LP on near-best basin
            yl = _lp_polish(yf, seconds=0.9)
            vl = _max_cos(yl)
            if vl < best_val:
                best_val = vl
                best = yl.copy()
                pool.append((vl, yl.copy()))

    # final stage: LP polish pool sorted
    pool.sort(key=lambda z: z[0])
    # keep unique best few
    time_left = deadline - time.time()
    # allocate most time to best
    if time_left < 5:
        return _norm(best)
    # polish top-3 with increasing budget
    budgets = []
    if len(pool) >= 3 and time_left > 20:
        # e.g. 20% , 25%, rest
        budgets = [time_left * 0.2, time_left * 0.25, time_left - 6.0 - time_left * 0.45]
    elif len(pool) >= 2 and time_left > 12:
        budgets = [time_left * 0.3, time_left - 6.0 - time_left * 0.3]
    else:
        budgets = [time_left - 5.0]
    for k, bd in enumerate(budgets):
        if k >= len(pool):
            break
        if bd < 1.0:
            continue
        cand = pool[k][1]
        polished = _lp_polish(cand, seconds=float(bd))
        v = _max_cos(polished)
        if v < best_val:
            best_val = v
            best = polished.copy()
    # final safety polish of best with remaining time (at least a bit)
    rem = deadline - time.time() - 2.0
    if rem > 2.0:
        # short final squeeze with small radius for tight convergence
        best = _lp_polish(best, seconds=float(min(rem, 12.0)), radius0=1e-3)
    best = _norm(best)
    # guard NaN
    if not np.all(np.isfinite(best)):
        best = _norm(np.load("best.npy").astype(np.float64))
    return best
