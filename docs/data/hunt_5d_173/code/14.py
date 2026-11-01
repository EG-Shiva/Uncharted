import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contacts(X, tol=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    C = (G >= m - tol)
    cnt = C.sum(axis=1)
    return m, cnt, G

def _smooth_polish(X, steps, b0, b1, lr, verbose=False):
    X = X.copy()
    n, d = X.shape
    for s in range(steps):
        f = s / max(1, steps - 1)
        # exponential anneal of beta
        beta = b0 * (b1 / b0) ** f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gm = float(G.max())
        # weights, peak 1
        W = np.exp(beta * (G - gm))
        np.fill_diagonal(W, 0.0)
        # gradient of sum W : W @ X
        Gr = W @ X
        # tangent projection
        dot = (Gr * X).sum(axis=1, keepdims=True)
        Gt = Gr - dot * X
        # normalize step per point to keep stable
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        # scale: fixed lr, but reduce if huge
        step = lr / (1.0 + 0.5 * f)
        X = X - step * (Gt / np.maximum(1.0, gn * 0.5))
        X = _norm(X)
    return X

def _find_void(Xr, rng, ns=15000):
    C = rng.normal(size=(ns, 5))
    C = _norm(C)
    D = C @ Xr.T
    m = D.max(axis=1)
    return C[int(np.argmin(m))]

def _lp_polish(X, seconds, cap=3500):
    X = _norm(X.copy())
    n, d = X.shape
    best = _max_cos(X)
    if seconds <= 0.05:
        return X, best
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if radius < 1e-14:
            break
        G = X @ X.T
        cfull = G[iu0]
        m = float(cfull.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = cfull > thresh
        if int(sel.sum()) > cap:
            idx = np.where(sel)[0]
            order = np.argsort(cfull[idx])[-cap:]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I = iu0[0][sel]; J = iu0[1][sel]
        c = cfull[sel]
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
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
        else:
            radius *= 0.5
    return X, best

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
    # short initial LP to ensure tight local opt (2s)
    if deadline - time.time() > 15:
        best, best_val = _lp_polish(best, 2.0)

    trial = 0
    stag = 0
    while True:
        rem = deadline - time.time()
        if rem < 9.0:
            break
        trial += 1
        _, cnt, Gfull = _contacts(best, tol=1e-7)
        order_desc = np.argsort(-cnt)
        order_asc = np.argsort(cnt)
        mode = trial % 7
        y = best.copy()
        if mode == 0:
            # relocate worst point to void of rest
            victim = int(order_desc[0])
            mask = np.ones(n, dtype=bool); mask[victim] = False
            vdir = _find_void(y[mask], rng, ns=20000)
            y[victim] = vdir + 0.005 * rng.normal(size=(d,))
            # shake neighbours of victim a bit
            sims = Gfull[victim].copy()
            sims[victim] = -2
            nbrs = np.argsort(-sims)[:4]
            y[nbrs] += 0.02 * rng.normal(size=(4, d))
        elif mode == 1:
            # relocate rattler / low-contact to void
            victim = int(order_asc[rng.integers(0, min(8, n))])
            mask = np.ones(n, dtype=bool); mask[victim] = False
            vdir = _find_void(y[mask], rng, ns=20000)
            y[victim] = vdir
            y[victim] += 0.002 * rng.normal(size=(d,))
        elif mode == 2:
            # kick jammed cluster medium
            k = int(4 + (trial % 3))
            victims = order_desc[:k].copy()
            # add few random
            extra = rng.choice(n, 3, replace=False)
            victims = np.unique(np.concatenate([victims, extra]))
            sigma = 0.06 + 0.02 * (stag % 3)
            y[victims] += sigma * rng.normal(size=(len(victims), d))
        elif mode == 3:
            # global small shake
            sigma = 0.012 if stag < 15 else 0.03
            y += sigma * rng.normal(size=(n, d))
        elif mode == 4:
            # two-point void swap: remove 2 worst, place at two voids
            v1 = int(order_desc[0]); v2 = int(order_desc[1])
            mask = np.ones(n, dtype=bool); mask[v1] = False; mask[v2] = False
            Xr = y[mask]
            vdir1 = _find_void(Xr, rng, ns=15000)
            # second void far from first too
            C = _norm(rng.normal(size=(15000, d)))
            D = C @ Xr.T
            D2 = C @ vdir1
            m = np.maximum(D.max(axis=1), D2)
            vdir2 = C[int(np.argmin(m))]
            y[v1] = vdir1; y[v2] = vdir2
        elif mode == 5:
            # random subset strong kick (basin jump)
            k = 10
            victims = rng.choice(n, k, replace=False)
            y[victims] += 0.12 * rng.normal(size=(k, d))
        else:
            # single random reinit + neighbour shake
            victim = int(rng.integers(0, n))
            y[victim] = rng.normal(size=(d,))
            # shake 3 random others slightly
            extra = rng.choice(n, 3, replace=False)
            y[extra] += 0.03 * rng.normal(size=(3, d))
        y = _norm(y)
        # fast smooth anneal
        # vary schedule by mode
        if mode in (3,):
            steps, b0, b1, lr = 400, 25.0, 250.0, 0.02
        elif mode in (5,):
            steps, b0, b1, lr = 700, 20.0, 350.0, 0.025
        else:
            steps, b0, b1, lr = 550, 30.0, 300.0, 0.022
        y = _smooth_polish(y, steps, b0, b1, lr)
        v_smooth = _max_cos(y)
        # only LP-polish if promising (saves time)
        thresh_lp = best_val + (0.004 if stag < 20 else 0.008)
        if v_smooth > thresh_lp:
            # not promising, but occasionally still polish weak ones far from best?
            # skip LP
            if v_smooth < best_val:
                # unexpected: polish it
                pt = min(1.5, max(0.5, rem - 8.0))
                y2, v2 = _lp_polish(y, pt)
                if v2 < best_val:
                    best, best_val = y2, v2
                    stag = 0
                    continue
            stag += 1
            continue
        pt = 1.6
        if rem < 25:
            pt = max(0.7, rem - 8.0)
        pt = min(pt, rem - 8.0)
        if pt < 0.4:
            break
        y2, v2 = _lp_polish(y, pt)
        if v2 < best_val - 1e-12:
            best, best_val = y2, v2
            stag = 0
        else:
            stag += 1
            # accept equal to diversify contact topology rarely
            if v2 <= best_val + 1e-12 and (trial % 11 == 0):
                # move to equal basin to explore (keep best_val unchanged)
                best = y2.copy()
    # final short polish of best
    try:
        rem = deadline - time.time()
        if rem > 3.0:
            best, best_val = _lp_polish(best, min(rem - 1.0, 6.0))
    except Exception:
        pass
    best = _norm(best)
    return best
