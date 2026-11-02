import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _adam_target(X0, t, steps, lr0, lmbda=8e-4, Tlse=0.004, decay_to=0.2):
    n, d = X0.shape
    X = X0.copy()
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    Eye = np.eye(n)
    for k in range(steps):
        frac = k / max(1, steps - 1)
        lr = lr0 * (1.0 - (1.0 - decay_to) * frac)
        G = X @ X.T
        # penalty relu part (ignore diag)
        R = G - t
        # kill diagonal (G_ii=1): set to 0
        np.fill_diagonal(R, 0.0)
        R[R < 0] = 0.0
        grad_p = 2.0 * (R @ X)
        # tiny log-sum-exp tie breaker over off-diag pairs
        # off-diag max for shift
        m_off = float((G - 2.0 * Eye).max())
        E = (G - m_off) / max(Tlse, 1e-12)
        np.clip(E, -15.0, 0.0, out=E)
        np.fill_diagonal(E, -15.0)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        s = float(W.sum())
        if s > 0 and np.isfinite(s):
            Wn = W / s
            grad_l = Wn @ X
            # symmetrize (W symmetric already) -> scale to match pair counting
            grad_l = grad_l * 2.0
        else:
            grad_l = 0.0
        grad = grad_p + lmbda * grad_l
        # tangent projection
        dot = np.einsum('ij,ij->i', grad, X)[:, None]
        grad -= dot * X
        m = b1 * m + (1.0 - b1) * grad
        v = b2 * v + (1.0 - b2) * (grad * grad)
        mh = m / (1.0 - b1 ** (k + 2))
        vh = v / (1.0 - b2 ** (k + 2))
        step = lr * mh / (np.sqrt(vh) + eps)
        # keep tangent (remove normal of step)
        dot2 = np.einsum('ij,ij->i', step, X)[:, None]
        step -= dot2 * X
        X = X - step
        # renormalize
        nn = np.linalg.norm(X, axis=1, keepdims=True)
        nn[nn == 0] = 1.0
        X /= nn
        # guard
        if not np.all(np.isfinite(X)):
            return _norm(X0)
    return X

def _lp_polish(x, seconds, r0=1.2e-3, cap=700):
    x = _norm(np.asarray(x, dtype=float))
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        remain = seconds - (time.time() - start)
        if remain < 0.4:
            break
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.argsort(-c)[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        csv = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        Ccol = np.empty((p, 2 * d + 1), dtype=int)
        Ccol[:, :d] = I[:, None] * d + np.arange(d)
        Ccol[:, d:2 * d] = J[:, None] * d + np.arange(d)
        Ccol[:, 2 * d] = n * d
        cols = Ccol.ravel()
        Arow = np.empty((p, 2 * d + 1))
        Arow[:, :d] = x[J]
        Arow[:, d:2 * d] = x[I]
        Arow[:, 2 * d] = -1.0
        vals = Arow.ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": min(1.5, max(0.3, remain - 0.15))})
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
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    LIMIT = 166.0
    def tl():
        return LIMIT - (time.time() - t0)
    rng = np.random.RandomState(0)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.randn(54, 21)
    xb = _norm(np.asarray(xb, dtype=float))
    if xb.shape != (54, 21) or not np.all(np.isfinite(xb)):
        xb = _norm(rng.randn(54, 21))
    try:
        xr = _norm(np.asarray(np.load("record.npy"), dtype=float))
        if xr.shape == (54, 21) and np.all(np.isfinite(xr)):
            if _max_cos(xr) < _max_cos(xb):
                xb = xr
    except Exception:
        pass
    best = xb.copy()
    best_val = _max_cos(best)
    init_best_val = float(best_val)
    thresh = 0.0666634
    targets = [0.066, 0.064, 0.062, 0.060]
    sigmas = [0.03, 0.07, 0.015, 0.12, 0.05, 0.09, 0.02, 0.20]
    lrs = [0.015, 0.010, 0.020, 0.008]
    restart = 0
    # homotopy search, reserve time for LP
    while tl() > 32.0:
        restart += 1
        if restart > 300:
            break
        # init: perturbed best or random
        if restart % 6 == 0:
            X0 = _norm(rng.randn(54, 21))
        else:
            sigma = sigmas[(restart - 1) % len(sigmas)]
            # occasionally perturb random subset strongly to change topology
            X0 = best + sigma * rng.randn(54, 21)
            if restart % 4 == 0:
                idx = rng.choice(54, size=6, replace=False)
                X0[idx] = X0[idx] + 0.5 * rng.randn(6, 21)
            X0 = _norm(X0)
        lr0 = lrs[(restart - 1) % len(lrs)]
        X = X0
        for ti, tt in enumerate(targets):
            if tl() < 32.0:
                break
            # fewer steps for lower targets (only reached if feasible)
            if ti == 0:
                steps = 1600
            elif ti == 1:
                steps = 1200
            else:
                steps = 900
            # shrink steps if time low
            if tl() < 60.0:
                steps = 700
            X = _adam_target(X, tt, steps=steps, lr0=lr0, lmbda=8e-4, Tlse=0.004)
            v = _max_cos(X)
            if v < best_val - 1e-12:
                best_val = float(v)
                best = X.copy()
                # early exit to polish if we beat threshold
                if best_val < thresh - 2e-9:
                    pass
            # annealing only if penalty near zero: max <= t + tol
            if v <= tt + 5e-05:
                continue
            else:
                break
        # quick check deadline
        if tl() < 32.0:
            break
    # spend remaining on LP polish if we beat threshold, else short polish
    rem = tl()
    if best_val < thresh:
        budget = min(rem - 1.0, 24.0)
        if budget > 2.0:
            best = _lp_polish(best, seconds=budget, r0=1.2e-3, cap=700)
            best_val = _max_cos(best)
    else:
        # still try short polish to try to cross threshold (bounded)
        budget = min(rem - 1.0, 10.0)
        if budget > 3.0:
            best = _lp_polish(best, seconds=budget, r0=1.0e-3, cap=600)
            best_val = _max_cos(best)
    best = _norm(best)
    return best
