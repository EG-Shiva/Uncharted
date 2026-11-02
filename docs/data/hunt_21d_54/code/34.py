import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _proj(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    x = x / n
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    n2 = np.linalg.norm(x, axis=1, keepdims=True)
    n2[n2 == 0] = 1.0
    return x / n2

def _homotopy_one(X0, t_levels, steps_per_level, lr0, lse_T, lse_lam, deadline, rng):
    X = X0.copy()
    n, d = X.shape
    m1 = np.zeros((n, d))
    v1 = np.zeros((n, d))
    b1, b2 = 0.9, 0.999
    eps = 1e-8
    step_global = 0
    best_X = X.copy()
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best_val = float(G0.max())
    for li, t in enumerate(t_levels):
        # reset second moment partially for new level to allow jumps
        # keep first moment for continuity
        for s in range(steps_per_level):
            if (s & 127) == 0 and time.time() > deadline:
                return best_X, best_val
            G = X @ X.T
            # need max for LSE and tracking (diag excluded)
            # use copy with diag -2
            np.fill_diagonal(G, -2.0)
            cur_max = float(G.max())
            # track best by true max
            if cur_max < best_val:
                best_val = cur_max
                best_X = X.copy()
                if best_val < 0.0666634:
                    # early success signal; continue a bit to improve further
                    pass
            # penalty matrices: need G with diag 0 for excess
            # excess = relu(G - t), diag already -2 so zero after relu
            # compute H
            # G currently has -2 diag, so relu gives 0 there
            H = G - t
            np.maximum(H, 0.0, out=H)
            # LSE weights
            # stable softmax
            E = (G - cur_max) / lse_T
            np.clip(E, -20.0, 0.0, out=E)
            W = np.exp(E)
            # zero diag (diag E = (-2-cur_max)/T clipped -> small but set 0)
            # set diag 0
            # faster: flat indexing
            W[np.diag_indices(n)] = 0.0
            ssum = float(W.sum() * 0.5)  # sum over i<j
            if ssum < 1e-300:
                ssum = 1.0
            # gradients
            grad_pen = 2.0 * (H @ X)
            grad_lse = (W @ X) * (lse_lam / (2.0 * ssum + 1e-300))
            grad = grad_pen + grad_lse
            # tangent projection
            dot = np.einsum('ij,ij->i', grad, X)
            grad -= dot[:, None] * X
            # Adam update
            step_global += 1
            m1 = b1 * m1 + (1.0 - b1) * grad
            v1 = b2 * v1 + (1.0 - b2) * (grad * grad)
            bc1 = 1.0 - b1 ** step_global
            bc2 = 1.0 - b2 ** step_global
            mh = m1 / bc1
            vh = v1 / bc2
            # lr schedule: decay within level
            frac = s / steps_per_level
            lr = lr0 * (1.0 - 0.5 * frac)
            X = X - lr * mh / (np.sqrt(vh) + eps)
            # retract
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            X /= nr
        # end level: check feasibility to anneal
        Gc = X @ X.T
        np.fill_diagonal(Gc, -2.0)
        cm = float(Gc.max())
        if cm < best_val:
            best_val = cm
            best_X = X.copy()
        # only go to lower t if penalty near zero: max < t + tol
        if cm > t + 2e-4:
            # failed to meet this target, stop annealing this start
            break
        # small kick between levels to help topology change
        if li + 1 < len(t_levels) and time.time() < deadline:
            # tiny tangent noise
            nz = rng.normal(size=(n, d)) * 1e-4
            nz -= (nz * X).sum(axis=1, keepdims=True) * X
            X = _proj(X + nz)
    return best_X, best_val

def _lp_polish(x, deadline):
    x = _proj(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 2500:
            csv_full = c[sel]
            idx = np.argpartition(-csv_full, 2500)[:2500]
            I, J = I[idx], J[idx]
            csv = csv_full[idx]
            p = 2500
        else:
            csv = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        tl = deadline - time.time() - 0.3
        if tl < 0.5:
            break
        try:
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                              method="highs", options={"time_limit": min(4.0, tl)})
            except Exception:
                res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                              method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y = _proj(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    T0 = time.time()
    LIMIT = 168.0
    deadline_search = T0 + 120.0
    deadline_final = T0 + 166.0
    rng = np.random.default_rng(770031)
    try:
        b0 = np.load("best.npy")
    except Exception:
        b0 = np.load("record.npy")
    b0 = _proj(np.asarray(b0, dtype=float))
    if b0.shape != (54, 21):
        b0 = _proj(rng.normal(size=(54, 21)))
    try:
        r0 = _proj(np.asarray(np.load("record.npy"), dtype=float))
        if r0.shape == (54, 21):
            if _max_cos(r0) < _max_cos(b0):
                b0 = r0
    except Exception:
        pass
    n, d = 54, 21
    base = b0.copy()
    base_val = _max_cos(base)
    best = base.copy()
    best_val = base_val
    t_levels = [0.066, 0.064, 0.062, 0.060]
    # build start pool: perturbed copies + random
    sigmas = [0.01, 0.02, 0.04, 0.07, 0.12, 0.20]
    lrs = [0.02, 0.03, 0.045]
    # prepare perturbed starts on the fly to save time
    start_idx = 0
    # also include unperturbed base as first start with small lr
    def make_start(k):
        if k == 0:
            return base.copy()
        elif k <= 12:
            sig = sigmas[(k - 1) % len(sigmas)]
            nz = rng.normal(size=(n, d)) * sig
            nz -= (nz * base).sum(axis=1, keepdims=True) * base
            return _proj(base + nz)
        else:
            return _proj(rng.normal(size=(n, d)))
    n_starts = 16
    steps_per_level = 2200
    # adjust steps if time short
    for k in range(n_starts):
        if time.time() > deadline_search:
            break
        if best_val < 0.0666634:
            break
        X0 = make_start(k)
        lr0 = lrs[k % len(lrs)]
        # use slightly fewer steps for large k to fit budget
        spl = steps_per_level
        if k >= 8:
            spl = 1600
        # per-start deadline
        per_deadline = min(deadline_search, time.time() + 14.0)
        # vary LSE params slightly for diversity
        lse_T = 0.004 + 0.001 * ((k % 3))
        lse_lam = 5e-4
        Xopt, v = _homotopy_one(X0, t_levels, spl, lr0, lse_T, lse_lam, per_deadline, rng)
        if v < best_val:
            best_val = v
            best = Xopt.copy()
        start_idx += 1
    # if success, spend remaining budget on LP polish
    if best_val < 0.0666634:
        # polish the best; give almost all remaining time
        if time.time() < deadline_final - 1.0:
            best = _lp_polish(best, deadline_final)
            best_val = _max_cos(best)
    else:
        # no success: still try a very short LP only if it cannot timeout?
        # per instructions return best feasible iterate found (no LP)
        # do at most 2s safeguard polish only if we have lots of time and it improves tiny margin?
        # skip heavy LP to avoid timeout; do nothing
        pass
    best = _proj(best)
    return best
