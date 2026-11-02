import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _proj(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _lp_polish(X, deadline):
    X = _proj(X)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = X @ X.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 2500:
            idx = np.argpartition(-c[sel], 2500)[:2500]
            I, J = I[idx], J[idx]
            csv = c[sel][idx]
            p = 2500
        else:
            csv = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        Y = X + res.x[:-1].reshape(n, d)
        Y = _proj(Y)
        v = _max_cos(Y)
        if v < best:
            X, best = Y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X, best

def _adam_target(X0, t, steps, lr0, deadline, eps_tie=1e-3, T_lse=0.003):
    X = X0.copy()
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, ea = 0.9, 0.999, 1e-8
    cnt = 0
    best_X = X.copy()
    best_v = _max_cos(X)
    # bias correction powers
    for s in range(steps):
        if cnt % 200 == 0 and time.time() > deadline:
            break
        cnt += 1
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        # track best max
        cur_max = float(G.max())
        if cur_max < best_v:
            best_v = cur_max
            best_X = X.copy()
        # relu part
        R = G - t
        R[R < 0] = 0.0
        np.fill_diagonal(R, 0.0)
        A = 2.0 * R
        # tie breaker weights
        # log-sum-exp weights centered at cur_max
        E = (G - cur_max) / T_lse
        np.clip(E, -20, 0, out=E)
        np.fill_diagonal(E, -20)  # zero out diag after exp ~2e-9
        W = np.exp(E)
        # remove diag contribution
        np.fill_diagonal(W, 0.0)
        # sum over triu for normalization
        S = 0.0
        # fast triu sum
        S = np.triu(W, 1).sum() + 1e-30
        Ws = W / S
        grad = (A + eps_tie * Ws) @ X
        # tangent projection
        dot = np.einsum('ij,ij->i', grad, X)
        grad -= dot[:, None] * X
        m1 = b1 * m1 + (1 - b1) * grad
        m2 = b2 * m2 + (1 - b2) * (grad * grad)
        c1 = 1 - b1 ** cnt
        c2 = 1 - b2 ** cnt
        mh = m1 / c1
        vh = m2 / c2
        frac = s / max(1, steps)
        lr = lr0 * (1.0 - 0.6 * frac)
        X = X - lr * mh / (np.sqrt(vh) + ea)
        X = _proj(X)
    # final check
    v = _max_cos(X)
    if v < best_v:
        best_v = v
        best_X = X.copy()
    return best_X, best_v, X

def solve():
    T0 = time.time()
    HARD = T0 + 155.0  # hard internal deadline
    ADAM_DEADLINE = T0 + 110.0
    try:
        b0 = np.load("best.npy").astype(float)
    except Exception:
        b0 = np.load("record.npy").astype(float)
    b0 = _proj(np.nan_to_num(b0))
    if b0.shape != (54, 21):
        rng0 = np.random.default_rng(0)
        b0 = _proj(rng0.normal(size=(54, 21)))
    try:
        r0 = _proj(np.load("record.npy").astype(float))
    except Exception:
        r0 = b0.copy()
    bv0 = _max_cos(b0)
    rv0 = _max_cos(r0)
    if rv0 < bv0:
        glob_X, glob_v = r0.copy(), rv0
    else:
        glob_X, glob_v = b0.copy(), bv0
    init_v = glob_v
    thresh = 0.0666634
    # build starts: perturbed best + random, deterministic
    rng = np.random.default_rng(20261101)
    starts = []
    starts.append(b0.copy())
    starts.append(r0.copy())
    for sig in (0.01, 0.025, 0.06, 0.12):
        N = rng.normal(size=(54, 21))
        N -= (N * b0).sum(axis=1, keepdims=True) * b0
        Y = b0 + sig * N
        starts.append(_proj(Y))
    for k in range(2):
        starts.append(_proj(rng.normal(size=(54, 21))))
    targets = [0.066, 0.064, 0.062, 0.060]
    # diverse lrs
    lrs = [0.02, 0.02, 0.015, 0.015, 0.02, 0.02, 0.025, 0.025]
    steps_list = [2800, 2800, 2200, 2200, 2200, 1800, 1800, 1800]
    for si, X0 in enumerate(starts):
        if time.time() > ADAM_DEADLINE:
            break
        lr0 = lrs[si % len(lrs)]
        steps = steps_list[si % len(steps_list)]
        Xc = X0.copy()
        for ti, t in enumerate(targets):
            if time.time() > ADAM_DEADLINE:
                break
            best_run, best_v_run, Xend = _adam_target(Xc, t, steps, lr0, ADAM_DEADLINE)
            if best_v_run < glob_v:
                glob_v = best_v_run
                glob_X = best_run.copy()
            # feasibility: penalty near zero <=> max <= t + tol
            if best_v_run <= t + 2e-6 or _max_cos(Xend) <= t + 5e-4:
                Xc = Xend.copy()
                # shrink lr for next harder target
                lr0 *= 0.8
                continue
            else:
                # not feasible: keep best_run as carry for one more try? stop this start
                # but still allow Xc=best_run for next? spec says anneal only after near zero
                break
        # early exit if we already beat record by margin and time is short
        if glob_v < thresh - 1e-6 and time.time() > ADAM_DEADLINE - 10:
            break
    # decide polish budget
    now = time.time()
    if glob_v < thresh:
        # spend remaining on LP
        glob_X, glob_v = _lp_polish(glob_X, HARD - 3.0)
    else:
        # still try short squeeze if time allows and we improved or not
        if now < HARD - 15.0:
            # short 12s squeeze on glob_X
            glob_X, glob_v = _lp_polish(glob_X, min(now + 12.0, HARD - 3.0))
    # safety: if polish worsened or nan, fallback to initial best
    if not np.isfinite(glob_X).all() or glob_v > init_v + 1e-12:
        # return the better of init
        glob_X = b0.copy()
        if rv0 < bv0:
            glob_X = r0.copy()
    glob_X = _proj(np.nan_to_num(glob_X))
    # tiny shrink guard (no-op, keep unit)
    return glob_X
