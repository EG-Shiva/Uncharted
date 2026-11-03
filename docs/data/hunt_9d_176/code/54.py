import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def max_abs_cos(Y):
    G = Y @ Y.T
    np.fill_diagonal(G, 0.0)
    return float(np.max(np.abs(G)))

def antipodal_adam(Y, steps=3000, a0=40.0, a1=250.0, lr=0.004):
    n, d = Y.shape
    m = np.zeros_like(Y)
    v = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    Y = Y.copy()
    for t in range(steps):
        frac = t / max(1, steps - 1)
        alpha = a0 * ((a1 / a0) ** frac)
        G = Y @ Y.T
        np.fill_diagonal(G, 0.0)
        M = float(np.max(np.abs(G)))
        # scaled exps
        Epos = np.exp(alpha * (G - M))
        Eneg = np.exp(alpha * (-G - M))
        np.fill_diagonal(Epos, 0.0)
        np.fill_diagonal(Eneg, 0.0)
        S = float((Epos + Eneg).sum() * 0.5)
        if S <= 0 or not np.isfinite(S):
            break
        W = Epos - Eneg
        Grad = (W @ Y) / S
        dot = np.sum(Grad * Y, axis=1, keepdims=True)
        Gt = Grad - dot * Y
        m = b1 * m + (1 - b1) * Gt
        v = b2 * v + (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1 ** (t + 1))
        vhat = v / (1 - b2 ** (t + 1))
        step = mhat / (np.sqrt(vhat) + eps)
        np.clip(step, -3.0, 3.0, out=step)
        Y = Y - lr * step
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        if not np.all(np.isfinite(Y)):
            break
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def full_adam(X, steps=1000, a0=40.0, a1=320.0, lr=0.003):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    X = X.copy()
    for t in range(steps):
        frac = t / max(1, steps - 1)
        alpha = a0 * ((a1 / a0) ** frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        E = np.exp(alpha * (G - M))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if S <= 0 or not np.isfinite(S):
            break
        Grad = (E @ X) / S
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        m = b1 * m + (1 - b1) * Gt
        v = b2 * v + (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1 ** (t + 1))
        vhat = v / (1 - b2 ** (t + 1))
        step = mhat / (np.sqrt(vhat) + eps)
        np.clip(step, -3.0, 3.0, out=step)
        lrt = lr * (1.0 - 0.4 * frac)
        X = X - lrt * step
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X /= nr
        if not np.all(np.isfinite(X)):
            break
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def pair_to_Y(X):
    n, d = X.shape
    h = n // 2
    G = X @ X.T
    unmatched = set(range(n))
    Y = []
    while unmatched:
        i = min(unmatched)
        unmatched.remove(i)
        if len(unmatched) == 0:
            Y.append(X[i] / np.linalg.norm(X[i]))
            break
        # find most antipodal (minimal dot)
        best_j = None
        best_v = 1e100
        for j in unmatched:
            v = G[i, j]
            if v < best_v:
                best_v = v
                best_j = j
        unmatched.remove(best_j)
        y = X[i] - X[best_j]
        nv = np.linalg.norm(y)
        if nv < 1e-12 or not np.isfinite(nv):
            y = X[i]
            nv = np.linalg.norm(y)
        Y.append(y / nv)
    Y = np.array(Y)
    if Y.shape[0] > h:
        Y = Y[:h]
    return Y

def lp_polish_deadline(x, deadline, r_init=0.004, cap=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = r_init
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        mval = float(c_all.max())
        thresh = mval - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            # keep largest `cap`
            part = np.argpartition(-c_all[idx], cap)[:cap]
            keep = np.zeros_like(sel)
            keep[idx[part]] = True
            sel = keep
            cnt = cap
        I = iu0[0][sel]; J = iu0[1][sel]
        c_sel = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty((p, 2 * d + 1), dtype=int)
        cols[:, :d] = I[:, None] * d + np.arange(d)
        cols[:, d:2 * d] = J[:, None] * d + np.arange(d)
        cols[:, 2 * d] = n * d
        cols = cols.ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(nr)) or np.any(nr < 1e-12):
            radius *= 0.5
            continue
        y = y / nr
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
        if time.time() >= deadline:
            break
    return x

def solve():
    t0 = time.time()
    HARD_END = t0 + 176.0
    rng = np.random.default_rng(0)
    n, d, h = 176, 9, 88
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(nr > 1e-12):
                    cands.append(a / nr)
        except Exception:
            pass
    if len(cands) == 0:
        a = rng.standard_normal((n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    base_vals = [max_cos(c) for c in cands]
    best_X = cands[int(np.argmin(base_vals))].copy()
    best_v = float(np.min(base_vals))

    # Build Y seeds
    Yseeds = []
    for X in cands:
        try:
            Yseeds.append(pair_to_Y(X))
        except Exception:
            pass
    # lattice seeds: integer cube projections
    for s in range(2):
        r = np.random.default_rng(100 + s)
        A = r.integers(-2, 3, size=(h * 4, d)).astype(float)
        # drop zero rows
        nr = np.linalg.norm(A, axis=1)
        A = A[nr > 1e-9]
        A = A / np.linalg.norm(A, axis=1, keepdims=True)
        # subsample / pad
        if A.shape[0] >= h:
            Yseeds.append(A[:h].copy())
        else:
            extra = r.standard_normal((h - A.shape[0], d))
            extra /= np.linalg.norm(extra, axis=1, keepdims=True)
            Yseeds.append(np.vstack([A, extra]))
    # random seeds
    for s in range(3):
        r = np.random.default_rng(200 + s)
        A = r.standard_normal((h, d))
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        Yseeds.append(A)

    # Antipodal Adam: ~20-30s budget but steps are cheap; limit steps
    best_Y = None
    best_a = 1e100
    # time cap for this phase: 35s
    anti_deadline = t0 + 35.0
    for Y0 in Yseeds:
        if time.time() > anti_deadline:
            break
        Yopt = antipodal_adam(Y0, steps=3000, a0=40.0, a1=250.0, lr=0.004)
        v = max_abs_cos(Yopt)
        if v < best_a:
            best_a = v
            best_Y = Yopt.copy()
        # second quick refine with higher alpha
        if time.time() > anti_deadline:
            break
        Yopt2 = antipodal_adam(Yopt, steps=1500, a0=100.0, a1=400.0, lr=0.002)
        v2 = max_abs_cos(Yopt2)
        if v2 < best_a:
            best_a = v2
            best_Y = Yopt2.copy()

    candidates_full = [(best_v, best_X.copy())]
    if best_Y is not None:
        Xsym = np.vstack([best_Y, -best_Y])
        Xsym /= np.linalg.norm(Xsym, axis=1, keepdims=True)
        vsym = max_cos(Xsym)
        candidates_full.append((vsym, Xsym.copy()))
    else:
        best_Y = pair_to_Y(best_X)

    # Release symmetry gradually + 1000-step quench
    amps = [0.0, 0.01, 0.03, 0.08]
    for ai, amp in enumerate(amps):
        if time.time() > HARD_END - 40.0:
            break
        r = np.random.default_rng(1000 + ai)
        Yp = best_Y.copy()
        X0 = np.vstack([Yp, -Yp])
        if amp > 0:
            N1 = r.standard_normal((h, d))
            N2 = r.standard_normal((h, d))
            # tangent projection
            N1 = N1 - np.sum(N1 * Yp, axis=1, keepdims=True) * Yp
            N2 = N2 - np.sum(N2 * (-Yp), axis=1, keepdims=True) * (-Yp)
            n1 = np.linalg.norm(N1, axis=1, keepdims=True); n1[n1 == 0] = 1.0
            n2 = np.linalg.norm(N2, axis=1, keepdims=True); n2[n2 == 0] = 1.0
            # scale to amplitude (tangent length)
            N1 = N1 / n1 * amp
            N2 = N2 / n2 * amp
            top = Yp + N1
            bot = -Yp + N2
            top /= np.linalg.norm(top, axis=1, keepdims=True)
            bot /= np.linalg.norm(bot, axis=1, keepdims=True)
            X0 = np.vstack([top, bot])
        Xq = full_adam(X0, steps=1000, a0=40.0, a1=320.0, lr=0.003)
        vq = max_cos(Xq)
        candidates_full.append((vq, Xq))

    candidates_full.sort(key=lambda t: t[0])
    cur_best_v, cur_best_X = candidates_full[0]

    # Single deadline-aware LP polish on the best quenched
    lp_deadline = HARD_END - 2.0
    # reserve at least a few seconds if possible, cap polish to ~25s
    remaining = lp_deadline - time.time()
    if remaining > 4.0:
        polish_time = min(remaining, 25.0)
        # polish deadline
        pd = time.time() + polish_time
        # ensure we don't exceed HARD_END
        pd = min(pd, lp_deadline)
        Xpol = lp_polish_deadline(cur_best_X, pd, r_init=0.004, cap=2500)
        vpol = max_cos(Xpol)
        if vpol < cur_best_v:
            cur_best_v, cur_best_X = vpol, Xpol

    # Final selection against original best
    if best_v < cur_best_v:
        return best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    # tiny shrink to enforce strict feasibility (no-op for cosine but safe)
    cur_best_X = cur_best_X / np.linalg.norm(cur_best_X, axis=1, keepdims=True)
    return cur_best_X
