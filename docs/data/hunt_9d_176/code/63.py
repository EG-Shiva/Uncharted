import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def smooth_quench(X, steps, beta0, beta1, lr0, lr1, mu=0.85):
    n, d = X.shape
    V = np.zeros_like(X)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = beta0 * (beta1 / beta0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        try:
            m = float(G.max())
        except Exception:
            break
        # weights focused on top, low beta spreads uniformly
        W = np.exp((G - m) * beta)
        np.fill_diagonal(W, 0.0)
        # kill non-finite
        if not np.all(np.isfinite(W)):
            break
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        nrm = float(np.linalg.norm(Gt) / np.sqrt(n))
        if not np.isfinite(nrm) or nrm < 1e-16:
            # keep momentum decay
            V *= mu
            continue
        ghat = Gt / nrm
        V = mu * V + (1.0 - mu) * ghat
        # bias correction early
        # effective step lr
        X = X - lr * V
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[~np.isfinite(nr)] = 1.0
        nr[nr < 1e-12] = 1.0
        X = X / nr
    return X

def void_reinsert(X, idxs, rng, M=4000):
    n, d = X.shape
    for idx in np.atleast_1d(idxs):
        idx = int(idx)
        R = rng.standard_normal((M, d))
        nr = np.linalg.norm(R, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        R = R / nr
        dots = R @ X.T
        dots[:, idx] = -2.0
        worst = dots.max(axis=1)
        pick = int(np.argmin(worst))
        X[idx] = R[pick]
    return X

def lp_polish_deadline(x, deadline, r_init=0.002, cap=2500):
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
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > cap:
            idx_all = np.where(sel)[0]
            # keep largest cap
            part = np.argpartition(-c_all[idx_all], cap)[:cap]
            keep = np.zeros_like(sel)
            keep[idx_all[part]] = True
            sel = keep
        I = iu0[0][sel]; J = iu0[1][sel]
        c_sel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols_I = (I[:, None] * d + np.arange(d))
        cols_J = (J[:, None] * d + np.arange(d))
        cols = np.empty((p, 2 * d + 1), dtype=int)
        cols[:, :d] = cols_I
        cols[:, d:2*d] = cols_J
        cols[:, 2*d] = n * d
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
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y = y / yn
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    HARD_END = t0 + 176.0
    SCREEN_END = t0 + 128.0
    rng = np.random.default_rng(0)
    n, d = 176, 9
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(nr > 1e-12):
                    cands.append(a / nr)
        except Exception:
            pass
    if len(cands) == 0:
        X0 = rng.standard_normal((n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands.append(X0)
    vals = [max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_X = cands[bi].copy()
    best_v = float(vals[bi])
    curr_X = best_X.copy()
    curr_v = best_v
    T = 5e-4
    trial = 0
    # LP-free basin hopping with cheap quench
    while time.time() < SCREEN_END:
        trial += 1
        if trial % 30 == 0:
            curr_X = best_X.copy(); curr_v = best_v
        mode = (trial - 1) % 6
        try:
            Gc = curr_X @ curr_X.T
            np.fill_diagonal(Gc, -2.0)
            rowmax = Gc.max(axis=1)
            order = np.argsort(rowmax)
        except Exception:
            order = np.arange(n); rowmax = np.zeros(n)
        Xp = curr_X.copy()
        if mode == 0:
            # big global kick to leave jammed basin
            Xp = Xp + 0.35 * rng.standard_normal((n, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        elif mode == 1:
            # medium kick
            Xp = Xp + 0.12 * rng.standard_normal((n, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        elif mode == 2:
            # worst-point void reinsert
            idx = int(order[-1])
            Xp = void_reinsert(Xp, [idx], rng, M=4000)
            Xp = Xp + 0.005 * rng.standard_normal((n, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        elif mode == 3:
            # 3 worst void reinsert sequentially
            idxs = order[-3:]
            Xp = void_reinsert(Xp, idxs, rng, M=2500)
            Xp = Xp + 0.004 * rng.standard_normal((n, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        elif mode == 4:
            # block shake
            k = 32
            idxs = rng.choice(n, size=k, replace=False)
            Xp[idxs] += 0.20 * rng.standard_normal((k, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        else:
            # fresh uniform random (isotropic regular start)
            Xp = rng.standard_normal((n, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        # cheap quench: low beta spread -> high beta focus
        try:
            if mode == 5:
                Xp = smooth_quench(Xp, steps=650, beta0=8.0, beta1=280.0, lr0=0.05, lr1=0.004)
            else:
                Xp = smooth_quench(Xp, steps=550, beta0=12.0, beta1=300.0, lr0=0.04, lr1=0.004)
            v = max_cos(Xp)
        except Exception:
            continue
        if not np.isfinite(v):
            continue
        if v < best_v:
            best_v = v; best_X = Xp.copy()
            curr_X = Xp.copy(); curr_v = v
        else:
            if v < curr_v or rng.random() < np.exp((curr_v - v) / T):
                curr_X = Xp.copy(); curr_v = v
        # stop if time low
        if time.time() > HARD_END - 35.0:
            break
    # intensify best with longer high-beta quench (two variants, pick best)
    try:
        if time.time() < HARD_END - 30.0:
            X1 = smooth_quench(best_X.copy(), steps=1500, beta0=15.0, beta1=600.0, lr0=0.02, lr1=0.002)
            v1 = max_cos(X1)
            if v1 < best_v:
                best_v = v1; best_X = X1
        if time.time() < HARD_END - 28.0:
            X2 = smooth_quench(best_X.copy(), steps=1200, beta0=20.0, beta1=500.0, lr0=0.012, lr1=0.0015, mu=0.9)
            v2 = max_cos(X2)
            if v2 < best_v:
                best_v = v2; best_X = X2
    except Exception:
        pass
    # single final LP squeeze
    try:
        deadline = HARD_END - 1.5
        if time.time() < deadline - 2.0:
            best_X = lp_polish_deadline(best_X, deadline, r_init=0.002, cap=2500)
    except Exception:
        pass
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    return best_X
