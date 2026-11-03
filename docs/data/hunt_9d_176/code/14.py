import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def quench(X, steps, beta0, beta1, lr0, lr1):
    n, d = X.shape
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = beta0 * (beta1 / beta0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        W = np.exp((G - m) * beta)
        np.fill_diagonal(W, 0.0)
        Gr = W @ X
        proj = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - proj * X
        gn = float(np.linalg.norm(Gt) / np.sqrt(n))
        if not np.isfinite(gn) or gn < 1e-16:
            continue
        X = X - (lr / gn) * Gt
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
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
        if time.time() > deadline - 0.5:
            break
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > cap:
            idx = np.where(sel)[0]
            part = np.argpartition(-c_all[idx], cap)[:cap]
            keep = np.zeros_like(sel)
            keep[idx[part]] = True
            sel = keep
        I = iu0[0][sel]; J = iu0[1][sel]
        c_sel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty((p, 2 * d + 1), dtype=int)
        cols[:, :d] = I[:, None] * d + np.arange(d)
        cols[:, d:2*d] = J[:, None] * d + np.arange(d)
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
    LOOP_END = t0 + 152.0
    FINAL_RES = 18.0
    rng = np.random.default_rng(11)
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
    if not cands:
        X0 = rng.standard_normal((n, d)); X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands.append(X0)
    vals = [max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_X = cands[bi].copy()
    best_v = float(vals[bi])
    curr_X = best_X.copy()
    curr_v = best_v
    sigmas = [0.003, 0.006, 0.012, 0.025, 0.05, 0.10]
    T = 3e-4
    trial = 0
    while time.time() < LOOP_END:
        trial += 1
        if trial % 30 == 0:
            curr_X = best_X.copy(); curr_v = best_v
        sigma = sigmas[(trial - 1) % len(sigmas)]
        mode = trial % 5
        # row worstness for targeting
        try:
            Gc = curr_X @ curr_X.T
            np.fill_diagonal(Gc, -2.0)
            rowmax = Gc.max(axis=1)
            order = np.argsort(rowmax)
        except Exception:
            order = np.arange(n)
        Xp = curr_X.copy()
        if mode == 0:
            Xp = Xp + sigma * rng.standard_normal((n, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        elif mode == 1:
            k = 10
            idx_w = order[-k:]
            noise = 0.002 * rng.standard_normal((n, d))
            noise[idx_w] += (sigma * 1.5) * rng.standard_normal((k, d))
            Xp = Xp + noise
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        elif mode == 2:
            idx = int(order[-1])
            M = 2000
            R = rng.standard_normal((M, d)); R /= np.linalg.norm(R, axis=1, keepdims=True)
            dots = R @ Xp.T
            dots[:, idx] = -2.0
            worst = dots.max(axis=1)
            pick = int(np.argmin(worst))
            Xp[idx] = R[pick]
            Xp = Xp + 0.002 * rng.standard_normal((n, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        elif mode == 3:
            k = 3
            idxs = order[-k:]
            Xp = Xp + 0.003 * rng.standard_normal((n, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            for idx in idxs:
                M = 900
                R = rng.standard_normal((M, d)); R /= np.linalg.norm(R, axis=1, keepdims=True)
                dots = R @ Xp.T
                dots[:, int(idx)] = -2.0
                worst = dots.max(axis=1)
                pick = int(np.argmin(worst))
                Xp[int(idx)] = R[pick]
        else:
            k = 20
            idxs = rng.choice(n, size=k, replace=False)
            Xp[idxs] += sigma * rng.standard_normal((k, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        steps = 600 if sigma < 0.04 else 400
        try:
            Xp = quench(Xp, steps=steps, beta0=25.0, beta1=170.0, lr0=0.025, lr1=0.002)
        except Exception:
            pass
        try:
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
        if time.time() > LOOP_END - 0.3:
            break
        # leave time for final polish
        if HARD_END - time.time() < FINAL_RES + 2.0:
            break
    try:
        fd = HARD_END - 1.0
        if time.time() < fd:
            best_X = lp_polish_deadline(best_X, fd, r_init=0.0015, cap=2500)
    except Exception:
        pass
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    return best_X
