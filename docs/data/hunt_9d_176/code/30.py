import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _quench(X, steps=400, beta0=30.0, beta1=300.0, lr0=0.03, lr1=0.003):
    n, d = X.shape
    Y = X.copy()
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = beta0 * (beta1 / beta0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = Y @ Y.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        # weights focused on top
        W = np.exp((G - m) * beta)
        np.fill_diagonal(W, 0.0)
        Grad = W @ Y
        proj = np.sum(Grad * Y, axis=1, keepdims=True)
        Gt = Grad - proj * Y
        gn = float(np.linalg.norm(Gt) / np.sqrt(n))
        if not np.isfinite(gn) or gn < 1e-16:
            continue
        Y = Y - (lr / gn) * Gt
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr[nr < 1e-12] = 1.0
        Y = Y / nr
        if not np.all(np.isfinite(Y)):
            break
    return Y

def _hole_replace(Y, k, rng, ntrials=1500):
    n, d = Y.shape
    G = Y @ Y.T
    np.fill_diagonal(G, -2.0)
    pmax = G.max(axis=1)
    idx = np.argpartition(-pmax, k)[:k]
    idx = idx[np.argsort(-pmax[idx])]
    mask = np.ones(n, dtype=bool)
    mask[idx] = False
    cur = Y[mask].copy()
    Z = Y.copy()
    for j in idx:
        T = rng.standard_normal((ntrials, d))
        T /= np.linalg.norm(T, axis=1, keepdims=True)
        D = T @ cur.T
        mm = D.max(axis=1)
        pick = int(np.argmin(mm))
        Z[j] = T[pick]
        cur = np.vstack([cur, T[pick][None, :]])
    return Z

def _lp_polish_deadline(x, deadline, r_init=0.004, cap=2600):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
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
            break
        if cnt > cap:
            ia = np.where(sel)[0]
            part = np.argpartition(-c_all[ia], cap)[:cap]
            keep = np.zeros_like(sel)
            keep[ia[part]] = True
            sel = keep
            cnt = cap
        I = iu0[0][sel]; J = iu0[1][sel]
        c_sel = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
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
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    HARD = t0 + 174.0
    LOOP_END = t0 + 150.0
    FINAL_END = t0 + 173.0
    rng = np.random.default_rng(0)
    n, d = 176, 9
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
    if not cands:
        a = rng.standard_normal((n, d)); a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_X = cands[bi].copy()
    best_v = float(vals[bi])
    # quick diverse starts: random jittered copies quenched very briefly to seed
    it = 0
    sigmas = [0.015, 0.03, 0.06, 0.10, 0.16]
    while time.time() < LOOP_END:
        it += 1
        # choose mode
        r = rng.random()
        if r < 0.55:
            sigma = sigmas[rng.integers(0, len(sigmas))]
            Y = best_X + sigma * rng.standard_normal((n, d))
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        elif r < 0.8:
            # collective shear: small random matrix applied then jitter
            A = 0.06 * rng.standard_normal((d, d))
            A = (A - A.T) * 0.5  # skew ~ rotation
            Y = best_X @ (np.eye(d) + A)
            Y = Y + 0.02 * rng.standard_normal((n, d))
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        else:
            # hole replacement of worst few
            k = int(rng.integers(4, 14))
            sigma = 0.01 * rng.random()
            Yb = best_X + sigma * rng.standard_normal((n, d))
            Yb /= np.linalg.norm(Yb, axis=1, keepdims=True)
            try:
                Y = _hole_replace(Yb, k, rng, ntrials=1200)
            except Exception:
                Y = Yb
        steps = 350 if (it % 4) else 500
        # vary beta range slightly for diversity
        b1 = 250.0 + 150.0 * rng.random()
        try:
            Z = _quench(Y, steps=steps, beta0=25.0, beta1=b1, lr0=0.03, lr1=0.0025)
        except Exception:
            continue
        try:
            v = _max_cos(Z)
        except Exception:
            continue
        if v < best_v and np.isfinite(v):
            best_v = v
            best_X = Z.copy()
        # time check for loop exit to reserve final polish
        if time.time() >= LOOP_END:
            break
    # final squeeze
    try:
        if time.time() < FINAL_END:
            best_X = _lp_polish_deadline(best_X, FINAL_END)
    except Exception:
        pass
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    return best_X
