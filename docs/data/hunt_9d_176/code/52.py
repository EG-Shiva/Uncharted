import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _quench(X, steps, beta0, beta1, lr0, lr1):
    n = X.shape[0]
    sq = np.sqrt(float(n))
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = beta0 * (beta1 / beta0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        # weights focused on top
        W = np.exp((G - m) * beta)
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        proj = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - proj * X
        gnorm = float(np.linalg.norm(Gt) / sq)
        if gnorm < 1e-16 or not np.isfinite(gnorm):
            continue
        X -= (lr / gnorm) * Gt
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        # guard zero
        nr[~np.isfinite(nr)] = 1.0
        nr[nr < 1e-12] = 1.0
        X /= nr
    return X

def _void_reinsert(X, idxs, rng, M=2048):
    n, d = X.shape
    idxs = np.asarray(idxs, dtype=int)
    k = len(idxs)
    if k == 0:
        return X
    set_idx = set(idxs.tolist())
    # fixed points
    mask_fixed = np.ones(n, dtype=bool)
    mask_fixed[idxs] = False
    cur = X[mask_fixed].copy()  # (n-k) x d
    for j in idxs:
        R = rng.standard_normal((M, d))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        D = R @ cur.T
        worst = D.max(axis=1)
        pick = int(np.argmin(worst))
        X[j] = R[pick]
        cur = np.vstack([cur, X[j][None, :]])
    return X

def _lp_polish_deadline(x, deadline, r_init=0.002, cap=2600):
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
            radius *= 0.5
            continue
        if cnt > cap:
            idx_all = np.where(sel)[0]
            # keep top-cap
            vals = c_all[idx_all]
            ordp = np.argpartition(-vals, cap)[:cap]
            keep = np.zeros_like(sel)
            keep[idx_all[ordp]] = True
            sel = keep
        I = iu0[0][sel]; J = iu0[1][sel]
        c_sel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty((p, 2 * d + 1), dtype=int)
        cols[:, :d] = I[:, None] * d + np.arange(d)
        cols[:, d:2 * d] = J[:, None] * d + np.arange(d)
        cols[:, 2 * d] = n * d
        cols = cols.ravel()
        vals2 = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals2, (rows, cols)), shape=(p, nv)).tocsr()
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
    HARD_END = t0 + 176.0
    LOOP_END = t0 + 148.0
    rng = np.random.default_rng(12345)
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
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_X = cands[bi].copy()
    best_v = float(vals[bi])
    curr_X = best_X.copy()
    curr_v = best_v
    sigmas = [0.004, 0.008, 0.015, 0.03, 0.06, 0.10]
    T = 3e-4
    trial = 0
    # cheap quench params
    while time.time() < LOOP_END:
        trial += 1
        if trial % 40 == 0:
            curr_X = best_X.copy(); curr_v = best_v
        sigma = sigmas[(trial - 1) % len(sigmas)]
        mode = trial % 5
        Xp = curr_X.copy()
        try:
            Gc = curr_X @ curr_X.T
            np.fill_diagonal(Gc, -2.0)
            rowmax = Gc.max(axis=1)
            order = np.argsort(rowmax)
        except Exception:
            order = np.arange(n); rowmax = np.zeros(n)
        if mode == 0:
            Xp = Xp + sigma * rng.standard_normal((n, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            steps = 350
        elif mode == 1:
            k = 1 if (trial % 2 == 0) else 2
            idxs = order[-k:][::-1]
            Xp = _void_reinsert(Xp, idxs, rng, M=2048)
            Xp = Xp + 0.002 * rng.standard_normal((n, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            steps = 380
        elif mode == 2:
            k = 3 if (trial % 2 == 0) else 5
            # stress = rowmax biased + random among top
            # take worst 12, sample k
            pool = order[-12:]
            idxs = rng.choice(pool, size=k, replace=False)
            # order by rowmax descending for sequential insert
            idxs = idxs[np.argsort(-rowmax[idxs])]
            Xp = _void_reinsert(Xp, idxs, rng, M=1536)
            Xp = Xp + 0.002 * rng.standard_normal((n, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            steps = 380
        elif mode == 3:
            k = 20
            idxs = rng.choice(n, size=k, replace=False)
            Xp[idxs] += sigma * rng.standard_normal((k, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            steps = 320
        else:
            # targeted shake of worst 10 + global small
            k = 10
            idx_w = order[-k:]
            noise = 0.002 * rng.standard_normal((n, d))
            noise[idx_w] += (sigma * 1.2) * rng.standard_normal((k, d))
            Xp = Xp + noise
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            steps = 350
        # cheap quench (no LP)
        try:
            # vary beta slightly per trial for diversity
            b0 = 22.0 if (trial % 3 != 0) else 35.0
            b1 = 260.0 if (trial % 2 == 0) else 350.0
            lr0 = 0.022; lr1 = 0.0025
            Xp = _quench(Xp, steps=steps, beta0=b0, beta1=b1, lr0=lr0, lr1=lr1)
            v = _max_cos(Xp)
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
        # opportunistic early break if time low
        if time.time() > LOOP_END - 0.5:
            break
    try:
        final_deadline = HARD_END - 1.5
        if time.time() < final_deadline:
            # short LP squeeze of best only
            best_X = _lp_polish_deadline(best_X, final_deadline, r_init=0.0015, cap=2600)
    except Exception:
        pass
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    return best_X
