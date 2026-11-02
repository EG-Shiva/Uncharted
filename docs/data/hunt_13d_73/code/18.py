import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _rowwise_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)


def _anneal(X0, steps, lr0, lr1, T0, T1, mu=0.88):
    X = X0.copy()
    vel = np.zeros_like(X)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    lt0, lt1 = np.log(T0), np.log(T1)
    n = X.shape[0]
    for k in range(steps):
        frac = k / (steps - 1) if steps > 1 else 1.0
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        T = np.exp(lt0 + (lt1 - lt0) * frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            break
        grad = (E @ X) / S
        dot = (grad * X).sum(axis=1, keepdims=True)
        gt = grad - dot * X
        mn = float(np.sqrt((gt * gt).sum(axis=1)).mean() / np.sqrt(X.shape[1])) if True else 0
        # mean row norm
        rn = np.sqrt((gt * gt).sum(axis=1, keepdims=True))
        mean_n = float(rn.mean()) + 1e-12
        if not np.isfinite(mean_n) or mean_n < 1e-15:
            break
        sd = gt / mean_n
        vel = mu * vel + (1.0 - mu) * sd
        X = X - lr * vel
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        X /= nrm
    return X


def _find_hole(X_keep, rng, n_cand=4000):
    d = X_keep.shape[1]
    C = rng.normal(size=(n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = C @ X_keep.T
    m = dots.max(axis=1)
    return C[int(np.argmin(m))]


def _lp_polish(x, t_end):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < t_end and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    rng = np.random.default_rng(7)
    n, d = 73, 13
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.normal(size=(n, d))
    if xb.shape != (n, d):
        tmp = rng.normal(size=(n, d))
        m0 = min(xb.shape[0], n)
        m1 = min(xb.shape[1], d)
        tmp[:m0, :m1] = xb[:m0, :m1]
        xb = tmp
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    X_best = xb.copy()
    X_cur = xb.copy()
    cur_val = best_val
    EXPLORE_END = DEADLINE - 30.0
    it = 0
    sigmas = [0.008, 0.015, 0.03, 0.06, 0.12, 0.22]
    no_imp = 0
    while time.time() < EXPLORE_END:
        it += 1
        # reset to best periodically
        if no_imp >= 25:
            X_cur = X_best.copy()
            cur_val = best_val
            no_imp = 0
        use_best = rng.random() < 0.7
        base = X_best if use_best else X_cur
        op = rng.random()
        X_try = base.copy()
        if op < 0.42:
            sigma = sigmas[int(rng.integers(0, len(sigmas)))]
            X_try = base + sigma * rng.normal(size=(n, d))
            X_try /= np.linalg.norm(X_try, axis=1, keepdims=True)
            if sigma > 0.08:
                steps, lr0, lr1, T0, T1 = 2200, 0.030, 0.0012, 0.020, 0.0012
            elif sigma > 0.025:
                steps, lr0, lr1, T0, T1 = 1600, 0.020, 0.0010, 0.012, 0.0010
            else:
                steps, lr0, lr1, T0, T1 = 1100, 0.012, 0.0008, 0.008, 0.0008
        elif op < 0.68:
            # kill worst k and reinsert into holes
            k = 1
            r = rng.random()
            if r > 0.85:
                k = 3
            elif r > 0.55:
                k = 2
            rm = _rowwise_max(base)
            order = np.argsort(-rm)
            victims = order[:k].copy()
            if rng.random() < 0.25:
                victims[0] = int(rng.integers(0, n))
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[victims] = False
            X_keep = base[keep_mask]
            new_vecs = []
            cur_keep = X_keep
            for _ in range(k):
                h = _find_hole(cur_keep, rng, n_cand=4000)
                new_vecs.append(h)
                cur_keep = np.vstack([cur_keep, h[None, :]])
            for vi, hv in zip(victims, new_vecs):
                X_try[vi] = hv
            # small jitter to neighbours to unlock
            X_try += 0.004 * rng.normal(size=(n, d))
            X_try /= np.linalg.norm(X_try, axis=1, keepdims=True)
            steps, lr0, lr1, T0, T1 = 2000, 0.025, 0.0010, 0.015, 0.0010
        elif op < 0.86:
            # subset shake of most loaded points
            rm = _rowwise_max(base)
            order = np.argsort(-rm)
            s = int(rng.integers(4, 11))
            idx = order[:s]
            sigma = float(rng.choice([0.03, 0.06, 0.12]))
            X_try[idx] = base[idx] + sigma * rng.normal(size=(s, d))
            X_try /= np.linalg.norm(X_try, axis=1, keepdims=True)
            steps, lr0, lr1, T0, T1 = 1600, 0.020, 0.0010, 0.012, 0.0010
        else:
            # coherent cap shake
            u = rng.normal(size=d)
            u /= np.linalg.norm(u)
            proj = base @ u
            thr = float(rng.choice([0.1, 0.25, 0.4]))
            mask = proj > thr
            if mask.sum() < 2:
                mask = np.zeros(n, dtype=bool)
                mask[np.argsort(-proj)[:8]] = True
            shift = rng.normal(size=d) * float(rng.choice([0.05, 0.10, 0.18]))
            shift = shift - (shift @ u) * u
            X_try[mask] = base[mask] + shift[None, :] + 0.01 * rng.normal(size=(int(mask.sum()), d))
            X_try /= np.linalg.norm(X_try, axis=1, keepdims=True)
            steps, lr0, lr1, T0, T1 = 1800, 0.022, 0.0010, 0.014, 0.0010
        # adapt steps to remaining time
        if time.time() + steps * 0.00005 > EXPLORE_END:
            steps = max(400, int((EXPLORE_END - time.time()) / 0.00005))
            if steps <= 400:
                break
        try:
            X_opt = _anneal(X_try, steps, lr0, lr1, T0, T1)
        except Exception:
            continue
        try:
            v = _max_cos(X_opt)
        except Exception:
            continue
        if not np.isfinite(v):
            continue
        if v < best_val - 1e-12:
            best_val = v
            X_best = X_opt.copy()
            X_cur = X_opt.copy()
            cur_val = v
            no_imp = 0
        else:
            no_imp += 1
            if v < cur_val or rng.random() < 0.06:
                X_cur = X_opt.copy()
                cur_val = v
        # early stop if clearly improved a lot? keep searching for more
    # fine squeeze of best
    try:
        if time.time() < DEADLINE - 26.0:
            X_fine = _anneal(X_best, 2500, 0.010, 0.0004, 0.006, 0.0006, mu=0.9)
            vf = _max_cos(X_fine)
            if vf < best_val:
                best_val = vf
                X_best = X_fine
    except Exception:
        pass
    try:
        if time.time() < DEADLINE - 2.0:
            X_best = _lp_polish(X_best, DEADLINE - 0.5)
    except Exception:
        pass
    X_best = X_best / np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best
