import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def _lp_polish(X, deadline, r0=2e-3):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = r0
    iu0 = np.triu_indices(n, 1)
    try:
        while time.time() < deadline and radius > 1e-14:
            Gfull = X @ X.T
            c = Gfull[iu0]
            m = float(c.max())
            thr = m - max(20.0 * radius, 2e-9)
            sel = c > thr
            # cap constraints for speed
            if np.count_nonzero(sel) > 4000:
                # keep tightest 4000
                idx = np.where(sel)[0]
                order = np.argsort(-c[idx])[:4000]
                mask = np.zeros_like(sel, dtype=bool)
                mask[idx[order]] = True
                sel = mask
            I = iu0[0][sel]
            J = iu0[1][sel]
            p = len(I)
            if p == 0:
                break
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(),
                               (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            b_ub = -c[sel]
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq,
                              b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                              method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = X + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
            if v < best:
                X, best = y, v
                radius = min(radius * 1.5, 0.02)
            else:
                radius *= 0.5
    except Exception:
        pass
    return X


def _smooth_opt(X, iters, s0, s1, lr0, lr1, mu):
    n, d = X.shape
    V = np.zeros_like(X)
    for it in range(iters):
        frac = it / max(1, iters)
        s = s0 + (s1 - s0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        W = np.exp(s * (G - m))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True)
        Grad = W @ X
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        Gt /= (rs + 1.0)
        V = mu * V + (1.0 - mu) * Gt
        step = lr * V
        # clip per-row step to avoid blow up
        sn = np.sqrt((step * step).sum(axis=1, keepdims=True)) + 1e-12
        scale = np.minimum(1.0, 0.05 / sn)
        step = step * scale
        X = X - step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def _void_dir(Y, rng):
    # Y: (m,5) unit rows, find direction minimizing max dot
    S = rng.randn(6000, 5)
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    dots = S @ Y.T
    md = dots.max(axis=1)
    v = S[np.argmin(md)].copy()
    for _ in range(150):
        dd = Y @ v
        dm = float(dd.max())
        w = np.exp(60.0 * (dd - dm))
        w /= w.sum()
        g = Y.T @ w
        g = g - (g @ v) * v
        v = v - 0.08 * g
        nrm = float(np.linalg.norm(v))
        if nrm < 1e-12:
            break
        v /= nrm
    return v


def solve():
    t0 = time.time()
    t_end = t0 + 170.0
    np.random.seed(0)
    rng = np.random.RandomState(0)
    try:
        best = np.load("best.npy").astype(float)
    except Exception:
        try:
            best = np.load("record.npy").astype(float)
        except Exception:
            best = rng.randn(173, 5)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val

    trial = 0
    # reserve time for final polish
    while time.time() < t_end - 10.0:
        trial += 1
        X = cur.copy()
        n = X.shape[0]
        r = rng.rand()
        if r < 0.22:
            sigma = 0.008 + rng.rand() * 0.025
            X = X + rng.randn(n, 5) * sigma
        elif r < 0.48:
            k = int(rng.choice([2, 3, 5, 8, 15, 30]))
            idx = rng.choice(n, size=k, replace=False)
            sigma = 0.05 + rng.rand() * 0.16
            X[idx] = X[idx] + rng.randn(k, 5) * sigma
        elif r < 0.68:
            # rattler / low-contact relocation to voids
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cnt = (G > m - 1e-6).sum(axis=1)
            low = np.where(cnt <= 2)[0]
            if len(low) == 0:
                low = np.argsort(cnt)[:3]
            take = low[:2]
            for li in take:
                mask = np.ones(n, dtype=bool)
                mask[li] = False
                Y = X[mask]
                v = _void_dir(Y, rng)
                X[li] = v
            X = X + rng.randn(n, 5) * 0.008
        elif r < 0.85:
            # shake jammed neighbourhood
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cnt = (G > m - 1e-6).sum(axis=1)
            hi = int(np.argmax(cnt))
            dots = X @ X[hi]
            nn = np.argsort(-dots)[:12]
            X[nn] = X[nn] + rng.randn(len(nn), 5) * (0.05 + rng.rand() * 0.08)
            # small global jitter
            X = X + rng.randn(n, 5) * 0.004
        else:
            k = int(rng.randint(1, 4))
            idx = rng.choice(n, size=k, replace=False)
            R = rng.randn(k, 5)
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            X[idx] = R
            X = X + rng.randn(n, 5) * 0.005
        X /= np.linalg.norm(X, axis=1, keepdims=True)

        # smooth repulsion annealing; scale iters with remaining time
        iters = 1800
        if r < 0.3:
            iters = 1200
        X2 = _smooth_opt(X, iters=iters, s0=60.0, s1=380.0,
                         lr0=0.16, lr1=0.02, mu=0.8)
        v2 = _max_cos(X2)

        # decide if worth LP polishing
        thresh = min(best_val + 0.004, cur_val + 0.008)
        if v2 < thresh:
            X3 = _lp_polish(X2, deadline=min(time.time() + 2.2, t_end - 10.0), r0=2e-3)
            v3 = _max_cos(X3)
        else:
            X3, v3 = X2, v2

        if v3 < best_val - 1e-12:
            best, best_val = X3.copy(), float(v3)
            cur, cur_val = X3.copy(), float(v3)
        elif v3 < cur_val + 0.004 and v3 < 0.85:
            # basin-hopping random walk: accept slightly worse
            if rng.rand() < 0.9:
                cur, cur_val = X3.copy(), float(v3)
        # occasional restart cur to best to intensify
        if trial % 25 == 0 and rng.rand() < 0.5:
            cur, cur_val = best.copy(), best_val
        if best_val < 0.70:
            break

    # final intensive polish of best
    best = _lp_polish(best, deadline=t_end - 1.0, r0=1.5e-3)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    # safety: remove NaN / zero rows
    best = np.nan_to_num(best, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(best, axis=1, keepdims=True)
    nr[nr < 1e-12] = 1.0
    best = best / nr
    return best
