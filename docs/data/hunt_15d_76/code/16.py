import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contacts(x, thr=1e-9):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = g.max()
    cnt = (g > m - thr).sum(axis=1)
    return m, cnt

def _fast_descent(x, steps=300, beta0=40.0, beta1=250.0, s0=0.02, s1=0.001, seed=0):
    n, d = x.shape
    X = x.copy()
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** frac
        step = s0 + (s1 - s0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        # stabilized exp
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        # symmetrize (already symmetric) and gradient of LSE
        S = E.sum()
        if not np.isfinite(S) or S <= 0:
            break
        Grad = (E @ X) / S
        # tangent projection
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        rms = float(np.sqrt((Gt ** 2).mean())) + 1e-12
        if not np.isfinite(rms):
            break
        eta = step / rms
        # clip eta to avoid blowup
        if eta > 5.0:
            eta = 5.0
        X = X - eta * Gt
        # re-normalize
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[~np.isfinite(nr)] = 1.0
        nr[nr < 1e-12] = 1.0
        X = X / nr
    return X

def _hole_candidate(X, keep_mask, rng, K=2000):
    # find vector minimizing max dot to kept points
    n, d = X.shape
    R = rng.standard_normal((K, d))
    R /= np.linalg.norm(R, axis=1, keepdims=True) + 1e-12
    S = R @ X[keep_mask].T  # K x nkept
    scores = S.max(axis=1)
    return R[int(np.argmin(scores))]

def _lp_polish(x, seconds, t_deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < t_deadline and radius > 1e-14:
        if time.time() >= t_deadline:
            break
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
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
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr < 1e-12] = 1.0
        y = y / nr
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        # remaining time check for seconds budget
        # (outer deadline handles it)
    return x

def solve():
    t0 = time.time()
    T_LIMIT = 165.0
    deadline = t0 + T_LIMIT
    rng = np.random.default_rng(0)
    np.random.seed(0)
    # load starting points
    best_x = None
    best_v = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            a = np.load(f).astype(float)
            if a.shape != (76, 15):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            v = _max_cos(a)
            if v < best_v:
                best_v = v
                best_x = a.copy()
        except Exception:
            continue
    if best_x is None:
        r = rng.standard_normal((76, 15))
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        best_x = r
        best_v = _max_cos(r)
    base_x = best_x.copy()
    # contacts to find rattlers
    try:
        _, cnt0 = _contacts(base_x)
    except Exception:
        cnt0 = np.ones(76)
    rattlers = np.where(cnt0 < 0.5)[0].tolist()
    if len(rattlers) == 0:
        rattlers = np.argsort(cnt0)[:6].tolist()
    # basin hopping with fast descent; reserve ~35s for final LP
    lp_reserve = 35.0
    search_deadline = deadline - lp_reserve
    n_att = 0
    # schedule of sigmas
    sigmas = [0.02, 0.05, 0.09, 0.14, 0.05, 0.09, 0.14, 0.20, 0.03, 0.07,
              0.12, 0.18, 0.04, 0.08, 0.15, 0.25, 0.05, 0.10, 0.16, 0.06,
              0.11, 0.20, 0.04, 0.09, 0.14]
    for att, sigma in enumerate(sigmas):
        if time.time() > search_deadline:
            break
        # start from current best (elitist) with occasional restart from base
        if att % 7 == 6 and att > 0:
            cur = base_x.copy()
        else:
            cur = best_x.copy()
        n, d = cur.shape
        # choose subset to shake: rattlers + random + overloaded
        try:
            _, cnt = _contacts(cur)
        except Exception:
            cnt = np.ones(n)
        order_hi = np.argsort(-cnt)
        # perturb
        noise = rng.standard_normal((n, d))
        # stronger noise on rattlers and top overloaded
        scale = np.full(n, sigma * 0.5)
        for r in rattlers:
            if r < n:
                scale[r] = sigma * 1.5
        # top-3 overloaded get extra shake
        for k in range(min(3, n)):
            scale[order_hi[k]] = max(scale[order_hi[k]], sigma * 1.2)
        cur = cur + scale[:, None] * noise
        cur /= np.linalg.norm(cur, axis=1, keepdims=True) + 1e-12
        # hole-optimized re-insertion for 1-2 rattlers (cyclic)
        n_reins = 1 + (att % 2)
        for t in range(n_reins):
            idx = rattlers[(att + t) % len(rattlers)] if len(rattlers) > 0 else int(rng.integers(0, n))
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            try:
                h = _hole_candidate(cur, mask, rng, K=1500)
            except Exception:
                h = rng.standard_normal(d)
                h /= np.linalg.norm(h) + 1e-12
            cur[idx] = h
        # annealed descent
        try:
            # vary steps/beta slightly per attempt
            steps = 280 if att % 2 == 0 else 350
            b0 = 35.0 + (att % 5) * 5.0
            b1 = 220.0 + (att % 3) * 40.0
            cur2 = _fast_descent(cur, steps=steps, beta0=b0, beta1=b1, s0=0.018, s1=0.0008)
        except Exception:
            continue
        try:
            v = _max_cos(cur2)
        except Exception:
            continue
        n_att += 1
        if v < best_v - 1e-12:
            best_v = v
            best_x = cur2.copy()
            # update rattlers to new solution's rattlers for guidance
            try:
                _, cnt2 = _contacts(best_x)
                rattlers = np.where(cnt2 < 0.5)[0].tolist()
                if len(rattlers) == 0:
                    rattlers = np.argsort(cnt2)[:6].tolist()
            except Exception:
                pass
            # early exit if big win
            if best_v < 0.1665:
                break
        # time guard
        if time.time() > search_deadline:
            break
    # final polish: short LP on best
    try:
        remaining = deadline - time.time() - 2.0
        if remaining > 5.0:
            use = min(remaining, 32.0)
            best_x = _lp_polish(best_x, use, time.time() + use)
            best_v = _max_cos(best_x)
    except Exception:
        pass
    # safety normalize + tiny shrink not needed; just normalize
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    # ensure finite
    if not np.all(np.isfinite(best_x)):
        best_x = base_x.copy()
    return best_x
