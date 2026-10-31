import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _refine(X, steps, a0, a1, lr0, lr1, deadline=None):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2 = 0.9, 0.999
    eps = 1e-8
    p1 = 1.0
    p2 = 1.0
    for t in range(1, steps + 1):
        if deadline is not None and (t & 63) == 0:
            if time.time() > deadline:
                break
        frac = t / float(steps)
        alpha = a0 + (a1 - a0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        D = X @ X.T
        np.fill_diagonal(D, -10.0)
        dm = D.max()
        # weights focused on large dots
        W = np.exp(alpha * (D - dm))
        np.fill_diagonal(W, 0.0)
        G = W @ X
        s = (G * X).sum(axis=1, keepdims=True)
        Gt = G - s * X
        m *= b1
        m += (1.0 - b1) * Gt
        v *= b2
        v += (1.0 - b2) * (Gt * Gt)
        p1 *= b1
        p2 *= b2
        mhat = m / (1.0 - p1 + 1e-20)
        vhat = v / (1.0 - p2 + 1e-20)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sn = (step * X).sum(axis=1, keepdims=True)
        step -= sn * X
        X -= step
        nr = np.sqrt((X * X).sum(axis=1, keepdims=True))
        X /= np.maximum(nr, 1e-12)
    return X

def _greedy_reinsert(X_rem, k, rng, n_cand=12000):
    # random candidates on S^3
    cand = rng.normal(size=(n_cand, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    X_aug = X_rem
    for _ in range(k):
        best_score = 2.0
        best_pt = None
        for a in range(0, n_cand, 3000):
            cc = cand[a:a+3000]
            dd = cc @ X_aug.T
            mm = dd.max(axis=1)
            j = int(np.argmin(mm))
            if float(mm[j]) < best_score:
                best_score = float(mm[j])
                best_pt = cc[j]
        jp = best_pt + rng.normal(0, 0.01, size=4)
        nrm = np.linalg.norm(jp)
        if nrm < 1e-12:
            jp = best_pt.copy()
        else:
            jp = jp / nrm
        X_aug = np.vstack([X_aug, jp.reshape(1, 4)])
    return X_aug

def _lp_polish(x, deadline):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return x
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        mval = c.max()
        sel = c > mval - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if getattr(res, "status", 1) != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() > deadline:
            break
    return x

def solve():
    t0 = time.time()
    deadline_final = t0 + 172.0
    phase1_end = t0 + 132.0
    rng = np.random.default_rng(0)
    n_target = 292
    d_target = 4
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n_target, d_target):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)) and np.min(nr) > 1e-9:
                    a = a / nr
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        X_best = rng.normal(size=(n_target, d_target))
        X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
    else:
        vals = [_max_cos(c) for c in cands]
        X_best = cands[int(np.argmin(vals))].copy()
    best_val = _max_cos(X_best)
    it = 0
    # main basin-hopping
    while time.time() < phase1_end:
        it += 1
        remaining = phase1_end - time.time()
        if remaining < 1.0:
            break
        mode = it % 3  # 0: small shake, 1,2: cluster excision
        if mode == 0:
            sigma = 0.008 + 0.007 * float(rng.random())
            Xt = X_best + rng.normal(0, sigma, size=X_best.shape)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            steps = 700 if remaining > 5 else 300
            Xt = _refine(Xt, steps, 40.0, 150.0, 0.004, 0.001, deadline=phase1_end)
        else:
            # pick center: worst pair or random
            try:
                D = X_best @ X_best.T
                np.fill_diagonal(D, -2.0)
                flat = int(np.argmax(D))
                i0, j0 = divmod(flat, n_target)
            except Exception:
                i0, j0 = 0, 1
            if rng.random() < 0.55:
                center = i0 if rng.random() < 0.5 else j0
            else:
                center = int(rng.integers(0, n_target))
            k = int(rng.integers(5, 10))  # 5..9
            # k nearest to center (largest dot)
            dots = X_best @ X_best[center]
            order = np.argsort(-dots)
            rem_idx = order[k:]
            # ensure size correct
            if rem_idx.shape[0] != n_target - k:
                rem_idx = np.delete(np.arange(n_target), center)
                # trim/pad to size
                rem_idx = rem_idx[:n_target - k]
            X_rem = X_best[rem_idx]
            try:
                Xt = _greedy_reinsert(X_rem, k, rng, n_cand=12000)
            except Exception:
                continue
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            steps = 1100 if remaining > 8 else 500
            # vary alpha schedule slightly per iteration
            if (it // 3) % 2 == 0:
                Xt = _refine(Xt, steps, 50.0, 180.0, 0.004, 0.0008, deadline=phase1_end)
            else:
                Xt = _refine(Xt, steps, 30.0, 120.0, 0.005, 0.0012, deadline=phase1_end)
        try:
            v = _max_cos(Xt)
        except Exception:
            continue
        if v < best_val - 1e-12:
            X_best = Xt
            best_val = v
        # opportunistic extra squeeze if close and time permits
        # (don't waste time)
    # final high-alpha squeeze
    if time.time() < deadline_final - 8.0:
        try:
            X_best = _refine(X_best, 1200, 150.0, 420.0, 0.0022, 0.0003, deadline=deadline_final - 8.0)
            best_val = _max_cos(X_best)
        except Exception:
            pass
    # trust-region LP polish with remaining time
    if time.time() < deadline_final - 3.0:
        try:
            X_best = _lp_polish(X_best, deadline_final - 1.0)
        except Exception:
            pass
    X_best = X_best / np.linalg.norm(X_best, axis=1, keepdims=True)
    # safety: replace non-finite
    if not np.all(np.isfinite(X_best)):
        # fallback to loaded best
        try:
            fb = np.load("best.npy")
            fb = fb / np.linalg.norm(fb, axis=1, keepdims=True)
            return fb
        except Exception:
            X_best = np.where(np.isfinite(X_best), X_best, 0.0)
            nr = np.linalg.norm(X_best, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            X_best = X_best / nr
    return X_best
