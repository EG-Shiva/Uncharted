import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def _gd_smooth(X0, beta, steps, lr, mu=0.86):
    X = X0.copy()
    V = np.zeros_like(X)
    for it in range(steps):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        D = beta * (G - m)
        np.clip(D, -30.0, 0.0, out=D)
        W = np.exp(D)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        mean = (W @ X) / rs
        dot = np.sum(mean * X, axis=1, keepdims=True)
        tang = mean - dot * X
        V = mu * V + tang
        cur_lr = lr * (1.0 - 0.4 * it / max(1, steps))
        X = X - cur_lr * V
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        np.maximum(nr, 1e-12, out=nr)
        X /= nr
    return X


def _gd_riesz(X0, s_pow, steps, lr, mu=0.85):
    X = X0.copy()
    V = np.zeros_like(X)
    expo = -s_pow * 0.5 - 1.0
    for it in range(steps):
        G = X @ X.T
        np.fill_diagonal(G, 1.0 - 1e-9)
        np.clip(G, -1.0, 0.999999, out=G)
        D = 2.0 - 2.0 * G
        np.maximum(D, 1e-6, out=D)
        # W = D^{expo}
        W = np.power(D, expo)
        np.fill_diagonal(W, 0.0)
        # normalize rows to avoid scale blowup
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        mean = (W @ X) / rs
        dot = np.sum(mean * X, axis=1, keepdims=True)
        tang = mean - dot * X
        # clip extreme tang
        V = mu * V + tang
        cur_lr = lr * (1.0 - 0.4 * it / max(1, steps))
        X = X - cur_lr * V
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        np.maximum(nr, 1e-12, out=nr)
        X /= nr
        if not np.all(np.isfinite(X)):
            return X0.copy()
    return X


def _worst_reinsert(Y, k, Nc, rng):
    n, d = Y.shape
    G = Y @ Y.T
    np.fill_diagonal(G, -2.0)
    rm = G.max(axis=1)
    order = np.argsort(-rm)
    victims = [int(v) for v in order[:k]]
    for w in victims:
        cand = rng.standard_normal((Nc, d))
        cand /= np.maximum(np.linalg.norm(cand, axis=1, keepdims=True), 1e-12)
        sc = cand @ Y.T
        sc[:, w] = -2.0
        mx = sc.max(axis=1)
        bi = int(np.argmin(mx))
        Y[w] = cand[bi]
    return Y


def _lp_polish(x, best, deadline, k_top=4000, r0=2e-3):
    n, d = x.shape
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    try:
        bv = _max_cos(x)
    except Exception:
        return x, best
    if bv < best:
        best = bv
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    ar = np.arange(d)
    it = 0
    while time.time() < deadline - 1.0 and radius > 1e-13 and it < 80:
        it += 1
        Gall = x @ x.T
        c_all = Gall[iu0, iu1]
        m = float(np.max(c_all))
        window = max(20.0 * radius, 5e-7)
        sel = c_all > m - window
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > k_top:
            idx = np.argpartition(-c_all, k_top)[:k_top]
            I = iu0[idx]; J = iu1[idx]; c_sel = c_all[idx]
        else:
            I = iu0[sel]; J = iu1[sel]; c_sel = c_all[sel]
        p = I.shape[0]
        nv = n * d + 1
        try:
            cols = np.concatenate([I[:, None] * d + ar,
                                   J[:, None] * d + ar,
                                   np.full((p, 1), n * d)], axis=1).ravel()
            rows = np.repeat(np.arange(p), 2 * d + 1)
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            bounds = [(-radius, radius)] * (n * d) + [(None, float(m))]
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq,
                          b_eq=np.zeros(n), bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best - 1e-12:
            x = y; best = v
            radius = min(radius * 1.5, 0.012)
        else:
            radius *= 0.5
    return x, best


def solve():
    t_start = time.time()
    BUDGET = 166.0
    def remaining():
        return BUDGET - (time.time() - t_start)
    rng = np.random.default_rng(0)
    n, d = 250, 6
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            z = np.asarray(np.load(fn), dtype=np.float64)
            if z.shape == (n, d):
                z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
                cands.append(z)
        except Exception:
            pass
    if not cands:
        z = rng.standard_normal((n, d))
        z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
        cands.append(z)
    elite = min(cands, key=_max_cos).copy()
    elite_val = _max_cos(elite)
    pool = [elite.copy()]

    # Riesz diversified starts (global energy finds different shell topology)
    for si in range(6):
        if remaining() < 120:
            break
        try:
            X0 = rng.standard_normal((n, d))
            X0 /= np.maximum(np.linalg.norm(X0, axis=1, keepdims=True), 1e-12)
            s_pow = 14.0 + 2.0 * (si % 3)
            Xr = _gd_riesz(X0, s_pow, 220, 0.006)
            X1 = _gd_smooth(Xr, 1200.0, 280, 0.009)
            X2 = _gd_smooth(X1, 3500.0, 220, 0.004)
            v2 = _max_cos(X2)
            pool.append(X2.copy())
            if v2 < elite_val:
                elite = X2.copy(); elite_val = v2
        except Exception:
            continue

    # quick smooth starts to enrich pool
    for si in range(4):
        if remaining() < 110:
            break
        try:
            X0 = rng.standard_normal((n, d))
            X0 /= np.maximum(np.linalg.norm(X0, axis=1, keepdims=True), 1e-12)
            X1 = _gd_smooth(X0, 1000.0, 300, 0.011)
            X2 = _gd_smooth(X1, 4000.0, 250, 0.004)
            v2 = _max_cos(X2)
            pool.append(X2.copy())
            if v2 < elite_val:
                elite = X2.copy(); elite_val = v2
        except Exception:
            continue

    try:
        pool_sorted = sorted(pool, key=_max_cos)[:4]
    except Exception:
        pool_sorted = [elite.copy()]
    cur = elite.copy(); cur_val = elite_val
    trial = 0
    # main basin hopping: lower whole shell, not single pair
    while remaining() > 32 and trial < 90:
        trial += 1
        try:
            r = rng.random()
            if r < 0.5:
                base = cur
            elif r < 0.78:
                base = elite
            else:
                base = pool_sorted[int(rng.integers(0, len(pool_sorted)))]
            Y = base.copy()
            mode = trial % 5
            if mode == 0:
                Y += rng.standard_normal(Y.shape) * 0.009
                Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
                Y = _gd_smooth(Y, 1300.0, 280, 0.008)
                Y = _gd_smooth(Y, 5000.0, 200, 0.003)
            elif mode == 1:
                krep = 5 + (trial % 4)
                Y = _worst_reinsert(Y, krep, 3500, rng)
                Y += rng.standard_normal(Y.shape) * 0.0007
                Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
                Y = _gd_smooth(Y, 1600.0, 280, 0.007)
                Y = _gd_smooth(Y, 7000.0, 180, 0.0025)
            elif mode == 2:
                Y += rng.standard_normal(Y.shape) * 0.006
                Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
                Y = _gd_riesz(Y, 18.0, 140, 0.004)
                Y = _gd_smooth(Y, 2500.0, 260, 0.005)
            elif mode == 3:
                Y += rng.standard_normal(Y.shape) * 0.022
                Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
                Y = _gd_smooth(Y, 900.0, 300, 0.010)
                Y = _gd_smooth(Y, 3500.0, 220, 0.004)
            else:
                idx = rng.choice(n, 10, replace=False)
                R = rng.standard_normal((10, d))
                R /= np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
                Y[idx] = R
                Y += rng.standard_normal(Y.shape) * 0.0006
                Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
                Y = _gd_smooth(Y, 1400.0, 300, 0.008)
            v = _max_cos(Y)
            if v < elite_val - 1e-12:
                elite = Y.copy(); elite_val = v
                cur = Y.copy(); cur_val = v
                try:
                    pool_sorted = sorted(pool_sorted + [Y.copy()], key=_max_cos)[:4]
                except Exception:
                    pass
            elif v < cur_val:
                cur = Y.copy(); cur_val = v
            elif v < cur_val + 2e-4 and rng.random() < 0.22:
                cur = Y.copy(); cur_val = v
            if cur_val > elite_val + 5e-4 and rng.random() < 0.35:
                cur = elite.copy(); cur_val = elite_val
        except Exception:
            continue

    if remaining() > 10:
        try:
            nxt = _gd_smooth(elite, 9000.0, 350, 0.0022)
            v = _max_cos(nxt)
            if v < elite_val:
                elite = nxt; elite_val = v
            nxt2 = _gd_smooth(elite, 15000.0, 250, 0.0015)
            v2 = _max_cos(nxt2)
            if v2 < elite_val:
                elite = nxt2; elite_val = v2
        except Exception:
            pass
    if remaining() > 6:
        try:
            deadline = time.time() + remaining() - 1.0
            polished, pv = _lp_polish(elite, elite_val, deadline)
            if pv < elite_val:
                elite = polished; elite_val = pv
        except Exception:
            pass
    elite = elite / np.maximum(np.linalg.norm(elite, axis=1, keepdims=True), 1e-12)
    return elite
