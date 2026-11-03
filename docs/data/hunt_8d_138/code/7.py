import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _lp_polish(x, seconds, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline - 0.4 and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(np.max(c))
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        csel = c[sel]
        if p > 3500:
            keep = np.argpartition(csel, -3500)[-3500:]
            I = I[keep]
            J = J[keep]
            csel = csel[keep]
            p = 3500
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1
        ).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, n * d + 1),
            ).tocsr()
            obj = np.zeros(n * d + 1)
            obj[-1] = 1.0
            res = linprog(
                obj, A_ub=A_ub, b_ub=-csel,
                A_eq=A_eq, b_eq=np.zeros(n),
                bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                method="highs",
            )
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            yn = np.linalg.norm(y, axis=1, keepdims=True)
            yn = np.maximum(yn, 1e-12)
            y = y / yn
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _anneal(X0, steps, T0, T1, lr0, lr1, mom):
    X = X0.copy()
    n, d = X.shape
    V = np.zeros_like(X)
    lT0 = np.log(T0)
    lT1 = np.log(T1)
    ll0 = np.log(lr0)
    ll1 = np.log(lr1)
    denom = max(steps - 1, 1)
    max_step = 0.12
    for s in range(steps):
        frac = s / denom
        T = float(np.exp(lT0 + frac * (lT1 - lT0)))
        lr = float(np.exp(ll0 + frac * (ll1 - ll0)))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = np.max(G, axis=1, keepdims=True)
        E = np.exp((G - M) / T)
        np.fill_diagonal(E, 0.0)
        S = np.sum(E, axis=1, keepdims=True) + 1e-300
        P = E / S
        W = P @ X
        dot = np.sum(W * X, axis=1, keepdims=True)
        Gt = W - dot * X
        V = mom * V - lr * Gt
        vn = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
        sc = np.minimum(1.0, max_step / vn)
        V = V * sc
        X = X + V
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X /= nr
    return X


def _contact_counts(X, gap=0.002):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    C = (G > m - gap).sum(axis=1)
    return C, m


def solve():
    np.random.seed(0)
    t_start = time.time()
    deadline = t_start + 172.0
    n_dim = 8
    n_pts = 138
    try:
        Xbest = np.load("best.npy").astype(float)
    except Exception:
        try:
            Xbest = np.load("record.npy").astype(float)
        except Exception:
            rng0 = np.random.default_rng(0)
            Xbest = rng0.normal(size=(n_pts, n_dim))
    if Xbest.shape != (n_pts, n_dim):
        rng0 = np.random.default_rng(0)
        Xbest = rng0.normal(size=(n_pts, n_dim))
    Xbest = Xbest / np.maximum(np.linalg.norm(Xbest, axis=1, keepdims=True), 1e-12)
    best_val = _max_cos(Xbest)
    rng = np.random.default_rng(7)
    cycle = 0
    # quick initial LP to ensure tight baseline if time allows (short)
    # main basin-hopping loop
    while time.time() < deadline - 12.0:
        cycle += 1
        remaining = deadline - 12.0 - time.time()
        if remaining < 0.5:
            break
        # sample schedule
        sigma = float(10.0 ** rng.uniform(-2.3, -0.5))
        n_replace = int(rng.integers(0, 10))
        T0 = float(10.0 ** rng.uniform(-1.6, -1.0))
        T1 = float(10.0 ** rng.uniform(-2.6, -2.15))
        lr0 = float(10.0 ** rng.uniform(-2.3, -1.5))
        lr1 = lr0 * 0.15
        mom = float(rng.uniform(0.80, 0.92))
        steps = int(rng.integers(1500, 3500))
        if cycle % 8 == 0:
            # fresh random basin
            Xr = rng.normal(size=(n_pts, n_dim))
            Xr /= np.maximum(np.linalg.norm(Xr, axis=1, keepdims=True), 1e-12)
            Xinit = Xr
        else:
            # perturb best, bias to overloaded points
            try:
                cnt, _m = _contact_counts(Xbest)
                cmax = float(np.max(cnt)) + 1e-12
                w = 0.5 + 1.5 * (cnt.astype(float) / cmax)
            except Exception:
                w = np.ones(n_pts)
            Xinit = Xbest.copy()
            if n_replace > 0:
                idx = rng.choice(n_pts, size=n_replace, replace=False)
                R = rng.normal(size=(n_replace, n_dim))
                R /= np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
                Xinit[idx] = R
            if sigma > 0:
                N = rng.normal(size=(n_pts, n_dim)) * (sigma * w[:, None])
                Xinit = Xinit + N
                Xinit /= np.maximum(np.linalg.norm(Xinit, axis=1, keepdims=True), 1e-12)
        try:
            Xa = _anneal(Xinit, steps, T0, T1, lr0, lr1, mom)
            va = _max_cos(Xa)
        except Exception:
            continue
        # only LP-polish promising candidates
        if va > best_val + 0.015:
            # still allow occasional polish of worse to escape? skip mostly
            if rng.random() > 0.05:
                continue
        try:
            Xp = _lp_polish(Xa, seconds=1.6, deadline=deadline - 10.0)
            vp = _max_cos(Xp)
        except Exception:
            continue
        if vp < best_val:
            Xbest = Xp
            best_val = vp
            # early exit if big win? continue to squeeze more
        if time.time() >= deadline - 12.0:
            break
    # final intensive polish of best with remaining time
    try:
        tleft = deadline - time.time() - 1.0
        if tleft > 2.0:
            tpol = min(tleft, 11.0)
            Xbest = _lp_polish(Xbest, seconds=tpol, deadline=deadline)
    except Exception:
        pass
    Xbest = Xbest / np.maximum(np.linalg.norm(Xbest, axis=1, keepdims=True), 1e-12)
    # guard NaN
    Xbest = np.nan_to_num(Xbest, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(Xbest, axis=1, keepdims=True)
    bad = (nr.squeeze() < 1e-9)
    if np.any(bad):
        rngf = np.random.default_rng(1)
        R = rngf.normal(size=(int(np.sum(bad)), n_dim))
        R /= np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
        Xbest[bad] = R
        Xbest = Xbest / np.maximum(np.linalg.norm(Xbest, axis=1, keepdims=True), 1e-12)
    return Xbest
