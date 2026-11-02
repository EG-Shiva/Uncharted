import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _quench(X, steps, b0, b1, lr0, lr1, mu=0.85, clip=0.04):
    n, d = X.shape
    V = np.zeros_like(X)
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        sw = float(W.sum())
        if not np.isfinite(sw) or sw <= 0:
            break
        W /= sw
        Gr = W @ X
        Gt = Gr - np.sum(Gr * X, axis=1, keepdims=True) * X
        V = mu * V + Gt
        step = lr * V
        step = step - np.sum(step * X, axis=1, keepdims=True) * X
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        over = sn[:, 0] > clip
        if np.any(over):
            step[over] *= (clip / sn[over])
        X = X - step
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X


def _best_hole(Y, rng, n_cand=5000, beta=600.0, it=60, lr=0.08):
    n, d = Y.shape
    C = rng.standard_normal((n_cand, d))
    C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    dots = C @ Y.T
    worst = dots.max(axis=1)
    bi = int(np.argmin(worst))
    c = C[bi].copy()
    # local refinement: minimise soft-max of dots
    for t in range(it):
        dd = Y @ c
        m = float(dd.max())
        w = np.exp(beta * (dd - m))
        sw = float(w.sum())
        if not np.isfinite(sw) or sw <= 0:
            break
        w /= sw
        g = w @ Y
        gt = g - float(g @ c) * c
        c = c - lr * gt
        nr = float(np.linalg.norm(c))
        if nr == 0:
            break
        c /= nr
        lr *= 0.985
        beta *= 1.02
    c /= (np.linalg.norm(c) + 1e-12)
    return c


def _lp_polish(X, seconds, deadline):
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    n, d = X.shape
    best = _max_cos(X)
    radius = 5e-4
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    itn = 0
    while True:
        if time.time() - start >= seconds:
            break
        if time.time() >= deadline:
            break
        if radius < 1e-13:
            break
        itn += 1
        if itn > 30:
            break
        G = X @ X.T
        c = G[iu0]
        m = float(c.max())
        window = max(20.0 * radius, 4e-5)
        order = np.argsort(-c)
        sel_list = []
        for idx in order:
            if c[idx] > m - window and len(sel_list) < 500:
                sel_list.append(int(idx))
            elif c[idx] <= m - window:
                break
            if len(sel_list) >= 500:
                break
        if len(sel_list) == 0:
            break
        sel = np.array(sel_list, dtype=int)
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(sel)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-12)
        v = _max_cos(y)
        if v < best - 1e-15:
            X = y
            best = v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X


def solve():
    t0 = time.time()
    DEADLINE = t0 + 170.0
    HOP_END = t0 + 138.0
    rng = np.random.default_rng(0)
    n, d = 58, 22

    base = None
    base_val = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            v = np.load(f).astype(float)
            if v.shape != (58, 22):
                continue
            v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
            c = _max_cos(v)
            if c < base_val:
                base_val = c
                base = v.copy()
        except Exception:
            continue
    if base is None:
        z = rng.standard_normal((n, d))
        z /= (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
        base = z
        base_val = _max_cos(base)

    gbest = base.copy()
    gval = base_val

    # contact degrees of incumbent for hub targeting
    def degrees(X, gap=0.003):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        cnt = (G > m - gap).sum(axis=1)
        return cnt, m

    trial = 0
    k_cycle = [5, 6, 4, 7, 3, 5, 8, 4]
    while time.time() < HOP_END and trial < 400:
        trial += 1
        k = k_cycle[(trial - 1) % len(k_cycle)]
        # refresh degrees every few trials from current best
        if trial % 3 == 1:
            cnt, mm = degrees(gbest)
            order = np.argsort(-cnt, kind="stable")
        # pick victims: mostly top-degree hubs + some randomness for diversity
        ntop = min(n, 14)
        pool = order[:ntop].tolist()
        # deterministic pseudo-random choice using rng
        perm = rng.permutation(np.array(pool))
        victims = sorted(perm[:k].tolist())
        # occasional random victim to diversify
        if trial % 7 == 0:
            victims = sorted(rng.choice(n, size=k, replace=False).tolist())

        keep = np.ones(n, dtype=bool)
        keep[victims] = False
        Xr = gbest[keep].copy()
        # small kick to subset to leave jammed basin
        kick = 0.004 + 0.004 * ((trial % 5) / 4.0)
        Xr = Xr + kick * rng.standard_normal(Xr.shape)
        Xr /= (np.linalg.norm(Xr, axis=1, keepdims=True) + 1e-12)
        # re-optimise sparse subset (has slack) quickly
        Xr = _quench(Xr, steps=260, b0=80.0, b1=900.0, lr0=0.035, lr1=0.006, mu=0.85, clip=0.04)

        # reinsert victims one by one into largest holes
        Xfull_list = [Xr[i] for i in range(Xr.shape[0])]
        Y = Xr.copy()
        rorder = rng.permutation(k)
        for _ii in rorder:
            nc = 4000 if trial % 2 == 0 else 6000
            c = _best_hole(Y, rng, n_cand=nc, beta=600.0, it=50, lr=0.08)
            Y = np.vstack([Y, c.reshape(1, -1)])
        # full quench with annealed beta
        Y = _quench(Y, steps=380, b0=150.0, b1=2500.0, lr0=0.03, lr1=0.004, mu=0.88, clip=0.035)
        # short hard-max refinement: a few greedy pushes via high-beta quench
        Y = _quench(Y, steps=80, b0=3000.0, b1=6000.0, lr0=0.006, lr1=0.002, mu=0.80, clip=0.02)
        v = _max_cos(Y)
        if v < gval - 1e-12:
            gval = v
            gbest = Y.copy()
            cnt, mm = degrees(gbest)
            order = np.argsort(-cnt, kind="stable")

        # interleaved pure-shake quench every 4 trials (cheap basin move)
        if trial % 4 == 0 and time.time() < HOP_END:
            sig = 0.006 if trial % 8 == 0 else 0.012
            Z = gbest + sig * rng.standard_normal(gbest.shape)
            Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
            Z = _quench(Z, steps=300, b0=120.0, b1=2200.0, lr0=0.025, lr1=0.004, mu=0.88, clip=0.035)
            v2 = _max_cos(Z)
            if v2 < gval - 1e-12:
                gval = v2
                gbest = Z.copy()

    # final trust-region LP polish
    try:
        gbest = _lp_polish(gbest, seconds=24.0, deadline=DEADLINE - 2.0)
        gval = _max_cos(gbest)
    except Exception:
        pass

    gbest = gbest / (np.linalg.norm(gbest, axis=1, keepdims=True) + 1e-12)
    # tiny shrink guard against boundary ties (no-op normalisation safe)
    return gbest
