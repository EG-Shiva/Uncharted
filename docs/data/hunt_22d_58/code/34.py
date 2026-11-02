import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _pump(X, t, steps, lr, beta0, beta1):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lb0, lb1 = np.log(beta0), np.log(beta1)
    thr = t - 0.01
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        G = X @ X.T
        Act = G > thr
        np.fill_diagonal(Act, False)
        D = G - t
        zc = np.clip(beta * D, -30, 30)
        Sig = 1.0 / (1.0 + np.exp(-zc))
        SP = np.logaddexp(0, zc) / beta
        SP[~Act] = 0.0
        Sig[~Act] = 0.0
        Fp = SP * Sig
        np.fill_diagonal(Fp, 0.0)
        Grad = Fp @ X
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Rg = Grad - X * dot
        m = b1 * m + (1 - b1) * Rg
        v = b2 * v + (1 - b2) * (Rg * Rg)
        mhat = m / (1 - b1 ** (s + 2))
        vhat = v / (1 - b2 ** (s + 2))
        Step = mhat / (np.sqrt(vhat) + eps)
        sdot = (Step * X).sum(axis=1, keepdims=True)
        Step = Step - X * sdot
        X = X - lr * Step
        X = _norm(X)
    return X

def _softmax_quench(X, steps, lr0, lr1, b0, b1, mu=0.85, clip=0.05):
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
        W = np.exp(np.clip(beta * (G - m), -30, 0))
        np.fill_diagonal(W, 0.0)
        sw = float(W.sum())
        if not np.isfinite(sw) or sw <= 0:
            break
        W /= sw
        Gr = W @ X
        dot = (Gr * X).sum(axis=1, keepdims=True)
        Gt = Gr - dot * X
        V = mu * V + Gt
        step = lr * V
        sdot = (step * X).sum(axis=1, keepdims=True)
        step = step - sdot * X
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        step = np.where(sn > clip, step * (clip / sn), step)
        X = _norm(X - step)
    return X

def _lp_polish(X, deadline, max_it=12, cap=400):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return X, _max_cos(X)
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    iu = np.triu_indices(n, 1)
    radius = 8e-4
    it = 0
    while time.time() < deadline and it < max_it and radius > 1e-13:
        it += 1
        if time.time() >= deadline:
            break
        Gfull = (X @ X.T)[iu]
        m = float(Gfull.max())
        window = max(20.0 * radius, 5e-5)
        order = np.argsort(-Gfull)
        sel = []
        for idx in order:
            if Gfull[idx] > m - window and len(sel) < cap:
                sel.append(int(idx))
            elif Gfull[idx] <= m - window:
                break
            if len(sel) >= cap:
                break
        if len(sel) == 0:
            break
        sel = np.array(sel)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(sel), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-Gfull[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm(X + res.x[:-1].reshape(n, d))
        v = _max_cos(y)
        if v < best - 1e-12:
            X, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X, best

def solve():
    T0 = time.time()
    DEAD = time.time() + 150.0
    rng = np.random.default_rng(1)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            c = np.load(fn).astype(float)
            if c.shape == (58, 22):
                c = _norm(np.where(np.isfinite(c), c, 0.0))
                cands.append(c)
        except Exception:
            pass
    if not cands:
        cands.append(_norm(rng.standard_normal((58, 22))))
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    incumbent = cands[bi].copy()
    best = incumbent.copy()
    best_val = float(vals[bi])

    # multi-kick pump + quench trials, cheap
    kicks = [0.003, 0.006, 0.010, 0.016, 0.003, 0.008, 0.013, 0.020]
    for k, kick in enumerate(kicks):
        if time.time() > DEAD - 25:
            break
        t = best_val - (0.00025 + 0.00005 * (k % 3))
        X = _norm(incumbent + kick * rng.standard_normal(incumbent.shape))
        # fast quench then lexicographic pump
        X = _softmax_quench(X, steps=350, lr0=0.006, lr1=0.0015, b0=200.0, b1=1200.0)
        if time.time() > DEAD - 25:
            v = _max_cos(X)
            if v < best_val:
                best, best_val = X, v
            continue
        X = _pump(X, t, steps=500, lr=0.004, beta0=250.0, beta1=1800.0)
        v = _max_cos(X)
        if v < best_val - 1e-12:
            best, best_val = X, v
            incumbent = best.copy()
        elif v < best_val + 3e-4:
            # check de-jamming: degree at v-5e-4
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            deg = (G > v - 5e-4).sum(axis=1)
            if int(deg.min()) <= 6:
                # transplant low-degree point into incumbent and retest quickly
                j = int(np.argmin(deg))
                Xt = incumbent.copy()
                Xt[j] = X[j]
                Xt = _norm(Xt)
                Xt = _pump(Xt, t, steps=200, lr=0.002, beta0=400.0, beta1=1200.0)
                vt = _max_cos(Xt)
                if vt < best_val - 1e-12:
                    best, best_val = Xt, vt
                    incumbent = best.copy()

    # hill-climb tiny tangent perturbations on best (fast, helps shave 1e-7)
    if time.time() < DEAD - 22:
        for _ in range(60):
            if time.time() > DEAD - 20:
                break
            sig = 0.0015
            Y = _norm(best + sig * rng.standard_normal(best.shape))
            # one short quench
            Y = _softmax_quench(Y, steps=120, lr0=0.003, lr1=0.001, b0=400.0, b1=1500.0)
            v = _max_cos(Y)
            if v < best_val - 1e-12:
                best, best_val = Y, v
                incumbent = best.copy()

    # final capped LP polish
    remain_dead = min(DEAD, time.time() + 14.0)
    if time.time() < remain_dead - 3:
        try:
            Xf, vf = _lp_polish(best, deadline=remain_dead - 1.0, max_it=12, cap=400)
            if vf < best_val:
                best, best_val = Xf, vf
        except Exception:
            pass

    best = _norm(best)
    # safety: never return worse than loaded best
    try:
        bl = _norm(np.load("best.npy").astype(float))
        if _max_cos(bl) < _max_cos(best):
            return bl
    except Exception:
        pass
    return best
