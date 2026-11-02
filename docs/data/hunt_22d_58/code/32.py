import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _quench(X, steps, b0, b1, lr0, lr1, mu=0.85, clip=0.03):
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
        Gr = (W @ X) / sw
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        V = mu * V + Gt
        step = lr * V
        # retangent + clip
        sdot = np.sum(step * X, axis=1, keepdims=True)
        step = step - sdot * X
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        over = sn > clip
        # avoid in-place broadcast issues
        step = np.where(over, step * (clip / sn), step)
        X = X - step
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X


def _hole_reinsert(X, victim, rng, n_cand=2000):
    n, d = X.shape
    mask = np.ones(n, dtype=bool)
    mask[victim] = False
    Y = X[mask]
    C = rng.standard_normal((n_cand, d))
    C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    dots = C @ Y.T
    worst = dots.max(axis=1)
    bi = int(np.argmin(worst))
    X2 = X.copy()
    X2[victim] = C[bi]
    return X2


def _lp_finish(X, seconds, deadline):
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    n, d = X.shape
    best = _max_cos(X)
    radius = 5e-4
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    it = 0
    while True:
        if time.time() - start >= seconds:
            break
        if time.time() >= deadline:
            break
        if radius < 1e-13:
            break
        it += 1
        if it > 25:
            break
        G = X @ X.T
        c = G[iu0]
        m = float(c.max())
        window = max(20.0 * radius, 3e-5)
        order = np.argsort(-c)
        sel_list = []
        for idx in order:
            if c[idx] > m - window and len(sel_list) < 600:
                sel_list.append(int(idx))
            elif c[idx] <= m - window:
                break
            if len(sel_list) >= 600:
                break
        if len(sel_list) == 0:
            break
        sel = np.array(sel_list)
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
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 2.0})
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
            radius = min(radius * 1.5, 0.015)
        else:
            radius *= 0.5
    return X


def solve():
    t0 = time.time()
    DEADLINE = t0 + 165.0
    rng = np.random.default_rng(12345)
    n, d = 58, 22

    best = None
    best_val = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            v = np.load(f).astype(float)
            if v.shape != (n, d):
                continue
            v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
            c = _max_cos(v)
            if c < best_val:
                best_val = c
                best = v.copy()
        except Exception:
            continue
    if best is None:
        z = rng.standard_normal((n, d))
        z /= (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
        best = z
        best_val = _max_cos(best)

    bestX = best.copy()

    # fast quench params presets
    trial = 0
    # reserve ~12s for final LP
    while time.time() < DEADLINE - 14.0:
        trial += 1
        remaining = DEADLINE - 14.0 - time.time()
        if remaining <= 0:
            break
        mode = trial % 5
        X = bestX.copy()
        if mode == 0:
            # small global shake
            sig = 0.025 if (trial % 2 == 0) else 0.06
            X = X + sig * rng.standard_normal(X.shape)
            X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            steps = 700
            X = _quench(X, steps, 100, 1200, 0.08, 0.008, 0.85, 0.03)
        elif mode == 1:
            # subset random replacement (topology break)
            k = int(rng.integers(3, 7))
            idx = rng.choice(n, size=k, replace=False)
            R = rng.standard_normal((k, d))
            R /= (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
            X[idx] = R
            steps = 1200
            X = _quench(X, steps, 80, 2000, 0.10, 0.006, 0.88, 0.035)
        elif mode == 2:
            # worst-pair hub -> hole
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            flat = int(np.argmax(G))
            ii = flat // n
            jj = flat % n
            # degree with gap to pick victim
            m0 = float(G.max())
            cnt = (G > m0 - 0.004).sum(axis=1)
            victim = ii if cnt[ii] >= cnt[jj] else jj
            X = _hole_reinsert(X, victim, rng, n_cand=2500)
            # small shake of neighbours to allow rearrangement
            X = X + 0.01 * rng.standard_normal(X.shape)
            X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            steps = 1000
            X = _quench(X, steps, 120, 2500, 0.08, 0.005, 0.85, 0.03)
        elif mode == 3:
            # overloaded hub rewiring
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m0 = float(G.max())
            cnt = (G > m0 - 0.004).sum(axis=1)
            victim = int(np.argmax(cnt + rng.random(n) * 0.5))
            X = _hole_reinsert(X, victim, rng, n_cand=2500)
            steps = 1000
            X = _quench(X, steps, 90, 1800, 0.09, 0.006, 0.86, 0.032)
        else:
            # fresh random quench (different basin) occasionally
            if trial % 10 == 0:
                Z = rng.standard_normal((n, d))
                Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
                X = Z
                steps = 1800
                X = _quench(X, steps, 60, 1500, 0.12, 0.008, 0.88, 0.04)
            else:
                # medium shake of random subset
                k = int(rng.integers(6, 12))
                idx = rng.choice(n, size=k, replace=False)
                X[idx] = X[idx] + 0.35 * rng.standard_normal((k, d))
                X[idx] /= (np.linalg.norm(X[idx], axis=1, keepdims=True) + 1e-12)
                steps = 900
                X = _quench(X, steps, 100, 2000, 0.09, 0.006, 0.85, 0.03)

        v = _max_cos(X)
        if v < best_val - 1e-12:
            best_val = v
            bestX = X.copy()
        # early stop if clearly improved a lot? continue to polish further
        if time.time() >= DEADLINE - 14.0:
            break

    # final intensive quench on champion with high beta
    try:
        if time.time() < DEADLINE - 11.0:
            bestX = _quench(bestX.copy(), 800, 200, 4000, 0.04, 0.003, 0.85, 0.02)
            v = _max_cos(bestX)
            if v < best_val:
                best_val = v
    except Exception:
        pass

    # capped LP finish
    try:
        if time.time() < DEADLINE - 2.0:
            secs = max(1.0, min(9.0, DEADLINE - 2.0 - time.time()))
            Xlp = _lp_finish(bestX.copy(), secs, DEADLINE - 1.0)
            v = _max_cos(Xlp)
            if v < best_val:
                best_val = v
                bestX = Xlp
    except Exception:
        pass

    bestX = bestX / (np.linalg.norm(bestX, axis=1, keepdims=True) + 1e-12)
    # sanitize
    bestX = np.where(np.isfinite(bestX), bestX, rng.standard_normal((n, d)))
    bestX = bestX / (np.linalg.norm(bestX, axis=1, keepdims=True) + 1e-12)
    return bestX
