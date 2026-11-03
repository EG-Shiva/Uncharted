import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _quench(X, steps, b0, b1, lr, mu, rng=None, noise=0.0):
    n, d = X.shape
    V = np.zeros_like(X)
    lb0, lb1 = np.log(b0), np.log(b1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # softmax weights, avoid overflow: G-m <= 0
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        sw = float(W.sum())
        if not np.isfinite(sw) or sw <= 0:
            break
        W /= sw
        Gr = W @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        V = mu * V + Gt
        step = lr * V
        # retangent
        sdot = np.sum(step * X, axis=1, keepdims=True)
        step = step - sdot * X
        # clip
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        over = sn > 0.05
        step = np.where(over, step * (0.05 / sn), step)
        X = X - step
        if noise > 0:
            X = X + noise * (1.0 - f) * rng.standard_normal(X.shape) * 0.01
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X

def _hole_reinsert(X, victims, rng, n_cand=2000):
    n, d = X.shape
    keep = np.ones(n, dtype=bool)
    keep[victims] = False
    Y = X[keep]
    X2 = X.copy()
    # sequential greedy insertion using same candidate pool refreshed per victim
    for v in victims:
        C = rng.standard_normal((n_cand, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
        # dots to fixed points + already reinserted victims
        # build current set excluding v and not-yet-filled victims? use Y plus filled
        D = C @ Y.T
        worst = D.max(axis=1)
        bi = int(np.argmin(worst))
        X2[v] = C[bi]
        # add to Y for next victim in this batch
        Y = np.vstack([Y, C[bi][None, :]])
    return X2

def _lp_polish(X, seconds, deadline):
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    n, d = X.shape
    best = _max_cos(X)
    radius = 6e-4
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
        if itn > 35:
            break
        G = X @ X.T
        c = G[iu0]
        m = float(c.max())
        window = max(20.0 * radius, 5e-5)
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
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 3.0})
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
    DEADLINE = t0 + 165.0
    rng = np.random.default_rng(0)
    n, d = 58, 22
    base0 = None
    base_val = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            v = np.load(f).astype(float)
            if v.shape != (58, 22):
                continue
            v = v / np.linalg.norm(v, axis=1, keepdims=True)
            c = _max_cos(v)
            if c < base_val:
                base_val = c
                base0 = v.copy()
        except Exception:
            continue
    if base0 is None:
        z = rng.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        base0 = z
        base_val = _max_cos(base0)
    best = base0.copy()
    best_val = base_val
    orig = base0.copy()

    # contact degrees of incumbent for targeted victim choice
    def victims_for(X, k):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        cnt = (G > m - 0.004).sum(axis=1)
        mx = G.max(axis=1)
        # score: high count and high max -> overloaded
        order = np.argsort(-(cnt * 10 + mx * 100))
        nv = []
        # take 1-2 worst deterministically + random rest
        nworst = min(k, 2) if rng.random() < 0.8 else 0
        for i in range(nworst):
            nv.append(int(order[i]))
        while len(nv) < k:
            c = int(rng.integers(0, n))
            if c not in nv:
                nv.append(c)
        rng.shuffle(nv)
        return nv

    trial = 0
    # reserve ~28s for final LP
    while time.time() < DEADLINE - 28:
        trial += 1
        # choose base: mostly current best, sometimes original for diversity
        if trial % 7 == 0:
            src = orig
        else:
            src = best
        X = src.copy()
        mode = trial % 4
        if mode in (0, 1, 2):
            k = int(rng.integers(3, 9))  # 3..8
            if mode == 2:
                k = int(rng.integers(6, 12))
            vv = victims_for(X, k)
            # hole reinsert (cheap 1500 candidates)
            try:
                X = _hole_reinsert(X, vv, rng, n_cand=1500)
            except Exception:
                X[vv] = rng.standard_normal((len(vv), d)) / np.linalg.norm(rng.standard_normal((len(vv), d)), axis=1, keepdims=True)
            b0 = float(rng.choice([60.0, 100.0, 150.0]))
            b1 = float(rng.choice([600.0, 1200.0, 2500.0]))
            lr = float(rng.uniform(0.25, 0.7))
            mu = float(rng.uniform(0.80, 0.93))
            steps = int(rng.choice([700, 900, 1200]))
            X = _quench(X, steps, b0, b1, lr, mu)
            # second short hot quench focusing on max
            X = _quench(X, 150, 1500.0, 4000.0, lr * 0.5, 0.85)
        else:
            # global shake + quench
            sig = float(rng.choice([0.08, 0.15, 0.25]))
            X = X + sig * rng.standard_normal(X.shape)
            X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
            b0 = 80.0
            b1 = float(rng.choice([800.0, 1500.0]))
            lr = float(rng.uniform(0.3, 0.6))
            mu = 0.88
            X = _quench(X, 900, b0, b1, lr, mu)
        v = _max_cos(X)
        if v < best_val:
            best_val = v
            best = X.copy()
            # intensify: also quick extra quench
            if time.time() < DEADLINE - 35:
                X2 = _quench(X.copy(), 400, 400.0, 3000.0, 0.3, 0.9)
                v2 = _max_cos(X2)
                if v2 < best_val:
                    best_val = v2
                    best = X2.copy()
        # early exit if clearly new record by margin
        if best_val < 0.0655:
            break
    # final LP squeeze on best
    try:
        best = _lp_polish(best, 26.0, DEADLINE - 1.0)
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
