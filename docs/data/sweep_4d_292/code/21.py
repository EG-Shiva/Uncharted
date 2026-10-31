import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _anneal(X, steps, b0, b1, lr0, mu):
    n, d = X.shape
    v = np.zeros_like(X)
    lb = np.log(b1 / b0) if b1 > b0 else 0.0
    for t in range(steps):
        frac = t / max(1, steps - 1)
        beta = b0 * np.exp(lb * frac)
        lr = lr0 * (1.0 - 0.5 * frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-300
        A = (W @ X) / S
        dot = (A * X).sum(axis=1, keepdims=True)
        Gt = A - dot * X
        v = mu * v + Gt
        vd = (v * X).sum(axis=1, keepdims=True)
        v = v - vd * X
        X = X - lr * v
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        np.maximum(nr, 1e-12, out=nr)
        X /= nr
    return X

def _farthest_reinsert(X_rem, k, rng, n_cand=20000):
    # sequential greedy farthest-hole insertion
    cand = rng.standard_normal((n_cand, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    cur = X_rem
    for _ in range(k):
        # compute in chunks to limit memory
        best_val = 2.0
        best_pt = cand[0]
        for a in range(0, n_cand, 5000):
            cc = cand[a:a+5000]
            dd = cc @ cur.T
            mm = dd.max(axis=1)
            j = int(np.argmin(mm))
            if float(mm[j]) < best_val:
                best_val = float(mm[j])
                best_pt = cc[j]
        # small jitter to avoid exact ties
        jp = best_pt + 0.005 * rng.standard_normal(4)
        jp /= max(1e-12, np.linalg.norm(jp))
        cur = np.vstack([cur, jp.reshape(1, 4)])
    return cur

def _lp_polish(x, deadline):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return x
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
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
            obj = np.zeros(nv); obj[-1] = 1.0
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
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    PHASE1_END = t0 + 132.0
    rng = np.random.default_rng(42)
    n, d = 292, 4
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (n, d):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)) and float(nr.min()) > 1e-9:
                    cands.append(a / nr)
        except Exception:
            pass
    if not cands:
        X0 = rng.standard_normal((n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands.append(X0)
    vals = [_max_cos(c) for c in cands]
    x_best = cands[int(np.argmin(vals))].copy()
    best_val = float(min(vals))

    presets = [
        (900, 25.0, 400.0, 0.08, 0.85),
        (1200, 20.0, 800.0, 0.06, 0.90),
        (600, 50.0, 1000.0, 0.05, 0.82),
        (350, 100.0, 600.0, 0.035, 0.85),
        (1500, 15.0, 500.0, 0.07, 0.88),
    ]

    it = 0
    while time.time() < PHASE1_END:
        it += 1
        if DEADLINE - time.time() < 5:
            break
        # time for at least a short anneal?
        if time.time() + 0.6 > PHASE1_END and it > 10:
            break
        X = x_best.copy()
        r = rng.random()
        # choose kick strength
        if r < 0.55:
            # worst-point excision + farthest-hole reinsertion (topology change)
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            k = int(rng.integers(2, 10))  # 2..9
            # occasionally larger excision
            if rng.random() < 0.15:
                k = int(rng.integers(10, 18))
            worst = np.argsort(rowmax)[-k:]
            mask = np.ones(n, dtype=bool)
            mask[worst] = False
            X_rem = X[mask]
            # small shake of survivors to diversify
            sig = float(rng.choice([0.005, 0.01, 0.02]))
            X_rem = X_rem + sig * rng.standard_normal(X_rem.shape)
            X_rem /= np.linalg.norm(X_rem, axis=1, keepdims=True)
            X = _farthest_reinsert(X_rem, k, rng, n_cand=20000)
        elif r < 0.80:
            # Gaussian shake + random kill-reinsert
            sig = float(rng.choice([0.008, 0.02, 0.05, 0.12, 0.25]))
            X = X + sig * rng.standard_normal(X.shape)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            if rng.random() < 0.7:
                k = int(rng.integers(1, 8))
                idx = rng.choice(n, size=k, replace=False)
                R = rng.standard_normal((k, d))
                R /= np.linalg.norm(R, axis=1, keepdims=True)
                X[idx] = R
        else:
            # heavy shake / partial restart
            if rng.random() < 0.3:
                R = rng.standard_normal((n, d))
                R /= np.linalg.norm(R, axis=1, keepdims=True)
                if rng.random() < 0.5:
                    X = R
                else:
                    X = X + 0.6 * rng.standard_normal((n, d))
                    X /= np.linalg.norm(X, axis=1, keepdims=True)
            else:
                sig = 0.30
                X = X + sig * rng.standard_normal(X.shape)
                X /= np.linalg.norm(X, axis=1, keepdims=True)
                k = int(rng.integers(4, 12))
                idx = rng.choice(n, size=k, replace=False)
                R = rng.standard_normal((k, d))
                R /= np.linalg.norm(R, axis=1, keepdims=True)
                X[idx] = R

        pr = presets[rng.integers(len(presets))]
        # adapt steps to remaining time
        remain = PHASE1_END - time.time()
        steps = pr[0]
        if remain < 3 and steps > 400:
            steps = 400
        try:
            X = _anneal(X, steps, pr[1], pr[2], pr[3], pr[4])
        except Exception:
            continue
        try:
            v = _max_cos(X)
        except Exception:
            continue
        if v < best_val:
            best_val = v
            x_best = X.copy()
            # early exit if big win? keep going to polish further
        # opportunistic: if time very low break
        if time.time() > PHASE1_END:
            break

    # high-beta squeeze on best
    try:
        if time.time() < DEADLINE - 35:
            x_best = _anneal(x_best, 1200, 200.0, 2500.0, 0.02, 0.9)
            bv = _max_cos(x_best)
            if bv < best_val:
                best_val = bv
            else:
                # keep previous if squeeze worsened (anneal is deterministic move, may slightly worsen max but improve softmax)
                pass
    except Exception:
        pass
    # final trust-region LP polish
    try:
        lp_deadline = DEADLINE - 1.0
        # reserve at least 10s for LP if possible
        if time.time() < lp_deadline:
            x_best = _lp_polish(x_best, lp_deadline)
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
