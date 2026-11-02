import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, seconds, deadline):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 6e-4
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
        if it > 30:
            break
        G = X @ X.T
        c = G[iu0]
        m = float(c.max())
        window = max(20.0 * radius, 4e-5)
        order = np.argsort(-c)
        sel_list = []
        for idx in order:
            if c[idx] > m - window:
                sel_list.append(int(idx))
                if len(sel_list) >= 650:
                    break
            else:
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
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y = y / nr
        v = _max_cos(y)
        if v < best - 1e-15:
            X = y
            best = v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X

def _quench(X, steps, b0, b1, lr0, lr1, mu=0.8, clip=0.05):
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
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        V = mu * V + Gt
        step = lr * V
        sdot = np.sum(step * X, axis=1, keepdims=True)
        step = step - sdot * X
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        over = sn > clip
        step = np.where(over, step * (clip / sn), step)
        X = X - step
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X

def _refine_single(y, Y, iters=60):
    # minimize max(Y @ y) with Y fixed, |y|=1
    y = y / (np.linalg.norm(y) + 1e-12)
    for s in range(iters):
        f = s / max(1, iters - 1)
        beta = 400.0 + 1200.0 * f
        lr = 0.05 * (0.1) ** f  # 0.05 -> 0.005
        dots = Y @ y
        m = float(dots.max())
        w = np.exp(beta * (dots - m))
        sw = float(w.sum())
        if not np.isfinite(sw) or sw <= 0:
            break
        w /= sw
        g = w @ Y
        gt = g - float(g @ y) * y
        y = y - lr * gt
        nrm = float(np.linalg.norm(y))
        if nrm == 0 or not np.isfinite(nrm):
            break
        y = y / nrm
    return y

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    TRIAL_END = t0 + 130.0
    FINAL_BUDGET = 20.0
    rng = np.random.default_rng(12345)
    n, d = 58, 22
    best = None
    best_val = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            v = np.load(f).astype(float)
            if v.shape != (58, 22):
                continue
            v = v / np.linalg.norm(v, axis=1, keepdims=True)
            c = _max_cos(v)
            if c < best_val:
                best_val = c
                best = v.copy()
        except Exception:
            continue
    if best is None:
        z = rng.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        best = z
        best_val = _max_cos(best)
    incumbent = best.copy()
    incumb_val = float(best_val)
    Ks = [3, 4, 5, 6, 8]
    trial = 0
    NCAND = 8192
    while time.time() < TRIAL_END and time.time() < HARD_END - 22.0:
        trial += 1
        try:
            k = Ks[int(rng.integers(0, len(Ks)))]
            mode = trial % 3  # 0 random, 1 hubs, 2 cluster
            G = incumbent @ incumbent.T
            np.fill_diagonal(G, -2.0)
            m0 = float(G.max())
            if mode == 1:
                gap = 0.004
                deg = (G > m0 - gap).sum(axis=1)
                rem = np.argsort(-deg)[:k]
                # add a little randomness: sometimes swap last one
                if rng.random() < 0.5:
                    rem = rem.copy()
                    rem[-1] = int(rng.integers(0, n))
                    rem = np.unique(rem)
                    while len(rem) < k:
                        c2 = int(rng.integers(0, n))
                        if c2 not in rem:
                            rem = np.append(rem, c2)
            elif mode == 2:
                seed = int(rng.integers(0, n))
                row = G[seed].copy()
                row[seed] = -2.0
                nn = np.argsort(-row)[:k-1]
                rem = np.concatenate([[seed], nn]).astype(int)
            else:
                rem = rng.choice(n, size=k, replace=False)
            rem = np.array(sorted(set(int(a) for a in rem)), dtype=int)
            if len(rem) != k:
                # fill up
                extra = [i for i in range(n) if i not in set(rem.tolist())]
                rng.shuffle(extra)
                need = k - len(rem)
                rem = np.array(sorted(rem.tolist() + extra[:need]), dtype=int)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[rem] = False
            cur = incumbent[keep_mask].copy()  # (58-k, d)
            # sequential reinsertion
            for _ii in range(k):
                if time.time() >= TRIAL_END or time.time() >= HARD_END - 22.0:
                    break
                C = rng.standard_normal((NCAND, d))
                C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
                dots = C @ cur.T
                worst = dots.max(axis=1)
                bi = int(np.argmin(worst))
                y = C[bi].copy()
                y = _refine_single(y, cur, iters=50)
                cur = np.vstack([cur, y.reshape(1, d)])
            if cur.shape[0] != n:
                continue
            X = cur.copy()
            # shuffle order to avoid bias (deterministic per trial)
            perm = np.arange(n)
            # small deterministic shuffle using rng
            rng.shuffle(perm)
            X = X[perm]
            # short all-point soft-max quench
            X = _quench(X, steps=110, b0=300.0, b1=2500.0, lr0=0.02, lr1=0.003, mu=0.82, clip=0.04)
            if time.time() >= HARD_END - 21.0:
                vq = _max_cos(X)
                if vq < incumb_val - 1e-12:
                    incumbent = X.copy()
                    incumb_val = float(vq)
                continue
            # capped-pair LP polish, short budget
            remain = min(2.0, max(0.5, (TRIAL_END - time.time()) * 0.3, HARD_END - 21.0 - time.time()))
            if remain < 0.4:
                vq = _max_cos(X)
                if vq < incumb_val - 1e-12:
                    incumbent = X.copy()
                    incumb_val = float(vq)
                continue
            X2 = _lp_polish(X, seconds=float(remain), deadline=min(TRIAL_END + 5.0, HARD_END - 20.0))
            v2 = _max_cos(X2)
            if v2 < incumb_val - 1e-12:
                incumbent = X2.copy()
                incumb_val = float(v2)
        except Exception:
            continue
    # reserve final ~20s for LP polishing of incumbent
    try:
        tnow = time.time()
        budget = min(FINAL_BUDGET, max(0.0, HARD_END - tnow - 2.0))
        if budget > 0.5:
            incumbent = _lp_polish(incumbent, seconds=float(budget), deadline=HARD_END - 1.0)
            incumb_val = _max_cos(incumbent)
    except Exception:
        pass
    # also consider original best as fallback
    try:
        if best_val < incumb_val:
            return best / np.linalg.norm(best, axis=1, keepdims=True)
    except Exception:
        pass
    incumbent = incumbent / np.linalg.norm(incumbent, axis=1, keepdims=True)
    return incumbent
