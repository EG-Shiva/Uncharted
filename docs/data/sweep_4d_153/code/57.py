import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _smooth(X, steps, b0, b1, lr0, lr1, mu=0.9):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n = X.shape[0]
    V = np.zeros_like(X)
    eye2 = 2.0 * np.eye(n)
    for k in range(steps):
        f = k / (steps - 1) if steps > 1 else 0.0
        beta = b0 + (b1 - b0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        m = np.max(G - eye2)
        A = beta * (G - m)
        np.clip(A, -12.0, 0.0, out=A)
        np.fill_diagonal(A, -30.0)
        np.exp(A, out=A)
        Grad = A @ X
        d = np.sum(Grad * X, axis=1, keepdims=True)
        T = Grad - d * X
        s = float(np.mean(np.sqrt(np.sum(T * T, axis=1)))) + 1e-12
        T /= s
        V *= mu
        V += (1.0 - mu) * T
        X -= lr * V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def _lp_polish(X, budget):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    try:
        best = _max_cos(X)
    except Exception:
        return X
    best_x = X.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < budget and radius > 1e-14:
        G = X @ X.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        # cap problem size for speed
        if p > 1200:
            # keep most tight pairs
            order = np.argsort(-c[sel])[:1200]
            I, J = I[order], J[order]
            ssel = np.where(sel)[0][order]
            csc = c[ssel]
            p = len(I)
        else:
            csc = c[sel]
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csc, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = X + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        except Exception:
            radius *= 0.5
            continue
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best:
            X, best, best_x = y, v, y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x


def _contacts(X, off=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    mx = float(G.max())
    cnt = np.sum(G > mx - off, axis=1)
    return cnt, mx, G


def solve():
    t0 = time.time()
    HARD = 176.0
    n0, d = 153, 4
    rng0 = np.random.default_rng(20261031)
    best = None
    best_val = 1e9
    fallback = None
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n0, d):
                v /= np.linalg.norm(v, axis=1, keepdims=True)
                if np.all(np.isfinite(v)):
                    if fallback is None:
                        fallback = v.copy()
                    c = _max_cos(v)
                    if c < best_val:
                        best_val = c
                        best = v.copy()
        except Exception:
            continue
    if best is None:
        best = rng0.standard_normal((n0, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
        fallback = best.copy()
    if fallback is None:
        fallback = best.copy()
    global_best = best.copy()
    global_val = float(best_val)

    def time_used():
        return time.time() - t0

    def remain():
        return HARD - time_used()

    # short helpers using closure time checks
    def best_candidate(X, rng, n_cand=20000):
        n = X.shape[0]
        # chunked generation to bound memory
        best_i_val = 1e9
        best_vec = None
        done = 0
        while done < n_cand:
            bs = min(5000, n_cand - done)
            C = rng.standard_normal((bs, d))
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            M = C @ X.T
            mv = M.max(axis=1)
            j = int(np.argmin(mv))
            if float(mv[j]) < best_i_val:
                best_i_val = float(mv[j])
                best_vec = C[j].copy()
            done += bs
        return best_vec

    def insert_one(X, rng, n_cand=20000):
        v = best_candidate(X, rng, n_cand=n_cand)
        return np.vstack([X, v.reshape(1, d)])

    def prune_choice(X):
        # return indices sorted by removal quality (unpolished max without i)
        n = X.shape[0]
        try:
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            # for each i, max over pairs not involving i
            # brute force
            scores = np.empty(n)
            for i in range(n):
                # max of G excluding row/col i
                # use mask via max of two blocks
                m = -2.0
                # efficient: global max not involving i
                # just compute via copy-free: temporarily set row/col to -2
                row = G[i, :].copy()
                col = G[:, i].copy()
                G[i, :] = -2.0
                G[:, i] = -2.0
                m = float(G.max())
                G[i, :] = row
                G[:, i] = col
                scores[i] = m
            order = np.argsort(scores)
            return order, scores
        except Exception:
            return np.arange(X.shape[0]), np.zeros(X.shape[0])

    branches = [
        {"seed": 101, "mode": "down", "down": 151, "cand": 20000},
        {"seed": 202, "mode": "downR", "down": 150, "cand": 22000},
        {"seed": 303, "mode": "down", "down": 149, "cand": 20000},
        {"seed": 404, "mode": "up", "up": 155, "cand": 20000},
        {"seed": 505, "mode": "downR", "down": 152, "cand": 24000},
        {"seed": 606, "mode": "downR", "down": 148, "cand": 20000},
        {"seed": 707, "mode": "up", "up": 154, "cand": 22000},
        {"seed": 808, "mode": "down", "down": 150, "cand": 24000},
        {"seed": 909, "mode": "up", "up": 156, "cand": 20000},
        {"seed": 111, "mode": "downR", "down": 151, "cand": 20000},
    ]

    for bi, cfg in enumerate(branches):
        if remain() < 32.0:
            break
        try:
            rng = np.random.default_rng(int(cfg["seed"]))
            X = global_best.copy()
            mode = cfg["mode"]
            if mode in ("down", "downR"):
                target = int(cfg["down"])
                # removal loop
                while X.shape[0] > target:
                    if remain() < 31.0:
                        break
                    try:
                        cnt, mx, G = _contacts(X, off=1e-7)
                    except Exception:
                        break
                    ncur = X.shape[0]
                    if mode == "down":
                        idx = int(np.argmax(cnt))
                        # tie-break randomly among max
                        mcnt = int(cnt.max())
                        ties = np.where(cnt == mcnt)[0]
                        if len(ties) > 1:
                            idx = int(ties[rng.integers(0, len(ties))])
                    else:
                        # random among high-contact
                        order = np.argsort(-cnt)
                        K = min(12, ncur)
                        pool = order[:K]
                        idx = int(pool[rng.integers(0, len(pool))])
                    mask = np.ones(ncur, dtype=bool)
                    mask[idx] = False
                    X = X[mask]
                    # brief polish after removal
                    try:
                        X = _smooth(X, 350, 30.0, 250.0, 0.020, 0.002)
                    except Exception:
                        pass
                    if remain() < 30.5:
                        break
                    try:
                        X = _lp_polish(X, min(1.0, max(0.3, remain() - 30.0)))
                    except Exception:
                        pass
                # rebuild to 153
                while X.shape[0] < n0:
                    if remain() < 31.0:
                        break
                    try:
                        X = insert_one(X, rng, n_cand=int(cfg["cand"]))
                    except Exception:
                        break
                    try:
                        X = _smooth(X, 450, 30.0, 280.0, 0.018, 0.002)
                    except Exception:
                        pass
                    if remain() < 30.5:
                        break
                    try:
                        X = _lp_polish(X, min(1.1, max(0.3, remain() - 30.0)))
                    except Exception:
                        pass
                    if X.shape[0] == n0:
                        try:
                            v = _max_cos(X)
                            if v < global_val - 1e-12 and np.all(np.isfinite(X)):
                                global_val = float(v)
                                global_best = X.copy()
                        except Exception:
                            pass
                # final touch of this branch if at 153
                if X.shape[0] == n0 and remain() > 31.0:
                    try:
                        v = _max_cos(X)
                        if v < global_val + 0.005:
                            Xp = _lp_polish(X, min(1.5, max(0.4, remain() - 30.0)))
                            vp = _max_cos(Xp)
                            if vp < global_val - 1e-12:
                                global_val = float(vp)
                                global_best = Xp.copy()
                            elif vp < v and v < global_val - 1e-12:
                                global_val = float(vp)
                                global_best = Xp.copy()
                    except Exception:
                        pass
            elif mode == "up":
                up = int(cfg["up"])
                while X.shape[0] < up:
                    if remain() < 31.0:
                        break
                    try:
                        X = insert_one(X, rng, n_cand=int(cfg["cand"]))
                    except Exception:
                        break
                    try:
                        X = _smooth(X, 350, 30.0, 250.0, 0.018, 0.002)
                    except Exception:
                        pass
                    try:
                        X = _lp_polish(X, min(0.9, max(0.3, remain() - 30.0)))
                    except Exception:
                        pass
                # prune back to 153
                while X.shape[0] > n0:
                    if remain() < 31.0:
                        break
                    try:
                        order, scores = prune_choice(X)
                    except Exception:
                        break
                    # try top-3 polished
                    K = min(3, X.shape[0])
                    cands = order[:K]
                    best_rm_val = 1e9
                    best_rm_X = None
                    for ci in cands:
                        if remain() < 30.5:
                            break
                        try:
                            mask = np.ones(X.shape[0], dtype=bool)
                            mask[int(ci)] = False
                            Y = X[mask].copy()
                            Y = _smooth(Y, 250, 30.0, 220.0, 0.015, 0.002)
                            Y = _lp_polish(Y, min(0.8, max(0.3, remain() - 30.0)))
                            vy = _max_cos(Y)
                            if vy < best_rm_val:
                                best_rm_val = float(vy)
                                best_rm_X = Y.copy()
                        except Exception:
                            continue
                    if best_rm_X is None:
                        # fallback: remove worst unpolished
                        try:
                            idx = int(order[0])
                            mask = np.ones(X.shape[0], dtype=bool)
                            mask[idx] = False
                            X = X[mask]
                        except Exception:
                            break
                    else:
                        X = best_rm_X
                    if X.shape[0] == n0:
                        try:
                            v = _max_cos(X)
                            if v < global_val - 1e-12:
                                global_val = float(v)
                                global_best = X.copy()
                        except Exception:
                            pass
                if X.shape[0] == n0 and remain() > 31.0:
                    try:
                        Xp = _lp_polish(X, min(1.5, max(0.4, remain() - 30.0)))
                        vp = _max_cos(Xp)
                        if vp < global_val - 1e-12:
                            global_val = float(vp)
                            global_best = Xp.copy()
                    except Exception:
                        pass
        except Exception:
            continue

    # reserve final 25-30 s for polishing best
    try:
        r = remain() - 1.0
        if r > 2.0:
            budget = min(r, 28.0)
            try:
                Xp = _lp_polish(global_best, budget)
                vp = _max_cos(Xp)
                if vp < global_val and np.all(np.isfinite(Xp)):
                    global_val = float(vp)
                    global_best = Xp.copy()
            except Exception:
                pass
    except Exception:
        pass

    try:
        global_best /= np.linalg.norm(global_best, axis=1, keepdims=True)
    except Exception:
        return fallback / np.linalg.norm(fallback, axis=1, keepdims=True)
    # fallback safety
    try:
        if not np.all(np.isfinite(global_best)) or global_best.shape != (n0, d):
            return fallback / np.linalg.norm(fallback, axis=1, keepdims=True)
        gv = _max_cos(global_best)
        fv = _max_cos(fallback)
        if gv > fv + 1e-9:
            # keep global anyway if close, but ensure valid; prefer better
            if gv > fv:
                return fallback
    except Exception:
        pass
    return global_best
