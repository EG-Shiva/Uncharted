import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(X, deadline, radius_init=1e-3, cap=900, max_it=8):
    X = _norm(X.copy())
    n, d = X.shape
    best = _max_cos(X)
    radius = float(radius_init)
    iu = np.triu_indices(n, 1)
    it = 0
    while it < max_it and radius > 1e-14:
        it += 1
        if time.time() >= deadline:
            break
        G = X @ X.T
        c_all = G[iu]
        m = float(np.max(c_all))
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            vals = c_all[idx]
            part = np.argpartition(-vals, cap - 1)[:cap]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[part]] = True
            sel = keep
        I = iu[0][sel]; J = iu[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cd = np.arange(d)
            cols = np.concatenate([I[:, None] * d + cd, J[:, None] * d + cd,
                                   np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 0.8})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        Y = _norm(X + res.x[:-1].reshape(n, d))
        if not np.all(np.isfinite(Y)):
            radius *= 0.5
            continue
        v = _max_cos(Y)
        if v < best:
            X, best = Y, v
            radius = min(radius * 1.6, 0.02)
        else:
            radius *= 0.5
    return X

def _quench(X_init, steps, lr0, lr1, b0, b1, t_end=None):
    X = X_init.copy()
    n, d = X.shape
    m1 = np.zeros_like(X); m2 = np.zeros_like(X)
    b_1, b_2, eps = 0.9, 0.999, 1e-8
    p1, p2 = 1.0, 1.0
    l0, l1 = np.log(lr0), np.log(lr1)
    e0, e1 = np.log(b0), np.log(b1)
    for s in range(steps):
        if t_end is not None and s % 50 == 0 and time.time() >= t_end:
            break
        frac = s / max(steps - 1, 1)
        lr = float(np.exp(l0 + (l1 - l0) * frac))
        beta = float(np.exp(e0 + (e1 - e0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gm = float(G.max())
        E = np.exp(np.clip(beta * (G - gm), -30.0, 0.0))
        np.fill_diagonal(E, 0.0)
        Gr = E @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        # normalize grad scale to avoid explosion
        gs = float(np.linalg.norm(Gt) + 1e-12)
        if gs > 1e6:
            Gt *= (1e6 / gs)
        p1 *= b_1; p2 *= b_2
        m1 = b_1 * m1 + (1.0 - b_1) * Gt
        m2 = b_2 * m2 + (1.0 - b_2) * (Gt * Gt)
        upd = (m1 / (1.0 - p1)) / (np.sqrt(m2 / (1.0 - p2)) + eps)
        X = _norm(X - lr * upd)
    return X

def _far_hole(fixed, rng, n_try=3000):
    d = fixed.shape[1]
    Z = rng.standard_normal((n_try, d))
    Z = _norm(Z)
    S = Z @ fixed.T
    m = S.max(axis=1)
    return Z[int(np.argmin(m))]

def solve():
    t0 = time.time()
    HARD_END = t0 + 135.0
    rng = np.random.default_rng(7)
    n, d = 82, 14
    xb = None
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    xb = a.copy()
                    break
        except Exception:
            continue
    if xb is None:
        xb = _norm(rng.standard_normal((n, d)))
    # quick polish of start within 2s
    try:
        xb = _lp_polish(xb, min(HARD_END - 5, time.time() + 2.0), radius_init=8e-4, cap=900, max_it=6)
    except Exception:
        pass
    best_val = _max_cos(xb)
    X_best = xb.copy()
    iu = np.triu_indices(n, 1)

    it = 0
    # precompute loop
    while time.time() < HARD_END - 6.0:
        it += 1
        if time.time() >= HARD_END - 6.0:
            break
        mode = it % 8
        X = X_best.copy()
        if mode == 0:
            # single worst-point far-hole reinsertion
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            mx = G.max(axis=1)
            v = int(np.argmax(mx))
            fixed = np.delete(X, v, axis=0)
            X[v] = _far_hole(fixed, rng, n_try=4000)
        elif mode == 1:
            # multi (2-4) worst far-hole
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            mx = G.max(axis=1)
            k = int(rng.integers(2, 5))
            victims = np.argsort(-mx)[:k]
            keep_mask = np.ones(n, dtype=bool); keep_mask[victims] = False
            fixed = X[keep_mask]
            for v in victims:
                X[int(v)] = _far_hole(fixed, rng, n_try=2500)
                fixed = np.vstack([fixed, X[int(v)]])
        elif mode == 2:
            # medium gaussian kick whole code
            sig = float(rng.choice([0.01, 0.02, 0.04, 0.08]))
            X = _norm(X + sig * rng.standard_normal((n, d)))
        elif mode == 3:
            # subset kick 6-14 points large
            k = int(rng.integers(6, 15))
            idx = rng.choice(n, size=k, replace=False)
            sig = float(rng.choice([0.05, 0.12, 0.25]))
            X[idx] = _norm(X[idx] + sig * rng.standard_normal((k, d)))
        elif mode == 4:
            # random orthogonal mix + jitter to change PCA bands
            Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
            if rng.random() < 0.5:
                X = _norm((X @ Q) + 0.005 * rng.standard_normal((n, d)))
            else:
                perm = rng.permutation(n)
                X = _norm(X[perm] + 0.02 * rng.standard_normal((n, d)))
        elif mode == 5:
            # completely random fresh start (rare)
            if it % 40 == 0:
                X = _norm(rng.standard_normal((n, d)))
            else:
                X = _norm(X + 0.03 * rng.standard_normal((n, d)))
        elif mode == 6:
            # pair-split: take current max pair and separate + far hole one of them
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            k = int(np.argmax(G[iu]))
            i, j = int(iu[0][k]), int(iu[1][k])
            v = i if rng.random() < 0.5 else j
            fixed = np.delete(X, v, axis=0)
            X[v] = _far_hole(fixed, rng, n_try=4000)
            # also jitter its former partner
            u = j if v == i else i
            X[u] = _norm(X[u] + 0.01 * rng.standard_normal(d))
        else:
            X = _norm(X + 0.015 * rng.standard_normal((n, d)))

        # quench schedule varies by mode
        if time.time() >= HARD_END - 6.0:
            break
        steps = int(rng.choice([250, 350, 500]))
        lr0 = float(rng.choice([0.01, 0.02, 0.03]))
        lr1 = float(rng.choice([0.0008, 0.0015, 0.003]))
        b0 = float(rng.choice([200.0, 400.0, 800.0]))
        b1 = float(rng.choice([1500.0, 3000.0, 6000.0]))
        # occasionally low-beta smooth start then high-beta jam
        try:
            Xq = _quench(X, steps, lr0, lr1, b0, b1, t_end=HARD_END - 4.0)
        except Exception:
            continue
        vq = _max_cos(Xq)
        # polish only if promising (within 3e-3 of best) to save time
        if vq < best_val + 0.003 and time.time() < HARD_END - 5.0:
            try:
                Xp = _lp_polish(Xq, min(HARD_END - 4.0, time.time() + 1.2),
                                radius_init=6e-4, cap=900, max_it=5)
                vp = _max_cos(Xp)
                if vp < vq:
                    Xq, vq = Xp, vp
            except Exception:
                pass
        if vq < best_val - 1e-12:
            best_val = vq
            X_best = Xq.copy()
        # late-stage: keep exploring even if not improving (basin hop from Xq)
        # every 25 iters, hop from second-best perturbation to diversify
        if it % 25 == 0 and time.time() < HARD_END - 8.0:
            # extra exploitation around current Xq if close
            if vq < best_val + 0.001:
                try:
                    X2 = _quench(_norm(Xq + 0.008 * rng.standard_normal((n, d))),
                                 300, 0.015, 0.001, 600.0, 4000.0, t_end=HARD_END - 4.0)
                    v2 = _max_cos(X2)
                    if v2 < best_val + 0.002 and time.time() < HARD_END - 5.0:
                        X2p = _lp_polish(X2, min(HARD_END - 4.0, time.time() + 1.0),
                                         radius_init=4e-4, cap=900, max_it=5)
                        v2p = _max_cos(X2p)
                        if v2p < v2:
                            X2, v2 = X2p, v2p
                    if v2 < best_val - 1e-12:
                        best_val = v2
                        X_best = X2.copy()
                except Exception:
                    pass

    # final polish best with remaining time
    try:
        if time.time() < HARD_END:
            Xp = _lp_polish(X_best, HARD_END - 0.5, radius_init=5e-4, cap=1200, max_it=10)
            vp = _max_cos(Xp)
            if vp < best_val and np.all(np.isfinite(Xp)):
                best_val = vp
                X_best = Xp.copy()
    except Exception:
        pass
    X_best = _norm(np.asarray(X_best, dtype=float))
    return X_best
