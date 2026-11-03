import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def per_point_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)

def smooth_opt(X, steps, a0, a1, lr0, lr1, mu=0.85):
    X = X.copy()
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best_m = float(G0.max())
    best = X.copy()
    V = np.zeros_like(X)
    one_mu = 1.0 - mu
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < best_m:
            best_m = m
            best = X.copy()
        # softmax weights, shift for stability
        E = np.exp(alpha * (G - m))
        np.fill_diagonal(E, 0.0)
        S = E.sum() * 0.5
        if S == 0 or not np.isfinite(S):
            continue
        Grad = (E @ X) / S
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        # exponential moving average (heavy ball with same scale)
        V = mu * V + one_mu * Gt
        X -= lr * V
        nn = np.linalg.norm(X, axis=1, keepdims=True)
        nn[nn == 0] = 1.0
        X /= nn
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    if m < best_m:
        best = X.copy()
    return best

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def build_E8():
    vecs = []
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = np.zeros(8)
                    v[i] = si
                    v[j] = sj
                    vecs.append(v)
    for mask in range(256):
        cnt = 0
        s = np.ones(8)
        for k in range(8):
            if (mask >> k) & 1:
                s[k] = -1.0
                cnt += 1
        if cnt % 2 == 0:
            vecs.append(0.5 * s)
    E = np.array(vecs, dtype=float)
    E /= np.sqrt(2.0)
    return E

def farthest_reinsert(keep, k, rng_n=4000):
    # keep: (n-k, 8) normalized; generate k new points sequentially
    nkeep = keep.shape[0]
    new_pts = []
    cur = keep.copy()
    for _ in range(k):
        C = np.random.randn(rng_n, 8)
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        # score = max dot to cur
        D = C @ cur.T
        scores = D.max(axis=1)
        order = np.argsort(scores)
        # take best few and refine by local jitter search
        best_c = C[order[0]]
        best_s = scores[order[0]]
        # local refinement around top 8 candidates
        for cand in C[order[:8]]:
            for _ in range(60):
                jit = cand + 0.02 * np.random.randn(8)
                jit /= np.linalg.norm(jit)
                s = float(np.max(jit @ cur.T))
                if s < best_s:
                    best_s = s
                    best_c = jit
        new_pts.append(best_c)
        cur = np.vstack([cur, best_c.reshape(1, 8)])
    return np.array(new_pts)

def solve():
    np.random.seed(7)
    t0 = time.time()
    TOTAL = 172.0
    try:
        X_best = np.load("best.npy").astype(float)
    except Exception:
        try:
            X_best = np.load("record.npy").astype(float)
        except Exception:
            X_best = np.random.randn(168, 8)
    X_best = X_best.reshape(168, 8)
    X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
    # also consider record
    try:
        X_rec = np.load("record.npy").astype(float).reshape(168, 8)
        X_rec /= np.linalg.norm(X_rec, axis=1, keepdims=True)
        if max_cos(X_rec) < max_cos(X_best):
            X_best = X_rec.copy()
    except Exception:
        pass
    best_val = max_cos(X_best)
    X_cur = X_best.copy()
    cur_val = best_val
    E8 = build_E8()
    T = 0.0025
    it = 0
    # exploration until TOTAL-34
    while time.time() - t0 < TOTAL - 34.0:
        it += 1
        r = np.random.rand()
        if r < 0.10 and it > 3:
            # pure random restart
            X_init = np.random.randn(168, 8)
            X_init /= np.linalg.norm(X_init, axis=1, keepdims=True)
            steps, a0, a1, lr0, lr1, mu = 2000, 10.0, 350.0, 0.06, 0.003, 0.85
        elif r < 0.24:
            # E8 subset restart
            idx = np.random.choice(240, 168, replace=False)
            X_init = E8[idx] + 0.015 * np.random.randn(168, 8)
            # random orthogonal scramble occasionally
            if np.random.rand() < 0.25:
                Q, _ = np.linalg.qr(np.random.randn(8, 8))
                if np.linalg.det(Q) < 0:
                    Q[:, 0] = -Q[:, 0]
                X_init = X_init @ Q.T
            X_init /= np.linalg.norm(X_init, axis=1, keepdims=True)
            steps, a0, a1, lr0, lr1, mu = 1600, 12.0, 420.0, 0.05, 0.0025, 0.85
        elif r < 0.55:
            # sigma perturbation of best/cur + overload kick
            base = X_cur if np.random.rand() < 0.65 else X_best
            sigma = float(np.random.choice([0.004, 0.008, 0.015, 0.03, 0.06, 0.12]))
            X_init = base + sigma * np.random.randn(168, 8)
            if np.random.rand() < 0.6:
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                mx = G.max(axis=1)
                # contacts count
                m = float(G.max())
                cnt = (G > m - 1e-9).sum(axis=1)
                score = mx + 0.002 * cnt
                worst = np.argsort(score)[-6:]
                X_init[worst] += np.random.randn(6, 8) * sigma * 2.5
            X_init /= np.linalg.norm(X_init, axis=1, keepdims=True)
            if sigma >= 0.06:
                steps, a0, a1, lr0, lr1, mu = 2000, 10.0, 450.0, 0.05, 0.002, 0.88
            else:
                steps, a0, a1, lr0, lr1, mu = 1400, 18.0, 520.0, 0.045, 0.002, 0.85
        else:
            # surgical ejection + farthest reinsert (topology change)
            base = X_cur if np.random.rand() < 0.6 else X_best
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            mx = G.max(axis=1)
            cnt = (G > m - 1e-7).sum(axis=1)
            score = mx + 0.0015 * cnt
            k = int(np.random.choice([1, 2, 3, 4], p=[0.35, 0.3, 0.2, 0.15]))
            worst = np.argsort(score)[-k:]
            mask = np.ones(168, dtype=bool)
            mask[worst] = False
            keep = base[mask]
            try:
                newp = farthest_reinsert(keep, k, rng_n=4000)
            except Exception:
                newp = np.random.randn(k, 8)
                newp /= np.linalg.norm(newp, axis=1, keepdims=True)
            X_init = np.vstack([keep, newp])
            # small jitter all
            X_init += 0.004 * np.random.randn(168, 8)
            X_init /= np.linalg.norm(X_init, axis=1, keepdims=True)
            steps, a0, a1, lr0, lr1, mu = 1700, 12.0, 480.0, 0.05, 0.002, 0.86
        X_opt = smooth_opt(X_init, steps, a0, a1, lr0, lr1, mu=mu)
        v = max_cos(X_opt)
        if v < best_val:
            best_val = v
            X_best = X_opt.copy()
            X_cur = X_opt.copy()
            cur_val = v
        else:
            if v < cur_val or np.random.rand() < np.exp(-(v - cur_val) / max(T, 1e-7)):
                X_cur = X_opt.copy()
                cur_val = v
        T = max(T * 0.997, 0.0002)
        if time.time() - t0 > TOTAL - 36:
            break
    # intense squeeze of best
    try:
        if time.time() - t0 < TOTAL - 30:
            X_best = smooth_opt(X_best, 1000, 80.0, 900.0, 0.012, 0.001, mu=0.7)
    except Exception:
        pass
    remain = TOTAL - (time.time() - t0) - 1.0
    if remain > 4:
        try:
            X_best = lp_polish(X_best, seconds=min(remain, 28.0))
        except Exception:
            pass
    X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best
