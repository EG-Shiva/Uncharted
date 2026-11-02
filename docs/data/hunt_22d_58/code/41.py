import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _quench(X, steps, b0, b1, lr0, lr1, mu=0.8, clip=0.04):
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
        step = np.where(sn > clip, step * (clip / sn), step)
        X = X - step
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X

def _greedy_push(X, steps, e0, e1):
    n = X.shape[0]
    for s in range(steps):
        f = s / max(1, steps - 1)
        eps = e0 * (e1 / e0) ** f if e0 != e1 else e0
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        k = int(np.argmax(G))
        i = k // n
        j = k % n
        g = float(G[i, j])
        xi = X[i].copy()
        xj = X[j].copy()
        ti = xj - g * xi
        tj = xi - g * xj
        ni = float(np.linalg.norm(ti)) + 1e-12
        nj = float(np.linalg.norm(tj)) + 1e-12
        X[i] = xi - eps * ti / ni
        X[j] = xj - eps * tj / nj
        X[i] /= (np.linalg.norm(X[i]) + 1e-12)
        X[j] /= (np.linalg.norm(X[j]) + 1e-12)
    return X

def _hole_best(X, victim, rng, n_cand=2000):
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

def _lp_polish(X, seconds, deadline):
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    n, d = X.shape
    best = _max_cos(X)
    radius = 8e-4
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
            if c[idx] > m - window and len(sel_list) < 500:
                sel_list.append(int(idx))
            elif c[idx] <= m - window:
                break
            if len(sel_list) >= 500:
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
    DEADLINE = t0 + 165.0
    rng = np.random.default_rng(0)
    n, d = 58, 22
    best = None
    best_val = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            v = np.load(f).astype(float)
            if v.shape != (58, 22):
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
    # basin hopping with fast quench only (no LP inside loop)
    n_iter = 0
    # precompute schedule variety
    while time.time() < DEADLINE - 25.0 and n_iter < 120:
        n_iter += 1
        G = best @ best.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # contact degree with small gap to find hubs
        deg = (G > m - 0.004).sum(axis=1)
        worst_max = G.max(axis=1)
        # choose number of victims 1-3
        r = rng.random()
        if r < 0.5:
            k = 1
        elif r < 0.85:
            k = 2
        else:
            k = 3
        # victims: mix of high-degree and high-max
        score = deg.astype(float) + 3.0 * (worst_max - m + 0.01) * 100.0
        # add random tie break
        score = score + rng.standard_normal(n) * 0.5
        victims = np.argsort(-score)[:k]
        X = best.copy()
        for v in victims:
            # adapt candidates to save time
            X = _hole_best(X, int(v), rng, n_cand=1500)
        # small shake of victims neighbourhood to diversify
        sig = 0.01 + 0.02 * rng.random()
        noise = rng.standard_normal((k, d)) * sig
        for ii, v in enumerate(victims):
            X[int(v)] = X[int(v)] + noise[ii]
            X[int(v)] /= (np.linalg.norm(X[int(v)]) + 1e-12)
        # quench with varying schedule
        if n_iter % 3 == 0:
            X = _quench(X, steps=500, b0=80.0, b1=1500.0, lr0=0.02, lr1=0.002, mu=0.82)
        elif n_iter % 3 == 1:
            X = _quench(X, steps=700, b0=150.0, b1=2500.0, lr0=0.015, lr1=0.001, mu=0.8)
        else:
            X = _quench(X, steps=400, b0=50.0, b1=800.0, lr0=0.03, lr1=0.003, mu=0.85)
        X = _greedy_push(X, steps=30, e0=0.01, e1=0.001)
        # short extra quench after push
        X = _quench(X, steps=150, b0=500.0, b1=3000.0, lr0=0.005, lr1=0.001, mu=0.7)
        v = _max_cos(X)
        if v < best_val - 1e-12:
            best = X
            best_val = v
        # occasional global shake from best to escape
        if n_iter % 15 == 0 and time.time() < DEADLINE - 40:
            Xs = best + rng.standard_normal(best.shape) * 0.03
            Xs /= (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-12)
            Xs = _quench(Xs, steps=600, b0=100.0, b1=2000.0, lr0=0.02, lr1=0.002, mu=0.8)
            vs = _max_cos(Xs)
            if vs < best_val - 1e-12:
                best = Xs
                best_val = vs
    # final squeeze: longer quench then capped LP
    try:
        best = _quench(best, steps=800, b0=200.0, b1=4000.0, lr0=0.008, lr1=0.0008, mu=0.8)
        best = _greedy_push(best, steps=50, e0=0.005, e1=0.0005)
    except Exception:
        pass
    try:
        if time.time() < DEADLINE - 3:
            remain = min(18.0, DEADLINE - time.time() - 1.0)
            if remain > 2:
                best = _lp_polish(best, remain, DEADLINE - 1.0)
    except Exception:
        pass
    best = best / (np.linalg.norm(best, axis=1, keepdims=True) + 1e-12)
    # tiny shrink guard against NaN
    best = np.nan_to_num(best, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    best = best / nr
    return best
