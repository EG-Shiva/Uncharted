import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        iu_c = G[iu]
        m = iu_c.max()
        thresh = m - max(20 * radius, 1e-9)
        sel = iu_c > thresh
        if np.count_nonzero(sel) == 0:
            break
        I = iu[0][sel]; J = iu[1][sel]
        csub = iu_c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y /= yn
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _quench(X_init, steps, T0, T1, lr0, lr1):
    X = X_init.copy()
    n, d = X.shape
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lT0 = np.log(T0); lT1 = np.log(T1)
    llr0 = np.log(lr0); llr1 = np.log(lr1)
    best_m = np.inf
    best_X = X.copy()
    # init best
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best_m = float(G0.max())
    best_X = X.copy()
    for s in range(1, steps + 1):
        frac = (s - 1) / max(steps - 1, 1)
        T = float(np.exp(lT0 * (1 - frac) + lT1 * frac))
        lr = float(np.exp(llr0 * (1 - frac) + llr1 * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < best_m:
            best_m = m
            best_X = X.copy()
        D = (G - m) / T
        # exp, D<=0
        E = np.exp(D)
        np.fill_diagonal(E, 0.0)
        S = E.sum() * 0.5
        if S <= 0 or not np.isfinite(S):
            S = 1.0
        Grad = (E @ X) / S
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        if not np.all(np.isfinite(Gt)):
            break
        m1 = b1 * m1 + (1 - b1) * Gt
        v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
        c1 = 1 - b1 ** s
        c2 = 1 - b2 ** s
        mhat = m1 / c1
        vhat = v1 / c2
        step = mhat / (np.sqrt(vhat) + eps)
        # clip extreme steps
        # global clip
        sn = np.linalg.norm(step)
        maxn = 10.0 * np.sqrt(n * d)
        if sn > maxn:
            step *= (maxn / sn)
        X = X - lr * step
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[~np.isfinite(nrm)] = 1.0
        nrm[nrm == 0] = 1.0
        X /= nrm
    # final check
    Gf = X @ X.T
    np.fill_diagonal(Gf, -2.0)
    mf = float(Gf.max())
    if mf < best_m:
        best_m = mf
        best_X = X.copy()
    return best_X

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    FINAL_POLISH = 26.0
    HOP_END = HARD_END - FINAL_POLISH - 1.0
    np.random.seed(0)
    rng = np.random.default_rng(12345)
    # load start
    best_X = None
    try:
        best_X = np.load("best.npy").astype(float)
    except Exception:
        try:
            best_X = np.load("record.npy").astype(float)
        except Exception:
            pass
    if best_X is None or best_X.shape != (71, 12) or not np.all(np.isfinite(best_X)):
        r = rng.standard_normal((71, 12))
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        best_X = r
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    best_score = _max_cos(best_X)
    n, d = 71, 12
    it = 0
    # precompute contacts for targeting (updated on improvement)
    contacts = None
    def _contacts(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        cnt = np.sum(G > (m - 0.003), axis=1)
        return cnt, m
    try:
        contacts, _ = _contacts(best_X)
    except Exception:
        contacts = np.ones(n)
    while time.time() < HOP_END:
        it += 1
        # time left check
        tleft = HOP_END - time.time()
        if tleft < 0.5:
            break
        # hyperparams
        u = rng.random()
        if it % 37 == 0:
            # pure random restart
            X0 = rng.standard_normal((n, d))
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            sigma = 0.0
            k_rep = 0
        else:
            # sigma log-uniform 0.003 .. 0.35
            sigma = float(10 ** rng.uniform(-2.5, -0.45))
            rpick = rng.random()
            if rpick < 0.55:
                k_rep = 0
            elif rpick < 0.85:
                k_rep = int(rng.integers(1, 4))  # 1-3
            else:
                k_rep = int(rng.integers(4, 7))  # 4-6 bigger jump
            if k_rep > 0:
                # choose indices biased to overloaded
                try:
                    order = np.argsort(-contacts)
                    top10 = order[:10]
                except Exception:
                    top10 = np.arange(10)
                idx = []
                # one from top
                idx.append(int(rng.choice(top10)))
                while len(idx) < k_rep:
                    cand = int(rng.integers(0, n))
                    if cand not in idx:
                        idx.append(cand)
                idx = np.array(idx, dtype=int)
                X0 = best_X.copy()
                rnd = rng.standard_normal((k_rep, d))
                rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
                X0[idx] = rnd
                # small noise to rest
                noise = rng.standard_normal((n, d)) * (sigma * 0.5)
                mask = np.ones(n, dtype=bool)
                mask[idx] = False
                X0[mask] += noise[mask]
                X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            else:
                noise = rng.standard_normal((n, d)) * sigma
                X0 = best_X + noise
                X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        # quench params
        T0 = float(10 ** rng.uniform(-2.4, -1.5))  # ~0.004..0.03
        T1 = float(10 ** rng.uniform(-4.0, -3.0))  # 0.0001..0.001
        lr0 = float(10 ** rng.uniform(-2.7, -2.1))  # ~0.002..0.008
        lr1 = lr0 * float(10 ** rng.uniform(-1.3, -0.7))  # decay ~10x
        # steps adapt to time
        if tleft < 5:
            steps = 1200
        elif tleft < 20:
            steps = 1800
        else:
            steps = 2200
            if rng.random() < 0.2:
                steps = 3500
        try:
            X1 = _quench(X0, steps, T0, T1, lr0, lr1)
            s1 = _max_cos(X1)
        except Exception:
            continue
        if not np.isfinite(s1):
            continue
        # promising? quench best is already tight; allow small slack
        if s1 < best_score + 0.0008:
            # short LP polish to reveal true local optimum (2s max, respect HOP_END)
            secs = 2.0
            if tleft < secs + 0.5:
                secs = max(0.5, tleft - 0.5)
            try:
                X2 = _lp_polish(X1, seconds=secs, deadline=HOP_END)
                s2 = _max_cos(X2)
            except Exception:
                continue
            if np.isfinite(s2) and s2 < best_score - 1e-12:
                best_X = X2 / np.linalg.norm(X2, axis=1, keepdims=True)
                best_score = float(s2)
                try:
                    contacts, _ = _contacts(best_X)
                except Exception:
                    pass
        else:
            # far worse, ignore (but if random restart found good basin, s1 would be close)
            pass
        # occasional contacts refresh
        if it % 20 == 0:
            try:
                contacts, _ = _contacts(best_X)
            except Exception:
                pass
    # final squeeze of best
    try:
        trem = HARD_END - time.time() - 0.8
        if trem > 1.0:
            best_X = _lp_polish(best_X, seconds=min(trem, 40.0), deadline=HARD_END - 0.5)
    except Exception:
        pass
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    # safety: replace non-finite rows
    for i in range(best_X.shape[0]):
        if not np.all(np.isfinite(best_X[i])) or np.linalg.norm(best_X[i]) < 1e-12:
            v = rng.standard_normal(d)
            v /= np.linalg.norm(v)
            best_X[i] = v
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    return best_X
