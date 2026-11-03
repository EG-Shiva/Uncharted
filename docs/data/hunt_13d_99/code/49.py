import time
import numpy as np
from scipy.optimize import minimize

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _anneal(X0, alphas, miters, deadline):
    N, d = X0.shape
    U0 = X0.ravel().astype(float).copy()
    Xcur = X0.copy()
    for alpha, mit in zip(alphas, miters):
        if time.time() > deadline:
            break
        a = float(alpha)
        def fun(Uf, _a=a):
            U = Uf.reshape(N, d)
            nr = np.linalg.norm(U, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            X = U / nr
            G = X @ X.T
            np.fill_diagonal(G, -10.0)
            m = float(np.max(G))
            E = np.exp(_a * (G - m))
            np.fill_diagonal(E, 0.0)
            S = float(E.sum() * 0.5) + 1e-300
            L = m + np.log(S) / _a
            GX = (E @ X) / S
            dot = np.sum(GX * X, axis=1, keepdims=True)
            GU = (GX - dot * X) / nr
            return float(L), GU.ravel().astype(float)
        try:
            res = minimize(fun, U0, method='L-BFGS-B', jac=True,
                           options={'maxiter': int(mit), 'ftol': 1e-20,
                                    'gtol': 1e-7, 'maxfun': int(mit)*10, 'maxcor': 10})
            U0 = res.x.copy()
        except Exception:
            break
        Xcur = _norm_rows(U0.reshape(N, d))
    return Xcur

def _soft_refine(X0, a0, a1, steps, lr, beta=0.9):
    cur = X0.copy()
    best = cur.copy()
    bs = _max_cos(cur)
    mom = np.zeros_like(cur)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        alpha = a0 * ((a1 / max(a0, 1e-12)) ** frac)
        G = cur @ cur.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ cur
        proj = np.sum(Grad * cur, axis=1, keepdims=True)
        Gt = Grad - proj * cur
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        Gt = Gt / rs
        mom = beta * mom + (1 - beta) * Gt
        bc = 1 - beta ** (s + 1)
        cur = cur - lr * (1 - 0.5 * frac) * (mom / bc)
        cur = _norm_rows(cur)
        if s % 200 == 199 or s == steps - 1:
            sc = _max_cos(cur)
            if sc < bs:
                bs = sc
                best = cur.copy()
    return best, bs

def _void_replace(X, idx, rng, n_cand=2000):
    Y = X.copy()
    d = X.shape[1]
    for k in idx:
        cand = _norm_rows(rng.standard_normal(size=(n_cand, d)))
        others = np.delete(Y, int(k), axis=0)
        md = (cand @ others.T).max(axis=1)
        Y[int(k)] = cand[np.argmin(md)]
    return Y

def solve():
    t0 = time.time()
    BUDGET = 172.0
    T_END = t0 + BUDGET
    T_EXPL = t0 + 138.0
    rng = np.random.default_rng(12345)
    cands = []
    try:
        xb = np.asarray(np.load("best.npy"), float)
        if xb.shape == (99, 13):
            cands.append(_norm_rows(xb))
    except Exception:
        pass
    try:
        xr = np.asarray(np.load("record.npy"), float)
        if xr.shape == (99, 13):
            cands.append(_norm_rows(xr))
    except Exception:
        pass
    if not cands:
        cands.append(_norm_rows(rng.standard_normal(size=(99, 13))))
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_X = cands[bi].copy()
    best_v = float(vals[bi])
    n, d = 99, 13
    # initial plateau collapse from incumbent
    try:
        Xc = _anneal(best_X, [35, 100, 280, 600], [180, 180, 180, 140], min(T_EXPL, t0 + 32))
        vc = _max_cos(Xc)
        if vc < best_v:
            best_v = vc
            best_X = Xc.copy()
        # light soft refine
        q, vq = _soft_refine(Xc, 30.0, 200.0, 500, 0.04)
        if vq < best_v:
            best_v = vq
            best_X = q.copy()
    except Exception:
        pass
    cur_X = best_X.copy()
    cur_v = best_v
    elite = [best_X.copy()]
    it = 0
    while time.time() < T_EXPL and it < 200:
        it += 1
        remain = T_EXPL - time.time()
        if remain < 3:
            break
        mode = rng.random()
        # per-point maxima to find overloaded
        try:
            Gc = cur_X @ cur_X.T
            np.fill_diagonal(Gc, -2.0)
            rowmax = Gc.max(axis=1)
            order = np.argsort(rowmax)
        except Exception:
            order = np.arange(n)
        if mode < 0.38:
            sigma = float(rng.choice([0.008, 0.015, 0.03, 0.06, 0.11]))
            base = best_X if rng.random() < 0.7 else cur_X
            if len(elite) > 1 and rng.random() < 0.2:
                base = elite[int(rng.integers(0, len(elite)))]
            X0 = _norm_rows(base + rng.standard_normal(size=(n, d)) * sigma)
            alphas = [40.0, 140.0, 420.0]
            mits = [110, 110, 110]
        elif mode < 0.68:
            kk = int(rng.integers(2, 7))
            idx = order[-kk:]
            sigma = float(rng.uniform(0.06, 0.28))
            X0 = cur_X.copy()
            X0[idx] = X0[idx] + rng.standard_normal(size=(kk, d)) * sigma
            X0 = _norm_rows(X0)
            if rng.random() < 0.5:
                pk = rng.choice(idx, size=min(2, kk), replace=False)
                X0 = _void_replace(X0, pk, rng, 1500)
                X0 = _norm_rows(X0 + rng.standard_normal(size=(n, d)) * 0.008)
            alphas = [55.0, 160.0, 500.0]
            mits = [120, 120, 110]
        else:
            kk = int(rng.integers(1, 4))
            idx = order[-kk:]
            X0 = _void_replace(cur_X, idx, rng, 2500)
            X0 = _norm_rows(X0 + rng.standard_normal(size=(n, d)) * 0.012)
            alphas = [45.0, 150.0, 450.0]
            mits = [130, 120, 110]
        deadline = min(time.time() + 6.0, T_EXPL)
        try:
            X1 = _anneal(X0, alphas, mits, deadline)
        except Exception:
            continue
        # quick soft polish, cheap
        try:
            a0 = float(rng.choice([25.0, 40.0]))
            X1b, _ = _soft_refine(X1, a0, a0 * 5.0, 350, float(rng.uniform(0.02, 0.05)))
            # keep better of the two by max_cos
            if _max_cos(X1b) < _max_cos(X1):
                X1 = X1b
        except Exception:
            pass
        try:
            v1 = _max_cos(X1)
        except Exception:
            continue
        if v1 < best_v - 1e-12:
            best_v = v1
            best_X = X1.copy()
            cur_X = X1.copy()
            cur_v = v1
            elite.append(X1.copy())
            if len(elite) > 5:
                elite = elite[-5:]
        elif v1 < cur_v + 0.0012 and rng.random() < 0.25:
            # accept move to different basin (simulated annealing)
            cur_X = X1.copy()
            cur_v = v1
        # occasional reset to best
        if it % 12 == 0:
            cur_X = best_X.copy()
            cur_v = best_v
    # final high-alpha polish on best
    try:
        remain = T_END - time.time() - 1.0
        if remain > 5:
            Xp = _anneal(best_X, [700.0, 1400.0], [200, 180], time.time() + min(remain, 28.0))
            vp = _max_cos(Xp)
            if vp < best_v:
                best_v = vp
                best_X = Xp.copy()
    except Exception:
        pass
    best_X = _norm_rows(best_X)
    return best_X
