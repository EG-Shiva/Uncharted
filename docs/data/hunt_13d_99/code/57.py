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

def _lse_fun_factory(N, d, alpha):
    def fun(Uf):
        U = Uf.reshape(N, d)
        nr = np.linalg.norm(U, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = U / nr
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        E = np.exp(alpha * (G - m))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5) + 1e-300
        L = m + np.log(S) / alpha
        GX = (E @ X) / S
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GU = (GX - dot * X) / nr
        return float(L), GU.ravel().astype(float)
    return fun

def _anneal(X0, alphas, miters):
    N, d = X0.shape
    U0 = X0.ravel().astype(float).copy()
    for alpha, mit in zip(alphas, miters):
        fun = _lse_fun_factory(N, d, float(alpha))
        try:
            res = minimize(fun, U0, method='L-BFGS-B', jac=True,
                           options={'maxiter': int(mit), 'ftol': 1e-20,
                                    'gtol': 1e-7, 'maxfun': int(mit)*8, 'maxcor': 5})
            U0 = res.x.copy()
        except Exception:
            break
    return _norm_rows(U0.reshape(N, d))

def _hole_replace(X, idx, rng, n_cand=2000):
    Y = X.copy()
    d = X.shape[1]
    for k in idx:
        kk = int(k)
        cand = _norm_rows(rng.standard_normal(size=(n_cand, d)))
        mask = np.ones(X.shape[0], dtype=bool)
        mask[kk] = False
        others = Y[mask]
        md = cand @ others.T
        m = md.max(axis=1)
        Y[kk] = cand[np.argmin(m)]
    return Y

def solve():
    t0 = time.time()
    BUDGET = 170.0
    T_END = t0 + BUDGET
    T_EXPL = t0 + 143.0
    rng = np.random.default_rng(7)
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
    # quick initial collapse of plateau with low-mid alpha
    try:
        if time.time() < T_EXPL:
            Xc = _anneal(best_X, [60.0, 220.0, 650.0], [60, 60, 60])
            vc = _max_cos(Xc)
            if vc < best_v:
                best_v = vc
                best_X = Xc.copy()
    except Exception:
        pass
    cur_X = best_X.copy()
    cur_v = best_v
    n, d = 99, 13
    it = 0
    # precompute for speed: schedule depends on remaining time
    while time.time() < T_EXPL and it < 300:
        it += 1
        remain = T_EXPL - time.time()
        if remain < 4:
            break
        # choose base
        r0 = rng.random()
        if r0 < 0.65:
            base = best_X
        elif r0 < 0.9:
            base = cur_X
        else:
            base = None
        # per-point maxima for targeted moves
        try:
            Gc = cur_X @ cur_X.T
            np.fill_diagonal(Gc, -2.0)
            rowmax = Gc.max(axis=1)
            order = np.argsort(rowmax)
            Gw = best_X @ best_X.T
            np.fill_diagonal(Gw, -2.0)
            border = np.argsort(Gw.max(axis=1))
        except Exception:
            order = np.arange(n)
            border = np.arange(n)
        mode = rng.random()
        if base is None:
            # fresh random basin (rare)
            X0 = _norm_rows(rng.standard_normal(size=(n, d)))
            alphas = [50.0, 180.0, 550.0]
            mits = [70, 70, 80]
        elif mode < 0.35:
            sigma = float(rng.choice([0.01, 0.025, 0.06, 0.12, 0.20]))
            X0 = _norm_rows(base + rng.standard_normal(size=(n, d)) * sigma)
            alphas = [60.0, 220.0, 650.0]
            mits = [60, 60, 70]
        elif mode < 0.60:
            kk = int(rng.integers(2, 8))
            # worst points of cur (or best)
            idx = order[-kk:] if rng.random() < 0.6 else border[-kk:]
            X0 = base.copy()
            # strong shake on worst, tiny elsewhere
            X0[idx] = _norm_rows(X0[idx] + rng.standard_normal(size=(kk, d)) * 0.25)
            X0 = _norm_rows(X0 + rng.standard_normal(size=(n, d)) * 0.005)
            alphas = [60.0, 220.0, 650.0]
            mits = [60, 60, 70]
        elif mode < 0.85:
            kk = int(rng.integers(2, 7))
            idx = order[-kk:] if rng.random() < 0.7 else border[-kk:]
            X0 = _hole_replace(base, idx, rng, n_cand=1500)
            # small jitter after reinsertion
            X0 = _norm_rows(X0 + rng.standard_normal(size=(n, d)) * 0.008)
            alphas = [60.0, 220.0, 650.0]
            mits = [60, 60, 70]
        else:
            # pairwise repulsion kick on top pairs then anneal
            try:
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                iu0, iu1 = np.triu_indices(n, 1)
                c = G[iu0, iu1]
                k = min(30, c.size)
                top = np.argpartition(-c, k)[:k]
                X0 = base.copy()
                for t in top[:12]:
                    i = int(iu0[t]); j = int(iu1[t])
                    v = X0[i] - X0[j]
                    nv = np.linalg.norm(v)
                    if nv < 1e-12:
                        v = rng.standard_normal(size=(d,))
                        nv = np.linalg.norm(v)
                    v = v / nv
                    step = float(rng.uniform(0.005, 0.03))
                    X0[i] = _norm_rows((X0[i] + step * v).reshape(1, -1))[0]
                    X0[j] = _norm_rows((X0[j] - step * v).reshape(1, -1))[0]
                X0 = _norm_rows(X0 + rng.standard_normal(size=(n, d)) * 0.01)
            except Exception:
                X0 = _norm_rows(base + rng.standard_normal(size=(n, d)) * 0.03)
            alphas = [60.0, 220.0, 650.0]
            mits = [60, 60, 70]
        # shrink work if time low
        if remain < 25:
            mits = [40, 40, 40]
        try:
            Xn = _anneal(X0, alphas, mits)
            vn = _max_cos(Xn)
        except Exception:
            continue
        if vn < best_v:
            best_v = vn
            best_X = Xn.copy()
            cur_X = Xn.copy()
            cur_v = vn
        elif vn < cur_v or rng.random() < 0.12:
            # move current (allow sideways to change basin)
            if vn < cur_v + 0.004:
                cur_X = Xn.copy()
                cur_v = vn
        # opportunistic: if cur stuck much worse than best, reset cur to best
        if cur_v > best_v + 0.003 and rng.random() < 0.5:
            cur_X = best_X.copy()
            cur_v = best_v
    # final polish of best with higher alpha
    try:
        remain = T_END - time.time()
        if remain > 8:
            # two-stage final: mid then high
            Xf = _anneal(best_X, [250.0, 700.0], [120, 120])
            vf = _max_cos(Xf)
            if vf < best_v:
                best_v = vf
                best_X = Xf.copy()
            remain = T_END - time.time()
            if remain > 6:
                Xf2 = _anneal(best_X, [1500.0], [80])
                vf2 = _max_cos(Xf2)
                if vf2 < best_v:
                    best_v = vf2
                    best_X = Xf2.copy()
    except Exception:
        pass
    best_X = _norm_rows(best_X)
    return best_X
