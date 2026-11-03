import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _per_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1)

def _fast_relax(X, iters, lr0=0.02, margin0=0.004):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    best = _max_cos(X)
    bestX = X.copy()
    cur = X.copy()
    lr = lr0
    margin = margin0
    for it in range(iters):
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        tau = margin * 0.2 + 1e-12
        W = np.exp((G - m) / tau)
        W[G < m - margin * 5.0] = 0.0
        np.fill_diagonal(W, 0.0)
        Grad = W @ cur
        proj = np.sum(Grad * cur, axis=1, keepdims=True)
        Gt = Grad - proj * cur
        gn = float(np.max(np.linalg.norm(Gt, axis=1)))
        if not np.isfinite(gn) or gn < 1e-14:
            margin *= 0.5
            if margin < 1e-7:
                break
            continue
        Xn = cur - (lr / gn) * Gt
        Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
        G2 = Xn @ Xn.T
        np.fill_diagonal(G2, -2.0)
        mn = float(np.max(G2))
        if mn < m:
            cur = Xn
            if mn < best:
                best = mn
                bestX = Xn.copy()
            if it % 200 == 0:
                margin *= 0.85
        else:
            lr *= 0.6
            if lr < 1e-6:
                margin *= 0.7
                lr = lr0 * 0.3
                if margin < 5e-7:
                    break
    return bestX, best

def _lp_polish(x, seconds, t0, BUDGET):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    best = _max_cos(x)
    bx = x.copy()
    radius = 1e-3
    start = time.time()
    while time.time() - start < seconds and time.time() - t0 < BUDGET - 2:
        if radius < 1e-14:
            break
        Gf = x @ x.T
        c = Gf[iu0, iu1]
        m = float(np.max(c))
        thresh = m - max(20 * radius, 1e-9)
        sel = c > thresh
        I = iu0[sel]; J = iu1[sel]; csel = c[sel]
        p = len(I)
        if p == 0:
            break
        if p > 3500:
            k = 3500
            part = np.argpartition(-csel, k - 1)[:k]
            I, J, csel = I[part], J[part], csel[part]
            p = k
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
        obj = np.zeros(n * d + 1)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            bx = y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return bx, best

def _lbfgs_sm(X0, beta, maxiter):
    n, d = X0.shape
    iu0, iu1 = np.triu_indices(n, 1)
    y0 = X0.ravel().copy()
    def fun(y):
        Y = y.reshape(n, d)
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn = np.maximum(rn, 1e-12)
        X = Y / rn
        G = X @ X.T
        g = G[iu0, iu1]
        m = float(np.max(g))
        e = np.exp(beta * (g - m))
        S = float(np.sum(e))
        loss = m + np.log(S + 1e-300) / beta
        w = e / (S + 1e-300)
        W = np.zeros((n, n))
        W[iu0, iu1] = w
        W = W + W.T
        GX = W @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / rn
        return float(loss), GY.ravel()
    try:
        res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8, "maxfun": 20000})
        Y = res.x.reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    except Exception:
        return X0

def _refine_void(v, others, iters=150, lr0=0.06):
    v = v / np.linalg.norm(v)
    lr = lr0
    best = float(np.max(others @ v))
    bestv = v.copy()
    for _ in range(iters):
        dots = others @ v
        m = float(np.max(dots))
        tau = 0.002
        w = np.exp((dots - m) / tau)
        w[dots < m - 0.012] = 0.0
        g = w @ others
        gt = g - float(g @ v) * v
        nrm = float(np.linalg.norm(gt))
        if nrm < 1e-14 or not np.isfinite(nrm):
            break
        vn = v - (lr / nrm) * gt
        vn /= np.linalg.norm(vn)
        mn = float(np.max(others @ vn))
        if mn < m:
            v = vn
            if mn < best:
                best = mn
                bestv = vn.copy()
        else:
            lr *= 0.7
            if lr < 1e-5:
                break
    return bestv, best

def _find_void(X, rng, n_cand=4000, n_top=3):
    d = X.shape[1]
    C = rng.standard_normal((n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    D = X @ C.T
    mp = np.max(D, axis=0)
    idx = np.argpartition(mp, n_top - 1)[:n_top]
    bestv = None
    bests = 1e9
    for j in idx:
        v, s = _refine_void(C[j], X, iters=150)
        if s < bests:
            bests = s
            bestv = v
    return bestv

def solve():
    t0 = time.time()
    BUDGET = 170.0
    rng = np.random.default_rng(7)
    n0, d = 219, 6
    x_best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=float)
            if v.shape == (n0, d):
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    x_best = v.copy()
        except Exception:
            pass
    if x_best is None:
        x_best = rng.standard_normal((n0, d))
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
        best_val = _max_cos(x_best)

    def consider(X):
        nonlocal x_best, best_val
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        c = _max_cos(X)
        if c < best_val:
            best_val = c
            x_best = X.copy()
        return c

    # ---- Path A: down to 200 then rebuild ----
    try:
        per = _per_max(x_best)
        order = np.argsort(per)  # asc
        # delete 18 worst + 1 rattler (smallest)
        del_idx = set(order[-18:].tolist() + [int(order[0])])
        # ensure 19 unique
        while len(del_idx) < 19:
            del_idx.add(int(rng.integers(0, n0)))
        del_idx = np.array(sorted(del_idx))
        keep = np.array([i for i in range(n0) if i not in set(del_idx.tolist())])
        Xcur = x_best[keep].copy()  # 200
        Xcur, _ = _fast_relax(Xcur, iters=900, lr0=0.025, margin0=0.005)
        if time.time() - t0 < BUDGET - 60:
            Xcur = _lbfgs_sm(Xcur, beta=250, maxiter=80)
            Xcur, _ = _fast_relax(Xcur, iters=300, lr0=0.01, margin0=0.002)
        # sequential add back to 219
        need = n0 - Xcur.shape[0]
        for k in range(need):
            if time.time() - t0 > BUDGET - 55:
                break
            v = _find_void(Xcur, rng, n_cand=3500, n_top=3)
            Xcur = np.vstack([Xcur, v.reshape(1, d)])
            Xcur /= np.linalg.norm(Xcur, axis=1, keepdims=True)
            beta = 200.0 + 700.0 * (k + 1) / max(1, need)
            Xcur, _ = _fast_relax(Xcur, iters=220, lr0=0.018, margin0=0.003)
            if time.time() - t0 < BUDGET - 40:
                Xcur = _lbfgs_sm(Xcur, beta=beta, maxiter=35)
                Xcur /= np.linalg.norm(Xcur, axis=1, keepdims=True)
            if (k + 1) % 5 == 0 and time.time() - t0 < BUDGET - 35:
                Xcur, _ = _lp_polish(Xcur, seconds=3.0, t0=t0, BUDGET=BUDGET)
        if Xcur.shape[0] == n0:
            Xcur, _ = _fast_relax(Xcur, iters=300, lr0=0.01, margin0=0.0015)
            if time.time() - t0 < BUDGET - 30:
                Xcur = _lbfgs_sm(Xcur, beta=900, maxiter=60)
            if time.time() - t0 < BUDGET - 20:
                Xcur, _ = _lp_polish(Xcur, seconds=6.0, t0=t0, BUDGET=BUDGET)
            consider(Xcur)
    except Exception:
        pass

    # ---- Path B: overfill to 228 then prune ----
    try:
        if time.time() - t0 < BUDGET - 35:
            Xover = x_best.copy()
            for _ in range(9):
                if time.time() - t0 > BUDGET - 30:
                    break
                v = _find_void(Xover, rng, n_cand=3000, n_top=3)
                Xover = np.vstack([Xover, v.reshape(1, d)])
                Xover /= np.linalg.norm(Xover, axis=1, keepdims=True)
            Xover, _ = _fast_relax(Xover, iters=700, lr0=0.022, margin0=0.004)
            if time.time() - t0 < BUDGET - 25:
                Xover = _lbfgs_sm(Xover, beta=350, maxiter=70)
            # prune back to 219
            while Xover.shape[0] > n0 and time.time() - t0 < BUDGET - 15:
                per2 = _per_max(Xover)
                worst = int(np.argmax(per2))
                Xover = np.delete(Xover, worst, axis=0)
                Xover, _ = _fast_relax(Xover, iters=180, lr0=0.015, margin0=0.0025)
                if Xover.shape[0] % 3 == 0 and time.time() - t0 < BUDGET - 18:
                    Xover = _lbfgs_sm(Xover, beta=600, maxiter=25)
            if Xover.shape[0] == n0:
                if time.time() - t0 < BUDGET - 10:
                    Xover, _ = _lp_polish(Xover, seconds=5.0, t0=t0, BUDGET=BUDGET)
                consider(Xover)
    except Exception:
        pass

    # ---- final squeeze of global best ----
    try:
        remain = BUDGET - (time.time() - t0) - 1.5
        if remain > 4:
            x_best, best_val2 = _lp_polish(x_best, seconds=min(remain * 0.6, 12.0), t0=t0, BUDGET=BUDGET)
            if best_val2 < best_val:
                best_val = best_val2
            if time.time() - t0 < BUDGET - 4:
                Xf, vf = _fast_relax(x_best, iters=250, lr0=0.006, margin0=0.001)
                if vf < best_val:
                    best_val, x_best = vf, Xf
                if time.time() - t0 < BUDGET - 2:
                    Xl = _lbfgs_sm(x_best, beta=1200, maxiter=40)
                    consider(Xl)
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
