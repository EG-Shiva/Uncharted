import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _per_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1)

def _smooth_gd(X, beta, iters, lr0, rng=None, noise0=0.0):
    n, d = X.shape
    cur = X.copy()
    lr = lr0
    for it in range(iters):
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        # softmax weights on full matrix (symmetric)
        # use upper via full for speed
        E = np.exp(beta * (G - m))
        # zero out far pairs for stability/sparsity (optional)
        # keep all for smooth phase
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E)) * 0.5 + 1e-300
        # Grad = (E @ cur)/ (2S)  (factor 2 cancels, scale invariant due to normalization)
        Grad = E @ cur
        proj = np.sum(Grad * cur, axis=1, keepdims=True)
        Gt = Grad - proj * cur
        gn = float(np.max(np.linalg.norm(Gt, axis=1)))
        if not np.isfinite(gn) or gn < 1e-14:
            break
        step = (lr / (gn + 1e-300)) * Gt
        if noise0 > 0 and rng is not None:
            decay = 1.0 - it / float(iters)
            nz = rng.standard_normal(cur.shape)
            nz -= np.sum(nz * cur, axis=1, keepdims=True) * cur
            nn = np.linalg.norm(nz, axis=1, keepdims=True) + 1e-12
            step = step + (noise0 * decay) * (nz / nn) * lr * 0.5
        cur = cur - step
        cur = _norm(cur)
    return cur

def _greedy(X, iters, lr0=0.02, margin0=0.004):
    X = _norm(X)
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
        Xn = _norm(Xn)
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

def _riesz_gd(X, pwr=8.0, iters=250, lr0=0.06):
    cur = _norm(X.copy())
    lr = lr0
    n = cur.shape[0]
    for it in range(iters):
        G = cur @ cur.T
        np.fill_diagonal(G, 1.0 - 1e-9)
        # r^2 = 2-2dot ; energy (1-dot)^(-p/2)
        one_minus = np.clip(1.0 - G, 1e-9, 4.0)
        # weight = (1-dot)^(-p/2-1)
        W = one_minus ** (-pwr / 2.0 - 1.0)
        np.fill_diagonal(W, 0.0)
        Grad = W @ cur
        proj = np.sum(Grad * cur, axis=1, keepdims=True)
        Gt = Grad - proj * cur
        gn = float(np.max(np.linalg.norm(Gt, axis=1)))
        if not np.isfinite(gn) or gn < 1e-14:
            break
        Xn = cur - (lr / gn) * Gt
        Xn = _norm(Xn)
        cur = Xn
        if it % 100 == 0:
            lr *= 0.9
    return cur

def solve():
    t0 = time.time()
    BUDGET = 170.0
    HOP_END = t0 + 148.0
    rng = np.random.default_rng(0)
    n, d = 219, 6
    iu0, iu1 = np.triu_indices(n, 1)

    x_best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=float)
            if v.shape == (n, d):
                v = _norm(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    x_best = v.copy()
        except Exception:
            pass
    if x_best is None:
        x_best = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(x_best)

    def consider(X):
        nonlocal x_best, best_val
        X = _norm(X)
        c = _max_cos(X)
        if c < best_val:
            best_val = c
            x_best = X.copy()
        return c

    def run_lbfgs(x0, beta, maxiter):
        y0 = x0.ravel().copy()
        def fun(y):
            Y = y.reshape(n, d)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Y / rn
            G = X @ X.T
            g = G[iu0, iu1]
            m = float(np.max(g))
            e = np.exp(beta * (g - m))
            S = float(np.sum(e)) + 1e-300
            loss = m + np.log(S) / beta
            w = e / S
            W = np.zeros((n, n), dtype=float)
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            dot = np.sum(GX * X, axis=1, keepdims=True)
            GY = (GX - dot * X) / rn
            return float(loss), GY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 20000})
            Y = res.x.reshape(n, d)
            Y = _norm(Y)
            return Y
        except Exception:
            return _norm(x0)

    def lp_polish(x, deadline, r_init=1.2e-3):
        x = _norm(x.copy())
        best = _max_cos(x)
        bx = x.copy()
        radius = r_init
        while time.time() < deadline and time.time() - t0 < BUDGET - 1.0:
            if radius < 1e-14:
                break
            Gf = x @ x.T
            c = Gf[iu0, iu1]
            m = float(np.max(c))
            thr = m - max(18.0 * radius, 1e-9)
            sel = c > thr
            I = iu0[sel]; J = iu1[sel]; csel = c[sel]
            p = len(I)
            if p == 0:
                break
            if p > 3800:
                k2 = 3800
                part = np.argpartition(-csel, k2 - 1)[:k2]
                I, J, csel = I[part], J[part], csel[part]
                p = k2
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
            obj = np.zeros(n * d + 1); obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = _norm(x + res.x[:-1].reshape(n, d))
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                bx = y.copy()
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        return bx, best

    hop = 0
    # quick initial squeeze to get contact info
    while time.time() < HOP_END and time.time() - t0 < BUDGET - 20:
        hop += 1
        if hop % 8 == 0:
            # fresh Riesz basin (different topology)
            try:
                Xr = _norm(rng.standard_normal((n, d)))
                pw = float(rng.choice([6.0, 8.0, 10.0, 12.0]))
                Xr = _riesz_gd(Xr, pwr=pw, iters=220, lr0=0.08)
                Xr = _smooth_gd(Xr, beta=70.0, iters=120, lr0=0.07)
                Xr = run_lbfgs(Xr, beta=300, maxiter=40)
                Xr, _ = _greedy(Xr, iters=120, lr0=0.015, margin0=0.003)
                consider(Xr)
            except Exception:
                pass
            continue
        # kick from global best (large topology change)
        try:
            X = x_best.copy()
            per = _per_max(X)
            order_asc = np.argsort(per)
            order_desc = order_asc[::-1]
            k = int(rng.integers(3, 10))  # 3..9 points
            # victims: bias to worst + overloaded neighbourhood
            worst_pool = order_desc[:25]
            nv = min(k - 1, len(worst_pool))
            victims = rng.choice(worst_pool, size=nv, replace=False).tolist() if nv > 0 else []
            while len(victims) < k:
                victims.append(int(rng.integers(0, n)))
            # sometimes relocate rattler (min per)
            r = float(rng.random())
            if r < 0.30:
                victims[0] = int(order_asc[0])
            elif r < 0.45:
                # cap cluster: pick random direction = random point, take its nearest neighbours
                seed = int(rng.integers(0, n))
                dots = X @ X[seed]
                cap = np.argsort(-dots)[:k]
                victims = cap.tolist()
            sigma = float(rng.choice([0.25, 0.40, 0.60, 0.90]))
            for vi in victims:
                nz = rng.standard_normal(d) * sigma
                # tangential component dominates; just add then renormalize
                nv2 = X[vi] + nz
                nn = float(np.linalg.norm(nv2))
                if nn < 1e-9 or not np.isfinite(nn):
                    nv2 = rng.standard_normal(d)
                    nn = float(np.linalg.norm(nv2))
                X[vi] = nv2 / nn
            if rng.random() < 0.55:
                X = X + rng.standard_normal(X.shape) * 0.018
                X = _norm(X)
            # anneal through gap: low beta -> mid beta (allow uphill in max)
            b1 = float(rng.choice([50.0, 70.0, 90.0]))
            X = _smooth_gd(X, beta=b1, iters=130, lr0=0.085, rng=rng, noise0=0.012)
            X = _smooth_gd(X, beta=180.0, iters=130, lr0=0.05)
            # re-jam
            if time.time() - t0 > BUDGET - 25:
                consider(X)
                break
            beta2 = float(rng.choice([350.0, 500.0, 650.0]))
            X = run_lbfgs(X, beta=beta2, maxiter=45)
            X = _norm(X)
            # short greedy to read true max
            Xg, cg = _greedy(X, iters=100, lr0=0.012, margin0=0.002)
            # keep better of smooth vs greedy
            c_raw = _max_cos(X)
            if cg < c_raw:
                X = Xg
                c_raw = cg
            consider(X)
            # chain: if close to best, do extra sharpening + LP micro-step occasionally
            if c_raw < best_val + 0.0006 and hop % 5 == 0 and time.time() < HOP_END - 10:
                X2 = run_lbfgs(X, beta=900.0, maxiter=30)
                consider(X2)
        except Exception:
            continue

    # final squeeze of global best
    try:
        remain = BUDGET - (time.time() - t0) - 1.0
        if remain > 5:
            xb, vb = lp_polish(x_best, deadline=time.time() + min(remain * 0.55, 10.0))
            if vb < best_val:
                best_val = vb
                x_best = xb
            if time.time() - t0 < BUDGET - 4:
                Xf = run_lbfgs(x_best, beta=1200.0, maxiter=50)
                consider(Xf)
            if time.time() - t0 < BUDGET - 3:
                Xg, cg = _greedy(x_best, iters=250, lr0=0.008, margin0=0.0012)
                consider(Xg)
    except Exception:
        pass
    x_best = _norm(x_best)
    return x_best
