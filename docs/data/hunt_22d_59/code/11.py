import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def solve():
    import time as _tm
    import numpy as _np
    np = _np
    t0 = _tm.time()
    DEADLINE = t0 + 168.0
    np.random.seed(0)
    rng = np.random.default_rng(0)

    # load
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (59, 22) and np.all(np.isfinite(a)):
                a = _norm(a)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal((59, 22))
        cands.append(_norm(a))
    vals = [_max_cos(c) for c in cands]
    best = cands[int(np.argmin(vals))].copy()
    best_val = float(min(vals))
    n, d = 59, 22

    def adam_polish(X0, steps, a0, a1, lr0, lr1):
        X = X0.copy()
        m = np.zeros_like(X)
        v = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        la0, la1 = np.log(a0), np.log(a1)
        ll0, ll1 = np.log(lr0), np.log(lr1)
        cur_best = X.copy()
        cur_val = _max_cos(X)
        for t in range(1, steps + 1):
            if (t & 255) == 0 and _tm.time() > DEADLINE - 3:
                break
            frac = t / steps
            alpha = float(np.exp(la0 + (la1 - la0) * frac))
            lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            M = float(np.max(G))
            G -= M
            G *= alpha
            np.exp(G, out=G)
            np.fill_diagonal(G, 0.0)
            S = float(np.sum(G))
            if not np.isfinite(S) or S <= 0:
                continue
            Gr = (G @ X) / S
            dot = np.sum(Gr * X, axis=1, keepdims=True)
            Gt = Gr - dot * X
            m *= b1
            m += (1.0 - b1) * Gt
            v *= b2
            v += (1.0 - b2) * (Gt * Gt)
            c1 = 1.0 - (b1 ** t)
            c2 = 1.0 - (b2 ** t)
            mc = m / c1
            vc = v / c2
            X -= lr * mc / (np.sqrt(vc) + eps)
            nn = np.linalg.norm(X, axis=1, keepdims=True)
            nn[nn == 0] = 1.0
            X /= nn
            if (t % 400 == 0) or (t == steps):
                cv = _max_cos(X)
                if cv < cur_val:
                    cur_val = cv
                    cur_best = X.copy()
        fv = _max_cos(X)
        if fv < cur_val:
            cur_val = fv
            cur_best = X.copy()
        return cur_best, cur_val

    def lbfgs_polish(X0, beta, maxiter):
        try:
            from scipy.optimize import minimize
        except Exception:
            return X0.copy(), _max_cos(X0)
        Z0 = X0.copy().ravel()
        nn0, dd0 = X0.shape
        def fun_g(zf):
            Z = zf.reshape(nn0, dd0)
            rn = np.sqrt(np.sum(Z * Z, axis=1, keepdims=True))
            rn[rn == 0] = 1.0
            X = Z / rn
            G = X @ X.T
            np.fill_diagonal(G, -10.0)
            gm = float(np.max(G))
            E = G - gm
            E *= beta
            # clip low end for exp stability
            np.exp(E, out=E)
            np.fill_diagonal(E, 0.0)
            S = float(np.sum(E))
            if S <= 0 or not np.isfinite(S):
                return 1e9, np.zeros_like(zf)
            L = gm + np.log(S) / beta
            E /= S
            GX = E @ X
            GX *= 2.0
            dot = np.sum(GX * X, axis=1, keepdims=True)
            GXt = GX - dot * X
            DZ = GXt / rn
            return float(L), DZ.ravel()
        try:
            res = minimize(fun_g, Z0, jac=True, method='L-BFGS-B',
                           options={'maxiter': int(maxiter), 'ftol': 1e-22, 'gtol': 1e-14, 'maxfun': 100000})
            Z = res.x.reshape(nn0, dd0)
            X = _norm(Z)
            return X, _max_cos(X)
        except Exception:
            return X0.copy(), _max_cos(X0)

    def deep_polish(X, light=True):
        # one adam + lbfgs chain
        if light:
            Y, vy = adam_polish(X, steps=2200, a0=120, a1=900, lr0=0.004, lr1=0.0004)
            if _tm.time() > DEADLINE - 8:
                return Y, vy
            Y2, v2 = lbfgs_polish(Y, beta=900.0, maxiter=120)
            if v2 < vy:
                return Y2, v2
            return Y, vy
        else:
            Y, vy = adam_polish(X, steps=5000, a0=80, a1=1200, lr0=0.005, lr1=0.0003)
            for be, mi in ((500, 150), (1200, 150), (2500, 120)):
                if _tm.time() > DEADLINE - 8:
                    break
                Y2, v2 = lbfgs_polish(Y, beta=be, maxiter=mi)
                if v2 < vy:
                    vy = v2
                    Y = Y2
            return Y, vy

    def hole_replace(X, kill_idx, n_cand=3000):
        Y = X.copy()
        keep = [i for i in range(n) if i not in set(kill_idx.tolist() if hasattr(kill_idx, 'tolist') else list(kill_idx))]
        Xk = X[keep]
        for k in (kill_idx.tolist() if hasattr(kill_idx, 'tolist') else list(kill_idx)):
            C = rng.standard_normal((n_cand, d))
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            sc = C @ Xk.T
            mx = np.max(sc, axis=1)
            bi = int(np.argmin(mx))
            # local refine of hole: few projected steps toward minimizing max
            Y[k] = C[bi]
            Xk = np.vstack([Xk, Y[k:k+1]])
        return Y

    # initial deep polish of record
    best, best_val = adam_polish(best, steps=5000, a0=60, a1=700, lr0=0.006, lr1=0.0005)
    if _tm.time() < DEADLINE - 10:
        for be, mi in ((600, 150), (1500, 150)):
            Y2, v2 = lbfgs_polish(best, beta=be, maxiter=mi)
            if v2 < best_val:
                best, best_val = Y2, v2
            if _tm.time() > DEADLINE - 10:
                break

    it = 0
    sigmas = [0.006, 0.015, 0.035, 0.008, 0.06, 0.012]
    while _tm.time() < DEADLINE - 9:
        it += 1
        remain = DEADLINE - _tm.time()
        if remain < 8:
            break
        mode = it % 8
        G = best @ best.T
        np.fill_diagonal(G, -2.0)
        rowmax = np.max(G, axis=1)
        # contacts count with tol
        cnt = np.sum(G > (best_val - 0.004), axis=1)
        order_worst = np.argsort(-rowmax)
        order_loaded = np.argsort(-cnt)
        Y = None
        if mode in (0, 4):
            sigma = sigmas[(it // 8) % len(sigmas)]
            noise = rng.standard_normal((n, d))
            # boost shake on worst / overloaded rows
            boost = np.ones((n, 1))
            boost[order_worst[:6]] = 3.0
            Y = best + (sigma * boost) * noise
            Y = _norm(Y)
        elif mode in (1, 5):
            k = 1 if (it % 2 == 0) else 2
            kill = order_worst[:k]
            Y = hole_replace(best, kill, n_cand=4000)
            # small jitter others
            Y += 0.002 * rng.standard_normal((n, d))
            Y = _norm(Y)
        elif mode in (2, 6):
            k = 3
            # mix worst and loaded
            kill = np.unique(np.concatenate([order_worst[:2], order_loaded[:2]]))[:k]
            Y = hole_replace(best, kill, n_cand=4000)
            Y = _norm(Y)
        else:
            # single-point big jump to random hole + jitter rest
            k = int(rng.integers(0, n))
            kill = np.array([k])
            if (it % 3) == 0:
                # pure random reinit of one point
                Y = best.copy()
                Y[k] = rng.standard_normal(d)
                Y = _norm(Y)
            else:
                Y = hole_replace(best, kill, n_cand=6000)
            # light global jitter
            Y += 0.003 * rng.standard_normal((n, d))
            Y = _norm(Y)

        Yp, Yv = deep_polish(Y, light=True)
        if Yv < best_val - 1e-12:
            best, best_val = Yp, Yv
        # occasional intensify around new best
        if (it % 12) == 0 and _tm.time() < DEADLINE - 20:
            Yp2, Yv2 = deep_polish(best, light=False)
            if Yv2 < best_val:
                best, best_val = Yp2, Yv2

    # final squeeze
    if _tm.time() < DEADLINE - 2:
        Y, vy = adam_polish(best, steps=2500, a0=900, a1=2500, lr0=0.0012, lr1=0.00012)
        if vy < best_val:
            best, best_val = Y, vy
    if _tm.time() < DEADLINE - 1:
        for be in (1200, 2500):
            Y2, v2 = lbfgs_polish(best, beta=be, maxiter=150)
            if v2 < best_val:
                best, best_val = Y2, v2
            if _tm.time() > DEADLINE - 1:
                break

    best = _norm(best)
    return best
