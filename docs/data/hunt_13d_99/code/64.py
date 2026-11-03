import time
import numpy as np
from scipy.optimize import minimize

def _norm_rows(V):
    n = np.linalg.norm(V, axis=1, keepdims=True)
    n = np.where(n < 1e-15, 1.0, n)
    return V / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _per_point_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _polish_once(V0, alpha, maxiter):
    N, d = V0.shape
    x0 = V0.ravel().astype(float).copy()
    def fun(v):
        U = v.reshape(N, d)
        nr = np.linalg.norm(U, axis=1, keepdims=True)
        nr = np.where(nr < 1e-15, 1.0, nr)
        Y = U / nr
        G = Y @ Y.T
        np.fill_diagonal(G, -10.0)
        mx = float(G.max())
        E = np.exp(alpha * (G - mx))
        np.fill_diagonal(E, 0.0)
        Sm = float(E.sum() * 0.5) + 1e-300
        L = mx + np.log(Sm) / alpha
        GX = (E @ Y) / Sm
        dot = np.sum(GX * Y, axis=1, keepdims=True)
        GU = (GX - dot * Y) / nr
        return float(L), GU.ravel().astype(float)
    try:
        r = minimize(fun, x0, jac=True, method='L-BFGS-B',
                     options={'maxiter': int(maxiter), 'ftol': 1e-20,
                              'gtol': 1e-7, 'maxcor': 10,
                              'maxfun': int(maxiter) * 8})
        U = r.x.reshape(N, d)
        return _norm_rows(U)
    except Exception:
        return V0.copy()

def solve():
    t0 = time.time()
    deadline = t0 + 165.0
    rng = np.random.RandomState(0)
    # load incumbents
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (99, 13):
                cands.append(_norm_rows(a))
        except Exception:
            pass
    if not cands:
        cands.append(_norm_rows(rng.standard_normal(size=(99, 13))))
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best = cands[bi].copy()
    best_val = float(vals[bi])
    # quick baseline polish of incumbent with moderate-high alpha
    # (cheap, often shaves 1e-7..1e-6)
    try:
        if deadline - time.time() > 20:
            B = _polish_once(best, 150.0, 80)
            v = _max_cos(B)
            if v < best_val:
                best, best_val = B, v
            if deadline - time.time() > 15:
                B2 = _polish_once(best, 600.0, 100)
                v2 = _max_cos(B2)
                if v2 < best_val:
                    best, best_val = B2, v2
    except Exception:
        pass
    it = 0
    sigmas = [0.35, 0.18, 0.08, 0.03, 0.12, 0.25, 0.05, 0.01]
    while time.time() < deadline - 12.0:
        it += 1
        remain = deadline - time.time()
        if remain < 13.0:
            break
        mode = it % 5
        Y = None
        if mode == 0:
            # global shake
            s = sigmas[(it // 5) % len(sigmas)]
            Y = best + s * rng.standard_normal(size=(99, 13))
            Y = _norm_rows(Y)
        elif mode == 1:
            # subset shake
            Y = best.copy()
            k = 12 if (it % 2 == 0) else 6
            idx = rng.choice(99, size=k, replace=False)
            s = 0.4 if (it % 4 < 2) else 0.15
            Y[idx] = Y[idx] + s * rng.standard_normal(size=(k, 13))
            Y = _norm_rows(Y)
        elif mode == 2:
            # worst-point void reinsertion (attack overloaded / plateau points)
            Y = best.copy()
            ppm = _per_point_max(best)
            # pick 3 worst + 1 random rattler-ish
            order = np.argsort(-ppm)
            nw = 3
            targets = order[:nw].copy()
            for t in targets:
                # propose many random vectors, keep one minimizing max dot to rest
                mask = np.ones(99, dtype=bool); mask[t] = False
                R = _norm_rows(rng.standard_normal(size=(40, 13)))
                dots = R @ Y[mask].T
                md = dots.max(axis=1)
                Y[t] = R[int(np.argmin(md))]
            # small jitter others to break symmetry
            Y = Y + 0.01 * rng.standard_normal(size=(99, 13))
            Y = _norm_rows(Y)
        elif mode == 3:
            # random replacement of a few points
            Y = best.copy()
            k = 5 if (it % 2 == 0) else 2
            idx = rng.choice(99, size=k, replace=False)
            Y[idx] = _norm_rows(rng.standard_normal(size=(k, 13)))
            Y = _norm_rows(Y)
        else:
            # fresh random start occasionally to find new basin
            if it % 15 == 0:
                Y = _norm_rows(rng.standard_normal(size=(99, 13)))
            else:
                s = 0.07
                Y = best + s * rng.standard_normal(size=(99, 13))
                Y = _norm_rows(Y)
        # annealed polish: fast 2-stage for screening
        try:
            # stage 1 low alpha to rebalance
            Yp = _polish_once(Y, 80.0, 50)
            if time.time() > deadline - 12.0:
                v = _max_cos(Yp)
                if v < best_val:
                    best, best_val = Yp, v
                break
            Yp = _polish_once(Yp, 300.0, 70)
            v = _max_cos(Yp)
            if v < best_val - 2e-9:
                # extra high-alpha squeeze on promising candidate
                if time.time() < deadline - 20.0:
                    Yq = _polish_once(Yp, 900.0, 80)
                    vq = _max_cos(Yq)
                    if vq < v:
                        Yp, v = Yq, vq
                best, best_val = Yp, v
            elif v < best_val:
                best, best_val = Yp, v
        except Exception:
            continue
    # final squeeze of best with high alpha if time permits
    try:
        if time.time() < deadline - 10.0:
            B = _polish_once(best, 1200.0, 60)
            v = _max_cos(B)
            if v < best_val:
                best, best_val = B, v
    except Exception:
        pass
    best = _norm_rows(best)
    # guard NaN
    best = np.where(np.isfinite(best), best, 0.0)
    best = _norm_rows(best)
    return best
