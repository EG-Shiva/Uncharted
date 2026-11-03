import time
import numpy as np
from scipy.optimize import minimize

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _per_point_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1)

def _lse_polish(X0, alpha, maxiter, ftol=1e-20, gtol=1e-7):
    N, d = X0.shape
    U0 = X0.ravel().astype(float).copy()
    alpha = float(alpha)
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
    try:
        res = minimize(fun, U0, method='L-BFGS-B', jac=True,
                       options={'maxiter': int(maxiter), 'ftol': ftol,
                                'gtol': gtol, 'maxfun': int(maxiter)*6, 'maxcor': 8})
        U = res.x.reshape(N, d)
        return _norm_rows(U)
    except Exception:
        return X0.copy()

def solve():
    t0 = time.time()
    HARD = 172.0
    n, d = 99, 13
    rng = np.random.default_rng(0)

    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None

    cands = []
    if xb is not None:
        try:
            a = _norm_rows(np.asarray(xb, dtype=float))
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                cands.append(a)
        except Exception:
            pass
    if xr is not None:
        try:
            a = _norm_rows(np.asarray(xr, dtype=float))
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_norm_rows(rng.standard_normal(size=(n, d))))
    vals = [_max_cos(c) for c in cands]
    best = cands[int(np.argmin(vals))].copy()
    best_val = float(np.min(vals))
    best = _norm_rows(best)

    def time_left():
        return HARD - (time.time() - t0)

    # Phase A: polish incumbent with annealing
    if time_left() > 20:
        try:
            X1 = _lse_polish(best, 120.0, 250, gtol=1e-6)
            v1 = _max_cos(X1)
            if v1 < best_val and np.all(np.isfinite(X1)):
                best, best_val = X1, v1
        except Exception:
            pass
    if time_left() > 18:
        try:
            X1 = _lse_polish(best, 350.0, 250, gtol=1e-7)
            v1 = _max_cos(X1)
            if v1 < best_val and np.all(np.isfinite(X1)):
                best, best_val = X1, v1
        except Exception:
            pass

    # helper: largest-hole replacement for worst point(s)
    def hole_replace(Y, n_replace=1, n_trial=2000):
        # find worst points by per-point max
        try:
            ppm = _per_point_max(Y)
            worst = np.argsort(-ppm)[:n_replace]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[worst] = False
            Ykeep = Y[keep_mask]  # (n-n_replace, d)
            R = _norm_rows(rng.standard_normal(size=(n_trial, d)))
            # score = max dot to kept set
            S = R @ Ykeep.T  # (trial, kept)
            s_max = np.max(S, axis=1)
            # also penalize closeness among replacements: greedy pick
            chosen = []
            # pick sequentially with mutual repulsion
            pool = np.argsort(s_max)  # ascending best first
            # take first, then enforce separation for more than 1
            sel_idx = []
            for idx in pool:
                v = R[idx]
                ok = True
                for c in chosen:
                    if float(v @ c) > 0.5:
                        ok = False
                        break
                if ok:
                    chosen.append(v)
                    sel_idx.append(idx)
                    if len(chosen) >= n_replace:
                        break
            if len(chosen) < n_replace:
                for idx in pool:
                    if idx not in sel_idx:
                        chosen.append(R[idx])
                        if len(chosen) >= n_replace:
                            break
            Yn = Y.copy()
            for k, w in enumerate(worst):
                Yn[w] = chosen[k] / np.linalg.norm(chosen[k])
            return _norm_rows(Yn)
        except Exception:
            return Y

    def greedy_single(Y, cur_val, steps, sigma, rs):
        # in-place style hill climb on single rows
        G = Y @ Y.T
        np.fill_diagonal(G, -2.0)
        cur = cur_val
        for _ in range(steps):
            i = int(rs.integers(0, n))
            xi = Y[i].copy()
            pert = rs.standard_normal(size=(d,)) * sigma
            # make tangent-ish: remove radial component
            pert = pert - np.dot(pert, xi) * xi
            v = xi + pert
            nv = np.linalg.norm(v)
            if nv < 1e-12 or not np.finite(nv):
                continue
            v /= nv
            dots = Y @ v
            dots[i] = -2.0
            nm = float(np.max(dots))
            # also need max among pairs not involving i: precompute
            # cheap: max of G excluding row/col i
            # compute quickly every step? G max excl i can be maintained,
            # but n=99 small: compute max of G with row/col i masked
            # To avoid O(n^2) full recompute, estimate:
            # other_max = current G max excluding i (compute on demand every step via masked max, O(n^2)=10k, fine)
            # Use G matrix (stale except row i). Since Y unchanged except maybe i, G is valid.
            # masked max:
            # we can compute other_max = max of G without i
            # G is 99x99, masked max costs ~10k, fine for few thousand steps.
            Gwo = G.copy()
            Gwo[i, :] = -2.0
            Gwo[:, i] = -2.0
            other = float(np.max(Gwo))
            tot = nm if nm > other else other
            if tot < cur - 1e-14:
                Y[i] = v
                # update G row/col i
                G[i, :] = dots
                G[:, i] = dots
                G[i, i] = -2.0
                cur = tot
        return Y, cur

    # Phase B: basin hopping
    it = 0
    sigmas = [0.004, 0.009, 0.018, 0.032, 0.06]
    while time_left() > 8:
        it += 1
        tl = time_left()
        # allocate iters based on time left
        if tl < 12:
            # final emergency: only greedy refine best
            try:
                best, best_val = greedy_single(best.copy(), best_val, 3000, 1.5e-4, rng)
            except Exception:
                pass
            break
        sigma = sigmas[it % len(sigmas)]
        # with some probability use larger shake early, smaller late
        Y = best + sigma * rng.standard_normal(size=(n, d))
        Y = _norm_rows(Y)
        # hole replacement on ~40% of iterations
        if (it % 5 == 2) or (it % 7 == 3):
            try:
                nr = 1 if (it % 2 == 0) else 2
                Y = hole_replace(Y, n_replace=nr, n_trial=2500)
            except Exception:
                pass
        # quick annealed polish, fewer iters for speed
        try:
            Y = _lse_polish(Y, 130.0, 110, gtol=1e-6)
        except Exception:
            continue
        if time_left() < 6:
            v = _max_cos(Y)
            if v < best_val:
                best, best_val = Y, v
            break
        try:
            Y = _lse_polish(Y, 380.0, 110, gtol=1e-6)
        except Exception:
            continue
        v = _max_cos(Y)
        # greedy single-point shave (fast, targets the 1-pair bottleneck)
        if time_left() > 6:
            try:
                # adapt sigma: smaller when already good
                gs = 3e-4 if v < best_val + 1e-4 else 6e-4
                Y, v = greedy_single(Y, v, 1200, gs, rng)
            except Exception:
                pass
        if v < best_val - 1e-14:
            best, best_val = Y, v
        # also try to improve best itself with greedy occasionally even without shake
        if it % 6 == 0 and time_left() > 10:
            try:
                Bn, vn = greedy_single(best.copy(), best_val, 1500, 2e-4, rng)
                if vn < best_val - 1e-14:
                    best, best_val = Bn, vn
            except Exception:
                pass

    best = _norm_rows(best)
    return best
