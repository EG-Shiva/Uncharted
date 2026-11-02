import numpy as np
import time
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X, iu=None):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def solve():
    import time as _time
    import numpy as _np
    from scipy.optimize import linprog as _linprog
    from scipy.sparse import coo_matrix as _coo
    t0 = _time.time()
    HARD = 172.0  # leave margin for 180s
    rng = _np.random.default_rng(20261101)
    n, d = 78, 15
    # load start
    try:
        Xb = _np.load("best.npy")
    except Exception:
        try:
            Xb = _np.load("record.npy")
        except Exception:
            Xb = rng.standard_normal((n, d))
    Xb = _np.asarray(Xb, dtype=float)
    if Xb.shape != (n, d):
        # fallback reshape/random
        tmp = rng.standard_normal((n, d))
        Xb = tmp
    Xb = Xb / _np.linalg.norm(Xb, axis=1, keepdims=True)
    iu_all = _np.triu_indices(n, 1)
    iu_r, iu_c = iu_all

    def max_cos(X):
        G = X @ X.T
        _np.fill_diagonal(G, -2.0)
        return float(G.max())

    def lp_polish(X, seconds, r0=1e-3):
        X = X / _np.linalg.norm(X, axis=1, keepdims=True)
        best = max_cos(X)
        radius = r0
        start = _time.time()
        nv = n * d + 1
        obj = _np.zeros(nv); obj[-1] = 1.0
        bnds = None
        while _time.time() - start < seconds and _time.time() - t0 < HARD - 2:
            if radius < 1e-14:
                break
            Gc = (X @ X.T)[iu_r, iu_c]
            m = float(Gc.max())
            thr = m - max(20.0 * radius, 1e-9)
            sel = Gc > thr
            # limit constraints for speed
            if int(sel.sum()) > 1200:
                # keep tightest 1200
                idx = _np.argsort(Gc)
                keep = _np.zeros_like(sel, dtype=bool)
                keep[idx[-1200:]] = True
                sel = sel & keep
            I = iu_r[sel]; J = iu_c[sel]
            p = len(I)
            if p == 0:
                break
            csel = Gc[sel]
            rows = _np.repeat(_np.arange(p), 2 * d + 1)
            cols = _np.concatenate([I[:, None] * d + _np.arange(d),
                                    J[:, None] * d + _np.arange(d),
                                    _np.full((p, 1), n * d)], 1).ravel()
            vals = _np.concatenate([X[J], X[I], -_np.ones((p, 1))], 1).ravel()
            A_ub = _coo((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = _coo((X.ravel(), (_np.repeat(_np.arange(n), d), _np.arange(n * d))), shape=(n, nv)).tocsr()
            bnds = [(-radius, radius)] * (n * d) + [(None, None)]
            try:
                res = _linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=_np.zeros(n),
                               bounds=bnds, method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            Y = X + res.x[:-1].reshape(n, d)
            Y = Y / _np.linalg.norm(Y, axis=1, keepdims=True)
            v = max_cos(Y)
            if v < best:
                X, best = Y, v
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        return X

    def smooth_anneal(X, steps, T0, T1, lr):
        X = X / _np.linalg.norm(X, axis=1, keepdims=True)
        best_m = max_cos(X)
        best_s = X.copy()
        # geometric temp schedule
        for s in range(steps):
            if _time.time() - t0 > HARD - 5:
                break
            frac = s / max(steps - 1, 1)
            T = T0 * ((T1 / T0) ** frac)
            G = X @ X.T
            # max over upper
            m = float(G[iu_r, iu_c].max())
            if m < best_m:
                best_m = m
                best_s = X.copy()
            E = _np.exp((G - m) / T)
            _np.fill_diagonal(E, 0.0)
            S = float(E.sum() * 0.5)
            if S <= 0 or not _np.isfinite(S):
                break
            Grad = (E / S) @ X
            # tangent projection
            dot = (Grad * X).sum(axis=1, keepdims=True)
            Gt = Grad - dot * X
            # clip per-row to avoid blowup
            gn = _np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
            # limit step to 0.03
            scale = _np.minimum(1.0, 0.03 / (lr * gn + 1e-12))
            X = X - lr * Gt * scale
            # renorm
            nr = _np.linalg.norm(X, axis=1, keepdims=True)
            nr[~_np.isfinite(nr)] = 1.0
            nr[nr < 1e-12] = 1.0
            X = X / nr
        vf = max_cos(X)
        if vf < best_m:
            return X, vf
        return best_s, best_m

    def largest_holes(Y, k, M=15000):
        # find k hole directions w.r.t Y excluding replaced indices later
        R = rng.standard_normal((M, d))
        R = R / _np.linalg.norm(R, axis=1, keepdims=True)
        return R

    best_val = max_cos(Xb)
    # short initial polish
    Xb = lp_polish(Xb, seconds=2.0)
    best_val = max_cos(Xb)

    NITER = 80
    for it in range(NITER):
        if _time.time() - t0 > 150.0:
            break
        mode = it % 5
        Y = Xb.copy()
        if mode == 0:
            # tiny global shake
            Y = Y + 0.007 * rng.standard_normal(Y.shape)
            Y = Y / _np.linalg.norm(Y, axis=1, keepdims=True)
            Y, _ = smooth_anneal(Y, steps=200, T0=0.006, T1=0.0012, lr=0.4)
            Y = lp_polish(Y, seconds=1.0)
        elif mode == 1:
            # shake lowest-contact points hard
            G = Y @ Y.T
            _np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cnt = (G > m - 3e-4).sum(axis=1)
            idx = _np.argsort(cnt)[:6]
            Y[idx] = Y[idx] + 0.12 * rng.standard_normal((len(idx), d))
            Y[idx] = Y[idx] / _np.linalg.norm(Y[idx], axis=1, keepdims=True)
            mask = _np.ones(n, dtype=bool); mask[idx] = False
            Y[mask] = Y[mask] + 0.003 * rng.standard_normal((int(mask.sum()), d))
            Y = Y / _np.linalg.norm(Y, axis=1, keepdims=True)
            Y, _ = smooth_anneal(Y, steps=350, T0=0.014, T1=0.0018, lr=0.6)
            Y = lp_polish(Y, seconds=1.0)
        elif mode == 2:
            # hole reinsertion for worst rows
            G = Y @ Y.T
            _np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            idx_rep = _np.argsort(rowmax)[-3:]
            R = largest_holes(Y, len(idx_rep), M=15000)
            # sequential best-hole insertion
            remaining = set(idx_rep.tolist())
            for kk in idx_rep:
                D = R @ Y.T
                # ignore columns still to be replaced (except kk already? we want holes vs kept)
                for c in remaining:
                    D[:, c] = -10.0
                # also consider already inserted holes (they are in Y, not ignored)
                scores = D.max(axis=1)
                bi = int(_np.argmin(scores))
                Y[kk] = R[bi]
                remaining.discard(kk)
            Y = Y / _np.linalg.norm(Y, axis=1, keepdims=True)
            Y, _ = smooth_anneal(Y, steps=350, T0=0.012, T1=0.0015, lr=0.55)
            Y = lp_polish(Y, seconds=1.0)
        elif mode == 3:
            # medium global kick to change backbone topology
            Y = Y + 0.025 * rng.standard_normal(Y.shape)
            Y = Y / _np.linalg.norm(Y, axis=1, keepdims=True)
            Y, _ = smooth_anneal(Y, steps=300, T0=0.010, T1=0.0015, lr=0.5)
            Y = lp_polish(Y, seconds=1.0)
        else:
            # random subset shake
            k = 8
            idx = rng.choice(n, size=k, replace=False)
            Y[idx] = Y[idx] + 0.06 * rng.standard_normal((k, d))
            Y[idx] = Y[idx] / _np.linalg.norm(Y[idx], axis=1, keepdims=True)
            Y = Y / _np.linalg.norm(Y, axis=1, keepdims=True)
            Y, _ = smooth_anneal(Y, steps=300, T0=0.009, T1=0.0012, lr=0.5)
            Y = lp_polish(Y, seconds=1.0)
        v = max_cos(Y)
        if v < best_val - 1e-12:
            Xb = Y
            best_val = v

    # final long polish
    remain = HARD - (_time.time() - t0) - 1.0
    if remain > 3:
        Xb = lp_polish(Xb, seconds=min(remain, 20.0), r0=2e-3)
        # extra fine polish with tiny radius
        Xb = lp_polish(Xb, seconds=min(5.0, max(1.0, remain - 15.0)), r0=5e-4)
    Xb = Xb / _np.linalg.norm(Xb, axis=1, keepdims=True)
    return Xb
