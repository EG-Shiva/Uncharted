import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def solve():
    import time as _time
    import numpy as _np
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    t0 = _time.time()
    HARD_END = t0 + 175.0
    HOP_END = t0 + 135.0
    SQUEEZE_END = t0 + 155.0
    rng = _np.random.default_rng(12345)
    _np.random.seed(0)

    n, d = 250, 6
    try:
        best = _np.load("best.npy").astype(float)
    except Exception:
        best = rng.standard_normal((n, d))
    try:
        rec = _np.load("record.npy").astype(float)
    except Exception:
        rec = None
    if best.shape != (n, d):
        # try to reshape / truncate
        try:
            best = _np.resize(best, (n, d))
        except Exception:
            best = rng.standard_normal((n, d))
    # normalize
    nr = _np.linalg.norm(best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    best = best / nr
    best_val = _max_cos(best)
    if rec is not None and rec.shape == (n, d):
        nr2 = _np.linalg.norm(rec, axis=1, keepdims=True)
        nr2[nr2 == 0] = 1.0
        rec = rec / nr2
        rv = _max_cos(rec)
        if rv < best_val:
            best = rec.copy()
            best_val = rv

    iu0, iu1 = _np.triu_indices(n, 1)

    def refine(X_init, steps, b0, b1, lr, deadline):
        X = X_init.copy()
        m = _np.zeros_like(X)
        v = _np.zeros_like(X)
        b1m = 0.9
        b2m = 0.999
        eps = 1e-8
        tt = 0
        lr0 = lr
        for s in range(steps):
            if (s & 63) == 0 and _time.time() > deadline:
                break
            frac = s / (steps - 1) if steps > 1 else 1.0
            beta = b0 * ((b1 / b0) ** frac)
            G = X @ X.T
            cur_max = _np.max(G[iu0, iu1])
            G -= cur_max
            G *= beta
            _np.clip(G, -30.0, 0.0, out=G)
            _np.exp(G, out=G)
            _np.fill_diagonal(G, 0.0)
            s_sum = G.sum() * 0.5
            if s_sum < 1e-100:
                continue
            grad = (G @ X) / s_sum
            dot = _np.einsum('ij,ij->i', grad, X)[:, None]
            grad -= dot * X
            tt += 1
            m *= b1m
            m += (1.0 - b1m) * grad
            v *= b2m
            v += (1.0 - b2m) * (grad * grad)
            c1 = 1.0 - (b1m ** tt)
            c2 = 1.0 - (b2m ** tt)
            # Adam step
            vhat = v / c2
            mhat_scale = lr0 / c1
            # step = lr*mhat/(sqrt(vhat)+eps)
            step = mhat_scale * m / (_np.sqrt(vhat) + eps)
            # clip per-point step norm to 5*lr to avoid jumps at high beta
            sn = _np.linalg.norm(step, axis=1, keepdims=True) + 1e-16
            cap = 5.0 * lr0
            scale = _np.minimum(1.0, cap / (sn + 1e-16))
            # scale is (n,1)
            step = step * scale
            X -= step
            _np.divide(X, _np.linalg.norm(X, axis=1, keepdims=True), out=X)
            # keep momentum tangent
            md = _np.einsum('ij,ij->i', m, X)[:, None]
            m -= md * X
        return X

    def worst_order(X):
        G = X @ X.T
        _np.fill_diagonal(G, -2.0)
        rm = G.max(axis=1)
        return _np.argsort(-rm), rm

    # Stage 0: polish current best without kick (small)
    try:
        Xc = refine(best, 1200, 80.0, 800.0, 0.0015, HOP_END)
        vc = _max_cos(Xc)
        if vc < best_val:
            best = Xc
            best_val = vc
    except Exception:
        pass

    # Stage 1: basin hopping
    trial = 0
    sigmas = [0.004, 0.008, 0.015, 0.006, 0.025, 0.010]
    while _time.time() < HOP_END and trial < 60:
        trial += 1
        try:
            order, _ = worst_order(best)
        except Exception:
            order = _np.arange(n)
        X0 = best.copy()
        mode = trial % 6
        sigma = sigmas[trial % len(sigmas)]
        if mode == 4:
            # single-point reinsertion of worst point to random location
            w = int(order[0])
            # best void: try few random candidates pick farthest
            cand = rng.standard_normal((32, d))
            cand /= _np.linalg.norm(cand, axis=1, keepdims=True)
            # score by max cosine to others (excluding w)
            others = best.copy()
            # compute cos to candidates
            sc = cand @ others.T  # 32 x 250
            # for each cand, max over others except w
            # set column w to -2
            sc[:, w] = -2.0
            mx = sc.max(axis=1)
            bi = int(_np.argmin(mx))
            X0[w] = cand[bi]
            # also jitter neighbours slightly
            X0 += 0.002 * rng.standard_normal((n, d))
        elif mode == 5:
            # heavy kick on worst K
            K = 12 + (trial % 15)
            idx = order[:K]
            X0[idx] += (sigma * 3.0) * rng.standard_normal((K, d))
            mask = _np.ones(n, dtype=bool)
            mask[idx] = False
            X0[mask] += (sigma * 0.25) * rng.standard_normal((int(mask.sum()), d))
        else:
            X0 += sigma * rng.standard_normal((n, d))
            if mode == 1 or mode == 2:
                K = 20
                idx = order[:K]
                X0[idx] += sigma * rng.standard_normal((K, d))
        # normalize
        X0 /= _np.linalg.norm(X0, axis=1, keepdims=True)
        # choose anneal params deterministically varying
        if trial % 3 == 0:
            b0, b1, lr, steps = 50.0, 600.0, 0.003, 1400
        elif trial % 3 == 1:
            b0, b1, lr, steps = 80.0, 900.0, 0.002, 1600
        else:
            b0, b1, lr, steps = 40.0, 400.0, 0.004, 1200
        try:
            Xo = refine(X0, steps, b0, b1, lr, HOP_END)
            vo = _max_cos(Xo)
            if vo < best_val - 1e-12:
                best = Xo
                best_val = vo
                # exploit same basin a bit with tiny extra squeeze
                try:
                    Xe = refine(best, 600, 300.0, 1500.0, 0.0006, HOP_END)
                    ve = _max_cos(Xe)
                    if ve < best_val:
                        best = Xe
                        best_val = ve
                except Exception:
                    pass
        except Exception:
            continue

    # Stage 2: high-beta squeeze
    try:
        if _time.time() < SQUEEZE_END:
            Xs = refine(best, 1800, 400.0, 2500.0, 0.0005, SQUEEZE_END)
            vs = _max_cos(Xs)
            if vs < best_val:
                best = Xs
                best_val = vs
    except Exception:
        pass

    # Stage 3: trust-region LP polish with remaining time
    def lp_polish_lp(X, deadline):
        X = X / _np.linalg.norm(X, axis=1, keepdims=True)
        nn, dd = X.shape
        cur_best = _max_cos(X)
        radius = 1e-3
        iut0, iut1 = _np.triu_indices(nn, 1)
        Ktop = 2800
        while _time.time() < deadline and radius > 1e-14:
            Gfull = X @ X.T
            c = Gfull[iut0, iut1]
            mxx = c.max()
            # select top-K plus threshold
            if len(c) > Ktop:
                # argpartition for top K
                part = _np.argpartition(-c, Ktop)[:Ktop]
                # also ensure threshold inclusion for stability
                sel_idx = part
                I = iut0[sel_idx]
                J = iut1[sel_idx]
                cb = c[sel_idx]
            else:
                I, J = iut0, iut1
                cb = c
            p = len(I)
            nv = nn * dd + 1
            rows = _np.repeat(_np.arange(p), 2 * dd + 1)
            cols = _np.concatenate([I[:, None] * dd + _np.arange(dd), J[:, None] * dd + _np.arange(dd), _np.full((p, 1), nn * dd)], 1).ravel()
            vals = _np.concatenate([X[J], X[I], -_np.ones((p, 1))], 1).ravel()
            try:
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((X.ravel(), (_np.repeat(_np.arange(nn), dd), _np.arange(nn * dd))), shape=(nn, nv)).tocsr()
            except Exception:
                break
            obj = _np.zeros(nv)
            obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-cb, A_eq=A_eq, b_eq=_np.zeros(nn),
                              bounds=[(-radius, radius)] * (nn * dd) + [(None, None)], method="highs")
            except Exception:
                break
            if res.status != 0:
                radius *= 0.5
                continue
            y = X + res.x[:-1].reshape(nn, dd)
            y /= _np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
            if v < cur_best:
                X, cur_best, radius = y, v, min(radius * 1.5, 0.02)
            else:
                radius *= 0.5
            # early stop if time
            if _time.time() > deadline:
                break
        return X, cur_best

    try:
        if _time.time() < HARD_END - 5:
            Xlp, vlp = lp_polish_lp(best, HARD_END - 1.0)
            if vlp < best_val:
                best = Xlp
                best_val = vlp
    except Exception:
        pass

    best = best / _np.linalg.norm(best, axis=1, keepdims=True)
    # sanitize
    best = _np.nan_to_num(best, nan=0.0, posinf=0.0, neginf=0.0)
    nr = _np.linalg.norm(best, axis=1, keepdims=True)
    nr[nr < 1e-12] = 1.0
    best = best / nr
    return best
