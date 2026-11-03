import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    HARD_END = t0 + 175.0
    HOP_END = t0 + 95.0
    GREEDY_END = t0 + 160.0
    LP_END = t0 + 174.0
    rng = _np.random.default_rng(7)

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
        try:
            best = _np.resize(best, (n, d))
        except Exception:
            best = rng.standard_normal((n, d))
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
        for s in range(steps):
            if (s & 63) == 0 and _time.time() > deadline:
                break
            frac = s / max(1, steps - 1)
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
            vhat = v / c2
            step = (lr / c1) * m / (_np.sqrt(vhat) + eps)
            sn = _np.linalg.norm(step, axis=1, keepdims=True) + 1e-16
            cap = 5.0 * lr
            scale = _np.minimum(1.0, cap / (sn + 1e-16))
            step = step * scale
            X -= step
            _np.divide(X, _np.linalg.norm(X, axis=1, keepdims=True), out=X)
            md = _np.einsum('ij,ij->i', m, X)[:, None]
            m -= md * X
        return X

    def worst_order(X):
        G = X @ X.T
        _np.fill_diagonal(G, -2.0)
        rm = G.max(axis=1)
        return _np.argsort(-rm), rm, G

    # Stage 0: quick polish
    try:
        Xc = refine(best, 800, 80.0, 800.0, 0.0015, HOP_END)
        vc = _max_cos(Xc)
        if vc < best_val:
            best = Xc
            best_val = vc
    except Exception:
        pass

    # Stage 1: basin hopping (short)
    trial = 0
    sigmas = [0.004, 0.008, 0.015, 0.006, 0.025, 0.010]
    while _time.time() < HOP_END and trial < 40:
        trial += 1
        try:
            order, _, _ = worst_order(best)
        except Exception:
            order = _np.arange(n)
        X0 = best.copy()
        mode = trial % 6
        sigma = sigmas[trial % len(sigmas)]
        if mode == 4:
            w = int(order[0])
            cand = rng.standard_normal((32, d))
            cand /= _np.linalg.norm(cand, axis=1, keepdims=True)
            sc = cand @ best.T
            sc[:, w] = -2.0
            mx = sc.max(axis=1)
            bi = int(_np.argmin(mx))
            X0[w] = cand[bi]
            X0 += 0.002 * rng.standard_normal((n, d))
        elif mode == 5:
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
        X0 /= _np.linalg.norm(X0, axis=1, keepdims=True)
        if trial % 3 == 0:
            b0, b1, lr, steps = 50.0, 600.0, 0.003, 1300
        elif trial % 3 == 1:
            b0, b1, lr, steps = 80.0, 900.0, 0.002, 1500
        else:
            b0, b1, lr, steps = 40.0, 400.0, 0.004, 1100
        try:
            Xo = refine(X0, steps, b0, b1, lr, HOP_END)
            vo = _max_cos(Xo)
            if vo < best_val - 1e-12:
                best = Xo
                best_val = vo
        except Exception:
            continue

    # Stage 2: greedy true-minimax single-point peak shaving
    # Very fast: each trial needs only one matrix-vector product.
    try:
        # precompute G for exclusion values
        sig_list = [0.003, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002]
        # number of candidates per point per sigma
        while _time.time() < GREEDY_END:
            improved_sweep = False
            Gfull = best @ best.T
            _np.fill_diagonal(Gfull, -2.0)
            rm = Gfull.max(axis=1)
            order = _np.argsort(-rm)
            # global max and second max info
            # sweep worst 30 points
            Kworst = 30
            for kk in range(Kworst):
                if _time.time() > GREEDY_END:
                    break
                # refresh rm occasionally
                if kk % 10 == 0 and kk > 0:
                    Gfull = best @ best.T
                    _np.fill_diagonal(Gfull, -2.0)
                    rm = Gfull.max(axis=1)
                    order = _np.argsort(-rm)
                idx = int(order[kk % n])
                x = best[idx].copy()
                # max without idx: max over pairs not involving idx
                # compute quickly: mask
                # global max
                cur_global = float(Gfull.max())
                # max excluding idx
                # Gfull copy trick: max of all except row/col idx
                # compute via partition: take max of submatrix = max( max upper part)
                # brute: set row/col to -2 temporarily
                old_row = Gfull[idx, :].copy()
                old_col = Gfull[:, idx].copy()
                Gfull[idx, :] = -2.0
                Gfull[:, idx] = -2.0
                base_without = float(Gfull.max())
                Gfull[idx, :] = old_row
                Gfull[:, idx] = old_col
                # if base_without already >= best_val, this point not unique; still try
                for sigma in sig_list:
                    if _time.time() > GREEDY_END:
                        break
                    # try several random tangent directions, keep best
                    best_y = None
                    best_pm = float((best @ x)[_np.arange(n) != idx].max()) if n > 1 else -2.0
                    # actually pm of current x
                    # try candidates
                    Nt = 25
                    for t in range(Nt):
                        r = rng.standard_normal(d)
                        r -= float(r @ x) * x
                        rn = float(_np.linalg.norm(r))
                        if rn < 1e-12 or not _np.isfinite(rn):
                            continue
                        r /= rn
                        y = x + sigma * r
                        yn = float(_np.linalg.norm(y))
                        if yn == 0 or not _np.isfinite(yn):
                            continue
                        y /= yn
                        c = best @ y
                        c[idx] = -2.0
                        pm = float(c.max())
                        new_global = pm if pm > base_without else base_without
                        if new_global < best_val - 1e-13 and pm < best_pm:
                            best_pm = pm
                            best_y = y.copy()
                    if best_y is not None:
                        best[idx] = best_y
                        # update Gfull row/col
                        cnew = best @ best_y
                        cnew[idx] = -2.0
                        Gfull[idx, :] = cnew
                        Gfull[:, idx] = cnew
                        pm_new = float(cnew.max())
                        new_global = pm_new if pm_new > base_without else base_without
                        best_val = new_global
                        improved_sweep = True
                        # after accept, try to immediately refine same point with smaller sigma
                        # continue to next sigma (smaller) with updated x
                        x = best_y.copy()
                        # update base_without stays same
                    # if we improved a lot, break sigma loop to move to next point?
                # end sigma
            if not improved_sweep:
                # no improvement in sweep: try pair move on absolute worst pair + tiny global jitter
                Gfull = best @ best.T
                _np.fill_diagonal(Gfull, -2.0)
                am = int(_np.argmax(Gfull.max(axis=1)))
                # find partner
                bm = int(_np.argmax(Gfull[am]))
                # try moving both apart symmetrically with small steps
                found = False
                for sigma in [0.0005, 0.0002, 0.0001]:
                    for _ in range(60):
                        if _time.time() > GREEDY_END:
                            break
                        xa = best[am].copy()
                        xb = best[bm].copy()
                        ra = rng.standard_normal(d)
                        ra -= float(ra @ xa) * xa
                        ra /= (_np.linalg.norm(ra) + 1e-16)
                        rb = rng.standard_normal(d)
                        rb -= float(rb @ xb) * xb
                        rb /= (_np.linalg.norm(rb) + 1e-16)
                        ya = xa + sigma * ra
                        ya /= _np.linalg.norm(ya)
                        yb = xb + sigma * rb
                        yb /= _np.linalg.norm(yb)
                        # evaluate: compute max involving a,b
                        ca = best @ ya
                        ca[am] = -2.0
                        # replace bm entry with ya.yb
                        ca[bm] = float(ya @ yb)
                        cb = best @ yb
                        cb[bm] = -2.0
                        cb[am] = float(ya @ yb)
                        # base without a,b
                        # approximate: compute full max excluding a,b via stored Gfull
                        # quick upper bound: base_without_ab <= best_val, compute exactly rarely
                        # do exact check only if candidate promising
                        pa = float(ca.max())
                        pb = float(cb.max())
                        if max(pa, pb) >= best_val - 1e-13:
                            continue
                        # exact global: temporarily set
                        Xt = best.copy()
                        Xt[am] = ya
                        Xt[bm] = yb
                        vg = _max_cos(Xt)
                        if vg < best_val - 1e-13:
                            best = Xt
                            best_val = vg
                            found = True
                            break
                    if found:
                        break
                if not found:
                    # try small random jitter of a rattler to escape? just reduce sigma set and continue?
                    # do a null move: shrink sig_list (already minimal) -> break to LP
                    # instead do tiny basin kick on worst point and re-evaluate
                    # if stuck for 2 sweeps, break
                    break
    except Exception:
        pass

    # Stage 3: high-beta squeeze + LP polish with remaining time
    try:
        if _time.time() < GREEDY_END + 2 and _time.time() < LP_END:
            Xs = refine(best, 1200, 400.0, 2500.0, 0.0005, LP_END)
            vs = _max_cos(Xs)
            if vs < best_val:
                best = Xs
                best_val = vs
    except Exception:
        pass

    try:
        from scipy.optimize import linprog as _linprog
        from scipy.sparse import coo_matrix as _coo
        def lp_polish(X, deadline):
            X = X / _np.linalg.norm(X, axis=1, keepdims=True)
            nn, dd = X.shape
            cur_best = _max_cos(X)
            radius = 1e-3
            iut0, iut1 = _np.triu_indices(nn, 1)
            Ktop = 2800
            while _time.time() < deadline and radius > 1e-14:
                Gfull = X @ X.T
                c = Gfull[iut0, iut1]
                if len(c) > Ktop:
                    part = _np.argpartition(-c, Ktop)[:Ktop]
                    I = iut0[part]
                    J = iut1[part]
                    cb = c[part]
                else:
                    I, J = iut0, iut1
                    cb = c
                p = len(I)
                nv = nn * dd + 1
                rows = _np.repeat(_np.arange(p), 2 * dd + 1)
                cols = _np.concatenate([I[:, None] * dd + _np.arange(dd), J[:, None] * dd + _np.arange(dd), _np.full((p, 1), nn * dd)], 1).ravel()
                vals = _np.concatenate([X[J], X[I], -_np.ones((p, 1))], 1).ravel()
                try:
                    A_ub = _coo((vals, (rows, cols)), shape=(p, nv)).tocsr()
                    A_eq = _coo((X.ravel(), (_np.repeat(_np.arange(nn), dd), _np.arange(nn * dd))), shape=(nn, nv)).tocsr()
                except Exception:
                    break
                obj = _np.zeros(nv)
                obj[-1] = 1.0
                try:
                    res = _linprog(obj, A_ub=A_ub, b_ub=-cb, A_eq=A_eq, b_eq=_np.zeros(nn),
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
                if _time.time() > deadline:
                    break
            return X, cur_best
        if _time.time() < LP_END:
            Xlp, vlp = lp_polish(best, LP_END)
            if vlp < best_val:
                best = Xlp
                best_val = vlp
    except Exception:
        pass

    best = best / _np.linalg.norm(best, axis=1, keepdims=True)
    best = _np.nan_to_num(best, nan=0.0, posinf=0.0, neginf=0.0)
    nr = _np.linalg.norm(best, axis=1, keepdims=True)
    nr[nr < 1e-12] = 1.0
    best = best / nr
    return best
