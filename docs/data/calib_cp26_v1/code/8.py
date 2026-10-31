import numpy as np
import time

def solve():
    import time as _tm
    import numpy as _np
    from scipy.optimize import minimize, linprog
    n = 26
    t0 = _tm.time()
    DEADLINE = 124.0  # leave margin for 130s limit

    # pair indices and A_ub pattern (fixed)
    iu, ju = _np.triu_indices(n, 1)
    m = len(iu)
    A_ub0 = _np.zeros((m, n), dtype=float)
    _ar = _np.arange(m)
    A_ub0[_ar, iu] = 1.0
    A_ub0[_ar, ju] = 1.0
    c_obj = -_np.ones(n)

    def eval_lp(xc, yc, best_sum_ref=None):
        # quick bound prune
        b = _np.minimum(_np.minimum(xc, 1.0 - xc), _np.minimum(yc, 1.0 - yc))
        b = _np.maximum(b, 0.0)
        sb = _np.sum(b)
        if best_sum_ref is not None and sb <= best_sum_ref:
            return None, -1.0
        dx = xc[iu] - xc[ju]
        dy = yc[iu] - yc[ju]
        dvec = _np.sqrt(dx * dx + dy * dy)
        # avoid exact zero distance (linprog needs b_ub >=0)
        # dvec can be 0 if coincident; then constraint r_i+r_j<=0 forces 0, still ok
        bounds = [(0.0, float(bi)) for bi in b]
        try:
            res = linprog(c_obj, A_ub=A_ub0, b_ub=dvec, bounds=bounds, method='highs')
        except Exception:
            return None, -1.0
        if not res.success or res.x is None:
            return None, -1.0
        r = _np.array(res.fun is not None and res.x, dtype=float).flatten()
        # res.fun = -sum
        s = float(_np.sum(r))
        # clip tiny negatives
        r = _np.maximum(r, 0.0)
        return r, s

    def repair_radii(xc, yc, rc):
        eps = 1e-7
        b = _np.minimum(_np.minimum(xc, 1.0 - xc), _np.minimum(yc, 1.0 - yc))
        t = 1.0
        # box
        for i in range(n):
            if rc[i] > 0:
                ti = (b[i] - eps) / rc[i]
                if ti < t:
                    t = ti
        if t <= 0:
            # centres too close to border, shrink drastically
            rc = _np.full(n, 1e-4)
            return rc
        dx = xc[iu] - xc[ju]
        dy = yc[iu] - yc[ju]
        dvec = _np.sqrt(dx * dx + dy * dy)
        rs = rc[iu] + rc[ju]
        # only where rs>0
        mask = rs > 1e-12
        if _np.any(mask):
            ratios = (dvec[mask] - eps) / rs[mask]
            tpair = _np.min(ratios)
            if tpair < t:
                t = tpair
        if t > 1.0:
            t = 1.0
        if t <= 0:
            rc = _np.full(n, 1e-4)
            return rc
        # safety factor to be strictly feasible for 1e-12 tolerance
        t = t * (1.0 - 1e-9)
        return rc * t

    # bounds for L-BFGS-B
    bnds = [(0.0, 1.0)] * (2 * n) + [(1e-4, 0.35)] * n
    P_seq = [30.0, 300.0, 3000.0, 30000.0]

    best_sum = -1.0
    best_x = None
    best_y = None
    best_r = None

    # fallback parent grid in case everything fails
    def parent_fallback():
        k = int(_np.ceil(_np.sqrt(n)))
        rr = 0.5 / k
        pts = []
        for i in range(k):
            for j in range(k):
                if len(pts) < n:
                    pts.append([(2 * i + 1) * rr, (2 * j + 1) * rr, rr * 0.999])
        return _np.array(pts, dtype=float)

    restart_idx = 0
    # main gradient restarts
    while True:
        if _tm.time() - t0 > 82.0:
            break
        if restart_idx > 400:
            break
        rng = _np.random.RandomState(1000 + restart_idx * 7919)
        if restart_idx == 0:
            # 5x5 grid + gap circle
            pts = []
            for i in range(5):
                for j in range(5):
                    pts.append([0.1 + 0.2 * i, 0.1 + 0.2 * j])
            pts = _np.array(pts)
            extra = _np.array([[0.2, 0.2]])
            xy = _np.vstack([pts, extra])
            x0 = xy[:, 0].copy()
            y0 = xy[:, 1].copy()
            r0 = _np.full(n, 0.09)
            r0[-1] = 0.04
            # tiny jitter deterministic
            x0 += rng.randn(n) * 0.002
            y0 += rng.randn(n) * 0.002
            x0 = _np.clip(x0, 0.05, 0.95)
            y0 = _np.clip(y0, 0.05, 0.95)
        elif restart_idx == 1:
            # 6x5-ish dense grid init
            xs = ( _np.arange(6) + 0.5) / 6.0
            ys = ( _np.arange(5) + 0.5) / 5.0
            xx, yy = _np.meshgrid(xs, ys)
            xy = _np.column_stack([xx.ravel(), yy.ravel()])
            # pick 26 of 30
            perm = rng.permutation(len(xy))[:n]
            x0 = xy[perm, 0] + rng.randn(n) * 0.005
            y0 = xy[perm, 1] + rng.randn(n) * 0.005
            x0 = _np.clip(x0, 0.05, 0.95)
            y0 = _np.clip(y0, 0.05, 0.95)
            r0 = rng.uniform(0.06, 0.09, size=n)
        else:
            x0 = rng.uniform(0.05, 0.95, size=n)
            y0 = rng.uniform(0.05, 0.95, size=n)
            r0 = rng.uniform(0.06, 0.10, size=n)

        z = _np.concatenate([x0, y0, r0])

        # staged penalty L-BFGS-B
        feasible_time = True
        for P in P_seq:
            if _tm.time() - t0 > 82.0:
                feasible_time = False
                break
            # closure
            def fg(zz, P=P):
                x = zz[0:n]
                y = zz[n:2*n]
                r = zz[2*n:3*n]
                f = -_np.sum(r)
                gx = _np.zeros(n)
                gy = _np.zeros(n)
                gr = -_np.ones(n)
                v1 = r - x
                mm = v1 > 0
                if _np.any(mm):
                    f += P * _np.sum(v1[mm] * v1[mm])
                    co = 2.0 * P * v1[mm]
                    gx[mm] -= co
                    gr[mm] += co
                v2 = x + r - 1.0
                mm = v2 > 0
                if _np.any(mm):
                    f += P * _np.sum(v2[mm] * v2[mm])
                    co = 2.0 * P * v2[mm]
                    gx[mm] += co
                    gr[mm] += co
                v3 = r - y
                mm = v3 > 0
                if _np.any(mm):
                    f += P * _np.sum(v3[mm] * v3[mm])
                    co = 2.0 * P * v3[mm]
                    gy[mm] -= co
                    gr[mm] += co
                v4 = y + r - 1.0
                mm = v4 > 0
                if _np.any(mm):
                    f += P * _np.sum(v4[mm] * v4[mm])
                    co = 2.0 * P * v4[mm]
                    gy[mm] += co
                    gr[mm] += co
                # pairwise
                dxm = x[:, None] - x[None, :]
                dym = y[:, None] - y[None, :]
                D = _np.sqrt(dxm * dxm + dym * dym)
                Rsum = r[:, None] + r[None, :]
                S = Rsum - D
                _np.fill_diagonal(S, -1.0)
                mask = S > 0
                if _np.any(mask):
                    f += P * 0.5 * _np.sum(S[mask] * S[mask])
                    W = _np.zeros((n, n))
                    # safe divide
                    W[mask] = (2.0 * P * S[mask]) / (D[mask] + 1e-12)
                    rs = W.sum(axis=1)
                    gx += W.dot(x) - rs * x
                    gy += W.dot(y) - rs * y
                    Spos = _np.where(mask, S, 0.0)
                    gr += 2.0 * P * Spos.sum(axis=1)
                g = _np.concatenate([gx, gy, gr])
                return f, g
            try:
                res = minimize(fg, z, method='L-BFGS-B', jac=True, bounds=bnds,
                               options={'maxiter': 250, 'ftol': 1e-12, 'gtol': 1e-7})
                z = res.x
            except Exception:
                break
        # evaluate with LP for best radii
        try:
            xc = _np.clip(z[0:n], 0.0, 1.0)
            yc = _np.clip(z[n:2*n], 0.0, 1.0)
            r_opt, s = eval_lp(xc, yc, best_sum_ref=best_sum if best_sum > 0 else None)
            if r_opt is not None and s > best_sum:
                best_sum = float(s)
                best_x = xc.copy()
                best_y = yc.copy()
                best_r = r_opt.copy()
        except Exception:
            pass
        restart_idx += 1

    # hill climbing on LP objective (single-circle moves)
    if best_x is not None:
        rng_hc = _np.random.RandomState(9999)
        hc_iter = 0
        while _tm.time() - t0 < DEADLINE - 3.0 and hc_iter < 6000:
            sigma = 0.02 * _np.exp(-hc_iter / 1500.0) + 0.001
            idx = int(rng_hc.randint(n))
            nx = best_x.copy()
            ny = best_y.copy()
            nx[idx] += float(rng_hc.randn()) * sigma
            ny[idx] += float(rng_hc.randn()) * sigma
            # keep interior
            if nx[idx] < 0.01:
                nx[idx] = 0.01
            if nx[idx] > 0.99:
                nx[idx] = 0.99
            if ny[idx] < 0.01:
                ny[idx] = 0.01
            if ny[idx] > 0.99:
                ny[idx] = 0.99
            r_opt, s = eval_lp(nx, ny, best_sum_ref=best_sum)
            if r_opt is not None and s > best_sum + 1e-12:
                best_sum = float(s)
                best_x = nx
                best_y = ny
                best_r = r_opt.copy()
            hc_iter += 1
            # occasionally try full jitter
            if hc_iter % 500 == 0 and _tm.time() - t0 < DEADLINE - 10.0:
                nx2 = best_x + rng_hc.randn(n) * 0.003
                ny2 = best_y + rng_hc.randn(n) * 0.003
                nx2 = _np.clip(nx2, 0.01, 0.99)
                ny2 = _np.clip(ny2, 0.01, 0.99)
                r_opt2, s2 = eval_lp(nx2, ny2, best_sum_ref=best_sum)
                if r_opt2 is not None and s2 > best_sum + 1e-12:
                    best_sum = float(s2)
                    best_x = nx2
                    best_y = ny2
                    best_r = r_opt2.copy()

    if best_x is None or best_r is None:
        return parent_fallback()
    # final exact radii + strict repair
    r_final_opt, _ = eval_lp(best_x, best_y, best_sum_ref=None)
    if r_final_opt is not None:
        best_r = r_final_opt
    r_safe = repair_radii(best_x, best_y, _np.maximum(best_r, 0.0))
    # ensure positivity
    r_safe = _np.maximum(r_safe, 1e-6)
    # final clip of centres to allow strict interior with r_safe
    # if centre too close to border, nudge it inward slightly
    for i in range(n):
        if best_x[i] < r_safe[i] + 1e-9:
            best_x[i] = r_safe[i] + 1e-9
        if best_x[i] > 1.0 - r_safe[i] - 1e-9:
            best_x[i] = 1.0 - r_safe[i] - 1e-9
        if best_y[i] < r_safe[i] + 1e-9:
            best_y[i] = r_safe[i] + 1e-9
        if best_y[i] > 1.0 - r_safe[i] - 1e-9:
            best_y[i] = 1.0 - r_safe[i] - 1e-9
    # one more repair after nudging (tiny change, re-shrink if needed)
    r_safe = repair_radii(best_x, best_y, r_safe)
    r_safe = _np.maximum(r_safe, 1e-9)
    out = _np.column_stack([best_x, best_y, r_safe])
    return out
