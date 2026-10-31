import numpy as np
import time as _tm

def solve():
    import numpy as _np
    from scipy.optimize import minimize, linprog
    n = 26
    t0 = _tm.time()
    DEADLINE = 124.0

    iu, ju = _np.triu_indices(n, 1)
    m = len(iu)

    # primal pattern for final radii
    A_ub0 = _np.zeros((m, n), dtype=float)
    _ar = _np.arange(m)
    A_ub0[_ar, iu] = 1.0
    A_ub0[_ar, ju] = 1.0
    c_primal = -_np.ones(n)

    # dual pattern: A_ub_dual = -M^T, shape (n, m+n)
    A_dual = _np.zeros((n, m + n), dtype=float)
    for k in range(m):
        A_dual[iu[k], k] = -1.0
        A_dual[ju[k], k] = -1.0
    for i in range(n):
        A_dual[i, m + i] = -1.0
    b_dual = -_np.ones(n)
    bounds_dual = [(0.0, None)] * (m + n)

    def dual_solve(xc, yc):
        # returns S, y_pair, z or (None,...)
        dx = xc[iu] - xc[ju]
        dy = yc[iu] - yc[ju]
        dvec = _np.sqrt(dx * dx + dy * dy)
        bvec = _np.minimum(_np.minimum(xc, 1.0 - xc), _np.minimum(yc, 1.0 - yc))
        # centres must stay in [0,1]; clip tiny negative due to numerics
        if _np.any(bvec < -1e-12):
            return None, None, None, None, None
        bvec = _np.maximum(bvec, 0.0)
        c_dual = _np.concatenate([dvec, bvec])
        try:
            res = linprog(c_dual, A_ub=A_dual, b_ub=b_dual, bounds=bounds_dual, method='highs')
        except Exception:
            return None, None, None, None, None
        if (not res.success) or res.x is None:
            return None, None, None, None, None
        S = float(_np.dot(c_dual, res.x))
        y = _np.array(res.x, dtype=float).flatten()
        yp = y[:m]
        z = y[m:]
        return S, yp, z, dvec, bvec

    def fg_centres(v):
        xc = v[:n]
        yc = v[n:2*n]
        # clip to [0,1] for stability
        xc = _np.clip(xc, 0.0, 1.0)
        yc = _np.clip(yc, 0.0, 1.0)
        out = dual_solve(xc, yc)
        S, yp, z, dvec, bvec = out
        if S is None:
            return 10.0, _np.zeros(2*n)
        f = -S
        # gradients dS/dx, dS/dy
        gx = _np.zeros(n)
        gy = _np.zeros(n)
        # pair part
        # avoid div by zero
        invd = _np.zeros(m)
        maskd = dvec > 1e-12
        invd[maskd] = 1.0 / dvec[maskd]
        # contributions: y_ij * (x_i - x_j)/d
        dx = xc[iu] - xc[ju]
        dy = yc[iu] - yc[ju]
        cx = yp * invd * dx  # for i: +.., for j: -..
        cy = yp * invd * dy
        # accumulate
        _np.add.at(gx, iu[maskd], cx[maskd])
        _np.add.at(gx, ju[maskd], -cx[maskd])
        _np.add.at(gy, iu[maskd], cy[maskd])
        _np.add.at(gy, ju[maskd], -cy[maskd])
        # wall part: b = min(x,1-x,y,1-y)
        # subgradient
        for i in range(n):
            xi = xc[i]; yi = yc[i]
            # find minimal
            v0 = xi
            v1 = 1.0 - xi
            v2 = yi
            v3 = 1.0 - yi
            mn = v0
            if v1 < mn: mn = v1
            if v2 < mn: mn = v2
            if v3 < mn: mn = v3
            zi = z[i]
            if zi == 0.0:
                continue
            # tie: pick first minimal (deterministic)
            if mn == v0:
                gx[i] += zi * 1.0
            elif mn == v1:
                gx[i] += zi * (-1.0)
            elif mn == v2:
                gy[i] += zi * 1.0
            else:
                gy[i] += zi * (-1.0)
        g = -_np.concatenate([gx, gy])
        # clip huge grads
        g = _np.clip(g, -10.0, 10.0)
        return f, g

    def eval_primal(xc, yc):
        dx = xc[iu] - xc[ju]
        dy = yc[iu] - yc[ju]
        dvec = _np.sqrt(dx*dx + dy*dy)
        bvec = _np.minimum(_np.minimum(xc, 1.0-xc), _np.minimum(yc, 1.0-yc))
        bvec = _np.maximum(bvec, 0.0)
        bounds = [(0.0, float(bi)) for bi in bvec]
        try:
            res = linprog(c_primal, A_ub=A_ub0, b_ub=dvec, bounds=bounds, method='highs')
        except Exception:
            return None, -1.0
        if (not res.success) or res.x is None:
            return None, -1.0
        r = _np.maximum(_np.array(res.x, dtype=float).flatten(), 0.0)
        return r, float(_np.sum(r))

    def repair_radii(xc, yc, rc):
        eps = 1e-7
        b = _np.minimum(_np.minimum(xc, 1.0-xc), _np.minimum(yc, 1.0-yc))
        t = 1.0
        for i in range(n):
            if rc[i] > 0:
                ti = (b[i] - eps) / rc[i]
                if ti < t:
                    t = ti
        if t <= 0:
            return _np.full(n, 1e-4)
        dx = xc[iu]-xc[ju]; dy = yc[iu]-yc[ju]
        dvec = _np.sqrt(dx*dx+dy*dy)
        rs = rc[iu]+rc[ju]
        mask = rs > 1e-12
        if _np.any(mask):
            ratios = (dvec[mask]-eps)/rs[mask]
            tp = _np.min(ratios)
            if tp < t:
                t = tp
        if t > 1.0: t = 1.0
        if t <= 0:
            return _np.full(n, 1e-4)
        t = t*(1.0-1e-9)
        return rc*t

    best_S = -1.0
    best_x = None
    best_y = None

    bnds_c = [(0.0, 1.0)]*(2*n)
    restart = 0
    # allow up to ~60 restarts or 85s
    while True:
        if _tm.time()-t0 > 85.0: break
        if restart >= 70: break
        rng = _np.random.RandomState(1000+restart*7919)
        if restart == 0:
            pts = []
            for i in range(5):
                for j in range(5):
                    pts.append([0.1+0.2*i, 0.1+0.2*j])
            pts = _np.array(pts, dtype=float)
            extra = _np.array([[0.2, 0.2]])
            xy = _np.vstack([pts, extra])
            x0 = xy[:,0].copy()+rng.randn(n)*0.002
            y0 = xy[:,1].copy()+rng.randn(n)*0.002
            x0 = _np.clip(x0, 0.02, 0.98); y0 = _np.clip(y0, 0.02, 0.98)
        elif restart == 1:
            xs = (_np.arange(6)+0.5)/6.0
            ys = (_np.arange(5)+0.5)/5.0
            xx, yy = _np.meshgrid(xs, ys)
            xy = _np.column_stack([xx.ravel(), yy.ravel()])
            perm = rng.permutation(len(xy))[:n]
            x0 = xy[perm,0]+rng.randn(n)*0.004
            y0 = xy[perm,1]+rng.randn(n)*0.004
            x0 = _np.clip(x0, 0.02, 0.98); y0 = _np.clip(y0, 0.02, 0.98)
        elif restart == 2:
            # hex shifted
            pts = []
            s = 0.185
            y = 0.09
            row = 0
            while len(pts) < 36 and y < 0.99:
                xoff = 0.09 if (row%2==0) else 0.09+s/2.0
                x = xoff
                while x < 0.99:
                    pts.append([x, y])
                    x += s
                y += s*0.8660254
                row += 1
            pts = _np.array(pts)
            perm = rng.permutation(len(pts))[:n]
            x0 = pts[perm,0]+rng.randn(n)*0.003
            y0 = pts[perm,1]+rng.randn(n)*0.003
            x0 = _np.clip(x0, 0.02, 0.98); y0 = _np.clip(y0, 0.02, 0.98)
        else:
            x0 = rng.uniform(0.03, 0.97, size=n)
            y0 = rng.uniform(0.03, 0.97, size=n)
        v0 = _np.concatenate([x0, y0])
        # quick eval to skip hopeless? still optimize
        try:
            S0, _, _, _, _ = dual_solve(_np.clip(x0,0,1), _np.clip(y0,0,1))
        except Exception:
            S0 = None
        # choose maxiter adaptively
        mi = 80 if (S0 is not None and S0 > 2.2) else 50
        try:
            res = minimize(fg_centres, v0, method='L-BFGS-B', jac=True, bounds=bnds_c,
                           options={'maxiter': mi, 'ftol': 1e-10, 'gtol': 1e-6})
            v = res.x
            xc = _np.clip(v[:n], 0.0, 1.0); yc = _np.clip(v[n:], 0.0, 1.0)
            S, _, _, _, _ = dual_solve(xc, yc)
            if S is not None and S > best_S:
                best_S = float(S); best_x = xc.copy(); best_y = yc.copy()
        except Exception:
            pass
        restart += 1
        # also try to keep time for hill climbing
        if _tm.time()-t0 > 85.0:
            break

    # hill climbing on dual value (single + global jitter)
    if best_x is not None:
        rng_hc = _np.random.RandomState(9999)
        hc = 0
        # pre-evaluate to ensure best_S consistent
        Scheck, _, _, _, _ = dual_solve(best_x, best_y)
        if Scheck is not None:
            best_S = float(Scheck)
        while _tm.time()-t0 < DEADLINE-4.0 and hc < 8000:
            sigma = 0.02*_np.exp(-hc/1500.0)+0.0008
            idx = int(rng_hc.randint(n))
            nx = best_x.copy(); ny = best_y.copy()
            nx[idx] += float(rng_hc.randn())*sigma
            ny[idx] += float(rng_hc.randn())*sigma
            if nx[idx] < 0.005: nx[idx]=0.005
            if nx[idx] > 0.995: nx[idx]=0.995
            if ny[idx] < 0.005: ny[idx]=0.005
            if ny[idx] > 0.995: ny[idx]=0.995
            S, _, _, _, _ = dual_solve(nx, ny)
            if S is not None and S > best_S+1e-12:
                best_S = float(S); best_x = nx; best_y = ny
            hc += 1
            if hc % 600 == 0 and _tm.time()-t0 < DEADLINE-12.0:
                nx2 = best_x + rng_hc.randn(n)*0.0025
                ny2 = best_y + rng_hc.randn(n)*0.0025
                nx2 = _np.clip(nx2, 0.005, 0.995); ny2 = _np.clip(ny2, 0.005, 0.995)
                S2, _, _, _, _ = dual_solve(nx2, ny2)
                if S2 is not None and S2 > best_S+1e-12:
                    best_S = float(S2); best_x = nx2; best_y = ny2

    if best_x is None:
        k = int(_np.ceil(_np.sqrt(n)))
        rr = 0.5/k
        pts = []
        for i in range(k):
            for j in range(k):
                if len(pts) < n:
                    pts.append([(2*i+1)*rr, (2*j+1)*rr, rr*0.999])
        return _np.array(pts, dtype=float)

    r_opt, _ = eval_primal(best_x, best_y)
    if r_opt is None:
        r_opt = _np.full(n, 0.05)
    r_safe = repair_radii(best_x, best_y, _np.maximum(r_opt, 0.0))
    r_safe = _np.maximum(r_safe, 1e-6)
    for i in range(n):
        if best_x[i] < r_safe[i]+1e-9: best_x[i] = r_safe[i]+1e-9
        if best_x[i] > 1.0-r_safe[i]-1e-9: best_x[i] = 1.0-r_safe[i]-1e-9
        if best_y[i] < r_safe[i]+1e-9: best_y[i] = r_safe[i]+1e-9
        if best_y[i] > 1.0-r_safe[i]-1e-9: best_y[i] = 1.0-r_safe[i]-1e-9
    r_safe = repair_radii(best_x, best_y, r_safe)
    r_safe = _np.maximum(r_safe, 1e-9)
    out = _np.column_stack([best_x, best_y, r_safe])
    return out
