import numpy as np

def solve():
    import time as _tm
    import numpy as _np
    from scipy.optimize import minimize, linprog
    n = 26
    t0 = _tm.time()
    DEADLINE = 123.0
    PHASE_A_END = 58.0
    PHASE_B_END = 118.0

    iu, ju = _np.triu_indices(n, 1)
    m = len(iu)
    A_ub0 = _np.zeros((m, n), dtype=float)
    _ar = _np.arange(m)
    A_ub0[_ar, iu] = 1.0
    A_ub0[_ar, ju] = 1.0
    c_obj = -_np.ones(n)

    def eval_lp(xc, yc):
        try:
            b = _np.minimum(_np.minimum(xc, 1.0 - xc), _np.minimum(yc, 1.0 - yc))
            b = _np.maximum(b, 0.0)
            dx = xc[iu] - xc[ju]
            dy = yc[iu] - yc[ju]
            dvec = _np.sqrt(dx * dx + dy * dy) - 1e-9
            # quick infeasible prune
            if _np.min(b) < 0.004:
                # still try but bound tiny
                pass
            bounds = [(0.0, float(bi)) for bi in b]
            res = linprog(c_obj, A_ub=A_ub0, b_ub=dvec, bounds=bounds, method='highs')
            if not res.success or res.x is None:
                return None, -1.0
            r = _np.asarray(res.x, dtype=float).flatten()
            r = _np.maximum(r, 0.0)
            # strict shrink for eps used
            r = r * (1.0 - 2e-9)
            s = float(_np.sum(r))
            # verify roughly feasible
            if s < 0 or _np.any(r < 0):
                return None, -1.0
            return r, s
        except Exception:
            return None, -1.0

    def fallback():
        k = int(_np.ceil(_np.sqrt(n)))
        rr = 0.5 / k
        pts = []
        for i in range(k):
            for j in range(k):
                if len(pts) < n:
                    pts.append([(2 * i + 1) * rr, (2 * j + 1) * rr, rr * 0.998])
        return _np.array(pts, dtype=float)

    best_arr = fallback()
    # scale fallback strictly
    try:
        Xb, Yb, Rb = best_arr[:, 0], best_arr[:, 1], best_arr[:, 2]
        wall = _np.minimum(_np.minimum(Xb, 1 - Xb), _np.minimum(Yb, 1 - Yb)) / Rb
        Dx = Xb[:, None] - Xb[None, :]; Dy = Yb[:, None] - Yb[None, :]
        D = _np.sqrt(Dx * Dx + Dy * Dy)
        _np.fill_diagonal(D, _np.inf)
        S = Rb[:, None] + Rb[None, :]
        ratio = D / S
        a = min(_np.min(wall), _np.min(ratio))
        a = a * (1.0 - 1e-9)
        Rb = Rb * a
        best_arr = _np.column_stack([Xb, Yb, Rb])
    except Exception:
        pass
    best_sum = float(_np.sum(best_arr[:, 2]))
    best_x = best_arr[:, 0].copy(); best_y = best_arr[:, 1].copy(); best_r = best_arr[:, 2].copy()

    bnds = [(0.0, 1.0)] * (2 * n) + [(1e-4, 0.35)] * n
    P_seq_fast = [20.0, 200.0, 2000.0, 20000.0]

    def run_penalty(x0, y0, r0, maxiter_each=80, pseq=None, tlim=66.0):
        if pseq is None:
            pseq = P_seq_fast
        z = _np.concatenate([x0, y0, r0])
        for P in pseq:
            if _tm.time() - t0 > tlim:
                break
            def fg(zz, P=P):
                x = zz[0:n]; y = zz[n:2 * n]; r = zz[2 * n:3 * n]
                f = -_np.sum(r)
                gx = _np.zeros(n); gy = _np.zeros(n); gr = -_np.ones(n)
                v1 = r - x
                mm = v1 > 0
                if _np.any(mm):
                    f += P * _np.sum(v1[mm] * v1[mm])
                    co = 2.0 * P * v1[mm]
                    gx[mm] -= co; gr[mm] += co
                v2 = x + r - 1.0
                mm = v2 > 0
                if _np.any(mm):
                    f += P * _np.sum(v2[mm] * v2[mm])
                    co = 2.0 * P * v2[mm]
                    gx[mm] += co; gr[mm] += co
                v3 = r - y
                mm = v3 > 0
                if _np.any(mm):
                    f += P * _np.sum(v3[mm] * v3[mm])
                    co = 2.0 * P * v3[mm]
                    gy[mm] -= co; gr[mm] += co
                v4 = y + r - 1.0
                mm = v4 > 0
                if _np.any(mm):
                    f += P * _np.sum(v4[mm] * v4[mm])
                    co = 2.0 * P * v4[mm]
                    gy[mm] += co; gr[mm] += co
                dxm = x[:, None] - x[None, :]
                dym = y[:, None] - y[None, :]
                D = _np.sqrt(dxm * dxm + dym * dym)
                Rsum = r[:, None] + r[None, :]
                S = Rsum - D
                _np.fill_diagonal(S, -1.0)
                mask = S > 0
                if _np.any(mask):
                    f += P * 0.5 * _np.sum(S[mask] * S[mask])
                    Dsafe = D + 1e-12
                    W = _np.zeros((n, n))
                    W[mask] = (2.0 * P * S[mask]) / Dsafe[mask]
                    rs = W.sum(axis=1)
                    gx += W.dot(x) - rs * x
                    gy += W.dot(y) - rs * y
                    Spos = _np.where(mask, S, 0.0)
                    gr += 2.0 * P * Spos.sum(axis=1)
                g = _np.concatenate([gx, gy, gr])
                return f, g
            try:
                res = minimize(fg, z, method='L-BFGS-B', jac=True, bounds=bnds,
                               options={'maxiter': maxiter_each, 'ftol': 1e-10, 'gtol': 1e-6})
                z = res.x
            except Exception:
                break
        return z

    # ---------- Phase A: diversified restarts ----------
    restart_idx = 0
    # hex prototype points
    while (_tm.time() - t0) < PHASE_A_END and restart_idx < 400:
        rng = _np.random.RandomState(1000 + restart_idx * 7919)
        mode = restart_idx % 7
        try:
            if mode == 0:
                x0 = rng.rand(n); y0 = rng.rand(n)
            elif mode == 1:
                # 5x5 +1 jittered
                base = []
                for i in range(5):
                    for j in range(5):
                        base.append([0.1 + 0.2 * i, 0.1 + 0.2 * j])
                base = _np.array(base)
                # one extra random
                extra = rng.rand(1, 2)
                pts = _np.vstack([base, extra])  # 26 pts
                perm = rng.permutation(26)[:n]
                # actually base+extra =26 already
                x0 = pts[:, 0] + (rng.rand(n) - 0.5) * 0.10
                y0 = pts[:, 1] + (rng.rand(n) - 0.5) * 0.10
                x0 = _np.clip(x0, 0.02, 0.98); y0 = _np.clip(y0, 0.02, 0.98)
            elif mode == 2:
                # hex jitter
                xs = []; ys = []
                ny = 6; nx = 5
                for iy in range(ny):
                    for ix in range(nx):
                        if len(xs) >= n:
                            break
                        xx = (ix + 0.5 + (0.5 if iy % 2 == 1 else 0.0)) / 5.5
                        yy = (iy + 0.5) / 6.0
                        xs.append(xx); ys.append(yy)
                    if len(xs) >= n:
                        break
                x0 = _np.array(xs[:n]) + (rng.rand(n) - 0.5) * 0.08
                y0 = _np.array(ys[:n]) + (rng.rand(n) - 0.5) * 0.08
                x0 = _np.clip(x0, 0.02, 0.98); y0 = _np.clip(y0, 0.02, 0.98)
            elif mode == 3:
                # random row counts summing to 26 (staggered rows)
                # pick 5 or 6 rows
                nr = 5 if rng.rand() < 0.5 else 6
                # random composition
                cuts = sorted(rng.choice(range(1, 26), size=nr - 1, replace=False))
                counts = []
                prev = 0
                for c in cuts:
                    counts.append(c - prev); prev = c
                counts.append(26 - prev)
                xs = []; ys = []
                for ri, cnt in enumerate(counts):
                    yy = (ri + 0.5) / nr
                    for k2 in range(cnt):
                        xx = (k2 + 0.5 + rng.rand() * 0.3) / cnt
                        xs.append(xx); ys.append(yy + (rng.rand() - 0.5) * 0.05)
                # shuffle assign
                perm = rng.permutation(n)
                x0 = _np.array(xs)[perm] + (rng.rand(n) - 0.5) * 0.04
                y0 = _np.array(ys)[perm] + (rng.rand(n) - 0.5) * 0.04
                x0 = _np.clip(x0, 0.02, 0.98); y0 = _np.clip(y0, 0.02, 0.98)
            elif mode == 4:
                # two-size biased: cluster
                x0 = rng.rand(n); y0 = rng.rand(n)
            elif mode == 5:
                # column template
                nc = 5 if rng.rand() < 0.5 else 6
                xs = []; ys = []
                for ci in range(nc):
                    cnt = n // nc + (1 if ci < n % nc else 0)
                    xx = (ci + 0.5) / nc
                    for k2 in range(cnt):
                        yy = (k2 + 0.5 + rng.rand() * 0.2) / cnt
                        xs.append(xx + (rng.rand() - 0.5) * 0.05); ys.append(yy)
                perm = rng.permutation(n)
                x0 = _np.array(xs)[perm]; y0 = _np.array(ys)[perm]
                x0 = _np.clip(x0, 0.02, 0.98); y0 = _np.clip(y0, 0.02, 0.98)
            else:
                if best_x is not None and rng.rand() < 0.6:
                    x0 = _np.clip(best_x + 0.06 * rng.randn(n), 0.02, 0.98)
                    y0 = _np.clip(best_y + 0.06 * rng.randn(n), 0.02, 0.98)
                else:
                    x0 = rng.rand(n); y0 = rng.rand(n)
            # radii init diverse
            if mode == 4:
                # few large, many small
                r0 = 0.04 + 0.03 * rng.rand(n)
                nl = 3 + int(rng.rand() * 3)
                idxL = rng.choice(n, size=nl, replace=False)
                r0[idxL] = 0.12 + 0.04 * rng.rand(nl)
            elif restart_idx % 2 == 0:
                r0 = 0.06 + 0.05 * rng.rand(n)
            else:
                r0 = 0.04 + 0.10 * rng.rand(n)
            r0 = _np.clip(r0, 0.015, 0.28)
            # fast anneal, fewer iters for throughput
            mi = 60 if n > 20 else 80
            z = run_penalty(x0, y0, r0, maxiter_each=mi, tlim=PHASE_A_END)
            x = z[0:n]; y = z[n:2 * n]
            x = _np.clip(x, 0.006, 0.994); y = _np.clip(y, 0.006, 0.994)
            r_opt, s = eval_lp(x, y)
            if r_opt is not None and s > best_sum + 1e-12:
                best_sum = s
                best_x = x.copy(); best_y = y.copy(); best_r = r_opt.copy()
        except Exception:
            pass
        restart_idx += 1

    # ---------- Phase B: LP hill-climb + ruin/recreate ----------
    rngB = _np.random.RandomState(777)
    # coarse grid for void search
    gx1d = _np.linspace(0.01, 0.99, 45)
    GX, GY = _np.meshgrid(gx1d, gx1d)
    GP = _np.column_stack([GX.ravel(), GY.ravel()])
    nGP = GP.shape[0]

    def largest_void(xc, yc, rc):
        try:
            # vectorized clearance
            dx = GP[:, None, 0] - xc[None, :]
            dy = GP[:, None, 1] - yc[None, :]
            dist = _np.sqrt(dx * dx + dy * dy) - rc[None, :]
            best_c = _np.min(dist, axis=1)
            wall = _np.minimum(_np.minimum(GP[:, 0], 1 - GP[:, 0]), _np.minimum(GP[:, 1], 1 - GP[:, 1]))
            clear = _np.minimum(wall, best_c)
            k = int(_np.argmax(clear))
            return GP[k, 0], GP[k, 1], float(clear[k])
        except Exception:
            return 0.5, 0.5, 0.0

    sigmas = [0.02, 0.012, 0.008, 0.005, 0.003, 0.0015, 0.0008]
    it_outer = 0
    # estimate best order for victim selection
    while (_tm.time() - t0) < PHASE_B_END:
        it_outer += 1
        if it_outer > 60:
            break
        for sigma in sigmas:
            if (_tm.time() - t0) >= PHASE_B_END:
                break
            iters = 120 if sigma >= 0.005 else 90
            for _ in range(iters):
                if (_tm.time() - t0) >= PHASE_B_END:
                    break
                u = rngB.rand()
                try:
                    if u < 0.45:
                        xn = best_x + sigma * rngB.randn(n)
                        yn = best_y + sigma * rngB.randn(n)
                        xn = _np.clip(xn, 0.015, 0.985); yn = _np.clip(yn, 0.015, 0.985)
                        r_opt, s = eval_lp(xn, yn)
                        if r_opt is not None and s > best_sum + 1e-10:
                            best_sum = s; best_x = xn; best_y = yn; best_r = r_opt
                    elif u < 0.75:
                        xn = best_x.copy(); yn = best_y.copy()
                        idx = int(rngB.randint(n))
                        xn[idx] += sigma * 1.8 * rngB.randn()
                        yn[idx] += sigma * 1.8 * rngB.randn()
                        xn = _np.clip(xn, 0.015, 0.985); yn = _np.clip(yn, 0.015, 0.985)
                        r_opt, s = eval_lp(xn, yn)
                        if r_opt is not None and s > best_sum + 1e-10:
                            best_sum = s; best_x = xn; best_y = yn; best_r = r_opt
                    elif u < 0.85:
                        # 2-3 circles
                        xn = best_x.copy(); yn = best_y.copy()
                        kk = 2 if rngB.rand() < 0.7 else 3
                        idxs = rngB.choice(n, size=kk, replace=False)
                        xn[idxs] += sigma * 1.5 * rngB.randn(kk)
                        yn[idxs] += sigma * 1.5 * rngB.randn(kk)
                        xn = _np.clip(xn, 0.015, 0.985); yn = _np.clip(yn, 0.015, 0.985)
                        r_opt, s = eval_lp(xn, yn)
                        if r_opt is not None and s > best_sum + 1e-10:
                            best_sum = s; best_x = xn; best_y = yn; best_r = r_opt
                    elif u < 0.94:
                        # void teleport of a small circle
                        order = _np.argsort(best_r)
                        # pick among 6 smallest randomly
                        cand = order[:6]
                        victim = int(cand[int(rngB.randint(len(cand)))])
                        vx, vy, _ = largest_void(best_x, best_y, best_r)
                        xn = best_x.copy(); yn = best_y.copy()
                        xn[victim] = vx + 0.005 * rngB.randn()
                        yn[victim] = vy + 0.005 * rngB.randn()
                        xn = _np.clip(xn, 0.015, 0.985); yn = _np.clip(yn, 0.015, 0.985)
                        r_opt, s = eval_lp(xn, yn)
                        if r_opt is not None and s > best_sum + 1e-10:
                            best_sum = s; best_x = xn; best_y = yn; best_r = r_opt
                        else:
                            # short rejamm around teleported config (cheap)
                            if rngB.rand() < 0.35 and (_tm.time() - t0) < PHASE_B_END - 8:
                                r0t = best_r.copy()
                                # jitter radii a bit
                                zt = run_penalty(xn, yn, _np.clip(r0t, 0.01, 0.3), maxiter_each=25,
                                                 pseq=[200.0, 2000.0], tlim=PHASE_B_END)
                                xx = _np.clip(zt[0:n], 0.006, 0.994); yy = _np.clip(zt[n:2 * n], 0.006, 0.994)
                                r2, s2 = eval_lp(xx, yy)
                                if r2 is not None and s2 > best_sum + 1e-10:
                                    best_sum = s2; best_x = xx; best_y = yy; best_r = r2
                    else:
                        # swap two disparate + jitter then quick LP
                        a = int(rngB.randint(n)); b = int(rngB.randint(n))
                        if a == b:
                            continue
                        xn = best_x.copy(); yn = best_y.copy()
                        # swap
                        xn[a], xn[b] = xn[b], xn[a]
                        yn[a], yn[b] = yn[b], yn[a]
                        xn += 0.004 * rngB.randn(n); yn += 0.004 * rngB.randn(n)
                        xn = _np.clip(xn, 0.015, 0.985); yn = _np.clip(yn, 0.015, 0.985)
                        r_opt, s = eval_lp(xn, yn)
                        if r_opt is not None and s > best_sum + 1e-10:
                            best_sum = s; best_x = xn; best_y = yn; best_r = r_opt
                except Exception:
                    continue
        # basin-hop shake if time remains
        if (_tm.time() - t0) >= PHASE_B_END:
            break
        # occasional medium shake attempt
        try:
            improved = False
            for _ in range(25):
                if (_tm.time() - t0) >= PHASE_B_END:
                    break
                xn = _np.clip(best_x + 0.035 * rngB.randn(n), 0.015, 0.985)
                yn = _np.clip(best_y + 0.035 * rngB.randn(n), 0.015, 0.985)
                r_opt, s = eval_lp(xn, yn)
                if r_opt is not None and s > best_sum + 1e-10:
                    best_sum = s; best_x = xn; best_y = yn; best_r = r_opt
                    improved = True
                    break
            # if not improved, continue next sigma cycle (keeps best safeguarded)
        except Exception:
            pass

    # ---------- Phase C: joint polish ----------
    if (_tm.time() - t0) < DEADLINE - 6:
        try:
            rb = _np.minimum(best_r.copy(), 0.3)
            denom = _np.maximum(1.0 - 2.0 * rb, 0.2)
            ub = (best_x - rb) / denom
            vb = (best_y - rb) / denom
            ub = _np.clip(ub, 0.001, 0.999); vb = _np.clip(vb, 0.001, 0.999)
            z = _np.concatenate([ub, vb, rb])
            bounds_box = [(0.0, 1.0)] * (2 * n) + [(0.01, 0.30)] * n
            for W in (5000.0, 50000.0):
                if (_tm.time() - t0) >= DEADLINE - 3:
                    break
                def fun2(zz, W=W):
                    uu = zz[:n]; vv = zz[n:2 * n]; rr = zz[2 * n:]
                    x = rr * (1.0 - 2.0 * uu) + uu
                    y = rr * (1.0 - 2.0 * vv) + vv
                    Dx = x[:, None] - x[None, :]
                    Dy = y[:, None] - y[None, :]
                    D = _np.sqrt(Dx * Dx + Dy * Dy + 1e-16)
                    _np.fill_diagonal(D, 1.0)
                    E = rr[:, None] + rr[None, :] - D
                    _np.fill_diagonal(E, 0.0)
                    Ep = _np.maximum(E, 0.0)
                    Pen = 0.5 * _np.sum(Ep * Ep)
                    F = -_np.sum(rr) + W * Pen
                    InvD = 1.0 / D
                    _np.fill_diagonal(InvD, 0.0)
                    Wmat = Ep * InvD
                    Gx = -2.0 * _np.sum(Wmat * Dx, axis=1)
                    Gy = -2.0 * _np.sum(Wmat * Dy, axis=1)
                    sumEp = _np.sum(Ep, axis=1)
                    om2r = 1.0 - 2.0 * rr
                    dFdu = W * Gx * om2r
                    dFdv = W * Gy * om2r
                    dFdr = -_np.ones(n) + W * (2.0 * sumEp + Gx * (1.0 - 2.0 * uu) + Gy * (1.0 - 2.0 * vv))
                    g = _np.concatenate([dFdu, dFdv, dFdr])
                    return F, g
                res = minimize(fun2, z, method='L-BFGS-B', jac=True, bounds=bounds_box,
                               options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6})
                z = res.x
            uu = z[:n]; vv = z[n:2 * n]; rr = z[2 * n:]
            x = rr * (1.0 - 2.0 * uu) + uu
            y = rr * (1.0 - 2.0 * vv) + vv
            x = _np.clip(x, 0.006, 0.994); y = _np.clip(y, 0.006, 0.994)
            r_opt, s = eval_lp(x, y)
            if r_opt is not None and s > best_sum:
                best_sum = s; best_x = x; best_y = y; best_r = r_opt
        except Exception:
            pass

    # ---------- strict safety ----------
    try:
        x = best_x; y = best_y; r = best_r
        wall = _np.minimum(_np.minimum(x, 1 - x), _np.minimum(y, 1 - y))
        Dx = x[:, None] - x[None, :]; Dy = y[:, None] - y[None, :]
        D = _np.sqrt(Dx * Dx + Dy * Dy)
        fw = _np.min(wall / _np.maximum(r, 1e-12))
        fp = 1e100
        for i in range(n):
            for j in range(i + 1, n):
                v = D[i, j] / (r[i] + r[j])
                if v < fp:
                    fp = v
        f = min(fw, fp)
        if f < 1.0:
            r = r * f * (1.0 - 1e-9)
        else:
            r = r * (1.0 - 1e-12)
        # ensure positivity
        r = _np.maximum(r, 1e-9)
        out = _np.column_stack([x, y, r])
        return out
    except Exception:
        return fallback()
