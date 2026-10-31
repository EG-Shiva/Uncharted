import numpy as np

def solve():
    import time as _tm
    import numpy as _np
    from scipy.optimize import minimize, linprog
    n = 26
    t0 = _tm.time()
    HARD = 124.0
    iu, ju = _np.triu_indices(n, 1)
    m = len(iu)
    A0 = _np.zeros((m, n), dtype=float)
    _ar = _np.arange(m)
    A0[_ar, iu] = 1.0
    A0[_ar, ju] = 1.0
    c_obj = -_np.ones(n, dtype=float)
    rng = _np.random.RandomState(42)

    fx = []; fy = []
    for ix in range(6):
        for jy in range(5):
            fx.append((ix + 0.5) / 6.0); fy.append((jy + 0.5) / 5.0)
    fx = _np.array(fx[:n]); fy = _np.array(fy[:n])
    fr = _np.full(n, 0.08)
    best_xc = fx.copy(); best_yc = fy.copy(); best_r = fr.copy()
    best_sum = float(_np.sum(fr))

    bnds78 = [(0.0, 1.0)] * (2 * n) + [(0.015, 0.24)] * n
    P_full = [30.0, 300.0, 3000.0, 30000.0]
    P_short = [300.0, 3000.0, 30000.0]

    def eval_lp(xc, yc):
        try:
            b = _np.minimum(_np.minimum(xc, 1.0 - xc), _np.minimum(yc, 1.0 - yc))
            if _np.any(b < 1e-9):
                return -1.0, None
            b = _np.maximum(b, 0.0)
            dx = xc[iu] - xc[ju]
            dy = yc[iu] - yc[ju]
            dvec = _np.sqrt(dx * dx + dy * dy)
            dvec = _np.maximum(dvec, 1e-12)
            bounds = [(0.0, float(bi)) for bi in b]
            res = linprog(c_obj, A_ub=A0, b_ub=dvec, bounds=bounds, method='highs')
            if res.status != 0 or res.x is None:
                return -1.0, None
            r = _np.asarray(res.x, dtype=float).flatten()
            r = _np.maximum(r, 0.0)
            return float(_np.sum(r)), r
        except Exception:
            return -1.0, None

    def run_penalty(x0, y0, r0, P_seq, maxiter_each):
        z = _np.concatenate([x0, y0, r0])
        for P in P_seq:
            if _tm.time() - t0 > HARD - 12.0:
                break
            def fg(zz, P=P):
                x = zz[0:n]; y = zz[n:2*n]; r = zz[2*n:3*n]
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
                    Dm = D + 1e-12
                    W = _np.zeros((n, n))
                    W[mask] = (2.0 * P * S[mask]) / Dm[mask]
                    rs = W.sum(axis=1)
                    gx += W.dot(x) - rs * x
                    gy += W.dot(y) - rs * y
                    Spos = _np.where(mask, S, 0.0)
                    gr += 2.0 * P * Spos.sum(axis=1)
                g = _np.concatenate([gx, gy, gr])
                return f, g
            try:
                res = minimize(fg, z, method='L-BFGS-B', jac=True, bounds=bnds78,
                               options={'maxiter': maxiter_each, 'ftol': 1e-10, 'gtol': 1e-6})
                z = _np.asarray(res.x, dtype=float)
            except Exception:
                break
        return z

    def make_init(mode):
        if mode == 0:
            bx = []; by = []
            for ix in range(5):
                for jy in range(5):
                    bx.append((ix + 0.5) / 5.0); by.append((jy + 0.5) / 5.0)
            bx.append(0.5); by.append(0.5)
            bx = _np.array(bx); by = _np.array(by)
            jx = bx + rng.normal(0, 0.015, n)
            jy = by + rng.normal(0, 0.015, n)
            r0 = _np.full(n, 0.09) + rng.normal(0, 0.01, n)
            return _np.clip(jx, 0.05, 0.95), _np.clip(jy, 0.05, 0.95), _np.clip(r0, 0.02, 0.18)
        elif mode == 1:
            gx = []; gy = []
            for ix in range(6):
                for jy in range(5):
                    gx.append((ix + 0.5) / 6.0); gy.append((jy + 0.5) / 5.0)
            gx = _np.array(gx); gy = _np.array(gy)
            perm = rng.permutation(len(gx))[:n]
            jx = gx[perm] + rng.normal(0, 0.02, n)
            jy = gy[perm] + rng.normal(0, 0.02, n)
            r0 = _np.full(n, 0.085) + rng.normal(0, 0.012, n)
            return _np.clip(jx, 0.03, 0.97), _np.clip(jy, 0.03, 0.97), _np.clip(r0, 0.02, 0.18)
        elif mode == 2:
            counts = _np.array([5, 5, 5, 5, 6])
            rng.shuffle(counts)
            xs = []; ys = []
            for ri in range(5):
                cnt = int(counts[ri])
                y = (ri + 0.5) / 5.0 + rng.normal(0, 0.01)
                off = 0.5 / cnt if (ri % 2 == 1) else 0.0
                for k in range(cnt):
                    x = (k + 0.5 + off * 0.5) / (cnt + 0.2) + rng.normal(0, 0.01)
                    xs.append(x); ys.append(y)
            xs = _np.array(xs); ys = _np.array(ys)
            r0 = _np.full(n, 0.085) + rng.uniform(-0.02, 0.02, n)
            return _np.clip(xs, 0.03, 0.97), _np.clip(ys, 0.03, 0.97), _np.clip(r0, 0.02, 0.18)
        elif mode == 3:
            counts = _np.array([5, 5, 5, 5, 6])
            rng.shuffle(counts)
            xs = []; ys = []
            for ci in range(5):
                cnt = int(counts[ci])
                x = (ci + 0.5) / 5.0 + rng.normal(0, 0.01)
                for k in range(cnt):
                    y = (k + 0.5) / (cnt + 0.2) + rng.normal(0, 0.01)
                    xs.append(x); ys.append(y)
            xs = _np.array(xs); ys = _np.array(ys)
            r0 = _np.full(n, 0.085) + rng.uniform(-0.02, 0.02, n)
            return _np.clip(xs, 0.03, 0.97), _np.clip(ys, 0.03, 0.97), _np.clip(r0, 0.02, 0.18)
        elif mode == 4:
            x0 = rng.uniform(0.05, 0.95, n)
            y0 = rng.uniform(0.05, 0.95, n)
            r0 = _np.empty(n)
            r0[:6] = 0.11 + rng.normal(0, 0.008, 6)
            r0[6:20] = 0.088 + rng.normal(0, 0.008, 14)
            r0[20:] = 0.06 + rng.normal(0, 0.008, 6)
            rng.shuffle(r0)
            return x0, y0, _np.clip(r0, 0.02, 0.18)
        elif mode == 5:
            x0 = rng.uniform(0.06, 0.94, n)
            y0 = rng.uniform(0.06, 0.94, n)
            r0 = rng.uniform(0.05, 0.12, n)
            return x0, y0, r0
        else:
            if best_sum > 2.2 and rng.rand() < 0.7:
                sig = 0.03 if rng.rand() < 0.5 else 0.012
                xn = _np.clip(best_xc + rng.normal(0, sig, n), 0.02, 0.98)
                yn = _np.clip(best_yc + rng.normal(0, sig, n), 0.02, 0.98)
                rn = _np.clip(best_r * (1.0 + rng.normal(0, 0.08, n)), 0.02, 0.20)
                if rng.rand() < 0.3:
                    a, b = rng.choice(n, 2, replace=False)
                    xn[a], xn[b] = xn[b], xn[a]
                    yn[a], yn[b] = yn[b], yn[a]
                return xn, yn, rn
            else:
                x0 = rng.uniform(0.05, 0.95, n)
                y0 = rng.uniform(0.05, 0.95, n)
                r0 = rng.uniform(0.05, 0.115, n)
                return x0, y0, r0

    for _ in range(6):
        if _tm.time() - t0 > 15.0:
            break
        mx = _ % 4
        tx, ty, _tr = make_init(mx)
        s, rr = eval_lp(tx, ty)
        if s > best_sum + 1e-12 and rr is not None:
            best_sum = s; best_xc = tx.copy(); best_yc = ty.copy(); best_r = rr.copy()

    restart = 0
    while True:
        if _tm.time() - t0 > 55.0:
            break
        if restart > 80:
            break
        mode = restart % 7
        x0, y0, r0 = make_init(mode)
        mi = 90 if restart < 10 else 65
        z = run_penalty(x0, y0, r0, P_full, mi)
        xc = _np.clip(z[0:n], 0.005, 0.995); yc = _np.clip(z[n:2*n], 0.005, 0.995)
        s, rr = eval_lp(xc, yc)
        if s is not None and s > best_sum + 1e-12 and rr is not None:
            best_sum = s; best_xc = xc.copy(); best_yc = yc.copy(); best_r = rr.copy()
        restart += 1

    def largest_voids(xc, yc, rc, G=34, K=5):
        lin = _np.linspace(0.0, 1.0, G)
        PX, PY = _np.meshgrid(lin, lin)
        px = PX.ravel(); py = PY.ravel()
        wall = _np.minimum(_np.minimum(px, 1.0 - px), _np.minimum(py, 1.0 - py))
        dx = px[:, None] - xc[None, :]
        dy = py[:, None] - yc[None, :]
        D = _np.sqrt(dx * dx + dy * dy) - rc[None, :]
        mind = D.min(axis=1)
        clear = _np.minimum(wall, mind)
        idx = _np.argsort(clear)[::-1]
        picks = []
        for ii in idx:
            if clear[ii] < 0.008:
                break
            ok = True
            for (qx, qy) in picks:
                if (px[ii] - qx) ** 2 + (py[ii] - qy) ** 2 < 0.05 ** 2:
                    ok = False; break
            if ok:
                picks.append((float(px[ii]), float(py[ii]), float(clear[ii])))
                if len(picks) >= K:
                    break
        return picks

    n_ruin = 0
    while True:
        if _tm.time() - t0 > 92.0:
            break
        if n_ruin > 90:
            break
        try:
            voids = largest_voids(best_xc, best_yc, best_r, G=32, K=5)
        except Exception:
            voids = []
        op = n_ruin % 4
        xn = best_xc.copy(); yn = best_yc.copy(); rn = best_r.copy()
        if op == 0 and len(voids) > 0:
            order = _np.argsort(rn)
            k = min(len(voids), 3 if rng.rand() < 0.7 else 2)
            for t in range(k):
                ci = int(order[t]) if rng.rand() < 0.8 else int(rng.choice(n))
                vx, vy, vc = voids[t % len(voids)]
                xn[ci] = _np.clip(vx + rng.normal(0, 0.005), 0.02, 0.98)
                yn[ci] = _np.clip(vy + rng.normal(0, 0.005), 0.02, 0.98)
                rn[ci] = _np.clip(vc * 0.85, 0.02, 0.18)
            xn = _np.clip(xn + rng.normal(0, 0.004, n), 0.015, 0.985)
            yn = _np.clip(yn + rng.normal(0, 0.004, n), 0.015, 0.985)
        elif op == 1:
            a, b = rng.choice(n, 2, replace=False)
            xn[a], xn[b] = xn[b], xn[a]
            yn[a], yn[b] = yn[b], yn[a]
            xn = _np.clip(xn + rng.normal(0, 0.006, n), 0.015, 0.985)
            yn = _np.clip(yn + rng.normal(0, 0.006, n), 0.015, 0.985)
            rn = _np.clip(rn * (1.0 + rng.normal(0, 0.05, n)), 0.02, 0.20)
        elif op == 2:
            xn = _np.clip(xn + rng.normal(0, 0.012, n), 0.015, 0.985)
            yn = _np.clip(yn + rng.normal(0, 0.012, n), 0.015, 0.985)
            rn = _np.clip(rn * (1.0 + rng.normal(0, 0.06, n)), 0.02, 0.20)
        else:
            idx = rng.choice(n, max(2, n // 6), replace=False)
            for ci in idx:
                xn[ci] = rng.uniform(0.05, 0.95)
                yn[ci] = rng.uniform(0.05, 0.95)
                rn[ci] = rng.uniform(0.04, 0.12)
        z = run_penalty(xn, yn, rn, P_short, 45)
        xc = _np.clip(z[0:n], 0.005, 0.995); yc = _np.clip(z[n:2*n], 0.005, 0.995)
        s, rr = eval_lp(xc, yc)
        if s is not None and s > best_sum + 1e-12 and rr is not None:
            best_sum = s; best_xc = xc.copy(); best_yc = yc.copy(); best_r = rr.copy()
        n_ruin += 1

    # Phase 3: centre-only LP hill climbing (coordinate ascent)
    try:
        cur_x = best_xc.copy(); cur_y = best_yc.copy()
        cur_s, cur_r = eval_lp(cur_x, cur_y)
        if cur_r is not None:
            best_sum = float(cur_s); best_r = cur_r.copy()
            deltas = [0.02, 0.01, 0.005, 0.0025, 0.0012]
            dirs = _np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=float)
            for delta in deltas:
                if _tm.time() - t0 > HARD - 6.0:
                    break
                order = _np.arange(n)
                rng.shuffle(order)
                improved = True
                sweeps = 0
                while improved and sweeps < 3:
                    if _tm.time() - t0 > HARD - 6.0:
                        break
                    improved = False
                    sweeps += 1
                    for ii in order:
                        if _tm.time() - t0 > HARD - 6.0:
                            break
                        bx = cur_x[ii]; by = cur_y[ii]
                        best_local_s = cur_s
                        best_lx = bx; best_ly = by
                        for dd in range(8):
                            nx = bx + dirs[dd, 0] * delta
                            ny = by + dirs[dd, 1] * delta
                            if nx < 0.01 or nx > 0.99 or ny < 0.01 or ny > 0.99:
                                continue
                            cur_x[ii] = nx; cur_y[ii] = ny
                            s2, _ = eval_lp(cur_x, cur_y)
                            if s2 > best_local_s + 1e-10:
                                best_local_s = s2
                                best_lx = nx; best_ly = ny
                        cur_x[ii] = bx; cur_y[ii] = by
                        if best_local_s > cur_s + 1e-10:
                            cur_x[ii] = best_lx; cur_y[ii] = best_ly
                            s3, r3 = eval_lp(cur_x, cur_y)
                            cur_s = float(s3); cur_r = r3
                            if cur_s > best_sum:
                                best_sum = cur_s; best_xc = cur_x.copy(); best_yc = cur_y.copy(); best_r = r3.copy()
                            improved = True
            # final rejamm of best centres with short penalty to tighten
            if _tm.time() - t0 < HARD - 8.0:
                try:
                    z = run_penalty(best_xc, best_yc, best_r, P_short, 30)
                    xc = _np.clip(z[0:n], 0.005, 0.995); yc = _np.clip(z[n:2*n], 0.005, 0.995)
                    s, rr = eval_lp(xc, yc)
                    if s > best_sum and rr is not None:
                        best_sum = s; best_xc = xc; best_yc = yc; best_r = rr
                except Exception:
                    pass
    except Exception:
        pass

    # safe shrink
    try:
        xc = _np.clip(best_xc, 0.0, 1.0); yc = _np.clip(best_yc, 0.0, 1.0)
        r = _np.maximum(_np.asarray(best_r, dtype=float).flatten(), 1e-9)
        eps = 2e-9
        b = _np.minimum(_np.minimum(xc, 1.0 - xc), _np.minimum(yc, 1.0 - yc))
        b = _np.maximum(b, 1e-12)
        t = 1.0
        for i in range(n):
            if r[i] > 1e-12:
                ti = (b[i] - 1e-9) / r[i]
                if ti < t:
                    t = ti
        dx = xc[iu] - xc[ju]; dy = yc[iu] - yc[ju]
        dvec = _np.sqrt(dx * dx + dy * dy)
        rs = r[iu] + r[ju]
        mask = rs > 1e-12
        if _np.any(mask):
            ratios = (dvec[mask] - 2e-9) / rs[mask]
            tp = float(_np.min(ratios))
            if tp < t:
                t = tp
        if not _np.isfinite(t) or t <= 0:
            t = 0.5
        if t > 1.0:
            t = 1.0
        t = t * (1.0 - 1e-9)
        r2 = r * t
        r2 = _np.minimum(r2, b - 1e-9)
        r2 = _np.maximum(r2, 1e-6)
        # re-verify pairwise after cap; if any violation, scale once more
        dx = xc[iu] - xc[ju]; dy = yc[iu] - yc[ju]
        dvec = _np.sqrt(dx * dx + dy * dy)
        rs2 = r2[iu] + r2[ju]
        viol = rs2 - dvec
        mv = float(_np.max(viol)) if viol.size else -1.0
        if mv > -1e-9:
            t2 = float(_np.min((dvec - 2e-9) / _np.maximum(rs2, 1e-12)))
            if _np.isfinite(t2) and t2 < 1.0 and t2 > 0:
                r2 = r2 * t2 * (1.0 - 1e-9)
        out = _np.column_stack([xc, yc, r2])
        return out
    except Exception:
        out = _np.column_stack([best_xc, best_yc, best_r * 0.999])
        return out
