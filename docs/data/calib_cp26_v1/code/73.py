import numpy as np

def solve():
    import time as _tm
    import numpy as _np
    from scipy.optimize import minimize, linprog
    n = 26
    t0 = _tm.time()
    DEADLINE = 124.0
    PHASE1_END = 52.0

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
            dvec = _np.sqrt(dx * dx + dy * dy)
            bounds = [(0.0, float(bi)) for bi in b]
            res = linprog(c_obj, A_ub=A_ub0, b_ub=dvec, bounds=bounds, method='highs')
            if (not res.success) or res.x is None:
                return None, -1.0
            r = _np.asarray(res.x, dtype=float).flatten()
            r = _np.maximum(r, 0.0)
            s = float(_np.sum(r))
            if not _np.isfinite(s):
                return None, -1.0
            return r, s
        except Exception:
            return None, -1.0

    def safe_shrink(xc, yc, rc):
        rc = _np.maximum(np.asarray(rc, dtype=float), 1e-12)
        b = _np.minimum(_np.minimum(xc, 1.0 - xc), _np.minimum(yc, 1.0 - yc))
        t = 1.0
        for i in range(n):
            ti = b[i] / rc[i]
            if ti < t:
                t = ti
        dx = xc[iu] - xc[ju]
        dy = yc[iu] - yc[ju]
        dvec = _np.sqrt(dx * dx + dy * dy)
        rs = rc[iu] + rc[ju]
        mask = rs > 1e-12
        if _np.any(mask):
            tp = _np.min(dvec[mask] / rs[mask])
            if tp < t:
                t = tp
        t = t * (1.0 - 1e-7) - 1e-12
        if (not _np.isfinite(t)) or t <= 0:
            t = 0.5
        if t > 1.0:
            t = 1.0 - 1e-7
        return rc * t

    # fallback grid
    k = int(_np.ceil(_np.sqrt(n)))
    rr = 0.5 / k
    _pts = []
    for i in range(k):
        for j in range(k):
            if len(_pts) < n:
                _pts.append([(2 * i + 1) * rr, (2 * j + 1) * rr, rr * 0.999])
    _pts = _np.array(_pts, dtype=float)
    best_x = _pts[:, 0].copy(); best_y = _pts[:, 1].copy(); best_r = _pts[:, 2].copy()
    best_r = safe_shrink(best_x, best_y, best_r)
    best_sum = float(_np.sum(best_r))

    bnds = [(0.0, 1.0)] * (2 * n) + [(0.005, 0.34)] * n
    P_full = [20.0, 200.0, 2000.0, 20000.0]
    P_short = [300.0, 4000.0]

    def run_penalty(x0, y0, r0, P_seq, maxiter_each, tlim=1e9):
        z = _np.concatenate([x0, y0, r0])
        for P in P_seq:
            if _tm.time() - t0 > tlim:
                break
            if _tm.time() - t0 > DEADLINE - 3.0:
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
                dx = x[iu] - x[ju]
                dy = y[iu] - y[ju]
                D = _np.sqrt(dx * dx + dy * dy + 1e-20)
                S = r[iu] + r[ju] - D
                pos = S > 0
                if _np.any(pos):
                    Sp = S[pos]
                    f += P * 0.5 * _np.sum(Sp * Sp)
                    coeff = P * Sp / D[pos]
                    dxp = dx[pos]; dyp = dy[pos]
                    iup = iu[pos]; jup = ju[pos]
                    _np.add.at(gx, iup, -coeff * dxp)
                    _np.add.at(gx, jup, coeff * dxp)
                    _np.add.at(gy, iup, -coeff * dyp)
                    _np.add.at(gy, jup, coeff * dyp)
                    _np.add.at(gr, iup, P * Sp)
                    _np.add.at(gr, jup, P * Sp)
                g = _np.concatenate([gx, gy, gr])
                return f, g
            try:
                res = minimize(fg, z, method='L-BFGS-B', jac=True, bounds=bnds,
                               options={'maxiter': maxiter_each, 'ftol': 1e-10, 'gtol': 1e-6})
                z = res.x
            except Exception:
                break
        return z

    def make_init(ridx):
        rng = _np.random.RandomState(1000 + ridx * 7919)
        mode = ridx % 10
        if mode == 0:
            x0 = rng.rand(n); y0 = rng.rand(n)
            r0 = 0.055 + 0.055 * rng.rand(n)
        elif mode == 1:
            base = []
            for ix in range(5):
                for jx in range(5):
                    base.append([0.1 + 0.2 * ix, 0.1 + 0.2 * jx])
            base = _np.array(base)
            extra = rng.rand(1, 2)
            base = _np.vstack([base, extra])
            x0 = base[:, 0] + 0.035 * (rng.rand(n) - 0.5)
            y0 = base[:, 1] + 0.035 * (rng.rand(n) - 0.5)
            r0 = 0.065 + 0.045 * rng.rand(n)
        elif mode == 2:
            x0 = rng.rand(n); y0 = rng.rand(n)
            r0 = 0.04 + 0.03 * rng.rand(n)
            kl = 5 + int(rng.randint(0, 4))
            idx = rng.choice(n, kl, replace=False)
            r0[idx] = 0.105 + 0.045 * rng.rand(kl)
        elif mode == 3:
            pts = []
            yy = 0.08; row = 0
            while len(pts) < n and yy < 0.96:
                off = 0.08 if (row % 2 == 0) else 0.17
                xx = off
                while len(pts) < n and xx < 0.96:
                    pts.append([xx, yy]); xx += 0.18
                yy += 0.155; row += 1
            pts = _np.array(pts)
            if len(pts) < n:
                extra = rng.rand(n - len(pts), 2) * 0.8 + 0.1
                pts = _np.vstack([pts, extra])
            x0 = pts[:n, 0] + 0.03 * (rng.rand(n) - 0.5)
            y0 = pts[:n, 1] + 0.03 * (rng.rand(n) - 0.5)
            r0 = 0.05 + 0.06 * rng.rand(n)
        elif mode == 4:
            templates = [[6,5,5,5,5],[5,6,5,5,5],[5,5,6,5,5],[6,6,4,5,5],[6,5,6,4,5],[4,6,6,5,5],[5,4,6,6,5],[5,6,4,6,5]]
            counts = templates[rng.randint(len(templates))][:]
            rng.shuffle(counts)
            xs = []; ys = []
            for ri, cnt in enumerate(counts):
                yy = (ri + 0.5) / 5.0
                for kk in range(cnt):
                    xx = (kk + 0.5) / cnt
                    xs.append(xx); ys.append(yy)
            x0 = _np.array(xs) + 0.03 * (rng.rand(n) - 0.5)
            y0 = _np.array(ys) + 0.03 * (rng.rand(n) - 0.5)
            r0 = 0.06 + 0.05 * rng.rand(n)
        elif mode == 5:
            templates = [[6,5,5,5,5],[5,6,5,5,5],[6,6,4,5,5],[5,5,6,5,5]]
            counts = templates[rng.randint(len(templates))][:]
            xs = []; ys = []
            for ci, cnt in enumerate(counts):
                xx = (ci + 0.5) / 5.0
                for kk in range(cnt):
                    yy = (kk + 0.5) / cnt
                    xs.append(xx); ys.append(yy)
            # counts sum may not be 26; fix by sampling
            xs = _np.array(xs); ys = _np.array(ys)
            if len(xs) != n:
                # pad or trim with random
                if len(xs) < n:
                    ex = rng.rand(n - len(xs)); ey = rng.rand(n - len(xs))
                    xs = _np.concatenate([xs, ex]); ys = _np.concatenate([ys, ey])
                else:
                    perm = rng.permutation(len(xs))[:n]
                    xs = xs[perm]; ys = ys[perm]
            x0 = xs + 0.03 * (rng.rand(n) - 0.5)
            y0 = ys + 0.03 * (rng.rand(n) - 0.5)
            r0 = 0.06 + 0.05 * rng.rand(n)
        elif mode == 6:
            # left-right mirror
            h = n // 2
            xl = 0.05 + 0.45 * rng.rand(h)
            yl = 0.05 + 0.90 * rng.rand(h)
            rl = 0.05 + 0.07 * rng.rand(h)
            x0 = _np.concatenate([xl, 1.0 - xl])
            y0 = _np.concatenate([yl, yl])
            r0 = _np.concatenate([rl, rl])
            # jitter to break exact symmetry slightly for optimizer diversity
            x0 += 0.005 * (rng.rand(n) - 0.5)
            y0 += 0.005 * (rng.rand(n) - 0.5)
        elif mode == 7:
            h = n // 2
            xp = rng.rand(h); yp = rng.rand(h)
            rp = 0.05 + 0.07 * rng.rand(h)
            x0 = _np.concatenate([0.08 + 0.84 * xp, 0.08 + 0.84 * (1.0 - xp)])
            y0 = _np.concatenate([0.08 + 0.84 * yp, 0.08 + 0.84 * (1.0 - yp)])
            r0 = _np.concatenate([rp, rp])
            x0 += 0.005 * (rng.rand(n) - 0.5)
            y0 += 0.005 * (rng.rand(n) - 0.5)
        elif mode == 8:
            # corner anchored large
            x0 = 0.08 + 0.84 * rng.rand(n); y0 = 0.08 + 0.84 * rng.rand(n)
            r0 = 0.04 + 0.04 * rng.rand(n)
            corners = _np.array([[0.12, 0.12], [0.88, 0.12], [0.12, 0.88], [0.88, 0.88]])
            for ci in range(4):
                x0[ci] = corners[ci, 0] + 0.02 * (rng.rand() - 0.5)
                y0[ci] = corners[ci, 1] + 0.02 * (rng.rand() - 0.5)
                r0[ci] = 0.115 + 0.03 * rng.rand()
            # two more large in middle bottom
            r0[4] = 0.12 + 0.02 * rng.rand(); r0[5] = 0.12 + 0.02 * rng.rand()
        else:
            # clustered mixture
            nc = 5
            cx = rng.rand(nc); cy = rng.rand(nc)
            x0 = _np.zeros(n); y0 = _np.zeros(n)
            for i in range(n):
                c = rng.randint(nc)
                x0[i] = cx[c] + 0.09 * rng.randn()
                y0[i] = cy[c] + 0.09 * rng.randn()
            r0 = 0.05 + 0.06 * rng.rand(n)
        x0 = _np.clip(x0, 0.02, 0.98); y0 = _np.clip(y0, 0.02, 0.98)
        r0 = _np.clip(r0, 0.01, 0.30)
        return x0, y0, r0

    # Phase 1
    ridx = 0
    while True:
        if _tm.time() - t0 > PHASE1_END:
            break
        if ridx > 600:
            break
        x0, y0, r0 = make_init(ridx)
        z = run_penalty(x0, y0, r0, P_full, 90, tlim=PHASE1_END + 50.0)
        xp = _np.clip(z[0:n], 0.02, 0.98); yp = _np.clip(z[n:2 * n], 0.02, 0.98)
        r_opt, s = eval_lp(xp, yp)
        if r_opt is not None and _np.all(r_opt > 0.002) and s > best_sum:
            best_sum = s; best_x = xp.copy(); best_y = yp.copy(); best_r = r_opt.copy()
        ridx += 1
        # also try direct LP on initial centres (cheap, catches different topology)
        if ridx % 3 == 0:
            r2, s2 = eval_lp(x0, y0)
            if r2 is not None and _np.all(r2 > 0.002) and s2 > best_sum:
                best_sum = s2; best_x = x0.copy(); best_y = y0.copy(); best_r = r2.copy()

    rng2 = _np.random.RandomState(777)
    # Gaussian hill climb on LP surface
    for sigma in [0.02, 0.012, 0.006, 0.003]:
        for _ in range(70):
            if _tm.time() - t0 > DEADLINE - 55.0:
                break
            xn = _np.clip(best_x + sigma * rng2.randn(n), 0.02, 0.98)
            yn = _np.clip(best_y + sigma * rng2.randn(n), 0.02, 0.98)
            r_opt, s = eval_lp(xn, yn)
            if r_opt is not None and _np.all(r_opt > 0.002) and s > best_sum + 1e-10:
                best_sum = s; best_x = xn; best_y = yn; best_r = r_opt
        if _tm.time() - t0 > DEADLINE - 55.0:
            break

    def largest_void(xc, yc, rc, ncand=500):
        cx = rng2.rand(ncand); cy = rng2.rand(ncand)
        best_vx = 0.5; best_vy = 0.5; best_vd = -1e100
        for b in range(0, ncand, 200):
            px = cx[b:b + 200]; py = cy[b:b + 200]
            dx = px[:, None] - xc[None, :]
            dy = py[:, None] - yc[None, :]
            dd = _np.sqrt(dx * dx + dy * dy) - rc[None, :]
            wall = _np.minimum(_np.minimum(px, 1 - px), _np.minimum(py, 1 - py))
            vd = _np.minimum(_np.min(dd, axis=1), wall)
            j = int(_np.argmax(vd))
            if vd[j] > best_vd:
                best_vd = float(vd[j]); best_vx = float(px[j]); best_vy = float(py[j])
        return best_vx, best_vy, best_vd

    # ruin and recreate
    it = 0
    while _tm.time() - t0 < DEADLINE - 14.0:
        it += 1
        if it > 5000:
            break
        a = rng2.rand()
        if a < 0.42:
            vx, vy, _ = largest_void(best_x, best_y, best_r, ncand=500)
            order = _np.argsort(best_r)
            nv = 1 if rng2.rand() < 0.7 else 2
            victims = order[:nv].copy()
            if rng2.rand() < 0.25:
                victims = rng2.choice(n, nv, replace=False)
            xn = best_x.copy(); yn = best_y.copy()
            for vi, v in enumerate(victims):
                if vi == 0:
                    xn[v] = _np.clip(vx + 0.015 * rng2.randn(), 0.02, 0.98)
                    yn[v] = _np.clip(vy + 0.015 * rng2.randn(), 0.02, 0.98)
                else:
                    xn[v] = _np.clip(rng2.rand(), 0.02, 0.98)
                    yn[v] = _np.clip(rng2.rand(), 0.02, 0.98)
            # sometimes short rejamm
            if rng2.rand() < 0.30:
                r0 = _np.clip(best_r, 0.01, 0.30)
                # move victims radii to medium
                for v in victims:
                    r0[v] = 0.06 + 0.03 * rng2.rand()
                try:
                    z = run_penalty(xn, yn, r0, P_short, 35, tlim=DEADLINE - 12.0)
                    xn = _np.clip(z[0:n], 0.02, 0.98); yn = _np.clip(z[n:2 * n], 0.02, 0.98)
                except Exception:
                    pass
            r_opt, s = eval_lp(xn, yn)
            if r_opt is not None and _np.all(r_opt > 0.002) and s > best_sum + 1e-10:
                best_sum = s; best_x = xn; best_y = yn; best_r = r_opt
        elif a < 0.62:
            # swap two centres
            i, j = rng2.choice(n, 2, replace=False)
            xn = best_x.copy(); yn = best_y.copy()
            xn[i], xn[j] = best_x[j], best_x[i]
            yn[i], yn[j] = best_y[j], best_y[i]
            xn[i] += 0.004 * rng2.randn(); yn[i] += 0.004 * rng2.randn()
            xn[j] += 0.004 * rng2.randn(); yn[j] += 0.004 * rng2.randn()
            xn = _np.clip(xn, 0.02, 0.98); yn = _np.clip(yn, 0.02, 0.98)
            if rng2.rand() < 0.25:
                r0 = _np.clip(best_r, 0.01, 0.30)
                try:
                    z = run_penalty(xn, yn, r0, P_short, 30, tlim=DEADLINE - 12.0)
                    xn = _np.clip(z[0:n], 0.02, 0.98); yn = _np.clip(z[n:2 * n], 0.02, 0.98)
                except Exception:
                    pass
            r_opt, s = eval_lp(xn, yn)
            if r_opt is not None and _np.all(r_opt > 0.002) and s > best_sum + 1e-10:
                best_sum = s; best_x = xn; best_y = yn; best_r = r_opt
        else:
            sigma = 0.008 if rng2.rand() < 0.6 else 0.003
            xn = _np.clip(best_x + sigma * rng2.randn(n), 0.02, 0.98)
            yn = _np.clip(best_y + sigma * rng2.randn(n), 0.02, 0.98)
            r_opt, s = eval_lp(xn, yn)
            if r_opt is not None and _np.all(r_opt > 0.002) and s > best_sum + 1e-10:
                best_sum = s; best_x = xn; best_y = yn; best_r = r_opt

    # per-circle coordinate polish
    try:
        order0 = list(range(n))
        for sweep in range(3):
            if _tm.time() - t0 > DEADLINE - 5.0:
                break
            rng2.shuffle(order0)
            step = 0.006 / (1.5 ** sweep)
            for i in order0:
                if _tm.time() - t0 > DEADLINE - 5.0:
                    break
                improved = False
                for ang in [0, 0.7853981, 1.5707963, 2.3561945, 3.14159265, 3.9269908, 4.71238898, 5.49778714]:
                    xn = best_x.copy(); yn = best_y.copy()
                    xn[i] = _np.clip(best_x[i] + step * _np.cos(ang), 0.02, 0.98)
                    yn[i] = _np.clip(best_y[i] + step * _np.sin(ang), 0.02, 0.98)
                    r_opt, s = eval_lp(xn, yn)
                    if r_opt is not None and _np.all(r_opt > 0.002) and s > best_sum + 1e-10:
                        best_sum = s; best_x = xn; best_y = yn; best_r = r_opt
                        improved = True
                        break
                # extra smaller step if improved
                if improved:
                    for ang in [0, 1.5707963, 3.14159265, 4.71238898]:
                        xn = best_x.copy(); yn = best_y.copy()
                        xn[i] = _np.clip(best_x[i] + 0.4 * step * _np.cos(ang), 0.02, 0.98)
                        yn[i] = _np.clip(best_y[i] + 0.4 * step * _np.sin(ang), 0.02, 0.98)
                        r_opt, s = eval_lp(xn, yn)
                        if r_opt is not None and _np.all(r_opt > 0.002) and s > best_sum + 1e-10:
                            best_sum = s; best_x = xn; best_y = yn; best_r = r_opt
                            break
    except Exception:
        pass

    # final joint rejamm from best then LP
    try:
        if _tm.time() - t0 < DEADLINE - 4.0:
            r0 = _np.clip(best_r, 0.01, 0.30)
            z = run_penalty(best_x.copy(), best_y.copy(), r0, [500.0, 5000.0, 20000.0], 60, tlim=DEADLINE - 3.0)
            xp = _np.clip(z[0:n], 0.02, 0.98); yp = _np.clip(z[n:2 * n], 0.02, 0.98)
            r_opt, s = eval_lp(xp, yp)
            if r_opt is not None and _np.all(r_opt > 0.002) and s > best_sum:
                best_sum = s; best_x = xp; best_y = yp; best_r = r_opt
    except Exception:
        pass

    rf = safe_shrink(best_x, best_y, best_r)
    out = _np.column_stack([best_x, best_y, rf])
    return out
