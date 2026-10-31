import numpy as np

def solve():
    import time as _tm
    import numpy as _np
    from scipy.optimize import minimize, linprog
    n = 26
    t0 = _tm.time()
    DEADLINE = 124.0
    PHASE1_END = 58.0

    iu, ju = _np.triu_indices(n, 1)
    m = len(iu)
    A_ub0 = _np.zeros((m, n), dtype=float)
    _ar = _np.arange(m)
    A_ub0[_ar, iu] = 1.0
    A_ub0[_ar, ju] = 1.0
    c_obj = -_np.ones(n)

    MARGIN = 1e-9

    def eval_lp(xc, yc):
        # wall upper bounds with margin
        b = _np.minimum(_np.minimum(xc, 1.0 - xc), _np.minimum(yc, 1.0 - yc)) - MARGIN
        if _np.any(b < 0.004):
            return None, -1.0
        dx = xc[iu] - xc[ju]
        dy = yc[iu] - yc[ju]
        dvec = _np.sqrt(dx * dx + dy * dy) - MARGIN
        if _np.any(dvec < 0.008):
            return None, -1.0
        bounds = [(0.0, float(bi)) for bi in b]
        try:
            res = linprog(c_obj, A_ub=A_ub0, b_ub=dvec, bounds=bounds, method='highs')
        except Exception:
            return None, -1.0
        if (not res.success) or res.x is None:
            return None, -1.0
        r = _np.asarray(res.x, dtype=float).flatten()
        r = _np.maximum(r, 0.0)
        if _np.any(r < 1e-6):
            return None, -1.0
        # extra shrink for strict 1e-12 verifier
        r = r * (1.0 - 1e-7)
        # verify quickly
        s = float(_np.sum(r))
        return r, s

    def uniform_safe_scale(xc, yc, rc):
        b = _np.minimum(_np.minimum(xc, 1.0 - xc), _np.minimum(yc, 1.0 - yc))
        dx = xc[iu] - xc[ju]
        dy = yc[iu] - yc[ju]
        dvec = _np.sqrt(dx * dx + dy * dy)
        rs = rc[iu] + rc[ju]
        t = 1.0
        # wall
        for i in range(n):
            if rc[i] > 1e-12:
                ti = b[i] / rc[i]
                if ti < t:
                    t = ti
        mask = rs > 1e-12
        if _np.any(mask):
            tp = _np.min(dvec[mask] / rs[mask])
            if tp < t:
                t = tp
        t = t * (1.0 - 1e-9) - 1e-12
        if (not _np.isfinite(t)) or t <= 0:
            t = 0.5
        if t > 1.0:
            t = 1.0 - 1e-9
        return rc * t

    bnds = [(0.0, 1.0)] * (2 * n) + [(0.01, 0.32)] * n
    P_seq_full = [20.0, 200.0, 2000.0, 20000.0]
    P_seq_short = [200.0, 5000.0]

    def run_penalty(x0, y0, r0, P_seq, maxiter_each):
        z = _np.concatenate([x0, y0, r0])
        for P in P_seq:
            if _tm.time() - t0 > PHASE1_END + 55.0:
                # in phase2 allow longer, check deadline
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
                g = _np.concatenate([gx * P if False else gx, gy, gr])  # placeholder
                # NOTE: wall grads already include P factor via co; pairwise gx,gy already include P
                # gr currently = -1 + P*... ; gx,gy = P*...
                # So assemble: gx,gy as is, gr as is
                g = _np.concatenate([gx, gy, gr])
                return f, g
            try:
                res = minimize(fg, z, method='L-BFGS-B', jac=True, bounds=bnds,
                               options={'maxiter': maxiter_each, 'ftol': 1e-10, 'gtol': 1e-6})
                z = res.x
            except Exception:
                break
        return z

    best_sum = -1.0
    best_x = None; best_y = None; best_r = None

    def consider(xc, yc):
        global_vars = None
        r_opt, s = eval_lp(xc, yc)
        return r_opt, s

    # ---------- Phase 1 : diverse restarts ----------
    restart_idx = 0
    rng_master = _np.random.RandomState(20260214)
    # hex grid points for init
    while True:
        if _tm.time() - t0 > PHASE1_END:
            break
        if restart_idx > 1000:
            break
        rng = _np.random.RandomState(1000 + restart_idx * 7919)
        mode = restart_idx % 7
        if mode == 0:
            x0 = rng.rand(n); y0 = rng.rand(n)
            r0 = 0.06 + 0.05 * rng.rand(n)
        elif mode == 1:
            # perturbed 5x5 + 1
            base = []
            for ix in range(5):
                for jx in range(5):
                    base.append([0.1 + 0.2 * ix, 0.1 + 0.2 * jx])
            base = _np.array(base)
            # add one extra random
            extra = rng.rand(1, 2)
            base = _np.vstack([base, extra])
            perm = rng.permutation(n)
            # shuffle assignment
            x0 = base[:, 0] + 0.04 * (rng.rand(n) - 0.5)
            y0 = base[:, 1] + 0.04 * (rng.rand(n) - 0.5)
            x0 = _np.clip(x0, 0.05, 0.95); y0 = _np.clip(y0, 0.05, 0.95)
            r0 = 0.07 + 0.04 * rng.rand(n)
        elif mode == 2:
            x0 = rng.rand(n); y0 = rng.rand(n)
            # bimodal radii: few large
            r0 = 0.04 + 0.03 * rng.rand(n)
            klarge = 5 + int(rng.randint(0, 4))
            idx = rng.choice(n, klarge, replace=False)
            r0[idx] = 0.11 + 0.04 * rng.rand(klarge)
        elif mode == 3:
            # hex init
            pts = []
            yy = 0.08
            row = 0
            while len(pts) < n and yy < 0.95:
                off = 0.08 if (row % 2 == 0) else 0.17
                xx = off
                while len(pts) < n and xx < 0.95:
                    pts.append([xx, yy])
                    xx += 0.18
                yy += 0.155
                row += 1
            pts = _np.array(pts)
            if len(pts) < n:
                extra = rng.rand(n - len(pts), 2) * 0.8 + 0.1
                pts = _np.vstack([pts, extra])
            x0 = pts[:n, 0] + 0.03 * (rng.rand(n) - 0.5)
            y0 = pts[:n, 1] + 0.03 * (rng.rand(n) - 0.5)
            x0 = _np.clip(x0, 0.05, 0.95); y0 = _np.clip(y0, 0.05, 0.95)
            r0 = 0.05 + 0.06 * rng.rand(n)
        elif mode == 4:
            x0 = rng.rand(n); y0 = rng.rand(n)
            r0 = 0.03 + 0.10 * rng.rand(n)
        elif mode == 5:
            # rows with jitter
            counts = [5, 5, 6, 5, 5]  # sums 26
            rng.shuffle(counts)
            xs = []; ys = []
            for ri, cnt in enumerate(counts):
                yy = (ri + 0.5) / 5.0
                for k in range(cnt):
                    xx = (k + 0.5) / cnt
                    xs.append(xx); ys.append(yy)
            x0 = _np.array(xs) + 0.03 * (rng.rand(n) - 0.5)
            y0 = _np.array(ys) + 0.03 * (rng.rand(n) - 0.5)
            x0 = _np.clip(x0, 0.05, 0.95); y0 = _np.clip(y0, 0.05, 0.95)
            r0 = 0.06 + 0.05 * rng.rand(n)
        else:
            x0 = 0.1 + 0.8 * rng.rand(n); y0 = 0.1 + 0.8 * rng.rand(n)
            r0 = 0.055 + 0.045 * rng.rand(n)

        r0 = _np.clip(r0, 0.01, 0.30)
        x0 = _np.clip(x0, 0.02, 0.98); y0 = _np.clip(y0, 0.02, 0.98)

        z = run_penalty(x0, y0, r0, P_seq_full, 110)
        xp = z[0:n]; yp = z[n:2 * n]
        xp = _np.clip(xp, 0.02, 0.98); yp = _np.clip(yp, 0.02, 0.98)
        r_opt, s = eval_lp(xp, yp)
        if r_opt is not None and s > best_sum:
            best_sum = s
            best_x = xp.copy(); best_y = yp.copy(); best_r = r_opt.copy()
        restart_idx += 1

    # ---------- Phase 2 : ruin-and-recreate + hill climbing ----------
    if best_x is None:
        k = int(_np.ceil(_np.sqrt(n)))
        rr = 0.5 / k
        pts = []
        for i in range(k):
            for j in range(k):
                if len(pts) < n:
                    pts.append([(2 * i + 1) * rr, (2 * j + 1) * rr, rr * 0.999])
        return _np.array(pts, dtype=float)

    rng2 = _np.random.RandomState(777)
    # quick Gaussian hill climb first
    sigmas = [0.02, 0.01, 0.005, 0.0025]
    for sigma in sigmas:
        for _ in range(90):
            if _tm.time() - t0 > DEADLINE - 8.0:
                break
            xn = best_x + sigma * rng2.randn(n)
            yn = best_y + sigma * rng2.randn(n)
            xn = _np.clip(xn, 0.02, 0.98); yn = _np.clip(yn, 0.02, 0.98)
            r_opt, s = eval_lp(xn, yn)
            if r_opt is not None and s > best_sum + 1e-10:
                best_sum = s; best_x = xn; best_y = yn; best_r = r_opt
        if _tm.time() - t0 > DEADLINE - 8.0:
            break

    # ruin-and-recreate with largest void + swaps
    n_void_cand = 800
    it = 0
    while _tm.time() - t0 < DEADLINE - 4.0:
        it += 1
        if it > 4000:
            break
        action = rng2.rand()
        if action < 0.45:
            # largest void teleport of smallest circle(s)
            # find void via random candidates
            cand_x = rng2.rand(n_void_cand)
            cand_y = rng2.rand(n_void_cand)
            # also include midpoints of pairs
            best_vx = 0.5; best_vy = 0.5; best_vd = -1e100
            # vectorized void search in batches
            for b in range(0, n_void_cand, 200):
                px = cand_x[b:b + 200]; py = cand_y[b:b + 200]
                dx = px[:, None] - best_x[None, :]
                dy = py[:, None] - best_y[None, :]
                dd = _np.sqrt(dx * dx + dy * dy) - best_r[None, :]
                wall = _np.minimum(_np.minimum(px, 1 - px)[:, None], _np.minimum(py, 1 - py)[:, None])
                # wall is (B,1); dd is (B,n)
                vd = _np.minimum(_np.min(dd, axis=1), wall[:, 0])
                j = int(_np.argmax(vd))
                if vd[j] > best_vd:
                    best_vd = float(vd[j]); best_vx = float(px[j]); best_vy = float(py[j])
            # pick victim: smallest radius (or random small)
            order = _np.argsort(best_r)
            nvict = 1 if rng2.rand() < 0.7 else 2
            victims = order[:nvict]
            # sometimes random victim
            if rng2.rand() < 0.25:
                victims = rng2.choice(n, nvict, replace=False)
            xn = best_x.copy(); yn = best_y.copy()
            for vi, victim in enumerate(victims):
                jx = 0.02 * rng2.randn(); jy = 0.02 * rng2.randn()
                if vi == 0:
                    xn[victim] = _np.clip(best_vx + jx, 0.02, 0.98)
                    yn[victim] = _np.clip(best_vy + jy, 0.02, 0.98)
                else:
                    xn[victim] = _np.clip(rng2.rand(), 0.02, 0.98)
                    yn[victim] = _np.clip(rng2.rand(), 0.02, 0.98)
            # short rejamm: run penalty starting from these centres with current radii
            r_start = best_r.copy()
            # perturb radii slightly
            r_start = r_start * (1.0 + 0.05 * rng2.randn(n))
            r_start = _np.clip(r_start, 0.01, 0.30)
            z = run_penalty(xn, yn, r_start, P_seq_short, 45)
            xp = _np.clip(z[0:n], 0.02, 0.98); yp = _np.clip(z[n:2 * n], 0.02, 0.98)
            r_opt, s = eval_lp(xp, yp)
            if r_opt is not None and s > best_sum + 1e-10:
                best_sum = s; best_x = xp; best_y = yp; best_r = r_opt
            else:
                # try direct LP without jamming
                r_opt2, s2 = eval_lp(xn, yn)
                if r_opt2 is not None and s2 > best_sum + 1e-10:
                    best_sum = s2; best_x = xn; best_y = yn; best_r = r_opt2
        elif action < 0.65:
            # swap two circles' centres
            i, j = rng2.choice(n, 2, replace=False)
            xn = best_x.copy(); yn = best_y.copy()
            xn[i], xn[j] = xn[j], xn[i]
            yn[i], yn[j] = yn[j], yn[i]
            xn += 0.004 * rng2.randn(n); yn += 0.004 * rng2.randn(n)
            xn = _np.clip(xn, 0.02, 0.98); yn = _np.clip(yn, 0.02, 0.98)
            r_opt, s = eval_lp(xn, yn)
            if r_opt is not None and s > best_sum + 1e-10:
                best_sum = s; best_x = xn; best_y = yn; best_r = r_opt
        else:
            # Gaussian step with adaptive sigma
            sigma = [0.012, 0.006, 0.002][rng2.randint(0, 3)]
            xn = best_x + sigma * rng2.randn(n)
            yn = best_y + sigma * rng2.randn(n)
            xn = _np.clip(xn, 0.02, 0.98); yn = _np.clip(yn, 0.02, 0.98)
            r_opt, s = eval_lp(xn, yn)
            if r_opt is not None and s > best_sum + 1e-10:
                best_sum = s; best_x = xn; best_y = yn; best_r = r_opt

    # final polish: tiny hill climb around best
    rng3 = _np.random.RandomState(999)
    for _ in range(200):
        if _tm.time() - t0 > DEADLINE - 1.5:
            break
        sigma = 0.0015
        xn = best_x + sigma * rng3.randn(n)
        yn = best_y + sigma * rng3.randn(n)
        xn = _np.clip(xn, 0.015, 0.985); yn = _np.clip(yn, 0.015, 0.985)
        r_opt, s = eval_lp(xn, yn)
        if r_opt is not None and s > best_sum + 1e-12:
            best_sum = s; best_x = xn; best_y = yn; best_r = r_opt

    x = best_x; y = best_y; r = best_r
    # final strict safety scaling
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
    if (not _np.isfinite(f)) or f >= 1.0:
        f = 1.0 - 1e-9
    else:
        f = f * (1.0 - 1e-9)
        if f <= 0:
            f = 0.5
    r = r * f
    # ensure positivity
    r = _np.maximum(r, 1e-6)
    # re-check after floor (floor could violate); scale again if needed
    wall2 = _np.minimum(_np.minimum(x, 1 - x), _np.minimum(y, 1 - y))
    if _np.any(r > wall2):
        sc = _np.min(wall2 / r) * (1.0 - 1e-9)
        r = r * sc
    out = _np.column_stack([x, y, r])
    return out
