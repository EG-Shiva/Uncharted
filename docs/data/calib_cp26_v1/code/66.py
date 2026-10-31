import numpy as np

def solve():
    import time as _tm
    import numpy as _np
    from scipy.optimize import minimize, linprog
    n = 26
    t0 = _tm.time()
    HARD = 124.0  # must return before 130
    # ---------- LP ----------
    iu, ju = _np.triu_indices(n, 1)
    m = len(iu)
    A_ub0 = _np.zeros((m, n), dtype=float)
    _ar = _np.arange(m)
    A_ub0[_ar, iu] = 1.0
    A_ub0[_ar, ju] = 1.0
    c_obj = -_np.ones(n)

    def eval_lp(xc, yc):
        b = _np.minimum(_np.minimum(xc, 1.0 - xc), _np.minimum(yc, 1.0 - yc))
        b = _np.maximum(b, 0.0)
        dx = xc[iu] - xc[ju]
        dy = yc[iu] - yc[ju]
        dvec = _np.sqrt(dx*dx + dy*dy)
        bounds = [(0.0, float(bi)) for bi in b]
        try:
            res = linprog(c_obj, A_ub=A_ub0, b_ub=dvec, bounds=bounds, method='highs')
        except Exception:
            return None, -1.0
        if res.status != 0 or res.x is None:
            return None, -1.0
        r = _np.asarray(res.x, dtype=float).flatten()
        r[_np.isnan(r)] = 0.0
        r = _np.maximum(r, 0.0)
        return r, float(_np.sum(r))

    def feasible_scale(xc, yc, rc):
        rc = _np.maximum(_np.asarray(rc, float).flatten(), 0.0)
        b = _np.minimum(_np.minimum(xc, 1.0 - xc), _np.minimum(yc, 1.0 - yc))
        t = 1.0
        nz = rc > 1e-12
        if _np.any(nz):
            # wall ratios
            wr = b[nz] / rc[nz]
            t = min(t, float(_np.min(wr)))
        dx = xc[iu] - xc[ju]; dy = yc[iu] - yc[ju]
        dvec = _np.sqrt(dx*dx + dy*dy)
        ssum = rc[iu] + rc[ju]
        mask = ssum > 1e-12
        if _np.any(mask):
            pr = dvec[mask] / ssum[mask]
            t = min(t, float(_np.min(pr)))
        if not _np.isfinite(t) or t <= 0:
            return _np.full(n, 1e-4), 0.0
        if t > 1.0:
            t = 1.0
        t = t * (1.0 - 3e-7)  # strict margin >>1e-12
        # also absolute tiny shave
        r2 = rc * t
        r2 = _np.maximum(r2, 1e-6)
        # re-check after forcing minimum (may violate); rescale once more if needed
        b2 = b
        nz2 = r2 > 0
        t2 = 1.0
        wr2 = b2 / _np.maximum(r2, 1e-18)
        t2 = min(t2, float(_np.min(wr2)))
        ssum2 = r2[iu] + r2[ju]
        mask2 = ssum2 > 1e-18
        if _np.any(mask2):
            pr2 = dvec[mask2] / ssum2[mask2]
            t2 = min(t2, float(_np.min(pr2)))
        if t2 < 1.0:
            r2 = r2 * t2 * (1.0 - 1e-9)
        return r2, float(_np.sum(r2))

    # ---------- fallback ----------
    k = int(_np.ceil(_np.sqrt(n)))
    rr = 0.5 / k
    fb_x = []; fb_y = []
    for i in range(k):
        for j in range(k):
            if len(fb_x) < n:
                fb_x.append((2*i+1)*rr); fb_y.append((2*j+1)*rr)
    fb_x = _np.array(fb_x); fb_y = _np.array(fb_y)
    fb_r0, _ = eval_lp(fb_x, fb_y)
    if fb_r0 is None:
        fb_r0 = _np.full(n, rr*0.5)
    fb_r, fb_s = feasible_scale(fb_x, fb_y, fb_r0)
    best_x = fb_x.copy(); best_y = fb_y.copy(); best_r = fb_r.copy(); best_sum = fb_s
    elites = [(fb_s, fb_x.copy(), fb_y.copy())]

    bnds = [(0.0, 1.0)]*(2*n) + [(1e-4, 0.35)]*n
    P_seq = [20.0, 200.0, 2000.0, 20000.0]

    def run_penalty(x0, y0, r0, maxiter_each=80):
        z = _np.concatenate([x0, y0, r0])
        for P in P_seq:
            if _tm.time() - t0 > 66.0:
                break
            def fg(zz, P=P):
                x = zz[0:n]; y = zz[n:2*n]; r = zz[2*n:3*n]
                f = -_np.sum(r)
                gx = _np.zeros(n); gy = _np.zeros(n); gr = -_np.ones(n)
                v1 = r - x
                mm = v1 > 0
                if _np.any(mm):
                    f += P*_np.sum(v1[mm]*v1[mm])
                    co = 2.0*P*v1[mm]
                    gx[mm] -= co; gr[mm] += co
                v2 = x + r - 1.0
                mm = v2 > 0
                if _np.any(mm):
                    f += P*_np.sum(v2[mm]*v2[mm])
                    co = 2.0*P*v2[mm]
                    gx[mm] += co; gr[mm] += co
                v3 = r - y
                mm = v3 > 0
                if _np.any(mm):
                    f += P*_np.sum(v3[mm]*v3[mm])
                    co = 2.0*P*v3[mm]
                    gy[mm] -= co; gr[mm] += co
                v4 = y + r - 1.0
                mm = v4 > 0
                if _np.any(mm):
                    f += P*_np.sum(v4[mm]*v4[mm])
                    co = 2.0*P*v4[mm]
                    gy[mm] += co; gr[mm] += co
                dxm = x[:, None] - x[None, :]
                dym = y[:, None] - y[None, :]
                D = _np.sqrt(dxm*dxm + dym*dym)
                Rsum = r[:, None] + r[None, :]
                S = Rsum - D
                _np.fill_diagonal(S, -1.0)
                mask = S > 0
                if _np.any(mask):
                    f += P*0.5*_np.sum(S[mask]*S[mask])
                    Dm = D + 1e-12
                    W = _np.zeros((n, n))
                    W[mask] = (2.0*P*S[mask]) / Dm[mask]
                    rs = W.sum(axis=1)
                    gx += W.dot(x) - rs*x
                    gy += W.dot(y) - rs*y
                    Spos = _np.where(mask, S, 0.0)
                    gr += 2.0*P*Spos.sum(axis=1)
                g = _np.concatenate([gx, gy, gr])
                return f, g
            try:
                res = minimize(fg, z, method='L-BFGS-B', jac=True, bounds=bnds,
                               options={'maxiter': maxiter_each, 'ftol': 1e-10, 'gtol': 1e-6})
                z = res.x
            except Exception:
                break
        return z

    # ---------- phase 1: diverse penalty restarts ----------
    restart_idx = 0
    # hex template
    hex_x = []; hex_y = []
    for row in range(6):
        for col in range(6):
            if len(hex_x) >= 30:
                break
            xx = (col + 0.5*(row % 2))*0.17 + 0.05
            yy = row*0.17 + 0.06
            if xx < 0.98 and yy < 0.98:
                hex_x.append(xx); hex_y.append(yy)
    hex_x = _np.array(hex_x); hex_y = _np.array(hex_y)
    while True:
        if _tm.time() - t0 > 66.0:
            break
        if restart_idx > 600:
            break
        rng = _np.random.RandomState(1000 + restart_idx*7919)
        mode = restart_idx % 7
        if mode == 0:
            x0 = rng.uniform(0.05, 0.95, n); y0 = rng.uniform(0.05, 0.95, n)
        elif mode == 1:
            # jittered 6x5 grid
            gx = []; gy = []
            for ix in range(6):
                for jy in range(5):
                    gx.append((ix+0.5)/6.0); gy.append((jy+0.5)/5.0)
            gx = _np.array(gx); gy = _np.array(gy)
            perm = rng.permutation(len(gx))[:n]
            x0 = gx[perm] + rng.normal(0, 0.035, n)
            y0 = gy[perm] + rng.normal(0, 0.035, n)
            x0 = _np.clip(x0, 0.02, 0.98); y0 = _np.clip(y0, 0.02, 0.98)
        elif mode == 2:
            # 5x5 + extra jittered
            xs = []; ys = []
            for ix in range(5):
                for jy in range(5):
                    xs.append(0.1+0.2*ix); ys.append(0.1+0.2*jy)
            xs.append(0.5); ys.append(0.5)
            x0 = _np.array(xs[:n]) + rng.normal(0, 0.045, n)
            y0 = _np.array(ys[:n]) + rng.normal(0, 0.045, n)
            x0 = _np.clip(x0, 0.02, 0.98); y0 = _np.clip(y0, 0.02, 0.98)
        elif mode == 3:
            perm = rng.permutation(len(hex_x))[:n]
            x0 = hex_x[perm] + rng.normal(0, 0.03, n)
            y0 = hex_y[perm] + rng.normal(0, 0.03, n)
            x0 = _np.clip(x0, 0.02, 0.98); y0 = _np.clip(y0, 0.02, 0.98)
        elif mode == 4 and len(elites) > 0:
            bi = rng.integers(0, len(elites))
            _, bx, by = elites[bi % len(elites)]
            sig = 0.02 + 0.06*rng.random()
            x0 = bx + rng.normal(0, sig, n)
            y0 = by + rng.normal(0, sig, n)
            x0 = _np.clip(x0, 0.01, 0.99); y0 = _np.clip(y0, 0.01, 0.99)
        elif mode == 5:
            # corner compressed
            x0 = rng.uniform(0.05, 0.95, n); y0 = rng.uniform(0.05, 0.95, n)
            ccx = float(rng.choice([0.0, 1.0])); ccy = float(rng.choice([0.0, 1.0]))
            f0 = 0.80 + 0.12*rng.random()
            x0 = ccx + (x0-ccx)*f0; y0 = ccy + (y0-ccy)*f0
            x0 = _np.clip(x0, 0.02, 0.98); y0 = _np.clip(y0, 0.02, 0.98)
        else:
            x0 = rng.uniform(0.03, 0.97, n); y0 = rng.uniform(0.03, 0.97, n)
        # radii init bimodal to seed heterogeneity
        if rng.random() < 0.5:
            r0 = rng.uniform(0.04, 0.12, n)
        else:
            r0 = _np.empty(n)
            half = n//2
            r0[:half] = rng.uniform(0.09, 0.13, half)
            r0[half:] = rng.uniform(0.03, 0.07, n-half)
            rng.shuffle(r0)
        try:
            z = run_penalty(_np.clip(x0,0,1), _np.clip(y0,0,1), r0, maxiter_each=70)
            xc = z[0:n]; yc = z[n:2*n]
            xc = _np.clip(xc, 0.001, 0.999); yc = _np.clip(yc, 0.001, 0.999)
        except Exception:
            restart_idx += 1
            continue
        r_lp, s_lp = eval_lp(xc, yc)
        if r_lp is not None and s_lp > 0:
            r_feas, s_feas = feasible_scale(xc, yc, r_lp)
            elites.append((s_feas, xc.copy(), yc.copy()))
            elites = sorted(elites, key=lambda z2: -z2[0])[:10]
            if s_feas > best_sum:
                best_sum = s_feas; best_x = xc.copy(); best_y = yc.copy(); best_r = r_feas.copy()
        restart_idx += 1

    # ---------- phase 2: LP-scored ruin-and-recreate hill climbing ----------
    # uses only centres + eval_lp, very fast, changes topology
    hc_rng = _np.random.RandomState(777)
    it = 0
    while _tm.time() - t0 < 112.0 and it < 8000:
        if len(elites) == 0:
            break
        # pick elite (bias to best)
        if hc_rng.random() < 0.6:
            s0, bx, by = elites[0]
        else:
            bi = hc_rng.integers(0, len(elites))
            s0, bx, by = elites[bi]
        xc = bx.copy(); yc = by.copy()
        mv = hc_rng.random()
        if mv < 0.45:
            # gaussian jiggle of subset
            ksub = hc_rng.integers(1, 7)
            idx = hc_rng.choice(n, size=ksub, replace=False)
            sig = float(hc_rng.choice([0.003, 0.008, 0.02, 0.05]))
            xc[idx] += hc_rng.normal(0, sig, ksub)
            yc[idx] += hc_rng.normal(0, sig, ksub)
            xc = _np.clip(xc, 0.01, 0.99); yc = _np.clip(yc, 0.01, 0.99)
        elif mv < 0.70:
            # teleport 1-2 circles to random voids
            ksub = 1 if hc_rng.random() < 0.7 else 2
            # need radii to find small ones: eval current
            r_cur, _ = eval_lp(bx, by)
            if r_cur is not None:
                order = _np.argsort(r_cur)
                idx = order[:ksub]
            else:
                idx = hc_rng.choice(n, size=ksub, replace=False)
            xc[idx] = hc_rng.uniform(0.03, 0.97, ksub)
            yc[idx] = hc_rng.uniform(0.03, 0.97, ksub)
        elif mv < 0.85:
            # global small jitter
            sig = 0.004 + 0.008*hc_rng.random()
            xc += hc_rng.normal(0, sig, n)
            yc += hc_rng.normal(0, sig, n)
            xc = _np.clip(xc, 0.01, 0.99); yc = _np.clip(yc, 0.01, 0.99)
        else:
            # line compression / expansion toward random corner
            ccx = float(hc_rng.choice([0.0, 0.5, 1.0])); ccy = float(hc_rng.choice([0.0, 0.5, 1.0]))
            f = 0.94 + 0.12*hc_rng.random()  # 0.94..1.06
            xc = ccx + (xc-ccx)*f
            yc = ccy + (yc-ccy)*f
            xc = _np.clip(xc, 0.01, 0.99); yc = _np.clip(yc, 0.01, 0.99)
        r_lp, s_lp = eval_lp(xc, yc)
        it += 1
        if r_lp is None:
            continue
        r_feas, s_feas = feasible_scale(xc, yc, r_lp)
        if s_feas > s0 + 1e-10:
            elites.append((s_feas, xc.copy(), yc.copy()))
            elites = sorted(elites, key=lambda z2: -z2[0])[:10]
            if s_feas > best_sum:
                best_sum = s_feas; best_x = xc.copy(); best_y = yc.copy(); best_r = r_feas.copy()
        elif s_feas > elites[-1][0] and hc_rng.random() < 0.05 and len(elites) < 10:
            elites.append((s_feas, xc.copy(), yc.copy()))
            elites = sorted(elites, key=lambda z2: -z2[0])[:10]
        # occasional short rejamm from promising teleports
        if s_feas > best_sum - 0.02 and (_tm.time()-t0) < 105.0 and (it % 400 == 0):
            try:
                r_init = _np.clip(r_lp, 0.02, 0.2)
                z = run_penalty(xc, yc, r_init, maxiter_each=40)
                xc2 = _np.clip(z[0:n], 0.001, 0.999); yc2 = _np.clip(z[n:2*n], 0.001, 0.999)
                r2, s2 = eval_lp(xc2, yc2)
                if r2 is not None:
                    rf2, sf2 = feasible_scale(xc2, yc2, r2)
                    elites.append((sf2, xc2.copy(), yc2.copy()))
                    elites = sorted(elites, key=lambda z2: -z2[0])[:10]
                    if sf2 > best_sum:
                        best_sum = sf2; best_x = xc2.copy(); best_y = yc2.copy(); best_r = rf2.copy()
            except Exception:
                pass

    # ---------- phase 3: final jamming polish of top elites ----------
    try:
        cands = sorted(elites, key=lambda z2: -z2[0])[:3]
        for _, cx0, cy0 in cands:
            if _tm.time() - t0 > HARD - 6.0:
                break
            r_lp, _ = eval_lp(cx0, cy0)
            if r_lp is None:
                continue
            r_init = _np.clip(r_lp, 0.01, 0.25)
            try:
                z = run_penalty(cx0.copy(), cy0.copy(), r_init, maxiter_each=120)
                xc = _np.clip(z[0:n], 0.001, 0.999); yc = _np.clip(z[n:2*n], 0.001, 0.999)
            except Exception:
                continue
            r2, s2 = eval_lp(xc, yc)
            if r2 is None:
                continue
            rf, sf = feasible_scale(xc, yc, r2)
            if sf > best_sum:
                best_sum = sf; best_x = xc.copy(); best_y = yc.copy(); best_r = rf.copy()
    except Exception:
        pass

    out = _np.column_stack([best_x, best_y, best_r])
    # absolute final safety: rescale once more
    try:
        rf, _ = feasible_scale(out[:, 0], out[:, 1], out[:, 2])
        out[:, 2] = rf
    except Exception:
        pass
    return out
