import numpy as np
import time

def solve():
    n = 26
    t0 = time.time()
    T_PEN = 72.0
    T_HILL = 124.0
    rng = np.random.RandomState(0)

    # precompute pair incidence for LP
    I, J = np.triu_indices(n, 1)
    m = len(I)
    A0 = np.zeros((m, n), dtype=float)
    A0[np.arange(m), I] = 1.0
    A0[np.arange(m), J] = 1.0
    c = -np.ones(n, dtype=float)

    try:
        from scipy.optimize import minimize, linprog
        has_opt = True
    except Exception:
        has_opt = False

    # fallback grid
    k = int(np.ceil(np.sqrt(n)))
    rr0 = 0.5 / k
    fb = []
    for ix in range(k):
        for jy in range(k):
            if len(fb) < n:
                fb.append([(2*ix+1)*rr0, (2*jy+1)*rr0, rr0*0.999])
    fb_arr = np.array(fb, dtype=float)

    def lp_eval(xc, yc):
        b = np.minimum(np.minimum(xc, 1.0 - xc), np.minimum(yc, 1.0 - yc))
        if np.any(b < 0):
            return -1.0, None
        # clip tiny negatives
        b = np.maximum(b, 0.0)
        dx = xc[I] - xc[J]
        dy = yc[I] - yc[J]
        d = np.sqrt(dx*dx + dy*dy)
        bnds = [(0.0, float(bi)) for bi in b]
        try:
            res = linprog(c, A_ub=A0, b_ub=d, bounds=bnds, method='highs')
        except Exception:
            return -1.0, None
        if res.status != 0:
            return -1.0, None
        r = np.asarray(res.x, dtype=float)
        r = np.maximum(r, 0.0)
        s = float(np.sum(r))
        return s, r

    def feasible_scaled_sum(X, Y, R):
        R = np.maximum(np.asarray(R, float), 1e-12)
        X = np.asarray(X, float); Y = np.asarray(Y, float)
        wall = np.minimum(np.minimum(X, 1.0-X), np.minimum(Y, 1.0-Y))
        # if outside
        if np.any(wall < 0):
            return -1.0, None
        Dx = X[:, None] - X[None, :]
        Dy = Y[:, None] - Y[None, :]
        D = np.sqrt(Dx*Dx + Dy*Dy)
        S = R[:, None] + R[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_w = wall / R
            ratio_p = D / S
        np.fill_diagonal(ratio_p, np.inf)
        a = float(np.min(ratio_w))
        bb = float(np.min(ratio_p))
        alpha = min(1.0, a, bb)
        alpha = alpha * (1.0 - 1e-7) - 1e-12
        if not np.isfinite(alpha) or alpha <= 0:
            return -1.0, None
        return float(np.sum(R*alpha)), R*alpha

    best_sum = -1.0
    best_xc = None; best_yc = None; best_r = None
    # init fallback
    s_fb, r_fb = lp_eval(fb_arr[:, 0], fb_arr[:, 1])
    if s_fb > 0:
        best_sum = s_fb; best_xc = fb_arr[:, 0].copy(); best_yc = fb_arr[:, 1].copy(); best_r = r_fb.copy()
    # pool of top centres
    pool = []
    if best_xc is not None:
        pool.append((best_sum, best_xc.copy(), best_yc.copy()))

    if has_opt:
        # ---------- Phase A: penalized joint L-BFGS-B ----------
        bounds78 = [(0.0, 1.0)]*(2*n) + [(0.005, 0.20)]*n
        Ws = [50.0, 500.0, 5000.0, 50000.0]
        maxiters = [120, 120, 150, 200]

        def make_init(rid):
            if rid == 0:
                xs = []; ys = []
                for ix in range(5):
                    for jy in range(5):
                        xs.append((ix+0.5)/5.0); ys.append((jy+0.5)/5.0)
                xs.append(0.5); ys.append(0.5)
                xc = np.array(xs, float) + rng.normal(0, 0.02, n)
                yc = np.array(ys, float) + rng.normal(0, 0.02, n)
                xc = np.clip(xc, 0.02, 0.98); yc = np.clip(yc, 0.02, 0.98)
                R0 = 0.09 + 0.02*rng.rand(n)
                return xc, yc, R0
            elif rid == 1:
                # hex-ish lattice 6x5 subset
                pts = []
                sy = 0.175; sx = 0.185
                for ix in range(6):
                    for jy in range(5):
                        x = 0.09 + ix*sx + (0.5*sx if (jy % 2 == 1) else 0.0)
                        y = 0.09 + jy*sy
                        if x < 0.95 and y < 0.95:
                            pts.append((x, y))
                pts = np.array(pts)
                perm = rng.permutation(len(pts))[:n]
                # if not enough points pad random
                xc = np.zeros(n); yc = np.zeros(n)
                for k2 in range(n):
                    if k2 < len(perm):
                        xc[k2], yc[k2] = pts[perm[k2]]
                    else:
                        xc[k2] = rng.uniform(0.05, 0.95); yc[k2] = rng.uniform(0.05, 0.95)
                xc = xc + rng.normal(0, 0.01, n); yc = yc + rng.normal(0, 0.01, n)
                xc = np.clip(xc, 0.01, 0.99); yc = np.clip(yc, 0.01, 0.99)
                R0 = 0.08 + 0.04*rng.rand(n)
                return xc, yc, R0
            else:
                xc = rng.uniform(0.03, 0.97, n)
                yc = rng.uniform(0.03, 0.97, n)
                if rid % 3 == 0:
                    R0 = 0.07 + 0.04*rng.rand(n)
                else:
                    R0 = 0.05 + 0.07*rng.rand(n)
                return xc, yc, R0

        restart = 0
        while time.time() - t0 < T_PEN and restart < 80:
            xc0, yc00, R00 = make_init(restart)
            v = np.concatenate([xc0, yc00, R00])
            v = np.clip(v, 0.0, 1.0)
            # keep R in bounds
            v[2*n:] = np.clip(v[2*n:], 0.005, 0.20)
            aborted = False
            for wi, W in enumerate(Ws):
                if time.time() - t0 >= T_PEN:
                    aborted = True
                    break
                def fun(vv, W=W):
                    X = vv[0:n]; Y = vv[n:2*n]; R = vv[2*n:3*n]
                    # wall pen
                    pen = 0.0
                    gx = np.zeros(n); gy = np.zeros(n); gr_pen = np.zeros(n)
                    v1 = R - X
                    m1 = v1 > 0
                    if np.any(m1):
                        pen += np.sum(v1[m1]*v1[m1])
                        gx[m1] += -2.0*v1[m1]
                        gr_pen[m1] += 2.0*v1[m1]
                    v2 = X + R - 1.0
                    m2 = v2 > 0
                    if np.any(m2):
                        pen += np.sum(v2[m2]*v2[m2])
                        gx[m2] += 2.0*v2[m2]
                        gr_pen[m2] += 2.0*v2[m2]
                    v3 = R - Y
                    m3 = v3 > 0
                    if np.any(m3):
                        pen += np.sum(v3[m3]*v3[m3])
                        gy[m3] += -2.0*v3[m3]
                        gr_pen[m3] += 2.0*v3[m3]
                    v4 = Y + R - 1.0
                    m4 = v4 > 0
                    if np.any(m4):
                        pen += np.sum(v4[m4]*v4[m4])
                        gy[m4] += 2.0*v4[m4]
                        gr_pen[m4] += 2.0*v4[m4]
                    Dx = X[:, None] - X[None, :]
                    Dy = Y[:, None] - Y[None, :]
                    D = np.sqrt(Dx*Dx + Dy*Dy + 1e-16)
                    S = R[:, None] + R[None, :]
                    V = S - D
                    np.fill_diagonal(V, 0.0)
                    Vp = np.maximum(V, 0.0)
                    np.fill_diagonal(Vp, 0.0)
                    pen += 0.25*np.sum(Vp*Vp)
                    # grad pair
                    np.fill_diagonal(D, 1.0)
                    # Q = Vp / D
                    Q = Vp / D
                    gx += -np.sum(Q*Dx, axis=1)
                    gy += -np.sum(Q*Dy, axis=1)
                    gr_pen += np.sum(Vp, axis=1)
                    loss = -np.sum(R) + W*pen
                    g = np.empty_like(vv)
                    g[0:n] = W*gx
                    g[n:2*n] = W*gy
                    g[2*n:3*n] = -1.0 + W*gr_pen
                    return loss, g
                try:
                    res = minimize(fun, v, method='L-BFGS-B', jac=True, bounds=bounds78,
                                   options={'maxiter': maxiters[wi], 'ftol': 1e-9, 'gtol': 1e-5})
                    v = res.x
                except Exception:
                    aborted = True
                    break
            if not aborted:
                X = np.clip(v[0:n], 0.0, 1.0); Y = np.clip(v[n:2*n], 0.0, 1.0); R = np.clip(v[2*n:], 0.005, 0.20)
                # LP optimal for these centres
                s_lp, r_lp = lp_eval(np.clip(X, 0.0, 1.0), np.clip(Y, 0.0, 1.0))
                s_sc, _ = feasible_scaled_sum(X, Y, R)
                # consider LP
                if s_lp is not None and s_lp > best_sum:
                    best_sum = s_lp; best_xc = X.copy(); best_yc = Y.copy(); best_r = r_lp.copy()
                    pool.append((best_sum, X.copy(), Y.copy()))
                    pool = sorted(pool, key=lambda t: -t[0])[:5]
                if s_sc is not None and s_sc > best_sum:
                    # need LP radii for scaled? use scaled R as fallback centres keep LP best
                    # evaluate LP again? already have; if scaled better than LP (should not) keep scaled
                    # but LP should dominate; still keep centres
                    pass
                # also keep near-best for diversity
                if s_lp is not None and s_lp > 0 and len(pool) < 5:
                    pool.append((s_lp, X.copy(), Y.copy()))
            restart += 1

        # ---------- Phase B: LP hill climbing on centres ----------
        if best_xc is not None:
            # start from best pool[0]
            pool = sorted(pool, key=lambda t: -t[0])[:5]
            cur_xc, cur_yc = pool[0][1].copy(), pool[0][2].copy()
            cur_sum, cur_r = lp_eval(cur_xc, cur_yc)
            if cur_sum < 0:
                cur_xc, cur_yc = best_xc.copy(), best_yc.copy()
                cur_sum, cur_r = best_sum, best_r.copy()
            if cur_sum > best_sum:
                best_sum = cur_sum; best_xc = cur_xc.copy(); best_yc = cur_yc.copy(); best_r = cur_r.copy()
            step = 0
            noimp = 0
            # precompute for tight detection? just gaussian + global
            while time.time() - t0 < T_HILL:
                # sigma schedule
                base = 0.035 * np.exp(-step / 3500.0) + 0.0008
                if noimp > 800:
                    base = 0.012
                if noimp > 2000:
                    # large kick then reset to best
                    if time.time() - t0 > T_HILL - 5:
                        break
                    # restart from random pool member with perturbation
                    pi = rng.randint(len(pool))
                    cur_xc = pool[pi][1] + rng.normal(0, 0.01, n)
                    cur_yc = pool[pi][2] + rng.normal(0, 0.01, n)
                    cur_xc = np.clip(cur_xc, 0.002, 0.998)
                    cur_yc = np.clip(cur_yc, 0.002, 0.998)
                    cur_sum, cur_r = lp_eval(cur_xc, cur_yc)
                    if cur_sum < 0:
                        cur_xc, cur_yc = best_xc.copy(), best_yc.copy()
                        cur_sum, cur_r = best_sum, best_r.copy()
                    noimp = 0
                    step += 1
                    continue
                u = rng.rand()
                txc = cur_xc.copy(); tyc = cur_yc.copy()
                if u < 0.82:
                    i = rng.randint(n)
                    sigma = base
                    txc[i] = np.clip(cur_xc[i] + rng.normal(0, sigma), 0.002, 0.998)
                    tyc[i] = np.clip(cur_yc[i] + rng.normal(0, sigma), 0.002, 0.998)
                elif u < 0.92:
                    # two-circle move
                    i = rng.randint(n); j = rng.randint(n)
                    sigma = base
                    txc[i] = np.clip(cur_xc[i] + rng.normal(0, sigma), 0.002, 0.998)
                    tyc[i] = np.clip(cur_yc[i] + rng.normal(0, sigma), 0.002, 0.998)
                    txc[j] = np.clip(cur_xc[j] + rng.normal(0, sigma), 0.002, 0.998)
                    tyc[j] = np.clip(cur_yc[j] + rng.normal(0, sigma), 0.002, 0.998)
                else:
                    sigma = base * 0.4
                    txc = np.clip(cur_xc + rng.normal(0, sigma, n), 0.002, 0.998)
                    tyc = np.clip(cur_yc + rng.normal(0, sigma, n), 0.002, 0.998)
                s, r = lp_eval(txc, tyc)
                step += 1
                if s > cur_sum + 1e-12:
                    cur_xc, cur_yc, cur_sum, cur_r = txc, tyc, s, r
                    noimp = 0
                    if s > best_sum:
                        best_sum = s; best_xc = txc.copy(); best_yc = tyc.copy(); best_r = r.copy()
                        # update pool
                        pool.append((s, txc.copy(), tyc.copy()))
                        pool = sorted(pool, key=lambda t: -t[0])[:5]
                else:
                    noimp += 1

    if best_xc is None:
        return fb_arr

    xc, yc, r = best_xc.copy(), best_yc.copy(), np.maximum(best_r.copy(), 1e-9)
    # exact feasibility scaling
    b = np.minimum(np.minimum(xc, 1.0 - xc), np.minimum(yc, 1.0 - yc))
    dx = xc[I] - xc[J]; dy = yc[I] - yc[J]
    d = np.sqrt(dx*dx + dy*dy)
    Rsum = r[I] + r[J]
    mask = Rsum > 1e-14
    s_pair = 1.0
    if np.any(mask):
        ratios = d[mask] / (Rsum[mask] + 1e-30)
        s_pair = float(np.min(ratios))
    s_box = float(np.min(b / (r + 1e-30)))
    s = min(s_pair, s_box, 1.0)
    scale = s * (1.0 - 1e-7)
    if scale > 1.0:
        scale = 1.0 - 1e-7
    if scale <= 0 or not np.isfinite(scale):
        scale = 0.99
    # tiny extra shrink to be strictly inside for 1e-12
    scale = scale * (1.0 - 1e-9)
    r = r * scale
    r = np.maximum(r, 1e-9)
    arr = np.column_stack((xc, yc, r))
    return arr
