import numpy as np
import time

def solve():
    n = 26
    t0 = time.time()
    DEADLINE = 122.0
    T_PEN = 38.0  # phase A budget
    rng = np.random.default_rng(0)

    # pair matrix for LP
    I, J = np.triu_indices(n, 1)
    m = len(I)
    A0 = np.zeros((m, n), dtype=float)
    A0[np.arange(m), I] = 1.0
    A0[np.arange(m), J] = 1.0
    cobj = -np.ones(n, dtype=float)
    try:
        from scipy.optimize import minimize, linprog
        has_opt = True
    except Exception:
        has_opt = False

    def lp_eval(xc, yc):
        # returns (sum, r) ; always feasible if xc,yc in [0,1]
        xc = np.asarray(xc, dtype=float)
        yc = np.asarray(yc, dtype=float)
        wall = np.minimum(np.minimum(xc, 1.0 - xc), np.minimum(yc, 1.0 - yc))
        if np.any(wall < 0):
            return -1.0, None
        wall = np.maximum(wall, 0.0)
        dx = xc[I] - xc[J]
        dy = yc[I] - yc[J]
        d = np.sqrt(dx * dx + dy * dy)
        bnds = [(0.0, float(w)) for w in wall]
        try:
            res = linprog(cobj, A_ub=A0, b_ub=d, bounds=bnds, method='highs')
        except Exception:
            return -1.0, None
        if res.status != 0:
            return -1.0, None
        r = np.asarray(res.x, dtype=float)
        r = np.maximum(r, 0.0)
        s = float(np.sum(r))
        return s, r

    def uniform_scale(X, Y, R):
        R = np.maximum(np.asarray(R, float), 1e-12)
        X = np.asarray(X, float); Y = np.asarray(Y, float)
        wall = np.minimum(np.minimum(X, 1.0 - X), np.minimum(Y, 1.0 - Y))
        Dx = X[:, None] - X[None, :]
        Dy = Y[:, None] - Y[None, :]
        D = np.sqrt(Dx * Dx + Dy * Dy)
        S = R[:, None] + R[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            rw = wall / R
            rp = D / S
        np.fill_diagonal(rp, np.inf)
        a = float(np.min(rw)); b = float(np.min(rp))
        alpha = min(1.0, a, b)
        alpha = alpha * (1.0 - 1e-7) - 1e-12
        if not np.finite(alpha) or alpha <= 0:
            alpha = 0.5
        return R * alpha, alpha

    # fallback
    k = int(np.ceil(np.sqrt(n)))
    r0 = 0.5 / k
    fb = []
    for ix in range(k):
        for jy in range(k):
            if len(fb) < n:
                fb.append([(2 * ix + 1) * r0, (2 * jy + 1) * r0])
    fb = np.array(fb, dtype=float)
    best_sum = -1.0
    best_xc = fb[:, 0].copy(); best_yc = fb[:, 1].copy(); best_r = None
    s_fb, r_fb = lp_eval(best_xc, best_yc)
    if s_fb > 0:
        best_sum = s_fb; best_r = r_fb

    # ---------- Phase A: short penalized L-BFGS ----------
    if has_opt:
        def make_fun(nn):
            def fun(v, W):
                X = v[0:nn]; Y = v[nn:2*nn]; R = v[2*nn:3*nn]
                gx = np.zeros(nn); gy = np.zeros(nn); pen_gr = np.zeros(nn)
                pen = 0.0
                v1 = R - X
                mm = v1 > 0
                if np.any(mm):
                    pen += np.sum(v1[mm]**2)
                    gx[mm] += -2.0*v1[mm]; pen_gr[mm] += 2.0*v1[mm]
                v2 = X + R - 1.0
                mm = v2 > 0
                if np.any(mm):
                    pen += np.sum(v2[mm]**2)
                    gx[mm] += 2.0*v2[mm]; pen_gr[mm] += 2.0*v2[mm]
                v3 = R - Y
                mm = v3 > 0
                if np.any(mm):
                    pen += np.sum(v3[mm]**2)
                    gy[mm] += -2.0*v3[mm]; pen_gr[mm] += 2.0*v3[mm]
                v4 = Y + R - 1.0
                mm = v4 > 0
                if np.any(mm):
                    pen += np.sum(v4[mm]**2)
                    gy[mm] += 2.0*v4[mm]; pen_gr[mm] += 2.0*v4[mm]
                Dx = X[:, None] - X[None, :]
                Dy = Y[:, None] - Y[None, :]
                D = np.sqrt(Dx*Dx + Dy*Dy + 1e-20)
                S = R[:, None] + R[None, :]
                V = S - D
                np.fill_diagonal(V, 0.0)
                np.fill_diagonal(D, 1.0)
                Mask = V > 0
                if np.any(Mask):
                    pen += 0.5*np.sum(V[Mask]**2)
                    Q = np.zeros((nn, nn))
                    Q[Mask] = -2.0*V[Mask]/D[Mask]
                    gx += np.sum(Q*Dx, axis=1)
                    gy += np.sum(Q*Dy, axis=1)
                    G = np.zeros((nn, nn))
                    G[Mask] = 2.0*V[Mask]
                    pen_gr += np.sum(G, axis=1)
                loss = -np.sum(R) + W*pen
                g = np.empty_like(v)
                g[0:nn] = W*gx
                g[nn:2*nn] = W*gy
                g[2*nn:3*nn] = -1.0 + W*pen_gr
                return loss, g
            return fun
        fun = make_fun(n)
        bounds78 = [(0.0, 1.0)]*(2*n) + [(0.02, 0.20)]*n
        Ws = [1e2, 1e3, 1e4, 1e5]
        restart = 0
        while (time.time() - t0 < T_PEN) and restart < 40:
            if restart == 0:
                X0 = np.zeros(n); Y0 = np.zeros(n)
                idx = 0
                for ii in range(5):
                    for jj in range(5):
                        X0[idx] = (ii+0.5)/5.0 + rng.normal(0, 0.006)
                        Y0[idx] = (jj+0.5)/5.0 + rng.normal(0, 0.006)
                        idx += 1
                X0[25] = 0.9 + rng.normal(0, 0.01); Y0[25] = 0.1 + rng.normal(0, 0.01)
                X0 = np.clip(X0, 0.05, 0.95); Y0 = np.clip(Y0, 0.05, 0.95)
                R0 = np.full(n, 0.095) + rng.normal(0, 0.005, n)
            elif restart == 1:
                pts = []
                for ix in range(6):
                    for jy in range(5):
                        xx = (ix+0.5)/6.0; yy = (jy+0.5)/5.0
                        if jy % 2 == 1:
                            xx += 0.04
                            if xx > 0.96: xx -= 1.0/6.0
                        pts.append((xx, yy))
                pts = np.array(pts)
                perm = rng.permutation(len(pts))[:n]
                X0 = pts[perm, 0] + rng.normal(0, 0.006, n)
                Y0 = pts[perm, 1] + rng.normal(0, 0.006, n)
                X0 = np.clip(X0, 0.03, 0.97); Y0 = np.clip(Y0, 0.03, 0.97)
                R0 = 0.085 + 0.02*rng.random(n)
            elif restart % 3 == 0:
                # unequal radii init: few large + many medium
                X0 = rng.random(n)*0.9+0.05; Y0 = rng.random(n)*0.9+0.05
                R0 = 0.07 + 0.05*rng.random(n)
                # make 4 of them small filler-like
                idxs = rng.choice(n, size=5, replace=False)
                R0[idxs] = 0.03 + 0.02*rng.random(5)
            else:
                X0 = rng.random(n)*0.9+0.05; Y0 = rng.random(n)*0.9+0.05
                R0 = 0.08 + 0.03*rng.random(n)
            R0 = np.clip(R0, 0.02, 0.20)
            v = np.concatenate([X0, Y0, R0])
            for W in Ws:
                if time.time() - t0 > T_PEN:
                    break
                try:
                    res = minimize(fun, v, args=(W,), jac=True, method='L-BFGS-B',
                                   bounds=bounds78, options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-6})
                    v = res.x
                except Exception:
                    break
            X = np.clip(v[0:n], 0.0, 1.0); Y = np.clip(v[n:2*n], 0.0, 1.0)
            s_lp, r_lp = lp_eval(X, Y)
            if s_lp > best_sum:
                best_sum = s_lp; best_xc = X.copy(); best_yc = Y.copy(); best_r = r_lp.copy()
            restart += 1

    # ---------- Phase B: annealing centre search on LP objective ----------
    if best_r is None:
        s0, r0v = lp_eval(best_xc, best_yc)
        best_r = r0v; best_sum = s0
    cur_x = best_xc.copy(); cur_y = best_yc.copy(); cur_r = best_r.copy(); cur_s = best_sum

    # temperature schedule
    T0 = 0.0025
    T1 = 0.00002
    sigmas = np.array([0.03, 0.015, 0.008, 0.0035, 0.0015], dtype=float)
    # order small circles for biased moves
    it = 0
    # pre-allocate candidate buffer for void search
    while time.time() - t0 < DEADLINE - 6:
        elapsed = time.time() - t0
        total = DEADLINE - 6 - T_PEN
        if total < 1: total = 1
        frac = max(0.0, min(1.0, (elapsed - T_PEN) / total))
        Temp = T0 * (T1 / T0) ** frac
        # choose move
        u = rng.random()
        xn = cur_x.copy(); yn = cur_y.copy()
        if u < 0.62:
            # single Gaussian perturb, bias to small circles
            if rng.random() < 0.5:
                # pick among smallest 8 with higher prob
                order = np.argsort(cur_r)
                if rng.random() < 0.7:
                    ii = rng.choice(order[:8])
                else:
                    ii = rng.integers(0, n)
            else:
                ii = rng.integers(0, n)
            sig = rng.choice(sigmas)
            # larger sigma early, smaller late
            if frac > 0.75 and sig > 0.008:
                sig = 0.008
            xn[ii] += rng.normal(0, sig)
            yn[ii] += rng.normal(0, sig)
            xn[ii] = min(max(xn[ii], 0.0), 1.0)
            yn[ii] = min(max(yn[ii], 0.0), 1.0)
        elif u < 0.74:
            # uniform teleport of a (prefer small) circle
            order = np.argsort(cur_r)
            if rng.random() < 0.7:
                ii = rng.choice(order[:10])
            else:
                ii = rng.integers(0, n)
            xn[ii] = rng.random()
            yn[ii] = rng.random()
        elif u < 0.90:
            # void jump: place ii at largest empty spot
            order = np.argsort(cur_r)
            ii = rng.choice(order[:10]) if rng.random() < 0.75 else rng.integers(0, n)
            # sample candidates
            ncand = 120
            cx = rng.random(ncand); cy = rng.random(ncand)
            # score = min(wall, min_dist/2)
            best_sc = -1; best_cx = cx[0]; best_cy = cy[0]
            ox = np.delete(cur_x, ii); oy = np.delete(cur_y, ii)
            for kk in range(ncand):
                ddx = ox - cx[kk]; ddy = oy - cy[kk]
                dm = float(np.min(np.sqrt(ddx*ddx + ddy*ddy)))
                wl = float(min(cx[kk], 1-cx[kk], cy[kk], 1-cy[kk]))
                sc = wl if wl < dm*0.5 else dm*0.5
                if sc > best_sc:
                    best_sc = sc; best_cx = cx[kk]; best_cy = cy[kk]
            xn[ii] = best_cx + rng.normal(0, 0.003)
            yn[ii] = best_cy + rng.normal(0, 0.003)
            xn[ii] = min(max(xn[ii], 0.0), 1.0)
            yn[ii] = min(max(yn[ii], 0.0), 1.0)
        else:
            # global shake
            sig = 0.004 * (1-frac) + 0.0008
            xn = xn + rng.normal(0, sig, n)
            yn = yn + rng.normal(0, sig, n)
            xn = np.clip(xn, 0.0, 1.0); yn = np.clip(yn, 0.0, 1.0)
        s_new, r_new = lp_eval(xn, yn)
        if s_new < 0:
            it += 1
            continue
        if s_new > cur_s:
            cur_x = xn; cur_y = yn; cur_r = r_new; cur_s = s_new
            if s_new > best_sum:
                best_sum = s_new; best_xc = xn.copy(); best_yc = yn.copy(); best_r = r_new.copy()
        else:
            if Temp > 1e-12 and rng.random() < np.exp((s_new - cur_s) / Temp):
                cur_x = xn; cur_y = yn; cur_r = r_new; cur_s = s_new
            # occasional restart from best if stuck
            if it % 700 == 699 and rng.random() < 0.25:
                cur_x = best_xc.copy(); cur_y = best_yc.copy(); cur_r = best_r.copy(); cur_s = best_sum
        it += 1
        # greedy intensification at very end
        if frac > 0.92:
            # only accept improvements (Temp ~ tiny already)
            pass

    # ---------- final greedy polish (coordinate descent) ----------
    t_end = DEADLINE - 1.5
    sigs = [0.004, 0.002, 0.001, 0.0005]
    for sig in sigs:
        if time.time() - t0 > t_end:
            break
        for _ in range(250):
            if time.time() - t0 > t_end:
                break
            ii = rng.integers(0, n)
            xn = best_xc.copy(); yn = best_yc.copy()
            xn[ii] += rng.normal(0, sig); yn[ii] += rng.normal(0, sig)
            xn[ii] = min(max(xn[ii], 0.0), 1.0); yn[ii] = min(max(yn[ii], 0.0), 1.0)
            s_new, r_new = lp_eval(xn, yn)
            if s_new > best_sum + 1e-12:
                best_sum = s_new; best_xc = xn; best_yc = yn; best_r = r_new

    # final safety shrink
    try:
        Rs, _ = uniform_scale(best_xc, best_yc, best_r)
        out = np.column_stack([best_xc, best_yc, Rs])
    except Exception:
        Rs = np.asarray(best_r, float) * (1.0 - 1e-7)
        out = np.column_stack([best_xc, best_yc, Rs])
    return out
