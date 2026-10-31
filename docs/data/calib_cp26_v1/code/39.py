import numpy as np
import time

def solve():
    n = 26
    t0 = time.time()
    HARD = 127.0
    rng = np.random.RandomState(0)
    try:
        from scipy.optimize import minimize, linprog
        has_opt = True
    except Exception:
        has_opt = False

    I, J = np.triu_indices(n, 1)
    m_pair = len(I)
    # full matrix for primal with walls: (m_pair+n) x n
    A_full = np.zeros((m_pair + n, n), dtype=float)
    for k in range(m_pair):
        A_full[k, I[k]] = 1.0
        A_full[k, J[k]] = 1.0
    for i in range(n):
        A_full[m_pair + i, i] = 1.0

    # ---------- helpers ----------
    def lp_primal(xc, yc):
        b = np.minimum(np.minimum(xc, 1.0 - xc), np.minimum(yc, 1.0 - yc))
        if np.any(b < 0):
            return -1.0, None
        b = np.maximum(b, 0.0)
        dx = xc[I] - xc[J]
        dy = yc[I] - yc[J]
        d = np.sqrt(dx * dx + dy * dy)
        b_full = np.concatenate([d, b])
        c = -np.ones(n)
        bounds = [(0.0, None)] * n
        try:
            res = linprog(c, A_ub=A_full, b_ub=b_full, bounds=bounds, method='highs')
        except Exception:
            return -1.0, None
        if res.status != 0:
            return -1.0, None
        r = np.asarray(res.x, dtype=float)
        r = np.maximum(r, 0.0)
        return float(np.sum(r)), r

    def lp_primal_dual(xc, yc):
        b = np.minimum(np.minimum(xc, 1.0 - xc), np.minimum(yc, 1.0 - yc))
        if np.any(b <= 1e-12):
            return -1.0, None, None, None
        b = np.maximum(b, 0.0)
        dx = xc[I] - xc[J]
        dy = yc[I] - yc[J]
        d = np.sqrt(dx * dx + dy * dy)
        # avoid degenerate zero distance
        if np.any(d < 1e-12):
            return -1.0, None, None, None
        b_full = np.concatenate([d, b])
        c = -np.ones(n)
        bounds = [(0.0, None)] * n
        try:
            res = linprog(c, A_ub=A_full, b_ub=b_full, bounds=bounds, method='highs')
        except Exception:
            return -1.0, None, None, None
        if res.status != 0:
            return -1.0, None, None, None
        r = np.maximum(np.asarray(res.x, float), 0.0)
        s = float(np.sum(r))
        # dual: min b_full^T y s.t. -A_full^T y <= -1, y>=0
        M = m_pair + n
        c_d = b_full.copy()
        A_d = (-A_full.T).copy()  # n x M
        b_d = -np.ones(n)
        try:
            rd = linprog(c_d, A_ub=A_d, b_ub=b_d, bounds=[(0.0, None)] * M, method='highs')
        except Exception:
            return s, r, None, None
        if rd.status != 0:
            return s, r, None, None
        y = np.maximum(np.asarray(rd.x, float), 0.0)
        # zero tiny
        y[y < 1e-11] = 0.0
        lam = y[:m_pair]
        mu = y[m_pair:]
        # sanity: dual objective should match primal
        # if badly mismatched, discard dual
        try:
            dob = float(np.dot(b_full, y))
            if abs(dob - s) > 1e-5 + 1e-4 * max(1.0, s):
                # still keep but threshold
                pass
        except Exception:
            pass
        return s, r, lam, mu

    def feasible_shrink(X, Y, R):
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
        a = float(np.min(rw)); bb = float(np.min(rp))
        alpha = min(1.0, a, bb)
        alpha = alpha * (1.0 - 1e-9) - 1e-12
        if not np.finite(alpha) or alpha <= 0:
            return None, -1
        return R * alpha, alpha

    # fallback grid
    k = int(np.ceil(np.sqrt(n)))
    rr0 = 0.5 / k
    fb = []
    for ix in range(k):
        for jy in range(k):
            if len(fb) < n:
                fb.append([(2 * ix + 1) * rr0, (2 * jy + 1) * rr0, rr0 * 0.999])
    fb_arr = np.array(fb, dtype=float)
    best_sum = -1.0
    best_xc = None; best_yc = None; best_r = None
    s_fb, r_fb = lp_primal(fb_arr[:, 0], fb_arr[:, 1])
    if s_fb > 0:
        Rs, _ = feasible_shrink(fb_arr[:, 0], fb_arr[:, 1], r_fb)
        if Rs is not None:
            r_fb = Rs
            s_fb = float(np.sum(Rs))
        best_sum = s_fb; best_xc = fb_arr[:, 0].copy(); best_yc = fb_arr[:, 1].copy(); best_r = r_fb.copy()
    pool = []
    if best_xc is not None:
        pool.append((best_sum, best_xc.copy(), best_yc.copy()))

    # ---------- Phase A: penalized joint search (fast) ----------
    if has_opt:
        T_PEN = t0 + 52.0
        bounds78 = [(0.0, 1.0)] * (2 * n) + [(0.005, 0.20)] * n
        Ws = [80.0, 800.0, 8000.0]
        maxiters = [80, 80, 110]

        def make_init(rid):
            if rid == 0:
                xs = []; ys = []
                for ix in range(5):
                    for jy in range(5):
                        xs.append((ix + 0.5) / 5.0); ys.append((jy + 0.5) / 5.0)
                xs.append(0.5); ys.append(0.5)
                xc = np.array(xs, float) + rng.normal(0, 0.02, n)
                yc = np.array(ys, float) + rng.normal(0, 0.02, n)
                xc = np.clip(xc, 0.02, 0.98); yc = np.clip(yc, 0.02, 0.98)
                R0 = 0.09 + 0.02 * rng.rand(n)
                return xc, yc, R0
            elif rid == 1:
                xc = rng.uniform(0.05, 0.95, n); yc = rng.uniform(0.05, 0.95, n)
                R0 = 0.08 + 0.03 * rng.rand(n)
                return xc, yc, R0
            else:
                xc = rng.uniform(0.03, 0.97, n)
                yc = rng.uniform(0.03, 0.97, n)
                R0 = 0.05 + 0.06 * rng.rand(n)
                return xc, yc, R0

        restart = 0
        while time.time() < T_PEN and restart < 60:
            xc0, yc0, R00 = make_init(restart)
            v = np.concatenate([xc0, yc0, R00])
            v[2 * n:] = np.clip(v[2 * n:], 0.005, 0.20)
            aborted = False
            for wi, W in enumerate(Ws):
                if time.time() >= T_PEN:
                    aborted = True; break
                def fun(vv, W=W):
                    X = vv[0:n]; Y = vv[n:2 * n]; R = vv[2 * n:3 * n]
                    pen = 0.0
                    gx = np.zeros(n); gy = np.zeros(n); grp = np.zeros(n)
                    v1 = R - X
                    m1 = v1 > 0
                    if np.any(m1):
                        pen += np.sum(v1[m1] * v1[m1])
                        gx[m1] += -2.0 * v1[m1]; grp[m1] += 2.0 * v1[m1]
                    v2 = X + R - 1.0
                    m2 = v2 > 0
                    if np.any(m2):
                        pen += np.sum(v2[m2] * v2[m2])
                        gx[m2] += 2.0 * v2[m2]; grp[m2] += 2.0 * v2[m2]
                    v3 = R - Y
                    m3 = v3 > 0
                    if np.any(m3):
                        pen += np.sum(v3[m3] * v3[m3])
                        gy[m3] += -2.0 * v3[m3]; grp[m3] += 2.0 * v3[m3]
                    v4 = Y + R - 1.0
                    m4 = v4 > 0
                    if np.any(m4):
                        pen += np.sum(v4[m4] * v4[m4])
                        gy[m4] += 2.0 * v4[m4]; grp[m4] += 2.0 * v4[m4]
                    Dx = X[:, None] - X[None, :]
                    Dy = Y[:, None] - Y[None, :]
                    D = np.sqrt(Dx * Dx + Dy * Dy + 1e-16)
                    S = R[:, None] + R[None, :]
                    V = S - D
                    np.fill_diagonal(V, 0.0)
                    Vp = np.maximum(V, 0.0)
                    np.fill_diagonal(Vp, 0.0)
                    pen += 0.25 * np.sum(Vp * Vp)
                    np.fill_diagonal(D, 1.0)
                    Q = Vp / D
                    gx += -np.sum(Q * Dx, axis=1)
                    gy += -np.sum(Q * Dy, axis=1)
                    grp += np.sum(Vp, axis=1)
                    loss = -np.sum(R) + W * pen
                    g = np.empty_like(vv)
                    g[0:n] = W * gx; g[n:2 * n] = W * gy
                    g[2 * n:3 * n] = -1.0 + W * grp
                    return loss, g
                try:
                    res = minimize(fun, v, method='L-BFGS-B', jac=True, bounds=bounds78,
                                   options={'maxiter': maxiters[wi], 'ftol': 1e-9, 'gtol': 1e-5})
                    v = res.x
                except Exception:
                    aborted = True; break
            if not aborted:
                X = np.clip(v[0:n], 0.0, 1.0); Y = np.clip(v[n:2 * n], 0.0, 1.0)
                s_lp, r_lp = lp_primal(np.clip(X, 0.0, 1.0), np.clip(Y, 0.0, 1.0))
                if s_lp is not None and s_lp > 0:
                    if s_lp > best_sum:
                        best_sum = s_lp; best_xc = X.copy(); best_yc = Y.copy(); best_r = r_lp.copy()
                        pool.append((best_sum, X.copy(), Y.copy()))
                        pool = sorted(pool, key=lambda t: -t[0])[:6]
                    elif len(pool) < 6 and s_lp > best_sum - 0.05:
                        pool.append((s_lp, X.copy(), Y.copy()))
            restart += 1

    if best_xc is None:
        best_xc = fb_arr[:, 0].copy(); best_yc = fb_arr[:, 1].copy(); best_r = r_fb.copy()
        best_sum = s_fb

    # ---------- build symmetric + perturbed candidates ----------
    def transforms(xc, yc):
        t = []
        t.append((xc, yc))
        t.append((1 - xc, yc))
        t.append((xc, 1 - yc))
        t.append((1 - xc, 1 - yc))
        t.append((yc, xc))
        t.append((1 - yc, xc))
        t.append((yc, 1 - xc))
        t.append((1 - yc, 1 - xc))
        return t

    cands = []
    try:
        base_pool = sorted(pool, key=lambda t: -t[0])[:2]
    except Exception:
        base_pool = [(best_sum, best_xc, best_yc)]
    for _, bx, by in base_pool:
        for (tx, ty) in transforms(bx, by):
            cands.append((tx.copy(), ty.copy()))
    # perturb
    rng2 = np.random.RandomState(1)
    pcands = []
    for (tx, ty) in cands:
        for sig in (0.0015, 0.004):
            nx = np.clip(tx + rng2.normal(0, sig, n), 0.005, 0.995)
            ny = np.clip(ty + rng2.normal(0, sig, n), 0.005, 0.995)
            pcands.append((nx, ny))
    # evaluate quickly, keep top 4
    allc = cands[:4] + pcands
    scored = []
    for (tx, ty) in allc:
        if time.time() - t0 > 60.0:
            break
        s, _ = lp_primal(tx, ty)
        if s > 0:
            scored.append((s, tx, ty))
    scored = sorted(scored, key=lambda t: -t[0])[:4]
    if len(scored) == 0:
        scored = [(best_sum, best_xc, best_yc)]

    # ---------- Phase B: exact active-set dual-gradient polishing ----------
    # surrogate with fixed lambda,mu: maximize lam^T d + mu^T b
    best_global = (best_sum, best_xc.copy(), best_yc.copy(), best_r.copy())
    T_ACTIVE_END = t0 + 100.0
    for si, (s0, cx0, cy0) in enumerate(scored):
        if time.time() > T_ACTIVE_END:
            break
        xc = cx0.copy(); yc = cy0.copy()
        # outer refresh loop
        for outer in range(14):
            if time.time() > T_ACTIVE_END:
                break
            s, r, lam, mu = lp_primal_dual(xc, yc)
            if s > 0 and s > best_global[0]:
                best_global = (s, xc.copy(), yc.copy(), r.copy())
            if lam is None or mu is None:
                # fallback: small random kick + re-eval
                xc = np.clip(xc + rng2.normal(0, 0.002, n), 0.005, 0.995)
                yc = np.clip(yc + rng2.normal(0, 0.002, n), 0.005, 0.995)
                continue
            # active pairs
            act = np.where(lam > 1e-12)[0]
            if len(act) == 0:
                # no pair active: push apart? kick
                xc = np.clip(xc + rng2.normal(0, 0.003, n), 0.005, 0.995)
                yc = np.clip(yc + rng2.normal(0, 0.003, n), 0.005, 0.995)
                continue
            lam_act = lam[act]
            Ia = I[act]; Ja = J[act]
            mu_v = mu.copy()

            # define surrogate objective for L-BFGS-B (minimize -S)
            def surr(vv):
                X = vv[0:n]; Y = vv[n:2 * n]
                dx = X[Ia] - X[Ja]
                dy = Y[Ia] - Y[Ja]
                d = np.sqrt(dx * dx + dy * dy + 1e-24)
                b = np.minimum(np.minimum(X, 1.0 - X), np.minimum(Y, 1.0 - Y))
                S = float(np.sum(lam_act * d) + np.dot(mu_v, b))
                # gradient of S
                inv = lam_act / d
                gx = np.zeros(n); gy = np.zeros(n)
                # accumulate
                # for each active pair k: contrib to Ia[k] += inv*dx, to Ja[k] -= inv*dx
                np.add.at(gx, Ia, inv * dx)
                np.add.at(gx, Ja, -inv * dx)
                np.add.at(gy, Ia, inv * dy)
                np.add.at(gy, Ja, -inv * dy)
                # wall derivatives (subgradient)
                # determine closest wall per circle
                # walls: 0:X,1:1-X,2:Y,3:1-Y
                w0 = X; w1 = 1.0 - X; w2 = Y; w3 = 1.0 - Y
                # stack
                # for xc derivative: +1 if w0 minimal, -1 if w1 minimal, else 0; split ties
                mins = np.minimum(np.minimum(w0, w1), np.minimum(w2, w3))
                tol = 1e-12
                m0 = (w0 <= mins + tol); m1 = (w1 <= mins + tol)
                m2 = (w2 <= mins + tol); m3 = (w3 <= mins + tol)
                cnt = m0.astype(float) + m1.astype(float) + m2.astype(float) + m3.astype(float)
                cnt = np.maximum(cnt, 1.0)
                dbdx = ((m0.astype(float) - m1.astype(float)) / cnt) * mu_v
                dbdy = ((m2.astype(float) - m3.astype(float)) / cnt) * mu_v
                gx += dbdx
                gy += dbdy
                loss = -S
                g = np.empty_like(vv)
                g[0:n] = -gx
                g[n:2 * n] = -gy
                return loss, g

            v0 = np.concatenate([xc, yc])
            bnds = [(0.005, 0.995)] * (2 * n)
            try:
                ro = minimize(surr, v0, method='L-BFGS-B', jac=True, bounds=bnds,
                              options={'maxiter': 60, 'ftol': 1e-11, 'gtol': 1e-6})
                v1 = ro.x
            except Exception:
                break
            xn = v1[0:n]; yn = v1[n:2 * n]
            # check movement
            mv = float(np.max(np.abs(xn - xc)) + np.max(np.abs(yn - yc)))
            xc, yc = xn, yn
            if mv < 1e-7:
                # tiny move: add small kick to escape flat
                if outer > 4:
                    xc = np.clip(xc + rng2.normal(0, 0.0008, n), 0.005, 0.995)
                    yc = np.clip(yc + rng2.normal(0, 0.0008, n), 0.005, 0.995)
        # end outer
        # final lp eval
        s, r = lp_primal(xc, yc)
        if s > 0 and s > best_global[0]:
            best_global = (s, xc.copy(), yc.copy(), r.copy())

    best_sum, best_xc, best_yc, best_r = best_global[0], best_global[1].copy(), best_global[2].copy(), best_global[3].copy()

    # ---------- Phase C: SLSQP over all 78 vars with analytic Jacobians ----------
    if has_opt and (time.time() - t0) < HARD - 12.0:
        try:
            X0 = np.clip(best_xc, 0.01, 0.99); Y0 = np.clip(best_yc, 0.01, 0.99); R0 = np.maximum(best_r, 1e-4)
            # slight shrink to make strictly feasible start
            Rs0, _ = feasible_shrink(X0, Y0, R0)
            if Rs0 is not None:
                R0 = Rs0 * 0.999 + 1e-12
            v0 = np.concatenate([X0, Y0, R0])

            def obj_fun(vv):
                return -float(np.sum(vv[2 * n:])), -np.concatenate([np.zeros(2 * n), np.ones(n)])

            # wall constraints vector
            def wall_fun(vv):
                X = vv[0:n]; Y = vv[n:2 * n]; R = vv[2 * n:3 * n]
                return np.concatenate([X - R, 1.0 - X - R, Y - R, 1.0 - Y - R])

            def wall_jac(vv):
                Jm = np.zeros((4 * n, 3 * n))
                for i in range(n):
                    Jm[i, i] = 1.0; Jm[i, 2 * n + i] = -1.0
                    Jm[n + i, i] = -1.0; Jm[n + i, 2 * n + i] = -1.0
                    Jm[2 * n + i, n + i] = 1.0; Jm[2 * n + i, 2 * n + i] = -1.0
                    Jm[3 * n + i, n + i] = -1.0; Jm[3 * n + i, 2 * n + i] = -1.0
                return Jm

            def pair_fun(vv):
                X = vv[0:n]; Y = vv[n:2 * n]; R = vv[2 * n:3 * n]
                dx = X[I] - X[J]; dy = Y[I] - Y[J]
                d = np.sqrt(dx * dx + dy * dy)
                return d - R[I] - R[J]

            def pair_jac(vv):
                X = vv[0:n]; Y = vv[n:2 * n]
                dx = X[I] - X[J]; dy = Y[I] - Y[J]
                d = np.sqrt(dx * dx + dy * dy + 1e-24)
                Jm = np.zeros((m_pair, 3 * n))
                inv = 1.0 / d
                # X parts
                # use loop-vectorized via indexing
                rows = np.arange(m_pair)
                Jm[rows, I] += dx * inv
                Jm[rows, J] -= dx * inv
                Jm[rows, n + I] += dy * inv
                Jm[rows, n + J] -= dy * inv
                Jm[rows, 2 * n + I] -= 1.0
                Jm[rows, 2 * n + J] -= 1.0
                return Jm

            cons = [
                {'type': 'ineq', 'fun': wall_fun, 'jac': wall_jac},
                {'type': 'ineq', 'fun': pair_fun, 'jac': pair_jac},
            ]
            bnds78 = [(0.0, 1.0)] * (2 * n) + [(1e-5, 0.30)] * n
            # time-boxed SLSQP: first loose, then tight
            remaining = HARD - (time.time() - t0)
            it1 = 300 if remaining > 25 else 120
            res = minimize(lambda vv: (-float(np.sum(vv[2*n:])), -np.concatenate([np.zeros(2*n), np.ones(n)]))[0],
                           v0, method='SLSQP', jac=lambda vv: -np.concatenate([np.zeros(2*n), np.ones(n)]),
                           bounds=bnds78, constraints=cons,
                           options={'maxiter': it1, 'ftol': 1e-10, 'disp': False, 'eps': 1e-9})
            v1 = res.x if res is not None and hasattr(res, 'x') and res.x is not None else v0
            # second tight pass if time
            if (time.time() - t0) < HARD - 14.0:
                try:
                    res2 = minimize(lambda vv: (-float(np.sum(vv[2*n:])), -np.concatenate([np.zeros(2*n), np.ones(n)]))[0],
                                   v1, method='SLSQP', jac=lambda vv: -np.concatenate([np.zeros(2*n), np.ones(n)]),
                                   bounds=bnds78, constraints=cons,
                                   options={'maxiter': 300, 'ftol': 1e-13, 'disp': False, 'eps': 1e-10})
                    if res2 is not None and hasattr(res2, 'x') and res2.x is not None:
                        # accept if feasible and better
                        Xc = res2.x[0:n]; Yc = res2.x[n:2*n]; Rc = res2.x[2*n:3*n]
                        wall = np.minimum(np.minimum(Xc, 1-Xc), np.minimum(Yc, 1-Yc))
                        Dx = Xc[:, None]-Xc[None,:]; Dy = Yc[:, None]-Yc[None,:]
                        D = np.sqrt(Dx*Dx+Dy*Dy); np.fill_diagonal(D, np.inf)
                        S = Rc[:,None]+Rc[None,:]; np.fill_diagonal(S, 0)
                        if np.all(wall - Rc > -1e-9) and np.all(D - S > -1e-9):
                            if float(np.sum(Rc)) > float(np.sum(v1[2*n:])) - 1e-9:
                                v1 = res2.x
                except Exception:
                    pass
            Xf = np.clip(v1[0:n], 0.0, 1.0); Yf = np.clip(v1[n:2*n], 0.0, 1.0); Rf = np.maximum(v1[2*n:3*n], 1e-9)
            # verify and maybe keep best
            Rs, alpha = feasible_shrink(Xf, Yf, Rf)
            # compute sums
            s_slsqp = float(np.sum(Rs)) if Rs is not None else -1
            if s_slsqp > best_sum and Rs is not None:
                # final shrink already applied in Rs (alpha includes 1e-9 margin)
                best_xc, best_yc, best_r = Xf.copy(), Yf.copy(), Rs.copy()
                best_sum = s_slsqp
        except Exception:
            pass

    # ---------- final uniform feasibility shrink ~1e-9 ----------
    X = np.clip(np.asarray(best_xc, float), 0.0, 1.0)
    Y = np.clip(np.asarray(best_yc, float), 0.0, 1.0)
    R = np.maximum(np.asarray(best_r, float), 1e-12)
    Rf, _ = feasible_shrink(X, Y, R)
    if Rf is not None:
        R = Rf
    else:
        R = R * 0.99
    out = np.column_stack([X, Y, R])
    return out
