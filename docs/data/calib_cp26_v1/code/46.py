import numpy as np
import time

def solve():
    import numpy as np
    import time
    from scipy.optimize import minimize, linprog
    n = 26
    t0 = time.time()
    DEADLINE = 122.0
    rng = np.random.default_rng(0)

    I, J = np.triu_indices(n, 1)
    m = len(I)
    A0 = np.zeros((m, n), dtype=float)
    A0[np.arange(m), I] = 1.0
    A0[np.arange(m), J] = 1.0
    c_primal = -np.ones(n)

    # dual matrix M (n x (m+n))
    p = m + n
    M = np.zeros((n, p), dtype=float)
    for k in range(m):
        M[I[k], k] = 1.0
        M[J[k], k] = 1.0
    for i in range(n):
        M[i, m + i] = 1.0
    Aub_dual = -M
    bub_dual = -np.ones(n)
    dual_bounds = [(0.0, None)] * p

    def lp_primal(X, Y):
        wall = np.minimum(np.minimum(X, 1.0 - X), np.minimum(Y, 1.0 - Y))
        if np.any(wall < 0):
            return -1.0, None
        wall = np.maximum(wall, 0.0)
        dx = X[I] - X[J]
        dy = Y[I] - Y[J]
        d = np.sqrt(dx * dx + dy * dy)
        bnds = [(0.0, float(w)) for w in wall]
        try:
            res = linprog(c_primal, A_ub=A0, b_ub=d, bounds=bnds, method='highs')
        except Exception:
            return -1.0, None
        if res.status != 0:
            return -1.0, None
        r = np.asarray(res.x, dtype=float)
        r = np.maximum(r, 0.0)
        return float(np.sum(r)), r

    def lp_primal_dual(X, Y):
        wall = np.minimum(np.minimum(X, 1.0 - X), np.minimum(Y, 1.0 - Y))
        if np.any(wall < -1e-12):
            return -1.0, None, None, None
        wall = np.maximum(wall, 0.0)
        dx = X[I] - X[J]
        dy = Y[I] - Y[J]
        d = np.sqrt(dx * dx + dy * dy)
        bnds = [(0.0, float(w)) for w in wall]
        try:
            res = linprog(c_primal, A_ub=A0, b_ub=d, bounds=bnds, method='highs')
        except Exception:
            return -1.0, None, None, None
        if res.status != 0:
            return -1.0, None, None, None
        r = np.maximum(np.asarray(res.x, dtype=float), 0.0)
        s = float(np.sum(r))
        # dual
        c_dual = np.concatenate([d, wall])
        try:
            rd = linprog(c_dual, A_ub=Aub_dual, b_ub=bub_dual, bounds=dual_bounds, method='highs')
        except Exception:
            return s, r, None, None
        if rd.status != 0:
            return s, r, None, None
        y = np.asarray(rd.x, dtype=float)
        lam = y[:m]
        mu = y[m:]
        return s, r, lam, mu

    def uniform_shrink(X, Y, R):
        R = np.maximum(np.asarray(R, float), 1e-14)
        wall = np.minimum(np.minimum(X, 1.0 - X), np.minimum(Y, 1.0 - Y))
        Dx = X[:, None] - X[None, :]
        Dy = Y[:, None] - Y[None, :]
        D = np.sqrt(Dx * Dx + Dy * Dy)
        S = R[:, None] + R[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            rw = wall / R
            rp = D / S
        np.fill_diagonal(rp, np.inf)
        a = float(np.min(rw))
        b = float(np.min(rp))
        alpha = min(1.0, a, b)
        alpha = alpha * (1.0 - 1e-9) - 1e-12
        if not np.isfinite(alpha) or alpha <= 0:
            alpha = 0.5
        return R * alpha

    # fallback
    k0 = int(np.ceil(np.sqrt(n)))
    r0 = 0.5 / k0
    fb = []
    for ix in range(k0):
        for jy in range(k0):
            if len(fb) < n:
                fb.append([(2 * ix + 1) * r0, (2 * jy + 1) * r0])
    fb = np.array(fb, dtype=float)
    s_fb, r_fb = lp_primal(fb[:, 0], fb[:, 1])
    if s_fb < 0:
        # uniform grid feasible
        Rfb = np.full(n, r0 * 0.99)
        best_X = fb[:, 0].copy(); best_Y = fb[:, 1].copy(); best_R = Rfb
        best_sum = float(np.sum(Rfb))
    else:
        best_X = fb[:, 0].copy(); best_Y = fb[:, 1].copy()
        best_R = uniform_shrink(best_X, best_Y, r_fb)
        best_sum = float(np.sum(best_R))
    pool = [(best_sum, best_X.copy(), best_Y.copy())]

    # ---- Phase A: annealed penalty joint search ----
    bounds78 = [(0.0, 1.0)] * (2 * n) + [(0.03, 0.20)] * n

    def make_pen_fun(nn):
        def fun(v, W):
            X = v[0:nn]; Y = v[nn:2*nn]; R = v[2*nn:3*nn]
            gx = np.zeros(nn); gy = np.zeros(nn)
            pen_gr = np.zeros(nn)
            pen = 0.0
            v1 = R - X
            mm = v1 > 0
            if np.any(mm):
                pen += np.sum(v1[mm] ** 2)
                gx[mm] += -2.0 * v1[mm]; pen_gr[mm] += 2.0 * v1[mm]
            v2 = X + R - 1.0
            mm = v2 > 0
            if np.any(mm):
                pen += np.sum(v2[mm] ** 2)
                gx[mm] += 2.0 * v2[mm]; pen_gr[mm] += 2.0 * v2[mm]
            v3 = R - Y
            mm = v3 > 0
            if np.any(mm):
                pen += np.sum(v3[mm] ** 2)
                gy[mm] += -2.0 * v3[mm]; pen_gr[mm] += 2.0 * v3[mm]
            v4 = Y + R - 1.0
            mm = v4 > 0
            if np.any(mm):
                pen += np.sum(v4[mm] ** 2)
                gy[mm] += 2.0 * v4[mm]; pen_gr[mm] += 2.0 * v4[mm]
            Dx = X[:, None] - X[None, :]
            Dy = Y[:, None] - Y[None, :]
            D = np.sqrt(Dx * Dx + Dy * Dy + 1e-20)
            S = R[:, None] + R[None, :]
            V = S - D
            np.fill_diagonal(V, 0.0)
            np.fill_diagonal(D, 1.0)
            Mask = V > 0
            if np.any(Mask):
                pen += 0.5 * np.sum(V[Mask] ** 2)
                Q = np.zeros((nn, nn))
                Q[Mask] = -2.0 * V[Mask] / D[Mask]
                gx += np.sum(Q * Dx, axis=1)
                gy += np.sum(Q * Dy, axis=1)
                G = np.zeros((nn, nn))
                G[Mask] = 2.0 * V[Mask]
                pen_gr += np.sum(G, axis=1)
            loss = -np.sum(R) + W * pen
            g = np.empty_like(v)
            g[0:nn] = W * gx
            g[nn:2*nn] = W * gy
            g[2*nn:3*nn] = -1.0 + W * pen_gr
            return loss, g
        return fun
    pen_fun = make_pen_fun(n)
    Ws = [1e2, 1e3, 1e4, 1e5]
    restart = 0
    # run until ~45s
    while time.time() - t0 < 45.0 and restart < 40:
        if restart == 0:
            X0 = np.zeros(n); Y0 = np.zeros(n)
            idx = 0
            for ii in range(5):
                for jj in range(5):
                    X0[idx] = (ii + 0.5) / 5.0 + rng.normal(0, 0.008)
                    Y0[idx] = (jj + 0.5) / 5.0 + rng.normal(0, 0.008)
                    idx += 1
            X0[25] = 0.9 + rng.normal(0, 0.01); Y0[25] = 0.1 + rng.normal(0, 0.01)
            X0 = np.clip(X0, 0.05, 0.95); Y0 = np.clip(Y0, 0.05, 0.95)
            R0 = np.full(n, 0.095) + rng.normal(0, 0.003, n)
            R0 = np.clip(R0, 0.03, 0.20)
        elif restart == 1:
            pts = []
            for ix in range(6):
                for jy in range(5):
                    xx = (ix + 0.5) / 6.0; yy = (jy + 0.5) / 5.0
                    if jy % 2 == 1:
                        xx += 0.25 / 6.0
                    pts.append((xx, yy))
            pts = np.array(pts)
            perm = rng.permutation(len(pts))[:n]
            X0 = pts[perm, 0] + rng.normal(0, 0.006, n)
            Y0 = pts[perm, 1] + rng.normal(0, 0.006, n)
            X0 = np.clip(X0, 0.05, 0.95); Y0 = np.clip(Y0, 0.05, 0.95)
            R0 = 0.085 + 0.02 * rng.random(n)
        else:
            X0 = rng.random(n) * 0.9 + 0.05
            Y0 = rng.random(n) * 0.9 + 0.05
            R0 = 0.07 + 0.05 * rng.random(n)
        v = np.concatenate([X0, Y0, R0])
        for W in Ws:
            if time.time() - t0 > 46.0:
                break
            try:
                res = minimize(pen_fun, v, args=(W,), jac=True, method='L-BFGS-B',
                               bounds=bounds78, options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-6})
                v = res.x
            except Exception:
                break
        X = np.clip(v[0:n], 0.0, 1.0); Y = np.clip(v[n:2*n], 0.0, 1.0)
        s_lp, r_lp = lp_primal(X, Y)
        if s_lp > 0:
            rs = uniform_shrink(X, Y, r_lp)
            s = float(np.sum(rs))
            if s > best_sum:
                best_sum = s; best_X = X.copy(); best_Y = Y.copy(); best_R = rs.copy()
            pool.append((s, X.copy(), Y.copy()))
        restart += 1

    pool = sorted(pool, key=lambda t: -t[0])[:6]

    # add symmetries of best
    try:
        bx, by = best_X.copy(), best_Y.copy()
        syms = [(bx, by), (1-bx, by), (bx, 1-by), (1-bx, 1-by),
                (by, bx), (1-by, bx), (by, 1-bx), (1-by, 1-bx)]
        for sx, sy in syms:
            s_lp, _ = lp_primal(sx, sy)
            if s_lp > 0:
                pool.append((s_lp, sx.copy(), sy.copy()))
        pool = sorted(pool, key=lambda t: -t[0])[:8]
    except Exception:
        pass

    # ---- Phase B: LP-dual envelope gradient centre optimisation ----
    bounds52 = [(0.02, 0.98)] * (2 * n)

    def centre_fun_grad(v):
        X = v[0:n]; Y = v[n:2*n]
        s, r, lam, mu = lp_primal_dual(X, Y)
        if s < 0 or r is None:
            return 1e3, np.zeros_like(v)
        f = -s
        g = np.zeros_like(v)
        if lam is None or mu is None:
            return f, g
        dx = X[I] - X[J]
        dy = Y[I] - Y[J]
        d = np.sqrt(dx * dx + dy * dy)
        # pair contributions
        gx = np.zeros(n); gy = np.zeros(n)
        nz = d > 1e-12
        contrib = np.zeros(m)
        contrib[nz] = lam[nz] / d[nz]
        gx += np.bincount(I, weights=contrib * dx, minlength=n)
        gx -= np.bincount(J, weights=contrib * dx, minlength=n)
        # careful: above double counts? bincount add then subtract: net correct
        # Actually first line adds for I, second subtracts for J? Let's do explicit loop-free:
        # gx[I] += lam*dx/d ; gx[J] -= lam*dx/d
        # bincount approach: gx = bincount(I,...) - bincount(J,...) ; same for gy
        gx = np.bincount(I, weights=contrib * dx, minlength=n) - np.bincount(J, weights=contrib * dx, minlength=n)
        # Wait contrib*dx already = lam*dx/d, need separate for dy
        gy = np.bincount(I, weights=contrib * dy, minlength=n) - np.bincount(J, weights=contrib * dy, minlength=n)
        # wall contributions
        for i in range(n):
            mui = mu[i]
            if mui == 0:
                continue
            xi = X[i]; yi = Y[i]
            cands = (xi, 1.0 - xi, yi, 1.0 - yi)
            k = int(np.argmin(cands))
            if k == 0:
                gx[i] += mui * 1.0
            elif k == 1:
                gx[i] += mui * (-1.0)
            elif k == 2:
                gy[i] += mui * 1.0
            else:
                gy[i] += mui * (-1.0)
        g[0:n] = -gx
        g[n:2*n] = -gy
        # clip huge grads
        g = np.clip(g, -10.0, 10.0)
        return f, g

    # polish top few
    n_polish = min(4, len(pool))
    for pi in range(n_polish):
        if time.time() - t0 > 95.0:
            break
        _, px, py = pool[pi]
        # small perturbation to escape kink
        for pert in [0.0, 0.002]:
            if time.time() - t0 > 95.0:
                break
            if pert > 0:
                v0 = np.concatenate([px + rng.normal(0, pert, n), py + rng.normal(0, pert, n)])
            else:
                v0 = np.concatenate([px, py])
            v0 = np.clip(v0, 0.02, 0.98)
            try:
                res = minimize(centre_fun_grad, v0, jac=True, method='L-BFGS-B',
                               bounds=bounds52, options={'maxiter': 120, 'ftol': 1e-12, 'gtol': 1e-7})
                Xp = res.x[0:n]; Yp = res.x[n:2*n]
                s_lp, r_lp = lp_primal(Xp, Yp)
                if s_lp > 0:
                    rs = uniform_shrink(Xp, Yp, r_lp)
                    s = float(np.sum(rs))
                    if s > best_sum:
                        best_sum = s; best_X = Xp.copy(); best_Y = Yp.copy(); best_R = rs.copy()
                    pool.append((s, Xp.copy(), Yp.copy()))
            except Exception:
                continue

    # refresh best
    pool = sorted(pool, key=lambda t: -t[0])
    best_sum, best_X, best_Y = pool[0][0], pool[0][1].copy(), pool[0][2].copy()
    s_lp, r_lp = lp_primal(best_X, best_Y)
    if s_lp > 0:
        best_R = uniform_shrink(best_X, best_Y, r_lp)
        best_sum = float(np.sum(best_R))
    else:
        best_R = uniform_shrink(best_X, best_Y, np.full(n, 0.08))

    # ---- Phase C: SLSQP KKT polishing over 78 vars ----
    if time.time() - t0 < DEADLINE - 10:
        z0 = np.concatenate([best_X, best_Y, best_R])
        # ensure strictly feasible interior
        z0[2*n:3*n] = z0[2*n:3*n] * 0.9999 + 1e-10

        def obj(z):
            return -np.sum(z[2*n:3*n])
        def obj_jac(z):
            g = np.zeros_like(z)
            g[2*n:3*n] = -1.0
            return g
        def wall_vec(z):
            X = z[0:n]; Y = z[n:2*n]; R = z[2*n:3*n]
            return np.concatenate([X - R, 1.0 - X - R, Y - R, 1.0 - Y - R])
        def wall_jac(z):
            Jm = np.zeros((4*n, 3*n))
            for i in range(n):
                Jm[i, i] = 1.0; Jm[i, 2*n+i] = -1.0
                Jm[n+i, i] = -1.0; Jm[n+i, 2*n+i] = -1.0
                Jm[2*n+i, n+i] = 1.0; Jm[2*n+i, 2*n+i] = -1.0
                Jm[3*n+i, n+i] = -1.0; Jm[3*n+i, 2*n+i] = -1.0
            return Jm
        def pair_vec(z):
            X = z[0:n]; Y = z[n:2*n]; R = z[2*n:3*n]
            dx = X[I] - X[J]; dy = Y[I] - Y[J]
            d = np.sqrt(dx*dx + dy*dy)
            return d - R[I] - R[J]
        def pair_jac(z):
            X = z[0:n]; Y = z[n:2*n]
            dx = X[I] - X[J]; dy = Y[I] - Y[J]
            d = np.sqrt(dx*dx + dy*dy + 1e-24)
            Jm = np.zeros((m, 3*n))
            inv = 1.0 / np.maximum(d, 1e-12)
            ux = dx * inv; uy = dy * inv
            for k in range(m):
                i = I[k]; j = J[k]
                Jm[k, i] = ux[k]; Jm[k, j] = -ux[k]
                Jm[k, n+i] = uy[k]; Jm[k, n+j] = -uy[k]
                Jm[k, 2*n+i] = -1.0; Jm[k, 2*n+j] = -1.0
            return Jm
        cons = [{'type': 'ineq', 'fun': wall_vec, 'jac': wall_jac},
                {'type': 'ineq', 'fun': pair_vec, 'jac': pair_jac}]
        b78 = [(0.0, 1.0)]*(2*n) + [(0.005, 0.25)]*n
        try:
            res = minimize(obj, z0, jac=obj_jac, method='SLSQP', bounds=b78,
                           constraints=cons,
                           options={'maxiter': 400, 'ftol': 1e-13, 'eps': 1e-10, 'disp': False})
            if res.success or res.fun < -best_sum:
                z = np.asarray(res.x)
                X = np.clip(z[0:n], 0.0, 1.0); Y = np.clip(z[n:2*n], 0.0, 1.0); R = np.maximum(z[2*n:3*n], 1e-12)
                # verify feasibility with margin
                wall = np.minimum(np.minimum(X, 1-X), np.minimum(Y, 1-Y))
                dx = X[I]-X[J]; dy = Y[I]-Y[J]; d = np.sqrt(dx*dx+dy*dy)
                if np.all(wall - R >= -1e-9) and np.all(d - R[I] - R[J] >= -1e-9):
                    Rs = uniform_shrink(X, Y, R)
                    s = float(np.sum(Rs))
                    if s > best_sum:
                        best_sum = s; best_X, best_Y, best_R = X, Y, Rs
        except Exception:
            pass
        # second SLSQP from slightly perturbed best if time remains
        if time.time() - t0 < DEADLINE - 15:
            try:
                z1 = np.concatenate([best_X, best_Y, best_R * 1.0005])
                # re-shrink to feasible
                Rs0 = uniform_shrink(z1[0:n], z1[n:2*n], z1[2*n:3*n])
                z1[2*n:3*n] = Rs0
                res2 = minimize(obj, z1, jac=obj_jac, method='SLSQP', bounds=b78,
                                constraints=cons,
                                options={'maxiter': 300, 'ftol': 1e-14, 'eps': 1e-11, 'disp': False})
                z = np.asarray(res2.x)
                X = np.clip(z[0:n], 0.0, 1.0); Y = np.clip(z[n:2*n], 0.0, 1.0); R = np.maximum(z[2*n:3*n], 1e-12)
                wall = np.minimum(np.minimum(X, 1-X), np.minimum(Y, 1-Y))
                dx = X[I]-X[J]; dy = Y[I]-Y[J]; d = np.sqrt(dx*dx+dy*dy)
                if np.all(wall - R >= -1e-9) and np.all(d - R[I] - R[J] >= -1e-9):
                    Rs = uniform_shrink(X, Y, R)
                    s = float(np.sum(Rs))
                    if s > best_sum:
                        best_sum = s; best_X, best_Y, best_R = X, Y, Rs
            except Exception:
                pass

    best_R = uniform_shrink(best_X, best_Y, best_R)
    out = np.column_stack([best_X, best_Y, best_R])
    return out
