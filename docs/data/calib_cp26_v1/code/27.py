import numpy as np
import time

def solve():
    import numpy as np
    import time
    from scipy.optimize import minimize, linprog
    N = 26
    t0 = time.time()
    BUDGET = 123.0
    rng = np.random.default_rng(1)

    iu, ju = np.triu_indices(N, 1)
    m = len(iu)
    # constant constraint matrix: pair rows + wall rows (eye)
    A_pair = np.zeros((m, N), dtype=float)
    ar = np.arange(m)
    A_pair[ar, iu] = 1.0
    A_pair[ar, ju] = 1.0
    A_full = np.vstack([A_pair, np.eye(N, dtype=float)])
    c_obj = -np.ones(N, dtype=float)
    bounds0 = [(0.0, None)] * N

    def lp_sum_only(xc, yc):
        wall = np.minimum(np.minimum(xc, 1.0 - xc), np.minimum(yc, 1.0 - yc))
        if np.min(wall) <= 1e-12:
            return None, -1.0
        dx = xc[iu] - xc[ju]
        dy = yc[iu] - yc[ju]
        dvec = np.sqrt(dx * dx + dy * dy)
        b_ub = np.concatenate([dvec, wall])
        try:
            res = linprog(c_obj, A_ub=A_full, b_ub=b_ub, bounds=bounds0, method='highs')
        except Exception:
            return None, -1.0
        if not res.success:
            return None, -1.0
        r = np.maximum(np.asarray(res.x, dtype=float).flatten(), 0.0)
        return r, float(np.sum(r))

    def lp_dual(xc, yc):
        wall = np.minimum(np.minimum(xc, 1.0 - xc), np.minimum(yc, 1.0 - yc))
        if np.min(wall) <= 1e-12:
            return None, -1.0, None, None
        dx = xc[iu] - xc[ju]
        dy = yc[iu] - yc[ju]
        dvec = np.sqrt(dx * dx + dy * dy)
        # avoid exact zero (still feasible)
        dvec = np.maximum(dvec, 0.0)
        b_ub = np.concatenate([dvec, wall])
        try:
            res = linprog(c_obj, A_ub=A_full, b_ub=b_ub, bounds=bounds0, method='highs')
        except Exception:
            return None, -1.0, None, None
        if not res.success:
            return None, -1.0, None, None
        r = np.maximum(np.asarray(res.x, dtype=float).flatten(), 0.0)
        s = float(np.sum(r))
        # duals
        try:
            marg = np.asarray(res.ineqlin.marginals, dtype=float).flatten()
        except Exception:
            try:
                marg = np.asarray(res.ineqlin['marginals'], dtype=float).flatten()
            except Exception:
                return r, s, np.zeros(N), np.zeros(N)
        if marg.size != m + N:
            return r, s, np.zeros(N), np.zeros(N)
        wvec = np.abs(marg[:m])
        uvec = np.abs(marg[m:])
        # zero out non-active (slack large) to kill numerical noise
        # compute slacks
        rs = r[iu] + r[ju]
        slack_pair = dvec - rs
        slack_wall = wall - r
        wvec[slack_pair > 1e-7] = 0.0
        uvec[slack_wall > 1e-7] = 0.0
        # build gradient
        Dmat = np.sqrt((xc[:, None] - xc[None, :]) ** 2 + (yc[:, None] - yc[None, :]) ** 2)
        gx = np.zeros(N)
        gy = np.zeros(N)
        # pair part
        # D for pairs
        for k in range(m):
            w = wvec[k]
            if w == 0.0:
                continue
            i = iu[k]; j = ju[k]
            d = dvec[k]
            if d < 1e-12:
                continue
            ddx = (xc[i] - xc[j]) / d
            ddy = (yc[i] - yc[j]) / d
            gx[i] += w * ddx
            gx[j] -= w * ddx
            gy[i] += w * ddy
            gy[j] -= w * ddy
        # wall part: wall = min(x,1-x,y,1-y)
        for i in range(N):
            u = uvec[i]
            if u == 0.0:
                continue
            xi = xc[i]; yi = yc[i]
            candidates = (xi, 1.0 - xi, yi, 1.0 - yi)
            kmin = int(np.argmin(candidates))
            if kmin == 0:
                gx[i] += u * 1.0
            elif kmin == 1:
                gx[i] += u * (-1.0)
            elif kmin == 2:
                gy[i] += u * 1.0
            else:
                gy[i] += u * (-1.0)
        return r, s, gx, gy

    best_sum = -1.0
    best_x = None; best_y = None; best_r = None

    # ---- phase 1: random centre sampling ----
    n_init = 350
    for _ in range(n_init):
        if time.time() - t0 > 25.0:
            break
        xc = rng.uniform(0.05, 0.95, size=N)
        yc = rng.uniform(0.05, 0.95, size=N)
        r, s = lp_sum_only(xc, yc)
        if r is not None and s > best_sum:
            best_sum = s
            best_x = xc.copy(); best_y = yc.copy(); best_r = r.copy()
    # also try grid-biased inits
    for _ in range(60):
        if time.time() - t0 > 30.0:
            break
        xs = (np.arange(6) + 0.5) / 6.0
        ys = (np.arange(5) + 0.5) / 5.0
        xx, yy = np.meshgrid(xs, ys)
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        perm = rng.permutation(len(pts))[:N]
        xc = pts[perm, 0] + rng.standard_normal(N) * 0.01
        yc = pts[perm, 1] + rng.standard_normal(N) * 0.01
        xc = np.clip(xc, 0.03, 0.97); yc = np.clip(yc, 0.03, 0.97)
        r, s = lp_sum_only(xc, yc)
        if r is not None and s > best_sum:
            best_sum = s
            best_x = xc.copy(); best_y = yc.copy(); best_r = r.copy()

    steps = [0.02, 0.01, 0.005, 0.002, 0.001, 5e-4, 2e-4, 1e-4]

    def dual_ascent(xc, yc, rcur, scur, max_iter):
        xc = xc.copy(); yc = yc.copy(); scur_cur = scur; rcur_cur = rcur
        stagn = 0
        for it in range(max_iter):
            if time.time() - t0 > BUDGET - 30.0:
                break
            rlp, s, gx, gy = lp_dual(xc, yc)
            if rlp is None:
                # nudge
                xc = np.clip(xc + rng.standard_normal(N) * 0.002, 0.02, 0.98)
                yc = np.clip(yc + rng.standard_normal(N) * 0.002, 0.02, 0.98)
                continue
            if s > scur_cur + 1e-12:
                scur_cur = s; rcur_cur = rlp.copy()
            ng = np.sqrt(np.sum(gx * gx + gy * gy))
            if ng < 1e-10 or not np.isfinite(ng):
                # random kick
                xn = np.clip(xc + rng.standard_normal(N) * 0.003, 0.02, 0.98)
                yn = np.clip(yc + rng.standard_normal(N) * 0.003, 0.02, 0.98)
                rn, sn = lp_sum_only(xn, yn)
                if rn is not None and sn > scur_cur + 1e-10:
                    xc, yc, scur_cur, rcur_cur = xn, yn, sn, rn
                    stagn = 0
                else:
                    stagn += 1
                if stagn > 25:
                    # bigger shake
                    xn = np.clip(xc + rng.standard_normal(N) * 0.015, 0.02, 0.98)
                    yn = np.clip(yc + rng.standard_normal(N) * 0.015, 0.02, 0.98)
                    rn, sn = lp_sum_only(xn, yn)
                    if rn is not None and sn > scur_cur - 0.01:
                        xc, yc = xn, yn
                        if sn > scur_cur:
                            scur_cur, rcur_cur = sn, rn
                    stagn = 0
                continue
            improved = False
            for st in steps:
                xn = xc + st * gx
                yn = yc + st * gy
                np.clip(xn, 0.02, 0.98, out=xn)
                np.clip(yn, 0.02, 0.98, out=yn)
                rn, sn = lp_sum_only(xn, yn)
                if rn is not None and sn > scur_cur + 1e-10:
                    xc, yc, scur_cur, rcur_cur = xn, yn, sn, rn
                    improved = True
                    stagn = 0
                    break
            if not improved:
                stagn += 1
                # try normalized step
                if stagn % 3 == 0:
                    nn = ng + 1e-12
                    for st in (0.005, 0.002, 0.001):
                        xn = np.clip(xc + st * gx / nn * np.sqrt(float(N)), 0.02, 0.98)
                        yn = np.clip(yc + st * gy / nn * np.sqrt(float(N)), 0.02, 0.98)
                        rn, sn = lp_sum_only(xn, yn)
                        if rn is not None and sn > scur_cur + 1e-10:
                            xc, yc, scur_cur, rcur_cur = xn, yn, sn, rn
                            improved = True
                            stagn = 0
                            break
                if not improved and stagn >= 12:
                    # random perturbation to change topology
                    xn = np.clip(xc + rng.standard_normal(N) * 0.008, 0.02, 0.98)
                    yn = np.clip(yc + rng.standard_normal(N) * 0.008, 0.02, 0.98)
                    rn, sn = lp_sum_only(xn, yn)
                    if rn is not None and sn > scur_cur - 0.005:
                        xc, yc = xn, yn
                        if sn > scur_cur:
                            scur_cur, rcur_cur = sn, rn
                    stagn = 0
        return xc, yc, rcur_cur, scur_cur

    # ---- phase 2: dual ascent on best ----
    if best_x is not None:
        best_x, best_y, best_r, best_sum = dual_ascent(best_x, best_y, best_r, best_sum, 400)

    # ---- phase 3: occasional exact NLP polishing (joint penalty L-BFGS-B) ----
    def joint_polish(x0, y0, r0, t_lim):
        n = N
        v0 = np.concatenate([x0, y0, r0])
        bnds = [(0.0, 1.0)] * (2 * n) + [(0.005, 0.30)] * n
        v = v0.copy()
        for W in (200.0, 2000.0, 20000.0):
            if time.time() - t0 > t_lim:
                break
            def fun(vv, W=W):
                X = vv[:n]; Y = vv[n:2*n]; R = vv[2*n:]
                loss = -np.sum(R)
                gx = np.zeros(n); gy = np.zeros(n); grpen = np.zeros(n)
                pen = 0.0
                v1 = R - X
                mm = v1 > 0
                if np.any(mm):
                    pen += np.sum(v1[mm]**2); gx[mm] += -2*v1[mm]; grpen[mm] += 2*v1[mm]
                v2 = X + R - 1.0
                mm = v2 > 0
                if np.any(mm):
                    pen += np.sum(v2[mm]**2); gx[mm] += 2*v2[mm]; grpen[mm] += 2*v2[mm]
                v3 = R - Y
                mm = v3 > 0
                if np.any(mm):
                    pen += np.sum(v3[mm]**2); gy[mm] += -2*v3[mm]; grpen[mm] += 2*v3[mm]
                v4 = Y + R - 1.0
                mm = v4 > 0
                if np.any(mm):
                    pen += np.sum(v4[mm]**2); gy[mm] += 2*v4[mm]; grpen[mm] += 2*v4[mm]
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
                    Q = np.zeros((n, n))
                    Q[Mask] = -2.0*V[Mask]/D[Mask]
                    gx += np.sum(Q*Dx, axis=1)
                    gy += np.sum(Q*Dy, axis=1)
                    G = np.zeros((n, n)); G[Mask] = 2.0*V[Mask]
                    grpen += np.sum(G, axis=1)
                loss = loss + W*pen
                g = np.empty_like(vv)
                g[:n] = W*gx; g[n:2*n] = W*gy; g[2*n:] = -1.0 + W*grpen
                return loss, g
            try:
                res = minimize(fun, v, method='L-BFGS-B', jac=True, bounds=bnds,
                               options={'maxiter': 150, 'ftol': 1e-9, 'gtol': 1e-5})
                v = res.x
            except Exception:
                break
        X = v[:n]; Y = v[n:2*n]
        X = np.clip(X, 0.02, 0.98); Y = np.clip(Y, 0.02, 0.98)
        r, s = lp_sum_only(X, Y)
        return X, Y, r, s

    if best_x is not None and (time.time() - t0) < BUDGET - 45.0:
        Xp, Yp, rp, sp = joint_polish(best_x, best_y, best_r, BUDGET - 35.0)
        if rp is not None and sp > best_sum:
            best_sum = sp; best_x = Xp; best_y = Yp; best_r = rp
        # second dual ascent after polish
        best_x, best_y, best_r, best_sum = dual_ascent(best_x, best_y, best_r, best_sum, 250)

    # ---- phase 4: LP hill climbing with shrinking sigma ----
    if best_x is not None:
        for sigma in (0.01, 0.005, 0.002, 0.001, 0.0005):
            iters = 300
            for _ in range(iters):
                if time.time() - t0 > BUDGET:
                    break
                xn = np.clip(best_x + rng.standard_normal(N) * sigma, 0.02, 0.98)
                yn = np.clip(best_y + rng.standard_normal(N) * sigma, 0.02, 0.98)
                rn, sn = lp_sum_only(xn, yn)
                if rn is not None and sn > best_sum + 1e-10:
                    best_sum = sn; best_x = xn; best_y = yn; best_r = rn
            if time.time() - t0 > BUDGET:
                break

    if best_x is None:
        k = int(np.ceil(np.sqrt(N)))
        rr = 0.5 / k
        pts = []
        for i in range(k):
            for j in range(k):
                if len(pts) < N:
                    pts.append([(2*i+1)*rr, (2*j+1)*rr, rr*0.999])
        return np.array(pts, dtype=float)

    out = np.column_stack([best_x, best_y, best_r])
    # strict safety shrink
    x = out[:, 0]; y = out[:, 1]; r = out[:, 2]
    wall = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
    Dx = x[:, None] - x[None, :]; Dy = y[:, None] - y[None, :]
    D = np.sqrt(Dx*Dx + Dy*Dy)
    fw = np.min(wall / np.maximum(r, 1e-18))
    fp = 1e100
    for i in range(N):
        for j in range(i+1, N):
            v = D[i, j] / (r[i] + r[j]) if (r[i]+r[j]) > 0 else 1e100
            if v < fp:
                fp = v
    f = min(fw, fp)
    if (not np.isfinite(f)) or f >= 1.0:
        f = 1.0 - 1e-9
        r = r * f
        out = np.column_stack([x, y, r])
    else:
        r = r * f * (1.0 - 1e-9)
        out = np.column_stack([x, y, r])
    return out
