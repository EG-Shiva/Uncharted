import numpy as np
import time

def solve():
    import numpy as np
    import time
    from scipy.optimize import minimize, linprog
    N = 26
    t0 = time.time()
    TIME_BUDGET = 118.0
    np.random.seed(12345)

    def radii_lp(x, y):
        wall = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        # quick infeasibility
        if np.min(wall) < 0.005 - 1e-12:
            return None, -1.0
        Dx = x[:, None] - x[None, :]
        Dy = y[:, None] - y[None, :]
        D = np.sqrt(Dx * Dx + Dy * Dy)
        # build constraints: r_i + r_j <= D_ij - eps, r_i <= wall_i - eps
        eps = 1e-9
        npair = N * (N - 1) // 2
        A_ub = np.zeros((npair + N, N), dtype=np.float64)
        b_ub = np.zeros(npair + N, dtype=np.float64)
        k = 0
        for i in range(N):
            Di = D[i]
            for j in range(i + 1, N):
                A_ub[k, i] = 1.0
                A_ub[k, j] = 1.0
                b_ub[k] = Di[j] - eps
                k += 1
        for i in range(N):
            A_ub[k, i] = 1.0
            b_ub[k] = wall[i] - eps
            k += 1
        # need b >= 0.01 otherwise infeasible with lower 0.005
        if np.min(b_ub[:npair]) < 0.01 - 1e-12:
            return None, -1.0
        c = -np.ones(N, dtype=np.float64)
        bounds = [(0.005, None)] * N
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        except Exception:
            return None, -1.0
        if not res.success:
            return None, -1.0
        r = res.x
        # safety shrink for strict 1e-12
        r = r * (1.0 - 5e-9)
        # verify
        # walls already ok, pair ok by eps margin
        return r, float(np.sum(r))

    best_sum = -1.0
    best_x = None
    best_y = None
    best_r = None

    bounds_box = [(0.0, 1.0)] * (2 * N) + [(0.01, 0.30)] * N

    restart = 0
    max_restart = 80
    # pre-grid for diverse init
    gx = []
    gy = []
    for ix in range(6):
        for jy in range(5):
            gx.append((ix + 0.5) / 6.0)
            gy.append((jy + 0.5) / 5.0)
    gx = np.array(gx); gy = np.array(gy)

    while restart < max_restart and (time.time() - t0) < 85.0:
        # init
        if restart % 2 == 0:
            u0 = np.random.rand(N)
            v0 = np.random.rand(N)
        else:
            perm = np.random.permutation(len(gx))[:N]
            u0 = gx[perm] + (np.random.rand(N) - 0.5) * 0.06
            v0 = gy[perm] + (np.random.rand(N) - 0.5) * 0.06
            u0 = np.clip(u0, 0.0, 1.0)
            v0 = np.clip(v0, 0.0, 1.0)
        if restart % 3 == 0:
            r0 = 0.07 + 0.04 * np.random.rand(N)
        else:
            r0 = 0.04 + 0.09 * np.random.rand(N)
            r0 = np.clip(r0, 0.015, 0.28)
        z = np.concatenate([u0, v0, r0])

        for W in (20.0, 200.0, 2000.0, 20000.0):
            if (time.time() - t0) >= 85.0:
                break
            def fun(zz, W=W):
                u = zz[:N]; vv = zz[N:2*N]; rr = zz[2*N:]
                x = rr * (1.0 - 2.0 * u) + u
                y = rr * (1.0 - 2.0 * vv) + vv
                Dx = x[:, None] - x[None, :]
                Dy = y[:, None] - y[None, :]
                D = np.sqrt(Dx*Dx + Dy*Dy + 1e-16)
                np.fill_diagonal(D, 1.0)
                E = rr[:, None] + rr[None, :] - D
                np.fill_diagonal(E, 0.0)
                Ep = E
                # in-place maximum
                Ep = np.maximum(E, 0.0)
                # fix diag after maximum (E diag was 0 -> Ep diag 0)
                Pen = 0.5 * np.sum(Ep * Ep)
                F = -np.sum(rr) + W * Pen
                InvD = 1.0 / D
                np.fill_diagonal(InvD, 0.0)
                Wmat = Ep * InvD
                Gx = -2.0 * np.sum(Wmat * Dx, axis=1)
                Gy = -2.0 * np.sum(Wmat * Dy, axis=1)
                sumEp = np.sum(Ep, axis=1)
                om2r = 1.0 - 2.0 * rr
                dFdu = W * Gx * om2r
                dFdv = W * Gy * om2r
                dFdr = -np.ones(N) + W * (2.0 * sumEp + Gx * (1.0 - 2.0 * u) + Gy * (1.0 - 2.0 * vv))
                g = np.concatenate([dFdu, dFdv, dFdr])
                return F, g
            try:
                res = minimize(fun, z, method='L-BFGS-B', jac=True, bounds=bounds_box,
                               options={'maxiter': 250, 'ftol': 1e-9, 'gtol': 1e-5})
                z = res.x
            except Exception:
                break
        # extract centres
        u = z[:N]; vv = z[N:2*N]; rr = z[2*N:]
        x = rr * (1.0 - 2.0 * u) + u
        y = rr * (1.0 - 2.0 * vv) + vv
        x = np.clip(x, 1e-6, 1.0 - 1e-6)
        y = np.clip(y, 1e-6, 1.0 - 1e-6)
        r_opt, s = radii_lp(x, y)
        if r_opt is not None and s > best_sum:
            best_sum = s
            best_x = x.copy(); best_y = y.copy(); best_r = r_opt.copy()
        restart += 1

    # hill climbing on centres with LP evaluation
    if best_x is not None:
        sigmas = [0.02, 0.01, 0.005, 0.002]
        for sigma in sigmas:
            iters = 120
            for _ in range(iters):
                if (time.time() - t0) >= TIME_BUDGET:
                    break
                xn = best_x + sigma * np.random.randn(N)
                yn = best_y + sigma * np.random.randn(N)
                xn = np.clip(xn, 0.02, 0.98)
                yn = np.clip(yn, 0.02, 0.98)
                r_opt, s = radii_lp(xn, yn)
                if r_opt is not None and s > best_sum + 1e-10:
                    best_sum = s
                    best_x = xn; best_y = yn; best_r = r_opt
            if (time.time() - t0) >= TIME_BUDGET:
                break

    if best_x is None:
        # fallback grid
        n = 26
        k = int(np.ceil(np.sqrt(n)))
        r = 0.5 / k
        pts = []
        for i in range(k):
            for j in range(k):
                if len(pts) < n:
                    pts.append([(2*i+1)*r, (2*j+1)*r, r*0.999999])
        return np.array(pts, dtype=np.float64)

    out = np.column_stack([best_x, best_y, best_r])
    # final strict safety: shrink 1e-9 (already have eps margin, keep sum)
    # verify and if violated, uniform shrink
    x = out[:, 0]; y = out[:, 1]; r = out[:, 2]
    wall = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
    Dx = x[:, None] - x[None, :]; Dy = y[:, None] - y[None, :]
    D = np.sqrt(Dx*Dx + Dy*Dy)
    fw = np.min(wall / r)
    fp = 1e100
    for i in range(N):
        for j in range(i+1, N):
            fp = min(fp, D[i, j] / (r[i] + r[j]))
    f = min(fw, fp)
    if f < 1.0:
        r = r * f * (1.0 - 1e-9)
        out = np.column_stack([x, y, r])
    return out
