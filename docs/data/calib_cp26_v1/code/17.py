import numpy as np
import time

def solve():
    import numpy as np
    import time as _tm
    from scipy.optimize import minimize, linprog
    N = 26
    t0 = _tm.time()
    DEADLINE = 124.0
    # time split: joint restarts until 70s, refinement rest of time
    JOINT_END = 70.0
    rng_global = np.random.RandomState(7)

    iu, ju = np.triu_indices(N, 1)
    m = len(iu)
    A_ub0 = np.zeros((m, N), dtype=float)
    ar = np.arange(m)
    A_ub0[ar, iu] = 1.0
    A_ub0[ar, ju] = 1.0
    c_obj = -np.ones(N, dtype=float)

    def eval_lp(xc, yc, best_ref=None):
        b = np.minimum(np.minimum(xc, 1.0 - xc), np.minimum(yc, 1.0 - yc))
        # infeasible centres outside
        if np.min(b) <= 0:
            return None, -1.0
        sb = np.sum(b)
        if best_ref is not None and sb <= best_ref:
            return None, -1.0
        dx = xc[iu] - xc[ju]
        dy = yc[iu] - yc[ju]
        dvec = np.sqrt(dx * dx + dy * dy)
        # if any pair coincident, still feasible but force zero; keep
        bounds = [(0.0, float(bi)) for bi in b]
        try:
            res = linprog(c_obj, A_ub=A_ub0, b_ub=dvec, bounds=bounds, method='highs')
        except Exception:
            return None, -1.0
        if not res.success or res.x is None:
            return None, -1.0
        r = np.maximum(np.array(res.x, dtype=float).flatten(), 0.0)
        s = float(np.sum(r))
        return r, s

    best_sum = -1.0
    best_x = None
    best_y = None
    best_r = None

    def try_update(xc, yc):
        global_vars = None
        r, s = eval_lp(xc, yc, best_sum)
        return r, s

    # ---------- Stage 1: joint penalized L-BFGS-B restarts ----------
    bounds_box = [(0.0, 1.0)] * (2 * N) + [(0.01, 0.30)] * N
    # grid pool for init
    gx = []; gy = []
    for ix in range(6):
        for jy in range(5):
            gx.append((ix + 0.5) / 6.0); gy.append((jy + 0.5) / 5.0)
    gx = np.array(gx); gy = np.array(gy)

    restart = 0
    while (_tm.time() - t0) < JOINT_END and restart < 200:
        rs = np.random.RandomState(12345 + restart * 977)
        if restart == 0:
            # 5x5 + extra
            xs = []; ys = []
            for i in range(5):
                for j in range(5):
                    xs.append(0.1 + 0.2 * i); ys.append(0.1 + 0.2 * j)
            xs = np.array(xs); ys = np.array(ys)
            # add 26th near centre gap
            x0c = np.append(xs, [0.5]); y0c = np.append(ys, [0.55])
            # convert to u,v approx? we init u,v directly as centres then r
            u0 = x0c + rs.randn(N) * 0.003
            v0 = y0c + rs.randn(N) * 0.003
            u0 = np.clip(u0, 0.01, 0.99); v0 = np.clip(v0, 0.01, 0.99)
            r0 = np.full(N, 0.09); r0[-1] = 0.05
        elif restart % 2 == 1:
            perm = rs.permutation(len(gx))[:N]
            u0 = gx[perm] + (rs.rand(N) - 0.5) * 0.08
            v0 = gy[perm] + (rs.rand(N) - 0.5) * 0.08
            u0 = np.clip(u0, 0.0, 1.0); v0 = np.clip(v0, 0.0, 1.0)
            r0 = 0.06 + 0.05 * rs.rand(N)
        else:
            u0 = rs.rand(N); v0 = rs.rand(N)
            r0 = 0.05 + 0.06 * rs.rand(N)
        r0 = np.clip(r0, 0.015, 0.28)
        z = np.concatenate([u0, v0, r0])
        # staged penalties
        for W in (30.0, 300.0, 3000.0, 30000.0):
            if (_tm.time() - t0) >= JOINT_END:
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
                Ep = np.maximum(E, 0.0)
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
                               options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-5})
                z = res.x
            except Exception:
                break
        u = z[:N]; vv = z[N:2*N]; rr = z[2*N:]
        x = rr * (1.0 - 2.0 * u) + u
        y = rr * (1.0 - 2.0 * vv) + vv
        x = np.clip(x, 0.02, 0.98); y = np.clip(y, 0.02, 0.98)
        r_opt, s = eval_lp(x, y, best_sum)
        if r_opt is not None and s > best_sum:
            best_sum = s
            best_x = x.copy(); best_y = y.copy(); best_r = r_opt.copy()
        restart += 1

    if best_x is None:
        k = int(np.ceil(np.sqrt(N)))
        rr = 0.5 / k
        pts = []
        for i in range(k):
            for j in range(k):
                if len(pts) < N:
                    pts.append([(2*i+1)*rr, (2*j+1)*rr, rr*0.999])
        arr = np.array(pts, dtype=float)
        # shrink safe
        arr[:,2] *= (1.0 - 1e-7)
        return arr

    # ---------- Stage 2: greedy full-vector + single-circle hill climbing ----------
    # full vector gaussian
    sigmas = [0.015, 0.008, 0.004, 0.002]
    for sigma in sigmas:
        iters = 250
        for _ in range(iters):
            if (_tm.time() - t0) >= DEADLINE - 18.0:
                break
            xn = best_x + sigma * rng_global.randn(N)
            yn = best_y + sigma * rng_global.randn(N)
            xn = np.clip(xn, 0.02, 0.98); yn = np.clip(yn, 0.02, 0.98)
            r_opt, s = eval_lp(xn, yn, best_sum + 1e-12)
            if r_opt is not None and s > best_sum + 1e-10:
                best_sum = s
                best_x = xn; best_y = yn; best_r = r_opt
        if (_tm.time() - t0) >= DEADLINE - 18.0:
            break

    # single-circle moves (large number, cheap: 1 LP each)
    for _ in range(2500):
        if (_tm.time() - t0) >= DEADLINE - 12.0:
            break
        # decaying sigma
        sigma = 0.01 if (_ % 3) else 0.003
        i = rng_global.integers(0, N)
        xn = best_x.copy(); yn = best_y.copy()
        xn[i] += sigma * rng_global.standard_normal()
        yn[i] += sigma * rng_global.standard_normal()
        if xn[i] < 0.02 or xn[i] > 0.98 or yn[i] < 0.02 or yn[i] > 0.98:
            continue
        r_opt, s = eval_lp(xn, yn, best_sum + 1e-12)
        if r_opt is not None and s > best_sum + 1e-10:
            best_sum = s
            best_x = xn; best_y = yn; best_r = r_opt

    # ---------- Stage 3: per-circle Powell coordinate ascent ----------
    # sweep circles in random order, 2-D derivative-free optimisation of LP sum
    order = rng_global.permutation(N)
    n_sweeps = 2
    for sweep in range(n_sweeps):
        for idx in order:
            if (_tm.time() - t0) >= DEADLINE - 2.0:
                break
            i = int(idx)
            x0 = np.array([best_x[i], best_y[i]], dtype=float)
            def f2d(p):
                xn = best_x.copy(); yn = best_y.copy()
                xn[i] = float(np.clip(p[0], 0.02, 0.98))
                yn[i] = float(np.clip(p[1], 0.02, 0.98))
                _, s = eval_lp(xn, yn, None)
                # eval_lp returns -1 if infeasible; penalise
                return -s
            try:
                res = minimize(f2d, x0, method='Powell',
                               options={'maxiter': 25, 'disp': False})
                xn = best_x.copy(); yn = best_y.copy()
                xn[i] = float(np.clip(res.x[0], 0.02, 0.98))
                yn[i] = float(np.clip(res.x[1], 0.02, 0.98))
                r_opt, s = eval_lp(xn, yn, best_sum + 1e-9)
                if r_opt is not None and s > best_sum + 1e-9:
                    best_sum = s
                    best_x = xn; best_y = yn; best_r = r_opt
            except Exception:
                continue
        # reshuffle
        order = rng_global.permutation(N)
        if (_tm.time() - t0) >= DEADLINE - 2.0:
            break

    # final small full-vector polish
    for _ in range(300):
        if (_tm.time() - t0) >= DEADLINE - 0.5:
            break
        sigma = 0.0015
        xn = best_x + sigma * rng_global.randn(N)
        yn = best_y + sigma * rng_global.randn(N)
        xn = np.clip(xn, 0.02, 0.98); yn = np.clip(yn, 0.02, 0.98)
        r_opt, s = eval_lp(xn, yn, best_sum + 1e-12)
        if r_opt is not None and s > best_sum + 1e-10:
            best_sum = s
            best_x = xn; best_y = yn; best_r = r_opt

    x = best_x; y = best_y; r = best_r.copy()
    # strict safety shrink
    r = np.maximum(r, 1e-9)
    wall = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    Dx = x[:, None] - x[None, :]; Dy = y[:, None] - y[None, :]
    D = np.sqrt(Dx*Dx + Dy*Dy)
    # compute feasible scale
    alpha = 1.0
    wr = wall / r
    alpha = float(np.min(wr))
    S = r[:, None] + r[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = D / S
    np.fill_diagonal(ratio, np.inf)
    alpha = min(alpha, float(np.min(ratio)))
    if not np.isfinite(alpha) or alpha <= 0:
        alpha = 0.5
    if alpha < 1.0:
        r = r * alpha
    # tiny margin for 1e-12 strictness
    r = r * (1.0 - 1e-7)
    # ensure positivity
    r = np.maximum(r, 1e-9)
    out = np.column_stack([x, y, r])
    return out
