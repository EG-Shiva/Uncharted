import numpy as np
import time

def solve():
    import numpy as np
    import time
    from scipy.optimize import minimize, linprog
    N = 26
    t0 = time.time()
    BUDGET = 124.0
    PHASE_A_END = 58.0
    np.random.seed(7)
    # fixed pair matrix: r_i + r_j <= D_ij
    npair = N * (N - 1) // 2
    A_pair = np.zeros((npair, N), dtype=np.float64)
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            A_pair[k, i] = 1.0
            A_pair[k, j] = 1.0
            k += 1
    cvec = -np.ones(N, dtype=np.float64)
    iu = np.triu_indices(N, 1)
    LB = 0.005
    EPS = 1e-9

    def lp_opt(x, y):
        # walls
        wall = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y)) - EPS
        if np.min(wall) < LB:
            return None, -1.0
        Dx = x[:, None] - x[None, :]
        Dy = y[:, None] - y[None, :]
        D = np.sqrt(Dx * Dx + Dy * Dy)
        b_vec = D[iu] - EPS
        if np.min(b_vec) < 2 * LB - 1e-12:
            return None, -1.0
        bounds = [(LB, float(w)) for w in wall]
        try:
            res = linprog(cvec, A_ub=A_pair, b_ub=b_vec, bounds=bounds, method='highs')
        except Exception:
            return None, -1.0
        if not res.success:
            return None, -1.0
        r = res.x * (1.0 - 2e-9)
        # quick sum
        return r, float(np.sum(r))

    best_sum = -1.0
    best_x = None
    best_y = None
    best_r = None

    bounds_box = [(0.0, 1.0)] * (2 * N) + [(0.01, 0.30)] * N

    # grid base for diversity
    gx = []
    gy = []
    for ix in range(6):
        for jy in range(5):
            gx.append((ix + 0.5) / 6.0)
            gy.append((jy + 0.5) / 5.0)
    gx = np.array(gx)
    gy = np.array(gy)

    restart = 0
    while (time.time() - t0) < PHASE_A_END:
        restart += 1
        rmode = restart % 4
        if rmode == 0:
            u0 = np.random.rand(N)
            v0 = np.random.rand(N)
        elif rmode == 1:
            perm = np.random.permutation(len(gx))[:N]
            u0 = gx[perm] + (np.random.rand(N) - 0.5) * 0.08
            v0 = gy[perm] + (np.random.rand(N) - 0.5) * 0.08
            u0 = np.clip(u0, 0.0, 1.0)
            v0 = np.clip(v0, 0.0, 1.0)
        elif rmode == 2:
            # jittered 5x5 + extra
            u0 = np.random.rand(N)
            v0 = np.random.rand(N)
            # force roughly uniform
            for i in range(N):
                u0[i] = ((i % 5) + 0.5) / 5.0 + (np.random.rand() - 0.5) * 0.1
                v0[i] = ((i // 5) + 0.5) / 6.0 + (np.random.rand() - 0.5) * 0.1
            # last one random
            u0 = np.clip(u0, 0.0, 1.0)
            v0 = np.clip(v0, 0.0, 1.0)
        else:
            if best_x is not None and np.random.rand() < 0.5:
                # perturb best
                u0 = np.clip(best_x + 0.05 * np.random.randn(N), 0.0, 1.0)
                v0 = np.clip(best_y + 0.05 * np.random.randn(N), 0.0, 1.0)
            else:
                u0 = np.random.rand(N)
                v0 = np.random.rand(N)
        if restart % 2 == 0:
            r0 = 0.06 + 0.05 * np.random.rand(N)
        else:
            r0 = 0.04 + 0.10 * np.random.rand(N)
        r0 = np.clip(r0, 0.015, 0.28)
        z = np.concatenate([u0, v0, r0])

        for W in (30.0, 300.0, 3000.0, 30000.0):
            if (time.time() - t0) >= PHASE_A_END:
                break
            def fun(zz, W=W):
                u = zz[:N]; vv = zz[N:2 * N]; rr = zz[2 * N:]
                x = rr * (1.0 - 2.0 * u) + u
                y = rr * (1.0 - 2.0 * vv) + vv
                Dx = x[:, None] - x[None, :]
                Dy = y[:, None] - y[None, :]
                D = np.sqrt(Dx * Dx + Dy * Dy + 1e-16)
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
                               options={'maxiter': 180, 'ftol': 1e-9, 'gtol': 1e-5})
                z = res.x
            except Exception:
                break
        u = z[:N]; vv = z[N:2 * N]; rr = z[2 * N:]
        x = rr * (1.0 - 2.0 * u) + u
        y = rr * (1.0 - 2.0 * vv) + vv
        x = np.clip(x, 0.006, 0.994)
        y = np.clip(y, 0.006, 0.994)
        r_opt, s = lp_opt(x, y)
        if r_opt is not None and s > best_sum:
            best_sum = s
            best_x = x.copy(); best_y = y.copy(); best_r = r_opt.copy()
        # also try raw shrunk candidate without LP (in case LP infeasible due to LB)
        # skip

    if best_x is None:
        kk = int(np.ceil(np.sqrt(N)))
        rr = 0.5 / kk
        pts = []
        for i in range(kk):
            for j in range(kk):
                if len(pts) < N:
                    pts.append([(2 * i + 1) * rr, (2 * j + 1) * rr, rr * 0.999999])
        return np.array(pts, dtype=np.float64)

    # Phase B: hill climbing on LP value, single + global moves
    sigmas = [0.025, 0.015, 0.01, 0.007, 0.005, 0.003, 0.002, 0.0012, 0.0007]
    # repeat cycles until budget
    n_cycle = 0
    while (time.time() - t0) < (BUDGET - 6.0):
        n_cycle += 1
        improved_cycle = False
        for sigma in sigmas:
            if (time.time() - t0) >= (BUDGET - 6.0):
                break
            # iters scaled by sigma: larger sigma fewer iters
            iters = 180 if sigma >= 0.005 else 140
            for _ in range(iters):
                if (time.time() - t0) >= (BUDGET - 6.0):
                    break
                rnd = np.random.rand()
                if rnd < 0.45:
                    # global move
                    xn = best_x + sigma * np.random.randn(N)
                    yn = best_y + sigma * np.random.randn(N)
                elif rnd < 0.80:
                    # single circle move
                    xn = best_x.copy(); yn = best_y.copy()
                    idx = np.random.randint(N)
                    xn[idx] += sigma * 1.6 * np.random.randn()
                    yn[idx] += sigma * 1.6 * np.random.randn()
                else:
                    # 2-3 circles move
                    xn = best_x.copy(); yn = best_y.copy()
                    kk2 = 2 if np.random.rand() < 0.7 else 3
                    idxs = np.random.choice(N, size=kk2, replace=False)
                    xn[idxs] += sigma * 1.4 * np.random.randn(kk2)
                    yn[idxs] += sigma * 1.4 * np.random.randn(kk2)
                xn = np.clip(xn, 0.02, 0.98)
                yn = np.clip(yn, 0.02, 0.98)
                r_opt, s = lp_opt(xn, yn)
                if r_opt is not None and s > best_sum + 1e-10:
                    best_sum = s
                    best_x = xn; best_y = yn; best_r = r_opt
                    improved_cycle = True
        # occasional big shake if stuck (basin hopping)
        if not improved_cycle:
            # shake with medium sigma and accept even if slightly worse to escape? here only accept if not too bad
            # try random restarts around best with larger noise, keep best
            for _ in range(60):
                if (time.time() - t0) >= (BUDGET - 6.0):
                    break
                xn = np.clip(best_x + 0.04 * np.random.randn(N), 0.02, 0.98)
                yn = np.clip(best_y + 0.04 * np.random.randn(N), 0.02, 0.98)
                r_opt, s = lp_opt(xn, yn)
                if r_opt is not None and s > best_sum + 1e-10:
                    best_sum = s
                    best_x = xn; best_y = yn; best_r = r_opt
                    improved_cycle = True
                    break
        if n_cycle > 30:
            break

    # Phase C: joint fine tune from best with high penalty
    if (time.time() - t0) < (BUDGET - 4.0):
        try:
            # encode best to u,v,r
            rb = best_r.copy()
            # ensure rb <0.5
            rb = np.minimum(rb, 0.3)
            # compute u = (x - r)/(1-2r)
            denomx = 1.0 - 2.0 * rb
            denomx = np.maximum(denomx, 0.2)
            ub = (best_x - rb) / denomx
            vb = (best_y - rb) / denomx
            ub = np.clip(ub, 0.001, 0.999)
            vb = np.clip(vb, 0.001, 0.999)
            z = np.concatenate([ub, vb, rb])
            for W in (5000.0, 50000.0):
                if (time.time() - t0) >= (BUDGET - 3.0):
                    break
                def fun2(zz, W=W):
                    u = zz[:N]; vv = zz[N:2 * N]; rr = zz[2 * N:]
                    x = rr * (1.0 - 2.0 * u) + u
                    y = rr * (1.0 - 2.0 * vv) + vv
                    Dx = x[:, None] - x[None, :]
                    Dy = y[:, None] - y[None, :]
                    D = np.sqrt(Dx * Dx + Dy * Dy + 1e-16)
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
                res = minimize(fun2, z, method='L-BFGS-B', jac=True, bounds=bounds_box,
                               options={'maxiter': 120, 'ftol': 1e-10, 'gtol': 1e-6})
                z = res.x
            u = z[:N]; vv = z[N:2 * N]; rr = z[2 * N:]
            x = rr * (1.0 - 2.0 * u) + u
            y = rr * (1.0 - 2.0 * vv) + vv
            x = np.clip(x, 0.006, 0.994)
            y = np.clip(y, 0.006, 0.994)
            r_opt, s = lp_opt(x, y)
            if r_opt is not None and s > best_sum:
                best_sum = s
                best_x = x; best_y = y; best_r = r_opt
        except Exception:
            pass

    x = best_x; y = best_y; r = best_r
    out = np.column_stack([x, y, r])
    # final strict safety
    wall = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    Dx = x[:, None] - x[None, :]; Dy = y[:, None] - y[None, :]
    D = np.sqrt(Dx * Dx + Dy * Dy)
    fw = np.min(wall / r)
    fp = 1e100
    for i in range(N):
        for j in range(i + 1, N):
            fp = min(fp, D[i, j] / (r[i] + r[j]))
    f = min(fw, fp)
    if f < 1.0:
        r = r * f * (1.0 - 1e-9)
        out = np.column_stack([x, y, r])
    else:
        # tiny shrink for 1e-12 guard
        r = r * (1.0 - 1e-12)
        out = np.column_stack([x, y, r])
    return out
