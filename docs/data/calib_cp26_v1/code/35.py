import numpy as np
import time

def solve():
    import numpy as np
    import time as _tm
    from scipy.optimize import minimize, linprog
    N = 26
    t0 = _tm.time()
    DEADLINE = 124.0

    rng = np.random.default_rng(7)

    # pair matrix for LP: r_i + r_j <= D_ij, 0<=r_i<=wall_i via bounds
    iu, ju = np.triu_indices(N, 1)
    m = len(iu)
    A_ub0 = np.zeros((m, N), dtype=float)
    ar = np.arange(m)
    A_ub0[ar, iu] = 1.0
    A_ub0[ar, ju] = 1.0
    c_obj = -np.ones(N)

    def lp_opt(xc, yc):
        # walls as bounds, always feasible with r>=0
        b = np.minimum(np.minimum(xc, 1.0 - xc), np.minimum(yc, 1.0 - yc))
        # centres must stay inside (0,1); clip negative to 0
        b = np.maximum(b, 0.0)
        # quick prune not needed
        dx = xc[iu] - xc[ju]
        dy = yc[iu] - yc[ju]
        dvec = np.sqrt(dx * dx + dy * dy)
        # avoid exact duplicate issues: keep as is (0 allowed)
        bounds = [(0.0, float(bi)) for bi in b]
        try:
            res = linprog(c_obj, A_ub=A_ub0, b_ub=dvec, bounds=bounds, method='highs')
        except Exception:
            return None, -1.0
        if not res.success or res.x is None:
            return None, -1.0
        r = np.asarray(res.x, dtype=float).flatten()
        r = np.maximum(r, 0.0)
        s = float(np.sum(r))
        return r, s

    best_sum = -1.0
    best_x = None
    best_y = None
    best_r = None

    def consider(xc, yc):
        global_best = [best_sum]
        # use closure via nonlocal
        return None

    # ---------- Phase 1: joint penalized L-BFGS with wall-free parametrization ----------
    # x = r*(1-2u)+u, y = r*(1-2v)+v, u,v in [0,1] => wall auto-satisfied
    bounds_box = [(0.0, 1.0)] * (2 * N) + [(0.008, 0.26)] * N

    # diverse grid templates
    gx6 = []; gy6 = []
    for ix in range(6):
        for jy in range(5):
            gx6.append((ix + 0.5) / 6.0); gy6.append((jy + 0.5) / 5.0)
    gx6 = np.array(gx6); gy6 = np.array(gy6)

    restart = 0
    # time for phase1
    T_PHASE1 = 52.0
    while (_tm.time() - t0) < T_PHASE1 and restart < 80:
        rs = np.random.RandomState(12345 + restart * 10007)
        mode = restart % 5
        if mode == 0 and restart == 0:
            # 5x5 + extra
            pts = []
            for i in range(5):
                for j in range(5):
                    pts.append([0.1 + 0.2 * i, 0.1 + 0.2 * j])
            pts = np.array(pts)
            extra = np.array([[0.5 + rs.randn()*0.05, 0.5 + rs.randn()*0.05]])
            xy = np.vstack([pts, extra])
            x0 = xy[:, 0] + rs.randn(N) * 0.003
            y0 = xy[:, 1] + rs.randn(N) * 0.003
            r0 = np.full(N, 0.09) + rs.randn(N) * 0.005
        elif mode == 1:
            perm = rs.permutation(len(gx6))[:N]
            x0 = gx6[perm] + (rs.rand(N) - 0.5) * 0.08
            y0 = gy6[perm] + (rs.rand(N) - 0.5) * 0.08
            r0 = 0.06 + 0.05 * rs.rand(N)
        elif mode == 2:
            # hex jitter
            x0 = rs.uniform(0.06, 0.94, size=N)
            y0 = rs.uniform(0.06, 0.94, size=N)
            # log-uniform radii to force hierarchy (some large, some tiny)
            r0 = np.exp(rs.uniform(np.log(0.02), np.log(0.16), size=N))
        elif mode == 3:
            x0 = rs.uniform(0.05, 0.95, size=N)
            y0 = rs.uniform(0.05, 0.95, size=N)
            r0 = 0.07 + 0.04 * rs.rand(N)
        else:
            x0 = rs.uniform(0.08, 0.92, size=N)
            y0 = rs.uniform(0.08, 0.92, size=N)
            r0 = rs.uniform(0.03, 0.13, size=N)
        x0 = np.clip(x0, 0.05, 0.95)
        y0 = np.clip(y0, 0.05, 0.95)
        r0 = np.clip(r0, 0.008, 0.25)
        # convert to u,v
        denom = 1.0 - 2.0 * r0
        denom = np.maximum(denom, 0.2)
        u0 = (x0 - r0) / denom
        v0 = (y0 - r0) / denom
        u0 = np.clip(u0, 0.0, 1.0); v0 = np.clip(v0, 0.0, 1.0)
        z = np.concatenate([u0, v0, r0])

        for W in (30.0, 300.0, 3000.0, 30000.0):
            if (_tm.time() - t0) >= T_PHASE1:
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
                               options={'maxiter': 120, 'ftol': 1e-9, 'gtol': 1e-5})
                z = res.x
            except Exception:
                break
        u = z[:N]; vv = z[N:2*N]; rr = z[2*N:]
        x = rr * (1.0 - 2.0 * u) + u
        y = rr * (1.0 - 2.0 * vv) + vv
        x = np.clip(x, 0.005, 0.995); y = np.clip(y, 0.005, 0.995)
        r_opt, s = lp_opt(x, y)
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
        # safety shrink
        x = arr[:,0]; y = arr[:,1]; r = arr[:,2]*0.999
        return np.column_stack([x,y,r])

    # ---------- Phase 2: topology-changing centre-only basin hopping with LP ----------
    # intensive: global Gaussian + single-circle uniform relocation + multi-try void fill
    sigmas = [0.03, 0.02, 0.012, 0.007, 0.0035, 0.0015]
    # outer loops until deadline
    # precompute order bias
    while (_tm.time() - t0) < (DEADLINE - 6.0):
        improved_outer = False
        for sigma in sigmas:
            if (_tm.time() - t0) >= (DEADLINE - 6.0):
                break
            iters = 160 if sigma > 0.005 else 220
            for _ in range(iters):
                if (_tm.time() - t0) >= (DEADLINE - 6.0):
                    break
                p = rng.random()
                if p < 0.55:
                    xn = best_x + sigma * rng.standard_normal(N)
                    yn = best_y + sigma * rng.standard_normal(N)
                    xn = np.clip(xn, 0.008, 0.992)
                    yn = np.clip(yn, 0.008, 0.992)
                    r_opt, s = lp_opt(xn, yn)
                    if r_opt is not None and s > best_sum + 1e-10:
                        best_sum = s; best_x = xn; best_y = yn; best_r = r_opt
                        improved_outer = True
                elif p < 0.80:
                    # single circle uniform jump (biased to small radii)
                    if rng.random() < 0.6:
                        order = np.argsort(best_r)
                        kk = int(rng.choice(order[:7]))
                    else:
                        kk = int(rng.integers(N))
                    xn = best_x.copy(); yn = best_y.copy()
                    xn[kk] = float(rng.uniform(0.03, 0.97))
                    yn[kk] = float(rng.uniform(0.03, 0.97))
                    r_opt, s = lp_opt(xn, yn)
                    if r_opt is not None and s > best_sum + 1e-10:
                        best_sum = s; best_x = xn; best_y = yn; best_r = r_opt
                        improved_outer = True
                else:
                    # void multi-try: pick victim, try K positions, keep best
                    if rng.random() < 0.6:
                        order = np.argsort(best_r)
                        kk = int(rng.choice(order[:8]))
                    else:
                        kk = int(rng.integers(N))
                    # try 10 candidates
                    for _t in range(10):
                        if (_tm.time() - t0) >= (DEADLINE - 6.0):
                            break
                        xn = best_x.copy(); yn = best_y.copy()
                        xn[kk] = float(rng.uniform(0.02, 0.98))
                        yn[kk] = float(rng.uniform(0.02, 0.98))
                        r_opt, s = lp_opt(xn, yn)
                        if r_opt is not None and s > best_sum + 1e-10:
                            best_sum = s; best_x = xn; best_y = yn; best_r = r_opt
                            improved_outer = True
                            break
        if not improved_outer:
            # diversification: shake worst 3 circles randomly without requiring improvement once
            if (_tm.time() - t0) >= (DEADLINE - 6.0):
                break
            # perturb best slightly to escape plateau and continue
            xn = best_x + 0.008 * rng.standard_normal(N)
            yn = best_y + 0.008 * rng.standard_normal(N)
            xn = np.clip(xn, 0.008, 0.992); yn = np.clip(yn, 0.008, 0.992)
            r_opt, s = lp_opt(xn, yn)
            if r_opt is not None and s > best_sum - 0.005:
                # accept sideways move to explore (but keep best separately? we move best only if not too bad)
                # keep elite: only move if within tolerance to diversify
                if s > best_sum + 1e-10:
                    best_sum = s; best_x = xn; best_y = yn; best_r = r_opt
                else:
                    # random restart from best + larger shake as new base for next loop?
                    # do nothing, continue; to avoid stagnation, force a single-circle jump evaluation
                    pass

    # ---------- final strict feasibility repair ----------
    x = best_x.copy(); y = best_y.copy(); r = best_r.copy()
    # bump zeros to tiny positive
    r = np.maximum(r, 1e-7)
    # ensure centres strictly inside
    x = np.clip(x, 1e-9, 1.0 - 1e-9); y = np.clip(y, 1e-9, 1.0 - 1e-9)
    wall = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    Dx = x[:, None] - x[None, :]
    Dy = y[:, None] - y[None, :]
    D = np.sqrt(Dx*Dx + Dy*Dy)
    # compute max feasible scale
    alpha = 1.0
    # wall ratios
    for i in range(N):
        if r[i] > 0:
            a = wall[i] / r[i]
            if a < alpha:
                alpha = a
    for i in range(N):
        for j in range(i+1, N):
            s_ = r[i] + r[j]
            if s_ > 1e-18:
                a = D[i, j] / s_
                if a < alpha:
                    alpha = a
    # strict margin
    alpha = alpha * (1.0 - 1e-7) - 1e-12
    if not np.isfinite(alpha) or alpha <= 0:
        alpha = 0.9
    if alpha < 1.0:
        r = r * alpha
    elif alpha > 1.0:
        # already strictly feasible, still shrink tiny bit for 1e-12 tolerance
        r = r * (1.0 - 1e-9)
    # final clip to guarantee r>0
    r = np.maximum(r, 1e-9)
    out = np.column_stack([x, y, r])
    return out
