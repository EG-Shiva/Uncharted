import numpy as np
import time

def solve():
    import numpy as np
    import time
    from scipy.optimize import minimize, linprog
    N = 26
    t0 = time.time()
    BUDGET = 126.0
    PHASE_A = 72.0
    rng = np.random.RandomState(7)

    LB = 0.0015
    EPS = 2e-9

    def lp_opt(x, y):
        wall = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        if np.min(wall) < LB + EPS:
            return None, -1.0
        Dx = x[:, None] - x[None, :]
        Dy = y[:, None] - y[None, :]
        D = np.sqrt(Dx * Dx + Dy * Dy)
        # quick infeasibility
        # collect pairs that matter
        rows = []
        rhs = []
        for i in range(N):
            for j in range(i + 1, N):
                dij = D[i, j]
                # redundant if dij >= wall[i]+wall[j]
                if dij >= wall[i] + wall[j] - 1e-12:
                    continue
                lim = dij - EPS
                if lim < 2 * LB - 1e-12:
                    return None, -1.0
                rows.append((i, j, lim))
        m = len(rows)
        if m == 0:
            r = wall - EPS
            r = np.maximum(r, LB)
            return r * (1.0 - 1e-9), float(np.sum(r))
        A_ub = np.zeros((m, N), dtype=np.float64)
        b_ub = np.zeros(m, dtype=np.float64)
        for k, (i, j, lim) in enumerate(rows):
            A_ub[k, i] = 1.0
            A_ub[k, j] = 1.0
            b_ub[k] = lim
        c = -np.ones(N, dtype=np.float64)
        bounds = [(LB, float(wall[i] - EPS)) for i in range(N)]
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                          options={'primal_feasibility_tolerance': 1e-9})
        except Exception:
            return None, -1.0
        if not res.success:
            return None, -1.0
        r = res.x * (1.0 - 5e-9)
        return r, float(np.sum(r))

    best_sum = -1.0
    best_x = None
    best_y = None
    best_r = None

    def consider(x, y):
        global_best = None
        nonlocal best_sum, best_x, best_y, best_r
        r, s = lp_opt(x, y)
        if r is not None and s > best_sum + 1e-12:
            best_sum = s
            best_x = x.copy()
            best_y = y.copy()
            best_r = r.copy()
            return True
        return False

    # ---------- Phase A: joint penalty ----------
    bounds_box = [(0.0, 1.0)] * (2 * N) + [(0.008, 0.28)] * N
    # grid seeds
    gx = []; gy = []
    for ix in range(6):
        for jy in range(5):
            gx.append((ix + 0.5) / 6.0); gy.append((jy + 0.5) / 5.0)
    gx = np.array(gx); gy = np.array(gy)

    restart = 0
    while (time.time() - t0) < PHASE_A and restart < 200:
        if restart % 3 == 0:
            u0 = rng.rand(N); v0 = rng.rand(N)
        elif restart % 3 == 1:
            perm = rng.permutation(len(gx))[:N]
            u0 = gx[perm] + (rng.rand(N) - 0.5) * 0.08
            v0 = gy[perm] + (rng.rand(N) - 0.5) * 0.08
            u0 = np.clip(u0, 0.0, 1.0); v0 = np.clip(v0, 0.0, 1.0)
        else:
            # hex jitter
            u0 = rng.rand(N); v0 = rng.rand(N)
            # force spread: sort + jitter
            u0 = np.sort(u0) + (rng.rand(N) - 0.5) * 0.03
            v0 = np.sort(v0) + (rng.rand(N) - 0.5) * 0.03
            u0 = np.clip(u0, 0.0, 1.0); v0 = np.clip(v0, 0.0, 1.0)
        if restart % 2 == 0:
            r0 = 0.06 + 0.05 * rng.rand(N)
        else:
            r0 = 0.04 + 0.09 * rng.rand(N)
        r0 = np.clip(r0, 0.01, 0.27)
        z = np.concatenate([u0, v0, r0])
        for W in (30.0, 300.0, 3000.0, 30000.0):
            if (time.time() - t0) >= PHASE_A:
                break
            def fun(zz, W=W):
                u = zz[:N]; vv = zz[N:2*N]; rr = zz[2*N:]
                om2r = 1.0 - 2.0 * rr
                x = rr + om2r * u
                y = rr + om2r * vv
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
                dFdu = W * Gx * om2r
                dFdv = W * Gy * om2r
                dFdr = -np.ones(N) + W * (2.0*sumEp + Gx*(1.0-2.0*u) + Gy*(1.0-2.0*vv))
                g = np.concatenate([dFdu, dFdv, dFdr])
                return F, g
            try:
                res = minimize(fun, z, method='L-BFGS-B', jac=True, bounds=bounds_box,
                               options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-5})
                z = res.x
            except Exception:
                break
        u = z[:N]; vv = z[N:2*N]; rr = z[2*N:]
        x = rr + (1.0 - 2.0*rr) * u
        y = rr + (1.0 - 2.0*rr) * vv
        x = np.clip(x, 0.005, 0.995); y = np.clip(y, 0.005, 0.995)
        consider(x, y)
        restart += 1

    if best_x is None:
        k = int(np.ceil(np.sqrt(N)))
        r = 0.5 / k
        pts = []
        for i in range(k):
            for j in range(k):
                if len(pts) < N:
                    pts.append([(2*i+1)*r, (2*j+1)*r, r*0.999999])
        return np.array(pts, dtype=np.float64)

    # ---------- Phase B: targeted LNS ----------
    # order small circles first for relocation
    it = 0
    sigmas = [0.025, 0.012, 0.006, 0.0025, 0.001]
    # precompute
    while (time.time() - t0) < BUDGET - 4.0:
        # alternate move types
        mode = it % 10
        if mode < 6:
            # single circle relocation (prefer small)
            # pick index: mostly smallest
            if best_r is not None:
                order = np.argsort(best_r)
                # choose among 8 smallest randomly
                cand_pool = order[:8]
                idx = cand_pool[rng.randint(len(cand_pool))]
            else:
                idx = rng.randint(N)
            # propose new position: uniform + void-seeking
            # generate few candidates, pick most isolated
            ncand = 6
            px = 0.03 + 0.94 * rng.rand(ncand)
            py = 0.03 + 0.94 * rng.rand(ncand)
            # score by min dist to others
            best_c = -1
            best_px = px[0]; best_py = py[0]
            for c in range(ncand):
                dx = best_x - px[c]; dy = best_y - py[c]
                dx[idx] = 1e9; dy[idx] = 1e9
                d = np.min(np.sqrt(dx*dx + dy*dy))
                w = min(px[c], 1-px[c], py[c], 1-py[c])
                score = min(d, 2*w)
                if score > best_c:
                    best_c = score; best_px = px[c]; best_py = py[c]
            # with 50% use best void, else pure random
            if rng.rand() < 0.5:
                nx, ny = px[rng.randint(ncand)], py[rng.randint(ncand)]
            else:
                nx, ny = best_px, best_py
            xn = best_x.copy(); yn = best_y.copy()
            xn[idx] = np.clip(nx, 0.02, 0.98); yn[idx] = np.clip(ny, 0.02, 0.98)
            consider(xn, yn)
        elif mode < 9:
            sigma = sigmas[(it // 10) % len(sigmas)]
            # perturb all, but smaller perturb for large circles
            xn = best_x + sigma * rng.randn(N)
            yn = best_y + sigma * rng.randn(N)
            xn = np.clip(xn, 0.02, 0.98); yn = np.clip(yn, 0.02, 0.98)
            consider(xn, yn)
        else:
            # joint refine from best
            if (time.time() - t0) > BUDGET - 15.0:
                it += 1
                continue
            # encode best
            rr = np.clip(best_r * 1.02, 0.008, 0.28)
            # avoid div by zero when rr~0.5
            denomx = (1.0 - 2.0*rr)
            # if denom small, clip
            u = (best_x - rr) / denomx
            vv = (best_y - rr) / denomx
            u = np.clip(u, 0.0, 1.0); vv = np.clip(vv, 0.0, 1.0)
            z = np.concatenate([u, vv, rr])
            W = 5000.0
            def fun2(zz):
                u_ = zz[:N]; vv_ = zz[N:2*N]; rr_ = zz[2*N:]
                om2r = 1.0 - 2.0*rr_
                x_ = rr_ + om2r*u_
                y_ = rr_ + om2r*vv_
                Dx = x_[:, None]-x_[None,:]; Dy = y_[:, None]-y_[None,:]
                D = np.sqrt(Dx*Dx+Dy*Dy+1e-16)
                np.fill_diagonal(D, 1.0)
                E = rr_[:, None]+rr_[None,:]-D
                np.fill_diagonal(E, 0.0)
                Ep = np.maximum(E, 0.0)
                Pen = 0.5*np.sum(Ep*Ep)
                F = -np.sum(rr_)+W*Pen
                InvD = 1.0/D; np.fill_diagonal(InvD, 0.0)
                Wmat = Ep*InvD
                Gx = -2.0*np.sum(Wmat*Dx, axis=1)
                Gy = -2.0*np.sum(Wmat*Dy, axis=1)
                sumEp = np.sum(Ep, axis=1)
                dFdu = W*Gx*om2r; dFdv = W*Gy*om2r
                dFdr = -np.ones(N)+W*(2.0*sumEp+Gx*(1.0-2.0*u_)+Gy*(1.0-2.0*vv_))
                return F, np.concatenate([dFdu, dFdv, dFdr])
            try:
                res = minimize(fun2, z, method='L-BFGS-B', jac=True, bounds=bounds_box,
                               options={'maxiter': 120, 'ftol': 1e-9, 'gtol': 1e-5})
                z = res.x
            except Exception:
                pass
            u_ = z[:N]; vv_ = z[N:2*N]; rr_ = z[2*N:]
            x_ = rr_ + (1.0-2.0*rr_)*u_; y_ = rr_ + (1.0-2.0*rr_)*vv_
            x_ = np.clip(x_, 0.005, 0.995); y_ = np.clip(y_, 0.005, 0.995)
            consider(x_, y_)
        it += 1
        # occasional early exit check
        if it % 200 == 0 and (time.time() - t0) > BUDGET - 10.0:
            break

    out = np.column_stack([best_x, best_y, best_r])
    x = out[:, 0]; y = out[:, 1]; r = out[:, 2]
    wall = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
    Dx = x[:, None]-x[None,:]; Dy = y[:, None]-y[None,:]
    D = np.sqrt(Dx*Dx+Dy*Dy)
    fw = np.min(wall / np.maximum(r, 1e-18))
    fp = 1e100
    for i in range(N):
        for j in range(i+1, N):
            fp = min(fp, D[i, j] / (r[i]+r[j]))
    f = min(fw, fp)
    if f < 1.0:
        r = r * f * (1.0 - 1e-9)
        out = np.column_stack([x, y, r])
    else:
        # tiny safety shrink to guarantee strict 1e-12
        r = r * (1.0 - 1e-9)
        out = np.column_stack([x, y, r])
    return out
