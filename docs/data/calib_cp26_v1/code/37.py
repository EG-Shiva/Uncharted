import numpy as np
import time

def solve():
    import numpy as np
    import time
    from scipy.optimize import minimize, linprog
    N = 26
    t0 = time.time()
    BUDGET = 126.0
    PHASE_A_END = 48.0
    rng = np.random.RandomState(7)
    LB = 0.001
    EPS = 1e-9

    def lp_opt(x, y):
        wall = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        if np.min(wall) < LB - 1e-12:
            return None, -1.0
        Dx = x[:, None] - x[None, :]
        Dy = y[:, None] - y[None, :]
        D = np.sqrt(Dx * Dx + Dy * Dy)
        rows = []
        for i in range(N):
            for j in range(i + 1, N):
                dij = D[i, j]
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
            r = r * (1.0 - 1e-9)
            return r, float(np.sum(r))
        A_ub = np.zeros((m, N), dtype=np.float64)
        b_ub = np.zeros(m, dtype=np.float64)
        for k, (i, j, lim) in enumerate(rows):
            A_ub[k, i] = 1.0
            A_ub[k, j] = 1.0
            b_ub[k] = lim
        c = -np.ones(N, dtype=np.float64)
        bounds = [(LB, float(wall[i] - EPS)) for i in range(N)]
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
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
        nonlocal best_sum, best_x, best_y, best_r
        r, s = lp_opt(x, y)
        if r is not None and s > best_sum + 1e-12:
            best_sum = s
            best_x = x.copy()
            best_y = y.copy()
            best_r = r.copy()
            return True
        return False

    bounds_box = [(0.0, 1.0)] * (2 * N) + [(0.008, 0.28)] * N
    gx = []
    gy = []
    for ix in range(6):
        for jy in range(5):
            gx.append((ix + 0.5) / 6.0)
            gy.append((jy + 0.5) / 5.0)
    gx = np.array(gx)
    gy = np.array(gy)

    # Phase A: joint penalty restarts
    restart = 0
    while (time.time() - t0) < PHASE_A_END and restart < 60:
        if restart % 3 == 0:
            u0 = rng.rand(N); v0 = rng.rand(N)
        elif restart % 3 == 1:
            perm = rng.permutation(len(gx))[:N]
            u0 = gx[perm] + (rng.rand(N) - 0.5) * 0.08
            v0 = gy[perm] + (rng.rand(N) - 0.5) * 0.08
            u0 = np.clip(u0, 0.0, 1.0); v0 = np.clip(v0, 0.0, 1.0)
        else:
            u0 = np.sort(rng.rand(N)) + (rng.rand(N) - 0.5) * 0.03
            v0 = np.sort(rng.rand(N)) + (rng.rand(N) - 0.5) * 0.03
            u0 = np.clip(u0, 0.0, 1.0); v0 = np.clip(v0, 0.0, 1.0)
        if restart % 2 == 0:
            r0 = 0.06 + 0.05 * rng.rand(N)
        else:
            r0 = 0.04 + 0.09 * rng.rand(N)
        r0 = np.clip(r0, 0.01, 0.27)
        z = np.concatenate([u0, v0, r0])
        for W in (30.0, 300.0, 3000.0, 20000.0):
            if (time.time() - t0) >= PHASE_A_END:
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
                return F, np.concatenate([dFdu, dFdv, dFdr])
            try:
                res = minimize(fun, z, method='L-BFGS-B', jac=True, bounds=bounds_box,
                               options={'maxiter': 150, 'ftol': 1e-9, 'gtol': 1e-5})
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

    # helpers for surgery
    GN = 201
    _lin = np.linspace(0.0, 1.0, GN)
    _PX, _PY = np.meshgrid(_lin, _lin)
    PXf = _PX.ravel()
    PYf = _PY.ravel()
    WALLg = np.minimum(np.minimum(PXf, 1.0-PXf), np.minimum(PYf, 1.0-PYf))

    def find_k_voids(xc, yc, rc, k):
        # vectorized clearance
        # M x N distances
        dx = PXf[:, None] - xc[None, :]
        dy = PYf[:, None] - yc[None, :]
        dist = np.sqrt(dx*dx + dy*dy) - rc[None, :]
        cpair = np.min(dist, axis=1)
        clear = np.minimum(WALLg, cpair)
        voids = []
        # greedy with suppression
        cur_clear = clear.copy()
        # to avoid picking border zeros, require interior? just pick max
        synth_x = []
        synth_y = []
        synth_r = []
        for _ in range(k):
            idx = int(np.argmax(cur_clear))
            vx = float(PXf[idx]); vy = float(PYf[idx]); vc = float(cur_clear[idx])
            voids.append((vx, vy, vc))
            # suppress neighbourhood: update cur_clear = min(cur_clear, dist_to_void - vc)
            ddx = PXf - vx
            ddy = PYf - vy
            d = np.sqrt(ddx*ddx + ddy*ddy) - max(vc, 1e-9)
            cur_clear = np.minimum(cur_clear, d)
        return voids

    def contact_counts(xc, yc, rc, tol=0.004):
        wall = np.minimum(np.minimum(xc, 1.0-xc), np.minimum(yc, 1.0-yc))
        cnt = np.zeros(N, dtype=int)
        for i in range(N):
            c = 0
            if wall[i] - rc[i] < tol:
                c += 1
            Dx = xc - xc[i]; Dy = yc - yc[i]
            D = np.sqrt(Dx*Dx + Dy*Dy)
            for j in range(N):
                if j == i:
                    continue
                if D[j] - rc[i] - rc[j] < tol:
                    c += 1
            cnt[i] = c
        return cnt

    def polish_centres(x0, y0, r0, maxiter=60, W=8000.0):
        rr0 = np.clip(r0 * 1.01, 0.008, 0.28)
        den = (1.0 - 2.0 * rr0)
        den = np.where(np.abs(den) < 0.2, 0.2*np.sign(den+1e-12), den)
        u0 = (x0 - rr0) / den
        v0 = (y0 - rr0) / den
        u0 = np.clip(u0, 0.0, 1.0); v0 = np.clip(v0, 0.0, 1.0)
        z0 = np.concatenate([u0, v0, rr0])
        def f2(zz):
            u_ = zz[:N]; vv_ = zz[N:2*N]; rr_ = zz[2*N:]
            om = 1.0 - 2.0*rr_
            x_ = rr_ + om*u_
            y_ = rr_ + om*vv_
            Dx = x_[:, None]-x_[None, :]; Dy = y_[:, None]-y_[None, :]
            D = np.sqrt(Dx*Dx+Dy*Dy+1e-16)
            np.fill_diagonal(D, 1.0)
            E = rr_[:, None]+rr_[None, :]-D
            np.fill_diagonal(E, 0.0)
            Ep = np.maximum(E, 0.0)
            Pen = 0.5*np.sum(Ep*Ep)
            F = -np.sum(rr_)+W*Pen
            InvD = 1.0/D; np.fill_diagonal(InvD, 0.0)
            Wm = Ep*InvD
            Gx = -2.0*np.sum(Wm*Dx, axis=1)
            Gy = -2.0*np.sum(Wm*Dy, axis=1)
            sEp = np.sum(Ep, axis=1)
            dFdu = W*Gx*om; dFdv = W*Gy*om
            dFdr = -np.ones(N)+W*(2.0*sEp+Gx*(1.0-2.0*u_)+Gy*(1.0-2.0*vv_))
            return F, np.concatenate([dFdu, dFdv, dFdr])
        try:
            res = minimize(f2, z0, method='L-BFGS-B', jac=True, bounds=bounds_box,
                           options={'maxiter': maxiter, 'ftol': 1e-9, 'gtol': 1e-5})
            z = res.x
        except Exception:
            return x0, y0
        u_ = z[:N]; vv_ = z[N:2*N]; rr_ = z[2*N:]
        x_ = rr_ + (1.0-2.0*rr_)*u_; y_ = rr_ + (1.0-2.0*rr_)*vv_
        return np.clip(x_, 0.005, 0.995), np.clip(y_, 0.005, 0.995)

    # Phase B: void surgery + basin hopping SA
    cur_x = best_x.copy(); cur_y = best_y.copy(); cur_r = best_r.copy()
    cur_sum = best_sum
    it = 0
    last_imp = 0
    T0 = 0.005
    Tend = 0.00015
    sigmas = [0.025, 0.012, 0.006, 0.0025]
    # decay constant: estimate ~1500 iters
    while (time.time() - t0) < BUDGET - 5.0:
        frac = it / 1500.0
        T = T0 * (Tend / T0) ** min(frac, 1.0)
        # elite restore
        if it - last_imp > 250:
            cur_x = best_x.copy(); cur_y = best_y.copy(); cur_r = best_r.copy(); cur_sum = best_sum
            last_imp = it
            # small kick to avoid loop
            cur_x = np.clip(cur_x + 0.002*rng.randn(N), 0.01, 0.99)
            cur_y = np.clip(cur_y + 0.002*rng.randn(N), 0.01, 0.99)
            rr, ss = lp_opt(cur_x, cur_y)
            if rr is not None:
                cur_r, cur_sum = rr, ss
                if ss > best_sum + 1e-12:
                    best_sum, best_x, best_y, best_r = ss, cur_x.copy(), cur_y.copy(), rr.copy()
            it += 1
            continue
        u = rng.rand()
        xn = None; yn = None
        do_polish = False
        if u < 0.45:
            # single void relocation
            try:
                voids = find_k_voids(cur_x, cur_y, cur_r, 1)
            except Exception:
                voids = [(rng.rand(), rng.rand(), 0.0)]
            vx, vy, _ = voids[0]
            if rng.rand() < 0.65:
                order = np.argsort(cur_r)
                pool = order[:6]
                idx = pool[rng.randint(len(pool))]
            else:
                try:
                    cc = contact_counts(cur_x, cur_y, cur_r)
                    minc = np.min(cc)
                    cand = np.where(cc <= minc + 1)[0]
                    cand = cand[np.argsort(cur_r[cand])]
                    idx = cand[rng.randint(min(len(cand), 4))]
                except Exception:
                    order = np.argsort(cur_r)
                    idx = order[rng.randint(6)]
            xn = cur_x.copy(); yn = cur_y.copy()
            xn[idx] = np.clip(vx + (rng.rand()-0.5)*0.004, 0.01, 0.99)
            yn[idx] = np.clip(vy + (rng.rand()-0.5)*0.004, 0.01, 0.99)
            do_polish = (rng.rand() < 0.45)
        elif u < 0.70:
            kk = 2 if rng.rand() < 0.6 else 3
            try:
                voids = find_k_voids(cur_x, cur_y, cur_r, kk)
            except Exception:
                voids = [(rng.rand(), rng.rand(), 0.0)]*kk
            order = np.argsort(cur_r)
            if rng.rand() < 0.7:
                idxs = order[:kk]
            else:
                pool = order[:8]
                idxs = rng.choice(pool, kk, replace=False)
            xn = cur_x.copy(); yn = cur_y.copy()
            for t_, idx in enumerate(idxs):
                vx, vy, _ = voids[t_]
                xn[idx] = np.clip(vx + (rng.rand()-0.5)*0.005, 0.01, 0.99)
                yn[idx] = np.clip(vy + (rng.rand()-0.5)*0.005, 0.01, 0.99)
            do_polish = (rng.rand() < 0.5)
        elif u < 0.86:
            sigma = sigmas[rng.randint(len(sigmas))]
            xn = np.clip(cur_x + sigma*rng.randn(N), 0.01, 0.99)
            yn = np.clip(cur_y + sigma*rng.randn(N), 0.01, 0.99)
            do_polish = False
        else:
            # pure polish exploitation from cur/best
            if rng.rand() < 0.5:
                bx, by, br = best_x, best_y, best_r
            else:
                bx, by, br = cur_x, cur_y, cur_r
            if (time.time() - t0) > BUDGET - 18.0:
                it += 1
                continue
            xp, yp = polish_centres(bx, by, br, maxiter=80, W=6000.0)
            rp, sp = lp_opt(xp, yp)
            if rp is not None:
                delta = sp - cur_sum
                if delta >= 0 or rng.rand() < np.exp(delta / max(T, 1e-9)):
                    cur_x, cur_y, cur_r, cur_sum = xp, yp, rp, sp
                if sp > best_sum + 1e-12:
                    best_sum, best_x, best_y, best_r = sp, xp.copy(), yp.copy(), rp.copy()
                    last_imp = it
            it += 1
            continue
        # evaluate proposal
        rnew, snew = lp_opt(xn, yn)
        if do_polish and rnew is not None and snew > cur_sum - 0.03:
            if (time.time() - t0) < BUDGET - 12.0:
                try:
                    xp, yp = polish_centres(xn, yn, rnew, maxiter=40, W=8000.0)
                    rp, sp = lp_opt(xp, yp)
                    if rp is not None and sp > snew:
                        xn, yn, rnew, snew = xp, yp, rp, sp
                except Exception:
                    pass
        if rnew is not None:
            delta = snew - cur_sum
            if delta >= 0 or rng.rand() < np.exp(delta / max(T, 1e-9)):
                cur_x, cur_y, cur_r, cur_sum = xn.copy(), yn.copy(), rnew.copy(), snew
            if snew > best_sum + 1e-12:
                best_sum, best_x, best_y, best_r = xn.copy(), yn.copy(), rnew.copy(), snew
                last_imp = it
        it += 1

    out = np.column_stack([best_x, best_y, best_r])
    x = out[:, 0]; y = out[:, 1]; r = out[:, 2]
    wall = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
    Dx = x[:, None]-x[None, :]; Dy = y[:, None]-y[None, :]
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
        r = r * (1.0 - 1e-9)
        out = np.column_stack([x, y, r])
    return out
