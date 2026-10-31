import numpy as np
import time

def solve():
    import numpy as np
    import time
    from scipy.optimize import minimize, linprog
    N = 26
    t0 = time.time()
    DEADLINE = 122.0
    rng = np.random.default_rng(7)

    # pair matrix for LP: r_i + r_j <= d_ij
    I, J = np.triu_indices(N, 1)
    m = len(I)
    A0 = np.zeros((m, N), dtype=float)
    A0[np.arange(m), I] = 1.0
    A0[np.arange(m), J] = 1.0
    cvec = -np.ones(N, dtype=float)

    def lp_opt(xc, yc):
        xc = np.asarray(xc, float); yc = np.asarray(yc, float)
        wall = np.minimum(np.minimum(xc, 1.0 - xc), np.minimum(yc, 1.0 - yc))
        if np.min(wall) <= 1e-9:
            return None, -1.0
        dx = xc[I] - xc[J]
        dy = yc[I] - yc[J]
        d = np.sqrt(dx*dx + dy*dy)
        if np.min(d) <= 1e-9:
            return None, -1.0
        eps = 1e-9
        b_ub = d - eps
        # need b_ub >=0
        if np.min(b_ub) <= 0:
            return None, -1.0
        bounds = [(0.0, float(wi - eps)) for wi in wall]
        # quick check upper bounds positive
        if min(b[1] for b in bounds) <= 0:
            return None, -1.0
        try:
            res = linprog(cvec, A_ub=A0, b_ub=b_ub, bounds=bounds, method='highs')
        except Exception:
            return None, -1.0
        if res.status != 0:
            return None, -1.0
        r = np.asarray(res.x, dtype=float)
        r = np.maximum(r, 0.0)
        if np.min(r) < 1e-7:
            # degenerate, still feasible but avoid zero radius
            pass
        return r, float(np.sum(r))

    # penalized joint loss+grad
    def make_fun():
        def fun(v, W):
            X = v[0:N]; Y = v[N:2*N]; R = v[2*N:3*N]
            gx = np.zeros(N); gy = np.zeros(N); grpen = np.zeros(N)
            pen = 0.0
            v1 = R - X
            mm = v1 > 0
            if np.any(mm):
                pen += np.sum(v1[mm]**2)
                gx[mm] += -2.0*v1[mm]
                grpen[mm] += 2.0*v1[mm]
            v2 = X + R - 1.0
            mm = v2 > 0
            if np.any(mm):
                pen += np.sum(v2[mm]**2)
                gx[mm] += 2.0*v2[mm]
                grpen[mm] += 2.0*v2[mm]
            v3 = R - Y
            mm = v3 > 0
            if np.any(mm):
                pen += np.sum(v3[mm]**2)
                gy[mm] += -2.0*v3[mm]
                grpen[mm] += 2.0*v3[mm]
            v4 = Y + R - 1.0
            mm = v4 > 0
            if np.any(mm):
                pen += np.sum(v4[mm]**2)
                gy[mm] += 2.0*v4[mm]
                grpen[mm] += 2.0*v4[mm]
            Dx = X[:, None] - X[None, :]
            Dy = Y[:, None] - Y[None, :]
            D = np.sqrt(Dx*Dx + Dy*Dy + 1e-20)
            S = R[:, None] + R[None, :]
            V = S - D
            np.fill_diagonal(V, 0.0)
            np.fill_diagonal(D, 1.0)
            Mask = V > 0
            if np.any(Mask):
                pen += 0.25*np.sum(V[Mask]**2)
                Q = np.zeros((N, N), dtype=float)
                # -V/D
                Q[Mask] = -V[Mask] / D[Mask]
                gx += np.sum(Q*Dx, axis=1)
                gy += np.sum(Q*Dy, axis=1)
                # d pen / d R
                grpen += np.sum(V*Mask, axis=1)
            loss = -np.sum(R) + W*pen
            g = np.empty_like(v)
            g[0:N] = W*gx
            g[N:2*N] = W*gy
            g[2*N:3*N] = -1.0 + W*grpen
            return loss, g
        return fun
    pfun = make_fun()
    boundsJR = [(0.0, 1.0)]*(2*N) + [(0.015, 0.22)]*N

    def short_polish(xc, yc, rc, time_left):
        # one short annealed polish, return new xc,yc,rc
        v0 = np.concatenate([xc, yc, np.clip(rc, 0.015, 0.22)])
        # choose stages based on time
        if time_left > 30:
            Ws = [5e3, 5e4]
            mi = [60, 60]
        else:
            Ws = [1e4]
            mi = [50]
        v = v0
        for W, mit in zip(Ws, mi):
            if time.time()-t0 > DEADLINE-6:
                break
            try:
                res = minimize(pfun, v, args=(W,), method='L-BFGS-B', jac=True,
                               bounds=boundsJR, options={'maxiter': mit, 'ftol':1e-9, 'gtol':1e-5})
                v = res.x
            except Exception:
                break
        X = v[0:N]; Y = v[N:2*N]; R = v[2*N:3*N]
        # clip to inside (centres may be outside due to penalty tradeoff, clamp)
        X = np.clip(X, 0.01, 0.99); Y = np.clip(Y, 0.01, 0.99)
        r2, s2 = lp_opt(X, Y)
        if r2 is None:
            return xc, yc, rc
        return X, Y, r2

    # ---------- Phase 1: few penalized restarts ----------
    best_sum = -1.0
    best_x = None; best_y = None; best_r = None
    # grid pts for init diversity
    gx = []; gy = []
    for ix in range(6):
        for jy in range(5):
            gx.append((ix+0.5)/6.0); gy.append((jy+0.5)/5.0)
    gx = np.array(gx); gy = np.array(gy)

    n_restart = 0
    Ws_full = [1e2, 1e3, 1e4, 1e5]
    mits_full = [80, 80, 100, 120]
    while (time.time()-t0) < 38.0 and n_restart < 14:
        if n_restart % 2 == 0:
            X0 = rng.uniform(0.03, 0.97, N); Y0 = rng.uniform(0.03, 0.97, N)
        else:
            perm = rng.permutation(len(gx))[:N]
            X0 = gx[perm] + (rng.random(N)-0.5)*0.08
            Y0 = gy[perm] + (rng.random(N)-0.5)*0.08
            X0 = np.clip(X0, 0.02, 0.98); Y0 = np.clip(Y0, 0.02, 0.98)
        R0 = 0.06 + 0.05*rng.random(N)
        v = np.concatenate([X0, Y0, R0])
        for W, mit in zip(Ws_full, mits_full):
            if time.time()-t0 >= 38.0:
                break
            try:
                res = minimize(pfun, v, args=(W,), method='L-BFGS-B', jac=True,
                               bounds=boundsJR, options={'maxiter': mit, 'ftol':1e-9, 'gtol':1e-5})
                v = res.x
            except Exception:
                break
        X = np.clip(v[0:N], 0.01, 0.99); Y = np.clip(v[N:2*N], 0.01, 0.99)
        r_opt, s = lp_opt(X, Y)
        if r_opt is not None and s > best_sum:
            best_sum = s; best_x = X.copy(); best_y = Y.copy(); best_r = r_opt.copy()
        n_restart += 1

    if best_x is None:
        # fallback grid
        k = int(np.ceil(np.sqrt(N)))
        rr0 = 0.5/k
        pts = []
        for ix in range(k):
            for jy in range(k):
                if len(pts) < N:
                    pts.append([(2*ix+1)*rr0, (2*jy+1)*rr0, rr0*0.999])
        arr = np.array(pts, float)
        # strict shrink
        X = arr[:,0]; Y = arr[:,1]; R = arr[:,2]*0.999
        return np.column_stack([X, Y, R])

    # ---------- helpers for surgery ----------
    G = 201
    lin = np.linspace(0.0, 1.0, G)
    PX, PY = np.meshgrid(lin, lin)
    PXf = PX.ravel(); PYf = PY.ravel()
    Mpts = PXf.size

    def contact_counts(xc, yc, rc):
        wall = np.minimum(np.minimum(xc, 1-xc), np.minimum(yc, 1-yc))
        cw = (wall - rc) < 2e-3
        Dx = xc[:,None]-xc[None,:]; Dy = yc[:,None]-yc[None,:]
        D = np.sqrt(Dx*Dx+Dy*Dy)
        S = rc[:,None]+rc[None,:]
        tight = (D - S) < 2e-3
        # count per i (exclude self)
        cnt = cw.astype(int)
        for i in range(N):
            cnt[i] += int(np.sum(tight[i]) - 1)  # minus self (Dii-Sii<0 true)
        return cnt

    def find_k_voids(xc, yc, rc, fixed_mask, k):
        # clearance w.r.t fixed circles only, then greedy
        # fixed_mask True = fixed
        xf = xc[fixed_mask]; yf = yc[fixed_mask]; rf = rc[fixed_mask]
        # compute base dist min
        dmin = np.full(Mpts, 1e100, dtype=float)
        for (xj, yj, rj) in zip(xf, yf, rf):
            dx = PXf - xj; dy = PYf - yj
            d = np.sqrt(dx*dx+dy*dy) - rj
            np.minimum(dmin, d, out=dmin)
        wallp = np.minimum(np.minimum(PXf, 1.0-PXf), np.minimum(PYf, 1.0-PYf))
        voids = []
        dcur = dmin.copy()
        wcur = wallp
        for _ in range(k):
            clear = np.minimum(wcur, dcur)
            idx = int(np.argmax(clear))
            cl = float(clear[idx])
            vx = float(PXf[idx]); vy = float(PYf[idx])
            voids.append((vx, vy, cl))
            # insert virtual circle
            dx = PXf - vx; dy = PYf - vy
            d = np.sqrt(dx*dx+dy*dy) - cl
            np.minimum(dcur, d, out=dcur)
        return voids

    cur_x = best_x.copy(); cur_y = best_y.copy(); cur_r = best_r.copy(); cur_sum = best_sum
    T0 = 0.004; T1 = 0.00015
    it = 0
    # estimate total iters for cooling: use time-based
    t_phase2_start = time.time()-t0

    while (time.time()-t0) < DEADLINE-8:
        it += 1
        frac = (time.time()-t0 - t_phase2_start) / max(1e-6, (DEADLINE-8 - t_phase2_start))
        frac = min(max(frac, 0.0), 1.0)
        T = T0*(1-frac) + T1*frac
        # periodically restore elite
        if it % 25 == 0 and (rng.random() < 0.7):
            cur_x = best_x.copy(); cur_y = best_y.copy(); cur_r = best_r.copy(); cur_sum = best_sum
        # decide surgery size
        mode = it % 5
        if mode == 0:
            k = 1
        elif mode == 1:
            k = 2
        elif mode == 2:
            k = 3
        elif mode == 3:
            k = 1
        else:
            k = 2
        # choose indices to move
        order_small = np.argsort(cur_r)
        if mode == 3:
            cnt = contact_counts(cur_x, cur_y, cur_r)
            order_contact = np.argsort(cnt*10.0 + cur_r*50.0)
            move_idx = order_contact[:k]
        elif mode == 4:
            # random among bottom 8
            bottom = np.argsort(cur_r)[:8]
            sel = rng.choice(bottom, size=k, replace=False)
            move_idx = np.array(sel)
        else:
            move_idx = order_small[:k]
        fixed_mask = np.ones(N, dtype=bool)
        fixed_mask[move_idx] = False
        voids = find_k_voids(cur_x, cur_y, cur_r, fixed_mask, k)
        new_x = cur_x.copy(); new_y = cur_y.copy()
        for t_, idx_ in enumerate(move_idx):
            vx, vy, cl = voids[t_]
            # jitter deterministically via rng
            jx = (rng.random()-0.5)*0.004; jy = (rng.random()-0.5)*0.004
            nx = min(max(vx+jx, 0.02), 0.98); ny = min(max(vy+jy, 0.02), 0.98)
            new_x[idx_] = nx; new_y[idx_] = ny
        # occasional pure Gaussian hill move instead (10%)
        if rng.random() < 0.10:
            sig = 0.006 if rng.random()<0.5 else 0.015
            new_x = cur_x + sig*rng.standard_normal(N)
            new_y = cur_y + sig*rng.standard_normal(N)
            new_x = np.clip(new_x, 0.02, 0.98); new_y = np.clip(new_y, 0.02, 0.98)
        r_try, s_try = lp_opt(new_x, new_y)
        if r_try is None:
            continue
        # short polish
        time_left = DEADLINE-8-(time.time()-t0)
        # polish every move if promising (within 0.03) else polish only sometimes to save time
        do_polish = (s_try > cur_sum - 0.03) and (it % 2 == 0 or s_try > cur_sum)
        if do_polish:
            tstart = time.time()
            # limit polish time
            new_x2, new_y2, r_try2 = short_polish(new_x, new_y, r_try, time_left)
            r_try, s_try = r_try2, float(np.sum(r_try2))
            # lp_opt already inside; recompute s from r_try
            # need to get centres new_x2
            new_x, new_y = new_x2, new_y2
        else:
            pass
        if s_try > best_sum + 1e-12:
            best_sum = s_try; best_x = new_x.copy(); best_y = new_y.copy(); best_r = r_try.copy()
            cur_x = new_x.copy(); cur_y = new_y.copy(); cur_r = r_try.copy(); cur_sum = s_try
        else:
            delta = s_try - cur_sum
            if delta >= 0:
                cur_x = new_x.copy(); cur_y = new_y.copy(); cur_r = r_try.copy(); cur_sum = s_try
            else:
                if T > 1e-12 and rng.random() < np.exp(delta / T):
                    cur_x = new_x.copy(); cur_y = new_y.copy(); cur_r = r_try.copy(); cur_sum = s_try
        # extra greedy hill on best occasionally
        if it % 30 == 0:
            # try small Gaussian around best
            for _ in range(5):
                if (time.time()-t0) > DEADLINE-8:
                    break
                sig = 0.003
                tx = best_x + sig*rng.standard_normal(N)
                ty = best_y + sig*rng.standard_normal(N)
                tx = np.clip(tx, 0.02, 0.98); ty = np.clip(ty, 0.02, 0.98)
                rr, ss = lp_opt(tx, ty)
                if rr is not None and ss > best_sum:
                    best_sum = ss; best_x = tx; best_y = ty; best_r = rr
                    cur_x = tx; cur_y = ty; cur_r = rr; cur_sum = ss

    # final strict feasibility scaling
    x = best_x; y = best_y; r = best_r.copy()
    r = np.maximum(r, 1e-9)
    wall = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
    Dx = x[:,None]-x[None,:]; Dy = y[:,None]-y[None,:]
    D = np.sqrt(Dx*Dx+Dy*Dy)
    np.fill_diagonal(D, np.inf)
    S = r[:,None]+r[None,:]
    with np.errstate(divide='ignore', invalid='ignore'):
        a = float(np.min(wall / r))
        b = float(np.min(D / S))
    alpha = min(1.0, a, b)
    alpha = alpha*(1.0-1e-7)-1e-12
    if not np.isfinite(alpha) or alpha <= 0:
        alpha = 0.9
    r = r*alpha
    # ensure positivity
    r = np.maximum(r, 1e-6)
    # re-check after max (if max pushed violation, rescale once more)
    wall2 = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
    if np.min(wall2 - r) < 0 or np.min(D - (r[:,None]+r[None,:])) < 0:
        # uniform shrink again
        with np.errstate(divide='ignore', invalid='ignore'):
            a2 = float(np.min(wall2 / r))
            b2 = float(np.min(D / (r[:,None]+r[None,:])))
        al2 = min(1.0, a2, b2)*(1.0-1e-9)-1e-12
        r = r*al2
    out = np.column_stack([x, y, r])
    return out
