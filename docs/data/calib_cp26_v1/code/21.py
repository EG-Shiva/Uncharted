import numpy as np
import time

def solve():
    import numpy as np
    import time
    import scipy.optimize as so
    n = 26
    t0 = time.time()
    BUDGET = 124.0

    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    m = len(pairs)
    pi = np.array([p[0] for p in pairs], dtype=int)
    pj = np.array([p[1] for p in pairs], dtype=int)

    A_pair = np.zeros((m, n), dtype=float)
    for k, (i, j) in enumerate(pairs):
        A_pair[k, i] = 1.0
        A_pair[k, j] = 1.0
    A_ub = np.vstack([np.eye(n), A_pair])
    c_obj = -np.ones(n)
    bnds_r = [(0, None)] * n
    At_neg = -A_ub.T  # for dual if needed

    # ---------- LP primal ----------
    def lp_primal(x, y):
        dw = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        if np.min(dw) <= 1e-12:
            return -1.0, None, None
        Dx = x[:, None] - x[None, :]
        Dy = y[:, None] - y[None, :]
        D = np.sqrt(Dx * Dx + Dy * Dy)
        dp = D[pi, pj]
        if np.min(dp) <= 1e-12:
            return -1.0, None, None
        b_ub = np.concatenate([dw, dp])
        try:
            res = so.linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bnds_r, method='highs')
        except Exception:
            return -1.0, None, None
        if not res.success:
            return -1.0, None, None
        r = np.array(res.x, dtype=float)
        s = float(np.sum(r))
        lam = None
        try:
            iq = getattr(res, 'ineqlin', None)
            marg = None
            if iq is not None:
                if hasattr(iq, 'marginals'):
                    marg = iq.marginals
                elif isinstance(iq, dict) and 'marginals' in iq:
                    marg = iq['marginals']
            if marg is not None:
                marg = np.asarray(marg, dtype=float)
                if marg.shape[0] == n + m:
                    lam = np.maximum(-marg, 0.0)
        except Exception:
            lam = None
        return s, r, lam

    def lam_via_dual(b_ub):
        # dual of max{1'r : Ar<=b, r>=0} is min{b'l : A'l>=1, l>=0}
        c_d = np.array(b_ub, dtype=float)
        b_d = -np.ones(n, dtype=float)
        try:
            res = so.linprog(c_d, A_ub=At_neg, b_ub=b_d,
                             bounds=[(0, None)] * (n + m), method='highs')
        except Exception:
            return None
        if not res.success:
            return None
        return np.maximum(np.array(res.x, dtype=float), 0.0)

    # function for L-BFGS-B: p = [x,y]
    def fun_grad(p):
        x = p[:n]
        y = p[n:]
        s, r, lam = lp_primal(x, y)
        if r is None:
            # infeasible: push inside
            return 0.0, np.zeros_like(p)
        if lam is None:
            dw0 = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
            Dx0 = x[:, None] - x[None, :]
            Dy0 = y[:, None] - y[None, :]
            D0 = np.sqrt(Dx0 * Dx0 + Dy0 * Dy0)
            dp0 = D0[pi, pj]
            b0 = np.concatenate([dw0, dp0])
            lam = lam_via_dual(b0)
            if lam is None:
                return -s, np.zeros_like(p)
        lam_w = lam[:n]
        lam_p = lam[n:]
        gx = np.zeros(n)
        gy = np.zeros(n)
        # wall parts
        for i in range(n):
            xi = x[i]; yi = y[i]
            a1 = xi; a2 = 1.0 - xi; b1 = yi; b2 = 1.0 - yi
            w = min(a1, a2, b1, b2)
            lw = lam_w[i]
            if lw == 0:
                continue
            if w == a1:
                gx[i] += lw
            elif w == a2:
                gx[i] -= lw
            elif w == b1:
                gy[i] += lw
            else:
                gy[i] -= lw
        # pair parts
        # compute distances
        Dx = x[:, None] - x[None, :]
        Dy = y[:, None] - y[None, :]
        D = np.sqrt(Dx * Dx + Dy * Dy + 1e-20)
        # only where lam>0
        idx = np.where(lam_p > 1e-12)[0]
        for k in idx:
            i = pi[k]; j = pj[k]
            d = D[i, j]
            if d < 1e-12:
                continue
            lp = lam_p[k]
            ux = (x[i] - x[j]) / d
            uy = (y[i] - y[j]) / d
            gx[i] += lp * ux
            gx[j] -= lp * ux
            gy[i] += lp * uy
            gy[j] -= lp * uy
        grad_f = np.concatenate([gx, gy])
        return -s, -grad_f

    rng = np.random.default_rng(0)
    best_sum = -1.0
    best_x = None
    best_y = None
    best_r = None

    def consider(x, y):
        global_vars = None
        s, r, _ = lp_primal(x, y)
        return s, r

    # fallback grid
    try:
        xs = []; ys = []
        for ix in range(5):
            for iy in range(5):
                xs.append(0.1 + 0.2 * ix); ys.append(0.1 + 0.2 * iy)
        xs.append(0.5); ys.append(0.5)
        xs = np.array(xs); ys = np.array(ys)
        # jitter slightly to avoid degeneracy
        xs[-1] = 0.5; ys[-1] = 0.02
        ys[-1] = 0.08
        s0, r0, _ = lp_primal(np.clip(xs, 0.01, 0.99), np.clip(ys, 0.01, 0.99))
        if r0 is not None:
            best_sum = s0; best_x = np.clip(xs, 0.01, 0.99).copy(); best_y = np.clip(ys, 0.01, 0.99).copy(); best_r = r0.copy()
    except Exception:
        pass

    bounds_xy = [(0.01, 0.99)] * (2 * n)

    # ---- Phase 1: gradient-based centre optimisation ----
    PHASE1_END = 68.0
    restart = 0
    while time.time() - t0 < PHASE1_END:
        restart += 1
        if time.time() - t0 > BUDGET - 30:
            break
        mode = restart % 4
        if mode == 0 and best_x is not None and restart > 3:
            # perturb best
            sig = 0.03
            x0 = np.clip(best_x + rng.normal(0, sig, n), 0.01, 0.99)
            y0 = np.clip(best_y + rng.normal(0, sig, n), 0.01, 0.99)
        elif mode == 1:
            # row-structured init 5,5,6,5,5
            rows = [5, 5, 6, 5, 5]
            bx = []; by = []
            for ri, cnt in enumerate(rows):
                yy = (ri + 0.5) / 5.0
                for k in range(cnt):
                    xx = (k + 0.5) / cnt
                    bx.append(xx); by.append(yy)
            bx = np.array(bx); by = np.array(by)
            x0 = np.clip(bx + rng.normal(0, 0.03, n), 0.01, 0.99)
            y0 = np.clip(by + rng.normal(0, 0.03, n), 0.01, 0.99)
        elif mode == 2:
            # 6x? grid subsample
            gx = []; gy = []
            for ix in range(6):
                for jy in range(5):
                    gx.append((ix + 0.5) / 6.0); gy.append((jy + 0.5) / 5.0)
            gx = np.array(gx); gy = np.array(gy)
            perm = rng.permutation(len(gx))[:n]
            x0 = np.clip(gx[perm] + rng.normal(0, 0.02, n), 0.01, 0.99)
            y0 = np.clip(gy[perm] + rng.normal(0, 0.02, n), 0.01, 0.99)
        else:
            x0 = rng.uniform(0.05, 0.95, n)
            y0 = rng.uniform(0.05, 0.95, n)
        p0 = np.concatenate([x0, y0])
        try:
            res = so.minimize(fun_grad, p0, jac=True, method='L-BFGS-B',
                              bounds=bounds_xy,
                              options={'maxiter': 120, 'ftol': 1e-9, 'gtol': 1e-5})
            p_opt = res.x
        except Exception:
            continue
        x_opt = p_opt[:n]; y_opt = p_opt[n:]
        try:
            s, r, _ = lp_primal(x_opt, y_opt)
        except Exception:
            continue
        if r is not None and s > best_sum:
            best_sum = s; best_x = x_opt.copy(); best_y = y_opt.copy(); best_r = r.copy()

    # ---- Phase 2: single-circle hill climbing on LP value ----
    if best_x is None:
        k = int(np.ceil(np.sqrt(n)))
        rr = 0.5 / k
        pts = []
        for i in range(k):
            for j in range(k):
                if len(pts) < n:
                    pts.append([(2 * i + 1) * rr, (2 * j + 1) * rr, rr * 0.999])
        arr = np.array(pts)
        arr[:, 2] *= (1 - 1e-7)
        return arr

    # re-polish best
    try:
        s, r, _ = lp_primal(best_x, best_y)
        if r is not None and s > best_sum:
            best_sum = s; best_r = r
    except Exception:
        pass

    sigma = 0.02
    it = 0
    # precompute for speed: local func without dual
    while time.time() - t0 < BUDGET - 1.5:
        it += 1
        if it % 400 == 0:
            sigma *= 0.90
            if sigma < 0.002:
                sigma = 0.002
        # choose subset size
        u = rng.random()
        if u < 0.65:
            ksub = 1
        elif u < 0.88:
            ksub = 2
        else:
            ksub = 3
        idx = rng.choice(n, size=ksub, replace=False)
        # occasionally use smaller step for fine tune
        s_local = sigma * (0.3 if (it % 7 == 0) else 1.0)
        xn = best_x.copy(); yn = best_y.copy()
        xn[idx] += rng.normal(0, s_local, ksub)
        yn[idx] += rng.normal(0, s_local, ksub)
        xn = np.clip(xn, 0.01, 0.99); yn = np.clip(yn, 0.01, 0.99)
        try:
            s, r, _ = lp_primal(xn, yn)
        except Exception:
            continue
        if r is not None and s > best_sum + 1e-12:
            best_sum = s; best_x = xn; best_y = yn; best_r = r
            sigma *= 1.015
            if sigma > 0.035:
                sigma = 0.035

    x = best_x; y = best_y; r = np.maximum(np.array(best_r, dtype=float), 1e-9)
    # strict feasibility shrink
    Dx = x[:, None] - x[None, :]
    Dy = y[:, None] - y[None, :]
    D = np.sqrt(Dx * Dx + Dy * Dy)
    np.fill_diagonal(D, 1e9)
    denom = r[:, None] + r[None, :]
    s_pair = np.min(D / np.maximum(denom, 1e-12))
    dw = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    s_wall = np.min(dw / np.maximum(r, 1e-12))
    s = min(s_pair, s_wall, 1.0 - 1e-9)
    s = s * (1.0 - 1e-7)
    r = r * s
    r = np.maximum(r, 1e-7)
    out = np.zeros((n, 3))
    out[:, 0] = x; out[:, 1] = y; out[:, 2] = r
    return out
