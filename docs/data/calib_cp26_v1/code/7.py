import numpy as np
import time

def solve():
    import numpy as np
    import time
    import scipy.optimize as so
    n = 26
    t0 = time.time()
    TIME_BUDGET = 122.0
    # prebuild pair matrix for LP
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    m = len(pairs)
    A_pair = np.zeros((m, n), dtype=float)
    for k, (i, j) in enumerate(pairs):
        A_pair[k, i] = 1.0
        A_pair[k, j] = 1.0
    A_ub_base = np.vstack([np.eye(n), A_pair])
    c_obj = -np.ones(n)

    def lp_polish(x, y):
        # returns r_opt feasible, sum or None
        dw = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        if np.any(dw < 1e-9):
            # infeasible positions outside, clip will fail
            return None, -1
        Dx = x[:, None] - x[None, :]
        Dy = y[:, None] - y[None, :]
        D = np.sqrt(Dx*Dx + Dy*Dy)
        # upper triangle distances
        dp = np.array([D[i, j] for (i, j) in pairs], dtype=float)
        b_ub = np.concatenate([dw, dp])
        bounds = [(0, None)]*n
        try:
            res = so.linprog(c_obj, A_ub=A_ub_base, b_ub=b_ub, bounds=bounds, method='highs')
        except Exception:
            return None, -1
        if not res.success:
            return None, -1
        r = np.array(res.x, dtype=float)
        # enforce strictly positive and feasible with margin
        r = np.maximum(r, 1e-7)
        # compute scaling to feasibility
        # wall scale
        # avoid div by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            s_wall = np.min(np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y)) / np.maximum(r, 1e-12))
            # pair scale
            denom = r[:, None] + r[None, :]
            # dist matrix
            # compute s_pair
            # only where denom>0
            ratio = D / np.maximum(denom, 1e-12)
            # set diag to large
            np.fill_diagonal(ratio, 1e9)
            s_pair = np.min(ratio)
        s = min(s_wall, s_pair)
        if not np.isfinite(s) or s <= 0:
            return None, -1
        s = s * (1.0 - 1e-9) - 1e-12
        if s <= 0:
            return None, -1
        # also cap at 1 to leave margin
        if s > 1.0:
            s = 1.0 - 1e-9
        r = r * s
        # final tiny shrink for tolerance 1e-12
        r = r * (1.0 - 1e-9)
        return r, float(np.sum(r))

    def feasible_scale(x, y, r):
        r = np.maximum(np.array(r, dtype=float), 1e-9)
        Dx = x[:, None] - x[None, :]
        Dy = y[:, None] - y[None, :]
        D = np.sqrt(Dx*Dx + Dy*Dy)
        np.fill_diagonal(D, 1e9)
        denom = r[:, None] + r[None, :]
        with np.errstate(divide='ignore'):
            s_pair = np.min(D / np.maximum(denom, 1e-12))
            dw = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
            s_wall = np.min(dw / np.maximum(r, 1e-12))
        s = min(s_wall, s_pair)
        if not np.isfinite(s) or s <= 0:
            s = 0.5
        if s > 1.0:
            s = 1.0 - 1e-9
        else:
            s = s * (1.0 - 1e-9) - 1e-12
            if s <= 0:
                s = 0.9
        return r * s, float(np.sum(r * s)), s

    # penalty objective factory
    def make_obj(P):
        def obj(v):
            x = v[0::3]
            y = v[1::3]
            r = v[2::3]
            v1 = r - x
            v2 = x + r - 1.0
            v3 = r - y
            v4 = y + r - 1.0
            p1 = np.maximum(v1, 0.0)
            p2 = np.maximum(v2, 0.0)
            p3 = np.maximum(v3, 0.0)
            p4 = np.maximum(v4, 0.0)
            wall_pen = np.sum(p1*p1 + p2*p2 + p3*p3 + p4*p4)
            Dx = x[:, None] - x[None, :]
            Dy = y[:, None] - y[None, :]
            D = np.sqrt(Dx*Dx + Dy*Dy + 1e-12)
            S = r[:, None] + r[None, :]
            Viol = S - D
            np.fill_diagonal(Viol, 0.0)
            V = np.maximum(Viol, 0.0)
            over_pen = 0.5*np.sum(V*V)
            f = -np.sum(r) + P*(wall_pen + over_pen)
            gx_w = -2.0*p1 + 2.0*p2
            gy_w = -2.0*p3 + 2.0*p4
            gr_w = 2.0*(p1+p2+p3+p4)
            invD = np.zeros_like(D)
            mask = V > 0
            invD[mask] = 1.0 / D[mask]
            W = V * invD
            gx_o = -2.0*np.sum(W*Dx, axis=1)
            gy_o = -2.0*np.sum(W*Dy, axis=1)
            gr_o = 2.0*np.sum(V, axis=1)
            gx_tot = P*(gx_w + gx_o)
            gy_tot = P*(gy_w + gy_o)
            gr_tot = -np.ones(n) + P*(gr_w + gr_o)
            g = np.empty_like(v)
            g[0::3] = gx_tot
            g[1::3] = gy_tot
            g[2::3] = gr_tot
            return f, g
        return obj

    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((0.01, 0.32))

    rng = np.random.default_rng(0)
    best_sum = -1
    best_x = None
    best_y = None
    best_r = None

    # seed baseline grid 5x5 + extra to guarantee something
    try:
        gx0 = np.array([0.1+0.2*i for i in range(5)]*5 + [0.5])
        gy0 = np.array([0.1+0.2*i for i in range(5) for _ in range(5)] + [0.1] if False else None)
        # simpler: 5x5 grid
        xs = []
        ys = []
        for ix in range(5):
            for iy in range(5):
                xs.append(0.1+0.2*ix)
                ys.append(0.1+0.2*iy)
        xs.append(0.5); ys.append(0.02)
        # adjust to fit 26: shrink a bit
        xs = np.array(xs); ys = np.array(ys)
        # use small radius feasible
        r0 = np.full(n, 0.08)
        # ensure inside: y last =0.08?
        ys[-1] = 0.08
        r_feas, s_sum, _ = feasible_scale(xs, ys, r0)
        # lp polish it
        rlp, s_lp = lp_polish(xs, ys)
        if rlp is not None and s_lp > s_sum:
            r_feas = rlp; s_sum = s_lp
        best_sum = s_sum
        best_x, best_y, best_r = xs.copy(), ys.copy(), r_feas.copy()
    except Exception:
        pass

    # Phase A: penalty restarts
    n_restart = 0
    # time split: phase A until ~70s
    PHASE_A_END = 72.0
    while time.time() - t0 < PHASE_A_END and time.time() - t0 < TIME_BUDGET - 20:
        n_restart += 1
        # make init
        v0 = np.empty(3*n)
        mode = n_restart % 5
        if mode == 0:
            # grid perturbed
            base_x = []
            base_y = []
            # 5 rows varied counts to fill: try 5,5,6,5,5 =26
            rows = [5,5,6,5,5]
            ycoords = [0.1,0.3,0.5,0.7,0.9]
            # adjust y for 5 rows
            for ri, cnt in enumerate(rows):
                yy = (ri+0.5)/5.0
                for k in range(cnt):
                    xx = (k+0.5)/cnt
                    base_x.append(xx); base_y.append(yy)
            base_x = np.array(base_x); base_y = np.array(base_y)
            # jitter
            jx = rng.normal(0, 0.02, n)
            jy = rng.normal(0, 0.02, n)
            x0 = np.clip(base_x+jx, 0.05, 0.95)
            y0 = np.clip(base_y+jy, 0.05, 0.95)
            r0 = rng.uniform(0.06, 0.11, n)
        elif mode == 1:
            x0 = rng.uniform(0.05, 0.95, n)
            y0 = rng.uniform(0.05, 0.95, n)
            r0 = rng.uniform(0.07, 0.11, n)
        else:
            x0 = rng.uniform(0.05, 0.95, n)
            y0 = rng.uniform(0.05, 0.95, n)
            r0 = rng.uniform(0.04, 0.10, n)
        v0[0::3] = x0; v0[1::3] = y0; v0[2::3] = r0
        # two-stage P
        v_cur = v0
        for P, it in [(2000.0, 350), (20000.0, 250)]:
            if time.time() - t0 > TIME_BUDGET - 10:
                break
            obj = make_obj(P)
            try:
                res = so.minimize(obj, v_cur, jac=True, method='L-BFGS-B', bounds=bounds,
                                  options={'maxiter': it, 'ftol': 1e-9, 'gtol': 1e-7})
                v_cur = res.x
            except Exception:
                break
        x_cur = v_cur[0::3]; y_cur = v_cur[1::3]; r_cur = v_cur[2::3]
        # clip positions inside
        x_cur = np.clip(x_cur, 1e-6, 1-1e-6); y_cur = np.clip(y_cur, 1e-6, 1-1e-6)
        # evaluate feasible scaled
        try:
            r_s, s_s, _ = feasible_scale(x_cur, y_cur, r_cur)
            if s_s > best_sum:
                best_sum = s_s; best_x, best_y, best_r = x_cur.copy(), y_cur.copy(), r_s.copy()
        except Exception:
            pass
        # LP polish
        try:
            rlp, s_lp = lp_polish(x_cur, y_cur)
            if rlp is not None and s_lp > best_sum:
                best_sum = s_lp; best_x, best_y, best_r = x_cur.copy(), y_cur.copy(), rlp.copy()
        except Exception:
            pass
        # early exit if time
        if time.time() - t0 > PHASE_A_END:
            break

    # Phase B: hill climbing on positions with LP
    if best_x is not None:
        sigma = 0.025
        it_b = 0
        # ensure best is LP-optimal for its positions
        try:
            rlp0, s0 = lp_polish(best_x, best_y)
            if rlp0 is not None and s0 > best_sum:
                best_sum = s0; best_r = rlp0
        except Exception:
            pass
        while time.time() - t0 < TIME_BUDGET:
            it_b += 1
            # decay sigma slowly
            if it_b % 300 == 0:
                sigma *= 0.85
                if sigma < 0.003:
                    sigma = 0.003
            # propose
            xn = best_x + rng.normal(0, sigma, n)
            yn = best_y + rng.normal(0, sigma, n)
            xn = np.clip(xn, 0.001, 0.999)
            yn = np.clip(yn, 0.001, 0.999)
            try:
                rlp, s_lp = lp_polish(xn, yn)
            except Exception:
                continue
            if rlp is not None and s_lp > best_sum + 1e-12:
                best_sum = s_lp
                best_x, best_y, best_r = xn, yn, rlp
                # slight sigma increase on success
                sigma *= 1.02
                if sigma > 0.04:
                    sigma = 0.04
            # periodic time check is in while
        # final assembly with safety shrink
        out = np.zeros((n, 3))
        out[:, 0] = best_x
        out[:, 1] = best_y
        out[:, 2] = best_r
        # final strict feasibility guarantee
        x = out[:, 0]; y = out[:, 1]; r = out[:, 2]
        r = np.maximum(r, 1e-6)
        Dx = x[:, None]-x[None, :]
        Dy = y[:, None]-y[None, :]
        D = np.sqrt(Dx*Dx+Dy*Dy)
        np.fill_diagonal(D, 1e9)
        denom = r[:, None]+r[None, :]
        s_pair = np.min(D/np.maximum(denom, 1e-12))
        dw = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
        s_wall = np.min(dw/np.maximum(r, 1e-12))
        s = min(s_pair, s_wall, 1.0-1e-9)
        s = s*(1.0-1e-9)
        r = r*s
        # ensure tiny margin and >0
        r = np.maximum(r, 1e-7)
        out[:, 2] = r
        return out
    else:
        # fallback grid
        k = int(np.ceil(np.sqrt(n)))
        rr = 0.5/k
        pts = []
        for i in range(k):
            for j in range(k):
                if len(pts) < n:
                    pts.append([(2*i+1)*rr, (2*j+1)*rr, rr*0.999999])
        return np.array(pts)
