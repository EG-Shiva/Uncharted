import numpy as np
import time

def solve():
    n = 26
    t0 = time.time()
    DEADLINE = 124.0
    T_TEMPL_END = t0 + 38.0
    T_HILL_END = t0 + 112.0
    rng = np.random.RandomState(0)

    I, J = np.triu_indices(n, 1)
    m = len(I)
    # fixed A_ub: pair rows + wall rows
    A_full = np.zeros((m + n, n), dtype=float)
    ar = np.arange(m)
    A_full[ar, I] = 1.0
    A_full[ar, J] = 1.0
    for i in range(n):
        A_full[m + i, i] = 1.0
    cvec = -np.ones(n, dtype=float)
    bnds = [(0.0, None)] * n

    try:
        from scipy.optimize import minimize, linprog
        has_opt = True
    except Exception:
        has_opt = False
        linprog = None

    EPS = 1e-9

    def lp_eval(xc, yc):
        # returns (sum, r) or (-1, None)
        try:
            wall = np.minimum(np.minimum(xc, 1.0 - xc), np.minimum(yc, 1.0 - yc))
        except Exception:
            return -1.0, None
        if np.any(~np.isfinite(wall)):
            return -1.0, None
        if np.min(wall) <= EPS:
            return -1.0, None
        dx = xc[I] - xc[J]
        dy = yc[I] - yc[J]
        d = np.sqrt(dx * dx + dy * dy)
        if np.min(d) <= 2 * EPS:
            return -1.0, None
        b_ub = np.empty(m + n, dtype=float)
        b_ub[:m] = d - EPS
        b_ub[m:] = wall - EPS
        # quick infeasible if any b<0 (should not)
        if np.min(b_ub) < 0:
            return -1.0, None
        try:
            res = linprog(cvec, A_ub=A_full, b_ub=b_ub, bounds=bnds, method='highs')
        except Exception:
            return -1.0, None
        if res.status != 0:
            return -1.0, None
        try:
            r = np.asarray(res.x, dtype=float)
        except Exception:
            return -1.0, None
        r = np.maximum(r, 0.0)
        s = float(np.sum(r))
        if not np.isfinite(s) or s <= 0:
            return -1.0, None
        return s, r

    def sigmoid(z):
        # stable
        out = np.empty_like(z, dtype=float)
        pos = z >= 0
        neg = ~pos
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        e = np.exp(z[neg])
        out[neg] = e / (1.0 + e)
        return out

    def logit(u):
        u = np.clip(u, 1e-6, 1 - 1e-6)
        return np.log(u / (1.0 - u))

    def centres_from_template(p, counts):
        R = len(counts)
        p = np.asarray(p, dtype=float).reshape(R, 4)
        py = p[:, 0]; px0 = p[:, 1]; px1 = p[:, 2]; pw = p[:, 3]
        y_raw = sigmoid(py)
        x0_raw = sigmoid(px0)
        x1_raw = sigmoid(px1)
        warp = 0.08 * np.tanh(pw)
        # sort rows by y to enforce bottom-to-top order; x travels with y
        order = np.argsort(y_raw, kind='stable')
        y_s = y_raw[order]
        x0_s = x0_raw[order]
        x1_s = x1_raw[order]
        warp_s = warp[order]
        # clip y to interior
        y_s = np.clip(y_s, 0.015, 0.985)
        xc = np.empty(n, dtype=float)
        yc = np.empty(n, dtype=float)
        ptr = 0
        for k in range(R):
            ck = int(counts[k])
            yk = float(y_s[k])
            xa = float(x0_s[k]); xb = float(x1_s[k])
            lo = xa if xa < xb else xb
            hi = xb if xa < xb else xa
            # ensure minimal width
            if hi - lo < 1e-6:
                hi = lo + 1e-6
            lo = float(np.clip(lo, 0.01, 0.99))
            hi = float(np.clip(hi, 0.01, 0.99))
            if hi < lo:
                lo, hi = hi, lo
            if hi - lo < 0.02:
                mid = 0.5 * (lo + hi)
                lo = max(0.01, mid - 0.35)
                hi = min(0.99, mid + 0.35)
            wk = float(warp_s[k])
            for j in range(ck):
                if ck == 1:
                    t = 0.5
                else:
                    t = j / (ck - 1.0)
                x = lo + t * (hi - lo) + wk * np.sin(np.pi * t)
                x = float(np.clip(x, 0.005, 0.995))
                xc[ptr] = x
                yc[ptr] = yk
                ptr += 1
        xc = np.clip(xc, 0.005, 0.995)
        yc = np.clip(yc, 0.005, 0.995)
        return xc, yc

    # templates compatible with 26
    templates = [
        [6, 5, 5, 5, 5],
        [5, 6, 5, 5, 5],
        [5, 5, 6, 5, 5],
        [5, 5, 5, 6, 5],
        [5, 5, 5, 5, 6],
        [6, 5, 6, 5, 4],
        [5, 6, 5, 6, 4],
        [4, 6, 5, 6, 5],
        [6, 5, 5, 6, 4],
        [5, 4, 5, 4, 4, 4],  # 6 rows variants (sums 26? check)
        [5, 5, 4, 4, 4, 4],
        [5, 4, 5, 4, 5, 3],  # will be fixed if sum!=26
        [7, 6, 7, 6],
        [6, 7, 6, 7],
        [7, 6, 6, 7],
    ]
    # fix sums to 26 (adjust last entry)
    fixed = []
    for cc in templates:
        s = sum(cc)
        cc = list(cc)
        if s != 26:
            cc[-1] += 26 - s
            if cc[-1] <= 0:
                continue
        fixed.append(cc)
    templates = fixed

    def p0_for_template(counts):
        R = len(counts)
        p = np.zeros((R, 4), dtype=float)
        for k in range(R):
            u = (k + 0.5) / R
            p[k, 0] = logit(u)
            p[k, 1] = logit(0.10)
            p[k, 2] = logit(0.90)
            p[k, 3] = 0.0
        # stagger: offset alternate rows slightly
        for k in range(R):
            if k % 2 == 1:
                # shift interior a bit
                p[k, 1] = logit(0.13)
                p[k, 2] = logit(0.93)
        return p.reshape(-1)

    best_sum = -1.0
    best_xc = None; best_yc = None; best_r = None
    # fallback grid
    kgrid = int(np.ceil(np.sqrt(n)))
    rr0 = 0.5 / kgrid
    fb = []
    for ix in range(kgrid):
        for jy in range(kgrid):
            if len(fb) < n:
                fb.append([(2 * ix + 1) * rr0, (2 * jy + 1) * rr0])
    fb = np.array(fb, dtype=float)
    if has_opt:
        s_fb, r_fb = lp_eval(fb[:, 0], fb[:, 1])
        if s_fb > 0:
            best_sum = s_fb; best_xc = fb[:, 0].copy(); best_yc = fb[:, 1].copy(); best_r = r_fb.copy()
    else:
        # no scipy: return grid
        out = np.column_stack([fb[:, 0], fb[:, 1], np.full(n, rr0 * 0.999)])
        return out

    pool = []
    if best_xc is not None:
        pool.append((best_sum, best_xc.copy(), best_yc.copy()))

    # ---- Phase T: score templates initial ----
    init_scores = []
    if has_opt:
        for ti, counts in enumerate(templates):
            if time.time() - t0 > T_TEMPL_END - 15.0:
                break
            p0 = p0_for_template(counts)
            xc0, yc0 = centres_from_template(p0, counts)
            s0, _ = lp_eval(xc0, yc0)
            init_scores.append((s0, ti))
            if s0 > best_sum:
                _, r0 = lp_eval(xc0, yc0)
                if r0 is not None:
                    best_sum = s0; best_xc = xc0.copy(); best_yc = yc0.copy(); best_r = r0.copy()
                    pool.append((best_sum, xc0.copy(), yc0.copy()))
        # rank templates
        init_scores.sort(key=lambda t: -t[0])
        ranked = [ti for (s, ti) in init_scores]
        # stochastic hill climbing on template params
        for rank, ti in enumerate(ranked):
            if time.time() - t0 > T_TEMPL_END:
                break
            counts = templates[ti]
            R = len(counts)
            D = 4 * R
            p_best = p0_for_template(counts) + rng.normal(0, 0.25, D)
            # evaluate
            xc_b, yc_b = centres_from_template(p_best, counts)
            s_best, r_best = lp_eval(xc_b, yc_b)
            if s_best < 0:
                s_best = -1.0
            # iterations budget by rank
            if rank < 5:
                Niter = 130
            elif rank < 9:
                Niter = 60
            else:
                Niter = 30
            for it in range(Niter):
                if time.time() - t0 > T_TEMPL_END:
                    break
                sigma = 0.55 * np.exp(-it / 55.0) + 0.06
                # occasionally larger kick
                if rng.rand() < 0.08:
                    sigma = 0.35
                p_new = p_best + rng.normal(0, sigma, D)
                xc_n, yc_n = centres_from_template(p_new, counts)
                s_n, r_n = lp_eval(xc_n, yc_n)
                if s_n > s_best:
                    s_best = s_n; p_best = p_new
                    if s_best > best_sum:
                        best_sum = s_best; best_xc = xc_n.copy(); best_yc = yc_n.copy(); best_r = r_n.copy()
                        pool.append((best_sum, xc_n.copy(), yc_n.copy()))
            # also try symmetric variant: enforce vertical symmetry by averaging? quick try mirror
            # small jitter break ties attempt
            if rank < 5 and time.time() - t0 < T_TEMPL_END:
                # break row ties: jitter centres slightly and score
                xc_j = best_xc.copy() if best_xc is not None else xc_b
                # not needed
                pass
        pool = sorted(pool, key=lambda t: -t[0])[:6]

    # ---- Phase H: centre-only LP hill climbing from template seeds ----
    if best_xc is None:
        best_xc = fb[:, 0].copy(); best_yc = fb[:, 1].copy()
        best_sum, best_r = lp_eval(best_xc, best_yc)
    # start from best pool
    pool = sorted(pool, key=lambda t: -t[0])[:6]
    cur_xc = pool[0][1].copy(); cur_yc = pool[0][2].copy()
    cur_sum, cur_r = lp_eval(cur_xc, cur_yc)
    if cur_sum < 0:
        cur_xc, cur_yc = best_xc.copy(), best_yc.copy()
        cur_sum, cur_r = best_sum, best_r.copy()
    if cur_sum > best_sum:
        best_sum = cur_sum; best_xc = cur_xc.copy(); best_yc = cur_yc.copy(); best_r = cur_r.copy()

    step = 0
    noimp = 0
    # add slight break of exact row alignment for top seeds to allow gradient moves
    # jitter current slightly once
    cur_xc = np.clip(cur_xc + rng.normal(0, 0.0015, n), 0.003, 0.997)
    cur_yc = np.clip(cur_yc + rng.normal(0, 0.0015, n), 0.003, 0.997)
    s_try, r_try = lp_eval(cur_xc, cur_yc)
    if s_try > cur_sum:
        cur_sum = s_try; cur_r = r_try
        if cur_sum > best_sum:
            best_sum = cur_sum; best_xc = cur_xc.copy(); best_yc = cur_yc.copy(); best_r = cur_r.copy()

    while time.time() - t0 < T_HILL_END:
        base = 0.032 * np.exp(-step / 4200.0) + 0.0007
        if noimp > 900:
            base = 0.011
        if noimp > 2500:
            # restart from pool member with perturbation or best
            if time.time() - t0 > T_HILL_END - 4:
                break
            pi = rng.randint(len(pool))
            cur_xc = pool[pi][1] + rng.normal(0, 0.008, n)
            cur_yc = pool[pi][2] + rng.normal(0, 0.008, n)
            cur_xc = np.clip(cur_xc, 0.003, 0.997)
            cur_yc = np.clip(cur_yc, 0.003, 0.997)
            cur_sum, cur_r = lp_eval(cur_xc, cur_yc)
            if cur_sum < 0:
                cur_xc, cur_yc = best_xc.copy(), best_yc.copy()
                cur_sum, cur_r = best_sum, best_r.copy()
            noimp = 0
            step += 1
            continue
        u = rng.rand()
        txc = cur_xc.copy(); tyc = cur_yc.copy()
        if u < 0.68:
            i = rng.randint(n)
            sigma = base
            txc[i] = np.clip(cur_xc[i] + rng.normal(0, sigma), 0.003, 0.997)
            tyc[i] = np.clip(cur_yc[i] + rng.normal(0, sigma), 0.003, 0.997)
        elif u < 0.80:
            i = rng.randint(n); j = rng.randint(n)
            sigma = base
            txc[i] = np.clip(cur_xc[i] + rng.normal(0, sigma), 0.003, 0.997)
            tyc[i] = np.clip(cur_yc[i] + rng.normal(0, sigma), 0.003, 0.997)
            txc[j] = np.clip(cur_xc[j] + rng.normal(0, sigma), 0.003, 0.997)
            tyc[j] = np.clip(cur_yc[j] + rng.normal(0, sigma), 0.003, 0.997)
        elif u < 0.88:
            sigma = base * 0.35
            txc = np.clip(cur_xc + rng.normal(0, sigma, n), 0.003, 0.997)
            tyc = np.clip(cur_yc + rng.normal(0, sigma, n), 0.003, 0.997)
        elif u < 0.94:
            # swap two centres (topology change)
            i = rng.randint(n); j = rng.randint(n)
            if i != j:
                txc[i], txc[j] = cur_xc[j], cur_xc[i]
                tyc[i], tyc[j] = cur_yc[j], cur_yc[i]
                # small jitter after swap
                txc[i] = np.clip(txc[i] + rng.normal(0, base * 0.5), 0.003, 0.997)
                tyc[i] = np.clip(tyc[i] + rng.normal(0, base * 0.5), 0.003, 0.997)
            else:
                continue
        else:
            # teleport one circle to random void
            i = rng.randint(n)
            txc[i] = rng.uniform(0.02, 0.98)
            tyc[i] = rng.uniform(0.02, 0.98)
        s, r = lp_eval(txc, tyc)
        step += 1
        if s > cur_sum + 1e-12:
            cur_xc, cur_yc, cur_sum, cur_r = txc, tyc, s, r
            noimp = 0
            if s > best_sum + 1e-12:
                best_sum = s; best_xc = txc.copy(); best_yc = tyc.copy(); best_r = r.copy()
                pool.append((best_sum, txc.copy(), tyc.copy()))
                pool = sorted(pool, key=lambda t: -t[0])[:6]
        else:
            noimp += 1

    # ---- Phase J: short joint penalized L-BFGS-B polish + re-LP ----
    if time.time() - t0 < DEADLINE - 6 and has_opt:
        try:
            X0 = best_xc.copy(); Y0 = best_yc.copy(); R0 = best_r.copy()
            R0 = np.clip(R0, 0.005, 0.20)
            v0 = np.concatenate([X0, Y0, R0])
            bounds78 = [(0.0, 1.0)] * (2 * n) + [(0.005, 0.20)] * n
            # single weight schedule with limited iters to fit time
            for W in [500.0, 5000.0, 50000.0]:
                if time.time() - t0 > DEADLINE - 5:
                    break
                def fun(vv, W=W):
                    X = vv[0:n]; Y = vv[n:2 * n]; R = vv[2 * n:3 * n]
                    pen = 0.0
                    gx = np.zeros(n); gy = np.zeros(n); grp = np.zeros(n)
                    v1 = R - X
                    m1 = v1 > 0
                    if np.any(m1):
                        pen += np.sum(v1[m1] * v1[m1])
                        gx[m1] += -2.0 * v1[m1]
                        grp[m1] += 2.0 * v1[m1]
                    v2 = X + R - 1.0
                    m2 = v2 > 0
                    if np.any(m2):
                        pen += np.sum(v2[m2] * v2[m2])
                        gx[m2] += 2.0 * v2[m2]
                        grp[m2] += 2.0 * v2[m2]
                    v3 = R - Y
                    m3 = v3 > 0
                    if np.any(m3):
                        pen += np.sum(v3[m3] * v3[m3])
                        gy[m3] += -2.0 * v3[m3]
                        grp[m3] += 2.0 * v3[m3]
                    v4 = Y + R - 1.0
                    m4 = v4 > 0
                    if np.any(m4):
                        pen += np.sum(v4[m4] * v4[m4])
                        gy[m4] += 2.0 * v4[m4]
                        grp[m4] += 2.0 * v4[m4]
                    Dx = X[:, None] - X[None, :]
                    Dy = Y[:, None] - Y[None, :]
                    D = np.sqrt(Dx * Dx + Dy * Dy + 1e-16)
                    S = R[:, None] + R[None, :]
                    V = S - D
                    np.fill_diagonal(V, 0.0)
                    Vp = np.maximum(V, 0.0)
                    np.fill_diagonal(Vp, 0.0)
                    pen += 0.25 * np.sum(Vp * Vp)
                    np.fill_diagonal(D, 1.0)
                    Q = Vp / D
                    gx += -np.sum(Q * Dx, axis=1)
                    gy += -np.sum(Q * Dy, axis=1)
                    grp += np.sum(Vp, axis=1)
                    loss = -np.sum(R) + W * pen
                    g = np.empty_like(vv)
                    g[0:n] = W * gx
                    g[n:2 * n] = W * gy
                    g[2 * n:3 * n] = -1.0 + W * grp
                    return loss, g
                res = minimize(fun, v0, method='L-BFGS-B', jac=True, bounds=bounds78,
                               options={'maxiter': 120, 'ftol': 1e-9, 'gtol': 1e-5})
                v0 = res.x
            Xf = np.clip(v0[0:n], 0.0, 1.0); Yf = np.clip(v0[n:2 * n], 0.0, 1.0)
            sf, rf = lp_eval(np.clip(Xf, 0.003, 0.997), np.clip(Yf, 0.003, 0.997))
            if sf is not None and sf > best_sum:
                best_sum = sf; best_xc = Xf.copy(); best_yc = Yf.copy(); best_r = rf.copy()
            # also try LP hill micro-polish around Xf for remaining time
            cur_xc, cur_yc, cur_sum, cur_r = best_xc.copy(), best_yc.copy(), best_sum, best_r.copy()
            while time.time() - t0 < DEADLINE - 2:
                i = rng.randint(n)
                sigma = 0.0012
                txc = cur_xc.copy(); tyc = cur_yc.copy()
                txc[i] = np.clip(cur_xc[i] + rng.normal(0, sigma), 0.003, 0.997)
                tyc[i] = np.clip(cur_yc[i] + rng.normal(0, sigma), 0.003, 0.997)
                s, r = lp_eval(txc, tyc)
                if s > cur_sum + 1e-12:
                    cur_xc, cur_yc, cur_sum, cur_r = txc, tyc, s, r
                    best_sum = cur_sum; best_xc = cur_xc.copy(); best_yc = cur_yc.copy(); best_r = cur_r.copy()
                else:
                    # count to break? just continue until time
                    pass
        except Exception:
            pass

    # final shrink for strict feasibility
    Xf = np.clip(best_xc, 0.0, 1.0); Yf = np.clip(best_yc, 0.0, 1.0)
    Rf = np.maximum(np.asarray(best_r, dtype=float), 1e-9)
    Rf = Rf * (1.0 - 2e-7)
    Rf = np.maximum(Rf, 1e-9)
    # verify and if needed uniform scale down until feasible with margin
    for _ in range(5):
        wall = np.minimum(np.minimum(Xf, 1.0 - Xf), np.minimum(Yf, 1.0 - Yf))
        Dx = Xf[:, None] - Xf[None, :]
        Dy = Yf[:, None] - Yf[None, :]
        D = np.sqrt(Dx * Dx + Dy * Dy)
        S = Rf[:, None] + Rf[None, :]
        # margins
        min_wall = np.min(wall - Rf)
        # pairwise margin ignoring diag
        np.fill_diagonal(D, np.inf)
        min_pair = np.min(D - S)
        if min_wall > 1e-9 and min_pair > 1e-9:
            break
        Rf = Rf * 0.9999
    out = np.column_stack([Xf, Yf, Rf])
    return out
