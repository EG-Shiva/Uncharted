import numpy as np
import time
from scipy.optimize import minimize, linprog

def solve():
    np.random.seed(0)
    n = 26
    t0 = time.time()
    DEADLINE = 122.0
    ROW_BUDGET = 52.0
    LB = 0.005
    EPS = 1e-9

    # ---------- LP optimal radii for fixed centres ----------
    def lp_radii(X, Y):
        wall = np.minimum(np.minimum(X, 1.0 - X), np.minimum(Y, 1.0 - Y))
        if np.any(wall < LB + 1e-12):
            return None
        DX = X[:, None] - X[None, :]
        DY = Y[:, None] - Y[None, :]
        D = np.sqrt(DX * DX + DY * DY)
        # quick infeasibility via lower bound
        # check upper triangle
        triu = D[np.triu_indices(n, 1)]
        if np.min(triu) < 2 * LB - 1e-12:
            return None
        m = n * (n - 1) // 2
        A_ub = np.zeros((m, n), dtype=float)
        b_ub = np.zeros(m, dtype=float)
        k = 0
        for i in range(n):
            Di = D[i]
            for j in range(i + 1, n):
                A_ub[k, i] = 1.0
                A_ub[k, j] = 1.0
                b_ub[k] = Di[j] - EPS
                k += 1
        c = -np.ones(n)
        bounds = [(LB, float(wi) - EPS) for wi in wall]
        for lo, hi in bounds:
            if hi < lo - 1e-12:
                return None
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        except Exception:
            return None
        if not res.success:
            return None
        return res.x

    def lp_sum(X, Y):
        r = lp_radii(X, Y)
        if r is None:
            return -1.0, None
        return float(np.sum(r)), r

    def uniform_shrink(X, Y, R):
        R = np.maximum(np.asarray(R, float), 1e-12)
        wall = np.minimum(np.minimum(X, 1.0 - X), np.minimum(Y, 1.0 - Y))
        a = np.min(wall / np.maximum(R, 1e-12))
        DX = X[:, None] - X[None, :]
        DY = Y[:, None] - Y[None, :]
        D = np.sqrt(DX * DX + DY * DY)
        S = R[:, None] + R[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = D / S
        np.fill_diagonal(ratio, np.inf)
        b = np.min(ratio)
        alpha = min(1.0, float(a), float(b))
        alpha = alpha * (1.0 - 1e-7) - 1e-12
        if not np.isfinite(alpha) or alpha <= 0:
            alpha = 0.5
        return R * alpha, alpha

    # best tracker
    k0 = int(np.ceil(np.sqrt(n)))
    r0 = 0.5 / k0
    pts = []
    for i in range(k0):
        for j in range(k0):
            if len(pts) < n:
                pts.append([(2 * i + 1) * r0, (2 * j + 1) * r0, r0 * 0.999])
    best_arr = np.array(pts, float)
    Rb_s, _ = uniform_shrink(best_arr[:, 0], best_arr[:, 1], best_arr[:, 2])
    best_arr = np.column_stack([best_arr[:, 0], best_arr[:, 1], Rb_s])
    best_sum = float(np.sum(Rb_s))

    def consider(X, Y, R):
        nonlocal best_sum, best_arr
        try:
            X = np.asarray(X, float).copy()
            Y = np.asarray(Y, float).copy()
            R = np.asarray(R, float).copy()
            Rs, _ = uniform_shrink(X, Y, R)
            s = float(np.sum(Rs))
            if s > best_sum:
                best_sum = s
                best_arr = np.column_stack([X, Y, Rs])
        except Exception:
            pass

    def consider_lp(X, Y):
        try:
            r = lp_radii(np.asarray(X, float), np.asarray(Y, float))
            if r is not None:
                # tiny safety shrink
                r = r * (1.0 - 5e-9)
                consider(X, Y, r)
                return float(np.sum(r))
        except Exception:
            pass
        return -1.0

    # ---------- row template machinery ----------
    templates = [
        (6, 5, 6, 5, 4),
        (5, 6, 5, 6, 4),
        (6, 5, 5, 5, 5),
        (5, 5, 6, 5, 5),
        (5, 5, 5, 6, 5),
        (5, 5, 5, 5, 6),
        (6, 5, 6, 4, 5),
        (4, 5, 6, 5, 6),
        (5, 6, 4, 6, 5),
        (6, 6, 5, 5, 4),
        (6, 5, 4, 5, 6),
        (5, 4, 6, 5, 6),
        (7, 6, 7, 6),
        (6, 7, 6, 7),
        (5, 4, 4, 5, 4, 4),
        (4, 4, 5, 4, 5, 4),
    ]

    def centers_from_params(p, counts):
        kk = len(counts)
        yv = p[0:kk]
        lv = p[kk:2 * kk]
        sv = p[2 * kk:3 * kk]
        wv = p[3 * kk:4 * kk]
        Xs, Ys = [], []
        for j, cj in enumerate(counts):
            yj = float(yv[j])
            lj = float(lv[j])
            sj = float(sv[j])
            wj = float(wv[j])
            for i in range(cj):
                dx = i - (cj - 1) / 2.0
                x = lj + i * sj + wj * dx * dx
                Xs.append(x)
                Ys.append(yj)
        return np.array(Xs), np.array(Ys)

    def row_fun(p, counts):
        try:
            X, Y = centers_from_params(p, counts)
            if np.any(X < 0.0) or np.any(X > 1.0) or np.any(Y < 0.0) or np.any(Y > 1.0):
                return 0.0
            s, _ = lp_sum(X, Y)
            if s < 0:
                return 0.0
            return -s
        except Exception:
            return 0.0

    row_bests = []  # (sum, X, Y, counts)

    # symmetric special case helper: counts must be palindromic
    def expand_symmetric(ph, counts):
        kk = len(counts)
        h = (kk + 1) // 2
        # ph layout: y_h, l_h, s_h, w_h
        yh = ph[0:h]; lh = ph[h:2*h]; sh = ph[2*h:3*h]; wh = ph[3*h:4*h]
        yv = np.zeros(kk); lv = np.zeros(kk); sv = np.zeros(kk); wv = np.zeros(kk)
        for j in range(kk):
            mj = min(j, kk - 1 - j)
            yv[j] = yh[mj]; lv[j] = lh[mj]; sv[j] = sh[mj]; wv[j] = wh[mj]
            if kk % 2 == 1 and j == kk // 2:
                pass
            elif j >= (kk + 1) // 2:
                yv[j] = 1.0 - yh[mj]
        return np.concatenate([yv, lv, sv, wv])

    for tc in templates:
        if time.time() - t0 > ROW_BUDGET:
            break
        kk = len(tc)
        dim = 4 * kk
        # init: equally spaced y, centred rows with stagger
        y0 = np.array([(j + 0.5) / kk for j in range(kk)])
        s0v = np.full(kk, 0.165)
        l0v = np.zeros(kk)
        for j, cj in enumerate(tc):
            l0v[j] = (1.0 - (cj - 1) * s0v[j]) / 2.0
            if j % 2 == 1:
                # stagger half pitch, keep inside
                l0v[j] += 0.02
                # re-centre clip
                if l0v[j] + (cj - 1) * s0v[j] > 0.95:
                    l0v[j] = 0.95 - (cj - 1) * s0v[j]
        w0v = np.zeros(kk)
        p0 = np.concatenate([y0, l0v, s0v, w0v])
        bnds = []
        for j in range(kk):
            bnds.append((0.04, 0.96))
        for j in range(kk):
            bnds.append((0.02, 0.45))
        for j in range(kk):
            bnds.append((0.09, 0.26))
        for j in range(kk):
            bnds.append((-0.015, 0.015))
        # adjust l upper for large counts
        try:
            res = minimize(row_fun, p0, args=(tc,), method='Powell',
                           bounds=bnds, options={'maxiter': 45, 'disp': False})
            pstar = res.x
            Xb, Yb = centers_from_params(pstar, tc)
            s, r = lp_sum(Xb, Yb)
            if s > 0:
                row_bests.append((s, Xb.copy(), Yb.copy(), tc))
                consider(Xb, Yb, r)
            # second jittered start (small)
            if time.time() - t0 < ROW_BUDGET - 3:
                rng = np.random.RandomState(hash(tc) % (2**31 - 1))
                p1 = p0 + rng.randn(dim) * np.array([0.02]*kk + [0.02]*kk + [0.01]*kk + [0.005]*kk)
                # clip to bounds for start
                for di in range(dim):
                    lo, hi = bnds[di]
                    p1[di] = min(max(p1[di], lo), hi)
                res2 = minimize(row_fun, p1, args=(tc,), method='Powell',
                                bounds=bnds, options={'maxiter': 30, 'disp': False})
                Xb2, Yb2 = centers_from_params(res2.x, tc)
                s2, r2 = lp_sum(Xb2, Yb2)
                if s2 > 0:
                    row_bests.append((s2, Xb2.copy(), Yb2.copy(), tc))
                    consider(Xb2, Yb2, r2)
        except Exception:
            continue
        # symmetric special case for palindromic templates
        if list(tc) == list(tc)[::-1] and (time.time() - t0 < ROW_BUDGET - 3):
            try:
                h = (kk + 1) // 2
                yh0 = np.array([(j + 0.5) / kk for j in range(h)])
                # mirror consistent: for upper half use y0
                lh0 = l0v[:h]; sh0 = s0v[:h]; wh0 = w0v[:h]
                ph0 = np.concatenate([yh0, lh0, sh0, wh0])
                hbnds = bnds[:h] + bnds[kk:kk+h] + bnds[2*kk:2*kk+h] + bnds[3*kk:3*kk+h]
                def sym_fun(ph, counts):
                    pfull = expand_symmetric(ph, counts)
                    return row_fun(pfull, counts)
                ress = minimize(sym_fun, ph0, args=(tc,), method='Powell',
                                bounds=hbnds, options={'maxiter': 30, 'disp': False})
                pfull = expand_symmetric(ress.x, tc)
                Xs, Ys = centers_from_params(pfull, tc)
                ss, rs = lp_sum(Xs, Ys)
                if ss > 0:
                    row_bests.append((ss, Xs.copy(), Ys.copy(), tc))
                    consider(Xs, Ys, rs)
            except Exception:
                pass

    row_bests.sort(key=lambda t: -t[0])
    seeds = []
    for s, Xb, Yb, tc in row_bests[:4]:
        seeds.append((Xb, Yb))
        # broken-tie jittered copies
        rng = np.random.RandomState(1000 + len(seeds))
        seeds.append((Xb + rng.randn(n) * 0.004, Yb + rng.randn(n) * 0.004))
    # clip seeds
    clipped = []
    for Xb, Yb in seeds:
        Xb = np.clip(np.asarray(Xb, float), 0.02, 0.98)
        Yb = np.clip(np.asarray(Yb, float), 0.02, 0.98)
        clipped.append((Xb, Yb))
    seeds = clipped

    # ---------- centre-only LP hill climbing on best rows ----------
    try:
        if seeds and (time.time() - t0 < DEADLINE - 45):
            # pick top 2
            for si in range(min(2, len(seeds))):
                if time.time() - t0 > DEADLINE - 45:
                    break
                Xc, Yc = seeds[si]
                cur_s, cur_r = lp_sum(Xc, Yc)
                if cur_s < 0:
                    continue
                consider(Xc, Yc, cur_r)
                rng = np.random.RandomState(777 + si)
                for sigma, iters in [(0.015, 60), (0.007, 60), (0.003, 60)]:
                    if time.time() - t0 > DEADLINE - 42:
                        break
                    for _ in range(iters):
                        if time.time() - t0 > DEADLINE - 42:
                            break
                        # move single circle or all
                        if rng.rand() < 0.5:
                            idx = rng.integers(0, n)
                            Xn, Yn = Xc.copy(), Yc.copy()
                            Xn[idx] += rng.normal(0, sigma)
                            Yn[idx] += rng.normal(0, sigma)
                        else:
                            Xn = Xc + rng.normal(0, sigma * 0.5, size=n)
                            Yn = Yc + rng.normal(0, sigma * 0.5, size=n)
                        Xn = np.clip(Xn, 0.01, 0.99)
                        Yn = np.clip(Yn, 0.01, 0.99)
                        ns, nr = lp_sum(Xn, Yn)
                        if ns > cur_s:
                            cur_s, Xc, Yc, cur_r = ns, Xn, Yn, nr
                            consider(Xc, Yc, cur_r)
                seeds.append((Xc, Yc))
    except Exception:
        pass

    # ---------- full joint SLSQP/L-BFGS polish (Direction 1 style) ----------
    def make_joint_fun(W):
        def fun(v):
            X = v[0:n]; Y = v[n:2*n]; R = v[2*n:3*n]
            loss = -np.sum(R)
            gx = np.zeros(n); gy = np.zeros(n); gr = -np.ones(n)
            pen = 0.0
            v1 = R - X
            m1 = v1 > 0
            if np.any(m1):
                pen += np.sum(v1[m1]**2); gx[m1] += -2.0*v1[m1]; gr[m1] += 2.0*v1[m1]
            v2 = X + R - 1.0
            m2 = v2 > 0
            if np.any(m2):
                pen += np.sum(v2[m2]**2); gx[m2] += 2.0*v2[m2]; gr[m2] += 2.0*v2[m2]
            v3 = R - Y
            m3 = v3 > 0
            if np.any(m3):
                pen += np.sum(v3[m3]**2); gy[m3] += -2.0*v3[m3]; gr[m3] += 2.0*v3[m3]
            v4 = Y + R - 1.0
            m4 = v4 > 0
            if np.any(m4):
                pen += np.sum(v4[m4]**2); gy[m4] += 2.0*v4[m4]; gr[m4] += 2.0*v4[m4]
            DX = X[:, None] - X[None, :]
            DY = Y[:, None] - Y[None, :]
            D = np.sqrt(DX*DX + DY*DY + 1e-20)
            S = R[:, None] + R[None, :]
            V = S - D
            np.fill_diagonal(V, 0.0)
            np.fill_diagonal(D, 1.0)
            Mask = V > 0
            if np.any(Mask):
                pen += 0.5*np.sum(V[Mask]**2)
                Q = np.zeros((n, n))
                Q[Mask] = -2.0*V[Mask]/D[Mask]
                gx += np.sum(Q*DX, axis=1)
                gy += np.sum(Q*DY, axis=1)
                G = np.zeros((n, n)); G[Mask] = 2.0*V[Mask]
                gr += np.sum(G, axis=1)
            # convert pen grads to full grad with weight W
            g = np.empty_like(v)
            g[0:n] = W*gx
            g[n:2*n] = W*gy
            # gr = -1 + pen_grad  => final = -1 + W*pen_grad
            pen_gr = gr + 1.0
            g[2*n:3*n] = -1.0 + W*pen_gr
            F = -np.sum(R) + W*pen
            return F, g
        return fun

    bounds_j = [(0.0, 1.0)]*(2*n) + [(0.015, 0.22)]*n
    Ws = [5e2, 5e3, 5e4, 5e5]

    # build start list: row seeds first, then a few random/grid
    starts = []
    for Xb, Yb in seeds[:6]:
        r = lp_radii(Xb, Yb)
        if r is None:
            r = np.full(n, 0.09)
        starts.append((Xb.copy(), Yb.copy(), np.clip(r, 0.02, 0.2)))
    rngG = np.random.RandomState(0)
    # add 2 generic random starts if time permits
    for k in range(3):
        starts.append((rngG.uniform(0.07, 0.93, n), rngG.uniform(0.07, 0.93, n),
                       0.08 + rngG.rand(n)*0.05))

    for si, (X0, Y0, R0) in enumerate(starts):
        if time.time() - t0 > DEADLINE - 6:
            break
        X0 = np.clip(np.asarray(X0, float), 0.05, 0.95)
        Y0 = np.clip(np.asarray(Y0, float), 0.05, 0.95)
        R0 = np.clip(np.asarray(R0, float), 0.02, 0.18)
        v = np.concatenate([X0, Y0, R0])
        for W in Ws:
            if time.time() - t0 > DEADLINE - 6:
                break
            fun = make_joint_fun(W)
            try:
                res = minimize(fun, v, jac=True, method='L-BFGS-B', bounds=bounds_j,
                               options={'maxiter': 220, 'ftol': 1e-10, 'gtol': 1e-6})
                v = res.x
            except Exception:
                break
        X = v[0:n]; Y = v[n:2*n]; R = v[2*n:3*n]
        consider(X, Y, R)
        consider_lp(X, Y)
        # small kick + re-polish at high W for first seeds
        if si < 2 and (time.time() - t0 < DEADLINE - 10):
            try:
                rng2 = np.random.RandomState(5000 + si)
                Xk = X + rng2.randn(n)*0.002
                Yk = Y + rng2.randn(n)*0.002
                rlp = lp_radii(X, Y)
                Rk = np.clip(rlp if rlp is not None else R, 0.02, 0.2)
                vk = np.concatenate([np.clip(Xk,0,1), np.clip(Yk,0,1), Rk])
                funH = make_joint_fun(5e5)
                res2 = minimize(funH, vk, jac=True, method='L-BFGS-B', bounds=bounds_j,
                                options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-6})
                v2 = res2.x
                consider(v2[0:n], v2[n:2*n], v2[2*n:3*n])
                consider_lp(v2[0:n], v2[n:2*n])
            except Exception:
                pass

    best_arr[:, 2] *= (1.0 - 1e-9)
    best_arr[:, 2] = np.maximum(best_arr[:, 2], 1e-6)
    return best_arr
