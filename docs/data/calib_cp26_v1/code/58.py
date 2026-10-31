import numpy as np
import time as _time

def solve():
    import time
    import numpy as np
    from scipy.optimize import minimize, linprog
    t0 = time.time()
    DEADLINE = 115.0
    n = 26
    rng = np.random.default_rng(0)

    # ---------- helpers ----------
    iu, ju = np.triu_indices(n, 1)
    m_pair = len(iu)
    A_ub0 = np.zeros((m_pair, n), dtype=float)
    _ar = np.arange(m_pair)
    A_ub0[_ar, iu] = 1.0
    A_ub0[_ar, ju] = 1.0
    c_obj = -np.ones(n)

    def uniform_shrink(X, Y, R):
        R = np.asarray(R, dtype=float).copy()
        R = np.maximum(R, 1e-12)
        wall = np.minimum(np.minimum(X, 1.0 - X), np.minimum(Y, 1.0 - Y))
        # wall ratio
        with np.errstate(divide='ignore', invalid='ignore'):
            wr = wall / R
        a = float(np.min(wr))
        Dx = X[:, None] - X[None, :]
        Dy = Y[:, None] - Y[None, :]
        D = np.sqrt(Dx * Dx + Dy * Dy)
        S = R[:, None] + R[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = D / S
        np.fill_diagonal(ratio, np.inf)
        b = float(np.min(ratio))
        alpha = min(1.0, a, b)
        alpha = alpha * (1.0 - 1e-9)
        # extra tiny guard to clear 1e-12 strictly
        alpha = alpha - 1e-12
        if not np.isfinite(alpha) or alpha <= 1e-6:
            return None, -1.0
        Rs = R * alpha
        Rs = np.maximum(Rs, 1e-9)
        # re-verify strictly (with 5e-12 margin)
        wall2 = np.minimum(np.minimum(X, 1.0 - X), np.minimum(Y, 1.0 - Y)) - Rs
        if np.min(wall2) < 1e-12:
            # shrink a bit more
            Rs = Rs * (1.0 - 1e-7)
            wall2 = np.minimum(np.minimum(X, 1.0 - X), np.minimum(Y, 1.0 - Y)) - Rs
            if np.min(wall2) < 1e-12:
                return None, -1.0
        Dx2 = X[iu] - X[ju]
        Dy2 = Y[iu] - Y[ju]
        Dv = np.sqrt(Dx2 * Dx2 + Dy2 * Dy2) - (Rs[iu] + Rs[ju])
        if np.min(Dv) < 1e-12:
            Rs = Rs * (1.0 - 1e-7)
            Dv = np.sqrt(Dx2 * Dx2 + Dy2 * Dy2) - (Rs[iu] + Rs[ju])
            if np.min(Dv) < 1e-12:
                return None, -1.0
        return Rs, float(np.sum(Rs))

    def lp_opt(X, Y):
        try:
            wall = np.minimum(np.minimum(X, 1.0 - X), np.minimum(Y, 1.0 - Y))
            wall = np.maximum(wall, 0.0)
            dx = X[iu] - X[ju]
            dy = Y[iu] - Y[ju]
            dvec = np.sqrt(dx * dx + dy * dy)
            bounds = [(0.0, float(bi)) for bi in wall]
            res = linprog(c_obj, A_ub=A_ub0, b_ub=dvec, bounds=bounds, method='highs')
            if not res.success or res.x is None:
                return None, -1.0
            r = np.asarray(res.x, dtype=float).flatten()
            r = np.maximum(r, 0.0)
            # uniform shrink for strict feasibility
            Rs, s = uniform_shrink(X, Y, np.maximum(r, 1e-9))
            if Rs is None:
                return None, -1.0
            return Rs, s
        except Exception:
            return None, -1.0

    # fallback grid
    k = int(np.ceil(np.sqrt(n)))
    r0 = 0.5 / k
    pts = []
    for i in range(k):
        for j in range(k):
            if len(pts) < n:
                pts.append([(2 * i + 1) * r0, (2 * j + 1) * r0, r0 * 0.999])
    best_arr = np.array(pts, dtype=float)
    Xb, Yb, Rb = best_arr[:, 0], best_arr[:, 1], best_arr[:, 2]
    Rs0, s0 = uniform_shrink(Xb, Yb, Rb)
    if Rs0 is not None:
        best_arr = np.column_stack([Xb, Yb, Rs0])
        best_sum = float(np.sum(Rs0))
    else:
        best_sum = -1.0

    # ---------- Phase 1: reproduce incumbent ~2.631, at most 25 s ----------
    PHASE1 = 25.0

    def make_fun(n_):
        def fun(v, W):
            X = v[0:n_]
            Y = v[n_:2 * n_]
            R = v[2 * n_:3 * n_]
            loss = -np.sum(R)
            gx = np.zeros(n_)
            gy = np.zeros(n_)
            gr = -np.ones(n_)
            v1 = R - X
            m1 = v1 > 0
            if np.any(m1):
                gx[m1] += -2.0 * v1[m1]
                gr[m1] += 2.0 * v1[m1]
            v2 = X + R - 1.0
            m2 = v2 > 0
            if np.any(m2):
                gx[m2] += 2.0 * v2[m2]
                gr[m2] += 2.0 * v2[m2]
            v3 = R - Y
            m3 = v3 > 0
            if np.any(m3):
                gy[m3] += -2.0 * v3[m3]
                gr[m3] += 2.0 * v3[m3]
            v4 = Y + R - 1.0
            m4 = v4 > 0
            if np.any(m4):
                gy[m4] += 2.0 * v4[m4]
                gr[m4] += 2.0 * v4[m4]
            pen = 0.0
            if np.any(m1):
                pen += np.sum(v1[m1] ** 2)
            if np.any(m2):
                pen += np.sum(v2[m2] ** 2)
            if np.any(m3):
                pen += np.sum(v3[m3] ** 2)
            if np.any(m4):
                pen += np.sum(v4[m4] ** 2)
            Dx = X[:, None] - X[None, :]
            Dy = Y[:, None] - Y[None, :]
            D = np.sqrt(Dx * Dx + Dy * Dy + 1e-20)
            S = R[:, None] + R[None, :]
            V = S - D
            np.fill_diagonal(V, 0.0)
            np.fill_diagonal(D, 1.0)
            Mask = V > 0
            if np.any(Mask):
                pen += 0.5 * np.sum(V[Mask] ** 2)
                Q = np.zeros((n_, n_))
                Q[Mask] = -2.0 * V[Mask] / D[Mask]
                Hx = Q * Dx
                Hy = Q * Dy
                gx += np.sum(Hx, axis=1)
                gy += np.sum(Hy, axis=1)
                G = np.zeros((n_, n_))
                G[Mask] = 2.0 * V[Mask]
                pen_gr = np.sum(G, axis=1)
            else:
                pen_gr = np.zeros(n_)
            # assemble grads: gx,gy are pen grads; pen grad r = contributions from walls + pen_gr
            # recompute wall pen grad for r cleanly:
            pgr = np.zeros(n_)
            if np.any(m1):
                pgr[m1] += 2.0 * v1[m1]
            if np.any(m2):
                pgr[m2] += 2.0 * v2[m2]
            if np.any(m3):
                pgr[m3] += 2.0 * v3[m3]
            if np.any(m4):
                pgr[m4] += 2.0 * v4[m4]
            pgr += pen_gr
            # wall gx,gy pen grads already in gx,gy (they are pen grads)
            # But gx currently holds wall+pair pen grads (we added). Good.
            loss = -np.sum(R) + W * pen
            g = np.empty_like(v)
            g[0:n_] = W * gx
            g[n_:2 * n_] = W * gy
            g[2 * n_:3 * n_] = -1.0 + W * pgr
            return loss, g
        return fun

    fun = make_fun(n)
    Ws = [1e2, 1e3, 1e4, 1e5, 1e6]
    bounds_pb = [(0.0, 1.0)] * (2 * n) + [(0.03, 0.20)] * n
    restart = 0
    while True:
        if time.time() - t0 > PHASE1:
            break
        if restart > 60:
            break
        if time.time() - t0 > DEADLINE - 30:
            break
        if restart == 0:
            X0 = np.zeros(n); Y0 = np.zeros(n); R0 = np.full(n, 0.095)
            idx = 0
            for i in range(5):
                for j in range(5):
                    if idx < n:
                        X0[idx] = (i + 0.5) / 5.0 + rng.normal(0, 0.005)
                        Y0[idx] = (j + 0.5) / 5.0 + rng.normal(0, 0.005)
                        idx += 1
            if n > 25:
                X0[25] = 0.9 + rng.normal(0, 0.01)
                Y0[25] = 0.1 + rng.normal(0, 0.01)
            X0 = np.clip(X0, 0.05, 0.95); Y0 = np.clip(Y0, 0.05, 0.95)
        elif restart == 1:
            pts_l = []
            for ix in range(6):
                for iy in range(5):
                    if len(pts_l) < n:
                        xx = (ix + 0.5) / 6.0
                        yy = (iy + 0.5) / 5.0
                        if iy % 2 == 1:
                            xx += 0.5 / 6.0 * 0.5
                            if xx > 0.95:
                                xx -= 1.0 / 6.0
                        pts_l.append((xx, yy))
            arr = np.array(pts_l)
            X0 = arr[:, 0] + rng.normal(0, 0.005, n)
            Y0 = arr[:, 1] + rng.normal(0, 0.005, n)
            X0 = np.clip(X0, 0.05, 0.95); Y0 = np.clip(Y0, 0.05, 0.95)
            R0 = 0.085 + 0.02 * rng.random(n)
        else:
            X0 = rng.random(n) * 0.9 + 0.05
            Y0 = rng.random(n) * 0.9 + 0.05
            R0 = 0.085 + 0.025 * rng.random(n)
        v = np.concatenate([X0, Y0, R0])
        for W in Ws:
            if time.time() - t0 > PHASE1:
                break
            try:
                res = minimize(fun, v, args=(W,), jac=True, method='L-BFGS-B',
                               bounds=bounds_pb,
                               options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-7})
                v = res.x
            except Exception:
                break
        try:
            X = np.clip(v[0:n], 0.0, 1.0)
            Y = np.clip(v[n:2 * n], 0.0, 1.0)
            R = np.clip(v[2 * n:3 * n], 0.01, 0.25)
            Rs, _ = uniform_shrink(X, Y, R)
            if Rs is not None:
                s_uni = float(np.sum(Rs))
                if s_uni > best_sum:
                    best_sum = s_uni
                    best_arr = np.column_stack([X, Y, Rs])
        except Exception:
            pass
        try:
            X = np.clip(v[0:n], 0.0, 1.0)
            Y = np.clip(v[n:2 * n], 0.0, 1.0)
            Rlp, s_lp = lp_opt(X, Y)
            if Rlp is not None and s_lp > best_sum:
                best_sum = float(s_lp)
                best_arr = np.column_stack([X, Y, Rlp])
        except Exception:
            pass
        restart += 1

    # record incumbent centres, active contacts within 1e-4
    try:
        Xc = best_arr[:, 0].copy(); Yc = best_arr[:, 1].copy(); Rc = best_arr[:, 2].copy()
    except Exception:
        Xc = best_arr[:, 0]; Yc = best_arr[:, 1]; Rc = best_arr[:, 2]
    # active sets (for bookkeeping)
    active_pairs = set()
    try:
        Dx0 = Xc[iu] - Xc[ju]; Dy0 = Yc[iu] - Yc[ju]
        D0 = np.sqrt(Dx0 * Dx0 + Dy0 * Dy0)
        slack0 = D0 - (Rc[iu] + Rc[ju])
        for k in range(m_pair):
            if slack0[k] <= 1e-4:
                active_pairs.add((int(iu[k]), int(ju[k])))
    except Exception:
        pass

    # ---------- Phase 2: exact SLSQP polishing (10 rounds) ----------
    # objective
    def obj_f(v):
        s = float(np.sum(v[2 * n:3 * n]))
        return -s

    def obj_j(v):
        g = np.zeros_like(v)
        g[2 * n:3 * n] = -1.0
        return g

    def wall_f(v):
        X = v[0:n]; Y = v[n:2 * n]; R = v[2 * n:3 * n]
        out = np.empty(4 * n)
        out[0:n] = X - R
        out[n:2 * n] = 1.0 - X - R
        out[2 * n:3 * n] = Y - R
        out[3 * n:4 * n] = 1.0 - Y - R
        return out

    def wall_j(v):
        J = np.zeros((4 * n, 3 * n))
        idx = np.arange(n)
        J[idx, idx] = 1.0
        J[idx, 2 * n + idx] = -1.0
        J[n + idx, idx] = -1.0
        J[n + idx, 2 * n + idx] = -1.0
        J[2 * n + idx, n + idx] = 1.0
        J[2 * n + idx, 2 * n + idx] = -1.0
        J[3 * n + idx, n + idx] = -1.0
        J[3 * n + idx, 2 * n + idx] = -1.0
        return J

    def pair_f(v):
        X = v[0:n]; Y = v[n:2 * n]; R = v[2 * n:3 * n]
        dx = X[iu] - X[ju]
        dy = Y[iu] - Y[ju]
        D = np.sqrt(dx * dx + dy * dy)
        return D - (R[iu] + R[ju])

    def pair_j(v):
        X = v[0:n]; Y = v[n:2 * n]
        dx = X[iu] - X[ju]
        dy = Y[iu] - Y[ju]
        D = np.sqrt(dx * dx + dy * dy)
        # avoid div by zero
        Dsafe = np.maximum(D, 1e-12)
        ux = dx / Dsafe
        uy = dy / Dsafe
        # zero out where D tiny
        tiny = D < 1e-12
        ux[tiny] = 0.0
        uy[tiny] = 0.0
        J = np.zeros((m_pair, 3 * n))
        rows = np.arange(m_pair)
        J[rows, iu] = ux
        J[rows, ju] = -ux
        J[rows, n + iu] = uy
        J[rows, n + ju] = -uy
        J[rows, 2 * n + iu] = -1.0
        J[rows, 2 * n + ju] = -1.0
        return J

    cons = [{'type': 'ineq', 'fun': wall_f, 'jac': wall_j},
            {'type': 'ineq', 'fun': pair_f, 'jac': pair_j}]
    bounds_slsqp = [(0.0, 1.0)] * (2 * n) + [(0.005, 0.30)] * n

    # current iterate = checkpointed incumbent
    curX, curY, curR = Xc.copy(), Yc.copy(), Rc.copy()
    v_start = np.concatenate([curX, curY, curR])

    # deterministic perturbation rng
    prng = np.random.RandomState(12345)

    for rnd in range(10):
        if time.time() - t0 > DEADLINE - 8:
            break
        # --- SLSQP round ---
        try:
            res = minimize(lambda vv: (float(np.sum(vv[2 * n:3 * n])) * -1.0, obj_j(vv)),
                           v_start, jac=True, method='SLSQP',
                           bounds=bounds_slsqp, constraints=cons,
                           options={'maxiter': 200, 'ftol': 1e-9, 'disp': False, 'eps': 1e-8})
            v_new = np.asarray(res.x, dtype=float).flatten()
        except Exception:
            v_new = v_start.copy()
        # checkpoint SLSQP iterate via uniform shrink
        try:
            Xn = np.clip(v_new[0:n], 0.0, 1.0)
            Yn = np.clip(v_new[n:2 * n], 0.0, 1.0)
            Rn = np.clip(v_new[2 * n:3 * n], 1e-6, 0.35)
            Rsh, ssh = uniform_shrink(Xn, Yn, Rn)
            if Rsh is not None and ssh > best_sum:
                best_sum = float(ssh)
                best_arr = np.column_stack([Xn, Yn, Rsh])
                curX, curY, curR = Xn.copy(), Yn.copy(), Rsh.copy()
            else:
                # keep iterate for LP even if not improved
                curX, curY = Xn.copy(), Yn.copy()
                # keep radii as shrunk if feasible else old
                if Rsh is not None:
                    curR = Rsh.copy()
        except Exception:
            pass
        # --- LP re-optimization of radii ---
        try:
            Rlp2, s_lp2 = lp_opt(curX, curY)
            if Rlp2 is not None and s_lp2 > best_sum:
                best_sum = float(s_lp2)
                best_arr = np.column_stack([curX, curY, Rlp2])
                curR = Rlp2.copy()
            elif Rlp2 is not None:
                # use LP radii as next base (they are optimal for centres)
                curR = Rlp2.copy()
        except Exception:
            pass
        # --- promote contacts active within 1e-7 ---
        try:
            Dx = curX[iu] - curX[ju]; Dy = curY[iu] - curY[ju]
            Dv = np.sqrt(Dx * Dx + Dy * Dy)
            slack = Dv - (curR[iu] + curR[ju])
            for k in range(m_pair):
                if slack[k] <= 1e-7:
                    active_pairs.add((int(iu[k]), int(ju[k])))
            # counts per circle
            wall_slacks = np.stack([curX - curR, 1.0 - curX - curR,
                                    curY - curR, 1.0 - curY - curR], axis=1)
            cnt = np.sum(wall_slacks <= 1e-7, axis=1)
            # pair active counts
            pair_active = slack <= 1e-7
            for k in np.where(pair_active)[0]:
                cnt[iu[k]] += 1
                cnt[ju[k]] += 1
        except Exception:
            cnt = np.full(n, 3)
        # --- perturb only centres with <3 active contacts ---
        try:
            step0 = 0.004 * (0.85 ** rnd)
            Xn2 = curX.copy(); Yn2 = curY.copy()
            # precompute nearest for direction
            DxM = curX[:, None] - curX[None, :]
            DyM = curY[:, None] - curY[None, :]
            DM = np.sqrt(DxM * DxM + DyM * DyM)
            np.fill_diagonal(DM, np.inf)
            for i in range(n):
                if cnt[i] < 3:
                    # local candidates: 4 walls + 5 nearest neighbours
                    wsl = [(curX[i] - curR[i], (-1.0, 0.0)),
                           (1.0 - curX[i] - curR[i], (1.0, 0.0)),
                           (curY[i] - curR[i], (0.0, -1.0)),
                           (1.0 - curY[i] - curR[i], (0.0, 1.0))]
                    order = np.argsort(DM[i])[:5]
                    best_gap = None
                    best_dir = None
                    # find largest slack among local set
                    for name_idx, (sl, d) in enumerate(wsl):
                        if best_gap is None or sl > best_gap:
                            best_gap = sl
                            best_dir = d
                    best_j = -1
                    for j in order:
                        sl = float(np.sqrt((curX[i] - curX[j]) ** 2 + (curY[i] - curY[j]) ** 2) - (curR[i] + curR[j]))
                        if sl > best_gap:
                            best_gap = sl
                            dx_ = curX[j] - curX[i]
                            dy_ = curY[j] - curY[i]
                            nn = float(np.sqrt(dx_ * dx_ + dy_ * dy_) + 1e-12)
                            best_dir = (dx_ / nn, dy_ / nn)
                            best_j = j
                    # move along best_dir to reduce largest slack
                    if best_dir is not None:
                        st = step0 * (0.5 + prng.rand() * 0.5)
                        Xn2[i] += st * best_dir[0]
                        Yn2[i] += st * best_dir[1]
                    else:
                        Xn2[i] += prng.normal(0, step0 * 0.5)
                        Yn2[i] += prng.normal(0, step0 * 0.5)
            Xn2 = np.clip(Xn2, 0.01, 0.99)
            Yn2 = np.clip(Yn2, 0.01, 0.99)
            # next start uses perturbed centres + current radii
            v_start = np.concatenate([Xn2, Yn2, curR])
        except Exception:
            v_start = np.concatenate([curX, curY, curR])

    # final safety: ensure best_arr strictly feasible via shrink
    try:
        Xf = np.asarray(best_arr[:, 0], dtype=float)
        Yf = np.asarray(best_arr[:, 1], dtype=float)
        Rf = np.asarray(best_arr[:, 2], dtype=float)
        Rfs, _ = uniform_shrink(Xf, Yf, Rf)
        if Rfs is not None:
            best_arr = np.column_stack([Xf, Yf, Rfs])
    except Exception:
        pass
    return np.asarray(best_arr, dtype=float)
