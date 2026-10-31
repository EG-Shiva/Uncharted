import numpy as np
import time

def solve():
    n = 26
    t0 = time.time()
    DEADLINE = 115.0
    T_INC = 25.0  # incumbent reproduction budget

    # pair indices
    I, J = np.triu_indices(n, 1)
    m = len(I)
    A0 = np.zeros((m, n), dtype=float)
    A0[np.arange(m), I] = 1.0
    A0[np.arange(m), J] = 1.0
    cobj = -np.ones(n, dtype=float)

    try:
        from scipy.optimize import minimize, linprog
        has_opt = True
    except Exception:
        has_opt = False
        minimize = None
        linprog = None

    def lp_eval(xc, yc):
        try:
            xc = np.asarray(xc, float).flatten()
            yc = np.asarray(yc, float).flatten()
            if np.any(xc < 0) or np.any(xc > 1) or np.any(yc < 0) or np.any(yc > 1):
                return -1.0, None
            b = np.minimum(np.minimum(xc, 1.0 - xc), np.minimum(yc, 1.0 - yc))
            b = np.maximum(b, 0.0)
            dx = xc[I] - xc[J]
            dy = yc[I] - yc[J]
            d = np.sqrt(dx * dx + dy * dy)
            bnds = [(0.0, float(bi)) for bi in b]
            res = linprog(cobj, A_ub=A0, b_ub=d, bounds=bnds, method='highs')
            if res.status != 0 or res.x is None:
                return -1.0, None
            r = np.asarray(res.x, float).flatten()
            r = np.maximum(r, 0.0)
            return float(np.sum(r)), r
        except Exception:
            return -1.0, None

    def strictify(X, Y, R):
        try:
            X = np.asarray(X, float).flatten()
            Y = np.asarray(Y, float).flatten()
            R = np.asarray(R, float).flatten()
            R = np.maximum(R, 1e-12)
            Xc = np.clip(X, 0.0, 1.0)
            Yc = np.clip(Y, 0.0, 1.0)
            wall = np.minimum(np.minimum(Xc, 1.0 - Xc), np.minimum(Yc, 1.0 - Yc))
            # per-circle ratio
            with np.errstate(divide='ignore', invalid='ignore'):
                rw = wall / R
            a_w = float(np.min(rw)) if np.all(np.isfinite(rw)) else -1.0
            Dx = Xc[:, None] - Xc[None, :]
            Dy = Yc[:, None] - Yc[None, :]
            D = np.sqrt(Dx * Dx + Dy * Dy)
            S = R[:, None] + R[None, :]
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = D / S
            np.fill_diagonal(ratio, np.inf)
            a_p = float(np.min(ratio)) if np.all(np.isfinite(ratio)) else -1.0
            alpha = min(1.0, a_w, a_p)
            alpha = alpha * (1.0 - 1e-9) - 1e-12
            if not np.isfinite(alpha) or alpha <= 0:
                return None
            Rs = R * alpha
            Rs = np.maximum(Rs, 1e-9)
            # verify strict
            if np.any(Rs > Xc) or np.any(Xc + Rs > 1.0) or np.any(Rs > Yc) or np.any(Yc + Rs > 1.0):
                return None
            # pair verify with 1e-12 tolerance
            Dx2 = Xc[:, None] - Xc[None, :]
            Dy2 = Yc[:, None] - Yc[None, :]
            D2 = np.sqrt(Dx2 * Dx2 + Dy2 * Dy2)
            S2 = Rs[:, None] + Rs[None, :]
            # allow 1e-12 tolerance
            if np.any(D2 + 1e-12 < S2 - 1e-12):
                # need to be careful diagonal: D2=0, S2>0 -> would trigger; ignore diag
                tmp = D2 + np.eye(n) * 10.0
                if np.any(tmp + 1e-12 < S2):
                    return None
            return Xc, Yc, Rs, float(np.sum(Rs))
        except Exception:
            return None

    # fallback grid checkpoint
    k = int(np.ceil(np.sqrt(n)))
    rr0 = 0.5 / k
    fb = []
    for ix in range(k):
        for jy in range(k):
            if len(fb) < n:
                fb.append([(2 * ix + 1) * rr0, (2 * jy + 1) * rr0, rr0 * 0.999])
    fb_arr = np.array(fb, dtype=float)
    try:
        s_fb, r_fb = lp_eval(fb_arr[:, 0], fb_arr[:, 1]) if has_opt else (-1.0, None)
    except Exception:
        s_fb, r_fb = -1.0, None
    if s_fb is not None and s_fb > 0 and r_fb is not None:
        st = strictify(fb_arr[:, 0], fb_arr[:, 1], r_fb)
        if st is not None:
            best_sum = st[3]
            best_xc, best_yc, best_r = st[0], st[1], st[2]
        else:
            best_sum = -1.0
            best_xc = best_yc = best_r = None
    else:
        # uniform scaled grid
        Rb = fb_arr[:, 2] * 0.9
        st = strictify(fb_arr[:, 0], fb_arr[:, 1], Rb)
        if st is not None:
            best_sum = st[3]
            best_xc, best_yc, best_r = st[0], st[1], st[2]
        else:
            best_sum = -1.0
            best_xc = best_yc = best_r = None
            best_xc = fb_arr[:, 0].copy(); best_yc = fb_arr[:, 1].copy(); best_r = fb_arr[:, 2] * 0.5

    # checkpointed incumbent array (strictly feasible)
    if best_xc is not None:
        checkpoint_sum = best_sum
        checkpoint_arr = np.column_stack([best_xc, best_yc, best_r])
    else:
        checkpoint_sum = 0.1
        checkpoint_arr = fb_arr.copy()
        checkpoint_arr[:, 2] *= 0.5
        best_xc = checkpoint_arr[:, 0].copy(); best_yc = checkpoint_arr[:, 1].copy(); best_r = checkpoint_arr[:, 2].copy()
        best_sum = checkpoint_sum

    def try_checkpoint(X, Y, R):
        global_best = [checkpoint_sum]
        try:
            st = strictify(X, Y, R)
            if st is None:
                return False
            s = st[3]
            if s > checkpoint_arr[:, 2].sum() + 1e-12:
                # update outer via mutable
                checkpoint_arr[:, 0] = st[0]
                checkpoint_arr[:, 1] = st[1]
                checkpoint_arr[:, 2] = st[2]
                return True
            return False
        except Exception:
            return False

    # wrap checkpoint updates without global confusion: use list
    # Phase 1: reproduce incumbent near 2.631 using deterministic program-6 pipeline for <=25s
    if has_opt:
        rng = np.random.default_rng(0)
        # penalty fun factory
        def pen_fun(v, W):
            X = v[0:n]; Y = v[n:2 * n]; R = v[2 * n:3 * n]
            gx = np.zeros(n); gy = np.zeros(n); gr_pen = np.zeros(n)
            pen = 0.0
            v1 = R - X
            mm = v1 > 0
            if np.any(mm):
                pen += np.sum(v1[mm] * v1[mm])
                gx[mm] += -2.0 * v1[mm]
                gr_pen[mm] += 2.0 * v1[mm]
            v2 = X + R - 1.0
            mm = v2 > 0
            if np.any(mm):
                pen += np.sum(v2[mm] * v2[mm])
                gx[mm] += 2.0 * v2[mm]
                gr_pen[mm] += 2.0 * v2[mm]
            v3 = R - Y
            mm = v3 > 0
            if np.any(mm):
                pen += np.sum(v3[mm] * v3[mm])
                gy[mm] += -2.0 * v3[mm]
                gr_pen[mm] += 2.0 * v3[mm]
            v4 = Y + R - 1.0
            mm = v4 > 0
            if np.any(mm):
                pen += np.sum(v4[mm] * v4[mm])
                gy[mm] += 2.0 * v4[mm]
                gr_pen[mm] += 2.0 * v4[mm]
            Dx = X[:, None] - X[None, :]
            Dy = Y[:, None] - Y[None, :]
            D = np.sqrt(Dx * Dx + Dy * Dy + 1e-20)
            S = R[:, None] + R[None, :]
            V = S - D
            np.fill_diagonal(V, 0.0)
            np.fill_diagonal(D, 1.0)
            Mask = V > 0
            if np.any(Mask):
                pen += 0.5 * np.sum(V[Mask] * V[Mask])
                Q = np.zeros((n, n), float)
                Q[Mask] = -2.0 * V[Mask] / D[Mask]
                gx += np.sum(Q * Dx, axis=1)
                gy += np.sum(Q * Dy, axis=1)
                G = np.zeros((n, n), float)
                G[Mask] = 2.0 * V[Mask]
                gr_pen += np.sum(G, axis=1)
            loss = -np.sum(R) + W * pen
            g = np.empty_like(v)
            g[0:n] = W * gx
            g[n:2 * n] = W * gy
            g[2 * n:3 * n] = -1.0 + W * gr_pen
            return loss, g

        Ws = [1e2, 1e3, 1e4, 5e4]
        maxiters = [80, 80, 100, 120]
        bounds78 = [(0.0, 1.0)] * (2 * n) + [(0.03, 0.20)] * n
        restart = 0
        while time.time() - t0 < T_INC and restart < 60:
            if time.time() - t0 > T_INC - 2:
                break
            try:
                if restart == 0:
                    X0 = np.zeros(n); Y0 = np.zeros(n)
                    cnt = 0
                    for ix in range(5):
                        for jy in range(5):
                            if cnt < 25:
                                X0[cnt] = 0.1 + 0.2 * ix
                                Y0[cnt] = 0.1 + 0.2 * jy
                                cnt += 1
                    X0[25] = 0.5; Y0[25] = 0.5
                    X0 = X0 + rng.normal(0, 0.015, n)
                    Y0 = Y0 + rng.normal(0, 0.015, n)
                    X0 = np.clip(X0, 0.02, 0.98); Y0 = np.clip(Y0, 0.02, 0.98)
                    R0 = 0.09 + 0.02 * rng.random(n)
                else:
                    X0 = rng.uniform(0.03, 0.97, n)
                    Y0 = rng.uniform(0.03, 0.97, n)
                    R0 = 0.06 + 0.05 * rng.random(n)
                v = np.concatenate([X0, Y0, R0])
                v = np.clip(v, 0.0, 1.0)
                v[2 * n:] = np.clip(v[2 * n:], 0.03, 0.20)
                aborted = False
                for wi, W in enumerate(Ws):
                    if time.time() - t0 >= T_INC:
                        aborted = True
                        break
                    try:
                        res = minimize(pen_fun, v, args=(W,), method='L-BFGS-B', jac=True,
                                       bounds=bounds78,
                                       options={'maxiter': maxiters[wi], 'ftol': 1e-9, 'gtol': 1e-5})
                        v = res.x
                    except Exception:
                        aborted = True
                        break
                if not aborted:
                    X = np.clip(v[0:n], 0.0, 1.0); Y = np.clip(v[n:2 * n], 0.0, 1.0); R = np.clip(v[2 * n:], 0.03, 0.20)
                    try:
                        s_lp, r_lp = lp_eval(np.clip(X, 0.0, 1.0), np.clip(Y, 0.0, 1.0))
                    except Exception:
                        s_lp, r_lp = -1.0, None
                    if s_lp is not None and r_lp is not None and s_lp > 0:
                        st = strictify(X, r_lp is not None and Y or Y, r_lp)
                        # strictify uses X,Y,r_lp
                        try:
                            st2 = strictify(X, Y, r_lp)
                        except Exception:
                            st2 = None
                        if st2 is not None and st2[3] > checkpoint_arr[:, 2].sum() + 1e-12:
                            checkpoint_arr[:, 0] = st2[0]; checkpoint_arr[:, 1] = st2[1]; checkpoint_arr[:, 2] = st2[2]
                        if st2 is not None and st2[3] > best_sum:
                            best_sum = st2[3]; best_xc = st2[0].copy(); best_yc = st2[1].copy(); best_r = st2[2].copy()
            except Exception:
                pass
            restart += 1

    # record centres, radii, active contacts within 1e-4 (for preferred set)
    try:
        cur_xc = checkpoint_arr[:, 0].copy(); cur_yc = checkpoint_arr[:, 1].copy(); cur_r = checkpoint_arr[:, 2].copy()
        best_xc = cur_xc.copy(); best_yc = cur_yc.copy(); best_r = cur_r.copy()
        best_sum = float(np.sum(cur_r))
    except Exception:
        pass

    def contact_info(X, Y, R, tol):
        try:
            s_xmin = X - R; s_xmax = 1.0 - X - R; s_ymin = Y - R; s_ymax = 1.0 - Y - R
            Dx = X[:, None] - X[None, :]; Dy = Y[:, None] - Y[None, :]
            D = np.sqrt(Dx * Dx + Dy * Dy)
            S = R[:, None] + R[None, :]
            slack = D - S
            np.fill_diagonal(slack, np.inf)
            wall_active = [(s_xmin[i] <= tol, s_xmax[i] <= tol, s_ymin[i] <= tol, s_ymax[i] <= tol) for i in range(n)]
            pair_list = []
            for a in range(n):
                for b in range(a + 1, n):
                    if slack[a, b] <= tol:
                        pair_list.append((a, b))
            counts = np.zeros(n, dtype=int)
            for i in range(n):
                counts[i] += int(s_xmin[i] <= tol) + int(s_xmax[i] <= tol) + int(s_ymin[i] <= tol) + int(s_ymax[i] <= tol)
            for (a, b) in pair_list:
                counts[a] += 1; counts[b] += 1
            return counts, pair_list, slack, (s_xmin, s_xmax, s_ymin, s_ymax)
        except Exception:
            return np.full(n, 4), [], np.full((n, n), np.inf), (np.ones(n), np.ones(n), np.ones(n), np.ones(n))

    # preferred active set (within 1e-7)
    preferred_set = set()
    try:
        _c, _pl, _, _w = contact_info(best_xc, best_yc, best_r, 1e-4)
        _c7, _pl7, _, _ = contact_info(best_xc, best_yc, best_r, 1e-7)
        for (a, b) in _pl7:
            preferred_set.add((a, b))
    except Exception:
        pass

    # SLSQP exact-constraint polishing, 10 rounds alternating with LP
    if has_opt:
        # analytic Jacobians
        def obj_f(v):
            return -np.sum(v[2 * n:3 * n])
        def obj_j(v):
            g = np.zeros_like(v)
            g[2 * n:3 * n] = -1.0
            return g
        def wall_f(v):
            X = v[0:n]; Y = v[n:2 * n]; R = v[2 * n:3 * n]
            out = np.empty(4 * n, float)
            out[0:n] = X - R
            out[n:2 * n] = 1.0 - X - R
            out[2 * n:3 * n] = Y - R
            out[3 * n:4 * n] = 1.0 - Y - R
            return out
        def wall_j(v):
            Jm = np.zeros((4 * n, 3 * n), float)
            for i in range(n):
                Jm[i, i] = 1.0; Jm[i, 2 * n + i] = -1.0
                Jm[n + i, i] = -1.0; Jm[n + i, 2 * n + i] = -1.0
                Jm[2 * n + i, n + i] = 1.0; Jm[2 * n + i, 2 * n + i] = -1.0
                Jm[3 * n + i, n + i] = -1.0; Jm[3 * n + i, 2 * n + i] = -1.0
            return Jm
        def pair_f(v):
            X = v[0:n]; Y = v[n:2 * n]; R = v[2 * n:3 * n]
            dx = X[I] - X[J]; dy = Y[I] - Y[J]
            d = np.sqrt(dx * dx + dy * dy)
            return d - R[I] - R[J]
        def pair_j(v):
            X = v[0:n]; Y = v[n:2 * n]
            dx = X[I] - X[J]; dy = Y[I] - Y[J]
            d = np.sqrt(dx * dx + dy * dy)
            Jm = np.zeros((m, 3 * n), float)
            # avoid div by zero
            safe = d > 1e-12
            ux = np.zeros_like(d); uy = np.zeros_like(d)
            ux[safe] = dx[safe] / d[safe]
            uy[safe] = dy[safe] / d[safe]
            rows = np.arange(m)
            Jm[rows, I] = ux
            Jm[rows, J] = -ux
            Jm[rows, n + I] = uy
            Jm[rows, n + J] = -uy
            Jm[rows, 2 * n + I] = -1.0
            Jm[rows, 2 * n + J] = -1.0
            return Jm

        bounds_slsqp = [(0.0, 1.0)] * (2 * n) + [(0.005, 0.25)] * n
        # starting point from checkpoint (strictly feasible)
        cur_x = checkpoint_arr[:, 0].copy(); cur_y = checkpoint_arr[:, 1].copy(); cur_rr = checkpoint_arr[:, 2].copy()
        for rnd in range(10):
            if time.time() - t0 > DEADLINE - 8:
                break
            v0 = np.concatenate([cur_x, cur_y, cur_rr])
            # ensure bounds
            v0[0:2 * n] = np.clip(v0[0:2 * n], 0.0, 1.0)
            v0[2 * n:] = np.clip(v0[2 * n:], 0.005, 0.25)
            try:
                cons = [{'type': 'ineq', 'fun': wall_f, 'jac': wall_j},
                        {'type': 'ineq', 'fun': pair_f, 'jac': pair_j}]
                res = minimize(obj_f, v0, method='SLSQP', jac=obj_j, bounds=bounds_slsqp,
                               constraints=cons, options={'maxiter': 150, 'ftol': 1e-12, 'disp': False, 'eps': 1e-9})
                v1 = res.x
            except Exception:
                v1 = None
            if v1 is not None:
                try:
                    X1 = np.asarray(v1[0:n], float); Y1 = np.asarray(v1[n:2 * n], float); R1 = np.asarray(v1[2 * n:3 * n], float)
                    st = strictify(X1, Y1, R1)
                    if st is not None and st[3] > float(np.sum(checkpoint_arr[:, 2])) + 1e-12:
                        checkpoint_arr[:, 0] = st[0]; checkpoint_arr[:, 1] = st[1]; checkpoint_arr[:, 2] = st[2]
                    # LP re-optimization of radii for fixed centres
                    try:
                        s_lp, r_lp = lp_eval(np.clip(X1, 0.0, 1.0), np.clip(Y1, 0.0, 1.0))
                        if s_lp is not None and s_lp > 0 and r_lp is not None:
                            st2 = strictify(np.clip(X1, 0.0, 1.0), np.clip(Y1, 0.0, 1.0), r_lp)
                            if st2 is not None and st2[3] > float(np.sum(checkpoint_arr[:, 2])) + 1e-12:
                                checkpoint_arr[:, 0] = st2[0]; checkpoint_arr[:, 1] = st2[1]; checkpoint_arr[:, 2] = st2[2]
                            # choose better centre/radii as next start
                            # compare st vs st2
                            best_next = None
                            best_next_sum = -1
                            if st is not None:
                                best_next = (st[0], st[1], st[2]); best_next_sum = st[3]
                            if st2 is not None and st2[3] > best_next_sum:
                                best_next = (st2[0], st2[1], st2[2]); best_next_sum = st2[3]
                            if best_next is not None:
                                cur_x, cur_y, cur_rr = best_next[0].copy(), best_next[1].copy(), best_next[2].copy()
                            else:
                                cur_x, cur_y, cur_rr = np.clip(X1, 0.0, 1.0), np.clip(Y1, 0.0, 1.0), np.clip(R1, 0.005, 0.25)
                        else:
                            if st is not None:
                                cur_x, cur_y, cur_rr = st[0].copy(), st[1].copy(), st[2].copy()
                    except Exception:
                        if st is not None:
                            cur_x, cur_y, cur_rr = st[0].copy(), st[1].copy(), st[2].copy()
                    # promote contacts active within 1e-7
                    try:
                        counts, plist, slack_mat, walls = contact_info(cur_x, cur_y, cur_rr, 1e-7)
                        for (a, b) in plist:
                            preferred_set.add((a, b))
                        # perturb only centres of circles with <3 active contacts along direction reducing largest slack
                        s_xmin, s_xmax, s_ymin, s_ymax = walls
                        step = 0.004 * (0.7 ** rnd) + 0.0005
                        for i in range(n):
                            if counts[i] < 3:
                                # find largest slack among its walls and pairs
                                # walls
                                cand_val = -1e100
                                cand_dir = None
                                # wall candidates: value and move direction to reduce it
                                # xmin: slack = X-R, reduce by dx=-step
                                vals = [s_xmin[i], s_xmax[i], s_ymin[i], s_ymax[i]]
                                # pair slacks for i
                                row = slack_mat[i, :].copy()
                                # largest pair index
                                jmax = int(np.argmax(row)) if n > 1 else -1
                                jmax_val = float(row[jmax]) if jmax >= 0 else -1e100
                                wmax = max(vals)
                                if jmax_val > wmax:
                                    # move towards jmax
                                    dx_ = cur_x[jmax] - cur_x[i]; dy_ = cur_y[jmax] - cur_y[i]
                                    dd = np.sqrt(dx_ * dx_ + dy_ * dy_) + 1e-12
                                    cur_x[i] += step * dx_ / dd
                                    cur_y[i] += step * dy_ / dd
                                else:
                                    k = int(np.argmax(vals))
                                    if k == 0:
                                        cur_x[i] -= step
                                    elif k == 1:
                                        cur_x[i] += step
                                    elif k == 2:
                                        cur_y[i] -= step
                                    else:
                                        cur_y[i] += step
                                cur_x[i] = float(np.clip(cur_x[i], 0.002, 0.998))
                                cur_y[i] = float(np.clip(cur_y[i], 0.002, 0.998))
                        # re-LP after perturbation? will be done next round's LP; also quick LP now to keep cur_rr consistent
                        try:
                            s2, r2 = lp_eval(cur_x, cur_y)
                            if s2 is not None and s2 > 0 and r2 is not None:
                                st3 = strictify(cur_x, cur_y, r2)
                                if st3 is not None:
                                    if st3[3] > float(np.sum(checkpoint_arr[:, 2])) + 1e-12:
                                        checkpoint_arr[:, 0] = st3[0]; checkpoint_arr[:, 1] = st3[1]; checkpoint_arr[:, 2] = st3[2]
                                    cur_x, cur_y, cur_rr = st3[0].copy(), st3[1].copy(), st3[2].copy()
                        except Exception:
                            pass
                    except Exception:
                        pass
                except Exception:
                    pass
            # if SLSQP failed, try small random perturbation of low-contact circles to continue
            if v1 is None:
                try:
                    counts, _, _, _ = contact_info(cur_x, cur_y, cur_rr, 1e-7)
                    nrng = np.random.RandomState(1234 + rnd)
                    for i in range(n):
                        if counts[i] < 3:
                            cur_x[i] = np.clip(cur_x[i] + nrng.normal(0, 0.003), 0.002, 0.998)
                            cur_y[i] = np.clip(cur_y[i] + nrng.normal(0, 0.003), 0.002, 0.998)
                except Exception:
                    pass

    # final return: checkpointed incumbent (strictly feasible, shrunk)
    try:
        out = checkpoint_arr.copy()
        # final uniform shrinking by 1-1e-9 is already in strictify, ensure once more
        out[:, 2] = out[:, 2] * (1.0 - 1e-9)
        # clip centres just in case
        out[:, 0] = np.clip(out[:, 0], 0.0, 1.0)
        out[:, 1] = np.clip(out[:, 1], 0.0, 1.0)
        out[:, 2] = np.maximum(out[:, 2], 1e-9)
        # verify, if fails return previous checkpoint without extra shrink
        Xf, Yf, Rf = out[:, 0], out[:, 1], out[:, 2]
        wall = np.minimum(np.minimum(Xf, 1.0 - Xf), np.minimum(Yf, 1.0 - Yf))
        ok = np.all(wall + 1e-12 >= Rf)
        if ok:
            Dx = Xf[:, None] - Xf[None, :]; Dy = Yf[:, None] - Yf[None, :]
            D = np.sqrt(Dx * Dx + Dy * Dy)
            S = Rf[:, None] + Rf[None, :]
            np.fill_diagonal(D, 1e10)
            ok = np.all(D + 1e-12 >= S)
        if not ok:
            out = checkpoint_arr.copy()
        # final shape check
        assert out.shape == (26, 3)
        return out
    except Exception:
        try:
            return checkpoint_arr.copy()
        except Exception:
            k2 = int(np.ceil(np.sqrt(n)))
            rr = 0.5 / k2
            pts = []
            for i in range(k2):
                for j in range(k2):
                    if len(pts) < n:
                        pts.append([(2 * i + 1) * rr, (2 * j + 1) * rr, rr * 0.999 * 0.99])
            return np.array(pts, dtype=float)
