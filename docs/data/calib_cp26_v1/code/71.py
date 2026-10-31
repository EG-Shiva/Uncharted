import numpy as np
import time

def solve():
    import time as _tm
    import numpy as _np
    from scipy.optimize import minimize, linprog
    n = 26
    t0 = _tm.time()
    DEADLINE = 115.0
    # time budget for phase A (incumbent reproduction)
    PHASE_A = 25.0

    rng = _np.random.default_rng(0)

    # ---------- helpers ----------
    iu, ju = _np.triu_indices(n, 1)
    m_pair = len(iu)
    A_pair = _np.zeros((m_pair, n), dtype=float)
    _ar = _np.arange(m_pair)
    A_pair[_ar, iu] = 1.0
    A_pair[_ar, ju] = 1.0
    c_obj = -_np.ones(n, dtype=float)

    def uniform_scale(X, Y, R):
        R = _np.maximum(R, 1e-12)
        wall = _np.minimum(_np.minimum(X, 1.0 - X), _np.minimum(Y, 1.0 - Y))
        Dx = X[:, None] - X[None, :]
        Dy = Y[:, None] - Y[None, :]
        D = _np.sqrt(Dx * Dx + Dy * Dy)
        with _np.errstate(divide='ignore', invalid='ignore'):
            rw = wall / R
            rp = D / (R[:, None] + R[None, :])
        _np.fill_diagonal(rp, _np.inf)
        try:
            a = float(_np.min(rw)); b = float(_np.min(rp))
        except Exception:
            return R * 0.5, 0.5
        alpha = min(1.0, a, b)
        alpha = alpha * (1.0 - 1e-7) - 1e-12
        if not _np.isfinite(alpha) or alpha <= 0:
            alpha = 0.5
        return R * alpha, alpha

    # fallback grid
    k = int(_np.ceil(_np.sqrt(n)))
    r0f = 0.5 / k
    _pts = []
    for _i in range(k):
        for _j in range(k):
            if len(_pts) < n:
                _pts.append([(2 * _i + 1) * r0f, (2 * _j + 1) * r0f, r0f * 0.999])
    fallback = _np.array(_pts, dtype=float)
    Xfb, Yfb, Rfb = fallback[:, 0].copy(), fallback[:, 1].copy(), fallback[:, 2].copy()
    Rfb_s, _ = uniform_scale(Xfb, Yfb, Rfb)
    # checkpointed incumbent (strictly feasible via shrink)
    SHRINK = 1.0 - 1e-9
    best_x = Xfb.copy(); best_y = Yfb.copy(); best_r = Rfb_s.copy() * SHRINK
    best_sum = float(_np.sum(best_r))
    best_arr = _np.column_stack([best_x, best_y, best_r])

    def lp_opt(xc, yc, eps=1e-9):
        try:
            wall = _np.minimum(_np.minimum(xc, 1.0 - xc), _np.minimum(yc, 1.0 - yc))
            if _np.any(wall <= 1e-9):
                return None, -1.0
            dx = xc[iu] - xc[ju]
            dy = yc[iu] - yc[ju]
            dvec = _np.sqrt(dx * dx + dy * dy) - eps
            if _np.any(dvec <= 1e-12):
                return None, -1.0
            bounds = []
            for w in wall:
                ub = float(w - eps * 0.5)
                if ub <= 0:
                    return None, -1.0
                bounds.append((0.0, ub))
            res = linprog(c_obj, A_ub=A_pair, b_ub=dvec, bounds=bounds, method='highs')
            if not res.success:
                return None, -1.0
            r = _np.asarray(res.x, dtype=float).flatten() * (1.0 - 5e-9)
            r = _np.maximum(r, 0.0)
            return r, float(_np.sum(r))
        except Exception:
            return None, -1.0

    def is_strictly_feasible(X, Y, R):
        try:
            if _np.any(R <= 0):
                return False
            if _np.any(R - X > 1e-12): return False
            if _np.any(X + R - 1.0 > 1e-12): return False
            if _np.any(R - Y > 1e-12): return False
            if _np.any(Y + R - 1.0 > 1e-12): return False
            Dx = X[:, None] - X[None, :]
            Dy = Y[:, None] - Y[None, :]
            D = _np.sqrt(Dx * Dx + Dy * Dy)
            S = R[:, None] + R[None, :] - D
            _np.fill_diagonal(S, -1.0)
            if _np.any(S > 1e-12):
                return False
            return True
        except Exception:
            return False

    def checkpoint(Xc, Yc, Rc):
        nonlocal best_x, best_y, best_r, best_sum, best_arr
        try:
            Rc = _np.asarray(Rc, dtype=float).flatten()
            Xc = _np.asarray(Xc, dtype=float).flatten()
            Yc = _np.asarray(Yc, dtype=float).flatten()
            # uniform shrinking
            Rs = Rc * SHRINK
            # tiny extra guard for fp error
            Rs = _np.maximum(Rs, 1e-12)
            if not is_strictly_feasible(Xc, Yc, Rs):
                # try stronger shrink
                Rs2 = Rc * (1.0 - 1e-7)
                if is_strictly_feasible(Xc, Yc, Rs2):
                    Rs = Rs2
                else:
                    return False
            s = float(_np.sum(Rs))
            if s > best_sum + 1e-14:
                best_sum = s
                best_x = Xc.copy(); best_y = Yc.copy(); best_r = Rs.copy()
                best_arr = _np.column_stack([best_x, best_y, best_r])
                return True
            return False
        except Exception:
            return False

    # ---------- Phase A: program-6-like annealed penalty (<=25s) ----------
    def make_fun():
        def fun(v, W):
            X = v[0:n]; Y = v[n:2*n]; R = v[2*n:3*n]
            gx = _np.zeros(n); gy = _np.zeros(n)
            gr_pen = _np.zeros(n)
            pen = 0.0
            v1 = R - X
            m1 = v1 > 0
            if _np.any(m1):
                pen += _np.sum(v1[m1] ** 2)
                gx[m1] += -2.0 * v1[m1]; gr_pen[m1] += 2.0 * v1[m1]
            v2 = X + R - 1.0
            m2 = v2 > 0
            if _np.any(m2):
                pen += _np.sum(v2[m2] ** 2)
                gx[m2] += 2.0 * v2[m2]; gr_pen[m2] += 2.0 * v2[m2]
            v3 = R - Y
            m3 = v3 > 0
            if _np.any(m3):
                pen += _np.sum(v3[m3] ** 2)
                gy[m3] += -2.0 * v3[m3]; gr_pen[m3] += 2.0 * v3[m3]
            v4 = Y + R - 1.0
            m4 = v4 > 0
            if _np.any(m4):
                pen += _np.sum(v4[m4] ** 2)
                gy[m4] += 2.0 * v4[m4]; gr_pen[m4] += 2.0 * v4[m4]
            Dx = X[:, None] - X[None, :]
            Dy = Y[:, None] - Y[None, :]
            D = _np.sqrt(Dx * Dx + Dy * Dy + 1e-20)
            S = R[:, None] + R[None, :]
            V = S - D
            _np.fill_diagonal(V, 0.0)
            _np.fill_diagonal(D, 1.0)
            Mask = V > 0
            if _np.any(Mask):
                pen += 0.5 * _np.sum(V[Mask] ** 2)
                Q = _np.zeros((n, n), dtype=float)
                Q[Mask] = -2.0 * V[Mask] / D[Mask]
                gx += _np.sum(Q * Dx, axis=1)
                gy += _np.sum(Q * Dy, axis=1)
                G = _np.zeros((n, n), dtype=float)
                G[Mask] = 2.0 * V[Mask]
                gr_pen += _np.sum(G, axis=1)
            loss = -_np.sum(R) + W * pen
            g = _np.empty_like(v)
            g[0:n] = W * gx
            g[n:2*n] = W * gy
            g[2*n:3*n] = -1.0 + W * gr_pen
            return loss, g
        return fun
    penfun = make_fun()
    boundsJR = [(0.0, 1.0)] * (2 * n) + [(0.012, 0.20)] * n
    WsA = [1e2, 1e3, 1e4, 1e5]

    restart = 0
    try:
        while _tm.time() - t0 < PHASE_A and (_tm.time() - t0) < DEADLINE - 90:
            if restart > 60:
                break
            try:
                if restart == 0:
                    X0 = _np.zeros(n); Y0 = _np.zeros(n)
                    idx = 0
                    for ii in range(5):
                        for jj in range(5):
                            if idx < 25:
                                X0[idx] = (ii + 0.5) / 5.0 + rng.normal(0, 0.008)
                                Y0[idx] = (jj + 0.5) / 5.0 + rng.normal(0, 0.008)
                                idx += 1
                    X0[25] = rng.random() * 0.9 + 0.05
                    Y0[25] = rng.random() * 0.9 + 0.05
                    X0 = _np.clip(X0, 0.03, 0.97); Y0 = _np.clip(Y0, 0.03, 0.97)
                    R0 = _np.full(n, 0.095)
                    v = _np.concatenate([X0, Y0, R0])
                else:
                    X0 = rng.random(n) * 0.94 + 0.03
                    Y0 = rng.random(n) * 0.94 + 0.03
                    R0 = 0.07 + 0.04 * rng.random(n)
                    v = _np.concatenate([X0, Y0, R0])
                for W in WsA:
                    if _tm.time() - t0 > PHASE_A:
                        break
                    try:
                        res = minimize(penfun, v, args=(W,), jac=True, method='L-BFGS-B',
                                       bounds=boundsJR, options={'maxiter': 60, 'ftol': 1e-9, 'gtol': 1e-5})
                        v = res.x
                    except Exception:
                        break
                Xc = _np.clip(v[0:n], 0.02, 0.98); Yc = _np.clip(v[n:2*n], 0.02, 0.98)
                r, s = lp_opt(Xc, Yc)
                if r is not None and s > 0:
                    # checkpoint with shrink
                    checkpoint(Xc, Yc, r)
            except Exception:
                pass
            restart += 1
    except Exception:
        pass

    # record incumbent centres, active contacts within 1e-4 (for diagnostics / active set)
    try:
        cur_x = best_x.copy(); cur_y = best_y.copy(); cur_r = best_r.copy()
        # neighbour pairs within 1e-4
        Dx0 = cur_x[:, None] - cur_x[None, :]
        Dy0 = cur_y[:, None] - cur_y[None, :]
        D0 = _np.sqrt(Dx0 * Dx0 + Dy0 * Dy0)
        S0 = cur_r[:, None] + cur_r[None, :]
        Gap0 = D0 - S0
        wallL = cur_x - cur_r; wallR = 1.0 - cur_x - cur_r
        wallB = cur_y - cur_r; wallT = 1.0 - cur_y - cur_r
        # (stored implicitly; used to seed active set below)
    except Exception:
        cur_x = best_x.copy(); cur_y = best_y.copy(); cur_r = best_r.copy()

    # ---------- Phase B: SLSQP exact-constraint polishing ----------
    Nvar = 3 * n
    # constraint ordering: 4*n walls + m_pair
    Nc = 4 * n + m_pair
    bnds_slsqp = [(0.0, 1.0)] * (2 * n) + [(0.005, 0.25)] * n

    def obj_fun(v):
        return -_np.sum(v[2*n:3*n])
    def obj_jac(v):
        g = _np.zeros_like(v)
        g[2*n:3*n] = -1.0
        return g

    def cons_fun(v):
        X = v[0:n]; Y = v[n:2*n]; R = v[2*n:3*n]
        out = _np.empty(Nc, dtype=float)
        out[0:n] = X - R
        out[n:2*n] = 1.0 - X - R
        out[2*n:3*n] = Y - R
        out[3*n:4*n] = 1.0 - Y - R
        dx = X[iu] - X[ju]
        dy = Y[iu] - Y[ju]
        d = _np.sqrt(dx*dx + dy*dy)
        out[4*n:] = d - R[iu] - R[ju]
        return out

    def cons_jac(v):
        X = v[0:n]; Y = v[n:2*n]
        J = _np.zeros((Nc, Nvar), dtype=float)
        idx = _np.arange(n)
        # left: X - R
        J[idx, idx] = 1.0
        J[idx, 2*n+idx] = -1.0
        # right: 1 - X - R
        J[n+idx, idx] = -1.0
        J[n+idx, 2*n+idx] = -1.0
        # bottom: Y - R
        J[2*n+idx, n+idx] = 1.0
        J[2*n+idx, 2*n+idx] = -1.0
        # top
        J[3*n+idx, n+idx] = -1.0
        J[3*n+idx, 2*n+idx] = -1.0
        # pairs
        dx = X[iu] - X[ju]
        dy = Y[iu] - Y[ju]
        d = _np.sqrt(dx*dx + dy*dy)
        # avoid div0
        safe = d > 1e-12
        ux = _np.zeros_like(dx); uy = _np.zeros_like(dy)
        ux[safe] = dx[safe] / d[safe]
        uy[safe] = dy[safe] / d[safe]
        rows = 4*n + _np.arange(m_pair)
        J[rows, iu] = ux
        J[rows, ju] = -ux
        J[rows, n+iu] = uy
        J[rows, n+ju] = -uy
        J[rows, 2*n+iu] = -1.0
        J[rows, 2*n+ju] = -1.0
        return J

    cons_dict = {'type': 'ineq', 'fun': cons_fun, 'jac': cons_jac}

    # helper: active counts within tol
    def active_counts(X, Y, R, tol):
        cc = _np.zeros(n, dtype=int)
        gL = X - R; gR = 1.0 - X - R; gB = Y - R; gT = 1.0 - Y - R
        walls = _np.stack([gL, gR, gB, gT], axis=1)
        for i in range(n):
            cc[i] += int(_np.sum(walls[i] < tol))
        Dx = X[:, None] - X[None, :]
        Dy = Y[:, None] - Y[None, :]
        D = _np.sqrt(Dx*Dx + Dy*Dy)
        S = R[:, None] + R[None, :]
        Gap = D - S
        _np.fill_diagonal(Gap, _np.inf)
        for i in range(n):
            cc[i] += int(_np.sum(Gap[i] < tol))
        return cc, walls, Gap

    # alternate ten SLSQP rounds with LP re-opt
    try:
        v_cur = _np.concatenate([best_x, best_y, best_r])
    except Exception:
        v_cur = _np.concatenate([cur_x, cur_y, cur_r])

    for rnd in range(10):
        if _tm.time() - t0 > DEADLINE - 8:
            break
        # SLSQP round
        try:
            # ensure start feasible (use checkpointed best as start for first rounds, else perturbed)
            if rnd == 0:
                v_start = _np.concatenate([best_x, best_y, best_r])
            else:
                v_start = v_cur.copy()
            # clip to bounds
            v_start[0:n] = _np.clip(v_start[0:n], 0.0, 1.0)
            v_start[n:2*n] = _np.clip(v_start[n:2*n], 0.0, 1.0)
            v_start[2*n:3*n] = _np.clip(v_start[2*n:3*n], 0.005, 0.25)
            res = minimize(obj_fun, v_start, jac=obj_jac, method='SLSQP',
                           bounds=bnds_slsqp, constraints=[cons_dict],
                           options={'maxiter': 200, 'ftol': 1e-12, 'disp': False, 'eps': 1e-9})
            v_new = res.x
            Xn = _np.clip(v_new[0:n], 0.0, 1.0)
            Yn = _np.clip(v_new[n:2*n], 0.0, 1.0)
            Rn = _np.clip(v_new[2*n:3*n], 1e-9, 0.3)
            # checkpoint if strictly feasible (SLSQP may be slightly infeasible; checkpoint does shrink+check)
            # first try direct checkpoint of SLSQP radii
            try:
                checkpoint(Xn, Yn, Rn)
            except Exception:
                pass
            # LP re-optimization of radii for these centres
            try:
                rlp, slp = lp_opt(Xn, Yn)
                if rlp is not None:
                    checkpoint(Xn, Yn, rlp)
                    # update v_cur to LP optimum (shrunk? use unshrunk for next start but feasible)
                    # use LP radii slightly shrunk to stay feasible as start
                    v_cur = _np.concatenate([Xn, Yn, rlp * (1.0 - 1e-9)])
                else:
                    v_cur = _np.concatenate([Xn, Yn, Rn])
            except Exception:
                try:
                    v_cur = _np.concatenate([Xn, Yn, Rn])
                except Exception:
                    pass
        except Exception:
            # keep v_cur, try LP on best
            try:
                rlp, slp = lp_opt(best_x, best_y)
                if rlp is not None:
                    checkpoint(best_x, best_y, rlp)
            except Exception:
                pass
            continue

        # promote contacts active within 1e-7, perturb loose circles (<3 contacts)
        if _tm.time() - t0 > DEADLINE - 6:
            break
        try:
            Xc = v_cur[0:n].copy(); Yc = v_cur[n:2*n].copy(); Rc = v_cur[2*n:3*n].copy()
            # re-LP to get consistent radii for gap computation? use Rc
            cc, walls, Gap = active_counts(Xc, Yc, Rc, tol=1e-7)
            # also compute with 1e-4 for record (preferred active set concept)
            # perturb only centres of circles with <3 active contacts
            loose = _np.where(cc < 3)[0]
            if len(loose) > 0:
                for i in loose:
                    if _tm.time() - t0 > DEADLINE - 5:
                        break
                    # find largest slack among its 4 walls + pairs
                    gL = walls[i, 0]; gR = walls[i, 1]; gB = walls[i, 2]; gT = walls[i, 3]
                    # pair gaps for i
                    pg = Gap[i, :].copy()
                    pg[i] = -_np.inf
                    # largest slack value
                    max_wall = max(gL, gR, gB, gT)
                    # argmax pair
                    jmax = int(_np.argmax(pg))
                    max_pair = float(pg[jmax])
                    step = 0.0025
                    dxm = 0.0; dym = 0.0
                    if max_pair >= max_wall:
                        # move towards jmax to reduce that gap
                        dx_ = Xc[jmax] - Xc[i]; dy_ = Yc[jmax] - Yc[i]
                        dd = float(_np.sqrt(dx_*dx_ + dy_*dy_))
                        if dd > 1e-12:
                            dxm = dx_ / dd * step
                            dym = dy_ / dd * step
                        else:
                            dxm = rng.normal(0, 0.001); dym = rng.normal(0, 0.001)
                    else:
                        # move towards wall with largest slack
                        warg = int(_np.argmax(walls[i]))
                        if warg == 0:  # left: decrease X
                            dxm = -step; dym = rng.normal(0, 0.0005)
                        elif warg == 1:  # right: increase X
                            dxm = step; dym = rng.normal(0, 0.0005)
                        elif warg == 2:  # bottom
                            dxm = rng.normal(0, 0.0005); dym = -step
                        else:
                            dxm = rng.normal(0, 0.0005); dym = step
                    # add tiny random jitter
                    dxm += rng.normal(0, 0.0004); dym += rng.normal(0, 0.0004)
                    Xc[i] += dxm; Yc[i] += dym
                Xc = _np.clip(Xc, 0.02, 0.98); Yc = _np.clip(Yc, 0.02, 0.98)
                try:
                    r2, s2 = lp_opt(Xc, Yc)
                    if r2 is not None:
                        # don't checkpoint yet with possibly worse; set as next start
                        v_cur = _np.concatenate([Xc, Yc, r2 * (1.0 - 1e-9)])
                        # if it already improves, checkpoint
                        checkpoint(Xc, Yc, r2)
                    else:
                        # keep previous v_cur
                        pass
                except Exception:
                    pass
        except Exception:
            pass

    # final safety: ensure best_arr strictly feasible, else fallback
    try:
        if not is_strictly_feasible(best_x, best_y, best_r):
            Rs, _ = uniform_scale(best_x, best_y, _np.maximum(best_r, 1e-9))
            best_x = _np.clip(best_x, 0.0, 1.0); best_y = _np.clip(best_y, 0.0, 1.0)
            best_r = Rs * (1.0 - 1e-7)
            best_arr = _np.column_stack([best_x, best_y, best_r])
    except Exception:
        best_arr = _np.column_stack([best_x, best_y, best_r])

    # enforce deadline-independent return of checkpointed incumbent
    try:
        out = _np.asarray(best_arr, dtype=float).reshape(26, 3)
        # final clip + shrink guard
        out[:, 0] = _np.clip(out[:, 0], 0.0, 1.0)
        out[:, 1] = _np.clip(out[:, 1], 0.0, 1.0)
        out[:, 2] = _np.maximum(out[:, 2], 1e-9)
        return out
    except Exception:
        return _np.asarray(best_arr, dtype=float).reshape(26, 3)
