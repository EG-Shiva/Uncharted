import numpy as np
import time as _time

def solve():
    import time as _tm
    import numpy as _np
    from scipy.optimize import minimize, linprog
    n = 26
    t0 = _tm.time()
    DEADLINE = 115.0
    def elapsed():
        return _tm.time() - t0

    iu, ju = _np.triu_indices(n, 1)
    m = len(iu)
    A_ub0 = _np.zeros((m, n), dtype=float)
    _ar = _np.arange(m)
    A_ub0[_ar, iu] = 1.0
    A_ub0[_ar, ju] = 1.0
    c_obj = -_np.ones(n)

    def eval_lp(xc, yc, eps=2e-9):
        b = _np.minimum(_np.minimum(xc, 1.0 - xc), _np.minimum(yc, 1.0 - yc)) - eps
        b = _np.maximum(b, 0.0)
        dx = xc[iu] - xc[ju]
        dy = yc[iu] - yc[ju]
        dvec = _np.sqrt(dx*dx + dy*dy) - eps
        bounds = [(0.0, float(bi)) for bi in b]
        try:
            res = linprog(c_obj, A_ub=A_ub0, b_ub=dvec, bounds=bounds, method='highs')
        except Exception:
            return None, -1.0
        if not res.success or res.x is None:
            return None, -1.0
        r = _np.asarray(res.x, dtype=float).flatten()
        r = _np.maximum(r, 0.0)
        return r, float(_np.sum(r))

    def max_viol(xc, yc, rc):
        v = -1e100
        v = max(v, _np.max(rc - xc))
        v = max(v, _np.max(xc + rc - 1.0))
        v = max(v, _np.max(rc - yc))
        v = max(v, _np.max(yc + rc - 1.0))
        dx = xc[iu]-xc[ju]; dy = yc[iu]-yc[ju]
        d = _np.sqrt(dx*dx+dy*dy)
        v = max(v, float(_np.max(rc[iu]+rc[ju]-d)) if m>0 else -1)
        return float(v)

    def shrink_check(xc, yc, rc):
        # uniform shrinking by 1-1e-9 then extra if needed
        rc2 = rc * (1.0 - 1e-9)
        # ensure strictly feasible with margin 1e-9
        for _ in range(5):
            mv = max_viol(xc, yc, rc2)
            if mv < -1e-9:
                break
            rc2 = rc2 * (1.0 - 1e-7) - 1e-12
            rc2 = _np.maximum(rc2, 1e-6)
        return rc2

    # fallback grid
    k = int(_np.ceil(_np.sqrt(n)))
    rr0 = 0.5 / k
    pts = []
    for i in range(k):
        for j in range(k):
            if len(pts) < n:
                pts.append([(2*i+1)*rr0, (2*j+1)*rr0, rr0*0.999])
    fb = _np.array(pts, dtype=float)
    Xb0, Yb0, Rb0 = fb[:,0].copy(), fb[:,1].copy(), fb[:,2].copy()
    Rb0s = shrink_check(Xb0, Yb0, Rb0*0.99)
    # make strictly feasible: scale until feasible
    # simple uniform scale
    try:
        r_lp, s_lp = eval_lp(Xb0, Yb0)
        if r_lp is not None:
            r_lp = shrink_check(Xb0, Yb0, r_lp)
            s_lp = float(_np.sum(r_lp))
            best_x, best_y, best_r = Xb0.copy(), Yb0.copy(), r_lp.copy()
            best_sum = s_lp
        else:
            best_x, best_y, best_r = Xb0.copy(), Yb0.copy(), Rb0s.copy()
            best_sum = float(_np.sum(Rb0s))
    except Exception:
        best_x, best_y, best_r = Xb0.copy(), Yb0.copy(), Rb0s.copy()
        best_sum = float(_np.sum(Rb0s))

    best_arr = _np.column_stack([best_x, best_y, best_r])
    def checkpoint(xc, yc, rc):
        global_best = None
        nonlocal_best = None
        return None
    def try_checkpoint(xc, yc, rc):
        nonlocal best_x, best_y, best_r, best_sum, best_arr
        try:
            rc2 = shrink_check(_np.asarray(xc,float), _np.asarray(yc,float), _np.asarray(rc,float))
            xc2 = _np.clip(_np.asarray(xc,float), 0.0, 1.0)
            yc2 = _np.clip(_np.asarray(yc,float), 0.0, 1.0)
            # need centres inside [r,1-r] strictly; if violation, nudge centres inward
            # shrink already helps, but check
            mv = max_viol(xc2, yc2, rc2)
            if mv >= -1e-12:
                # further shrink
                rc2 = rc2 * 0.999999
                mv = max_viol(xc2, yc2, rc2)
                if mv >= -1e-12:
                    return False
            s = float(_np.sum(rc2))
            if s > best_sum + 1e-12:
                best_sum = s
                best_x = xc2.copy(); best_y = yc2.copy(); best_r = rc2.copy()
                best_arr = _np.column_stack([best_x, best_y, best_r])
                return True
            return False
        except Exception:
            return False

    # ---------- Phase 1: deterministic program-6-like pipeline, max 25 s ----------
    PHASE1_END = 25.0
    bnds_pen = [(0.0,1.0)]*(2*n)+[(0.03,0.20)]*n
    P_seq = [1e2, 1e3, 1e4, 1e5, 1e6]
    rng = _np.random.default_rng(0)

    def make_init(restart):
        if restart == 0:
            # perturbed 5x5 + one extra (program 6 style)
            X0 = _np.zeros(n); Y0 = _np.zeros(n)
            idx = 0
            for i in range(5):
                for j in range(5):
                    if idx < 25:
                        X0[idx] = (i+0.5)/5.0 + float(rng.standard_normal())*0.004
                        Y0[idx] = (j+0.5)/5.0 + float(rng.standard_normal())*0.004
                        idx += 1
            X0[25] = 0.5 + float(rng.standard_normal())*0.01
            Y0[25] = 0.5 + float(rng.standard_normal())*0.01
            X0 = _np.clip(X0, 0.05, 0.95); Y0 = _np.clip(Y0, 0.05, 0.95)
            R0 = _np.full(n, 0.085) + rng.standard_normal(n)*0.005
            R0 = _np.clip(R0, 0.03, 0.20)
            return X0, Y0, R0
        else:
            X0 = rng.uniform(0.05, 0.95, size=n)
            Y0 = rng.uniform(0.05, 0.95, size=n)
            R0 = rng.uniform(0.05, 0.11, size=n)
            return X0, Y0, R0

    def run_penalty_once(x0, y0, r0):
        z = _np.concatenate([x0, y0, r0])
        for P in P_seq:
            if elapsed() > PHASE1_END - 0.5:
                break
            def fg(zz, P=P):
                x = zz[0:n]; y = zz[n:2*n]; r = zz[2*n:3*n]
                gx = _np.zeros(n); gy = _np.zeros(n)
                gr_pen = _np.zeros(n)
                pen = 0.0
                v1 = r - x
                mm = v1 > 0
                if _np.any(mm):
                    pen += _np.sum(v1[mm]*v1[mm])
                    gx[mm] += -2.0*v1[mm]
                    gr_pen[mm] += 2.0*v1[mm]
                v2 = x + r - 1.0
                mm = v2 > 0
                if _np.any(mm):
                    pen += _np.sum(v2[mm]*v2[mm])
                    gx[mm] += 2.0*v2[mm]
                    gr_pen[mm] += 2.0*v2[mm]
                v3 = r - y
                mm = v3 > 0
                if _np.any(mm):
                    pen += _np.sum(v3[mm]*v3[mm])
                    gy[mm] += -2.0*v3[mm]
                    gr_pen[mm] += 2.0*v3[mm]
                v4 = y + r - 1.0
                mm = v4 > 0
                if _np.any(mm):
                    pen += _np.sum(v4[mm]*v4[mm])
                    gy[mm] += 2.0*v4[mm]
                    gr_pen[mm] += 2.0*v4[mm]
                Dx = x[:,None]-x[None,:]
                Dy = y[:,None]-y[None,:]
                D = _np.sqrt(Dx*Dx+Dy*Dy+1e-20)
                S = r[:,None]+r[None,:]-D
                _np.fill_diagonal(S, 0.0)
                _np.fill_diagonal(D, 1.0)
                Mask = S > 0
                if _np.any(Mask):
                    pen += 0.5*_np.sum(S[Mask]*S[Mask])
                    Q = _np.zeros((n,n)); Q[Mask] = -2.0*S[Mask]/D[Mask]
                    gx += _np.sum(Q*Dx, axis=1)
                    gy += _np.sum(Q*Dy, axis=1)
                    G = _np.zeros((n,n)); G[Mask] = 2.0*S[Mask]
                    gr_pen += _np.sum(G, axis=1)
                loss = -_np.sum(r) + P*pen
                g = _np.empty_like(zz)
                g[0:n] = P*gx
                g[n:2*n] = P*gy
                g[2*n:3*n] = -1.0 + P*gr_pen
                return loss, g
            try:
                res = minimize(fg, z, method='L-BFGS-B', jac=True, bounds=bnds_pen,
                               options={'maxiter': 60, 'ftol': 1e-9, 'gtol': 1e-5})
                z = res.x
            except Exception:
                break
            if elapsed() > PHASE1_END - 0.3:
                break
        return z

    restart = 0
    while elapsed() < PHASE1_END - 1.0 and restart < 40:
        try:
            X0, Y0, R0 = make_init(restart)
            z = run_penalty_once(X0, Y0, R0)
            xc = _np.clip(z[0:n], 0.0, 1.0); yc = _np.clip(z[n:2*n], 0.0, 1.0)
            bq = _np.minimum(_np.minimum(xc,1.0-xc), _np.minimum(yc,1.0-yc))
            if _np.sum(bq) <= best_sum - 0.05:
                restart += 1
                continue
            r_opt, s = eval_lp(xc, yc)
            if r_opt is not None and s > best_sum:
                try_checkpoint(xc, yc, r_opt)
        except Exception:
            pass
        restart += 1

    # record active wall contacts and neighbour pairs within 1e-4
    try:
        xc0, yc0, rc0 = best_x.copy(), best_y.copy(), best_r.copy()
        wall_slack = _np.stack([xc0-rc0, 1.0-xc0-rc0, yc0-rc0, 1.0-yc0-rc0], axis=1)
        dx0 = xc0[iu]-xc0[ju]; dy0 = yc0[iu]-yc0[ju]
        d0 = _np.sqrt(dx0*dx0+dy0*dy0)
        pair_slack = d0 - rc0[iu] - rc0[ju]
        active_wall_1e4 = wall_slack < 1e-4
        active_pair_1e4 = pair_slack < 1e-4
    except Exception:
        active_wall_1e4 = None
        active_pair_1e4 = None

    # ---------- Phase 2: exact SLSQP polishing, 10 rounds alternating with LP ----------
    Nvars = 3*n
    bnds_slsqp = [(0.0,1.0)]*(2*n) + [(1e-4,0.35)]*n

    def obj_fun(z):
        f = -_np.sum(z[2*n:3*n])
        g = _np.zeros_like(z)
        g[2*n:3*n] = -1.0
        return f, g

    def wall_fun(z):
        x = z[0:n]; y = z[n:2*n]; r = z[2*n:3*n]
        c1 = x - r
        c2 = 1.0 - x - r
        c3 = y - r
        c4 = 1.0 - y - r
        c = _np.concatenate([c1,c2,c3,c4])
        return c
    def wall_jac(z):
        J = _np.zeros((4*n, Nvars))
        for i in range(n):
            # c1_i = x_i - r_i
            J[i, i] = 1.0; J[i, 2*n+i] = -1.0
            # c2_i = 1 - x_i - r_i
            J[n+i, i] = -1.0; J[n+i, 2*n+i] = -1.0
            # c3_i = y_i - r_i
            J[2*n+i, n+i] = 1.0; J[2*n+i, 2*n+i] = -1.0
            # c4_i = 1 - y_i - r_i
            J[3*n+i, n+i] = -1.0; J[3*n+i, 2*n+i] = -1.0
        return J

    def pair_fun(z):
        x = z[0:n]; y = z[n:2*n]; r = z[2*n:3*n]
        dx = x[iu]-x[ju]; dy = y[iu]-y[ju]
        d = _np.sqrt(dx*dx+dy*dy+1e-24)
        return d - r[iu] - r[ju]
    def pair_jac(z):
        x = z[0:n]; y = z[n:2*n]
        dx = x[iu]-x[ju]; dy = y[iu]-y[ju]
        d = _np.sqrt(dx*dx+dy*dy+1e-24)
        d = _np.maximum(d, 1e-12)
        J = _np.zeros((m, Nvars))
        inv = 1.0/d
        gx = dx*inv; gy = dy*inv
        # vectorized scatter
        J[_np.arange(m), iu] = gx
        J[_np.arange(m), ju] = -gx
        J[_np.arange(m), n+iu] = gy
        J[_np.arange(m), n+ju] = -gy
        J[_np.arange(m), 2*n+iu] = -1.0
        J[_np.arange(m), 2*n+ju] = -1.0
        return J

    cur_x, cur_y, cur_r = best_x.copy(), best_y.copy(), best_r.copy()

    for rnd in range(10):
        if elapsed() > DEADLINE - 6.0:
            break
        z0 = _np.concatenate([cur_x, cur_y, cur_r])
        # SLSQP round
        try:
            cons = (
                {'type': 'ineq', 'fun': wall_fun, 'jac': wall_jac},
                {'type': 'ineq', 'fun': pair_fun, 'jac': pair_jac},
            )
            res = minimize(lambda zz: (float(-_np.sum(zz[2*n:3*n])), None) if False else obj_fun(zz),
                           z0, method='SLSQP', jac=True, bounds=bnds_slsqp,
                           constraints=list(cons),
                           options={'maxiter': 200, 'ftol': 1e-12, 'disp': False, 'eps': 1e-9})
            zf = res.x
            xf = _np.clip(zf[0:n], 0.0, 1.0); yf = _np.clip(zf[n:2*n], 0.0, 1.0)
            rf = _np.clip(zf[2*n:3*n], 1e-4, 0.35)
            # LP re-optimization of radii for these centres
            try:
                r2, s2 = eval_lp(xf, yf)
                if r2 is not None:
                    try_checkpoint(xf, yf, r2)
                    # also checkpoint raw SLSQP iterate (shrunk)
                    try_checkpoint(xf, yf, rf)
                    # choose best centres for next round: use LP centres if improved else SLSQP
                    # keep centres with higher LP sum
                    r_cur_lp, s_cur_lp = eval_lp(cur_x, cur_y)
                    if s2 >= (s_cur_lp if r_cur_lp is not None else -1):
                        cur_x, cur_y = xf.copy(), yf.copy()
                        # set cur_r to LP radii (for warm start)
                        cur_r = r2.copy()
                    # else keep cur as is (but still consider SLSQP move if feasible and better)
                    else:
                        # check raw SLSQP sum
                        mv = max_viol(xf, yf, shrink_check(xf,yf,rf))
                        if mv < -1e-12 and float(_np.sum(rf)) > best_sum:
                            try_checkpoint(xf, yf, rf)
                else:
                    # no LP, try raw
                    try_checkpoint(xf, yf, rf)
            except Exception:
                try:
                    try_checkpoint(xf, yf, rf)
                except Exception:
                    pass
        except Exception:
            pass

        if elapsed() > DEADLINE - 5.0:
            break

        # promote contacts active within 1e-7 into preferred set & count
        try:
            dx = cur_x[iu]-cur_x[ju]; dy = cur_y[iu]-cur_y[ju]
            d = _np.sqrt(dx*dx+dy*dy)
            # use current best radii? use cur_r
            pair_c = cur_r[iu]+cur_r[ju]-d  # <=0 feasible, active if >-1e-7
            wall_c = _np.stack([cur_r-cur_x, cur_x+cur_r-1.0, cur_r-cur_y, cur_y+cur_r-1.0], axis=1)
            deg = _np.zeros(n, dtype=int)
            for i in range(n):
                cnt = int(_np.sum(wall_c[i] > -1e-7))
                deg[i] += cnt
            # pair actives
            act_mask = pair_c > -1e-7
            act_idx = _np.where(act_mask)[0]
            for kk in act_idx:
                deg[iu[kk]] += 1
                deg[ju[kk]] += 1
            # perturb only centres with <3 active contacts
            step = 0.0015 * (0.7 ** rnd) + 0.0002
            rng2 = _np.random.RandomState(1000+rnd)
            nx, ny = cur_x.copy(), cur_y.copy()
            for i in range(n):
                if deg[i] < 3:
                    # find largest slack direction
                    sL = cur_x[i]-cur_r[i]; sR = 1.0-cur_x[i]-cur_r[i]
                    sB = cur_y[i]-cur_r[i]; sT = 1.0-cur_y[i]-cur_r[i]
                    # pair slacks
                    # distances to others
                    dxi = cur_x[i]-cur_x; dyi = cur_y[i]-cur_y
                    di = _np.sqrt(dxi*dxi+dyi*dyi)
                    di[i] = _np.inf
                    sj = di - cur_r[i] - cur_r
                    # largest slack
                    best_s = sL; direc = (-1.0, 0.0)
                    if sR > best_s:
                        best_s = sR; direc = (1.0, 0.0)
                    if sB > best_s:
                        best_s = sB; direc = (0.0, -1.0)
                    if sT > best_s:
                        best_s = sT; direc = (0.0, 1.0)
                    jstar = int(_np.argmax(sj))
                    if sj[jstar] > best_s:
                        dd = max(di[jstar], 1e-12)
                        direc = ((cur_x[jstar]-cur_x[i])/dd, (cur_y[jstar]-cur_y[i])/dd)
                    # move along direc to reduce largest slack
                    nx[i] += direc[0]*step + rng2.randn()*step*0.3
                    ny[i] += direc[1]*step + rng2.randn()*step*0.3
                    nx[i] = float(_np.clip(nx[i], 0.01, 0.99))
                    ny[i] = float(_np.clip(ny[i], 0.01, 0.99))
            # test perturbed centres via LP
            try:
                rp, sp = eval_lp(nx, ny)
                if rp is not None and sp > best_sum + 1e-12:
                    try_checkpoint(nx, ny, rp)
                # adopt perturbed centres for next SLSQP start regardless (to explore),
                # but only if not much worse (within 0.5%)
                if rp is not None:
                    rcur, scur = eval_lp(cur_x, cur_y)
                    if scur is None or sp > scur - 0.01:
                        cur_x, cur_y = nx.copy(), ny.copy()
                        cur_r = rp.copy()
                    else:
                        # small random kick anyway for low-degree circles
                        cur_x, cur_y = nx.copy(), ny.copy()
                        # keep cur_r as LP of new centres if feasible
                        cur_r = rp.copy()
                else:
                    cur_x, cur_y = nx.copy(), ny.copy()
            except Exception:
                pass
        except Exception:
            pass

        # refresh cur from best occasionally to safeguard incumbent
        try:
            if rnd % 3 == 2:
                cur_x, cur_y, cur_r = best_x.copy(), best_y.copy(), best_r.copy()
        except Exception:
            pass

    # final safety: return checkpointed incumbent
    try:
        out = _np.column_stack([best_x, best_y, best_r]).astype(float)
        # final strict check; if violated, shrink further
        mv = max_viol(out[:,0], out[:,1], out[:,2])
        if mv >= -1e-12:
            r2 = shrink_check(out[:,0], out[:,1], out[:,2])
            out[:,2] = r2
        # ensure shape and positivity
        out[:,2] = _np.maximum(out[:,2], 1e-6)
        return out
    except Exception:
        try:
            return best_arr.astype(float)
        except Exception:
            kk = int(_np.ceil(_np.sqrt(n)))
            rr = 0.5/kk
            pp = []
            for i in range(kk):
                for j in range(kk):
                    if len(pp) < n:
                        pp.append([(2*i+1)*rr,(2*j+1)*rr,rr*0.999*0.999])
            return _np.array(pp, dtype=float)
