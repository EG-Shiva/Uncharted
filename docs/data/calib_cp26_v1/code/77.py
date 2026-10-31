import numpy as np
import time as _time

def solve():
    import time as _tm
    import numpy as _np
    from scipy.optimize import minimize, linprog
    n = 26
    t0 = _tm.time()
    HARD = 124.0
    PHASE1_END = 30.0  # template screening budget
    PHASE2_END = 52.0  # homotopy budget end (from t0)

    rng = _np.random.RandomState(7)

    iu, ju = _np.triu_indices(n, 1)
    m = len(iu)
    A0 = _np.zeros((m, n), dtype=float)
    _ar = _np.arange(m)
    A0[_ar, iu] = 1.0
    A0[_ar, ju] = 1.0
    c_obj = -_np.ones(n, dtype=float)

    def eval_lp(xc, yc, lb_idx=-1, lb_val=0.0):
        try:
            b = _np.minimum(_np.minimum(xc, 1.0 - xc), _np.minimum(yc, 1.0 - yc))
            b = _np.maximum(b, 0.0)
            if lb_idx >= 0:
                if b[lb_idx] < lb_val:
                    return -1.0, None
            dx = xc[iu] - xc[ju]
            dy = yc[iu] - yc[ju]
            dvec = _np.sqrt(dx*dx + dy*dy)
            dvec = _np.maximum(dvec, 1e-12)
            bounds = [(0.0, float(bi)) for bi in b]
            if lb_idx >= 0:
                bounds[lb_idx] = (float(lb_val), float(b[lb_idx]))
            res = linprog(c_obj, A_ub=A0, b_ub=dvec, bounds=bounds, method='highs')
            if res.status != 0 or res.x is None:
                return -1.0, None
            r = _np.asarray(res.x, dtype=float).flatten()
            r = _np.maximum(r, 0.0)
            return float(_np.sum(r)), r
        except Exception:
            return -1.0, None

    def run_penalty(x0, y0, r0, P_seq, maxiter_each, r_low_val=0.015, r_low_arr=None, r_high=0.24):
        if r_low_arr is None:
            rl = _np.full(n, r_low_val, dtype=float)
        else:
            rl = _np.asarray(r_low_arr, dtype=float)
        bnds = [(0.0, 1.0)]*(2*n) + [(float(rl[i]), float(r_high)) for i in range(n)]
        z = _np.concatenate([_np.asarray(x0, dtype=float), _np.asarray(y0, dtype=float), _np.asarray(r0, dtype=float)])
        # clip r0 into bounds
        z[2*n:3*n] = _np.clip(z[2*n:3*n], rl+1e-9, r_high-1e-9)
        z[0:2*n] = _np.clip(z[0:2*n], 0.0, 1.0)
        for P in P_seq:
            if _tm.time() - t0 > HARD - 8.0:
                break
            def fg(zz, P=P):
                x = zz[0:n]; y = zz[n:2*n]; r = zz[2*n:3*n]
                f = -_np.sum(r)
                gx = _np.zeros(n); gy = _np.zeros(n); gr = -_np.ones(n)
                v1 = r - x
                mm = v1 > 0
                if _np.any(mm):
                    f += P*_np.sum(v1[mm]*v1[mm])
                    co = 2.0*P*v1[mm]
                    gx[mm] -= co; gr[mm] += co
                v2 = x + r - 1.0
                mm = v2 > 0
                if _np.any(mm):
                    f += P*_np.sum(v2[mm]*v2[mm])
                    co = 2.0*P*v2[mm]
                    gx[mm] += co; gr[mm] += co
                v3 = r - y
                mm = v3 > 0
                if _np.any(mm):
                    f += P*_np.sum(v3[mm]*v3[mm])
                    co = 2.0*P*v3[mm]
                    gy[mm] -= co; gr[mm] += co
                v4 = y + r - 1.0
                mm = v4 > 0
                if _np.any(mm):
                    f += P*_np.sum(v4[mm]*v4[mm])
                    co = 2.0*P*v4[mm]
                    gy[mm] += co; gr[mm] += co
                dxm = x[:, None] - x[None, :]
                dym = y[:, None] - y[None, :]
                D = _np.sqrt(dxm*dxm + dym*dym)
                Rsum = r[:, None] + r[None, :]
                S = Rsum - D
                _np.fill_diagonal(S, -1.0)
                mask = S > 0
                if _np.any(mask):
                    f += P*0.5*_np.sum(S[mask]*S[mask])
                    Dm = D + 1e-12
                    W = _np.zeros((n, n))
                    W[mask] = (2.0*P*S[mask])/Dm[mask]
                    rs = W.sum(axis=1)
                    gx += W.dot(x) - rs*x
                    gy += W.dot(y) - rs*y
                    Spos = _np.where(mask, S, 0.0)
                    gr += 2.0*P*Spos.sum(axis=1)
                g = _np.concatenate([gx, gy, gr])
                return f, g
            try:
                res = minimize(fg, z, method='L-BFGS-B', jac=True, bounds=bnds,
                               options={'maxiter': maxiter_each, 'ftol': 1e-10, 'gtol': 1e-6})
                z = _np.asarray(res.x, dtype=float)
            except Exception:
                break
        return z[0:n], z[n:2*n], z[2*n:3*n]

    def gen_centres(part, orient, yv, off):
        xc = _np.zeros(n); yc = _np.zeros(n)
        idx = 0
        if orient == 'row':
            for k, c in enumerate(part):
                yk = float(yv[k]); ox = float(off[k])
                for j in range(c):
                    bx = (j + 0.5)/float(c) + ox
                    if bx < 0.02: bx = 0.02
                    if bx > 0.98: bx = 0.98
                    xc[idx] = bx; yc[idx] = yk
                    idx += 1
        else:
            for k, c in enumerate(part):
                xk = (k + 0.5)/5.0
                oy = float(off[k]); 
                # yv[k] unused as x? reuse yv as x offset? For col, yv holds x positions
                xk2 = float(yv[k])  # allow x to move
                # Actually use yv as x positions for col orientation
                for j in range(c):
                    by = (j + 0.5)/float(c) + oy
                    if by < 0.02: by = 0.02
                    if by > 0.98: by = 0.98
                    xc[idx] = xk2; yc[idx] = by
                    idx += 1
        xc = _np.clip(xc, 0.03, 0.97)
        yc = _np.clip(yc, 0.03, 0.97)
        return xc, yc

    # global best (fallback grid)
    k = int(_np.ceil(_np.sqrt(n)))
    r0f = 0.5/k
    fb = []
    for i in range(k):
        for j in range(k):
            if len(fb) < n:
                fb.append([(2*i+1)*r0f, (2*j+1)*r0f, r0f*0.999*0.999])
    best_arr = _np.array(fb, dtype=float)
    best_xc = best_arr[:,0].copy(); best_yc = best_arr[:,1].copy(); best_r = best_arr[:,2].copy()
    best_sum = float(_np.sum(best_r))

    def consider(xc, yc, r):
        global_best = [best_sum]
        return None

    def try_update(xc, yc, r, s):
        nonlocal best_sum, best_xc, best_yc, best_r
        if r is None: return False
        if s > best_sum:
            best_sum = float(s)
            best_xc = xc.copy(); best_yc = yc.copy(); best_r = r.copy()
            return True
        return False

    # ---------- Phase 1: structured defect templates ----------
    parts = [[6,5,6,5,4],[6,6,5,5,4],[5,6,5,5,5],[4,6,5,6,5],[6,5,4,6,5]]
    templates = []
    for p in parts:
        templates.append({'part': p, 'orient': 'row'})
    for p in parts:
        templates.append({'part': p, 'orient': 'col'})
    T = len(templates)
    # per-template state
    t_y = []
    t_off = []
    t_score = []
    t_xc = []
    t_yc = []
    t_r = []
    for ti in range(T):
        y0 = _np.array([(kk+0.5)/5.0 for kk in range(5)], dtype=float)
        o0 = _np.zeros(5, dtype=float)
        xc0, yc0 = gen_centres(templates[ti]['part'], templates[ti]['orient'], y0, o0)
        s0, r0 = eval_lp(xc0, yc0)
        if s0 < 0: s0 = -1.0; r0 = None
        t_y.append(y0); t_off.append(o0); t_score.append(float(s0))
        t_xc.append(xc0); t_yc.append(yc0); t_r.append(r0)
        if r0 is not None and s0 > best_sum:
            try_update(xc0, yc0, r0, s0)

    # stochastic hill climbing on low-dim params, round robin, total <=35s
    sig_y0 = 0.025; sig_o0 = 0.015
    it = 0
    while _tm.time() - t0 < PHASE1_END - 2.0:
        if _tm.time() - t0 > HARD - 70:
            break
        ti = it % T
        # decay
        decay = 1.0/(1.0 + it*0.001)
        sig_y = sig_y0*decay + 0.004
        sig_o = sig_o0*decay + 0.002
        # propose
        by = t_y[ti]; bo = t_off[ti]
        ny = by + rng.normal(0, sig_y, size=5)
        no = bo + rng.normal(0, sig_o, size=5)
        ny = _np.clip(ny, 0.06, 0.94)
        ny = _np.sort(ny)
        # keep separation
        # enforce min gap 0.08
        for kk in range(1,5):
            if ny[kk] - ny[kk-1] < 0.07:
                ny[kk] = ny[kk-1] + 0.07
        # rescale if overflow
        if ny[4] > 0.94:
            shift = ny[4]-0.94
            ny -= shift
            ny = _np.clip(ny, 0.06, 0.94)
            ny = _np.sort(ny)
        no = _np.clip(no, -0.07, 0.07)
        xc, yc = gen_centres(templates[ti]['part'], templates[ti]['orient'], ny, no)
        s, r = eval_lp(xc, yc)
        it += 1
        if s > t_score[ti]:
            t_score[ti] = float(s)
            t_y[ti] = ny; t_off[ti] = no
            t_xc[ti] = xc; t_yc[ti] = yc; t_r[ti] = r
            if r is not None:
                try_update(xc, yc, r, s)
        # occasional random restart of worst template
        if it % 200 == 0:
            # re-evaluate global to avoid stall: perturb best template more
            pass
        if it > 8000:
            break

    # pick best two templates
    order = _np.argsort(_np.array(t_score))[::-1]
    best_two = [int(order[0]), int(order[1])] if T >= 2 else [int(order[0])]

    # ---------- Phase 2: homotopy from 5x5 ----------
    homo_xc = None; homo_yc = None; homo_r = None; homo_sum = -1
    try:
        # 25 grid centres
        g25x = []; g25y = []
        for ix in range(5):
            for jy in range(5):
                g25x.append((ix+0.5)/5.0); g25y.append((jy+0.5)/5.0)
        g25x = _np.array(g25x); g25y = _np.array(g25y)
        # LP for 25 to get radii for void search (build quick via equal)
        # approximate radii 0.095
        rapp = _np.full(25, 0.095)
        # find largest void by dense sampling
        gridN = 60
        xs = _np.linspace(0.02, 0.98, gridN)
        ys = _np.linspace(0.02, 0.98, gridN)
        best_clear = -1; bxv = 0.5; byv = 0.5
        for gx in xs[::2]:
            # vectorize over ys
            for gy in ys[::2]:
                wall = min(gx, 1-gx, gy, 1-gy)
                d = _np.sqrt((g25x-gx)**2 + (g25y-gy)**2) - rapp
                cl = min(float(_np.min(d)), float(wall))
                if cl > best_clear:
                    best_clear = cl; bxv = float(gx); byv = float(gy)
        # refine around best with finer search
        for _ in range(200):
            cx = bxv + rng.normal(0, 0.02); cy = byv + rng.normal(0, 0.02)
            cx = min(max(cx, 0.02), 0.98); cy = min(max(cy, 0.02), 0.98)
            wall = min(cx, 1-cx, cy, 1-cy)
            d = _np.sqrt((g25x-cx)**2 + (g25y-cy)**2) - rapp
            cl = min(float(_np.min(d)), float(wall))
            if cl > best_clear:
                best_clear = cl; bxv = cx; byv = cy
        # initial 26 centres
        hx = _np.concatenate([g25x, _np.array([bxv])])
        hy = _np.concatenate([g25y, _np.array([byv])])
        s_init, r_init = eval_lp(hx, hy)
        if r_init is None:
            r_init = _np.full(n, 0.07)
            s_init = float(_np.sum(r_init))
        # stages forcing growth of inserted (idx 25)
        lbs = [0.02, 0.04, 0.06, 0.08]
        cur_x, cur_y, cur_r = hx.copy(), hy.copy(), r_init.copy()
        cur_r[25] = max(cur_r[25]*0.2, 0.005)
        for lb in lbs:
            if _tm.time() - t0 > PHASE2_END:
                break
            rl_arr = _np.full(n, 0.015); rl_arr[25] = float(lb)
            # short penalized optimization forcing growth
            px, py, pr = run_penalty(cur_x, cur_y, cur_r, P_seq=[300.0, 3000.0, 30000.0],
                                     maxiter_each=120, r_low_arr=rl_arr)
            # LP with lower bound to reoptimize others while keeping defect
            s_lb, r_lb = eval_lp(px, py, lb_idx=25, lb_val=float(lb))
            if r_lb is None:
                # fallback to penalty radii scaled? keep penalty result repaired
                cur_x, cur_y, cur_r = px, py, pr
            else:
                cur_x, cur_y, cur_r = px, py, r_lb
                try_update(cur_x, cur_y, cur_r, float(_np.sum(cur_r)))
            # also try unconstrained LP (may be better, keep for tracking)
            s2, r2 = eval_lp(cur_x, cur_y)
            if r2 is not None:
                try_update(cur_x, cur_y, r2, s2)
        homo_xc, homo_yc, homo_r = cur_x, cur_y, cur_r
        sH, rH = eval_lp(homo_xc, homo_yc)
        if rH is not None:
            homo_sum = sH
            try_update(homo_xc, homo_yc, rH, sH)
        else:
            homo_sum = float(_np.sum(homo_r))
    except Exception:
        pass

    # ---------- Phase 3: release best two + homotopy to full optimization ----------
    cands = []
    for ti in best_two:
        cands.append((t_xc[ti].copy(), t_yc[ti].copy()))
    if homo_xc is not None:
        cands.append((homo_xc.copy(), homo_yc.copy()))
    # also add current global best centres as candidate
    cands.append((best_xc.copy(), best_yc.copy()))

    P_full = [30.0, 300.0, 3000.0, 30000.0]
    # round-robin polish
    rounds = 0
    # give each candidate at least one full polish
    for ci, (cx0, cy0) in enumerate(list(cands)):
        if _tm.time() - t0 > HARD - 10.0:
            break
        s0, r0 = eval_lp(cx0, cy0)
        if r0 is None:
            r0 = _np.full(n, 0.085)
        # small jitter to break symmetry
        jx = cx0 + rng.normal(0, 0.002, size=n)
        jy = cy0 + rng.normal(0, 0.002, size=n)
        jx = _np.clip(jx, 0.02, 0.98); jy = _np.clip(jy, 0.02, 0.98)
        px, py, pr = run_penalty(jx, jy, r0, P_seq=P_full, maxiter_each=200)
        s1, r1 = eval_lp(px, py)
        if r1 is not None and s1 > best_sum:
            try_update(px, py, r1, s1)
            # update candidate for further rounds
            cands[ci] = (px.copy(), py.copy())
        else:
            # keep penalty result if LP failed but penalty feasible?
            pass
        rounds += 1

    # extra ruin-and-recreate / random restarts with remaining time
    extra = 0
    while _tm.time() - t0 < HARD - 9.0 and extra < 40:
        extra += 1
        mode = extra % 4
        if mode == 0 and len(cands) > 0:
            # perturb global best slightly and rejamm
            sx = best_xc + rng.normal(0, 0.008, size=n)
            sy = best_yc + rng.normal(0, 0.008, size=n)
            sx = _np.clip(sx, 0.02, 0.98); sy = _np.clip(sy, 0.02, 0.98)
            s0, r0 = eval_lp(sx, sy)
            if r0 is None:
                continue
            px, py, pr = run_penalty(sx, sy, r0, P_seq=[300.0, 3000.0, 30000.0], maxiter_each=120)
            s1, r1 = eval_lp(px, py)
            if r1 is not None:
                try_update(px, py, r1, s1)
        elif mode == 1:
            # random uniform restart short
            rx = rng.random(n)*0.9+0.05
            ry = rng.random(n)*0.9+0.05
            s0, r0 = eval_lp(rx, ry)
            if r0 is None:
                continue
            # only pursue promising (>2.3)
            if s0 < 2.2:
                continue
            px, py, pr = run_penalty(rx, ry, r0, P_seq=P_full, maxiter_each=150)
            s1, r1 = eval_lp(px, py)
            if r1 is not None:
                try_update(px, py, r1, s1)
        elif mode == 2:
            # void teleport: move smallest circle to largest void of best
            xc = best_xc.copy(); yc = best_yc.copy()
            # smallest radius idx
            si = int(_np.argmin(best_r))
            # find void (sample)
            best_v = -1; vx = 0.5; vy = 0.5
            for _ in range(400):
                qx = rng.random(); qy = rng.random()
                wall = min(qx, 1-qx, qy, 1-qy)
                if wall < 0.02:
                    continue
                d = _np.sqrt((xc-qx)**2 + (yc-qy)**2)
                # exclude self
                d[si] = 1e9
                clear = min(float(_np.min(d - best_r)), float(wall))
                if clear > best_v:
                    best_v = clear; vx = qx; vy = qy
            xc[si] = vx; yc[si] = vy
            s0, r0 = eval_lp(xc, yc)
            if r0 is None:
                continue
            px, py, pr = run_penalty(xc, yc, r0, P_seq=[300.0, 3000.0, 30000.0], maxiter_each=120)
            s1, r1 = eval_lp(px, py)
            if r1 is not None:
                try_update(px, py, r1, s1)
        else:
            # swap large-small positions
            xc = best_xc.copy(); yc = best_yc.copy()
            o = _np.argsort(best_r)
            a = int(o[0]); b = int(o[-1])
            tx = xc[a]; ty = yc[a]
            xc[a] = xc[b]; yc[a] = yc[b]
            xc[b] = tx; yc[b] = ty
            xc += rng.normal(0, 0.003, size=n); yc += rng.normal(0, 0.003, size=n)
            xc = _np.clip(xc, 0.02, 0.98); yc = _np.clip(yc, 0.02, 0.98)
            s0, r0 = eval_lp(xc, yc)
            if r0 is None:
                continue
            px, py, pr = run_penalty(xc, yc, r0, P_seq=[300.0, 3000.0, 30000.0], maxiter_each=120)
            s1, r1 = eval_lp(px, py)
            if r1 is not None:
                try_update(px, py, r1, s1)

    # final safety shrink to guarantee strict feasibility 1e-12
    Xf = _np.clip(best_xc, 0.0, 1.0); Yf = _np.clip(best_yc, 0.0, 1.0); Rf = _np.maximum(best_r, 1e-12)
    # uniform scale
    wall_ratio = _np.minimum(_np.minimum(Xf, 1.0-Xf), _np.minimum(Yf, 1.0-Yf)) / _np.maximum(Rf, 1e-12)
    Dx = Xf[:, None] - Xf[None, :]
    Dy = Yf[:, None] - Yf[None, :]
    D = _np.sqrt(Dx*Dx + Dy*Dy)
    S = Rf[:, None] + Rf[None, :]
    with _np.errstate(divide='ignore', invalid='ignore'):
        ratio = D / S
    _np.fill_diagonal(ratio, _np.inf)
    a = float(_np.min(wall_ratio)); b = float(_np.min(ratio))
    alpha = min(1.0, a, b)
    alpha = alpha*(1.0-1e-7) - 1e-12
    if not _np.isfinite(alpha) or alpha <= 0:
        alpha = 0.99
    Rf = Rf*alpha
    # extra tiny guard
    Rf = _np.maximum(Rf - 1e-13, 1e-9)
    # re-check: if still violating due to rounding, shrink a bit more
    out = _np.column_stack([Xf, Yf, Rf])
    return out
