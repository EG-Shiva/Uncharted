import numpy as np
import time

def solve():
    import time as _tm
    import numpy as _np
    from scipy.optimize import minimize, linprog
    n = 26
    t0 = _tm.time()
    DEADLINE = 124.0

    iu, ju = _np.triu_indices(n, 1)
    m = len(iu)
    A_primal = _np.zeros((m, n), dtype=float)
    ar = _np.arange(m)
    A_primal[ar, iu] = 1.0
    A_primal[ar, ju] = 1.0
    c_primal = -_np.ones(n, dtype=float)

    def primal_eval(xc, yc):
        b = _np.minimum(_np.minimum(xc, 1.0 - xc), _np.minimum(yc, 1.0 - yc))
        if _np.min(b) <= 1e-9:
            return None, -1.0
        dx = xc[iu] - xc[ju]
        dy = yc[iu] - yc[ju]
        dvec = _np.sqrt(dx * dx + dy * dy)
        if _np.min(dvec) <= 1e-9:
            return None, -1.0
        bounds = [(0.0, float(bi)) for bi in b]
        try:
            res = linprog(c_primal, A_ub=A_primal, b_ub=dvec, bounds=bounds, method='highs')
        except Exception:
            return None, -1.0
        if not res.success:
            return None, -1.0
        r = _np.maximum(_np.array(res.x, dtype=float).flatten(), 0.0)
        s = float(_np.sum(r))
        return r, s

    # joint penalty fun (uses iu,ju)
    def pen_fun(v, W):
        X = v[0:n]
        Y = v[n:2*n]
        R = v[2*n:3*n]
        gx = _np.zeros(n)
        gy = _np.zeros(n)
        gr = _np.zeros(n)
        # walls
        v1 = R - X
        m1 = v1 > 0
        if _np.any(m1):
            gx[m1] += -2.0 * v1[m1]
            gr[m1] += 2.0 * v1[m1]
        v2 = X + R - 1.0
        m2 = v2 > 0
        if _np.any(m2):
            gx[m2] += 2.0 * v2[m2]
            gr[m2] += 2.0 * v2[m2]
        v3 = R - Y
        m3 = v3 > 0
        if _np.any(m3):
            gy[m3] += -2.0 * v3[m3]
            gr[m3] += 2.0 * v3[m3]
        v4 = Y + R - 1.0
        m4 = v4 > 0
        if _np.any(m4):
            gy[m4] += 2.0 * v4[m4]
            gr[m4] += 2.0 * v4[m4]
        # pairs via iu,ju
        dx = X[iu] - X[ju]
        dy = Y[iu] - Y[ju]
        d = _np.sqrt(dx*dx + dy*dy + 1e-20)
        V = R[iu] + R[ju] - d
        M = V > 0
        if _np.any(M):
            # grad w.r.t centres
            inv = _np.zeros_like(d)
            inv[M] = 1.0 / d[M]
            # dV/dx_iu = -(x_iu-x_ju)/d etc.
            cx = V[M] * inv[M] * dx[M]  # V*(dx/d)
            cy = V[M] * inv[M] * dy[M]
            # gx[iu] += -cx, gx[ju]+= +cx
            _np.add.at(gx, iu[M], -cx)
            _np.add.at(gx, ju[M], cx)
            _np.add.at(gy, iu[M], -cy)
            _np.add.at(gy, ju[M], cy)
            # grad w.r.t radii
            _np.add.at(gr, iu[M], V[M])
            _np.add.at(gr, ju[M], V[M])
        # penalty value
        pen = 0.0
        if _np.any(m1):
            pen += _np.sum(v1[m1]**2)
        if _np.any(m2):
            pen += _np.sum(v2[m2]**2)
        if _np.any(m3):
            pen += _np.sum(v3[m3]**2)
        if _np.any(m4):
            pen += _np.sum(v4[m4]**2)
        if _np.any(M):
            pen += 0.5 * _np.sum(V[M]**2)
        loss = -_np.sum(R) + W * pen
        g = _np.empty_like(v)
        g[0:n] = W * gx
        g[n:2*n] = W * gy
        g[2*n:3*n] = -1.0 + W * gr
        return loss, g

    bounds_joint = [(0.0, 1.0)]*(2*n) + [(0.015, 0.25)]*n

    def joint_polish(x0, y0, r0, Ws, maxiter_each):
        v = _np.concatenate([x0, y0, r0])
        for W in Ws:
            if _tm.time() - t0 > DEADLINE - 3.0:
                break
            def f(vv, W=W):
                return pen_fun(vv, W)
            try:
                res = minimize(f, v, method='L-BFGS-B', jac=True, bounds=bounds_joint,
                               options={'maxiter': maxiter_each, 'ftol': 1e-9, 'gtol': 1e-5})
                v = res.x
            except Exception:
                break
        X = _np.clip(v[0:n], 0.02, 0.98)
        Y = _np.clip(v[n:2*n], 0.02, 0.98)
        r, s = primal_eval(X, Y)
        if r is None:
            # fallback to penalty radii
            R = _np.clip(v[2*n:3*n], 1e-4, 0.25)
            return X, Y, R, -1.0
        return X, Y, r, s

    # grid for void search
    GN = 201
    gx1 = _np.linspace(0.0, 1.0, GN)
    gy1 = _np.linspace(0.0, 1.0, GN)
    XXm, YYm = _np.meshgrid(gx1, gy1)
    XX = XXm.ravel()
    YY = YYm.ravel()
    wall_clear_grid = _np.minimum(_np.minimum(XX, 1.0-XX), _np.minimum(YY, 1.0-YY))

    def find_void(X, Y, R, keep_mask):
        # keep_mask bool (n,) True = obstacle
        best_c = -1e100
        bx = 0.5; by = 0.5
        # compute min dist
        md = _np.full(XX.shape, 1e100, dtype=float)
        idx = _np.where(keep_mask)[0]
        for j in idx:
            dx = XX - X[j]
            dy = YY - Y[j]
            d = _np.sqrt(dx*dx + dy*dy) - R[j]
            # in-place min
            _np.minimum(md, d, out=md)
        if len(idx) == 0:
            md[:] = 1e100
        clear = _np.minimum(wall_clear_grid, md)
        k = int(_np.argmax(clear))
        return float(XX[k]), float(YY[k]), float(clear[k])

    def contact_counts(X, Y, R):
        b = _np.minimum(_np.minimum(X, 1.0-X), _np.minimum(Y, 1.0-Y))
        cnt = _np.zeros(n, dtype=int)
        # wall contact
        wc = (b - R) < 5e-4
        cnt[wc] += 1
        dx = X[:, None] - X[None, :]
        dy = Y[:, None] - Y[None, :]
        D = _np.sqrt(dx*dx + dy*dy)
        S = R[:, None] + R[None, :]
        C = (D - S) < 5e-4
        # count neighbours
        for i in range(n):
            c = int(_np.sum(C[i]) - 1)  # exclude self (Dii-Sii negative -> true, so -1)
            # self: Dii=0,Sii=2Ri => D-S=-2R<0 => True, so subtract
            if c < 0:
                c = 0
            cnt[i] += c
        return cnt

    rng = _np.random.RandomState(0)
    best_S = -1.0
    best_X = None; best_Y = None; best_R = None

    # ---- Phase A: annealed restarts ----
    # fallback grid
    k0 = int(_np.ceil(_np.sqrt(n)))
    rr0 = 0.5 / k0
    pts = []
    for i in range(k0):
        for j in range(k0):
            if len(pts) < n:
                pts.append([(2*i+1)*rr0, (2*j+1)*rr0, rr0*0.9])
    fb = _np.array(pts, dtype=float)
    rfb, sfb = primal_eval(fb[:,0], fb[:,1])
    if rfb is not None:
        best_S = sfb; best_X = fb[:,0].copy(); best_Y = fb[:,1].copy(); best_R = rfb.copy()

    T_PHASE_A = 45.0
    n_restarts_done = 0
    restart = 0
    while _tm.time() - t0 < T_PHASE_A:
        if restart > 40:
            break
        if restart == 0:
            # 5x5 + extra
            xs = []; ys = []
            for i in range(5):
                for j in range(5):
                    xs.append(0.1 + 0.2*i); ys.append(0.1 + 0.2*j)
            xs = _np.array(xs); ys = _np.array(ys)
            xs = _np.append(xs, 0.5); ys = _np.append(ys, 0.62)
            X0 = xs + rng.randn(n)*0.008
            Y0 = ys + rng.randn(n)*0.008
            R0 = _np.full(n, 0.07) + rng.randn(n)*0.01
        elif restart == 1:
            # 6x5 grid subset
            xs6 = (_np.arange(6)+0.5)/6.0
            ys5 = (_np.arange(5)+0.5)/5.0
            xx, yy = _np.meshgrid(xs6, ys5)
            xy = _np.column_stack([xx.ravel(), yy.ravel()])
            perm = rng.permutation(len(xy))[:n]
            X0 = xy[perm,0] + rng.randn(n)*0.01
            Y0 = xy[perm,1] + rng.randn(n)*0.01
            R0 = 0.06 + 0.03*rng.rand(n)
        else:
            X0 = rng.uniform(0.05, 0.95, size=n)
            Y0 = rng.uniform(0.05, 0.95, size=n)
            R0 = 0.05 + 0.05*rng.rand(n)
        X0 = _np.clip(X0, 0.03, 0.97)
        Y0 = _np.clip(Y0, 0.03, 0.97)
        R0 = _np.clip(R0, 0.02, 0.2)
        # quick polish with 3 weights
        Xp, Yp, Rp, Sp = joint_polish(X0, Y0, R0, [5e2, 5e3, 5e4], 80)
        if Sp > best_S:
            best_S = Sp; best_X = Xp.copy(); best_Y = Yp.copy(); best_R = Rp.copy()
        restart += 1
        n_restarts_done += 1

    if best_X is None:
        return fb

    cur_X = best_X.copy(); cur_Y = best_Y.copy(); cur_R = best_R.copy(); cur_S = best_S
    rng2 = _np.random.RandomState(1)

    # ---- Phase B: topology surgery / basin hopping ----
    T0 = 0.004
    T1 = 0.00015
    it = 0
    no_improve = 0
    # estimate total iters for cooling: use time-based cooling
    t_start_B = _tm.time()
    # run until deadline
    while _tm.time() - t0 < DEADLINE - 4.0:
        it += 1
        frac = min(1.0, (_tm.time() - t_start_B) / max(1.0, (DEADLINE - 4.0 - t_start_B)))
        Temp = T0 * (T1/T0)**frac  # exponential cooling
        # choose k
        u = rng2.rand()
        if u < 0.6:
            kk = 1
        elif u < 0.85:
            kk = 2
        else:
            kk = 3
        # choose victims
        # order by radius
        order_small = _np.argsort(cur_R)
        cc = contact_counts(cur_X, cur_Y, cur_R)
        order_loose = _np.argsort(cc)
        if rng2.rand() < 0.7:
            victims = order_small[:kk]
        else:
            # mix smallest + loosest
            cand = list(order_small[:6]) + list(order_loose[:6])
            # dedup preserve
            uniq = []
            for c in cand:
                if c not in uniq:
                    uniq.append(c)
            victims = _np.array(rng2.choice(uniq, size=min(kk, len(uniq)), replace=False))
            kk = len(victims)
        victims = _np.array(victims, dtype=int)
        keep = _np.ones(n, dtype=bool)
        keep[victims] = False
        # greedy reinsertion
        newX = cur_X.copy(); newY = cur_Y.copy(); newR = cur_R.copy()
        # for clearance use cur_R for keepers
        tmpX = newX.copy(); tmpY = newY.copy(); tmpR = newR.copy()
        # we will place victims one by one; keep mask grows
        cur_keep = keep.copy()
        # sort victims by placement order (smallest last? place larger first) - keep as is
        placed_pos = []
        for v in victims:
            bx, by, bc = find_void(tmpX, tmpY, tmpR, cur_keep)
            # jitter
            bx += rng2.randn()*0.002
            by += rng2.randn()*0.002
            bx = float(_np.clip(bx, 0.02, 0.98))
            by = float(_np.clip(by, 0.02, 0.98))
            placed_pos.append((bx, by, bc))
            tmpX[v] = bx; tmpY[v] = by
            # provisional radius for further void search
            tmpR[v] = max(float(bc)*0.9, 0.02)
            cur_keep[v] = True
        for v, (bx, by, bc) in zip(victims, placed_pos):
            newX[v] = bx; newY[v] = by
            # init radius guess
            newR[v] = max(float(bc)*0.85, 0.02)
            newR[v] = float(_np.clip(newR[v], 0.015, 0.2))
        # also tiny jitter to keepers occasionally to escape
        if rng2.rand() < 0.3:
            jit = 0.0015
            newX[keep] += rng2.randn(_np.sum(keep))*jit
            newY[keep] += rng2.randn(_np.sum(keep))*jit
            newX = _np.clip(newX, 0.02, 0.98)
            newY = _np.clip(newY, 0.02, 0.98)
        # LP eval before polish
        r0, s0 = primal_eval(newX, newY)
        if r0 is None:
            no_improve += 1
            continue
        # short penalty re-opt
        # use lighter polish for speed
        Xp, Yp, Rp, Sp = joint_polish(newX, newY, _np.clip(r0, 0.015, 0.2), [2e4], 60)
        # joint_polish already does LP; Sp is LP sum
        if Sp < 0:
            # fallback to s0
            Xp, Yp, Rp, Sp = newX, newY, r0, s0
        # acceptance
        delta = Sp - cur_S
        accept = False
        if delta > 1e-12:
            accept = True
        else:
            if Temp > 1e-12:
                p = _np.exp(delta / Temp) if delta > -0.05 else 0.0
                if rng2.rand() < p:
                    accept = True
        if accept:
            cur_X, cur_Y, cur_R, cur_S = Xp, Yp, Rp, Sp
            no_improve = 0 if delta > 0 else no_improve + 1
        else:
            no_improve += 1
        if Sp > best_S + 1e-12:
            best_S = Sp; best_X = Xp.copy(); best_Y = Yp.copy(); best_R = Rp.copy()
            no_improve = 0
        # periodic elite restore
        if no_improve >= 30:
            # restore elite with perturbation
            cur_X = best_X + rng2.randn(n)*0.002
            cur_Y = best_Y + rng2.randn(n)*0.002
            cur_X = _np.clip(cur_X, 0.02, 0.98)
            cur_Y = _np.clip(cur_Y, 0.02, 0.98)
            rr, ss = primal_eval(cur_X, cur_Y)
            if rr is not None:
                cur_R, cur_S = rr, ss
            else:
                cur_X, cur_Y, cur_R, cur_S = best_X.copy(), best_Y.copy(), best_R.copy(), best_S
            no_improve = 0
        # occasional hill-climb micro step on best (exploit)
        if it % 25 == 0 and (_tm.time() - t0 < DEADLINE - 10.0):
            # try small Gaussian move on best
            sig = 0.003
            xn = best_X + rng2.randn(n)*sig
            yn = best_Y + rng2.randn(n)*sig
            xn = _np.clip(xn, 0.02, 0.98); yn = _np.clip(yn, 0.02, 0.98)
            rr, ss = primal_eval(xn, yn)
            if rr is not None and ss > best_S:
                best_S = ss; best_X = xn; best_Y = yn; best_R = rr
                cur_X, cur_Y, cur_R, cur_S = xn.copy(), yn.copy(), rr.copy(), ss

    x = best_X; y = best_Y; r = best_R.copy()
    r = _np.maximum(r, 1e-7)
    wall = _np.minimum(_np.minimum(x, 1.0 - x), _np.minimum(y, 1.0 - y))
    Dx = x[:, None] - x[None, :]
    Dy = y[:, None] - y[None, :]
    D = _np.sqrt(Dx*Dx + Dy*Dy)
    fw = _np.min(wall / r)
    fp = 1e100
    for i in range(n):
        for j in range(i+1, n):
            v = D[i, j] / (r[i] + r[j])
            if v < fp:
                fp = v
    f = fw if fw < fp else fp
    if not _np.isfinite(f):
        f = fw
    if f < 1.0:
        r = r * f * (1.0 - 1e-9)
    else:
        r = r * (1.0 - 1e-9)
    out = _np.column_stack([x, y, r])
    return out
