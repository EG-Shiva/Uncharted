import numpy as np
import time as _time

def solve():
    import time as _tm
    import numpy as _np
    from scipy.optimize import minimize, linprog
    n = 26
    t0 = _tm.time()
    DEADLINE = 124.0

    iu, ju = _np.triu_indices(n, 1)
    m = len(iu)
    A_ub0 = _np.zeros((m, n), dtype=float)
    _ar = _np.arange(m)
    A_ub0[_ar, iu] = 1.0
    A_ub0[_ar, ju] = 1.0
    c_obj = -_np.ones(n)

    def eval_lp(xc, yc, best_ref=None):
        b = _np.minimum(_np.minimum(xc, 1.0 - xc), _np.minimum(yc, 1.0 - yc))
        b = _np.maximum(b, 0.0)
        if best_ref is not None and _np.sum(b) <= best_ref:
            return None, -1.0
        dx = xc[iu] - xc[ju]
        dy = yc[iu] - yc[ju]
        dvec = _np.sqrt(dx*dx + dy*dy)
        bounds = [(0.0, float(bi)) for bi in b]
        try:
            res = linprog(c_obj, A_ub=A_ub0, b_ub=dvec, bounds=bounds, method='highs')
        except Exception:
            return None, -1.0
        if not res.success or res.x is None:
            return None, -1.0
        r = _np.array(res.x, dtype=float).flatten()
        r = _np.maximum(r, 0.0)
        s = float(_np.sum(r))
        return r, s

    def repair_radii(xc, yc, rc):
        eps = 1e-9
        b = _np.minimum(_np.minimum(xc, 1.0 - xc), _np.minimum(yc, 1.0 - yc))
        t = 1.0
        for i in range(n):
            if rc[i] > 0:
                ti = (b[i] - eps) / rc[i]
                if ti < t:
                    t = ti
        if t <= 0:
            return _np.full(n, 1e-4)
        dx = xc[iu] - xc[ju]
        dy = yc[iu] - yc[ju]
        dvec = _np.sqrt(dx*dx + dy*dy)
        rs = rc[iu] + rc[ju]
        mask = rs > 1e-12
        if _np.any(mask):
            ratios = (dvec[mask] - eps) / rs[mask]
            tp = _np.min(ratios)
            if tp < t:
                t = tp
        if t > 1.0:
            t = 1.0
        if t <= 0:
            return _np.full(n, 1e-4)
        t = t * (1.0 - 1e-9)
        return rc * t

    best_sum = -1.0
    best_x = None
    best_y = None
    best_r = None

    def consider(xc, yc):
        global_vars = [best_sum]
        # use closure via nonlocal
        return None

    # ---------- Phase A: penalty with wall-free transform ----------
    # x = u + r*(1-2u), y = v + r*(1-2v), u,v in [0,1]
    bnds_tr = [(0.0, 1.0)]*(2*n) + [(0.02, 0.20)]*n
    Ws = [20.0, 200.0, 2000.0, 20000.0]

    def make_fg():
        def fg(zz, P):
            u = zz[0:n]; v = zz[n:2*n]; r = zz[2*n:3*n]
            om2u = 1.0 - 2.0*u
            om2v = 1.0 - 2.0*v
            om2r = 1.0 - 2.0*r
            x = u + r*om2u
            y = v + r*om2v
            dxm = x[:, None] - x[None, :]
            dym = y[:, None] - y[None, :]
            D = _np.sqrt(dxm*dxm + dym*dym)
            _np.fill_diagonal(D, 1.0)
            Rsum = r[:, None] + r[None, :]
            E = Rsum - D
            _np.fill_diagonal(E, 0.0)
            Ep = _np.maximum(E, 0.0)
            # Pen = sum_{i<j} Ep^2
            Pen = 0.5*_np.sum(Ep*Ep)
            F = -_np.sum(r) + P*Pen
            # gradients w.r.t x,y,r-pen
            InvD = 1.0/D
            _np.fill_diagonal(InvD, 0.0)
            Wmat = Ep*InvD  # symmetric
            # dPen/dx = -2*sum_j Wmat*k? careful factor:
            # Pen=0.5 sum_all Ep^2, dPen/dx_k = sum_j 2*Ep*...? Let's compute:
            # d/d x_k of 0.5*sum_{a,b} Ep_ab^2 = sum_j 2*Ep_kj * (-(x_k-x_j)/D) for j!=k, but 0.5*2? Actually derivative of 0.5*Ep^2 is Ep*dEp, sum over both (k,j) and (j,k) gives 2*Ep*dEp. So:
            Gx = -2.0*_np.sum(Wmat*dxm, axis=1)
            Gy = -2.0*_np.sum(Wmat*dym, axis=1)
            sumEp = _np.sum(Ep, axis=1)  # =2*sum_{j>...}? No sum_all =2*sum_{j!=k single}. Need dPen/dr: 0.5*sum Ep^2 derivative = sum_{j} 2*Ep_kj*1? Since both (k,j) and (j,k) contribute: sum_all Ep*dEp/dr = 2*sum_j Ep_kj
            dPendr = 2.0*sumEp
            dFdx = P*Gx
            dFdy = P*Gy
            dFdu = dFdx*om2r
            dFdv = dFdy*om2r
            dFdr = -_np.ones(n) + P*dPendr + dFdx*om2u + dFdy*om2v
            g = _np.concatenate([dFdu, dFdv, dFdr])
            return F, g
        return fg
    fg_fun = make_fg()

    restart_idx = 0
    # grid for init
    gx6 = []; gy6 = []
    for ix in range(6):
        for jy in range(5):
            gx6.append((ix+0.5)/6.0); gy6.append((jy+0.5)/5.0)
    gx6 = _np.array(gx6); gy6 = _np.array(gy6)

    while True:
        if _tm.time() - t0 > 58.0:
            break
        if restart_idx > 300:
            break
        rng = _np.random.RandomState(2000 + restart_idx*6133)
        if restart_idx == 0:
            # 5x5 grid + extra mapped to u,v ~ centres
            ptsx = []; ptsy = []
            for i in range(5):
                for j in range(5):
                    ptsx.append(0.1+0.2*i); ptsy.append(0.1+0.2*j)
            ptsx = _np.array(ptsx); ptsy = _np.array(ptsy)
            # add 26th
            ptsx = _np.append(ptsx, 0.5); ptsy = _np.append(ptsy, 0.5)
            r0 = _np.full(n, 0.09)
            # invert transform approx: u ~ x (since r small)
            u0 = _np.clip(ptsx + rng.randn(n)*0.003, 0.0, 1.0)
            v0 = _np.clip(ptsy + rng.randn(n)*0.003, 0.0, 1.0)
        elif restart_idx == 1:
            perm = rng.permutation(len(gx6))[:n]
            u0 = _np.clip(gx6[perm] + rng.randn(n)*0.01, 0.0, 1.0)
            v0 = _np.clip(gy6[perm] + rng.randn(n)*0.01, 0.0, 1.0)
            r0 = rng.uniform(0.06, 0.10, size=n)
        else:
            if restart_idx % 3 == 1:
                perm = rng.permutation(len(gx6))[:n]
                u0 = _np.clip(gx6[perm] + (rng.rand(n)-0.5)*0.08, 0.0, 1.0)
                v0 = _np.clip(gy6[perm] + (rng.rand(n)-0.5)*0.08, 0.0, 1.0)
            else:
                u0 = rng.rand(n); v0 = rng.rand(n)
            r0 = rng.uniform(0.05, 0.11, size=n)
        z = _np.concatenate([u0, v0, r0])
        for P in Ws:
            if _tm.time() - t0 > 58.0:
                break
            def clos(zz, P=P):
                return fg_fun(zz, P)
            try:
                res = minimize(clos, z, method='L-BFGS-B', jac=True, bounds=bnds_tr,
                               options={'maxiter': 180, 'ftol': 1e-9, 'gtol': 1e-5})
                z = res.x
            except Exception:
                break
        try:
            u = z[0:n]; v = z[n:2*n]; rr = z[2*n:3*n]
            xc = u + rr*(1.0-2.0*u)
            yc = v + rr*(1.0-2.0*v)
            xc = _np.clip(xc, 0.0, 1.0); yc = _np.clip(yc, 0.0, 1.0)
            r_opt, s = eval_lp(xc, yc, best_ref=best_sum if best_sum>0 else None)
            if r_opt is not None and s > best_sum:
                best_sum = float(s); best_x = xc.copy(); best_y = yc.copy(); best_r = r_opt.copy()
        except Exception:
            pass
        restart_idx += 1

    if best_x is None:
        k = int(_np.ceil(_np.sqrt(n)))
        rr = 0.5/k
        pts = []
        for i in range(k):
            for j in range(k):
                if len(pts) < n:
                    pts.append([(2*i+1)*rr, (2*j+1)*rr, rr*0.999])
        arr = _np.array(pts)
        return arr

    # ---------- Phase B: guided hill climbing on LP value ----------
    rng_hc = _np.random.RandomState(7777)
    hc_iter = 0
    # precompute for force moves: need best_r
    no_improve = 0
    while _tm.time() - t0 < DEADLINE - 4.0 and hc_iter < 30000:
        hc_iter += 1
        # choose move
        do_directed = (hc_iter % 17 == 0)
        do_global = (hc_iter % 400 == 0)
        if do_global:
            sigma_g = 0.003
            nx = best_x + rng_hc.randn(n)*sigma_g
            ny = best_y + rng_hc.randn(n)*sigma_g
            nx = _np.clip(nx, 0.01, 0.99); ny = _np.clip(ny, 0.01, 0.99)
        elif do_directed:
            try:
                # contact forces from current best
                X = best_x; Y = best_y; R = best_r
                Dx = X[:, None]-X[None, :]
                Dy = Y[:, None]-Y[None, :]
                D = _np.sqrt(Dx*Dx+Dy*Dy)
                _np.fill_diagonal(D, 1.0)
                Rs = R[:, None]+R[None, :]
                gap = D - Rs
                # tight threshold
                tight = (gap < 5e-4)
                _np.fill_diagonal(tight, False)
                Fx = _np.zeros(n); Fy = _np.zeros(n)
                # pairwise repulsion for tight only
                # vectorized
                Inv = _np.where(tight, 1.0/(_np.maximum(D,1e-9)), 0.0)
                Fx = _np.sum(Dx*Inv, axis=1)
                Fy = _np.sum(Dy*Inv, axis=1)
                # wall tight
                bwall = _np.minimum(_np.minimum(X,1-X), _np.minimum(Y,1-Y))
                walltight = (bwall - R) < 5e-4
                for i in range(n):
                    if walltight[i]:
                        # nearest wall
                        dL = X[i]; dR = 1-X[i]; dB = Y[i]; dT = 1-Y[i]
                        m = min(dL,dR,dB,dT)
                        if m == dL: Fx[i] += 1.0
                        elif m == dR: Fx[i] -= 1.0
                        elif m == dB: Fy[i] += 1.0
                        else: Fy[i] -= 1.0
                # normalize
                nrm = _np.sqrt(Fx*Fx+Fy*Fy)+1e-12
                # step
                step = 0.004 * _np.exp(-hc_iter/8000.0) + 0.0005
                # scale per-circle by nrm? use direction
                nx = best_x + (Fx/(nrm))*step + rng_hc.randn(n)*0.0008
                ny = best_y + (Fy/(nrm))*step + rng_hc.randn(n)*0.0008
                # if forces zero, add noise
                if _np.all(nrm < 1e-9):
                    nx = best_x + rng_hc.randn(n)*0.002
                    ny = best_y + rng_hc.randn(n)*0.002
                nx = _np.clip(nx, 0.01, 0.99); ny = _np.clip(ny, 0.01, 0.99)
            except Exception:
                sigma = 0.005
                idx = int(rng_hc.randint(n))
                nx = best_x.copy(); ny = best_y.copy()
                nx[idx] += float(rng_hc.randn())*sigma
                ny[idx] += float(rng_hc.randn())*sigma
                nx = _np.clip(nx, 0.01, 0.99); ny = _np.clip(ny, 0.01, 0.99)
        else:
            sigma = 0.018*_np.exp(-hc_iter/3500.0) + 0.0007
            # 80% single, 20% two circles
            if rng_hc.rand() < 0.85:
                idx = int(rng_hc.randint(n))
                nx = best_x.copy(); ny = best_y.copy()
                nx[idx] += float(rng_hc.randn())*sigma
                ny[idx] += float(rng_hc.randn())*sigma
                nx[idx] = min(max(nx[idx],0.01),0.99)
                ny[idx] = min(max(ny[idx],0.01),0.99)
            else:
                i1 = int(rng_hc.randint(n)); i2 = int(rng_hc.randint(n))
                nx = best_x.copy(); ny = best_y.copy()
                nx[i1] += float(rng_hc.randn())*sigma
                ny[i1] += float(rng_hc.randn())*sigma
                nx[i2] += float(rng_hc.randn())*sigma
                ny[i2] += float(rng_hc.randn())*sigma
                nx = _np.clip(nx, 0.01, 0.99); ny = _np.clip(ny, 0.01, 0.99)
        r_opt, s = eval_lp(nx, ny, best_ref=best_sum)
        if r_opt is not None and s > best_sum + 1e-12:
            best_sum = float(s); best_x = nx; best_y = ny; best_r = r_opt.copy()
            no_improve = 0
        else:
            no_improve += 1
        # occasional large shake if stuck
        if no_improve > 1200 and _tm.time()-t0 < DEADLINE-12.0:
            nx = best_x + rng_hc.randn(n)*0.006
            ny = best_y + rng_hc.randn(n)*0.006
            nx = _np.clip(nx, 0.01, 0.99); ny = _np.clip(ny, 0.01, 0.99)
            r_opt2, s2 = eval_lp(nx, ny, best_ref=None)
            if r_opt2 is not None and s2 > best_sum - 0.02:  # accept sideways to diversify if not too bad
                # only jump if it gives new basin? keep best but move centre to explore? Use random restart from shaken point with few hill steps?
                # Instead, set current probe to shaken point for next iters without losing best: we keep best, but we will continue from shaken if promising
                # To implement, temporarily set best to shaken if within 0.005? No, keep best. Instead do nothing, just reset counter with small perturbation of best?
                pass
            no_improve = 0

    # final exact + repair
    r_final, _ = eval_lp(best_x, best_y, best_ref=None)
    if r_final is not None:
        best_r = r_final
    r_safe = repair_radii(best_x, best_y, _np.maximum(best_r, 0.0))
    r_safe = _np.maximum(r_safe, 1e-9)
    for i in range(n):
        if best_x[i] < r_safe[i] + 1e-9:
            best_x[i] = r_safe[i] + 1e-9
        if best_x[i] > 1.0 - r_safe[i] - 1e-9:
            best_x[i] = 1.0 - r_safe[i] - 1e-9
        if best_y[i] < r_safe[i] + 1e-9:
            best_y[i] = r_safe[i] + 1e-9
        if best_y[i] > 1.0 - r_safe[i] - 1e-9:
            best_y[i] = 1.0 - r_safe[i] - 1e-9
    r_safe = repair_radii(best_x, best_y, r_safe)
    r_safe = _np.maximum(r_safe, 1e-9)
    out = _np.column_stack([best_x, best_y, r_safe])
    return out
