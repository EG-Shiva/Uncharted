import numpy as np
import time

def solve():
    import time as _tm
    import numpy as _np
    from scipy.optimize import minimize, linprog
    n = 26
    t0 = _tm.time()
    DEADLINE = 124.0
    rng = _np.random.default_rng(1)

    iu, ju = _np.triu_indices(n, 1)
    m = len(iu)
    A0 = _np.zeros((m, n), dtype=float)
    ar = _np.arange(m)
    A0[ar, iu] = 1.0
    A0[ar, ju] = 1.0
    cobj = -_np.ones(n)

    best_sum = -1.0
    best_x = None
    best_y = None
    best_r = None

    def eval_lp(xc, yc):
        nonlocal best_sum
        w = _np.minimum(_np.minimum(xc, 1.0 - xc), _np.minimum(yc, 1.0 - yc))
        if _np.any(w < -1e-12):
            return None, -1.0
        w = _np.maximum(w, 0.0)
        sw = _np.sum(w)
        if sw <= best_sum:
            return None, -1.0
        dx = xc[iu] - xc[ju]
        dy = yc[iu] - yc[ju]
        dvec = _np.sqrt(dx * dx + dy * dy)
        # upper bound sum <= sum(d)/(n-1)
        ub = _np.sum(dvec) / (n - 1)
        if ub < sw:
            ub_sw = ub
        else:
            ub_sw = sw
        if ub_sw <= best_sum:
            return None, -1.0
        bounds = [(0.0, float(wi)) for wi in w]
        try:
            res = linprog(cobj, A_ub=A0, b_ub=dvec, bounds=bounds, method='highs')
        except Exception:
            return None, -1.0
        if (not res.success) or (res.x is None):
            return None, -1.0
        r = _np.array(res.x, dtype=float).ravel()
        r = _np.maximum(r, 0.0)
        s = float(_np.sum(r))
        return r, s

    # fallback grid best
    k = int(_np.ceil(_np.sqrt(n)))
    rr0 = 0.5 / k
    gx0 = []
    for i in range(k):
        for j in range(k):
            if len(gx0) < n:
                gx0.append([(2*i+1)*rr0, (2*j+1)*rr0])
    gx0 = _np.array(gx0)
    _x0 = gx0[:, 0].copy()
    _y0 = gx0[:, 1].copy()
    _r0, _s0 = eval_lp(_x0, _y0)
    if _r0 is not None:
        best_sum = _s0
        best_x, best_y, best_r = _x0.copy(), _y0.copy(), _r0.copy()

    # ---- Phase 1: fast joint penalty restarts ----
    def joint_fun(v, W):
        X = v[0:n]; Y = v[n:2*n]; R = v[2*n:3*n]
        gx = _np.zeros(n); gy = _np.zeros(n)
        gr_pen = _np.zeros(n)
        # walls
        v1 = R - X
        mm = v1 > 0
        if _np.any(mm):
            gx[mm] += -2.0*v1[mm]
            gr_pen[mm] += 2.0*v1[mm]
        v2 = X + R - 1.0
        mm = v2 > 0
        if _np.any(mm):
            gx[mm] += 2.0*v2[mm]
            gr_pen[mm] += 2.0*v2[mm]
        v3 = R - Y
        mm = v3 > 0
        if _np.any(mm):
            gy[mm] += -2.0*v3[mm]
            gr_pen[mm] += 2.0*v3[mm]
        v4 = Y + R - 1.0
        mm = v4 > 0
        if _np.any(mm):
            gy[mm] += 2.0*v4[mm]
            gr_pen[mm] += 2.0*v4[mm]
        Dx = X[:, None] - X[None, :]
        Dy = Y[:, None] - Y[None, :]
        D = _np.sqrt(Dx*Dx + Dy*Dy + 1e-20)
        S = R[:, None] + R[None, :]
        V = S - D
        _np.fill_diagonal(V, 0.0)
        _np.fill_diagonal(D, 1.0)
        Mask = V > 0
        if _np.any(Mask):
            Q = _np.zeros((n, n))
            Q[Mask] = -2.0*V[Mask]/D[Mask]
            gx += _np.sum(Q*Dx, axis=1)
            gy += _np.sum(Q*Dy, axis=1)
            G = _np.zeros((n, n))
            G[Mask] = 2.0*V[Mask]
            gr_pen += _np.sum(G, axis=1)
        pen = 0.0
        # recompute pen value quickly (squared)
        # wall part
        # use above v's (recompute masked squares)
        # for speed approximate from grad? recompute:
        # (we have V)
        pen_w = 0.0
        # walls squares
        t1 = R - X
        t1 = t1[t1 > 0]
        if t1.size: pen_w += _np.sum(t1*t1)
        t2 = X + R - 1.0
        t2 = t2[t2 > 0]
        if t2.size: pen_w += _np.sum(t2*t2)
        t3 = R - Y
        t3 = t3[t3 > 0]
        if t3.size: pen_w += _np.sum(t3*t3)
        t4 = Y + R - 1.0
        t4 = t4[t4 > 0]
        if t4.size: pen_w += _np.sum(t4*t4)
        pen_p = 0.5*_np.sum(V[Mask]*V[Mask]) if _np.any(Mask) else 0.0
        pen = pen_w + pen_p
        loss = -_np.sum(R) + W*pen
        g = _np.empty_like(v)
        g[0:n] = W*gx
        g[n:2*n] = W*gy
        g[2*n:3*n] = -1.0 + W*gr_pen
        return loss, g

    bnds = [(0.0, 1.0)]*(2*n) + [(0.02, 0.20)]*n
    nrest = 0
    while _tm.time() - t0 < 32.0 and nrest < 25:
        if nrest == 0:
            X0 = _np.zeros(n); Y0 = _np.zeros(n)
            idx = 0
            for i in range(5):
                for j in range(5):
                    if idx < 25:
                        X0[idx] = (i+0.5)/5.0 + rng.normal(0, 0.004)
                        Y0[idx] = (j+0.5)/5.0 + rng.normal(0, 0.004)
                        idx += 1
            X0 = _np.append(X0[:25] if n == 26 else X0, 0.9) if n == 26 else X0
            # fix for n=26: we built 25, need 26
            if len(X0) == 25:
                X0 = _np.append(X0, 0.9 + rng.normal(0, 0.005))
                Y0 = _np.append(Y0, 0.1 + rng.normal(0, 0.005))
            else:
                # general fallback
                X0 = _np.clip(X0, 0.05, 0.95); Y0 = _np.clip(Y0, 0.05, 0.95)
            X0 = _np.clip(X0, 0.05, 0.95); Y0 = _np.clip(Y0, 0.05, 0.95)
            R0 = _np.full(n, 0.095) + rng.normal(0, 0.005, n)
            R0 = _np.clip(R0, 0.03, 0.18)
        elif nrest == 1:
            xs = (_np.arange(6)+0.5)/6.0
            ys = (_np.arange(5)+0.5)/5.0
            xx, yy = _np.meshgrid(xs, ys)
            xy = _np.column_stack([xx.ravel(), yy.ravel()])
            perm = rng.permutation(len(xy))[:n]
            X0 = xy[perm, 0] + rng.normal(0, 0.004, n)
            Y0 = xy[perm, 1] + rng.normal(0, 0.004, n)
            X0 = _np.clip(X0, 0.05, 0.95); Y0 = _np.clip(Y0, 0.05, 0.95)
            R0 = 0.085 + 0.02*rng.random(n)
        else:
            X0 = rng.random(n)*0.9+0.05
            Y0 = rng.random(n)*0.9+0.05
            R0 = 0.08 + 0.03*rng.random(n)
        v = _np.concatenate([X0, Y0, R0])
        for W in (1e3, 3e4):
            if _tm.time() - t0 > 33.0:
                break
            try:
                res = minimize(joint_fun, v, args=(W,), jac=True, method='L-BFGS-B',
                               bounds=bnds, options={'maxiter': 150, 'ftol': 1e-9, 'gtol': 1e-6})
                v = res.x
            except Exception:
                break
        X = _np.clip(v[0:n], 0.0, 1.0)
        Y = _np.clip(v[n:2*n], 0.0, 1.0)
        rlp, slp = eval_lp(X, Y)
        if rlp is not None and slp > best_sum:
            best_sum = slp
            best_x, best_y, best_r = X.copy(), Y.copy(), rlp.copy()
        nrest += 1

    if best_x is None:
        best_x = rng.random(n)*0.8+0.1
        best_y = rng.random(n)*0.8+0.1
        rlp, slp = eval_lp(best_x, best_y)
        if rlp is not None:
            best_sum = slp; best_r = rlp
        else:
            best_r = _np.full(n, 0.05)

    # ---- Phase 2: annealed LP hill climbing + void kicks ----
    def largest_void_px(px_n, py_n, xc, yc, rc, excl):
        # vectorized clearance
        wall = _np.minimum(_np.minimum(px_n, 1.0-px_n), _np.minimum(py_n, 1.0-py_n))
        dx = px_n[:, None] - xc[None, :]
        dy = py_n[:, None] - yc[None, :]
        d = _np.sqrt(dx*dx + dy*dy)
        cp = d - rc[None, :]
        if excl >= 0:
            cp[:, excl] = _np.inf
        mp = _np.min(cp, axis=1)
        clear = _np.minimum(wall, mp)
        k = int(_np.argmax(clear))
        return float(px_n[k]), float(py_n[k]), float(clear[k])

    no_improve = 0
    it = 0
    # ensure best_r aligned
    if best_r is None or len(best_r) != n:
        rtmp, _ = eval_lp(best_x, best_y)
        best_r = rtmp if rtmp is not None else _np.full(n, 0.05)

    while _tm.time() - t0 < 115.0 and it < 40000:
        it += 1
        elapsed = _tm.time() - t0
        sigma_base = 0.02*_np.exp(-elapsed/45.0) + 0.0012
        if no_improve > 800:
            sigma_base = 0.05
        if no_improve > 2000:
            # big random restart kick from best but keep best
            no_improve = 0
        u = rng.random()
        xn = None; yn = None
        if u < 0.62:
            # single circle move
            i = int(rng.integers(0, n))
            s = sigma_base
            xn = best_x.copy(); yn = best_y.copy()
            xn[i] += rng.normal(0, s)
            yn[i] += rng.normal(0, s)
            if xn[i] < 0.0: xn[i] = 0.0
            elif xn[i] > 1.0: xn[i] = 1.0
            if yn[i] < 0.0: yn[i] = 0.0
            elif yn[i] > 1.0: yn[i] = 1.0
        elif u < 0.82:
            s = sigma_base*0.35
            xn = best_x + rng.normal(0, s, n)
            yn = best_y + rng.normal(0, s, n)
            xn = _np.clip(xn, 0.0, 1.0)
            yn = _np.clip(yn, 0.0, 1.0)
        else:
            # rattler relocation to void or random
            # pick victim among smallest 5
            order = _np.argsort(best_r)
            # random choice weighted to smallest
            cand = order[:5]
            vi = int(cand[rng.integers(0, len(cand))])
            xn = best_x.copy(); yn = best_y.copy()
            if rng.random() < 0.6:
                ns = 1500
                px = rng.random(ns); py = rng.random(ns)
                bx, by, _ = largest_void_px(px, py, best_x, best_y, best_r, vi)
                # add tiny jitter
                bx += rng.normal(0, 0.001); by += rng.normal(0, 0.001)
                bx = min(max(bx, 0.0), 1.0); by = min(max(by, 0.0), 1.0)
                xn[vi] = bx; yn[vi] = by
            else:
                xn[vi] = rng.random(); yn[vi] = rng.random()
        rnew, snew = eval_lp(xn, yn)
        if rnew is not None and snew > best_sum + 1e-12:
            best_sum = snew
            best_x, best_y, best_r = xn, yn, rnew
            no_improve = 0
        else:
            no_improve += 1

    # ---- final safety shrink ----
    try:
        Xf = _np.clip(best_x, 0.0, 1.0)
        Yf = _np.clip(best_y, 0.0, 1.0)
        Rf = _np.maximum(best_r, 1e-12)
        w = _np.minimum(_np.minimum(Xf, 1.0-Xf), _np.minimum(Yf, 1.0-Yf))
        alpha = 1.0
        # wall ratios
        for i in range(n):
            if Rf[i] > 0:
                ti = w[i]/Rf[i]
                if ti < alpha:
                    alpha = ti
        dx = Xf[iu]-Xf[ju]; dy = Yf[iu]-Yf[ju]
        dvec = _np.sqrt(dx*dx+dy*dy)
        rs = Rf[iu]+Rf[ju]
        mask = rs > 1e-15
        if _np.any(mask):
            rats = dvec[mask]/rs[mask]
            tp = _np.min(rats)
            if tp < alpha:
                alpha = tp
        alpha = alpha*(1.0-1e-7)-1e-12
        if (not _np.isfinite(alpha)) or alpha <= 0:
            alpha = 0.5
        if alpha > 1.0:
            alpha = 1.0*(1.0-1e-9)
        Rf = Rf*alpha
        Rf = _np.maximum(Rf, 1e-9)
        # re-verify strict feasibility, shrink again if needed
        # wall
        w2 = _np.minimum(_np.minimum(Xf, 1.0-Xf), _np.minimum(Yf, 1.0-Yf))
        if _np.any(Rf > w2):
            # further shrink
            alpha2 = _np.min(w2/Rf)
            Rf = Rf*alpha2*(1.0-1e-7)
        # pair
        dx = Xf[iu]-Xf[ju]; dy = Yf[iu]-Yf[ju]
        dvec = _np.sqrt(dx*dx+dy*dy)
        rs = Rf[iu]+Rf[ju]
        viol = rs - dvec
        mv = _np.max(viol)
        if mv > -1e-12:
            # shrink to fix
            # need scale t s.t. t*rs <= dvec -1e-9
            need = (dvec-2e-9)/rs
            # only where rs>0
            need = need[rs > 1e-15]
            if need.size:
                t = _np.min(need)
                if t < 1.0:
                    Rf = Rf*max(t, 1e-6)
        best_arr = _np.column_stack([Xf, Yf, Rf])
    except Exception:
        best_arr = _np.column_stack([best_x, best_y, best_r*0.99])
    return best_arr
