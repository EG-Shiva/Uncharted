import numpy as np
import time as _time

def solve():
    import time as _tm
    import numpy as _np
    from scipy.optimize import minimize, linprog
    n = 26
    t0 = _tm.time()
    HARD = 124.0
    iu, ju = _np.triu_indices(n, 1)
    m = len(iu)
    A0 = _np.zeros((m, n), dtype=float)
    _ar = _np.arange(m)
    A0[_ar, iu] = 1.0
    A0[_ar, ju] = 1.0
    c_obj = -_np.ones(n, dtype=float)
    rng = _np.random.RandomState(42)

    # fallback feasible 6x5 grid first 26
    fx = []; fy = []
    for ix in range(6):
        for jy in range(5):
            fx.append((ix + 0.5) / 6.0); fy.append((jy + 0.5) / 5.0)
    fx = _np.array(fx[:n]); fy = _np.array(fy[:n])
    fr = _np.full(n, 0.08)
    best_xc = fx.copy(); best_yc = fy.copy(); best_r = fr.copy()
    best_sum = float(_np.sum(fr))  # 2.08

    bnds78 = [(0.0, 1.0)] * (2 * n) + [(0.015, 0.24)] * n
    P_seq_full = [30.0, 300.0, 3000.0, 30000.0]
    P_seq_short = [300.0, 3000.0, 30000.0]

    def eval_lp(xc, yc):
        # returns (sum, r) or (-1, None)
        try:
            b = _np.minimum(_np.minimum(xc, 1.0 - xc), _np.minimum(yc, 1.0 - yc))
            if _np.any(b < 1e-9):
                return -1.0, None
            b = _np.maximum(b, 0.0)
            dx = xc[iu] - xc[ju]
            dy = yc[iu] - yc[ju]
            dvec = _np.sqrt(dx * dx + dy * dy)
            # need dvec>0
            dvec = _np.maximum(dvec, 1e-12)
            bounds = [(0.0, float(bi)) for bi in b]
            res = linprog(c_obj, A_ub=A0, b_ub=dvec, bounds=bounds, method='highs')
            if res.status != 0 or res.x is None:
                return -1.0, None
            r = _np.asarray(res.x, dtype=float).flatten()
            r = _np.maximum(r, 0.0)
            return float(_np.sum(r)), r
        except Exception:
            return -1.0, None

    def run_penalty(x0, y0, r0, P_seq, maxiter_each):
        z = _np.concatenate([x0, y0, r0])
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
                    f += P * _np.sum(v1[mm] * v1[mm])
                    co = 2.0 * P * v1[mm]
                    gx[mm] -= co; gr[mm] += co
                v2 = x + r - 1.0
                mm = v2 > 0
                if _np.any(mm):
                    f += P * _np.sum(v2[mm] * v2[mm])
                    co = 2.0 * P * v2[mm]
                    gx[mm] += co; gr[mm] += co
                v3 = r - y
                mm = v3 > 0
                if _np.any(mm):
                    f += P * _np.sum(v3[mm] * v3[mm])
                    co = 2.0 * P * v3[mm]
                    gy[mm] -= co; gr[mm] += co
                v4 = y + r - 1.0
                mm = v4 > 0
                if _np.any(mm):
                    f += P * _np.sum(v4[mm] * v4[mm])
                    co = 2.0 * P * v4[mm]
                    gy[mm] += co; gr[mm] += co
                dxm = x[:, None] - x[None, :]
                dym = y[:, None] - y[None, :]
                D = _np.sqrt(dxm * dxm + dym * dym)
                Rsum = r[:, None] + r[None, :]
                S = Rsum - D
                _np.fill_diagonal(S, -1.0)
                mask = S > 0
                if _np.any(mask):
                    f += P * 0.5 * _np.sum(S[mask] * S[mask])
                    Dm = D + 1e-12
                    W = _np.zeros((n, n))
                    W[mask] = (2.0 * P * S[mask]) / Dm[mask]
                    rs = W.sum(axis=1)
                    gx += W.dot(x) - rs * x
                    gy += W.dot(y) - rs * y
                    Spos = _np.where(mask, S, 0.0)
                    gr += 2.0 * P * Spos.sum(axis=1)
                g = _np.concatenate([gx, gy, gr])
                return f, g
            try:
                res = minimize(fg, z, method='L-BFGS-B', jac=True, bounds=bnds78,
                               options={'maxiter': maxiter_each, 'ftol': 1e-10, 'gtol': 1e-6})
                z = _np.asarray(res.x, dtype=float)
            except Exception:
                break
        return z

    def make_init(mode):
        # returns x0,y0,r0
        if mode == 0:
            # 5x5 + center
            bx = []; by = []
            for ix in range(5):
                for jy in range(5):
                    bx.append((ix + 0.5) / 5.0); by.append((jy + 0.5) / 5.0)
            bx.append(0.5); by.append(0.5)
            bx = _np.array(bx); by = _np.array(by)
            jx = bx + rng.normal(0, 0.015, n)
            jy = by + rng.normal(0, 0.015, n)
            r0 = _np.full(n, 0.09) + rng.normal(0, 0.01, n)
            return _np.clip(jx, 0.05, 0.95), _np.clip(jy, 0.05, 0.95), _np.clip(r0, 0.02, 0.18)
        elif mode == 1:
            # 6x5 subset shuffled + jitter
            gx = []; gy = []
            for ix in range(6):
                for jy in range(5):
                    gx.append((ix + 0.5) / 6.0); gy.append((jy + 0.5) / 5.0)
            gx = _np.array(gx); gy = _np.array(gy)
            perm = rng.permutation(len(gx))[:n]
            # if n==26 from 30, perm gives 26
            jx = gx[perm] + rng.normal(0, 0.02, n)
            jy = gy[perm] + rng.normal(0, 0.02, n)
            r0 = _np.full(n, 0.085) + rng.normal(0, 0.012, n)
            return _np.clip(jx, 0.03, 0.97), _np.clip(jy, 0.03, 0.97), _np.clip(r0, 0.02, 0.18)
        elif mode == 2:
            # staggered rows template random counts summing to 26
            # choose 5 rows
            # random composition
            counts = _np.array([5, 5, 5, 5, 6])
            rng.shuffle(counts)
            # counts sum 26
            xs = []; ys = []
            for ri in range(5):
                cnt = int(counts[ri])
                y = (ri + 0.5) / 5.0 + rng.normal(0, 0.01)
                off = 0.5 / cnt if (ri % 2 == 1) else 0.0
                for k in range(cnt):
                    x = (k + 0.5 + off * 0.5) / (cnt + 0.2) + rng.normal(0, 0.01)
                    xs.append(x); ys.append(y)
            xs = _np.array(xs); ys = _np.array(ys)
            # ensure length n (it is 26)
            r0 = _np.full(n, 0.085) + rng.uniform(-0.02, 0.02, n)
            return _np.clip(xs, 0.03, 0.97), _np.clip(ys, 0.03, 0.97), _np.clip(r0, 0.02, 0.18)
        elif mode == 3:
            # staggered columns (transpose)
            counts = _np.array([5, 5, 5, 5, 6])
            rng.shuffle(counts)
            xs = []; ys = []
            for ci in range(5):
                cnt = int(counts[ci])
                x = (ci + 0.5) / 5.0 + rng.normal(0, 0.01)
                for k in range(cnt):
                    y = (k + 0.5) / (cnt + 0.2) + rng.normal(0, 0.01)
                    xs.append(x); ys.append(y)
            xs = _np.array(xs); ys = _np.array(ys)
            r0 = _np.full(n, 0.085) + rng.uniform(-0.02, 0.02, n)
            return _np.clip(xs, 0.03, 0.97), _np.clip(ys, 0.03, 0.97), _np.clip(r0, 0.02, 0.18)
        elif mode == 4:
            # hex-ish + random radii tiered (few large, many mid, few small)
            x0 = rng.uniform(0.05, 0.95, n)
            y0 = rng.uniform(0.05, 0.95, n)
            # tiered radii
            r0 = _np.empty(n)
            # 6 large 0.11, 14 mid 0.09, 6 small 0.06
            r0[:6] = 0.11 + rng.normal(0, 0.008, 6)
            r0[6:20] = 0.088 + rng.normal(0, 0.008, 14)
            r0[20:] = 0.06 + rng.normal(0, 0.008, 6)
            rng.shuffle(r0)
            return x0, y0, _np.clip(r0, 0.02, 0.18)
        elif mode == 5:
            # uniform random
            x0 = rng.uniform(0.06, 0.94, n)
            y0 = rng.uniform(0.06, 0.94, n)
            r0 = rng.uniform(0.05, 0.12, n)
            return x0, y0, r0
        else:
            # perturbed best if available else random
            if best_sum > 2.2 and rng.rand() < 0.7:
                sig = 0.03 if rng.rand() < 0.5 else 0.012
                xn = _np.clip(best_xc + rng.normal(0, sig, n), 0.02, 0.98)
                yn = _np.clip(best_yc + rng.normal(0, sig, n), 0.02, 0.98)
                rn = _np.clip(best_r * (1.0 + rng.normal(0, 0.08, n)), 0.02, 0.20)
                # occasionally shuffle two centres
                if rng.rand() < 0.3:
                    a, b = rng.choice(n, 2, replace=False)
                    xn[a], xn[b] = xn[b], xn[a]
                    yn[a], yn[b] = yn[b], yn[a]
                return xn, yn, rn
            else:
                x0 = rng.uniform(0.05, 0.95, n)
                y0 = rng.uniform(0.05, 0.95, n)
                r0 = rng.uniform(0.05, 0.115, n)
                return x0, y0, r0

    # quick screen: evaluate LP for a few structured inits to seed best
    for _ in range(6):
        if _tm.time() - t0 > 20.0:
            break
        mx = _ % 4
        tx, ty, _tr = make_init(mx)
        s, rr = eval_lp(tx, ty)
        if s > best_sum and rr is not None:
            best_sum = s; best_xc = tx.copy(); best_yc = ty.copy(); best_r = rr.copy()

    # Phase 1: fast penalty restarts
    restart = 0
    while True:
        if _tm.time() - t0 > 62.0:
            break
        if restart > 90:
            break
        mode = restart % 7
        # first few force diverse modes
        x0, y0, r0 = make_init(mode)
        # adapt maxiter: early larger, later smaller
        mi = 90 if restart < 12 else 70
        z = run_penalty(x0, y0, r0, P_seq_full, mi)
        xc = _np.clip(z[0:n], 0.005, 0.995); yc = _np.clip(z[n:2*n], 0.005, 0.995)
        s, rr = eval_lp(xc, yc)
        if s > best_sum + 1e-12 and rr is not None:
            best_sum = s; best_xc = xc.copy(); best_yc = yc.copy(); best_r = rr.copy()
        restart += 1

    # helper: largest voids
    def largest_voids(xc, yc, rc, G=36, K=6):
        lin = _np.linspace(0.0, 1.0, G)
        PX, PY = _np.meshgrid(lin, lin)
        px = PX.ravel(); py = PY.ravel()
        # wall clearance
        wall = _np.minimum(_np.minimum(px, 1.0 - px), _np.minimum(py, 1.0 - py))
        # distance to circles
        # (G*G, n)
        dx = px[:, None] - xc[None, :]
        dy = py[:, None] - yc[None, :]
        D = _np.sqrt(dx * dx + dy * dy) - rc[None, :]
        mind = D.min(axis=1)
        clear = _np.minimum(wall, mind)
        idx = _np.argsort(clear)[::-1]
        picks = []
        for ii in idx:
            if clear[ii] < 0.01:
                break
            ok = True
            for (qx, qy) in picks:
                if (px[ii] - qx) ** 2 + (py[ii] - qy) ** 2 < 0.05 ** 2:
                    ok = False; break
            if ok:
                picks.append((float(px[ii]), float(py[ii]), float(clear[ii])))
                if len(picks) >= K:
                    break
        return picks

    # Phase 2: ruin-and-recreate on best
    n_ruin = 0
    while True:
        if _tm.time() - t0 > 102.0:
            break
        if n_ruin > 120:
            break
        # need valid best_r
        try:
            voids = largest_voids(best_xc, best_yc, best_r, G=34, K=5)
        except Exception:
            voids = []
        op = n_ruin % 4
        xn = best_xc.copy(); yn = best_yc.copy(); rn = best_r.copy()
        if op == 0 and len(voids) > 0:
            # teleport smallest circles to voids
            order = _np.argsort(rn)  # smallest first
            k = min(len(voids), 3 if rng.rand() < 0.7 else 2)
            for t in range(k):
                ci = int(order[t]) if rng.rand() < 0.8 else int(rng.choice(n))
                vx, vy, vc = voids[t % len(voids)]
                xn[ci] = _np.clip(vx + rng.normal(0, 0.005), 0.02, 0.98)
                yn[ci] = _np.clip(vy + rng.normal(0, 0.005), 0.02, 0.98)
                rn[ci] = _np.clip(vc * 0.85, 0.02, 0.18)
            # small shake others
            xn = _np.clip(xn + rng.normal(0, 0.004, n), 0.015, 0.985)
            yn = _np.clip(yn + rng.normal(0, 0.004, n), 0.015, 0.985)
        elif op == 1:
            # swap large-small positions
            order = _np.argsort(rn)
            # swap 2-3 pairs
            for _ in range(2):
                a = int(order[rng.randint(0, 6)])  # small
                b = int(order[rng.randint(n - 6, n)])  # large
                xn[a], xn[b] = xn[b], xn[a]
                yn[a], yn[b] = yn[b], yn[a]
            xn = _np.clip(xn + rng.normal(0, 0.003, n), 0.015, 0.985)
            yn = _np.clip(yn + rng.normal(0, 0.003, n), 0.015, 0.985)
        elif op == 2:
            # shake all moderately
            sig = 0.015 if rng.rand() < 0.5 else 0.008
            xn = _np.clip(xn + rng.normal(0, sig, n), 0.015, 0.985)
            yn = _np.clip(yn + rng.normal(0, sig, n), 0.015, 0.985)
            rn = _np.clip(rn * (1.0 + rng.normal(0, 0.05, n)), 0.02, 0.20)
        else:
            # single-circle teleport to best void + jitter
            if len(voids) > 0:
                ci = int(rng.choice(n))
                vx, vy, vc = voids[0]
                # pick worst circle: minimal (r / wall?) use smallest r
                if rng.rand() < 0.6:
                    ci = int(_np.argmin(rn))
                xn[ci] = _np.clip(vx + rng.normal(0, 0.004), 0.02, 0.98)
                yn[ci] = _np.clip(vy + rng.normal(0, 0.004), 0.02, 0.98)
                rn[ci] = _np.clip(vc * 0.9, 0.02, 0.18)
            else:
                xn = _np.clip(xn + rng.normal(0, 0.01, n), 0.015, 0.985)
                yn = _np.clip(yn + rng.normal(0, 0.01, n), 0.015, 0.985)
        z = run_penalty(xn, yn, rn, P_seq_short, 55)
        xc = _np.clip(z[0:n], 0.005, 0.995); yc = _np.clip(z[n:2*n], 0.005, 0.995)
        s, rr = eval_lp(xc, yc)
        if s > best_sum + 1e-12 and rr is not None:
            best_sum = s; best_xc = xc.copy(); best_yc = yc.copy(); best_r = rr.copy()
        n_ruin += 1

    # Phase 3: LP hill-climbing on centres (gradient-free, exact objective)
    sig = 0.012
    it = 0
    while True:
        if _tm.time() - t0 > HARD - 2.0:
            break
        if it > 2500:
            break
        # anneal sigma
        if it % 300 == 0 and it > 0:
            sig *= 0.7
            sig = max(sig, 0.0015)
        # propose: mostly global small shake, sometimes single-circle jump to void
        if it % 25 == 24:
            try:
                voids = largest_voids(best_xc, best_yc, best_r, G=30, K=3)
            except Exception:
                voids = []
            if len(voids) > 0 and rng.rand() < 0.8:
                xn = best_xc.copy(); yn = best_yc.copy()
                ci = int(_np.argmin(best_r)) if rng.rand() < 0.7 else int(rng.choice(n))
                vx, vy, _vc = voids[int(rng.choice(len(voids)))]
                xn[ci] = _np.clip(vx + rng.normal(0, 0.003), 0.01, 0.99)
                yn[ci] = _np.clip(vy + rng.normal(0, 0.003), 0.01, 0.99)
            else:
                xn = _np.clip(best_xc + rng.normal(0, sig, n), 0.01, 0.99)
                yn = _np.clip(best_yc + rng.normal(0, sig, n), 0.01, 0.99)
        else:
            # move subset of circles for higher acceptance
            xn = best_xc.copy(); yn = best_yc.copy()
            if rng.rand() < 0.6:
                # move 1-3 circles
                k = 1 if rng.rand() < 0.6 else (2 if rng.rand() < 0.7 else 4)
                idxs = rng.choice(n, k, replace=False)
                xn[idxs] = _np.clip(xn[idxs] + rng.normal(0, sig * 1.5, k), 0.01, 0.99)
                yn[idxs] = _np.clip(yn[idxs] + rng.normal(0, sig * 1.5, k), 0.01, 0.99)
            else:
                xn = _np.clip(xn + rng.normal(0, sig, n), 0.01, 0.99)
                yn = _np.clip(yn + rng.normal(0, sig, n), 0.01, 0.99)
        s, rr = eval_lp(xn, yn)
        if s > best_sum + 1e-10 and rr is not None:
            best_sum = s; best_xc = xn.copy(); best_yc = yn.copy(); best_r = rr.copy()
        it += 1

    # finalize with strict shrink
    try:
        xb = _np.clip(best_xc, 0.0, 1.0); yb = _np.clip(best_yc, 0.0, 1.0)
        rb = _np.maximum(best_r, 1e-12)
        wall = _np.minimum(_np.minimum(xb, 1.0 - xb), _np.minimum(yb, 1.0 - yb))
        Dx = xb[:, None] - xb[None, :]
        Dy = yb[:, None] - yb[None, :]
        D = _np.sqrt(Dx * Dx + Dy * Dy)
        with _np.errstate(divide='ignore', invalid='ignore'):
            wr = wall / rb
            pr = D / (rb[:, None] + rb[None, :])
        _np.fill_diagonal(pr, _np.inf)
        a = float(_np.min(wr)); b = float(_np.min(pr))
        alpha = min(1.0, a, b)
        if not _np.isfinite(alpha) or alpha <= 0:
            raise ValueError("infeasible")
        alpha = alpha * (1.0 - 1e-7) - 1e-12
        if alpha <= 0 or not _np.isfinite(alpha):
            alpha = 0.99
        rb = rb * alpha
        rb = _np.maximum(rb, 1e-12)
    except Exception:
        # ultra-safe fallback grid
        xb = fx; yb = fy; rb = _np.full(n, 0.08 * 0.999)
    out = _np.column_stack([xb, yb, rb])
    return out
