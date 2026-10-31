import numpy as np
import time as _time

def solve():
    import time as _tm
    import numpy as _np
    from scipy.optimize import minimize, linprog
    n = 26
    t0 = _tm.time()
    HARD_END = 119.0
    # ---------- preallocated LP data ----------
    iu, ju = _np.triu_indices(n, 1)
    m = len(iu)
    A_ub0 = _np.zeros((m, n), dtype=float)
    _ar = _np.arange(m)
    A_ub0[_ar, iu] = 1.0
    A_ub0[_ar, ju] = 1.0
    c_obj = -_np.ones(n)

    def eval_lp(xc, yc):
        b = _np.minimum(_np.minimum(xc, 1.0 - xc), _np.minimum(yc, 1.0 - yc))
        b = _np.maximum(b, 0.0)
        dx = xc[iu] - xc[ju]
        dy = yc[iu] - yc[ju]
        dvec = _np.sqrt(dx * dx + dy * dy)
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

    def safe_radii(xc, yc, r):
        # uniform shrink to satisfy strictly with margin 1e-9
        eps = 1e-9
        r = _np.maximum(r, 0.0) * (1.0 - 1e-7)
        # wall
        b = _np.minimum(_np.minimum(xc, 1.0 - xc), _np.minimum(yc, 1.0 - yc))
        # avoid zero division
        for _ in range(2):
            dx = xc[iu] - xc[ju]; dy = yc[iu] - yc[ju]
            dvec = _np.sqrt(dx * dx + dy * dy)
            rs = r[iu] + r[ju]
            worst = 1.0
            # wall factor
            pos = r > 1e-12
            if _np.any(pos):
                wf = _np.min((b[pos] - eps) / r[pos]) if _np.any(pos) else 1.0
                if wf < worst:
                    worst = wf
            mask = rs > 1e-12
            if _np.any(mask):
                pf = _np.min((dvec[mask] - eps) / rs[mask])
                if pf < worst:
                    worst = pf
            if worst >= 1.0:
                break
            if worst <= 0:
                r = r * 0.5
                break
            r = r * worst * (1.0 - 1e-9)
        r = _np.maximum(r, 1e-6)
        return r

    def fallback():
        k = int(_np.ceil(_np.sqrt(n)))
        rr = 0.5 / k
        pts = []
        for i in range(k):
            for j in range(k):
                if len(pts) < n:
                    pts.append([(2 * i + 1) * rr, (2 * j + 1) * rr, rr * 0.999 * 0.99])
        return _np.array(pts, dtype=float)

    # ---------- Phase A: fast penalty restarts for incumbent ----------
    bnds = [(0.0, 1.0)] * (2 * n) + [(1e-4, 0.35)] * n
    P_seq = [30.0, 300.0, 3000.0]
    best_sum = -1.0
    best_x = None; best_y = None; best_r = None

    def run_penalty(x0, y0, r0):
        z = _np.concatenate([x0, y0, r0])
        for P in P_seq:
            if _tm.time() - t0 > 32.0:
                break
            def fg(zz, P=P):
                x = zz[0:n]; y = zz[n:2 * n]; r = zz[2 * n:3 * n]
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
                    W = _np.zeros((n, n))
                    W[mask] = (2.0 * P * S[mask]) / (D[mask] + 1e-12)
                    rs = W.sum(axis=1)
                    gx += W.dot(x) - rs * x
                    gy += W.dot(y) - rs * y
                    Spos = _np.where(mask, S, 0.0)
                    gr += 2.0 * P * Spos.sum(axis=1)
                g = _np.concatenate([gx, gy, gr])
                return f, g
            try:
                res = minimize(fg, z, method='L-BFGS-B', jac=True, bounds=bnds,
                               options={'maxiter': 80, 'ftol': 1e-9, 'gtol': 1e-5})
                z = res.x
            except Exception:
                break
        return z

    restart_idx = 0
    rngA = _np.random.RandomState(12345)
    while _tm.time() - t0 < 32.0 and restart_idx < 60:
        mode = restart_idx % 5
        if restart_idx == 0:
            pts = []
            for i in range(5):
                for j in range(5):
                    pts.append([0.1 + 0.2 * i, 0.1 + 0.2 * j])
            pts = _np.array(pts)
            xy = _np.vstack([pts, _np.array([[0.3, 0.3]])])
            x0 = xy[:, 0].copy(); y0 = xy[:, 1].copy()
            r0 = _np.full(n, 0.09); r0[-1] = 0.04
            x0 += rngA.randn(n) * 0.003; y0 += rngA.randn(n) * 0.003
            x0 = _np.clip(x0, 0.05, 0.95); y0 = _np.clip(y0, 0.05, 0.95)
        elif mode == 1:
            x0 = rngA.uniform(0.05, 0.95, size=n)
            y0 = rngA.uniform(0.05, 0.95, size=n)
            r0 = rngA.uniform(0.05, 0.10, size=n)
        elif mode == 2:
            xs = (_np.arange(6) + 0.5) / 6.0; ys = (_np.arange(5) + 0.5) / 5.0
            xx, yy = _np.meshgrid(xs, ys)
            xy = _np.column_stack([xx.ravel(), yy.ravel()])
            perm = rngA.permutation(len(xy))[:n]
            x0 = xy[perm, 0] + rngA.randn(n) * 0.006
            y0 = xy[perm, 1] + rngA.randn(n) * 0.006
            x0 = _np.clip(x0, 0.05, 0.95); y0 = _np.clip(y0, 0.05, 0.95)
            r0 = rngA.uniform(0.06, 0.095, size=n)
        elif mode == 3:
            xy = []
            nr = 5
            for rr_ in range(nr):
                y_ = (rr_ + 0.5) / nr
                for cc in range(6):
                    x_ = (cc + 0.5 + (0.5 if rr_ % 2 == 1 else 0.0)) / 6.0
                    if x_ > 0.97:
                        continue
                    xy.append([x_, y_])
            xy = _np.array(xy)
            perm = rngA.permutation(len(xy))[:n]
            x0 = xy[perm, 0] + rngA.randn(n) * 0.008
            y0 = xy[perm, 1] + rngA.randn(n) * 0.008
            x0 = _np.clip(x0, 0.03, 0.97); y0 = _np.clip(y0, 0.03, 0.97)
            r0 = rngA.uniform(0.06, 0.10, size=n)
        else:
            if best_x is not None and rngA.rand() < 0.5:
                x0 = best_x + rngA.randn(n) * 0.03
                y0 = best_y + rngA.randn(n) * 0.03
                x0 = _np.clip(x0, 0.03, 0.97); y0 = _np.clip(y0, 0.03, 0.97)
                r0 = _np.maximum(best_r, 0.02) + rngA.randn(n) * 0.01
                r0 = _np.clip(r0, 0.02, 0.20)
            else:
                x0 = rngA.uniform(0.03, 0.97, size=n)
                y0 = rngA.uniform(0.03, 0.97, size=n)
                r0 = rngA.uniform(0.04, 0.11, size=n)
        try:
            z = run_penalty(x0, y0, r0)
            xc = _np.clip(z[0:n], 0.0, 1.0); yc = _np.clip(z[n:2 * n], 0.0, 1.0)
            bq = _np.minimum(_np.minimum(xc, 1.0 - xc), _np.minimum(yc, 1.0 - yc))
            if _np.sum(bq) <= best_sum:
                restart_idx += 1
                continue
            r_opt, s = eval_lp(xc, yc)
            if r_opt is not None and s > best_sum:
                best_sum = float(s); best_x = xc.copy(); best_y = yc.copy(); best_r = r_opt.copy()
        except Exception:
            pass
        restart_idx += 1

    if best_x is None:
        return fallback()

    # ---------- Phase B: centre-only basin-hopping ----------
    rng = _np.random.RandomState(999)
    # elite pool of 10 best distinct
    elites = [(float(best_sum), best_x.copy(), best_y.copy(), best_r.copy())]
    cur_x = best_x.copy(); cur_y = best_y.copy(); cur_r = best_r.copy(); cur_s = float(best_sum)

    # 41x41 grid
    gl = _np.linspace(0.02, 0.98, 41)
    GXm, GYm = _np.meshgrid(gl, gl)
    Gx = GXm.ravel(); Gy = GYm.ravel()
    nG = Gx.size

    def elite_add(s, nx, ny, nr):
        for (es, ex, ey, er) in elites:
            if abs(es - s) < 1e-9:
                dd = _np.min([_np.max(_np.abs(nx - ex)) + _np.max(_np.abs(ny - ey)),
                              _np.max(_np.abs(nx - (1 - ex))) + _np.max(_np.abs(ny - ey))])
                if dd < 1e-3:
                    return
        elites.append((float(s), nx.copy(), ny.copy(), nr.copy()))
        elites.sort(key=lambda e: -e[0])
        del elites[10:]

    def dihedral(xc, yc, k):
        if k == 1:
            return yc.copy(), 1.0 - xc.copy()
        elif k == 2:
            return 1.0 - xc.copy(), 1.0 - yc.copy()
        elif k == 3:
            return 1.0 - yc.copy(), xc.copy()
        elif k == 4:
            return 1.0 - xc.copy(), yc.copy()
        elif k == 5:
            return xc.copy(), 1.0 - yc.copy()
        elif k == 6:
            return yc.copy(), xc.copy()
        else:
            return 1.0 - yc.copy(), 1.0 - xc.copy()

    T_start = 0.002
    T_end = 0.00005
    phase_start = _tm.time() - t0
    budget = HARD_END - phase_start - 1.5
    if budget < 5:
        budget = 5.0
    it = 0
    ruin_count = 0
    sigmas = [0.03, 0.01, 0.003]

    while _tm.time() - t0 < HARD_END - 1.0 and it < 60000:
        it += 1
        el = _tm.time() - t0 - phase_start
        prog = min(1.0, max(0.0, el / budget))
        # geometric cooling
        Temp = T_start * ((T_end / T_start) ** prog)
        u = float(rng.rand())
        try:
            if u < 0.70:
                # single / pair translations bursts
                if float(rng.rand()) < 0.78:
                    sigma = sigmas[int(rng.randint(3))]
                    idx = int(rng.randint(n))
                    nx = cur_x.copy(); ny = cur_y.copy()
                    nx[idx] += float(rng.randn()) * sigma
                    ny[idx] += float(rng.randn()) * sigma
                    nx[idx] = min(0.99, max(0.01, float(nx[idx])))
                    ny[idx] = min(0.99, max(0.01, float(ny[idx])))
                else:
                    sigma = sigmas[int(rng.randint(3))]
                    i1, i2 = rng.choice(n, 2, replace=False)
                    i1 = int(i1); i2 = int(i2)
                    nx = cur_x.copy(); ny = cur_y.copy()
                    nx[i1] += float(rng.randn()) * sigma
                    ny[i1] += float(rng.randn()) * sigma
                    nx[i2] += float(rng.randn()) * sigma
                    ny[i2] += float(rng.randn()) * sigma
                    nx[i1] = min(0.99, max(0.01, float(nx[i1])))
                    ny[i1] = min(0.99, max(0.01, float(ny[i1])))
                    nx[i2] = min(0.99, max(0.01, float(nx[i2])))
                    ny[i2] = min(0.99, max(0.01, float(ny[i2])))
                r_opt, s = eval_lp(nx, ny)
                if r_opt is None:
                    continue
                if s > cur_s or float(rng.rand()) < _np.exp((s - cur_s) / max(Temp, 1e-9)):
                    cur_x, cur_y, cur_r, cur_s = nx, ny, r_opt, float(s)
                    if s > elites[0][0] + 1e-12:
                        elite_add(float(s), nx, ny, r_opt)
            elif u < 0.80:
                # swap large-small
                order = _np.argsort(cur_r)
                small_pool = order[:8]; large_pool = order[-8:]
                i_s = int(small_pool[int(rng.randint(len(small_pool)))])
                i_L = int(large_pool[int(rng.randint(len(large_pool)))])
                if i_s == i_L:
                    continue
                nx = cur_x.copy(); ny = cur_y.copy()
                tx, ty = float(nx[i_s]), float(ny[i_s])
                nx[i_s], ny[i_s] = float(nx[i_L]), float(ny[i_L])
                nx[i_L], ny[i_L] = tx, ty
                nx[i_s] += float(rng.randn()) * 0.002
                ny[i_s] += float(rng.randn()) * 0.002
                nx[i_L] += float(rng.randn()) * 0.002
                ny[i_L] += float(rng.randn()) * 0.002
                nx = _np.clip(nx, 0.01, 0.99); ny = _np.clip(ny, 0.01, 0.99)
                r_opt, s = eval_lp(nx, ny)
                if r_opt is None:
                    continue
                if s > cur_s or float(rng.rand()) < _np.exp((s - cur_s) / max(Temp, 1e-9)):
                    cur_x, cur_y, cur_r, cur_s = nx, ny, r_opt, float(s)
                    if s > elites[0][0] + 1e-12:
                        elite_add(float(s), nx, ny, r_opt)
            elif u < 0.85:
                # reflect / rotate whole set by multiples of 90 deg
                k = int(rng.randint(1, 8))
                nx, ny = dihedral(cur_x, cur_y, k)
                nx = nx + rng.randn(n) * 0.0005
                ny = ny + rng.randn(n) * 0.0005
                nx = _np.clip(nx, 0.01, 0.99); ny = _np.clip(ny, 0.01, 0.99)
                r_opt, s = eval_lp(nx, ny)
                if r_opt is None:
                    continue
                if s > cur_s or float(rng.rand()) < _np.exp((s - cur_s) / max(Temp, 1e-9)):
                    cur_x, cur_y, cur_r, cur_s = nx, ny, r_opt, float(s)
                    if s > elites[0][0] + 1e-12:
                        elite_add(float(s), nx, ny, r_opt)
            elif u < 0.95:
                # relocate one large circle into largest empty region from dense grid
                # compute clearance
                DX = Gx[:, None] - cur_x[None, :]
                DY = Gy[:, None] - cur_y[None, :]
                D = _np.sqrt(DX * DX + DY * DY)
                Clear = D - cur_r[None, :]
                minc = _np.min(Clear, axis=1)
                wall = _np.minimum(_np.minimum(Gx, 1.0 - Gx), _np.minimum(Gy, 1.0 - Gy))
                score_g = _np.minimum(wall, minc)
                pick = int(_np.argmax(score_g))
                order = _np.argsort(cur_r)
                # one large circle: random among top half
                cand_large = order[-13:]
                idx = int(cand_large[int(rng.randint(len(cand_large)))])
                nx = cur_x.copy(); ny = cur_y.copy()
                nx[idx] = float(Gx[pick]) + float(rng.randn()) * 0.002
                ny[idx] = float(Gy[pick]) + float(rng.randn()) * 0.002
                nx[idx] = min(0.99, max(0.01, float(nx[idx])))
                ny[idx] = min(0.99, max(0.01, float(ny[idx])))
                r_opt, s = eval_lp(nx, ny)
                if r_opt is None:
                    continue
                if s > cur_s or float(rng.rand()) < _np.exp((s - cur_s) / max(Temp, 1e-9)):
                    cur_x, cur_y, cur_r, cur_s = nx, ny, r_opt, float(s)
                    if s > elites[0][0] + 1e-12:
                        elite_add(float(s), nx, ny, r_opt)
                # occasional restart from elite if stuck: jump to random elite
                if float(rng.rand()) < 0.02 and len(elites) > 1:
                    e = elites[int(rng.randint(len(elites)))]
                    # Metropolis jump to elite regardless
                    cur_x, cur_y, cur_r, cur_s = e[1].copy(), e[2].copy(), e[3].copy(), float(e[0])
            else:
                # ruin: remove three smallest and greedily reinsert on 41x41 grid
                if ruin_count > 12:
                    continue
                if _tm.time() - t0 > HARD_END - 15.0:
                    continue
                ruin_count += 1
                order = _np.argsort(cur_r)
                removed = [int(order[0]), int(order[1]), int(order[2])]
                base_x = cur_x.copy(); base_y = cur_y.copy()
                # greedy reinsert one at a time
                for ridx, idx in enumerate(removed):
                    best_g = -1.0
                    best_px = float(base_x[idx]); best_py = float(base_y[idx])
                    best_rr = None
                    # scan grid; to save time, stride 1 but early wall filter
                    for g in range(nG):
                        px = float(Gx[g]); py = float(Gy[g])
                        # quick wall upper bound: if wall too small skip? keep all
                        tx = base_x.copy(); ty = base_y.copy()
                        tx[idx] = px; ty[idx] = py
                        r_opt2, s2 = eval_lp(tx, ty)
                        if r_opt2 is None:
                            continue
                        if s2 > best_g:
                            best_g = float(s2); best_px = px; best_py = py; best_rr = r_opt2
                        if _tm.time() - t0 > HARD_END - 2.0:
                            break
                    base_x[idx] = best_px; base_y[idx] = best_py
                    # update cur_r to best for next insertion evaluation
                    if best_rr is not None:
                        cur_r = best_rr
                    if _tm.time() - t0 > HARD_END - 2.0:
                        break
                r_opt, s = eval_lp(base_x, base_y)
                if r_opt is None:
                    continue
                # accept with Metropolis (ruin often worsens then improves via greedy so accept greedily if better than cur)
                if s > cur_s or float(rng.rand()) < _np.exp((s - cur_s) / max(Temp, 1e-9)):
                    cur_x, cur_y, cur_r, cur_s = base_x, base_y, r_opt, float(s)
                    if s > elites[0][0] + 1e-12:
                        elite_add(float(s), base_x, base_y, r_opt)
        except Exception:
            continue
        # periodic elite refresh: if cur much worse than best, jump back sometimes
        if it % 500 == 0 and len(elites) > 0:
            if cur_s < elites[0][0] - 0.01 and float(rng.rand()) < 0.5:
                e = elites[0]
                cur_x, cur_y, cur_r, cur_s = e[1].copy(), e[2].copy(), e[3].copy(), float(e[0])

    # ---------- return best feasible elite ----------
    elites.sort(key=lambda e: -e[0])
    bx, by, br = elites[0][1], elites[0][2], elites[0][3]
    br_safe = safe_radii(bx, by, br)
    # final clip centres to respect walls strictly
    for i in range(n):
        ri = float(br_safe[i])
        bx[i] = min(1.0 - ri - 1e-9, max(ri + 1e-9, float(bx[i])))
        by[i] = min(1.0 - ri - 1e-9, max(ri + 1e-9, float(by[i])))
    # one more shrink after clipping
    br_safe = safe_radii(bx, by, br_safe)
    out = _np.column_stack([bx, by, br_safe])
    return out
