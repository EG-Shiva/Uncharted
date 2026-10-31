import numpy as np
import time

def solve():
    n = 26
    t0 = time.time()
    DEADLINE = 120.0

    rng = np.random.default_rng(0)
    rng2 = np.random.default_rng(1)

    # ---------- helpers ----------
    def uniform_scale(X, Y, R):
        R = np.maximum(np.asarray(R, dtype=float), 1e-12)
        X = np.asarray(X, dtype=float); Y = np.asarray(Y, dtype=float)
        wall_ratio = np.minimum(np.minimum(X, 1.0 - X), np.minimum(Y, 1.0 - Y)) / R
        Dx = X[:, None] - X[None, :]
        Dy = Y[:, None] - Y[None, :]
        D = np.sqrt(Dx*Dx + Dy*Dy)
        S = R[:, None] + R[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = D / S
        np.fill_diagonal(ratio, np.inf)
        a = float(np.min(wall_ratio))
        b = float(np.min(ratio))
        alpha = min(1.0, a, b)
        alpha = alpha * (1.0 - 1e-7) - 1e-12
        if (not np.isfinite(alpha)) or alpha <= 0:
            alpha = 0.5
        return R * alpha, alpha

    # prebuilt LP pattern
    iu, ju = np.triu_indices(n, 1)
    m = len(iu)
    A_ub0 = np.zeros((m, n), dtype=float)
    _ar = np.arange(m)
    A_ub0[_ar, iu] = 1.0
    A_ub0[_ar, ju] = 1.0
    c_obj = -np.ones(n)

    from scipy.optimize import minimize, linprog

    def lp_opt(Xc, Yc):
        Xc = np.asarray(Xc, dtype=float); Yc = np.asarray(Yc, dtype=float)
        wall = np.minimum(np.minimum(Xc, 1.0 - Xc), np.minimum(Yc, 1.0 - Yc))
        if np.min(wall) < 1e-9:
            return None, -1.0
        dx = Xc[iu] - Xc[ju]
        dy = Yc[iu] - Yc[ju]
        dvec = np.sqrt(dx*dx + dy*dy)
        # need strictly positive upper bounds; linprog tolerates 0
        bounds = [(0.0, float(bi)) for bi in np.maximum(wall, 0.0)]
        try:
            res = linprog(c_obj, A_ub=A_ub0, b_ub=dvec, bounds=bounds, method='highs')
        except Exception:
            return None, -1.0
        if (not res.success) or res.x is None:
            return None, -1.0
        r = np.maximum(np.asarray(res.x, dtype=float).flatten(), 0.0)
        # shrink for strict 1e-12 feasibility
        r = r * (1.0 - 1e-7)
        r = np.maximum(r - 5e-13, 0.0)
        # quick verify and rescale if violated by numerics
        # use uniform_scale guard
        try:
            rs, _ = uniform_scale(Xc, Yc, np.maximum(r, 1e-12))
            # uniform_scale applies extra shrink; use it as safe value
            # but avoid double shrink if already safe: check if scale ~1
            # just return rs
            return rs, float(np.sum(rs))
        except Exception:
            return r, float(np.sum(r))

    def make_fun(n_):
        def fun(v, W):
            X = v[0:n_]
            Y = v[n_:2*n_]
            R = v[2*n_:3*n_]
            loss = -np.sum(R)
            gx = np.zeros(n_)
            gy = np.zeros(n_)
            gr_pen = np.zeros(n_)
            # walls: using squared penalty
            v1 = R - X
            m1 = v1 > 0
            if np.any(m1):
                gx[m1] += -2.0*v1[m1]
                gr_pen[m1] += 2.0*v1[m1]
            v2 = X + R - 1.0
            m2 = v2 > 0
            if np.any(m2):
                gx[m2] += 2.0*v2[m2]
                gr_pen[m2] += 2.0*v2[m2]
            v3 = R - Y
            m3 = v3 > 0
            if np.any(m3):
                gy[m3] += -2.0*v3[m3]
                gr_pen[m3] += 2.0*v3[m3]
            v4 = Y + R - 1.0
            m4 = v4 > 0
            if np.any(m4):
                gy[m4] += 2.0*v4[m4]
                gr_pen[m4] += 2.0*v4[m4]
            Dx = X[:, None] - X[None, :]
            Dy = Y[:, None] - Y[None, :]
            D = np.sqrt(Dx*Dx + Dy*Dy + 1e-20)
            S = R[:, None] + R[None, :]
            V = S - D
            np.fill_diagonal(V, 0.0)
            np.fill_diagonal(D, 1.0)
            Mask = V > 0
            pen_wall = 0.0
            # wall pen value
            if np.any(m1): pen_wall += np.sum(v1[m1]**2)
            if np.any(m2): pen_wall += np.sum(v2[m2]**2)
            if np.any(m3): pen_wall += np.sum(v3[m3]**2)
            if np.any(m4): pen_wall += np.sum(v4[m4]**2)
            pen_pair = 0.0
            Gx = np.zeros(n_); Gy = np.zeros(n_); Gr = np.zeros(n_)
            if np.any(Mask):
                pen_pair = 0.5*np.sum(V[Mask]**2)
                Q = np.zeros((n_, n_))
                Q[Mask] = -2.0*V[Mask]/D[Mask]
                Hx = Q*Dx; Hy = Q*Dy
                Gx = np.sum(Hx, axis=1)
                Gy = np.sum(Hy, axis=1)
                G = np.zeros((n_, n_))
                G[Mask] = 2.0*V[Mask]
                Gr = np.sum(G, axis=1)
            pen = pen_wall + pen_pair
            loss = loss + W*pen
            g = np.empty_like(v)
            g[0:n_] = W*(gx + Gx)
            g[n_:2*n_] = W*(gy + Gy)
            g[2*n_:3*n_] = -1.0 + W*(gr_pen + Gr)
            return loss, g
        return fun

    fun = make_fun(n)
    bounds = [(0.0, 1.0)]*(2*n) + [(0.03, 0.20)]*n
    bounds_short = [(0.0, 1.0)]*(2*n) + [(0.01, 0.25)]*n

    # fallback feasible
    k = int(np.ceil(np.sqrt(n)))
    r0f = 0.5 / k
    pts = []
    for i in range(k):
        for j in range(k):
            if len(pts) < n:
                pts.append([(2*i+1)*r0f, (2*j+1)*r0f, r0f*0.999])
    best_arr = np.array(pts, dtype=float)
    Xb, Yb, Rb = best_arr[:,0], best_arr[:,1], best_arr[:,2]
    Rb_s, _ = uniform_scale(Xb, Yb, Rb)
    best_arr = np.column_stack([Xb, Yb, Rb_s])
    best_sum = float(np.sum(Rb_s))
    best_X = Xb.copy(); best_Y = Yb.copy(); best_R = Rb_s.copy()

    Ws = [1e2, 1e3, 1e4, 1e5]

    # ---------- Phase A: restarts ----------
    PHASE_A_END = 52.0
    restart = 0
    while True:
        if time.time() - t0 > PHASE_A_END:
            break
        if restart > 60:
            break
        if restart == 0:
            X0 = np.zeros(n); Y0 = np.zeros(n); R0 = np.full(n, 0.095)
            idx = 0
            for i in range(5):
                for j in range(5):
                    if idx < 25:
                        X0[idx] = (i+0.5)/5.0 + rng.normal(0, 0.005)
                        Y0[idx] = (j+0.5)/5.0 + rng.normal(0, 0.005)
                        idx += 1
            X0[25] = 0.9 + rng.normal(0, 0.01)
            Y0[25] = 0.1 + rng.normal(0, 0.01)
            X0 = np.clip(X0, 0.05, 0.95); Y0 = np.clip(Y0, 0.05, 0.95)
        elif restart == 1:
            pts_l = []
            for ix in range(6):
                for iy in range(5):
                    if len(pts_l) < n:
                        xx = (ix+0.5)/6.0
                        yy = (iy+0.5)/5.0
                        if iy % 2 == 1:
                            xx += 0.5/6.0*0.5
                            if xx > 0.95: xx -= 1.0/6.0
                        pts_l.append((xx, yy))
            arr = np.array(pts_l)
            X0 = arr[:,0] + rng.normal(0, 0.005, n)
            Y0 = arr[:,1] + rng.normal(0, 0.005, n)
            X0 = np.clip(X0, 0.05, 0.95); Y0 = np.clip(Y0, 0.05, 0.95)
            R0 = 0.085 + 0.02*rng.random(n)
        else:
            X0 = rng.random(n)*0.9+0.05
            Y0 = rng.random(n)*0.9+0.05
            R0 = 0.085 + 0.025*rng.random(n)
        v = np.concatenate([X0, Y0, R0])
        for W in Ws:
            if time.time() - t0 > PHASE_A_END:
                break
            try:
                res = minimize(fun, v, args=(W,), jac=True, method='L-BFGS-B',
                               bounds=bounds,
                               options={'maxiter': 220, 'ftol': 1e-9, 'gtol': 1e-7})
                v = res.x
            except Exception:
                break
        X = np.clip(v[0:n], 0.0, 1.0)
        Y = np.clip(v[n:2*n], 0.0, 1.0)
        R = np.clip(v[2*n:3*n], 0.01, 0.25)
        try:
            Rs, _ = uniform_scale(X, Y, R)
            s_uni = float(np.sum(Rs))
            if s_uni > best_sum:
                best_sum = s_uni
                best_X, best_Y, best_R = X.copy(), Y.copy(), Rs.copy()
        except Exception:
            pass
        try:
            Ropt, s_lp = lp_opt(X, Y)
            if Ropt is not None and s_lp > best_sum:
                best_sum = s_lp
                best_X, best_Y, best_R = X.copy(), Y.copy(), Ropt.copy()
        except Exception:
            pass
        restart += 1

    # ---------- helpers for surgery ----------
    # grid cache
    G = 121
    gx1 = np.linspace(0, 1, G)
    gy1 = np.linspace(0, 1, G)
    XXm, YYm = np.meshgrid(gx1, gy1)
    PX = XXm.ravel(); PY = YYm.ravel()
    NP = PX.size
    wall_grid = np.minimum(np.minimum(PX, 1.0-PX), np.minimum(PY, 1.0-PY))

    def best_void(Xc, Yc, Rc, exclude_mask=None):
        # clearance of each grid point w.r.t circles not excluded
        # returns idx, clearance
        if exclude_mask is None:
            Xv = Xc; Yv = Yc; Rv = Rc
        else:
            keep = ~exclude_mask
            if np.sum(keep) == 0:
                # empty: clearance = wall only
                idx = int(np.argmax(wall_grid))
                return idx, float(wall_grid[idx])
            Xv = Xc[keep]; Yv = Yc[keep]; Rv = Rc[keep]
        # vectorised in chunks to limit memory
        best_val = -1.0
        best_idx = 0
        chunk = 4096
        for s in range(0, NP, chunk):
            e = min(s+chunk, NP)
            px = PX[s:e, None]; py = PY[s:e, None]
            dx = px - Xv[None, :]
            dy = py - Yv[None, :]
            d = np.sqrt(dx*dx + dy*dy) - Rv[None, :]
            mnd = np.min(d, axis=1)
            clr = np.minimum(wall_grid[s:e], mnd)
            j = int(np.argmax(clr))
            if float(clr[j]) > best_val:
                best_val = float(clr[j])
                best_idx = s + j
        return best_idx, best_val

    def contact_counts(Xc, Yc, Rc):
        wall = np.minimum(np.minimum(Xc, 1.0-Xc), np.minimum(Yc, 1.0-Yc))
        wg = wall - Rc
        Dx = Xc[:, None] - Xc[None, :]
        Dy = Yc[:, None] - Yc[None, :]
        D = np.sqrt(Dx*Dx + Dy*Dy)
        S = Rc[:, None] + Rc[None, :]
        gap = D - S
        np.fill_diagonal(gap, np.inf)
        cnt = np.zeros(n, dtype=int)
        for i in range(n):
            c = 0
            if wg[i] < 0.004:
                # count walls that are tight (approx by min wall)
                c += 1
            c += int(np.sum(gap[i] < 0.004))
            cnt[i] = c
        return cnt

    def short_refine(Xs, Ys, Rs, maxiter=120, W=2e4):
        v0 = np.concatenate([Xs, Ys, Rs])
        try:
            res = minimize(fun, v0, args=(W,), jac=True, method='L-BFGS-B',
                           bounds=bounds_short,
                           options={'maxiter': maxiter, 'ftol': 1e-9, 'gtol': 1e-6})
            v = res.x
        except Exception:
            return Xs, Ys, Rs
        return np.clip(v[0:n], 0, 1), np.clip(v[n:2*n], 0, 1), np.clip(v[2*n:3*n], 0.005, 0.3)

    # ---------- Phase B: topology surgery + SA ----------
    cur_X, cur_Y, cur_R = best_X.copy(), best_Y.copy(), best_R.copy()
    cur_sum = best_sum
    # SA schedule
    T0 = 0.004
    T1 = 0.00025
    max_surg = 400
    # estimate total time left
    for it in range(max_surg):
        if time.time() - t0 > DEADLINE - 6:
            break
        frac = it / float(max_surg)
        # exponential cooling
        T = T0 * (T1 / T0) ** frac
        # periodic elite restore
        if it % 30 == 29:
            # restore elite with small jitter to diversify
            cur_X, cur_Y, cur_R = best_X.copy(), best_Y.copy(), best_R.copy()
            cur_sum = best_sum
            # tiny jitter one random circle to escape exact lock
            jj = int(rng2.integers(0, n))
            cur_X[jj] = np.clip(cur_X[jj] + rng2.normal(0, 0.004), 0.02, 0.98)
            cur_Y[jj] = np.clip(cur_Y[jj] + rng2.normal(0, 0.004), 0.02, 0.98)
            Rtmp, stmp = lp_opt(cur_X, cur_Y)
            if Rtmp is not None:
                cur_R = Rtmp; cur_sum = stmp
            continue
        # choose k
        u = rng2.random()
        if u < 0.60:
            kk = 1
        elif u < 0.85:
            kk = 2
        else:
            kk = 3
        # choose victims
        order_small = np.argsort(cur_R)
        cc = contact_counts(cur_X, cur_Y, cur_R)
        # least-contacted ordering
        order_contact = np.lexsort((cur_R, cc))  # sorts by cc then R? lexsort keys reversed: last is primary
        # actually want primary cc, secondary R
        order_contact = np.array(sorted(range(n), key=lambda i: (cc[i], cur_R[i])))
        rpick = rng2.random()
        if rpick < 0.55:
            victims = order_small[:kk]
        elif rpick < 0.8:
            # random among 6 smallest
            pool = order_small[:6]
            victims = rng2.choice(pool, size=kk, replace=False)
        else:
            victims = order_contact[:kk]
        victims = np.asarray(victims, dtype=int)
        # greedy reinsertion
        Xp = cur_X.copy(); Yp = cur_Y.copy(); Rp = cur_R.copy()
        # mark removed
        removed = np.zeros(n, dtype=bool)
        removed[victims] = True
        # for greedy, we keep remaining radii as cur_R, victims to be placed one by one
        # order victims by placing smallest clearance need first? place in given order
        # shuffle placement order randomly for diversity when kk>1
        porder = victims[np.argsort(rng2.random(kk))] if kk > 1 else victims
        for vdx in porder:
            excl = removed.copy()
            # temporarily keep vdx as excluded, but already-placed victims are included
            # removed currently True for all victims not yet placed; for current vdx keep excluded,
            # for already placed set False
            # We maintain: removed True = not yet placed (excluding current). Placed ones set False before search? 
            # Simplest: build mask of circles to consider = all except not-yet-placed (including current)
            # Already placed have removed=False now? Let's update: start all victims removed=True.
            # When we place one, set removed[that]=False after placing.
            # So for current vdx, excluded set = removed (includes vdx + unplaced others), which is correct.
            bidx, bval = best_void(Xp, Yp, Rp, exclude_mask=removed)
            bx = float(PX[bidx]); by = float(PY[bidx])
            # add small jitter to avoid grid lock
            jx = float(rng2.normal(0, 0.002)); jy = float(rng2.normal(0, 0.002))
            bx = float(np.clip(bx + jx, bval*0.5 if bval < 0.5 else 0.02, 1.0 - (bval*0.5 if bval < 0.5 else 0.02))) if bval > 1e-4 else bx
            by = float(np.clip(by + jy, bval*0.5 if bval < 0.5 else 0.02, 1.0 - (bval*0.5 if bval < 0.5 else 0.02))) if bval > 1e-4 else by
            # ensure wall feasibility for bval
            bval = max(float(bval), 0.005)
            # also ensure not too close to existing due to jitter: recompute safe radius quickly
            dx = bx - Xp[~removed]; dy = by - Yp[~removed]
            # ~removed includes remaining + already placed? Actually ~removed = available circles
            if np.sum(~removed) > 0:
                dd = np.sqrt(dx*dx + dy*dy) - Rp[~removed]
                safe = float(min(np.min(dd), min(bx, 1-bx, by, 1-by)))
                safe = max(safe, 0.005)
            else:
                safe = float(min(bx, 1-bx, by, 1-by))
            Xp[vdx] = bx; Yp[vdx] = by; Rp[vdx] = safe * 0.98
            removed[vdx] = False
        # LP polish
        Rlp, slp = lp_opt(Xp, Yp)
        if Rlp is None:
            continue
        # short penalty re-opt to move centres
        # use LP radii as start
        Xo, Yo, Ro = short_refine(Xp, Yp, Rlp, maxiter=90, W=2e4)
        Rlp2, slp2 = lp_opt(Xo, Yo)
        if Rlp2 is None:
            # fall back to pre-refine
            Xo, Yo, Rlp2, slp2 = Xp, Yp, Rlp, slp
        delta = slp2 - cur_sum
        accept = False
        if delta >= 0:
            accept = True
        else:
            if T > 1e-12 and rng2.random() < np.exp(delta / T):
                accept = True
        if accept:
            cur_X, cur_Y, cur_R, cur_sum = Xo.copy(), Yo.copy(), Rlp2.copy(), float(slp2)
            if cur_sum > best_sum:
                best_sum = cur_sum
                best_X, best_Y, best_R = cur_X.copy(), cur_Y.copy(), cur_R.copy()

    # final safety
    try:
        Rf, _ = uniform_scale(best_X, best_Y, best_R)
        out = np.column_stack([best_X, best_Y, Rf])
    except Exception:
        out = np.column_stack([best_X, best_Y, best_R])
    return out
