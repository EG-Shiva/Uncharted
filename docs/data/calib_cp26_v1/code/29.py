import numpy as np
import time

def solve():
    import numpy as np
    import time as _tm
    from scipy.optimize import minimize, linprog
    try:
        from scipy.spatial import Delaunay
        _has_del = True
    except Exception:
        _has_del = False

    n = 26
    t0 = _tm.time()
    DEADLINE = 124.0
    PHASE_A_END = 42.0

    iu, ju = np.triu_indices(n, 1)
    m = len(iu)
    A_ub0 = np.zeros((m, n), dtype=float)
    ar = np.arange(m)
    A_ub0[ar, iu] = 1.0
    A_ub0[ar, ju] = 1.0
    c_obj = -np.ones(n)

    rng_global = np.random.RandomState(7)

    def feasible_scale(X, Y, R):
        wall = np.minimum(np.minimum(X, 1.0 - X), np.minimum(Y, 1.0 - Y))
        Dx = X[:, None] - X[None, :]
        Dy = Y[:, None] - Y[None, :]
        D = np.sqrt(Dx * Dx + Dy * Dy)
        with np.errstate(divide='ignore', invalid='ignore'):
            wr = wall / np.maximum(R, 1e-12)
            pr = D / (R[:, None] + R[None, :] + 1e-30)
        np.fill_diagonal(pr, np.inf)
        a = float(np.min(wr))
        b = float(np.min(pr))
        alpha = min(1.0, a, b)
        alpha = alpha * (1.0 - 1e-9) - 1e-12
        if not np.isfinite(alpha) or alpha <= 0:
            alpha = 0.9
        return R * alpha, alpha

    def eval_lp(xc, yc, best_ref=-1.0):
        b = np.minimum(np.minimum(xc, 1.0 - xc), np.minimum(yc, 1.0 - yc))
        if np.min(b) < 1e-9:
            return None, -1.0
        if float(np.sum(b)) <= best_ref + 1e-12:
            return None, -1.0
        dx = xc[iu] - xc[ju]
        dy = yc[iu] - yc[ju]
        dvec = np.sqrt(dx * dx + dy * dy)
        # prune: if any pair too close, still feasible but sum limited; skip only if clearly bad
        bounds = [(0.0, float(bi)) for bi in b]
        try:
            res = linprog(c_obj, A_ub=A_ub0, b_ub=dvec, bounds=bounds, method='highs')
        except Exception:
            return None, -1.0
        if not res.success or res.x is None:
            return None, -1.0
        r = np.maximum(np.array(res.x, dtype=float).flatten(), 0.0)
        # strict feasibility shrink
        rf, alpha = feasible_scale(xc, yc, np.maximum(r, 1e-12))
        # if linprog was infeasible-ish alpha tiny, reject tiny sums
        s = float(np.sum(rf))
        if s <= best_ref + 1e-12 and best_ref > 0:
            # still return for completeness but caller checks improvement
            pass
        return rf, s

    # joint penalty fun factory
    def make_fun(nn):
        def fun(v, W):
            X = v[0:nn]; Y = v[nn:2*nn]; R = v[2*nn:3*nn]
            gx = np.zeros(nn); gy = np.zeros(nn); pen_gr = np.zeros(nn)
            pen = 0.0
            v1 = R - X
            mm = v1 > 0
            if np.any(mm):
                pen += np.sum(v1[mm]**2); gx[mm] += -2.0*v1[mm]; pen_gr[mm] += 2.0*v1[mm]
            v2 = X + R - 1.0
            mm = v2 > 0
            if np.any(mm):
                pen += np.sum(v2[mm]**2); gx[mm] += 2.0*v2[mm]; pen_gr[mm] += 2.0*v2[mm]
            v3 = R - Y
            mm = v3 > 0
            if np.any(mm):
                pen += np.sum(v3[mm]**2); gy[mm] += -2.0*v3[mm]; pen_gr[mm] += 2.0*v3[mm]
            v4 = Y + R - 1.0
            mm = v4 > 0
            if np.any(mm):
                pen += np.sum(v4[mm]**2); gy[mm] += 2.0*v4[mm]; pen_gr[mm] += 2.0*v4[mm]
            Dx = X[:, None] - X[None, :]
            Dy = Y[:, None] - Y[None, :]
            D = np.sqrt(Dx*Dx + Dy*Dy + 1e-20)
            S = R[:, None] + R[None, :]
            V = S - D
            np.fill_diagonal(V, 0.0)
            np.fill_diagonal(D, 1.0)
            Mask = V > 0
            if np.any(Mask):
                pen += 0.5*np.sum(V[Mask]**2)
                Q = np.zeros((nn, nn), dtype=float)
                Q[Mask] = -2.0*V[Mask]/D[Mask]
                gx += np.sum(Q*Dx, axis=1)
                gy += np.sum(Q*Dy, axis=1)
                G = np.zeros((nn, nn), dtype=float)
                G[Mask] = 2.0*V[Mask]
                pen_gr += np.sum(G, axis=1)
            loss = -np.sum(R) + W*pen
            g = np.empty_like(v)
            g[0:nn] = W*gx
            g[nn:2*nn] = W*gy
            g[2*nn:3*nn] = -1.0 + W*pen_gr
            return loss, g
        return fun
    fun = make_fun(n)
    bounds = [(0.0, 1.0)]*(2*n) + [(0.015, 0.22)]*n
    Ws = [1e2, 1e3, 1e4, 1e5]

    # fallback
    k = int(np.ceil(np.sqrt(n)))
    rr0 = 0.5 / k
    pts = []
    for i in range(k):
        for j in range(k):
            if len(pts) < n:
                pts.append([(2*i+1)*rr0, (2*j+1)*rr0, rr0*0.999])
    best_arr = np.array(pts, dtype=float)
    Rb_s, _ = feasible_scale(best_arr[:,0], best_arr[:,1], best_arr[:,2])
    best_arr[:,2] = Rb_s
    best_sum = float(np.sum(Rb_s))
    best_x = best_arr[:,0].copy(); best_y = best_arr[:,1].copy(); best_r = best_arr[:,2].copy()
    pool = []

    restart = 0
    while True:
        if _tm.time() - t0 > PHASE_A_END:
            break
        if restart > 60:
            break
        rs = np.random.RandomState(1000 + restart*977)
        if restart == 0:
            # 5x5 + extra jitter
            X0 = np.zeros(n); Y0 = np.zeros(n)
            idx = 0
            for i in range(5):
                for j in range(5):
                    if idx < 25:
                        X0[idx] = 0.1+0.2*i; Y0[idx] = 0.1+0.2*j; idx+=1
            X0[25] = 0.5; Y0[25] = 0.5
            X0 += rs.randn(n)*0.01; Y0 += rs.randn(n)*0.01
            R0 = np.full(n, 0.085) + rs.randn(n)*0.005
        elif restart % 3 == 1:
            X0 = rs.uniform(0.05, 0.95, size=n)
            Y0 = rs.uniform(0.05, 0.95, size=n)
            R0 = rs.uniform(0.05, 0.11, size=n)
        elif restart % 3 == 2:
            xs = (np.arange(6)+0.5)/6.0; ys = (np.arange(5)+0.5)/5.0
            xx, yy = np.meshgrid(xs, ys)
            xy = np.column_stack([xx.ravel(), yy.ravel()])
            perm = rs.permutation(len(xy))[:n]
            X0 = xy[perm,0]+rs.randn(n)*0.008
            Y0 = xy[perm,1]+rs.randn(n)*0.008
            R0 = rs.uniform(0.04, 0.10, size=n)
        else:
            X0 = rs.uniform(0.08, 0.92, size=n)
            Y0 = rs.uniform(0.08, 0.92, size=n)
            R0 = 0.07+0.04*rs.rand(n)
        X0 = np.clip(X0, 0.03, 0.97); Y0 = np.clip(Y0, 0.03, 0.97)
        R0 = np.clip(R0, 0.015, 0.22)
        v = np.concatenate([X0, Y0, R0])
        for W in Ws:
            if _tm.time() - t0 > PHASE_A_END:
                break
            try:
                res = minimize(fun, v, args=(W,), method='L-BFGS-B', jac=True,
                               bounds=bounds, options={'maxiter': 120, 'ftol':1e-9, 'gtol':1e-5})
                v = res.x
            except Exception:
                break
        X = np.clip(v[0:n], 0.0, 1.0); Y = np.clip(v[n:2*n], 0.0, 1.0)
        X = np.clip(X, 0.02, 0.98); Y = np.clip(Y, 0.02, 0.98)
        r_opt, s = eval_lp(X, Y, best_sum - 1e-9)
        if r_opt is not None and s > best_sum + 1e-10:
            best_sum = s; best_x = X.copy(); best_y = Y.copy(); best_r = r_opt.copy()
        if r_opt is not None:
            pool.append((s, X.copy(), Y.copy()))
        restart += 1

    # keep diverse pool: top 6
    pool = sorted(pool, key=lambda t: -t[0])[:8]

    def void_candidates(xc, yc, rc):
        cands = []
        # grid
        gx = np.linspace(0.04, 0.96, 10)
        gy = np.linspace(0.04, 0.96, 10)
        for a in gx:
            for b in gy:
                cands.append([a, b])
        cands = np.array(cands, dtype=float)
        # Delaunay circumcenters
        if _has_del:
            try:
                pts2 = np.column_stack([xc, yc])
                tri = Delaunay(pts2)
                for s in tri.simplices:
                    p1, p2, p3 = pts2[s[0]], pts2[s[1]], pts2[s[2]]
                    d = 2.0*(p1[0]*(p2[1]-p3[1])+p2[0]*(p3[1]-p1[1])+p3[0]*(p1[1]-p2[1]))
                    if abs(d) < 1e-12:
                        continue
                    s1 = p1[0]**2+p1[1]**2; s2 = p2[0]**2+p2[1]**2; s3 = p3[0]**2+p3[1]**2
                    ux = (s1*(p2[1]-p3[1])+s2*(p3[1]-p1[1])+s3*(p1[1]-p2[1]))/d
                    uy = (s1*(p3[0]-p2[0])+s2*(p1[0]-p3[0])+s3*(p2[0]-p1[0]))/d
                    if 0.02 < ux < 0.98 and 0.02 < uy < 0.98:
                        cands = np.vstack([cands, [ux, uy]])
            except Exception:
                pass
        # score by clearance to centres and walls
        scores = []
        for (cx, cy) in cands:
            dd = np.sqrt((xc-cx)**2 + (yc-cy)**2)
            wall = min(cx, 1-cx, cy, 1-cy)
            # clearance using centre distances (generous)
            scores.append(min(float(np.min(dd)), wall))
        scores = np.array(scores)
        order = np.argsort(-scores)
        # return top 24
        return cands[order[:24]]

    # ---- Phase B: alternate void jumps + single-circle compass ----
    sigmas = [0.02, 0.01, 0.005, 0.0025, 0.0012, 0.0006, 0.0003]
    # also try pool members as starting points for compass if time allows
    # first improve best with void jumps
    n_void_rounds = 0
    dirs4 = np.array([[1,0],[-1,0],[0,1],[0,-1]], dtype=float)
    dirs8 = np.array([[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]], dtype=float)/np.array([[1],[1],[1],[1],[np.sqrt(2)],[np.sqrt(2)],[np.sqrt(2)],[np.sqrt(2)]])

    rs_local = np.random.RandomState(1234)
    # quick compass function
    def compass_pass(xc, yc, sigma, cur_sum, order, use8=False):
        dirs = dirs8 if use8 else dirs4
        improved = False
        for i in order:
            if _tm.time() - t0 > DEADLINE - 1.0:
                break
            best_local_sum = cur_sum
            best_px, best_py = xc[i], yc[i]
            for d in dirs:
                nx = xc[i] + sigma*d[0]; ny = yc[i] + sigma*d[1]
                if nx < 0.02 or nx > 0.98 or ny < 0.02 or ny > 0.98:
                    continue
                xt = xc.copy(); yt = yc.copy()
                xt[i] = nx; yt[i] = ny
                r2, s2 = eval_lp(xt, yt, best_local_sum)
                if r2 is not None and s2 > best_local_sum + 1e-11:
                    best_local_sum = s2
                    best_px, best_py = nx, ny
                    best_r2 = r2
            if best_local_sum > cur_sum + 1e-11:
                xc[i] = best_px; yc[i] = best_py
                cur_sum = best_local_sum
                # refresh radii for subsequent scoring (re-eval to get r)
                # eval already gave r for best; re-evaluate cheap? keep
                r_new, _ = eval_lp(xc, yc, -1.0)
                # update rc reference via closure variable? return it
                improved = True
        # get final radii
        rf, sf = eval_lp(xc, yc, -1.0)
        return xc, yc, rf, sf, improved

    # try each pool member briefly with compass at small sigma to find better topology
    # (topology selection)
    if len(pool) > 1 and (_tm.time()-t0) < DEADLINE-30:
        for (s0, px0, py0) in pool[1:]:
            if _tm.time() - t0 > DEADLINE - 30:
                break
            # two quick compass passes at 0.005
            xc = px0.copy(); yc = py0.copy()
            rf0, sf0 = eval_lp(xc, yc, -1.0)
            if rf0 is None:
                continue
            order = rs_local.permutation(n)
            xc, yc, rf, sf, imp = compass_pass(xc, yc, 0.005, sf0, order, use8=False)
            if sf > best_sum + 1e-10:
                best_sum = sf; best_x = xc.copy(); best_y = yc.copy(); best_r = rf.copy()

    # main loop
    outer = 0
    while _tm.time() - t0 < DEADLINE - 1.5:
        outer += 1
        # --- void jump round ---
        if outer % 2 == 1:
            vc = void_candidates(best_x, best_y, best_r)
            # small circles first
            order_small = np.argsort(best_r)[:10]
            best_trial_sum = best_sum
            best_trial_x = None; best_trial_y = None; best_trial_r = None
            # limit trials: iterate voids x smalls
            for si in order_small:
                if _tm.time() - t0 > DEADLINE - 2.0:
                    break
                for vj in range(len(vc)):
                    if _tm.time() - t0 > DEADLINE - 2.0:
                        break
                    xt = best_x.copy(); yt = best_y.copy()
                    xt[si] = float(np.clip(vc[vj,0] + rs_local.randn()*0.002, 0.02, 0.98))
                    yt[si] = float(np.clip(vc[vj,1] + rs_local.randn()*0.002, 0.02, 0.98))
                    r2, s2 = eval_lp(xt, yt, best_trial_sum)
                    if r2 is not None and s2 > best_trial_sum + 1e-10:
                        best_trial_sum = s2
                        best_trial_x = xt; best_trial_y = yt; best_trial_r = r2
            if best_trial_x is not None:
                best_sum = best_trial_sum; best_x = best_trial_x; best_y = best_trial_y; best_r = best_trial_r
                continue  # re-do void jumps while they pay
        # --- compass sweep at current sigma level ---
        # pick sigma cyclically decreasing; after full cycle with no improvement, do jitter
        progressed = False
        for sigma in sigmas:
            if _tm.time() - t0 > DEADLINE - 1.5:
                break
            order = rs_local.permutation(n)
            # two passes per sigma
            for rep in range(2):
                if _tm.time() - t0 > DEADLINE - 1.5:
                    break
                use8 = (sigma < 0.002)
                nbx, nby, nbr, nbs, imp = compass_pass(best_x.copy(), best_y.copy(), sigma, best_sum, order, use8=use8)
                if nbs > best_sum + 1e-11:
                    best_x, best_y, best_r, best_sum = nbx, nby, nbr, nbs
                    progressed = True
                if not imp:
                    break
                order = rs_local.permutation(n)
        if not progressed:
            # small global jitter escape (single attempt)
            if _tm.time() - t0 > DEADLINE - 5:
                break
            xt = best_x + rs_local.randn(n)*0.003
            yt = best_y + rs_local.randn(n)*0.003
            xt = np.clip(xt, 0.02, 0.98); yt = np.clip(yt, 0.02, 0.98)
            r2, s2 = eval_lp(xt, yt, best_sum)
            if r2 is not None and s2 > best_sum + 1e-11:
                best_x, best_y, best_r, best_sum = xt, yt, r2, s2
            else:
                # if many outers with no progress, break
                if outer > 12:
                    break
        if outer > 40:
            break

    x = np.clip(best_x, 0.0, 1.0); y = np.clip(best_y, 0.0, 1.0); r = np.maximum(best_r, 1e-9)
    # final strict repair
    rf, _ = feasible_scale(x, y, r)
    out = np.column_stack([x, y, rf])
    return out
