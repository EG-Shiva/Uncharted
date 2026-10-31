import numpy as np
import time

def solve():
    import numpy as np
    import time
    from scipy.optimize import minimize, linprog
    try:
        from scipy.spatial import Delaunay
        HAS_DEL = True
    except Exception:
        HAS_DEL = False
    N = 26
    t0 = time.time()
    BUDGET = 125.0
    PHASE_A_END = 34.0
    rng = np.random.RandomState(7)
    LB = 0.001
    EPS = 1e-9

    def lp_opt(x, y):
        wall = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        if np.min(wall) < LB + EPS:
            return None, -1.0
        Dx = x[:, None] - x[None, :]
        Dy = y[:, None] - y[None, :]
        D = np.sqrt(Dx * Dx + Dy * Dy)
        rows_i = []
        rows_j = []
        rows_b = []
        for i in range(N):
            for j in range(i + 1, N):
                dij = D[i, j]
                if dij >= wall[i] + wall[j] - 1e-12:
                    continue
                lim = dij - EPS
                if lim < 2 * LB - 1e-12:
                    return None, -1.0
                rows_i.append(i); rows_j.append(j); rows_b.append(lim)
        m = len(rows_b)
        if m == 0:
            r = wall - EPS * 2.0
            r = np.maximum(r, LB)
            r = r * (1.0 - 1e-9)
            return r, float(np.sum(r))
        A_ub = np.zeros((m, N), dtype=np.float64)
        b_ub = np.array(rows_b, dtype=np.float64)
        for k in range(m):
            A_ub[k, rows_i[k]] = 1.0
            A_ub[k, rows_j[k]] = 1.0
        c = -np.ones(N, dtype=np.float64)
        bounds = [(LB, float(wall[i] - EPS)) for i in range(N)]
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        except Exception:
            return None, -1.0
        if not res.success:
            return None, -1.0
        r = np.array(res.x, dtype=np.float64) * (1.0 - 5e-9)
        return r, float(np.sum(r))

    best_sum = -1.0
    best_x = None; best_y = None; best_r = None

    def consider(x, y):
        global_best = None
        nonlocal_best = [best_sum]
        # use closure via lists
        return False

    # inline consider
    def try_consider(x, y):
        nonlocal best_sum, best_x, best_y, best_r
        r, s = lp_opt(x, y)
        if r is not None and s > best_sum + 1e-12:
            best_sum = s
            best_x = x.copy(); best_y = y.copy(); best_r = r.copy()
            return True
        return False

    bounds_box = [(0.0, 1.0)] * (2 * N) + [(0.008, 0.28)] * N
    gx = []; gy = []
    for ix in range(6):
        for jy in range(5):
            gx.append((ix + 0.5) / 6.0); gy.append((jy + 0.5) / 5.0)
    gx = np.array(gx); gy = np.array(gy)

    restart = 0
    while (time.time() - t0) < PHASE_A_END and restart < 40:
        if restart % 3 == 0:
            u0 = rng.rand(N); v0 = rng.rand(N)
        elif restart % 3 == 1:
            perm = rng.permutation(len(gx))[:N]
            u0 = gx[perm] + (rng.rand(N) - 0.5) * 0.08
            v0 = gy[perm] + (rng.rand(N) - 0.5) * 0.08
            u0 = np.clip(u0, 0.0, 1.0); v0 = np.clip(v0, 0.0, 1.0)
        else:
            u0 = rng.rand(N); v0 = rng.rand(N)
        r0 = 0.06 + 0.05 * rng.rand(N) if (restart % 2 == 0) else 0.04 + 0.09 * rng.rand(N)
        r0 = np.clip(r0, 0.01, 0.27)
        z = np.concatenate([u0, v0, r0])
        for W in (30.0, 300.0, 3000.0, 20000.0):
            if (time.time() - t0) >= PHASE_A_END:
                break
            def fun(zz, W=W):
                u = zz[:N]; vv = zz[N:2*N]; rr = zz[2*N:]
                om2r = 1.0 - 2.0 * rr
                x = rr + om2r * u; y = rr + om2r * vv
                Dx = x[:, None] - x[None, :]; Dy = y[:, None] - y[None, :]
                D = np.sqrt(Dx*Dx + Dy*Dy + 1e-16)
                np.fill_diagonal(D, 1.0)
                E = rr[:, None] + rr[None, :] - D
                np.fill_diagonal(E, 0.0)
                Ep = np.maximum(E, 0.0)
                Pen = 0.5 * np.sum(Ep * Ep)
                F = -np.sum(rr) + W * Pen
                InvD = 1.0 / D; np.fill_diagonal(InvD, 0.0)
                Wmat = Ep * InvD
                Gx = -2.0 * np.sum(Wmat * Dx, axis=1)
                Gy = -2.0 * np.sum(Wmat * Dy, axis=1)
                sumEp = np.sum(Ep, axis=1)
                dFdu = W * Gx * om2r; dFdv = W * Gy * om2r
                dFdr = -np.ones(N) + W * (2.0*sumEp + Gx*(1.0-2.0*u) + Gy*(1.0-2.0*vv))
                return F, np.concatenate([dFdu, dFdv, dFdr])
            try:
                res = minimize(fun, z, method='L-BFGS-B', jac=True, bounds=bounds_box,
                               options={'maxiter': 120, 'ftol': 1e-9, 'gtol': 1e-5})
                z = res.x
            except Exception:
                break
        u = z[:N]; vv = z[N:2*N]; rr = z[2*N:]
        x = rr + (1.0 - 2.0*rr) * u; y = rr + (1.0 - 2.0*rr) * vv
        x = np.clip(x, 0.005, 0.995); y = np.clip(y, 0.005, 0.995)
        try_consider(x, y)
        restart += 1

    if best_x is None:
        k = int(np.ceil(np.sqrt(N)))
        r = 0.5 / k
        pts = []
        for i in range(k):
            for j in range(k):
                if len(pts) < N:
                    pts.append([(2*i+1)*r, (2*j+1)*r, r*0.999999])
        return np.array(pts, dtype=np.float64)

    def contact_counts(xc, yc, rc, tol=0.004):
        wall = np.minimum(np.minimum(xc, 1.0-xc), np.minimum(yc, 1.0-yc))
        cnt = np.zeros(len(xc), dtype=int)
        Dx = xc[:, None] - xc[None, :]; Dy = yc[:, None] - yc[None, :]
        D = np.sqrt(Dx*Dx + Dy*Dy)
        for i in range(len(xc)):
            c = 0
            if wall[i] - rc[i] < tol:
                c += 1
            for j in range(len(xc)):
                if j == i: continue
                if D[i, j] - rc[i] - rc[j] < tol:
                    c += 1
            cnt[i] = c
        return cnt

    def delaunay_voids(xc, yc, rc, kneed):
        # xc,yc,rc are remaining (M)
        M = len(xc)
        cands = []
        # corners
        corners = np.array([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]])
        try:
            if HAS_DEL and M >= 2:
                P = np.vstack([np.column_stack([xc, yc]), corners])
                tri = Delaunay(P)
                # circumcentres
                for s in tri.simplices:
                    i0, i1, i2 = int(s[0]), int(s[1]), int(s[2])
                    ax, ay = float(P[i0,0]), float(P[i0,1])
                    bx, by = float(P[i1,0]), float(P[i1,1])
                    cx, cy = float(P[i2,0]), float(P[i2,1])
                    Dd = 2.0*(ax*(by-cy)+bx*(cy-ay)+cx*(ay-by))
                    if abs(Dd) < 1e-12:
                        continue
                    a2 = ax*ax+ay*ay; b2 = bx*bx+by*by; c2 = cx*cx+cy*cy
                    ux = (a2*(by-cy)+b2*(cy-ay)+c2*(ay-by))/Dd
                    uy = (a2*(cx-bx)+b2*(ax-cx)+c2*(bx-ax))/Dd
                    ux = min(max(ux, 0.0), 1.0); uy = min(max(uy, 0.0), 1.0)
                    wall = min(ux, 1.0-ux, uy, 1.0-uy)
                    dx = ux - xc; dy = uy - yc
                    dd = np.sqrt(dx*dx+dy*dy) - rc
                    dm = float(np.min(dd)) if M>0 else wall
                    clear = wall if wall < dm else dm
                    cands.append((ux, uy, clear))
                # edge midpoints (unique)
                edges = set()
                for s in tri.simplices:
                    for a,b in [(s[0],s[1]),(s[1],s[2]),(s[2],s[0])]:
                        a=int(a); b=int(b)
                        if a>b: a,b=b,a
                        edges.add((a,b))
                nP = P.shape[0]
                for (a,b) in edges:
                    if a >= nP-4 and b >= nP-4:
                        continue
                    mx = (float(P[a,0])+float(P[b,0]))*0.5
                    my = (float(P[a,1])+float(P[b,1]))*0.5
                    mx = min(max(mx,0.0),1.0); my = min(max(my,0.0),1.0)
                    wall = min(mx,1.0-mx,my,1.0-my)
                    dx = mx - xc; dy = my - yc
                    dd = np.sqrt(dx*dx+dy*dy) - rc
                    dm = float(np.min(dd)) if M>0 else wall
                    clear = wall if wall < dm else dm
                    cands.append((mx, my, clear))
        except Exception:
            pass
        # add random samples for robustness
        try:
            R = 300
            rx = rng.rand(R); ry = rng.rand(R)
            for t in range(R):
                ux=float(rx[t]); uy=float(ry[t])
                wall = min(ux,1.0-ux,uy,1.0-uy)
                dx = ux - xc; dy = uy - yc
                dd = np.sqrt(dx*dx+dy*dy) - rc
                dm = float(np.min(dd))
                clear = wall if wall < dm else dm
                cands.append((ux, uy, clear))
        except Exception:
            pass
        if len(cands)==0:
            return [(0.5,0.5,0.0)]*kneed
        cands.sort(key=lambda t: t[2], reverse=True)
        picked = []
        for (ux,uy,cl) in cands:
            if len(picked) >= kneed*3:
                break
            ok = True
            for (px,py,_pc) in picked:
                if (ux-px)*(ux-px)+(uy-py)*(uy-py) < 0.03*0.03:
                    ok=False; break
            if ok or len(picked)<kneed:
                # allow close if still need
                if not ok and len(picked)>=kneed:
                    continue
                picked.append((ux,uy,cl))
            if len(picked) >= kneed:
                # continue to get extras for choice but stop early
                if len(picked) >= kneed+2:
                    break
        # greedy largest with suppression already; return top kneed
        picked.sort(key=lambda t: t[2], reverse=True)
        # ensure kneed
        while len(picked) < kneed:
            picked.append(cands[0])
        return picked[:max(kneed,1)]

    def short_relax(x0, y0, steps=3, sigma=0.003):
        # random local centre relaxation scored by LP
        r0, s0 = lp_opt(x0, y0)
        if r0 is None:
            return x0, y0, None, -1.0
        bx, by, br, bs = x0.copy(), y0.copy(), r0, s0
        cx, cy = bx.copy(), by.copy()
        for _ in range(steps):
            tx = np.clip(cx + sigma*rng.randn(N), 0.01, 0.99)
            ty = np.clip(cy + sigma*rng.randn(N), 0.01, 0.99)
            rr, ss = lp_opt(tx, ty)
            if rr is not None and ss > bs + 1e-12:
                bx, by, br, bs = tx.copy(), ty.copy(), rr, ss
                cx, cy = tx, ty
        return bx, by, br, bs

    cur_x = best_x.copy(); cur_y = best_y.copy(); cur_r = best_r.copy(); cur_sum = best_sum
    # elite archive: list of (sig, x, y, r, s)
    archive = []
    def signature(xc, yc, rc):
        try:
            cc = contact_counts(xc, yc, rc)
            return tuple(sorted(cc.tolist())) + tuple(sorted([int(round(v*2000)) for v in rc]))
        except Exception:
            return tuple(sorted([int(round(v*2000)) for v in rc]))

    archive.append((signature(best_x,best_y,best_r), best_x.copy(), best_y.copy(), best_r.copy(), best_sum))

    T0 = 0.004; Tend = 0.00008
    it = 0; last_imp = 0
    sigmas = [0.015, 0.007, 0.003]
    while (time.time() - t0) < BUDGET - 4.0:
        frac = it / 2200.0
        if frac > 1.0: frac = 1.0
        T = T0 * ((Tend/T0) ** frac)
        if it - last_imp > 300:
            # restore to elite (best or random archive)
            if len(archive) > 1 and rng.rand() < 0.4:
                idx = rng.randint(len(archive))
                _, ax, ay, ar, asum = archive[idx]
            else:
                ax, ay, ar, asum = best_x, best_y, best_r, best_sum
            cur_x = np.clip(ax + 0.002*rng.randn(N), 0.01, 0.99)
            cur_y = np.clip(ay + 0.002*rng.randn(N), 0.01, 0.99)
            rr, ss = lp_opt(cur_x, cur_y)
            if rr is not None:
                cur_r, cur_sum = rr, ss
                if ss > best_sum + 1e-12:
                    best_sum = ss; best_x=cur_x.copy(); best_y=cur_y.copy(); best_r=rr.copy()
                    archive.append((signature(best_x,best_y,best_r), best_x.copy(), best_y.copy(), best_r.copy(), best_sum))
                    if len(archive) > 12: archive.pop(1)
                    last_imp = it
            else:
                cur_x, cur_y = ax.copy(), ay.copy(); cur_r, cur_sum = ar, asum
                last_imp = it
            it += 1
            continue
        u = rng.rand()
        xn = None; yn = None
        # choose victims
        if u < 0.48:
            # single void surgery
            if rng.rand() < 0.65:
                order = np.argsort(cur_r)
                pool = order[:6]
                idx = int(pool[rng.randint(len(pool))])
            else:
                try:
                    cc = contact_counts(cur_x, cur_y, cur_r)
                    # lowest pressure
                    minc = int(np.min(cc))
                    cand = np.where(cc <= minc+1)[0]
                    # sort by radius
                    cand = cand[np.argsort(cur_r[cand])]
                    idx = int(cand[rng.randint(min(len(cand),4))])
                except Exception:
                    order = np.argsort(cur_r)
                    idx = int(order[rng.randint(6)])
            mask = np.ones(N, dtype=bool); mask[idx]=False
            xr = cur_x[mask]; yr = cur_y[mask]; rr_ = cur_r[mask]
            voids = delaunay_voids(xr, yr, rr_, 3)
            # pick among top 2 randomly for diversity
            vi = 0
            if len(voids)>1 and rng.rand()<0.3:
                vi = 1
            vx, vy, _vc = voids[vi]
            xn = cur_x.copy(); yn = cur_y.copy()
            xn[idx] = float(np.clip(vx + (rng.rand()-0.5)*0.006, 0.01, 0.99))
            yn[idx] = float(np.clip(vy + (rng.rand()-0.5)*0.006, 0.01, 0.99))
            # short relax sometimes
            if rng.rand() < 0.5:
                # tiny dual-gradient-like jitter hill climb on neighbours only? do global small
                bx, by, br, bs = short_relax(xn, yn, steps=2, sigma=0.0025)
                if br is not None:
                    # evaluate SA on relaxed
                    xn, yn = bx, by
                    rnew, snew = br, bs
                else:
                    rnew, snew = lp_opt(xn, yn)
            else:
                rnew, snew = lp_opt(xn, yn)
        elif u < 0.74:
            kk = 2 if rng.rand() < 0.6 else 3
            order = np.argsort(cur_r)
            if rng.rand() < 0.7:
                idxs = order[:kk].copy()
            else:
                pool = order[:8]
                idxs = rng.choice(pool, kk, replace=False)
            mask = np.ones(N, dtype=bool); mask[idxs]=False
            xr = cur_x[mask]; yr = cur_y[mask]; rr_ = cur_r[mask]
            voids = delaunay_voids(xr, yr, rr_, kk+2)
            xn = cur_x.copy(); yn = cur_y.copy()
            # assign largest voids to smallest circles? sort idxs by radius ascending, voids descending
            idxs_sorted = idxs[np.argsort(cur_r[idxs])]
            for t_, idx in enumerate(idxs_sorted):
                vx, vy, _vc = voids[t_ % len(voids)]
                xn[int(idx)] = float(np.clip(vx + (rng.rand()-0.5)*0.007, 0.01, 0.99))
                yn[int(idx)] = float(np.clip(vy + (rng.rand()-0.5)*0.007, 0.01, 0.99))
            if rng.rand() < 0.4:
                bx, by, br, bs = short_relax(xn, yn, steps=2, sigma=0.003)
                if br is not None:
                    xn, yn = bx, by
                    rnew, snew = br, bs
                else:
                    rnew, snew = lp_opt(xn, yn)
            else:
                rnew, snew = lp_opt(xn, yn)
        elif u < 0.86:
            # swap filler and large
            order = np.argsort(cur_r)
            i_small = int(order[rng.randint(5)])
            i_large = int(order[N-1-rng.randint(5)])
            if i_small == i_large:
                it += 1
                continue
            xn = cur_x.copy(); yn = cur_y.copy()
            tx, ty = float(xn[i_small]), float(yn[i_small])
            xn[i_small], yn[i_small] = float(xn[i_large]), float(yn[i_large])
            xn[i_large], yn[i_large] = tx, ty
            xn += np.clip(0.0015*rng.randn(N), -0.005, 0.005)
            yn += np.clip(0.0015*rng.randn(N), -0.005, 0.005)
            xn = np.clip(xn, 0.01, 0.99); yn = np.clip(yn, 0.01, 0.99)
            rnew, snew = lp_opt(xn, yn)
        else:
            sigma = sigmas[rng.randint(len(sigmas))]
            xn = np.clip(cur_x + sigma*rng.randn(N), 0.01, 0.99)
            yn = np.clip(cur_y + sigma*rng.randn(N), 0.01, 0.99)
            rnew, snew = lp_opt(xn, yn)
        it += 1
        if rnew is None:
            continue
        dS = snew - cur_sum
        if dS >= 0 or rng.rand() < np.exp(dS / max(T, 1e-9)):
            cur_x, cur_y, cur_r, cur_sum = xn.copy(), yn.copy(), rnew, snew
            if snew > best_sum + 1e-12:
                best_sum = snew; best_x = xn.copy(); best_y = yn.copy(); best_r = rnew.copy()
                sig = signature(best_x, best_y, best_r)
                # distinct archive
                exists = False
                for (s_,_,_,_,_) in archive:
                    if s_ == sig:
                        exists=True; break
                if not exists:
                    archive.append((sig, best_x.copy(), best_y.copy(), best_r.copy(), best_sum))
                    if len(archive) > 12:
                        archive.pop(0)
                last_imp = it
        # time check
        if it % 50 == 0 and (time.time()-t0) >= BUDGET-4.0:
            break

    # final safety shrink + return
    try:
        X = best_x.copy(); Y = best_y.copy(); R = best_r.copy()* (1.0 - 1e-7)
        # verify and rescale if needed
        wall = np.minimum(np.minimum(X,1.0-X), np.minimum(Y,1.0-Y))
        Dx = X[:,None]-X[None,:]; Dy=Y[:,None]-Y[None,:]
        D = np.sqrt(Dx*Dx+Dy*Dy)
        np.fill_diagonal(D, np.inf)
        S = R[:,None]+R[None,:]
        # ratios
        wr = np.min(wall / np.maximum(R,1e-12))
        pr = np.min(D / np.maximum(S,1e-12))
        alpha = min(1.0, wr, pr)
        alpha = alpha*(1.0-1e-9)-1e-12
        if not np.isfinite(alpha) or alpha<=0:
            alpha=0.99
        if alpha < 1.0:
            R = R*alpha
        # ensure positivity
        R = np.maximum(R, 1e-6)
        out = np.column_stack([X,Y,R])
        return out
    except Exception:
        # fallback
        X = best_x; Y = best_y; R = best_r*(1.0-1e-7)
        return np.column_stack([X,Y,R])
