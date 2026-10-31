import numpy as np

def solve():
    import time as _tm
    import numpy as _np
    from scipy.optimize import minimize, linprog
    n = 26
    t0 = _tm.time()
    DEADLINE = 124.0
    TPLATE_END = 30.0
    HOMO_END = 48.0
    JAM_END = 88.0

    iu, ju = _np.triu_indices(n, 1)
    m = len(iu)
    A_ub0 = _np.zeros((m, n), dtype=float)
    _ar = _np.arange(m)
    A_ub0[_ar, iu] = 1.0
    A_ub0[_ar, ju] = 1.0
    c_obj = -_np.ones(n)
    MARGIN = 1e-9

    def eval_lp(xc, yc):
        b = _np.minimum(_np.minimum(xc, 1.0 - xc), _np.minimum(yc, 1.0 - yc)) - MARGIN
        if _np.any(b < 0.004):
            return None, -1.0
        dx = xc[iu] - xc[ju]
        dy = yc[iu] - yc[ju]
        dvec = _np.sqrt(dx*dx + dy*dy) - MARGIN
        if _np.any(dvec < 0.008):
            return None, -1.0
        bounds = [(0.0, float(bi)) for bi in b]
        try:
            res = linprog(c_obj, A_ub=A_ub0, b_ub=dvec, bounds=bounds, method='highs')
        except Exception:
            return None, -1.0
        if (not res.success) or res.x is None:
            return None, -1.0
        r = _np.asarray(res.x, dtype=float).flatten()
        r = _np.maximum(r, 0.0)
        if _np.any(r < 5e-4):
            return None, -1.0
        r = r * (1.0 - 1e-7)
        return r, float(_np.sum(r))

    def safe_scale(xc, yc, rc):
        b = _np.minimum(_np.minimum(xc, 1.0 - xc), _np.minimum(yc, 1.0 - yc))
        dx = xc[iu] - xc[ju]
        dy = yc[iu] - yc[ju]
        dvec = _np.sqrt(dx*dx + dy*dy)
        t = 1.0
        for i in range(n):
            if rc[i] > 1e-12:
                ti = b[i] / rc[i]
                if ti < t:
                    t = ti
        rs = rc[iu] + rc[ju]
        mask = rs > 1e-12
        if _np.any(mask):
            tp = _np.min(dvec[mask] / rs[mask])
            if tp < t:
                t = tp
        t = t * (1.0 - 1e-9) - 1e-12
        if (not _np.isfinite(t)) or t <= 0:
            t = 0.5
        if t > 1.0:
            t = 1.0 - 1e-9
        return rc * t

    bnds_std = [(0.0, 1.0)]*(2*n) + [(0.01, 0.32)]*n

    def run_penalty(x0, y0, r0, P_seq, maxiter_each, bnds=None, tlim=None):
        if bnds is None:
            bnds = bnds_std
        z = _np.concatenate([x0, y0, r0])
        for P in P_seq:
            if _tm.time() - t0 > (tlim if tlim is not None else DEADLINE - 3.0):
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
                dx = x[iu] - x[ju]
                dy = y[iu] - y[ju]
                D = _np.sqrt(dx*dx + dy*dy + 1e-20)
                S = r[iu] + r[ju] - D
                pos = S > 0
                if _np.any(pos):
                    Sp = S[pos]
                    f += P*0.5*_np.sum(Sp*Sp)
                    coeff = P*Sp / D[pos]
                    dxp = dx[pos]; dyp = dy[pos]
                    iup = iu[pos]; jup = ju[pos]
                    _np.add.at(gx, iup, -coeff*dxp)
                    _np.add.at(gx, jup, coeff*dxp)
                    _np.add.at(gy, iup, -coeff*dyp)
                    _np.add.at(gy, jup, coeff*dyp)
                    _np.add.at(gr, iup, P*Sp)
                    _np.add.at(gr, jup, P*Sp)
                g = _np.concatenate([gx, gy, gr])
                return f, g
            try:
                res = minimize(fg, z, method='L-BFGS-B', jac=True, bounds=bnds,
                               options={'maxiter': maxiter_each, 'ftol': 1e-10, 'gtol': 1e-6})
                z = res.x
            except Exception:
                break
        return z

    best_sum = -1.0
    best_x = None; best_y = None; best_r = None
    def feed(xc, yc):
        global best_sum, best_x, best_y, best_r
        r_opt, s = eval_lp(xc, yc)
        if r_opt is not None and s > best_sum:
            best_sum = s
            best_x = xc.copy(); best_y = yc.copy(); best_r = r_opt.copy()
            return True
        return False

    # ---------- helpers for templates ----------
    def centres_from(counts, rows_y, offs, tilts, orient):
        xs = []; ys = []
        if orient == 'row':
            for ri, cnt in enumerate(counts):
                y = float(rows_y[ri])
                for k in range(cnt):
                    nom = (k + 0.5) / cnt
                    x = nom + float(offs[ri]) + float(tilts[ri])*(nom - 0.5)
                    xs.append(x); ys.append(y)
        else:
            for ci, cnt in enumerate(counts):
                x = float(rows_y[ci])
                for k in range(cnt):
                    nom = (k + 0.5) / cnt
                    y = nom + float(offs[ci]) + float(tilts[ci])*(nom - 0.5)
                    xs.append(y if False else nom + float(offs[ci]) + float(tilts[ci])*(nom-0.5))
                    # x fixed, y varying -> need correct order
                    ys.append(0)  # placeholder
            # rebuild correctly for col
            xs = []; ys = []
            for ci, cnt in enumerate(counts):
                x = float(rows_y[ci])
                for k in range(cnt):
                    nom = (k + 0.5) / cnt
                    y = nom + float(offs[ci]) + float(tilts[ci])*(nom - 0.5)
                    xs.append(x); ys.append(y)
        xc = _np.clip(_np.array(xs, dtype=float), 0.03, 0.97)
        yc = _np.clip(_np.array(ys, dtype=float), 0.03, 0.97)
        return xc, yc

    bases = [(6,5,6,5,4),(6,6,5,5,4),(5,6,5,5,5),(4,6,5,6,5),(6,5,4,6,5)]
    tmpls = []
    for bb in bases:
        tmpls.append(('row', tuple(bb)))
        tmpls.append(('row', tuple(reversed(bb))))
        tmpls.append(('col', tuple(bb)))
        tmpls.append(('col', tuple(reversed(bb))))

    # ---------- Phase A: low-dim template search (<=35s) ----------
    tpl_best = []  # list of (s, xc, yc)
    rngA = _np.random.RandomState(12345)
    for ti, (orient, counts) in enumerate(tmpls):
        if _tm.time() - t0 > TPLATE_END:
            break
        R = len(counts)
        rows0 = _np.array([(r+0.5)/R for r in range(R)], dtype=float)
        offs0 = _np.zeros(R); tilts0 = _np.zeros(R)
        xc0, yc0 = centres_from(counts, rows0, offs0, tilts0, orient)
        r0, s0 = eval_lp(xc0, yc0)
        loc_best_s = -1.0; loc_best_xy = None
        if r0 is not None:
            loc_best_s = s0; loc_best_xy = (xc0, yc0)
            feed(xc0, yc0)
        # perturbations of low-dim params
        npert = 12
        # fewer for later templates if time short
        for pp in range(npert):
            if _tm.time() - t0 > TPLATE_END:
                break
            ry = rows0 + 0.035*rngA.randn(R)
            ry = _np.clip(ry, 0.03, 0.97)
            ry = _np.sort(ry)
            # enforce min separation
            offs = 0.05*rngA.randn(R)
            offs = _np.clip(offs, -0.09, 0.09)
            tilts = 0.12*rngA.randn(R)
            tilts = _np.clip(tilts, -0.3, 0.3)
            # smooth left-to-right radius gradient analogue: small systematic x-shift
            # already via tilts; add global gradient shift
            xc, yc = centres_from(counts, ry, offs, tilts, orient)
            # tiny full-centre jitter to break symmetry
            xc = xc + 0.004*rngA.randn(n)
            yc = yc + 0.004*rngA.randn(n)
            xc = _np.clip(xc, 0.02, 0.98); yc = _np.clip(yc, 0.02, 0.98)
            r_opt, s = eval_lp(xc, yc)
            if r_opt is not None:
                feed(xc, yc)
                if s > loc_best_s:
                    loc_best_s = s; loc_best_xy = (xc.copy(), yc.copy())
        if loc_best_xy is not None:
            tpl_best.append((loc_best_s, loc_best_xy[0], loc_best_xy[1]))
    tpl_best.sort(key=lambda t: -t[0])
    top2 = tpl_best[:2] if len(tpl_best) >= 2 else tpl_best

    # ---------- Phase B: homotopy from 25-circle 5x5 ----------
    try:
        # good 25 base
        b25 = []
        for ix in range(5):
            for jx in range(5):
                b25.append([0.1+0.2*ix, 0.1+0.2*jx])
        b25 = _np.array(b25, dtype=float)
        rngH = _np.random.RandomState(999)
        b25 = b25 + 0.008*rngH.randn(25,2)
        b25 = _np.clip(b25, 0.06, 0.94)
        x25 = b25[:,0]; y25 = b25[:,1]
        # nominal radii for void search
        rnom = _np.full(25, 0.095)
        # largest void search
        cand = rngH.rand(3000,2)*0.92+0.04
        # add midpoints
        best_vx, best_vy, best_vd = 0.5, 0.5, -1e100
        for bb in range(0, 3000, 500):
            px = cand[bb:bb+500,0]; py = cand[bb:bb+500,1]
            dx = px[:,None]-x25[None,:]; dy = py[:,None]-y25[None,:]
            dd = _np.sqrt(dx*dx+dy*dy)-rnom[None,:]
            wall = _np.minimum(_np.minimum(px,1-px), _np.minimum(py,1-py))
            vd = _np.minimum(_np.min(dd,axis=1), wall)
            j = int(_np.argmax(vd))
            if vd[j] > best_vd:
                best_vd=float(vd[j]); best_vx=float(px[j]); best_vy=float(py[j])
        xh = _np.concatenate([x25, [best_vx]])
        yh = _np.concatenate([y25, [best_vy]])
        rh = _np.concatenate([_np.full(25,0.085), [0.012]])
        xh = _np.clip(xh,0.02,0.98); yh=_np.clip(yh,0.02,0.98)
        lbs = [0.02,0.04,0.06,0.08]
        for sidx, lb in enumerate(lbs):
            if _tm.time()-t0 > HOMO_END:
                break
            bnds_h = [(0.0,1.0)]*(2*n)+[(0.01,0.32)]*(n-1)+[(float(lb),0.32)]
            z = run_penalty(xh, yh, rh, P_seq=[200.0,2000.0], maxiter_each=70, bnds=bnds_h, tlim=HOMO_END)
            xh = _np.clip(z[0:n],0.02,0.98); yh=_np.clip(z[n:2*n],0.02,0.98)
            rh = _np.clip(z[2*n:3*n],1e-4,0.32)
            rh = safe_scale(xh, yh, rh)
            # enforce lower bound still
            if rh[-1] < lb*0.999:
                rh[-1] = lb
                rh = safe_scale(xh, yh, rh)
            # LP reoptimize others but keep inserted >= lb*0.95 via post-fix
            r_opt, s = eval_lp(xh, yh)
            if r_opt is not None and r_opt[-1] > 0.015:
                # accept LP if it keeps inserted reasonably large
                feed(xh, yh)
                # continue homotopy from LP radii
                rh = r_opt
            else:
                # keep penalty radii, still record via safe scale sum
                ss = float(_np.sum(rh))
                if ss > best_sum and rh[-1] >= lb*0.9:
                    best_sum = ss; best_x=xh.copy(); best_y=yh.copy(); best_r=rh.copy()
        homo_xy = (xh.copy(), yh.copy())
    except Exception:
        homo_xy = None

    # ---------- Phase C: release best two templates + homotopy to full jam ----------
    jam_cands = []
    for _, xc, yc in top2:
        jam_cands.append((xc, yc))
    if homo_xy is not None:
        jam_cands.append(homo_xy)
    # also add current best
    if best_x is not None:
        jam_cands.append((best_x.copy(), best_y.copy()))
    # deduplicate to at most 4
    jam_cands = jam_cands[:4]
    P_full = [20.0, 200.0, 2000.0, 20000.0]
    for (xc0, yc0) in jam_cands:
        if _tm.time()-t0 > JAM_END:
            break
        r0lp, _ = eval_lp(xc0, yc0)
        if r0lp is None:
            r_init = _np.full(n, 0.08)
        else:
            r_init = _np.clip(r0lp, 0.01, 0.30)
        z = run_penalty(_np.clip(xc0,0.02,0.98), _np.clip(yc0,0.02,0.98), r_init, P_seq=P_full, maxiter_each=110, tlim=JAM_END)
        xp = _np.clip(z[0:n],0.02,0.98); yp=_np.clip(z[n:2*n],0.02,0.98)
        feed(xp, yp)
        # second short rejamm with tighter
        if _tm.time()-t0 > JAM_END:
            break
        # small jitter rejamm
        rngJ = _np.random.RandomState(7)
        xp2 = xp + 0.003*rngJ.randn(n); yp2 = yp + 0.003*rngJ.randn(n)
        xp2=_np.clip(xp2,0.02,0.98); yp2=_np.clip(yp2,0.02,0.98)
        feed(xp2, yp2)

    # ---------- Phase D: polish hill-climb + void teleport ----------
    if best_x is None:
        k = int(_np.ceil(_np.sqrt(n))); rr = 0.5/k
        pts=[]
        for i in range(k):
            for j in range(k):
                if len(pts)<n:
                    pts.append([(2*i+1)*rr,(2*j+1)*rr,rr*0.999])
        return _np.array(pts,dtype=float)
    rng2 = _np.random.RandomState(777)
    for sigma in [0.015, 0.008, 0.004, 0.002]:
        for _ in range(70):
            if _tm.time()-t0 > DEADLINE-4.0:
                break
            xn = best_x + sigma*rng2.randn(n)
            yn = best_y + sigma*rng2.randn(n)
            xn=_np.clip(xn,0.02,0.98); yn=_np.clip(yn,0.02,0.98)
            r_opt, s = eval_lp(xn, yn)
            if r_opt is not None and s > best_sum+1e-10:
                best_sum=s; best_x=xn; best_y=yn; best_r=r_opt
    # void teleport of small circles
    it=0
    while _tm.time()-t0 < DEADLINE-3.0 and it < 250:
        it+=1
        # find void
        nc=400
        cx = rng2.rand(nc); cy = rng2.rand(nc)
        best_vx=0.5; best_vy=0.5; best_vd=-1e100
        for b in range(0,nc,200):
            px=cx[b:b+200]; py=cy[b:b+200]
            dx=px[:,None]-best_x[None,:]; dy=py[:,None]-best_y[None,:]
            dd=_np.sqrt(dx*dx+dy*dy)-best_r[None,:]
            wall=_np.minimum(_np.minimum(px,1-px),_np.minimum(py,1-py))
            vd=_np.minimum(_np.min(dd,axis=1),wall)
            j=int(_np.argmax(vd))
            if vd[j]>best_vd:
                best_vd=float(vd[j]); best_vx=float(px[j]); best_vy=float(py[j])
        order=_np.argsort(best_r)
        victim=int(order[0] if rng2.rand()<0.75 else rng2.choice(n))
        xn=best_x.copy(); yn=best_y.copy()
        xn[victim]=_np.clip(best_vx+0.005*rng2.randn(),0.02,0.98)
        yn[victim]=_np.clip(best_vy+0.005*rng2.randn(),0.02,0.98)
        # occasional swap large-small
        if rng2.rand()<0.3:
            a=int(order[0]); b2=int(order[-1])
            xn[a],xn[b2]=xn[b2],xn[a]; yn[a],yn[b2]=yn[b2],yn[a]
        r_opt,s=eval_lp(xn,yn)
        if r_opt is not None and s>best_sum+1e-10:
            best_sum=s; best_x=xn; best_y=yn; best_r=r_opt
            continue
        # short rejamm occasionally if teleport failed
        if it%25==0 and _tm.time()-t0 < DEADLINE-10.0:
            r_init=_np.clip(best_r,0.01,0.30)
            z=run_penalty(best_x,best_y,r_init,P_seq=[200.0,2000.0],maxiter_each=60,tlim=DEADLINE-3.0)
            xp=_np.clip(z[0:n],0.02,0.98); yp=_np.clip(z[n:2*n],0.02,0.98)
            feed(xp,yp)

    # final safe scale
    best_r = safe_scale(best_x, best_y, best_r)
    # extra shrink
    best_r = best_r*(1.0-1e-9)
    out = _np.column_stack([best_x, best_y, best_r])
    return out
