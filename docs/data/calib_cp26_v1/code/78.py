import numpy as np

def solve():
    import time as _tm
    import numpy as _np
    from scipy.optimize import minimize, linprog
    n = 26
    t0 = _tm.time()
    HARD = 123.0

    iu, ju = _np.triu_indices(n, 1)
    m = len(iu)
    A0 = _np.zeros((m, n), dtype=float)
    _ar = _np.arange(m)
    A0[_ar, iu] = 1.0
    A0[_ar, ju] = 1.0
    c_obj = -_np.ones(n, dtype=float)

    rng_global = _np.random.RandomState(7)

    # fallback
    def fallback():
        k = int(_np.ceil(_np.sqrt(n)))
        rr = 0.5 / k
        pts = []
        for i in range(k):
            for j in range(k):
                if len(pts) < n:
                    pts.append([(2*i+1)*rr, (2*j+1)*rr, rr*0.999])
        return _np.array(pts, dtype=float)

    def eval_lp(xc, yc, best_ref=None):
        b = _np.minimum(_np.minimum(xc, 1.0 - xc), _np.minimum(yc, 1.0 - yc))
        b = _np.maximum(b, 0.0)
        if best_ref is not None:
            if _np.sum(b) <= best_ref + 1e-12:
                return None, -1.0
        dx = xc[iu] - xc[ju]
        dy = yc[iu] - yc[ju]
        dvec = _np.sqrt(dx*dx + dy*dy)
        # dvec >=0 feasible (0 forces zero radii)
        bounds = [(0.0, float(bi)) for bi in b]
        try:
            res = linprog(c_obj, A_ub=A0, b_ub=dvec, bounds=bounds, method='highs')
        except Exception:
            return None, -1.0
        if res.status != 0 or res.x is None:
            return None, -1.0
        r = _np.asarray(res.x, dtype=float).flatten()
        r = _np.maximum(r, 0.0)
        return r, float(_np.sum(r))

    def repair(xc, yc, rc):
        eps = 2e-7
        b = _np.minimum(_np.minimum(xc, 1.0 - xc), _np.minimum(yc, 1.0 - yc))
        t = 1.0
        for i in range(n):
            if rc[i] > 1e-12:
                ti = (b[i]-eps)/rc[i]
                if ti < t:
                    t = ti
        if t <= 0 or not _np.isfinite(t):
            return _np.full(n, 1e-4)
        dx = xc[iu]-xc[ju]; dy = yc[iu]-yc[ju]
        dvec = _np.sqrt(dx*dx+dy*dy)
        rs = rc[iu]+rc[ju]
        mask = rs > 1e-12
        if _np.any(mask):
            tp = _np.min((dvec[mask]-eps)/rs[mask])
            if tp < t:
                t = tp
        if t > 1.0:
            t = 1.0
        if t <= 0 or not _np.isfinite(t):
            return _np.full(n, 1e-4)
        t = t*(1.0-1e-7)
        return rc*t

    bnds = [(0.0,1.0)]*(2*n)+[(1e-4,0.35)]*n

    def run_penalty(x0, y0, r0, P_seq, maxiter_each):
        z = _np.concatenate([x0, y0, r0])
        for P in P_seq:
            if _tm.time()-t0 > 62.0:
                break
            def fg(zz, P=P):
                x = zz[0:n]; y = zz[n:2*n]; r = zz[2*n:3*n]
                f = -_np.sum(r)
                gx = _np.zeros(n); gy = _np.zeros(n); gr = -_np.ones(n)
                v1 = r-x
                mm = v1>0
                if _np.any(mm):
                    f += P*_np.sum(v1[mm]*v1[mm])
                    co = 2.0*P*v1[mm]
                    gx[mm]-=co; gr[mm]+=co
                v2 = x+r-1.0
                mm = v2>0
                if _np.any(mm):
                    f += P*_np.sum(v2[mm]*v2[mm])
                    co = 2.0*P*v2[mm]
                    gx[mm]+=co; gr[mm]+=co
                v3 = r-y
                mm = v3>0
                if _np.any(mm):
                    f += P*_np.sum(v3[mm]*v3[mm])
                    co = 2.0*P*v3[mm]
                    gy[mm]-=co; gr[mm]+=co
                v4 = y+r-1.0
                mm = v4>0
                if _np.any(mm):
                    f += P*_np.sum(v4[mm]*v4[mm])
                    co = 2.0*P*v4[mm]
                    gy[mm]+=co; gr[mm]+=co
                dxm = x[:,None]-x[None,:]
                dym = y[:,None]-y[None,:]
                D = _np.sqrt(dxm*dxm+dym*dym)
                Rsum = r[:,None]+r[None,:]
                S = Rsum-D
                _np.fill_diagonal(S,-1.0)
                mask = S>0
                if _np.any(mask):
                    f += P*0.5*_np.sum(S[mask]*S[mask])
                    W = _np.zeros((n,n))
                    W[mask]=(2.0*P*S[mask])/(D[mask]+1e-12)
                    rs = W.sum(axis=1)
                    gx += W.dot(x)-rs*x
                    gy += W.dot(y)-rs*y
                    Spos = _np.where(mask,S,0.0)
                    gr += 2.0*P*Spos.sum(axis=1)
                g = _np.concatenate([gx,gy,gr])
                return f,g
            try:
                res = minimize(fg,z,method='L-BFGS-B',jac=True,bounds=bnds,
                               options={'maxiter':maxiter_each,'ftol':1e-10,'gtol':1e-6})
                z = _np.asarray(res.x,dtype=float)
            except Exception:
                break
        return z

    best_sum = 2.08
    # init fallback 6x5
    fx=[]; fy=[]
    for ix in range(6):
        for jy in range(5):
            fx.append((ix+0.5)/6.0); fy.append((jy+0.5)/5.0)
    best_x = _np.array(fx[:n]); best_y = _np.array(fy[:n])
    br,_ = eval_lp(best_x,best_y,None)
    if br is not None:
        best_r = br; best_sum = float(_np.sum(br))
    else:
        best_r = _np.full(n,0.08)

    P_short = [100.0, 1000.0, 10000.0]
    P_full = [30.0, 300.0, 3000.0, 30000.0]

    restart = 0
    # Phase 1: diverse jams
    while True:
        if _tm.time()-t0 > 62.0:
            break
        if restart > 90:
            break
        rgg = _np.random.RandomState(1000+restart*7919)
        mode = restart % 7
        if restart == 0:
            bx=[]; by=[]
            for ix in range(5):
                for jy in range(5):
                    bx.append((ix+0.5)/5.0); by.append((jy+0.5)/5.0)
            bx.append(0.5); by.append(0.5)
            bx=_np.array(bx); by=_np.array(by)
            x0 = bx + rgg.normal(0,0.012,n); y0 = by + rgg.normal(0,0.012,n)
            r0 = _np.full(n,0.09)+rgg.normal(0,0.008,n)
        elif mode == 1:
            xs = (rgg.permutation(30)[:n] if False else None)
            gx=[]; gy=[]
            for ix in range(6):
                for jy in range(5):
                    gx.append((ix+0.5)/6.0); gy.append((jy+0.5)/5.0)
            gx=_np.array(gx); gy=_np.array(gy)
            perm = rgg.permutation(30)[:n]
            x0 = gx[perm]+rgg.normal(0,0.01,n); y0 = gy[perm]+rgg.normal(0,0.01,n)
            r0 = rgg.uniform(0.05,0.10,size=n)
        elif mode == 2:
            # hex
            x0=[]; y0=[]
            row=0
            yy=0.06
            while len(x0)<n and yy<0.97:
                off = 0.055 if (row%2==1) else 0.0
                xx=0.06+off
                while len(x0)<n and xx<0.97:
                    x0.append(xx); y0.append(yy)
                    xx+=0.11+rgg.normal(0,0.005)
                yy+=0.095+rgg.normal(0,0.005); row+=1
            x0=_np.array(x0[:n])+rgg.normal(0,0.008,n); y0=_np.array(y0[:n])+rgg.normal(0,0.008,n)
            r0=rgg.uniform(0.05,0.10,size=n)
        elif mode == 3:
            x0=rgg.uniform(0.05,0.95,size=n); y0=rgg.uniform(0.05,0.95,size=n)
            r0=rgg.uniform(0.05,0.10,size=n)
        elif mode == 4:
            # rows with random counts (deformed grid) to break 5x5 basin
            counts=[5,5,6,5,5]
            rgg.shuffle(counts)
            # adjust to sum 26
            while sum(counts)!=26:
                if sum(counts)<26: counts[rgg.randint(5)]+=1
                else:
                    ii=rgg.randint(5)
                    if counts[ii]>4: counts[ii]-=1
            xs=[]; ys=[]
            for ri,cn in enumerate(counts):
                yy=(ri+0.5)/5.0 + rgg.normal(0,0.01)
                for k in range(cn):
                    xx=(k+0.5)/cn + rgg.normal(0,0.01)
                    xs.append(xx); ys.append(yy)
            xs=_np.array(xs[:n]); ys=_np.array(ys[:n])
            # if short pad random
            while len(xs)<n:
                xs=_np.append(xs,rgg.uniform(0.1,0.9)); ys=_np.append(ys,rgg.uniform(0.1,0.9))
            x0=xs; y0=ys; r0=rgg.uniform(0.05,0.10,size=n)
        elif mode == 5:
            # jitter of current best (exploit)
            x0=best_x+rgg.normal(0,0.03,n); y0=best_y+rgg.normal(0,0.03,n)
            r0=_np.clip(best_r+rgg.normal(0,0.01,n),0.02,0.18)
        else:
            x0=rgg.uniform(0.08,0.92,size=n); y0=rgg.uniform(0.08,0.92,size=n)
            r0=rgg.uniform(0.03,0.11,size=n)
        x0=_np.clip(x0,0.02,0.98); y0=_np.clip(y0,0.02,0.98)
        r0=_np.clip(r0,0.015,0.24)
        use_full = (restart<6)
        z = run_penalty(x0,y0,r0, P_full if use_full else P_short, 200 if use_full else 90)
        xc=_np.clip(z[0:n],0.0,1.0); yc=_np.clip(z[n:2*n],0.0,1.0)
        r_opt,s = eval_lp(xc,yc,best_ref=best_sum)
        if r_opt is not None and s>best_sum+1e-12:
            best_sum=float(s); best_x=xc.copy(); best_y=yc.copy(); best_r=r_opt.copy()
        restart+=1

    # Phase 2: LP hill climbing + void teleport ruin-and-recreate
    rng_hc = _np.random.RandomState(9999)
    # precompute grid candidates for void search
    gx_lin = _np.linspace(0.02,0.98,28)
    gy_lin = _np.linspace(0.02,0.98,28)
    XX,YY = _np.meshgrid(gx_lin,gy_lin)
    grid_pts = _np.column_stack([XX.ravel(),YY.ravel()])

    def largest_void(xc,yc,r):
        # vectorized over grid + random pts
        cand = grid_pts
        # clearance
        # compute min over circles: dist - r_j
        # chunk to save memory
        best_c = 0.0; best_p = None
        # chunk 196 pts at a time
        for s in range(0,len(cand),196):
            blk = cand[s:s+196]
            dx = blk[:,0,None]-xc[None,:]
            dy = blk[:,1,None]-yc[None,:]
            dd = _np.sqrt(dx*dx+dy*dy)-r[None,:]
            wall = _np.minimum(_np.minimum(blk[:,0],1.0-blk[:,0]),_np.minimum(blk[:,1],1.0-blk[:,1]))
            clr = _np.minimum(wall, _np.min(dd,axis=1))
            ii=_np.argmax(clr)
            if clr[ii]>best_c:
                best_c=float(clr[ii]); best_p=(float(blk[ii,0]),float(blk[ii,1]))
        # plus random
        rnd = rng_hc.uniform(0.02,0.98,size=(600,2))
        dx = rnd[:,0,None]-xc[None,:]
        dy = rnd[:,1,None]-yc[None,:]
        dd = _np.sqrt(dx*dx+dy*dy)-r[None,:]
        wall = _np.minimum(_np.minimum(rnd[:,0],1.0-rnd[:,0]),_np.minimum(rnd[:,1],1.0-rnd[:,1]))
        clr = _np.minimum(wall,_np.min(dd,axis=1))
        ii=_np.argmax(clr)
        if clr[ii]>best_c:
            best_c=float(clr[ii]); best_p=(float(rnd[ii,0]),float(rnd[ii,1]))
        return best_p, best_c

    it = 0
    # order indices by radius for ruin
    while _tm.time()-t0 < HARD-6.0 and it < 25000:
        # periodic time check already
        do_void = (it % 60 == 0)
        if do_void:
            # ruin: move 1 smallest circle to largest void
            order = _np.argsort(best_r)
            vp, vc = largest_void(best_x,best_y,best_r)
            if vp is None:
                it+=1; continue
            # pick smallest among random choice of 3 smallest to diversify
            cand_idx = order[0] if (rng_hc.rand()<0.7) else order[int(rng_hc.randint(0,min(4,n)))]
            nx=best_x.copy(); ny=best_y.copy()
            nx[cand_idx]=vp[0]+rng_hc.normal(0,0.003); ny[cand_idx]=vp[1]+rng_hc.normal(0,0.003)
            nx=_np.clip(nx,0.01,0.99); ny=_np.clip(ny,0.01,0.99)
            r_opt,s = eval_lp(nx,ny,best_ref=best_sum)
            if r_opt is not None and s>best_sum+1e-12:
                best_sum=float(s); best_x=nx; best_y=ny; best_r=r_opt.copy()
            else:
                # try second smallest to second void with jitter
                if rng_hc.rand()<0.3:
                    nx2=best_x.copy(); ny2=best_y.copy()
                    j2=order[int(rng_hc.randint(0,5))]
                    nx2[j2]+=rng_hc.normal(0,0.02); ny2[j2]+=rng_hc.normal(0,0.02)
                    nx2=_np.clip(nx2,0.01,0.99); ny2=_np.clip(ny2,0.01,0.99)
                    r2,s2=eval_lp(nx2,ny2,best_ref=best_sum)
                    if r2 is not None and s2>best_sum+1e-12:
                        best_sum=float(s2); best_x=nx2; best_y=ny2; best_r=r2.copy()
        else:
            # single-circle gaussian move with annealed sigma
            cyc = (it % 800)/800.0
            sigma = 0.025*_np.exp(-it/4000.0)*(0.5+cyc)+0.0012
            if rng_hc.rand()<0.12:
                # global jitter
                nx=best_x+rng_hc.normal(0,0.004,n); ny=best_y+rng_hc.normal(0,0.004,n)
                nx=_np.clip(nx,0.01,0.99); ny=_np.clip(ny,0.01,0.99)
            elif rng_hc.rand()<0.10:
                # swap two centres (topology change for unequal circles)
                a=int(rng_hc.randint(n)); b=int(rng_hc.randint(n))
                if a==b:
                    it+=1; continue
                nx=best_x.copy(); ny=best_y.copy()
                nx[a],nx[b]=nx[b],nx[a]; ny[a],ny[b]=ny[b],ny[a]
                nx[a]+=rng_hc.normal(0,0.002); ny[a]+=rng_hc.normal(0,0.002)
            else:
                idx=int(rng_hc.randint(n))
                nx=best_x.copy(); ny=best_y.copy()
                nx[idx]+=float(rng_hc.randn())*sigma
                ny[idx]+=float(rng_hc.randn())*sigma
                if nx[idx]<0.01: nx[idx]=0.01
                if nx[idx]>0.99: nx[idx]=0.99
                if ny[idx]<0.01: ny[idx]=0.01
                if ny[idx]>0.99: ny[idx]=0.99
            r_opt,s = eval_lp(nx,ny,best_ref=best_sum)
            if r_opt is not None and s>best_sum+1e-12:
                best_sum=float(s); best_x=nx; best_y=ny; best_r=r_opt.copy()
        it+=1

    # Phase 3: fine polish with tiny moves
    it2=0
    while _tm.time()-t0 < HARD-2.0 and it2 < 4000:
        idx=int(rng_hc.randint(n))
        sigma=0.0015
        nx=best_x.copy(); ny=best_y.copy()
        nx[idx]+=float(rng_hc.randn())*sigma
        ny[idx]+=float(rng_hc.randn())*sigma
        nx=_np.clip(nx,0.01,0.99); ny=_np.clip(ny,0.01,0.99)
        r_opt,s = eval_lp(nx,ny,best_ref=best_sum)
        if r_opt is not None and s>best_sum+1e-12:
            best_sum=float(s); best_x=nx; best_y=ny; best_r=r_opt.copy()
        it2+=1

    # final exact + repair
    try:
        r_fin,s_fin = eval_lp(best_x,best_y,None)
        if r_fin is not None:
            best_r=r_fin
    except Exception:
        pass
    r_safe = repair(best_x,best_y,_np.maximum(best_r,0.0))
    r_safe=_np.maximum(r_safe,1e-6)
    for i in range(n):
        if best_x[i] < r_safe[i]+1e-9:
            best_x[i]=r_safe[i]+1e-9
        if best_x[i] > 1.0-r_safe[i]-1e-9:
            best_x[i]=1.0-r_safe[i]-1e-9
        if best_y[i] < r_safe[i]+1e-9:
            best_y[i]=r_safe[i]+1e-9
        if best_y[i] > 1.0-r_safe[i]-1e-9:
            best_y[i]=1.0-r_safe[i]-1e-9
    r_safe = repair(best_x,best_y,r_safe)
    r_safe=_np.maximum(r_safe,1e-9)
    out=_np.column_stack([best_x,best_y,r_safe])
    return out
