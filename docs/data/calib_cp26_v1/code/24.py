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
        r = _np.asarray(res.x, dtype=float).flatten()
        r = _np.maximum(r, 0.0)
        s = float(_np.sum(r))
        if best_ref is not None and s <= best_ref:
            return None, -1.0
        return r, s

    def repair(xc, yc, rc):
        eps = 1e-7
        b = _np.minimum(_np.minimum(xc, 1.0-xc), _np.minimum(yc, 1.0-yc))
        t = 1.0
        for i in range(n):
            if rc[i] > 1e-12:
                ti = (b[i]-eps)/rc[i]
                if ti < t: t = ti
        dx = xc[iu]-xc[ju]; dy = yc[iu]-yc[ju]
        dvec = _np.sqrt(dx*dx+dy*dy)
        rs = rc[iu]+rc[ju]
        mask = rs > 1e-12
        if _np.any(mask):
            ratios = (dvec[mask]-eps)/rs[mask]
            tp = _np.min(ratios)
            if tp < t: t = tp
        if t <= 0 or not _np.isfinite(t):
            return _np.full(n, 1e-4)
        if t > 1.0: t = 1.0
        t = t*(1.0-1e-9)
        return rc*t

    bnds = [(0.0,1.0)]*(2*n)+[(0.01,0.28)]*n
    P_seq = [50.0, 1500.0, 25000.0]

    best_sum = -1.0
    best_x = None; best_y = None; best_r = None

    def try_update(xc, yc):
        nonlocal best_sum, best_x, best_y, best_r
        r,s = eval_lp(xc, yc, best_ref=best_sum if best_sum>0 else None)
        if r is not None and s > best_sum:
            best_sum = float(s)
            best_x = xc.copy(); best_y = yc.copy(); best_r = r.copy()
            return True
        return False

    restart_idx = 0
    # Phase A: joint penalty restarts
    while True:
        if _tm.time()-t0 > 68.0: break
        if restart_idx >= 300: break
        rng = _np.random.RandomState(2000+restart_idx*6133)
        if restart_idx == 0:
            pts=[]
            for i in range(5):
                for j in range(5):
                    pts.append([0.1+0.2*i,0.1+0.2*j])
            xy=_np.array(pts); extra=_np.array([[0.5,0.5]])
            xy=_np.vstack([xy,extra])
            x0=xy[:,0].copy(); y0=xy[:,1].copy()
            r0=_np.full(n,0.085)
            x0+=rng.randn(n)*0.002; y0+=rng.randn(n)*0.002
            x0=_np.clip(x0,0.05,0.95); y0=_np.clip(y0,0.05,0.95)
        elif restart_idx == 1:
            xs=(_np.arange(6)+0.5)/6.0; ys=(_np.arange(5)+0.5)/5.0
            xx,yy=_np.meshgrid(xs,ys)
            xy=_np.column_stack([xx.ravel(),yy.ravel()])
            perm=rng.permutation(len(xy))[:n]
            x0=xy[perm,0]+rng.randn(n)*0.004
            y0=xy[perm,1]+rng.randn(n)*0.004
            x0=_np.clip(x0,0.05,0.95); y0=_np.clip(y0,0.05,0.95)
            r0=rng.uniform(0.06,0.09,size=n)
        elif restart_idx % 4 == 2:
            # hex-like
            x0=rng.uniform(0.05,0.95,size=n); y0=rng.uniform(0.05,0.95,size=n)
            r0=rng.uniform(0.07,0.10,size=n)
        elif restart_idx % 4 == 3:
            # jitter of current best
            if best_x is not None and rng.rand()<0.6:
                x0=best_x+rng.randn(n)*0.02; y0=best_y+rng.randn(n)*0.02
                x0=_np.clip(x0,0.02,0.98); y0=_np.clip(y0,0.02,0.98)
                r0=_np.maximum(best_r,0.02)+rng.randn(n)*0.005
                r0=_np.clip(r0,0.02,0.2)
            else:
                x0=rng.uniform(0.03,0.97,size=n); y0=rng.uniform(0.03,0.97,size=n)
                r0=rng.uniform(0.05,0.10,size=n)
        else:
            x0=rng.uniform(0.03,0.97,size=n); y0=rng.uniform(0.03,0.97,size=n)
            r0=rng.uniform(0.05,0.10,size=n)
        z=_np.concatenate([x0,y0,r0])
        for P in P_seq:
            if _tm.time()-t0 > 68.0: break
            def fg(zz, P=P):
                x=zz[0:n]; y=zz[n:2*n]; r=zz[2*n:3*n]
                f=-_np.sum(r)
                gx=_np.zeros(n); gy=_np.zeros(n); gr=-_np.ones(n)
                v1=r-x; mm=v1>0
                if _np.any(mm):
                    f+=P*_np.sum(v1[mm]*v1[mm]); co=2.0*P*v1[mm]
                    gx[mm]-=co; gr[mm]+=co
                v2=x+r-1.0; mm=v2>0
                if _np.any(mm):
                    f+=P*_np.sum(v2[mm]*v2[mm]); co=2.0*P*v2[mm]
                    gx[mm]+=co; gr[mm]+=co
                v3=r-y; mm=v3>0
                if _np.any(mm):
                    f+=P*_np.sum(v3[mm]*v3[mm]); co=2.0*P*v3[mm]
                    gy[mm]-=co; gr[mm]+=co
                v4=y+r-1.0; mm=v4>0
                if _np.any(mm):
                    f+=P*_np.sum(v4[mm]*v4[mm]); co=2.0*P*v4[mm]
                    gy[mm]+=co; gr[mm]+=co
                dxm=x[:,None]-x[None,:]; dym=y[:,None]-y[None,:]
                D=_np.sqrt(dxm*dxm+dym*dym)
                Rsum=r[:,None]+r[None,:]
                S=Rsum-D; _np.fill_diagonal(S,-1.0)
                mask=S>0
                if _np.any(mask):
                    f+=P*0.5*_np.sum(S[mask]*S[mask])
                    W=_np.zeros((n,n))
                    W[mask]=(2.0*P*S[mask])/(D[mask]+1e-12)
                    rs=W.sum(axis=1)
                    gx+=W.dot(x)-rs*x
                    gy+=W.dot(y)-rs*y
                    Spos=_np.where(mask,S,0.0)
                    gr+=2.0*P*Spos.sum(axis=1)
                g=_np.concatenate([gx,gy,gr])
                return f,g
            try:
                res=minimize(fg,z,method='L-BFGS-B',jac=True,bounds=bnds,options={'maxiter':180,'ftol':1e-12,'gtol':1e-7})
                z=res.x
            except Exception:
                break
        try:
            xc=_np.clip(z[0:n],0.0,1.0); yc=_np.clip(z[n:2*n],0.0,1.0)
            try_update(xc,yc)
        except Exception:
            pass
        restart_idx+=1

    if best_x is None:
        k=int(_np.ceil(_np.sqrt(n))); rr=0.5/k; pts=[]
        for i in range(k):
            for j in range(k):
                if len(pts)<n: pts.append([(2*i+1)*rr,(2*j+1)*rr,rr*0.999])
        return _np.array(pts,dtype=float)

    # Phase B: LP hill climbing + void filling + pattern search
    rng_hc=_np.random.RandomState(7777)
    # precompute grid for void search
    gx_lin=_np.linspace(0.01,0.99,40); gy_lin=_np.linspace(0.01,0.99,40)
    GXX,GYY=_np.meshgrid(gx_lin,gy_lin)
    Gpts=_np.column_stack([GXX.ravel(),GYY.ravel()])
    ng=Gpts.shape[0]
    it=0
    # order circles by radius for void moves
    sigmas=[0.02,0.01,0.005,0.002,0.0008]
    while _tm.time()-t0 < DEADLINE-4.0 and it < 12000:
        # periodic void-filling move
        if it % 70 == 0 and it>0:
            # find largest void w.r.t current best
            try:
                # distances from grid to centres
                dxg=Gpts[:,0,None]-best_x[None,:]
                dyg=Gpts[:,1,None]-best_y[None,:]
                Dg=_np.sqrt(dxg*dxg+dyg*dyg)
                clearance=Dg-best_r[None,:]
                wall=_np.minimum(_np.minimum(Gpts[:,0],1.0-Gpts[:,0]),_np.minimum(Gpts[:,1],1.0-Gpts[:,1]))
                clear=_np.minimum(clearance.min(axis=1),wall)
                vi=int(_np.argmax(clear))
                # pick a small circle to move (among 6 smallest, random)
                order=_np.argsort(best_r)
                cand=order[rng_hc.randint(0,6)]
                nx=best_x.copy(); ny=best_y.copy()
                nx[cand]=float(Gpts[vi,0]); ny[cand]=float(Gpts[vi,1])
                # tiny jitter
                nx[cand]+=(rng_hc.rand()-0.5)*0.002; ny[cand]+=(rng_hc.rand()-0.5)*0.002
                nx=_np.clip(nx,0.01,0.99); ny=_np.clip(ny,0.01,0.99)
                try_update(nx,ny)
            except Exception:
                pass
            it+=1
            continue
        if it % 500 == 499:
            # global small jitter
            nx=_np.clip(best_x+rng_hc.randn(n)*0.003,0.01,0.99)
            ny=_np.clip(best_y+rng_hc.randn(n)*0.003,0.01,0.99)
            try_update(nx,ny)
            it+=1
            continue
        # single-circle gaussian move with cycling sigma
        sigma=sigmas[it % len(sigmas)]*(0.5+ rng_hc.rand()*0.5)
        # bias to move smaller / more constrained circles more often? uniform for determinism
        idx=int(rng_hc.randint(n))
        nx=best_x.copy(); ny=best_y.copy()
        nx[idx]+=float(rng_hc.randn())*sigma
        ny[idx]+=float(rng_hc.randn())*sigma
        if nx[idx]<0.01: nx[idx]=0.01
        if nx[idx]>0.99: nx[idx]=0.99
        if ny[idx]<0.01: ny[idx]=0.01
        if ny[idx]>0.99: ny[idx]=0.99
        try_update(nx,ny)
        # occasional compass refinement on same idx: try 4 directions at sigma
        if it % 10 == 0:
            for ang in [0,_np.pi/2,_np.pi,3*_np.pi/2]:
                if _tm.time()-t0 >= DEADLINE-4.0: break
                nx2=best_x.copy(); ny2=best_y.copy()
                nx2[idx]+=float(_np.cos(ang))*sigma*0.7
                ny2[idx]+=float(_np.sin(ang))*sigma*0.7
                nx2=_np.clip(nx2,0.01,0.99); ny2=_np.clip(ny2,0.01,0.99)
                try_update(nx2,ny2)
        it+=1

    # final
    r_opt,_=eval_lp(best_x,best_y,None)
    if r_opt is not None: best_r=r_opt
    r_safe=repair(best_x,best_y,_np.maximum(best_r,0.0))
    r_safe=_np.maximum(r_safe,1e-9)
    for i in range(n):
        if best_x[i]<r_safe[i]+1e-9: best_x[i]=r_safe[i]+1e-9
        if best_x[i]>1.0-r_safe[i]-1e-9: best_x[i]=1.0-r_safe[i]-1e-9
        if best_y[i]<r_safe[i]+1e-9: best_y[i]=r_safe[i]+1e-9
        if best_y[i]>1.0-r_safe[i]-1e-9: best_y[i]=1.0-r_safe[i]-1e-9
    r_safe=repair(best_x,best_y,r_safe)
    r_safe=_np.maximum(r_safe,1e-9)
    out=_np.column_stack([best_x,best_y,r_safe])
    return out
