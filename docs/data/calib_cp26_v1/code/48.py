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

    def eval_lp(xc, yc):
        b = _np.minimum(_np.minimum(xc, 1.0 - xc), _np.minimum(yc, 1.0 - yc))
        b = _np.maximum(b, 0.0)
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
        return r, float(_np.sum(r))

    def repair_radii(xc, yc, rc):
        eps = 1e-7
        b = _np.minimum(_np.minimum(xc, 1.0 - xc), _np.minimum(yc, 1.0 - yc))
        t = 1.0
        for i in range(n):
            if rc[i] > 0:
                ti = (b[i]-eps)/rc[i]
                if ti < t:
                    t = ti
        if t <= 0:
            return _np.full(n, 1e-4)
        dx = xc[iu]-xc[ju]; dy = yc[iu]-yc[ju]
        dvec = _np.sqrt(dx*dx+dy*dy)
        rs = rc[iu]+rc[ju]
        mask = rs > 1e-12
        if _np.any(mask):
            ratios = (dvec[mask]-eps)/rs[mask]
            tp = _np.min(ratios)
            if tp < t:
                t = tp
        if t > 1.0:
            t = 1.0
        if t <= 0:
            return _np.full(n, 1e-4)
        t = t*(1.0-1e-9)
        return rc*t

    def parent_fallback():
        k = int(_np.ceil(_np.sqrt(n)))
        rr = 0.5/k
        pts = []
        for i in range(k):
            for j in range(k):
                if len(pts) < n:
                    pts.append([(2*i+1)*rr,(2*j+1)*rr,rr*0.999])
        return _np.array(pts, dtype=float)

    bnds = [(0.0,1.0)]*(2*n)+[(1e-4,0.35)]*n
    P_seq = [20.0, 200.0, 2000.0, 20000.0]

    best_sum = -1.0
    best_x = None; best_y = None; best_r = None

    def run_penalty(x0, y0, r0, maxiter_each=120):
        z = _np.concatenate([x0, y0, r0])
        for P in P_seq:
            if _tm.time()-t0 > 66.0:
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
                    W[mask]=(2.0*P*S[mask])/ (D[mask]+1e-12)
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
                z = res.x
            except Exception:
                break
        return z

    # precompute hex grid for init diversity
    restart_idx = 0
    while True:
        if _tm.time()-t0 > 66.0:
            break
        if restart_idx > 500:
            break
        rng = _np.random.RandomState(1000+restart_idx*7919)
        mode = restart_idx % 6
        if restart_idx == 0:
            pts=[]
            for i in range(5):
                for j in range(5):
                    pts.append([0.1+0.2*i,0.1+0.2*j])
            pts=_np.array(pts)
            xy=_np.vstack([pts,_np.array([[0.2,0.2]])])
            x0=xy[:,0].copy(); y0=xy[:,1].copy()
            r0=_np.full(n,0.09); r0[-1]=0.04
            x0+=rng.randn(n)*0.002; y0+=rng.randn(n)*0.002
            x0=_np.clip(x0,0.05,0.95); y0=_np.clip(y0,0.05,0.95)
        elif mode == 1:
            xs=(_np.arange(6)+0.5)/6.0; ys=(_np.arange(5)+0.5)/5.0
            xx,yy=_np.meshgrid(xs,ys)
            xy=_np.column_stack([xx.ravel(),yy.ravel()])
            perm=rng.permutation(len(xy))[:n]
            x0=xy[perm,0]+rng.randn(n)*0.006
            y0=xy[perm,1]+rng.randn(n)*0.006
            x0=_np.clip(x0,0.05,0.95); y0=_np.clip(y0,0.05,0.95)
            r0=rng.uniform(0.06,0.095,size=n)
        elif mode == 2:
            # staggered hexagonal rows (5-6 rows)
            xy=[]
            nr=5
            for rr_ in range(nr):
                y_=(rr_+0.5)/nr
                cnt=5 if rr_%2==0 else 6
                # need total ~26: adjust
                for cc in range(6):
                    if len(xy)>=30: break
                    x_=(cc+0.5+(0.5 if rr_%2==1 else 0.0))/6.0
                    if x_>0.97: continue
                    xy.append([x_,y_])
            xy=_np.array(xy)
            perm=rng.permutation(len(xy))[:n]
            x0=xy[perm,0]+rng.randn(n)*0.008
            y0=xy[perm,1]+rng.randn(n)*0.008
            x0=_np.clip(x0,0.03,0.97); y0=_np.clip(y0,0.03,0.97)
            r0=rng.uniform(0.06,0.10,size=n)
        elif mode == 3:
            x0=rng.uniform(0.05,0.95,size=n)
            y0=rng.uniform(0.05,0.95,size=n)
            r0=rng.uniform(0.05,0.10,size=n)
        elif mode == 4:
            # column-biased: 5 columns with jitter
            x0=_np.repeat(((_np.arange(5)+0.5)/5.0),6)[:n]+rng.randn(n)*0.02
            y0=rng.uniform(0.05,0.95,size=n)
            x0=_np.clip(x0,0.03,0.97); y0=_np.clip(y0,0.03,0.97)
            r0=rng.uniform(0.06,0.10,size=n)
        else:
            # reuse best with big kick to explore nearby basin
            if best_x is not None and rng.rand()<0.5:
                x0=best_x+rng.randn(n)*0.03
                y0=best_y+rng.randn(n)*0.03
                x0=_np.clip(x0,0.03,0.97); y0=_np.clip(y0,0.03,0.97)
                r0=_np.maximum(best_r,0.02)+rng.randn(n)*0.01
                r0=_np.clip(r0,0.02,0.20)
            else:
                x0=rng.uniform(0.03,0.97,size=n)
                y0=rng.uniform(0.03,0.97,size=n)
                r0=rng.uniform(0.04,0.11,size=n)
        z = run_penalty(x0,y0,r0, maxiter_each=100)
        try:
            xc=_np.clip(z[0:n],0.0,1.0); yc=_np.clip(z[n:2*n],0.0,1.0)
            # prune by box sum
            bq=_np.minimum(_np.minimum(xc,1.0-xc),_np.minimum(yc,1.0-yc))
            if _np.sum(bq) <= best_sum:
                restart_idx+=1
                continue
            r_opt,s = eval_lp(xc,yc)
            if r_opt is not None and s>best_sum:
                best_sum=float(s); best_x=xc.copy(); best_y=yc.copy(); best_r=r_opt.copy()
        except Exception:
            pass
        restart_idx+=1

    if best_x is None:
        return parent_fallback()

    # ---- Phase B: ruin-and-recreate ----
    rng_hc = _np.random.RandomState(777)
    # coarse grid for void search
    gx_lin = _np.linspace(0.01,0.99,60)
    gy_lin = _np.linspace(0.01,0.99,60)
    GX, GY = _np.meshgrid(gx_lin, gy_lin)
    Gpts_x = GX.ravel(); Gpts_y = GY.ravel()
    nG = Gpts_x.size

    def try_lp(nx, ny):
        global_best = best_sum
        try:
            r_opt,s = eval_lp(nx,ny)
        except Exception:
            return False
        if r_opt is not None and s > best_sum + 1e-12:
            # update outer via mutable closure trick: use lists
            return (r_opt,s)
        return False

    it = 0
    # precompute for speed: reuse arrays
    while _tm.time()-t0 < DEADLINE-3.0 and it < 25000:
        it+=1
        rchoice = rng_hc.rand()
        if rchoice < 0.45:
            # single gaussian move, decaying sigma
            sigma = 0.02*_np.exp(-it/2500.0)+0.001
            idx = int(rng_hc.randint(n))
            nx=best_x.copy(); ny=best_y.copy()
            nx[idx]+=float(rng_hc.randn())*sigma
            ny[idx]+=float(rng_hc.randn())*sigma
            nx[idx]=_np.clip(nx[idx],0.01,0.99); ny[idx]=_np.clip(ny[idx],0.01,0.99)
            res = try_lp(nx,ny)
            if res is not False:
                r_opt,s=res; best_sum=float(s); best_x=nx; best_y=ny; best_r=r_opt.copy()
        elif rchoice < 0.70:
            # void teleport of one circle (prefer small circles)
            # pick index: 70% among 8 smallest, else random
            order = _np.argsort(best_r)
            if rng_hc.rand()<0.7:
                cand = order[:8]
                idx = int(cand[rng_hc.randint(len(cand))])
            else:
                idx = int(rng_hc.randint(n))
            # compute clearance of grid pts w.r.t. others
            dx = Gpts_x[:,None]-best_x[None,:] if False else None
            # vectorized: G x n distances
            # do in chunks to avoid big memory? 3600x26 fine
            DX = Gpts_x[:,None]-best_x[None,:]
            DY = Gpts_y[:,None]-best_y[None,:]
            D = _np.sqrt(DX*DX+DY*DY)
            # subtract radii of others, ignore idx
            Rmat = best_r[None,:].copy()
            # for others, clearance = D - r_other; for idx ignore (set inf)
            Clear = D - Rmat
            Clear[:,idx]=1e9
            wall = _np.minimum(_np.minimum(Gpts_x,1.0-Gpts_x),_np.minimum(Gpts_y,1.0-Gpts_y))
            best_clear = _np.minimum(wall, _np.min(Clear,axis=1))
            # pick among top-K randomly
            K=8
            top = _np.argpartition(-best_clear,K)[:K]
            # choose one with prob proportional to clearance
            pick = int(top[rng_hc.randint(K)])
            nx=best_x.copy(); ny=best_y.copy()
            nx[idx]=float(Gpts_x[pick])+float(rng_hc.randn())*0.002
            ny[idx]=float(Gpts_y[pick])+float(rng_hc.randn())*0.002
            nx[idx]=_np.clip(nx[idx],0.01,0.99); ny[idx]=_np.clip(ny[idx],0.01,0.99)
            res = try_lp(nx,ny)
            if res is not False:
                r_opt,s=res; best_sum=float(s); best_x=nx; best_y=ny; best_r=r_opt.copy()
            # else try second best pick once
            elif rng_hc.rand()<0.3:
                pick2=int(top[(rng_hc.randint(K)+1)%K])
                nx2=best_x.copy(); ny2=best_y.copy()
                nx2[idx]=float(Gpts_x[pick2]); ny2[idx]=float(Gpts_y[pick2])
                res2=try_lp(nx2,ny2)
                if res2 is not False:
                    r_opt,s=res2; best_sum=float(s); best_x=nx2; best_y=ny2; best_r=r_opt.copy()
        elif rchoice < 0.82:
            # swap two circles (role exchange)
            i1=int(rng_hc.randint(n)); i2=int(rng_hc.randint(n))
            if i1==i2: continue
            nx=best_x.copy(); ny=best_y.copy()
            nx[i1],nx[i2]=nx[i2],nx[i1]
            ny[i1],ny[i2]=ny[i2],ny[i1]
            nx+=rng_hc.randn(n)*0.001; ny+=rng_hc.randn(n)*0.001
            nx=_np.clip(nx,0.01,0.99); ny=_np.clip(ny,0.01,0.99)
            res=try_lp(nx,ny)
            if res is not False:
                r_opt,s=res; best_sum=float(s); best_x=nx; best_y=ny; best_r=r_opt.copy()
        elif rchoice < 0.92:
            # full small jitter
            nx=_np.clip(best_x+rng_hc.randn(n)*0.003,0.01,0.99)
            ny=_np.clip(best_y+rng_hc.randn(n)*0.003,0.01,0.99)
            res=try_lp(nx,ny)
            if res is not False:
                r_opt,s=res; best_sum=float(s); best_x=nx; best_y=ny; best_r=r_opt.copy()
        else:
            # shake + short rejamm (changes topology, jams contacts)
            if _tm.time()-t0 > DEADLINE-15.0:
                continue
            nx0=_np.clip(best_x+rng_hc.randn(n)*0.012,0.02,0.98)
            ny0=_np.clip(best_y+rng_hc.randn(n)*0.012,0.02,0.98)
            r0j=_np.maximum(best_r,0.02)
            # one-stage quick penalty polish with moderate P
            z=_np.concatenate([nx0,ny0,r0j])
            P=3000.0
            def fg2(zz):
                x=zz[0:n]; y=zz[n:2*n]; r=zz[2*n:3*n]
                f=-_np.sum(r)
                gx=_np.zeros(n); gy=_np.zeros(n); gr=-_np.ones(n)
                v1=r-x; mm=v1>0
                if _np.any(mm):
                    f+=P*_np.sum(v1[mm]*v1[mm]); co=2.0*P*v1[mm]; gx[mm]-=co; gr[mm]+=co
                v2=x+r-1.0; mm=v2>0
                if _np.any(mm):
                    f+=P*_np.sum(v2[mm]*v2[mm]); co=2.0*P*v2[mm]; gx[mm]+=co; gr[mm]+=co
                v3=r-y; mm=v3>0
                if _np.any(mm):
                    f+=P*_np.sum(v3[mm]*v3[mm]); co=2.0*P*v3[mm]; gy[mm]-=co; gr[mm]+=co
                v4=y+r-1.0; mm=v4>0
                if _np.any(mm):
                    f+=P*_np.sum(v4[mm]*v4[mm]); co=2.0*P*v4[mm]; gy[mm]+=co; gr[mm]+=co
                dxm=x[:,None]-x[None,:]; dym=y[:,None]-y[None,:]
                D=_np.sqrt(dxm*dxm+dym*dym)
                Rsum=r[:,None]+r[None,:]
                S=Rsum-D; _np.fill_diagonal(S,-1.0)
                mask=S>0
                if _np.any(mask):
                    f+=P*0.5*_np.sum(S[mask]*S[mask])
                    W=_np.zeros((n,n)); W[mask]=(2.0*P*S[mask])/(D[mask]+1e-12)
                    rs=W.sum(axis=1); gx+=W.dot(x)-rs*x; gy+=W.dot(y)-rs*y
                    Spos=_np.where(mask,S,0.0); gr+=2.0*P*Spos.sum(axis=1)
                return f,_np.concatenate([gx,gy,gr])
            try:
                res_o=minimize(fg2,z,method='L-BFGS-B',jac=True,bounds=bnds,
                               options={'maxiter':60,'ftol':1e-9,'gtol':1e-5})
                z=res_o.x
                xc=_np.clip(z[0:n],0.0,1.0); yc=_np.clip(z[n:2*n],0.0,1.0)
                r_opt,s=eval_lp(xc,yc)
                if r_opt is not None and s>best_sum+1e-12:
                    best_sum=float(s); best_x=xc.copy(); best_y=yc.copy(); best_r=r_opt.copy()
            except Exception:
                pass

    # final exact radii + strict repair
    r_final,_ = eval_lp(best_x,best_y)
    if r_final is not None:
        best_r = r_final
    r_safe = repair_radii(best_x,best_y,_np.maximum(best_r,0.0))
    r_safe = _np.maximum(r_safe,1e-6)
    for i in range(n):
        if best_x[i] < r_safe[i]+1e-9:
            best_x[i]=r_safe[i]+1e-9
        if best_x[i] > 1.0-r_safe[i]-1e-9:
            best_x[i]=1.0-r_safe[i]-1e-9
        if best_y[i] < r_safe[i]+1e-9:
            best_y[i]=r_safe[i]+1e-9
        if best_y[i] > 1.0-r_safe[i]-1e-9:
            best_y[i]=1.0-r_safe[i]-1e-9
    r_safe = repair_radii(best_x,best_y,r_safe)
    r_safe = _np.maximum(r_safe,1e-9)
    out = _np.column_stack([best_x,best_y,r_safe])
    return out
