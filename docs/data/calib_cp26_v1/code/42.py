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
    # primal pair matrix m x n
    A_pair = _np.zeros((m, n), dtype=float)
    _ar = _np.arange(m)
    A_pair[_ar, iu] = 1.0
    A_pair[_ar, ju] = 1.0
    # incidence n x m for dual
    Inc = _np.zeros((n, m), dtype=float)
    for k in range(m):
        Inc[iu[k], k] = 1.0
        Inc[ju[k], k] = 1.0
    Cmat = _np.hstack([Inc, _np.eye(n, dtype=float)])  # n x (m+n)
    A_dual = -Cmat
    b_dual = -_np.ones(n, dtype=float)
    c_primal = -_np.ones(n, dtype=float)

    def eval_primal(xc, yc):
        b = _np.minimum(_np.minimum(xc, 1.0 - xc), _np.minimum(yc, 1.0 - yc))
        b = _np.maximum(b, 0.0)
        dx = xc[iu] - xc[ju]
        dy = yc[iu] - yc[ju]
        dvec = _np.sqrt(dx*dx + dy*dy)
        bounds = [(0.0, float(bi)) for bi in b]
        try:
            res = linprog(c_primal, A_ub=A_pair, b_ub=dvec, bounds=bounds, method='highs')
        except Exception:
            return None, None, -1.0, None, None
        if (not res.success) or (res.x is None):
            return None, None, -1.0, None, None
        r = _np.array(res.x, dtype=float).flatten()
        r = _np.maximum(r, 0.0)
        s = float(_np.sum(r))
        return r, s, dvec, b, res

    def eval_dual(dvec, bvec):
        c_dual = _np.concatenate([dvec, bvec]).astype(float)
        # all zero edge case
        try:
            res = linprog(c_dual, A_ub=A_dual, b_ub=b_dual, bounds=(0.0, None), method='highs')
        except Exception:
            return None, None
        if (not res.success) or (res.x is None):
            return None, None
        y = _np.array(res.x, dtype=float).flatten()
        w = y[:m]
        u = y[m:]
        # clean tiny
        w = _np.where(w < 1e-12, 0.0, w)
        u = _np.where(u < 1e-12, 0.0, u)
        return w, u

    def centre_grad(xc, yc, dvec, bvec, w, u):
        gx = _np.zeros(n)
        gy = _np.zeros(n)
        # pair part
        # avoid div by zero
        invD = _np.zeros(m)
        mask = dvec > 1e-12
        invD[mask] = w[mask] / dvec[mask]
        # accumulate
        dx = xc[iu] - xc[ju]
        dy = yc[iu] - yc[ju]
        contribx = invD * dx
        contributy = invD * dy
        # scatter
        for k in range(m):
            if invD[k] == 0.0:
                continue
            i = int(iu[k]); j = int(ju[k])
            cx = contribx[k]; cy = contributy[k]
            gx[i] += cx
            gx[j] -= cx
            gy[i] += cy
            gy[j] -= cy
        # wall part via u * db/dx
        for i in range(n):
            if u[i] <= 0:
                continue
            xl = xc[i]; xr = 1.0 - xc[i]; yb = yc[i]; yt = 1.0 - yc[i]
            bb = min(xl, xr, yb, yt)
            # tie: prefer x if tie between x and y? pick first min
            if bb == xl:
                gx[i] += u[i] * 1.0
            elif bb == xr:
                gx[i] += u[i] * (-1.0)
            elif bb == yb:
                gy[i] += u[i] * 1.0
            else:
                gy[i] += u[i] * (-1.0)
        return gx, gy

    def repair_radii(xc, yc, rc):
        eps = 1e-7
        b = _np.minimum(_np.minimum(xc, 1.0 - xc), _np.minimum(yc, 1.0 - yc))
        t = 1.0
        for i in range(n):
            if rc[i] > 1e-15:
                ti = (b[i] - eps) / rc[i]
                if ti < t:
                    t = ti
        if t <= 0:
            return _np.full(n, 1e-4)
        dx = xc[iu] - xc[ju]
        dy = yc[iu] - yc[ju]
        dvec = _np.sqrt(dx*dx + dy*dy)
        rs = rc[iu] + rc[ju]
        mask = rs > 1e-12
        if _np.any(mask):
            ratios = (dvec[mask] - eps) / rs[mask]
            tpair = _np.min(ratios)
            if tpair < t:
                t = tpair
        if t > 1.0:
            t = 1.0
        if t <= 0:
            return _np.full(n, 1e-4)
        t = t * (1.0 - 1e-9)
        return rc * t

    best_sum = -1.0
    best_x = None; best_y = None; best_r = None

    # ---------- Phase 1: short penalised joint restarts ----------
    bnds = [(0.0, 1.0)] * (2*n) + [(1e-4, 0.35)] * n
    P_seq_short = [300.0, 3000.0, 30000.0]
    restart_idx = 0
    # use ~40 sec for phase1
    PHASE1_END = t0 + 45.0
    while _tm.time() < PHASE1_END and restart_idx < 60:
        if _tm.time() - t0 > 100.0:
            break
        rng = _np.random.RandomState(1000 + restart_idx * 7919)
        if restart_idx == 0:
            pts = []
            for i in range(5):
                for j in range(5):
                    pts.append([0.1 + 0.2*i, 0.1 + 0.2*j])
            pts = _np.array(pts)
            extra = _np.array([[0.5, 0.5]])
            xy = _np.vstack([pts, extra])
            x0 = xy[:, 0].copy(); y0 = xy[:, 1].copy()
            r0 = _np.full(n, 0.09); r0[-1] = 0.04
            x0 += rng.randn(n)*0.002; y0 += rng.randn(n)*0.002
            x0 = _np.clip(x0, 0.05, 0.95); y0 = _np.clip(y0, 0.05, 0.95)
        elif restart_idx == 1:
            xs = (_np.arange(6)+0.5)/6.0; ys = (_np.arange(5)+0.5)/5.0
            xx, yy = _np.meshgrid(xs, ys)
            xy = _np.column_stack([xx.ravel(), yy.ravel()])
            perm = rng.permutation(len(xy))[:n]
            x0 = xy[perm,0] + rng.randn(n)*0.005
            y0 = xy[perm,1] + rng.randn(n)*0.005
            x0 = _np.clip(x0,0.05,0.95); y0=_np.clip(y0,0.05,0.95)
            r0 = rng.uniform(0.06,0.09,size=n)
        else:
            x0 = rng.uniform(0.05,0.95,size=n)
            y0 = rng.uniform(0.05,0.95,size=n)
            r0 = rng.uniform(0.06,0.10,size=n)
        z = _np.concatenate([x0,y0,r0])
        for P in P_seq_short:
            if _tm.time() >= PHASE1_END:
                break
            def fg(zz, P=P):
                x = zz[0:n]; y = zz[n:2*n]; r = zz[2*n:3*n]
                f = -_np.sum(r)
                gx = _np.zeros(n); gy=_np.zeros(n); gr=-_np.ones(n)
                v1 = r - x
                mm = v1>0
                if _np.any(mm):
                    f += P*_np.sum(v1[mm]*v1[mm])
                    co = 2.0*P*v1[mm]
                    gx[mm]-=co; gr[mm]+=co
                v2 = x+r-1.0
                mm=v2>0
                if _np.any(mm):
                    f += P*_np.sum(v2[mm]*v2[mm])
                    co=2.0*P*v2[mm]
                    gx[mm]+=co; gr[mm]+=co
                v3=r-y
                mm=v3>0
                if _np.any(mm):
                    f+=P*_np.sum(v3[mm]*v3[mm])
                    co=2.0*P*v3[mm]
                    gy[mm]-=co; gr[mm]+=co
                v4=y+r-1.0
                mm=v4>0
                if _np.any(mm):
                    f+=P*_np.sum(v4[mm]*v4[mm])
                    co=2.0*P*v4[mm]
                    gy[mm]+=co; gr[mm]+=co
                dxm=x[:,None]-x[None,:]
                dym=y[:,None]-y[None,:]
                D=_np.sqrt(dxm*dxm+dym*dym)
                Rsum=r[:,None]+r[None,:]
                S=Rsum-D
                _np.fill_diagonal(S,-1.0)
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
                res = minimize(fg, z, method='L-BFGS-B', jac=True, bounds=bnds,
                               options={'maxiter':120,'ftol':1e-12,'gtol':1e-7})
                z=res.x
            except Exception:
                break
        try:
            xc=_np.clip(z[0:n],0.0,1.0); yc=_np.clip(z[n:2*n],0.0,1.0)
            r_opt,s,_,_,_ = eval_primal(xc,yc)
            if r_opt is not None and s>best_sum:
                best_sum=float(s); best_x=xc.copy(); best_y=yc.copy(); best_r=r_opt.copy()
        except Exception:
            pass
        restart_idx+=1

    if best_x is None:
        # fallback grid
        k=int(_np.ceil(_np.sqrt(n))); rr=0.5/k; pts=[]
        for i in range(k):
            for j in range(k):
                if len(pts)<n:
                    pts.append([(2*i+1)*rr,(2*j+1)*rr,rr*0.999])
        arr=_np.array(pts,dtype=float)
        return arr

    # clip to [0.02,0.98] for dual phase
    best_x=_np.clip(best_x,0.02,0.98); best_y=_np.clip(best_y,0.02,0.98)
    # re-eval after clip
    r_tmp,s_tmp,_,_,_ = eval_primal(best_x,best_y)
    if r_tmp is not None:
        best_r=r_tmp; best_sum=float(s_tmp)

    # ---------- Phase 2: dual-guided projected gradient ascent ----------
    steps = [0.02, 0.01, 0.005, 0.002, 0.001, 5e-4, 2e-4, 1e-4]
    rng_hc=_np.random.RandomState(7777)
    stall=0
    it=0
    # pre-evaluate dual for current best to start
    while _tm.time()-t0 < DEADLINE-4.0 and it < 10000:
        it+=1
        # compute primal + dual for current best
        _, s_cur, dvec_cur, bvec_cur, _ = eval_primal(best_x, best_y)
        if s_cur<0:
            break
        # keep best_sum in sync
        if s_cur>best_sum:
            best_sum=s_cur
            r_tmp2,_,_,_,_ = eval_primal(best_x,best_y)
            # already have r? re-eval
            rr,_d,_,_,_ = eval_primal(best_x,best_y)
            # avoid extra call: use eval again
            pass
        w,u = eval_dual(dvec_cur, bvec_cur)
        moved=False
        if w is not None:
            gx,gy = centre_grad(best_x,best_y,dvec_cur,bvec_cur,w,u)
            gn=_np.sqrt(_np.sum(gx*gx+gy*gy))
            if gn>1e-9 and gn<1e4:
                # optional normalise? keep raw, backtrack handles
                for st in steps:
                    if _tm.time()-t0>=DEADLINE-4.0:
                        break
                    nx=_np.clip(best_x+st*gx,0.02,0.98)
                    ny=_np.clip(best_y+st*gy,0.02,0.98)
                    _, s_try,_,_,_ = eval_primal(nx,ny)
                    if s_try>best_sum+1e-11:
                        # accept
                        r_new,s_new,_,_,_ = eval_primal(nx,ny)
                        best_x=nx; best_y=ny; best_r=r_new; best_sum=float(s_new)
                        moved=True
                        stall=0
                        break
        if moved:
            # occasional NLP polish
            if it%150==0 and _tm.time()-t0<DEADLINE-15.0:
                try:
                    z0=_np.concatenate([best_x,best_y,_np.maximum(best_r,1e-3)])
                    Ppol=20000.0
                    def fgpol(zz):
                        x=zz[0:n]; y=zz[n:2*n]; r=zz[2*n:3*n]
                        f=-_np.sum(r)
                        gx=_np.zeros(n); gy=_np.zeros(n); gr=-_np.ones(n)
                        v1=r-x; mm=v1>0
                        if _np.any(mm):
                            f+=Ppol*_np.sum(v1[mm]*v1[mm]); co=2.0*Ppol*v1[mm]; gx[mm]-=co; gr[mm]+=co
                        v2=x+r-1.0; mm=v2>0
                        if _np.any(mm):
                            f+=Ppol*_np.sum(v2[mm]*v2[mm]); co=2.0*Ppol*v2[mm]; gx[mm]+=co; gr[mm]+=co
                        v3=r-y; mm=v3>0
                        if _np.any(mm):
                            f+=Ppol*_np.sum(v3[mm]*v3[mm]); co=2.0*Ppol*v3[mm]; gy[mm]-=co; gr[mm]+=co
                        v4=y+r-1.0; mm=v4>0
                        if _np.any(mm):
                            f+=Ppol*_np.sum(v4[mm]*v4[mm]); co=2.0*Ppol*v4[mm]; gy[mm]+=co; gr[mm]+=co
                        dxm=x[:,None]-x[None,:]; dym=y[:,None]-y[None,:]
                        D=_np.sqrt(dxm*dxm+dym*dym); Rsum=r[:,None]+r[None,:]
                        S=Rsum-D; _np.fill_diagonal(S,-1.0); mask=S>0
                        if _np.any(mask):
                            f+=Ppol*0.5*_np.sum(S[mask]*S[mask])
                            W=_np.zeros((n,n)); W[mask]=(2.0*Ppol*S[mask])/(D[mask]+1e-12)
                            rs=W.sum(axis=1); gx+=W.dot(x)-rs*x; gy+=W.dot(y)-rs*y
                            Spos=_np.where(mask,S,0.0); gr+=2.0*Ppol*Spos.sum(axis=1)
                        return f,_np.concatenate([gx,gy,gr])
                    respol=minimize(fgpol,z0,method='L-BFGS-B',jac=True,bounds=bnds,options={'maxiter':80,'ftol':1e-12,'gtol':1e-7})
                    zp=respol.x
                    xcp=_np.clip(zp[0:n],0.02,0.98); ycp=_np.clip(zp[n:2*n],0.02,0.98)
                    r_opt2,s2,_,_,_ = eval_primal(xcp,ycp)
                    if r_opt2 is not None and s2>best_sum+1e-11:
                        best_x=xcp; best_y=ycp; best_r=r_opt2; best_sum=float(s2)
                except Exception:
                    pass
            continue
        # no dual move: stochastic single-circle hill climb
        stall+=1
        # adapt sigma
        sigma = 0.015*_np.exp(-it/2500.0)+0.0008
        improved=False
        for _trial in range(25):
            if _tm.time()-t0>=DEADLINE-4.0:
                break
            idx=int(rng_hc.randint(n))
            nx=best_x.copy(); ny=best_y.copy()
            nx[idx]+=float(rng_hc.randn())*sigma
            ny[idx]+=float(rng_hc.randn())*sigma
            nx[idx]=_np.clip(nx[idx],0.02,0.98); ny[idx]=_np.clip(ny[idx],0.02,0.98)
            _, s_try,_,_,_ = eval_primal(nx,ny)
            if s_try>best_sum+1e-11:
                r_new,_,_,_,_ = eval_primal(nx,ny)
                # second call wasteful but ok; reuse r from first? re-eval to get r
                rr2,s2,_,_,_ = eval_primal(nx,ny)
                best_x=nx; best_y=ny; best_r=rr2; best_sum=float(s2)
                improved=True
                stall=0
                break
        if improved:
            continue
        # occasional big kick of worst rattler if long stall
        if stall%80==0 and _tm.time()-t0<DEADLINE-10.0:
            # relocate smallest radius circle to random void
            if best_r is not None:
                idx=int(_np.argmin(best_r))
                nx=best_x.copy(); ny=best_y.copy()
                nx[idx]=float(rng_hc.uniform(0.05,0.95)); ny[idx]=float(rng_hc.uniform(0.05,0.95))
                _, s_try,_,_,_ = eval_primal(nx,ny)
                if s_try>best_sum+1e-11:
                    rr2,s2,_,_,_ = eval_primal(nx,ny)
                    best_x=nx; best_y=ny; best_r=rr2; best_sum=float(s2)
                    stall=0

    # final exact radii + strict repair
    r_final,_s,_,_,_ = eval_primal(best_x,best_y)
    if r_final is not None:
        best_r=r_final
    r_safe=repair_radii(best_x,best_y,_np.maximum(best_r,0.0))
    r_safe=_np.maximum(r_safe,1e-6)
    for i in range(n):
        if best_x[i]<r_safe[i]+1e-9:
            best_x[i]=r_safe[i]+1e-9
        if best_x[i]>1.0-r_safe[i]-1e-9:
            best_x[i]=1.0-r_safe[i]-1e-9
        if best_y[i]<r_safe[i]+1e-9:
            best_y[i]=r_safe[i]+1e-9
        if best_y[i]>1.0-r_safe[i]-1e-9:
            best_y[i]=1.0-r_safe[i]-1e-9
    r_safe=repair_radii(best_x,best_y,r_safe)
    r_safe=_np.maximum(r_safe,1e-9)
    out=_np.column_stack([best_x,best_y,r_safe])
    return out
