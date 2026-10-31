import numpy as np
import time as _tm

def solve():
    import time as _time
    import numpy as _np
    from scipy.optimize import minimize, linprog
    n = 26
    t0 = _time.time()
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
        # quick prune
        s_ub = _np.sum(_np.maximum(b, 0.0))
        if best_ref is not None and s_ub <= best_ref:
            return None, -1.0
        if _np.any(b < 1e-7):
            # infeasible for positive radii, but still allow tiny
            # shift check: if any centre outside [0,1] skip
            if _np.any(b < -1e-12):
                return None, -1.0
        dx = xc[iu] - xc[ju]
        dy = yc[iu] - yc[ju]
        dvec = _np.sqrt(dx*dx + dy*dy)
        # prune: sum of pairwise upper bounds? skip
        bpos = _np.maximum(b, 0.0)
        bounds = [(0.0, float(bi)) for bi in bpos]
        try:
            res = linprog(c_obj, A_ub=A_ub0, b_ub=dvec, bounds=bounds, method='highs')
        except Exception:
            return None, -1.0
        if not res.success or res.x is None:
            return None, -1.0
        r = _np.array(res.x, dtype=float).flatten()
        r = _np.maximum(r, 0.0)
        s = float(_np.sum(r))
        return r, s

    def repair(xc, yc, rc):
        eps = 1e-7
        b = _np.minimum(_np.minimum(xc, 1.0-xc), _np.minimum(yc, 1.0-yc))
        rc = _np.maximum(rc, 0.0)
        t = 1.0
        pos = rc > 1e-15
        if _np.any(pos):
            ti = _np.min((b[pos]-eps)/rc[pos])
            t = min(t, ti)
        if not _np.isfinite(t) or t <= 0:
            return _np.full(n, 1e-4)*0.1
        dx = xc[iu]-xc[ju]
        dy = yc[iu]-yc[ju]
        dvec = _np.sqrt(dx*dx+dy*dy)
        rs = rc[iu]+rc[ju]
        mask = rs > 1e-15
        if _np.any(mask):
            tpair = _np.min((dvec[mask]-eps)/rs[mask])
            if tpair < t:
                t = tpair
        if t > 1.0:
            t = 1.0
        if t <= 0 or not _np.isfinite(t):
            return _np.full(n, 1e-4)*0.1
        t = t*(1.0-1e-9)
        return rc*t

    best_sum = -1.0
    best_x = None; best_y = None; best_r = None

    def consider(xc, yc):
        global best_sum, best_x, best_y, best_r
        r, s = eval_lp(xc, yc, best_ref=best_sum if best_sum>0 else None)
        if r is not None and s > best_sum + 1e-12:
            best_sum = float(s)
            best_x = xc.copy(); best_y = yc.copy(); best_r = r.copy()
            return True
        return False

    rng_init = _np.random.RandomState(0)

    # ---- Phase 1: diverse inits ----
    # 0: 5x5+1
    base5 = []
    for i in range(5):
        for j in range(5):
            base5.append([0.1+0.2*i, 0.1+0.2*j])
    base5 = _np.array(base5)
    # evaluate base + extras
    for trial in range(260):
        if _time.time()-t0 > 20.0:
            break
        if trial == 0:
            xc = base5[:,0].copy(); yc = base5[:,1].copy()
            # add extra at gap
            xc = _np.append(xc, 0.30); yc = _np.append(yc, 0.30)
        elif trial < 70:
            jit = 0.018
            xc = base5[:,0] + rng_init.randn(25)*jit
            yc = base5[:,1] + rng_init.randn(25)*jit
            ex = rng_init.uniform(0.05,0.95, size=1); ey = rng_init.uniform(0.05,0.95, size=1)
            xc = _np.append(xc, ex); yc = _np.append(yc, ey)
            xc = _np.clip(xc, 0.02, 0.98); yc = _np.clip(yc, 0.02, 0.98)
        elif trial < 120:
            xs = ( _np.arange(6)+0.5)/6.0
            ys = ( _np.arange(5)+0.5)/5.0
            xx, yy = _np.meshgrid(xs, ys)
            xy = _np.column_stack([xx.ravel(), yy.ravel()])
            perm = rng_init.permutation(len(xy))[:n]
            xc = xy[perm,0]+rng_init.randn(n)*0.008
            yc = xy[perm,1]+rng_init.randn(n)*0.008
            xc = _np.clip(xc,0.02,0.98); yc=_np.clip(yc,0.02,0.98)
        elif trial < 180:
            # staggered rows, random row counts summing to 26
            # 5 rows, base 5 each +1 extra randomly in a row
            rows = 5
            counts = _np.array([5,5,5,5,5]); counts[rng_init.randint(5)] += 1
            # random permute counts
            rng_init.shuffle(counts)
            xc_l=[]; yc_l=[]
            for r_ in range(rows):
                cr = int(counts[r_])
                yrow = (r_+0.5)/rows + rng_init.randn()*0.005
                for k_ in range(cr):
                    xpos = (k_+0.5)/cr + rng_init.randn()*0.008
                    xc_l.append(xpos); yc_l.append(yrow+rng_init.randn()*0.008)
            xc=_np.array(xc_l); yc=_np.array(yc_l)
            xc=_np.clip(xc,0.02,0.98); yc=_np.clip(yc,0.02,0.98)
        elif trial < 220:
            # 5 cols x pattern with offset (brick)
            rows=5
            counts=_np.array([5,5,5,5,6]); rng_init.shuffle(counts)
            # need sum 26
            xc_l=[]; yc_l=[]
            for r_ in range(rows):
                cr=int(counts[r_])
                yrow=(r_+0.5)/rows
                off = 0.5/cr if (r_%2==1) else 0.0
                for k_ in range(cr):
                    xpos=(k_+0.5)/cr + (0.02 if r_%2==1 else 0.0)
                    # wrap offset
                    xpos = _np.clip(xpos+rng_init.randn()*0.008,0.02,0.98)
                    xc_l.append(xpos); yc_l.append(_np.clip(yrow+rng_init.randn()*0.008,0.02,0.98))
            # counts sum =26? 5+5+5+5+6=26 yes
            xc=_np.array(xc_l); yc=_np.array(yc_l)
        else:
            xc = rng_init.uniform(0.05,0.95,size=n)
            yc = rng_init.uniform(0.05,0.95,size=n)
        # ensure length n
        if len(xc)!=n:
            continue
        r,s = eval_lp(xc,yc, best_ref=best_sum if best_sum>0 else None)
        if r is not None and s>best_sum:
            best_sum=float(s); best_x=xc.copy(); best_y=yc.copy(); best_r=r.copy()

    if best_x is None:
        k=int(_np.ceil(_np.sqrt(n))); rr=0.5/k
        pts=[]
        for i in range(k):
            for j in range(k):
                if len(pts)<n:
                    pts.append([(2*i+1)*rr,(2*j+1)*rr,rr*0.999])
        return _np.array(pts,dtype=float)

    # ---- Phase 2: large neighbourhood basin hopping ----
    rng = _np.random.RandomState(12345)
    it = 0
    # precompute order of smallest for teleport bias
    while _time.time()-t0 < DEADLINE-22.0 and it < 12000:
        it+=1
        u = rng.rand()
        nx = None; ny = None
        if u < 0.55:
            # single small perturb
            idx = int(rng.randint(n))
            sigma = 0.012 if rng.rand()<0.6 else (0.004 if rng.rand()<0.5 else 0.025)
            nx = best_x.copy(); ny = best_y.copy()
            nx[idx] += float(rng.randn())*sigma
            ny[idx] += float(rng.randn())*sigma
            nx[idx]=_np.clip(nx[idx],0.01,0.99); ny[idx]=_np.clip(ny[idx],0.01,0.99)
        elif u < 0.78:
            # teleport one circle (bias to small radii)
            if rng.rand()<0.6:
                # among 6 smallest
                order = _np.argsort(best_r)
                kk = int(rng.randint(min(6,n)))
                idx = int(order[kk])
            else:
                idx = int(rng.randint(n))
            nx = best_x.copy(); ny = best_y.copy()
            # void-biased: with prob 0.5 sample 4 candidates and pick most isolated
            if rng.rand()<0.5:
                cand_x = rng.uniform(0.02,0.98,size=4)
                cand_y = rng.uniform(0.02,0.98,size=4)
                # score by min dist to others
                best_sc=-1; bx=cand_x[0]; by=cand_y[0]
                for c_ in range(4):
                    dx_=nx-cand_x[c_]; dy_=ny-cand_y[c_]
                    dx_[idx]=1e9; dy_[idx]=1e9
                    dmin=_np.min(_np.sqrt(dx_*dx_+dy_*dy_))
                    wmin=min(cand_x[c_],1-cand_x[c_],cand_y[c_],1-cand_y[c_])
                    sc=min(dmin*0.5,wmin)
                    if sc>best_sc:
                        best_sc=sc; bx=cand_x[c_]; by=cand_y[c_]
                nx[idx]=bx; ny[idx]=by
            else:
                nx[idx]=float(rng.uniform(0.02,0.98)); ny[idx]=float(rng.uniform(0.02,0.98))
        elif u < 0.90:
            # global jitter
            sig = 0.0025 if rng.rand()<0.7 else 0.006
            nx = best_x + rng.randn(n)*sig
            ny = best_y + rng.randn(n)*sig
            nx=_np.clip(nx,0.01,0.99); ny=_np.clip(ny,0.01,0.99)
        else:
            # double perturb
            idxs = rng.choice(n,2,replace=False)
            sigma=0.010
            nx=best_x.copy(); ny=best_y.copy()
            for idx in idxs:
                nx[idx]+=float(rng.randn())*sigma
                ny[idx]+=float(rng.randn())*sigma
            nx=_np.clip(nx,0.01,0.99); ny=_np.clip(ny,0.01,0.99)
        r,s = eval_lp(nx,ny,best_ref=best_sum)
        if r is not None and s>best_sum+1e-12:
            best_sum=float(s); best_x=nx; best_y=ny; best_r=r.copy()
        # occasional shake without improvement check to diversify? keep greedy

    # ---- Phase 3: joint penalty jam polish from best ----
    bnds = [(0.0,1.0)]*(2*n)+[(1e-4,0.35)]*n
    for polish in range(3):
        if _time.time()-t0 > DEADLINE-4.0:
            break
        if polish==0:
            x0=best_x.copy(); y0=best_y.copy(); r0=best_r.copy()
        else:
            rrng=_np.random.RandomState(900+polish)
            x0=best_x+rrng.randn(n)*0.003; y0=best_y+rrng.randn(n)*0.003
            x0=_np.clip(x0,0.02,0.98); y0=_np.clip(y0,0.02,0.98)
            r0=best_r.copy()
        z=_np.concatenate([x0,y0,r0])
        for P in [300.0, 3000.0, 30000.0]:
            if _time.time()-t0 > DEADLINE-4.0:
                break
            def fg(zz, P=P):
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
                    W=_np.zeros((n,n))
                    W[mask]=(2.0*P*S[mask])/ (D[mask]+1e-12)
                    rs=W.sum(axis=1)
                    gx+=W.dot(x)-rs*x
                    gy+=W.dot(y)-rs*y
                    Spos=_np.where(mask,S,0.0)
                    gr+=2.0*P*Spos.sum(axis=1)
                g=_np.concatenate([gx,gy,gr])
                return f,g
            try:
                res=minimize(fg,z,method='L-BFGS-B',jac=True,bounds=bnds,options={'maxiter':200,'ftol':1e-12,'gtol':1e-7})
                z=res.x
            except Exception:
                break
        try:
            xc=_np.clip(z[0:n],0.0,1.0); yc=_np.clip(z[n:2*n],0.0,1.0)
            r_opt,s=eval_lp(xc,yc,best_ref=None)
            if r_opt is not None and s>best_sum+1e-9:
                best_sum=float(s); best_x=xc.copy(); best_y=yc.copy(); best_r=r_opt.copy()
            # also try clipped joint radii directly with repair
            rc=_np.clip(z[2*n:3*n],0,0.35)
            # repair joint and evaluate sum
            r_rep=repair(xc,yc,rc)
            sr=float(_np.sum(r_rep))
            if sr>best_sum:
                # verify via exact check then re-LP
                r_opt2,s2=eval_lp(xc,yc,best_ref=None)
                if r_opt2 is not None and s2>best_sum:
                    best_sum=float(s2); best_x=xc.copy(); best_y=yc.copy(); best_r=r_opt2.copy()
        except Exception:
            pass

    # final
    r_final,_=eval_lp(best_x,best_y,best_ref=None)
    if r_final is not None:
        best_r=r_final
    r_safe=repair(best_x,best_y,_np.maximum(best_r,0.0))
    r_safe=_np.maximum(r_safe,1e-9)
    for i in range(n):
        if best_x[i]<r_safe[i]+1e-9:
            best_x[i]=r_safe[i]+1e-9
        if best_x[i]>1.0-r_safe[i]-1e-9:
            best_x[i]=1.0-r_safe[i]-1e-9
        if best_y[i]<r_safe[i]+1e-9:
            best_y[i]=r_safe[i]+1e-9
        if best_y[i]>1.0-r_safe[i]-1e-9:
            best_y[i]=1.0-r_safe[i]-1e-9
    r_safe=repair(best_x,best_y,r_safe)
    r_safe=_np.maximum(r_safe,1e-9)
    out=_np.column_stack([best_x,best_y,r_safe])
    return out
