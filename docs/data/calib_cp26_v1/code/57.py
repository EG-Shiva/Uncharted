import numpy as np
import time as _time

def solve():
    import time as _tm
    import numpy as _np
    from scipy.optimize import minimize, linprog
    n = 26
    t0 = _tm.time()
    DEADLINE = 124.0
    ROW_BUDGET = 42.0
    _np.random.seed(0)

    iu, ju = _np.triu_indices(n, 1)
    m = len(iu)
    A_ub0 = _np.zeros((m, n), dtype=float)
    _ar = _np.arange(m)
    A_ub0[_ar, iu] = 1.0
    A_ub0[_ar, ju] = 1.0
    c_obj = -_np.ones(n)
    EPS = 1e-9

    def eval_lp(Xc, Yc):
        try:
            Xc = _np.asarray(Xc, float).ravel()
            Yc = _np.asarray(Yc, float).ravel()
            if Xc.size != n or Yc.size != n:
                return None, -1.0
            if _np.any(Xc < -1e-9) or _np.any(Xc > 1+1e-9) or _np.any(Yc < -1e-9) or _np.any(Yc > 1+1e-9):
                return None, -1.0
            wall = _np.minimum(_np.minimum(Xc, 1.0-Xc), _np.minimum(Yc, 1.0-Yc))
            if _np.any(wall < 1e-9):
                return None, -1.0
            dx = Xc[iu]-Xc[ju]; dy = Yc[iu]-Yc[ju]
            dvec = _np.sqrt(dx*dx+dy*dy)
            if _np.any(dvec < 2e-9):
                return None, -1.0
            bnds = [(0.0, float(w)-EPS) for w in wall]
            for lo, hi in bnds:
                if hi < 0:
                    return None, -1.0
            try:
                res = linprog(c_obj, A_ub=A_ub0, b_ub=dvec-EPS, bounds=bnds, method='highs')
            except Exception:
                return None, -1.0
            if not res.success or res.x is None:
                return None, -1.0
            r = _np.asarray(res.x, float).ravel()
            r = _np.maximum(r, 0.0)
            return r, float(_np.sum(r))
        except Exception:
            return None, -1.0

    def uniform_shrink(Xc, Yc, Rc):
        Xc = _np.asarray(Xc, float); Yc = _np.asarray(Yc, float); Rc = _np.asarray(Rc, float)
        Rc = _np.maximum(Rc, 1e-12)
        wall = _np.minimum(_np.minimum(Xc, 1.0-Xc), _np.minimum(Yc, 1.0-Yc))
        a = _np.min(wall / _np.maximum(Rc, 1e-12))
        DX = Xc[:, None]-Xc[None, :]
        DY = Yc[:, None]-Yc[None, :]
        D = _np.sqrt(DX*DX+DY*DY)
        S = Rc[:, None]+Rc[None, :]
        with _np.errstate(divide='ignore', invalid='ignore'):
            ratio = D / S
        _np.fill_diagonal(ratio, _np.inf)
        b = _np.min(ratio)
        alpha = min(1.0, float(a), float(b))
        alpha = alpha*(1.0-1e-7)-1e-12
        if not _np.isfinite(alpha) or alpha <= 0:
            alpha = 0.5
        return Rc*alpha, alpha

    # fallback feasible
    k0 = int(_np.ceil(_np.sqrt(n)))
    r0 = 0.5/k0
    pts = []
    for i in range(k0):
        for j in range(k0):
            if len(pts) < n:
                pts.append([(2*i+1)*r0, (2*j+1)*r0, r0*0.999])
    best_arr = _np.array(pts, float)
    Rb_s, _ = uniform_shrink(best_arr[:,0], best_arr[:,1], best_arr[:,2])
    best_arr = _np.column_stack([best_arr[:,0], best_arr[:,1], Rb_s])
    best_sum = float(_np.sum(Rb_s))

    def consider(Xc, Yc, Rc):
        global_best = None
        nonlocal_best = None
        try:
            Xc = _np.asarray(Xc, float).copy(); Yc = _np.asarray(Yc, float).copy(); Rc = _np.asarray(Rc, float).copy()
            Rs, _ = uniform_shrink(Xc, Yc, Rc)
            # ensure strictly inside after shrink
            Rs = _np.maximum(Rs, 1e-6)
            # re-shrink if clipping to 1e-6 broke feasibility (tiny)
            Rs2, _ = uniform_shrink(Xc, Yc, Rs)
            # if original Rs already >1e-6, Rs2==Rs up to margin; use Rs2
            # only use Rs2 if it differs a lot (means 1e-6 clip infeasible)
            if _np.min(Rs) < 1e-6+1e-12:
                Rs = Rs2
            s = float(_np.sum(Rs))
            # update outer via closure
            global best_sum, best_arr
            if s > best_sum:
                best_sum = s
                best_arr = _np.column_stack([Xc, Yc, Rs])
        except Exception:
            pass

    def consider_lp(Xc, Yc):
        try:
            r, s = eval_lp(Xc, Yc)
            if r is not None and s > 0:
                r = r*(1.0-5e-9)
                # avoid exact zero radii
                r = _np.maximum(r, 1e-7)
                consider(Xc, Yc, r)
                return float(_np.sum(r))
        except Exception:
            pass
        return -1.0

    # ---------- staggered row templates ----------
    templates = [
        (6,5,6,5,4),
        (5,6,5,6,4),
        (6,5,5,6,4),
        (5,5,6,5,5),
        (6,6,5,5,4),
        (6,5,6,4,5),
        (7,6,6,7),
        (6,7,6,7),
    ]

    def centers_from_params(p, counts):
        kk = len(counts)
        yv = p[0:kk]; lv = p[kk:2*kk]; sv = p[2*kk:3*kk]; wv = p[3*kk:4*kk]
        Xs=[]; Ys=[]
        for j, cj in enumerate(counts):
            yj=float(yv[j]); lj=float(lv[j]); sj=float(sv[j]); wj=float(wv[j])
            for i in range(cj):
                dx = i-(cj-1)/2.0
                x = lj + i*sj + wj*dx*dx
                Xs.append(x); Ys.append(yj)
        return _np.array(Xs), _np.array(Ys)

    def row_obj(p, counts):
        try:
            X, Y = centers_from_params(p, counts)
            if _np.any(X < 0.0) or _np.any(X > 1.0) or _np.any(Y < 0.0) or _np.any(Y > 1.0):
                return 0.0
            _, s = eval_lp(X, Y)
            if s < 0:
                return 0.0
            return -s
        except Exception:
            return 0.0

    row_bests = []
    for tc in templates:
        if _tm.time()-t0 > ROW_BUDGET:
            break
        kk = len(tc)
        dim = 4*kk
        y0 = _np.array([(j+0.5)/kk for j in range(kk)])
        s0v = _np.full(kk, 0.165)
        l0v = _np.zeros(kk)
        for j, cj in enumerate(tc):
            l0v[j]=(1.0-(cj-1)*s0v[j])/2.0
            if j%2==1:
                l0v[j]+=0.02
                if l0v[j]+(cj-1)*s0v[j] > 0.95:
                    l0v[j]=0.95-(cj-1)*s0v[j]
        w0v = _np.zeros(kk)
        p0 = _np.concatenate([y0, l0v, s0v, w0v])
        bnds=[]
        for j in range(kk): bnds.append((0.04,0.96))
        for j in range(kk): bnds.append((0.02,0.45))
        for j in range(kk): bnds.append((0.09,0.26))
        for j in range(kk): bnds.append((-0.015,0.015))
        # first Powell (cheap)
        try:
            res = minimize(row_obj, p0, args=(tc,), method='Powell',
                           bounds=bnds, options={'maxiter':12,'maxfun':120,'disp':False})
            Xb, Yb = centers_from_params(res.x, tc)
            s, r = eval_lp(Xb, Yb)
            if r is not None and s>0:
                row_bests.append((s, Xb.copy(), Yb.copy(), tc))
                consider(Xb, Yb, r*(1.0-5e-9))
                # transpose candidate (swap axes)
                st, rt = eval_lp(Yb, Xb)
                if rt is not None and st>0:
                    row_bests.append((st, Yb.copy(), Xb.copy(), tc))
                    consider(Yb, Xb, rt*(1.0-5e-9))
        except Exception:
            pass
        if _tm.time()-t0 > ROW_BUDGET-2:
            break
        # second jittered start, cheaper
        try:
            rng = _np.random.RandomState((hash(tc)&0x7fffffff)%(2**31-1))
            sig = _np.array([0.02]*kk+[0.02]*kk+[0.01]*kk+[0.005]*kk)
            p1 = p0 + rng.randn(dim)*sig
            for di in range(dim):
                lo,hi=bnds[di]
                if p1[di]<lo: p1[di]=lo
                if p1[di]>hi: p1[di]=hi
            res2 = minimize(row_obj, p1, args=(tc,), method='Powell',
                            bounds=bnds, options={'maxiter':8,'maxfun':80,'disp':False})
            Xb2, Yb2 = centers_from_params(res2.x, tc)
            s2, r2 = eval_lp(Xb2, Yb2)
            if r2 is not None and s2>0:
                row_bests.append((s2, Xb2.copy(), Yb2.copy(), tc))
                consider(Xb2, Yb2, r2*(1.0-5e-9))
        except Exception:
            pass

    row_bests.sort(key=lambda t: -t[0])
    seeds=[]
    for s,Xb,Yb,tc in row_bests[:3]:
        seeds.append((Xb.copy(),Yb.copy()))
        rng=_np.random.RandomState(1000+len(seeds)*97)
        Xn=_np.clip(Xb+rng.randn(n)*0.004,0.02,0.98)
        Yn=_np.clip(Yb+rng.randn(n)*0.004,0.02,0.98)
        seeds.append((Xn,Yn))

    # ---------- centre-only LP hill climbing (individual displacements) ----------
    try:
        # keep top 3 unique
        uniq=[]
        for Xb,Yb in seeds:
            if len(uniq)>=6: break
            uniq.append((Xb,Yb))
        new_seeds=[]
        for si,(Xc,Yc) in enumerate(uniq[:3]):
            if _tm.time()-t0 > DEADLINE-55:
                break
            Xc=_np.array(Xc,float); Yc=_np.array(Yc,float)
            cur_r,cur_s = eval_lp(Xc,Yc)
            if cur_r is None: continue
            consider(Xc,Yc,cur_r)
            rng=_np.random.RandomState(777+si)
            for sigma,iters in [(0.015,40),(0.007,40),(0.003,30)]:
                if _tm.time()-t0 > DEADLINE-55: break
                for _ in range(iters):
                    if _tm.time()-t0 > DEADLINE-55: break
                    if rng.rand()<0.6:
                        idx=rng.integers(0,n)
                        Xn=Xc.copy(); Yn=Yc.copy()
                        Xn[idx]+=rng.normal(0,sigma); Yn[idx]+=rng.normal(0,sigma)
                    else:
                        Xn=Xc+rng.normal(0,sigma*0.5,size=n)
                        Yn=Yc+rng.normal(0,sigma*0.5,size=n)
                    Xn=_np.clip(Xn,0.01,0.99); Yn=_np.clip(Yn,0.01,0.99)
                    nr,ns=eval_lp(Xn,Yn)
                    if nr is not None and ns>cur_s:
                        cur_s=Xn=Xn  # placeholder
                        cur_s=ns; Xc=Xn; Yc=Yn; cur_r=nr
                        consider(Xc,Yc,cur_r*(1.0-5e-9))
            new_seeds.append((Xc.copy(),Yc.copy()))
        seeds.extend(new_seeds)
    except Exception:
        pass

    # ---------- short constrained joint polish ----------
    bndsJ=[(0.0,1.0)]*(2*n)+[(1e-4,0.35)]*n
    P_seq=[20.0,200.0,2000.0,20000.0]
    def run_penalty(x0,y0,r0,maxiter_each=70):
        z=_np.concatenate([_np.asarray(x0,float),_np.asarray(y0,float),_np.asarray(r0,float)])
        for P in P_seq:
            if _tm.time()-t0 > DEADLINE-6: break
            def fg(zz,P=P):
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
                g=_np.concatenate([gx,gy,gr])
                return f,g
            try:
                res=minimize(fg,z,method='L-BFGS-B',jac=True,bounds=bndsJ,options={'maxiter':maxiter_each,'ftol':1e-10,'gtol':1e-6})
                z=res.x
            except Exception:
                break
        return z

    try:
        # order seeds by lp sum
        scored=[]
        for Xc,Yc in seeds:
            _,s=eval_lp(Xc,Yc)
            scored.append((s,Xc,Yc))
        scored.sort(key=lambda t:-t[0])
        # also add a couple of random restarts if time allows (basin escape)
        # build polish starts: top 3
        starts=scored[:3]
        for si,(s,Xc,Yc) in enumerate(starts):
            if _tm.time()-t0 > DEADLINE-8: break
            r0,_=eval_lp(Xc,Yc)
            if r0 is None:
                r0=_np.full(n,0.05)
            z=run_penalty(Xc,Yc,r0, maxiter_each=80)
            Xp=z[0:n]; Yp=z[n:2*n]; Rp=z[2*n:3*n]
            Rp=_np.maximum(Rp,1e-7)
            consider(Xp,Yp,Rp)
            # LP-optimal radii for these centres (exact scoring)
            consider_lp(_np.clip(Xp,0.01,0.99), _np.clip(Yp,0.01,0.99))
    except Exception:
        pass

    # final safety: ensure best_arr strictly feasible with margin
    try:
        Xf=_np.clip(best_arr[:,0],0.0,1.0); Yf=_np.clip(best_arr[:,1],0.0,1.0); Rf=_np.maximum(best_arr[:,2],1e-6)
        Rf,_=uniform_shrink(Xf,Yf,Rf)
        Rf=Rf*(1.0-1e-9)
        Rf=_np.maximum(Rf,1e-7)
        # one more shrink check
        Rf,_=uniform_shrink(Xf,Yf,Rf)
        best_arr=_np.column_stack([Xf,Yf,Rf])
    except Exception:
        pass
    return best_arr
