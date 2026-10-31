import numpy as np
import time

def solve():
    import time as _tm
    import numpy as _np
    from scipy.optimize import minimize, linprog
    n = 26
    t0 = _tm.time()
    DEADLINE = 122.0

    iu, ju = _np.triu_indices(n, 1)
    m = len(iu)
    A_ub0 = _np.zeros((m, n), dtype=float)
    _ar = _np.arange(m)
    A_ub0[_ar, iu] = 1.0
    A_ub0[_ar, ju] = 1.0
    c_obj = -_np.ones(n)

    rng = _np.random.default_rng(0)

    def eval_lp(xc, yc, best_ref=-1.0):
        b = _np.minimum(_np.minimum(xc, 1.0 - xc), _np.minimum(yc, 1.0 - yc))
        if _np.any(b < 1e-9):
            return None, -1.0
        if _np.sum(b) <= best_ref:
            return None, -1.0
        dx = xc[iu] - xc[ju]
        dy = yc[iu] - yc[ju]
        dvec = _np.sqrt(dx*dx + dy*dy)
        # prune: sum of pairwise upper bound
        # quick necessary: if any dvec very small, still feasible but poor
        bounds = [(0.0, float(bi)) for bi in b]
        try:
            res = linprog(c_obj, A_ub=A_ub0, b_ub=dvec, bounds=bounds, method='highs')
        except Exception:
            return None, -1.0
        if not res.success or res.x is None:
            return None, -1.0
        r = _np.array(res.x, dtype=float).ravel()
        r = _np.maximum(r, 0.0)
        s = float(_np.sum(r))
        if s <= best_ref:
            return None, -1.0
        return r, s

    # fallback grid
    k = int(_np.ceil(_np.sqrt(n)))
    r0f = 0.5 / k
    fb = []
    for ii in range(k):
        for jj in range(k):
            if len(fb) < n:
                fb.append([(2*ii+1)*r0f, (2*jj+1)*r0f, r0f*0.999*0.999])
    best_arr = _np.array(fb, dtype=float)
    # scale fallback to be safe
    def uniform_scale(X, Y, R):
        wall = _np.minimum(_np.minimum(X, 1-X), _np.minimum(Y, 1-Y))
        t = 1.0
        for i in range(len(R)):
            if R[i] > 0:
                ti = wall[i]/R[i]
                if ti < t: t = ti
        Dx = X[:,None]-X[None,:]
        Dy = Y[:,None]-Y[None,:]
        D = _np.sqrt(Dx*Dx+Dy*Dy)
        S = R[:,None]+R[None,:]
        _np.fill_diagonal(D, _np.inf)
        _np.fill_diagonal(S, 1.0)
        ratio = D/S
        t2 = _np.min(ratio)
        if t2 < t: t = t2
        t = t*(1.0-1e-7)-1e-12
        if not _np.isfinite(t) or t<=0: t=0.9
        if t>1: t=1.0*(1.0-1e-7)
        return R*t, t
    Xb, Yb, Rb = best_arr[:,0], best_arr[:,1], best_arr[:,2]
    Rb_s, _ = uniform_scale(Xb, Yb, Rb)
    best_arr = _np.column_stack([Xb, Yb, Rb_s])
    best_sum = float(_np.sum(Rb_s))
    best_x = Xb.copy(); best_y = Yb.copy(); best_r = Rb_s.copy()

    bnds = [(0.0,1.0)]*(2*n)+[(0.02,0.22)]*n

    # ---- Phase 1: penalised joint optimisation with wall-free map ----
    # x = u + r*(1-2u), y = v + r*(1-2v)
    restart = 0
    PHASE1_END = 72.0
    Ws = [30.0, 300.0, 3000.0, 30000.0]
    while True:
        if _tm.time()-t0 > PHASE1_END: break
        if restart > 300: break
        # init
        if restart == 0:
            # 5x5 grid
            u0 = _np.zeros(n); v0=_np.zeros(n)
            idx=0
            for ix_ in range(5):
                for jy_ in range(5):
                    u0[idx]=(ix_+0.5)/5.0 + rng.normal(0,0.004)
                    v0[idx]=(jy_+0.5)/5.0 + rng.normal(0,0.004)
                    idx+=1
            u0[25]=0.85+rng.normal(0,0.01); v0[25]=0.15+rng.normal(0,0.01)
            u0=_np.clip(u0,0.02,0.98); v0=_np.clip(v0,0.02,0.98)
            r0 = _np.full(n,0.095)+rng.normal(0,0.005,n)
            r0=_np.clip(r0,0.03,0.15)
        elif restart == 1:
            xs = (_np.arange(6)+0.5)/6.0
            ys = (_np.arange(5)+0.5)/5.0
            xx,yy = _np.meshgrid(xs,ys)
            xy = _np.column_stack([xx.ravel(),yy.ravel()])
            perm = rng.permutation(len(xy))[:n]
            u0 = xy[perm,0]+rng.normal(0,0.006,n)
            v0 = xy[perm,1]+rng.normal(0,0.006,n)
            u0=_np.clip(u0,0.02,0.98); v0=_np.clip(v0,0.02,0.98)
            r0 = rng.uniform(0.06,0.11,size=n)
        else:
            u0 = rng.uniform(0.02,0.98,size=n)
            v0 = rng.uniform(0.02,0.98,size=n)
            if restart%3==0:
                r0 = _np.full(n,0.10)+rng.normal(0,0.015,n)
            else:
                r0 = rng.uniform(0.05,0.12,size=n)
            r0=_np.clip(r0,0.02,0.20)
        z = _np.concatenate([u0,v0,r0])
        for W in Ws:
            if _tm.time()-t0 > PHASE1_END: break
            def fun(zz, W=W):
                u = zz[:n]; vv = zz[n:2*n]; rr = zz[2*n:]
                om2r = 1.0-2.0*rr
                # x,y
                x = u + rr*(1.0-2.0*u)
                y = vv + rr*(1.0-2.0*vv)
                Dx = x[:,None]-x[None,:]
                Dy = y[:,None]-y[None,:]
                D = _np.sqrt(Dx*Dx+Dy*Dy+1e-20)
                _np.fill_diagonal(D,1.0)
                S = rr[:,None]+rr[None,:]
                E = S-D
                _np.fill_diagonal(E,0.0)
                Ep = _np.maximum(E,0.0)
                Pen = 0.5*_np.sum(Ep*Ep)
                F = -_np.sum(rr)+W*Pen
                InvD = 1.0/D
                _np.fill_diagonal(InvD,0.0)
                Wmat = Ep*InvD
                Gx = -_np.sum(Wmat*Dx,axis=1)
                Gy = -_np.sum(Wmat*Dy,axis=1)
                sumEp = _np.sum(Ep,axis=1)
                dFdu = W*Gx*om2r
                dFdv = W*Gy*om2r
                dFdr = -_np.ones(n)+W*(2.0*sumEp+Gx*(1.0-2.0*u)+Gy*(1.0-2.0*vv))
                g = _np.concatenate([dFdu,dFdv,dFdr])
                return F,g
            try:
                res = minimize(fun,z,method='L-BFGS-B',jac=True,bounds=bnds,
                               options={'maxiter':220,'ftol':1e-9,'gtol':1e-5})
                z=res.x
            except Exception:
                break
        u = z[:n]; vv=z[n:2*n]; rr=z[2*n:]
        rr=_np.clip(rr,0.01,0.25)
        x = u+rr*(1.0-2.0*u)
        y = vv+rr*(1.0-2.0*vv)
        x=_np.clip(x,1e-6,1-1e-6); y=_np.clip(y,1e-6,1-1e-6)
        r_opt,s = eval_lp(x,y,best_sum)
        if r_opt is not None and s>best_sum:
            best_sum=s; best_x=x.copy(); best_y=y.copy(); best_r=r_opt.copy()
            best_arr=_np.column_stack([x,y,r_opt])
        # also try uniform-scaled penalised radii as candidate
        try:
            Rs,_ = uniform_scale(x,y,rr)
            su=float(_np.sum(Rs))
            if su>best_sum:
                best_sum=su; best_x=x.copy(); best_y=y.copy(); best_r=Rs.copy()
                best_arr=_np.column_stack([x,y,Rs])
        except Exception:
            pass
        restart+=1

    # ---- Phase 2: hill climbing on centres with LP ----
    # keep elite list
    elites_x=[best_x.copy()]; elites_y=[best_y.copy()]; elites_r=[best_r.copy()]
    # add a few perturbed copies of best to diversify
    for _ in range(2):
        elites_x.append(_np.clip(best_x+rng.normal(0,0.01,n),0.01,0.99))
        elites_y.append(_np.clip(best_y+rng.normal(0,0.01,n),0.01,0.99))
        elites_r.append(best_r.copy())
    sigmas=[0.02,0.01,0.005,0.002,0.001]
    # estimate iterations: LP ~5ms, we have ~45 sec => ~5000 evals
    it=0
    while _tm.time()-t0 < DEADLINE-4:
        # cycle elites and sigmas
        ei = it % len(elites_x)
        si = (it//max(1,len(elites_x))) % len(sigmas)
        # occasionally restart sigma schedule
        sigma = sigmas[si]
        # propose
        cx = elites_x[ei]; cy = elites_y[ei]
        # move single circle more often for fine tune, else all
        if rng.random()<0.6:
            nx = cx.copy(); ny = cy.copy()
            kidx = rng.integers(0,n)
            nx[kidx]+=rng.normal(0,sigma*2.0)
            ny[kidx]+=rng.normal(0,sigma*2.0)
            nx[kidx]=_np.clip(nx[kidx],0.02,0.98)
            ny[kidx]=_np.clip(ny[kidx],0.02,0.98)
        else:
            nx = _np.clip(cx+rng.normal(0,sigma,n),0.015,0.985)
            ny = _np.clip(cy+rng.normal(0,sigma,n),0.015,0.985)
        r_opt,s = eval_lp(nx,ny,best_sum-1e-12)
        # accept if improves elite or global
        if r_opt is not None and s>best_sum+1e-12:
            best_sum=s; best_x=nx.copy(); best_y=ny.copy(); best_r=r_opt.copy()
            elites_x[ei]=nx.copy(); elites_y[ei]=ny.copy(); elites_r[ei]=r_opt.copy()
            best_arr=_np.column_stack([nx,ny,r_opt])
        else:
            # also allow elite-level improvement without beating global? check vs elite
            # evaluate elite sum quickly: need its sum
            # compute elite sum via LP? we stored r, compute sum
            es = float(_np.sum(elites_r[ei]))
            if r_opt is not None and s>es:
                elites_x[ei]=nx.copy(); elites_y[ei]=ny.copy(); elites_r[ei]=r_opt.copy()
                if s>best_sum:
                    best_sum=s; best_x=nx.copy(); best_y=ny.copy(); best_r=r_opt.copy()
                    best_arr=_np.column_stack([nx,ny,r_opt])
            else:
                # occasional random restart of worst elite
                if rng.random()<0.002:
                    elites_x[ei]=_np.clip(best_x+rng.normal(0,0.02,n),0.015,0.985)
                    elites_y[ei]=_np.clip(best_y+rng.normal(0,0.02,n),0.015,0.985)
        it+=1
        if it>20000:
            break

    # ---- final safety shrink ----
    try:
        Xf=best_arr[:,0]; Yf=best_arr[:,1]; Rf=best_arr[:,2]
        Rf_s,_=uniform_scale(Xf,Yf,Rf)
        best_arr=_np.column_stack([Xf,Yf,Rf_s])
    except Exception:
        pass
    # ensure shape and positivity
    best_arr=_np.array(best_arr,dtype=float).reshape(n,3)
    best_arr[:,2]=_np.maximum(best_arr[:,2],1e-6)
    best_arr[:,0]=_np.clip(best_arr[:,0],1e-6,1-1e-6)
    best_arr[:,1]=_np.clip(best_arr[:,1],1e-6,1-1e-6)
    return best_arr
