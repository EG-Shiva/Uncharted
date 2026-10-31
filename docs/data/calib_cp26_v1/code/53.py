import numpy as np
import time

def solve():
    import time
    import numpy as np
    t0 = time.time()
    DEADLINE = 124.0
    n = 26
    rng = np.random.default_rng(0)
    try:
        from scipy.optimize import minimize, linprog
        has_opt = True
    except Exception:
        has_opt = False

    I, J = np.triu_indices(n, 1)
    m = len(I)
    A0 = np.zeros((m, n), dtype=float)
    A0[np.arange(m), I] = 1.0
    A0[np.arange(m), J] = 1.0
    cobj = -np.ones(n, dtype=float)

    def uniform_scale(X, Y, R):
        R = np.maximum(np.asarray(R, float), 1e-12)
        X = np.asarray(X, float); Y = np.asarray(Y, float)
        wall = np.minimum(np.minimum(X, 1.0-X), np.minimum(Y, 1.0-Y))
        Dx = X[:, None]-X[None, :]
        Dy = Y[:, None]-Y[None, :]
        D = np.sqrt(Dx*Dx+Dy*Dy)
        S = R[:, None]+R[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            rw = wall / R
            rp = D / S
        np.fill_diagonal(rp, np.inf)
        a = float(np.min(rw)); b = float(np.min(rp))
        alpha = min(1.0, a, b)
        alpha = alpha*(1.0-1e-9)-1e-12
        if not np.finite(alpha) or alpha <= 0:
            alpha = 0.5
        return R*alpha, alpha

    def lp_solve(xc, yc, get_dual=False):
        xc = np.asarray(xc, float); yc = np.asarray(yc, float)
        wall = np.minimum(np.minimum(xc, 1.0-xc), np.minimum(yc, 1.0-yc))
        wall = np.maximum(wall, 0.0)
        dx = xc[I]-xc[J]; dy = yc[I]-yc[J]
        d = np.sqrt(dx*dx+dy*dy)
        bnds = [(0.0, float(w)) for w in wall]
        try:
            res = linprog(cobj, A_ub=A0, b_ub=d, bounds=bnds, method='highs')
        except Exception:
            return None, -1.0, None, None
        if res.status != 0:
            return None, -1.0, None, None
        r = np.maximum(np.asarray(res.x, float), 0.0)
        s = float(np.sum(r))
        yp = None; up = None
        if get_dual:
            try:
                # ineqlin marginals for pair constraints, upper for wall ub
                yp_raw = None; up_raw = None
                if hasattr(res, 'ineqlin') and res.ineqlin is not None:
                    try:
                        yp_raw = np.asarray(res.ineqlin.marginals, float).ravel()
                    except Exception:
                        try:
                            yp_raw = np.asarray(res.ineqlin['marginals'], float).ravel()
                        except Exception:
                            yp_raw = None
                if hasattr(res, 'upper') and res.upper is not None:
                    try:
                        up_raw = np.asarray(res.upper.marginals, float).ravel()
                    except Exception:
                        try:
                            up_raw = np.asarray(res.upper['marginals'], float).ravel()
                        except Exception:
                            up_raw = None
                # fallback dict access
                if yp_raw is None:
                    try:
                        yp_raw = np.asarray(res['ineqlin']['marginals'], float).ravel()
                    except Exception:
                        yp_raw = None
                if up_raw is None:
                    try:
                        up_raw = np.asarray(res['upper']['marginals'], float).ravel()
                    except Exception:
                        up_raw = None
                if yp_raw is not None and yp_raw.size == m:
                    yp = np.abs(yp_raw)
                    yp[~np.isfinite(yp)] = 0.0
                if up_raw is not None and up_raw.size == n:
                    up = np.abs(up_raw)
                    up[~np.isfinite(up)] = 0.0
            except Exception:
                yp = None; up = None
        return r, s, yp, up

    # fallback grid best
    k = int(np.ceil(np.sqrt(n)))
    r0 = 0.5/k
    fb = []
    for ix in range(k):
        for jy in range(k):
            if len(fb) < n:
                fb.append([(2*ix+1)*r0, (2*jy+1)*r0])
    fb = np.array(fb, float)
    best_sum = -1.0
    best_xc = fb[:,0].copy(); best_yc = fb[:,1].copy(); best_r = None
    try:
        rfb, sfb, _, _ = lp_solve(best_xc, best_yc, False)
        if rfb is not None:
            best_r = rfb; best_sum = sfb
    except Exception:
        pass
    if best_r is None:
        best_r = np.full(n, r0*0.9)
        best_sum = float(np.sum(best_r))
    pool = []
    def push_pool(s, xc, yc):
        pool.append((s, xc.copy(), yc.copy()))
        pool.sort(key=lambda t: -t[0])
        del pool[5:]

    try:
        push_pool(best_sum, best_xc, best_yc)
    except Exception:
        pass

    # ---------- Phase A: annealed penalized joint search ----------
    if has_opt:
        bounds78 = [(0.0,1.0)]*(2*n) + [(0.02,0.20)]*n
        def pen_fun(v, W):
            X = v[0:n]; Y = v[n:2*n]; R = v[2*n:3*n]
            gx = np.zeros(n); gy = np.zeros(n); grpen = np.zeros(n)
            pen = 0.0
            v1 = R-X
            mm = v1>0
            if np.any(mm):
                pen += np.sum(v1[mm]**2); gx[mm] += -2.0*v1[mm]; grpen[mm] += 2.0*v1[mm]
            v2 = X+R-1.0
            mm = v2>0
            if np.any(mm):
                pen += np.sum(v2[mm]**2); gx[mm] += 2.0*v2[mm]; grpen[mm] += 2.0*v2[mm]
            v3 = R-Y
            mm = v3>0
            if np.any(mm):
                pen += np.sum(v3[mm]**2); gy[mm] += -2.0*v3[mm]; grpen[mm] += 2.0*v3[mm]
            v4 = Y+R-1.0
            mm = v4>0
            if np.any(mm):
                pen += np.sum(v4[mm]**2); gy[mm] += 2.0*v4[mm]; grpen[mm] += 2.0*v4[mm]
            Dx = X[:,None]-X[None,:]; Dy = Y[:,None]-Y[None,:]
            D = np.sqrt(Dx*Dx+Dy*Dy+1e-20)
            S = R[:,None]+R[None,:]
            V = S-D
            np.fill_diagonal(V, 0.0)
            np.fill_diagonal(D, 1.0)
            Mask = V>0
            if np.any(Mask):
                pen += 0.5*np.sum(V[Mask]**2)
                Q = np.zeros((n,n)); Q[Mask] = -2.0*V[Mask]/D[Mask]
                gx += np.sum(Q*Dx, axis=1); gy += np.sum(Q*Dy, axis=1)
                G = np.zeros((n,n)); G[Mask] = 2.0*V[Mask]
                grpen += np.sum(G, axis=1)
            loss = -np.sum(R) + W*pen
            g = np.empty_like(v)
            g[0:n] = W*gx; g[n:2*n] = W*gy; g[2*n:3*n] = -1.0 + W*grpen
            return loss, g
        Ws = [5e2, 5e3, 5e4]
        restart = 0
        while time.time()-t0 < 58.0 and restart < 40:
            if restart == 0:
                X0 = np.zeros(n); Y0 = np.zeros(n)
                idx=0
                for ii in range(5):
                    for jj in range(5):
                        X0[idx]=(ii+0.5)/5.0+rng.normal(0,0.008)
                        Y0[idx]=(jj+0.5)/5.0+rng.normal(0,0.008)
                        idx+=1
                X0[25]=0.88+rng.normal(0,0.01); Y0[25]=0.12+rng.normal(0,0.01)
                X0=np.clip(X0,0.05,0.95); Y0=np.clip(Y0,0.05,0.95)
                R0=0.085+0.02*rng.random(n)
            elif restart == 1:
                X0=rng.random(n)*0.9+0.05; Y0=rng.random(n)*0.9+0.05; R0=0.08+0.03*rng.random(n)
            else:
                if restart%2==0:
                    perm=rng.permutation(n)
                    gx_=np.zeros(n); gy_=np.zeros(n)
                    # hex-ish base
                    cnt=0
                    for ix in range(6):
                        for jy in range(5):
                            if cnt>=n: break
                            gx_[cnt]=(ix+0.5)/6.0+(0.04 if jy%2 else 0.0)
                            gy_[cnt]=(jy+0.5)/5.0
                            cnt+=1
                    X0=gx_+rng.normal(0,0.015,n); Y0=gy_+rng.normal(0,0.015,n)
                    X0=np.clip(X0,0.03,0.97); Y0=np.clip(Y0,0.03,0.97)
                    R0=0.07+0.05*rng.random(n)
                else:
                    X0=rng.random(n)*0.9+0.05; Y0=rng.random(n)*0.9+0.05; R0=0.06+0.06*rng.random(n)
            v=np.concatenate([X0,Y0,R0])
            for W in Ws:
                if time.time()-t0>58.0: break
                try:
                    res= minimize(pen_fun, v, args=(W,), jac=True, method='L-BFGS-B', bounds=bounds78, options={'maxiter':180,'ftol':1e-9,'gtol':1e-6})
                    v=res.x
                except Exception:
                    break
            X=np.clip(v[0:n],0.0,1.0); Y=np.clip(v[n:2*n],0.0,1.0)
            try:
                rlp,slp,_,_=lp_solve(X,Y,False)
                if rlp is not None and slp>best_sum:
                    best_sum=slp; best_xc=X.copy(); best_yc=Y.copy(); best_r=rlp.copy()
                if rlp is not None:
                    push_pool(slp,X,Y)
                # uniform-scale safety candidate
                R_=np.clip(v[2*n:3*n],0.01,0.25)
                Rs,_=uniform_scale(X,Y,R_)
                ss=float(np.sum(Rs))
                if ss>best_sum:
                    best_sum=ss; best_xc=X.copy(); best_yc=Y.copy(); best_r=Rs.copy()
            except Exception:
                pass
            restart+=1

    # symmetries + perturbations of pool
    try:
        cand_centres=[]
        for s,xc,yc in list(pool):
            cand_centres.append((xc,yc))
            for tr in range(8):
                xt=xc.copy(); yt=yc.copy()
                if tr==1: xt=1-xc
                elif tr==2: yt=1-yc
                elif tr==3: xt=1-xc; yt=1-yc
                elif tr==4: xt=yc; yt=xc
                elif tr==5: xt=1-yc; yt=xc
                elif tr==6: xt=yc; yt=1-xc
                elif tr==7: xt=1-yc; yt=1-xc
                cand_centres.append((xt,yt))
        # evaluate symmetries quickly
        for xc,yc in cand_centres:
            if time.time()-t0>70.0: break
            try:
                r,s,_,_=lp_solve(xc,yc,False)
                if r is not None and s>best_sum:
                    best_sum=s; best_xc=xc.copy(); best_yc=yc.copy(); best_r=r.copy()
                    push_pool(s,xc,yc)
            except Exception:
                pass
    except Exception:
        pass

    # ---------- Phase B: LP-dual envelope gradient centre ascent ----------
    if has_opt:
        def centre_fg(z):
            xc=z[0:n]; yc=z[n:2*n]
            r,s,yp,up=lp_solve(xc,yc,True)
            if r is None:
                return 1e3, np.zeros_like(z)
            f=-s
            gx=np.zeros(n); gy=np.zeros(n)
            try:
                if yp is not None:
                    dx=xc[I]-xc[J]; dy=yc[I]-yc[J]
                    d=np.sqrt(dx*dx+dy*dy)+1e-16
                    # threshold tiny duals
                    msk=yp>1e-10
                    if np.any(msk):
                        w=yp[msk]/d[msk]
                        cx=dx[msk]*w; cy=dy[msk]*w
                        np.add.at(gx, I[msk], cx); np.add.at(gx, J[msk], -cx)
                        np.add.at(gy, I[msk], cy); np.add.at(gy, J[msk], -cy)
                if up is not None:
                    for i in range(n):
                        u=up[i]
                        if u<1e-10: continue
                        xi=xc[i]; yi=yc[i]
                        cands=[xi,1.0-xi,yi,1.0-yi]
                        mn=min(cands)
                        cnt=sum(1 for c in cands if abs(c-mn)<1e-9)
                        if abs(xi-mn)<1e-9:
                            gx[i]+=u*(1.0/cnt if xi<=1.0-xi+1e-12 else 0.0)
                        if abs((1.0-xi)-mn)<1e-9:
                            gx[i]+=u*(-1.0/cnt)
                        if abs(yi-mn)<1e-9:
                            gy[i]+=u*(1.0/cnt if yi<=1.0-yi+1e-12 else 0.0)
                        if abs((1.0-yi)-mn)<1e-9:
                            gy[i]+=u*(-1.0/cnt)
                gx=-gx; gy=-gy  # because f=-sum
            except Exception:
                gx=np.zeros(n); gy=np.zeros(n)
            g=np.concatenate([gx,gy])
            #clip huge
            g=np.clip(g,-5.0,5.0)
            return f,g
        bnds_c=[(0.015,0.985)]*(2*n)
        # polish top few
        tops=sorted(pool,key=lambda t:-t[0])[:3]
        # ensure best included
        tops.append((best_sum,best_xc.copy(),best_yc.copy()))
        seen=set()
        for s,xc0,yc0 in tops:
            if time.time()-t0>102.0: break
            # small random kicks: try 2 kicks
            for kick in range(3):
                if time.time()-t0>102.0: break
                if kick==0:
                    zx=np.concatenate([xc0,yc0])
                else:
                    zx=np.concatenate([xc0+rng.normal(0,0.003,n),yc0+rng.normal(0,0.003,n)])
                    zx[0:n]=np.clip(zx[0:n],0.015,0.985); zx[n:]=np.clip(zx[n:],0.015,0.985)
                try:
                    res=minimize(centre_fg,zx,jac=True,method='L-BFGS-B',bounds=bnds_c,options={'maxiter':70,'ftol':1e-12,'gtol':1e-8})
                    z=res.x
                    xc=z[0:n]; yc=z[n:2*n]
                    r,ss,_,_=lp_solve(xc,yc,False)
                    if r is not None and ss>best_sum+1e-12:
                        best_sum=ss; best_xc=xc.copy(); best_yc=yc.copy(); best_r=r.copy()
                        push_pool(ss,xc,yc)
                    # hill-climb tiny steps
                    for _ in range(30):
                        if time.time()-t0>102.0: break
                        prop_x=xc+rng.normal(0,0.0025,n); prop_y=yc+rng.normal(0,0.0025,n)
                        prop_x=np.clip(prop_x,0.015,0.985); prop_y=np.clip(prop_y,0.015,0.985)
                        r2,s2,_,_=lp_solve(prop_x,prop_y,False)
                        if r2 is not None and s2>ss+1e-10:
                            xc,yc,ss,r=prop_x,prop_y,s2,r2
                            if ss>best_sum:
                                best_sum=ss; best_xc=xc.copy(); best_yc=yc.copy(); best_r=r.copy()
                except Exception:
                    continue

    # ---------- Phase C: SLSQP jamming over 78 vars ----------
    if has_opt and time.time()-t0 < 118.0:
        try:
            Xb=np.clip(np.asarray(best_xc,float),0.0,1.0)
            Yb=np.clip(np.asarray(best_yc,float),0.0,1.0)
            Rb=np.maximum(np.asarray(best_r,float),1e-6)
            v0=np.concatenate([Xb,Yb,Rb])
            bnds78=[(0.0,1.0)]*(2*n)+[(0.005,0.30)]*n
            def obj(v):
                return -np.sum(v[2*n:3*n])
            def obj_jac(v):
                g=np.zeros_like(v); g[2*n:3*n]=-1.0
                return g
            def wall_fun(v):
                X=v[0:n]; Y=v[n:2*n]; R=v[2*n:3*n]
                return np.concatenate([X-R, 1.0-X-R, Y-R, 1.0-Y-R])
            def wall_jac(v):
                J=np.zeros((4*n,3*n))
                for i in range(n):
                    J[i,i]=1.0; J[i,2*n+i]=-1.0
                    J[n+i,i]=-1.0; J[n+i,2*n+i]=-1.0
                    J[2*n+i,n+i]=1.0; J[2*n+i,2*n+i]=-1.0
                    J[3*n+i,n+i]=-1.0; J[3*n+i,2*n+i]=-1.0
                return J
            def pair_fun(v):
                X=v[0:n]; Y=v[n:2*n]; R=v[2*n:3*n]
                dx=X[I]-X[J]; dy=Y[I]-Y[J]
                d=np.sqrt(dx*dx+dy*dy+1e-20)
                return d-R[I]-R[J]
            def pair_jac(v):
                X=v[0:n]; Y=v[n:2*n]
                dx=X[I]-X[J]; dy=Y[I]-Y[J]
                d=np.sqrt(dx*dx+dy*dy+1e-20)
                J=np.zeros((m,3*n))
                ux=dx/d; uy=dy/d
                J[np.arange(m),I]=ux; J[np.arange(m),J]=-ux
                J[np.arange(m),n+I]=uy; J[np.arange(m),n+J]=-uy
                J[np.arange(m),2*n+I]=-1.0; J[np.arange(m),2*n+J]=-1.0
                return J
            cons=[{'type':'ineq','fun':wall_fun,'jac':wall_jac},{'type':'ineq','fun':pair_fun,'jac':pair_jac}]
            try:
                res=minimize(obj,v0,jac=obj_jac,method='SLSQP',bounds=bnds78,constraints=cons,options={'maxiter':250,'ftol':1e-12,'eps':1e-9,'disp':False})
                v=res.x
                X=v[0:n]; Y=v[n:2*n]; R=v[2*n:3*n]
                # verify then shrink
                Rs,_=uniform_scale(np.clip(X,0,1),np.clip(Y,0,1),np.maximum(R,1e-9))
                # check strict feasibility quickly
                wall=np.minimum(np.minimum(X,1-X),np.minimum(Y,1-Y))-Rs
                Dx=X[:,None]-X[None,:]; Dy=Y[:,None]-Y[None,:]
                D=np.sqrt(Dx*Dx+Dy*Dy); S=Rs[:,None]+Rs[None,:]
                np.fill_diagonal(D,np.inf); np.fill_diagonal(S,0.0)
                if np.min(wall)>=-1e-9 and np.min(D-S)>=-1e-9:
                    ss=float(np.sum(Rs))
                    if ss>best_sum-1e-7:  # accept SLSQP even if tiny loss (jammed)
                        # only accept if better or close; keep best if worse
                        if ss>best_sum:
                            best_sum=ss; best_xc=X.copy(); best_yc=Y.copy(); best_r=Rs.copy()
                        else:
                            # keep SLSQP if it is feasible and within 1e-6? prefer best
                            pass
            except Exception:
                pass
        except Exception:
            pass

    # final safety shrink
    try:
        Xf=np.clip(np.asarray(best_xc,float),0.0,1.0)
        Yf=np.clip(np.asarray(best_yc,float),0.0,1.0)
        Rf=np.maximum(np.asarray(best_r,float),1e-9)
        Rf,_=uniform_scale(Xf,Yf,Rf)
        best_arr=np.column_stack([Xf,Yf,Rf])
    except Exception:
        best_arr=np.column_stack([best_xc,best_yc,best_r])
    return np.asarray(best_arr,float)
