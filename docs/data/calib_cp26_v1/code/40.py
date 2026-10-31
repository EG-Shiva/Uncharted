import numpy as np
import time

def solve():
    import numpy as np
    import time
    from scipy.optimize import minimize, linprog
    N = 26
    t0 = time.time()
    BUDGET = 126.0
    rng = np.random.RandomState(123)
    PAIRS = [(i, j) for i in range(N) for j in range(i+1, N)]

    # ---------- LP for fixed centres ----------
    def lp_solve(x, y):
        wall = np.minimum(np.minimum(x, 1.0-x), np.minimum(y, 1.0-y))
        if np.min(wall) <= 1e-12:
            return None, -1.0
        Dx = x[:, None]-x[None, :]
        Dy = y[:, None]-y[None, :]
        D = np.sqrt(Dx*Dx+Dy*Dy)
        # select relevant pairs
        rows = []
        for (i, j) in PAIRS:
            dij = D[i, j]
            if dij >= wall[i]+wall[j]-1e-12:
                continue
            rows.append((i, j, dij))
        if len(rows)==0:
            r = wall.copy()
            return r, float(np.sum(r))
        m = len(rows)
        A_ub = np.zeros((m, N))
        b_ub = np.zeros(m)
        for k,(i,j,dij) in enumerate(rows):
            A_ub[k,i]=1.0; A_ub[k,j]=1.0; b_ub[k]=dij
        c = -np.ones(N)
        bounds = [(0.0, float(wall[i])) for i in range(N)]
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        except Exception:
            return None, -1.0
        if not res.success:
            return None, -1.0
        r = np.maximum(res.x, 0.0)
        return r, float(np.sum(r))

    # ---------- LP value + dual envelope gradient ----------
    def lp_val_grad(x, y):
        wall = np.minimum(np.minimum(x, 1.0-x), np.minimum(y, 1.0-y))
        if np.min(wall) <= 1e-12:
            return -1.0, np.zeros(2*N), None
        Dx = x[:, None]-x[None, :]
        Dy = y[:, None]-y[None, :]
        D = np.sqrt(Dx*Dx+Dy*Dy)
        np.fill_diagonal(D, 1.0)
        r, s = lp_solve(x, y)
        if r is None:
            return -1.0, np.zeros(2*N), None
        # find tight constraints with adaptive tol
        tol = 1e-7
        lam = None
        A_t = None; b_t = None; tpair_idx=None; twall_idx=None
        for attempt in range(3):
            tpairs=[]
            for (i,j) in PAIRS:
                if r[i]+r[j] >= D[i,j]-tol:
                    # only consider pairs that were in LP (non-redundant) or close
                    # include all close pairs
                    tpairs.append((i,j))
            twalls=[]
            for i in range(N):
                if r[i] >= wall[i]-tol:
                    twalls.append(i)
            M = len(tpairs)+len(twalls)
            if M >= N:
                A_t = np.zeros((M,N))
                b_t = np.zeros(M)
                for k,(i,j) in enumerate(tpairs):
                    A_t[k,i]=1.0; A_t[k,j]=1.0; b_t[k]=D[i,j]
                for k2,i in enumerate(twalls):
                    k = len(tpairs)+k2
                    A_t[k,i]=1.0; b_t[k]=wall[i]
                # dual restricted: min b^T lam s.t. A^T lam = 1, lam>=0
                c_d = b_t
                A_eq = A_t.T
                b_eq = np.ones(N)
                bounds_d = [(0.0, None)]*M
                try:
                    resd = linprog(c_d, A_eq=A_eq, b_eq=b_eq, bounds=bounds_d, method='highs')
                except Exception:
                    resd=None
                if resd is not None and resd.success:
                    lam = resd.x
                    tpair_idx = tpairs
                    twall_idx = twalls
                    break
            tol *= 10.0
        gx = np.zeros(N); gy = np.zeros(N)
        if lam is None:
            return s, np.zeros(2*N), r
        # pair contributions
        for k,(i,j) in enumerate(tpair_idx):
            lk = lam[k]
            if lk <= 1e-12:
                continue
            dij = D[i,j]
            if dij < 1e-12:
                continue
            dx = (x[i]-x[j])/dij * lk
            dy = (y[i]-y[j])/dij * lk
            gx[i]+=dx; gx[j]-=dx
            gy[i]+=dy; gy[j]-=dy
        off = len(tpair_idx)
        for k2,i in enumerate(twalls if tpair_idx is not None else []):
            lk = lam[off+k2]
            if lk <= 1e-12:
                continue
            # derivative of wall wrt x,y
            xi=x[i]; yi=y[i]
            cands = [xi, 1.0-xi, yi, 1.0-yi]
            m0 = min(cands)
            # if tie, split equally among minima
            ties=[]
            if abs(xi-m0)<1e-12: ties.append('x+')
            if abs(1.0-xi-m0)<1e-12: ties.append('x-')
            if abs(yi-m0)<1e-12: ties.append('y+')
            if abs(1.0-yi-m0)<1e-12: ties.append('y-')
            # distribute
            for tt in ties:
                w = lk/len(ties)
                if tt=='x+': gx[i]+=w
                elif tt=='x-': gx[i]-=w
                elif tt=='y+': gy[i]+=w
                elif tt=='y-': gy[i]-=w
        g = np.concatenate([gx,gy])
        # clip huge grads
        g = np.clip(g, -10.0, 10.0)
        return s, g, r

    best_sum=-1.0; best_x=None; best_y=None; best_r=None
    cands=[]  # store top centres
    def consider(x,y,r,s):
        nonlocal best_sum,best_x,best_y,best_r
        if s>best_sum+1e-12:
            best_sum=s; best_x=x.copy(); best_y=y.copy(); best_r=r.copy()
        # keep pool
        cands.append((s,x.copy(),y.copy()))
    def time_left():
        return BUDGET-(time.time()-t0)

    # ---------- Phase A: fast joint penalty to get basins ----------
    bounds_box=[(0.0,1.0)]*(2*N)+[(0.008,0.28)]*N
    PHASE_A_END = t0+45.0
    restart=0
    while time.time()<PHASE_A_END and restart<60:
        u0=rng.rand(N); v0=rng.rand(N)
        if restart%3==1:
            # jittered grid 6x5 (30 pts, pick 26)
            gx=np.array([(ix+0.5)/6.0 for ix in range(6) for _ in range(5)])
            gy=np.array([(jy+0.5)/5.0 for _ in range(6) for jy in range(5)])
            perm=rng.permutation(len(gx))[:N]
            u0=gx[perm]+(rng.rand(N)-0.5)*0.06
            v0=gy[perm]+(rng.rand(N)-0.5)*0.06
            u0=np.clip(u0,0,1); v0=np.clip(v0,0,1)
        r0=0.04+0.08*rng.rand(N)
        z=np.concatenate([u0,v0,r0])
        for W in (50.0, 500.0, 5000.0):
            if time.time()>=PHASE_A_END: break
            def fun(zz,W=W):
                u=zz[:N]; vv=zz[N:2*N]; rr=zz[2*N:]
                om=1.0-2.0*rr
                x=rr+om*u; y=rr+om*vv
                Dx=x[:,None]-x[None,:]; Dy=y[:,None]-y[None,:]
                D=np.sqrt(Dx*Dx+Dy*Dy+1e-16)
                np.fill_diagonal(D,1.0)
                E=rr[:,None]+rr[None,:]-D
                np.fill_diagonal(E,0.0)
                Ep=np.maximum(E,0.0)
                Pen=0.5*np.sum(Ep*Ep)
                F=-np.sum(rr)+W*Pen
                Inv=1.0/D; np.fill_diagonal(Inv,0.0)
                Wm=Ep*Inv
                Gx=-2.0*np.sum(Wm*Dx,axis=1)
                Gy=-2.0*np.sum(Wm*Dy,axis=1)
                sEp=np.sum(Ep,axis=1)
                dFdu=W*Gx*om; dFdv=W*Gy*om
                dFdr=-np.ones(N)+W*(2.0*sEp+Gx*(1.0-2.0*u)+Gy*(1.0-2.0*vv))
                return F,np.concatenate([dFdu,dFdv,dFdr])
            try:
                res=minimize(fun,z,method='L-BFGS-B',jac=True,bounds=bounds_box,options={'maxiter':120,'ftol':1e-9,'gtol':1e-5})
                z=res.x
            except Exception:
                break
        u=z[:N]; vv=z[N:2*N]; rr=z[2*N:]
        x=rr+(1.0-2.0*rr)*u; y=rr+(1.0-2.0*rr)*vv
        x=np.clip(x,0.005,0.995); y=np.clip(y,0.005,0.995)
        r,s=lp_solve(x,y)
        if r is not None:
            consider(x,y,r,s)
        restart+=1

    if best_x is None:
        k=int(np.ceil(np.sqrt(N))); rr0=0.5/k
        pts=[]
        for i in range(k):
            for j in range(k):
                if len(pts)<N: pts.append([(2*i+1)*rr0,(2*j+1)*rr0,rr0*0.999999])
        return np.array(pts,float)

    # build seed list: best + symmetries + perturbations + top cands
    cands_sorted=sorted(cands,key=lambda t:-t[0])[:8]
    seeds=[]
    bx0,by0=best_x,best_y
    # 8 dihedral transforms of best
    transforms=[
        (bx0,by0),(1-bx0,by0),(bx0,1-by0),(1-bx0,1-by0),
        (by0,bx0),(1-by0,bx0),(by0,1-bx0),(1-by0,1-bx0)
    ]
    for (sx,sy) in transforms:
        seeds.append((sx.copy(),sy.copy()))
    for (_,cx,cy) in cands_sorted[:4]:
        seeds.append((cx,cy))
    # small perturbations of best
    for sig in (0.003,0.008):
        for _ in range(2):
            seeds.append((np.clip(bx0+sig*rng.randn(N),0.005,0.995),
                          np.clip(by0+sig*rng.randn(N),0.005,0.995)))

    # ---------- Phase B: LP-dual-gradient L-BFGS + SLSQP ----------
    bnds_c=[(0.005,0.995)]*(2*N)
    def polish_centres(sx,sy):
        z0=np.concatenate([sx,sy])
        def fg(zz):
            x=zz[:N]; y=zz[N:2*N]
            s,g,_=lp_val_grad(x,y)
            return -s, -g
        try:
            res=minimize(fg,z0,method='L-BFGS-B',jac=True,bounds=bnds_c,
                         options={'maxiter':80,'ftol':1e-12,'gtol':1e-7})
            x=res.x[:N]; y=res.x[N:2*N]
        except Exception:
            x,y=sx,sy
        r,s=lp_solve(x,y)
        if r is None:
            return sx,sy,None,-1
        return x,y,r,s

    def slsqp_full(x0,y0,r0):
        v0=np.concatenate([x0,y0,r0])
        # bounds
        bnds=[(0.0,1.0)]*(2*N)+[(0.001,0.30)]*N
        P=len(PAIRS)
        def obj(v):
            return -np.sum(v[2*N:])
        def obj_jac(v):
            g=np.zeros_like(v); g[2*N:]=-1.0
            return g
        def wall_f(v):
            X=v[:N]; Y=v[N:2*N]; R=v[2*N:]
            return np.concatenate([X-R, 1.0-X-R, Y-R, 1.0-Y-R])
        def wall_j(v):
            J=np.zeros((4*N,3*N))
            for i in range(N):
                J[i,i]=1.0; J[i,2*N+i]=-1.0
                J[N+i,i]=-1.0; J[N+i,2*N+i]=-1.0
                J[2*N+i,N+i]=1.0; J[2*N+i,2*N+i]=-1.0
                J[3*N+i,N+i]=-1.0; J[3*N+i,2*N+i]=-1.0
            return J
        def pair_f(v):
            X=v[:N]; Y=v[N:2*N]; R=v[2*N:]
            out=np.empty(P)
            for k,(i,j) in enumerate(PAIRS):
                dx=X[i]-X[j]; dy=Y[i]-Y[j]
                out[k]=np.sqrt(dx*dx+dy*dy+1e-20)-R[i]-R[j]
            return out
        def pair_j(v):
            X=v[:N]; Y=v[N:2*N]
            J=np.zeros((P,3*N))
            for k,(i,j) in enumerate(PAIRS):
                dx=X[i]-X[j]; dy=Y[i]-Y[j]
                d=np.sqrt(dx*dx+dy*dy+1e-20)
                if d<1e-12: d=1e-12
                ux=dx/d; uy=dy/d
                J[k,i]=ux; J[k,j]=-ux
                J[k,N+i]=uy; J[k,N+j]=-uy
                J[k,2*N+i]=-1.0; J[k,2*N+j]=-1.0
            return J
        cons=[{'type':'ineq','fun':wall_f,'jac':wall_j},
              {'type':'ineq','fun':pair_f,'jac':pair_j}]
        try:
            res=minimize(obj,v0,method='SLSQP',jac=obj_jac,constraints=cons,
                         bounds=bnds,options={'maxiter':200,'ftol':1e-12,'disp':False,'eps':1e-13})
            v=res.x
        except Exception:
            return x0,y0,r0,float(np.sum(r0))
        X=v[:N]; Y=v[N:2*N]; R=np.maximum(v[2*N:],1e-9)
        return X,Y,R,float(np.sum(R))

    # iterate seeds while time permits
    for (sx,sy) in seeds:
        if time_left()<12.0: break
        # centre polish
        px,py,pr,ps=polish_centres(np.clip(sx,0.005,0.995),np.clip(sy,0.005,0.995))
        if pr is not None and ps>0:
            if ps>best_sum+1e-12:
                best_sum=ps; best_x=px.copy(); best_y=py.copy(); best_r=pr.copy()
        if time_left()<10.0: break
        # SLSQP from polished (or from seed LP)
        if pr is not None:
            try:
                qx,qy,qr,qs=slsqp_full(px,py,pr)
            except Exception:
                continue
            # verify feasibility (allow tiny viol, will shrink)
            wall=np.minimum(np.minimum(qx,1-qx),np.minimum(qy,1-qy))
            Dx=qx[:,None]-qx[None,:]; Dy=qy[:,None]-qy[None,:]
            D=np.sqrt(Dx*Dx+Dy*Dy)
            feas=True
            if np.min(wall-qr)<-1e-7: feas=False
            md=1e9
            for (i,j) in PAIRS:
                md=min(md,D[i,j]-qr[i]-qr[j])
            if md<-1e-7: feas=False
            # re-solve LP centres to get clean radii if infeasible slightly
            if not feas:
                qr2,qs2=lp_solve(np.clip(qx,0.005,0.995),np.clip(qy,0.005,0.995))
                if qr2 is not None:
                    qx=np.clip(qx,0.005,0.995); qy=np.clip(qy,0.005,0.995); qr=qr2; qs=qs2
                else:
                    continue
            # consider with exact LP re-eval of centres? keep SLSQP radii if feasible
            # evaluate true feasible sum after shrink
            wall2=np.minimum(np.minimum(qx,1-qx),np.minimum(qy,1-qy))
            fw=np.min(wall2/np.maximum(qr,1e-18))
            fp=1e100
            for (i,j) in PAIRS:
                fp=min(fp,D[i,j]/(qr[i]+qr[j]+1e-30))
            f=min(fw,fp)
            if f>0 and np.isfinite(f):
                # feasible shrunken sum
                qsum=float(np.sum(qr))*min(1.0,f)
                if qsum>best_sum+1e-12:
                    if f<1.0:
                        qr=qr*f
                    best_sum=qsum; best_x=qx.copy(); best_y=qy.copy(); best_r=qr.copy()

    # final safety shrink
    x=best_x; y=best_y; r=best_r
    Dx=x[:,None]-x[None,:]; Dy=y[:,None]-y[None,:]
    D=np.sqrt(Dx*Dx+Dy*Dy)
    wall=np.minimum(np.minimum(x,1-x),np.minimum(y,1-y))
    fw=np.min(wall/np.maximum(r,1e-30))
    fp=1e100
    for (i,j) in PAIRS:
        fp=min(fp,D[i,j]/(r[i]+r[j]+1e-30))
    f=min(fw,fp)
    if f<1.0:
        r=r*f*(1.0-1e-9)
    else:
        r=r*(1.0-1e-9)
    out=np.column_stack([x,y,r])
    return out
