import numpy as np
import time

def solve():
    import numpy as np
    import time
    from scipy.optimize import minimize, linprog
    n = 26
    t0 = time.time()
    DEADLINE = 122.0
    rng = np.random.default_rng(7)

    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    m = len(pairs)
    A_pair = np.zeros((m, n), dtype=float)
    for k, (i, j) in enumerate(pairs):
        A_pair[k, i] = 1.0
        A_pair[k, j] = 1.0
    c_obj = -np.ones(n, dtype=float)
    pi = np.array([p[0] for p in pairs], dtype=int)
    pj = np.array([p[1] for p in pairs], dtype=int)

    def uniform_scale(X, Y, R):
        R = np.maximum(R, 1e-12)
        wall = np.minimum(np.minimum(X, 1-X), np.minimum(Y, 1-Y))
        Dx = X[:, None]-X[None, :]
        Dy = Y[:, None]-Y[None, :]
        D = np.sqrt(Dx*Dx+Dy*Dy)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_w = wall / R
            ratio_p = D / (R[:, None]+R[None, :])
        np.fill_diagonal(ratio_p, np.inf)
        a = float(np.min(ratio_w)); b = float(np.min(ratio_p))
        alpha = min(1.0, a, b)
        alpha = alpha*(1.0-1e-7)-1e-12
        if not np.isfinite(alpha) or alpha <= 0:
            alpha = 0.5
        return R*alpha, alpha

    k5 = 5
    r0 = 0.5/k5*0.999
    pts = []
    for i in range(k5):
        for j in range(k5):
            pts.append([(2*i+1)*r0, (2*j+1)*r0])
    pts = np.array(pts)
    Xf0 = np.concatenate([pts[:,0], np.array([0.5])])
    Yf0 = np.concatenate([pts[:,1], np.array([0.5])])
    Rf0 = np.full(n, r0*0.9)
    Rs0,_ = uniform_scale(Xf0, Yf0, Rf0)
    best_x = Xf0.copy(); best_y = Yf0.copy(); best_r = Rs0.copy()
    best_sum = float(np.sum(Rs0))
    best_arr = np.column_stack([best_x, best_y, best_r])

    def lp_opt(x, y, eps=1e-9):
        wall = np.minimum(np.minimum(x, 1.0-x), np.minimum(y, 1.0-y))
        if np.any(wall <= 1e-9):
            return None, -1.0
        dx = x[pi]-x[pj]
        dy = y[pi]-y[pj]
        b_pair = np.sqrt(dx*dx+dy*dy) - eps
        if np.any(b_pair <= 1e-12):
            return None, -1.0
        bounds = []
        for w in wall:
            ub = float(w - eps*0.5)
            if ub <= 0:
                return None, -1.0
            bounds.append((0.0, ub))
        try:
            res = linprog(c_obj, A_ub=A_pair, b_ub=b_pair, bounds=bounds, method='highs')
        except Exception:
            return None, -1.0
        if not res.success:
            return None, -1.0
        r = np.array(res.x, dtype=float)*(1.0-5e-9)
        r = np.maximum(r, 0.0)
        return r, float(np.sum(r))

    def gen_grid5(jit=0.01):
        bx = np.zeros(n); by = np.zeros(n)
        idx=0
        for i in range(5):
            for j in range(5):
                if idx<25:
                    bx[idx]=(i+0.5)/5.0+rng.normal(0,jit)
                    by[idx]=(j+0.5)/5.0+rng.normal(0,jit)
                    idx+=1
        bx[25]=rng.random()*0.9+0.05
        by[25]=rng.random()*0.9+0.05
        return np.clip(bx,0.03,0.97), np.clip(by,0.03,0.97)
    def gen_rand():
        return rng.random(n)*0.94+0.03, rng.random(n)*0.94+0.03

    # Phase 1: diverse LP pool
    for trial in range(70):
        if time.time()-t0 > DEADLINE-95:
            break
        if trial % 2 == 0:
            x0,y0 = gen_grid5(0.012)
        else:
            x0,y0 = gen_rand()
        r,s = lp_opt(x0,y0)
        if r is not None and s > best_sum:
            best_sum=s; best_x=x0.copy(); best_y=y0.copy(); best_r=r.copy()
            best_arr=np.column_stack([best_x,best_y,best_r])

    # penalty fun
    def make_pen():
        def fun(v,W):
            X=v[0:n]; Y=v[n:2*n]; R=v[2*n:3*n]
            gx=np.zeros(n); gy=np.zeros(n); pgr=np.zeros(n)
            pen=0.0
            v1=R-X; m1=v1>0
            if np.any(m1):
                pen+=np.sum(v1[m1]**2); gx[m1]+=-2*v1[m1]; pgr[m1]+=2*v1[m1]
            v2=X+R-1.0; m2=v2>0
            if np.any(m2):
                pen+=np.sum(v2[m2]**2); gx[m2]+=2*v2[m2]; pgr[m2]+=2*v2[m2]
            v3=R-Y; m3=v3>0
            if np.any(m3):
                pen+=np.sum(v3[m3]**2); gy[m3]+=-2*v3[m3]; pgr[m3]+=2*v3[m3]
            v4=Y+R-1.0; m4=v4>0
            if np.any(m4):
                pen+=np.sum(v4[m4]**2); gy[m4]+=2*v4[m4]; pgr[m4]+=2*v4[m4]
            Dx=X[:,None]-X[None,:]; Dy=Y[:,None]-Y[None,:]
            D=np.sqrt(Dx*Dx+Dy*Dy+1e-20)
            S=R[:,None]+R[None,:]
            V=S-D; np.fill_diagonal(V,0.0); np.fill_diagonal(D,1.0)
            Mask=V>0
            if np.any(Mask):
                pen+=0.5*np.sum(V[Mask]**2)
                Q=np.zeros((n,n)); Q[Mask]=-2.0*V[Mask]/D[Mask]
                gx+=np.sum(Q*Dx,axis=1); gy+=np.sum(Q*Dy,axis=1)
                G=np.zeros((n,n)); G[Mask]=2.0*V[Mask]
                pgr+=np.sum(G,axis=1)
            loss=-np.sum(R)+W*pen
            g=np.empty_like(v)
            g[0:n]=W*gx; g[n:2*n]=W*gy; g[2*n:3*n]=-1.0+W*pgr
            return loss,g
        return fun
    penfun=make_pen()
    boundsJR=[(0.0,1.0)]*(2*n)+[(0.012,0.20)]*n

    def short_refine(X0,Y0,R0,Ws,maxiter):
        v=np.concatenate([X0,Y0,R0])
        for W in Ws:
            if time.time()-t0>DEADLINE-4:
                break
            try:
                res=minimize(penfun,v,args=(W,),jac=True,method='L-BFGS-B',bounds=boundsJR,options={'maxiter':maxiter,'ftol':1e-9,'gtol':1e-5})
                v=res.x
            except Exception:
                break
        X=np.clip(v[0:n],0.0,1.0); Y=np.clip(v[n:2*n],0.0,1.0); R=np.clip(v[2*n:3*n],0.005,0.22)
        X=np.clip(X,0.015,0.985); Y=np.clip(Y,0.015,0.985)
        return X,Y,R

    # Phase 2: few joint restarts
    for restart in range(6):
        if time.time()-t0>DEADLINE-75:
            break
        if restart==0:
            X0,Y0=gen_grid5(0.005)
            R0=np.full(n,0.095)
            v=np.concatenate([X0,Y0,R0])
        else:
            v=np.concatenate([rng.random(n)*0.9+0.05, rng.random(n)*0.9+0.05, 0.07+0.04*rng.random(n)])
        for W in (800.0, 12000.0, 90000.0):
            if time.time()-t0>DEADLINE-75:
                break
            try:
                res=minimize(penfun,v,args=(W,),jac=True,method='L-BFGS-B',bounds=boundsJR,options={'maxiter':110,'ftol':1e-9,'gtol':1e-5})
                v=res.x
            except Exception:
                break
        X=np.clip(v[0:n],0.03,0.97); Y=np.clip(v[n:2*n],0.03,0.97)
        r,s=lp_opt(X,Y)
        if r is not None and s>best_sum:
            best_sum=s; best_x=X.copy(); best_y=Y.copy(); best_r=r.copy()
            best_arr=np.column_stack([best_x,best_y,best_r])

    cur_x=best_x.copy(); cur_y=best_y.copy(); cur_r=best_r.copy(); cur_sum=best_sum

    # helpers for surgery
    def contact_counts(X,Y,R,tol=0.0035):
        wall=np.minimum(np.minimum(X,1-X),np.minimum(Y,1-Y))-R
        cc=np.zeros(n,dtype=float)
        cc+=(wall<tol).astype(float)
        Dx=X[:,None]-X[None,:]; Dy=Y[:,None]-Y[None,:]
        D=np.sqrt(Dx*Dx+Dy*Dy)
        S=R[:,None]+R[None,:]
        Gap=D-S
        np.fill_diagonal(Gap,np.inf)
        for i in range(n):
            cc[i]+=np.sum(Gap[i,:]<tol)
        return cc

    _xs201 = np.linspace(0,1,201)
    _ys201 = np.linspace(0,1,201)
    _XX,_YY = np.meshgrid(_xs201,_ys201)
    _PX = _XX.ravel(); _PY = _YY.ravel()
    _wall0 = np.minimum(np.minimum(_PX,1-_PX), np.minimum(_PY,1-_PY))

    def find_voids(X,Y,R,topk=3):
        DX=_PX[:,None]-X[None,:]
        DY=_PY[:,None]-Y[None,:]
        D=np.sqrt(DX*DX+DY*DY)
        gap=D-R[None,:]
        mg=np.min(gap,axis=1)
        clear=np.minimum(_wall0, mg)
        outs=[]
        for _ in range(topk):
            idx=int(np.argmax(clear))
            cl=float(clear[idx])
            px=float(_PX[idx]); py=float(_PY[idx])
            if cl<1e-9:
                # no positive void; return random interior point
                outs.append((rng.random()*0.9+0.05, rng.random()*0.9+0.05, 0.0))
                clear[idx]=-1
                continue
            outs.append((px,py,cl))
            dd=np.sqrt((_PX-px)**2+(_PY-py)**2)
            clear[dd < max(cl*1.0, 0.02)] = -1.0
        return outs

    # Phase 3: topology surgery + basin hopping with SA
    T0=0.004; T1=0.00005
    t_phase3=time.time()
    budget=max(1.0, DEADLINE-4-(t_phase3-t0))
    stall=0
    it=0
    sigmas=[0.02,0.012,0.007,0.003,0.0012]
    while time.time()-t0 < DEADLINE-3:
        frac=(time.time()-t_phase3)/budget
        if frac>1: frac=1
        Temp=T0*((T1/T0)**frac)
        it+=1
        if rng.random()<0.55:
            # surgery: k=1-3
            u=rng.random()
            if u<0.5: k=1
            elif u<0.8: k=2
            else: k=3
            try:
                voids=find_voids(cur_x,cur_y,cur_r,topk=k)
            except Exception:
                stall+=1
                continue
            if rng.random()<0.5:
                order=np.argsort(cur_r)
            else:
                try:
                    cc=contact_counts(cur_x,cur_y,cur_r)
                except Exception:
                    cc=np.zeros(n)
                order=np.lexsort((cur_r, cc))
            targets=order[:k]
            nx=cur_x.copy(); ny=cur_y.copy()
            for ti,(vx,vy,vc) in zip(targets, voids):
                nx[ti]=vx+rng.normal(0,0.002)
                ny[ti]=vy+rng.normal(0,0.002)
            nx=np.clip(nx,0.02,0.98); ny=np.clip(ny,0.02,0.98)
            r0_,s0_=lp_opt(nx,ny)
            if r0_ is None:
                stall+=1
                # try single random void move as fallback
                continue
            # short penalty repair only if not terrible
            if s0_ > cur_sum-0.06 or rng.random()<0.3:
                X1,Y1,R1=short_refine(nx,ny,r0_,(6000.0,60000.0),60)
                r1,s1=lp_opt(X1,Y1)
                if r1 is None:
                    X1,Y1,R1,s1=nx,ny,r0_,s0_
                    r1=r0_
                else:
                    pass
            else:
                X1,Y1,R1,s1,r1=nx,ny,r0_,s0_,r0_
            delta=s1-cur_sum
            if delta>0 or rng.random()<np.exp(delta/max(Temp,1e-9)):
                cur_x,cur_y,cur_r,cur_sum=X1,Y1,r1,s1
                if s1>best_sum+1e-12:
                    best_sum=s1; best_x=X1.copy(); best_y=Y1.copy(); best_r=r1.copy()
                    best_arr=np.column_stack([best_x,best_y,best_r])
                    stall=0
                else:
                    stall+=1
            else:
                stall+=1
        else:
            sigma=sigmas[it%len(sigmas)]
            if rng.random()<0.8:
                idxs=[int(rng.integers(0,n))]
            else:
                idxs=list(rng.choice(n,size=2,replace=False))
            nx=cur_x.copy(); ny=cur_y.copy()
            for ii in idxs:
                nx[ii]+=rng.normal(0,sigma)
                ny[ii]+=rng.normal(0,sigma)
            nx=np.clip(nx,0.02,0.98); ny=np.clip(ny,0.02,0.98)
            r,s=lp_opt(nx,ny)
            if r is None:
                stall+=1
            else:
                delta=s-cur_sum
                if delta>1e-12 or rng.random()<np.exp(delta/max(Temp,1e-9)):
                    cur_x,cur_y,cur_r,cur_sum=nx,ny,r,s
                    if s>best_sum+1e-12:
                        best_sum=s; best_x=nx.copy(); best_y=ny.copy(); best_r=r.copy()
                        best_arr=np.column_stack([best_x,best_y,best_r])
                        stall=0
                    else:
                        stall+=1 if delta<=0 else 0
                else:
                    stall+=1
        if stall>300:
            # restore elite with kick
            nx=best_x+rng.normal(0,0.008,n)
            ny=best_y+rng.normal(0,0.008,n)
            nx=np.clip(nx,0.02,0.98); ny=np.clip(ny,0.02,0.98)
            r,s=lp_opt(nx,ny)
            if r is not None:
                cur_x,cur_y,cur_r,cur_sum=nx,ny,r,s
                if s>best_sum:
                    best_sum=s; best_x=nx.copy(); best_y=ny.copy(); best_r=r.copy()
                    best_arr=np.column_stack([best_x,best_y,best_r])
            else:
                cur_x,cur_y,cur_r,cur_sum=best_x.copy(),best_y.copy(),best_r.copy(),best_sum
            stall=0
        if it%350==0 and time.time()-t0 < DEADLINE-8:
            # elite polish: tiny all-point jitter hill climb
            nx=best_x+rng.normal(0,0.0012,n)
            ny=best_y+rng.normal(0,0.0012,n)
            nx=np.clip(nx,0.02,0.98); ny=np.clip(ny,0.02,0.98)
            r,s=lp_opt(nx,ny)
            if r is not None and s>best_sum:
                best_sum=s; best_x=nx.copy(); best_y=ny.copy(); best_r=r.copy()
                best_arr=np.column_stack([best_x,best_y,best_r])
                cur_x,cur_y,cur_r,cur_sum=nx.copy(),ny.copy(),r.copy(),s

    try:
        Xf=best_arr[:,0]; Yf=best_arr[:,1]; Rf=best_arr[:,2]
        Rfs,_=uniform_scale(Xf,Yf,Rf)
        best_arr=np.column_stack([Xf,Yf,Rfs])
    except Exception:
        pass
    return best_arr
