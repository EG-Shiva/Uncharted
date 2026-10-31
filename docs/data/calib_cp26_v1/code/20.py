import numpy as np
import time

def solve():
    n = 26
    t0 = time.time()
    DEADLINE = 124.0

    rng = np.random.default_rng(0)

    # fixed pair matrix
    iu, ju = np.triu_indices(n, 1)
    m = len(iu)
    A0 = np.zeros((m, n), dtype=float)
    ar = np.arange(m)
    A0[ar, iu] = 1.0
    A0[ar, ju] = 1.0
    c_obj = -np.ones(n, dtype=float)

    from scipy.optimize import minimize, linprog

    # ---------- LP evaluation ----------
    def eval_lp(xc, yc, best_ref=-1.0):
        wall = np.minimum(np.minimum(xc, 1.0 - xc), np.minimum(yc, 1.0 - yc))
        # centres must be inside
        if np.min(wall) <= 0:
            return None, -1.0
        sw = float(np.sum(wall))
        if sw <= best_ref:
            return None, -1.0
        dx = xc[iu] - xc[ju]
        dy = yc[iu] - yc[ju]
        dvec = np.sqrt(dx * dx + dy * dy)
        # quick necessary: sum of all pair bounds? skip
        bounds = [(0.0, float(w)) for w in wall]
        try:
            res = linprog(c_obj, A_ub=A0, b_ub=dvec, bounds=bounds, method='highs')
        except Exception:
            return None, -1.0
        if not res.success or res.x is None:
            return None, -1.0
        r = np.array(res.x, dtype=float).flatten()
        r = np.maximum(r, 0.0)
        s = float(np.sum(r))
        if s <= best_ref:
            return r, s
        return r, s

    # ---------- fallback ----------
    k = int(np.ceil(np.sqrt(n)))
    r0f = 0.5 / k
    pts = []
    for i in range(k):
        for j in range(k):
            if len(pts) < n:
                pts.append([(2*i+1)*r0f, (2*j+1)*r0f, r0f*0.999*0.999])
    best_arr = np.array(pts, dtype=float)
    # make strictly feasible by tiny shrink
    best_arr[:,2] *= (1.0 - 1e-7)
    best_sum = float(np.sum(best_arr[:,2]))
    best_x = best_arr[:,0].copy()
    best_y = best_arr[:,1].copy()
    best_r = best_arr[:,2].copy()

    # try to improve fallback via LP immediately
    try:
        rlp, s = eval_lp(best_x, best_y, -1.0)
        if rlp is not None and s > best_sum:
            best_sum = s
            best_r = rlp
    except Exception:
        pass

    # ---------- Phase A: penalized joint L-BFGS for seeds ----------
    def make_fun(nn):
        def fun(v, W):
            X = v[0:nn]; Y = v[nn:2*nn]; R = v[2*nn:3*nn]
            gx = np.zeros(nn); gy = np.zeros(nn)
            gr_pen = np.zeros(nn)
            pen = 0.0
            v1 = R - X
            m1 = v1 > 0
            if np.any(m1):
                pen += np.sum(v1[m1]**2)
                gx[m1] += -2.0*v1[m1]
                gr_pen[m1] += 2.0*v1[m1]
            v2 = X + R - 1.0
            m2 = v2 > 0
            if np.any(m2):
                pen += np.sum(v2[m2]**2)
                gx[m2] += 2.0*v2[m2]
                gr_pen[m2] += 2.0*v2[m2]
            v3 = R - Y
            m3 = v3 > 0
            if np.any(m3):
                pen += np.sum(v3[m3]**2)
                gy[m3] += -2.0*v3[m3]
                gr_pen[m3] += 2.0*v3[m3]
            v4 = Y + R - 1.0
            m4 = v4 > 0
            if np.any(m4):
                pen += np.sum(v4[m4]**2)
                gy[m4] += 2.0*v4[m4]
                gr_pen[m4] += 2.0*v4[m4]
            Dx = X[:,None]-X[None,:]
            Dy = Y[:,None]-Y[None,:]
            D = np.sqrt(Dx*Dx+Dy*Dy+1e-20)
            S = R[:,None]+R[None,:]
            V = S-D
            np.fill_diagonal(V, 0.0)
            np.fill_diagonal(D, 1.0)
            Mask = V>0
            if np.any(Mask):
                pen += 0.5*np.sum(V[Mask]**2)
                Q = np.zeros((nn,nn))
                Q[Mask] = -2.0*V[Mask]/D[Mask]
                gx += np.sum(Q*Dx,axis=1)
                gy += np.sum(Q*Dy,axis=1)
                G = np.zeros((nn,nn))
                G[Mask]=2.0*V[Mask]
                gr_pen += np.sum(G,axis=1)
            loss = -np.sum(R)+W*pen
            g = np.empty_like(v)
            g[0:nn]=W*gx
            g[nn:2*nn]=W*gy
            g[2*nn:3*nn]=-1.0+W*gr_pen
            return loss,g
        return fun
    funJ = make_fun(n)
    boundsJ = [(0.0,1.0)]*(2*n)+[(0.03,0.20)]*n
    Ws = [1e2,1e3,1e4,1e5]

    n_seed_restarts = 12
    for restart in range(n_seed_restarts):
        if time.time()-t0 > 42.0:
            break
        if restart==0:
            X0=np.zeros(n);Y0=np.zeros(n);R0=np.full(n,0.095)
            idx=0
            for i in range(5):
                for j in range(5):
                    if idx<25:
                        X0[idx]=(i+0.5)/5.0+rng.normal(0,0.005)
                        Y0[idx]=(j+0.5)/5.0+rng.normal(0,0.005)
                        idx+=1
            X0[25]=0.9;Y0[25]=0.1
            X0=np.clip(X0,0.05,0.95);Y0=np.clip(Y0,0.05,0.95)
        elif restart==1:
            xs=(np.arange(6)+0.5)/6.0; ys=(np.arange(5)+0.5)/5.0
            xx,yy=np.meshgrid(xs,ys)
            xy=np.column_stack([xx.ravel(),yy.ravel()])
            perm=rng.permutation(len(xy))[:n]
            X0=xy[perm,0]+rng.normal(0,0.005,n)
            Y0=xy[perm,1]+rng.normal(0,0.005,n)
            X0=np.clip(X0,0.05,0.95);Y0=np.clip(Y0,0.05,0.95)
            R0=0.085+0.02*rng.random(n)
        else:
            X0=rng.random(n)*0.9+0.05
            Y0=rng.random(n)*0.9+0.05
            R0=0.085+0.025*rng.random(n)
        v=np.concatenate([X0,Y0,R0])
        for W in Ws:
            if time.time()-t0>42.0:
                break
            try:
                res=minimize(funJ,v,args=(W,),jac=True,method='L-BFGS-B',bounds=boundsJ,
                             options={'maxiter':220,'ftol':1e-9,'gtol':1e-7})
                v=res.x
            except Exception:
                break
        X=np.clip(v[0:n],0.0,1.0);Y=np.clip(v[n:2*n],0.0,1.0)
        rlp,s=eval_lp(X,Y,best_sum-1e-12)
        if rlp is not None and s>best_sum:
            best_sum=s;best_x=X.copy();best_y=Y.copy();best_r=rlp.copy()

    # also try pure random centres via LP (cheap seeds)
    for _ in range(30):
        if time.time()-t0>48.0:
            break
        X0=rng.random(n)*0.9+0.05
        Y0=rng.random(n)*0.9+0.05
        rlp,s=eval_lp(X0,Y0,best_sum-1e-12)
        if rlp is not None and s>best_sum:
            best_sum=s;best_x=X0.copy();best_y=Y0.copy();best_r=rlp.copy()

    # ---------- Phase B: LP-guided hill climbing ----------
    # keep small elite pool
    elite_x=[best_x.copy()]; elite_y=[best_y.copy()]; elite_s=[best_sum]
    # add a couple jittered copies
    for _ in range(2):
        elite_x.append(np.clip(best_x+rng.normal(0,0.01,n),0.001,0.999))
        elite_y.append(np.clip(best_y+rng.normal(0,0.01,n),0.001,0.999))
        elite_s.append(-1.0)
    # evaluate them
    for i in range(1,len(elite_x)):
        _,s=eval_lp(elite_x[i],elite_y[i],-1.0)
        elite_s[i]=s

    # pick best as current
    bi=int(np.argmax(elite_s))
    best_x=elite_x[bi].copy(); best_y=elite_y[bi].copy()
    # re-eval best to get r
    rlp,s=eval_lp(best_x,best_y,-1.0)
    if rlp is not None:
        best_r=rlp; best_sum=s

    it=0
    stagn=0
    while True:
        if time.time()-t0>DEADLINE-4.0:
            break
        it+=1
        # sigma schedule
        if stagn>1500:
            sigma=0.025
        elif stagn>600:
            sigma=0.012
        else:
            # decay from 0.012 to 0.001
            sigma=0.012*np.exp(-it/6000.0)+0.0008
        u=rng.random()
        # choose base: mostly best, sometimes elite
        if u<0.85 or len(elite_x)==1:
            bx=best_x; by=best_y
        else:
            ei=rng.integers(0,len(elite_x))
            bx=elite_x[ei]; by=elite_y[ei]
        if rng.random()<0.72:
            # single circle move
            idx=rng.integers(0,n)
            nx=bx.copy(); ny=by.copy()
            nx[idx]+=rng.normal(0,sigma)
            ny[idx]+=rng.normal(0,sigma)
            nx[idx]=np.clip(nx[idx],0.001,0.999)
            ny[idx]=np.clip(ny[idx],0.001,0.999)
        elif rng.random()<0.6:
            # two circles
            i1=rng.integers(0,n); i2=rng.integers(0,n)
            nx=bx.copy(); ny=by.copy()
            nx[i1]+=rng.normal(0,sigma); ny[i1]+=rng.normal(0,sigma)
            nx[i2]+=rng.normal(0,sigma); ny[i2]+=rng.normal(0,sigma)
            nx=np.clip(nx,0.001,0.999); ny=np.clip(ny,0.001,0.999)
        else:
            # global small shake
            s2=sigma*0.5
            nx=np.clip(bx+rng.normal(0,s2,n),0.001,0.999)
            ny=np.clip(by+rng.normal(0,s2,n),0.001,0.999)
        rlp,s=eval_lp(nx,ny,best_sum+1e-12)
        if rlp is not None and s>best_sum:
            best_sum=s; best_x=nx.copy(); best_y=ny.copy(); best_r=rlp.copy()
            stagn=0
            # update elite
            elite_x[0]=best_x.copy(); elite_y[0]=best_y.copy(); elite_s[0]=best_sum
        else:
            stagn+=1
            # occasionally accept equal to diversify elite (no need)
            if stagn>2000 and rng.random()<0.002:
                # big kick from best
                nx2=np.clip(best_x+rng.normal(0,0.04,n),0.001,0.999)
                ny2=np.clip(best_y+rng.normal(0,0.04,n),0.001,0.999)
                r2,s2=eval_lp(nx2,ny2,-1.0)
                if r2 is not None:
                    # add to elite
                    elite_x.append(nx2); elite_y.append(ny2); elite_s.append(s2)
                    if len(elite_x)>5:
                        # drop worst
                        wi=int(np.argmin(elite_s))
                        elite_x.pop(wi); elite_y.pop(wi); elite_s.pop(wi)
                    stagn=1000
        # periodic coordinate line search with tiny steps
        if it%800==0 and (time.time()-t0<DEADLINE-8.0):
            # try 8-direction moves for each circle greedily, one sweep
            for ci in range(n):
                improved=False
                for ang in [0, np.pi/2, np.pi, 3*np.pi/2, np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]:
                    step=0.002 if stagn<800 else 0.005
                    nx=best_x.copy(); ny=best_y.copy()
                    nx[ci]=np.clip(nx[ci]+step*np.cos(ang),0.001,0.999)
                    ny[ci]=np.clip(ny[ci]+step*np.sin(ang),0.001,0.999)
                    rlp2,s2v=eval_lp(nx,ny,best_sum+1e-12)
                    if rlp2 is not None and s2v>best_sum:
                        best_sum=s2v; best_x=nx; best_y=ny; best_r=rlp2
                        improved=True
                        break
                if time.time()-t0>DEADLINE-5.0:
                    break

    # ---------- final safety shrink ----------
    try:
        Xf=best_x.copy(); Yf=best_y.copy(); Rf=best_r.copy()
        Rf=np.maximum(Rf,1e-12)
        wall=np.minimum(np.minimum(Xf,1.0-Xf),np.minimum(Yf,1.0-Yf))
        dx=Xf[:,None]-Xf[None,:]; dy=Yf[:,None]-Yf[None,:]
        D=np.sqrt(dx*dx+dy*dy)
        np.fill_diagonal(D,np.inf)
        S=Rf[:,None]+Rf[None,:]
        with np.errstate(divide='ignore',invalid='ignore'):
            ratio=D/S
        a=np.min(wall/Rf)
        b=np.min(ratio)
        alpha=min(1.0,a,b)
        alpha=alpha*(1.0-1e-7)-1e-12
        if not np.isfinite(alpha) or alpha<=0:
            alpha=0.999
        Rf=Rf*alpha
        Rf=np.maximum(Rf-1e-13,1e-12)
        best_arr=np.column_stack([Xf,Yf,Rf])
    except Exception:
        best_arr=np.column_stack([best_x,best_y,best_r])
    return best_arr
