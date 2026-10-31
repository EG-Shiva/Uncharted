import numpy as np
import time

def solve():
    n = 26
    t0 = time.time()
    DEADLINE = 122.0
    rng = np.random.default_rng(0)
    from scipy.optimize import minimize, linprog
    try:
        from scipy.optimize import nnls
        has_nnls = True
    except Exception:
        has_nnls = False

    I, J = np.triu_indices(n, 1)
    m = len(I)
    A0 = np.zeros((m, n), dtype=float)
    A0[np.arange(m), I] = 1.0
    A0[np.arange(m), J] = 1.0
    cobj = -np.ones(n, dtype=float)
    ones_n = np.ones(n, dtype=float)

    # ---------- helpers ----------
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

    def lp_solve(X, Y):
        wall = np.minimum(np.minimum(X, 1.0-X), np.minimum(Y, 1.0-Y))
        wall = np.maximum(wall, 0.0)
        dx = X[I]-X[J]; dy = Y[I]-Y[J]
        d = np.sqrt(dx*dx+dy*dy)
        bnds = [(0.0, float(w)) for w in wall]
        try:
            res = linprog(cobj, A_ub=A0, b_ub=d, bounds=bnds, method='highs')
        except Exception:
            return None, -1.0
        if res.status != 0:
            return None, -1.0
        r = np.asarray(res.x, dtype=float)
        r = np.maximum(r, 0.0)
        return r, float(np.sum(r))

    # fallback grid
    k = int(np.ceil(np.sqrt(n)))
    r0 = 0.5/k
    fb = []
    for ix in range(k):
        for jy in range(k):
            if len(fb) < n:
                fb.append([(2*ix+1)*r0, (2*jy+1)*r0, r0*0.999])
    fb_arr = np.array(fb, dtype=float)
    Xb, Yb, Rb = fb_arr[:,0].copy(), fb_arr[:,1].copy(), fb_arr[:,2].copy()
    Rb_s, _ = uniform_scale(Xb, Yb, Rb)
    best_arr = np.column_stack([Xb, Yb, Rb_s])
    best_sum = float(np.sum(Rb_s))
    # try LP on fallback
    try:
        rlp, slp = lp_solve(Xb, Yb)
        if rlp is not None:
            rs, _ = uniform_scale(Xb, Yb, rlp*(1.0-1e-9))
            # re-scale already; use shrunk
            s = float(np.sum(rs))
            if s > best_sum:
                best_sum = s
                best_arr = np.column_stack([Xb, Yb, rs])
    except Exception:
        pass

    pool = []  # list of (sum, X, Y)

    # ---------- Phase A: fast penalized joint search ----------
    def make_pen_fun(nn):
        def fun(v, W):
            X = v[0:nn]; Y = v[nn:2*nn]; R = v[2*nn:3*nn]
            gx = np.zeros(nn); gy = np.zeros(nn)
            pen_gr = np.zeros(nn)
            pen = 0.0
            v1 = R-X
            mm = v1>0
            if np.any(mm):
                pen += np.sum(v1[mm]**2)
                gx[mm] += -2.0*v1[mm]; pen_gr[mm] += 2.0*v1[mm]
            v2 = X+R-1.0
            mm = v2>0
            if np.any(mm):
                pen += np.sum(v2[mm]**2)
                gx[mm] += 2.0*v2[mm]; pen_gr[mm] += 2.0*v2[mm]
            v3 = R-Y
            mm = v3>0
            if np.any(mm):
                pen += np.sum(v3[mm]**2)
                gy[mm] += -2.0*v3[mm]; pen_gr[mm] += 2.0*v3[mm]
            v4 = Y+R-1.0
            mm = v4>0
            if np.any(mm):
                pen += np.sum(v4[mm]**2)
                gy[mm] += 2.0*v4[mm]; pen_gr[mm] += 2.0*v4[mm]
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
                gx += np.sum(Q*Dx, axis=1)
                gy += np.sum(Q*Dy, axis=1)
                G = np.zeros((nn,nn)); G[Mask]=2.0*V[Mask]
                pen_gr += np.sum(G, axis=1)
            loss = -np.sum(R)+W*pen
            g = np.empty_like(v)
            g[0:nn]=W*gx; g[nn:2*nn]=W*gy
            g[2*nn:3*nn]=-1.0+W*pen_gr
            return loss, g
        return fun
    pen_fun = make_pen_fun(n)
    bounds78 = [(0.0,1.0)]*(2*n)+[(0.03,0.20)]*n
    Ws = [1e2, 1e3, 1e4, 1e5]
    n_restart_A = 14
    for restart in range(n_restart_A):
        if time.time()-t0 > 55.0:
            break
        if restart==0:
            X0=np.zeros(n);Y0=np.zeros(n)
            idx=0
            for ii in range(5):
                for jj in range(5):
                    X0[idx]=(ii+0.5)/5.0+rng.normal(0,0.006)
                    Y0[idx]=(jj+0.5)/5.0+rng.normal(0,0.006)
                    idx+=1
            X0[25]=0.88+rng.normal(0,0.01);Y0[25]=0.12+rng.normal(0,0.01)
            X0=np.clip(X0,0.05,0.95);Y0=np.clip(Y0,0.05,0.95)
            R0=np.full(n,0.095)+rng.normal(0,0.005,n)
            R0=np.clip(R0,0.03,0.20)
        elif restart==1:
            pts=[]
            for ix in range(6):
                for jy in range(5):
                    xx=(ix+0.5)/6.0; yy=(jy+0.5)/5.0
                    if jy%2==1:
                        xx+=0.5/6.0*0.5
                        if xx>0.95: xx-=1.0/6.0
                    pts.append((xx,yy))
            arr=np.array(pts)
            perm=rng.permutation(len(arr))[:n]
            X0=arr[perm,0]+rng.normal(0,0.005,n)
            Y0=arr[perm,1]+rng.normal(0,0.005,n)
            X0=np.clip(X0,0.05,0.95);Y0=np.clip(Y0,0.05,0.95)
            R0=0.085+0.02*rng.random(n)
        else:
            X0=rng.random(n)*0.9+0.05
            Y0=rng.random(n)*0.9+0.05
            R0=0.08+0.03*rng.random(n)
        v=np.concatenate([X0,Y0,R0])
        for W in Ws:
            if time.time()-t0>57.0: break
            try:
                res=minimize(pen_fun,v,args=(W,),jac=True,method='L-BFGS-B',bounds=bounds78,options={'maxiter':220,'ftol':1e-9,'gtol':1e-6})
                v=res.x
            except Exception:
                break
        X=np.clip(v[0:n],0.0,1.0);Y=np.clip(v[n:2*n],0.0,1.0);R=np.clip(v[2*n:3*n],0.01,0.25)
        try:
            rlp,slp=lp_solve(X,Y)
            if rlp is not None:
                rsh=np.maximum(rlp*(1.0-1e-9)-1e-12,0.0)
                # quick feasibility via scale
                rsh,_=uniform_scale(X,Y,np.maximum(rlp,1e-12))
                s=float(np.sum(rsh))
                pool.append((s,X.copy(),Y.copy()))
                if s>best_sum:
                    best_sum=s;best_arr=np.column_stack([X,Y,rsh])
            else:
                rs,_=uniform_scale(X,Y,R)
                s=float(np.sum(rs))
                pool.append((s,X.copy(),Y.copy()))
                if s>best_sum:
                    best_sum=s;best_arr=np.column_stack([X,Y,rs])
        except Exception:
            pass
    # keep top
    pool=sorted(pool,key=lambda t:-t[0])[:6]
    # augment with rotations/reflections + small noise of best
    if len(pool)>0:
        aug=[]
        s0,X0,Y0=pool[0]
        for krot in [1,2,3]:
            Xr,Yr=X0.copy(),Y0.copy()
            for _ in range(krot):
                Xr,Yr=Yr,1.0-Xr
            aug.append((s0,Xr.copy(),Yr.copy()))
        # reflection
        aug.append((s0,1.0-X0,Y0.copy()))
        aug.append((s0,X0.copy(),1.0-Y0))
        for s_,xa,ya in aug:
            try:
                rl,sl=lp_solve(xa,ya)
                if rl is not None:
                    rs,_=uniform_scale(xa,ya,rl)
                    pool.append((float(np.sum(rs)),xa,ya))
            except Exception:
                pass
        pool=sorted(pool,key=lambda t:-t[0])[:6]

    # ---------- Phase B: exact active-set LP-dual centre polishing ----------
    def lp_val_grad(cxy):
        X=cxy[:n];Y=cxy[n:]
        wall=np.minimum(np.minimum(X,1.0-X),np.minimum(Y,1.0-Y))
        if np.min(wall)<1e-4:
            # penalize outside
            pen=np.sum(np.maximum(1e-4-wall,0.0)**2)
            g=np.zeros(2*n)
            # crude push inside
            for i in range(n):
                if wall[i]<1e-4:
                    # determine side
                    xi=X[i];yi=Y[i]
                    # gradient of wall wrt x,y
                    # wall = min(x,1-x,y,1-y)
                    vals=np.array([xi,1-xi,yi,1-yi])
                    a=np.argmin(vals)
                    if a==0: g[i]-=2*(1e-4-wall[i])
                    elif a==1: g[i]+=2*(1e-4-wall[i])
                    elif a==2: g[n+i]-=2*(1e-4-wall[i])
                    else: g[n+i]+=2*(1e-4-wall[i])
            return (-2.0+1e4*pen, 1e4*g)
        dx=X[I]-X[J];dy=Y[I]-Y[J]
        d=np.sqrt(dx*dx+dy*dy)
        d=np.maximum(d,1e-12)
        bnds=[(0.0,float(w)) for w in np.maximum(wall,0.0)]
        try:
            res=linprog(cobj,A_ub=A0,b_ub=d,bounds=bnds,method='highs')
        except Exception:
            return (0.0, np.zeros(2*n))
        if res.status!=0:
            return (0.0, np.zeros(2*n))
        r=np.asarray(res.x,float)
        S=float(np.sum(r))
        # active set
        slack_p=d-(r[I]+r[J])
        slack_w=wall-r
        tight_p=np.where(slack_p<1e-9)[0]
        tight_w=np.where(slack_w<1e-9)[0]
        t=len(tight_p)+len(tight_w)
        if t==0:
            return (-S, np.zeros(2*n))
        # build Atight (t x n)
        At=np.zeros((t,n),float)
        prow_I=[];prow_J=[];prow_d=[]
        row=0
        pair_rows=[]
        wall_rows=[]
        for kk in tight_p:
            i=int(I[kk]);j=int(J[kk])
            At[row,i]=1.0;At[row,j]=1.0
            pair_rows.append((row,i,j,float(d[kk])))
            row+=1
        for i in tight_w:
            At[row,int(i)]=1.0
            wall_rows.append((row,int(i)))
            row+=1
        # dual via NNLS: At.T lam ~= 1
        try:
            if has_nnls:
                M=At.T  # n x t
                lam,_=nnls(M,ones_n)
            else:
                # lstsq + clip
                lam,_ ,_,_=np.linalg.lstsq(At.T,ones_n,rcond=None)
                lam=np.maximum(lam,0.0)
        except Exception:
            return (-S, np.zeros(2*n))
        gx=np.zeros(n);gy=np.zeros(n)
        for (rr,i,j,dd) in pair_rows:
            l=float(lam[rr])
            if l<1e-12: continue
            if dd<1e-12: continue
            ux=(X[i]-X[j])/dd;uy=(Y[i]-Y[j])/dd
            gx[i]+=l*ux;gx[j]-=l*ux
            gy[i]+=l*uy;gy[j]-=l*uy
        for (rr,i) in wall_rows:
            l=float(lam[rr])
            if l<1e-12: continue
            xi=X[i];yi=Y[i]
            vals=[xi,1-xi,yi,1-yi]
            a=int(np.argmin(vals))
            if a==0: gx[i]+=l*1.0
            elif a==1: gx[i]+=l*(-1.0)
            elif a==2: gy[i]+=l*1.0
            else: gy[i]+=l*(-1.0)
        # we minimize -S, grad = -dS/dc
        return (-S, -np.concatenate([gx,gy]))

    cbnds=[(0.02,0.98)]*(2*n)
    new_pool=[]
    for (s_,Xc,Yc) in pool:
        if time.time()-t0>88.0: 
            new_pool.append((s_,Xc,Yc))
            continue
        # small perturbation to escape degeneracy
        c0=np.concatenate([Xc+ rng.normal(0,0.0015,n), Yc+rng.normal(0,0.0015,n)])
        c0=np.clip(c0,0.02,0.98)
        try:
            res=minimize(lp_val_grad,c0,jac=True,method='L-BFGS-B',bounds=cbnds,options={'maxiter':120,'ftol':1e-12,'gtol':1e-8})
            Xopt=np.clip(res.x[:n],0.0,1.0);Yopt=np.clip(res.x[n:],0.0,1.0)
            rl,sl=lp_solve(Xopt,Yopt)
            if rl is not None:
                rs,_=uniform_scale(Xopt,Yopt,rl)
                snew=float(np.sum(rs))
                new_pool.append((snew,Xopt,Yopt))
                if snew>best_sum:
                    best_sum=snew;best_arr=np.column_stack([Xopt,Yopt,rs])
            else:
                new_pool.append((s_,Xc,Yc))
        except Exception:
            new_pool.append((s_,Xc,Yc))
    pool=sorted(new_pool,key=lambda t:-t[0])[:6] if len(new_pool)>0 else pool

    # ---------- Phase C: SLSQP joint jam with analytic Jacobians ----------
    def slsqp_polish(X0,Y0,R0,maxiter=600):
        v0=np.concatenate([X0,Y0,R0])
        # ensure strictly feasible interior
        # shrink radii a bit
        v0[2*n:]=np.maximum(v0[2*n:]*0.999,1e-6)
        def obj(v):
            return -np.sum(v[2*n:3*n])
        def obj_jac(v):
            g=np.zeros_like(v);g[2*n:]=-1.0
            return g
        def wall_fun(v):
            X=v[0:n];Y=v[n:2*n];R=v[2*n:3*n]
            out=np.empty(4*n)
            out[0:n]=X-R
            out[n:2*n]=1.0-X-R
            out[2*n:3*n]=Y-R
            out[3*n:4*n]=1.0-Y-R
            return out
        def wall_jac(v):
            J=np.zeros((4*n,3*n))
            idx=np.arange(n)
            J[idx,idx]=1.0;J[idx,2*n+idx]=-1.0
            J[n+idx,idx]=-1.0;J[n+idx,2*n+idx]=-1.0
            J[2*n+idx,n+idx]=1.0;J[2*n+idx,2*n+idx]=-1.0
            J[3*n+idx,n+idx]=-1.0;J[3*n+idx,2*n+idx]=-1.0
            return J
        def pair_fun(v):
            X=v[0:n];Y=v[n:2*n];R=v[2*n:3*n]
            dx=X[I]-X[J];dy=Y[I]-Y[J]
            d=np.sqrt(dx*dx+dy*dy)
            return d-R[I]-R[J]
        def pair_jac(v):
            X=v[0:n];Y=v[n:2*n]
            dx=X[I]-X[J];dy=Y[I]-Y[J]
            d=np.sqrt(dx*dx+dy*dy)
            d=np.maximum(d,1e-14)
            ux=dx/d;uy=dy/d
            J=np.zeros((m,3*n))
            kk=np.arange(m)
            J[kk,I]=ux;J[kk,J]=-ux
            J[kk,n+I]=uy;J[kk,n+J]=-uy
            J[kk,2*n+I]=-1.0;J[kk,2*n+J]=-1.0
            return J
        cons=[{'type':'ineq','fun':wall_fun,'jac':wall_jac},
              {'type':'ineq','fun':pair_fun,'jac':pair_jac}]
        try:
            res=minimize(obj,v0,jac=obj_jac,method='SLSQP',constraints=cons,options={'maxiter':maxiter,'ftol':1e-13,'disp':False,'eps':1e-12})
        except Exception:
            return None, -1
        v=res.x
        X=v[0:n];Y=v[n:2*n];R=v[2*n:3*n]
        return (X,Y,R)
    # order pool best first
    for (s_,Xc,Yc) in pool:
        if time.time()-t0>DEADLINE-6: break
        try:
            rl,_=lp_solve(Xc,Yc)
            if rl is None: continue
            Rinit=np.maximum(rl,0.005)
        except Exception:
            continue
        # add tiny noise for diversity on later ones
        # first one no noise
        try:
            Xo,Yo,Ro=slsqp_polish(Xc,Yc,Rinit,maxiter=500)
        except Exception:
            continue
        if Xo is None: continue
        try:
            Rs,_=uniform_scale(Xo,Yo,Ro)
            # verify strict
            s=float(np.sum(Rs))
            # check feasibility quickly
            wall=np.minimum(np.minimum(Xo,1-Xo),np.minimum(Yo,1-Yo))
            if np.min(wall-Rs)<-1e-12: 
                continue
            dx=Xo[I]-Xo[J];dy=Yo[I]-Yo[J]
            d=np.sqrt(dx*dx+dy*dy)
            if np.min(d-(Rs[I]+Rs[J]))<-1e-12:
                continue
            if np.min(Rs)<=0: continue
            if s>best_sum:
                best_sum=s;best_arr=np.column_stack([Xo,Yo,Rs])
        except Exception:
            continue
        # second SLSQP pass with tighter from best (chain)
        if time.time()-t0>DEADLINE-6: break

    # final safety shrink
    try:
        Xf=best_arr[:,0];Yf=best_arr[:,1];Rf=best_arr[:,2]
        Rfs,_=uniform_scale(Xf,Yf,Rf)
        best_arr=np.column_stack([Xf,Yf,Rfs])
    except Exception:
        pass
    return best_arr
