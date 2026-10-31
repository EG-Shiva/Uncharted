import numpy as np
import time

def solve():
    n = 26
    t0 = time.time()
    BUDGET_A = 65.0
    BUDGET_TOTAL = 122.0

    r_lo = 0.015
    r_hi = 0.20

    # fallback feasible
    k = int(np.ceil(np.sqrt(n)))
    r0 = 0.5 / k
    pts = []
    for i in range(k):
        for j in range(k):
            if len(pts) < n:
                pts.append([(2*i+1)*r0, (2*j+1)*r0, r0*0.999])
    best_arr = np.array(pts, dtype=float)
    best_sum = float(np.sum(best_arr[:,2]) * 0.9)

    def sigmoid(t):
        t = np.clip(t, -15.0, 15.0)
        return 1.0 / (1.0 + np.exp(-t))

    def decode(p):
        A = p[0:n]; B = p[n:2*n]; C = p[2*n:3*n]
        rs = sigmoid(C)
        R = r_lo + (r_hi - r_lo) * rs
        sx = sigmoid(A); sy = sigmoid(B)
        X = R + (1.0 - 2.0*R)*sx
        Y = R + (1.0 - 2.0*R)*sy
        return X, Y, R, sx, sy, rs

    def encode(Xd, Yd, Rd):
        Rd = np.clip(Rd, r_lo+1e-6, r_hi-1e-6)
        rs = (Rd - r_lo)/(r_hi - r_lo)
        rs = np.clip(rs, 0.02, 0.98)
        C = np.log(rs/(1.0-rs))
        sx = (Xd - Rd)/(1.0-2.0*Rd)
        sy = (Yd - Rd)/(1.0-2.0*Rd)
        sx = np.clip(sx, 0.02, 0.98); sy = np.clip(sy, 0.02, 0.98)
        A = np.log(sx/(1.0-sx)); B = np.log(sy/(1.0-sy))
        return np.concatenate([A,B,C])

    def make_fun(P):
        def fun(p):
            X,Y,R,sx,sy,rs = decode(p)
            DX = X[:,None]-X[None,:]
            DY = Y[:,None]-Y[None,:]
            D = np.sqrt(DX**2+DY**2)
            Dsafe = np.where(np.eye(n,dtype=bool),1.0,np.maximum(D,1e-12))
            S = R[:,None]+R[None,:]
            V = S - D
            V = np.maximum(V,0.0)
            np.fill_diagonal(V,0.0)
            pen = 0.5*np.sum(V**2)
            F = -np.sum(R) + P*pen
            Gr = -np.ones(n) + P*np.sum(2.0*V,axis=1)
            W = np.zeros((n,n))
            mask = V>0
            W[mask] = 2.0*V[mask]/Dsafe[mask]
            Gx = P*np.sum(W*(-DX),axis=1)
            Gy = P*np.sum(W*(-DY),axis=1)
            drdc = (r_hi-r_lo)*rs*(1.0-rs)
            dxda = (1.0-2.0*R)*sx*(1.0-sx)
            dydb = (1.0-2.0*R)*sy*(1.0-sy)
            dxdr = (1.0-2.0*sx); dydr=(1.0-2.0*sy)
            dFda = Gx*dxda; dFdb = Gy*dydb
            dFdc = Gr*drdc + Gx*dxdr*drdc + Gy*dydr*drdc
            g = np.concatenate([dFda,dFdb,dFdc])
            return F,g
        return fun

    from scipy.optimize import minimize, linprog

    iu0, iu1 = np.triu_indices(n,1)
    m = len(iu0)
    A_pair = np.zeros((m,n), dtype=float)
    A_pair[np.arange(m), iu0] = 1.0
    A_pair[np.arange(m), iu1] = 1.0
    c_obj = -np.ones(n)

    def lp_opt(X, Y):
        wall = np.minimum(np.minimum(X,1.0-X), np.minimum(Y,1.0-Y))
        if np.min(wall) < 1.5e-6:
            return None, -1.0
        DX = X[:,None]-X[None,:]
        DY = Y[:,None]-Y[None,:]
        D = np.sqrt(DX*DX+DY*DY)
        b_ub = D[iu0,iu1] - 1e-9
        if np.min(b_ub) < 2e-7:
            return None, -1.0
        bounds = [(1e-7, float(wi)-1e-9) for wi in wall]
        # quick check bounds valid
        for lo,hi in bounds:
            if hi <= lo:
                return None, -1.0
        try:
            res = linprog(c_obj, A_ub=A_pair, b_ub=b_ub, bounds=bounds, method='highs')
        except Exception:
            return None, -1.0
        if not res.success:
            return None, -1.0
        r = np.array(res.x, dtype=float)
        return r, float(np.sum(r))

    best_X = best_arr[:,0].copy(); best_Y = best_arr[:,1].copy(); best_R = best_arr[:,2].copy()

    pool = []  # list of (sum, X, Y, R)

    def consider_pool(X,Y):
        global_best = [False]
        Rlp, s = lp_opt(X,Y)
        if Rlp is None:
            return
        nonlocal_best = [best_sum]
        return Rlp, s

    # Phase A: joint optimisation restarts
    P_list = [300.0, 3000.0, 30000.0]
    restart = 0
    # need best_sum mutable
    cur_best_sum = best_sum
    cur_best_X = best_X.copy(); cur_best_Y = best_Y.copy(); cur_best_R = best_R.copy()

    while True:
        if time.time()-t0 > BUDGET_A:
            break
        # init
        if restart == 0:
            xs = (np.arange(6)+0.5)/6.0
            ys = (np.arange(5)+0.5)/5.0
            X0=[];Y0=[]
            for yy in ys:
                for xx in xs:
                    X0.append(xx);Y0.append(yy)
                    if len(X0)>=n: break
                if len(X0)>=n: break
            X0=np.array(X0);Y0=np.array(Y0)
            rs0=np.random.RandomState(0)
            X0+=rs0.randn(n)*0.005; Y0+=rs0.randn(n)*0.005
            X0=np.clip(X0,0.06,0.94);Y0=np.clip(Y0,0.06,0.94)
            R0=np.full(n,0.10)
            p=encode(X0,Y0,R0)
        elif restart == 1:
            xs5=(np.arange(5)+0.5)/5.0; ys5=(np.arange(5)+0.5)/5.0
            X0=[];Y0=[]
            for yy in ys5:
                for xx in xs5:
                    X0.append(xx);Y0.append(yy)
            X0=np.array(X0[:25]);Y0=np.array(Y0[:25])
            X0=np.append(X0,0.5);Y0=np.append(Y0,0.4)
            rs0=np.random.RandomState(1)
            X0+=rs0.randn(n)*0.01;Y0+=rs0.randn(n)*0.01
            X0=np.clip(X0,0.06,0.94);Y0=np.clip(Y0,0.06,0.94)
            R0=np.full(n,0.10)+rs0.randn(n)*0.005
            R0=np.clip(R0,0.06,0.13)
            p=encode(X0,Y0,R0)
        else:
            rs0=np.random.RandomState(1000+restart)
            X0=rs0.uniform(0.07,0.93,size=n); Y0=rs0.uniform(0.07,0.93,size=n)
            if restart%3==0:
                nx,ny=5,6
                pts2=[]
                for iy in range(ny):
                    for ix in range(nx):
                        xx=(ix+0.5+(0.5 if iy%2==1 else 0.0))/(nx+0.5)
                        yy=(iy+0.5)/ny
                        pts2.append((xx,yy))
                        if len(pts2)>=n: break
                    if len(pts2)>=n: break
                pts2=np.array(pts2)
                X0=pts2[:,0]+rs0.randn(n)*0.015
                Y0=pts2[:,1]+rs0.randn(n)*0.015
                X0=np.clip(X0,0.06,0.94);Y0=np.clip(Y0,0.06,0.94)
            R0=0.09+rs0.rand(n)*0.04
            p=encode(X0,Y0,R0)
        cur=p
        for P in P_list:
            if time.time()-t0 > BUDGET_A:
                break
            fun=make_fun(P)
            try:
                res=minimize(fun,cur,jac=True,method='L-BFGS-B',options={'maxiter':220,'ftol':1e-10,'gtol':1e-6})
                cur=res.x
            except Exception:
                break
        X,Y,R,_,_,_=decode(cur)
        Rlp,s = lp_opt(X,Y)
        if Rlp is not None and s > cur_best_sum:
            cur_best_sum=s
            cur_best_X=X.copy();cur_best_Y=Y.copy();cur_best_R=Rlp.copy()
        # keep pool of top
        if Rlp is not None:
            pool.append((s,X.copy(),Y.copy(),Rlp.copy()))
            # keep only top 5
            pool.sort(key=lambda t: t[0], reverse=True)
            if len(pool)>5:
                pool.pop()
        restart+=1
        if restart>300:
            break

    best_sum = cur_best_sum
    best_X, best_Y, best_R = cur_best_X.copy(), cur_best_Y.copy(), cur_best_R.copy()

    # Phase B: hill climbing on LP value, starting from pool bests
    if len(pool)==0:
        pool=[(best_sum,best_X.copy(),best_Y.copy(),best_R.copy())]
    # order pool best first
    pool.sort(key=lambda t: t[0], reverse=True)
    # focus on overall best but occasionally try others if time
    rng_hill = np.random.RandomState(9999)
    sigmas = [0.015, 0.010, 0.006, 0.0035, 0.002, 0.001]
    trials_per_sigma = [350, 350, 350, 350, 300, 200]

    # start from best
    Xc, Yc, Rc = best_X.copy(), best_Y.copy(), best_R.copy()
    Sc = best_sum

    # if pool has diverse good, try quick tournament: evaluate small jitter around each? skip, just hill on best
    for si, sigma in enumerate(sigmas):
        ntrials = trials_per_sigma[si]
        for t in range(ntrials):
            if time.time()-t0 > BUDGET_TOTAL:
                break
            # choose move type
            u = rng_hill.rand()
            Xp = Xc.copy(); Yp = Yc.copy()
            if u < 0.70:
                i = rng_hill.randint(n)
                Xp[i] += rng_hill.randn()*sigma
                Yp[i] += rng_hill.randn()*sigma
                if Xp[i]<0.015 or Xp[i]>0.985 or Yp[i]<0.015 or Yp[i]>0.985:
                    continue
                # quick distance prune
                dx = Xp[i]-Xp; dy = Yp[i]-Yp
                dx[i]=1e9; dy[i]=1e9
                dmin = np.min(np.sqrt(dx*dx+dy*dy))
                if dmin < 0.015:
                    continue
            elif u < 0.90:
                # perturb two circles
                i1 = rng_hill.randint(n); i2 = rng_hill.randint(n)
                Xp[i1]+=rng_hill.randn()*sigma; Yp[i1]+=rng_hill.randn()*sigma
                Xp[i2]+=rng_hill.randn()*sigma; Yp[i2]+=rng_hill.randn()*sigma
                if np.any(Xp<0.012) or np.any(Xp>0.988) or np.any(Yp<0.012) or np.any(Yp>0.988):
                    continue
            else:
                # global small jitter
                Xp += rng_hill.randn(n)*sigma*0.4
                Yp += rng_hill.randn(n)*sigma*0.4
                if np.any(Xp<0.012) or np.any(Xp>0.988) or np.any(Yp<0.012) or np.any(Yp>0.988):
                    continue
            wall = np.minimum(np.minimum(Xp,1.0-Xp), np.minimum(Yp,1.0-Yp))
            if np.min(wall) < 0.02:
                # allow but require not too small; LP would be poor
                # skip if any wall <0.015 to save time
                if np.min(wall) < 0.015:
                    continue
            Rlp, s = lp_opt(Xp,Yp)
            if Rlp is not None and s > Sc + 1e-12:
                Sc = s; Xc=Xp; Yc=Yp; Rc=Rlp
                best_sum=Sc; best_X=Xc.copy(); best_Y=Yc.copy(); best_R=Rc.copy()
        if time.time()-t0 > BUDGET_TOTAL:
            break
        # after each sigma, try to also exploit second pool member by crossover? simple: if stuck, restart from 2nd best with small mix
        # (only once)
    # final small polish: extra 200 trials at tiny sigma on best
    if time.time()-t0 < BUDGET_TOTAL-2:
        sigma=0.0008
        for _ in range(300):
            if time.time()-t0 > BUDGET_TOTAL:
                break
            i = rng_hill.randint(n)
            Xp=Xc.copy();Yp=Yc.copy()
            Xp[i]+=rng_hill.randn()*sigma; Yp[i]+=rng_hill.randn()*sigma
            Rlp,s=lp_opt(Xp,Yp)
            if Rlp is not None and s>Sc+1e-12:
                Sc=s;Xc=Xp;Yc=Yp;Rc=Rlp
                best_sum=Sc;best_X=Xc.copy();best_Y=Yc.copy();best_R=Rc.copy()

    Xf, Yf, Rf = best_X.copy(), best_Y.copy(), best_R.copy()
    # strict safety shrink
    DX = Xf[:,None]-Xf[None,:]; DY=Yf[:,None]-Yf[None,:]
    D = np.sqrt(DX*DX+DY*DY)
    wall = np.minimum(np.minimum(Xf,1.0-Xf), np.minimum(Yf,1.0-Yf))
    fb = np.min(wall/np.maximum(Rf,1e-12))
    fp = 1e9
    for i in range(n):
        for j in range(i+1,n):
            denom=Rf[i]+Rf[j]
            if denom>0:
                fp=min(fp, D[i,j]/denom)
    f=min(1.0,fb,fp)
    f_safe=f*(1.0-1e-7)-1e-12
    if not np.isfinite(f_safe) or f_safe<=0:
        f_safe=0.99*min(1.0,fb if np.isfinite(fb) else 1.0, fp if np.isfinite(fp) else 1.0)
    Rf = Rf*f_safe
    Rf = np.maximum(Rf,1e-6)
    # extra tiny factor
    Rf *= (1.0-1e-9)
    arr=np.column_stack([Xf,Yf,Rf])
    return arr
