import numpy as np
import time

def solve():
    n = 26
    t0 = time.time()
    DEADLINE = 122.0
    rng = np.random.default_rng(1)

    # ---------- helpers ----------
    from scipy.optimize import minimize, linprog

    def lp_polish(X, Y):
        try:
            DX = X[:, None] - X[None, :]
            DY = Y[:, None] - Y[None, :]
            D = np.sqrt(DX*DX + DY*DY)
            b = np.minimum(np.minimum(X, 1.0-X), np.minimum(Y, 1.0-Y))
            if np.any(b < 8e-6):
                return None
            m = n*(n-1)//2
            if m > 0:
                A_ub = np.zeros((m, n))
                b_ub = np.zeros(m)
                k = 0
                for i in range(n):
                    for j in range(i+1, n):
                        A_ub[k, i] = 1.0
                        A_ub[k, j] = 1.0
                        b_ub[k] = D[i, j] - 1e-9
                        k += 1
                if np.any(b_ub < 1.6e-5):
                    return None
            else:
                A_ub = None; b_ub = None
            c = -np.ones(n)
            bounds = [(1e-5, float(bi) - 1e-9) for bi in b]
            # check bounds valid
            for lo, hi in bounds:
                if hi < lo:
                    return None
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if not res.success:
                return None
            r = np.asarray(res.x).ravel()
            r = r * (1.0 - 1e-7) - 1e-11
            r = np.maximum(r, 1e-6)
            # verify
            if np.any(r > b):
                return None
            # pairwise verify
            k = 0
            for i in range(n):
                for j in range(i+1, n):
                    if r[i]+r[j] > D[i,j] + 1e-12:
                        return None
            return r
        except Exception:
            return None

    def uniform_shrink(X, Y, R):
        wall = np.minimum(np.minimum(X, 1.0-X), np.minimum(Y, 1.0-Y))
        a = np.min(wall / np.maximum(R, 1e-12))
        DX = X[:, None]-X[None,:]
        DY = Y[:, None]-Y[None,:]
        D = np.sqrt(DX*DX+DY*DY)
        S = R[:,None]+R[None,:]
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = D/np.maximum(S,1e-12)
        np.fill_diagonal(ratio, np.inf)
        bmin = np.min(ratio)
        alpha = min(1.0, float(a), float(bmin))
        alpha = alpha*(1.0-1e-7)-1e-12
        if not np.isfinite(alpha) or alpha<=0:
            alpha=0.5
        return R*alpha, alpha

    # penalty fun factory (returns fun(v,W))
    def make_pen():
        def fun(v, W):
            X = v[0:n]; Y = v[n:2*n]; R = v[2*n:3*n]
            pen = 0.0
            pgx = np.zeros(n); pgy = np.zeros(n); pgr = np.zeros(n)
            v1 = R - X
            m = v1 > 0
            if np.any(m):
                pen += np.sum(v1[m]**2)
                pgx[m] += -2.0*v1[m]; pgr[m] += 2.0*v1[m]
            v2 = X + R - 1.0
            m = v2 > 0
            if np.any(m):
                pen += np.sum(v2[m]**2)
                pgx[m] += 2.0*v2[m]; pgr[m] += 2.0*v2[m]
            v3 = R - Y
            m = v3 > 0
            if np.any(m):
                pen += np.sum(v3[m]**2)
                pgy[m] += -2.0*v3[m]; pgr[m] += 2.0*v3[m]
            v4 = Y + R - 1.0
            m = v4 > 0
            if np.any(m):
                pen += np.sum(v4[m]**2)
                pgy[m] += 2.0*v4[m]; pgr[m] += 2.0*v4[m]
            DX = X[:,None]-X[None,:]
            DY = Y[:,None]-Y[None,:]
            D = np.sqrt(DX*DX+DY*DY+1e-20)
            np.fill_diagonal(D, 1.0)
            S = R[:,None]+R[None,:]
            V = S - D
            np.fill_diagonal(V, 0.0)
            Mask = V > 0
            if np.any(Mask):
                pen += 0.5*np.sum(V[Mask]**2)
                Q = np.zeros((n,n))
                Q[Mask] = 2.0*V[Mask]/D[Mask]
                pgx += np.sum(Q*(-DX), axis=1)
                pgy += np.sum(Q*(-DY), axis=1)
                G = np.zeros((n,n)); G[Mask]=2.0*V[Mask]
                pgr += np.sum(G, axis=1)
            F = -np.sum(R) + W*pen
            g = np.empty_like(v)
            g[0:n] = W*pgx
            g[n:2*n] = W*pgy
            g[2*n:3*n] = -1.0 + W*pgr
            return F, g
        return fun
    penfun = make_pen()
    boundsVR = [(0.0,1.0)]*(2*n) + [(0.015,0.20)]*n

    def refine(X0, Y0, R0, W, maxiter):
        v0 = np.concatenate([X0, Y0, R0])
        v0 = np.clip(v0, [0.0]*(2*n)+[0.015]*n, [1.0]*(2*n)+[0.20]*n)
        try:
            res = minimize(penfun, v0, args=(W,), jac=True, method='L-BFGS-B',
                           bounds=boundsVR, options={'maxiter': maxiter, 'ftol':1e-10, 'gtol':1e-5})
            v = res.x
            return v[0:n].copy(), v[n:2*n].copy(), v[2*n:3*n].copy()
        except Exception:
            return X0, Y0, R0

    def contacts(X, Y, R, tol=2.5e-3):
        DX = X[:,None]-X[None,:]; DY=Y[:,None]-Y[None,:]
        D = np.sqrt(DX*DX+DY*DY)
        S = R[:,None]+R[None,:]
        cnt = np.zeros(n, dtype=int)
        for i in range(n):
            c = 0
            for j in range(n):
                if j==i: continue
                if D[i,j] <= S[i,j]+tol:
                    c+=1
            if X[i]-R[i] <= tol: c+=1
            if (1.0-X[i])-R[i] <= tol: c+=1
            if Y[i]-R[i] <= tol: c+=1
            if (1.0-Y[i])-R[i] <= tol: c+=1
            cnt[i]=c
        return cnt

    # void finder: clearance grid
    Gv = 181
    xs1 = (np.arange(Gv)+0.5)/Gv
    ys1 = (np.arange(Gv)+0.5)/Gv
    Xg2, Yg2 = np.meshgrid(xs1, ys1)
    Xgf = Xg2.ravel(); Ygf = Yg2.ravel()
    wallf0 = np.minimum(np.minimum(Xgf,1.0-Xgf), np.minimum(Ygf,1.0-Ygf))

    def find_voids(X, Y, R, k=3):
        clear = wallf0.copy()
        # vectorized loop
        for i in range(n):
            dx = Xgf - X[i]; dy = Ygf - Y[i]
            d = np.sqrt(dx*dx+dy*dy) - R[i]
            np.minimum(clear, d, out=clear)
        out=[]
        cur = clear.copy()
        for _ in range(k):
            idx = int(np.argmax(cur))
            c = float(cur[idx])
            vx = float(Xgf[idx]); vy = float(Ygf[idx])
            if c < 1e-4:
                # still record but break-ish
                out.append((vx,vy,max(c,1e-4)))
                cur[idx] = -1e9
                continue
            out.append((vx,vy,c))
            # simulate inserting circle: clearance update
            dx = Xgf - vx; dy = Ygf - vy
            d = np.sqrt(dx*dx+dy*dy) - c
            np.minimum(cur, d, out=cur)
        return out, float(np.max(clear))

    # ---------- Phase A: multi-restart ----------
    # fallback
    ksq = int(np.ceil(np.sqrt(n)))
    r0f = 0.5/ksq
    pts=[]
    for i in range(ksq):
        for j in range(ksq):
            if len(pts)<n:
                pts.append([(2*i+1)*r0f,(2*j+1)*r0f,r0f*0.999])
    barr = np.array(pts,float)
    Rs,_ = uniform_shrink(barr[:,0],barr[:,1],barr[:,2])
    barr[:,2]=Rs
    best_sum=float(np.sum(Rs)); best_X=barr[:,0].copy(); best_Y=barr[:,1].copy(); best_R=barr[:,2].copy()

    WsA=[5e2,5e3,5e4]
    restart=0
    while time.time()-t0 < 38.0:
        # init
        if restart==0:
            xs=(np.arange(6)+0.5)/6.0; ys=(np.arange(5)+0.5)/5.0
            X0=[];Y0=[]
            for yy in ys:
                for xx in xs:
                    X0.append(xx);Y0.append(yy)
                    if len(X0)>=n: break
                if len(X0)>=n: break
            X0=np.array(X0);Y0=np.array(Y0)
            X0+=rng.standard_normal(n)*0.005; Y0+=rng.standard_normal(n)*0.005
            X0=np.clip(X0,0.06,0.94);Y0=np.clip(Y0,0.06,0.94)
            R0=np.full(n,0.10)
        elif restart==1:
            xs5=(np.arange(5)+0.5)/5.0; ys5=(np.arange(5)+0.5)/5.0
            X0=[];Y0=[]
            for yy in ys5:
                for xx in xs5:
                    X0.append(xx);Y0.append(yy)
            X0=np.array(X0[:25]);Y0=np.array(Y0[:25])
            X0=np.append(X0,0.5);Y0=np.append(Y0,0.4)
            X0+=rng.standard_normal(n)*0.01;Y0+=rng.standard_normal(n)*0.01
            X0=np.clip(X0,0.06,0.94);Y0=np.clip(Y0,0.06,0.94)
            R0=np.clip(0.10+rng.standard_normal(n)*0.005,0.06,0.13)
        else:
            X0=rng.uniform(0.07,0.93,size=n); Y0=rng.uniform(0.07,0.93,size=n)
            if restart%3==0:
                nx,ny=5,6; pts2=[]
                for iy in range(ny):
                    for ix in range(nx):
                        xx=(ix+0.5+(0.5 if iy%2==1 else 0.0))/(nx+0.5); yy=(iy+0.5)/ny
                        pts2.append((xx,yy))
                        if len(pts2)>=n: break
                    if len(pts2)>=n: break
                pts2=np.array(pts2)
                X0=pts2[:,0]+rng.standard_normal(n)*0.015
                Y0=pts2[:,1]+rng.standard_normal(n)*0.015
                X0=np.clip(X0,0.06,0.94);Y0=np.clip(Y0,0.06,0.94)
            R0=0.09+rng.random(n)*0.04
        Xc,Yc,Rc=X0.copy(),Y0.copy(),R0.copy()
        for W in WsA:
            if time.time()-t0>40.0: break
            Xc,Yc,Rc=refine(Xc,Yc,Rc,W,150)
        # shrink eval
        Rs2,_=uniform_shrink(Xc,Yc,Rc)
        s=float(np.sum(Rs2))
        if s>best_sum:
            best_sum=s; best_X=Xc.copy(); best_Y=Yc.copy(); best_R=Rs2.copy()
        Rlp=lp_polish(Xc,Yc)
        if Rlp is not None:
            s2=float(np.sum(Rlp))
            if s2>best_sum:
                best_sum=s2; best_X=Xc.copy(); best_Y=Yc.copy(); best_R=Rlp.copy()
            # extra kick + re-opt occasionally
            if restart%2==0 and (time.time()-t0<36.0):
                Xk=Xc+rng.standard_normal(n)*0.003; Yk=Yc+rng.standard_normal(n)*0.003
                Xk=np.clip(Xk,Rlp+1e-6,1.0-Rlp-1e-6); Yk=np.clip(Yk,Rlp+1e-6,1.0-Rlp-1e-6)
                X2,Y2,R2=refine(Xk,Yk,np.clip(Rlp,0.015,0.20),5e4,100)
                Rs3,_=uniform_shrink(X2,Y2,R2)
                if float(np.sum(Rs3))>best_sum:
                    best_sum=float(np.sum(Rs3)); best_X=X2.copy(); best_Y=Y2.copy(); best_R=Rs3.copy()
                Rlp2=lp_polish(X2,Y2)
                if Rlp2 is not None and float(np.sum(Rlp2))>best_sum:
                    best_sum=float(np.sum(Rlp2)); best_X=X2.copy(); best_Y=Y2.copy(); best_R=Rlp2.copy()
        restart+=1
        if restart>60: break

    # ---------- Phase B: topology surgery + SA ----------
    cur_X,cur_Y,cur_R = best_X.copy(), best_Y.copy(), best_R.copy()
    cur_sum = best_sum
    T0=0.004; T1=0.00008
    # estimate total iters
    it=0; maxit=100000
    since_best=0
    # cooling geometric
    alpha=0.997
    T=T0
    while time.time()-t0 < DEADLINE:
        it+=1
        # periodic elite restore
        if since_best>35:
            cur_X,cur_Y,cur_R=best_X.copy(),best_Y.copy(),best_R.copy()
            cur_sum=best_sum
            since_best=0
            T=max(T*1.2,T1)  # slight reheat
        T=max(T1, T0*(alpha**it))
        # choose operation
        u=rng.random()
        cnt=contacts(cur_X,cur_Y,cur_R)
        order_small=np.argsort(cur_R)  # ascending
        order_weak=np.lexsort((cur_R, cnt))  # least contacts first
        voids,_ = find_voids(cur_X,cur_Y,cur_R,k=3)[0:2] if False else (None,None)
        # compute voids (inline to avoid double call)
        # --- find voids ---
        # (recompute quickly)
        # we already have function; call it
        vlist,_maxc = find_voids(cur_X,cur_Y,cur_R,k=3)
        Xn=cur_X.copy(); Yn=cur_Y.copy(); Rn=cur_R.copy()
        if u<0.35:
            # move single weakest to largest void
            idx=int(order_weak[0])
            vx,vy,vc=vlist[0]
            Xn[idx]=np.clip(vx,vc+1e-6,1.0-vc-1e-6) if vc<0.5 else vx
            Yn[idx]=np.clip(vy,vc+1e-6,1.0-vc-1e-6) if vc<0.5 else vy
            Rn[idx]=max(min(vc*0.95,0.18),0.015)
        elif u<0.60:
            # move smallest to largest void
            idx=int(order_small[0])
            # occasionally second smallest too? no
            vx,vy,vc=vlist[0]
            Xn[idx]=np.clip(vx,0.02,0.98); Yn[idx]=np.clip(vy,0.02,0.98)
            # clamp inside with vc
            Xn[idx]=np.clip(Xn[idx],vc*0.5+1e-6,1.0-vc*0.5-1e-6)
            Yn[idx]=np.clip(Yn[idx],vc*0.5+1e-6,1.0-vc*0.5-1e-6)
            Rn[idx]=max(min(vc*0.95,0.18),0.015)
        elif u<0.80:
            # k=2 or 3 smallest to k best voids greedily
            kk=2 if rng.random()<0.6 else 3
            idxs=order_small[:kk]
            for t,idx in enumerate(idxs):
                vx,vy,vc=vlist[t % len(vlist)]
                Xn[int(idx)]=np.clip(vx,0.02,0.98); Yn[int(idx)]=np.clip(vy,0.02,0.98)
                Rn[int(idx)]=max(min(vc*0.9,0.18),0.012)
        else:
            # basin-hop kick: perturb all + move random circle to void
            sig=0.008+rng.random()*0.012
            Xn=Xn+rng.standard_normal(n)*sig; Yn=Yn+rng.standard_normal(n)*sig
            Xn=np.clip(Xn,0.03,0.97); Yn=np.clip(Yn,0.03,0.97)
            if rng.random()<0.5:
                idx=int(rng.integers(0,n))
                vx,vy,vc=vlist[0]
                Xn[idx]=np.clip(vx,0.02,0.98); Yn[idx]=np.clip(vy,0.02,0.98)
        # short penalty re-opt
        # choose W by time: high for accuracy
        Xn2,Yn2,Rn2=refine(Xn,Yn,Rn,2e4,80)
        Rlp=lp_polish(Xn2,Yn2)
        if Rlp is None:
            # try shrink eval
            Rs,_=uniform_shrink(Xn2,Yn2,Rn2)
            new_sum=float(np.sum(Rs))
            Xn2n,Yn2n,Rn2n=Xn2,Yn2,Rs
        else:
            new_sum=float(np.sum(Rlp))
            Xn2n,Yn2n,Rn2n=Xn2,Yn2,Rlp
        delta=new_sum-cur_sum
        accept=False
        if delta>=0:
            accept=True
        else:
            if rng.random()<np.exp(delta/max(T,1e-12)):
                accept=True
        if accept:
            cur_X,cur_Y,cur_R=Xn2n,Yn2n,Rn2n
            cur_sum=new_sum
        if new_sum>best_sum+1e-12:
            best_sum=new_sum; best_X=Xn2n.copy(); best_Y=Yn2n.copy(); best_R=Rn2n.copy()
            since_best=0
        else:
            since_best+=1
        # also try LP-only greedy without refine occasionally (cheap): if void large, direct insert
        if it%25==0 and (time.time()-t0>DEADLINE-5):
            break

    # final safety
    best_R=np.maximum(best_R,1e-6)
    # ensure strict feasibility with extra shrink
    Rs,_=uniform_shrink(best_X,best_Y,best_R)
    # one more tiny factor
    Rs=Rs*(1.0-1e-9)
    arr=np.column_stack([best_X,best_Y,Rs])
    return arr
