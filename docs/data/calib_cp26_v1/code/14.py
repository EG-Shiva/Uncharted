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

    # pair matrix (fixed)
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
        a = np.min(ratio_w)
        b = np.min(ratio_p)
        alpha = min(1.0, float(a), float(b))
        alpha = alpha*(1.0-1e-7)-1e-12
        if not np.isfinite(alpha) or alpha <= 0:
            alpha = 0.5
        return R*alpha, alpha

    # initial feasible fallback 5x5+1 grid
    k5 = 5
    r0 = 0.5/k5*0.999
    pts = []
    for i in range(k5):
        for j in range(k5):
            pts.append([(2*i+1)*r0, (2*j+1)*r0])
    pts = np.array(pts)  # 25
    # add 26th
    extra = np.array([[0.5, 0.5]])
    Xf0 = np.concatenate([pts[:,0], extra[:,0]])
    Yf0 = np.concatenate([pts[:,1], extra[:,1]])
    Rf0 = np.full(n, r0*0.9)
    Rs0,_ = uniform_scale(Xf0, Yf0, Rf0)
    best_arr = np.column_stack([Xf0, Yf0, Rs0])
    best_sum = float(np.sum(Rs0))
    best_x = Xf0.copy(); best_y = Yf0.copy(); best_r = Rs0.copy()

    def lp_opt(x, y, eps=1e-9):
        wall = np.minimum(np.minimum(x, 1.0-x), np.minimum(y, 1.0-y))
        if np.any(wall <= 1e-9):
            return None, -1.0
        # quick prune: need reasonable sizes to beat current best?
        # allow any but skip tiny walls
        if np.min(wall) < 0.015:
            # still evaluate but likely bad; skip if best already high
            if best_sum > 2.4:
                # check if even upper bound can beat best
                if np.sum(wall) < best_sum:
                    # upper bound is sum wall, cannot beat
                    return None, -1.0
        dx = x[pi]-x[pj]
        dy = y[pi]-y[pj]
        b_pair = np.sqrt(dx*dx+dy*dy) - eps
        # infeasible lower bound check: if any dist <=0 skip
        if np.any(b_pair <= 0):
            return None, -1.0
        # upper bound check
        # if sum of walls already < best, skip? already above
        bounds = [(0.0, float(w - eps*0.5)) if w > eps else (0.0, 0.0) for w in wall]
        # quick necessary: sum(min(wall, ...)) < best -> skip? use sum wall
        try:
            res = linprog(c_obj, A_ub=A_pair, b_ub=b_pair, bounds=bounds, method='highs')
        except Exception:
            return None, -1.0
        if not res.success:
            return None, -1.0
        r = np.array(res.x, dtype=float)
        # shrink for strict feasibility
        r = r * (1.0 - 5e-9)
        r = np.maximum(r, 0.0)
        s = float(np.sum(r))
        return r, s

    def consider(x, y):
        global_best = [best_sum]
        r, s = lp_opt(x, y)
        return r, s

    # ---- Phase 1: pool of candidates ----
    # generate diverse centres and evaluate LP
    def gen_grid5(jit=0.008):
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
    def gen_hex():
        pts=[]
        for ix in range(6):
            for iy in range(5):
                if len(pts)<n:
                    xx=(ix+0.5)/6.0
                    yy=(iy+0.5)/5.0
                    if iy%2==1:
                        xx+=0.5/6.0*0.5
                        if xx>0.97: xx-=1.0/6.0
                    pts.append((xx,yy))
        a=np.array(pts)
        a[:,0]+=rng.normal(0,0.008,n)
        a[:,1]+=rng.normal(0,0.008,n)
        return np.clip(a[:,0],0.03,0.97), np.clip(a[:,1],0.03,0.97)
    def gen_rand():
        return rng.random(n)*0.94+0.03, rng.random(n)*0.94+0.03

    npool = 120
    for trial in range(npool):
        if time.time()-t0 > DEADLINE-70:
            break
        if trial % 3 == 0:
            x0,y0 = gen_grid5(0.01)
        elif trial % 3 == 1:
            x0,y0 = gen_hex()
        else:
            x0,y0 = gen_rand()
        r,s = lp_opt(x0,y0)
        if r is not None and s > best_sum:
            best_sum = s; best_x=x0.copy(); best_y=y0.copy(); best_r=r.copy()
            best_arr = np.column_stack([best_x,best_y,best_r])

    # ---- Phase 2: short joint penalised L-BFGS to diversify ----
    def joint_seed():
        X0 = rng.random(n)*0.9+0.05
        Y0 = rng.random(n)*0.9+0.05
        R0 = 0.08+0.03*rng.random(n)
        return np.concatenate([X0,Y0,R0])
    # penalty fun factory
    def make_pen():
        def fun(v,W):
            X=v[0:n]; Y=v[n:2*n]; R=v[2*n:3*n]
            gx=np.zeros(n); gy=np.zeros(n); pen_gr=np.zeros(n)
            pen=0.0
            v1=R-X; m1=v1>0
            if np.any(m1):
                pen+=np.sum(v1[m1]**2); gx[m1]+=-2*v1[m1]; pen_gr[m1]+=2*v1[m1]
            v2=X+R-1.0; m2=v2>0
            if np.any(m2):
                pen+=np.sum(v2[m2]**2); gx[m2]+=2*v2[m2]; pen_gr[m2]+=2*v2[m2]
            v3=R-Y; m3=v3>0
            if np.any(m3):
                pen+=np.sum(v3[m3]**2); gy[m3]+=-2*v3[m3]; pen_gr[m3]+=2*v3[m3]
            v4=Y+R-1.0; m4=v4>0
            if np.any(m4):
                pen+=np.sum(v4[m4]**2); gy[m4]+=2*v4[m4]; pen_gr[m4]+=2*v4[m4]
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
                pen_gr+=np.sum(G,axis=1)
            loss=-np.sum(R)+W*pen
            g=np.empty_like(v)
            g[0:n]=W*gx; g[n:2*n]=W*gy; g[2*n:3*n]=-1.0+W*pen_gr
            return loss,g
        return fun
    penfun=make_pen()
    boundsJR=[(0.0,1.0)]*(2*n)+[(0.03,0.20)]*n
    for restart in range(12):
        if time.time()-t0>DEADLINE-55:
            break
        if restart==0:
            X0,Y0=gen_grid5(0.005)
            R0=np.full(n,0.095)
            v=np.concatenate([X0,Y0,R0])
        else:
            v=joint_seed()
        for W in (500.0, 8000.0, 80000.0):
            if time.time()-t0>DEADLINE-55:
                break
            try:
                res=minimize(penfun,v,args=(W,),jac=True,method='L-BFGS-B',bounds=boundsJR,options={'maxiter':120,'ftol':1e-9,'gtol':1e-5})
                v=res.x
            except Exception:
                break
        X=np.clip(v[0:n],0.0,1.0); Y=np.clip(v[n:2*n],0.0,1.0)
        X=np.clip(X,0.03,0.97); Y=np.clip(Y,0.03,0.97)
        r,s=lp_opt(X,Y)
        if r is not None and s>best_sum:
            best_sum=s; best_x=X.copy(); best_y=Y.copy(); best_r=r.copy()
            best_arr=np.column_stack([best_x,best_y,best_r])

    # ---- Phase 3: basin-hopping hill climbing on LP objective ----
    cur_x=best_x.copy(); cur_y=best_y.copy(); cur_r=best_r.copy(); cur_sum=best_sum
    # fine: ensure cur evaluated
    # hopping
    sigmas = [0.02, 0.012, 0.007, 0.0035, 0.0015, 0.0007]
    stall=0
    it=0
    # total iterations limited by time
    while time.time()-t0 < DEADLINE-3:
        # pick sigma cyclically with decay on stall
        sigma = sigmas[it % len(sigmas)]
        # occasionally global kick
        if stall>600:
            # kick all
            nx = cur_x + rng.normal(0,0.012,n)
            ny = cur_y + rng.normal(0,0.012,n)
            nx=np.clip(nx,0.03,0.97); ny=np.clip(ny,0.03,0.97)
            r,s=lp_opt(nx,ny)
            it+=1
            if r is not None and s>cur_sum:
                cur_x,cur_y,cur_r,cur_sum=nx,ny,r,s
                stall=0
                if s>best_sum:
                    best_sum=s; best_x=nx.copy(); best_y=ny.copy(); best_r=r.copy()
                    best_arr=np.column_stack([best_x,best_y,best_r])
            else:
                # random restart from best with kick if stuck long
                if stall>1500:
                    nx = best_x + rng.normal(0,0.02,n)
                    ny = best_y + rng.normal(0,0.02,n)
                    nx=np.clip(nx,0.03,0.97); ny=np.clip(ny,0.03,0.97)
                    r,s=lp_opt(nx,ny)
                    if r is not None:
                        cur_x,cur_y=nx,ny
                        if s>cur_sum:
                            cur_r,cur_sum=r,s
                        # even if worse, move to explore (accept)
                        else:
                            # accept as new cur with its radii if feasible
                            cur_r,cur_sum=r,s
                        stall=0
                        if s>best_sum:
                            best_sum=s; best_x=nx.copy(); best_y=ny.copy(); best_r=r.copy()
                            best_arr=np.column_stack([best_x,best_y,best_r])
                    else:
                        stall=0
            continue
        # single or double point move
        if rng.random()<0.8:
            idxs=[int(rng.integers(0,n))]
        else:
            idxs=rng.choice(n,size=2,replace=False)
        nx=cur_x.copy(); ny=cur_y.copy()
        for ii in idxs:
            nx[ii]+=rng.normal(0,sigma)
            ny[ii]+=rng.normal(0,sigma)
        nx=np.clip(nx,0.03,0.97); ny=np.clip(ny,0.03,0.97)
        r,s=lp_opt(nx,ny)
        it+=1
        if r is not None and s>cur_sum+1e-12:
            cur_x,cur_y,cur_r,cur_sum=nx,ny,r,s
            stall=0
            if s>best_sum+1e-12:
                best_sum=s; best_x=nx.copy(); best_y=ny.copy(); best_r=r.copy()
                best_arr=np.column_stack([best_x,best_y,best_r])
        else:
            stall+=1
        # occasional all-point fine tune
        if it%400==0:
            nx=cur_x+rng.normal(0,0.0015,n)
            ny=cur_y+rng.normal(0,0.0015,n)
            nx=np.clip(nx,0.03,0.97); ny=np.clip(ny,0.03,0.97)
            r,s=lp_opt(nx,ny)
            if r is not None and s>cur_sum:
                cur_x,cur_y,cur_r,cur_sum=nx,ny,r,s
                stall=0
                if s>best_sum:
                    best_sum=s; best_x=nx.copy(); best_y=ny.copy(); best_r=r.copy()
                    best_arr=np.column_stack([best_x,best_y,best_r])

    # final safety shrink
    try:
        Xf=best_arr[:,0]; Yf=best_arr[:,1]; Rf=best_arr[:,2]
        Rfs,_=uniform_scale(Xf,Yf,Rf)
        best_arr=np.column_stack([Xf,Yf,Rfs])
    except Exception:
        pass
    return best_arr
