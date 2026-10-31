import numpy as np
import time as _time

def solve():
    import time
    import numpy as np
    from scipy.optimize import minimize, linprog
    n = 26
    t0 = time.time()
    DEADLINE = 124.0
    rng = np.random.default_rng(7)

    iu, ju = np.triu_indices(n, 1)
    m = len(iu)
    A0 = np.zeros((m, n), dtype=float)
    ar = np.arange(m)
    A0[ar, iu] = 1.0
    A0[ar, ju] = 1.0
    cobj = -np.ones(n)

    def lp_eval(X, Y):
        X = np.asarray(X, dtype=float); Y = np.asarray(Y, dtype=float)
        wall = np.minimum(np.minimum(X, 1.0 - X), np.minimum(Y, 1.0 - Y))
        if np.any(wall < 1e-9):
            return None, -1.0
        dx = X[iu] - X[ju]; dy = Y[iu] - Y[ju]
        dvec = np.sqrt(dx*dx + dy*dy)
        if np.any(dvec < 1e-9):
            return None, -1.0
        bounds = [(0.0, float(bi)) for bi in wall]
        try:
            res = linprog(cobj, A_ub=A0, b_ub=dvec, bounds=bounds, method='highs')
        except Exception:
            return None, -1.0
        if not res.success or res.x is None:
            return None, -1.0
        r = np.asarray(res.x, dtype=float).flatten()
        r = np.maximum(r, 0.0)
        return r, float(np.sum(r))

    def safe_scale(X, Y, R):
        X = np.asarray(X, float); Y = np.asarray(Y, float); R = np.maximum(np.asarray(R, float), 0.0)
        wall = np.minimum(np.minimum(X, 1.0 - X), np.minimum(Y, 1.0 - Y))
        alpha = 1.0
        mask = R > 1e-12
        if np.any(mask):
            wr = np.min(wall[mask] / R[mask])
            alpha = min(alpha, wr)
            Dx = X[:, None] - X[None, :]
            Dy = Y[:, None] - Y[None, :]
            D = np.sqrt(Dx*Dx + Dy*Dy)
            S = R[:, None] + R[None, :]
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = D / S
            np.fill_diagonal(ratio, np.inf)
            # ignore pairs with zero sum
            ratio[S < 1e-14] = np.inf
            b = np.min(ratio)
            alpha = min(alpha, b)
        alpha = alpha * (1.0 - 1e-7) - 1e-12
        if not np.isfinite(alpha) or alpha <= 0:
            alpha = 0.5
        if alpha > 1.0:
            alpha = 1.0 * (1.0 - 1e-7)
        return R * alpha

    # fallback feasible
    k = int(np.ceil(np.sqrt(n)))
    r0 = 0.5 / k
    fb = []
    for i in range(k):
        for j in range(k):
            if len(fb) < n:
                fb.append([(2*i+1)*r0, (2*j+1)*r0, r0*0.999])
    fb = np.array(fb, dtype=float)
    Xb0, Yb0, Rb0 = fb[:,0], fb[:,1], fb[:,2]
    Rb0 = safe_scale(Xb0, Yb0, Rb0)
    best_sum = float(np.sum(Rb0))
    best_X, best_Y, best_R = Xb0.copy(), Yb0.copy(), Rb0.copy()

    # ---- diverse initial centres ----
    inits = []
    # random uniform
    for _ in range(60):
        inits.append((rng.random(n)*0.9+0.05, rng.random(n)*0.9+0.05))
    # 5x5 +1 perturbed
    for _ in range(10):
        xs = np.zeros(n); ys = np.zeros(n); idx=0
        for i in range(5):
            for j in range(5):
                xs[idx]=0.1+0.2*i+rng.normal(0,0.01); ys[idx]=0.1+0.2*j+rng.normal(0,0.01); idx+=1
        xs[25]=0.85+rng.normal(0,0.02); ys[25]=0.15+rng.normal(0,0.02)
        inits.append((np.clip(xs,0.03,0.97), np.clip(ys,0.03,0.97)))
    # row-count templates
    def comp_init(nrows, counts):
        xs=[]; ys=[]
        for r in range(nrows):
            c = counts[r]
            y = (r+0.5)/nrows + rng.normal(0,0.008)
            off = (0.5/c) if (r%2==1) else 0.0
            for kk in range(c):
                x = (kk+0.5)/c + off + rng.normal(0,0.008)
                if x>0.97: x-=1.0/c
                if x<0.03: x+=1.0/c
                xs.append(x); ys.append(y)
        xs=np.array(xs[:n]); ys=np.array(ys[:n])
        return np.clip(xs,0.03,0.97), np.clip(ys,0.03,0.97)
    fixed_comps = [
        (5,[5,5,6,5,5]),(5,[6,5,5,5,5]),(5,[5,6,5,5,5]),(5,[4,6,6,5,5]),
        (5,[5,5,5,6,5]),(5,[6,6,5,4,5]),(5,[4,5,6,6,5]),(5,[5,4,6,6,5]),
        (6,[4,5,4,5,4,4]),(6,[5,4,5,4,4,4]),(6,[4,4,5,5,4,4]),(6,[5,5,4,4,4,4]),
        (4,[6,7,7,6]),(4,[7,6,6,7]),(4,[6,6,7,7]),
        (5,[6,4,6,5,5]),(5,[5,6,4,6,5]),(6,[4,4,6,4,4,4]),
    ]
    for nr, cc in fixed_comps:
        for _ in range(3):
            inits.append(comp_init(nr, cc))
    # random compositions
    for _ in range(40):
        nr = int(rng.integers(4,7))
        # random counts summing to n, each 3..7
        counts = np.full(nr, n//nr)
        rem = n - int(np.sum(counts))
        ii=0
        while rem>0:
            counts[ii%nr]+=1; rem-=1; ii+=1
        perm = rng.permutation(nr)
        counts = counts[perm]
        # mutate a bit
        for __ in range(3):
            a=int(rng.integers(0,nr)); b=int(rng.integers(0,nr))
            if counts[a]>3 and counts[b]<7:
                counts[a]-=1; counts[b]+=1
        inits.append(comp_init(nr, list(counts)))
    # hex jitter
    for _ in range(10):
        pts=[]
        for ix in range(6):
            for iy in range(5):
                if len(pts)<n:
                    xx=(ix+0.5)/6.0; yy=(iy+0.5)/5.0
                    if iy%2==1:
                        xx+=0.5/6.0*0.5
                        if xx>0.97: xx-=1.0/6.0
                    pts.append((xx,yy))
        a=np.array(pts)
        inits.append((np.clip(a[:,0]+rng.normal(0,0.008,n),0.03,0.97),
                      np.clip(a[:,1]+rng.normal(0,0.008,n),0.03,0.97)))

    scored = []
    for (xx, yy) in inits:
        if time.time()-t0 > 30.0:
            break
        _, s = lp_eval(xx, yy)
        scored.append(s)
        if s > best_sum + 1e-12:
            r, _ = lp_eval(xx, yy)
            if r is not None:
                best_sum = s; best_X = xx.copy(); best_Y = yy.copy(); best_R = r.copy()
    # order inits by score
    try:
        order = np.argsort(np.array(scored))[::-1]
        top_inits = [inits[i] for i in order[:6]]
    except Exception:
        top_inits = [(best_X, best_Y)]

    # ---- short joint jam for top inits ----
    bnds = [(0.0,1.0)]*(2*n)+[(0.02,0.25)]*n
    Pseq = [50.0, 500.0, 5000.0]
    for (tx, ty) in top_inits:
        if time.time()-t0 > 62.0:
            break
        try:
            r0l, _ = lp_eval(tx, ty)
            if r0l is None:
                r0l = np.full(n, 0.09)
            else:
                r0l = np.clip(r0l, 0.02, 0.25)
                # shrink a bit to be interior for penalty start
                r0l = r0l*0.95+0.005
                r0l = np.clip(r0l, 0.02, 0.25)
            z = np.concatenate([tx, ty, r0l])
            for P in Pseq:
                if time.time()-t0 > 62.0:
                    break
                def fg(zz, P=P):
                    x=zz[0:n]; y=zz[n:2*n]; r=zz[2*n:3*n]
                    f=-np.sum(r)
                    gx=np.zeros(n); gy=np.zeros(n); gr=-np.ones(n)
                    v1=r-x; mm=v1>0
                    if np.any(mm):
                        f+=P*np.sum(v1[mm]*v1[mm]); co=2.0*P*v1[mm]; gx[mm]-=co; gr[mm]+=co
                    v2=x+r-1.0; mm=v2>0
                    if np.any(mm):
                        f+=P*np.sum(v2[mm]*v2[mm]); co=2.0*P*v2[mm]; gx[mm]+=co; gr[mm]+=co
                    v3=r-y; mm=v3>0
                    if np.any(mm):
                        f+=P*np.sum(v3[mm]*v3[mm]); co=2.0*P*v3[mm]; gy[mm]-=co; gr[mm]+=co
                    v4=y+r-1.0; mm=v4>0
                    if np.any(mm):
                        f+=P*np.sum(v4[mm]*v4[mm]); co=2.0*P*v4[mm]; gy[mm]+=co; gr[mm]+=co
                    dxm=x[:,None]-x[None,:]; dym=y[:,None]-y[None,:]
                    D=np.sqrt(dxm*dxm+dym*dym+1e-20)
                    S=r[:,None]+r[None,:]-D
                    np.fill_diagonal(S,-1.0)
                    mask=S>0
                    if np.any(mask):
                        f+=P*0.5*np.sum(S[mask]*S[mask])
                        Dm=np.copy(D); np.fill_diagonal(Dm,1.0)
                        W=np.zeros((n,n)); W[mask]=(2.0*P*S[mask])/(Dm[mask]+1e-12)
                        rs=W.sum(axis=1)
                        gx+=W.dot(x)-rs*x; gy+=W.dot(y)-rs*y
                        Sp=np.where(mask,S,0.0); gr+=2.0*P*Sp.sum(axis=1)
                    return f, np.concatenate([gx,gy,gr])
                try:
                    res = minimize(fg, z, method='L-BFGS-B', jac=True, bounds=bnds,
                                   options={'maxiter':150,'ftol':1e-9,'gtol':1e-6})
                    z=res.x
                except Exception:
                    break
            X=np.clip(z[0:n],0.0,1.0); Y=np.clip(z[n:2*n],0.0,1.0)
            r2,s2=lp_eval(X,Y)
            if r2 is not None and s2>best_sum+1e-12:
                best_sum=s2; best_X=X.copy(); best_Y=Y.copy(); best_R=r2.copy()
        except Exception:
            continue

    # ---- hill climbing with void teleport ----
    cur_X, cur_Y = best_X.copy(), best_Y.copy()
    _, cur_sum = lp_eval(cur_X, cur_Y)
    if cur_sum < 0:
        cur_sum = best_sum
    sigmas = [0.016, 0.008, 0.0035, 0.0012]
    gx_grid, gy_grid = np.meshgrid(np.linspace(0.02,0.98,42), np.linspace(0.02,0.98,42))
    gp = np.column_stack([gx_grid.ravel(), gy_grid.ravel()])
    it = 0
    stagn = 0
    while True:
        if time.time()-t0 > 114.0:
            break
        if it > 40000:
            break
        sigma = sigmas[(it//600)%4]
        it+=1
        # propose
        px = cur_X.copy(); py = cur_Y.copy()
        u = rng.random()
        try:
            if u < 0.75:
                idx = int(rng.integers(0,n))
                px[idx]+=rng.normal(0,sigma); py[idx]+=rng.normal(0,sigma)
                px[idx]=np.clip(px[idx],0.01,0.99); py[idx]=np.clip(py[idx],0.01,0.99)
            elif u < 0.9:
                px+=rng.normal(0,sigma*0.25,n); py+=rng.normal(0,sigma*0.25,n)
                px=np.clip(px,0.01,0.99); py=np.clip(py,0.01,0.99)
            else:
                # two-circle move
                a=int(rng.integers(0,n)); b=int(rng.integers(0,n))
                px[a]+=rng.normal(0,sigma); py[a]+=rng.normal(0,sigma)
                px[b]+=rng.normal(0,sigma); py[b]+=rng.normal(0,sigma)
                px=np.clip(px,0.01,0.99); py=np.clip(py,0.01,0.99)
        except Exception:
            continue
        try:
            _, s = lp_eval(px, py)
        except Exception:
            continue
        if s > cur_sum + 1e-12:
            cur_X, cur_Y, cur_sum = px, py, s
            stagn=0
            if s > best_sum + 1e-12:
                rtmp,_ = lp_eval(px,py)
                best_sum=s; best_X=px.copy(); best_Y=py.copy()
                if rtmp is not None: best_R=rtmp.copy()
        else:
            stagn+=1
        # void teleport on stagnation
        if stagn >= 450:
            stagn=0
            try:
                # largest void using best_R geometry at cur
                rc,_ = lp_eval(cur_X, cur_Y)
                if rc is None: rc=best_R
                # vectorized clearance
                dx = gp[:,None,0]-cur_X[None,:] if False else None
                # manual loop in chunks for memory
                best_pt=None; best_cl=-1
                # chunk grid
                for s0 in range(0, gp.shape[0], 441):
                    chunk=gp[s0:s0+441]
                    Dx=chunk[:,None,0]-cur_X[None,:] if False else (chunk[:,None,0] if False else None)
                    # compute properly
                    Dx = chunk[:,0,None]-cur_X[None,:]
                    Dy = chunk[:,1,None]-cur_Y[None,:]
                    Dd=np.sqrt(Dx*Dx+Dy*Dy)
                    clear = np.min(Dd-rc[None,:], axis=1)
                    wallc=np.minimum(np.minimum(chunk[:,0],1-chunk[:,0]),np.minimum(chunk[:,1],1-chunk[:,1]))
                    clear=np.minimum(clear,wallc)
                    j=int(np.argmax(clear))
                    if clear[j]>best_cl:
                        best_cl=float(clear[j]); best_pt=chunk[j].copy()
                # teleport smallest circle
                sid=int(np.argmin(rc))
                nx=cur_X.copy(); ny=cur_Y.copy()
                nx[sid]=float(np.clip(best_pt[0]+rng.normal(0,0.003),0.01,0.99))
                ny[sid]=float(np.clip(best_pt[1]+rng.normal(0,0.003),0.01,0.99))
                # also jitter a neighbour
                _, s2 = lp_eval(nx,ny)
                cur_X,cur_Y=nx,ny
                _, cur_sum = lp_eval(cur_X,cur_Y)
                if s2>best_sum+1e-12:
                    r2,_=lp_eval(nx,ny)
                    best_sum=s2; best_X=nx.copy(); best_Y=ny.copy()
                    if r2 is not None: best_R=r2.copy()
                # if teleport made things much worse, restart from best with kick
                if cur_sum < best_sum-0.08:
                    cur_X=best_X.copy()+rng.normal(0,0.008,n); cur_Y=best_Y.copy()+rng.normal(0,0.008,n)
                    cur_X=np.clip(cur_X,0.01,0.99); cur_Y=np.clip(cur_Y,0.01,0.99)
                    _,cur_sum=lp_eval(cur_X,cur_Y)
            except Exception:
                cur_X,cur_Y=best_X.copy(),best_Y.copy()
                _,cur_sum=lp_eval(cur_X,cur_Y)
                continue

    # ---- final jam from best ----
    try:
        if time.time()-t0 < 121.0:
            z = np.concatenate([best_X, best_Y, np.clip(best_R*0.96+0.004,0.02,0.25)])
            def fg2(zz):
                P=8000.0
                x=zz[0:n]; y=zz[n:2*n]; r=zz[2*n:3*n]
                f=-np.sum(r)
                gx=np.zeros(n); gy=np.zeros(n); gr=-np.ones(n)
                v1=r-x; mm=v1>0
                if np.any(mm):
                    f+=P*np.sum(v1[mm]*v1[mm]); co=2.0*P*v1[mm]; gx[mm]-=co; gr[mm]+=co
                v2=x+r-1.0; mm=v2>0
                if np.any(mm):
                    f+=P*np.sum(v2[mm]*v2[mm]); co=2.0*P*v2[mm]; gx[mm]+=co; gr[mm]+=co
                v3=r-y; mm=v3>0
                if np.any(mm):
                    f+=P*np.sum(v3[mm]*v3[mm]); co=2.0*P*v3[mm]; gy[mm]-=co; gr[mm]+=co
                v4=y+r-1.0; mm=v4>0
                if np.any(mm):
                    f+=P*np.sum(v4[mm]*v4[mm]); co=2.0*P*v4[mm]; gy[mm]+=co; gr[mm]+=co
                dxm=x[:,None]-x[None,:]; dym=y[:,None]-y[None,:]
                D=np.sqrt(dxm*dxm+dym*dym+1e-20)
                S=r[:,None]+r[None,:]-D
                np.fill_diagonal(S,-1.0)
                mask=S>0
                if np.any(mask):
                    f+=P*0.5*np.sum(S[mask]*S[mask])
                    Dm=np.copy(D); np.fill_diagonal(Dm,1.0)
                    W=np.zeros((n,n)); W[mask]=(2.0*P*S[mask])/(Dm[mask]+1e-12)
                    rs=W.sum(axis=1)
                    gx+=W.dot(x)-rs*x; gy+=W.dot(y)-rs*y
                    Sp=np.where(mask,S,0.0); gr+=2.0*P*Sp.sum(axis=1)
                return f, np.concatenate([gx,gy,gr])
            try:
                res=minimize(fg2,z,method='L-BFGS-B',jac=True,bounds=bnds,options={'maxiter':200,'ftol':1e-10,'gtol':1e-7})
                z=res.x
                X=np.clip(z[0:n],0,1); Y=np.clip(z[n:2*n],0,1)
                r2,s2=lp_eval(X,Y)
                if r2 is not None and s2>best_sum+1e-12:
                    best_sum=s2; best_X=X.copy(); best_Y=Y.copy(); best_R=r2.copy()
            except Exception:
                pass
    except Exception:
        pass

    # final safety
    try:
        rfin,_ = lp_eval(best_X, best_Y)
        if rfin is not None:
            best_R = rfin
        best_R = safe_scale(best_X, best_Y, best_R)
        out = np.column_stack([best_X, best_Y, best_R])
        return out
    except Exception:
        best_R = safe_scale(best_X, best_Y, best_R)
        return np.column_stack([best_X, best_Y, best_R])
