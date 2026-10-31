import numpy as np
import time as _time

def solve():
    import time as _tm
    import numpy as _np
    from scipy.optimize import minimize, linprog, least_squares
    try:
        from scipy.spatial import Delaunay as _Del
    except Exception:
        _Del = None
    n = 26
    t0 = _tm.time()
    HARD = 118.0
    def time_left():
        return HARD - (_tm.time() - t0)

    _np.random.seed(0)
    iu, ju = _np.triu_indices(n, 1)
    m = len(iu)
    A0 = _np.zeros((m, n), dtype=float)
    _ar = _np.arange(m)
    A0[_ar, iu] = 1.0
    A0[_ar, ju] = 1.0
    cobj = -_np.ones(n)

    def eval_lp(xc, yc, eps=5e-7):
        b = _np.minimum(_np.minimum(xc, 1.0-xc), _np.minimum(yc, 1.0-yc))
        b = b - eps
        b = _np.maximum(b, 0.0)
        dx = xc[iu]-xc[ju]; dy = yc[iu]-yc[ju]
        dvec = _np.sqrt(dx*dx+dy*dy) - eps
        bounds = [(0.0, float(bi)) for bi in b]
        try:
            res = linprog(cobj, A_ub=A0, b_ub=dvec, bounds=bounds, method='highs')
        except Exception:
            return None, -1.0
        if not res.success or res.x is None:
            return None, -1.0
        r = _np.asarray(res.x, dtype=float).flatten()
        r = _np.maximum(r, 0.0)
        return r, float(_np.sum(r))

    def uniform_scale(X, Y, R):
        R = _np.maximum(R, 1e-12)
        wr = _np.minimum(_np.minimum(X, 1.0-X), _np.minimum(Y, 1.0-Y)) / R
        Dx = X[:,None]-X[None,:]; Dy = Y[:,None]-Y[None,:]
        D = _np.sqrt(Dx*Dx+Dy*Dy)
        S = R[:,None]+R[None,:]
        with _np.errstate(divide='ignore', invalid='ignore'):
            ratio = D / S
        _np.fill_diagonal(ratio, _np.inf)
        a = _np.min(wr); b_ = _np.min(ratio)
        alpha = min(1.0, a, b_)
        alpha = alpha*(1.0-1e-7)-1e-12
        if alpha<=0 or not _np.isfinite(alpha):
            alpha=0.5
        return R*alpha

    # fallback grid
    k = int(_np.ceil(_np.sqrt(n)))
    r0f = 0.5/k
    pts=[]
    for i in range(k):
        for j in range(k):
            if len(pts)<n:
                pts.append([(2*i+1)*r0f,(2*j+1)*r0f,r0f*0.999])
    best_arr = _np.array(pts,dtype=float)
    Xb,Yb,Rb = best_arr[:,0],best_arr[:,1],best_arr[:,2]
    Rb2 = uniform_scale(Xb,Yb,Rb)
    best_arr=_np.column_stack([Xb,Yb,Rb2])
    best_sum=float(_np.sum(Rb2))
    best_x=Xb.copy(); best_y=Yb.copy(); best_r=Rb2.copy()

    bnds=[(0.0,1.0)]*(2*n)+[(1e-4,0.35)]*n
    P_seq=[50.0,500.0,5000.0]

    def run_penalty(x0,y0,r0,maxit=120):
        z=_np.concatenate([x0,y0,r0])
        for P in P_seq:
            if _tm.time()-t0>HARD-55:
                break
            def fg(zz,P=P):
                x=zz[0:n]; y=zz[n:2*n]; r=zz[2*n:3*n]
                f=-_np.sum(r)
                gx=_np.zeros(n); gy=_np.zeros(n); gr=-_np.ones(n)
                v1=r-x; mm=v1>0
                if _np.any(mm):
                    f+=P*_np.sum(v1[mm]*v1[mm]); co=2.0*P*v1[mm]; gx[mm]-=co; gr[mm]+=co
                v2=x+r-1.0; mm=v2>0
                if _np.any(mm):
                    f+=P*_np.sum(v2[mm]*v2[mm]); co=2.0*P*v2[mm]; gx[mm]+=co; gr[mm]+=co
                v3=r-y; mm=v3>0
                if _np.any(mm):
                    f+=P*_np.sum(v3[mm]*v3[mm]); co=2.0*P*v3[mm]; gy[mm]-=co; gr[mm]+=co
                v4=y+r-1.0; mm=v4>0
                if _np.any(mm):
                    f+=P*_np.sum(v4[mm]*v4[mm]); co=2.0*P*v4[mm]; gy[mm]+=co; gr[mm]+=co
                dxm=x[:,None]-x[None,:]; dym=y[:,None]-y[None,:]
                D=_np.sqrt(dxm*dxm+dym*dym)
                Rsum=r[:,None]+r[None,:]
                S=Rsum-D; _np.fill_diagonal(S,-1.0)
                mask=S>0
                if _np.any(mask):
                    f+=P*0.5*_np.sum(S[mask]*S[mask])
                    W=_np.zeros((n,n)); W[mask]=(2.0*P*S[mask])/(_np.maximum(D[mask],1e-12))
                    rs=W.sum(axis=1)
                    gx+=W.dot(x)-rs*x; gy+=W.dot(y)-rs*y
                    Spos=_np.where(mask,S,0.0); gr+=2.0*P*Spos.sum(axis=1)
                return f,_np.concatenate([gx,gy,gr])
            try:
                res=minimize(fg,z,method='L-BFGS-B',jac=True,bounds=bnds,options={'maxiter':maxit,'ftol':1e-10,'gtol':1e-6})
                z=res.x
            except Exception:
                break
        return z

    # Phase A: fast penalty restarts + lattice LP seeds
    rng=_np.random.RandomState(0)
    n_restart=0
    # lattice seeds evaluated immediately
    try:
        # 5x5+1 grid
        gx5=[];gy5=[]
        for i in range(5):
            for j in range(5):
                gx5.append((i+0.5)/5.0); gy5.append((j+0.5)/5.0)
        gx5=_np.array(gx5); gy5=_np.array(gy5)
        for extra in [(0.9,0.1),(0.1,0.9),(0.5,0.5)]:
            xc=_np.concatenate([gx5,[extra[0]]]); yc=_np.concatenate([gy5,[extra[1]]])
            xc+=rng.randn(n)*0.003; yc+=rng.randn(n)*0.003
            xc=_np.clip(xc,0.05,0.95); yc=_np.clip(yc,0.05,0.95)
            r,s=eval_lp(xc,yc)
            if s>best_sum and r is not None:
                best_sum=s; best_x=xc.copy(); best_y=yc.copy(); best_r=r.copy()
                best_arr=_np.column_stack([xc,yc,r])
    except Exception:
        pass

    while _tm.time()-t0 < 48.0 and n_restart<40:
        mode=n_restart%5
        if mode==0:
            X0=rng.rand(n)*0.9+0.05; Y0=rng.rand(n)*0.9+0.05; R0=0.085+0.025*rng.rand(n)
        elif mode==1:
            X0=_np.zeros(n);Y0=_np.zeros(n)
            idx=0
            for i in range(5):
                for j in range(5):
                    X0[idx]=(i+0.5)/5.0+rng.randn()*0.006; Y0[idx]=(j+0.5)/5.0+rng.randn()*0.006; idx+=1
            X0[25]=0.9+rng.randn()*0.01; Y0[25]=0.1+rng.randn()*0.01
            X0=_np.clip(X0,0.05,0.95);Y0=_np.clip(Y0,0.05,0.95); R0=0.08+0.03*rng.rand(n)
        elif mode==2:
            # hex 6x5 subset
            pl=[]
            for ix in range(6):
                for iy in range(5):
                    if len(pl)<n:
                        xx=(ix+0.5)/6.0; yy=(iy+0.5)/5.0
                        if iy%2==1:
                            xx+=0.5/6.0*0.5
                            if xx>0.95: xx-=1.0/6.0
                        pl.append((xx,yy))
            arr=_np.array(pl); X0=arr[:,0]+rng.randn(n)*0.006; Y0=arr[:,1]+rng.randn(n)*0.006
            X0=_np.clip(X0,0.05,0.95);Y0=_np.clip(Y0,0.05,0.95); R0=0.08+0.03*rng.rand(n)
        else:
            X0=rng.rand(n)*0.85+0.075; Y0=rng.rand(n)*0.85+0.075; R0=0.07+0.04*rng.rand(n)
        try:
            z=run_penalty(X0,Y0,R0,maxit=100)
            X=_np.clip(z[0:n],0.0,1.0); Y=_np.clip(z[n:2*n],0.0,1.0); R=_np.clip(z[2*n:3*n],0.01,0.3)
            Rs=uniform_scale(X,Y,R); sU=float(_np.sum(Rs))
            if sU>best_sum:
                best_sum=sU; best_x=X.copy(); best_y=Y.copy(); best_r=Rs.copy()
                best_arr=_np.column_stack([X,Y,Rs])
            rL,sL=eval_lp(X,Y)
            if rL is not None and sL>best_sum:
                # quick feasibility double check
                best_sum=sL; best_x=X.copy(); best_y=Y.copy(); best_r=rL.copy()
                best_arr=_np.column_stack([X,Y,rL])
        except Exception:
            pass
        n_restart+=1

    # helpers for graph phase
    def delaunay_edges(xc,yc):
        try:
            if _Del is None:
                raise RuntimeError("no del")
            pts=_np.column_stack([xc,yc])
            # jitter duplicates
            tri=_Del(pts)
            es=set()
            for s in tri.simplices:
                a,b,c=int(s[0]),int(s[1]),int(s[2])
                for u,v in ((a,b),(b,c),(c,a)):
                    if u>v: u,v=v,u
                    es.add((u,v))
            return list(es)
        except Exception:
            # kNN fallback: 6 nearest per node
            Dx=xc[:,None]-xc[None,:]; Dy=yc[:,None]-yc[None,:]
            D=Dx*Dx+Dy*Dy
            es=set()
            for i in range(n):
                idx=_np.argsort(D[i])[1:7]
                for j in idx:
                    u,v=(i,int(j)) if i<int(j) else (int(j),i)
                    es.add((u,v))
            return list(es)

    def select_walls(xc,yc,rc):
        walls=[]
        for i in range(n):
            dl=xc[i]; dr=1.0-xc[i]; db=yc[i]; dt=1.0-yc[i]
            m=min(dl,dr,db,dt)
            # only near-boundary circles get a wall contact
            if m - rc[i] < 0.03:
                if m==dl: w=0
                elif m==dr: w=1
                elif m==db: w=2
                else: w=3
                walls.append((i,w))
        # ensure at least 6 walls
        if len(walls)<6:
            order=_np.argsort(_np.minimum(_np.minimum(xc,1-xc),_np.minimum(yc,1-yc))-rc)
            for i in order:
                if (i,0) in walls or (i,1) in walls or (i,2) in walls or (i,3) in walls:
                    continue
                dl=xc[i]; dr=1-xc[i]; db=yc[i]; dt=1-yc[i]
                m=min(dl,dr,db,dt)
                if m==dl: w=0
                elif m==dr: w=1
                elif m==db: w=2
                else: w=3
                walls.append((int(i),w))
                if len(walls)>=8: break
        return walls

    def rigidity_count(xc,yc,rc,tol=0.002):
        cnt=0
        b=_np.minimum(_np.minimum(xc,1-xc),_np.minimum(yc,1-yc))-rc
        cnt+=_np.sum(b<tol)
        dx=xc[iu]-xc[ju]; dy=yc[iu]-yc[ju]
        d=_np.sqrt(dx*dx+dy*dy)-(rc[iu]+rc[ju])
        cnt+=_np.sum(d<tol)
        # per-circle >=3 contacts?
        per=_np.zeros(n,dtype=int)
        for k in range(m):
            if d[k]<tol:
                per[iu[k]]+=1; per[ju[k]]+=1
        for i in range(n):
            if b[i]<tol: per[i]+=1
        cnt3=_np.sum(per>=3)
        return int(cnt), int(cnt3)

    def graph_solve(edges,walls,x0,y0,r0):
        v0=_np.concatenate([x0,y0,r0])
        # residual
        def res(v):
            X=v[0:n]; Y=v[n:2*n]; R=v[2*n:3*n]
            out=[]
            for (i,j) in edges:
                out.append((X[i]-X[j])**2+(Y[i]-Y[j])**2-(R[i]+R[j])**2)
            for (i,w) in walls:
                if w==0: out.append(X[i]-R[i])
                elif w==1: out.append(1.0-X[i]-R[i])
                elif w==2: out.append(Y[i]-R[i])
                else: out.append(1.0-Y[i]-R[i])
            return _np.array(out,dtype=float)
        lb=_np.concatenate([_np.zeros(2*n), _np.full(n,0.02)])
        ub=_np.concatenate([_np.ones(2*n), _np.full(n,0.25)])
        try:
            sol=least_squares(res,v0,bounds=(lb,ub),method='trf',ftol=1e-9,xtol=1e-9,gtol=1e-9,max_nfev=120,verbose=0)
            v=sol.x
            X=v[0:n]; Y=v[n:2*n]; R=v[2*n:3*n]
            if _np.any(R<0): return None
            # reject wild
            if _np.any(R<0.005): return None
            cost=float(_np.sum(res(v)**2))
            return X,Y,R,cost
        except Exception:
            return None

    # collect seeds for graph repair
    seeds=[(best_x.copy(),best_y.copy(),best_r.copy())]
    # add 2 perturbed lattice seeds scored by LP
    try:
        for sdx in range(2):
            if _tm.time()-t0>HARD-60: break
            Xp=best_x+rng.randn(n)*0.01; Yp=best_y+rng.randn(n)*0.01
            Xp=_np.clip(Xp,0.05,0.95); Yp=_np.clip(Yp,0.05,0.95)
            r,s=eval_lp(Xp,Yp)
            if r is not None and s>best_sum-0.05:
                seeds.append((Xp.copy(),Yp.copy(),r.copy()))
    except Exception:
        pass

    rng2=_np.random.RandomState(1)
    # graph-repair loop
    for si,(sxc,syc,src) in enumerate(list(seeds)):
        if _tm.time()-t0>HARD-8: break
        xc=sxc.copy(); yc=syc.copy(); rc=src.copy()
        r0,s_cur=eval_lp(xc,yc)
        if r0 is not None:
            rc=r0.copy(); s_cur=float(s_cur)
            if s_cur>best_sum:
                best_sum=s_cur; best_x=xc.copy(); best_y=yc.copy(); best_r=rc.copy()
                best_arr=_np.column_stack([xc,yc,rc])
        else:
            s_cur=float(_np.sum(rc))
        edges=delaunay_edges(xc,yc)
        # keep spanning set: closest ~44 edges by gap
        try:
            dx=xc[iu]-xc[ju]; dy=yc[iu]-yc[ju]
            dd=_np.sqrt(dx*dx+dy*dy)-(rc[iu]+rc[ju])
            eset=set(edges)
            # add closest non-edges if too few, prune loosest if too many
            if len(edges)>48:
                # score edges by gap
                mp={}
                for k in range(m):
                    mp[(min(int(iu[k]),int(ju[k])),max(int(iu[k]),int(ju[k])))]=dd[k]
                edges=sorted(edges,key=lambda e: mp.get(e,0))[:48]
            elif len(edges)<40:
                order=_np.argsort(dd)
                eset=set(edges)
                for k in order:
                    e=(min(int(iu[k]),int(ju[k])),max(int(iu[k]),int(ju[k])))
                    if e not in eset:
                        eset.add(e); edges.append(e)
                    if len(edges)>=44: break
        except Exception:
            pass
        walls=select_walls(xc,yc,rc)
        _,rig3=rigidity_count(xc,yc,rc)
        it=0
        while it<60 and _tm.time()-t0<HARD-6:
            it+=1
            # propose modification every other iter, else resolve same graph from perturbed start
            mod_edges=list(edges); mod_walls=list(walls)
            do_flip=(it%2==0)
            if do_flip:
                choice=rng2.rand()
                if choice<0.6 and len(mod_edges)>5:
                    # replace loosest edge with closest non-edge
                    try:
                        Xc=xc; Yc=yc; Rc=rc
                        # gap for current edges
                        egap=[]
                        for (i,j) in mod_edges:
                            d=_np.sqrt((Xc[i]-Xc[j])**2+(Yc[i]-Yc[j])**2)-(Rc[i]+Rc[j])
                            egap.append(d)
                        rem=int(_np.argmax(egap))
                        eset=set(mod_edges)
                        # closest non-edge
                        dx2=xc[iu]-xc[ju]; dy2=yc[iu]-yc[ju]
                        dd2=_np.sqrt(dx2*dx2+dy2*dy2)-(rc[iu]+rc[ju])
                        order=_np.argsort(dd2)
                        added=False
                        for k in order:
                            e=(min(int(iu[k]),int(ju[k])),max(int(iu[k]),int(ju[k])))
                            if e not in eset:
                                mod_edges[rem]=e
                                added=True
                                break
                        if not added:
                            # random flip
                            a=rng2.integers(0,len(mod_edges)); b=rng2.integers(0,n)
                            c_2=rng2.integers(0,n)
                            if b!=c_2:
                                mod_edges[a]=(min(b,c_2),max(b,c_2))
                    except Exception:
                        pass
                else:
                    # replace one wall contact
                    try:
                        if len(mod_walls)>0 and rng2.rand()<0.8:
                            idx=rng2.integers(0,len(mod_walls))
                            ii,w0=mod_walls[idx]
                            nw=int(rng2.integers(0,4))
                            if nw==w0: nw=(nw+1)%4
                            mod_walls[idx]=(ii,nw)
                        else:
                            ii=int(rng2.integers(0,n))
                            nw=int(rng2.integers(0,4))
                            # avoid duplicate
                            if not any(a==ii for a,_ in mod_walls):
                                mod_walls.append((ii,nw))
                    except Exception:
                        pass
            # multiple perturbed starts (2, or 1 if late)
            ntry=2 if time_left()>20 else 1
            improved=False
            for tr in range(ntry):
                if _tm.time()-t0>HARD-5: break
                sig=0.004 if tr==0 else 0.010
                x0=_np.clip(xc+rng2.randn(n)*sig,0.02,0.98)
                y0=_np.clip(yc+rng2.randn(n)*sig,0.02,0.98)
                r0_= _np.clip(rc+rng2.randn(n)*0.004,0.025,0.20)
                got=graph_solve(mod_edges,mod_walls,x0,y0,r0_)
                if got is None: continue
                X1,Y1,R1,cost=got
                # reject non-graph big overlaps: check min pair ratio
                if _np.any(R1<=0): continue
                rL,sL=eval_lp(X1,Y1)
                if rL is None: continue
                c1,_=rigidity_count(X1,rL if False else Y1,rL) if False else (None,None)
                # compute rigidity properly
                tot,c3=rigidity_count(X1,Y1,rL)
                # retain if improves or rigidity up without losing >0.0002
                if sL>best_sum:
                    best_sum=float(sL); best_x=X1.copy(); best_y=Y1.copy(); best_r=rL.copy()
                    best_arr=_np.column_stack([X1,Y1,rL])
                if (sL>s_cur-1e-12) or (tot>=0 and sL>=s_cur-0.0002 and c3>rig3):
                    # accept as new centre
                    xc,yc,rc=X1.copy(),Y1.copy(),rL.copy()
                    s_cur=float(sL); rig3=int(c3)
                    edges=list(mod_edges); walls=list(mod_walls)
                    # refresh delaunay occasionally to track geometry
                    if it%7==0:
                        try:
                            de=delaunay_edges(xc,yc)
                            # merge: keep accepted + a few fresh
                            eset=set(edges)
                            for e in de:
                                if e not in eset and len(edges)<50:
                                    # add if tight
                                    i,j=e
                                    d=_np.sqrt((xc[i]-xc[j])**2+(yc[i]-yc[j])**2)-(rc[i]+rc[j])
                                    if d<0.02:
                                        edges.append(e); eset.add(e)
                        except Exception:
                            pass
                    improved=True
                    break
            if not improved and do_flip:
                # try resolving original graph once from small perturb (exploit)
                pass
            # checkpoint safety: ensure best_arr valid
            if it%10==0 and _tm.time()-t0>HARD-10:
                break

    # final hill-climb on centres with LP score (fast)
    try:
        sigmas=[0.006,0.003,0.0015]
        for sig in sigmas:
            if _tm.time()-t0>HARD-4: break
            for _ in range(120):
                if _tm.time()-t0>HARD-3: break
                i=int(rng.integers(0,n))
                xn=best_x.copy(); yn=best_y.copy()
                xn[i]+=rng.randn()*sig; yn[i]+=rng.randn()*sig
                xn[i]=float(_np.clip(xn[i],0.02,0.98)); yn[i]=float(_np.clip(yn[i],0.02,0.98))
                rL,sL=eval_lp(xn,yn)
                if rL is not None and sL>best_sum:
                    best_sum=float(sL); best_x=xn.copy(); best_y=yn.copy(); best_r=rL.copy()
                    best_arr=_np.column_stack([xn,yn,rL])
    except Exception:
        pass

    # final safety shrink
    try:
        Xf=best_arr[:,0]; Yf=best_arr[:,1]; Rf=best_arr[:,2]
        Rs=uniform_scale(Xf,Yf,Rf)
        # extra tiny shrink for 1e-12 strictness
        Rs=Rs*(1.0-1e-9)-1e-13
        Rs=_np.maximum(Rs,1e-6)
        # re-verify walls: clip centres if needed
        for i in range(n):
            if Rs[i]>Xf[i]: Rs[i]=Xf[i]*(1.0-1e-9)-1e-13
            if Rs[i]>1.0-Xf[i]: Rs[i]=(1.0-Xf[i])*(1.0-1e-9)-1e-13
            if Rs[i]>Yf[i]: Rs[i]=Yf[i]*(1.0-1e-9)-1e-13
            if Rs[i]>1.0-Yf[i]: Rs[i]=(1.0-Yf[i])*(1.0-1e-9)-1e-13
        best_arr=_np.column_stack([Xf,Yf,Rs])
    except Exception:
        pass
    return best_arr
