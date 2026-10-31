import numpy as np
import time

def solve():
    import time as _tm
    n = 26
    t0 = _tm.time()
    SAFE = 118.0
    rng = np.random.RandomState(20261031)

    try:
        from scipy.optimize import minimize, linprog, least_squares
        from scipy.spatial import Delaunay
        has_opt = True
    except Exception:
        has_opt = False
        Delaunay = None

    iu, ju = np.triu_indices(n, 1)
    m = len(iu)
    A_ub = np.zeros((m, n), dtype=float)
    A_ub[np.arange(m), iu] = 1.0
    A_ub[np.arange(m), ju] = 1.0
    c_obj = -np.ones(n, dtype=float)

    def lp_eval(xc, yc):
        try:
            b = np.minimum(np.minimum(xc, 1.0 - xc), np.minimum(yc, 1.0 - yc))
            if np.any(b < -1e-12):
                return -1.0, None
            b = np.maximum(b, 0.0)
            dx = xc[iu] - xc[ju]
            dy = yc[iu] - yc[ju]
            dvec = np.sqrt(dx*dx + dy*dy)
            bounds = [(0.0, float(bi)) for bi in b]
            res = linprog(c_obj, A_ub=A_ub, b_ub=dvec, bounds=bounds, method='highs')
            if res.status != 0 or res.x is None:
                return -1.0, None
            r = np.asarray(res.x, dtype=float).flatten()
            r = np.maximum(r, 0.0)
            return float(np.sum(r)), r
        except Exception:
            return -1.0, None

    # fallback grid checkpoint
    k = int(np.ceil(np.sqrt(n)))
    rr = 0.5 / k
    fb = []
    for ix in range(k):
        for jy in range(k):
            if len(fb) < n:
                fb.append([(2*ix+1)*rr, (2*jy+1)*rr, rr*0.995])
    fb_arr = np.array(fb, dtype=float)
    s_fb, r_fb = lp_eval(fb_arr[:,0], fb_arr[:,1]) if has_opt else (float(np.sum(fb_arr[:,2])), fb_arr[:,2].copy())
    if s_fb is None or s_fb < 0:
        s_fb = float(np.sum(fb_arr[:,2])); r_fb = fb_arr[:,2].copy()
    best_sum = float(s_fb)
    best_xc = fb_arr[:,0].copy(); best_yc = fb_arr[:,1].copy(); best_r = np.asarray(r_fb, dtype=float).copy()
    best_arr = np.column_stack([best_xc, best_yc, best_r])
    def checkpoint(xc, yc, r, s):
        global_best = globals().get('_unused', None)
        return s
    # helper to save
    def save_if_better(s, xc, yc, r):
        nonlocal best_sum, best_xc, best_yc, best_r, best_arr
        if s is not None and np.isfinite(s) and s > best_sum:
            best_sum = float(s)
            best_xc = xc.copy(); best_yc = yc.copy(); best_r = r.copy()
            best_arr = np.column_stack([best_xc, best_yc, best_r])
            return True
        return False

    pool = [(best_sum, best_xc.copy(), best_yc.copy())]

    # ---------- Phase A: fast penalized L-BFGS ----------
    if has_opt:
        bnds78 = [(0.0,1.0)]*(2*n) + [(0.005,0.22)]*n
        Ws = [50.0, 500.0, 5000.0, 30000.0]
        maxiters = [60, 60, 80, 100]
        T_PEN = 42.0
        restart = 0
        while _tm.time()-t0 < T_PEN and restart < 60:
            # diverse init
            if restart == 0:
                xs=[]; ys=[]
                for ix in range(5):
                    for jy in range(5):
                        xs.append((ix+0.5)/5.0); ys.append((jy+0.5)/5.0)
                xs.append(0.5); ys.append(0.5)
                xc0 = np.array(xs,float)+rng.normal(0,0.02,n)
                yc0 = np.array(ys,float)+rng.normal(0,0.02,n)
                xc0=np.clip(xc0,0.02,0.98); yc0=np.clip(yc0,0.02,0.98)
                R0 = 0.09+0.02*rng.rand(n)
            elif restart == 1:
                pts=[]
                sy=0.175; sx=0.185
                for ix in range(6):
                    for jy in range(5):
                        x=0.09+ix*sx+(0.5*sx if (jy%2==1) else 0.0)
                        y=0.09+jy*sy
                        if x<0.95 and y<0.95:
                            pts.append((x,y))
                pts=np.array(pts)
                perm=rng.permutation(len(pts))
                xc0=np.zeros(n); yc0=np.zeros(n)
                for kk in range(n):
                    if kk < len(perm):
                        xc0[kk],yc0[kk]=pts[perm[kk]]
                    else:
                        xc0[kk]=rng.uniform(0.05,0.95); yc0[kk]=rng.uniform(0.05,0.95)
                xc0+=rng.normal(0,0.01,n); yc0+=rng.normal(0,0.01,n)
                xc0=np.clip(xc0,0.01,0.99); yc0=np.clip(yc0,0.01,0.99)
                R0=0.08+0.04*rng.rand(n)
            else:
                xc0=rng.uniform(0.03,0.97,n); yc0=rng.uniform(0.03,0.97,n)
                if restart%3==0:
                    R0=0.07+0.04*rng.rand(n)
                else:
                    R0=0.05+0.07*rng.rand(n)
            v=np.concatenate([xc0,yc0,R0])
            v=np.clip(v,0.0,1.0); v[2*n:]=np.clip(v[2*n:],0.005,0.22)
            aborted=False
            for wi,W in enumerate(Ws):
                if _tm.time()-t0>=T_PEN:
                    aborted=True; break
                def fun(vv,W=W):
                    X=vv[0:n]; Y=vv[n:2*n]; R=vv[2*n:3*n]
                    pen=0.0
                    gx=np.zeros(n); gy=np.zeros(n); gr=np.zeros(n)
                    v1=R-X; mm=v1>0
                    if np.any(mm):
                        pen+=np.sum(v1[mm]*v1[mm]); gx[mm]+=-2.0*v1[mm]; gr[mm]+=2.0*v1[mm]
                    v2=X+R-1.0; mm=v2>0
                    if np.any(mm):
                        pen+=np.sum(v2[mm]*v2[mm]); gx[mm]+=2.0*v2[mm]; gr[mm]+=2.0*v2[mm]
                    v3=R-Y; mm=v3>0
                    if np.any(mm):
                        pen+=np.sum(v3[mm]*v3[mm]); gy[mm]+=-2.0*v3[mm]; gr[mm]+=2.0*v3[mm]
                    v4=Y+R-1.0; mm=v4>0
                    if np.any(mm):
                        pen+=np.sum(v4[mm]*v4[mm]); gy[mm]+=2.0*v4[mm]; gr[mm]+=2.0*v4[mm]
                    Dx=X[:,None]-X[None,:]; Dy=Y[:,None]-Y[None,:]
                    D=np.sqrt(Dx*Dx+Dy*Dy+1e-16)
                    S=R[:,None]+R[None,:]-D
                    np.fill_diagonal(S,0.0)
                    Vp=np.maximum(S,0.0); np.fill_diagonal(Vp,0.0)
                    pen+=0.25*np.sum(Vp*Vp)
                    np.fill_diagonal(D,1.0)
                    Q=Vp/D
                    gx+=-np.sum(Q*Dx,axis=1); gy+=-np.sum(Q*Dy,axis=1)
                    gr+=np.sum(Vp,axis=1)
                    loss=-np.sum(R)+W*pen
                    g=np.empty_like(vv); g[0:n]=W*gx; g[n:2*n]=W*gy; g[2*n:3*n]=-1.0+W*gr
                    return loss,g
                try:
                    res=minimize(fun,v,method='L-BFGS-B',jac=True,bounds=bnds78,options={'maxiter':maxiters[wi],'ftol':1e-9,'gtol':1e-5})
                    v=res.x
                except Exception:
                    aborted=True; break
            if not aborted:
                try:
                    X=np.clip(v[0:n],0.0,1.0); Y=np.clip(v[n:2*n],0.0,1.0)
                    s_lp,r_lp=lp_eval(np.clip(X,0.002,0.998),np.clip(Y,0.002,0.998))
                    if s_lp is not None and s_lp>0:
                        if s_lp>best_sum:
                            save_if_better(s_lp,X.copy(),Y.copy(),r_lp.copy())
                            pool.append((s_lp,X.copy(),Y.copy()))
                            pool=sorted(pool,key=lambda t:-t[0])[:6]
                        elif len(pool)<6 and s_lp>best_sum-0.05:
                            pool.append((s_lp,X.copy(),Y.copy()))
                except Exception:
                    pass
            restart+=1

    # helpers for graph phase
    def delaunay_edges(xc, yc):
        try:
            if Delaunay is None:
                raise Exception("no delaunay")
            pts=np.column_stack([xc,yc])
            # jitter duplicates slightly
            tri=Delaunay(pts)
            es=set()
            for s in tri.simplices:
                for a,b in [(s[0],s[1]),(s[1],s[2]),(s[2],s[0])]:
                    if a>b: a,b=b,a
                    es.add((int(a),int(b)))
            return list(es)
        except Exception:
            # k-nearest fallback: 3 nearest each
            Dx=xc[:,None]-xc[None,:]; Dy=yc[:,None]-yc[None,:]
            D=np.sqrt(Dx*Dx+Dy*Dy)
            es=set()
            for i in range(n):
                idx=np.argsort(D[i])
                for j in idx[1:4]:
                    a,b=(i,int(j)) if i<int(j) else (int(j),i)
                    es.add((a,b))
            return list(es)

    def wall_dict(xc, yc, thresh=0.035):
        wd={}
        for i in range(n):
            dl=xc[i]; dr=1.0-xc[i]; db=yc[i]; dt=1.0-yc[i]
            m=min(dl,dr,db,dt)
            if m < thresh:
                if m==dl: wd[i]=0
                elif m==dr: wd[i]=1
                elif m==db: wd[i]=2
                else: wd[i]=3
        return wd

    def second_wall(xc, yc, i, cur):
        dl=xc[i]; dr=1.0-xc[i]; db=yc[i]; dt=1.0-yc[i]
        order=np.argsort([dl,dr,db,dt])
        for w in order:
            if int(w)!=int(cur):
                return int(w)
        return int(cur)

    def count_contacts(xc, yc, r, tol=0.0025):
        c=0
        b=np.minimum(np.minimum(xc,1.0-xc),np.minimum(yc,1.0-yc))-r
        c+=np.sum(b<tol)
        dx=xc[iu]-xc[ju]; dy=yc[iu]-yc[ju]
        d=np.sqrt(dx*dx+dy*dy)
        gap=d-(r[iu]+r[ju])
        c+=np.sum(gap<tol)
        return int(c)

    def graph_solve(xc0, yc0, r0, edges, walls, pert=0.006):
        try:
            xp=np.clip(xc0+rng.normal(0,pert,n),0.005,0.995)
            yp=np.clip(yc0+rng.normal(0,pert,n),0.005,0.995)
            rp=np.clip(r0+rng.normal(0,pert*0.5,n),0.008,0.20)
            v0=np.concatenate([xp,yp,rp])
            E=[(int(a),int(b)) for a,b in edges]
            W=list(walls.items())
            ne=len(E); nw=len(W)
            # residual + jac
            def fun_jac(v):
                X=v[0:n]; Y=v[n:2*n]; R=v[2*n:3*n]
                res=np.empty(ne+nw,dtype=float)
                jac=np.zeros((ne+nw,3*n),dtype=float)
                for k,(a,b) in enumerate(E):
                    dx=X[a]-X[b]; dy=Y[a]-Y[b]; sr=R[a]+R[b]
                    val=dx*dx+dy*dy-sr*sr
                    # scale to ~order 0.01
                    res[k]=val*10.0
                    jac[k,a]+=20.0*dx; jac[k,b]+=-20.0*dx
                    jac[k,n+a]+=20.0*dy; jac[k,n+b]+=-20.0*dy
                    jac[k,2*n+a]+=-20.0*sr; jac[k,2*n+b]+=-20.0*sr
                for k2,(i,w) in enumerate(W):
                    k=ne+k2
                    if w==0: val=X[i]-R[i]
                    elif w==1: val=X[i]+R[i]-1.0
                    elif w==2: val=Y[i]-R[i]
                    else: val=Y[i]+R[i]-1.0
                    res[k]=val*10.0
                    if w==0: jac[k,i]=10.0; jac[k,2*n+i]=-10.0
                    elif w==1: jac[k,i]=10.0; jac[k,2*n+i]=10.0
                    elif w==2: jac[k,n+i]=10.0; jac[k,2*n+i]=-10.0
                    else: jac[k,n+i]=10.0; jac[k,2*n+i]=10.0
                return res, jac
            def f_only(v):
                r,j=fun_jac(v); return r
            def j_only(v):
                r,j=fun_jac(v); return j
            # bounds: centres [0,1], radii [0.005,0.25]
            lb=np.zeros(3*n); ub=np.ones(3*n)
            lb[2*n:3*n]=0.005; ub[2*n:3*n]=0.25
            sol=least_squares(f_only,v0,jac=j_only,bounds=(lb,ub),method='trf',ftol=1e-10,xtol=1e-10,gtol=1e-8,max_nfev=120,verbose=0)
            v=sol.x
            X=v[0:n]; Y=v[n:2*n]; R=v[2*n:3*n]
            if np.any(R<0.004):
                return None,None,None
            # reject non-graph heavy overlaps
            eset=set((a,b) if a<b else (b,a) for a,b in E)
            Dx=X[:,None]-X[None,:]; Dy=Y[:,None]-Y[None,:]
            D=np.sqrt(Dx*Dx+Dy*Dy)
            S=R[:,None]+R[None,:]
            # check any pair with deep overlap not in graph
            for i in range(n):
                for j in range(i+1,n):
                    if (i,j) not in eset:
                        if D[i,j] < (S[i,j]-0.01):
                            # allow but penalize: if very deep, reject
                            if D[i,j] < S[i,j]*0.85:
                                return None,None,None
            return X,Y,R
        except Exception:
            return None,None,None

    # ---------- Phase B: graph-repair loop from several seeds ----------
    if has_opt and (_tm.time()-t0 < SAFE-8):
        # build initial graphs from pool members + lattices
        seeds=[]
        for _,sx,sy in pool[:4]:
            seeds.append((sx.copy(),sy.copy()))
        # perturbed copies
        for _ in range(3):
            bx=pool[0][1]+rng.normal(0,0.008,n); by=pool[0][2]+rng.normal(0,0.008,n)
            seeds.append((np.clip(bx,0.005,0.995),np.clip(by,0.005,0.995)))
        # structured lattice seed
        gx=np.zeros(n); gy=np.zeros(n)
        cnt=0
        for ix in range(6):
            for jy in range(5):
                if cnt<n:
                    gx[cnt]=0.09+ix*0.165+rng.normal(0,0.005)
                    gy[cnt]=0.09+jy*0.175+rng.normal(0,0.005)
                    cnt+=1
        seeds.append((np.clip(gx,0.01,0.99),np.clip(gy,0.01,0.99)))

        for si,(sxc,syc) in enumerate(seeds):
            if _tm.time()-t0 > SAFE-10:
                break
            s0,r0c=lp_eval(np.clip(sxc,0.002,0.998),np.clip(syc,0.002,0.998))
            if s0<0:
                continue
            edges=delaunay_edges(sxc,syc)
            # spanning set: keep closest ~55 edges by gap
            try:
                dx=sxc[iu]-sxc[ju]; dy=syc[iu]-syc[ju]
                d=np.sqrt(dx*dx+dy*dy)
                est={e for e in edges}
                # compute gap using r0c
                gaps=[]
                for (a,b) in edges:
                    gaps.append((d[list(zip(iu,ju)).index((a,b))] if (a,b) in zip(iu,ju) else np.sqrt((sxc[a]-sxc[b])**2+(syc[a]-syc[b])**2))-(r0c[a]+r0c[b]))
                # Actually simpler: sort by distance
                order=np.argsort([np.sqrt((sxc[a]-sxc[b])**2+(syc[a]-syc[b])**2)-(r0c[a]+r0c[b]) for a,b in edges])
                keep=int(min(len(edges),2*n+4))
                edges=[edges[k] for k in order[:keep]]
            except Exception:
                pass
            walls=wall_dict(sxc,syc,0.045)
            cur_xc,cur_yc=sxc.copy(),syc.copy()
            cur_s,cur_r=s0,r0c
            cur_cont=count_contacts(cur_xc,cur_yc,cur_r)
            save_if_better(cur_s,cur_xc,cur_yc,cur_r)
            # repair iterations
            iters=14 if si<2 else 8
            for it in range(iters):
                if _tm.time()-t0>SAFE-8:
                    break
                # propose modification
                new_edges=list(edges); new_walls=dict(walls)
                u=rng.rand()
                if u<0.6 and len(new_edges)>5:
                    # flip one Delaunay edge
                    rm=rng.randint(len(new_edges))
                    removed=new_edges.pop(rm)
                    # find replacement: closest non-edge pair
                    Dx=cur_xc[:,None]-cur_xc[None,:]; Dy=cur_yc[:,None]-cur_yc[None,:]
                    Dd=np.sqrt(Dx*Dx+Dy*Dy)
                    eset=set((a,b) if a<b else (b,a) for a,b in new_edges)
                    best=None; bestd=1e9
                    # sample candidates
                    for _c in range(60):
                        a=rng.randint(n); b=rng.randint(n)
                        if a==b: continue
                        if a>b: a,b=b,a
                        if (a,b) in eset: continue
                        dd=Dd[a,b]-(cur_r[a]+cur_r[b]) if cur_r is not None else Dd[a,b]
                        if dd<bestd:
                            bestd=dd; best=(a,b)
                    if best is not None:
                        new_edges.append(best)
                    else:
                        new_edges.append(removed)
                else:
                    # replace one wall contact
                    if len(new_walls)>0 and rng.rand()<0.8:
                        key=rng.choice(list(new_walls.keys()))
                        new_walls[key]=second_wall(cur_xc,cur_yc,key,new_walls[key])
                    else:
                        # add wall to most boundary non-walled circle
                        bnds=np.minimum(np.minimum(cur_xc,1.0-cur_xc),np.minimum(cur_yc,1.0-cur_yc))
                        cand=np.argsort(bnds)
                        added=False
                        for cc in cand:
                            if cc not in new_walls:
                                dl=cur_xc[cc]; dr=1.0-cur_xc[cc]; db=cur_yc[cc]; dt=1.0-cur_yc[cc]
                                w=int(np.argmin([dl,dr,db,dt])); new_walls[int(cc)]=w; added=True; break
                        if not added:
                            pass
                # solve from perturbed start
                Xs,Ys,Rs=graph_solve(cur_xc,cur_yc,cur_r,new_edges,new_walls,pert=0.004+rng.rand()*0.004)
                if Xs is None:
                    continue
                Xc=np.clip(Xs,0.002,0.998); Yc=np.clip(Ys,0.002,0.998)
                ns,nr=lp_eval(Xc,Yc)
                if ns<0:
                    continue
                ncont=count_contacts(Xc,Yc,nr)
                # checkpoint every feasible
                improved = ns > cur_s + 1e-9
                rigid_gain = (ncont > cur_cont) and (ns >= cur_s - 0.0002)
                global_gain = ns > best_sum + 1e-9
                if improved or rigid_gain or global_gain:
                    cur_xc,cur_yc,cur_r,cur_s,cur_cont=Xc.copy(),Yc.copy(),nr.copy(),ns,ncont
                    edges, walls = new_edges, new_walls
                    save_if_better(ns,Xc,Yc,nr)
                    if len(pool)<6:
                        pool.append((ns,Xc.copy(),Yc.copy()))
                else:
                    # occasional neutral drift if very close
                    if ns >= cur_s - 0.00005 and rng.rand()<0.15:
                        cur_xc,cur_yc,cur_r=Xc.copy(),Yc.copy(),nr.copy()
                        # keep old graph
                # also try pure hill step on centres from cur
                if _tm.time()-t0>SAFE-8:
                    break

    # ---------- Phase C: LP hill climbing polish ----------
    if has_opt:
        cur_xc,cur_yc=best_xc.copy(),best_yc.copy()
        cur_sum,cur_r=lp_eval(cur_xc,cur_yc)
        if cur_sum<0:
            cur_sum,cur_r=best_sum,best_r.copy()
        step=0; noimp=0
        while _tm.time()-t0 < SAFE-1.0:
            base=0.03*np.exp(-step/4000.0)+0.0007
            if noimp>900:
                base=0.010
            if noimp>2200:
                # restart from pool
                pi=rng.randint(len(pool))
                cur_xc=pool[pi][1]+rng.normal(0,0.008,n)
                cur_yc=pool[pi][2]+rng.normal(0,0.008,n)
                cur_xc=np.clip(cur_xc,0.002,0.998); cur_yc=np.clip(cur_yc,0.002,0.998)
                cur_sum,cur_r=lp_eval(cur_xc,cur_yc)
                if cur_sum<0:
                    cur_xc,cur_yc=best_xc.copy(),best_yc.copy(); cur_sum,cur_r=best_sum,best_r.copy()
                noimp=0; step+=1
                continue
            u=rng.rand()
            txc=cur_xc.copy(); tyc=cur_yc.copy()
            if u<0.8:
                i=rng.randint(n)
                txc[i]=np.clip(cur_xc[i]+rng.normal(0,base),0.002,0.998)
                tyc[i]=np.clip(cur_yc[i]+rng.normal(0,base),0.002,0.998)
            elif u<0.92:
                i=rng.randint(n); j=rng.randint(n)
                txc[i]=np.clip(cur_xc[i]+rng.normal(0,base),0.002,0.998)
                tyc[i]=np.clip(cur_yc[i]+rng.normal(0,base),0.002,0.998)
                txc[j]=np.clip(cur_xc[j]+rng.normal(0,base),0.002,0.998)
                tyc[j]=np.clip(cur_yc[j]+rng.normal(0,base),0.002,0.998)
            else:
                txc=np.clip(cur_xc+rng.normal(0,base*0.4,n),0.002,0.998)
                tyc=np.clip(cur_yc+rng.normal(0,base*0.4,n),0.002,0.998)
            s,r=lp_eval(txc,tyc)
            step+=1
            if s>cur_sum+1e-12:
                cur_xc,cur_yc,cur_sum,cur_r=txc,tyc,s,r
                noimp=0
                if s>best_sum:
                    save_if_better(s,txc,tyc,r)
            else:
                noimp+=1

    # final strict shrink and return
    try:
        Xf=np.clip(best_xc,0.0,1.0); Yf=np.clip(best_yc,0.0,1.0); Rf=np.maximum(np.asarray(best_r,float),1e-9)
        # shrink for strict feasibility
        Rf = Rf*(1.0-2e-7)-1e-10
        Rf=np.maximum(Rf,1e-6)
        # ensure inside
        Xf=np.clip(Xf,Rf,1.0-Rf); Yf=np.clip(Yf,Rf,1.0-Rf)
        # extra pairwise safety: compute min ratio and scale if needed
        Dx=Xf[:,None]-Xf[None,:]; Dy=Yf[:,None]-Yf[None,:]
        D=np.sqrt(Dx*Dx+Dy*Dy)
        S=Rf[:,None]+Rf[None,:]
        np.fill_diagonal(D,np.inf); np.fill_diagonal(S,1.0)
        with np.errstate(divide='ignore',invalid='ignore'):
            ratio=D/S
        mr=float(np.min(ratio))
        wall=np.minimum(np.minimum(Xf,1.0-Xf),np.minimum(Yf,1.0-Yf))/Rf
        mw=float(np.min(wall))
        alpha=min(1.0,mr,mw)
        if alpha<1.0:
            Rf=Rf*alpha*0.9999995
            Xf=np.clip(Xf,Rf,1.0-Rf); Yf=np.clip(Yf,Rf,1.0-Rf)
        out=np.column_stack([Xf,Yf,Rf])
        return out
    except Exception:
        return best_arr
