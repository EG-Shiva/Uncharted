import numpy as np
import time

def solve():
    import time as _tm
    n = 26
    t0 = _tm.time()
    HARD = 118.0
    rng = np.random.default_rng(0)

    iu, ju = np.triu_indices(n, 1)
    m = len(iu)
    A0 = np.zeros((m, n), dtype=float)
    ar = np.arange(m)
    A0[ar, iu] = 1.0
    A0[ar, ju] = 1.0
    c_obj = -np.ones(n, dtype=float)
    from scipy.optimize import minimize, linprog
    try:
        from scipy.optimize import least_squares
        HAS_LS = True
    except Exception:
        HAS_LS = False
    try:
        from scipy.spatial import Delaunay
        HAS_DEL = True
    except Exception:
        HAS_DEL = False

    def eval_lp(xc, yc, best_ref=-1e100):
        try:
            wall = np.minimum(np.minimum(xc, 1.0 - xc), np.minimum(yc, 1.0 - yc))
            if np.min(wall) <= 1e-12:
                return None, -1.0
            if float(np.sum(wall)) <= best_ref:
                return None, -1.0
            dx = xc[iu] - xc[ju]
            dy = yc[iu] - yc[ju]
            dvec = np.sqrt(dx*dx + dy*dy)
            bounds = [(0.0, float(w)) for w in wall]
            res = linprog(c_obj, A_ub=A0, b_ub=dvec, bounds=bounds, method='highs')
            if not res.success or res.x is None:
                return None, -1.0
            r = np.asarray(res.x, dtype=float).flatten()
            r = np.maximum(r, 0.0)
            return r, float(np.sum(r))
        except Exception:
            return None, -1.0

    # fallback grid
    k = int(np.ceil(np.sqrt(n)))
    r0f = 0.5 / k
    pts = []
    for i in range(k):
        for j in range(k):
            if len(pts) < n:
                pts.append([(2*i+1)*r0f, (2*j+1)*r0f, r0f*0.999*0.999])
    best_arr0 = np.array(pts, dtype=float)
    best_x = best_arr0[:,0].copy(); best_y = best_arr0[:,1].copy()
    rlp, s0 = eval_lp(best_x, best_y, -1e100)
    if rlp is not None:
        best_r = rlp; best_sum = s0
    else:
        best_r = best_arr0[:,2].copy()*0.99; best_sum = float(np.sum(best_r))

    def time_left():
        return HARD - (_tm.time() - t0)

    # Phase A: penalized joint L-BFGS
    def make_fun(nn):
        def fun(v, W):
            X = v[0:nn]; Y = v[nn:2*nn]; R = v[2*nn:3*nn]
            gx = np.zeros(nn); gy = np.zeros(nn)
            gr_pen = np.zeros(nn)
            pen = 0.0
            v1 = R - X
            mm = v1 > 0
            if np.any(mm):
                pen += np.sum(v1[mm]**2)
                gx[mm] += -2.0*v1[mm]; gr_pen[mm] += 2.0*v1[mm]
            v2 = X + R - 1.0
            mm = v2 > 0
            if np.any(mm):
                pen += np.sum(v2[mm]**2)
                gx[mm] += 2.0*v2[mm]; gr_pen[mm] += 2.0*v2[mm]
            v3 = R - Y
            mm = v3 > 0
            if np.any(mm):
                pen += np.sum(v3[mm]**2)
                gy[mm] += -2.0*v3[mm]; gr_pen[mm] += 2.0*v3[mm]
            v4 = Y + R - 1.0
            mm = v4 > 0
            if np.any(mm):
                pen += np.sum(v4[mm]**2)
                gy[mm] += 2.0*v4[mm]; gr_pen[mm] += 2.0*v4[mm]
            Dx = X[:,None]-X[None,:]; Dy = Y[:,None]-Y[None,:]
            D = np.sqrt(Dx*Dx+Dy*Dy+1e-20)
            S = R[:,None]+R[None,:]
            V = S-D
            np.fill_diagonal(V, 0.0); np.fill_diagonal(D, 1.0)
            Mask = V>0
            if np.any(Mask):
                pen += 0.5*np.sum(V[Mask]**2)
                Q = np.zeros((nn,nn)); Q[Mask] = -2.0*V[Mask]/D[Mask]
                gx += np.sum(Q*Dx,axis=1); gy += np.sum(Q*Dy,axis=1)
                G = np.zeros((nn,nn)); G[Mask]=2.0*V[Mask]
                gr_pen += np.sum(G,axis=1)
            loss = -np.sum(R)+W*pen
            g = np.empty_like(v)
            g[0:nn]=W*gx; g[nn:2*nn]=W*gy; g[2*nn:3*nn]=-1.0+W*gr_pen
            return loss,g
        return fun
    funJ = make_fun(n)
    boundsJ = [(0.0,1.0)]*(2*n)+[(0.03,0.20)]*n
    Ws = [1e2,1e3,1e4,1e5]
    try:
        n_seed = 7
        for restart in range(n_seed):
            if _tm.time()-t0 > 30.0:
                break
            if restart==0:
                X0=np.zeros(n);Y0=np.zeros(n)
                idx=0
                for i in range(5):
                    for j in range(5):
                        if idx<25:
                            X0[idx]=(i+0.5)/5.0+rng.normal(0,0.004)
                            Y0[idx]=(j+0.5)/5.0+rng.normal(0,0.004)
                            idx+=1
                X0[25]=0.88;Y0[25]=0.12
                X0=np.clip(X0,0.05,0.95);Y0=np.clip(Y0,0.05,0.95)
                R0=np.full(n,0.095)
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
                if _tm.time()-t0>30.0:
                    break
                try:
                    res=minimize(funJ,v,args=(W,),jac=True,method='L-BFGS-B',bounds=boundsJ,
                                 options={'maxiter':180,'ftol':1e-9,'gtol':1e-7})
                    v=res.x
                except Exception:
                    break
            X=np.clip(v[0:n],0.001,0.999);Y=np.clip(v[n:2*n],0.001,0.999)
            rlp2,s2=eval_lp(X,Y,best_sum+1e-12)
            if rlp2 is not None and s2>best_sum:
                best_sum=s2;best_x=X.copy();best_y=Y.copy();best_r=rlp2.copy()
        # cheap random centres
        for _ in range(20):
            if _tm.time()-t0>34.0:
                break
            X0=rng.random(n)*0.9+0.05; Y0=rng.random(n)*0.9+0.05
            rlp2,s2=eval_lp(X0,Y0,best_sum+1e-12)
            if rlp2 is not None and s2>best_sum:
                best_sum=s2;best_x=X0.copy();best_y=Y0.copy();best_r=rlp2.copy()
    except Exception:
        pass

    # Phase B: LP hill climbing until ~62s
    try:
        it=0; stagn=0
        while True:
            if _tm.time()-t0 > 62.0:
                break
            if time_left() < 8.0:
                break
            it+=1
            if stagn>1200:
                sigma=0.022
            elif stagn>500:
                sigma=0.010
            else:
                sigma=0.011*np.exp(-it/6000.0)+0.0008
            bx=best_x; by=best_y
            rr = rng.random()
            if rr<0.72:
                idx=int(rng.integers(0,n))
                nx=bx.copy(); ny=by.copy()
                nx[idx]=np.clip(nx[idx]+rng.normal(0,sigma),0.001,0.999)
                ny[idx]=np.clip(ny[idx]+rng.normal(0,sigma),0.001,0.999)
            elif rr<0.88:
                i1=int(rng.integers(0,n)); i2=int(rng.integers(0,n))
                nx=bx.copy(); ny=by.copy()
                nx[i1]+=rng.normal(0,sigma); ny[i1]+=rng.normal(0,sigma)
                nx[i2]+=rng.normal(0,sigma); ny[i2]+=rng.normal(0,sigma)
                nx=np.clip(nx,0.001,0.999); ny=np.clip(ny,0.001,0.999)
            else:
                s2s=sigma*0.5
                nx=np.clip(bx+rng.normal(0,s2s,n),0.001,0.999)
                ny=np.clip(by+rng.normal(0,s2s,n),0.001,0.999)
            rlp2,s2=eval_lp(nx,ny,best_sum+1e-12)
            if rlp2 is not None and s2>best_sum:
                best_sum=s2;best_x=nx.copy();best_y=ny.copy();best_r=rlp2.copy()
                stagn=0
            else:
                stagn+=1
            if it%700==0 and (_tm.time()-t0 < 60.0):
                for ci in range(n):
                    if _tm.time()-t0>62.0:
                        break
                    for ang in (0, np.pi/2, np.pi, 3*np.pi/2):
                        step=0.002 if stagn<700 else 0.005
                        nx=best_x.copy(); ny=best_y.copy()
                        nx[ci]=np.clip(nx[ci]+step*np.cos(ang),0.001,0.999)
                        ny[ci]=np.clip(ny[ci]+step*np.sin(ang),0.001,0.999)
                        rlp2,s2=eval_lp(nx,ny,best_sum+1e-12)
                        if rlp2 is not None and s2>best_sum:
                            best_sum=s2;best_x=nx;best_y=ny;best_r=rlp2
                            break
    except Exception:
        pass

    # Phase C: Delaunay / power-diagram contact-graph repair
    # checkpoint best
    ck_x=best_x.copy(); ck_y=best_y.copy(); ck_r=best_r.copy(); ck_s=best_sum
    if HAS_LS:
        try:
            def delaunay_edges(Xc,Yc):
                try:
                    if HAS_DEL:
                        P=np.column_stack([Xc,Yc])
                        # tiny jitter to avoid degeneracy
                        tri=Delaunay(P)
                        es=set()
                        for s in tri.simplices:
                            for a,b in ((s[0],s[1]),(s[1],s[2]),(s[2],s[0])):
                                if a>b: a,b=b,a
                                es.add((int(a),int(b)))
                        return list(es)
                    raise RuntimeError("no del")
                except Exception:
                    # fallback: k nearest by distance (80 smallest)
                    Dx=Xc[:,None]-Xc[None,:]; Dy=Yc[:,None]-Yc[None,:]
                    D=np.sqrt(Dx*Dx+Dy*Dy)
                    pairs=[(D[i,j],i,j) for i in range(n) for j in range(i+1,n)]
                    pairs.sort(key=lambda t:t[0])
                    return [(p[1],p[2]) for p in pairs[:90]]

            def wall_choice(Xc,Yc,Rc):
                walls=[]
                for i in range(n):
                    d0=Xc[i]-Rc[i]; d1=1.0-Xc[i]-Rc[i]; d2=Yc[i]-Rc[i]; d3=1.0-Yc[i]-Rc[i]
                    walls.append(int(np.argmin([d0,d1,d2,d3])))
                return walls

            def contact_count(Xc,Yc,Rc,tol=2e-4):
                Dx=Xc[:,None]-Xc[None,:]; Dy=Yc[:,None]-Yc[None,:]
                D=np.sqrt(Dx*Dx+Dy*Dy)
                S=Rc[:,None]+Rc[None,:]
                G=D-S
                iu2,ju2=np.triu_indices(n,1)
                c=np.sum(G[iu2,ju2] < tol)
                wall=np.minimum(np.minimum(Xc-Rc,1.0-Xc-Rc),np.minimum(Yc-Rc,1.0-Yc-Rc))
                c+=np.sum(wall < tol)
                return int(c)

            def select_edges(Xc,Yc,Rc,allE,target=52):
                Dx=Xc[:,None]-Xc[None,:]; Dy=Yc[:,None]-Yc[None,:]
                D=np.sqrt(Dx*Dx+Dy*Dy)
                scored=[]
                for (i,j) in allE:
                    gap=D[i,j]-(Rc[i]+Rc[j])
                    scored.append((gap,i,j))
                scored.sort(key=lambda t:t[0])
                return [(s[1],s[2]) for s in scored[:target]]

            def solve_graph(X0,Y0,R0,edges,walls,noise,seed2):
                try:
                    rs2=np.random.RandomState(seed2)
                    Xn=np.clip(X0+rs2.normal(0,noise,n),0.02,0.98)
                    Yn=np.clip(Y0+rs2.normal(0,noise,n),0.02,0.98)
                    Rn=np.clip(R0+rs2.normal(0,noise*0.5,n),0.02,0.25)
                    z0=np.concatenate([Xn,Yn,Rn])
                    Earr=np.array(edges,dtype=int) if len(edges)>0 else np.zeros((0,2),dtype=int)
                    Warr=np.array(walls,dtype=int)
                    def resfun(z):
                        X=z[0:n];Y=z[n:2*n];R=z[2*n:3*n]
                        out=[]
                        if len(Earr)>0:
                            dx=X[Earr[:,0]]-X[Earr[:,1]]
                            dy=Y[Earr[:,0]]-Y[Earr[:,1]]
                            sr=R[Earr[:,0]]+R[Earr[:,1]]
                            out.append(dx*dx+dy*dy-sr*sr)
                        wres=np.empty(n)
                        for i in range(n):
                            w=Warr[i]
                            if w==0: wres[i]=(X[i]-R[i])*4.0
                            elif w==1: wres[i]=(1.0-X[i]-R[i])*4.0
                            elif w==2: wres[i]=(Y[i]-R[i])*4.0
                            else: wres[i]=(1.0-Y[i]-R[i])*4.0
                        out.append(wres)
                        return np.concatenate(out)
                    lb=np.concatenate([np.zeros(2*n),np.full(n,0.015)])
                    ub=np.concatenate([np.ones(2*n),np.full(n,0.30)])
                    sol=least_squares(resfun,z0,bounds=(lb,ub),max_nfev=180,xtol=1e-10,ftol=1e-10,gtol=1e-10,verbose=0)
                    z=sol.x
                    X=np.clip(z[0:n],0.001,0.999);Y=np.clip(z[n:2*n],0.001,0.999)
                    # require centres strictly inside
                    if np.min(np.minimum(np.minimum(X,1-X),np.minimum(Y,1-Y)))<=1e-9:
                        return None,None
                    if np.any(z[2*n:3*n]<=0):
                        return None,None
                    return X,Y
                except Exception:
                    return None,None

            # init current graph from incumbent
            cur_x=ck_x.copy(); cur_y=ck_y.copy(); cur_r=ck_r.copy(); cur_s=ck_s
            allE=delaunay_edges(cur_x,cur_y)
            cur_walls=wall_choice(cur_x,cur_y,cur_r)
            cur_edges=select_edges(cur_x,cur_y,cur_r,allE,target=52)
            try:
                cur_c=contact_count(cur_x,cur_y,cur_r)
            except Exception:
                cur_c=0
            seedc=1
            # a few structured lattice seeds merged: try jittered hex as alternative start for graph
            # main repair loop
            iters=0
            while (_tm.time()-t0) < (HARD-3.0) and iters < 400:
                iters+=1
                if (_tm.time()-t0)> (HARD-3.0):
                    break
                # propose modification: flip one edge or one wall
                prop_allE = allE
                # occasionally refresh Delaunay from cur
                if iters%25==0:
                    try:
                        allE=delaunay_edges(cur_x,cur_y)
                        prop_allE=allE
                    except Exception:
                        pass
                cand_edges=list(cur_edges); cand_walls=list(cur_walls)
                u=rng.random()
                if u<0.7:
                    # edge flip: remove one with largest gap, add one outside with smallest gap
                    try:
                        Dx=cur_x[:,None]-cur_x[None,:]; Dy=cur_y[:,None]-cur_y[None,:]
                        D=np.sqrt(Dx*Dx+Dy*Dy)
                        gaps=[(D[i,j]-(cur_r[i]+cur_r[j]),(i,j)) for (i,j) in cand_edges]
                        gaps.sort(key=lambda t:t[0])
                        # remove one of worst 5 randomly
                        rm_idx=int(rng.integers(max(1,len(gaps)-5),len(gaps)))
                        rm_idx=min(max(0,rm_idx),len(gaps)-1)
                        rm_pair=gaps[rm_idx][1]
                        # find candidates not in set
                        sset=set((min(a,b),max(a,b)) for a,b in cand_edges)
                        outs=[e for e in prop_allE if (min(e[0],e[1]),max(e[0],e[1])) not in sset]
                        if len(outs)==0:
                            continue
                        # score outs by gap
                        og=[(D[e[0],e[1]]-(cur_r[e[0]]+cur_r[e[1]]),e) for e in outs]
                        og.sort(key=lambda t:t[0])
                        # pick among best 5
                        pick=int(rng.integers(0,min(5,len(og))))
                        add_pair=(int(og[pick][1][0]),int(og[pick][1][1]))
                        # do swap
                        # remove rm_pair (find index)
                        for k2,(a,b) in enumerate(cand_edges):
                            if (a==rm_pair[0] and b==rm_pair[1]) or (a==rm_pair[1] and b==rm_pair[0]):
                                cand_edges[k2]=add_pair
                                break
                    except Exception:
                        continue
                else:
                    # wall flip for random circle
                    try:
                        ci=int(rng.integers(0,n))
                        # second best wall
                        d0=cur_x[ci]-cur_r[ci]; d1=1.0-cur_x[ci]-cur_r[ci]
                        d2=cur_y[ci]-cur_r[ci]; d3=1.0-cur_y[ci]-cur_r[ci]
                        order=np.argsort([d0,d1,d2,d3])
                        # flip to best or second best different from current
                        if cand_walls[ci]==int(order[0]):
                            cand_walls[ci]=int(order[1])
                        else:
                            cand_walls[ci]=int(order[0])
                    except Exception:
                        continue
                # solve graph (1-2 perturbed starts)
                best_prop=None; best_prop_s=-1; best_prop_xy=None
                nstarts=2 if iters<120 else 1
                for si in range(nstarts):
                    if (_tm.time()-t0)>(HARD-3.0):
                        break
                    Xg,Yg=solve_graph(cur_x,cur_y,cur_r,cand_edges,cand_walls,noise=0.004+0.002*si,seed2=10000+seedc)
                    seedc+=1
                    if Xg is None:
                        continue
                    rp,sp=eval_lp(Xg,Yg, -1e100)
                    if rp is None:
                        continue
                    if sp>best_prop_s:
                        best_prop_s=sp; best_prop_xy=(Xg,Yg,rp)
                if best_prop_xy is None:
                    continue
                Xg,Yg,rp=best_prop_xy; sp=best_prop_s
                try:
                    cc=contact_count(Xg,Yg,rp)
                except Exception:
                    cc=0
                # acceptance: improve cur or rigidity with small loss
                improved = sp > cur_s + 1e-12
                rigid_side = (cc > cur_c) and (sp > cur_s - 0.0002)
                if improved or rigid_side:
                    cur_x,cur_y,cur_r,cur_s = Xg,Yg,rp,sp
                    cur_edges=cand_edges; cur_walls=cand_walls; cur_c=cc
                    try:
                        allE2=delaunay_edges(cur_x,cur_y)
                        # merge to keep diversity
                        s0=set((min(a,b),max(a,b)) for a,b in allE)
                        for e in allE2:
                            k_=(min(e[0],e[1]),max(e[0],e[1]))
                            if k_ not in s0:
                                allE.append((int(e[0]),int(e[1])))
                                s0.add(k_)
                    except Exception:
                        pass
                    if sp > ck_s:
                        ck_x,ck_y,ck_r,ck_s = Xg.copy(),Yg.copy(),rp.copy(),sp
                else:
                    # elitist: if beats global best even without cur acceptance? (should have)
                    if sp > ck_s + 1e-12:
                        ck_x,ck_y,ck_r,ck_s = Xg.copy(),Yg.copy(),rp.copy(),sp
                        # also jump cur to best occasionally
                        if rng.random()<0.3:
                            cur_x,cur_y,cur_r,cur_s = Xg.copy(),Yg.copy(),rp.copy(),sp
                            cur_edges=cand_edges; cur_walls=cand_walls; cur_c=cc
                # update best tracking
                best_x, best_y, best_r, best_sum = ck_x.copy(), ck_y.copy(), ck_r.copy(), ck_s
            # restore best
            best_x, best_y, best_r, best_sum = ck_x.copy(), ck_y.copy(), ck_r.copy(), ck_s
        except Exception:
            # safeguard: keep checkpoint
            try:
                best_x, best_y, best_r, best_sum = ck_x.copy(), ck_y.copy(), ck_r.copy(), ck_s
            except Exception:
                pass

    # final safety shrink
    try:
        Xf=best_x.copy(); Yf=best_y.copy(); Rf=np.maximum(best_r.copy(),1e-12)
        wall=np.minimum(np.minimum(Xf,1.0-Xf),np.minimum(Yf,1.0-Yf))
        Dx=Xf[:,None]-Xf[None,:]; Dy=Yf[:,None]-Yf[None,:]
        D=np.sqrt(Dx*Dx+Dy*Dy); np.fill_diagonal(D,np.inf)
        S=Rf[:,None]+Rf[None,:]
        with np.errstate(divide='ignore',invalid='ignore'):
            ratio=D/S
        a=float(np.min(wall/Rf)); b=float(np.min(ratio))
        alpha=min(1.0,a,b)
        alpha=alpha*(1.0-1e-7)-1e-12
        if not np.isfinite(alpha) or alpha<=0:
            alpha=0.999
        Rf=Rf*alpha
        Rf=np.maximum(Rf-1e-13,1e-12)
        best_arr=np.column_stack([Xf,Yf,Rf])
    except Exception:
        try:
            best_arr=np.column_stack([best_x,best_y,best_r])
        except Exception:
            k2=int(np.ceil(np.sqrt(n))); rr2=0.5/k2
            pp=[]
            for i in range(k2):
                for j in range(k2):
                    if len(pp)<n:
                        pp.append([(2*i+1)*rr2,(2*j+1)*rr2,rr2*0.99])
            best_arr=np.array(pp,dtype=float)
    return best_arr
