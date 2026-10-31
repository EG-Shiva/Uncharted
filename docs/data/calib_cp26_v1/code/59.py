import numpy as np
import time

def solve():
    n = 26
    t0 = time.time()
    HARD = 118.0
    T_PEN = 62.0
    T_HILL = 100.0
    rng = np.random.RandomState(0)

    # fallback grid
    def fallback():
        k = int(np.ceil(np.sqrt(n)))
        rr = 0.5 / k
        pts = []
        for i in range(k):
            for j in range(k):
                if len(pts) < n:
                    pts.append([(2*i+1)*rr, (2*j+1)*rr, rr*0.999])
        return np.array(pts, dtype=float)

    try:
        from scipy.optimize import linprog, minimize, least_squares
        has_opt = True
    except Exception:
        has_opt = False
    if not has_opt:
        return fallback()

    try:
        from scipy.spatial import Delaunay
        has_del = True
    except Exception:
        has_del = False

    iu, ju = np.triu_indices(n, 1)
    m = len(iu)
    A0 = np.zeros((m, n), dtype=float)
    A0[np.arange(m), iu] = 1.0
    A0[np.arange(m), ju] = 1.0
    cobj = -np.ones(n, dtype=float)

    def lp_eval(xc, yc):
        b = np.minimum(np.minimum(xc, 1.0 - xc), np.minimum(yc, 1.0 - yc))
        if np.any(b <= 1e-12):
            return -1.0, None
        d = np.sqrt((xc[iu]-xc[ju])**2 + (yc[iu]-yc[ju])**2)
        bnds = [(0.0, float(bi)) for bi in b]
        try:
            res = linprog(cobj, A_ub=A0, b_ub=d, bounds=bnds, method='highs')
        except Exception:
            return -1.0, None
        if res.status != 0 or res.x is None:
            return -1.0, None
        r = np.asarray(res.x, dtype=float).flatten()
        r = np.maximum(r, 0.0)
        return float(np.sum(r)), r

    def feasible_scale(xc, yc, rc, eps=1e-7):
        b = np.minimum(np.minimum(xc, 1.0-xc), np.minimum(yc, 1.0-yc))
        rc = np.maximum(rc, 1e-12)
        t = 1.0
        # wall
        wr = (b - eps) / rc
        t = float(np.min(wr))
        d = np.sqrt((xc[iu]-xc[ju])**2 + (yc[iu]-yc[ju])**2)
        rs = rc[iu]+rc[ju]
        mask = rs > 1e-12
        if np.any(mask):
            tp = float(np.min((d[mask]-eps)/rs[mask]))
            if tp < t:
                t = tp
        if t > 1.0:
            t = 1.0
        if t <= 0:
            return rc*0.5, 0.5
        t = t*(1.0-1e-9)
        return rc*t, t

    # init best with fallback LP
    best_sum = -1.0
    best_xc = best_yc = best_r = None
    try:
        fx = fallback()
        s0, r0 = lp_eval(fx[:,0], fx[:,1])
        if s0 > 0:
            best_sum, best_xc, best_yc, best_r = s0, fx[:,0].copy(), fx[:,1].copy(), r0.copy()
    except Exception:
        pass

    # ---------- helpers for inits ----------
    def make_init(idx, rng_local):
        mode = idx % 6
        if mode == 0:
            xs=[]; ys=[]
            for ix in range(5):
                for jy in range(5):
                    xs.append((ix+0.5)/5.0); ys.append((jy+0.5)/5.0)
            xs.append(0.5); ys.append(0.5)
            xc = np.array(xs,float)+rng_local.normal(0,0.02,n)
            yc = np.array(ys,float)+rng_local.normal(0,0.02,n)
            xc = np.clip(xc,0.02,0.98); yc=np.clip(yc,0.02,0.98)
            r0 = np.clip(0.09+rng_local.normal(0,0.02,n),0.03,0.18)
            return xc,yc,r0
        elif mode == 1:
            xc=rng_local.uniform(0.03,0.97,n); yc=rng_local.uniform(0.03,0.97,n)
            r0=rng_local.uniform(0.05,0.14,n)
            return xc,yc,r0
        elif mode == 2:
            # hex-ish lattice
            pts=[]
            ny=5; nx=6
            dy=1.0/ny; dx=1.0/nx
            for iy in range(ny):
                for ix in range(nx):
                    if len(pts)>=n: break
                    x=(ix+0.5)*dx + (0.5*dx if (iy%2==1) else 0.0)
                    y=(iy+0.5)*dy
                    x=np.clip(x,0.03,0.97)
                    pts.append((x,y))
                if len(pts)>=n: break
            pts=np.array(pts[:n])
            # if short, pad random
            if len(pts)<n:
                extra=np.column_stack((rng_local.uniform(0.05,0.95,n-len(pts)),rng_local.uniform(0.05,0.95,n-len(pts))))
                pts=np.vstack([pts,extra])
            xc=pts[:,0]+rng_local.normal(0,0.02,n); yc=pts[:,1]+rng_local.normal(0,0.02,n)
            xc=np.clip(xc,0.02,0.98); yc=np.clip(yc,0.02,0.98)
            r0=np.clip(0.09+rng_local.normal(0,0.02,n),0.03,0.18)
            return xc,yc,r0
        elif mode == 3:
            # rows template random counts
            xc=rng_local.uniform(0.05,0.95,n); yc=rng_local.uniform(0.05,0.95,n)
            r0=np.full(n,0.10)
            return xc,yc,r0
        elif mode == 4:
            # staggered columns
            xc=rng_local.uniform(0.03,0.97,n); yc=rng_local.uniform(0.03,0.97,n)
            r0=rng_local.uniform(0.03,0.17,n)
            return xc,yc,r0
        else:
            if best_xc is not None and rng_local.rand()<0.6:
                xc=np.clip(best_xc+rng_local.normal(0,0.03,n),0.01,0.99)
                yc=np.clip(best_yc+rng_local.normal(0,0.03,n),0.01,0.99)
                r0=np.clip(best_r+rng_local.normal(0,0.01,n),0.02,0.20)
                return xc,yc,r0
            xc=rng_local.uniform(0.03,0.97,n); yc=rng_local.uniform(0.03,0.97,n)
            r0=rng_local.uniform(0.04,0.15,n)
            return xc,yc,r0

    bnds_pen = [(0.0,1.0)]*(2*n)+[(0.015,0.28)]*n
    P_seq = [30.0, 300.0, 3000.0, 30000.0]

    def run_penalty(x0,y0,r0, maxiter_each=90):
        z = np.concatenate([x0,y0,r0])
        for P in P_seq:
            if time.time()-t0 > T_PEN:
                break
            def fg(zz, P=P):
                x=zz[0:n]; y=zz[n:2*n]; r=zz[2*n:3*n]
                f=-np.sum(r)
                gx=np.zeros(n); gy=np.zeros(n); gr=-np.ones(n)
                v1=r-x
                mm=v1>0
                if np.any(mm):
                    f+=P*np.sum(v1[mm]*v1[mm])
                    co=2.0*P*v1[mm]
                    gx[mm]-=co; gr[mm]+=co
                v2=x+r-1.0
                mm=v2>0
                if np.any(mm):
                    f+=P*np.sum(v2[mm]*v2[mm])
                    co=2.0*P*v2[mm]
                    gx[mm]+=co; gr[mm]+=co
                v3=r-y
                mm=v3>0
                if np.any(mm):
                    f+=P*np.sum(v3[mm]*v3[mm])
                    co=2.0*P*v3[mm]
                    gy[mm]-=co; gr[mm]+=co
                v4=y+r-1.0
                mm=v4>0
                if np.any(mm):
                    f+=P*np.sum(v4[mm]*v4[mm])
                    co=2.0*P*v4[mm]
                    gy[mm]+=co; gr[mm]+=co
                dxm=x[:,None]-x[None,:]
                dym=y[:,None]-y[None,:]
                D=np.sqrt(dxm*dxm+dym*dym)
                Rsum=r[:,None]+r[None,:]
                S=Rsum-D
                np.fill_diagonal(S,-1.0)
                mask=S>0
                if np.any(mask):
                    f+=P*0.5*np.sum(S[mask]*S[mask])
                    W=np.zeros((n,n))
                    Dm=D[mask]+1e-12
                    W[mask]=(2.0*P*S[mask])/Dm
                    rs=W.sum(axis=1)
                    gx+=W.dot(x)-rs*x
                    gy+=W.dot(y)-rs*y
                    Spos=np.where(mask,S,0.0)
                    gr+=2.0*P*Spos.sum(axis=1)
                g=np.concatenate([P*gx if False else gx*P*0+gx*0, gx*0, gr*0])  # placeholder
                # build correct grad: x,y parts scaled by P? we already included P in gx,gy? Actually gx above includes 2P.. so no extra scale; gr includes -1 + P parts
                # Need: grad_x = gx_pen (already has P), grad_y similarly, grad_r = -1 + pen_r
                # Our gx,gy currently are pen grads with P factor; gr = -1 + pen_r with P
                # So:
                gg=np.concatenate([gx, gy, gr])
                # Note f already = -sum + P*pen, consistent
                return f, gg
            # fix closure: fg defined with correct math above (gx already includes P)
            try:
                res = minimize(fg, z, method='L-BFGS-B', jac=True, bounds=bnds_pen,
                               options={'maxiter': maxiter_each, 'ftol':1e-10, 'gtol':1e-6})
                z = res.x
            except Exception:
                break
        return z

    # Phase A: penalty restarts
    restart = 0
    try:
        while time.time()-t0 < T_PEN and restart < 300:
            rng_local = np.random.RandomState(1000+restart*7919)
            x0,y0,r0 = make_init(restart, rng_local)
            z = run_penalty(x0,y0,r0)
            xp=z[0:n]; yp=z[n:2*n]; rp=z[2*n:3*n]
            xp=np.clip(xp,0.001,0.999); yp=np.clip(yp,0.001,0.999)
            rp=np.clip(rp,0.005,0.28)
            # repair then LP
            try:
                rrep,_ = feasible_scale(xp,yp,rp)
                s,r = lp_eval(xp,yp)
                if s>best_sum:
                    best_sum=s; best_xc=xp.copy(); best_yc=yp.copy(); best_r=r.copy()
                else:
                    # also try repaired sum
                    sr=float(np.sum(rrep))
                    # check feasibility of rrep is guaranteed by scale; keep if LP failed?
                    if s<0 and sr>best_sum:
                        # verify strictly feasible then LP would have found; skip
                        pass
            except Exception:
                pass
            restart+=1
    except Exception:
        pass

    # Phase B: LP hill climbing polish
    try:
        if best_xc is not None:
            cur_xc,cur_yc,cur_r,cur_sum = best_xc.copy(),best_yc.copy(),best_r.copy(),best_sum
            rng2=np.random.RandomState(42)
            step=0
            while time.time()-t0 < T_HILL:
                step+=1
                # annealed sigma
                if step<2000:
                    sigma=0.03*np.exp(-step/800.0)+0.0015
                else:
                    sigma=0.004
                if rng2.rand()<0.9:
                    i=rng2.randint(n)
                    txc=cur_xc.copy(); tyc=cur_yc.copy()
                    txc[i]=np.clip(cur_xc[i]+rng2.normal(0,sigma),0.005,0.995)
                    tyc[i]=np.clip(cur_yc[i]+rng2.normal(0,sigma),0.005,0.995)
                else:
                    txc=np.clip(cur_xc+rng2.normal(0,sigma*0.4,n),0.005,0.995)
                    tyc=np.clip(cur_yc+rng2.normal(0,sigma*0.4,n),0.005,0.995)
                s,r = lp_eval(txc,tyc)
                if s>cur_sum+1e-12:
                    cur_xc,cur_yc,cur_sum,cur_r=txc,tyc,s,r
                    if s>best_sum:
                        best_sum=s; best_xc=txc.copy(); best_yc=tyc.copy(); best_r=r.copy()
                if step>30000:
                    break
    except Exception:
        pass

    # Phase C: Delaunay / power-graph least-squares jamming + edge flip
    def rigidity_count(xc,yc,r):
        try:
            d=np.sqrt((xc[iu]-xc[ju])**2+(yc[iu]-yc[ju])**2)
            gap=d-(r[iu]+r[ju])
            c1=int(np.sum(gap<5e-4))
            b=np.minimum(np.minimum(xc,1-xc),np.minimum(yc,1-yc))
            slack=b-r
            c2=int(np.sum(slack<5e-4))
            return c1+c2, c1, c2
        except Exception:
            return 0,0,0

    def build_graph(xc,yc,r, E_target=44, W_target=12):
        # neighbour candidate edges
        edges=set()
        if has_del:
            try:
                pts=np.column_stack((xc,yc))
                # jitter duplicate points slightly for qhull
                tri=Delaunay(pts)
                for s in tri.simplices:
                    for a in range(3):
                        i=int(s[a]); j=int(s[(a+1)%3])
                        if i!=j:
                            if i>j: i,j=j,i
                            edges.add((i,j))
            except Exception:
                pass
        # compute all gaps
        dAll=np.sqrt((xc[iu]-xc[ju])**2+(yc[iu]-yc[ju])**2)
        gaps=dAll-(r[iu]+r[ju])
        order=np.argsort(gaps)
        # start from Delaunay edges sorted by gap, take best E_target; if not enough, fill from global order
        if len(edges)>0:
            elist=list(edges)
            # map to gap
            def gap_of(e):
                i,j=e
                # find index? compute directly
                dd=np.sqrt((xc[i]-xc[j])**2+(yc[i]-yc[j])**2)-(r[i]+r[j])
                return dd
            elist_sorted=sorted(elist, key=gap_of)
            chosen=[]
            for e in elist_sorted:
                chosen.append(e)
                if len(chosen)>=E_target:
                    break
            # ensure degree>=2
            deg=np.zeros(n,int)
            for (i,j) in chosen:
                deg[i]+=1; deg[j]+=1
            # add nearest for low-degree
            ptr=0
            while np.any(deg<2) and ptr<len(order):
                k=int(order[ptr]); ptr+=1
                i=int(iu[k]); j=int(ju[k])
                if (i,j) in set(chosen) or (j,i) in set(chosen):
                    continue
                if deg[i]<2 or deg[j]<2:
                    chosen.append((i,j)); deg[i]+=1; deg[j]+=1
            # if still short, fill
            for k in order:
                if len(chosen)>=E_target: break
                i=int(iu[int(k)]); j=int(ju[int(k)])
                if (i,j) in set(chosen): continue
                chosen.append((i,j))
            edges_chosen=chosen[:max(E_target,len(chosen))]
        else:
            edges_chosen=[(int(iu[int(k)]),int(ju[int(k))]) if False else (int(iu[int(k)]),int(ju[int(k)])) for k in order[:E_target]]
        # wall contacts: smallest slack
        b=np.minimum(np.minimum(xc,1-xc),np.minimum(yc,1-yc))
        slack=b-r
        worder=np.argsort(slack)
        walls=[]
        for k in worder[:W_target]:
            i=int(k)
            # closest side
            xi=xc[i]; yi=yc[i]
            dists=[xi,1-xi,yi,1-yi]
            side=int(np.argmin(dists))
            walls.append((i,side))
        return edges_chosen, walls

    def graph_solve(xc0,yc0,r0, edges, walls):
        # variables [x,y,r]
        E=len(edges); W=len(walls)
        M=E+W
        def fun(v):
            x=v[0:n]; y=v[n:2*n]; r=v[2*n:3*n]
            res=np.empty(M)
            for k,(i,j) in enumerate(edges):
                dx=x[i]-x[j]; dy=y[i]-y[j]
                d=np.sqrt(dx*dx+dy*dy+1e-18)
                res[k]=d-(r[i]+r[j])
            for k,(i,side) in enumerate(walls):
                if side==0: res[E+k]=x[i]-r[i]
                elif side==1: res[E+k]=(1-x[i])-r[i]
                elif side==2: res[E+k]=y[i]-r[i]
                else: res[E+k]=(1-y[i])-r[i]
            return res
        def jac(v):
            x=v[0:n]; y=v[n:2*n]
            J=np.zeros((M,3*n))
            for k,(i,j) in enumerate(edges):
                dx=x[i]-x[j]; dy=y[i]-y[j]
                d=np.sqrt(dx*dx+dy*dy+1e-18)
                if d<1e-12: d=1e-12
                ux=dx/d; uy=dy/d
                J[k,i]+=ux; J[k,j]-=ux
                J[k,n+i]+=uy; J[k,n+j]-=uy
                J[k,2*n+i]-=1.0; J[k,2*n+j]-=1.0
            for k,(i,side) in enumerate(walls):
                if side==0:
                    J[E+k,i]=1.0; J[E+k,2*n+i]=-1.0
                elif side==1:
                    J[E+k,i]=-1.0; J[E+k,2*n+i]=-1.0
                elif side==2:
                    J[E+k,n+i]=1.0; J[E+k,2*n+i]=-1.0
                else:
                    J[E+k,n+i]=-1.0; J[E+k,2*n+i]=-1.0
            return J
        v0=np.concatenate([xc0,yc0,r0])
        try:
            sol=least_squares(fun, v0, jac=jac, method='trf', ftol=1e-10, xtol=1e-10, gtol=1e-8,
                              max_nfev=250, verbose=0,
                              bounds=(np.concatenate([np.zeros(2*n), np.full(n,1e-4)]),
                                      np.concatenate([np.ones(2*n), np.full(n,0.30)])))
            v=sol.x
            return v[0:n], v[n:2*n], v[2*n:3*n], float(np.sum(sol.fun**2))
        except Exception:
            return None

    try:
        if best_xc is not None:
            base_rig,_ ,_ = rigidity_count(best_xc,best_yc,best_r)
            # several seeds
            nseeds=6
            rng3=np.random.RandomState(7)
            for sidx in range(nseeds):
                if time.time()-t0 > HARD-2:
                    break
                # perturbed start centres
                if sidx==0:
                    pxc, pyc, pr = best_xc.copy(), best_yc.copy(), best_r.copy()
                else:
                    pxc=np.clip(best_xc+rng3.normal(0,0.004,n),0.01,0.99)
                    pyc=np.clip(best_yc+rng3.normal(0,0.004,n),0.01,0.99)
                    pr=np.clip(best_r+rng3.normal(0,0.003,n),0.01,0.25)
                edges,walls=build_graph(pxc,pyc,pr)
                out=graph_solve(pxc,pyc,pr,edges,walls)
                if out is None:
                    continue
                sxc,syc,sr,_=out
                # reject negative / bad
                if np.any(sr<0):
                    continue
                sxc=np.clip(sxc,0.005,0.995); syc=np.clip(syc,0.005,0.995)
                s_lp,r_lp=lp_eval(sxc,syc)
                if s_lp<0:
                    continue
                rig,_,_=rigidity_count(sxc,syc,r_lp)
                if s_lp>best_sum+1e-9:
                    best_sum=s_lp; best_xc=sxc.copy(); best_yc=syc.copy(); best_r=r_lp.copy()
                    base_rig=rig
                elif rig>base_rig and s_lp>best_sum-0.0002:
                    # keep more rigid without losing much? keep as base for flips but not overwrite best unless better?
                    # retain for further flips: set perturbed base
                    base_rig=rig
                    # flip loop: try one edge flip from here
                    pass
                # edge-flip repair: try 2 flips
                for _fl in range(2):
                    if time.time()-t0 > HARD-2:
                        break
                    # pick random edge to replace
                    if len(edges)==0:
                        break
                    fi=rng3.randint(len(edges))
                    # find replacement: nearest non-edge by gap
                    dAll=np.sqrt((sxc[iu]-sxc[ju])**2+(syc[iu]-syc[ju])**2)
                    gaps=dAll-(r_lp[iu]+r_lp[ju])
                    order=np.argsort(gaps)
                    eset=set(edges)
                    rep=None
                    for k in order:
                        cand=(int(iu[int(k)]),int(ju[int(k)]))
                        if cand not in eset:
                            rep=cand
                            break
                    if rep is None:
                        break
                    new_edges=list(edges); new_edges[fi]=rep
                    # optionally flip one wall
                    new_walls=list(walls)
                    if rng3.rand()<0.4:
                        wi=rng3.randint(len(new_walls))
                        i_old,side_old=new_walls[wi]
                        # replace side with second-closest side for same circle
                        xi=sxc[i_old]; yi=syc[i_old]
                        dists=[xi,1-xi,yi,1-yi]
                        side2=int(np.argsort(dists)[1])
                        new_walls[wi]=(i_old,side2)
                    out2=graph_solve(sxc+rng3.normal(0,0.002,n),syc+rng3.normal(0,0.002,n),r_lp, new_edges,new_walls)
                    if out2 is None:
                        continue
                    fxc,fyc,fr,_=out2
                    fxc=np.clip(fxc,0.005,0.995); fyc=np.clip(fyc,0.005,0.995)
                    s2,r2=lp_eval(fxc,fyc)
                    if s2<0:
                        continue
                    rig2,_,_=rigidity_count(fxc,fyc,r2)
                    if s2>best_sum+1e-9:
                        best_sum=s2; best_xc=fxc.copy(); best_yc=fyc.copy(); best_r=r2.copy()
                        base_rig=rig2; edges=new_edges; walls=new_walls
                        sxc,syc,sr=sxc.copy(),syc.copy(),r_lp  # update
                    elif rig2>base_rig and s2>best_sum-0.0002:
                        base_rig=rig2; edges=new_edges; walls=new_walls
    except Exception:
        pass

    # final return with strict scaling
    try:
        if best_xc is None:
            return fallback()
        xc,yc,r = best_xc.copy(), best_yc.copy(), best_r.copy()
        b=np.minimum(np.minimum(xc,1-xc),np.minimum(yc,1-yc))
        d=np.sqrt((xc[iu]-xc[ju])**2+(yc[iu]-yc[ju])**2)
        rs=r[iu]+r[ju]
        mask=rs>1e-14
        s_pair=1.0
        if np.any(mask):
            s_pair=float(np.min(d[mask]/(rs[mask]+1e-30)))
        s_box=float(np.min(b/(r+1e-30)))
        s=min(s_pair,s_box,1.0)
        scale=s*(1.0-1e-7)
        if scale>1.0: scale=1.0-1e-7
        if (not np.isfinite(scale)) or scale<=0: scale=0.99
        r=r*scale
        r=np.maximum(r,1e-9)
        arr=np.column_stack((xc,yc,r))
        return arr
    except Exception:
        try:
            return fallback()
        except Exception:
            return np.array([[0.5,0.5,0.01]]*26, dtype=float)
