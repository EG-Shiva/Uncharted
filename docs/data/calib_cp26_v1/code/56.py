import numpy as np
import time as _time

def solve():
    import time as _tm
    import numpy as _np
    from scipy.optimize import minimize, linprog
    n = 26
    t0 = _tm.time()
    DEADLINE = 124.0
    rng = _np.random.RandomState(7)
    rng2 = _np.random.RandomState(123)

    iu, ju = _np.triu_indices(n, 1)
    m = len(iu)
    A_primal = _np.zeros((m, n), dtype=float)
    ar = _np.arange(m)
    A_primal[ar, iu] = 1.0
    A_primal[ar, ju] = 1.0
    c_primal = -_np.ones(n, dtype=float)

    def primal_eval(xc, yc):
        b = _np.minimum(_np.minimum(xc, 1.0 - xc), _np.minimum(yc, 1.0 - yc))
        if _np.min(b) <= 1e-9:
            return None, -1.0
        dx = xc[iu] - xc[ju]
        dy = yc[iu] - yc[ju]
        dvec = _np.sqrt(dx*dx + dy*dy) - 1e-9
        if _np.min(dvec) <= 1e-9:
            return None, -1.0
        # upper bounds for r are b- eps; use bounds
        b2 = b - 1e-9
        if _np.min(b2) <= 1e-12:
            return None, -1.0
        bounds = [(0.0, float(bi)) for bi in b2]
        try:
            res = linprog(c_primal, A_ub=A_primal, b_ub=dvec, bounds=bounds, method='highs')
        except Exception:
            return None, -1.0
        if not res.success:
            return None, -1.0
        r = _np.maximum(_np.array(res.x, dtype=float).flatten(), 0.0)
        s = float(_np.sum(r))
        return r, s

    def pen_fun(v, W):
        X = v[0:n]; Y = v[n:2*n]; R = v[2*n:3*n]
        gx = _np.zeros(n); gy = _np.zeros(n); gr = _np.zeros(n)
        v1 = R - X
        m1 = v1 > 0
        if _np.any(m1):
            gx[m1] += -2.0*v1[m1]; gr[m1] += 2.0*v1[m1]
        v2 = X + R - 1.0
        m2 = v2 > 0
        if _np.any(m2):
            gx[m2] += 2.0*v2[m2]; gr[m2] += 2.0*v2[m2]
        v3 = R - Y
        m3 = v3 > 0
        if _np.any(m3):
            gy[m3] += -2.0*v3[m3]; gr[m3] += 2.0*v3[m3]
        v4 = Y + R - 1.0
        m4 = v4 > 0
        if _np.any(m4):
            gy[m4] += 2.0*v4[m4]; gr[m4] += 2.0*v4[m4]
        dx = X[iu] - X[ju]
        dy = Y[iu] - Y[ju]
        d = _np.sqrt(dx*dx + dy*dy + 1e-20)
        V = R[iu] + R[ju] - d
        M = V > 0
        if _np.any(M):
            inv = _np.zeros_like(d)
            inv[M] = 1.0/d[M]
            cx = V[M]*inv[M]*dx[M]
            cy = V[M]*inv[M]*dy[M]
            _np.add.at(gx, iu[M], -cx)
            _np.add.at(gx, ju[M], cx)
            _np.add.at(gy, iu[M], -cy)
            _np.add.at(gy, ju[M], cy)
            _np.add.at(gr, iu[M], V[M])
            _np.add.at(gr, ju[M], V[M])
        pen = 0.0
        if _np.any(m1): pen += _np.sum(v1[m1]**2)
        if _np.any(m2): pen += _np.sum(v2[m2]**2)
        if _np.any(m3): pen += _np.sum(v3[m3]**2)
        if _np.any(m4): pen += _np.sum(v4[m4]**2)
        if _np.any(M): pen += 0.5*_np.sum(V[M]**2)
        loss = -_np.sum(R) + W*pen
        g = _np.empty_like(v)
        g[0:n] = W*gx
        g[n:2*n] = W*gy
        g[2*n:3*n] = -1.0 + W*gr
        return loss, g

    bounds_joint = [(0.0,1.0)]*(2*n) + [(0.008,0.25)]*n

    def joint_polish(x0,y0,r0,Ws,maxiter_each):
        v = _np.concatenate([x0,y0,r0])
        for W in Ws:
            if _tm.time()-t0 > DEADLINE-3.0:
                break
            def f(vv, W=W):
                return pen_fun(vv,W)
            try:
                res = minimize(f, v, method='L-BFGS-B', jac=True, bounds=bounds_joint,
                               options={'maxiter':maxiter_each,'ftol':1e-9,'gtol':1e-5})
                v = res.x
            except Exception:
                break
        X = _np.clip(v[0:n],0.005,0.995)
        Y = _np.clip(v[n:2*n],0.005,0.995)
        r,s = primal_eval(X,Y)
        if r is None:
            R = _np.clip(v[2*n:3*n],1e-4,0.25)
            return X,Y,R,-1.0
        return X,Y,r,s

    # void grid
    GN = 161
    gx1 = _np.linspace(0.0,1.0,GN)
    XXm, YYm = _np.meshgrid(gx1,gx1)
    XX = XXm.ravel(); YY = YYm.ravel()
    wall_clear_grid = _np.minimum(_np.minimum(XX,1.0-XX), _np.minimum(YY,1.0-YY))

    def find_void(X,Y,R,keep_mask):
        md = _np.full(XX.shape, 1e100, dtype=float)
        idx = _np.where(keep_mask)[0]
        for j in idx:
            dx = XX - X[j]; dy = YY - Y[j]
            d = _np.sqrt(dx*dx+dy*dy) - R[j]
            _np.minimum(md,d,out=md)
        if len(idx)==0:
            md[:] = 1e100
        clear = _np.minimum(wall_clear_grid, md)
        k = int(_np.argmax(clear))
        return float(XX[k]), float(YY[k]), float(clear[k])

    # fallback
    k0 = int(_np.ceil(_np.sqrt(n)))
    rr0 = 0.5/k0
    pts=[]
    for i in range(k0):
        for j in range(k0):
            if len(pts)<n:
                pts.append([(2*i+1)*rr0,(2*j+1)*rr0,rr0*0.9])
    fb=_np.array(pts,dtype=float)
    rfb,sfb = primal_eval(fb[:,0],fb[:,1])
    if rfb is None:
        best_S=0.0; best_X=fb[:,0].copy(); best_Y=fb[:,1].copy(); best_R=_np.full(n,0.05)
    else:
        best_S=sfb; best_X=fb[:,0].copy(); best_Y=fb[:,1].copy(); best_R=rfb.copy()

    def make_hex(nrow_list, jitter):
        xs=[]; ys=[]
        nr=len(nrow_list)
        for ri,cnt in enumerate(nrow_list):
            y=(ri+0.5)/nr
            off = 0.5/cnt if (ri%2==1) else 0.0
            for ci in range(cnt):
                x=(ci+0.5+off*0.5)/cnt
                # keep inside
                x=min(max(x,0.03),0.97)
                xs.append(x+jitter*rng.randn())
                ys.append(y+jitter*rng.randn())
        xs=_np.array(xs); ys=_np.array(ys)
        if len(xs)<n:
            ex=n-len(xs)
            xs=_np.concatenate([xs, rng.uniform(0.05,0.95,size=ex)])
            ys=_np.concatenate([ys, rng.uniform(0.05,0.95,size=ex)])
        return xs[:n], ys[:n]

    # Phase A restarts
    T_A = 42.0
    restart=0
    while _tm.time()-t0 < T_A:
        if restart>60: break
        typ = restart % 7
        if typ==0:
            xs=[]; ys=[]
            for i in range(5):
                for j in range(5):
                    xs.append(0.1+0.2*i); ys.append(0.1+0.2*j)
            xs=_np.array(xs); ys=_np.array(ys)
            xs=_np.append(xs,0.5+rng.randn()*0.05); ys=_np.append(ys,0.5+rng.randn()*0.05)
            X0=xs+rng.randn(n)*0.008; Y0=ys+rng.randn(n)*0.008
            R0=_np.full(n,0.075)+rng.randn(n)*0.008
        elif typ==1:
            X0=rng.uniform(0.05,0.95,size=n); Y0=rng.uniform(0.05,0.95,size=n)
            R0=0.05+0.05*rng.rand(n)
        elif typ==2:
            X0,Y0=make_hex([5,5,6,5,5],0.008)
            R0=0.06+0.03*rng.rand(n)
        elif typ==3:
            X0,Y0=make_hex([5,6,5,5,5],0.01)
            R0=0.06+0.03*rng.rand(n)
        elif typ==4:
            # 6 cols x 5 rows staggered
            xx=[]; yy=[]
            for ix in range(6):
                for jy in range(5):
                    xx.append((ix+0.5)/6.0); yy.append((jy+0.5)/5.0)
            xx=_np.array(xx); yy=_np.array(yy)
            perm=rng.permutation(len(xx))[:n]
            X0=xx[perm]+rng.randn(n)*0.012; Y0=yy[perm]+rng.randn(n)*0.012
            R0=0.055+0.035*rng.rand(n)
        elif typ==5:
            # perturb best heavily
            if best_X is not None:
                X0=best_X+rng.randn(n)*0.03; Y0=best_Y+rng.randn(n)*0.03
                R0=best_R+rng.randn(n)*0.01
            else:
                X0=rng.uniform(0.05,0.95,size=n); Y0=rng.uniform(0.05,0.95,size=n)
                R0=0.05+0.05*rng.rand(n)
        else:
            X0,Y0=make_hex([4,5,4,5,4,4],0.01)
            R0=0.06+0.04*rng.rand(n)
        X0=_np.clip(X0,0.03,0.97); Y0=_np.clip(Y0,0.03,0.97)
        R0=_np.clip(R0,0.02,0.2)
        Xp,Yp,Rp,Sp=joint_polish(X0,Y0,R0,[5e2,5e3,5e4],70)
        if Sp>best_S:
            best_S=Sp; best_X=Xp.copy(); best_Y=Yp.copy(); best_R=Rp.copy()
        restart+=1

    cur_X=best_X.copy(); cur_Y=best_Y.copy(); cur_R=best_R.copy(); cur_S=best_S
    elites=[(best_S,best_X.copy(),best_Y.copy(),best_R.copy())]

    def try_update(Xp,Yp,Rp,Sp):
        global_best=False
        nonlocal_best=False
        return Sp

    # hill climbing helper (LP only, single centre moves)
    def lp_hill(X,Y,R,S, budget, sigma0):
        nonlocal_t0 = _tm.time()
        curX=X.copy(); curY=Y.copy(); curR=R.copy(); curS=S
        for k in range(budget):
            if _tm.time()-t0>DEADLINE-8.0: break
            i = rng2.randint(n)
            sig = sigma0*(1.0-0.5*k/max(1,budget))
            px = curX[i]+rng2.randn()*sig
            py = curY[i]+rng2.randn()*sig
            if px<0.02 or px>0.98 or py<0.02 or py>0.98:
                continue
            nx=curX.copy(); ny=curY.copy()
            nx[i]=px; ny[i]=py
            r0,s0=primal_eval(nx,ny)
            if r0 is not None and s0>curS+1e-12:
                curX,curY,curR,curS=nx,ny,r0,s0
        return curX,curY,curR,curS

    # quick hill on best
    if _tm.time()-t0 < DEADLINE-10:
        bx,by,br,bs = lp_hill(best_X,best_Y,best_R,best_S,300,0.015)
        if bs>best_S:
            best_S=bs; best_X=bx.copy(); best_Y=by.copy(); best_R=br.copy()
            cur_X,cur_Y,cur_R,cur_S=bx.copy(),by.copy(),br.copy(),bs

    T0=0.004; T1=0.00012
    t_start_B=_tm.time()
    it=0; no_improve=0
    while _tm.time()-t0 < DEADLINE-6.0:
        it+=1
        frac=min(1.0,(_tm.time()-t_start_B)/max(1.0,(DEADLINE-6.0-t_start_B)))
        Temp=T0*(T1/T0)**frac
        # choose move type
        u=rng2.rand()
        newX=cur_X.copy(); newY=cur_Y.copy(); newR=cur_R.copy()
        if u<0.55:
            # void relocation of 1-3 smallest
            kk=1 if rng2.rand()<0.6 else (2 if rng2.rand()<0.7 else 3)
            order=_np.argsort(cur_R)
            if rng2.rand()<0.25:
                victims=_np.array(rng2.choice(n,size=kk,replace=False))
            else:
                victims=order[:kk].copy()
                if rng2.rand()<0.3:
                    # replace one with random
                    victims[rng2.randint(kk)]=rng2.randint(n)
            keep=_np.ones(n,dtype=bool); keep[victims]=False
            tmpX=newX.copy(); tmpY=newY.copy(); tmpR=newR.copy()
            cur_keep=keep.copy()
            placed=[]
            for v in victims:
                bx,by,bc=find_void(tmpX,tmpY,tmpR,cur_keep)
                bx+=rng2.randn()*0.002; by+=rng2.randn()*0.002
                bx=float(_np.clip(bx,0.02,0.98)); by=float(_np.clip(by,0.02,0.98))
                placed.append((bx,by,bc))
                tmpX[v]=bx; tmpY[v]=by; tmpR[v]=max(float(bc)*0.9,0.02)
                cur_keep[v]=True
            for v,(bx,by,bc) in zip(victims,placed):
                newX[v]=bx; newY[v]=by
                newR[v]=float(_np.clip(max(float(bc)*0.85,0.02),0.015,0.2))
            if rng2.rand()<0.3:
                jit=0.0015
                newX[keep]+=rng2.randn(_np.sum(keep))*jit
                newY[keep]+=rng2.randn(_np.sum(keep))*jit
                newX=_np.clip(newX,0.02,0.98); newY=_np.clip(newY,0.02,0.98)
        elif u<0.70:
            # swap large-small positions
            order=_np.argsort(cur_R)
            a=order[-1-rng2.randint(3)]; b=order[rng2.randint(4)]
            newX[a],newX[b]=newX[b],newX[a]
            newY[a],newY[b]=newY[b],newY[a]
            newX+=rng2.randn(n)*0.0015; newY+=rng2.randn(n)*0.0015
            newX=_np.clip(newX,0.02,0.98); newY=_np.clip(newY,0.02,0.98)
            r0,s0=primal_eval(newX,newY)
            if r0 is not None:
                newR=_np.clip(r0,0.015,0.2)
            else:
                continue
        elif u<0.85:
            # global jitter
            sig=0.008+rng2.rand()*0.012
            newX+=rng2.randn(n)*sig; newY+=rng2.randn(n)*sig
            newX=_np.clip(newX,0.02,0.98); newY=_np.clip(newY,0.02,0.98)
            r0,s0=primal_eval(newX,newY)
            if r0 is None: continue
            newR=_np.clip(r0,0.015,0.2)
        else:
            # LP hill step directly on cur
            i=rng2.randint(n)
            sig=0.006
            nx=cur_X.copy(); ny=cur_Y.copy()
            nx[i]+=rng2.randn()*sig; ny[i]+=rng2.randn()*sig
            nx[i]=float(_np.clip(nx[i],0.02,0.98)); ny[i]=float(_np.clip(ny[i],0.02,0.98))
            r0,s0=primal_eval(nx,ny)
            if r0 is None: continue
            delta=s0-cur_S
            if delta>0 or (Temp>0 and rng2.rand()<_np.exp(delta/Temp) if delta>-0.05 else False):
                cur_X,cur_Y,cur_R,cur_S=nx,ny,r0,s0
                if s0>best_S+1e-12:
                    best_S=s0; best_X=nx.copy(); best_Y=ny.copy(); best_R=r0.copy()
                    no_improve=0
                else:
                    no_improve+=1
            else:
                no_improve+=1
            continue

        r0,s0=primal_eval(newX,newY)
        if r0 is None:
            no_improve+=1
            continue
        # light polish
        Xp,Yp,Rp,Sp=joint_polish(newX,newY,_np.clip(r0,0.015,0.2),[2e4],50)
        if Sp<0:
            Xp,Yp,Rp,Sp=newX,newY,r0,s0
        delta=Sp-cur_S
        accept=False
        if delta>1e-12:
            accept=True
        else:
            if Temp>1e-12:
                p=_np.exp(delta/Temp) if delta>-0.05 else 0.0
                if rng2.rand()<p: accept=True
        if accept:
            cur_X,cur_Y,cur_R,cur_S=Xp,Yp,Rp,Sp
            no_improve=0 if delta>0 else no_improve+1
        else:
            no_improve+=1
        if Sp>best_S+1e-12:
            best_S=Sp; best_X=Xp.copy(); best_Y=Yp.copy(); best_R=Rp.copy()
            no_improve=0
            # intensify: short LP hill from new best
            hx,hy,hr,hs=lp_hill(Xp,Yp,Rp,Sp,80,0.008)
            if hs>best_S:
                best_S=hs; best_X=hx.copy(); best_Y=hy.copy(); best_R=hr.copy()
                cur_X,cur_Y,cur_R,cur_S=hx.copy(),hy.copy(),hr.copy(),hs
        if no_improve>=40:
            # restore elite with perturbation
            cur_X=best_X+rng2.randn(n)*0.004
            cur_Y=best_Y+rng2.randn(n)*0.004
            cur_X=_np.clip(cur_X,0.02,0.98); cur_Y=_np.clip(cur_Y,0.02,0.98)
            r0,s0=primal_eval(cur_X,cur_Y)
            if r0 is not None:
                cur_R=r0; cur_S=s0
            else:
                cur_X,cur_Y,cur_R,cur_S=best_X.copy(),best_Y.copy(),best_R.copy(),best_S
            no_improve=0

    # final jamming polish on best
    for Ws,mi in [([1e5,1e6],120),([5e6],120)]:
        if _tm.time()-t0>DEADLINE-2.0: break
        Xp,Yp,Rp,Sp=joint_polish(best_X,best_Y,_np.clip(best_R,0.008,0.25),Ws,mi)
        if Sp>best_S:
            best_S=Sp; best_X=Xp.copy(); best_Y=Yp.copy(); best_R=Rp.copy()
    # final LP hill
    if _tm.time()-t0<DEADLINE-1.0:
        hx,hy,hr,hs=lp_hill(best_X,best_Y,best_R,best_S,200,0.005)
        if hs>best_S:
            best_S=hs; best_X=hx; best_Y=hy; best_R=hr

    # safety shrink to guarantee strict feasibility
    Xf=_np.clip(best_X,0.0,1.0); Yf=_np.clip(best_Y,0.0,1.0)
    Rf=_np.maximum(best_R,1e-9)
    # compute scaling
    wall=_np.minimum(_np.minimum(Xf,1.0-Xf),_np.minimum(Yf,1.0-Yf))
    # avoid div zero
    with _np.errstate(divide='ignore',invalid='ignore'):
        a=float(_np.min(wall/Rf))
        Dx=Xf[:,None]-Xf[None,:]; Dy=Yf[:,None]-Yf[None,:]
        D=_np.sqrt(Dx*Dx+Dy*Dy)
        S=Rf[:,None]+Rf[None,:]
        ratio=D/S
        _np.fill_diagonal(ratio,_np.inf)
        b=float(_np.min(ratio))
    alpha=min(1.0,a,b)
    alpha=alpha*(1.0-1e-7)-1e-12
    if not _np.isfinite(alpha) or alpha<=0:
        alpha=0.99
    Rf=Rf*alpha
    Rf=_np.maximum(Rf,1e-7)
    out=_np.column_stack([Xf,Yf,Rf])
    return out
