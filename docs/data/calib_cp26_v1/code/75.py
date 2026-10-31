import numpy as np
import time as _time

def solve():
    import time as _tm
    import numpy as _np
    from scipy.optimize import minimize, linprog
    n = 26
    t0 = _tm.time()
    DEADLINE = 124.0

    iu, ju = _np.triu_indices(n, 1)
    m = len(iu)
    A_ub0 = _np.zeros((m, n), dtype=float)
    _ar = _np.arange(m)
    A_ub0[_ar, iu] = 1.0
    A_ub0[_ar, ju] = 1.0
    c_obj = -_np.ones(n)

    def eval_lp(xc, yc):
        b = _np.minimum(_np.minimum(xc, 1.0 - xc), _np.minimum(yc, 1.0 - yc))
        b = _np.maximum(b, 0.0)
        # small margin so LP solution is interior
        b = _np.maximum(b - 1e-9, 0.0)
        dx = xc[iu] - xc[ju]
        dy = yc[iu] - yc[ju]
        dvec = _np.sqrt(dx*dx + dy*dy) - 1e-9
        dvec = _np.maximum(dvec, 0.0)
        bounds = [(0.0, float(bi)) for bi in b]
        try:
            res = linprog(c_obj, A_ub=A_ub0, b_ub=dvec, bounds=bounds, method='highs')
        except Exception:
            return None, -1.0
        if (not res.success) or res.x is None:
            return None, -1.0
        r = _np.asarray(res.x, dtype=float).flatten()
        r = _np.maximum(r, 0.0)
        return r, float(_np.sum(r))

    def strictify(xc, yc, rc):
        xc = _np.clip(_np.asarray(xc, float), 0.0, 1.0)
        yc = _np.clip(_np.asarray(yc, float), 0.0, 1.0)
        rc = _np.maximum(_np.asarray(rc, float).flatten(), 1e-7)
        # wall fix with margin
        for i in range(n):
            bw = min(xc[i], 1.0-xc[i], yc[i], 1.0-yc[i])
            maxr = bw - 2e-9
            if maxr < 1e-7:
                # move centre slightly inside if jammed at edge (should not happen)
                xc[i] = min(max(xc[i], 1e-6), 1.0-1e-6)
                yc[i] = min(max(yc[i], 1e-6), 1.0-1e-6)
                bw = min(xc[i], 1.0-xc[i], yc[i], 1.0-yc[i])
                maxr = bw - 2e-9
            if rc[i] > maxr:
                rc[i] = max(maxr, 1e-7)
        rc = rc * (1.0 - 2e-7)
        # pairwise fix, need ri+rj + 3e-9 <= d
        for _ in range(20):
            viol = False
            for i in range(n):
                xi = xc[i]; yi = yc[i]
                for j in range(i+1, n):
                    dx = xi - xc[j]; dy = yi - yc[j]
                    d = float(_np.sqrt(dx*dx + dy*dy))
                    need = rc[i] + rc[j] + 3e-9 - d
                    if need > 0:
                        # shrink both proportionally
                        s = (d - 3e-9) / (rc[i] + rc[j])
                        if s <= 0:
                            s = 0.5
                        # shrink a bit more
                        s *= (1.0 - 1e-9)
                        rc[i] *= s
                        rc[j] *= s
                        viol = True
            if not viol:
                break
        rc = _np.maximum(rc, 1e-7)
        return _np.column_stack([xc, yc, rc])

    # fallback feasible
    def fallback_arr():
        k = int(_np.ceil(_np.sqrt(n)))
        rr = 0.5 / k
        pts = []
        for i in range(k):
            for j in range(k):
                if len(pts) < n:
                    pts.append([(2*i+1)*rr, (2*j+1)*rr, rr*0.995])
        return _np.array(pts, dtype=float)

    best_arr = fallback_arr()
    # ensure fallback strictly feasible via strictify
    best_arr = strictify(best_arr[:,0], best_arr[:,1], best_arr[:,2])
    best_sum = float(_np.sum(best_arr[:,2]))
    best_x = best_arr[:,0].copy(); best_y = best_arr[:,1].copy(); best_r = best_arr[:,2].copy()

    bnds = [(0.0,1.0)]*(2*n)+[(1e-4,0.35)]*n
    P_seq_full = [30.0, 300.0, 3000.0, 30000.0]
    P_seq_short = [300.0, 8000.0]

    def run_penalty(x0, y0, r0, P_seq, maxiter_each=80):
        z = _np.concatenate([x0, y0, r0])
        for P in P_seq:
            if _tm.time()-t0 > DEADLINE-8:
                break
            def fg(zz, P=P):
                x = zz[0:n]; y = zz[n:2*n]; r = zz[2*n:3*n]
                f = -_np.sum(r)
                gx = _np.zeros(n); gy = _np.zeros(n); gr = -_np.ones(n)
                v1 = r-x
                mm = v1>0
                if _np.any(mm):
                    f += P*_np.sum(v1[mm]*v1[mm])
                    co = 2.0*P*v1[mm]
                    gx[mm]-=co; gr[mm]+=co
                v2 = x+r-1.0
                mm = v2>0
                if _np.any(mm):
                    f += P*_np.sum(v2[mm]*v2[mm])
                    co = 2.0*P*v2[mm]
                    gx[mm]+=co; gr[mm]+=co
                v3 = r-y
                mm = v3>0
                if _np.any(mm):
                    f += P*_np.sum(v3[mm]*v3[mm])
                    co = 2.0*P*v3[mm]
                    gy[mm]-=co; gr[mm]+=co
                v4 = y+r-1.0
                mm = v4>0
                if _np.any(mm):
                    f += P*_np.sum(v4[mm]*v4[mm])
                    co = 2.0*P*v4[mm]
                    gy[mm]+=co; gr[mm]+=co
                dxm = x[:,None]-x[None,:]
                dym = y[:,None]-y[None,:]
                D = _np.sqrt(dxm*dxm+dym*dym)
                Rsum = r[:,None]+r[None,:]
                S = Rsum-D
                _np.fill_diagonal(S,-1.0)
                mask = S>0
                if _np.any(mask):
                    f += P*0.5*_np.sum(S[mask]*S[mask])
                    W = _np.zeros((n,n))
                    Dm = D[mask]+1e-12
                    W[mask]=(2.0*P*S[mask])/Dm
                    rs = W.sum(axis=1)
                    gx += W.dot(x)-rs*x
                    gy += W.dot(y)-rs*y
                    Spos = _np.where(mask,S,0.0)
                    gr += 2.0*P*Spos.sum(axis=1)
                g = _np.concatenate([gx,gy,gr])
                return f,g
            try:
                res = minimize(fg,z,method='L-BFGS-B',jac=True,bounds=bnds,
                               options={'maxiter':maxiter_each,'ftol':1e-9,'gtol':1e-5})
                z = res.x
            except Exception:
                break
        return z[0:n], z[n:2*n], z[2*n:3*n]

    def try_update(xc, yc):
        global_best = [False]
        r_try, s_try = eval_lp(xc, yc)
        if r_try is None:
            return False
        if s_try > best_sum + 1e-12:
            return (r_try, s_try)
        return False

    try:
        # ---- Stage 1: diverse restarts ----
        restart_idx = 0
        # time budget 62s for stage1
        while True:
            if _tm.time()-t0 > 62.0:
                break
            if restart_idx > 300:
                break
            rng = _np.random.RandomState(9000+restart_idx*131071)
            mode = restart_idx % 8
            if restart_idx == 0:
                # jittered 5x5+1
                xs=[]; ys=[]
                for ii in range(5):
                    for jj in range(5):
                        xs.append((ii+0.5)/5.0); ys.append((jj+0.5)/5.0)
                xs.append(0.5); ys.append(0.5)
                x0=_np.array(xs)+rng.normal(0,0.015,n)
                y0=_np.array(ys)+rng.normal(0,0.015,n)
                x0=_np.clip(x0,0.05,0.95); y0=_np.clip(y0,0.05,0.95)
                r0=_np.full(n,0.09)+rng.normal(0,0.008,n)
                r0=_np.clip(r0,0.05,0.13)
            elif mode == 1:
                x0=rng.uniform(0.05,0.95,n); y0=rng.uniform(0.05,0.95,n)
                r0=rng.uniform(0.06,0.12,n)
            elif mode == 2:
                # hex init
                x0=_np.zeros(n); y0=_np.zeros(n)
                # hex spacing
                sx=0.185; sy=0.165
                cnt=0
                row=0
                yy=0.08
                while cnt<n and yy<0.97:
                    xx=0.08+(0.09 if (row%2==1) else 0.0)
                    while xx<0.97 and cnt<n:
                        x0[cnt]=xx+rng.normal(0,0.01); y0[cnt]=yy+rng.normal(0,0.01)
                        cnt+=1; xx+=sx
                    yy+=sy; row+=1
                while cnt<n:
                    x0[cnt]=rng.uniform(0.1,0.9); y0[cnt]=rng.uniform(0.1,0.9); cnt+=1
                x0=_np.clip(x0,0.05,0.95); y0=_np.clip(y0,0.05,0.95)
                r0=_np.full(n,0.095)+rng.normal(0,0.01,n)
                r0=_np.clip(r0,0.05,0.13)
            elif mode == 3:
                # bimodal large-small
                x0=rng.uniform(0.05,0.95,n); y0=rng.uniform(0.05,0.95,n)
                r0=_np.zeros(n)
                perm=rng.permutation(n)
                nL=13
                r0[perm[:nL]]=rng.uniform(0.10,0.14,nL)
                r0[perm[nL:]]=rng.uniform(0.04,0.08,n-nL)
            elif mode == 4:
                # random rows template
                nrows=rng.randint(4,7)
                # random row counts summing to 26
                cuts=sorted(rng.choice(range(1,26),nrows-1,replace=False))
                counts=_np.diff([0]+cuts+[26])
                rng.shuffle(counts)
                x0=_np.zeros(n); y0=_np.zeros(n)
                idx=0
                for ri,cnt_ in enumerate(counts):
                    yy=(ri+0.5)/nrows+rng.normal(0,0.01)
                    for k in range(cnt_):
                        xx=(k+0.5)/cnt_+rng.normal(0,0.02)
                        x0[idx]=xx; y0[idx]=yy; idx+=1
                x0=_np.clip(x0+rng.normal(0,0.005,n),0.05,0.95)
                y0=_np.clip(y0+rng.normal(0,0.005,n),0.05,0.95)
                r0=rng.uniform(0.07,0.11,n)
            elif mode == 5:
                # border-heavy
                x0=_np.zeros(n); y0=_np.zeros(n)
                for i in range(n):
                    if i<14:
                        # near border
                        side=rng.randint(0,4)
                        t=rng.uniform(0.08,0.92)
                        off=rng.uniform(0.06,0.16)
                        if side==0: x0[i]=off; y0[i]=t
                        elif side==1: x0[i]=1-off; y0[i]=t
                        elif side==2: x0[i]=t; y0[i]=off
                        else: x0[i]=t; y0[i]=1-off
                    else:
                        x0[i]=rng.uniform(0.2,0.8); y0[i]=rng.uniform(0.2,0.8)
                r0=rng.uniform(0.06,0.12,n)
            elif mode == 6:
                # lognormal varied
                x0=rng.uniform(0.05,0.95,n); y0=rng.uniform(0.05,0.95,n)
                r0=_np.clip(rng.lognormal(mean=np.log(0.085),sigma=0.45,size=n),0.03,0.16)
            else:
                x0=rng.uniform(0.05,0.95,n); y0=rng.uniform(0.05,0.95,n)
                r0=_np.full(n,0.10)

            # choose iterations: first few longer, rest shorter for speed
            mi = 100 if restart_idx<12 else 60
            x1,y1,r1 = run_penalty(_np.clip(x0,0,1),_np.clip(y0,0,1),_np.clip(r0,1e-4,0.35),P_seq_full, maxiter_each=mi)
            r_opt, s_opt = eval_lp(_np.clip(x1,0.002,0.998), _np.clip(y1,0.002,0.998))
            if r_opt is not None and s_opt > best_sum + 1e-12:
                best_sum = s_opt
                best_x = _np.clip(x1,0.002,0.998).copy()
                best_y = _np.clip(y1,0.002,0.998).copy()
                best_r = r_opt.copy()
            restart_idx+=1

        # ---- Stage 2: ruin and recreate ----
        # need current best; if still fallback, continue
        rng_global = _np.random.RandomState(12345)
        it2 = 0
        # precompute grid for void search
        gx_grid, gy_grid = _np.meshgrid(_np.linspace(0.02,0.98,36), _np.linspace(0.02,0.98,36))
        cand = _np.column_stack([gx_grid.ravel(), gy_grid.ravel()])
        while True:
            if _tm.time()-t0 > 108.0:
                break
            if it2 > 400:
                break
            it2+=1
            rng = _np.random.RandomState(50000+it2*9973)
            base_x = best_x.copy(); base_y = best_y.copy(); base_r = best_r.copy()
            op = it2 % 5
            nx = base_x.copy(); ny = base_y.copy(); nr = base_r.copy()
            if op == 0:
                # largest-void teleport of a small circle
                # compute clearance using base
                # vectorized over candidates
                # wall clearance
                wall = _np.minimum(_np.minimum(cand[:,0],1-cand[:,0]), _np.minimum(cand[:,1],1-cand[:,1]))
                # dist to circles
                # chunk to save memory
                best_c = -1e9; best_p = None
                # subsample candidates for speed (every other)
                sub = cand[::2]
                dx = sub[:,None,0]-base_x[None,:] if False else None
                # loop in batches
                bsize=648
                bc=-1e9; bp=None
                for s in range(0, len(sub), bsize):
                    blk=sub[s:s+bsize]
                    Dx=blk[:,None,0]-base_x[None,:]
                    Dy=blk[:,None,1]-base_y[None,:]
                    D=_np.sqrt(Dx*Dx+Dy*Dy)-base_r[None,:]
                    clr=_np.minimum(_np.minimum(blk[:,0],1-blk[:,0]),_np.minimum(blk[:,1],1-blk[:,1]))
                    clr=_np.minimum(clr, _np.min(D,axis=1))
                    j=_np.argmax(clr)
                    if clr[j]>bc:
                        bc=clr[j]; bp=blk[j].copy()
                # choose victim: smallest radius (or random among 6 smallest)
                order=_np.argsort(base_r)
                victim=order[rng.randint(0,6)]
                nx[victim]=float(_np.clip(bp[0]+rng.normal(0,0.005),0.01,0.99))
                ny[victim]=float(_np.clip(bp[1]+rng.normal(0,0.005),0.01,0.99))
                # jitter others slightly
                nx+=rng.normal(0,0.004,n); ny+=rng.normal(0,0.004,n)
                nx=_np.clip(nx,0.01,0.99); ny=_np.clip(ny,0.01,0.99)
                nr=_np.clip(base_r+rng.normal(0,0.005,n),1e-4,0.35)
            elif op == 1:
                # large-small swap + jitter
                order=_np.argsort(base_r)
                a=order[rng.randint(n-6,n)]; b=order[rng.randint(0,6)]
                nx[a],nx[b]=nx[b],nx[a]
                ny[a],ny[b]=ny[b],ny[a]
                nx+=rng.normal(0,0.006,n); ny+=rng.normal(0,0.006,n)
                nx=_np.clip(nx,0.01,0.99); ny=_np.clip(ny,0.01,0.99)
                nr=_np.clip(base_r,1e-4,0.35)
            elif op == 2:
                # ruin k circles randomly
                k=rng.randint(3,8)
                victims=rng.choice(n,k,replace=False)
                for v in victims:
                    nx[v]=rng.uniform(0.05,0.95); ny[v]=rng.uniform(0.05,0.95)
                # small shake of rest
                mask=_np.ones(n,bool); mask[victims]=False
                nx[mask]+=rng.normal(0,0.005,mask.sum()); ny[mask]+=rng.normal(0,0.005,mask.sum())
                nx=_np.clip(nx,0.01,0.99); ny=_np.clip(ny,0.01,0.99)
                nr=_np.clip(base_r+rng.normal(0,0.004,n),1e-4,0.35)
            elif op == 3:
                # global shake
                sig=0.015 if (it2%10==0) else 0.007
                nx=base_x+rng.normal(0,sig,n); ny=base_y+rng.normal(0,sig,n)
                nx=_np.clip(nx,0.01,0.99); ny=_np.clip(ny,0.01,0.99)
                nr=_np.clip(base_r+rng.normal(0,0.003,n),1e-4,0.35)
            else:
                # pair teleport to void: move two small circles to two best voids
                order=_np.argsort(base_r)
                victims=order[:2]
                # find voids sequentially (rough: random candidates scored)
                trial=rng.uniform(0.02,0.98,size=(800,2))
                wall=_np.minimum(_np.minimum(trial[:,0],1-trial[:,0]),_np.minimum(trial[:,1],1-trial[:,1]))
                Dx=trial[:,None,0]-base_x[None,:]
                Dy=trial[:,None,1]-base_y[None,:]
                D=_np.sqrt(Dx*Dx+Dy*Dy)-base_r[None,:]
                clr=_np.minimum(wall,_np.min(D,axis=1))
                idx=_np.argsort(clr)[-2:]
                for vi,ti in zip(victims, idx):
                    nx[vi]=trial[ti,0]; ny[vi]=trial[ti,1]
                nx+=rng.normal(0,0.003,n); ny+=rng.normal(0,0.003,n)
                nx=_np.clip(nx,0.01,0.99); ny=_np.clip(ny,0.01,0.99)
                nr=_np.clip(base_r,1e-4,0.35)

            # short rejamming
            x1,y1,r1 = run_penalty(nx,ny,nr,P_seq_short,maxiter_each=50)
            r_opt,s_opt = eval_lp(_np.clip(x1,0.002,0.998),_np.clip(y1,0.002,0.998))
            if r_opt is not None and s_opt > best_sum + 1e-12:
                best_sum=s_opt; best_x=_np.clip(x1,0.002,0.998).copy(); best_y=_np.clip(y1,0.002,0.998).copy(); best_r=r_opt.copy()
            else:
                # also try LP directly on perturbed centres without penalty (fast topology test)
                if _tm.time()-t0 < 100.0:
                    r2,s2 = eval_lp(nx,ny)
                    if r2 is not None and s2 > best_sum + 1e-9:
                        # jam this promising topology a bit more
                        x3,y3,r3 = run_penalty(nx,ny,nr,P_seq_full,maxiter_each=60)
                        r_opt2,s_opt2 = eval_lp(_np.clip(x3,0.002,0.998),_np.clip(y3,0.002,0.998))
                        if r_opt2 is not None and s_opt2 > best_sum + 1e-12:
                            best_sum=s_opt2; best_x=_np.clip(x3,0.002,0.998).copy(); best_y=_np.clip(y3,0.002,0.998).copy(); best_r=r_opt2.copy()

        # ---- Stage 3: hill climbing on centres with LP ----
        rng = _np.random.RandomState(777)
        # estimate remaining time
        nHill = 0
        while True:
            if _tm.time()-t0 > DEADLINE-4:
                break
            if nHill>600:
                break
            # decaying sigma
            frac = nHill/600.0
            sigma = 0.012*(1-frac)+0.0008
            # sometimes shake subset
            if nHill%7==0:
                xt=best_x+rng.normal(0,sigma,n); yt=best_y+rng.normal(0,sigma,n)
            else:
                # shake 4 random circles more
                xt=best_x.copy(); yt=best_y.copy()
                sel=rng.choice(n,4,replace=False)
                xt+=rng.normal(0,sigma*0.5,n); yt+=rng.normal(0,sigma*0.5,n)
                xt[sel]+=rng.normal(0,sigma*1.5,4); yt[sel]+=rng.normal(0,sigma*1.5,4)
            xt=_np.clip(xt,0.002,0.998); yt=_np.clip(yt,0.002,0.998)
            r_try,s_try=eval_lp(xt,yt)
            if r_try is not None and s_try>best_sum+1e-12:
                best_sum=s_try; best_x=xt; best_y=yt; best_r=r_try
            nHill+=1

        # final tight rejam of best with full penalty then LP
        if _tm.time()-t0 < DEADLINE-6:
            try:
                x1,y1,r1=run_penalty(best_x,best_y,_np.maximum(best_r,0.02),P_seq_full,maxiter_each=80)
                r_opt,s_opt=eval_lp(_np.clip(x1,0.002,0.998),_np.clip(y1,0.002,0.998))
                if r_opt is not None and s_opt>best_sum+1e-12:
                    best_sum=s_opt; best_x=_np.clip(x1,0.002,0.998).copy(); best_y=_np.clip(y1,0.002,0.998).copy(); best_r=r_opt.copy()
            except Exception:
                pass
    except Exception:
        pass

    out = strictify(best_x, best_y, best_r)
    return out
