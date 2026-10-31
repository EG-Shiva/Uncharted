import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def rowmax_info(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    rm = G.max(axis=1)
    return rm, float(rm.max()), G

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt > 4000:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], 4000-1)[:4000]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[order]] = True
            sel = keep
        c = c_all[sel]
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)],1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
        A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
        A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d), np.arange(n*d))), shape=(n,nv)).tocsr()
        obj = np.zeros(nv); obj[-1]=1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n,d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius*1.5, 0.05)
        else:
            radius *= 0.5
        if time.time()-start >= seconds:
            break
    return x

def smooth_gd(x, steps, a0, a1, s0, s1, mu=0.85):
    for it in range(steps):
        frac = it/max(steps-1,1)
        alpha = a0*(a1/a0)**frac
        smax = s0*(s1/s0)**frac if s0!=s1 else s0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        W = np.exp(alpha*(G-m))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad*x, axis=1, keepdims=True)
        gt = grad - dot*x
        # momentum via vel stored in closure? use simple heavy ball with local vel
        # we implement vel outside loop for speed: use running vel variable
        # to avoid extra alloc, keep vel array
        # This function creates vel internally
        # (unrolled below with vel)
        break
    # actual loop with vel to avoid double computation
    vel = np.zeros_like(x)
    for it in range(steps):
        frac = it/max(steps-1,1)
        alpha = a0*(a1/a0)**frac
        smax = s0*(s1/s0)**frac if s0!=s1 else s0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        W = np.exp(alpha*(G-m))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad*x, axis=1, keepdims=True)
        gt = grad - dot*x
        vel = mu*vel + gt
        nrm = float(np.linalg.norm(vel, axis=1).max())
        if nrm > 1e-14:
            x = x - (smax/nrm)*vel
            x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def hard_descent(x, steps, step0, step1, tol0=2e-3, tol1=2e-4):
    n,d = x.shape
    for it in range(steps):
        frac = it/max(steps-1,1)
        step = step0*(step1/step0)**frac if step0!=step1 else step0
        tol = tol0*(tol1/tol0)**frac if tol0!=tol1 else tol0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        M = (G > (m-tol)).astype(np.float64)
        grad = M @ x
        dot = np.sum(grad*x, axis=1, keepdims=True)
        gt = grad - dot*x
        nrm = float(np.linalg.norm(gt, axis=1).max())
        if nrm < 1e-14:
            continue
        x = x - step*gt/(nrm+1e-14)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def find_holes(x, rng, k=4, n_sample=16000):
    n,d = x.shape
    S = rng.standard_normal((n_sample,d))
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    vals = np.empty(n_sample)
    chunk=4000
    for a in range(0,n_sample,chunk):
        b=min(a+chunk,n_sample)
        vals[a:b]= (S[a:b] @ x.T).max(axis=1)
    order = np.argsort(vals)
    holes=[]
    for idx in order:
        h = S[idx]
        ok=True
        for hh in holes:
            if float(h @ hh) > 0.86:
                ok=False; break
        if ok:
            # refine
            hc = h.copy()
            for _ in range(60):
                dots = x @ hc
                m = float(dots.max())
                sel = dots > (m-0.03)
                g = np.sum(x[sel]-dots[sel,None]*hc[None,:], axis=0)
                g = g - np.dot(g,hc)*hc
                ng=float(np.linalg.norm(g))
                if ng<1e-12: break
                hn = hc - 0.01*g/ng
                hn/=np.linalg.norm(hn)
                mn=float((x@hn).max())
                if mn < m: hc=hn
                else:
                    hn2=hc-0.002*g/ng; hn2/=np.linalg.norm(hn2)
                    if float((x@hn2).max())<m: hc=hn2
                    else: break
            holes.append(hc)
            if len(holes)>=k: break
    return holes

def solve():
    t0=time.time()
    BUDGET=168.0
    rng=np.random.default_rng(7)
    try:
        bx=np.load("best.npy")
    except Exception:
        try: bx=np.load("record.npy")
        except Exception: bx=rng.standard_normal((272,4))
    bx=np.asarray(bx,dtype=np.float64)
    if bx.shape!=(272,4):
        bx=np.resize(bx,(272,4))
    bx/=np.linalg.norm(bx,axis=1,keepdims=True)
    best_x=bx.copy(); best_v=max_cos(best_x)
    try:
        rx=np.load("record.npy")
        rx=np.asarray(rx,dtype=np.float64)
        rx/=np.linalg.norm(rx,axis=1,keepdims=True)
        rv=max_cos(rx)
        if rv<best_v: best_x, best_v=rx.copy(), rv
    except Exception: pass
    cur_x=best_x.copy(); cur_v=best_v
    # state for selective LP
    pending_x=None; pending_v=np.inf
    it=0
    sigmas=[0.004,0.009,0.02,0.045,0.085,0.14,0.012]
    while time.time()-t0 < BUDGET-14.0:
        it+=1
        remain=BUDGET-14.0-(time.time()-t0)
        if remain<3: break
        progress=1.0-remain/(BUDGET-14.0)
        T = 0.0016*(1-progress)+0.00015
        # info of walker
        try:
            rowmax,mcur,Gcur=rowmax_info(cur_x)
            deg=np.count_nonzero(Gcur>(mcur-1e-3),axis=1)
            is_rat=rowmax<(mcur-5e-4)
        except Exception:
            rowmax=None; deg=np.zeros(272,int); is_rat=np.zeros(272,bool)
            mcur=cur_v
        sigma=sigmas[(it-1)%len(sigmas)]
        # occasionally reset walker if too bad
        if cur_v > best_v+0.006 and rng.random()<0.7:
            cur_x=best_x.copy(); cur_v=best_v
            try:
                rowmax,mcur,Gcur=rowmax_info(cur_x)
                deg=np.count_nonzero(Gcur>(mcur-1e-3),axis=1)
                is_rat=rowmax<(mcur-5e-4)
            except Exception: pass
        # generate 2 candidates from walker, keep better
        ncand=2 if remain>25 else 1
        cands=[]
        # get holes once per iteration (cheap-ish) every other iter
        holes=None
        need_hole=(rng.random()<0.55)
        if need_hole:
            try: holes=find_holes(cur_x,rng,k=3,n_sample=12000 if remain>40 else 8000)
            except Exception: holes=None
        for kk in range(ncand):
            mode=rng.random()
            s=sigma*(0.7+0.6*rng.random())
            xp=None
            if mode<0.42 and holes is not None and len(holes)>0:
                xp=cur_x.copy()
                # victims: worst pool
                if rowmax is not None:
                    order=np.argsort(-(rowmax+1e-3*deg))
                else: order=np.arange(272)
                nvict=1
                r=rng.random()
                if r<0.25: nvict=2
                if r>0.88: nvict=3
                nvict=min(nvict,len(holes))
                pool=order[:14]
                vics=rng.choice(pool,size=nvict,replace=False)
                for vi,hh in zip(vics,holes[:nvict]):
                    xp[int(vi)]=hh+rng.standard_normal(4)*(0.002+s*0.25)
                    xp[int(vi)]/=np.linalg.norm(xp[int(vi)])
                xp+=rng.standard_normal(xp.shape)*(s*0.32)
                if np.any(is_rat):
                    xp[is_rat]+=rng.standard_normal((int(is_rat.sum()),4))*(s*0.6)
                xp/=np.linalg.norm(xp,axis=1,keepdims=True)
            elif mode<0.62:
                # rattler scramble
                xp=cur_x.copy()+rng.standard_normal(cur_x.shape)*s
                if np.any(is_rat) and rng.random()<0.8:
                    # fully randomize a subset of rattlers to random sphere points
                    idx=np.where(is_rat)[0]
                    nr=min(len(idx), 1+int(rng.integers(0,3)))
                    pick=rng.choice(idx,size=nr,replace=False)
                    R=rng.standard_normal((nr,4)); R/=np.linalg.norm(R,axis=1,keepdims=True)
                    xp[pick]=R
                else:
                    # push top contact pair
                    G=cur_x@cur_x.T; np.fill_diagonal(G,-2.0)
                    iu=np.triu_indices(272,1)
                    cg=G[iu]
                    top=np.argpartition(-cg,9)[:10]
                    ch=top[rng.integers(len(top))]
                    i=int(iu[0][ch]); j=int(iu[1][ch])
                    push=0.5*s+0.005
                    xp[i]+=push*(cur_x[i]-cur_x[j]); xp[j]+=push*(cur_x[j]-cur_x[i])
                xp/=np.linalg.norm(xp,axis=1,keepdims=True)
            elif mode<0.82:
                xp=cur_x+rng.standard_normal(cur_x.shape)*s
                if np.any(is_rat): xp[is_rat]+=rng.standard_normal((int(is_rat.sum()),4))*(s*1.2)
                # pair push
                if rng.random()<0.6:
                    G=cur_x@cur_x.T; np.fill_diagonal(G,-2.0)
                    iu=np.triu_indices(272,1); cg=G[iu]
                    top=np.argpartition(-cg,19)[:20]
                    ch=top[rng.integers(len(top))]
                    i=int(iu[0][ch]); j=int(iu[1][ch])
                    push=0.5*s+0.005
                    xp[i]+=push*(cur_x[i]-cur_x[j]); xp[j]+=push*(cur_x[j]-cur_x[i])
                xp/=np.linalg.norm(xp,axis=1,keepdims=True)
            else:
                # large shake
                ls=0.12+0.12*rng.random()
                xp=cur_x+rng.standard_normal(cur_x.shape)*ls
                xp/=np.linalg.norm(xp,axis=1,keepdims=True)
            a0=18.0+14.0*rng.random()
            a1=130.0+150.0*rng.random()
            s0=0.007+0.007*rng.random()
            s1=0.0005+0.001*rng.random()
            steps=520+int(rng.integers(0,320))
            if remain<30: steps=min(steps,420)
            mu=0.82+0.08*rng.random()
            xp=smooth_gd(xp,steps,a0,a1,s0,s1,mu=mu)
            xp=hard_descent(xp,70,0.0025,0.00035)
            v=max_cos(xp)
            cands.append((v,xp))
        cands.sort(key=lambda t:t[0])
        v0,x0=cands[0]
        # walker acceptance (basin hopping)
        if v0 < cur_v or (v0 < cur_v+T and rng.random()<0.35):
            cur_x,cur_v=x0,v0
        # track pending for LP
        if v0 < pending_v:
            pending_v,pending_x=v0,x0.copy()
        # immediate LP if strictly better than best (potential record)
        if v0 < best_v-1e-7:
            rt=min(5.0,max(2.0,(BUDGET-14.0-(time.time()-t0))/5.0))
            yp=lp_polish(x0,rt)
            vp=max_cos(yp)
            if vp<best_v-1e-12:
                best_x,best_v=yp,vp
                cur_x,cur_v=yp.copy(),vp
                pending_v=np.inf; pending_x=None
                continue
            elif vp<pending_v:
                pending_v,pending_x=vp,yp.copy()
        # periodic selective LP every 4 iters
        if it%4==0 and pending_x is not None:
            if pending_v < best_v+0.0012:
                rt2=min(4.0,max(2.0,(BUDGET-14.0-(time.time()-t0))/6.0))
                yp=lp_polish(pending_x,rt2)
                vp=max_cos(yp)
                if vp<best_v-1e-12:
                    best_x,best_v=yp,vp
                    cur_x,cur_v=yp.copy(),vp
                elif vp < cur_v-2e-4 and rng.random()<0.5:
                    cur_x,cur_v=yp.copy(),vp
                pending_v=np.inf; pending_x=None
            else:
                pending_v=np.inf; pending_x=None
        # safety: if pending very stale reset
        if it%12==0:
            pending_v=np.inf; pending_x=None
    remain=BUDGET-(time.time()-t0)
    if remain>3:
        try:
            best_x=lp_polish(best_x,min(remain-1.0,11.0))
        except Exception: pass
    best_x/=np.linalg.norm(best_x,axis=1,keepdims=True)
    return best_x
