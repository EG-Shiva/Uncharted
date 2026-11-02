import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _lp_polish(x, deadline, radius_init=3e-4, cap=1500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(radius_init)
    iu = np.triu_indices(n, 1)
    while radius > 1e-14:
        if time.time() >= deadline:
            break
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        ns = int(np.sum(sel))
        if ns == 0:
            break
        if ns > cap:
            idx = np.argpartition(-c, cap)[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = sel & mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None]*d + np.arange(d),
                               J[:, None]*d + np.arange(d),
                               np.full((p,1), n*d)],1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
        A_ub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
        A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        obj = np.zeros(nv); obj[-1]=1.0
        try:
            res = linprog(obj,A_ub=A_ub,b_ub=-c[sel],A_eq=A_eq,b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        y = x + res.x[:-1].reshape(n,d)
        y /= np.linalg.norm(y,axis=1,keepdims=True)
        v = _max_cos(y)
        if v < best:
            x,best=y,v
            radius=min(radius*1.5,0.02)
        else:
            radius*=0.5
        if time.time()>=deadline:
            break
    return x

def _tangent_bases(x):
    n,d = x.shape
    Bs=[]
    for i in range(n):
        xi=x[i]
        km=int(np.argmax(np.abs(xi)))
        # columns: xi + all e_j j!=km
        M=np.zeros((d,d))
        M[:,0]=xi
        c=1
        for j in range(d):
            if j==km: continue
            e=np.zeros(d); e[j]=1.0
            M[:,c]=e; c+=1
        Q,R=np.linalg.qr(M)
        # ensure first col aligns with xi (sign)
        if np.dot(Q[:,0],xi)<0:
            Q[:,0]=-Q[:,0]
        # Q[:,1:] spans tangent (orthonormal)
        Bs.append(Q[:,1:])
    return Bs  # list n of d x (d-1)

def _build_R(x,Bs,pairs):
    n,d=x.shape
    dm=d-1
    N=n*dm
    p=len(pairs[0])
    R=np.zeros((p,N),dtype=float)
    for k in range(p):
        i=int(pairs[0][k]); j=int(pairs[1][k])
        Ti=Bs[i]; Tj=Bs[j]
        # a_j = Ti^T x_j , a_i = Tj^T x_i
        ai_j=Ti.T @ x[j]  # dm
        aj_i=Tj.T @ x[i]
        R[k,i*dm:(i+1)*dm]=ai_j
        R[k,j*dm:(j+1)*dm]=aj_i
    return R

def _smooth_grad_ambient(x,beta):
    n=x.shape[0]
    G=x@x.T
    np.fill_diagonal(G,-10.0)
    m=float(G.max())
    E=np.exp(np.clip(beta*(G-m),-30.0,0.0))
    np.fill_diagonal(E,0.0)
    Gr=E@x
    dot=np.sum(Gr*x,axis=1,keepdims=True)
    Gt=Gr-dot*x
    return Gt,m,G

def solve():
    t0=time.time()
    LIMIT=168.0
    t_end=t0+LIMIT
    rng=np.random.default_rng(7)
    n,d=82,14
    cands=[]
    for fn in ("best.npy","record.npy"):
        try:
            a=np.load(fn)
            a=np.asarray(a,dtype=float)
            if a.shape==(n,d):
                a=a/np.linalg.norm(a,axis=1,keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands)==0:
        xb=rng.standard_normal(size=(n,d)); xb/=np.linalg.norm(xb,axis=1,keepdims=True)
    else:
        cands.sort(key=lambda c:_max_cos(c))
        xb=cands[0].copy()
    # brief initial squeeze
    if t_end-time.time()>14:
        xb=_lp_polish(xb,deadline=min(time.time()+3.0,t_end-10.0),radius_init=3e-4,cap=1500)
    best_val=_max_cos(xb)
    Xb=xb.copy()
    iu_all=np.triu_indices(n,1)
    # rigidity event-driven loop
    it=0
    stall=0
    while time.time()<t_end-8.0:
        it+=1
        if time.time()>=t_end-8.0: break
        G=Xb@Xb.T
        np.fill_diagonal(G,-2.0)
        m=float(G.max())
        # adaptive slack for rigidity matrix: target 350-900 pairs
        slack=5e-6
        for _ in range(5):
            cnt_pairs=int(np.sum(G[iu_all]>m-slack))
            if cnt_pairs<250: slack*=2.0
            elif cnt_pairs>1000: slack*=0.5
            else: break
        slack=min(max(slack,2e-6),2e-5)
        sel=G[iu_all]>m-slack
        Pi=(iu_all[0][sel],iu_all[1][sel])
        p=len(Pi[0])
        if p<10 or p>1600:
            # fallback: take top 600
            flat=G[iu_all]
            k=min(600,len(flat))
            idx=np.argpartition(-flat,k)[:k]
            Pi=(iu_all[0][idx],iu_all[1][idx])
            p=k
        # degrees at 1e-5
        tolD=1e-5
        deg=np.sum(G>m-tolD,axis=1)
        avg_deg=float(deg.mean())
        # build tangent bases + rigidity matrix
        try:
            Bs=_tangent_bases(Xb)
        except Exception:
            break
        try:
            R=_build_R(Xb,Bs,Pi)
        except Exception:
            break
        # rank via singular values (economy, no vectors)
        try:
            svals=np.linalg.svd(R,compute_uv=False)
            rank=int(np.sum(svals>1e-7))
            smin=float(svals.min()) if len(svals)>0 else 0.0
        except Exception:
            rank=p; smin=0.0
        # smooth gradient in reduced coords
        beta=float(10**rng.uniform(3.0,3.9))  # 1000-8000
        if stall>6:
            beta=float(10**rng.uniform(2.4,3.2))
        Gt,m2,G2=_smooth_grad_ambient(Xb,beta)
        dm=d-1
        N=n*dm
        g_red=np.zeros(N)
        for i in range(n):
            g_red[i*dm:(i+1)*dm]=Bs[i].T @ Gt[i]
        nr=float(np.linalg.norm(g_red))
        if not np.isfinite(nr) or nr<1e-18:
            # random tangent kick for rattlers
            low=np.argsort(deg)[:4]
            Xc=Xb.copy()
            sig=1e-3
            Xc[low]+=rng.standard_normal((len(low),d))*sig
            Xc/=np.linalg.norm(Xc,axis=1,keepdims=True)
            v=_max_cos(Xc)
            if v<best_val:
                Xb=Xc; best_val=v; stall=0
            else:
                stall+=1
            continue
        # project gradient onto nullspace of R: solve R^T y ~= g
        try:
            # lstsq: R.T (N x p) y = g
            ysol,_,_,_=np.linalg.lstsq(R.T,g_red,rcond=None)
            g_row=R.T @ ysol
            g_null=g_red-g_row
            nnull=float(np.linalg.norm(g_null))
            if not np.isfinite(nnull) or nnull<1e-14:
                g_null=g_red
                nnull=float(np.linalg.norm(g_null))
        except Exception:
            g_null=g_red
            nnull=float(np.linalg.norm(g_null))
        # floppy direction
        d_flop=-g_null/(nnull+1e-18)
        # amplify lowest-degree points (relocate along descent, not random teleport)
        low_order=np.argsort(deg,kind="stable")
        # weight: rattlers get boost
        boost=np.ones(n)
        for r in range(min(10,n)):
            idx=low_order[r]
            if deg[idx]<6:
                boost[idx]=3.0 if deg[idx]==0 else 2.0
        Dmat=d_flop.reshape(n,dm)
        for i in range(n):
            Dmat[i]*=boost[i]
        # renormalize global
        Dmat=Dmat/(np.linalg.norm(Dmat)+1e-18)*np.sqrt(float(n))
        # line search along floppy mode (event-driven: until new pair enters)
        steps=[3e-4,8e-4,1.5e-3,3e-4*0.3,2.5e-3]
        # adapt steps by smin/rank: if rigid (high rank) use smaller
        if rank>700:
            steps=[s*0.5 for s in steps]
        best_step=None; best_X=None; best_v=best_val
        # precompute Ti for mapping
        for st in steps:
            if time.time()>=t_end-7.5: break
            Y=Xb.copy()
            for i in range(n):
                dx=Bs[i] @ Dmat[i]
                Y[i]=Y[i]+st*dx
            Y/=np.linalg.norm(Y,axis=1,keepdims=True)
            v=_max_cos(Y)
            if v<best_v:
                best_v=v; best_step=st; best_X=Y.copy()
        if best_X is not None:
            # accept floppy slide, then micro-LP to jam new contact
            Xb=best_X; best_val=best_v; stall=0
            sec=1.0 if (t_end-time.time())>30 else 0.6
            Xb=_lp_polish(Xb,deadline=min(time.time()+sec,t_end-6.5),radius_init=1.2e-4,cap=1500)
            nv=_max_cos(Xb)
            best_val=nv
            continue
        # no improvement: relocate worst rattlers along descent + hole bias
        stall+=1
        # pick 1-3 lowest degree points
        kpick=1 if stall<4 else int(rng.integers(2,5))
        victims=low_order[:kpick].copy()
        Xc=Xb.copy()
        # descent step for victims with larger stride
        stride=float(10**rng.uniform(-3.0,-2.0))*(2.0 if stall>3 else 1.0)
        for vix in victims:
            dx=Bs[vix] @ Dmat[vix]
            dn=float(np.linalg.norm(dx))
            if dn>1e-12:
                dx=dx/dn
            else:
                z=rng.standard_normal(d); z-=np.dot(z,Xc[vix])*Xc[vix]; dn2=np.linalg.norm(z); dx=z/(dn2+1e-18)
            # mix with far-hole direction occasionally
            if rng.random()<0.5:
                # sample candidates, pick hole most aligned with -descent (descent-guided hole)
                cand=rng.standard_normal((1500,d)); cand/=np.linalg.norm(cand,axis=1,keepdims=True)
                # hole score = max dot to others
                S=cand @ Xc.T
                # exclude self column? set victims col to -inf via recompute without victim
                # max over j!=victim
                S[:,vix]=-10.0
                mx=S.max(axis=1)
                # prefer small mx but also alignment with descent target
                target=Xc[vix]+(-Gt[vix])*5.0  # descent target (unnormed)
                tn=np.linalg.norm(target)
                if tn>1e-12: target/=tn
                else: target=Xc[vix]
                align=cand @ target
                score=mx - 0.15*align
                hole=cand[int(np.argmin(score))]
                # geodesic interpolation toward hole biased by stride
                alpha=min(1.0,stride*8.0)
                # slerp approx: linear + normalize
                new=(1-alpha)*Xc[vix]+alpha*hole + stride*0.5*dx
                new/=np.linalg.norm(new)
                Xc[vix]=new
            else:
                Xc[vix]=(Xc[vix]+stride*dx)
                Xc[vix]/=np.linalg.norm(Xc[vix])
        Xc/=np.linalg.norm(Xc,axis=1,keepdims=True)
        # short smooth quench (few Adam-like steps) along floppy-projected gradient to settle
        # 60 steps simple normalized GD with small lr
        for _ in range(60):
            Gt2,_,_= _smooth_grad_ambient(Xc,beta)
            # project per step cheap? skip projection, just tangent GD for victims+neighbors
            Xc=Xc-1.5e-3*Gt2
            Xc/=np.linalg.norm(Xc,axis=1,keepdims=True)
            if time.time()>=t_end-7.0: break
        v=_max_cos(Xc)
        if v<best_val+0.002:
            sec=0.7 if v>=best_val else 1.2
            Xp=_lp_polish(Xc,deadline=min(time.time()+sec,t_end-6.5),radius_init=1.5e-4,cap=1500)
            vp=_max_cos(Xp)
            if vp<best_val-1e-12:
                Xb=Xp; best_val=vp; stall=0
                continue
        # occasional full random small shake to avoid deadlock
        if stall>=8:
            Xc=Xb+rng.standard_normal((n,d))*0.002
            Xc/=np.linalg.norm(Xc,axis=1,keepdims=True)
            Xp=_lp_polish(Xc,deadline=min(time.time()+0.7,t_end-6.5),radius_init=2e-4,cap=1200)
            vp=_max_cos(Xp)
            if vp<best_val:
                Xb=Xp; best_val=vp; stall=0
            else:
                stall=0  # reset to try floppy again
        # loop time check
    rem=(t_end-1.0)-time.time()
    if rem>1.2:
        Xb=_lp_polish(Xb,deadline=time.time()+min(rem,6.0),radius_init=8e-5,cap=1500)
    Xb=Xb/np.linalg.norm(Xb,axis=1,keepdims=True)
    return Xb
