import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def adam_quench(x_init, steps, T0, T1, lr):
    x = x_init.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    l0 = np.log(T0); l1 = np.log(T1)
    for s in range(1, steps+1):
        T = float(np.exp(l0 + (l1-l0)*(s/steps)))
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        M = G.max()
        W = np.exp((G-M)/T)
        np.fill_diagonal(W, 0.0)
        Grad = (W @ x) / T
        dot = np.sum(Grad*x, axis=1, keepdims=True)
        Gt = Grad - dot*x
        m = b1*m + (1.0-b1)*Gt
        v = b2*v + (1.0-b2)*(Gt*Gt)
        mh = m/(1.0-b1**s)
        vh = v/(1.0-b2**s)
        upd = mh/(np.sqrt(vh)+eps)
        # tangent
        upd = upd - np.sum(upd*x, axis=1, keepdims=True)*x
        x = x - lr*upd
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr==0]=1.0
        x/=nr
    return x

def mover_quench(x_init, movers, steps, T0, T1, lr):
    x = x_init.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    l0=np.log(T0); l1=np.log(T1)
    mov = np.zeros(n, dtype=bool); mov[movers]=True
    for s in range(1, steps+1):
        T = float(np.exp(l0+(l1-l0)*(s/steps)))
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        M = G.max()
        W = np.exp((G-M)/T)
        np.fill_diagonal(W, 0.0)
        Grad = (W @ x)/T
        dot = np.sum(Grad*x, axis=1, keepdims=True)
        Gt = Grad - dot*x
        # only movers get momentum update, but keep m,v for all for bias simplicity
        m = 0.9*m + 0.1*Gt
        v = 0.999*v + 0.001*(Gt*Gt)
        mh = m/(1.0-0.9**s)
        vh = v/(1.0-0.999**s)
        upd = mh/(np.sqrt(vh)+1e-8)
        upd = upd - np.sum(upd*x, axis=1, keepdims=True)*x
        upd[~mov]=0.0
        x = x - lr*upd
        nr=np.linalg.norm(x,axis=1,keepdims=True); nr[nr==0]=1.0; x/=nr
    return x

def lp_polish(x, seconds, t_end):
    x = x/np.linalg.norm(x,axis=1,keepdims=True)
    n,d = x.shape
    best = max_cos(x)
    radius=1e-3
    start=time.time()
    iu=np.triu_indices(n,1)
    while time.time()-start<seconds and time.time()<t_end:
        if radius<1e-14: break
        c=(x@x.T)[iu]
        mcur=c.max()
        sel=c>mcur-max(20*radius,1e-9)
        I,J=iu[0][sel],iu[1][sel]
        p=len(I)
        if p==0: break
        nv=n*d+1
        rows=np.repeat(np.arange(p),2*d+1)
        cols=np.concatenate([I[:,None]*d+np.arange(d),J[:,None]*d+np.arange(d),np.full((p,1),n*d)],1).ravel()
        vals=np.concatenate([x[J],x[I],-np.ones((p,1))],1).ravel()
        A_ub=coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
        A_eq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        obj=np.zeros(nv); obj[-1]=1.0
        try:
            res=linprog(obj,A_ub=A_ub,b_ub=-c[sel],A_eq=A_eq,b_eq=np.zeros(n),
                        bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        y=x+res.x[:-1].reshape(n,d)
        y/=np.linalg.norm(y,axis=1,keepdims=True)
        vv=max_cos(y)
        if vv<best:
            x,best,radius=y,vv,min(radius*1.5,0.05)
        else:
            radius*=0.5
    return x

def refine_hole(q, Xrem, iters=250, T=0.004, lr=0.05):
    q=q.copy()/np.linalg.norm(q)
    for _ in range(iters):
        dots=Xrem@q
        mm=dots.max()
        w=np.exp((dots-mm)/T)
        g=(w[:,None]*Xrem).sum(axis=0)
        g=g-np.dot(g,q)*q
        n=np.linalg.norm(g)
        if n==0 or not np.isfinite(n): break
        g/=n
        q=q-lr*g
        q/=np.linalg.norm(q)
        lr*=0.995
    return q

def find_hole(Xrem, rng, n_samp=12000, Tref=0.004):
    d=Xrem.shape[1]
    S=rng.standard_normal((n_samp,d))
    S/=np.linalg.norm(S,axis=1,keepdims=True)
    dots=S@Xrem.T
    worst=dots.max(axis=1)
    bi=int(np.argmin(worst))
    q=S[bi]
    q=refine_hole(q,Xrem,iters=250,T=Tref,lr=0.05)
    return q

def solve():
    t0=time.time()
    T_LIM=172.0
    t_end=t0+T_LIM
    rng=np.random.default_rng(7)
    cands=[]
    for fn in ("best.npy","record.npy"):
        try:
            a=np.load(fn).astype(float)
            a/=np.linalg.norm(a,axis=1,keepdims=True)
            if a.shape==(102,12):
                cands.append((max_cos(a),a))
        except Exception:
            pass
    if len(cands)==0:
        a=rng.standard_normal((102,12)); a/=np.linalg.norm(a,axis=1,keepdims=True)
        x_best=a; best=max_cos(a)
    else:
        cands.sort(key=lambda t:t[0])
        best,x_best=cands[0][0],cands[0][1].copy()
    n,d=102,12
    x_cur=x_best.copy(); v_cur=best
    basin_T=0.003
    it=0
    # explore until reserve for final polish
    while time.time()<t_end-22.0 and it<500:
        it+=1
        mode=it%6
        # stats of cur
        Gc=x_cur@x_cur.T
        np.fill_diagonal(Gc,-2.0)
        rowmax=Gc.max(axis=1)
        order=np.argsort(-rowmax)
        # contact count for overload
        mval=Gc.max()
        cnt=(Gc>mval-0.004).sum(axis=1)
        over=np.argsort(-cnt)
        x_try=x_cur.copy()
        movers=None
        if mode==0:
            x_try=x_cur+0.012*rng.standard_normal((n,d))
            steps,T0,T1,lr=600,0.02,0.004,0.0025
        elif mode==1:
            k=int(rng.integers(2,4))  # 2..3
            idx=order[:k]
            x_try[idx]+=0.09*rng.standard_normal((k,d))
            x_try+=0.003*rng.standard_normal((n,d))
            movers=idx
            steps,T0,T1,lr=850,0.025,0.0035,0.0022
        elif mode==2:
            ci=int(rng.integers(n))
            Gr=(x_cur@x_cur[ci]); Gr[ci]=-2
            nbr=np.argsort(-Gr)[:5]
            cl=np.append(nbr,ci)
            x_try[cl]+=0.055*rng.standard_normal((len(cl),d))
            x_try+=0.003*rng.standard_normal((n,d))
            movers=cl
            steps,T0,T1,lr=850,0.02,0.003,0.0025
        elif mode==3:
            nk=int(rng.integers(1,3))  #1-2
            # prefer overloaded among worst
            pool=np.concatenate([order[:6],over[:6]])
            # unique
            _,ui=np.unique(pool,return_index=True)
            pool=np.sort(ui)
            # actually simpler: take worst by rowmax
            worst=order[:nk]
            mask=np.ones(n,dtype=bool); mask[worst]=False
            Xrem=x_cur[mask]
            newpts=[]
            Xc=Xrem.copy()
            for wi in worst:
                q=find_hole(Xc,rng,n_samp=12000,Tref=0.005)
                newpts.append(q)
                Xc=np.vstack([Xc,q[None,:]])
            for k2,q in zip(worst,newpts):
                x_try[k2]=q
            # small jitter others
            x_try[mask]+=0.002*rng.standard_normal((n-nk,d))
            movers=np.array(worst)
            steps,T0,T1,lr=900,0.018,0.003,0.002
        elif mode==4:
            k=int(rng.integers(6,10))
            idx=rng.choice(n,k,replace=False)
            # full random replacement for half, jitter for rest
            for j in idx[:max(1,k//3)]:
                v=rng.standard_normal(d); v/=np.linalg.norm(v); x_try[j]=v
            x_try[idx]+=0.02*rng.standard_normal((k,d))
            movers=idx
            steps,T0,T1,lr=1000,0.03,0.004,0.0025
        else:
            x_try=x_cur+0.025*rng.standard_normal((n,d))
            steps,T0,T1,lr=700,0.035,0.006,0.003

        x_try/=np.linalg.norm(x_try,axis=1,keepdims=True)
        # mover-only pre-refine if hole/cluster
        if movers is not None and len(movers)<=10 and mode in (1,2,3,4):
            # quick mover focus at higher T
            try:
                x_try=mover_quench(x_try, movers, steps=250, T0=0.008, T1=0.003, lr=0.01)
            except Exception:
                pass
        try:
            x_q=adam_quench(x_try, steps=steps, T0=T0, T1=T1, lr=lr)
        except Exception:
            continue
        vq=max_cos(x_q)
        # accept to cur with annealing
        if vq<v_cur or rng.random()<np.exp((v_cur-vq)/basin_T):
            x_cur=x_q; v_cur=vq
            # cool slowly
            basin_T=max(0.0008,basin_T*0.9995)
        # if promising, LP snap
        improve = vq < best - 2e-9
        near = vq < best + 0.002
        if improve or (near and (it%7==0)):
            # quick polish 2.5s
            try:
                x_p=lp_polish(x_q, seconds=2.5, t_end=t_end-20.0)
            except Exception:
                x_p=x_q
            vp=max_cos(x_p)
            if vp<best:
                best=x_best=x_p.copy() if False else x_p.copy()
                x_best=x_p.copy(); best=vp
                x_cur=x_p.copy(); v_cur=vp
            elif vp<v_cur and rng.random()<0.5:
                x_cur=x_p.copy(); v_cur=vp
        # opportunistic global best from cur wandering even without polish
        if vq<best-2e-9:
            # will have been polished above, but keep raw if polish failed
            pass
        if time.time()>=t_end-22.0:
            break
    # final intensive polish of best
    try:
        # low-T quench then LP
        x_best=adam_quench(x_best, steps=1200, T0=0.006, T1=0.0012, lr=0.0015)
        best=max_cos(x_best)
    except Exception:
        pass
    remain=max(4.0, t_end-time.time())
    try:
        x_best=lp_polish(x_best, seconds=remain-1.0, t_end=t_end)
    except Exception:
        pass
    x_best/=np.linalg.norm(x_best,axis=1,keepdims=True)
    return x_best
