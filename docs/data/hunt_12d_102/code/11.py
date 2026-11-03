import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def quench_softmax(x_init, steps, T0, T1, lr, rng=None):
    x = x_init.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    l0 = np.log(T0); l1 = np.log(T1)
    for s in range(1, steps+1):
        T = float(np.exp(l0 + (l1-l0)*(s/steps)))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        M = G.max()
        E = np.exp((G-M)/T)
        np.fill_diagonal(E, 0.0)
        S = E.sum()
        if S <= 0 or not np.isfinite(S):
            break
        P = E / S
        Grad = P @ x
        dot = np.sum(Grad*x, axis=1, keepdims=True)
        Gt = Grad - dot*x
        m = b1*m + (1-b1)*Gt
        v = b2*v + (1-b2)*(Gt*Gt)
        mh = m/(1-b1**s); vh = v/(1-b2**s)
        upd = mh/(np.sqrt(vh)+eps)
        # keep tangential
        upd = upd - np.sum(upd*x,axis=1,keepdims=True)*x
        x = x - lr*upd
        x /= np.linalg.norm(x,axis=1,keepdims=True)
    return x

def quench_power(x_init, steps, p0, p1, lr):
    x = x_init.copy()
    n,d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1,b2,eps = 0.9,0.999,1e-8
    for s in range(1, steps+1):
        p = p0 + (p1-p0)*(s/steps)
        G = x @ x.T
        np.fill_diagonal(G, 0.0)
        # only repulsive for positive dots to focus on max; shift
        Gp = np.maximum(G, 0.0)
        # avoid overflow: scale by max
        M = Gp.max()
        if M <= 1e-12:
            break
        W = (Gp/M)**(p-1.0) / (M+1e-12)
        np.fill_diagonal(W, 0.0)
        Grad = W @ x
        dot = np.sum(Grad*x,axis=1,keepdims=True)
        Gt = Grad - dot*x
        # normalize grad scale
        gn = np.linalg.norm(Gt)
        if not np.isfinite(gn) or gn==0:
            break
        # scale to comparable magnitude
        Gt = Gt/(gn/np.sqrt(n)+1e-12)*0.1
        m = b1*m+(1-b1)*Gt
        v = b2*v+(1-b2)*(Gt*Gt)
        mh=m/(1-b1**s); vh=v/(1-b2**s)
        upd=mh/(np.sqrt(vh)+eps)
        upd = upd - np.sum(upd*x,axis=1,keepdims=True)*x
        x = x - lr*upd
        nr=np.linalg.norm(x,axis=1,keepdims=True); nr[nr==0]=1.0
        x/=nr
    return x

def lp_polish(x, seconds, t_end):
    x = x/np.linalg.norm(x,axis=1,keepdims=True)
    n,d = x.shape
    best = max_cos(x)
    radius=1e-3
    start=time.time()
    iu=np.triu_indices(n,1)
    while time.time()-start<seconds and time.time()<t_end:
        if radius<1e-14:
            break
        c=(x@x.T)[iu]
        mcur=c.max()
        sel=c>mcur-max(20*radius,1e-9)
        I,J=iu[0][sel],iu[1][sel]
        p=len(I)
        if p==0:
            break
        nv=n*d+1
        rows=np.repeat(np.arange(p),2*d+1)
        cols=np.concatenate([I[:,None]*d+np.arange(d),J[:,None]*d+np.arange(d),np.full((p,1),n*d)],1).ravel()
        vals=np.concatenate([x[J],x[I],-np.ones((p,1))],1).ravel()
        A_ub=coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
        A_eq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        obj=np.zeros(nv);obj[-1]=1.0
        try:
            res=linprog(obj,A_ub=A_ub,b_ub=-c[sel],A_eq=A_eq,b_eq=np.zeros(n),
                        bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5;continue
        if res.status!=0:
            radius*=0.5;continue
        y=x+res.x[:-1].reshape(n,d)
        y/=np.linalg.norm(y,axis=1,keepdims=True)
        vv=max_cos(y)
        if vv<best:
            x,best,radius=y,vv,min(radius*1.5,0.05)
        else:
            radius*=0.5
    return x

def best_hole(X_rem, rng, n_samp=8000):
    d=X_rem.shape[1]
    S=rng.standard_normal((n_samp,d))
    S/=np.linalg.norm(S,axis=1,keepdims=True)
    dots=S@X_rem.T
    worst=dots.max(axis=1)
    q=S[int(np.argmin(worst))].copy()
    # refine: gradient descent on softmax max
    for _ in range(250):
        dv=X_rem@q
        M=dv.max()
        w=np.exp((dv-M)/0.004)
        ws=w.sum()
        if ws==0 or not np.isfinite(ws):
            break
        g=(w/ws)@X_rem
        g=g-np.dot(g,q)*q
        q=q-0.02*g
        nq=np.linalg.norm(q)
        if nq==0 or not np.isfinite(nq):
            break
        q/=nq
    return q

def solve():
    t0=time.time()
    T_LIMIT=172.0
    t_end=t0+T_LIMIT
    rng=np.random.default_rng(42)
    cands=[]
    for f in ("best.npy","record.npy"):
        try:
            a=np.array(np.load(f),dtype=float)
            a/=np.linalg.norm(a,axis=1,keepdims=True)
            if a.shape==(102,12):
                cands.append((max_cos(a),a))
        except Exception:
            pass
    if len(cands)==0:
        a=rng.standard_normal((102,12));a/=np.linalg.norm(a,axis=1,keepdims=True)
        x_best=a;best=max_cos(a)
    else:
        cands.sort(key=lambda t:t[0])
        best,x_best=cands[0][0],cands[0][1].copy()
    n,d=102,12
    # tabu memory of accepted side moves to diversify
    stuck=0
    it=0
    # reserve final polish
    while time.time()<t_end-22.0 and it<600:
        it+=1
        G0=x_best@x_best.T
        np.fill_diagonal(G0,-2.0)
        rowmax=G0.max(axis=1)
        order=np.argsort(-rowmax)
        mval=G0.max()
        cnt=(G0>mval-0.002).sum(axis=1)
        over=np.argsort(-cnt)
        under=np.argsort(cnt)
        x_try=x_best.copy()
        boost=1.0+min(stuck,60)/20.0
        mode=it%6
        use_power=(it%3==2)
        if mode==0:
            sig=0.012*boost
            x_try=x_best+sig*rng.standard_normal((n,d))
            steps,T0,T1,lr=600,0.018,0.0035,0.0022
        elif mode==1:
            k=3
            idx=order[:k]
            x_try[idx]+=0.10*boost*rng.standard_normal((k,d))
            x_try+=0.004*rng.standard_normal((n,d))
            steps,T0,T1,lr=850,0.024,0.003,0.0022
        elif mode==2:
            # overloaded hub kick
            k=4
            idx=over[:k]
            x_try[idx]+=0.07*boost*rng.standard_normal((k,d))
            x_try+=0.003*rng.standard_normal((n,d))
            steps,T0,T1,lr=850,0.02,0.0028,0.002
        elif mode==3:
            # loose pole / hole re-insertion (1-2 points)
            nk=2 if (it%2==0) else 1
            victims=under[:nk]
            # also include worst
            if rng.random()<0.5:
                victims=np.array([order[0]])
                if nk==2:
                    victims=np.array([order[0],under[0]])
            mask=np.ones(n,dtype=bool);mask[victims]=False
            Xrem=x_best[mask]
            for vi in victims:
                h=best_hole(Xrem,rng,n_samp=8000)
                x_try[vi]=h+0.004*rng.standard_normal(d)
                # update Xrem for second victim
                Xrem=np.vstack([Xrem,h[None,:]])
            x_try[mask]+=0.004*rng.standard_normal((n-len(victims),d))
            steps,T0,T1,lr=900,0.019,0.0028,0.002
        elif mode==4:
            # cluster surgery: random center + 6 nearest
            ci=int(rng.integers(n))
            Gr=(x_best@x_best[ci]);Gr[ci]=-2.0
            nbr=np.argsort(-Gr)[:6]
            cl=np.append(nbr,ci)
            x_try[cl]+=0.065*boost*rng.standard_normal((len(cl),d))
            x_try+=0.003*rng.standard_normal((n,d))
            steps,T0,T1,lr=850,0.021,0.003,0.002
        else:
            # big patch reinit: 8-12 random points jitter large
            k=int(rng.integers(8,13))
            idx=rng.choice(n,k,replace=False)
            x_try[idx]+=0.12*boost*rng.standard_normal((k,d))
            x_try+=0.005*rng.standard_normal((n,d))
            steps,T0,T1,lr=1000,0.028,0.0035,0.0024
        x_try/=np.linalg.norm(x_try,axis=1,keepdims=True)
        if use_power:
            p0,p1=60.0,220.0
            # vary p range randomly a bit
            p1=p1*(0.8+0.4*rng.random())
            x_q=quench_power(x_try,steps,p0,p1,0.003)
            # follow with short softmax cool
            x_q=quench_softmax(x_q,200,T1,0.0015,0.0015)
        else:
            x_q=quench_softmax(x_try,steps,T0,T1,lr)
        vq=max_cos(x_q)
        if vq<best-1e-12:
            budget=min(4.0,max(1.0,(t_end-22.0-time.time())/4.0))
            if time.time()+budget<t_end-20.0:
                x_q=lp_polish(x_q,budget,t_end-20.0)
                vq=max_cos(x_q)
            if vq<best:
                x_best,best=x_q,vq
                stuck=0
                continue
        # side-move acceptance to change basin when stuck: accept small worsening rarely
        if stuck>25 and vq<best+0.002 and (it%7==0):
            # accept as new base to diversify (tabu)
            x_best=x_q
            best=vq  # allow drift; will recover
            stuck=0
            continue
        if vq<best+0.004 and (it%6==0):
            budget=min(2.2,max(0.7,(t_end-22.0-time.time())/8.0))
            if budget>0.6 and time.time()+budget<t_end-20.0:
                x_p=lp_polish(x_q,budget,t_end-20.0)
                vp=max_cos(x_p)
                if vp<best:
                    x_best,best=x_p,vp
                    stuck=0
                    continue
        stuck+=1
        if time.time()>=t_end-22.0:
            break
    remain=t_end-time.time()-2.0
    if remain>2.0:
        x_best=lp_polish(x_best,min(remain,18.0),t_end-1.0)
    x_best/=np.linalg.norm(x_best,axis=1,keepdims=True)
    return x_best
