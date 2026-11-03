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
        Mx = G.max()
        W = np.exp((G-Mx)/T)
        np.fill_diagonal(W, 0.0)
        Grad = (W @ x)/max(T,1e-12)
        dot = np.sum(Grad*x, axis=1, keepdims=True)
        Gt = Grad - dot*x
        m = b1*m + (1.0-b1)*Gt
        v = b2*v + (1.0-b2)*(Gt*Gt)
        mh = m/(1.0-b1**s); vh = v/(1.0-b2**s)
        upd = mh/(np.sqrt(vh)+eps)
        x = x - lr*upd
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr==0]=1.0
        x /= nr
    return x

def lp_polish(x, seconds, t_end):
    x = x/np.linalg.norm(x, axis=1, keepdims=True)
    n,d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n,1)
    while time.time()-start < seconds and time.time() < t_end:
        if radius < 1e-13:
            break
        c = (x@x.T)[iu]
        mcur = c.max()
        sel = c > mcur - max(20*radius, 2e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p==0:
            break
        if p>2500:
            # keep most-violated to bound LP size
            idx = np.argsort(-c[sel])[:2500]
            I,J = I[idx],J[idx]
            p = len(I)
            csv = c[sel][idx]
        else:
            csv = c[sel]
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1),n*d)],1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
        A_ub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
        A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        obj = np.zeros(nv); obj[-1]=1.0
        try:
            res = linprog(obj,A_ub=A_ub,b_ub=-csv,A_eq=A_eq,b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        y = x + res.x[:-1].reshape(n,d)
        y /= np.linalg.norm(y,axis=1,keepdims=True)
        vv = max_cos(y)
        if vv < best:
            x,best,radius = y,vv,min(radius*1.5,0.02)
        else:
            radius*=0.5
    return x

def gen_pool(rng, target=9000, scale2=0.5):
    d=12
    P=[]
    # weight-1
    for i in range(d):
        for s in (-1.0,1.0):
            v=np.zeros(d); v[i]=s; P.append(v)
    # weight-2 (1,1)/sqrt2 all
    import itertools
    for i,j in itertools.combinations(range(d),2):
        for si in (-1.0,1.0):
            for sj in (-1.0,1.0):
                v=np.zeros(d); v[i]=si/np.sqrt(2); v[j]=sj/np.sqrt(2); P.append(v)
    P=np.array(P)
    # random weight-3,4,6 signed
    def rand_w(k,n):
        V=np.zeros((n,d))
        for t in range(n):
            idx=rng.choice(d,k,replace=False)
            sg=rng.choice(np.array([-1.0,1.0]),k)
            V[t,idx]=sg/np.sqrt(k)
        return V
    # two-scale layered: support 6, 2 entries amp 1, 4 entries amp s
    def rand_twoscale(n, s):
        V=np.zeros((n,d))
        for t in range(n):
            idx=rng.choice(d,6,replace=False)
            big=rng.choice(6,2,replace=False)
            sg=rng.choice(np.array([-1.0,1.0]),6)
            v=np.zeros(d)
            for k2 in range(6):
                amp=1.0 if k2 in big else s
                v[idx[k2]]=sg[k2]*amp
            v/=np.linalg.norm(v)
            V[t]=v
        return V
    A=rand_w(3,1500); B=rand_w(4,2500); C=rand_w(6,1500)
    D1=rand_twoscale(1200,scale2); D2=rand_twoscale(800,0.3 if scale2!=0.3 else 0.7)
    # random dense uniform sphere
    R=rng.standard_normal((800,d)); R/=np.linalg.norm(R,axis=1,keepdims=True)
    P=np.vstack([P,A,B,C,D1,D2,R])
    # dedup roughly by rounding? skip
    return P

def greedy_select(P, rng, ktie=16):
    M=P.shape[0]; n=102
    # normalize pool
    Pn=P/np.linalg.norm(P,axis=1,keepdims=True)
    start=int(rng.integers(M))
    sel=[start]
    curmax=Pn@Pn[start]  # size M
    curmax[start]=np.inf  # avoid reselect
    # mask selected
    selected_mask=np.zeros(M,bool); selected_mask[start]=True
    for _ in range(1,n):
        # find ktie smallest curmax among unselected
        # argpartition for speed
        idx=np.argpartition(curmax, ktie+1)[:ktie+1]
        # filter selected (rare)
        idx=idx[~selected_mask[idx]]
        if len(idx)==0:
            idx=np.where(~selected_mask)[0][:ktie]
        # randomized tie-break: softmax over -curmax
        vals=curmax[idx]
        # prefer smaller; add randomness
        # random choice weighted by rank
        # temperature
        w=np.exp(-(vals-vals.min())/max(1e-4, (vals.max()-vals.min()+1e-12)*0.3))
        w/=w.sum()
        pick=int(rng.choice(idx,p=w))
        sel.append(pick)
        selected_mask[pick]=True
        d2=Pn@Pn[pick]
        d2[selected_mask]=np.inf  # keep inf for selected? but curmax for selected unused
        # update: curmax = max(curmax, d2) for unselected
        # for selected entries keep inf
        curmax=np.maximum(curmax,d2)
        curmax[selected_mask]=np.inf
    X=Pn[np.array(sel)]
    # random global sign flips (pool already symmetric but keep)
    return X

def apply_layer_params(X, s, phi, ax=(0,6)):
    # s: scale second half (layer scale), phi: Givens rotation in plane ax
    Y=X.copy()
    Y[:,6:]*=s
    i,j=ax
    ci=np.cos(phi); si=np.sin(phi)
    c1=Y[:,i].copy(); c2=Y[:,j].copy()
    Y[:,i]=ci*c1-si*c2; Y[:,j]=si*c1+ci*c2
    Y/=np.linalg.norm(Y,axis=1,keepdims=True)
    return Y

def solve():
    t0=time.time(); T_END=t0+172.0
    rng=np.random.default_rng(12345)
    # load incumbent (only for final comparison)
    x_inc=None; v_inc=np.inf
    for fn in ("best.npy","record.npy"):
        try:
            a=np.load(fn).astype(float)
            a/=np.linalg.norm(a,axis=1,keepdims=True)
            if a.shape==(102,12):
                vv=max_cos(a)
                if vv<v_inc:
                    v_inc=vv; x_inc=a.copy()
        except Exception:
            pass
    if x_inc is None:
        x_inc=rng.standard_normal((102,12)); x_inc/=np.linalg.norm(x_inc,axis=1,keepdims=True)
        v_inc=max_cos(x_inc)
    x_best=x_inc.copy(); best=v_inc

    # ---- Phase 1: structured independent search ----
    best_struct=None; best_struct_val=np.inf
    # few template families with different two-scale
    scales=[0.5,0.35,0.65]
    ntrials=6
    for t in range(ntrials):
        if time.time()>T_END-85.0:
            break
        sc=scales[t%len(scales)]
        P=gen_pool(rng,target=9000,scale2=sc)
        X0=greedy_select(P,rng,ktie=14+ (t%3)*4)
        v0=max_cos(X0)
        # jointly optimize small layer params (s, phi) + positions:
        # try few (s,phi) candidates, short quench each, keep best
        cands=[(1.0,0.0),(0.85,0.1),(1.15,-0.08)]
        if t%2==0:
            cands=[(1.0,0.0),(0.9,0.15),(1.1,0.2)]
        bx=None; bv=np.inf
        for (sm,ph) in cands:
            Y=apply_layer_params(X0,sm,ph)
            Yq=adam_quench(Y,steps=500,T0=0.02,T1=0.004,lr=0.0025)
            vv=max_cos(Yq)
            if vv<bv:
                bv=vv; bx=Yq
            if time.time()>T_END-85.0:
                break
        # longer anneal on best layer choice
        if bx is not None:
            bx2=adam_quench(bx,steps=700,T0=0.012,T1=0.0025,lr=0.002)
            bv2=max_cos(bx2)
            if bv2<bv:
                bx,bv=bx2,bv2
        if bv<best_struct_val:
            best_struct_val=bv; best_struct=bx.copy()
    # refine best structured basin with stronger anneal + short LP
    if best_struct is not None and time.time()<T_END-70.0:
        # annealed descent
        best_struct=adam_quench(best_struct,steps=1200,T0=0.015,T1=0.002,lr=0.0022)
        best_struct_val=max_cos(best_struct)
        # snap with LP (short)
        budget=min(6.0,max(2.0,T_END-70.0-time.time()))
        if time.time()+budget<T_END-60.0:
            best_struct=lp_polish(best_struct,budget,T_END-60.0)
            best_struct_val=max_cos(best_struct)
        if best_struct_val<best:
            x_best,best=best_struct,best_struct_val

    # ---- Phase 2: use remaining time to polish global best ----
    # alternate small kicks on current best + quench + LP snap
    it=0
    while time.time()<T_END-12.0 and it<40:
        it+=1
        if time.time()>T_END-12.0:
            break
        # choose source: mostly x_best, occasionally best_struct for diversity
        src=x_best
        G0=src@src.T; np.fill_diagonal(G0,-2.0)
        rowmax=G0.max(axis=1); order=np.argsort(-rowmax)
        x_try=src.copy()
        mode=it%3
        if mode==0:
            x_try+=0.008*rng.standard_normal(x_try.shape)
            steps,T0,T1,lr=600,0.015,0.003,0.002
        elif mode==1:
            k=3; idx=order[:k]
            x_try[idx]+=0.05*rng.standard_normal((k,12))
            x_try+=0.003*rng.standard_normal(x_try.shape)
            steps,T0,T1,lr=700,0.018,0.003,0.002
        else:
            wi=int(order[0])
            mask=np.ones(102,bool); mask[wi]=False
            # best hole among samples
            S=rng.standard_normal((4000,12)); S/=np.linalg.norm(S,axis=1,keepdims=True)
            dots=S@src[mask].T; worst=dots.max(axis=1)
            hole=S[int(np.argmin(worst))]
            x_try[wi]=hole+0.004*rng.standard_normal(12)
            x_try[mask]+=0.003*rng.standard_normal((101,12))
            steps,T0,T1,lr=700,0.015,0.003,0.002
        x_try/=np.linalg.norm(x_try,axis=1,keepdims=True)
        x_q=adam_quench(x_try,steps,T0,T1,lr)
        vq=max_cos(x_q)
        if vq<best-1e-12:
            budget=min(3.0,max(1.0,(T_END-12.0-time.time())/6.0))
            if time.time()+budget<T_END-10.0:
                x_q=lp_polish(x_q,budget,T_END-10.0)
                vq=max_cos(x_q)
            if vq<best:
                x_best,best=x_q,vq
        if time.time()>=T_END-12.0:
            break
    # final snap of incumbent
    remain=T_END-time.time()-1.5
    if remain>2.0:
        x_best=lp_polish(x_best,min(remain,10.0),T_END-0.5)
        best=max_cos(x_best)
    x_best/=np.linalg.norm(x_best,axis=1,keepdims=True)
    # safety: if somehow worse than loaded incumbent, return incumbent
    if best>v_inc+1e-12 and x_inc is not None:
        # re-polish incumbent briefly? just return it
        return x_inc/np.linalg.norm(x_inc,axis=1,keepdims=True)
    return x_best
