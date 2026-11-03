import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _build_e8():
    pts = []
    for i in range(8):
        for j in range(i+1, 8):
            for s1 in (-1.0, 1.0):
                for s2 in (-1.0, 1.0):
                    v = np.zeros(8); v[i]=s1; v[j]=s2
                    pts.append(v)
    for mask in range(256):
        v = np.ones(8)*0.5
        nminus=0
        for k in range(8):
            if (mask>>k)&1:
                v[k]=-0.5; nminus+=1
        if nminus%2==0:
            pts.append(v.copy())
    E=np.array(pts,float)
    E/=np.linalg.norm(E,axis=1,keepdims=True)
    return E

def _build_group(m, freqs):
    # freqs len 4, 0 => identity plane; else rotation by 2pi*k*f/m
    Gs=np.zeros((m,8,8))
    for k in range(m):
        G=np.zeros((8,8))
        for p,f in enumerate(freqs):
            a=2.0*np.pi*k*float(f)/float(m) if f!=0 else 0.0
            c=np.cos(a); s=np.sin(a)
            G[2*p,2*p]=c; G[2*p,2*p+1]=-s
            G[2*p+1,2*p]=s; G[2*p+1,2*p+1]=c
            if f==0:
                G[2*p,2*p]=1.0; G[2*p,2*p+1]=0.0
                G[2*p+1,2*p]=0.0; G[2*p+1,2*p+1]=1.0
        Gs[k]=G
    return Gs

def _expand(S, Gs):
    ns=S.shape[0]; m=Gs.shape[0]
    X=np.empty((ns*m,8))
    for k in range(m):
        X[k::m] if False else None
    # ordering j*m+k
    for j in range(ns):
        s=S[j]
        for k in range(m):
            X[j*m+k]=Gs[k]@s
    return X

def _expand_fast(S, Gs, GsT=None):
    # S (ns,8) row vectors, X[j,k]=S[j]@Gk.T
    ns,m=S.shape[0],Gs.shape[0]
    X=np.empty((ns*m,8))
    for k in range(m):
        Gk=Gs[k]
        # S @ Gk.T
        Xk=S@Gk.T
        for j in range(ns):
            X[j*m+k]=Xk[j]
    return X

def _sym_quench(S0, Gs, steps, b0, b1, lr0, lr1, rng, noise=0.0, mu=0.8, t_lim=None):
    S=S0.copy()
    S/=np.linalg.norm(S,axis=1,keepdims=True)
    ns=S.shape[0]; m=Gs.shape[0]; n=ns*m
    V=np.zeros_like(S)
    lb0=np.log(b0); lb1=np.log(b1); ll0=np.log(lr0); ll1=np.log(lr1)
    bestV=1e9; bestS=S.copy()
    for k in range(steps):
        if t_lim is not None and (k&63)==0 and time.time()>t_lim:
            break
        f=k/max(steps-1,1)
        beta=float(np.exp(lb0+(lb1-lb0)*f))
        lr=float(np.exp(ll0+(ll1-ll0)*f))
        X=_expand_fast(S,Gs)
        G=X@X.T
        np.fill_diagonal(G,-2.0)
        mv=float(G.max())
        if mv<bestV:
            bestV=mv; bestS=S.copy()
        D=G-mv
        cut=9.0/beta+0.012
        W=np.zeros_like(D)
        mk=D>-cut
        W[mk]=np.exp(beta*D[mk])
        np.fill_diagonal(W,0.0)
        Grad=W@X
        dot=np.sum(Grad*X,axis=1,keepdims=True)
        Gt=Grad-dot*X
        Gtr=Gt.reshape(ns,m,8)
        # Ds[j,:]=sum_k Gt[j,k,:]@Gk
        Ds=np.einsum('jkl,klm->jm',Gtr,Gs)
        # project to tangent of seeds
        d2=np.sum(Ds*S,axis=1,keepdims=True)
        Dst=Ds-d2*S
        rms=float(np.sqrt(np.mean(Dst*Dst)))+1e-12
        Dst/=rms
        V=mu*V+(1.0-mu)*Dst
        S=S-lr*V
        if noise>0:
            ns_ = noise*(1.0-f)*lr
            if ns_>1e-12:
                Z=rng.standard_normal(size=S.shape)
                Z-=np.sum(Z*S,axis=1,keepdims=True)*S
                S=S+ns_*Z
        S/=np.linalg.norm(S,axis=1,keepdims=True)
    # evaluate best
    Xb=_expand_fast(bestS,Gs)
    bv=_max_cos(Xb)
    return bestS,bv

def _free_anneal(X0, steps, b0, b1, l0, l1, rng, noise=0.0, mu=0.8, t_lim=None):
    X=X0.copy(); n,d=X.shape
    V=np.zeros_like(X)
    G0=X@X.T; np.fill_diagonal(G0,-2.0)
    bestV=float(G0.max()); bestX=X.copy()
    lb0=np.log(b0); lb1=np.log(b1); ll0=np.log(l0); ll1=np.log(l1)
    for k in range(steps):
        if t_lim is not None and (k&127)==0 and time.time()>t_lim:
            break
        f=k/max(steps-1,1)
        beta=float(np.exp(lb0+(lb1-lb0)*f)); lr=float(np.exp(ll0+(ll1-ll0)*f))
        G=X@X.T; np.fill_diagonal(G,-2.0)
        m=float(G.max())
        if m<bestV:
            bestV=m; bestX=X.copy()
        D=G-m
        cut=9.0/beta+0.012
        W=np.zeros_like(D)
        mk=D>-cut
        W[mk]=np.exp(beta*D[mk])
        np.fill_diagonal(W,0.0)
        Grad=W@X
        dot=np.sum(Grad*X,axis=1,keepdims=True)
        Gt=Grad-dot*X
        rms=float(np.sqrt(np.mean(Gt*Gt)))+1e-12)
        Gt/=rms
        V=mu*V+(1.0-mu)*Gt
        X=X-lr*V
        if noise>0:
            ns_=noise*(1.0-f)*lr
            if ns_>1e-12:
                Z=rng.standard_normal(size=X.shape)
                Z-=np.sum(Z*X,axis=1,keepdims=True)*X
                X=X+ns_*Z
        X/=np.linalg.norm(X,axis=1,keepdims=True)
    G=X@X.T; np.fill_diagonal(G,-2.0); m=float(G.max())
    if m<bestV:
        bestV=m; bestX=X.copy()
    return bestX,bestV

def _lp_polish(x, seconds, deadline=None):
    x=x/np.linalg.norm(x,axis=1,keepdims=True)
    n,d=x.shape
    best=_max_cos(x)
    radius=1e-3
    start=time.time()
    iu=np.triu_indices(n,1)
    while time.time()-start<seconds:
        if deadline is not None and time.time()>deadline:
            break
        if radius<1e-14:
            break
        xxT=x@x.T
        c_all=xxT[iu]
        m=float(c_all.max())
        sel=c_all>m-max(20*radius,1e-9)
        idx=np.where(sel)[0]
        if len(idx)>4000:
            order=np.argsort(-c_all[idx])[:4000]
            idx=idx[order]
        if len(idx)==0:
            radius*=0.5; continue
        I=iu[0][idx]; J=iu[1][idx]
        csel=c_all[idx]; p=len(I)
        nv=n*d+1
        rows=np.repeat(np.arange(p),2*d+1)
        cols=np.concatenate([I[:,None]*d+np.arange(d),J[:,None]*d+np.arange(d),np.full((p,1),n*d)],1).ravel()
        vals=np.concatenate([x[J],x[I],-np.ones((p,1))],1).ravel()
        A_ub=coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
        A_eq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        obj=np.zeros(nv); obj[-1]=1.0
        try:
            res=linprog(obj,A_ub=A_ub,b_ub=-csel,A_eq=A_eq,b_eq=np.zeros(n),bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        y=x+res.x[:-1].reshape(n,d)
        y/=np.linalg.norm(y,axis=1,keepdims=True)
        v=_max_cos(y)
        if v<best:
            x,best,radius=y,v,min(radius*1.5,0.05)
        else:
            radius*=0.5
    return x

def _greedy_e8_subset(E8, target=168):
    # iteratively delete vector with most ~0.5 contacts
    idx=np.arange(len(E8))
    cur=np.arange(len(E8))
    G=E8@E8.T
    # count pairs with dot>0.49
    while len(cur)>target:
        sub=G[np.ix_(cur,cur)]
        np.fill_diagonal(sub, -2.0)
        cnt=np.sum(sub>0.49,axis=1)
        # also consider per max
        worst=int(np.argmax(cnt))
        cur=np.delete(cur,worst)
    return E8[cur].copy()

def solve():
    t0=time.time(); DEADLINE=t0+175.0
    rng=np.random.default_rng(12345)
    try:
        bestX=np.load("best.npy").astype(float)
    except Exception:
        try:
            bestX=np.load("record.npy").astype(float)
        except Exception:
            bestX=rng.standard_normal(size=(168,8))
    if bestX.shape!=(168,8):
        tmp=rng.standard_normal(size=(168,8))
        try:
            m=min(len(bestX),168); dd=min(bestX.shape[1],8)
            tmp[:m,:dd]=bestX[:m,:dd]
        except Exception:
            pass
        bestX=tmp
    bestX/=np.linalg.norm(bestX,axis=1,keepdims=True)
    bestV=_max_cos(bestX)
    E8=_build_e8()
    # symmetry configs: (m, freqs)
    configs=[(7,[1,2,3,0]),(7,[1,2,3,4]),(14,[1,3,5,7]),(21,[1,4,7,10]),(28,[1,3,5,7])]
    sym_cands=[]  # (val, Xfull, S, Gs)
    sym_end=DEADLINE-110.0
    if sym_end<t0+20: sym_end=t0+20
    for (m,freqs) in configs:
        ns=168//m
        if ns*m!=168: continue
        nrep=3 if m==7 else 2
        Gs=_build_group(m,freqs)
        for r in range(nrep):
            if time.time()>sym_end: break
            S0=rng.standard_normal(size=(ns,8)); S0/=np.linalg.norm(S0,axis=1,keepdims=True)
            S1,v1=_sym_quench(S0,Gs,steps=650,b0=25.0,b1=500.0,lr0=0.03,lr1=0.0025,rng=rng,noise=0.5,mu=0.78,t_lim=sym_end)
            # refine high beta
            S2,v2=_sym_quench(S1,Gs,steps=450,b0=80.0,b1=1100.0,lr0=0.010,lr1=0.0012,rng=rng,noise=0.08,mu=0.82,t_lim=sym_end)
            Xf=_expand_fast(S2,Gs)
            v=_max_cos(Xf)
            sym_cands.append((v,Xf.copy(),S2.copy(),Gs.copy()))
            # track global via released? keep best symmetric if surprisingly good
            if v<bestV-1e-12:
                bestV=v; bestX=Xf.copy()
    # E8 greedy subset candidate
    if time.time()<DEADLINE-100.0:
        try:
            Es=_greedy_e8_subset(E8,168)
            Es=Es+0.015*rng.standard_normal(size=Es.shape)
            Es/=np.linalg.norm(Es,axis=1,keepdims=True)
            Eq,vq=_free_anneal(Es,steps=500,b0=30.0,b1=600.0,l0=0.02,l1=0.002,rng=rng,noise=0.3,t_lim=DEADLINE-95.0)
            sym_cands.append((vq,Eq.copy(),None,None))
            if vq<bestV:
                bestV=vq; bestX=Eq.copy()
        except Exception:
            pass
    # sort candidates
    sym_cands.sort(key=lambda t:t[0])
    # staged release: take top K
    K=min(4,len(sym_cands))
    rel_end=DEADLINE-32.0
    for ci in range(K):
        if time.time()>rel_end: break
        v0,X0=sym_cands[ci][0],sym_cands[ci][1].copy()
        # only release promising ones (within 0.02 of best) plus best symmetric
        if v0>bestV+0.02 and ci>=2: continue
        # stage amplitudes
        curX=X0; curV=v0
        for amp,steps,b0,b1,lr0,lr1,noise in [
            (1e-3,650,50.0,850.0,0.009,0.0010,0.10),
            (1e-2,650,55.0,950.0,0.009,0.0010,0.08),
            (5e-2,800,35.0,700.0,0.012,0.0012,0.12),
        ]:
            if time.time()>rel_end: break
            Y=curX+amp*rng.standard_normal(size=curX.shape)
            Y/=np.linalg.norm(Y,axis=1,keepdims=True)
            Y2,v2=_free_anneal(Y,steps=steps,b0=b0,b1=b1,l0=lr0,l1=lr1,rng=rng,noise=noise,t_lim=rel_end)
            if v2<curV:
                curX,curV=Y2,v2
            # also try without extra kick: quench cur
            if time.time()>rel_end: break
        # fully free extra quench
        if time.time()<rel_end-2:
            Xf2,vf2=_free_anneal(curX,steps=700,b0=60.0,b1=1100.0,l0=0.008,l1=0.0009,rng=rng,noise=0.03,t_lim=rel_end)
            if vf2<curV:
                curX,curV=Xf2,vf2
        if curV<bestV-1e-12:
            bestV=curV; bestX=curX.copy()
    # final free polish of incumbent (small kicks to change basin but keep best)
    if time.time()<DEADLINE-34.0:
        # short diverse kick on bestX to try escape, but keep fallback
        Y=bestX+0.02*rng.standard_normal(size=bestX.shape)
        Y/=np.linalg.norm(Y,axis=1,keepdims=True)
        Y2,v2=_free_anneal(Y,steps=900,b0=45.0,b1=1000.0,l0=0.009,l1=0.0009,rng=rng,noise=0.05,t_lim=DEADLINE-32.0)
        if v2<bestV:
            bestV=v2; bestX=Y2
    rem=DEADLINE-time.time()-1.0
    if rem>5.0:
        bestX=_lp_polish(bestX,seconds=min(rem,30.0),deadline=DEADLINE-0.5)
        bestV=_max_cos(bestX)
    try:
        ref=np.load("best.npy").astype(float)
        ref/=np.linalg.norm(ref,axis=1,keepdims=True)
        rv=_max_cos(ref)
        if bestV>rv:
            bestX=ref; bestV=rv
    except Exception:
        pass
    bestX/=np.linalg.norm(bestX,axis=1,keepdims=True)
    return bestX
