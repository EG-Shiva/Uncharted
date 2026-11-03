import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _balance_std(X, tol=5e-4):
    try:
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        deg = (G > m - tol).sum(axis=1).astype(float)
        return float(deg.std()), float(deg.max()), float(m)
    except Exception:
        return 1e9, 1e9, 1e9

def _full_quench_best(X, steps, beta0, beta1, lr0, lr1):
    n, d = X.shape
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    m_ad = np.zeros_like(X)
    v_ad = np.zeros_like(X)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    best = X.copy()
    try:
        best_m = _max_cos(X)
    except Exception:
        best_m = 1e9
    for t in range(1, steps+1):
        frac = t / steps
        beta = np.exp(lb0 + (lb1-lb0)*frac)
        lr = np.exp(ll0 + (ll1-ll0)*frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = float(G.max())
        if gmax < best_m:
            best_m = gmax
            best = X.copy()
        W = np.exp(beta*(G-gmax))
        np.fill_diagonal(W, 0.0)
        Z = float(W.sum())
        if Z == 0 or not np.isfinite(Z):
            continue
        Grad = (W @ X) / Z
        dot = (Grad*X).sum(axis=1, keepdims=True)
        Gt = Grad - dot*X
        m_ad *= b1; m_ad += (1.0-b1)*Gt
        v_ad *= b2; v_ad += (1.0-b2)*(Gt*Gt)
        bc1 = 1.0 - b1**t; bc2 = 1.0 - b2**t
        step = (m_ad/bc1)/(np.sqrt(v_ad/bc2)+eps)
        X = X - lr*step
        nrm = np.sqrt((X*X).sum(axis=1, keepdims=True))
        nrm[nrm==0]=1.0
        X /= nrm
    try:
        vm = _max_cos(X)
        if vm < best_m:
            best = X.copy(); best_m = vm
    except Exception:
        pass
    return best

def _orbit_build(seeds, freqs, k):
    m = seeds.shape[0]
    t = np.arange(k)
    ang = 2.0*np.pi*np.outer(t, np.asarray(freqs, dtype=float))/float(k)
    cosA = np.cos(ang); sinA = np.sin(ang)
    se = seeds[:, 0::2]
    so = seeds[:, 1::2]
    Xe = se[:, None, :]*cosA[None, :, :] - so[:, None, :]*sinA[None, :, :]
    Xo = se[:, None, :]*sinA[None, :, :] + so[:, None, :]*cosA[None, :, :]
    X = np.empty((m, k, 8), dtype=float)
    X[:, :, 0::2] = Xe
    X[:, :, 1::2] = Xo
    return X.reshape(m*k, 8), cosA, sinA

def _quench_anti(Yinit, steps, beta0, beta1, lr0, lr1):
    Y = Yinit / np.linalg.norm(Yinit, axis=1, keepdims=True)
    m_ad = np.zeros_like(Y); v_ad = np.zeros_like(Y)
    b1=0.9; b2=0.999; eps=1e-8
    lb0=np.log(beta0); lb1=np.log(beta1); ll0=np.log(lr0); ll1=np.log(lr1)
    bestY = Y.copy()
    Xb = np.concatenate([Y, -Y], axis=0)
    best_m = _max_cos(Xb)
    n = 84
    for t in range(1, steps+1):
        frac=t/steps
        beta=np.exp(lb0+(lb1-lb0)*frac); lr=np.exp(ll0+(ll1-ll0)*frac)
        X = np.concatenate([Y, -Y], axis=0)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax=float(G.max())
        if gmax < best_m:
            best_m=gmax; bestY=Y.copy()
        W=np.exp(beta*(G-gmax))
        np.fill_diagonal(W,0.0)
        Z=float(W.sum())
        if Z==0 or not np.isfinite(Z):
            continue
        Grad=(W @ X)/Z
        Gseed = Grad[:n]-Grad[n:]
        dot=(Gseed*Y).sum(axis=1, keepdims=True)
        Gt=Gseed-dot*Y
        m_ad*=b1; m_ad+=(1.0-b1)*Gt
        v_ad*=b2; v_ad+=(1.0-b2)*(Gt*Gt)
        bc1=1.0-b1**t; bc2=1.0-b2**t
        step=(m_ad/bc1)/(np.sqrt(v_ad/bc2)+eps)
        Y=Y-lr*step
        nrm=np.sqrt((Y*Y).sum(axis=1, keepdims=True)); nrm[nrm==0]=1.0; Y/=nrm
    return bestY

def _quench_orbit(Sinit, freqs, k, steps, beta0, beta1, lr0, lr1):
    S = Sinit/np.linalg.norm(Sinit, axis=1, keepdims=True)
    m_ad=np.zeros_like(S); v_ad=np.zeros_like(S)
    b1=0.9; b2=0.999; eps=1e-8
    lb0=np.log(beta0); lb1=np.log(beta1); ll0=np.log(lr0); ll1=np.log(lr1)
    bestS=S.copy()
    X0,_c,_s=_orbit_build(S,freqs,k)
    best_m=_max_cos(X0)
    mnum=S.shape[0]
    for t in range(1, steps+1):
        frac=t/steps
        beta=np.exp(lb0+(lb1-lb0)*frac); lr=np.exp(ll0+(ll1-ll0)*frac)
        X,cosA,sinA=_orbit_build(S,freqs,k)
        G=X@X.T
        np.fill_diagonal(G,-10.0)
        gmax=float(G.max())
        if gmax<best_m:
            best_m=gmax; bestS=S.copy()
        W=np.exp(beta*(G-gmax))
        np.fill_diagonal(W,0.0)
        Z=float(W.sum())
        if Z==0 or not np.isfinite(Z):
            continue
        Grad=(W@X)/Z
        Gr=Grad.reshape(mnum,k,8)
        Ge=Gr[:,:,0::2]; Go=Gr[:,:,1::2]
        cA=cosA[None,:,:]; sA=sinA[None,:,:]
        dSe=(Ge*cA+Go*sA).sum(axis=1)
        dSo=(-Ge*sA+Go*cA).sum(axis=1)
        Gseed=np.empty_like(S)
        Gseed[:,0::2]=dSe; Gseed[:,1::2]=dSo
        dot=(Gseed*S).sum(axis=1,keepdims=True)
        Gt=Gseed-dot*S
        m_ad*=b1; m_ad+=(1.0-b1)*Gt
        v_ad*=b2; v_ad+=(1.0-b2)*(Gt*Gt)
        bc1=1.0-b1**t; bc2=1.0-b2**t
        step=(m_ad/bc1)/(np.sqrt(v_ad/bc2)+eps)
        S=S-lr*step
        nrm=np.sqrt((S*S).sum(axis=1,keepdims=True)); nrm[nrm==0]=1.0; S/=nrm
    return bestS

def _lp_polish(x, seconds, r0=1.2e-3):
    x=x/np.linalg.norm(x,axis=1,keepdims=True)
    try:
        best=_max_cos(x)
    except Exception:
        return x
    n,d=x.shape
    radius=r0
    start=time.time()
    iu=np.triu_indices(n,1)
    while time.time()-start<seconds and radius>1e-14:
        try:
            c=(x@x.T)[iu]
        except Exception:
            break
        m=float(c.max())
        sel=c>m-max(20*radius,1e-9)
        I,J=iu[0][sel],iu[1][sel]
        p=len(I)
        if p==0:
            radius*=0.5; continue
        if p>2800:
            idx=np.argsort(c[sel])[-2800:]
            I,J=I[idx],J[idx]; cc=c[sel][idx]; p=2800
        else:
            cc=c[sel]
        nv=n*d+1
        rows=np.repeat(np.arange(p),2*d+1)
        cols=np.concatenate([I[:,None]*d+np.arange(d),J[:,None]*d+np.arange(d),np.full((p,1),n*d)],1).ravel()
        vals=np.concatenate([x[J],x[I],-np.ones((p,1))],1).ravel()
        try:
            A_ub=coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        except Exception:
            radius*=0.5; continue
        obj=np.zeros(nv); obj[-1]=1.0
        try:
            res=linprog(obj,A_ub=A_ub,b_ub=-cc,A_eq=A_eq,b_eq=np.zeros(n),bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        y=x+res.x[:-1].reshape(n,d)
        y/=np.linalg.norm(y,axis=1,keepdims=True)
        try:
            vv=_max_cos(y)
        except Exception:
            radius*=0.5; continue
        if vv<best:
            x,best=y,vv
            radius=min(radius*1.5,0.05)
        else:
            radius*=0.5
    return x

def _build_E8():
    vecs=[]
    for i in range(8):
        for j in range(i+1,8):
            for si in (1.0,-1.0):
                for sj in (1.0,-1.0):
                    v=np.zeros(8); v[i]=si; v[j]=sj; vecs.append(v)
    for mask in range(256):
        cnt=0; s=np.ones(8)
        for kk in range(8):
            if (mask>>kk)&1:
                s[kk]=-1.0; cnt+=1
        if cnt%2==0:
            vecs.append(0.5*s)
    E=np.array(vecs,float)
    E/=np.linalg.norm(E,axis=1,keepdims=True)
    return E

def solve():
    t0=time.time()
    BUDGET=175.0
    SYM_DEADLINE=t0+95.0
    REL_DEADLINE=t0+145.0
    rng=np.random.RandomState(12345)
    best=None; best_val=1e9
    for fn in ("best.npy","record.npy"):
        try:
            a=np.load(fn).astype(np.float64)
            if a.shape!=(168,8):
                continue
            a/=np.linalg.norm(a,axis=1,keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            va=_max_cos(a)
            if va<best_val:
                best_val=va; best=a.copy()
        except Exception:
            pass
    if best is None:
        best=rng.randn(168,8); best/=np.linalg.norm(best,axis=1,keepdims=True)
        best_val=_max_cos(best)
    fallback=best.copy(); fallback_val=best_val
    try:
        E8=_build_E8()
    except Exception:
        E8=None
    cands=[]
    def push(X, tag):
        try:
            X=X/np.linalg.norm(X,axis=1,keepdims=True)
            if X.shape!=(168,8) or not np.all(np.isfinite(X)):
                return
            mv=_max_cos(X)
            std,mx,_=_balance_std(X)
            cands.append((mv,std,mx,X.copy(),tag))
            global_best_update(X,mv)
        except Exception:
            pass
    def global_best_update(X,mv):
        global best, best_val
        # use closure via list
        pass
    # we manage best manually
    def try_update(X):
        global_best=None
        nonlocal_best=[0]
        return None
    # redefine push with best update
    cands=[]
    def record(X, tag):
        nonlocal best, best_val
        try:
            X=np.asarray(X,float)
            if X.shape!=(168,8):
                return
            X/=np.linalg.norm(X,axis=1,keepdims=True)
            if not np.all(np.isfinite(X)):
                return
            mv=_max_cos(X)
            std,mx,_=_balance_std(X)
            cands.append((mv,std,mx,X.copy(),tag))
            if mv+1e-12<best_val:
                best_val=mv; best=X.copy()
        except Exception:
            pass
    # helper to get seeds init
    def init_seeds(m, kind):
        if kind=='e8' and E8 is not None:
            try:
                idx=rng.choice(E8.shape[0], m, replace=False)
                return E8[idx].copy()+rng.randn(m,8)*0.05
            except Exception:
                pass
        if kind=='bestproj':
            try:
                idx=rng.choice(168, m, replace=False)
                return best[idx].copy()+rng.randn(m,8)*0.05
            except Exception:
                pass
        return rng.randn(m,8)
    # (i) antipodal 84 pairs
    try:
        for trial in range(4):
            if time.time()>SYM_DEADLINE:
                break
            kind='e8' if trial%2==0 else ('bestproj' if trial%3==0 else 'rand')
            Y0=init_seeds(84, kind if trial!=3 else 'rand')
            Y0/=np.linalg.norm(Y0,axis=1,keepdims=True)
            steps=550
            b0=25+rng.rand()*30; b1=300+rng.rand()*400
            lr0=0.004+rng.rand()*0.004; lr1=lr0*0.1
            try:
                Yo=_quench_anti(Y0, steps, b0, b1, lr0, lr1)
                X=np.concatenate([Yo,-Yo],axis=0)
                X/=np.linalg.norm(X,axis=1,keepdims=True)
                record(X, f'anti{trial}')
            except Exception:
                continue
    except Exception:
        pass
    # (ii) C7 C14 C21
    orbit_cfgs=[
        ('C7',24,7,[1,2,3,4]),
        ('C7b',24,7,[1,2,3,5]),
        ('C14',12,14,[1,3,5,7]),
        ('C14b',12,14,[1,3,5,9]),
        ('C21',8,21,[1,4,7,10]),
        ('C21b',8,21,[1,2,5,11]),
    ]
    try:
        for ci,(tag,m,k,freqs) in enumerate(orbit_cfgs):
            for rep in range(2):
                if time.time()>SYM_DEADLINE:
                    break
                kind='e8' if rep==0 else 'rand'
                # vary freqs slightly on second rep for some
                fr=list(freqs)
                if rep==1 and k>7:
                    try:
                        fr[3]=(fr[3]+2)%k
                        if fr[3]==0: fr[3]=1
                    except Exception:
                        pass
                S0=init_seeds(m, kind)
                S0/=np.linalg.norm(S0,axis=1,keepdims=True)
                steps=550
                b0=25+rng.rand()*30; b1=300+rng.rand()*400
                lr0=0.004+rng.rand()*0.004; lr1=lr0*0.12
                try:
                    So=_quench_orbit(S0, fr, k, steps, b0, b1, lr0, lr1)
                    X,_,__=_orbit_build(So/np.linalg.norm(So,axis=1,keepdims=True), fr, k)
                    X/=np.linalg.norm(X,axis=1,keepdims=True)
                    record(X, f'{tag}r{rep}')
                except Exception:
                    continue
    except Exception:
        pass
    # (iii) C24 C28
    big_cfgs=[
        ('C24',7,24,[1,5,7,11]),
        ('C24b',7,24,[1,5,7,13]),
        ('C28',6,28,[1,3,5,11]),
        ('C28b',6,28,[1,3,9,13]),
    ]
    try:
        for tag,m,k,freqs in big_cfgs:
            if time.time()>SYM_DEADLINE:
                break
            for rep in range(1):
                if time.time()>SYM_DEADLINE:
                    break
                S0=init_seeds(m, 'rand' if rep==0 else 'e8')
                S0/=np.linalg.norm(S0,axis=1,keepdims=True)
                steps=500
                b0=25+rng.rand()*30; b1=300+rng.rand()*400
                lr0=0.004+rng.rand()*0.004; lr1=lr0*0.12
                try:
                    So=_quench_orbit(S0, freqs, k, steps, b0, b1, lr0, lr1)
                    X,_,__=_orbit_build(So/np.linalg.norm(So,axis=1,keepdims=True), freqs, k)
                    X/=np.linalg.norm(X,axis=1,keepdims=True)
                    record(X, tag)
                except Exception:
                    continue
    except Exception:
        pass
    # rank: prefer low max but also balanced
    selected=[]
    try:
        if len(cands)>0:
            # sort by max
            by_max=sorted(cands, key=lambda z: z[0])
            # sort by balance among near-best
            thr=by_max[0][0]+0.003
            near=[z for z in cands if z[0]<thr]
            by_bal=sorted(near, key=lambda z: z[1])
            seen=set()
            for z in by_max[:3]:
                if id(z) not in seen:
                    selected.append(z); seen.add(id(z))
            for z in by_bal[:2]:
                if id(z) not in seen:
                    selected.append(z); seen.add(id(z))
            # cap to 4
            selected=selected[:4]
        else:
            selected=[]
    except Exception:
        selected=[]
    # staged release: small noise + full quench, then slightly larger
    try:
        for si,z in enumerate(list(selected)):
            if time.time()>REL_DEADLINE:
                break
            mv,std,mx,X0,tag=z
            for stage,sig in enumerate([0.008, 0.02]):
                if time.time()>REL_DEADLINE:
                    break
                if time.time()+3.0>t0+BUDGET:
                    break
                try:
                    Xp=X0+rng.randn(168,8)*sig
                    Xp/=np.linalg.norm(Xp,axis=1,keepdims=True)
                    steps=500 if stage==0 else 600
                    b0=30+rng.rand()*20; b1=350+rng.rand()*400
                    lr0=0.004+rng.rand()*0.003; lr1=lr0*0.1
                    Xq=_full_quench_best(Xp, steps, b0, b1, lr0, lr1)
                    record(Xq, f'{tag}_rel{stage}')
                except Exception:
                    continue
    except Exception:
        pass
    # short full-dimensional hops from current best to ensure progress if symmetric failed
    try:
        hops=0
        while time.time()<REL_DEADLINE and hops<6:
            if time.time()+4.0>t0+BUDGET:
                break
            hops+=1
            try:
                X=best.copy()
                sig=[0.01,0.025,0.05][rng.randint(3)]
                X=X+rng.randn(168,8)*sig
                X/=np.linalg.norm(X,axis=1,keepdims=True)
                Xq=_full_quench_best(X, 700, 30+rng.rand()*30, 350+rng.rand()*400, 0.004+rng.rand()*0.004, 0.0004)
                mv=_max_cos(Xq)
                if mv+1e-12<best_val:
                    best_val=mv; best=Xq.copy()
                    cands.append((mv,0,0,Xq.copy(),'hop'))
            except Exception:
                continue
    except Exception:
        pass
    # final LP polish reserving last ~30s
    try:
        remaining=(t0+BUDGET)-time.time()
        if remaining>7.0:
            secs=min(remaining-2.0, 30.0)
            if secs>3.0:
                try:
                    polished=_lp_polish(best, secs, r0=1.2e-3)
                    vp=_max_cos(polished)
                    if vp<best_val and np.all(np.isfinite(polished)):
                        best=polished; best_val=vp
                except Exception:
                    pass
    except Exception:
        pass
    try:
        best=np.asarray(best,float)
        best/=np.linalg.norm(best,axis=1,keepdims=True)
        bad=~np.all(np.isfinite(best),axis=1)
        if np.any(bad):
            rr=rng.randn(int(bad.sum()),8); rr/=np.linalg.norm(rr,axis=1,keepdims=True)
            best[bad]=rr
        best/=np.linalg.norm(best,axis=1,keepdims=True)
        if best.shape!=(168,8) or not np.all(np.isfinite(best)):
            best=fallback.copy()
    except Exception:
        try:
            best=fallback.copy()
        except Exception:
            rr=np.random.RandomState(0).randn(168,8); rr/=np.linalg.norm(rr,axis=1,keepdims=True)
            best=rr
    return best
