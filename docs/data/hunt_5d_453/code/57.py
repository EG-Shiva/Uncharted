import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _build_orbit_X(seed_params, m, a, q, n_free_fixed_poles, free_vecs):
    # seed_params: (q,3) theta,phi,delta
    # free_vecs: (n_free_free,5) unit or None
    n = q*m
    X = np.empty((n + n_free_fixed_poles + (0 if free_vecs is None else free_vecs.shape[0]), 5), dtype=np.float64)
    ks = np.arange(m)
    ang1 = 2*np.pi*ks/m
    ang2 = 2*np.pi*a*ks/m
    c1 = np.cos(ang1); s1 = np.sin(ang1)
    c2 = np.cos(ang2); s2 = np.sin(ang2)
    pos = 0
    for o in range(q):
        th, ph, de = seed_params[o]
        z = np.cos(th); R = np.sin(th)
        r1 = R*np.cos(ph); r2 = R*np.sin(ph)
        cd = np.cos(de); sd = np.sin(de)
        # rotate r2 vector by de+ang2
        # r2*cos(de+ang2)=r2*(cd*c2-sd*s2)
        c2d = cd*c2 - sd*s2
        s2d = sd*c2 + cd*s2
        Xo = np.empty((m,5))
        Xo[:,0]=r1*c1; Xo[:,1]=r1*s1
        Xo[:,2]=r2*c2d; Xo[:,3]=r2*s2d
        Xo[:,4]=z
        X[pos:pos+m]=Xo
        pos+=m
    if n_free_fixed_poles>=1:
        X[pos]=np.array([0,0,0,0,1.0]); pos+=1
    if n_free_fixed_poles>=2:
        X[pos]=np.array([0,0,0,0,-1.0]); pos+=1
    if free_vecs is not None and len(free_vecs)>0:
        X[pos:pos+len(free_vecs)]=free_vecs
        pos+=len(free_vecs)
    # normalize (should be unit)
    nrm=np.linalg.norm(X,axis=1,keepdims=True); nrm[nrm==0]=1
    X/=nrm
    return X

def _softmax_polish(X, best_val, best_X, deadline, steps=600, tau0=0.008, tau1=0.0015, lr0=0.03, lr1=0.003, mu=0.7, rng=None):
    n,d=X.shape
    v=np.zeros_like(X)
    X=X.copy()
    import math
    ltau=math.log(tau1/tau0) if tau0>0 else 0
    llr=math.log(lr1/lr0) if lr0>0 else 0
    for it in range(steps):
        if it%60==0 and time.time()>deadline:
            break
        frac=it/max(steps-1,1)
        tau=tau0*math.exp(ltau*frac)
        lr=lr0*math.exp(llr*frac)
        G=X@X.T
        np.fill_diagonal(G,-2.0)
        m=float(np.max(G))
        if m<best_val:
            best_val=m; best_X=X.copy()
        # softmax weights
        # clip for overflow
        E=np.exp((G-m)/tau)
        # zero diag approx (exp((-2-m)/tau) ~0)
        Grad=E@X
        dot=np.sum(Grad*X,axis=1,keepdims=True)
        Gt=Grad-dot*X
        # normalize grad scale
        gn=float(np.linalg.norm(Gt))/np.sqrt(n)+1e-30
        # scale to avoid blowup
        v=mu*v+(1-mu)* (Gt/gn)
        X=X-lr*0.5*v
        nrm=np.linalg.norm(X,axis=1,keepdims=True); nrm[nrm==0]=1
        X/=nrm
    # final eval
    m=_max_cos(X)
    if m<best_val:
        best_val=m; best_X=X.copy()
    return X,best_val,best_X

def _greedy_shave(X,best_val,best_X,deadline,steps=1500,eps0=2e-3,rng=None):
    n,d=X.shape
    X=X.copy()
    eps=float(eps0)
    for it in range(steps):
        if it%200==0 and time.time()>deadline:
            break
        G=X@X.T
        np.fill_diagonal(G,-2.0)
        m=float(np.max(G))
        if m<best_val:
            best_val=m; best_X=X.copy()
        flat=int(np.argmax(G))
        i=flat//n; j=flat%n
        cij=float(X[i]@X[j])
        cij=max(-1.0,min(1.0,cij))
        ti=X[j]-cij*X[i]; ni=float(np.linalg.norm(ti))
        if ni<1e-12 or not np.isfinite(ni):
            if rng is None: continue
            ti=rng.standard_normal(d); ti-=float(ti@X[i])*X[i]; ni=float(np.linalg.norm(ti))+1e-30
        ti/=ni
        tj=X[i]-cij*X[j]; nj=float(np.linalg.norm(tj))
        if nj<1e-12 or not np.isfinite(nj):
            if rng is None: continue
            tj=rng.standard_normal(d); tj-=float(tj@X[j])*X[j]; nj=float(np.linalg.norm(tj))+1e-30
        tj/=nj
        improved=False
        for s in (eps,eps*0.4):
            inv=1.0/np.sqrt(1.0+s*s)
            vi=(X[i]-s*ti)*inv; vj=(X[j]-s*tj)*inv
            xi_old=X[i].copy(); xj_old=X[j].copy()
            X[i]=vi; X[j]=vj
            G2=X@X.T; np.fill_diagonal(G2,-2.0)
            m2=float(np.max(G2))
            if m2<m-1e-13:
                m=m2
                if m2<best_val:
                    best_val=m2; best_X=X.copy()
                eps=min(eps*1.05,3e-3)
                improved=True
                break
            else:
                X[i]=xi_old; X[j]=xj_old
        if not improved:
            eps*=0.94
            if eps<2e-7:
                eps=2e-7
                # tiny jitter
                if rng is not None and (it%50==0):
                    k=i if (it%2==0) else j
                    rnd=rng.standard_normal(d); rnd-=float(rnd@X[k])*X[k]
                    rn=float(np.linalg.norm(rnd))
                    if rn>1e-12:
                        X[k]=(X[k]+1e-7*rnd/rn); X[k]/=np.linalg.norm(X[k])
    return X,best_val,best_X

def solve():
    t0=time.time()
    DEAD=t0+170.0
    rng=np.random.default_rng(0)
    np.random.seed(0)
    # load incumbents
    cands=[]
    for fn in ("best.npy","record.npy"):
        try:
            a=np.load(fn)
            cands.append(np.asarray(a,dtype=float))
        except Exception:
            pass
    best_val=float("inf"); best_X=None
    for c in cands:
        try:
            if c.shape!=(453,5): continue
            c=c/np.linalg.norm(c,axis=1,keepdims=True)
            if not np.all(np.isfinite(c)): continue
            v=_max_cos(c)
            if v<best_val:
                best_val=v; best_X=c.copy()
        except Exception:
            continue
    if best_X is None:
        best_X=rng.standard_normal((453,5)); best_X/=np.linalg.norm(best_X,axis=1,keepdims=True)
        best_val=_max_cos(best_X)
    # keep incumbent copy
    incumbent_X=best_X.copy(); incumbent_V=best_val
    try:
        from scipy.optimize import differential_evolution
        has_de=True
    except Exception:
        has_de=False
    # families: (m,a,q,n_pole_fixed,n_free)
    families=[
        (151,2,3,0,0),
        (151,3,3,0,0),
        (113,2,4,1,0),
        (113,5,4,1,0),
        (90,7,5,2,1),
        (75,2,6,2,1),
    ]
    sym_candidates=[]  # (val,X,fam)
    # staged symmetric optimisation
    for (m,a,q,npol,nfree) in families:
        if time.time()>t0+85.0: break
        if has_de:
            # dims: q*3 + nfree*5
            dim=q*3+nfree*5
            # bounds
            bounds=[]
            for o in range(q):
                bounds.append((0.05,np.pi-0.05))  # theta avoid exact poles (dup)
                bounds.append((0.0,np.pi/2))      # phi
                bounds.append((0.0,2*np.pi))      # delta
            for k in range(nfree*5):
                bounds.append((-1.0,1.0))
            def obj(vec):
                try:
                    seeds=np.reshape(vec[:q*3],(q,3))
                    if nfree>0:
                        fw=np.reshape(vec[q*3:],(nfree,5))
                        nrm=np.linalg.norm(fw,axis=1,keepdims=True)
                        # avoid zero
                        bad=np.any(~np.isfinite(fw)) or np.any(nrm.squeeze()<1e-8)
                        if bad: return 2.0
                        fw=fw/nrm
                    else:
                        fw=None
                    X=_build_orbit_X(seeds,m,a,q,npol,fw)
                    # duplicate check via max (if dup ->1.0)
                    return _max_cos(X)
                except Exception:
                    return 2.0
            # budget depends on dim
            try:
                maxiter=18 if dim<=12 else 12
                pop=8 if dim<=12 else 7
                res=differential_evolution(obj,bounds,maxiter=maxiter,popsize=pop,seed=0,workers=1,updating='deferred',tol=1e-9,disp=False)
                xbest=res.x
            except Exception:
                xbest=None
            if xbest is None: continue
            if time.time()>t0+95.0:
                # still record
                pass
            seeds=np.reshape(xbest[:q*3],(q,3))
            if nfree>0:
                fw=np.reshape(xbest[q*3:],(nfree,5)); fw=fw/np.linalg.norm(fw,axis=1,keepdims=True)
            else:
                fw=None
            Xsym=_build_orbit_X(seeds,m,a,q,npol,fw)
            vsym=_max_cos(Xsym)
            # orbit-restricted hill climb polish to <1e-10 style
            step=np.array([0.02]* (q*3) + [0.05]*(nfree*5))
            cur_vec=xbest.copy(); cur_val=vsym
            # simple adaptive hill climb
            hc_iters=600 if dim<=12 else 300
            for hi in range(hc_iters):
                if time.time()>t0+100.0: break
                prop=cur_vec+rng.standard_normal(dim)*step
                # wrap/clip bounds
                # clip theta/phi/delta properly
                for o in range(q):
                    b0=prop[o*3]
                    if b0<0.02: b0=0.02
                    if b0>np.pi-0.02: b0=np.pi-0.02
                    prop[o*3]=b0
                    b1=prop[o*3+1]
                    if b1<0: b1=0
                    if b1>np.pi/2: b1=np.pi/2
                    prop[o*3+1]=b1
                    b2=prop[o*3+2]%(2*np.pi)
                    prop[o*3+2]=b2
                for k in range(nfree*5):
                    idx=q*3+k
                    if prop[idx]<-1: prop[idx]=-1
                    if prop[idx]>1: prop[idx]=1
                v=obj(prop)
                if v<cur_val-1e-13:
                    cur_val=v; cur_vec=prop.copy()
                    step*=1.03
                    # update Xsym
                    seeds2=np.reshape(cur_vec[:q*3],(q,3))
                    if nfree>0:
                        fw2=np.reshape(cur_vec[q*3:],(nfree,5)); fw2=fw2/np.linalg.norm(fw2,axis=1,keepdims=True)
                    else: fw2=None
                    Xsym=_build_orbit_X(seeds2,m,a,q,npol,fw2)
                else:
                    step*=0.995
                if np.max(step)<1e-7:
                    break
            vsym=cur_val
            # retain symmetric code itself as candidate
            if vsym<best_val:
                best_val=vsym; best_X=Xsym.copy()
            sym_candidates.append((vsym,Xsym.copy(),(m,a,q,npol,nfree),cur_vec.copy()))
        # end de
    # sort sym candidates best first, keep top 3 for breaking
    sym_candidates.sort(key=lambda t:t[0])
    # staged symmetry breaking by Fourier modes
    # helper to apply mode-f perturbation
    def fourier_break(Xsym,fam,mode,amp,seed_off):
        m,a,q,npol,nfree=fam
        r2=np.random.default_rng(1000+seed_off)
        X=Xsym.copy()
        # map orbit structure: first q*m rows are orbits
        for o in range(q):
            U=r2.standard_normal(5); V=r2.standard_normal(5)
            for k in range(m):
                idx=o*m+k
                ang=2*np.pi*mode*k/m
                d=np.cos(ang)*U+np.sin(ang)*V
                x=X[idx]
                d=d-float(d@x)*x
                dn=float(np.linalg.norm(d))
                if dn<1e-12 or not np.isfinite(dn):
                    continue
                d/=dn
                # scale: amp * (0.5+0.5 random?) keep deterministic amp
                xn=x+amp*d
                xn/=np.linalg.norm(xn)
                X[idx]=xn
        # perturb tail points with plain noise
        for idx in range(q*m,X.shape[0]):
            x=X[idx]
            d=r2.standard_normal(5); d-=float(d@x)*x
            dn=float(np.linalg.norm(d))
            if dn<1e-12: continue
            d/=dn
            xn=x+amp*0.7*d; xn/=np.linalg.norm(xn)
            X[idx]=xn
        return X
    # try breaking for top candidates
    nbreak=min(3,len(sym_candidates))
    for bi in range(nbreak):
        if time.time()>t0+135.0: break
        vsym,Xsym,fam,vec=sym_candidates[bi]
        # stages: (mode, amp)
        stages=[(1,0.005),(1,0.02),(2,0.01),(0,0.01)]
        for si,(mode,amp) in enumerate(stages):
            if time.time()>t0+140.0: break
            if mode==0:
                # full random tangent perturbation
                r3=np.random.default_rng(2000+bi*10+si)
                Xp=Xsym.copy()
                for idx in range(Xp.shape[0]):
                    x=Xp[idx]
                    d=r3.standard_normal(5); d-=float(d@x)*x
                    dn=float(np.linalg.norm(d))
                    if dn<1e-12: continue
                    Xp[idx]=(x+amp*d/dn)/np.linalg.norm(x+amp*d/dn)
            else:
                Xp=fourier_break(Xsym,fam,mode,amp,seed_off=bi*10+si)
            # re-polish with LP-like softmax after each stage
            dl=min(DEAD-4.0,time.time()+7.0)
            Xp,_,_=_softmax_polish(Xp,best_val,best_X,dl,steps=350,tau0=0.006,tau1=0.0012,lr0=0.025,lr1=0.002,mu=0.6,rng=rng)
            # update global best inside polish via closure? we passed best_val/best_X but function returns copies of best? Actually _softmax_polish tracks internal best but needs to propagate.
            # Recompute: it returns (Xp, best_val_new?, best_X_new?) we ignored; instead re-evaluate
            # Let's instead run polish starting from Xp tracking global best via explicit vars:
            # (we already passed best_val/best_X, but python passes floats by value; need to capture)
            # Workaround: after polish, evaluate and also the polish updated its own copy; we redo tracking here:
            # To properly track, call again with explicit handling:
            # Actually _softmax_polish returns X and updated best; we discarded. Fix: capture.
            # We called with best_val,best_X but didn't capture. Re-capture by checking:
            # Simplest: evaluate Xp and update.
            v=_max_cos(Xp)
            if v<best_val:
                best_val=v; best_X=Xp.copy()
            # also greedy shave brief
            dl2=min(DEAD-4.0,time.time()+3.0)
            Xp2, nv, nb=_softmax_polish(Xp,best_val,best_X,dl2,steps=150,tau0=0.0015,tau1=0.0006,lr0=0.006,lr1=0.001,mu=0.5,rng=rng)
            # _softmax_polish returns updated best; adopt
            best_val=nv; best_X=nb.copy()
            # if this stage beat symmetric, promote Xsym for next stage (staged)
            if _max_cos(Xp2)<vsym-1e-9:
                Xsym=Xp2.copy(); vsym=_max_cos(Xsym)
    # final polish of global best + incumbent to ensure record
    # polish incumbent and best
    for start in [best_X.copy(), incumbent_X.copy()]:
        if time.time()>DEAD-12.0: break
        dl=min(DEAD-3.0,time.time()+5.0)
        _,nv,nb=_softmax_polish(start,best_val,best_X,dl,steps=400,tau0=0.005,tau1=0.0008,lr0=0.02,lr1=0.0015,mu=0.65,rng=rng)
        best_val=nv; best_X=nb.copy()
        if time.time()>DEAD-6.0: break
        dl2=min(DEAD-2.0,time.time()+4.0)
        _,nv2,nb2=_greedy_shave(best_X.copy(),best_val,best_X,dl2,steps=1200,eps0=1e-3,rng=rng)
        best_val=nv2; best_X=nb2.copy()
    # strict normalize, shrink margin, guard
    best_X=np.asarray(best_X,dtype=float)
    nrm=np.linalg.norm(best_X,axis=1,keepdims=True); nrm[nrm==0]=1
    best_X/=nrm
    best_X=np.clip(best_X,-1.0,1.0)
    # re-normalize after clip
    nrm=np.linalg.norm(best_X,axis=1,keepdims=True); nrm[nrm==0]=1
    best_X/=nrm
    if not np.all(np.isfinite(best_X)):
        best_X=incumbent_X.copy()
    return best_X
