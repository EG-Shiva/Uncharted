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
        Grad = (W @ x) / T
        dot = np.sum(Grad*x, axis=1, keepdims=True)
        Gt = Grad - dot*x
        m = b1*m + (1.0-b1)*Gt
        v = b2*v + (1.0-b2)*(Gt*Gt)
        mh = m/(1.0-b1**s); vh = v/(1.0-b2**s)
        upd = mh/(np.sqrt(vh)+eps)
        # remove normal comp for stability
        upd = upd - np.sum(upd*x,axis=1,keepdims=True)*x
        x = x - lr*upd
        nr = np.linalg.norm(x,axis=1,keepdims=True)
        nr[nr==0]=1.0
        x /= nr
    return x

def lp_polish(x, seconds, t_end=None):
    x = x/np.linalg.norm(x,axis=1,keepdims=True)
    n,d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n,1)
    while True:
        if time.time()-start>=seconds: break
        if t_end is not None and time.time()>=t_end: break
        if radius<1e-14: break
        c = (x@x.T)[iu]
        mcur = c.max()
        sel = c > mcur - max(20*radius,1e-9)
        I,J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p==0: break
        nv = n*d+1
        rows = np.repeat(np.arange(p),2*d+1)
        cols = np.concatenate([I[:,None]*d+np.arange(d),J[:,None]*d+np.arange(d),np.full((p,1),n*d)],1).ravel()
        vals = np.concatenate([x[J],x[I],-np.ones((p,1))],1).ravel()
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
        vv = max_cos(y)
        if vv<best:
            x,best,radius = y,vv,min(radius*1.5,0.05)
        else:
            radius*=0.5
    return x

def kmeans_labels(X,k,rng,iters=12):
    n=X.shape[0]
    centers=X[rng.choice(n,k,replace=False)].copy()
    labels=np.zeros(n,dtype=int)
    for _ in range(iters):
        d2=((X[:,None,:]-centers[None,:,:])**2).sum(axis=2)
        labels=d2.argmin(axis=1)
        for c in range(k):
            m=labels==c
            if np.any(m):
                centers[c]=X[m].mean(axis=0)
            else:
                centers[c]=X[int(rng.integers(n))]
    return labels

def cluster_rotate(x,rng):
    n,d=x.shape
    y=x.copy()
    use_pc1=rng.random()<0.5
    if use_pc1:
        try:
            _,_,Vt=np.linalg.svd(x,full_matrices=False)
            v1=Vt[0]
        except Exception:
            v1=rng.standard_normal(d); v1/=np.linalg.norm(v1)
        s=x@v1
        med=np.median(s)
        mask=s>med
        if mask.all() or (~mask).all():
            mask=rng.random(n)<0.5
        # pick one side
        if rng.random()<0.5:
            cl=np.where(mask)[0]
        else:
            cl=np.where(~mask)[0]
    else:
        k=int(rng.integers(4,7))
        try:
            lab=kmeans_labels(x,k,rng)
        except Exception:
            lab=rng.integers(0,k,size=n)
        pick=int(rng.integers(k))
        cl=np.where(lab==pick)[0]
        if len(cl)==0 or len(cl)==n:
            cl=np.where(rng.random(n)<0.3)[0]
        if len(cl)==0:
            cl=np.array([int(rng.integers(n))])
    a,b=rng.choice(d,2,replace=False)
    th=float(rng.uniform(0.03,0.2))*(1.0 if rng.random()<0.5 else -1.0)
    co,si=np.cos(th),np.sin(th)
    xa=y[cl,a].copy(); xb=y[cl,b].copy()
    y[cl,a]=co*xa-si*xb
    y[cl,b]=si*xa+co*xb
    nr=np.linalg.norm(y,axis=1,keepdims=True); nr[nr==0]=1.0
    y/=nr
    return y

def spectral_perturb(x,rng,amp,k_modes=30,Tsoft=0.002):
    n,d=x.shape
    N=n*d
    try:
        G=x@x.T
        np.fill_diagonal(G,-2.0)
        M=float(G.max())
        iu=np.triu_indices(n,1)
        c=(x@x.T)[iu]
        # active graph within 2e-3
        sel=c>M-2e-3
        I=iu[0][sel]; J=iu[1][sel]
        # cap to tightest 3000 for speed
        if len(I)>3000:
            order=np.argsort(-c[sel])[:3000]
            I,J=I[order],J[order]
        p=len(I)
        if p<10:
            raise ValueError("too few contacts")
        cij=np.array([(x[i]@x[j]) for i,j in zip(I,J)])
        w=np.exp((cij-M)/Tsoft)
        # scale to avoid huge numbers: divide by 1/T^2
        f1=w*Tsoft  # = (w/T)*T^2
        f2=w        # = (w/T^2)*T^2
        H=np.zeros((N,N),dtype=float)
        eye=np.eye(d)
        for t in range(p):
            i=int(I[t]); j=int(J[t])
            xi=x[i]; xj=x[j]
            oii=np.outer(xj,xj)*f2[t]
            ojj=np.outer(xi,xi)*f2[t]
            oij=eye*f1[t]+np.outer(xj,xi)*f2[t]
            bi=i*d; bj=j*d
            H[bi:bi+d,bi:bi+d]+=oii
            H[bj:bj+d,bj:bj+d]+=ojj
            H[bi:bi+d,bj:bj+d]+=oij
            H[bj:bj+d,bi:bi+d]+=oij.T
        # tangent basis U (N x Nt), Nt=n*(d-1)
        Nt=n*(d-1)
        U=np.zeros((N,Nt),dtype=float)
        for i in range(n):
            xi=x[i]
            k=int(np.argmax(np.abs(xi)))
            V=np.zeros((d,d-1),dtype=float)
            col=0
            for jj in range(d):
                if jj==k: continue
                V[jj,col]=1.0
                V[k,col]=-xi[jj]/(xi[k] if abs(xi[k])>1e-12 else 1e-12)
                col+=1
            Q,_=np.linalg.qr(V)
            U[i*d:(i+1)*d, i*(d-1):(i+1)*(d-1)]=Q
        HU=H@U
        Ht=U.T@HU
        # symmetrize
        Ht=(Ht+Ht.T)*0.5
        vals,vecs=np.linalg.eigh(Ht)
        k_eff=int(min(max(k_modes,20),40, Nt))
        # lowest k_eff modes
        low=vecs[:,:k_eff]
        coeff=rng.standard_normal(k_eff)
        nrm=np.linalg.norm(coeff)
        if nrm==0: coeff[0]=1.0; nrm=1.0
        coeff/=nrm
        combo=low@coeff  # Nt
        full=U@combo    # N
        y=x+ (amp*full).reshape(n,d)
        y/=np.linalg.norm(y,axis=1,keepdims=True)
        return y
    except Exception:
        # fallback jitter
        y=x+0.01*rng.standard_normal(x.shape)
        y/=np.linalg.norm(y,axis=1,keepdims=True)
        return y

def contact_mask(x,thr):
    G=x@x.T
    np.fill_diagonal(G,-2.0)
    iu=np.triu_indices(x.shape[0],1)
    return G[iu]>thr

def solve():
    t0=time.time()
    T_END=t0+172.0
    FINAL_RESERVE=16.0
    rng=np.random.default_rng(7)
    try:
        xb=np.load("best.npy").astype(float)
    except Exception:
        xb=None
    try:
        xr=np.load("record.npy").astype(float)
    except Exception:
        xr=None
    cands=[]
    for c in (xb,xr):
        if c is not None:
            try:
                a=np.array(c,dtype=float)
                a/=np.linalg.norm(a,axis=1,keepdims=True)
                if a.shape==(102,12):
                    cands.append((max_cos(a),a))
            except Exception:
                pass
    if len(cands)==0:
        a=rng.standard_normal((102,12)); a/=np.linalg.norm(a,axis=1,keepdims=True)
        best=float(max_cos(a)); x_best=a
    else:
        cands.sort(key=lambda t:t[0])
        best,x_best=cands[0][0],cands[0][1].copy()
    n,d=102,12
    cur=x_best.copy(); cur_val=best
    # contact reference threshold (absolute) for edit distance
    def edit_dist(a,b,thr):
        try:
            ma=contact_mask(a,thr); mb=contact_mask(b,thr)
            return int(np.count_nonzero(ma!=mb))
        except Exception:
            return 0
    it=0
    n_spec=0
    while time.time()<T_END-FINAL_RESERVE:
        it+=1
        time_left=T_END-FINAL_RESERVE-time.time()
        if time_left<2.0: break
        do_spec=(it%3==1)  # every 3rd is spectral
        # choose amplitude
        if do_spec:
            amp=float(rng.uniform(0.02,0.15))
            k_modes=int(rng.integers(20,41))
            try:
                x_try=spectral_perturb(cur,rng,amp,k_modes)
            except Exception:
                x_try=cur+0.01*rng.standard_normal((n,d))
                x_try/=np.linalg.norm(x_try,axis=1,keepdims=True)
            steps=int(rng.integers(400,801))
            T0=float(rng.uniform(0.015,0.03)); T1=float(rng.uniform(0.0025,0.005)); lr=float(rng.uniform(0.0018,0.003))
            n_spec+=1
        else:
            x_try=cluster_rotate(cur,rng)
            # small extra jitter to break exact rigidity
            x_try=x_try+ float(rng.uniform(0.0,0.004))*rng.standard_normal((n,d))
            x_try/=np.linalg.norm(x_try,axis=1,keepdims=True)
            steps=int(rng.integers(400,801))
            T0=float(rng.uniform(0.015,0.03)); T1=float(rng.uniform(0.0025,0.005)); lr=float(rng.uniform(0.0018,0.003))
        x_q=adam_quench(x_try,steps,T0,T1,lr)
        qval=max_cos(x_q)
        x_new=x_q; nval=qval
        # short LP only for promising
        if qval<0.269 and time.time()<T_END-FINAL_RESERVE-3.0:
            # only polish if close to incumbent or better than cur
            if qval<cur_val+0.001 or qval<best+0.001:
                x_lp=lp_polish(x_q,seconds=1.2,t_end=T_END-FINAL_RESERVE)
                lval=max_cos(x_lp)
                if lval<nval:
                    x_new,nval=x_lp,lval
        if nval<best-1e-12:
            best,x_best=nval,x_new.copy()
            cur,cur_val=x_new.copy(),nval
        elif nval<cur_val:
            cur,cur_val=x_new.copy(),nval
        else:
            # allow uphill within 0.002 if topology differs
            if nval<best+0.002:
                thr=best-0.002
                ed=edit_dist(cur,x_new,thr)
                # favour large edit: threshold ~80
                if ed>80:
                    cur,cur_val=x_new.copy(),nval
                elif nval<cur_val+0.0005 and rng.random()<0.2:
                    cur,cur_val=x_new.copy(),nval
            # occasional random restart from best to avoid drift
            if it%25==0 and rng.random()<0.5:
                # pull back toward best with small spectral kick
                cur=x_best.copy(); cur_val=best
    # final polish of best with remaining time
    remain=T_END-time.time()-1.0
    if remain>2.0:
        try:
            x_best=lp_polish(x_best,seconds=min(remain,14.0),t_end=T_END-0.5)
            best=max_cos(x_best)
        except Exception:
            pass
    x_best/=np.linalg.norm(x_best,axis=1,keepdims=True)
    return x_best
