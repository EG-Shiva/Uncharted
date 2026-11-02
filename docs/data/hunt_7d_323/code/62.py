import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _build_cross():
    C = np.zeros((14,7))
    for i in range(7):
        C[2*i,i]=1.0; C[2*i+1,i]=-1.0
    return C

def _build_d7():
    rows=[]
    for i in range(7):
        for j in range(i+1,7):
            for s1 in (1.0,-1.0):
                for s2 in (1.0,-1.0):
                    v=np.zeros(7); v[i]=s1; v[j]=s2; v/=np.sqrt(2.0)
                    rows.append(v)
    return np.array(rows)  # 84x7

def _orbits_from_bases(bases):
    # bases Bx7 -> (B*7)x7 cyclic rolls
    B = bases.shape[0]
    out = np.empty((B*7,7))
    for k in range(7):
        out[k*B:(k+1)*B] = np.roll(bases, k, axis=1)
    # reorder to group by base (doesn't matter)
    return out / np.linalg.norm(out, axis=1, keepdims=True)

def _random_bases(rng, B, a, b):
    M = np.zeros((B,7))
    for r in range(B):
        k = int(rng.integers(2,5))  # 2..4 nonzeros
        pos = rng.choice(7, size=k, replace=False)
        for p in pos:
            t = int(rng.integers(0,3))
            s = 1.0 if rng.random()<0.5 else -1.0
            if t==0: M[r,p]=s*1.0
            elif t==1: M[r,p]=s*a
            else: M[r,p]=s*b
    n = np.linalg.norm(M,axis=1,keepdims=True); n[n==0]=1.0
    return M/n

def _gd_polish(x, steps, tau0, tau1, lr0, lr1, mu, deadline, seed=0):
    n,d = x.shape
    iuI, iuJ = np.triu_indices(n,1)
    vel = np.zeros_like(x)
    best = _max_cos(x); best_x = x.copy()
    for s in range(steps):
        if time.time()>=deadline: break
        frac = s/max(1,steps-1)
        tau = tau0+(tau1-tau0)*frac
        lr = lr0+(lr1-lr0)*frac
        G = x @ x.T
        c = G[iuI,iuJ]
        m = float(c.max())
        e = np.exp((c-m)/tau)
        S = float(e.sum())
        if not np.isfinite(S) or S==0: break
        w = e/S
        Wx = np.zeros_like(x)
        np.add.at(Wx, iuI, w[:,None]*x[iuJ])
        np.add.at(Wx, iuJ, w[:,None]*x[iuI])
        dot = np.sum(Wx*x,axis=1,keepdims=True)
        Gt = Wx - dot*x
        vel = mu*vel + Gt
        x = x - lr*vel
        xn = np.linalg.norm(x,axis=1,keepdims=True); xn[xn==0]=1.0
        x = x/xn
        if (s%80==79) or s==steps-1:
            v=_max_cos(x)
            if v<best:
                best=v; best_x=x.copy()
    return best_x, best

def _lp_polish(x, deadline, init_radius=4e-3, maxc=2500):
    x = x/np.linalg.norm(x,axis=1,keepdims=True)
    n,d = x.shape
    best=_max_cos(x)
    radius=float(init_radius)
    iuI,iuJ=np.triu_indices(n,1)
    while time.time()<deadline and radius>1e-14:
        G=x@x.T
        c=G[iuI,iuJ]
        m=float(c.max())
        thresh=m-max(20.0*radius,1e-9)
        sel=c>thresh
        cnt=int(sel.sum())
        if cnt==0: break
        if cnt>maxc:
            idx=np.where(sel)[0]; cidx=c[idx]
            part=np.argpartition(cidx,-maxc)[-maxc:]
            sel=np.zeros_like(sel,dtype=bool); sel[idx[part]]=True; cnt=maxc
        I=iuI[sel]; J=iuJ[sel]; csel=c[sel]
        p=cnt; nv=n*d+1
        rows=np.repeat(np.arange(p),2*d+1)
        cols=np.empty(p*(2*d+1),dtype=np.int64)
        vals=np.empty(p*(2*d+1),dtype=np.float64)
        Cm=cols.reshape(p,2*d+1); Vm=vals.reshape(p,2*d+1)
        ar=np.arange(d)
        Cm[:,:d]=I[:,None]*d+ar; Cm[:,d:2*d]=J[:,None]*d+ar; Cm[:,2*d]=n*d
        Vm[:,:d]=x[J]; Vm[:,d:2*d]=x[I]; Vm[:,2*d]=-1.0
        A_ub=coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
        A_eq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        obj=np.zeros(nv); obj[-1]=1.0
        try:
            res=linprog(obj,A_ub=A_ub,b_ub=-csel,A_eq=A_eq,b_eq=np.zeros(n),
                        bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        y=x+res.x[:-1].reshape(n,d)
        yn=np.linalg.norm(y,axis=1,keepdims=True); yn[yn==0]=1.0; y=y/yn
        if not np.all(np.isfinite(y)):
            radius*=0.5; continue
        v=_max_cos(y)
        if v<best:
            x=y; best=v; radius=min(radius*1.5,0.03)
        else:
            radius*=0.5
    return x

def solve():
    t0=time.time()
    HARD=t0+172.0
    rng=np.random.default_rng(0)
    CROSS=_build_cross()
    D7=_build_d7()
    ALL1=np.ones((1,7))/np.sqrt(7.0)
    # precompute weight-3 triple list for fam2
    import itertools
    triples=list(itertools.combinations(range(7),3))
    triples=np.array(triples)  # 35x3
    # ---- structured random search ----
    STRUCT_END=min(t0+32.0, HARD-90.0)
    top=[]  # (val, code)
    def push(v,c):
        top.append((v,c.copy()))
        top.sort(key=lambda t:t[0])
        del top[3:]
    trial=0
    # quick estimate loop
    while time.time()<STRUCT_END and trial<6000:
        fam=trial%4
        try:
            if fam==0:
                a=float(rng.uniform(0.3,1.8)); b=float(rng.uniform(0.3,1.8))
                bases=_random_bases(rng,32,a,b)
                orb=_orbits_from_bases(bases)
                X=np.vstack([CROSS,D7,orb,ALL1])
            elif fam==1:
                a=float(rng.uniform(0.3,1.8)); b=float(rng.uniform(0.3,1.8))
                bases=_random_bases(rng,44,a,b)
                orb=_orbits_from_bases(bases)
                X=np.vstack([CROSS,orb,ALL1])
            elif fam==2:
                a=float(rng.uniform(0.3,1.8)); b=float(rng.uniform(0.3,1.8))
                # w2: (1,a) on pairs
                w2=np.empty((84,7))
                k=0
                na=np.sqrt(1+a*a)
                for i in range(7):
                    for j in range(i+1,7):
                        for s1 in (1.0,-1.0):
                            for s2 in (1.0,-1.0):
                                v=np.zeros(7); v[i]=s1*1.0/na; v[j]=s2*a/na
                                w2[k]=v; k+=1
                # w3 subset 225 of 280
                nb=np.sqrt(1+a*a+b*b)
                # all 280
                all3=np.empty((280,7))
                k=0
                for ti in range(35):
                    i,j,l=triples[ti]
                    for s1 in (1.0,-1.0):
                        for s2 in (1.0,-1.0):
                            for s3 in (1.0,-1.0):
                                v=np.zeros(7); v[i]=s1/nb; v[j]=s2*a/nb; v[l]=s3*b/nb
                                all3[k]=v; k+=1
                sel=rng.choice(280,size=225,replace=False)
                w3=all3[sel]
                X=np.vstack([CROSS,w2,w3])
            else:
                N1=int(rng.integers(120,204))
                N2=323-N1
                h1=float(rng.uniform(-0.7,0.7)); h2=float(rng.uniform(-0.7,0.7))
                s1=np.sqrt(max(1e-6,1-h1*h1)); s2=np.sqrt(max(1e-6,1-h2*h2))
                A=rng.standard_normal(size=(N1,6)); A/=np.linalg.norm(A,axis=1,keepdims=True)
                B=rng.standard_normal(size=(N2,6)); B/=np.linalg.norm(B,axis=1,keepdims=True)
                X1=np.hstack([np.full((N1,1),h1), A*s1])
                X2=np.hstack([np.full((N2,1),h2), B*s2])
                # random orthogonal-ish mix: random permute coords per row? apply fixed random perm
                perm=rng.permutation(7)
                X=np.vstack([X1,X2])[:,perm]
            if X.shape!=(323,7) or not np.all(np.isfinite(X)):
                trial+=1; continue
            X=X/np.linalg.norm(X,axis=1,keepdims=True)
            v=_max_cos(X)
            if len(top)<3 or v<top[-1][0]:
                push(v,X)
        except Exception:
            pass
        trial+=1
    # ---- free-polish best 3 structured ----
    polished=[]
    for idx in range(len(top)):
        if time.time()>HARD-35: break
        X=top[idx][1].copy()
        dl=min(time.time()+12.0, HARD-30)
        # two-stage gd: coarse then fine
        X,_=_gd_polish(X, steps=500, tau0=0.015, tau1=0.003, lr0=4.0, lr1=0.6, mu=0.8, deadline=dl)
        if time.time()<dl-1:
            X,_=_gd_polish(X, steps=300, tau0=0.006, tau1=0.002, lr0=2.0, lr1=0.4, mu=0.75, deadline=dl)
        v=_max_cos(X)
        polished.append((v,X))
    # ---- load incumbent as fallback ----
    inc=None; incv=np.inf
    for fn in ("best.npy","record.npy"):
        try:
            a=np.load(fn).astype(np.float64)
            if a.shape==(323,7):
                a=a/np.linalg.norm(a,axis=1,keepdims=True)
                v=_max_cos(a)
                if v<incv: incv=v; inc=a.copy()
        except Exception:
            pass
    cands=list(polished)
    if inc is not None:
        cands.append((incv,inc))
    if len(cands)==0:
        a=rng.standard_normal(size=(323,7)); a/=np.linalg.norm(a,axis=1,keepdims=True)
        cands=[(_max_cos(a),a)]
    cands.sort(key=lambda t:t[0])
    x_best=cands[0][1].copy(); best=cands[0][0]
    # ---- at most one final 20s LP on overall best ----
    if time.time()<HARD-3:
        dl=min(time.time()+20.0, HARD-1.5)
        try:
            x_best=_lp_polish(x_best, deadline=dl, init_radius=3e-3)
            best=_max_cos(x_best)
        except Exception:
            pass
    x_best=x_best/np.linalg.norm(x_best,axis=1,keepdims=True)
    # sanitize
    bad=~np.isfinite(x_best).all(axis=1)
    if np.any(bad):
        for i in np.where(bad)[0]:
            v=rng.standard_normal(size=(7,)); v/=np.linalg.norm(v); x_best[i]=v
    x_best=x_best/np.linalg.norm(x_best,axis=1,keepdims=True)
    return x_best
