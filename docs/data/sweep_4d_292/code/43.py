import time
import itertools
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n==0]=1.0
    return x/n

def _max_cos(x):
    G=x@x.T
    np.fill_diagonal(G,-2.0)
    return float(G.max())

# ---------- quaternion / group construction ----------
def quat_left_mat(a):
    aw,ax,ay,az=a
    return np.array([
        [aw,-ax,-ay,-az],
        [ax,aw,-az,ay],
        [ay,az,aw,-ax],
        [az,-ay,ax,aw]],dtype=float)
def quat_right_mat(b):
    bw,bx,by,bz=b
    return np.array([
        [bw,-bx,-by,-bz],
        [bx,bw,bz,-by],
        [by,-bz,bw,bx],
        [bz,by,-bx,bw]],dtype=float)

def binary_tetra():
    pts=[]
    pts.append(np.array([1.,0,0,0])); pts.append(np.array([-1.,0,0,0]))
    for i in range(3):
        v=np.zeros(4); v[1+i]=1.0; pts.append(v.copy()); pts.append(-v.copy())
    for s in itertools.product([-0.5,0.5],repeat=4):
        pts.append(np.array(s,dtype=float))
    return np.array(pts)  # 24
def binary_octa():
    T=list(binary_tetra())
    # 24 elements (1/sqrt2,1/sqrt2,0,0) permutations
    import math
    v=1.0/math.sqrt(2.0)
    for pos in itertools.combinations(range(4),2):
        for s1 in (-v,v):
            for s2 in (-v,v):
                q=np.zeros(4); q[pos[0]]=s1; q[pos[1]]=s2
                T.append(q)
    return np.array(T)  # 48
def binary_icosa():
    pts=list(binary_tetra())
    phi=(1.0+np.sqrt(5.0))/2.0
    a=0.5; b=phi/2.0; c=1.0/(2.0*phi)
    base=np.array([0.0,a,b,c])
    # even permutations of 4 coords: generate all perms, keep even
    import math
    def perm_parity(p):
        inv=0
        for i in range(len(p)):
            for j in range(i+1,len(p)):
                if p[i]>p[j]: inv+=1
        return inv%2
    seen=set()
    for perm in itertools.permutations(range(4)):
        if perm_parity(perm)!=0: continue
        for s1 in (-1,1):
            for s2 in (-1,1):
                for s3 in (-1,1):
                    q=np.zeros(4)
                    # base = [0, +-1/2, +-phi/2, +-1/(2phi)] permuted
                    vals=np.array([0.0, s1*a, s2*b, s3*c])
                    for i,pp in enumerate(perm):
                        q[pp]=vals[i]
                    # dedup roughly
                    pts.append(q.copy())
    # dedup to 120 (includes duplicates of 2T, dedup by rounding)
    arr=np.array(pts)
    # unique
    r=np.round(arr,8)
    uniq=[]
    seen=set()
    for i,row in enumerate(r):
        t=tuple(row)
        if t not in seen:
            seen.add(t); uniq.append(arr[i])
    uniq=np.array(uniq)
    # should be 120
    # normalize
    uniq=uniq/np.linalg.norm(uniq,axis=1,keepdims=True)
    return uniq

def hyperoctahedral_mats():
    mats=[]
    for perm in itertools.permutations(range(4)):
        # parity for det? keep all, det may be -1 but still orthogonal; for SO restrict det=+1*sign?
        # hyperoctahedral includes O(4); we restrict to SO(4) det=+1 for orientation
        for signs in itertools.product([-1.0,1.0],repeat=4):
            M=np.zeros((4,4))
            for i,pp in enumerate(perm):
                M[i,pp]=signs[i]
            if round(np.linalg.det(M))!=1:
                continue
            mats.append(M)
    return np.array(mats)  # 192 in SO(4)

def cyclic_mats(m,f1,f2):
    mats=[]
    for k in range(m):
        a1=2*np.pi*f1*k/m; a2=2*np.pi*f2*k/m
        c1=np.cos(a1); s1=np.sin(a1); c2=np.cos(a2); s2=np.sin(a2)
        M=np.zeros((4,4))
        M[0,0]=c1; M[0,1]=-s1; M[1,0]=s1; M[1,1]=c1
        M[2,2]=c2; M[2,3]=-s2; M[3,2]=s2; M[3,3]=c2
        mats.append(M)
    return np.array(mats)

def orbit_of_seed(seed,G):
    # G: (g,4,4), seed (4,)
    pts=np.einsum('gij,j->gi',G,seed)
    # dedup 1e-8
    # round to 1e-8 for dedup
    r=np.round(pts,8)
    _,idx=np.unique(r,axis=0,return_index=True)
    idx=np.sort(idx)
    return pts[idx]

# ---------- symmetric LSE objective ----------
class SymProblem:
    def __init__(self,G,s):
        self.G=G; self.g=len(G); self.s=s
        self.N=self.g*s
    def expand(self,seeds):
        # seeds (s,4) normalized
        # X (N,4)
        X=np.einsum('gij,sj->gsi',self.G,seeds).reshape(-1,4)
        return X
    def val_grad(self,yflat,alpha):
        s=self.s
        y=yflat.reshape(s,4)
        nrm=np.linalg.norm(y,axis=1,keepdims=True)
        nrm=np.maximum(nrm,1e-12)
        seeds=y/nrm
        X=self.expand(seeds)  # N,4
        N=X.shape[0]
        G=X@X.T
        np.fill_diagonal(G,-2.0)
        m=G.max()
        E=np.exp(alpha*(G-m))
        np.fill_diagonal(E,0.0)
        S=E.sum()
        if S<=0 or not np.isfinite(S):
            return 1e6, np.zeros_like(yflat)
        L=m+np.log(S)/alpha - np.log(N*(N-1)/2.0)/alpha  # shift invariant
        # weights
        W=E/S
        # grad wrt X
        GX=W@X  # (N,4)
        # map to seeds: X index = si*g + k? our reshape: gs indexing: g major? einsum g,s -> gs order g,s,i ; reshape gives (g*s?) Actually shape (g,s,4) reshape -> (g*s,4) with g outer. Need careful.
        # recompute mapping: X[gi*s+si]? Let's reconstruct: tmp[g,s,i]; reshape (g*s) row = g*s + s
        GX3=GX.reshape(self.g,self.s,4)
        # seeds grad: sum_g G_g^T @ GX
        GS=np.zeros((self.s,4))
        for gi in range(self.g):
            GS+= GX3[gi] @ self.G[gi]  # (s,4) @? GX row is grad, need G^T*grad: (G^T g) ; g is 4-vector: G^T@g = g@G
            # GX3[gi] is (s,4), each row grad; multiply by G on right: grad @ G
        # project through normalization: grad_y = (I-ss^T)/nrm * grad_s
        GY=np.zeros_like(y)
        for si in range(s):
            gvec=GS[si]
            sv=seeds[si]
            GY[si]=(gvec - (gvec@sv)*sv)/nrm[si,0]
        return float(L), GY.ravel()

def optimize_sym(G,s,rng,alphas=(50.0,200.0,1000.0),iters=(250,250,300)):
    prob=SymProblem(G,s)
    bestL=np.inf; bestY=None
    # single restart inside; caller loops
    y=rng.standard_normal((s,4))
    y/=np.linalg.norm(y,axis=1,keepdims=True)
    yflat=y.ravel()
    for alpha,it in zip(alphas,iters):
        def fg(yf):
            v,g=prob.val_grad(yf,alpha)
            return v,g
        try:
            res=minimize(fg,yflat,jac=True,method='L-BFGS-B',options={'maxiter':it,'ftol':1e-12,'gtol':1e-8})
            yflat=res.x
        except Exception:
            break
    v,_=prob.val_grad(yflat,alphas[-1])
    seeds=(yflat.reshape(s,4))
    seeds/=np.linalg.norm(seeds,axis=1,keepdims=True)
    X=prob.expand(seeds)
    # dedup check
    Gg=X@X.T
    np.fill_diagonal(Gg,-2)
    # if duplicates closer than 1e-8 (dot>1-5e-17) discard? check min dist
    return X,float(_max_cos(X)),yflat

# ---------- anneal + LP ----------
def _anneal(x,steps,beta0,beta1,lr0,mu,rng=None):
    n,d=x.shape
    v=np.zeros_like(x)
    lb=np.log(beta1/beta0) if beta1>beta0 else 0.0
    for t in range(steps):
        frac=t/max(1,steps-1)
        beta=beta0*np.exp(lb*frac)
        lr=lr0*(1.0-0.5*frac)
        G=x@x.T
        np.fill_diagonal(G,-2.0)
        gmax=G.max()
        W=np.exp(beta*(G-gmax))
        np.fill_diagonal(W,0.0)
        S=W.sum(axis=1,keepdims=True)+1e-300
        A=(W@x)/S
        dot=(A*x).sum(axis=1,keepdims=True)
        Gt=A-dot*x
        v=mu*v+Gt
        vd=(v*x).sum(axis=1,keepdims=True)
        v-=vd*x
        x=x-lr*v
        nrm=np.linalg.norm(x,axis=1,keepdims=True)
        np.maximum(nrm,1e-12,out=nrm)
        x/=nrm
    return x

def _lp_polish(x,seconds,r0=8e-4):
    x=_norm(np.asarray(x,dtype=float).copy())
    n,d=x.shape
    best=_max_cos(x); xb=x.copy()
    if seconds<=0: return xb
    import time as _t
    start=_t.time()
    radius=r0
    iu0=np.triu_indices(n,1)
    while _t.time()-start<seconds and radius>1e-14:
        G=x@x.T
        c=G[iu0]
        m=c.max()
        thresh=m-max(20.0*radius,1e-9)
        sel=c>thresh
        if not np.any(sel): break
        I=iu0[0][sel]; J=iu0[1][sel]; p=len(I); nv=n*d+1
        rows=np.repeat(np.arange(p),2*d+1)
        ci=I[:,None]*d+np.arange(d); cj=J[:,None]*d+np.arange(d)
        cols=np.concatenate([ci,cj,np.full((p,1),n*d)],axis=1).ravel()
        vals=np.concatenate([x[J],x[I],-np.ones((p,1))],axis=1).ravel()
        try:
            A_ub=coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
            obj=np.zeros(nv); obj[-1]=1.0
            res=linprog(obj,A_ub=A_ub,b_ub=-c[sel],A_eq=A_eq,b_eq=np.zeros(n),
                        bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        y=x+res.x[:-1].reshape(n,d); y=_norm(y)
        v=_max_cos(y)
        if v<best:
            best=v; xb=y.copy(); x=y; radius=min(radius*1.5,0.05)
        else:
            radius*=0.5
    return xb

def solve():
    t0=time.time(); TL=172.0
    rng=np.random.default_rng(292292)
    try:
        xb=np.load("best.npy").astype(float)
    except Exception:
        try: xb=np.load("record.npy").astype(float)
        except Exception: xb=rng.standard_normal((292,4))
    if xb.shape!=(292,4) or not np.all(np.isfinite(xb)):
        xb=rng.standard_normal((292,4))
    xb=_norm(xb)
    try:
        rb=np.load("record.npy").astype(float)
        if rb.shape==(292,4) and np.all(np.isfinite(rb)):
            rb=_norm(rb)
            if _max_cos(rb)<_max_cos(xb): xb=rb
    except Exception: pass
    best_val=_max_cos(xb); x_best=xb.copy()

    # --- symmetry-restricted search (ignore best except fallback) ---
    cands=[]  # list of (val,X)
    # demonstrate discard of binary-polyhedral left-only unions (cannot reach 292)
    try:
        T2=binary_tetra()
        # left action free => orbit size =24 always => 292/24 not integer => discard
        # (record check, no optimization)
        assert 292%24!=0
        O2=binary_octa()
        assert 292%48!=0
        I2=binary_icosa()
        # icosa 120: 292%120 !=0
        assert 292%120!=0
        H=hyperoctahedral_mats()
        # generic orbit 192? (SO hyperoct 192) >292/2 etc; sample to show cannot trivially tile
        # we just note discard: try subset-sum of observed orbit sizes quickly
        # sample few special seeds
        specials=[np.array([1.,0,0,0]),np.array([0.5,0.5,0.5,0.5]),
                  np.array([1/np.sqrt(2),1/np.sqrt(2),0,0]),
                  np.array([1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3),0])]
        sizes=[]
        for s0 in specials:
            s0=s0/np.linalg.norm(s0)
            o=orbit_of_seed(s0,H)
            sizes.append(len(o))
        # try to see if any combination sums to 292 (bounded search); if not, discard
        # (we don't need exact; just proceed to cyclic families)
    except Exception:
        pass

    sym_end=t0+62.0
    # cyclic families that can reach exactly 292
    families=[]
    for r in (2,5,9,17):
        families.append((cyclic_mats(73,1,r),4,f"73_{r}"))
    for r in (3,7):
        families.append((cyclic_mats(146,1,r),2,f"146_{r}"))
    for r in (3,11):
        families.append((cyclic_mats(292,1,r),1,f"292_{r}"))
    # Klein V4 order4 with 73 seeds (low-dim but symmetric)
    try:
        g0=np.eye(4)
        g1=np.diag([-1.,-1.,1.,1.]); g2=np.diag([-1.,1.,-1.,1.]); g3=np.diag([1.,-1.,-1.,1.])
        families.append((np.array([g0,g1,g2,g3]),73,"V4"))
    except Exception: pass

    # budget restarts per family
    for (G,s,tag) in families:
        if time.time()>sym_end: break
        if s*len(G)!=292: continue  # discard unions that cannot reach 292
        # number of restarts based on s (smaller s => more restarts)
        if s<=2: nrep=6
        elif s==4: nrep=5
        else: nrep=2
        for rep in range(nrep):
            if time.time()>sym_end: break
            # deduplicate check tolerance 1e-8 inside optimize
            try:
                X,val,_=optimize_sym(G,s,rng)
            except Exception:
                continue
            if not np.all(np.isfinite(X)) or X.shape!=(292,4): continue
            # dedup closer than 1e-8: check max off-diag close to 1
            Gg=X@X.T; np.fill_diagonal(Gg,-2)
            if Gg.max()>1-1e-12:  # duplicate points
                continue
            cands.append((val,X.copy()))
            # early: if symmetric already competitive, keep
    # pick best few symmetric candidates, break symmetry + LP polish
    cands=sorted(cands,key=lambda t:t[0])[:4]
    sym_best_val=np.inf; sym_best_X=None
    for (val,X) in cands:
        # tiny tangent perturbation 1e-4 to break symmetry
        try:
            pert=rng.standard_normal(X.shape)*1e-4
            # project to tangent
            pert-= (pert*X).sum(axis=1,keepdims=True)*X
            Y=_norm(X+pert)
            # short anneal to leave symmetric subspace slightly? keep tiny
            # directly LP-polish
            remain=min(6.0,max(1.0,sym_end-time.time()+8.0))
            # use small portion
            Yp=_lp_polish(Y,min(3.0,remain))
            v=_max_cos(Yp)
            if v<sym_best_val:
                sym_best_val=v; sym_best_X=Yp.copy()
        except Exception:
            continue
    # compare symmetric-polished vs incumbent later; keep incumbent for now
    if sym_best_X is not None and sym_best_val<best_val:
        best_val=sym_best_val; x_best=sym_best_X.copy()

    # --- basin-hopping improvement of incumbent (topology changes) ---
    explore_end=t0+TL-30.0
    it=0
    sigmas=[0.004,0.01,0.025,0.06,0.12]
    while time.time()<explore_end:
        it+=1
        if time.time()+1.0>explore_end: break
        X=x_best.copy()
        r=rng.random()
        n=292
        if r<0.5:
            # kill worst + fill holes
            Gc=X@X.T; np.fill_diagonal(Gc,-2.0)
            ppm=Gc.max(axis=1)
            k=int(rng.integers(1,7))
            kill=np.argsort(-ppm+ rng.standard_normal(n)*1e-3)[:k]
            mask=np.ones(n,bool); mask[kill]=False
            Yk=X[mask]
            # shake keepers slightly
            Yk=Yk+0.01*rng.standard_normal(Yk.shape); Yk=_norm(Yk)
            # fill by random candidates
            Nc=4000
            cand=rng.standard_normal((Nc,4)); cand=_norm(cand)
            for _ in range(k):
                D=cand@Yk.T
                worst=D.max(axis=1); bi=int(np.argmin(worst))
                c=cand[bi].copy()
                for s_ in range(30):
                    dots=Yk@c; md=dots.max()
                    W=np.exp(30.0*(dots-md)); W/=W.sum()+1e-300
                    g=W@Yk; g-=(g@c)*c
                    c-=0.12*g
                    nn=np.linalg.norm(c)
                    if nn<1e-12: c=cand[bi].copy(); break
                    c/=nn
                Yk=np.vstack([Yk,c.reshape(1,4)])
            X=Yk
            X+=0.004*rng.standard_normal(X.shape); X=_norm(X)
        elif r<0.8:
            # cluster shake around random active edge
            Gc=X@X.T; np.fill_diagonal(Gc,-2.0); m=Gc.max()
            iu=np.triu_indices(n,1); cc=Gc[iu]
            sel=np.where(cc>m-2e-3)[0]
            if len(sel)>0:
                e=sel[int(rng.integers(len(sel)))]
                i,j=int(iu[0][e]),int(iu[1][e])
                tgt=np.unique(np.concatenate([[i,j],np.argsort(-Gc[i])[:6],np.argsort(-Gc[j])[:6]]))
                X[tgt]+=0.07*rng.standard_normal((len(tgt),4)); X=_norm(X)
            else:
                X+=0.02*rng.standard_normal(X.shape); X=_norm(X)
        else:
            sigma=sigmas[int(rng.integers(len(sigmas)))]
            X=X+sigma*rng.standard_normal(X.shape); X=_norm(X)
            if rng.random()<0.4:
                k=int(rng.integers(1,4))
                idx=rng.choice(n,size=k,replace=False)
                R=rng.standard_normal((k,4)); R=_norm(R); X[idx]=R
        # anneal params
        u=rng.random()
        if u<0.4: steps,b0,b1,lr0,mu=1000,25.0,350.0,0.06,0.85
        elif u<0.7: steps,b0,b1,lr0,mu=1400,15.0,250.0,0.09,0.88
        else: steps,b0,b1,lr0,mu=700,10.0,150.0,0.12,0.90
        remain=explore_end-time.time()
        maxsteps=int(remain/0.00028)-150
        if maxsteps<300: break
        steps=min(steps,maxsteps)
        Xo=_anneal(X,steps,b0,b1,lr0,mu)
        val=_max_cos(Xo)
        if val<best_val+0.001:
            try:
                Xo2=_lp_polish(Xo,min(1.5,max(0.5,(explore_end-time.time())*0.1)))
                v2=_max_cos(Xo2)
                if np.all(np.isfinite(Xo2)): Xo,val=Xo2,v2
            except Exception: pass
        if val<best_val:
            best_val=val; x_best=Xo.copy()
            # if symmetric candidate was best, keep competing
    # final polish
    rem=TL-(time.time()-t0)-1.0
    rem=max(4.0,min(28.0,rem))
    try:
        x_best=_lp_polish(x_best,rem*0.7,r0=8e-4)
        best_val=_max_cos(x_best)
    except Exception: pass
    # ensure better than loaded incumbent: fallback
    try:
        inc=np.load("best.npy").astype(float); inc=_norm(inc)
        if _max_cos(inc)<_max_cos(x_best): x_best=inc
    except Exception: pass
    x_best=_norm(x_best)
    return x_best
