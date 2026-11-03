import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _signed_D(s):
    # s length 13, +-1 ; D[k,j] for P^k
    d = 13
    D = np.ones((13, d))
    for k in range(1, 13):
        # D_k[j] = s_j * D_{k-1}[j-1]
        D[k] = s * np.roll(D[k-1], 1)
    return D

def _build_unsigned(orb, free):
    # orb (7,13), free (3,13)
    X = np.empty((94, 13))
    for i in range(orb.shape[0]):
        v = orb[i]
        for k in range(13):
            X[i*13+k] = np.roll(v, k)
    X[91:94] = free
    return X

def _build_signed(orb, free, D):
    X = np.empty((94, 13))
    for i in range(orb.shape[0]):
        v = orb[i]
        for k in range(13):
            X[i*13+k] = D[k] * np.roll(v, k)
    X[91:94] = free
    return X

def _build_mixed(orb5, vant, free):
    # 5x13 + 26 (orbit+antipode) +3 =94
    X = np.empty((94, 13))
    for i in range(5):
        v = orb5[i]
        for k in range(13):
            X[i*13+k] = np.roll(v, k)
    v = vant[0]
    for k in range(13):
        X[65+k] = np.roll(v, k)
        X[65+13+k] = -np.roll(v, k)
    X[91:94] = free
    return X

def _build_mixed_signed(orb5, vant, free, D):
    X = np.empty((94, 13))
    for i in range(5):
        v = orb5[i]
        for k in range(13):
            X[i*13+k] = D[k] * np.roll(v, k)
    v = vant[0]
    for k in range(13):
        X[65+k] = D[k] * np.roll(v, k)
        X[65+13+k] = -D[k] * np.roll(v, k)
    X[91:94] = free
    return X

def _lse_grad_X(X, kk):
    n = X.shape[0]
    G = X @ X.T
    m = float(np.max(G - np.eye(n)*2.0))
    W = np.exp(kk*(G-m))
    np.fill_diagonal(W, 0.0)
    S = float(W.sum())*0.5
    if S<=0 or not np.isfinite(S):
        return None, m
    Grad = (W @ X)/S
    dot = (Grad*X).sum(axis=1,keepdims=True)
    Gt = Grad - dot*X
    return Gt, m

def _agg_unsigned(Gt, norb):
    # Gt (94,d) -> g_orb (norb,13), g_free (3,13)
    g_orb = np.zeros((norb,13))
    for i in range(norb):
        s = np.zeros(13)
        for k in range(13):
            s += np.roll(Gt[i*13+k], -k)
        g_orb[i]=s
    g_free = Gt[91:94].copy()
    return g_orb, g_free

def _agg_signed(Gt, norb, D):
    g_orb = np.zeros((norb,13))
    for i in range(norb):
        s = np.zeros(13)
        for k in range(13):
            s += np.roll(D[k]*Gt[i*13+k], -k)
        g_orb[i]=s
    g_free = Gt[91:94].copy()
    return g_orb, g_free

def _agg_mixed(Gt, D=None):
    # 5 orb + 1 antipodal + free
    g5 = np.zeros((5,13))
    for i in range(5):
        s=np.zeros(13)
        for k in range(13):
            g = Gt[i*13+k]
            if D is not None:
                g = D[k]*g
            s += np.roll(g,-k)
        g5[i]=s
    gant = np.zeros((1,13))
    s=np.zeros(13)
    for k in range(13):
        ga = Gt[65+k]; gb = Gt[65+13+k]
        if D is not None:
            ga = D[k]*ga; gb = D[k]*gb
        s += np.roll(ga,-k) - np.roll(gb,-k)
    gant[0]=s
    g_free = Gt[91:94].copy()
    return g5, gant, g_free

def _sym_adam_run(Zorb0, Zfree0, build, agg, steps, k0,k1,lr0,lr1, seed_rng=None):
    orb = Zorb0.copy(); free = Zfree0.copy()
    # normalize
    orb/=np.linalg.norm(orb,axis=1,keepdims=True)+1e-30
    free/=np.linalg.norm(free,axis=1,keepdims=True)+1e-30
    m1o=np.zeros_like(orb); v1o=np.zeros_like(orb)
    m1f=np.zeros_like(free); v1f=np.zeros_like(free)
    b1,b2,eps=0.9,0.999,1e-8
    best_val=1e9; best_orb=orb.copy(); best_free=free.copy()
    is_mixed = isinstance(orb, tuple)
    for s_ in range(steps):
        frac=s_/max(1,steps-1)
        kk=k0*(k1/k0)**frac
        lr=lr0*(lr1/lr0)**frac
        if is_mixed:
            X=build(orb[0],orb[1],free)
        else:
            X=build(orb,free)
        Gt,m = _lse_grad_X(X,kk)
        if Gt is None:
            break
        if is_mixed:
            g5,gant,gfree = agg(Gt)
            # tangent project
            g5 = g5 - (g5*orb[0]).sum(axis=1,keepdims=True)*orb[0]
            gant = gant - (gant*orb[1]).sum(axis=1,keepdims=True)*orb[1]
            gfree = gfree - (gfree*free).sum(axis=1,keepdims=True)*free
            # adam update (share step counters)
            for (g, m1, v1, p) in [(g5,m1o[0] if False else None,None,None)]:
                pass
            # manual adam for three groups with same b
            # use separate moments stored in closure lists
            # simplify: SGD-Adam inline
            Gtup=(g5,gant,gfree)
            # we keep moments in outer vars: need structures
            # fallback: simple gradient step with lr normalisation
            # implement per-group adam using persistent arrays
            # (handled below via generic path - instead unpack)
            pass
        # non-mixed path handled outside; mixed handled separately
        break
    return best_orb,best_free,best_val

# Simpler: generic symmetric Adam that works on flat stacked Z with custom build/agg closures
def _sym_adam(Z0_list, build_fn, agg_fn, steps, k0,k1,lr0,lr1):
    Zs=[z.copy() for z in Z0_list]
    for z in Zs:
        z/=np.linalg.norm(z,axis=1,keepdims=True)+1e-30
    Ms=[np.zeros_like(z) for z in Zs]
    Vs=[np.zeros_like(z) for z in Zs]
    b1,b2,eps=0.9,0.999,1e-8
    best_val=1e9
    best=[z.copy() for z in Zs]
    for s_ in range(steps):
        frac=s_/max(1,steps-1)
        kk=k0*(k1/k0)**frac
        lr=lr0*(lr1/lr0)**frac
        X=build_fn(Zs)
        Gt,_=_lse_grad_X(X,kk)
        if Gt is None:
            break
        Gs=agg_fn(Gt)
        # project
        for idx in range(len(Zs)):
            Gs[idx]=Gs[idx]-(Gs[idx]*Zs[idx]).sum(axis=1,keepdims=True)*Zs[idx]
        for idx in range(len(Zs)):
            Ms[idx]=b1*Ms[idx]+(1-b1)*Gs[idx]
            Vs[idx]=b2*Vs[idx]+(1-b2)*(Gs[idx]*Gs[idx])
            bc1=1.0-b1**(s_+1); bc2=1.0-b2**(s_+1)
            upd=lr*(Ms[idx]/bc1)/(np.sqrt(Vs[idx]/bc2)+eps)
            Zs[idx]=Zs[idx]-upd
            Zs[idx]/=np.linalg.norm(Zs[idx],axis=1,keepdims=True)+1e-30
        if (s_%25==0) or s_==steps-1:
            Xc=build_fn(Zs)
            vv=_max_cos(Xc)
            if vv<best_val:
                best_val=vv
                best=[z.copy() for z in Zs]
    return best,best_val

def _global_adam_lse(x,steps,k0,k1,lr0,lr1):
    n,d=x.shape
    m1=np.zeros_like(x); v1=np.zeros_like(x)
    b1,b2,eps=0.9,0.999,1e-8
    for s in range(steps):
        frac=s/max(1,steps-1)
        kk=k0*(k1/k0)**frac; lr=lr0*(lr1/lr0)**frac
        Gt,_=_lse_grad_X(x,kk)
        if Gt is None: break
        m1=b1*m1+(1-b1)*Gt; v1=b2*v1+(1-b2)*(Gt*Gt)
        bc1=1-b1**(s+1); bc2=1-b2**(s+1)
        upd=lr*(m1/bc1)/(np.sqrt(v1/bc2)+eps)
        x=x-upd
        x/=np.linalg.norm(x,axis=1,keepdims=True)+1e-30
    return x

def _global_adam_quant(x,steps,M,lr0,lr1):
    n,d=x.shape
    m1=np.zeros_like(x); v1=np.zeros_like(x)
    b1,b2,eps=0.9,0.999,1e-8
    iu=np.triu_indices(n,1)
    for s in range(steps):
        frac=s/max(1,steps-1)
        lr=lr0*(lr1/lr0)**frac
        G=x@x.T
        vals=G[iu]
        if M>=vals.size: t=float(vals.min())
        else: t=float(np.partition(vals,-M)[-M])
        Mm=G-t; Mm[Mm<0]=0.0
        np.fill_diagonal(Mm,0.0)
        cnt=float((Mm>0).sum()*0.5)
        if cnt<1: cnt=1.0
        Grad=(Mm@x)/cnt
        dot=(Grad*x).sum(axis=1,keepdims=True)
        Gt=Grad-dot*x
        m1=b1*m1+(1-b1)*Gt; v1=b2*v1+(1-b2)*(Gt*Gt)
        bc1=1-b1**(s+1); bc2=1-b2**(s+1)
        upd=lr*(m1/bc1)/(np.sqrt(v1/bc2)+eps)
        x=x-upd
        x/=np.linalg.norm(x,axis=1,keepdims=True)+1e-30
    return x

def _lp_polish(x,deadline):
    x=x/np.linalg.norm(x,axis=1,keepdims=True)
    n,d=x.shape
    best=_max_cos(x)
    radius=1e-3
    iu=np.triu_indices(n,1)
    while time.time()<deadline and radius>1e-14:
        cmat=(x@x.T)[iu]
        m=float(cmat.max())
        sel=cmat>m-max(20.0*radius,1e-9)
        I,J=iu[0][sel],iu[1][sel]
        p=len(I)
        if p==0: break
        nv=n*d+1
        rows=np.repeat(np.arange(p),2*d+1)
        cols=np.concatenate([I[:,None]*d+np.arange(d),J[:,None]*d+np.arange(d),np.full((p,1),n*d)],1).ravel()
        vals=np.concatenate([x[J],x[I],-np.ones((p,1))],1).ravel()
        try:
            A_ub=coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        except Exception: break
        obj=np.zeros(nv); obj[-1]=1.0
        try:
            res=linprog(obj,A_ub=A_ub,b_ub=-cmat[sel],A_eq=A_eq,b_eq=np.zeros(n),bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        y=x+res.x[:-1].reshape(n,d)
        y/=np.linalg.norm(y,axis=1,keepdims=True)
        v=_max_cos(y)
        if v<best: x,best,radius=y,v,min(radius*1.5,0.05)
        else: radius*=0.5
    return x,best

def _tangent_perturb(X,sigma,rng):
    N=rng.standard_normal(size=X.shape)
    N=N-(N*X).sum(axis=1,keepdims=True)*X
    Y=X+sigma*N
    Y/=np.linalg.norm(Y,axis=1,keepdims=True)+1e-30
    return Y

def solve():
    T0=time.time()
    BUDGET=165.0
    END=T0+BUDGET
    rng=np.random.default_rng(0)
    rec=None; bst=None
    for fn in ["record.npy","best.npy"]:
        try:
            a=np.load(fn).astype(float)
            a/=np.linalg.norm(a,axis=1,keepdims=True)
            if a.shape==(94,13) and np.all(np.isfinite(a)):
                if fn=="record.npy": rec=a.copy()
                else: bst=a.copy()
        except Exception: pass
    if rec is None and bst is None:
        r=rng.standard_normal(size=(94,13)); r/=np.linalg.norm(r,axis=1,keepdims=True)
        rec=r.copy(); bst=r.copy()
    if rec is None: rec=bst.copy()
    if bst is None: bst=rec.copy()
    rv=_max_cos(rec); bv=_max_cos(bst)
    if bv<rv: global_best=bst.copy(); global_val=bv
    else: global_best=rec.copy(); global_val=rv
    # signed pattern: two flips -> product +1
    s_pat=np.ones(13); s_pat[0]=-1.0; s_pat[5]=-1.0
    D=_signed_D(s_pat)
    sym_cands=[]
    def time_left(): return END-time.time()
    # helper to run one construction
    # Construction A: unsigned 7x13+3
    constructions=[]
    constructions.append(("unsigned",30.0))
    constructions.append(("signed",30.0))
    constructions.append(("mixed",30.0))
    constructions.append(("mixed_signed",30.0))
    for cname,clim in constructions:
        c_end=min(time.time()+clim, END-70.0)
        if time.time()>c_end: continue
        # number of inits depends on remaining
        n_init=4
        for init in range(n_init):
            if time.time()>c_end: break
            if time_left()<75: break
            # random init with sign/perm variants
            orb_n = 7 if cname in ("unsigned","signed") else 5
            Zorb=rng.standard_normal(size=(orb_n,13))
            # sign variant: randomly flip signs of some reps
            if init%2==1:
                Zorb*=(rng.choice([-1.0,1.0],size=(orb_n,1)))
            # permutation variant: permute coords of init
            if init%3==2:
                p=rng.permutation(13); Zorb=Zorb[:,p]
            Zfree=rng.standard_normal(size=(3,13))
            if cname in ("mixed","mixed_signed"):
                Zant=rng.standard_normal(size=(1,13))
            if cname=="unsigned":
                build=lambda Zs: _build_unsigned(Zs[0],Zs[1])
                agg=lambda Gt: list(_agg_unsigned(Gt,7))
                Z0=[Zorb,Zfree]
            elif cname=="signed":
                build=lambda Zs,D=D: _build_signed(Zs[0],Zs[1],D)
                agg=lambda Gt,D=D: list(_agg_signed(Gt,7,D))
                Z0=[Zorb,Zfree]
            elif cname=="mixed":
                build=lambda Zs: _build_mixed(Zs[0],Zs[1],Zs[2])
                def agg(Gt):
                    g5,gant,gf=_agg_mixed(Gt,None)
                    return [g5,gant,gf]
                Z0=[Zorb,Zant,Zfree]
            else:
                build=lambda Zs,D=D: _build_mixed_signed(Zs[0],Zs[1],Zs[2],D)
                def agg(Gt,D=D):
                    g5,gant,gf=_agg_mixed(Gt,D)
                    return [g5,gant,gf]
                Z0=[Zorb,Zant,Zfree]
            try:
                bestZ,bv2=_sym_adam(Z0,build,agg,steps=450,k0=30.0,k1=400.0,lr0=0.02,lr1=0.002)
                Xsym=build(bestZ)
                Xsym/=np.linalg.norm(Xsym,axis=1,keepdims=True)
                vv=_max_cos(Xsym)
                sym_cands.append((vv,Xsym.copy(),cname))
            except Exception:
                continue
    sym_cands.sort(key=lambda t:t[0])
    # symmetry-breaking polish on top symmetric (compare only after polish)
    polished=[]
    for idx,(sv,SX,cn) in enumerate(sym_cands[:4]):
        if time_left()<25: break
        for sg in [0.008,0.03]:
            if time_left()<20: break
            try:
                Y=_tangent_perturb(SX,sg,rng)
                Y=_global_adam_quant(Y,steps=250,M=600,lr0=0.015,lr1=0.002)
                Y=_global_adam_lse(Y,steps=250,k0=40.0,k1=500.0,lr0=0.012,lr1=0.0012)
                vv=_max_cos(Y)
                polished.append((vv,Y.copy()))
                if vv<global_val:
                    global_val=vv; global_best=Y.copy()
            except Exception: continue
    # also direct incumbent quantile crushing (small, to not dominate)
    if time_left()>20:
        try:
            Y=_tangent_perturb(global_best,0.02,rng)
            Y=_global_adam_quant(Y,steps=200,M=600,lr0=0.012,lr1=0.0015)
            Y=_global_adam_lse(Y,steps=200,k0=50.0,k1=500.0,lr0=0.01,lr1=0.001)
            vv=_max_cos(Y)
            polished.append((vv,Y.copy()))
            if vv<global_val: global_val=vv; global_best=Y.copy()
        except Exception: pass
    # final LP polish on best few
    cands=[(global_val,global_best.copy())]
    polished.sort(key=lambda t:t[0])
    for vv,XX in polished[:3]:
        cands.append((vv,XX.copy()))
    # dedup
    uniq=[]
    for vv,XX in sorted(cands,key=lambda t:t[0]):
        dup=False
        for _,YY in uniq:
            try:
                if np.linalg.norm(XX-YY)<1e-6: dup=True; break
            except Exception: pass
        if not dup:
            uniq.append((vv,XX))
        if len(uniq)>=4: break
    if time_left()>6 and len(uniq)>0:
        per=min(max(time_left()/max(1,len(uniq)),4.0),18.0)
        fb=global_best.copy(); fv=global_val
        for vv,XX in uniq:
            if time.time()>END-1.5: break
            dl=min(time.time()+per,END-0.5)
            try:
                P,pv=_lp_polish(XX.copy(),dl)
            except Exception: continue
            if pv<fv: fv=pv; fb=P.copy()
        global_best=fb; global_val=fv
    global_best/=np.linalg.norm(global_best,axis=1,keepdims=True)+1e-30
    return global_best
