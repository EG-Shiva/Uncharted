import time
import itertools
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _adam_full(X0, steps, b0, b1, lr0, lr1):
    X = X0.copy()
    m = np.zeros_like(X); v = np.zeros_like(X)
    b1m, b2m, eps = 0.9, 0.999, 1e-8
    t = 0
    lb0, lb1 = np.log(b0), np.log(b1)
    for s in range(steps):
        frac = s / max(steps-1,1)
        beta = float(np.exp(lb0+(lb1-lb0)*frac))
        lr = float(lr0+(lr1-lr0)*frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cm = float(np.max(G))
        G -= cm; G *= beta
        np.clip(G, -30.0, 0.0, out=G)
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        S = float(np.sum(G))
        if S < 1e-100:
            continue
        GX = (G @ X) * (2.0/S)
        GX -= np.sum(GX*X, axis=1, keepdims=True)*X
        t += 1
        m = b1m*m+(1.0-b1m)*GX
        v = b2m*v+(1.0-b2m)*(GX*GX)
        c1 = 1.0-b1m**t; c2 = 1.0-b2m**t
        step = (lr/c1)*m/(np.sqrt(v/c2)+eps)
        sn = np.linalg.norm(step, axis=1, keepdims=True)+1e-16
        cap = 5.0*lr+1e-12
        step *= np.minimum(1.0, cap/sn)
        X -= step
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[nrm==0]=1.0
        X /= nrm
        md = np.sum(m*X, axis=1, keepdims=True)
        m -= md*X
    return X

def _lp_polish(x, deadline, k_top=4500, r0=1.2e-3):
    x = x/np.linalg.norm(x,axis=1,keepdims=True)
    n,d = x.shape
    best = _max_cos(x)
    radius = r0
    iu0,iu1 = np.triu_indices(n,1)
    while time.time() < deadline-1.0 and radius > 1e-14:
        Gall = x@x.T
        c_all = Gall[iu0,iu1]
        m = float(np.max(c_all))
        sel = c_all > m-max(20.0*radius,1e-6)
        cnt = int(np.count_nonzero(sel))
        if cnt==0: break
        if cnt > k_top:
            idx = np.argpartition(-c_all,k_top)[:k_top]
            I=iu0[idx];J=iu1[idx];c_sel=c_all[idx]
        else:
            I=iu0[sel];J=iu1[sel];c_sel=c_all[sel]
        p=len(I); nv=n*d+1
        cols=np.concatenate([I[:,None]*d+np.arange(d),J[:,None]*d+np.arange(d),np.full((p,1),n*d)],axis=1).ravel()
        rows=np.repeat(np.arange(p),2*d+1)
        vals=np.concatenate([x[J],x[I],-np.ones((p,1))],axis=1).ravel()
        try:
            A_ub=coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
            obj=np.zeros(nv);obj[-1]=1.0
            res=linprog(obj,A_ub=A_ub,b_ub=-c_sel,A_eq=A_eq,b_eq=np.zeros(n),bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        y=x+res.x[:-1].reshape(n,d)
        y/=np.linalg.norm(y,axis=1,keepdims=True)
        vv=_max_cos(y)
        if vv<best:
            x,best,radius=y,vv,min(radius*1.5,0.008)
        else:
            radius*=0.5
    return x

def _smooth_opt(X0, beta, maxiter, maxcor=8):
    n,d=X0.shape; Y0=X0.copy()
    def fg(Yflat):
        Y=Yflat.reshape(n,d)
        nrm=np.linalg.norm(Y,axis=1,keepdims=True); nrm=np.maximum(nrm,1e-12)
        X=Y/nrm
        G=X@X.T; np.fill_diagonal(G,-2.0)
        M=float(np.max(G))
        D=beta*(G-M); np.clip(D,-30.0,0.0,out=D)
        E=np.empty_like(D); np.exp(D,out=E); np.fill_diagonal(E,0.0)
        S=float(np.sum(E))
        if S<=0 or not np.isfinite(S): S=1e-300
        loss=M+np.log(S)/beta
        W=E/S; GX=2.0*(W@X)
        dot=np.sum(GX*X,axis=1,keepdims=True)
        GY=(GX-dot*X)/nrm
        return loss,GY.ravel()
    try:
        res=minimize(fg,Y0.ravel(),jac=True,method="L-BFGS-B",options={"maxiter":maxiter,"ftol":1e-20,"gtol":1e-12,"maxfun":maxiter*5,"maxcor":maxcor})
        Y=res.x.reshape(n,d)
    except Exception:
        Y=Y0
    Y=Y/np.maximum(np.linalg.norm(Y,axis=1,keepdims=True),1e-12)
    return Y

# ---- symmetric orbit helpers ----
def _fixed_block(kind):
    d=6
    if kind=='CROSS':
        M=np.zeros((12,d)); r=0
        for i in range(6):
            for s in (-1.0,1.0):
                M[r,i]=s; r+=1
        return M
    if kind=='PAIR':
        rows=[]
        for i in range(6):
            for j in range(i+1,6):
                for s1 in (-1.0,1.0):
                    for s2 in (-1.0,1.0):
                        v=np.zeros(d); v[i]=s1/np.sqrt(2); v[j]=s2/np.sqrt(2); rows.append(v)
        return np.array(rows)
    if kind=='HAD':
        rows=[]
        for mask in range(64):
            v=np.ones(d)
            for k in range(6):
                if (mask>>k)&1: v[k]=-1.0
            v/=np.sqrt(6); rows.append(v)
        return np.array(rows)
    if kind=='TRIPLE':
        rows=[]
        for comb in itertools.combinations(range(6),3):
            for mask in range(8):
                v=np.zeros(d)
                for t,k in enumerate(comb):
                    v[k]=-1.0/np.sqrt(3) if (mask>>t)&1 else 1.0/np.sqrt(3)
                rows.append(v)
        return np.array(rows)
    raise ValueError(kind)

def _ab_pattern():
    I=[];J=[];S1=[];S2=[]
    for i in range(6):
        for j in range(6):
            if i==j: continue
            for s1 in (-1.0,1.0):
                for s2 in (-1.0,1.0):
                    I.append(i);J.append(j);S1.append(s1);S2.append(s2)
    return np.array(I),np.array(J),np.array(S1),np.array(S2)

def _abc_pattern():
    I=[];J=[];K=[]
    for i in range(6):
        for j in range(6):
            if j==i: continue
            for k in range(6):
                if k==i or k==j: continue
                I.append(i);J.append(j);K.append(k)
    return np.array(I),np.array(J),np.array(K)

_AB_I,_AB_J,_AB_S1,_AB_S2=_ab_pattern()
_ABC_I,_ABC_J,_ABC_K=_abc_pattern()

def solve():
    t_start=time.time(); BUDGET=168.0
    def rem(): return BUDGET-(time.time()-t_start)
    rng=np.random.default_rng(0)
    n,d=250,6
    cands=[]
    for fn in ("best.npy","record.npy"):
        try:
            z=np.asarray(np.load(fn),dtype=np.float64)
            if z.shape==(n,d):
                z=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
                cands.append(z)
        except Exception: pass
    if not cands:
        z=rng.standard_normal((n,d)); z/=np.linalg.norm(z,axis=1,keepdims=True); cands.append(z)
    elite=min(cands,key=_max_cos).copy(); elite_val=_max_cos(elite)
    # precompute fixed
    F_PAIR=_fixed_block('PAIR'); F_HAD=_fixed_block('HAD'); F_CROSS=_fixed_block('CROSS')
    F_TRIPLE=_fixed_block('TRIPLE')
    # configs: list of (name, blocks)
    # blocks: list of tuples; ('AB',), ('ABC',), ('FIX',mat), ('FREE',k)
    configs=[]
    configs.append(('AB+PAIR+HAD+6f',[('AB',),('FIX',F_PAIR),('FIX',F_HAD),('FREE',6)]))
    configs.append(('ABC+PAIR+HAD+6f',[('ABC',),('FIX',F_PAIR),('FIX',F_HAD),('FREE',6)]))
    configs.append(('AB+ABC+10f',[('AB',),('ABC',),('FREE',10)]))
    configs.append(('AB+AB+10f',[('AB',),('AB2',),('FREE',10)]))
    configs.append(('ABC+ABC+10f',[('ABC',),('ABC2',),('FREE',10)]))
    configs.append(('HAD+AB+66f',[('FIX',F_HAD),('AB',),('FREE',66)]))
    configs.append(('TRIPLE+CROSS+12+FREE',[('FIX',F_TRIPLE),('FIX',F_CROSS),('FREE',78)]))
    # symmetric search
    sym_best=None; sym_val=np.inf
    beta_sym=2000.0
    deadline_sym=t_start+70.0
    # helper to build evaluator
    def make_eval(blocks):
        # count params: each AB->1, ABC->2, FREE k*6
        # param layout
        sizes=[]; kinds=[]
        for b in blocks:
            if b[0]=='AB' or b[0]=='AB2': sizes.append(1); kinds.append(b[0])
            elif b[0]=='ABC' or b[0]=='ABC2': sizes.append(2); kinds.append(b[0])
            elif b[0]=='FREE': sizes.append(b[1]*6); kinds.append('FREE')
            else: kinds.append('FIX'); sizes.append(0)
        # offsets in X
        xrows=[]
        for b in blocks:
            if b[0] in ('AB','AB2'): xrows.append(120)
            elif b[0] in ('ABC','ABC2'): xrows.append(120)
            elif b[0]=='FIX': xrows.append(b[1].shape[0])
            elif b[0]=='FREE': xrows.append(b[1])
        assert sum(xrows)==250, str(xrows)
        def build(p):
            X=np.empty((250,6),dtype=np.float64)
            ro=0; po=0
            cache={}
            for bi,b in enumerate(blocks):
                nr=xrows[bi]
                if b[0]=='AB' or b[0]=='AB2':
                    t=float(p[po]); po+=1
                    a=np.cos(t); bb=np.sin(t)
                    Bl=np.zeros((120,6))
                    Bl[np.arange(120),_AB_I]=a*_AB_S1
                    Bl[np.arange(120),_AB_J]+=bb*_AB_S2
                    X[ro:ro+nr]=Bl; cache[bi]=(a,bb,t); ro+=nr
                elif b[0]=='ABC' or b[0]=='ABC2':
                    th=float(p[po]); ph=float(p[po+1]); po+=2
                    a=np.cos(th); bb=np.sin(th)*np.cos(ph); c=np.sin(th)*np.sin(ph)
                    Bl=np.zeros((120,6))
                    Bl[np.arange(120),_ABC_I]=a
                    Bl[np.arange(120),_ABC_J]=bb
                    Bl[np.arange(120),_ABC_K]=c
                    X[ro:ro+nr]=Bl; cache[bi]=(a,bb,c,th,ph); ro+=nr
                elif b[0]=='FIX':
                    X[ro:ro+nr]=b[1]; ro+=nr
                elif b[0]=='FREE':
                    k=b[1]; Y=p[po:po+k*6].reshape(k,6); po+=k*6
                    nn=np.linalg.norm(Y,axis=1,keepdims=True); nn=np.maximum(nn,1e-12)
                    X[ro:ro+nr]=Y/nn; cache[bi]=Y.copy(); ro+=nr
            return X,cache
        def fg(p):
            X,cache=build(p)
            G=X@X.T; np.fill_diagonal(G,-2.0)
            M=float(np.max(G))
            D=beta_sym*(G-M); np.clip(D,-30.0,0.0,out=D)
            E=np.empty_like(D); np.exp(D,out=E); np.fill_diagonal(E,0.0)
            S=float(np.sum(E))
            if S<=0 or not np.isfinite(S): S=1e-300
            loss=M+np.log(S)/beta_sym
            W=E/S; GX=2.0*(W@X)
            grad=np.zeros_like(p)
            ro=0; po=0
            for bi,b in enumerate(blocks):
                nr=xrows[bi]
                Gb=GX[ro:ro+nr]
                if b[0]=='AB' or b[0]=='AB2':
                    a,bb,t=cache[bi]
                    da=float(np.sum(Gb[np.arange(120),_AB_I]*_AB_S1))
                    db=float(np.sum(Gb[np.arange(120),_AB_J]*_AB_S2))
                    dt=da*(-np.sin(t))+db*(np.cos(t))
                    grad[po]=dt; po+=1; ro+=nr
                elif b[0]=='ABC' or b[0]=='ABC2':
                    a,bb,c,th,ph=cache[bi]
                    da=float(np.sum(Gb[np.arange(120),_ABC_I]))
                    db=float(np.sum(Gb[np.arange(120),_ABC_J]))
                    dc=float(np.sum(Gb[np.arange(120),_ABC_K]))
                    st=np.sin(th); ct=np.cos(th)
                    dth=da*(-st)+db*(ct*np.cos(ph))+dc*(ct*np.sin(ph))
                    dph=db*(-st*np.sin(ph))+dc*(st*np.cos(ph))
                    grad[po]=dth; grad[po+1]=dph; po+=2; ro+=nr
                elif b[0]=='FIX':
                    ro+=nr
                elif b[0]=='FREE':
                    k=b[1]; Y=cache[bi]
                    nn=np.linalg.norm(Y,axis=1,keepdims=True); nn=np.maximum(nn,1e-12)
                    Xf=Y/nn
                    Gf=Gb
                    dot=np.sum(Gf*Xf,axis=1,keepdims=True)
                    GY=(Gf-dot*Xf)/nn
                    grad[po:po+k*6]=GY.ravel(); po+=k*6; ro+=nr
            return loss,grad
        return fg,build
    # enumerate restarts
    try:
        for cname,blocks in configs:
            if time.time()>deadline_sym-2: break
            # number of restarts by size
            nrest=4
            if '66f' in cname or 'FREE' in cname and blocks[-1][0]=='FREE' and blocks[-1][1]>20:
                nrest=2
            for r in range(nrest):
                if time.time()>deadline_sym-2: break
                fg,build=make_eval(blocks)
                # init p
                parts=[]
                for b in blocks:
                    if b[0] in ('AB','AB2'): parts.append(rng.uniform(0.1,0.7,1))
                    elif b[0] in ('ABC','ABC2'): parts.append(np.array([rng.uniform(0.5,1.2),rng.uniform(0,2*np.pi)]))
                    elif b[0]=='FREE':
                        parts.append(rng.standard_normal(b[1]*6)*0.8)
                p0=np.concatenate([np.ravel(a) for a in parts]).astype(float) if parts else np.zeros(0)
                try:
                    res=minimize(lambda p: fg(p),p0,jac=True,method="L-BFGS-B",options={"maxiter":220,"ftol":1e-20,"gtol":1e-9,"maxfun":1200,"maxcor":8})
                    p=res.x
                except Exception:
                    continue
                try:
                    X,_=build(p)
                    # dedup check: if duplicate rows reduce count, skip (still evaluate)
                    v=_max_cos(X)
                except Exception: continue
                if v<sym_val:
                    sym_val=v; sym_best=X.copy()
    except Exception:
        pass
    # break symmetry: polish symmetric best and elite
    cands_polish=[]
    cands_polish.append(elite.copy())
    if sym_best is not None and np.all(np.isfinite(sym_best)):
        # add small noise to break exact symmetry
        try:
            nb=sym_best+rng.standard_normal(sym_best.shape)*1e-4
            nb/=np.linalg.norm(nb,axis=1,keepdims=True)
            cands_polish.append(nb)
            cands_polish.append(sym_best.copy())
        except Exception: pass
    best_overall=elite.copy(); best_val=elite_val
    if sym_val<best_val:
        best_overall=sym_best.copy(); best_val=sym_val
    # full Adam + LP polish with remaining time
    # allocate: first elite, then others if time
    for idx,X0 in enumerate(cands_polish):
        if rem()<12: break
        try:
            steps=350 if idx==0 else 220
            X1=_adam_full(X0,steps=steps,b0=2000.0,b1=25000.0,lr0=0.01,lr1=0.0015)
            v1=_max_cos(X1)
            if v1<best_val: best_overall=X1.copy(); best_val=v1
            # smooth LBFGS short
            if rem()>20:
                X2=_smooth_opt(X1 if v1<best_val+5e-4 else best_overall,20000.0,60)
                v2=_max_cos(X2)
                if v2<best_val: best_overall=X2.copy(); best_val=v2
            # LP
            if rem()>8:
                X3=_lp_polish(best_overall if best_val<v1 else X1, time.time()+min(10.0,rem()-2.0))
                v3=_max_cos(X3)
                if v3<best_val: best_overall=X3.copy(); best_val=v3
        except Exception:
            continue
    # final squeeze
    if rem()>5:
        try:
            Xf=_smooth_opt(best_overall,30000.0,80)
            vf=_max_cos(Xf)
            if vf<best_val: best_overall=Xf; best_val=vf
        except Exception: pass
    if rem()>4:
        try:
            Xp=_lp_polish(best_overall,time.time()+rem()-1.0)
            vp=_max_cos(Xp)
            if vp<best_val: best_overall=Xp; best_val=vp
        except Exception: pass
    best_overall=best_overall/np.maximum(np.linalg.norm(best_overall,axis=1,keepdims=True),1e-12)
    best_overall=np.where(np.isfinite(best_overall),best_overall,0.0)
    best_overall=best_overall/np.maximum(np.linalg.norm(best_overall,axis=1,keepdims=True),1e-12)
    return best_overall
