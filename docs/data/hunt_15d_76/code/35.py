import time
import numpy as np

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

_FIX = np.ones(15) / np.sqrt(15.0)

def _build_c15(seeds):
    # seeds (5,15) unit
    X = np.empty((76, 15))
    for k in range(5):
        s = seeds[k]
        for sh in range(15):
            X[k*15+sh] = np.roll(s, sh)
    X[75] = _FIX
    return X

def _grad_c15(Gfull, seeds):
    # Gfull (76,15) tangent grad on full code
    gs = np.zeros_like(seeds)
    for k in range(5):
        acc = np.zeros(15)
        for sh in range(15):
            acc += np.roll(Gfull[k*15+sh], -sh)
        gs[k] = acc
    # project to seed tangents
    dot = np.sum(gs*seeds, axis=1, keepdims=True)
    gs = gs - dot*seeds
    return gs

def _S_op(x, k):
    # signed cyclic S^k : S(x)[0]=-x[14], S(x)[i]=x[i-1]
    # apply k times (k may be >=0)
    k = k % 30
    y = x.copy()
    for _ in range(k):
        y = np.concatenate(([-y[-1]], y[:-1]))
    return y

def _S_inv(g, k):
    # inverse of S^k = S^{-k} = S^{30-k}; S^{-1}(y)[i]=y[i+1], last=-y[0]
    k = k % 30
    y = g.copy()
    for _ in range(k):
        y = np.concatenate((y[1:], [-y[0]]))
    return y

def _build_signed(seeds30a, seeds30b, seed15):
    X = np.empty((76, 15))
    for sh in range(30):
        X[sh] = _S_op(seeds30a, sh)
    for sh in range(30):
        X[30+sh] = _S_op(seeds30b, sh)
    for sh in range(15):
        X[60+sh] = np.roll(seed15, sh)
    X[75] = _FIX
    return X

def _grad_signed(Gfull, sa, sb, sc):
    ga = np.zeros(15); gb = np.zeros(15); gc = np.zeros(15)
    for sh in range(30):
        ga += _S_inv(Gfull[sh], sh)
        gb += _S_inv(Gfull[30+sh], sh)
    for sh in range(15):
        gc += np.roll(Gfull[60+sh], -sh)
    for g, s in ((ga, sa), (gb, sb), (gc, sc)):
        d = float(g @ s)
        g -= d*s
    return ga, gb, gc

def _build_anti(seeds38):
    return np.concatenate([seeds38, -seeds38], axis=0)

def _grad_anti(Gfull):
    ga = Gfull[:38] - Gfull[38:]
    return ga

def _adam_step(par, grad, m, v, b1t, b2t, lr, b1=0.9, b2=0.999, eps=1e-8):
    m[:] = b1*m + (1-b1)*grad
    v[:] = b2*v + (1-b2)*(grad*grad)
    mh = m/(1-b1t); vh = v/(1-b2t)
    par[:] = par - lr*mh/(np.sqrt(vh)+eps)

def _soft_weights(X, beta):
    G = X @ X.T
    np.fill_diagonal(G, -10.0)
    gm = float(G.max())
    E = np.exp(beta*(G-gm))
    np.fill_diagonal(E, 0.0)
    S = float(E.sum())
    return E, S, gm

def _anneal_c15(seeds, steps, T0, T1, lr0, lr1, rng=None):
    seeds = seeds.copy()
    m = np.zeros_like(seeds); v = np.zeros_like(seeds)
    b1t = 1.0; b2t = 1.0
    b1, b2 = 0.9, 0.999
    lb = np.log((1.0/T1)/(1.0/T0)) if T0!=T1 else 0.0
    ll = np.log(lr1/lr0) if lr0!=lr1 else 0.0
    beta0 = 1.0/T0
    for it in range(steps):
        f = it/(steps-1) if steps>1 else 0.0
        beta = beta0*np.exp(lb*f)
        lr = lr0*np.exp(ll*f)
        X = _build_c15(seeds)
        E, S, gm = _soft_weights(X, beta)
        if S==0 or not np.isfinite(S):
            break
        Gr = 2.0*(E @ X)/S
        # tangent project full
        Gr = Gr - np.sum(Gr*X,axis=1,keepdims=True)*X
        gs = _grad_c15(Gr, seeds)
        b1t*=b1; b2t*=b2
        _adam_step(seeds, gs, m, v, b1t, b2t, lr)
        seeds = _norm(seeds)
    return seeds

def _anneal_signed(sa, sb, sc, steps, T0, T1, lr0, lr1):
    sa=sa.copy(); sb=sb.copy(); sc=sc.copy()
    ma=np.zeros_like(sa); va=np.zeros_like(sa)
    mb=np.zeros_like(sb); vb=np.zeros_like(sb)
    mc=np.zeros_like(sc); vc=np.zeros_like(sc)
    b1,b2=0.9,0.999
    b1t=1.0; b2t=1.0
    lb=np.log((1.0/T1)/(1.0/T0)) if T0!=T1 else 0.0
    ll=np.log(lr1/lr0) if lr0!=lr1 else 0.0
    beta0=1.0/T0
    sa=sa.reshape(1,15); sb=sb.reshape(1,15); sc=sc.reshape(1,15)
    ma=ma.reshape(1,15); va=va.reshape(1,15)
    mb=mb.reshape(1,15); vb=vb.reshape(1,15)
    mc=mc.reshape(1,15); vc=vc.reshape(1,15)
    for it in range(steps):
        f=it/(steps-1) if steps>1 else 0.0
        beta=beta0*np.exp(lb*f); lr=lr0*np.exp(ll*f)
        X=_build_signed(sa[0],sb[0],sc[0])
        E,S,gm=_soft_weights(X,beta)
        if S==0 or not np.isfinite(S):
            break
        Gr=2.0*(E@X)/S
        Gr=Gr-np.sum(Gr*X,axis=1,keepdims=True)*X
        ga,gb,gc=_grad_signed(Gr,sa[0],sb[0],sc[0])
        ga=ga.reshape(1,15); gb=gb.reshape(1,15); gc=gc.reshape(1,15)
        b1t*=b1; b2t*=b2
        _adam_step(sa,ga,ma,va,b1t,b2t,lr)
        _adam_step(sb,gb,mb,vb,b1t,b2t,lr)
        _adam_step(sc,gc,mc,vc,b1t,b2t,lr)
        sa=_norm(sa); sb=_norm(sb); sc=_norm(sc)
    return sa[0],sb[0],sc[0]

def _anneal_anti(seeds, steps, T0, T1, lr0, lr1):
    seeds=seeds.copy()
    m=np.zeros_like(seeds); v=np.zeros_like(seeds)
    b1,b2=0.9,0.999; b1t=1.0; b2t=1.0
    lb=np.log((1.0/T1)/(1.0/T0)) if T0!=T1 else 0.0
    ll=np.log(lr1/lr0) if lr0!=lr1 else 0.0
    beta0=1.0/T0
    for it in range(steps):
        f=it/(steps-1) if steps>1 else 0.0
        beta=beta0*np.exp(lb*f); lr=lr0*np.exp(ll*f)
        X=_build_anti(seeds)
        E,S,gm=_soft_weights(X,beta)
        if S==0 or not np.isfinite(S):
            break
        Gr=2.0*(E@X)/S
        Gr=Gr-np.sum(Gr*X,axis=1,keepdims=True)*X
        gs=_grad_anti(Gr)
        dot=np.sum(gs*seeds,axis=1,keepdims=True)
        gs=gs-dot*seeds
        b1t*=b1; b2t*=b2
        _adam_step(seeds,gs,m,v,b1t,b2t,lr)
        seeds=_norm(seeds)
    return seeds

def _anneal_full(X, steps, T0, T1, lr0, lr1):
    X=X.copy()
    m=np.zeros_like(X); v=np.zeros_like(X)
    b1,b2=0.9,0.999; b1t=1.0; b2t=1.0
    lb=np.log((1.0/T1)/(1.0/T0)) if T0!=T1 else 0.0
    ll=np.log(lr1/lr0) if lr0!=lr1 else 0.0
    beta0=1.0/T0
    for it in range(steps):
        f=it/(steps-1) if steps>1 else 0.0
        beta=beta0*np.exp(lb*f); lr=lr0*np.exp(ll*f)
        E,S,gm=_soft_weights(X,beta)
        if S==0 or not np.isfinite(S):
            break
        Gr=2.0*(E@X)/S
        Gr=Gr-np.sum(Gr*X,axis=1,keepdims=True)*X
        b1t*=b1; b2t*=b2
        _adam_step(X,Gr,m,v,b1t,b2t,lr)
        X=_norm(X)
    return X

def _light_lp(x, t_end):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return x
    x=_norm(x.copy())
    n,d=x.shape
    best=_maxcos(x)
    radius=8e-4
    iu=np.triu_indices(n,1)
    it=0
    while time.time()<t_end-0.6 and radius>1e-13 and it<12:
        it+=1
        c=(x@x.T)[iu]
        m=float(c.max())
        thr=max(20*radius,1e-9)
        sel=c>m-thr
        if int(sel.sum())>1200:
            idx=np.argpartition(-c,1200)[:1200]
            mask=np.zeros_like(sel); mask[idx]=True; sel=mask
        I,J=iu[0][sel],iu[1][sel]
        p=len(I)
        if p==0:
            break
        nv=n*d+1
        rows=np.repeat(np.arange(p),2*d+1)
        cols=np.concatenate([I[:,None]*d+np.arange(d),J[:,None]*d+np.arange(d),np.full((p,1),n*d)],1).ravel()
        vals=np.concatenate([x[J],x[I],-np.ones((p,1))],1).ravel()
        try:
            A_ub=coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
            obj=np.zeros(nv); obj[-1]=1.0
            res=linprog(obj,A_ub=A_ub,b_ub=-c[sel],A_eq=A_eq,b_eq=np.zeros(n),
                        bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            break
        if res.status!=0:
            radius*=0.5; continue
        y=x+res.x[:-1].reshape(n,d)
        y=_norm(y)
        val=_maxcos(y)
        if val<best:
            x,best,radius=y,val,min(radius*1.5,0.01)
        else:
            radius*=0.5
    return x

def solve():
    import time
    t0=time.time()
    deadline=t0+170.0
    rng=np.random.default_rng(7)
    try:
        xb=np.load("best.npy").astype(float)
    except Exception:
        xb=None
    try:
        xr=np.load("record.npy").astype(float)
    except Exception:
        xr=None
    def ok(a):
        return a is not None and a.shape==(76,15) and np.all(np.isfinite(a))
    cands=[]
    if ok(xb):
        cands.append(_norm(xb))
    if ok(xr):
        cands.append(_norm(xr))
    if not cands:
        r=rng.normal(size=(76,15)); return _norm(r)
    best=cands[0].copy(); bv=_maxcos(best)
    for c in cands[1:]:
        vv=_maxcos(c)
        if vv<bv:
            best,bv=c.copy(),vv
    best0=best.copy(); bv0=bv

    def rand_seed5(rg):
        s=rg.normal(size=(5,15))
        s=_norm(s)
        # avoid near-fixed seeds
        for k in range(5):
            if abs(float(s[k]@_FIX))>0.92:
                s[k]=_norm(rg.normal(size=(1,15)))[0]
        return s

    sym_best=None; sym_bv=1e9
    # --- C15 search ---
    # 4 restarts, coarse then refine best
    n_rest=4
    for r in range(n_rest):
        if time.time()>deadline-70:
            break
        rg=np.random.default_rng(100+r)
        seeds=rand_seed5(rg)
        seeds=_anneal_c15(seeds, steps=1200, T0=0.025, T1=0.003, lr0=0.02, lr1=0.003)
        if time.time()>deadline-70:
            X=_build_c15(_norm(seeds)); vv=_maxcos(X)
            if vv<sym_bv:
                sym_bv,sym_best=vv,X.copy()
            break
        seeds=_anneal_c15(seeds, steps=1200, T0=0.004, T1=0.0008, lr0=0.008, lr1=0.001)
        X=_build_c15(_norm(seeds))
        vv=_maxcos(X)
        if vv<sym_bv:
            sym_bv,sym_best=vv,X.copy()
    # --- signed cyclic search (2 restarts, cheaper) ---
    for r in range(2):
        if time.time()>deadline-55:
            break
        rg=np.random.default_rng(300+r)
        sa=_norm(rg.normal(size=(1,15)))[0]
        sb=_norm(rg.normal(size=(1,15)))[0]
        sc=_norm(rg.normal(size=(1,15)))[0]
        sa,sb,sc=_anneal_signed(sa,sb,sc,steps=1000,T0=0.025,T1=0.003,lr0=0.02,lr1=0.003)
        X=_build_signed(sa,sb,sc)
        vv=_maxcos(X)
        if vv<sym_bv:
            sym_bv,sym_best=vv,X.copy()
    # --- antipodal 38-pair search (2 restarts) ---
    for r in range(2):
        if time.time()>deadline-45:
            break
        rg=np.random.default_rng(500+r)
        s38=_norm(rg.normal(size=(38,15)))
        s38=_anneal_anti(s38,steps=1000,T0=0.025,T1=0.003,lr0=0.02,lr1=0.003)
        s38=_anneal_anti(s38,steps=800,T0=0.004,T1=0.0008,lr0=0.008,lr1=0.001)
        X=_build_anti(_norm(s38))
        vv=_maxcos(X)
        if vv<sym_bv:
            sym_bv,sym_best=vv,X.copy()
    # global-rotation reduction for antipodal: already invariant; skip explicit

    # choose release start: best symmetric found
    rel=None
    if sym_best is not None:
        rel=sym_best.copy()
    else:
        rel=best.copy()
    # unconstrained release with annealed full optimization
    if time.time()<deadline-30:
        # two-stage release
        rel=_anneal_full(rel, steps=1200, T0=0.008, T1=0.001, lr0=0.01, lr1=0.001)
        if time.time()<deadline-25:
            rel=_anneal_full(rel, steps=800, T0=0.0015, T1=0.0004, lr0=0.003, lr1=0.0005)
    rv=_maxcos(rel)
    # light LP polish on released symmetric code
    if time.time()<deadline-12:
        t_end=min(deadline-2.0, time.time()+10.0)
        try:
            rel=_light_lp(rel, t_end)
        except Exception:
            pass
        rv=_maxcos(rel)
    # final selection vs record best
    if rv<bv:
        best, bv = rel, rv
    # also try quick polish of original best if symmetric worse (cheap hill climb)
    if bv>=bv0-1e-12 and time.time()<deadline-8:
        # tiny full anneal starting from best0 to ensure no regression and maybe gain
        tmp=_anneal_full(best0, steps=400, T0=0.002, T1=0.0005, lr0=0.002, lr1=0.0005)
        vv=_maxcos(tmp)
        if vv<bv:
            best,bv=tmp,vv
    if _maxcos(best)>=bv0+1e-12 or not np.all(np.isfinite(best)):
        best=best0.copy()
    best=_norm(best)
    return best
