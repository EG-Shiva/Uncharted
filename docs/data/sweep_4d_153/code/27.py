import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _trig(M,a,b):
    k = np.arange(M)
    t1 = 2*np.pi*a*k/M
    t2 = 2*np.pi*b*k/M
    return np.cos(t1), np.sin(t1), np.cos(t2), np.sin(t2)

def expand_orbits(R, M, a, b, c1=None,s1=None,c2=None,s2=None):
    r = R.shape[0]
    if c1 is None:
        c1,s1,c2,s2 = _trig(M,a,b)
    out = np.empty((r, M, 4), dtype=float)
    r0 = R[:,0,None]; r1=R[:,1,None]; r2=R[:,2,None]; r3=R[:,3,None]
    out[:,:,0] = r0*c1[None,:] - r1*s1[None,:]
    out[:,:,1] = r0*s1[None,:] + r1*c1[None,:]
    out[:,:,2] = r2*c2[None,:] - r3*s2[None,:]
    out[:,:,3] = r2*s2[None,:] + r3*c2[None,:]
    return out.reshape(r*M,4), (c1,s1,c2,s2)

def sym_relax(Rinit, M,a,b, iters, K0,K1, lr0,lr1, mu):
    R = Rinit.copy()
    R /= np.linalg.norm(R,axis=1,keepdims=True)
    r = R.shape[0]
    N = r*M
    c1,s1,c2,s2 = _trig(M,a,b)
    v = np.zeros_like(R)
    for t in range(iters):
        frac = t/max(1,iters-1)
        K = K0*(K1/K0)**frac
        lr = lr0*(lr1/lr0)**frac
        # expand
        Rout = np.empty((r,M,4))
        Rout[:,:,0]=R[:,0,None]*c1[None,:]-R[:,1,None]*s1[None,:]
        Rout[:,:,1]=R[:,0,None]*s1[None,:]+R[:,1,None]*c1[None,:]
        Rout[:,:,2]=R[:,2,None]*c2[None,:]-R[:,3,None]*s2[None,:]
        Rout[:,:,3]=R[:,2,None]*s2[None,:]+R[:,3,None]*c2[None,:]
        X = Rout.reshape(N,4)
        G = X @ X.T
        np.fill_diagonal(G,-2.0)
        m = G.max()
        # avoid overflow
        W = np.exp(K*(G-m))
        np.fill_diagonal(W,0.0)
        GX = W @ X  # (N,4)
        GXr = GX.reshape(r,M,4)
        GR = np.empty_like(R)
        GR[:,0]=(GXr[:,:,0]*c1[None,:]+GXr[:,:,1]*s1[None,:]).sum(axis=1)
        GR[:,1]=(-GXr[:,:,0]*s1[None,:]+GXr[:,:,1]*c1[None,:]).sum(axis=1)
        GR[:,2]=(GXr[:,:,2]*c2[None,:]+GXr[:,:,3]*s2[None,:]).sum(axis=1)
        GR[:,3]=(-GXr[:,:,2]*s2[None,:]+GXr[:,:,3]*c2[None,:]).sum(axis=1)
        tot = W.sum(axis=1).reshape(r,M).sum(axis=1,keepdims=True)+1e-18
        GR = GR/tot
        dot = (GR*R).sum(axis=1,keepdims=True)
        Gt = GR - dot*R
        # clip large steps
        nrm = np.linalg.norm(Gt,axis=1,keepdims=True)+1e-18
        cap = np.minimum(1.0, 3.0/nrm)
        # cap keeps direction but limits magnitude to 3
        Gt = Gt*cap
        v = mu*v + Gt
        denom = (1.0-mu**(t+1)) if mu>0 else 1.0
        R = R - lr*(v/denom)
        R /= np.linalg.norm(R,axis=1,keepdims=True)+1e-18
    return R

def smooth_relax(x, iters, k0,k1, lr0,lr1, mu, noise, rng):
    n = x.shape[0]
    v = np.zeros_like(x)
    for t in range(iters):
        frac = t/max(1,iters-1)
        k = k0*(k1/k0)**frac
        lr = lr0*(lr1/lr0)**frac
        G = x @ x.T
        np.fill_diagonal(G,-2.0)
        moff = G.max()
        W = np.exp(k*(G-moff))
        np.fill_diagonal(W,0.0)
        rs = W.sum(axis=1,keepdims=True)+1e-18
        grad = (W @ x)/rs
        dot = (grad*x).sum(axis=1,keepdims=True)
        gt = grad - dot*x
        v = mu*v + gt
        denom = (1.0-mu**(t+1)) if mu>0 else 1.0
        x = x - lr*(v/denom)
        if noise>0 and frac<0.6:
            amp = noise*(1.0-frac/0.6)
            pert = rng.standard_normal(x.shape)
            pert -= (pert*x).sum(axis=1,keepdims=True)*x
            x = x + amp*pert
        x /= np.linalg.norm(x,axis=1,keepdims=True)+1e-18
    return x

def lp_polish(x, seconds):
    x = x/np.linalg.norm(x,axis=1,keepdims=True)
    n,d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n,1)
    while time.time()-start < seconds and radius>1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20*radius,1e-9)
        I,J = iu[0][sel], iu[1][sel]
        p,nv = len(I), n*d+1
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
        if vv < best:
            x,best,radius = y,vv,min(radius*1.5,0.05)
        else:
            radius*=0.5
    return x

def union_relax_51(Ginit, phi_init, iters, K0,K1,lr0,lr1,mu):
    # M=51,a=1,b=3, 2 generic (51) + 3 special z1=0 (17)
    G = Ginit.copy(); phi = phi_init.copy()
    G/=np.linalg.norm(G,axis=1,keepdims=True)
    M=51; a=1; b=3
    c1,s1,c2,s2=_trig(M,a,b)
    vG=np.zeros_like(G); vphi=np.zeros_like(phi)
    for t in range(iters):
        frac=t/max(1,iters-1)
        K=K0*(K1/K0)**frac; lr=lr0*(lr1/lr0)**frac
        # generic expand
        rg=G.shape[0]
       Rout=np.empty((rg,M,4))
        Rout[:,:,0]=G[:,0,None]*c1[None,:]-G[:,1,None]*s1[None,:]
        Rout[:,:,1]=G[:,0,None]*s1[None,:]+G[:,1,None]*c1[None,:]
        Rout[:,:,2]=G[:,2,None]*c2[None,:]-G[:,3,None]*s2[None,:]
        Rout[:,:,3]=G[:,2,None]*s2[None,:]+G[:,3,None]*c2[None,:]
        Xg=Rout.reshape(rg*M,4)
        # special expand: 3x17
        Ms=17
        th = phi[:,None]+2*np.pi*np.arange(Ms)[None,:]/Ms
        Xs=np.zeros((len(phi)*Ms,4))
        Xs[:,2]=np.cos(th).ravel(); Xs[:,3]=np.sin(th).ravel()
        X=np.concatenate([Xg,Xs],axis=0)
        N=X.shape[0]
        Gr=X@X.T; np.fill_diagonal(Gr,-2.0); m=Gr.max()
        W=np.exp(K*(Gr-m)); np.fill_diagonal(W,0.0)
        GX=W@X
        tot_all=W.sum()
        # generic grad
        GXg=GX[:rg*M].reshape(rg,M,4)
        GR=np.empty_like(G)
        GR[:,0]=(GXg[:,:,0]*c1[None,:]+GXg[:,:,1]*s1[None,:]).sum(axis=1)
        GR[:,1]=(-GXg[:,:,0]*s1[None,:]+GXg[:,:,1]*c1[None,:]).sum(axis=1)
        GR[:,2]=(GXg[:,:,2]*c2[None,:]+GXg[:,:,3]*s2[None,:]).sum(axis=1)
        GR[:,3]=(-GXg[:,:,2]*s2[None,:]+GXg[:,:,3]*c2[None,:]).sum(axis=1)
        # scale by row sums
        totg=W.sum(axis=1)[:rg*M].reshape(rg,M).sum(axis=1,keepdims=True)+1e-18
        GR=GR/totg
        dot=(GR*G).sum(axis=1,keepdims=True); Gt=GR-dot*G
        # spec grad w.r.t phi
        GXs=GX[rg*M:].reshape(len(phi),Ms,4)
        # Xs reshaped
        Xsr=Xs.reshape(len(phi),Ms,4)
        # dp/dphi = [0,0,-sin,cos]
        dphi = (-GXs[:,:,2]*Xsr[:,:,3]+GXs[:,:,3]*Xsr[:,:,2]).sum(axis=1)
        # scale by sums
        tots=W.sum(axis=1)[rg*M:].reshape(len(phi),Ms).sum(axis=1)+1e-18
        dphi=dphi/tots
        # update
        vG=mu*vG+Gt
        vphi=mu*vphi+dphi
        denom=(1.0-mu**(t+1)) if mu>0 else 1.0
        G=G-lr*(vG/denom); G/=np.linalg.norm(G,axis=1,keepdims=True)+1e-18
        phi=phi-lr*(vphi/denom)*2.0
    # final expand
    rg=G.shape[0]
    Rout=np.empty((rg,M,4))
    Rout[:,:,0]=G[:,0,None]*c1[None,:]-G[:,1,None]*s1[None,:]
    Rout[:,:,1]=G[:,0,None]*s1[None,:]+G[:,1,None]*c1[None,:]
    Rout[:,:,2]=G[:,2,None]*c2[None,:]-G[:,3,None]*s2[None,:]
    Rout[:,:,3]=G[:,2,None]*s2[None,:]+G[:,3,None]*c2[None,:]
    Xg=Rout.reshape(rg*M,4)
    th = phi[:,None]+2*np.pi*np.arange(Ms)[None,:]/Ms
    Xs=np.zeros((len(phi)*Ms,4)); Xs[:,2]=np.cos(th).ravel(); Xs[:,3]=np.sin(th).ravel()
    return np.concatenate([Xg,Xs],axis=0)

def solve():
    t0=time.time()
    BUDGET=172.0
    rng=np.random.default_rng(0)
    n,d=153,4
    try:
        best0=np.load("best.npy")
    except Exception:
        try: best0=np.load("record.npy")
        except Exception: best0=rng.standard_normal((n,d))
    best0=np.asarray(best0,dtype=float)
    if best0.shape!=(n,d):
        best0=rng.standard_normal((n,d))
    best0/=np.linalg.norm(best0,axis=1,keepdims=True)
    best_val=max_cos(best0)
    incumbent=best0.copy(); inc_val=best_val

    configs=[
        (17,1,1,9),(17,1,2,9),(17,1,3,9),(17,1,4,9),(17,1,0,9),
        (51,1,1,3),(51,1,2,3),(51,1,5,3),
        (9,1,1,17),(9,1,2,17),
        (153,1,2,1),(153,1,7,1),
    ]
    # cheap screening
    screened=[]
    # time limit for screening: 75s
    SCREEN_END=t0+75
    for (M,a,b,r) in configs:
        if time.time()>SCREEN_END: break
        # number of inits depends on size
        ninit = 10 if r>=9 else 12
        if M==153: ninit=12
        for si in range(ninit):
            if time.time()>SCREEN_END: break
            rs=rng.standard_normal((r,d)); rs/=np.linalg.norm(rs,axis=1,keepdims=True)
            # cheap relax
            Ropt=sym_relax(rs,M,a,b,iters=350,K0=25,K1=140,lr0=0.08,lr1=0.008,mu=0.75)
            X,_=expand_orbits(Ropt,M,a,b)
            if X.shape[0]!=153: continue
            v=max_cos(X)
            screened.append((v,X.copy(),(M,a,b)))
            if v<inc_val:
                inc_val=v; incumbent=X.copy()
    # union special: 2x51 + 3x17
    if time.time()<SCREEN_END:
        for si in range(6):
            if time.time()>SCREEN_END: break
            G0=rng.standard_normal((2,4)); G0/=np.linalg.norm(G0,axis=1,keepdims=True)
            phi0=rng.uniform(0,2*np.pi,size=(3,))
            Xu=union_relax_51(G0,phi0,iters=350,K0=25,K1=140,lr0=0.08,lr1=0.008,mu=0.75)
            v=max_cos(Xu)
            screened.append((v,Xu.copy(),('union',)))
            if v<inc_val:
                inc_val=v; incumbent=Xu.copy()
    # pick top few for full relax
    screened.sort(key=lambda t:t[0])
    top = screened[:4] if len(screened)>=4 else screened
    FULL_END=t0+115
    cands=[(inc_val,incumbent.copy())]
    for (v,X0,cfg) in top:
        if time.time()>FULL_END: break
        # fully relax symmetrically a bit more if it came from symmetric cfg
        # Instead break symmetry slightly and relax unconstrained
        for amp,seed in [(0.0,1),(0.02,2)]:
            if time.time()>FULL_END: break
            if amp==0.0 and not isinstance(cfg,tuple):
                continue
            # if amp==0 and cfg is symmetric, do longer sym relax from its reps?
            # For simplicity do unconstrained relax
            Y=X0.copy()
            if amp>0:
                Y=Y+amp*rng.standard_normal(Y.shape)
                Y/=np.linalg.norm(Y,axis=1,keepdims=True)
            Y=smooth_relax(Y,iters=900,k0=30,k1=250,lr0=0.06,lr1=0.004,mu=0.85,noise=0.0,rng=rng)
            vv=max_cos(Y)
            cands.append((vv,Y.copy()))
            if vv<inc_val:
                inc_val=vv; incumbent=Y.copy()
    # symmetry-breaking with larger amplitude as assigned 0.05-0.15
    BREAK_END=t0+140
    amps=[0.05,0.10,0.15,0.07]
    bi=0
    while time.time()<BREAK_END and bi<6:
        base = incumbent
        amp = amps[bi%len(amps)]
        Y=base+amp*rng.standard_normal(base.shape)
        Y/=np.linalg.norm(Y,axis=1,keepdims=True)
        # annealed repulsion
        Y=smooth_relax(Y,iters=700,k0=25,k1=300,lr0=0.05,lr1=0.003,mu=0.85,noise=0.01 if bi%2==0 else 0.0,rng=rng)
        vv=max_cos(Y)
        if vv<inc_val:
            inc_val=vv; incumbent=Y.copy()
        cands.append((vv,Y))
        bi+=1
    # choose best candidate (including original best0) for final polish
    cands.append((best_val,best0))
    cands.sort(key=lambda t:t[0])
    final=cands[0][1].copy()
    # use remaining time for LP polish (at least 15s)
    remain = BUDGET-(time.time()-t0)
    pol_time = max(10.0,min(35.0,remain-2.0))
    if pol_time>3:
        final=lp_polish(final,pol_time)
        fv=max_cos(final)
        # keep best among polished and unpolished
        if fv> cands[0][0]:
            # polishing sometimes numerically worsens? keep original
            # re-polish incumbent alternative
            alt=lp_polish(incumbent,min(8.0,pol_time/3))
            if max_cos(alt)<fv:
                final=alt
    final/=np.linalg.norm(final,axis=1,keepdims=True)
    # safety: if NaN or worse than best0 by margin, return best0 polished briefly
    if not np.all(np.isfinite(final)) or final.shape!=(n,d):
        return best0
    # ensure we never return worse than loaded best (to protect scoreboard)
    if max_cos(final) > best_val+1e-12:
        # return the better one
        return best0
    return final
