import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _sig(s):
    # stable sigmoid
    out = np.empty_like(s, dtype=float)
    pos = s >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-s[pos]))
    e = np.exp(s[~pos])
    out[~pos] = e / (1.0 + e)
    return out

# ---------- fixed orbit generators (normalized) ----------
def _orbit_12():
    X = np.zeros((12, 6))
    k = 0
    for p in range(6):
        for sgn in (1.0, -1.0):
            v = np.zeros(6); v[p] = sgn
            X[k] = v; k += 1
    return X

def _orbit_60():
    pts = []
    for p in range(6):
        for q in range(p+1, 6):
            for s1 in (1.0, -1.0):
                for s2 in (1.0, -1.0):
                    v = np.zeros(6)
                    v[p] = s1 / np.sqrt(2.0); v[q] = s2 / np.sqrt(2.0)
                    pts.append(v)
    return np.array(pts)

def _orbit_64():
    pts = []
    for mask in range(64):
        v = np.ones(6)
        for j in range(6):
            if (mask >> j) & 1:
                v[j] = -1.0
        v = v / np.sqrt(6.0)
        pts.append(v)
    return np.array(pts)

def _orbit_160():
    pts = []
    for a in range(6):
        for b in range(a+1, 6):
            for c in range(b+1, 6):
                for s1 in (1.0, -1.0):
                    for s2 in (1.0, -1.0):
                        for s3 in (1.0, -1.0):
                            v = np.zeros(6)
                            v[a] = s1/np.sqrt(3.0); v[b]=s2/np.sqrt(3.0); v[c]=s3/np.sqrt(3.0)
                            pts.append(v)
    return np.array(pts)

def _pattern_120():
    posA=[]; posB=[]; sA=[]; sB=[]
    for p in range(6):
        for q in range(p+1,6):
            # two orders: (p gets cos, q gets sin) and swapped
            for order in (0,1):
                for sa in (1.0,-1.0):
                    for sb in (1.0,-1.0):
                        if order==0:
                            posA.append(p); posB.append(q)
                        else:
                            posA.append(q); posB.append(p)
                        sA.append(sa); sB.append(sb)
    return (np.array(posA,dtype=int), np.array(posB,dtype=int),
            np.array(sA,dtype=float), np.array(sB,dtype=float))

_PAT120 = None
def _get_pat120():
    global _PAT120
    if _PAT120 is None:
        _PAT120 = _pattern_120()
    return _PAT120

def _build_120(phi):
    posA,posB,sA,sB = _get_pat120()
    c = np.cos(phi); s = np.sin(phi)
    X = np.zeros((120,6))
    X[np.arange(120),posA]=sA*c
    X[np.arange(120),posB]=sB*s
    return X

# ---------- symmetric low-dim optimisation ----------
def _sym_opt(Kp, fixed_X, m_free, beta, maxiter, rng, s_init=None, Y_init=None):
    # Kp = number of 120-orbits (0,1,2), fixed_X (nf,6) or None, m_free free points
    import numpy as _np
    posA,posB,sA,sB = _get_pat120()
    nf = 0 if fixed_X is None else fixed_X.shape[0]
    n_tot = Kp*120 + nf + m_free
    # init
    if s_init is None:
        s_init = rng.uniform(-2.0, 2.0, size=Kp)
    if Y_init is None:
        Y_init = rng.standard_normal((m_free,6)) if m_free>0 else np.zeros((0,6))
    v0 = np.concatenate([s_init, Y_init.ravel()]) if m_free>0 else s_init.copy()
    # bounds: s in [-6,6], Y free unbounded (use None -> L-BFGS-B needs finite? use large)
    bounds = [(-6.0,6.0)]*Kp + [(-5.0,5.0)]*(m_free*6)
    PI4 = np.pi/4.0
    def fg(v):
        s = v[:Kp] if Kp>0 else np.zeros(0)
        sig = _sig(s) if Kp>0 else s
        phis = PI4*sig
        cs = np.cos(phis) if Kp>0 else np.zeros(0)
        sns = np.sin(phis) if Kp>0 else np.zeros(0)
        # build full X
        blocks=[]
        for k in range(Kp):
            Xk = np.zeros((120,6))
            Xk[np.arange(120),posA]=sA*cs[k]
            Xk[np.arange(120),posB]=sB*sns[k]
            blocks.append(Xk)
        if nf>0:
            blocks.append(fixed_X)
        if m_free>0:
            Yf = v[Kp:].reshape(m_free,6)
            nrm = np.sqrt(np.sum(Yf*Yf,axis=1,keepdims=True))
            nrm = np.maximum(nrm,1e-12)
            Xf = Yf/nrm
            blocks.append(Xf)
        else:
            Yf=None; Xf=None
        X = np.concatenate(blocks,axis=0) if len(blocks)>0 else np.zeros((0,6))
        n = X.shape[0]
        G = X @ X.T
        np.fill_diagonal(G,-2.0)
        M = float(np.max(G))
        D = beta*(G-M)
        np.clip(D,-35.0,0.0,out=D)
        E = np.exp(D)
        np.fill_diagonal(E,0.0)
        S = float(np.sum(E))
        if S<=0 or not np.isfinite(S):
            S=1e-300
        loss = M + np.log(S)/beta
        W = E/S
        GX = 2.0*(W @ X)
        # grads
        grad = np.zeros_like(v)
        # phi grads
        if Kp>0:
            # dphi/ds
            dphi = PI4*sig*(1.0-sig)
            # for each k, sum over its rows
            for k in range(Kp):
                Gb = GX[k*120:(k+1)*120,:]  # (120,6)
                # dX/dphi: at posA: -sA*sin, at posB: sB*cos
                dphi_dot = Gb[np.arange(120),posA]*(-sA*sns[k]) + Gb[np.arange(120),posB]*(sB*cs[k])
                grad[k] = float(np.sum(dphi_dot))*dphi[k]
        if m_free>0:
            Gf = GX[Kp*120+nf: , : ]  # (m,6)
            # chain through normalisation
            # Xf already computed
            dot = np.sum(Gf*Xf,axis=1,keepdims=True)
            nrmf = np.sqrt(np.sum(Yf*Yf,axis=1,keepdims=True))
            nrmf = np.maximum(nrmf,1e-12)
            GY = (Gf - dot*Xf)/nrmf
            grad[Kp:] = GY.ravel()
        return loss, grad
    try:
        res = minimize(fg, v0, jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter":maxiter,"ftol":1e-20,"gtol":1e-9,
                                "maxfun":maxiter*5,"maxcor":10})
        v = res.x
    except Exception:
        v = v0
    # rebuild X
    s = v[:Kp] if Kp>0 else np.zeros(0)
    sig = _sig(s) if Kp>0 else s
    phis = PI4*sig
    blocks=[]
    for k in range(Kp):
        blocks.append(_build_120(float(phis[k])))
    if nf>0:
        blocks.append(fixed_X)
    if m_free>0:
        Yf = v[Kp:].reshape(m_free,6)
        nrm = np.linalg.norm(Yf,axis=1,keepdims=True); nrm=np.maximum(nrm,1e-12)
        blocks.append(Yf/nrm)
    Xfull = np.concatenate(blocks,axis=0)
    # safety: if n_tot !=250 due to combo, caller ensures 250
    return Xfull

def _smooth_opt(X0, beta, maxiter, maxcor=10):
    n,d = X0.shape
    Y0 = X0.copy()
    def fg(Yflat):
        Y = Yflat.reshape(n,d)
        nrm = np.linalg.norm(Y,axis=1,keepdims=True)
        nrm = np.maximum(nrm,1e-12)
        X = Y/nrm
        G = X @ X.T
        np.fill_diagonal(G,-2.0)
        M = float(np.max(G))
        D = beta*(G-M)
        np.clip(D,-35.0,0.0,out=D)
        E = np.empty_like(D); np.exp(D,out=E)
        np.fill_diagonal(E,0.0)
        S = float(np.sum(E))
        if S<=0 or not np.isfinite(S):
            S=1e-300
        loss = M + np.log(S)/beta
        W = E/S
        GX = 2.0*(W @ X)
        dot = np.sum(GX*X,axis=1,keepdims=True)
        GY = (GX-dot*X)/nrm
        return loss, GY.ravel()
    try:
        res = minimize(fg, Y0.ravel(), jac=True, method="L-BFGS-B",
                       options={"maxiter":maxiter,"ftol":1e-20,"gtol":1e-12,
                                "maxfun":maxiter*5,"maxcor":maxcor})
        Y = res.x.reshape(n,d)
    except Exception:
        Y = Y0
    nrm = np.linalg.norm(Y,axis=1,keepdims=True); nrm=np.maximum(nrm,1e-12)
    return Y/nrm

def _lp_polish(x, deadline, k_top=4000, radius_init=2e-3):
    x = x/np.linalg.norm(x,axis=1,keepdims=True)
    n,d = x.shape
    best = _max_cos(x)
    radius = radius_init
    iu0,iu1 = np.triu_indices(n,1)
    ar = np.arange(d)
    while time.time() < deadline-1.0 and radius>1e-14:
        Gall = x @ x.T
        c_all = Gall[iu0,iu1]
        m = float(np.max(c_all))
        sel = c_all > m - max(20.0*radius, 6e-7)
        cnt = int(np.count_nonzero(sel))
        if cnt==0:
            break
        if cnt>k_top:
            idx = np.argpartition(-c_all,k_top)[:k_top]
            I=iu0[idx]; J=iu1[idx]; c_sel=c_all[idx]
        else:
            I=iu0[sel]; J=iu1[sel]; c_sel=c_all[sel]
        p=I.shape[0]
        nv=n*d+1
        try:
            cols=np.concatenate([I[:,None]*d+ar, J[:,None]*d+ar, np.full((p,1),n*d)],axis=1).ravel()
            rows=np.repeat(np.arange(p),2*d+1)
            vals=np.concatenate([x[J],x[I],-np.ones((p,1))],axis=1).ravel()
            A_ub=coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
            obj=np.zeros(nv); obj[-1]=1.0
            bounds=[(-radius,radius)]*(n*d)+[(None,None)]
            res=linprog(obj,A_ub=A_ub,b_ub=-c_sel,A_eq=A_eq,b_eq=np.zeros(n),bounds=bounds,method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        y=x+res.x[:-1].reshape(n,d)
        y=y/np.linalg.norm(y,axis=1,keepdims=True)
        if not np.all(np.isfinite(y)):
            radius*=0.5; continue
        v=_max_cos(y)
        if v<best-1e-12:
            x,best=y,v
            radius=min(radius*1.5,0.012)
        else:
            radius*=0.5
    return x

def solve():
    t0=time.time()
    HARD=t0+172.0
    SYM_END=t0+62.0
    rng=np.random.default_rng(12345)
    n,d=250,6
    cands=[]
    for fn in ("best.npy","record.npy"):
        try:
            a=np.asarray(np.load(fn),dtype=float)
            if a.shape==(n,d):
                a=np.nan_to_num(a)
                a/=np.linalg.norm(a,axis=1,keepdims=True)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        tmp=rng.standard_normal((n,d)); tmp/=np.linalg.norm(tmp,axis=1,keepdims=True)
        cands.append(tmp)
    incumbent=min(cands,key=_max_cos).copy()
    inc_val=_max_cos(incumbent)
    # fixed blocks
    F12=_orbit_12(); F60=_orbit_60(); F64=_orbit_64(); F160=_orbit_160()
    # symmetric combos totalling 250
    combos=[]
    combos.append({"Kp":2,"fixed":None,"m":10})  # 240+10
    combos.append({"Kp":1,"fixed":np.concatenate([F64,F60],axis=0),"m":6})  #120+124+6
    combos.append({"Kp":0,"fixed":np.concatenate([F160,F60,F12],axis=0),"m":18}) #232+18
    combos.append({"Kp":0,"fixed":np.concatenate([F160,F64,F12],axis=0),"m":14}) #236+14
    combos.append({"Kp":1,"fixed":F64,"m":66})  #120+64+66 feasible
    combos.append({"Kp":0,"fixed":np.concatenate([F12,F160],axis=0),"m":78}) #172+78
    sym_best=None; sym_val=np.inf
    sym_pool=[]
    # quick enumeration: ~2 inits per combo with short opt
    for ci,cb in enumerate(combos):
        if time.time()>SYM_END-2:
            break
        Kp=cb["Kp"]; fixed=cb["fixed"]; m=cb["m"]
        for init in range(2):
            if time.time()>SYM_END-2:
                break
            try:
                Xsym=_sym_opt(Kp,fixed,m,beta=2000.0,maxiter=70,rng=rng)
            except Exception:
                continue
            if Xsym.shape!=(250,6):
                continue
            # dedup safety: if duplicates (max~1) skip
            try:
                v=_max_cos(Xsym)
            except Exception:
                continue
            if not np.isfinite(v):
                continue
            sym_pool.append((v,Xsym.copy()))
            if v<sym_val:
                sym_val=v; sym_best=Xsym.copy()
            if time.time()>SYM_END-1:
                break
    # candidate pool for breaking symmetry: incumbent + best 2 sym
    pool=[(inc_val,incumbent.copy())]
    if sym_pool:
        sym_pool.sort(key=lambda t:t[0])
        for v,Xs in sym_pool[:2]:
            pool.append((v,Xs.copy()))
    # break symmetry: full smooth polish of each (short), pick best, then deep polish
    polished=[]
    for v0,X0 in pool:
        if time.time()>HARD-45:
            break
        try:
            # Adam-like break: small noise to destroy exact symmetry
            Xn=X0+1e-4*rng.standard_normal((n,d))
            Xn/=np.linalg.norm(Xn,axis=1,keepdims=True)
            Xn=_smooth_opt(Xn,beta=3500.0,maxiter=60)
            if time.time()>HARD-40:
                v=_max_cos(Xn)
                polished.append((v,Xn))
                continue
            Xn=_smooth_opt(Xn,beta=12000.0,maxiter=60)
            v=_max_cos(Xn)
            polished.append((v,Xn))
        except Exception:
            polished.append((v0,X0))
    if not polished:
        best=incumbent
    else:
        polished.sort(key=lambda t:t[0])
        best=polished[0][1].copy()
        best_val=polished[0][0]
        # extra basin hops from best (a few perturbations + smooth) if time
        trial=0
        while time.time()<HARD-28 and trial<6:
            trial+=1
            try:
                sig=[0.004,0.008,0.002,0.012,0.006,0.01][trial%6]
                Xp=best+sig*rng.standard_normal((n,d))
                Xp/=np.linalg.norm(Xp,axis=1,keepdims=True)
                Xc=_smooth_opt(Xp,beta=8000.0,maxiter=50)
                vc=_max_cos(Xc)
                if vc<best_val:
                    best,best_val=Xc,vc
            except Exception:
                pass
        # final LP polish on best
        try:
            if time.time()<HARD-4:
                best=_lp_polish(best,min(HARD-1.0,time.time()+22.0),k_top=4000,radius_init=2e-3)
        except Exception:
            pass
    best=np.nan_to_num(best)
    best/=np.linalg.norm(best,axis=1,keepdims=True)
    return best
