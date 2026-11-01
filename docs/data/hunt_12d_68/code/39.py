import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _per_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _full_fg(n, d, IU, beta):
    def fg(yf):
        Y = yf.reshape(n, d)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = Y / nr
        g = (X @ X.T)[IU]
        m = float(g.max())
        e = np.exp(beta * (g - m))
        S = float(e.sum()) + 1e-300
        F = m + np.log(S) / beta
        w = e / S
        P = np.zeros((n, n))
        P[IU] = w
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return F, GY.ravel()
    return fg

def _run_full(x0, beta, maxiter):
    n, d = x0.shape
    IU = np.triu_indices(n, 1)
    fg = _full_fg(n, d, IU, beta)
    y0 = (x0 / np.linalg.norm(x0, axis=1, keepdims=True)).ravel().copy()
    try:
        r = minimize(fg, y0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-8, 'maxfun': maxiter*3, 'maxcor': 10})
        Y = r.x.reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    except Exception:
        return x0

def _lp_polish(x, deadline, r0=8e-4, cap=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline - 0.3 and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20.0*radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        csel = c[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > cap:
            od = np.argsort(-csel)[:cap]
            I, J, csel = I[od], J[od], csel[od]
            p = cap
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)],1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
        try:
            A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d), np.arange(n*d))), shape=(n,nv)).tocsr()
            obj = np.zeros(nv); obj[-1]=1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        step = res.x[:-1].reshape(n,d)
        improved=False
        for sc in (1.0,0.3,0.1):
            yy = x + step*sc
            yy /= np.linalg.norm(yy,axis=1,keepdims=True)
            vv = _max_cos(yy)
            if vv < best - 1e-15:
                x,best = yy,vv
                radius = min(radius*1.5,0.02)
                improved=True
                break
        if not improved:
            radius *= 0.5
    return x

def _build_M(D):
    d=len(D); M=np.ones((d,d))
    for k in range(1,d):
        prev=M[k-1]; cur=np.empty(d)
        for i in range(d):
            cur[i]=D[(i-1)%d]*prev[(i-1)%d]
        M[k]=cur
    return M

def _expand_raw(y_raw, M):
    d=12
    S=[]; Sn=[]
    for i in range(5):
        v=y_raw[i*12:(i+1)*12]
        n=max(float(np.linalg.norm(v)),1e-12)
        Sn.append(n); S.append(v/n)
    Araw=[y_raw[60:64], y_raw[64:68]]
    B=[]; Bn=[]
    Full=[]
    for v in Araw:
        n=max(float(np.linalg.norm(v)),1e-12)
        Bn.append(n); b=v/n; B.append(b)
        Full.append(np.tile(b,3)/np.sqrt(3.0))
    X=np.empty((68,d))
    ar=np.arange(d)
    for i in range(5):
        s=S[i]
        for k in range(12):
            idx=(ar-k)%d
            X[i*12+k]=M[k]*s[idx]
    for j in range(2):
        f=Full[j]
        for k in range(4):
            idx=(ar-k)%d
            X[60+j*4+k]=M[k]*f[idx]
    return X, S, Sn, B, Bn

def _seed_fg_factory(M, beta):
    ar=np.arange(12)
    def fg(y_raw):
        y_raw=np.asarray(y_raw,float)
        X,S,Sn,B,Bn=_expand_raw(y_raw,M)
        n=68
        IU=np.triu_indices(n,1)
        g=(X@X.T)[IU]
        m=float(g.max())
        e=np.exp(beta*(g-m))
        Ss=float(e.sum())+1e-300
        F=m+np.log(Ss)/beta
        w=e/Ss
        P=np.zeros((n,n)); P[IU]=w; P=P+P.T
        GX=P@X
        grad=np.empty_like(y_raw)
        for i in range(5):
            s=S[i]
            h=np.zeros(12)
            base=i*12
            for k in range(12):
                tmp=M[k]*GX[base+k]
                h+=np.roll(tmp,-k)
            hn=float(h@s)
            gv=(h-hn*s)/Sn[i]
            grad[i*12:(i+1)*12]=gv
        for j in range(2):
            b=B[j]
            hfull=np.zeros(12)
            base=60+j*4
            for k in range(4):
                tmp=M[k]*GX[base+k]
                hfull+=np.roll(tmp,-k)
            hb=(hfull[0:4]+hfull[4:8]+hfull[8:12])/np.sqrt(3.0)
            hn=float(hb@b)
            ga=(hb-hn*b)/Bn[j]
            grad[60+j*4:64+j*4]=ga
        return F, grad
    return fg

def _run_seed(y0, M, beta, maxiter):
    fg=_seed_fg_factory(M,beta)
    try:
        r=minimize(fg,y0,jac=True,method='L-BFGS-B',
                   options={'maxiter':maxiter,'ftol':1e-16,'gtol':1e-8,'maxfun':maxiter*3,'maxcor':10})
        return r.x
    except Exception:
        return y0

def _hole_for_rest(Xr, rng, ncand):
    d=Xr.shape[1]
    cand=rng.standard_normal((ncand,d))
    cand/=np.linalg.norm(cand,axis=1,keepdims=True)
    sc=(cand@Xr.T).max(axis=1)
    return cand[int(np.argmin(sc))]

def _hole_sweep(x, rng, k, ncand):
    n=x.shape[0]
    cur=_max_cos(x)
    pm=_per_max(x)
    order=np.argsort(-pm)[:k]
    for idx in order:
        mask=np.ones(n,bool); mask[idx]=False
        Xr=x[mask]
        G=Xr@Xr.T; np.fill_diagonal(G,-2.0)
        rest=float(G.max())
        hole=_hole_for_rest(Xr,rng,ncand)
        hmax=float((Xr@hole).max())
        nm=hmax if hmax>rest else rest
        if nm < cur-1e-13:
            x[idx]=hole; cur=nm
    return x,cur

def solve():
    t0=time.time(); deadline=t0+172.0
    rng=np.random.default_rng(777)
    cands=[]
    try:
        xb=np.load("best.npy").astype(float); cands.append(xb)
    except Exception: xb=None
    try:
        xr=np.load("record.npy").astype(float); cands.append(xr)
    except Exception: xr=None
    if len(cands)==0:
        x0=rng.standard_normal((68,12)); x0/=np.linalg.norm(x0,axis=1,keepdims=True)
        best_val=_max_cos(x0); x_best=x0
    else:
        best_val=1e9; x_best=None
        for c in cands:
            c=np.asarray(c,float); c/=np.linalg.norm(c,axis=1,keepdims=True)
            v=_max_cos(c)
            if v<best_val: best_val=v; x_best=c.copy()
    x_best=x_best/np.linalg.norm(x_best,axis=1,keepdims=True)
    # ---- symmetry-constrained search ----
    d0_list=[[1,1,1,1],[-1,-1,1,1],[-1,1,-1,1],[-1,1,1,-1],[1,-1,1,-1]]
    Ms=[]
    for d0 in d0_list:
        D=np.tile(np.array(d0,float),3)
        Ms.append(_build_M(D))
    sym_cands=[]
    sym_end=t0+52.0
    for mi,M in enumerate(Ms):
        ninit=3 if time.time()<sym_end-8 else 1
        for ii in range(ninit):
            if time.time()>sym_end: break
            if time.time()>deadline-125: break
            y0=rng.standard_normal(68)
            # avoid tiny norms
            y0=_run_seed(y0,M,60.0,120)
            y0=_run_seed(y0,M,200.0,150)
            X,_,_,_,_=_expand_raw(y0,M)
            v=_max_cos(X)
            sym_cands.append((v,X.copy(),mi))
            if v<best_val-2e-6:
                # promising symmetric basin, release immediately a bit
                Xr=_run_full(X,150.0,120)
                vr=_max_cos(Xr)
                if vr<best_val:
                    best_val=vr; x_best=Xr.copy()
    sym_cands.sort(key=lambda t:t[0])
    # release top symmetric in different basin
    for si in range(min(3,len(sym_cands))):
        if time.time()>deadline-110: break
        v,X,_=sym_cands[si]
        # skip if essentially same as best basin (too good match?) still polish
        Xr=X.copy()
        # escalating full polish
        Xr=_run_full(Xr,80.0,150)
        Xr=_run_full(Xr,300.0,150)
        vr=_max_cos(Xr)
        # small hole shake to break residual symmetry
        if time.time()<deadline-95:
            Xr,_=_hole_sweep(Xr,rng,k=6,ncand=8000)
            Xr=_run_full(Xr,600.0,80)
            vr=_max_cos(Xr)
        sym_cands[si]=(vr,Xr,sym_cands[si][2])
        if vr<best_val:
            best_val=vr; x_best=Xr.copy()
    # update sym order after release
    # ---- diverse full-basin hops around released + best ----
    explore_end=deadline-32.0
    trial=0
    while time.time()<explore_end:
        trial+=1
        if time.time()>deadline-35: break
        # choose base: mostly global best, sometimes released sym
        if len(sym_cands)>0 and rng.random()<0.35:
            base=sym_cands[int(rng.integers(0,min(3,len(sym_cands))))][1].copy()
        else:
            base=x_best.copy()
        x=base.copy()
        n,d=x.shape
        u=rng.random()
        pm=_per_max(x)
        order=np.argsort(-pm)
        if u<0.4:
            sig=float(rng.choice(np.array([0.03,0.08,0.18,0.35])))
            boost=np.ones((n,1)); boost[order[:10]]*=1.8
            x=x+sig*boost*rng.standard_normal(x.shape)
            x/=np.linalg.norm(x,axis=1,keepdims=True)
        elif u<0.65:
            k=int(rng.integers(2,9))
            idxs=set()
            for j in range(min(2,k)): idxs.add(int(order[j]))
            while len(idxs)<k: idxs.add(int(rng.integers(0,n)))
            for j in idxs:
                mask=np.ones(n,bool); mask[j]=False
                x[j]=_hole_for_rest(x[mask],rng,6000)
        elif u<0.8:
            k=int(rng.integers(8,20))
            sel=rng.choice(n,size=k,replace=False)
            A=rng.standard_normal((d,d)); Q,_=np.linalg.qr(A)
            x[sel]=x[sel]@Q.T
            x[sel]+=float(rng.uniform(0.02,0.10))*rng.standard_normal((k,d))
            x/=np.linalg.norm(x,axis=1,keepdims=True)
        else:
            k=int(rng.integers(4,12))
            sel=rng.choice(n,size=k,replace=False)
            for j in sel:
                vv=rng.standard_normal(d); vv/=np.linalg.norm(vv); x[j]=vv
        b1=float(rng.choice(np.array([60.0,120.0,200.0])))
        b2=float(rng.choice(np.array([350.0,600.0])))
        tl=explore_end-time.time()
        m1,m2=(100,80) if tl>15 else (60,60)
        x=_run_full(x,b1,m1); x=_run_full(x,b2,m2)
        v=_max_cos(x)
        if v<best_val+0.003:
            x,vv=_hole_sweep(x,rng,k=6,ncand=6000)
            v=vv
            if v<best_val-1e-12:
                best_val=v; x_best=x.copy()
                if time.time()<explore_end-2:
                    xl=_lp_polish(x_best,deadline-30.0,r0=5e-4)
                    vl=_max_cos(xl)
                    if vl<best_val: best_val=vl; x_best=xl.copy()
    # final intensive polish
    try:
        x_best,_=_hole_sweep(x_best,rng,k=12,ncand=15000)
    except Exception: pass
    try:
        if deadline-time.time()>8:
            x_best=_run_full(x_best,400.0,250)
            x_best=_run_full(x_best,800.0,200)
    except Exception: pass
    try:
        x_best,_=_hole_sweep(x_best,rng,k=14,ncand=15000)
    except Exception: pass
    rem=deadline-time.time()-0.7
    if rem>3:
        try:
            xl=_lp_polish(x_best,deadline-0.4,r0=8e-4)
            if _max_cos(xl)<_max_cos(x_best): x_best=xl
        except Exception: pass
    x_best=np.asarray(x_best,float)
    x_best/=np.linalg.norm(x_best,axis=1,keepdims=True)
    return x_best
