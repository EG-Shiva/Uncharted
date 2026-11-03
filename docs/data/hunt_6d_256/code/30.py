import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _build_group(kind="even_swap"):
    # returns perms (g,6), signs (g,6) with action y_l = sign[l]*s[perm[l]]
    if kind == "even_swap":
        P = np.array([1,0,3,2,5,4], dtype=int)
        perms=[]; signs=[]
        for mask in range(64):
            s = np.array([1.0 if ((mask>>i)&1)==0 else -1.0 for i in range(6)])
            if np.prod(s) < 0:
                continue
            # b=0
            perms.append(np.arange(6)); signs.append(s.copy())
            # b=1 : D_s @ P
            perms.append(P.copy()); signs.append(s.copy())
        return np.array(perms), np.array(signs)
    elif kind == "full_sign":
        perms=[]; signs=[]
        for mask in range(64):
            s = np.array([1.0 if ((mask>>i)&1)==0 else -1.0 for i in range(6)])
            perms.append(np.arange(6)); signs.append(s)
        return np.array(perms), np.array(signs)
    elif kind == "even_blockswap":
        P = np.array([2,3,0,1,4,5], dtype=int)
        perms=[]; signs=[]
        for mask in range(64):
            s = np.array([1.0 if ((mask>>i)&1)==0 else -1.0 for i in range(6)])
            if np.prod(s) < 0:
                continue
            perms.append(np.arange(6)); signs.append(s.copy())
            perms.append(P.copy()); signs.append(s.copy())
        return np.array(perms), np.array(signs)
    else:
        raise ValueError

def _expand(S, perms, signs):
    ns = S.shape[0]; g = perms.shape[0]
    Xr = np.empty((ns, g, 6))
    for k in range(g):
        p = perms[k]; s = signs[k]
        Xr[:, k, :] = S[:, p] * s[None, :]
    return Xr.reshape(ns*g, 6)

def _sym_adam(S0, perms, signs, steps, K0, K1, lr):
    S = S0.copy()
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    ns = S.shape[0]; g = perms.shape[0]
    m = np.zeros_like(S); v = np.zeros_like(S)
    b1,b2,eps = 0.9,0.999,1e-8
    b1p,b2p = 1.0,1.0
    la0,la1 = np.log(K0), np.log(K1)
    best_val = 1e9
    best_S = S.copy()
    for t in range(1, steps+1):
        K = float(np.exp(la0+(la1-la0)*t/steps))
        X = _expand(S, perms, signs)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(K*(G-gmax))
        np.fill_diagonal(W, 0.0)
        Ssum = float(W.sum()*0.5)
        if not np.isfinite(Ssum) or Ssum<=0:
            continue
        gradX = (W @ X)/Ssum
        GXr = gradX.reshape(ns, g, 6)
        gradS = np.zeros_like(S)
        for k in range(g):
            p = perms[k]; s = signs[k]
            Gk = GXr[:, k, :]
            # gradS[:,p[l]] += s[l]*Gk[:,l]
            for l in range(6):
                gradS[:, p[l]] += s[l]*Gk[:, l]
        dot = np.sum(gradS*S, axis=1, keepdims=True)
        gt = gradS - dot*S
        b1p*=b1; b2p*=b2
        m*=b1; m+=(1-b1)*gt
        v*=b2; v+=(1-b2)*(gt*gt)
        mh = m/(1-b1p); vh = v/(1-b2p)
        step = lr*mh/(np.sqrt(vh)+eps)
        sdot = np.sum(step*S, axis=1, keepdims=True)
        step = step - sdot*S
        S = S - step
        S /= np.linalg.norm(S, axis=1, keepdims=True)
        if t%200==0 or t==steps:
            Xc = _expand(S, perms, signs)
            val = _max_cos(Xc)
            if val < best_val:
                best_val = val
                best_S = S.copy()
    Xb = _expand(best_S, perms, signs)
    return best_S, Xb, best_val

def _adam_sm(x0, steps, K0, K1, lr):
    x = x0.copy()
    n,d = x.shape
    m = np.zeros_like(x); v = np.zeros_like(x)
    b1,b2,eps = 0.9,0.999,1e-8
    b1p,b2p = 1.0,1.0
    la0,la1 = np.log(K0), np.log(K1)
    best_val = _max_cos(x)
    best_x = x.copy()
    for t in range(1, steps+1):
        K = float(np.exp(la0+(la1-la0)*t/steps))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(K*(G-gmax))
        np.fill_diagonal(W, 0.0)
        Ssum = float(W.sum()*0.5)
        if not np.isfinite(Ssum) or Ssum<=0:
            continue
        grad = (W @ x)/Ssum
        dot = np.sum(grad*x, axis=1, keepdims=True)
        gt = grad - dot*x
        b1p*=b1; b2p*=b2
        m*=b1; m+=(1-b1)*gt
        v*=b2; v+=(1-b2)*(gt*gt)
        mh = m/(1-b1p); vh = v/(1-b2p)
        step = lr*mh/(np.sqrt(vh)+eps)
        sdot = np.sum(step*x, axis=1, keepdims=True)
        step = step - sdot*x
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if t%200==0 or t==steps:
            val = _max_cos(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
    return best_x, best_val

def _lp_polish(x, deadline, r0=2.5e-3, cap=3500):
    x = x/np.linalg.norm(x, axis=1, keepdims=True)
    n,d = x.shape
    best = _max_cos(x)
    radius = r0
    iuI, iuJ = np.triu_indices(n,1)
    ar = np.arange(d)
    eq_rows = np.repeat(np.arange(n), d)
    eq_cols = np.arange(n*d)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        margin = max(20.0*radius, 5e-5)
        sel = c > (m-margin)
        cnt = int(np.count_nonzero(sel))
        if cnt==0:
            radius*=0.5
            continue
        if cnt > cap:
            idx = np.where(sel)[0]
            part = np.argpartition(-c[idx], cap)[:cap]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[part]] = True
            sel = keep
            cnt = cap
        I = iuI[sel]; J = iuJ[sel]
        csel = c[sel]
        p = cnt
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d+ar, J[:,None]*d+ar, np.full((p,1), n*d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),(eq_rows,eq_cols)), shape=(n,nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1]=1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
        except Exception:
            radius*=0.5
            continue
        if res.status!=0:
            radius*=0.5
            continue
        y = x + res.x[:-1].reshape(n,d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        val = _max_cos(y)
        if val < best:
            x,best = y,val
            radius = min(radius*1.4, 0.02)
        else:
            radius*=0.5
    return x

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    SYM_END = t0 + 48.0
    ANN_END = t0 + 150.0
    rng = np.random.default_rng(0)
    n,d = 256,6
    x_best = None; best = 1e9
    for fn in ("best.npy","record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape==(256,6) and np.all(np.isfinite(a)):
                a = a/np.linalg.norm(a, axis=1, keepdims=True)
                vv = _max_cos(a)
                if vv < best:
                    best = vv; x_best = a.copy()
        except Exception:
            continue
    if x_best is None:
        x_best = rng.normal(size=(n,d))
        x_best/=np.linalg.norm(x_best,axis=1,keepdims=True)
        best = _max_cos(x_best)
    # symmetry search
    groups = []
    try:
        groups.append(_build_group("even_swap"))
        groups.append(_build_group("full_sign"))
        groups.append(_build_group("even_blockswap"))
    except Exception:
        groups=[]
    x_sym_best=None; sym_best=1e9
    if groups and time.time()<SYM_END:
        # ns=4 for g=64
        trial=0
        # allocate trials: mostly group0
        while time.time()<SYM_END:
            trial+=1
            if trial<=24:
                gi=0
            elif trial<=32:
                gi=1
            elif trial<=40:
                gi=2
            else:
                gi=trial%3
            perms,signs = groups[gi]
            g = perms.shape[0]
            ns = 256//g
            S0 = rng.normal(size=(ns,d))
            S0/=np.linalg.norm(S0,axis=1,keepdims=True)
            # annealed LogSumExp Adam K 100->3000
            steps=1100
            # shorten if low time
            if SYM_END-time.time()<4:
                steps=600
            try:
                _,Xb,vb = _sym_adam(S0,perms,signs,steps,100.0,3000.0,0.02)
            except Exception:
                continue
            if vb<sym_best:
                sym_best=vb; x_sym_best=Xb.copy()
            # early stop if already competitive
            if sym_best<best-5e-4 and trial>10:
                pass
            if time.time()>=SYM_END:
                break
    # release symmetry: refine symmetric best
    if x_sym_best is not None:
        try:
            if time.time()<ANN_END:
                xr,_ = _adam_sm(x_sym_best, 700, 80.0, 2500.0, 0.008)
                vr=_max_cos(xr)
                if vr<best:
                    best=vr; x_best=xr.copy()
                if vr<sym_best:
                    sym_best=vr; x_sym_best=xr.copy()
        except Exception:
            pass
        # extra perturbed releases
        for _ in range(2):
            if time.time()>ANN_END-8:
                break
            try:
                xc=x_sym_best.copy()
                xc+=0.02*rng.normal(size=xc.shape)
                xc/=np.linalg.norm(xc,axis=1,keepdims=True)
                xr,_=_adam_sm(xc,500,60.0,2000.0,0.009)
                vr=_max_cos(xr)
                if vr<best:
                    best=vr; x_best=xr.copy()
            except Exception:
                pass
    # main unconstrained improvement of incumbent (different basins via kicks)
    cyc=0
    while time.time()<ANN_END:
        rem=ANN_END-time.time()
        if rem<4:
            break
        cyc+=1
        x=x_best.copy()
        # topology-changing kick
        if cyc%4==1:
            G=x@x.T; np.fill_diagonal(G,-2.0)
            fi=int(np.argmax(G)); ai=fi//n; aj=fi%n
            eps=0.03+0.02*rng.random()
            x[ai]=x[ai]-eps*x[aj]; x[aj]=x[aj]-eps*x[ai]
            x[ai]+=0.01*rng.normal(size=(6,)); x[aj]+=0.01*rng.normal(size=(6,))
            prow=G.max(axis=1); over=np.argsort(-prow)[:6]
            x[over]+=0.02*rng.normal(size=(6,6))
            x/=np.linalg.norm(x,axis=1,keepdims=True)
            steps,K0,K1,lr=550,80.0,2200.0,0.008
        elif cyc%4==2:
            G=x@x.T; np.fill_diagonal(G,-2.0)
            prow=G.max(axis=1); order=np.argsort(-prow)[:4]
            for r in order:
                Z=rng.normal(size=(4000,6)); Z/=np.linalg.norm(Z,axis=1,keepdims=True)
                S=Z@x.T; S[:,r]=-2.0
                mm=S.max(axis=1); k2=int(np.argmin(mm))
                if mm[k2]<prow[r]-5e-5:
                    x[r]=Z[k2]
            x/=np.linalg.norm(x,axis=1,keepdims=True)
            steps,K0,K1,lr=500,60.0,1800.0,0.009
        elif cyc%4==3:
            G=x@x.T; np.fill_diagonal(G,-2.0)
            fi=int(np.argmax(G)); ai=fi//n
            nb=np.argsort(-G[ai])[:12]
            x[nb]+=0.10*rng.normal(size=(12,6))
            rnd=rng.choice(n,size=4,replace=False)
            x[rnd]+=0.04*rng.normal(size=(4,6))
            x/=np.linalg.norm(x,axis=1,keepdims=True)
            steps,K0,K1,lr=600,40.0,1200.0,0.011
        else:
            over=rng.choice(n,size=12,replace=False)
            x[over]+=0.05*rng.normal(size=(12,6))
            x/=np.linalg.norm(x,axis=1,keepdims=True)
            steps,K0,K1,lr=500,70.0,2000.0,0.008
        try:
            xr,vr=_adam_sm(x,steps,K0,K1,lr)
        except Exception:
            continue
        if vr<best:
            best=vr; x_best=xr.copy()
    # final wide LP polish
    try:
        if time.time()<T_END-2:
            x_best=_lp_polish(x_best,min(time.time()+max(4.0,T_END-time.time()-0.5),T_END),r0=2.5e-3,cap=3500)
            best=_max_cos(x_best)
    except Exception:
        pass
    x_best=x_best/np.linalg.norm(x_best,axis=1,keepdims=True)
    return x_best
