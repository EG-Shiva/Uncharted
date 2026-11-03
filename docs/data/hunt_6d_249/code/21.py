import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

_IU = None
def _iu(n):
    global _IU
    if _IU is None or _IU[0].shape[0] != n*(n-1)//2:
        _IU = np.triu_indices(n, 1)
    return _IU

def _soft_lbfgs(Xinit, tau, maxiter, creg=0.05):
    n, d = Xinit.shape
    I0, J0 = _iu(n)
    def fg(xf):
        X = xf.reshape(n, d)
        r = np.sqrt(np.sum(X*X, axis=1))
        r = np.maximum(r, 1e-12)
        Xn = X / r[:, None]
        G = Xn @ Xn.T
        g = G[I0, J0]
        gmax = np.max(g)
        e = np.exp((g - gmax) / tau)
        S = np.sum(e)
        Lsoft = gmax + tau * np.log(S)
        w = e / S
        W = np.zeros((n, n), dtype=float)
        W[I0, J0] = w
        W = W + W.T
        A = W @ Xn
        proj = np.sum(A * Xn, axis=1)
        Grad0 = (A - proj[:, None] * Xn) / r[:, None]
        diff = r - 1.0
        L = Lsoft + creg * np.sum(diff*diff)
        Grad = Grad0 + (2.0*creg*diff)[:, None] * (X / r[:, None])
        return float(L), Grad.ravel()
    Xf = Xinit.ravel().copy()
    try:
        res = minimize(fg, Xf, jac=True, method='L-BFGS-B',
                       options={'maxiter': int(maxiter), 'maxfun': int(maxiter*4),
                                'ftol': 1e-14, 'gtol': 1e-7})
        Xf = res.x
    except Exception:
        pass
    Xout = Xf.reshape(n, d)
    rn = np.linalg.norm(Xout, axis=1, keepdims=True)
    rn[rn==0]=1.0
    Xout /= rn
    return Xout

def _relax_two(X, tau_a, tau_b, it_a, it_b):
    X = _soft_lbfgs(X, tau_a, it_a)
    X = _soft_lbfgs(X, tau_b, it_b)
    return X

def _largest_hole(Xo, rng, n_cand=16000, chunk=4000):
    n1, d = Xo.shape
    best_v = 2.0
    best_c = None
    rem = n_cand
    while rem > 0:
        m = min(chunk, rem)
        rem -= m
        C = rng.standard_normal(size=(m, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        D = C @ Xo.T
        cmax = np.max(D, axis=1)
        i = int(np.argmin(cmax))
        if float(cmax[i]) < best_v:
            best_v = float(cmax[i])
            best_c = C[i].copy()
    return best_c, best_v

def _lp_polish(x, deadline):
    n, d = x.shape
    best = _max_cos(x)
    radius = 1.5e-3
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    rep = np.repeat(np.arange(n), d)
    while time.time() < deadline and radius > 1e-13:
        Gfull = x @ x.T
        c_all = Gfull[iu0]
        m = float(c_all.max())
        thresh = m - max(20.0*radius, 1e-9)
        sel = np.where(c_all > thresh)[0]
        if sel.size == 0:
            break
        if sel.size > 4000:
            part = np.argpartition(-c_all[sel], 4000)[:4000]
            sel = sel[part]
        I = iu0[0][sel]; J = iu0[1][sel]
        c_sel = c_all[sel]
        p = sel.size
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols_block = np.concatenate([(I[:,None]*d+ar),(J[:,None]*d+ar),np.full((p,1),n*d)],axis=1).ravel()
        vals_block = np.concatenate([x[J],x[I],-np.ones((p,1))],axis=1).ravel()
        try:
            A_ub = coo_matrix((vals_block,(rows,cols_block)),shape=(p,nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),(rep,np.arange(n*d))),shape=(n,nv)).tocsr()
            obj = np.zeros(nv); obj[-1]=1.0
            bounds = [(-radius,radius)]*(n*d)+[(None,None)]
            res = linprog(obj,A_ub=A_ub,b_ub=-c_sel,A_eq=A_eq,b_eq=np.zeros(n),bounds=bounds,method="highs")
        except Exception:
            radius*=0.5
            continue
        if res.status!=0:
            radius*=0.5
            continue
        y = x + res.x[:-1].reshape(n,d)
        nr = np.linalg.norm(y,axis=1,keepdims=True); nr[nr==0]=1.0
        y/=nr
        v=_max_cos(y)
        if v<best:
            x,best=y,v
            radius=min(radius*1.5,0.02)
        else:
            radius*=0.5
    return x

def solve():
    t0=time.time()
    dead_hop=t0+138.0
    dead_lp=t0+178.0
    rng=np.random.default_rng(7)
    try:
        xb=np.load("best.npy").astype(float)
    except Exception:
        try:
            xb=np.load("record.npy").astype(float)
        except Exception:
            xb=rng.standard_normal(size=(249,6))
    xb=np.asarray(xb,dtype=float)
    if xb.shape!=(249,6):
        xb=rng.standard_normal(size=(249,6))
    nr=np.linalg.norm(xb,axis=1,keepdims=True); nr[nr==0]=1.0
    xb/=nr
    n,d=xb.shape
    best=_max_cos(xb)
    # initial squeeze
    try:
        if time.time()<dead_hop:
            x0=_relax_two(xb,0.008,0.002,70,90)
            v0=_max_cos(x0)
            if v0<best:
                xb,best=x0,v0
    except Exception:
        pass
    sigmas=[0.008,0.015,0.03,0.06,0.12,0.20]
    ks=[3,5,8,12,18]
    it=0
    taus_a=[0.008,0.010,0.006,0.012]
    while time.time()<dead_hop:
        it+=1
        mode=it%3  # 0: hole-worst, 1: hole-random, 2: shake
        try:
            G=xb@xb.T
            np.fill_diagonal(G,-2.0)
            row_max=np.max(G,axis=1)
            order=np.argsort(-row_max)
        except Exception:
            order=np.arange(n)
            row_max=np.zeros(n)
        glob=best
        x_try=None
        if mode==0 or mode==1:
            # hole reinsertion
            if mode==0:
                # pick among top 8 worst
                cand_top=order[:8]
                idx=int(cand_top[rng.integers(0,len(cand_top))])
            else:
                # random point (bias to good/rattler half)
                idx=int(rng.integers(0,n))
            mask=np.ones(n,dtype=bool); mask[idx]=False
            Xo=xb[mask]
            # largest hole w.r.t Xo
            try:
                cstar,bv=_largest_hole(Xo,rng,n_cand=12000,chunk=4000)
            except Exception:
                continue
            G2=Xo@Xo.T
            np.fill_diagonal(G2,-2.0)
            glob_excl=float(np.max(G2)) if G2.size else -2.0
            x_try=xb.copy()
            # small jitter on neighbours of hole to help fit?
            x_try[idx]=cstar
            # tiny global jitter
            x_try+=rng.normal(scale=0.001,size=(n,d))
            nn=np.linalg.norm(x_try,axis=1,keepdims=True); nn[nn==0]=1.0
            x_try/=nn
            tau_a=float(taus_a[it%len(taus_a)])
            x_try=_relax_two(x_try,tau_a,0.0018,70,70)
        else:
            sigma=sigmas[it%len(sigmas)]
            k=ks[(it//3)%len(ks)]
            top30=order[:30]
            k1=min(k//2+1,k,30)
            k2=k-k1
            sel=set()
            if k1>0:
                p1=rng.choice(top30,size=k1,replace=False)
                for v in p1: sel.add(int(v))
            if k2>0:
                cand=rng.choice(n,size=min(n,k2*3),replace=False)
                c=0
                for v in cand:
                    if int(v) not in sel:
                        sel.add(int(v)); c+=1
                        if c>=k2: break
            sel=list(sel)
            tau_a=float(rng.uniform(0.005,0.012))
            lr_steps=90 if sigma<0.05 else 130
            x_try=xb.copy()
            if len(sel)>0:
                x_try[sel]+=rng.normal(scale=sigma,size=(len(sel),d))
            x_try+=rng.normal(scale=0.0012,size=(n,d))
            nn=np.linalg.norm(x_try,axis=1,keepdims=True); nn[nn==0]=1.0
            x_try/=nn
            x_try=_relax_two(x_try,tau_a,0.0018,70,lr_steps)
        if x_try is None:
            continue
        v=_max_cos(x_try)
        if v<best:
            xb,best=x_try,v
        # opportunistic quick LP if big gain? skip to save time
    try:
        if time.time()<dead_lp:
            xb=_lp_polish(xb,dead_lp)
    except Exception:
        pass
    nn=np.linalg.norm(xb,axis=1,keepdims=True); nn[nn==0]=1.0
    xb/=nn
    xb=np.nan_to_num(xb,nan=0.0,posinf=0.0,neginf=0.0)
    nn2=np.linalg.norm(xb,axis=1)
    for i in np.where(nn2<1e-12)[0]:
        v=rng.normal(size=d); v/=np.linalg.norm(v)
        xb[i]=v
    return xb
