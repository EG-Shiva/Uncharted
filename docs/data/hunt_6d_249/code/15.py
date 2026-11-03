import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _row_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return np.max(g, axis=1), g

def _lbfgs_soft(Xinit, tau, maxiter, creg=0.15):
    n, d = Xinit.shape
    iu0 = np.triu_indices(n, 1)
    I0, J0 = iu0
    def fg(xf):
        X = xf.reshape(n, d)
        r = np.sqrt(np.sum(X*X, axis=1))
        r = np.maximum(r, 1e-12)
        Xn = X / r[:, None]
        G = Xn @ Xn.T
        g = G[I0, J0]
        gmax = np.max(g)
        e = np.exp((g-gmax)/tau)
        S = np.sum(e)
        Lsoft = gmax + tau*np.log(S)
        w = e/S
        W = np.zeros((n,n))
        W[I0,J0]=w
        W = W+W.T
        A = W @ Xn
        proj = np.sum(A*Xn,axis=1)
        Grad0 = (A-proj[:,None]*Xn)/r[:,None]
        diff = r-1.0
        L = Lsoft + creg*np.sum(diff*diff)
        Grad = Grad0 + (2.0*creg*diff)[:,None]*(X/r[:,None])
        return float(L), Grad.ravel()
    Xf = Xinit.ravel().copy()
    try:
        res = minimize(fg, Xf, jac=True, method='L-BFGS-B',
                       options={'maxiter':int(maxiter),'maxfun':int(maxiter*4),'ftol':1e-14,'gtol':1e-7})
        Xf = res.x
    except Exception:
        pass
    Xout = Xf.reshape(n,d)
    Xout /= np.linalg.norm(Xout,axis=1,keepdims=True)+1e-16
    return Xout

def _exact_polish(Xinit, deadline, step0=2e-3):
    cur = Xinit.copy()
    cur /= np.linalg.norm(cur,axis=1,keepdims=True)
    curVal = _max_cos(cur)
    step = step0
    while time.time() < deadline:
        if step < 5e-14:
            break
        G = cur @ cur.T
        np.fill_diagonal(G,-2.0)
        m = float(G.max())
        window = max(10.0*step, 3e-4)
        if step < 5e-5:
            window = max(1.5e-4, 10.0*step)
        mask = G > (m-window)
        F = mask.astype(float) @ cur
        dot = np.sum(F*cur,axis=1)
        Ft = F-dot[:,None]*cur
        trial = cur - step*Ft
        trial /= np.maximum(np.linalg.norm(trial,axis=1,keepdims=True),1e-12)
        v = _max_cos(trial)
        if v < curVal:
            cur=trial; curVal=v
            step=min(step*1.25,0.01)
        else:
            step*=0.5
            if step<1e-7:
                break
    return cur, curVal

def _greedy_multi_insert(Xo, k, rng, n_cand=20000):
    n0,d = Xo.shape
    cur_set = Xo.copy()
    new_pts=[]
    C = rng.standard_normal(size=(n_cand,d))
    C/=np.linalg.norm(C,axis=1,keepdims=True)
    for _ in range(k):
        D = C @ cur_set.T
        cmax = np.max(D,axis=1)
        bi=int(np.argmin(cmax))
        bp=C[bi]
        new_pts.append(bp)
        cur_set=np.vstack([cur_set,bp[None,:]])
        # remove near-duplicate candidates to diversify
        # mask out candidates too close to chosen one occasionally
        if _ < k-1:
            dd = C @ bp
            keep = dd < 0.995
            # keep size stable: resample rejected ones
            nr = int(np.sum(~keep))
            if nr>0:
                R=rng.standard_normal(size=(nr,d))
                R/=np.linalg.norm(R,axis=1,keepdims=True)
                C[~keep]=R
    return np.vstack([Xo,np.array(new_pts)])

def _lp_polish(x, t_end):
    x=x/np.linalg.norm(x,axis=1,keepdims=True)
    n,d=x.shape
    best=_max_cos(x)
    radius=8e-4
    iu=np.triu_indices(n,1)
    while time.time()<t_end-1.0:
        if radius<1e-14:
            break
        Gfull=x@x.T
        c=Gfull[iu]
        m=float(np.max(c))
        sel=c>m-max(20*radius,5e-7)
        cnt=int(np.count_nonzero(sel))
        if cnt==0:
            radius*=0.5; continue
        I=iu[0][sel]; J=iu[1][sel]
        csv=c[sel]
        if cnt>6000:
            idx=np.argsort(-csv)[:6000]
            I=I[idx];J=J[idx];csv=csv[idx]
            cnt=6000
        p=cnt; nv=n*d+1
        rows=np.repeat(np.arange(p),2*d+1)
        cols=np.concatenate([I[:,None]*d+np.arange(d),J[:,None]*d+np.arange(d),np.full((p,1),n*d)],1).ravel()
        vals=np.concatenate([x[J],x[I],-np.ones((p,1))],1).ravel()
        try:
            A_ub=coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
            obj=np.zeros(nv);obj[-1]=1.0
            res=linprog(obj,A_ub=A_ub,b_ub=-csv,A_eq=A_eq,b_eq=np.zeros(n),
                        bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        y=x+res.x[:-1].reshape(n,d)
        y/=np.linalg.norm(y,axis=1,keepdims=True)
        v=_max_cos(y)
        if v<best:
            x,best=y,v
            radius=min(radius*1.5,0.02)
        else:
            radius*=0.5
    return x

def solve():
    t0=time.time()
    BUDGET_END=t0+172.0
    rng=np.random.default_rng(123)
    np.random.seed(123)
    n,d=249,6
    cands=[]
    for fn in ("best.npy","record.npy"):
        try:
            a=np.load(fn); a=np.asarray(a,dtype=float)
            if a.shape==(n,d):
                a=a/np.linalg.norm(a,axis=1,keepdims=True)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        X0=rng.standard_normal(size=(n,d)); X0/=np.linalg.norm(X0,axis=1,keepdims=True)
        cands.append(X0)
    vals=[_max_cos(a) for a in cands]
    bi=int(np.argmin(vals))
    bestX=cands[bi].copy(); bestVal=float(vals[bi])
    Xcur=bestX.copy(); curVal=bestVal
    def note(X):
        nonlocal bestX,bestVal
        v=_max_cos(X)
        if v<bestVal:
            bestVal=v; bestX=X.copy()
        return v
    # quick initial soften
    if time.time()<BUDGET_END-60:
        Xcur=_lbfgs_soft(Xcur,0.004,80)
        curVal=_max_cos(Xcur); note(Xcur)
        Xcur,_=_exact_polish(Xcur,time.time()+1.5,step0=1.5e-3)
        curVal=_max_cos(Xcur); note(Xcur)
    it=0
    sigmas=[0.03,0.015,0.05,0.02,0.008]
    while time.time()<BUDGET_END-38:
        it+=1
        rm,G=_row_max(Xcur)
        m=float(np.max(G))
        order=np.argsort(-rm)
        # contact counts for overload detection
        cc=np.sum(G>m-2e-4,axis=1)
        omax=np.argsort(-cc)
        r=np.random.rand() if False else rng.random()
        if r<0.55:
            # cluster ejection: center + its nearest neighbors
            if rng.random()<0.7:
                center=int(order[int(rng.integers(0,5))])
            else:
                center=int(omax[int(rng.integers(0,5))])
            k=int(rng.integers(2,6))  # 2..5
            nbr=np.argsort(-G[center])
            # exclude center itself (first is -2 diag? diag is -2 so not first; need filter)
            cand=[c for c in nbr if c!=center][:k-1]
            idxs=[center]+cand
            mask=np.ones(n,bool); mask[idxs]=False
            Xo=Xcur[mask]
            Xtry_full=_greedy_multi_insert(Xo,k,rng,n_cand=16000)
            # Xtry_full is Xo + new; need to map back to full ordering: just use it (order irrelevant)
            Xtry=Xtry_full
            Xtry=_lbfgs_soft(Xtry,0.0012,60)
            Xtry,_=_exact_polish(Xtry,time.time()+1.0,step0=1.2e-3)
            vtry=_max_cos(Xtry)
        elif r<0.75:
            # worst-k ejection (spatially dispersed worst)
            k=int(rng.integers(2,5))
            idxs=list(order[:k])
            mask=np.ones(n,bool); mask[idxs]=False
            Xo=Xcur[mask]
            Xtry=_greedy_multi_insert(Xo,k,rng,n_cand=16000)
            Xtry=_lbfgs_soft(Xtry,0.001,60)
            Xtry,_=_exact_polish(Xtry,time.time()+1.0,step0=1.2e-3)
            vtry=_max_cos(Xtry)
        else:
            # joint tangent shake of a contact cluster
            center=int(order[int(rng.integers(0,8))])
            k=int(rng.integers(4,11))
            nbr=np.argsort(-G[center])[:k]
            # include center
            sset=list(set([center]+list(nbr)))[:k+1]
            sigma=sigmas[it%len(sigmas)]
            Xtry=Xcur.copy()
            pert=rng.standard_normal(size=(len(sset),d))*sigma
            for ii,idx in enumerate(sset):
                v=pert[ii]-np.dot(pert[ii],Xtry[idx])*Xtry[idx]
                Xtry[idx]=Xtry[idx]+v
            # tiny global jitter
            j=rng.standard_normal(size=(n,d))*(sigma*0.1)
            j-=np.sum(j*Xtry,axis=1,keepdims=True)*Xtry
            Xtry=Xtry+j
            Xtry/=np.linalg.norm(Xtry,axis=1,keepdims=True)
            Xtry=_lbfgs_soft(Xtry,0.001,50)
            Xtry,_=_exact_polish(Xtry,time.time()+0.8,step0=1e-3)
            vtry=_max_cos(Xtry)
        if vtry<bestVal-2e-12:
            Xcur=Xtry; curVal=vtry; note(Xcur)
        elif vtry<curVal-1e-10:
            Xcur=Xtry; curVal=vtry
            if rng.random()<0.2:
                note(Xcur)
        else:
            if rng.random()<0.06:
                Xcur=Xtry; curVal=vtry
        if it%12==0:
            Xcur=bestX.copy(); curVal=bestVal
            # big kick from best to new basin
            k=int(rng.integers(3,7))
            # pick random center among top 20 worst of best
            rmb,_=_row_max(bestX)
            o2=np.argsort(-rmb)
            center=int(o2[int(rng.integers(0,20))])
            Gb=bestX@bestX.T; np.fill_diagonal(Gb,-2.0)
            nbr=np.argsort(-Gb[center])[:k]
            sset=list(set([center]+list(nbr)))[:k+1]
            sigma=0.06
            pert=rng.standard_normal(size=(len(sset),d))*sigma
            for ii,idx in enumerate(sset):
                v=pert[ii]-np.dot(pert[ii],Xcur[idx])*Xcur[idx]
                Xcur[idx]=Xcur[idx]+v
            Xcur/=np.linalg.norm(Xcur,axis=1,keepdims=True)
            curVal=_max_cos(Xcur)
        # fast single-hole opportunism every few iters
        if it%3==0 and time.time()<BUDGET_END-40:
            rm2,G2=_row_max(Xcur)
            glob=float(np.max(G2))
            o2=np.argsort(-rm2)
            for kk in range(2):
                idx=int(o2[kk])
                mask=np.ones(n,bool); mask[idx]=False
                Xo=Xcur[mask]
                Cc=rng.standard_normal(size=(8000,d)); Cc/=np.linalg.norm(Cc,axis=1,keepdims=True)
                D=Cc@Xo.T; cm=np.max(D,axis=1); bi2=int(np.argmin(cm))
                if float(cm[bi2])<glob-1e-6:
                    Xt=Xcur.copy(); Xt[idx]=Cc[bi2]
                    vg=_max_cos(Xt)
                    if vg<curVal:
                        Xcur=Xt; curVal=vg
                        if vg<bestVal: note(Xcur)
                        rm2,G2=_row_max(Xcur); glob=float(np.max(G2)); o2=np.argsort(-rm2)
    # final polish best
    Xfin=bestX.copy()
    if time.time()<BUDGET_END-30:
        Xfin=_lbfgs_soft(Xfin,0.0006,120)
        v=_max_cos(Xfin)
        if v<bestVal:
            bestVal=v; bestX=Xfin.copy()
        else:
            Xfin=bestX.copy()
    rem=BUDGET_END-time.time()-1.5
    if rem>6:
        try:
            Xlp=_lp_polish(Xfin,BUDGET_END-1.0)
            vlp=_max_cos(Xlp)
            if vlp<bestVal:
                bestX=Xlp; bestVal=vlp
        except Exception:
            pass
    bestX=bestX/np.linalg.norm(bestX,axis=1,keepdims=True)
    return bestX
