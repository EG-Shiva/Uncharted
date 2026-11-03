import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n==0]=1.0
    return x/n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _greedy_jam(x, steps, step0):
    x = x.copy()
    best = _max_cos(x)
    step = float(step0)
    n = x.shape[0]
    for _ in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        k = int(np.argmax(G))
        i = k // n; j = k % n
        c = float(G[i,j])
        # tangent directions decreasing <xi,xj>
        vi = c*x[i]-x[j]
        vi = vi - float(vi @ x[i])*x[i]
        vj = c*x[j]-x[i]
        vj = vj - float(vj @ x[j])*x[j]
        ni = float(np.linalg.norm(vi)); nj = float(np.linalg.norm(vj))
        if ni < 1e-14 or nj < 1e-14:
            step *= 0.5
            if step < 1e-12:
                step = step0*0.5
            continue
        vi/=ni; vj/=nj
        yi = x[i]+step*vi; yi/=np.linalg.norm(yi)+1e-30
        yj = x[j]+step*vj; yj/=np.linalg.norm(yj)+1e-30
        y = x.copy(); y[i]=yi; y[j]=yj
        v = _max_cos(y)
        if v < best:
            x=y; best=v
            step=min(step*1.08, 5e-3)
        else:
            # try single-sided
            y2=x.copy(); y2[i]=yi; v2=_max_cos(y2)
            if v2 < best:
                x=y2; best=v2
                step=min(step*1.05,5e-3)
                continue
            y3=x.copy(); y3[j]=yj; v3=_max_cos(y3)
            if v3 < best:
                x=y3; best=v3
                step=min(step*1.05,5e-3)
                continue
            step*=0.5
            if step<1e-12:
                step=step0*0.3
    return x, best

def _adam_lowT(x, steps, T, lr):
    n,d=x.shape
    m1=np.zeros_like(x); m2=np.zeros_like(x)
    b1,b2,eps=0.9,0.999,1e-8
    for it in range(steps):
        G=x@x.T
        np.fill_diagonal(G,-10.0)
        m=float(G.max())
        E=np.exp((G-m)/T)
        np.fill_diagonal(E,0.0)
        s=float(E.sum()*0.5)
        if not np.isfinite(s) or s<=0:
            continue
        Grad=(E@x)/max(s,1e-30)
        Gt=Grad-(Grad*x).sum(axis=1,keepdims=True)*x
        m1=b1*m1+(1-b1)*Gt
        m2=b2*m2+(1-b2)*(Gt*Gt)
        c1=1-b1**(it+2); c2=1-b2**(it+2)
        st=(m1/c1)/(np.sqrt(m2/c2)+eps)
        st=st-(st*x).sum(axis=1,keepdims=True)*x
        nm=np.linalg.norm(st,axis=1,keepdims=True)
        sc=np.minimum(1.0,2.0/(nm+1e-12))
        x=_norm(x-lr*st*sc)
    return x

def _void_replace(x,w,K):
    n,d=x.shape
    R=np.random.randn(K,d)
    R/=np.linalg.norm(R,axis=1,keepdims=True)+1e-30
    S=R@x.T
    S[:,w]=-2.0
    sm=S.max(axis=1)
    k=int(np.argmin(sm))
    return R[k],float(sm[k])

def _lp_polish(x,best,deadline):
    n,d=x.shape
    iu=np.triu_indices(n,1)
    radius=8e-4
    stall=0
    while time.time()<deadline and radius>1e-14:
        Gfull=x@x.T
        cfull=Gfull[iu]
        m=float(cfull.max())
        if m<best:
            best=m
        thresh=max(20.0*radius,4e-7)
        sel=cfull>m-thresh
        p=int(sel.sum())
        if p<50:
            kk=min(1000,len(cfull))
            idx=np.argpartition(-cfull,kk-1)[:kk]
            sel=np.zeros_like(sel,dtype=bool); sel[idx]=True; p=kk
        if p>4000:
            idx=np.where(sel)[0]
            order=np.argsort(cfull[idx])[-4000:]
            nsel=np.zeros_like(sel,dtype=bool); nsel[idx[order]]=True
            sel=nsel; p=4000
        I=iu[0][sel]; J=iu[1][sel]
        csel=cfull[sel]
        rows=np.repeat(np.arange(p),2*d+1)
        cols=np.concatenate([I[:,None]*d+np.arange(d),J[:,None]*d+np.arange(d),np.full((p,1),n*d)],1).ravel()
        vals=np.concatenate([x[J],x[I],-np.ones((p,1))],1).ravel()
        try:
            A_ub=coo_matrix((vals,(rows,cols)),shape=(p,n*d+1)).tocsr()
            A_eq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,n*d+1)).tocsr()
            obj=np.zeros(n*d+1); obj[-1]=1.0
            res=linprog(obj,A_ub=A_ub,b_ub=-csel,A_eq=A_eq,b_eq=np.zeros(n),bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; stall+=1
            if stall>=7:
                rnd=np.random.randn(n,d)*1e-6
                rnd=rnd-(rnd*x).sum(axis=1,keepdims=True)*x
                x=_norm(x+rnd); stall=0
            continue
        y=_norm(x+res.x[:-1].reshape(n,d))
        v=_max_cos(y)
        if v<best-1e-15:
            x=y; best=v; radius=min(radius*1.4,1.5e-3); stall=0
        else:
            radius*=0.5; stall+=1
            if stall>=8:
                rnd=np.random.randn(n,d)*(radius*0.5+1e-7)
                rnd=rnd-(rnd*x).sum(axis=1,keepdims=True)*x
                yt=_norm(x+rnd); vt=_max_cos(yt)
                if vt<best+3e-7:
                    x=yt
                    if vt<best: best=vt
                stall=0
    return x,best

def solve():
    np.random.seed(1)
    t0=time.time()
    BUDGET=170.0
    t_end=t0+BUDGET
    tA=t0+128.0
    best_x=None; best_v=1e99
    for fn in ["best.npy","record.npy"]:
        try:
            a=np.load(fn).astype(np.float64)
            if a.shape==(182,9):
                a=_norm(a); v=_max_cos(a)
                if v<best_v: best_v=v; best_x=a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x=_norm(np.random.randn(182,9)); best_v=_max_cos(best_x)
    n,d=182,9
    # immediate greedy equalization of loaded best
    try:
        gx,gv=_greedy_jam(best_x,800,4e-4)
        if gv<best_v: best_v=gv; best_x=gx.copy()
    except Exception:
        pass
    x_cur=best_x.copy(); cur_v=best_v
    it=0
    while time.time()<tA:
        it+=1
        if it%6==0:
            # void reinsertion of worst point
            G=x_cur@x_cur.T; np.fill_diagonal(G,-2.0)
            rowmax=G.max(axis=1)
            w=int(np.argmax(rowmax))
            # alternate second worst occasionally
            K=25000
            bv,_=_void_replace(x_cur,w,K)
            x_try=x_cur.copy(); x_try[w]=bv
            x_try,vt=_greedy_jam(x_try,350,3e-4)
            # short low-T smoothing
            if time.time()<tA:
                x_try=_adam_lowT(x_try,120,3e-4,2e-4)
                x_try,_=_greedy_jam(x_try,150,1.5e-4)
                vt=_max_cos(x_try)
        elif it%3==0:
            sig=0.002 if (it%9==0) else 0.0004
            rnd=np.random.randn(n,d)*sig
            rnd=rnd-(rnd*x_cur).sum(axis=1,keepdims=True)*x_cur
            x_try=_norm(x_cur+rnd)
            x_try=_adam_lowT(x_try,200,5e-4,4e-4)
            x_try,vt=_greedy_jam(x_try,300,2.5e-4)
        else:
            # pure greedy from jittered current
            sig=0.0003
            rnd=np.random.randn(n,d)*sig
            rnd=rnd-(rnd*x_cur).sum(axis=1,keepdims=True)*x_cur
            x_try=_norm(x_cur+rnd)
            x_try,vt=_greedy_jam(x_try,500,3e-4)
        if vt<best_v:
            best_v=vt; best_x=x_try.copy()
            x_cur=x_try.copy(); cur_v=vt
        elif vt<cur_v or (vt<best_v+6e-4 and np.random.rand()<0.2):
            x_cur=x_try.copy(); cur_v=vt
        else:
            sig=0.003 if (it%4!=0) else 0.015
            rnd=np.random.randn(n,d)*sig
            rnd=rnd-(rnd*best_x).sum(axis=1,keepdims=True)*best_x
            x_cur=_norm(best_x+rnd); cur_v=best_v+sig*0.05
        if time.time()>=tA:
            break
    x=best_x.copy()
    # smooth then LP
    if time.time()<t_end-25:
        try:
            x=_adam_lowT(x,250,4e-4,2e-4)
            v=_max_cos(x)
            if v<best_v: best_v=v; best_x=x.copy()
            else: x=best_x.copy()
            # extra greedy to tie contacts before LP
            x,gv=_greedy_jam(x,600,2e-4)
            if gv<best_v: best_v=gv; best_x=x.copy()
            else: x=best_x.copy() if _max_cos(x)>best_v else x
        except Exception:
            x=best_x.copy()
    if time.time()<t_end-2:
        try:
            x,best_v=_lp_polish(x,best_v,t_end-1.0)
            best_x=x.copy()
        except Exception:
            pass
    best_x=_norm(best_x)
    return best_x
