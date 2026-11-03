import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, deadline, p_cap=1500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() > deadline:
            break
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        window = max(20.0 * radius, 1e-9)
        sel = c_all > m - window
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > p_cap:
            idx = np.where(sel)[0]
            part = np.argpartition(-c_all[idx], p_cap - 1)[:p_cap]
            keep = np.zeros_like(sel)
            keep[idx[part]] = True
            sel = keep
        I = iu0[sel]; J = iu1[sel]; csel = c_all[sel]
        p = int(sel.sum())
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([(I[:, None]*d+np.arange(d)).reshape(p,d),
                               (J[:, None]*d+np.arange(d)).reshape(p,d),
                               np.full((p,1), n*d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            erows = np.repeat(np.arange(n),d); ecols=np.arange(n*d)
            A_eq = coo_matrix((x.ravel(),(erows,ecols)),shape=(n,nv)).tocsr()
            obj=np.zeros(nv); obj[-1]=1.0
            res=linprog(obj,A_ub=A_ub,b_ub=-csel,A_eq=A_eq,b_eq=np.zeros(n),
                        bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        y=x+res.x[:-1].reshape(n,d)
        y/=np.linalg.norm(y,axis=1,keepdims=True)
        v=_max_cos(y)
        if v<best:
            best=v; best_x=y.copy(); x=y
            radius=min(radius*1.5,0.05)
        else:
            radius*=0.5
            if radius<1e-7:
                x=best_x.copy()
        if time.time()-start>=seconds:
            break
    return best_x

def _tangent_bases(x):
    n,d=x.shape; dm=d-1
    Ts=[]
    for i in range(n):
        xi=x[i]
        k=int(np.argmax(np.abs(xi)))
        idx=[j for j in range(d) if j!=k]
        Xj=xi[idx]
        V=-np.outer(xi,Xj).copy()
        for t,j in enumerate(idx):
            V[j,t]+=1.0
        Q,R=np.linalg.qr(V)
        Ts.append(Q)
    return Ts

def solve():
    t0=time.time()
    HARD=168.0
    deadline=t0+HARD
    HOP_END=t0+148.0
    np.random.seed(0)
    rng=np.random.default_rng(0)
    n,d=123,13
    dm=d-1; N=n*dm
    best=None; best_val=1e9
    for fn in ("best.npy","record.npy"):
        try:
            a=np.load(fn)
            a=np.asarray(a,float)
            if a.shape==(n,d):
                a/=np.linalg.norm(a,axis=1,keepdims=True)
                v=_max_cos(a)
                if v<best_val:
                    best_val=v; best=a.copy()
        except Exception:
            pass
    if best is None:
        best=rng.normal(size=(n,d)); best/=np.linalg.norm(best,axis=1,keepdims=True)
        best_val=_max_cos(best)
    # quick initial squeeze
    try:
        if time.time()<HOP_END:
            c=_lp_polish(best,2.0,deadline)
            v=_max_cos(c)
            if v<best_val:
                best_val=v; best=c.copy()
    except Exception:
        pass
    iu0,iu1=np.triu_indices(n,1)
    it=0
    thetas=[0.04,0.07,0.11,0.15,0.05,0.09,0.13,0.03]
    while time.time()<HOP_END and time.time()<deadline-12:
        it+=1
        try:
            x=best.copy()
            G=x@x.T
            np.fill_diagonal(G,-2.0)
            m=float(G.max())
            thr=m-1e-7
            deg=(G>thr).sum(axis=1)
            c_all=G[iu0,iu1]
            sel=c_all>thr
            I=iu0[sel]; J=iu1[sel]
            p=len(I)
            if p==0 or p>4000:
                # fallback random shake
                theta=thetas[it%len(thetas)]
                pert=rng.normal(size=(n,d))*0.02
                pert-= (pert*x).sum(1,keepdims=True)*x
                y=x+pert
                y/=np.linalg.norm(y,axis=1,keepdims=True)
                c=_lp_polish(y,1.5,deadline)
                v=_max_cos(c)
                if v<best_val:
                    best_val=v; best=c.copy()
                continue
            hubs=np.argsort(-deg)[:6]
            hub=int(hubs[(it-1)%len(hubs)])
            inc=np.where((I==hub)|(J==hub))[0]
            if len(inc)==0:
                hub=int(hubs[0]); inc=np.where((I==hub)|(J==hub))[0]
                if len(inc)==0:
                    continue
            other=np.where(I[inc]==hub,J[inc],I[inc])
            order=np.argsort(-deg[other])
            kdrop=min(len(inc),4+((it-1)%3))
            rot=(it*3)%len(inc)
            pick_idx=[]
            for t in range(kdrop):
                pick_idx.append(inc[order[(rot+t)%len(inc)]])
            pick_idx=np.array(pick_idx,dtype=int)
            drop_mask=np.zeros(p,bool); drop_mask[pick_idx]=True
            keep_mask=~drop_mask
            Ts=_tangent_bases(x)
            # build reduced rigidity matrix
            # precompute Tx = Ts[i].T @ x[j] quickly
            # Build M_full p x N
            M=np.zeros((p,N),dtype=float)
            # cache projections: proj[i,j]=Ts[i].T@x[j]  -> n x n x dm too big; compute on fly per contact
            for r in range(p):
                a=int(I[r]); b=int(J[r])
                va=Ts[a].T.dot(x[b])
                vb=Ts[b].T.dot(x[a])
                M[r,a*dm:(a+1)*dm]=va
                M[r,b*dm:(b+1)*dm]=vb
            Mk=M[keep_mask]; Md=M[drop_mask]
            # random null mode of Mk via projection
            best_v=None; best_score=1e9
            for trial in range(3):
                rvec=rng.standard_normal(N)
                yv=Mk.dot(rvec)
                # solve (Mk MkT + reg) w = yv
                try:
                    GMT=Mk.dot(Mk.T)
                    reg=1e-10*(np.trace(GMT)/max(Mk.shape[0],1)+1e-12)
                    GMT.flat[::GMT.shape[0]+1]+=reg+1e-12
                    w=np.linalg.solve(GMT,yv)
                except Exception:
                    try:
                        w,_ ,_,_=np.linalg.lstsq(Mk.T,rvec,rcond=None)
                        # lstsq gives w? actually solves MkT w = rvec; need adjust
                    except Exception:
                        continue
                    # recompute residual form
                    continue
                vvec=rvec-Mk.T.dot(w)
                nv=np.linalg.norm(vvec)
                if nv<1e-12 or not np.isfinite(nv):
                    continue
                vvec/=nv
                dpos=Md.dot(vvec)
                # choose sign minimizing max
                s1=float(dpos.max()) if dpos.size else 0.0
                s2=float((-dpos).max()) if dpos.size else 0.0
                if s2<s1:
                    vvec=-vvec; sc=s2
                else:
                    sc=s1
                # tie-break by mean
                mean=float((Md.dot(vvec)).mean()) if dpos.size else 0.0
                score=sc+0.1*mean
                if score<best_score:
                    best_score=score; best_v=vvec.copy()
            if best_v is None:
                continue
            # map to dx
            DX=np.zeros((n,d))
            maxn=1e-20
            for i in range(n):
                blk=best_v[i*dm:(i+1)*dm]
                dxi=Ts[i].dot(blk)
                DX[i]=dxi
                nn=float(np.linalg.norm(dxi))
                if nn>maxn: maxn=nn
            theta=thetas[(it-1)%len(thetas)]
            # vary theta slightly with trial
            scale=theta/maxn
            DX*=scale
            norms=np.linalg.norm(DX,axis=1)
            y=np.empty_like(x)
            for i in range(n):
                ni=float(norms[i])
                if ni<1e-14:
                    y[i]=x[i]
                else:
                    y[i]=np.cos(ni)*x[i]+np.sin(ni)*DX[i]/ni
            y/=np.linalg.norm(y,axis=1,keepdims=True)
            c=_lp_polish(y,1.6,deadline)
            v=_max_cos(c)
            if v<best_val:
                best_val=v; best=c.copy()
        except Exception:
            continue
    # final polish with remaining time
    try:
        rem=deadline-time.time()-1.0
        if rem>2:
            c=_lp_polish(best,min(rem,12.0),deadline)
            v=_max_cos(c)
            if v<best_val:
                best=c.copy()
    except Exception:
        pass
    best/=np.linalg.norm(best,axis=1,keepdims=True)
    return best
