import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _e8_roots():
    d = 8
    pts = []
    # D8: two +-1
    for i in range(d):
        for j in range(i+1, d):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = np.zeros(d)
                    v[i] = si
                    v[j] = sj
                    pts.append(v)
    # half-integer with even number of minus (product +1)
    for mask in range(256):
        v = np.ones(d) * 0.5
        nminus = 0
        for k in range(d):
            if (mask >> k) & 1:
                v[k] = -0.5
                nminus += 1
        if nminus % 2 == 0:
            pts.append(v)
    A = np.array(pts, dtype=np.float64)  # 240 x 8
    A = A / np.sqrt(2.0)  # unit norm (raw norm sqrt2)
    # renormalize exactly
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    return A

def _balanced_quench(X, steps, beta0, beta1, lr0, lr1, lam, t_lim):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for t in range(1, steps+1):
        if (t & 127) == 0 and time.time() > t_lim:
            break
        frac = t / steps
        beta = np.exp(lb0 + (lb1-lb0)*frac)
        lr = np.exp(ll0 + (ll1-ll0)*frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = G.max()
        W = np.exp(beta*(G-gmax))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-18
        Grad_s = (W @ X) / S
        # weak one-sided p=2 background to move rattlers
        Gb = X @ X.T
        np.fill_diagonal(Gb, 0.0)
        np.maximum(Gb, 0.0, out=Gb)
        # Wb = 2*Gb
        Grad_b = (Gb @ X) * (2.0 / n)
        Grad = Grad_s + lam * Grad_b
        dot = (Grad*X).sum(axis=1, keepdims=True)
        Gt = Grad - dot*X
        m *= b1; m += (1.0-b1)*Gt
        v *= b2; v += (1.0-b2)*(Gt*Gt)
        bc1 = 1.0 - b1**t; bc2 = 1.0 - b2**t
        step = (m/bc1) / (np.sqrt(v/bc2)+eps)
        X = X - lr*step
        X /= np.sqrt((X*X).sum(axis=1, keepdims=True))
    return X

def _antipodal_quench(Y, steps, beta0, beta1, lr0, lr1, lam, t_lim):
    n2, d = Y.shape
    n = n2*2
    mY = np.zeros_like(Y)
    vY = np.zeros_like(Y)
    b1=0.9; b2=0.999; eps=1e-8
    lb0=np.log(beta0); lb1=np.log(beta1)
    ll0=np.log(lr0); ll1=np.log(lr1)
    for t in range(1, steps+1):
        if (t & 127)==0 and time.time()>t_lim:
            break
        frac=t/steps
        beta=np.exp(lb0+(lb1-lb0)*frac)
        lr=np.exp(ll0+(ll1-ll0)*frac)
        X=np.vstack([Y,-Y])
        G=X@X.T
        np.fill_diagonal(G,-10.0)
        gmax=G.max()
        W=np.exp(beta*(G-gmax))
        np.fill_diagonal(W,0.0)
        S=W.sum(axis=1,keepdims=True)+1e-18
        Grad_s=(W@X)/S
        Gb=X@X.T
        np.fill_diagonal(Gb,0.0)
        np.maximum(Gb,0.0,out=Gb)
        Grad_b=(Gb@X)*(2.0/n)
        Grad=Grad_s+lam*Grad_b
        # project to tangent
        dot=(Grad*X).sum(axis=1,keepdims=True)
        Gt=Grad-dot*X
        GtY=Gt[:n2]-Gt[n2:]
        # Adam on Y
        mY*=b1; mY+=(1.0-b1)*GtY
        vY*=b2; vY+=(1.0-b2)*(GtY*GtY)
        bc1=1.0-b1**t; bc2=1.0-b2**t
        step=(mY/bc1)/(np.sqrt(vY/bc2)+eps)
        Y=Y-lr*step
        Y/=np.sqrt((Y*Y).sum(axis=1,keepdims=True))
    X=np.vstack([Y,-Y])
    return X

def _refine_hole(Y, p, iters=200, beta=90.0, lr0=0.12, lr1=0.004):
    p=p/np.linalg.norm(p)
    for it in range(iters):
        frac=it/max(iters-1,1)
        lr=lr0*(lr1/lr0)**frac
        g=Y@p
        gm=g.max()
        w=np.exp(beta*(g-gm))
        s=w.sum()+1e-18
        A=(w@Y)/s
        At=A-float(A@p)*p
        p=p-lr*At
        nr=np.linalg.norm(p)
        if nr==0:
            break
        p/=nr
    return p

def _find_holes(Y, k, rng, beta=90.0, n_probe=12000):
    d=Y.shape[1]
    P=rng.randn(n_probe,d)
    P/=np.linalg.norm(P,axis=1,keepdims=True)
    CH=4000
    bs=np.empty(n_probe)
    for a in range(0,n_probe,CH):
        b=min(a+CH,n_probe)
        D=P[a:b]@Y.T
        bs[a:b]=D.max(axis=1)
    order=np.argsort(bs)
    holes=[]
    for idx in order:
        p=P[idx]
        ok=True
        for q in holes:
            if abs(float(p@q))>0.88:
                ok=False; break
        if ok:
            Yext=Y if len(holes)==0 else np.vstack([Y,np.array(holes)])
            pr=_refine_hole(Yext,p,iters=200,beta=beta+rng.rand()*40.0)
            holes.append(pr)
            if len(holes)>=k:
                break
        if len(holes)<k and idx>9000 and ok is False:
            continue
    j=0
    while len(holes)<k:
        p=P[order[j]]; j+=1
        Yext=Y if len(holes)==0 else np.vstack([Y,np.array(holes)])
        holes.append(_refine_hole(Yext,p,iters=160,beta=beta))
    return holes

def _equalize(X, steps, tol, lr0, lr1):
    n,d=X.shape
    for s in range(steps):
        frac=s/max(steps-1,1)
        tolcur=tol*(1.0-frac*0.5)
        lr=lr0*(lr1/lr0)**frac
        G=X@X.T
        np.fill_diagonal(G,-2.0)
        m=G.max()
        M=(G>(m-tolcur)).astype(np.float64)
        np.fill_diagonal(M,0.0)
        cnt=M.sum(axis=1)
        active=cnt>0.5
        Gt=np.zeros_like(X)
        if np.any(active):
            Grad=(M@X)/np.maximum(cnt[:,None],1.0)
            dot=(Grad*X).sum(axis=1,keepdims=True)
            GtA=Grad-dot*X
            nn=np.sqrt((GtA*GtA).sum(axis=1,keepdims=True))+1e-12
            Gt[active]=GtA[active]/nn[active]
        # background drift for inactive (rattlers): push from crowding
        if np.any(~active):
            Gb=X@X.T
            np.fill_diagonal(Gb,0.0)
            np.maximum(Gb,0.0,out=Gb)
            Gradb=(Gb@X)*(2.0/n)
            dotb=(Gradb*X).sum(axis=1,keepdims=True)
            GtB=Gradb-dotb*X
            nnb=np.sqrt((GtB*GtB).sum(axis=1,keepdims=True))+1e-12
            Gt[~active]=0.3*GtB[~active]/nnb[~active]
        X=X-lr*Gt
        X/=np.sqrt((X*X).sum(axis=1,keepdims=True))
    return X

def _lp_polish(x, seconds):
    x=x/np.linalg.norm(x,axis=1,keepdims=True)
    n,d=x.shape
    best=_max_cos(x)
    radius=1e-3
    start=time.time()
    iu=np.triu_indices(n,1)
    while time.time()-start<seconds and radius>1e-14:
        c=(x@x.T)[iu]
        m=c.max()
        sel=c>m-max(20*radius,1e-9)
        I,J=iu[0][sel],iu[1][sel]
        p=len(I)
        nv=n*d+1
        if p==0:
            break
        rows=np.repeat(np.arange(p),2*d+1)
        cols=np.concatenate([I[:,None]*d+np.arange(d),J[:,None]*d+np.arange(d),np.full((p,1),n*d)],1).ravel()
        vals=np.concatenate([x[J],x[I],-np.ones((p,1))],1).ravel()
        A_ub=coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
        A_eq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        obj=np.zeros(nv); obj[-1]=1.0
        try:
            res=linprog(obj,A_ub=A_ub,b_ub=-c[sel],A_eq=A_eq,b_eq=np.zeros(n),bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        y=x+res.x[:-1].reshape(n,d)
        y/=np.linalg.norm(y,axis=1,keepdims=True)
        vv=_max_cos(y)
        if vv<best:
            x,best,radius=y,vv,min(radius*1.5,0.05)
        else:
            radius*=0.5
    return x

def solve():
    t0=time.time()
    BUDGET=174.0
    t_explore=t0+124.0
    n,d=168,8
    rng=np.random.RandomState(7)
    best=None; best_val=1e9
    for fn in ("best.npy","record.npy"):
        try:
            a=np.load(fn).astype(np.float64)
            if a.shape!=(n,d):
                continue
            a=a/np.linalg.norm(a,axis=1,keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            va=_max_cos(a)
            if va<best_val:
                best_val=va; best=a.copy()
        except Exception:
            pass
    if best is None:
        best=rng.randn(n,d); best/=np.linalg.norm(best,axis=1,keepdims=True)
        best_val=_max_cos(best)
    E8=_e8_roots()
    elite=[(best_val,best.copy())]
    def push_elite(v,X):
        global_best=False
        for _,Xe in elite:
            if np.max(np.abs(X-Xe))<1e-9 and abs(v-best_val)<1e-12:
                pass
        elite.append((v,X.copy()))
        elite.sort(key=lambda e:e[0])
        del elite[4:]
        return v<best_val-1e-15
    it=0
    while time.time()<t_explore:
        if time.time()+2.5>t0+BUDGET:
            break
        it+=1
        r=rng.rand()
        cur_best=elite[0][1]
        if r<0.38:
            # large hole surgery on best
            X=cur_best.copy()
            G=X@X.T; np.fill_diagonal(G,-2.0)
            mcur=G.max()
            imax=G.max(axis=1)
            cnt=(G>mcur-5e-7).sum(axis=1)
            k=6+int(rng.randint(9))  # 6..14
            order_w=np.argsort(imax); order_c=np.argsort(cnt)
            victims=set()
            for _ in range(k):
                u=rng.rand()
                if u<0.5:
                    for idx in order_w[::-1]:
                        if idx not in victims:
                            victims.add(int(idx)); break
                elif u<0.8:
                    for idx in order_c[::-1]:
                        if idx not in victims:
                            victims.add(int(idx)); break
                else:
                    vv=int(rng.randint(n))
                    while vv in victims:
                        vv=int(rng.randint(n))
                    victims.add(vv)
            victims=sorted(victims)
            keep=np.ones(n,bool); keep[victims]=False
            Y=X[keep]
            holes=_find_holes(Y,k,rng,beta=70.0+rng.rand()*60.0)
            for vi,h in zip(victims,holes):
                X[vi]=h
            X=X+rng.randn(n,d)*0.0035
            X/=np.linalg.norm(X,axis=1,keepdims=True)
            beta0=25.0+rng.rand()*25.0
            beta1=280.0+rng.rand()*340.0
            lr0=0.004+rng.rand()*0.004
            steps=850+int(rng.randint(500))
            lam=0.03+rng.rand()*0.09
            X=_balanced_quench(X,steps,beta0,beta1,lr0,lr0*0.12,lam,t_explore)
            X=_equalize(X,120,3e-4,0.002,0.0002)
        elif r<0.56:
            # E8 subset restart
            idx=rng.choice(240,n,replace=False)
            X=E8[idx].copy()
            sigma=0.008+rng.rand()*0.012
            X=X+rng.randn(n,d)*sigma
            X/=np.linalg.norm(X,axis=1,keepdims=True)
            beta0=20.0+rng.rand()*20.0
            beta1=300.0+rng.rand()*300.0
            steps=1000+int(rng.randint(500))
            lam=0.04+rng.rand()*0.08
            X=_balanced_quench(X,steps,beta0,beta1,0.005,0.0005,lam,t_explore)
            X=_equalize(X,150,4e-4,0.0025,0.0002)
        elif r<0.70:
            # pure random restart
            X=rng.randn(n,d); X/=np.linalg.norm(X,axis=1,keepdims=True)
            # two-stage: broad then sharp
            X=_balanced_quench(X,700,15.0,120.0,0.006,0.001,0.08,t_explore)
            if time.time()>t_explore:
                continue
            X=_balanced_quench(X,900,80.0,500.0+rng.rand()*200.0,0.005,0.0004,0.04,t_explore)
            X=_equalize(X,120,3e-4,0.002,0.0002)
        elif r<0.84:
            # core dispersal: kick worst cluster hard
            X=cur_best.copy()
            G=X@X.T; np.fill_diagonal(G,-2.0)
            imax=G.max(axis=1)
            worst=np.argsort(imax)[-24:]
            sigma_big=0.07+rng.rand()*0.09
            X[worst]+=rng.randn(len(worst),d)*sigma_big
            mask=np.ones(n,bool); mask[worst]=False
            X[mask]+=rng.randn(mask.sum(),d)*0.004
            X/=np.linalg.norm(X,axis=1,keepdims=True)
            beta0=30.0+rng.rand()*20.0
            beta1=260.0+rng.rand()*320.0
            steps=900+int(rng.randint(500))
            lam=0.02+rng.rand()*0.07
            X=_balanced_quench(X,steps,beta0,beta1,0.005,0.0005,lam,t_explore)
            X=_equalize(X,150,3e-4,0.002,0.0002)
        else:
            # antipodal search
            # init Y from random or symmetrized best
            if rng.rand()<0.5:
                Y=rng.randn(84,d); Y/=np.linalg.norm(Y,axis=1,keepdims=True)
            else:
                # symmetrize best: greedy pair removal
                Xb=cur_best.copy()
                # pick 84 diverse via farthest point sampling
                sel=[int(rng.randint(n))]
                D=Xb@Xb.T
                for _ in range(83):
                    # maximize min |dot|? use min dot
                    mnd=np.min(np.abs(D[:,sel]),axis=1)
                    # actually want diverse: minimize max abs? pick smallest max-abs
                    mx=np.max(np.abs(D[:,sel]),axis=1)
                    mx[sel]=1e9
                    # also avoid antipodal duplicates: if we pick i, forbid near -i?
                    sel.append(int(np.argmin(mx)))
                Y=Xb[np.array(sel)]
                # fix signs randomly to break antipodal duplication
                s=rng.choice([-1.0,1.0],size=84)[:,None]
                Y=Y*s
                Y/=np.linalg.norm(Y,axis=1,keepdims=True)
            beta0=25.0+rng.rand()*20.0
            beta1=300.0+rng.rand()*300.0
            X=_antipodal_quench(Y,1000,beta0,beta1,0.005,0.0005,0.05,t_explore)
        if not np.all(np.isfinite(X)):
            continue
        vq=_max_cos(X)
        if vq<elite[-1][0]-1e-12 or (vq<elite[0][0] and len(elite)<4):
            elite.append((vq,X.copy()))
            elite.sort(key=lambda e:e[0])
            del elite[4:]
            if vq<best_val:
                best_val=vq; best=X.copy()
    # final polish elites
    elite.sort(key=lambda e:e[0])
    t_left=(t0+BUDGET-2.0)-time.time()
    if t_left<6:
        t_left=6
    per=max(4.0,t_left/min(3,len(elite)))
    for ei in range(min(3,len(elite))):
        if time.time()+3.0>t0+BUDGET:
            break
        v,Xe=elite[ei]
        Xp=_lp_polish(Xe,seconds=per)
        vp=_max_cos(Xp)
        if vp<best_val:
            best_val=vp; best=Xp.copy()
        elite[ei]=(vp,Xp)
    best=best/np.linalg.norm(best,axis=1,keepdims=True)
    # tiny shrink guard (no-op, keep unit)
    return best
