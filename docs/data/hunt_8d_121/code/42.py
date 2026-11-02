import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def solve():
    t0 = time.time()
    BUDGET = 169.0
    DEADLINE = t0 + BUDGET
    rng = np.random.default_rng(0)
    np.random.seed(0)
    n, d = 121, 8
    iu = np.triu_indices(n, 1)
    Iall = iu[0]; Jall = iu[1]

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        X0 = rng.standard_normal((n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands = [X0]
    vals = [_max_cos(a) for a in cands]
    Xbest = cands[int(np.argmin(vals))].copy()
    Vbest = float(min(vals))

    def lbfgs_anneal(X, betas, maxiters):
        Xf = X.ravel().copy()
        for beta, mi in zip(betas, maxiters):
            if time.time() > DEADLINE - 9.0:
                break
            b = float(beta)
            def fun(flat):
                Xr = flat.reshape(n, d)
                nr = np.linalg.norm(Xr, axis=1, keepdims=True)
                nr = np.maximum(nr, 1e-12)
                Y = Xr / nr
                G = Y @ Y.T
                c = G[iu]
                m = float(c.max())
                e = np.exp(b * (c - m))
                s = float(e.sum())
                if not np.isfinite(s) or s <= 0:
                    return float(m), np.zeros_like(flat)
                f = m + np.log(s) / b
                w = e / s
                W = np.zeros((n, n))
                W[Iall, Jall] = w
                W[Jall, Iall] = w
                GY = W @ Y
                dot = (GY * Y).sum(axis=1, keepdims=True)
                GX = (GY - dot * Y) / nr
                return float(f), GX.ravel()
            try:
                res = minimize(fun, Xf, jac=True, method="L-BFGS-B",
                               options={"maxiter": int(mi), "ftol": 1e-20, "gtol": 1e-12,
                                        "maxfun": int(mi*8)})
                Xf = np.asarray(res.x, dtype=float)
            except Exception:
                break
        Xn = Xf.reshape(n, d)
        Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
        return Xn

    def lp_polish(x, seconds, radius_init=4e-4):
        start = time.time()
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        best = _max_cos(x)
        radius = radius_init
        while time.time() - start < seconds and time.time() < DEADLINE - 1.0:
            if radius < 1e-14:
                break
            G = x @ x.T
            cf = G[iu]
            m = float(cf.max())
            thr = max(20.0 * radius, 2e-9)
            sel = cf > m - thr
            cnt = int(sel.sum())
            if cnt == 0:
                radius *= 0.5
                continue
            if cnt > 2500:
                kth = np.sort(cf)[-2500]
                sel = cf >= kth
                cnt = int(sel.sum())
                if cnt > 3000:
                    idxp = np.argpartition(-cf, 3000)[:3000]
                    mk = np.zeros_like(sel)
                    mk[idxp] = True
                    sel = mk
                    cnt = 3000
            c = cf[sel]
            I = iu[0][sel]; J = iu[1][sel]
            p = cnt
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None]*d + np.arange(d),
                                   J[:, None]*d + np.arange(d),
                                   np.full((p, 1), n*d)], 1).ravel()
            valsA = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            try:
                A_ub = coo_matrix((valsA, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            obj = np.zeros(nv); obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            yn = np.linalg.norm(y, axis=1, keepdims=True); yn[yn==0]=1.0
            y = y/yn
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius*1.6, 0.02)
            else:
                radius *= 0.5
        return x, best

    def contact_degrees(X, tol=1e-9):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        B = G >= m - tol
        deg = B.sum(axis=1)
        rm = G.max(axis=1)
        return deg, rm, float(m), G

    def pick_victims(X, k, tol=2e-7):
        deg, rm, m, G = contact_degrees(X, tol=tol)
        order = np.argsort(-deg)
        victims = []
        used = np.zeros(n, dtype=bool)
        # seed from most overloaded
        for s in order:
            if len(victims) >= k:
                break
            s = int(s)
            if used[s]:
                continue
            victims.append(s); used[s]=True
            if len(victims) >= k:
                break
            # add its tightest neighbours
            nbr = np.argsort(-G[s])
            for nb in nbr[:6]:
                nb=int(nb)
                if len(victims)>=k: break
                if not used[nb] and G[s,nb] > m-5e-6:
                    victims.append(nb); used[nb]=True
        # fill remainder with worst rowmax
        if len(victims)<k:
            o2=np.argsort(-rm)
            for v in o2:
                v=int(v)
                if not used[v]:
                    victims.append(v); used[v]=True
                if len(victims)>=k: break
        return np.array(victims[:k], dtype=int)

    def maximin_fill(Xfixed, k, nsample):
        # sequential farthest insertion + hill climb
        base = Xfixed.copy()
        new_pts=[]
        for t in range(k):
            C = rng.standard_normal((nsample, d))
            C/=np.linalg.norm(C,axis=1,keepdims=True)
            D = C @ base.T
            cm = D.max(axis=1)
            bi=int(np.argmin(cm))
            bc=C[bi].copy(); cur=float(cm[bi])
            for st in range(200):
                sig=0.04*(1.0-st/200.0)+0.001
                prop=bc+sig*rng.standard_normal(d)
                pn=np.linalg.norm(prop)
                if pn==0: continue
                prop/=pn
                md=float((prop @ base.T).max())
                if md<cur:
                    bc=prop; cur=md
            new_pts.append(bc)
            base=np.vstack([base, bc[None,:]])
        return np.stack(new_pts)

    def subset_opt(Fixed, Yinit, beta, maxiter):
        # Fixed: (nf,d), Yinit: (k,d); optimize Y only
        nf = Fixed.shape[0]; k = Yinit.shape[0]
        iu_k = np.triu_indices(k,1)
        Ik = iu_k[0]; Jk = iu_k[1]
        Xf = Yinit.ravel().copy()
        b=float(beta)
        def fun(flat):
            Xr=flat.reshape(k,d)
            nr=np.linalg.norm(Xr,axis=1,keepdims=True)
            nr=np.maximum(nr,1e-12)
            Y=Xr/nr
            C=Y@Fixed.T  # k x nf
            m1=float(C.max()) if C.size else -2.0
            if k>1:
                Gk=Y@Y.T
                ck=Gk[iu_k]
                m2=float(ck.max())
            else:
                m2=-2.0; ck=None
            m=max(m1,m2)
            e1=np.exp(b*(C-m)); s1=float(e1.sum())
            if k>1:
                e2=np.exp(b*(ck-m)); s2=float(e2.sum())
            else:
                s2=0.0; e2=None
            s=s1+s2
            f=m+np.log(s)/b
            # grads
            w1=e1/s
            GYf = w1.T @ Y  # nf x d? actually w1 (k,nf): w1.T @ Y -> nf x d
            # gradient w.r.t Y: w1 @ Fixed
            Grad_Y = w1 @ Fixed
            if k>1:
                W=np.zeros((k,k)); W[Ik,Jk]=e2/s; W[Jk,Ik]=e2/s
                Grad_Y = Grad_Y + W @ Y
                # need dot correction later
            dot=(Grad_Y*Y).sum(axis=1,keepdims=True)
            GX=(Grad_Y-dot*Y)/nr
            return float(f), GX.ravel()
        try:
            res=minimize(fun,Xf,jac=True,method="L-BFGS-B",
                         options={"maxiter":int(maxiter),"ftol":1e-20,"gtol":1e-12,"maxfun":int(maxiter*6)})
            Xf=np.asarray(res.x,dtype=float)
        except Exception:
            pass
        Yn=Xf.reshape(k,d); Yn/=np.linalg.norm(Yn,axis=1,keepdims=True)
        return Yn

    def balanced_reseed(X, k, nsample):
        victims=pick_victims(X,k)
        keep=np.ones(n,dtype=bool); keep[victims]=False
        Fixed=X[keep]
        New=maximin_fill(Fixed,k,nsample)
        # joint subset polish at moderate beta
        for b,mi in [(300,60),(1200,80)]:
            if time.time()>DEADLINE-10: break
            New=subset_opt(Fixed,New,b,mi)
            # re hill-climb mix: one maximin pass already done
        Y=X.copy()
        Y[victims]=New
        return Y

    def cap_rotate(X, cap_size, theta):
        # random plane rotation applied coherently to a cap
        rvec=rng.standard_normal(d); rvec/=np.linalg.norm(rvec)
        dots=X@rvec
        # sometimes center on existing point cluster instead
        if rng.random()<0.5:
            seed=int(rng.integers(0,n))
            dots=X@X[seed]
        cap=np.argsort(-dots)[:cap_size]
        A=rng.standard_normal((d,2))
        Q,_=np.linalg.qr(A)
        u=Q[:,0]; v=Q[:,1]
        c=np.cos(theta); s=np.sin(theta)
        R=np.eye(d)+(c-1.0)*(np.outer(u,u)+np.outer(v,v))+s*(np.outer(v,u)-np.outer(u,v))
        Y=X.copy()
        Y[cap]=(R@X[cap].T).T
        Y/=np.linalg.norm(Y,axis=1,keepdims=True)
        return Y

    def greedy_single(Y, trials):
        G=Y@Y.T; np.fill_diagonal(G,-2.0)
        rowmax=G.max(axis=1); cur_global=float(G.max())
        order=np.argsort(-rowmax)
        for idx in order[:15]:
            cur_row=float(rowmax[idx])
            C=rng.standard_normal((trials,d)); C/=np.linalg.norm(C,axis=1,keepdims=True)
            D=C@Y.T; D[:,idx]=-2.0
            cm=D.max(axis=1); kk=int(np.argmin(cm))
            if cm[kk]<cur_row-1e-12:
                Y2=Y.copy(); Y2[idx]=C[kk]
                ng=_max_cos(Y2)
                if ng<cur_global-1e-12:
                    return Y2,ng,True
        return Y,cur_global,False

    sched_coarse=([120.0,300.0,700.0,1500.0],[100,100,120,150])
    sched_fine=([400.0,900.0,1800.0,3000.0],[80,100,120,150])
    sched_hot=([60.0,150.0,400.0,900.0],[120,120,120,150])

    # initial quick balanced jumps
    for k,ns in [(6,20000),(9,18000),(4,22000)]:
        if time.time()>DEADLINE-120: break
        try:
            Y=balanced_reseed(Xbest,k,ns)
        except Exception:
            continue
        Y=lbfgs_anneal(Y,sched_coarse[0],sched_coarse[1])
        v=_max_cos(Y)
        for _ in range(3):
            Y2,v2,ok=greedy_single(Y,2500)
            if ok and v2<v: Y,v=Y2,v2
            else: break
        if v<Vbest-1e-12:
            Vbest,Xbest=v,Y.copy()
            Xp,vp=lp_polish(Xbest,2.0,3e-4)
            if vp<Vbest: Vbest,Xbest=vp,Xp

    it=0
    while time.time()<DEADLINE-30:
        it+=1
        mode=it%6
        try:
            if mode==0:
                k=int(rng.integers(6,13))
                Y0=balanced_reseed(Xbest,k,14000)
                betas,mis=sched_coarse
            elif mode==1:
                cs=int(rng.integers(8,19)); th=float(rng.uniform(0.08,0.55))
                Y0=cap_rotate(Xbest,cs,th)
                # add small jitter to cap to break symmetry
                Y0=Y0+float(rng.uniform(0.004,0.015))*rng.standard_normal((n,d))
                Y0/=np.linalg.norm(Y0,axis=1,keepdims=True)
                betas,mis=sched_coarse if (it%2==0) else sched_hot
            elif mode==2:
                cs=int(rng.integers(10,22)); th=float(rng.uniform(0.25,0.8))
                Y0=cap_rotate(Xbest,cs,th)
                # follow with small balanced reseed on top
                if rng.random()<0.5:
                    # reseed 3 worst after rotation
                    Y0=balanced_reseed(Y0,3,10000)
                betas,mis=sched_fine
            elif mode==3:
                k=int(rng.integers(2,5))
                Y0=balanced_reseed(Xbest,k,16000)
                betas,mis=sched_fine
            elif mode==4:
                sig=float(rng.uniform(0.01,0.05))
                Y0=Xbest+sig*rng.standard_normal((n,d))
                Y0/=np.linalg.norm(Y0,axis=1,keepdims=True)
                m=int(rng.integers(3,9))
                idx=rng.choice(n,size=m,replace=False)
                R=rng.standard_normal((m,d)); R/=np.linalg.norm(R,axis=1,keepdims=True)
                Y0[idx]=R
                betas,mis=sched_hot
            else:
                sig=float(rng.uniform(0.008,0.03))
                Y0=Xbest+sig*rng.standard_normal((n,d))
                Y0/=np.linalg.norm(Y0,axis=1,keepdims=True)
                betas,mis=sched_fine
        except Exception:
            continue
        Y=lbfgs_anneal(Y0,betas,mis)
        v=_max_cos(Y)
        for _ in range(3):
            if time.time()>DEADLINE-28: break
            Y2,v2,ok=greedy_single(Y,1800)
            if ok and v2<v: Y,v=Y2,v2
            else: break
        if v<Vbest-1e-12:
            Vbest,Xbest=v,Y.copy()
            if time.time()<DEADLINE-32:
                Xp,vp=lp_polish(Xbest,2.0,3e-4)
                if vp<Vbest: Vbest,Xbest=vp,Xp
        if it%10==0 and time.time()<DEADLINE-40:
            if v<Vbest+3e-6:
                Xp,vp=lp_polish(Y,1.2,2e-4)
                if vp<Vbest-1e-12: Vbest,Xbest=vp,Xp.copy()

    if time.time()<DEADLINE-18:
        Y=lbfgs_anneal(Xbest,[800.0,2000.0,4000.0],[150,200,250])
        v=_max_cos(Y)
        if v<Vbest: Vbest,Xbest=v,Y
    if time.time()<DEADLINE-10:
        Xp,vp=lp_polish(Xbest,6.0,3e-4)
        if vp<Vbest: Vbest,Xbest=vp,Xp
        # final tiny shrink to enforce strict feasibility is not needed (unit norm only)
    Xbest=Xbest/np.linalg.norm(Xbest,axis=1,keepdims=True)
    return Xbest
