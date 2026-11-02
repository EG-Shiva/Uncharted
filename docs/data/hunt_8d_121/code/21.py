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
    BUDGET = 170.0
    DEADLINE = t0 + BUDGET
    np.random.seed(1)
    rng = np.random.default_rng(1)
    n, d = 121, 8
    iu0 = np.triu_indices(n, 1)
    Iall = iu0[0]; Jall = iu0[1]

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

    def contact_degrees(X, eps=3e-4):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        B = G > (m - eps)
        deg = B.sum(axis=1).astype(float)
        return deg, m, G

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
                c = G[Iall, Jall]
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
                               options={"maxiter": int(mi), "ftol": 1e-20, "gtol": 1e-12, "maxfun": int(mi*8)})
                Xf = np.asarray(res.x, dtype=float)
            except Exception:
                break
        Xn = Xf.reshape(n, d)
        Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
        return Xn

    def lp_polish(x, seconds, radius_init=5e-4):
        start = time.time()
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        best = _max_cos(x)
        bx = x.copy()
        radius = radius_init
        while time.time() - start < seconds and time.time() < DEADLINE - 1.5:
            if radius < 1e-14:
                break
            G = x @ x.T
            cf = G[iu0]
            m = float(cf.max())
            thr = max(25.0*radius, 3e-9)
            sel = cf > m - thr
            cnt = int(sel.sum())
            if cnt == 0:
                radius *= 0.5
                continue
            if cnt > 2200:
                kth = np.sort(cf)[-2200]
                sel = cf >= kth
                cnt = int(sel.sum())
                if cnt > 2600:
                    idxp = np.argpartition(-cf, 2600)[:2600]
                    mk = np.zeros_like(sel); mk[idxp]=True; sel=mk; cnt=2600
            c = cf[sel]
            I = iu0[0][sel]; J = iu0[1][sel]
            p = cnt; nv = n*d+1
            rows = np.repeat(np.arange(p), 2*d+1)
            cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)],1).ravel()
            valsA = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
            try:
                A_ub = coo_matrix((valsA,(rows,cols)),shape=(p,nv)).tocsr()
                A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
            except Exception:
                radius*=0.5; continue
            obj = np.zeros(nv); obj[-1]=1.0
            try:
                res = linprog(obj,A_ub=A_ub,b_ub=-c,A_eq=A_eq,b_eq=np.zeros(n),
                              bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
            except Exception:
                radius*=0.5; continue
            if res.status!=0:
                radius*=0.5; continue
            y = x + res.x[:-1].reshape(n,d)
            yn = np.linalg.norm(y,axis=1,keepdims=True); yn[yn==0]=1.0; y/=yn
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                bx = y.copy()
                radius = min(radius*1.6, 0.02)
            else:
                radius*=0.5
        return bx, best

    def greedy_single(Y, trials):
        G = Y @ Y.T
        np.fill_diagonal(G, -2.0)
        rowmax = G.max(axis=1)
        cur_global = float(G.max())
        order = np.argsort(-rowmax)
        for idx in order[:14]:
            cur_row = float(rowmax[idx])
            C = rng.standard_normal((trials, d))
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            D = C @ Y.T
            D[:, idx] = -2.0
            cm = D.max(axis=1)
            k = int(np.argmin(cm))
            if cm[k] < cur_row - 1e-12:
                Y2 = Y.copy(); Y2[idx]=C[k]
                ng = _max_cos(Y2)
                if ng < cur_global - 1e-12:
                    return Y2, ng, True
        return Y, cur_global, False

    def find_void(base, nsample, refine=220):
        C = rng.standard_normal((nsample, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        D = C @ base.T
        cm = D.max(axis=1)
        kbest = int(np.argmin(cm))
        bc = C[kbest].copy()
        cur = float(cm[kbest])
        for st in range(refine):
            sig = 0.04*(1.0-st/refine)+0.001
            prop = bc + sig*rng.standard_normal(d)
            pn = np.linalg.norm(prop)
            if pn==0: continue
            prop/=pn
            md = float((prop @ base.T).max())
            if md < cur:
                bc=prop; cur=md
        return bc, cur

    def hub_void_reseed(X, k, nsample):
        deg, m, G = contact_degrees(X, eps=3e-4)
        hubs = np.argsort(-deg)[:k]
        keep_mask = np.ones(n,bool); keep_mask[hubs]=False
        fixed = X[keep_mask]
        new_pts=[]
        for t in range(k):
            if new_pts:
                base = np.vstack([fixed, np.stack(new_pts)])
            else:
                base = fixed
            bc,_ = find_void(base, nsample, refine=200)
            new_pts.append(bc)
        Y = X.copy()
        for ii,wi in enumerate(hubs):
            Y[wi]=new_pts[ii]
        return Y

    def lowdeg_void_relocate(X, k, nsample):
        deg, m, G = contact_degrees(X, eps=3e-4)
        lows = np.argsort(deg)[:max(k*3,8)]
        pick = rng.choice(lows, size=k, replace=False)
        Y = X.copy()
        for wi in pick:
            base = np.delete(Y, wi, axis=0)
            bc,_ = find_void(base, nsample, refine=200)
            Y[wi]=bc
        return Y

    def contact_cluster_kick(X, ksize, sigma_c, sigma_g):
        deg, m, G = contact_degrees(X, eps=5e-4)
        order = np.argsort(-deg)
        pool = min(18, n)
        seed = int(order[rng.integers(0, pool)])
        nbr = np.argsort(-G[seed])[:ksize]
        Y = X.copy() + sigma_g*rng.standard_normal((n,d))
        Y[nbr] += (sigma_c-sigma_g)*rng.standard_normal((len(nbr),d))
        if rng.random()<0.45:
            s2 = int(rng.integers(0,n))
            n2 = np.argsort(-G[s2])[:max(3,ksize//2)]
            Y[n2]+= sigma_c*0.7*rng.standard_normal((len(n2),d))
        Y/=np.linalg.norm(Y,axis=1,keepdims=True)
        return Y

    sched_coarse=([120.0,300.0,700.0,1500.0],[110,110,140,180])
    sched_fine=([400.0,900.0,1800.0,3000.0],[90,110,140,220])
    sched_hot=([60.0,150.0,400.0,900.0],[140,140,140,180])
    sched_ultra=([200.0,600.0,1500.0,4000.0,8000.0],[70,70,90,110,130])

    # initial squeeze of incumbent
    if time.time() < DEADLINE-60:
        try:
            Y = lbfgs_anneal(Xbest, [300.0,800.0,2000.0,5000.0],[70,70,100,130])
            v=_max_cos(Y)
            if v < Vbest:
                Vbest,Xbest=v,Y.copy()
            Xp,vp=lp_polish(Xbest,2.0,4e-4)
            if vp < Vbest:
                Vbest,Xbest=vp,Xp
        except Exception:
            pass

    elites=[(Vbest,Xbest.copy())]

    # early hub surgery attempts
    for k,ns in [(3,22000),(5,18000),(4,22000)]:
        if time.time()>DEADLINE-110: break
        try:
            Y0=hub_void_reseed(Xbest,k,ns)
            Y=lbfgs_anneal(Y0,sched_coarse[0],sched_coarse[1])
            v=_max_cos(Y)
            for _ in range(4):
                Y2,v2,ok=greedy_single(Y,2500)
                if ok and v2<v: Y,v=Y2,v2
                else: break
            if v<Vbest-1e-12:
                Vbest,Xbest=v,Y.copy()
                elites.append((v,Xbest.copy())); elites=sorted(elites,key=lambda t:t[0])[:3]
                Xp,vp=lp_polish(Xbest,2.0,3e-4)
                if vp<Vbest: Vbest,Xbest=vp,Xp; elites[0]=(Vbest,Xbest.copy())
        except Exception:
            continue

    it=0
    while time.time() < DEADLINE-28:
        it+=1
        mode=it%6
        try:
            base = elites[0][1] if rng.random()<0.8 else elites[int(rng.integers(0,len(elites)))][1]
            if mode==0:
                Y0=hub_void_reseed(base,int(rng.integers(2,6)),14000)
                betas,mis=sched_coarse
            elif mode==1:
                ks=int(rng.integers(7,14)); sc=float(rng.uniform(0.20,0.48)); sg=float(rng.uniform(0.004,0.02))
                Y0=contact_cluster_kick(base,ks,sc,sg)
                betas,mis=sched_coarse if (it%2==0) else sched_hot
            elif mode==2:
                ks=int(rng.integers(5,10)); sc=float(rng.uniform(0.10,0.26)); sg=float(rng.uniform(0.008,0.03))
                Y0=contact_cluster_kick(base,ks,sc,sg)
                betas,mis=sched_fine
            elif mode==3:
                Y0=lowdeg_void_relocate(base,int(rng.integers(1,4)),12000)
                # add small shake
                Y0 = Y0 + 0.012*rng.standard_normal((n,d))
                Y0/=np.linalg.norm(Y0,axis=1,keepdims=True)
                betas,mis=sched_hot
            elif mode==4:
                sig=float(rng.uniform(0.01,0.05))
                Y0=base+sig*rng.standard_normal((n,d))
                Y0/=np.linalg.norm(Y0,axis=1,keepdims=True)
                m=int(rng.integers(3,9))
                idx=rng.choice(n,size=m,replace=False)
                R=rng.standard_normal((m,d)); R/=np.linalg.norm(R,axis=1,keepdims=True)
                Y0[idx]=R
                betas,mis=sched_hot
            else:
                sig=float(rng.uniform(0.006,0.025))
                Y0=base+sig*rng.standard_normal((n,d))
                Y0/=np.linalg.norm(Y0,axis=1,keepdims=True)
                betas,mis=sched_fine if (it%2==0) else sched_ultra
        except Exception:
            continue
        try:
            Y=lbfgs_anneal(Y0,betas,mis)
            v=_max_cos(Y)
        except Exception:
            continue
        if v < Vbest+0.003 and time.time()<DEADLINE-26:
            for _ in range(3):
                try:
                    Y2,v2,ok=greedy_single(Y,1800)
                except Exception:
                    break
                if ok and v2<v: Y,v=Y2,v2
                else: break
        if v < Vbest-1e-12:
            Vbest,Xbest=v,Y.copy()
            elites.append((v,Xbest.copy())); elites=sorted(elites,key=lambda t:t[0])[:3]
            if time.time()<DEADLINE-32:
                try:
                    Xp,vp=lp_polish(Xbest,2.0,3e-4)
                    if vp<Vbest:
                        Vbest,Xbest=vp,Xp; elites[0]=(Vbest,Xbest.copy())
                except Exception:
                    pass
        elif it%11==0 and time.time()<DEADLINE-38:
            if v < Vbest+2e-6:
                try:
                    Xp,vp=lp_polish(Y,1.2,2e-4)
                    if vp<Vbest-1e-12:
                        Vbest,Xbest=vp,Xp.copy()
                        elites.append((vp,Xbest.copy())); elites=sorted(elites,key=lambda t:t[0])[:3]
                except Exception:
                    pass

    if time.time()<DEADLINE-18:
        try:
            Y=lbfgs_anneal(Xbest,sched_ultra[0],sched_ultra[1])
            v=_max_cos(Y)
            if v<Vbest: Vbest,Xbest=v,Y
            for _ in range(5):
                Y2,v2,ok=greedy_single(Xbest,4000)
                if ok and v2<Vbest: Xbest,Vbest=Y2,v2
                else: break
        except Exception:
            pass
    if time.time()<DEADLINE-8:
        try:
            remain=max(1.0,DEADLINE-time.time()-1.0)
            Xp,vp=lp_polish(Xbest,min(remain,8.0),3e-4)
            if vp<Vbest: Vbest,Xbest=vp,Xp
        except Exception:
            pass
    Xbest/=np.linalg.norm(Xbest,axis=1,keepdims=True)
    return Xbest
