import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def build_orb(m_list, b_list, a_val=1):
    # m_list: list of orbit sizes, b_list: list of b exponents, a=1 for all
    N = int(sum(m_list))
    R = len(m_list)
    rep_idx = np.zeros(N, dtype=int)
    c1 = np.zeros(N); s1 = np.zeros(N); c2 = np.zeros(N); s2 = np.zeros(N)
    p = 0
    for r, m in enumerate(m_list):
        a = a_val
        b = int(b_list[r]) % int(m)
        for k in range(int(m)):
            th1 = 2*np.pi*a*k/m
            th2 = 2*np.pi*b*k/m
            rep_idx[p] = r
            c1[p] = np.cos(th1); s1[p] = np.sin(th1)
            c2[p] = np.cos(th2); s2[p] = np.sin(th2)
            p += 1
    return {"rep_idx": rep_idx, "c1": c1, "s1": s1, "c2": c2, "s2": s2, "R": R, "N": N, "m_list": list(m_list), "b_list": list(b_list)}

def gen_X(reps, orb):
    ri = orb["rep_idx"]; c1 = orb["c1"]; s1 = orb["s1"]; c2 = orb["c2"]; s2 = orb["s2"]
    R0 = reps[ri, 0]; R1 = reps[ri, 1]; R2 = reps[ri, 2]; R3 = reps[ri, 3]
    N = ri.shape[0]
    X = np.empty((N, 4))
    X[:, 0] = c1*R0 - s1*R1
    X[:, 1] = s1*R0 + c1*R1
    X[:, 2] = c2*R2 - s2*R3
    X[:, 3] = s2*R2 + c2*R3
    return X

def sym_relax(reps, orb, iters, k0, k1, lr0, lr1, mu, rng, noise=0.0):
    reps = reps / np.linalg.norm(reps, axis=1, keepdims=True)
    R = reps.shape[0]
    ri = orb["rep_idx"]; c1 = orb["c1"]; s1 = orb["s1"]; c2 = orb["c2"]; s2 = orb["s2"]
    N = ri.shape[0]
    vel = np.zeros_like(reps)
    for t in range(iters):
        frac = t / max(1, iters-1)
        k = k0 * (k1/k0)**frac
        lr = lr0 * (lr1/lr0)**frac
        # generate
        R0 = reps[ri,0]; R1 = reps[ri,1]; R2 = reps[ri,2]; R3 = reps[ri,3]
        X0 = c1*R0 - s1*R1
        X1 = s1*R0 + c1*R1
        X2 = c2*R2 - s2*R3
        X3 = s2*R2 + c2*R3
        X = np.column_stack([X0,X1,X2,X3])
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        moff = float(G.max())
        W = np.exp(k*(G-moff))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True)+1e-12
        Gx = (W @ X)/rs  # (N,4)
        g0 = Gx[:,0]; g1 = Gx[:,1]; g2 = Gx[:,2]; g3 = Gx[:,3]
        # pullback via inverse rotation
        h0 =  c1*g0 + s1*g1
        h1 = -s1*g0 + c1*g1
        h2 =  c2*g2 + s2*g3
        h3 = -s2*g2 + c2*g3
        grad = np.zeros_like(reps)
        np.add.at(grad[:,0], ri, h0)
        np.add.at(grad[:,1], ri, h1)
        np.add.at(grad[:,2], ri, h2)
        np.add.at(grad[:,3], ri, h3)
        # tangent project
        dot = np.sum(grad*reps, axis=1, keepdims=True)
        grad = grad - dot*reps
        # normalize grad scale by orbit size approx to keep lr stable
        # divide by N/R average multiplicity
        grad = grad / (N/float(R))
        vel = mu*vel + grad
        if mu > 0:
            vc = vel/(1.0 - mu**(t+1))
        else:
            vc = vel
        reps = reps - lr*vc
        if noise > 0 and frac < 0.6:
            amp = noise*(1.0-frac/0.6)
            pert = rng.standard_normal(reps.shape)
            pert -= np.sum(pert*reps,axis=1,keepdims=True)*reps
            reps = reps + amp*pert
        reps = reps/np.linalg.norm(reps,axis=1,keepdims=True)
    return reps

def smooth_relax(x, iters, k0, k1, lr0, lr1, mu, noise, rng):
    n,d = x.shape
    v = np.zeros_like(x)
    for t in range(iters):
        frac = t/max(1,iters-1)
        k = k0*(k1/k0)**frac
        lr = lr0*(lr1/lr0)**frac
        G = x @ x.T
        np.fill_diagonal(G,-2.0)
        moff = float(G.max())
        W = np.exp(k*(G-moff))
        np.fill_diagonal(W,0.0)
        rs = W.sum(axis=1,keepdims=True)+1e-12
        grad = (W@x)/rs
        grad -= np.sum(grad*x,axis=1,keepdims=True)*x
        v = mu*v+grad
        vc = v/(1.0-mu**(t+1)) if mu>0 else v
        x = x - lr*vc
        if noise>0 and frac<0.6:
            amp = noise*(1.0-frac/0.6)
            pert = rng.standard_normal(x.shape)
            pert -= np.sum(pert*x,axis=1,keepdims=True)*x
            x = x+amp*pert
        x = x/np.linalg.norm(x,axis=1,keepdims=True)
    return x

def lp_polish(x, seconds):
    x = x/np.linalg.norm(x,axis=1,keepdims=True)
    n,d = x.shape
    best = max_cos(x)
    radius,start = 1e-3,time.time()
    iu = np.triu_indices(n,1)
    while time.time()-start < seconds and radius>1e-14:
        c = (x@x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20*radius,1e-9)
        I,J = iu[0][sel],iu[1][sel]
        p,nv = len(I), n*d+1
        if p==0:
            break
        rows = np.repeat(np.arange(p),2*d+1)
        cols = np.concatenate([I[:,None]*d+np.arange(d),J[:,None]*d+np.arange(d),np.full((p,1),n*d)],1).ravel()
        vals = np.concatenate([x[J],x[I],-np.ones((p,1))],1).ravel()
        try:
            A_ub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
            obj = np.zeros(nv); obj[-1]=1.0
            res = linprog(obj,A_ub=A_ub,b_ub=-c[sel],A_eq=A_eq,b_eq=np.zeros(n),bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5
            continue
        if res.status!=0:
            radius*=0.5
            continue
        y = x+res.x[:-1].reshape(n,d)
        y/=np.linalg.norm(y,axis=1,keepdims=True)
        vv=max_cos(y)
        if vv<best:
            x,best,radius=y,vv,min(radius*1.5,0.05)
        else:
            radius*=0.5
    return x

def solve():
    t0=time.time()
    rng=np.random.default_rng(0)
    n,d=153,4
    try:
        best=np.load("best.npy")
    except Exception:
        try:
            best=np.load("record.npy")
        except Exception:
            best=rng.standard_normal((n,d))
    best=np.asarray(best,dtype=float)
    if best.shape!=(n,d):
        best=rng.standard_normal((n,d))
    best=best/np.linalg.norm(best,axis=1,keepdims=True)
    best_val=max_cos(best)
    incumbent=best.copy(); inc_val=best_val

    # ---- symmetric screening ----
    # time budget for screening
    SCREEN_END=t0+65
    # keep top symmetric candidates (reps+orb -> X)
    sym_cands=[]  # (val, Xsym, reps copy, orb copy)
    trial=0
    # define families to cycle
    while time.time()<SCREEN_END:
        trial+=1
        fam=rng.random()
        if fam<0.55:
            # 9x17
            m_list=[17]*9
            b_list=[int(rng.integers(0,17)) for _ in range(9)]
            # ensure not all zero; set a=1 so fine
            # randomize: force at least some nonzero
            orb=build_orb(m_list,b_list,a_val=1)
            R=9
        elif fam<0.8:
            # 3x51
            m_list=[51]*3
            b_list=[int(rng.integers(0,51)) for _ in range(3)]
            orb=build_orb(m_list,b_list,a_val=1)
            R=3
        else:
            # mixed 2x51 + 3x17 =153
            m_list=[51,51,17,17,17]
            # b must be mod m
            b_list=[int(rng.integers(0,m)) for m in m_list]
            orb=build_orb(m_list,b_list,a_val=1)
            R=5
        reps=rng.standard_normal((R,4))
        # enforce both complex coords nonzero to keep full orbit size
        # if a component too small, add offset
        for r in range(R):
            n1=np.linalg.norm(reps[r,0:2]); n2=np.linalg.norm(reps[r,2:4])
            if n1<0.2:
                reps[r,0]+=0.5; reps[r,1]+=0.3
            if n2<0.2:
                reps[r,2]+=0.5; reps[r,3]+=0.3
        reps/=np.linalg.norm(reps,axis=1,keepdims=True)
        # cheap relax
        k0=float(rng.choice(np.array([8.0,12.0,18.0])))
        k1=float(rng.choice(np.array([120.0,200.0,300.0])))
        lr0=float(rng.choice(np.array([0.05,0.1,0.2])))
        lr1=float(rng.choice(np.array([0.005,0.01,0.02])))
        mu=float(rng.choice(np.array([0.0,0.8,0.9])))
        iters=int(rng.choice(np.array([250,350,500])))
        try:
            reps2=sym_relax(reps,orb,iters,k0,k1,lr0,lr1,mu,rng,noise=0.0)
            X=gen_X(reps2,orb)
            v=max_cos(X)
        except Exception:
            continue
        if len(sym_cands)<6 or v<max(z[0] for z in sym_cands):
            sym_cands.append((v,X.copy(),reps2.copy(),orb))
            sym_cands=sorted(sym_cands,key=lambda z:z[0])[:6]
        # also track global incumbent if symmetric already good (rare)
        if v<inc_val-1e-9:
            # will refine later; keep
            pass
        if trial>250:
            break
        if time.time()>SCREEN_END-1:
            break

    # ---- fully relax best few symmetric ----
    REFINE_END=t0+100
    refined=[]
    for (v,X0,reps0,orb) in sym_cands[:4]:
        if time.time()>REFINE_END-3:
            break
        try:
            reps1=sym_relax(reps0.copy(),orb,1200,12.0,350.0,0.08,0.004,0.85,rng,noise=0.0)
            X1=gen_X(reps1,orb)
            # second stage higher k
            reps2=sym_relax(reps1,orb,800,30.0,600.0,0.03,0.002,0.9,rng,noise=0.0)
            X2=gen_X(reps2,orb)
            v2=max_cos(X2)
            refined.append((v2,X2,reps2,orb))
        except Exception:
            continue
    if len(refined)==0 and len(sym_cands)>0:
        for (v,X0,reps0,orb) in sym_cands[:2]:
            refined.append((v,X0,reps0,orb))
    refined=sorted(refined,key=lambda z:z[0]) if refined else []

    # ---- break symmetry gradually + smooth relax ----
    # take top 3 refined, break with amplitudes 0.05-0.15 schedule
    broken=[]
    for idx,(v,Xs,_,_) in enumerate(refined[:3]):
        if time.time()>t0+125:
            break
        try:
            # gradual: two steps
            Xs_n=Xs/np.linalg.norm(Xs,axis=1,keepdims=True)
            # step1 small perturb + relax
            amp1=0.03+0.02*idx  # 0.03..0.07
            Xb=Xs_n+amp1*rng.standard_normal(Xs_n.shape)
            Xb/=np.linalg.norm(Xb,axis=1,keepdims=True)
            Xb=smooth_relax(Xb,700,12.0,250.0,0.04,0.002,0.85,0.0,rng)
            # step2 larger perturb + relax
            amp2=0.05+0.03*idx  # up to ~0.11
            # perturb again but keep best?
            Xc=Xb+amp2*0.5*rng.standard_normal(Xb.shape)
            Xc/=np.linalg.norm(Xc,axis=1,keepdims=True)
            Xc=smooth_relax(Xc,900,15.0,300.0,0.03,0.0015,0.9,0.0,rng)
            vcb=max_cos(Xc)
            # also try direct larger single break
            amp3=float(rng.uniform(0.05,0.15))
            Xd=Xs_n+amp3*rng.standard_normal(Xs_n.shape)
            Xd/=np.linalg.norm(Xd,axis=1,keepdims=True)
            Xd=smooth_relax(Xd,900,15.0,300.0,0.035,0.0015,0.85,0.0,rng)
            vdb=max_cos(Xd)
            if vdb<vcb:
                Xc,vCb=Xd,vdb
            else:
                vCb=vcb
            broken.append((vCb,Xc))
            if vCb<inc_val-1e-12:
                inc_val=vCb; incumbent=Xc.copy()
        except Exception:
            continue

    # also do a couple of generic basin hops from incumbent/best to not regress
    # quick smooth on incumbent variants if time allows
    if time.time()<t0+130:
        for _ in range(2):
            if time.time()>t0+130:
                break
            try:
                base=incumbent if rng.random()<0.5 else best
                sigma=float(rng.choice(np.array([0.02,0.06,0.12])))
                Xx=base+sigma*rng.standard_normal(base.shape)
                Xx/=np.linalg.norm(Xx,axis=1,keepdims=True)
                Xx=smooth_relax(Xx,600,12.0,250.0,0.035,0.002,0.85,0.0,rng)
                vv=max_cos(Xx)
                broken.append((vv,Xx))
                if vv<inc_val:
                    inc_val=vv; incumbent=Xx.copy()
            except Exception:
                continue

    # ---- final LP polishing with remaining time ----
    cands=[]
    cands.append((best_val,best.copy()))
    cands.append((inc_val,incumbent.copy()))
    for (vv,xx) in broken:
        cands.append((vv,xx))
    # dedup sort
    cands=sorted(cands,key=lambda z:z[0])[:3]
    time_left=170-(time.time()-t0)
    if time_left<8:
        time_left=8
    per=time_left/len(cands)
    polished_best=None; polished_val=1e9
    for i,(vv,xx) in enumerate(cands):
        budget=per*(1.4 if i==0 else 1.0)
        budget=min(budget,50)
        if time.time()-t0>172:
            break
        try:
            xp=lp_polish(xx,budget)
        except Exception:
            xp=xx
        try:
            vp=max_cos(xp)
        except Exception:
            continue
        if vp<polished_val:
            polished_val=vp; polished_best=xp.copy()
    if polished_best is None:
        polished_best=best.copy()
    polished_best/=np.linalg.norm(polished_best,axis=1,keepdims=True)
    # safety: return best known if polish failed to improve
    # still return polished_best if it is valid (it is), but choose min
    if best_val<max_cos(polished_best):
        return best/np.linalg.norm(best,axis=1,keepdims=True)
    return polished_best
