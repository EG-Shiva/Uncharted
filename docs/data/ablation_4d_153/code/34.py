import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

_TRIU = None
def _triu(n):
    global _TRIU
    if _TRIU is None or _TRIU[0].shape[0] != n*(n-1)//2:
        _TRIU = np.triu_indices(n, 1)
    return _TRIU

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def lp_polish(x, t_end, r0=0.003):
    x = _norm_rows(x.copy())
    n, d = x.shape
    iu = _triu(n)
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = float(r0)
    it = 0
    while time.time() < t_end and radius > 1e-14 and it < 80:
        it += 1
        G = x @ x.T
        c = G[iu]
        m = float(np.max(c))
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        if int(np.sum(sel)) == 0:
            radius *= 0.5
            continue
        idx_all = np.where(sel)[0]
        if idx_all.shape[0] > 2500:
            part = np.argpartition(c[idx_all], -2500)[-2500:]
            idx = idx_all[part]
        else:
            idx = idx_all
        p = idx.shape[0]
        I = iu[0][idx]; J = iu[1][idx]
        csel = c[idx]
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None]*d + np.arange(d),
                                   J[:, None]*d + np.arange(d),
                                   np.full((p,1), n*d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p,1))], 1).ravel()
            A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))), shape=(n,nv)).tocsr()
            obj = np.zeros(nv); obj[-1]=1.0
            bnds = [(-radius,radius)]*(n*d)+[(None,None)]
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bnds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            dx = res.x[:-1].reshape(n,d)
        except Exception:
            radius *= 0.5
            continue
        y = _norm_rows(x+dx)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius*1.5, 0.05)
        else:
            radius *= 0.5
    return x

def build_orbits(S, m, q):
    # S: (K,4) unit, returns (K*m,4)
    K = S.shape[0]
    js = np.arange(m)
    a1 = 2*np.pi*js/m
    a2 = 2*np.pi*q*js/m
    ca = np.cos(a1); sa=np.sin(a1); cb=np.cos(a2); sb=np.sin(a2)
    s1 = S[:,0]; s2=S[:,1]; s3=S[:,2]; s4=S[:,3]
    # (K,m)
    X1 = s1[:,None]*ca[None,:] - s2[:,None]*sa[None,:]
    X2 = s1[:,None]*sa[None,:] + s2[:,None]*ca[None,:]
    X3 = s3[:,None]*cb[None,:] - s4[:,None]*sb[None,:]
    X4 = s3[:,None]*sb[None,:] + s4[:,None]*cb[None,:]
    X = np.empty((K,m,4), dtype=np.float64)
    X[:,:,0]=X1; X[:,:,1]=X2; X[:,:,2]=X3; X[:,:,3]=X4
    return X.reshape(K*m,4)

def seed_quench(S0, m, q, steps, beta0, beta1, lr0, lr1, max_step=0.02):
    S = _norm_rows(S0.copy())
    K = S.shape[0]
    n = K*m
    js = np.arange(m)
    a1 = 2*np.pi*js/m; a2=2*np.pi*q*js/m
    ca=np.cos(a1); sa=np.sin(a1); cb=np.cos(a2); sb=np.sin(a2)
    vel = np.zeros_like(S)
    mom = 0.85
    lb0, lb1 = np.log(beta0), np.log(beta1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    bestS = S.copy()
    # evaluate initial
    Xb = build_orbits(S,m,q)
    bestv = _max_cos(Xb)
    for t in range(int(steps)):
        f = t/max(1,int(steps)-1)
        beta = float(np.exp(lb0+(lb1-lb0)*f))
        lr = float(np.exp(ll0+(ll1-ll0)*f))
        X = build_orbits(S,m,q)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = float(np.max(G))
        W = np.exp(beta*(G-mx))
        np.fill_diagonal(W, 0.0)
        ssum = float(np.sum(W))
        if ssum <= 0:
            ssum = 1.0
        gradX = (W @ X)/ssum  # (n,4)
        g3 = gradX.reshape(K,m,4)
        g1 = g3[:,:,0]; g2=g3[:,:,1]; g3c=g3[:,:,2]; g4=g3[:,:,3]
        gs1 = np.sum(g1*ca[None,:]+g2*sa[None,:], axis=1)
        gs2 = np.sum(-g1*sa[None,:]+g2*ca[None,:], axis=1)
        gs3 = np.sum(g3c*cb[None,:]+g4*sb[None,:], axis=1)
        gs4 = np.sum(-g3c*sb[None,:]+g4*cb[None,:], axis=1)
        gradS = np.stack([gs1,gs2,gs3,gs4], axis=1)
        dot = np.sum(gradS*S, axis=1, keepdims=True)
        gradS = gradS - dot*S
        vel = mom*vel + (1.0-mom)*gradS
        step = lr*vel
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        over = sn > max_step
        if np.any(over):
            step = step * np.where(over, max_step/np.maximum(sn,1e-15), 1.0)
        S = _norm_rows(S - step)
        # track best by max cos occasionally
        if (t % 50 == 49) or (t == int(steps)-1):
            Xc = build_orbits(S,m,q)
            # duplicate-orbit rejection: if coincident points, skip
            vc = _max_cos(Xc)
            if vc < bestv:
                bestv = vc
                bestS = S.copy()
    return bestS, bestv

def full_quench(X0, steps, beta, lr, max_step=0.02):
    X = _norm_rows(X0.copy())
    for _ in range(int(steps)):
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = float(np.max(G))
        W = np.exp(beta*(G-mx))
        np.fill_diagonal(W, 0.0)
        grad = W @ X
        proj = np.sum(grad*X, axis=1, keepdims=True)
        gt = grad - proj*X
        # scale: normalize by sum to keep lr stable? use raw with small lr
        # divide by sum for stability when beta large
        ssum = float(np.sum(W))
        if ssum > 1e-12:
            gt = gt/ssum*X.shape[0]
        step = lr*gt
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        over = sn > max_step
        if np.any(over):
            step = step*np.where(over, max_step/np.maximum(sn,1e-15), 1.0)
        X = _norm_rows(X - step)
    return X

def solve():
    t0 = time.time()
    T_END = t0 + 168.0
    PHASE_A_END = t0 + 60.0
    rng = np.random.default_rng(20240517)
    n,d = 153,4
    x_best=None; best_val=1e9
    for fn in ("best.npy","record.npy"):
        try:
            a=np.load(fn)
            a=np.asarray(a,dtype=np.float64)
            if a.shape==(n,d):
                a=_norm_rows(a)
                v=_max_cos(a)
                if v<best_val:
                    best_val=v; x_best=a.copy()
        except Exception:
            pass
    if x_best is None:
        x_best=_norm_rows(rng.standard_normal(size=(n,d)))
        best_val=_max_cos(x_best)
    # quick baseline polish
    try:
        if time.time()<T_END-20:
            x_best = lp_polish(x_best, min(time.time()+4.0, T_END-15), r0=0.002)
            best_val=_max_cos(x_best)
    except Exception:
        pass
    # ---- Phase A: symmetry-constrained search ----
    configs=[]
    for q in [1,2,4,5,7,10,13]:
        configs.append((51,3,q))
    for q in [1,2,3,4,5,6]:
        configs.append((17,9,q))
    configs.append((153,1,4))
    configs.append((153,1,7))
    sym_cands=[]  # list of (val, X)
    # random order but deterministic
    for (m,K,q) in configs:
        if time.time()>PHASE_A_END:
            break
        # two restarts per config
        for rs in range(2):
            if time.time()>PHASE_A_END:
                break
            # init seeds with duplicate rejection
            S0=None
            for trial in range(20):
                St = _norm_rows(rng.standard_normal(size=(K,d)))
                # fix gauge: rotate first seed th1->0? enforce by rotating first seed's first plane
                # rotate seed0 so that its (x1,x2) angle =0
                ang = np.arctan2(St[0,1],St[0,0])
                ca0=np.cos(ang); sa0=np.sin(ang)
                # apply inverse rotation to seed0 only? Instead apply global? Just set:
                r = float(np.sqrt(St[0,0]**2+St[0,1]**2))
                St[0,0]=r; St[0,1]=0.0
                # renormalize (r unchanged norm of first plane, total norm still 1)
                Xt = build_orbits(St,m,q)
                if _max_cos(Xt) < 0.999:
                    S0=St
                    break
            if S0 is None:
                continue
            try:
                Sopt, vopt = seed_quench(S0,m,q,steps=550,beta0=25.0,beta1=160.0,lr0=3.0,lr1=0.6)
                Xopt = build_orbits(Sopt,m,q)
                v = _max_cos(Xopt)
                # light full-free quench to judge basin? no, keep symmetric value
                sym_cands.append((v,Xopt.copy(),(m,K,q)))
            except Exception:
                continue
    sym_cands.sort(key=lambda t: t[0])
    # keep top 3
    top_sym = sym_cands[:3] if len(sym_cands)>0 else []
    # ---- Phase B: staged symmetry breaking ----
    cur_sym_best = None
    if len(top_sym)>0:
        cur_sym_best = top_sym[0][1].copy()
        # staged scales
        scales=[0.001,0.003,0.01,0.03]
        Xcur = cur_sym_best.copy()
        for si,sc in enumerate(scales):
            if time.time()>T_END-25:
                break
            try:
                noise = rng.standard_normal(size=(n,d))
                dot=np.sum(noise*Xcur,axis=1,keepdims=True)
                noise=noise-dot*Xcur
                nn=np.linalg.norm(noise,axis=1,keepdims=True)+1e-15
                noise=noise/nn*sc*np.sqrt(3)  # scale ~ tangent magnitude
                # random magnitude variation
                Xp=_norm_rows(Xcur+noise)
                # softmax quench
                beta = 90.0 if si<2 else 70.0
                Xq=full_quench(Xp, steps=220, beta=beta, lr=0.35)
                # LP polish
                Xp2=lp_polish(Xq, min(time.time()+3.0, T_END-20), r0=0.004)
                v2=_max_cos(Xp2)
                # accept if not catastrophically worse to allow topology change,
                # but track global best
                if v2<best_val-1e-13:
                    best_val=v2; x_best=Xp2.copy()
                # continue chain from best of this stage (allow uphill with limit)
                # if stage improved over chain, use it, else use quenched with small uphill
                vc=_max_cos(Xcur)
                if v2<vc+0.005:
                    Xcur=Xp2.copy()
                else:
                    # retry with smaller noise once
                    pass
            except Exception:
                continue
        # also try breaking second-best sym seed if time
        if len(top_sym)>1 and time.time()<T_END-40:
            try:
                X2=top_sym[1][1].copy()
                noise=rng.standard_normal(size=(n,d))*0.008
                dot=np.sum(noise*X2,axis=1,keepdims=True)
                noise=noise-dot*X2
                X2n=_norm_rows(X2+noise)
                X2q=full_quench(X2n, steps=250, beta=80.0, lr=0.35)
                X2p=lp_polish(X2q, min(time.time()+3.0, T_END-20), r0=0.004)
                v2=_max_cos(X2p)
                if v2<best_val:
                    best_val=v2; x_best=X2p.copy()
                    Xcur=x_best.copy()
            except Exception:
                pass
    else:
        Xcur=x_best.copy()
    # ---- Remaining time: unrestricted polishing / basin hopping from best ----
    x_cur = x_best.copy(); cur_val=best_val
    it=0
    while time.time()<T_END-3.0:
        it+=1
        if time.time()>T_END-4:
            break
        try:
            r=rng.random()
            if r<0.5:
                sigma=float(rng.choice([0.004,0.01,0.025,0.06]))
                noise=rng.standard_normal(size=(n,d))
                dot=np.sum(noise*x_cur,axis=1,keepdims=True)
                noise=noise-dot*x_cur
                y=_norm_rows(x_cur+sigma*noise)
                y=full_quench(y, steps=90, beta=85.0, lr=0.35)
            elif r<0.75:
                # subset reinit worst rows
                G=x_cur@x_cur.T; np.fill_diagonal(G,-2.0)
                mrow=np.max(G,axis=1)
                k=int(rng.integers(1,4))
                order=np.argsort(-mrow)
                pool=order[:14]
                idx=rng.choice(pool,size=k,replace=False)
                y=x_cur.copy()
                y[idx]=rng.standard_normal(size=(k,d))
                y=_norm_rows(y)
                y=full_quench(y, steps=80, beta=70.0, lr=0.35)
            else:
                sigma=0.015
                noise=rng.standard_normal(size=(n,d))
                dot=np.sum(noise*top_sym[0][1] if len(top_sym)>0 else x_cur,axis=1,keepdims=True) if False else np.sum(noise*x_cur,axis=1,keepdims=True)
                noise=noise-dot*(x_cur)
                y=_norm_rows(x_cur+sigma*noise)
                y=full_quench(y, steps=120, beta=110.0, lr=0.3)
            y2=lp_polish(y, min(time.time()+1.6, T_END-2.5), r0=0.004)
            v2=_max_cos(y2)
            if v2<best_val-1e-13:
                best_val=v2; x_best=y2.copy(); x_cur=y2.copy(); cur_val=v2
            elif v2<cur_val or rng.random()<0.08:
                x_cur=y2.copy(); cur_val=v2
        except Exception:
            continue
    return _norm_rows(x_best)
