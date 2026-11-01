import time
import numpy as np

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _topk_sorted(x, k=50):
    g = x @ x.T
    iu = np.triu_indices(x.shape[0], 1)
    v = g[iu]
    if len(v) <= k:
        return np.sort(v)[::-1]
    part = np.partition(v, len(v)-k)[len(v)-k:]
    return np.sort(part)[::-1]

def _lex_better(a, b, tol=1e-12):
    # True if a < b lexicographically
    m = min(len(a), len(b))
    for i in range(m):
        d = float(a[i]) - float(b[i])
        if d < -tol:
            return True
        if d > tol:
            return False
    return False

def _quench_lse(X_init, steps, T0, T1, lr, mu=0.88, deadline=None):
    X = X_init.copy()
    n, d = X.shape
    V = np.zeros_like(X)
    lt0 = np.log(max(T0,1e-9)); lt1 = np.log(max(T1,1e-9))
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best_m = float(G0.max())
    best_X = X.copy()
    for s in range(steps):
        if deadline is not None and (s % 200 == 0) and time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -1e100)
        m = float(G.max())
        if m < best_m:
            best_m = m
            best_X = X.copy()
        Q = (G - m) / T
        np.clip(Q, -50, 0, out=Q)
        E = np.exp(Q)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        Grad = (E @ X) / rs
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        bad = gn[:,0] > 2.0
        if np.any(bad):
            Gt[bad] *= (2.0 / gn[bad])
        V = mu * V + Gt
        X = X - lr * V
        X = _norm_rows(X)
    Gf = X @ X.T
    np.fill_diagonal(Gf, -2.0)
    mf = float(Gf.max())
    if mf < best_m:
        return X
    return best_X

def _lp_eps_polish(x0, eps_schedule, seconds, deadline=None, verbose=False):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = _norm_rows(x0.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    start = time.time()
    iu = np.triu_indices(n, 1)
    for eps in eps_schedule:
        if time.time() - start > seconds:
            break
        if deadline is not None and time.time() > deadline:
            break
        radius = min(3e-3, max(5e-4, eps*5))
        radius = min(radius, 3e-3)
        it = 0
        while time.time() - start < seconds:
            if deadline is not None and time.time() > deadline:
                break
            if radius < 1e-13:
                break
            if it > 12:
                break
            it += 1
            G = x @ x.T
            cv = G[iu]
            m = float(cv.max())
            # epsilon-active set: slack < eps, always include tightest 50
            sel = cv > (m - eps)
            cnt = int(np.count_nonzero(sel))
            if cnt < 60:
                # pad with tightest
                order = np.argsort(-cv)[:60]
                msk = np.zeros_like(sel, dtype=bool)
                msk[order] = True
                sel = sel | msk
                cnt = int(np.count_nonzero(sel))
            if cnt > 1500:
                order = np.argsort(-cv)
                keep = np.zeros_like(sel, dtype=bool)
                # take tightest 1500 among selected? just tightest 1500
                kk = order[:1500]
                sel = np.zeros_like(sel, dtype=bool)
                sel[kk] = True
                cnt = 1500
            I = iu[0][sel]; J = iu[1][sel]
            csub = cv[sel]
            p = cnt
            nv = n*d + 1
            rows = np.repeat(np.arange(p), 2*d+1)
            cols = np.concatenate([I[:,None]*d + np.arange(d),
                                   J[:,None]*d + np.arange(d),
                                   np.full((p,1), n*d)],1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
            try:
                A_ub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
                A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            obj = np.zeros(nv); obj[-1]=1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius,radius)]*(n*d)+[(-2.0, m+1e-6)], method="highs",
                              options={"presolve":True})
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            dx = res.x[:-1].reshape(n,d)
            # exact-max line search along dx (alpha in [0,1])
            best_a = 0.0
            best_v = best
            best_X = x
            for a in (0.25,0.5,0.75,1.0,0.125,0.375,0.625,0.875):
                y = _norm_rows(x + a*dx)
                v = _max_cos(y)
                if v < best_v:
                    best_v = v; best_a = a; best_X = y
            if best_v < best - 1e-14:
                x, best = best_X, best_v
                radius = min(radius*1.6, 0.02)
            else:
                # try full step already evaluated; no gain
                radius *= 0.5
                if radius < 2e-7 and eps > 1e-6:
                    break
    return x

def _soft_directions(X, eps=1.2e-4, kkeep=40):
    n, d = X.shape
    G = X @ X.T
    iu = np.triu_indices(n,1)
    cv = G[iu]
    m = float(cv.max())
    sel = cv > (m - eps)
    I = iu[0][sel]; J = iu[1][sel]
    p = len(I)
    # tangent bases Bs: (n,d,d-1)
    dm1 = d-1
    Bs = np.zeros((n,d,dm1))
    Bbig = np.zeros((n*d, n*dm1))
    for i in range(n):
        xi = X[i]
        am = int(np.argmax(np.abs(xi)))
        cols = []
        for j in range(d):
            if j==am: continue
            e = np.zeros(d); e[j]=1.0
            pj = e - float(e@xi)*xi
            cols.append(pj)
        M = np.stack(cols,axis=1)  # d x dm1
        try:
            Q,R = np.linalg.qr(M)
        except Exception:
            Q,_ = np.linalg.qr(np.random.randn(d,dm1))
        Bs[i]=Q
        Bbig[i*d:(i+1)*d, i*dm1:(i+1)*dm1]=Q
    Nt = n*dm1
    # rigidity Jacobian projected: Jt p x Nt
    Jt = np.zeros((p,Nt))
    for r in range(p):
        i=int(I[r]); j=int(J[r])
        Jt[r, i*dm1:(i+1)*dm1] = X[j] @ Bs[i]
        Jt[r, j*dm1:(j+1)*dm1] = X[i] @ Bs[j]
    # global rotations: Rfull nd x 66
    pairs=[]
    for a in range(d):
        for b in range(a+1,d):
            pairs.append((a,b))
    nrot=len(pairs)
    Rfull = np.zeros((n*d,nrot))
    for c,(a,b) in enumerate(pairs):
        for i in range(n):
            Rfull[i*d+a,c]=X[i,b]
            Rfull[i*d+b,c]=-X[i,a]
    Rb = Bbig.T @ Rfull  # Nt x nrot
    try:
        Ur,Sr,Vtr = np.linalg.svd(Rb, full_matrices=True)
    except Exception:
        return None
    rank_rot = int(np.sum(Sr>1e-10))
    C = Ur[:,rank_rot:]  # Nt x (Nt-rank)
    if C.shape[1]==0:
        return None
    Jc = Jt @ C  # p x Nc
    try:
        Uj,Sj,Vtj = np.linalg.svd(Jc, full_matrices=False)
    except Exception:
        return None
    rank_j = int(np.sum(Sj>1e-9))
    # bottom kkeep
    Nc = C.shape[1]
    kk = min(kkeep, Nc)
    Vsmall_c = Vtj[Nc-kk:,:].T  # Nc x kk (smallest last)
    # order smallest first
    Vsmall_c = Vsmall_c[:,::-1]
    Vt_small = C @ Vsmall_c  # Nt x kk
    Vamb = Bbig @ Vt_small  # nd x kk
    # contacts at 5e-8
    contacts = int(np.sum(cv > (m-5e-8)))
    return {"I":I,"J":J,"p":p,"m":m,"Sj":Sj,"rank_j":rank_j,"rank_rot":rank_rot,
            "Vamb":Vamb,"contacts":contacts,"Nt":Nt}

def _single_refine(s, R, steps=200, T=0.004, lr=0.12, mu=0.8):
    s = s/(np.linalg.norm(s)+1e-12)
    v = np.zeros_like(s)
    for _ in range(steps):
        g = R @ s
        m = float(g.max())
        e = np.exp(np.clip((g-m)/T,-50,0))
        den = float(e.sum())+1e-30
        Grad = (R.T @ e)/den
        Gt = Grad - float(Grad@s)*s
        v = mu*v+Gt
        s = s - lr*v
        nn=float(np.linalg.norm(s))
        if not np.isfinite(nn) or nn==0:
            break
        s=s/nn
    return s

def _find_hole(R, rng, n_sample=6000, n_cand=2):
    d=R.shape[1]
    Z=_norm_rows(rng.standard_normal(size=(n_sample,d)))
    S=Z@R.T
    scores=S.max(axis=1)
    idx=np.argpartition(scores,n_cand)[:n_cand]
    best_s=None; best_v=1e100
    for kk in idx:
        s0=Z[kk]
        s1=_single_refine(s0,R,steps=180,T=0.005,lr=0.12)
        s1=_single_refine(s1,R,steps=180,T=0.0015,lr=0.07)
        v=float((R@s1).max())
        if v<best_v:
            best_v=v; best_s=s1
    return best_s/(np.linalg.norm(best_s)+1e-12)

def solve():
    t0=time.time()
    HARD_END=t0+172.0
    EXPLORE_END=t0+142.0
    rng=np.random.default_rng(0)
    cands=[]
    for fn in ("best.npy","record.npy"):
        try:
            a=np.asarray(np.load(fn),dtype=np.float64)
            if a.shape==(71,12):
                cands.append(_norm_rows(a))
        except Exception:
            pass
    if len(cands)==0:
        cands.append(_norm_rows(rng.standard_normal(size=(71,12))))
    bestX=min(cands,key=_max_cos).copy()
    bestVal=float(_max_cos(bestX))
    bestTop=_topk_sorted(bestX,50)
    curX=bestX.copy(); curTop=bestTop.copy(); curVal=bestVal
    n,d=bestX.shape
    # initial rigidity-aware polish
    if HARD_END-time.time()>6:
        try:
            Xp=_lp_eps_polish(bestX,[2e-4,5e-5,1e-5,2e-6],seconds=6.0,deadline=HARD_END-2)
            vp=float(_max_cos(Xp)); tp=_topk_sorted(Xp,50)
            if vp<bestVal-1e-14 or _lex_better(tp,bestTop):
                bestX, bestVal, bestTop = Xp, vp, tp
                curX, curTop, curVal = Xp.copy(), tp.copy(), vp
        except Exception:
            pass
    trial=0
    sigmas=[0.004,0.008,0.015,0.025,0.010]
    while time.time()<EXPLORE_END and time.time()<HARD_END-26 and trial<400:
        trial+=1
        # choose mode
        r=rng.random()
        X0=None
        use_soft=True
        if r<0.68:
            # rigidity null kick
            eps = float(rng.choice([8e-5,1.2e-4,2e-4,5e-5]))
            try:
                info=_soft_directions(curX,eps=eps,kkeep=40)
            except Exception:
                info=None
            if info is None or info["p"]<50:
                # fallback isotropic tangent kick
                sigma=sigmas[trial%len(sigmas)]
                noise=rng.standard_normal(size=curX.shape)
                noise= noise - (np.sum(noise*curX,axis=1,keepdims=True))*curX
                X0=_norm_rows(curX+sigma*noise)
            else:
                if trial%12==1:
                    print(f"[rigid] trial {trial} m={info['m']:.9f} p_eps={info['p']} contacts5e-8={info['contacts']} rankJ={info['rank_j']}/{info['Nt']-66} smin={float(info['Sj'][-1]):.2e} smax={float(info['Sj'][0]):.2e}")
                Vamb=info["Vamb"]  # nd x kk
                kk=Vamb.shape[1]
                coeff=rng.standard_normal(size=kk)
                coeff/= (np.linalg.norm(coeff)+1e-12)
                # mix a couple of smallest more strongly
                w=np.exp(-0.15*np.arange(kk))
                coeff=coeff*w
                coeff/= (np.linalg.norm(coeff)+1e-12)
                D=(Vamb@coeff).reshape(n,d)
                dn=float(np.linalg.norm(D))
                if dn>0: D/=dn
                # D is tangent already (norm 1 total); scale to per-point rms
                sigma=sigmas[trial%len(sigmas)]*(0.7+0.6*rng.random())
                # small isotropic tangent noise
                noise=rng.standard_normal(size=curX.shape)
                noise=noise-(np.sum(noise*curX,axis=1,keepdims=True))*curX
                nn=float(np.linalg.norm(noise))
                if nn>0: noise/=nn
                X0=_norm_rows(curX+sigma*np.sqrt(float(n))*D+ (sigma*0.35)*np.sqrt(float(n))*noise)
        elif r<0.86:
            # hole re-insertion of worst point
            G=curX@curX.T
            np.fill_diagonal(G,-2.0)
            mx=G.max(axis=1)
            cnt=(G>curVal-1e-3).sum(axis=1)
            score=mx+0.0005*cnt
            victim=int(np.argsort(-score)[rng.integers(0,4)])
            R=np.delete(curX,victim,axis=0)
            hole=_find_hole(R,rng,n_sample=6000,n_cand=2)
            X0=curX.copy(); X0[victim]=hole
            noise=rng.standard_normal(size=X0.shape)
            noise=noise-(np.sum(noise*X0,axis=1,keepdims=True))*X0
            X0=_norm_rows(X0+0.006*noise)
        else:
            sigma=0.05+rng.random()*0.08
            noise=rng.standard_normal(size=curX.shape)
            X0=_norm_rows(curX+sigma*noise)
        if X0 is None:
            continue
        T0=float(np.exp(rng.uniform(np.log(0.008),np.log(0.025))))
        T1=float(np.exp(rng.uniform(np.log(0.0007),np.log(0.003))))
        lr=float(rng.uniform(0.04,0.10))
        mu=float(rng.uniform(0.82,0.92))
        steps=int(rng.integers(500,1100))
        Xq=_quench_lse(X0,steps=steps,T0=T0,T1=T1,lr=lr,mu=mu,deadline=HARD_END-22)
        vq=float(_max_cos(Xq))
        # only LP if promising lexicographically-ish (within 2e-3 of best)
        if vq < bestVal+0.002:
            try:
                Xp=_lp_eps_polish(Xq,[1e-4,1e-5,2e-6],seconds=0.7,deadline=HARD_END-20)
            except Exception:
                Xp=Xq
            vp=float(_max_cos(Xp)); tp=_topk_sorted(Xp,50)
        else:
            vp=vq; tp=_topk_sorted(Xq,50); Xp=Xq
        if _lex_better(tp,bestTop):
            bestX,bestVal,bestTop=Xp.copy(),vp,tp.copy()
            curX,curTop,curVal=Xp.copy(),tp.copy(),vp
        elif _lex_better(tp,curTop):
            curX,curTop,curVal=Xp.copy(),tp.copy(),vp
        else:
            # occasional sideways move
            if vp<curVal+0.0015 and rng.random()<0.08:
                curX,curTop,curVal=Xp.copy(),tp.copy(),vp
        # restart if stuck: perturb best
        if trial%40==0 and time.time()<EXPLORE_END:
            # nudge cur toward best + soft kick
            if rng.random()<0.7:
                curX=_norm_rows(bestX+0.01*rng.standard_normal(size=bestX.shape))
                curTop=_topk_sorted(curX,50); curVal=float(_max_cos(curX))
    # final tightening
    remain=HARD_END-time.time()-2.0
    if remain>3:
        try:
            Xf=_lp_eps_polish(bestX,[2e-4,5e-5,1e-5,1e-6,5e-8],seconds=min(remain-1,18.0),deadline=HARD_END-1.0)
            vf=float(_max_cos(Xf)); tf=_topk_sorted(Xf,50)
            if _lex_better(tf,bestTop) or vf<bestVal:
                bestX,bestVal,bestTop=Xf,vf,tf
        except Exception:
            pass
    bestX=_norm_rows(bestX)
    return bestX
