import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best_val = _max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    iu_r, iu_c = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-13:
        G = x @ x.T
        c_full = G[iu_r, iu_c]
        m = c_full.max()
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_full > thr
        p = int(sel.sum())
        if p == 0:
            radius *= 0.5
            continue
        if p > 2500:
            idx = np.argpartition(-c_full, 2500)[:2500]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = sel & mask
            p = 2500
        I = iu_r[sel]; J = iu_c[sel]
        c_sel = c_full[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None]*d+np.arange(d), J[:, None]*d+np.arange(d), np.full((p,1), n*d)],1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
        try:
            A_ub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
            obj = np.zeros(nv); obj[-1]=1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n,d)
        yn = np.linalg.norm(y,axis=1,keepdims=True)
        if np.any(~np.isfinite(yn)) or np.any(yn<1e-12):
            radius *= 0.5
            continue
        y /= yn
        v = _max_cos(y)
        if v < best_val:
            best_val=v; best_x=y.copy(); x=y
            radius=min(radius*1.5,0.02)
        else:
            radius*=0.5
    return best_x

def _batch_adam(Xp, steps, lr0, a0, a1, scales, deadline):
    B,n,d = Xp.shape
    m_ad = np.zeros_like(Xp); v_ad=np.zeros_like(Xp)
    Eye=np.eye(n)
    b1,b2,eps=0.9,0.999,1e-8
    for s in range(steps):
        if time.time()>deadline:
            break
        frac=s/max(steps-1,1)
        base_alpha=a0*((a1/a0)**frac)
        alphas=base_alpha*scales  # (B,)
        lrs=lr0*(1.0-0.65*frac)
        G=np.matmul(Xp,Xp.transpose(0,2,1))
        Gm=G-3.0*Eye[None,:,:]
        mval=Gm.max(axis=(1,2))
        # per-batch alpha scaling: exp(alpha_b*(G-mval))
        # do in loop over B to allow different alphas (B small)
        Grad=np.empty_like(Xp)
        for b in range(B):
            W=np.exp(alphas[b]*(G[b]-mval[b]))
            np.fill_diagonal(W,0.0)
            S=W.sum()
            if S<=0 or not np.isfinite(S):
                Grad[b]=0.0
            else:
                Grad[b]=(W@Xp[b])/S
        dot=np.sum(Grad*Xp,axis=2,keepdims=True)
        Gt=Grad-dot*Xp
        m_ad=b1*m_ad+(1-b1)*Gt
        v_ad=b2*v_ad+(1-b2)*(Gt*Gt)
        bc1=1.0-b1**(s+1); bc2=1.0-b2**(s+1)
        step=lrs*(m_ad/bc1)/(np.sqrt(v_ad/bc2)+eps)
        sdot=np.sum(step*Xp,axis=2,keepdims=True)
        step=step-sdot*Xp
        Xp=Xp-step
        Xp/=np.linalg.norm(Xp,axis=2,keepdims=True)
    return Xp

def _match_align(A,B,iters=3):
    n,d=A.shape
    Q=np.eye(d)
    perm=np.arange(n)
    for _ in range(iters):
        BR=B@Q
        C=A@BR.T
        try:
            r,c=linear_sum_assignment(-C)
        except Exception:
            break
        p2=np.empty(n,dtype=int); p2[r]=c
        perm=p2
        Bp=B[perm]
        H=Bp.T@A
        try:
            U,S,Vt=np.linalg.svd(H)
        except Exception:
            break
        Q=U@Vt
    B_aligned=(B[perm]@Q)
    B_aligned/=np.linalg.norm(B_aligned,axis=1,keepdims=True)
    return perm,Q,B_aligned

def solve():
    t0=time.time()
    HARD_END=t0+172.0
    rng=np.random.default_rng(42)
    n,d=102,12
    try:
        rec=np.load("record.npy").astype(float)
    except Exception:
        rec=None
    try:
        bst=np.load("best.npy").astype(float)
    except Exception:
        bst=rec
    def _fix(X):
        X=np.asarray(X,dtype=float)
        if X.shape!=(n,d):
            tmp=rng.standard_normal((n,d))
            try:
                m0=min(X.shape[0],n); m1=min(X.shape[1],d)
                tmp[:m0,:m1]=X[:m0,:m1]
            except Exception:
                pass
            X=tmp
        X/=np.linalg.norm(X,axis=1,keepdims=True)
        return X
    if rec is None and bst is None:
        base=rng.standard_normal((n,d)); base/=np.linalg.norm(base,axis=1,keepdims=True)
    else:
        if rec is None: rec=bst
        if bst is None: bst=rec
        rec=_fix(rec); bst=_fix(bst)
        base=bst if _max_cos(bst)<_max_cos(rec) else rec
    base_score=_max_cos(base)

    P=10
    pop=np.empty((P,n,d))
    sigmas=[0.02,0.07,0.15,0.05,0.10]
    # helper farthest reinsert
    def _reinsert(X_rem,k,seed_rng):
        cur=X_rem.copy()
        cand=seed_rng.standard_normal((2500,d))
        cand/=np.linalg.norm(cand,axis=1,keepdims=True)
        new=[]
        for _ in range(k):
            if cur.shape[0]>0:
                sc=(cand@cur.T).max(axis=1)
            else:
                sc=np.zeros(cand.shape[0])
            bi=int(np.argmin(sc))
            q=cand[bi]
            new.append(q)
            cur=np.vstack([cur,q[None,:]]) if cur.size else q[None,:]
        return np.array(new)
    for b in range(P):
        if b==0:
            X=base+0.02*rng.standard_normal((n,d))
            X/=np.linalg.norm(X,axis=1,keepdims=True); pop[b]=X
        elif b==1:
            X=base+0.07*rng.standard_normal((n,d))
            X/=np.linalg.norm(X,axis=1,keepdims=True); pop[b]=X
        elif b==2:
            X=base+0.15*rng.standard_normal((n,d))
            X/=np.linalg.norm(X,axis=1,keepdims=True); pop[b]=X
        elif b==3:
            X=rng.standard_normal((n,d))
            X/=np.linalg.norm(X,axis=1,keepdims=True); pop[b]=X
        elif b==4:
            X=rng.standard_normal((n,d))
            X/=np.linalg.norm(X,axis=1,keepdims=True); pop[b]=X
        elif b==5:
            # delete 5 random, farthest reinsert
            idx=np.arange(n); rng.shuffle(idx)
            keep=idx[5:]
            Xr=base[keep]
            new=_reinsert(Xr,5,rng)
            X=np.vstack([Xr,new])
            X=X+0.01*rng.standard_normal((n,d))
            X/=np.linalg.norm(X,axis=1,keepdims=True); pop[b]=X
        elif b==6:
            idx=np.arange(n); rng.shuffle(idx)
            keep=idx[10:]
            Xr=base[keep]
            new=_reinsert(Xr,10,rng)
            X=np.vstack([Xr,new])
            X=X+0.01*rng.standard_normal((n,d))
            X/=np.linalg.norm(X,axis=1,keepdims=True); pop[b]=X
        elif b==7:
            X=base+0.05*rng.standard_normal((n,d))
            # anisotropic stretch then renorm
            A=np.eye(d)+0.08*rng.standard_normal((d,d))
            X=X@A
            X/=np.linalg.norm(X,axis=1,keepdims=True); pop[b]=X
        elif b==8:
            X=base+0.10*rng.standard_normal((n,d))
            X/=np.linalg.norm(X,axis=1,keepdims=True); pop[b]=X
        else:
            # greedy farthest from pool
            pool=2500
            cand=rng.standard_normal((pool,d)); cand/=np.linalg.norm(cand,axis=1,keepdims=True)
            first=int(rng.integers(0,pool))
            chosen=[first]
            dm=cand@cand[first]
            dm[first]=np.inf
            for _ in range(1,n):
                nxt=int(np.argmin(dm))
                chosen.append(nxt)
                dm[nxt]=np.inf
                nd=cand@cand[nxt]
                # update max
                for c in chosen:
                    nd[c]=np.inf
                dm=np.maximum(np.where(np.isinf(dm),-np.inf,dm),np.where(np.isinf(nd),-np.inf,nd))
                for c in chosen:
                    dm[c]=np.inf
            X=cand[np.array(chosen)]+0.005*rng.standard_normal((n,d))
            X/=np.linalg.norm(X,axis=1,keepdims=True); pop[b]=X
    # cheap polish without LP, different annealing scales
    scales=np.array([0.6,0.8,1.0,1.4,1.8,0.7,1.2,1.0,1.5,0.9])
    Xp=pop.copy()
    Xp=_batch_adam(Xp,steps=900,lr0=0.010,a0=12.0,a1=160.0,scales=scales,deadline=min(HARD_END-60,t0+55))
    # scores
    scores=np.array([_max_cos(Xp[b]) for b in range(P)])
    order=np.argsort(scores)
    # fingerprint: sorted pair-cosine histogram downsampled
    iu_r,iu_c=np.triu_indices(n,1)
    descs=[]
    for b in range(P):
        G=Xp[b]@Xp[b].T
        c=G[iu_r,iu_c]
        cs=np.sort(c)[::-1]
        descs.append(cs[::8])
    descs=np.array(descs)
    # diverse retain K=6
    K=6
    sel=[]
    thresh=2.5e-4
    for idx in order:
        if len(sel)==0:
            sel.append(int(idx)); continue
        dd=descs[idx]
        dists=[float(np.mean(np.abs(dd-descs[s]))) for s in sel]
        if min(dists)>thresh:
            sel.append(int(idx))
        if len(sel)>=K:
            break
    if len(sel)<3:
        sel=[int(i) for i in order[:K]]
    else:
        # fill up to K with best remaining if threshold too strict
        if len(sel)<K:
            for idx in order:
                if int(idx) not in sel:
                    sel.append(int(idx))
                if len(sel)>=K:
                    break
    elites=[Xp[s] for s in sel]
    # include base as extra diversity candidate if not close
    # pairwise recombination: match + align
    children=[]
    # all pairs among elites
    pairs=[(i,j) for i in range(len(elites)) for j in range(i+1,len(elites))]
    rng_c=np.random.default_rng(7)
    for (i,j) in pairs:
        if time.time()>HARD_END-55:
            break
        A=elites[i]; B=elites[j]
        try:
            perm,Q,Bal=_match_align(A,B,iters=3)
        except Exception:
            continue
        # child type 1: slerp midpoint
        mid=A+Bal; mid/=np.linalg.norm(mid,axis=1,keepdims=True)
        children.append(mid)
        # child type 2: subset mix half/half with jitter
        mask=np.zeros(n,dtype=bool)
        idxs=np.arange(n); rng_c.shuffle(idxs)
        mask[idxs[:n//2]]=True
        mix=np.where(mask[:,None],A,Bal)
        mix=mix+0.005*rng_c.standard_normal((n,d))
        mix/=np.linalg.norm(mix,axis=1,keepdims=True)
        children.append(mix)
        if len(children)>=30:
            break
    if len(children)==0:
        best_idx=int(np.argmin(scores))
        return Xp[best_idx]/np.linalg.norm(Xp[best_idx],axis=1,keepdims=True)
    # fast screen raw scores
    craw=np.array([_max_cos(c) for c in children])
    c_order=np.argsort(craw)
    # take top 6 for quench
    QN=min(6,len(children))
    q_idx=[int(c_order[k]) for k in range(QN)]
    Qb=np.array([children[k] for k in q_idx])
    # quench
    qscales=np.ones(QN)
    Qb=_batch_adam(Qb,steps=500,lr0=0.008,a0=20.0,a1=220.0,scales=qscales,deadline=min(HARD_END-35,time.time()+25))
    qscores=np.array([_max_cos(Qb[b]) for b in range(QN)])
    qord=np.argsort(qscores)
    # pick best two for LP polish
    polished=[]
    for t in range(min(2,QN)):
        b=int(qord[t])
        Xc=Qb[b]
        remain=HARD_END-time.time()
        if remain<4:
            polished.append((float(qscores[b]),Xc)); continue
        budget=min(18.0,remain-2.0)
        if budget<2:
            polished.append((float(qscores[b]),Xc)); continue
        Xp_pol=_lp_polish(Xc,time.time()+budget)
        polished.append((_max_cos(Xp_pol),Xp_pol))
    # collect all candidates
    allc=[]
    allc.append((base_score,base))
    for b in range(P):
        allc.append((float(scores[b]),Xp[b]))
    for b in range(QN):
        allc.append((float(qscores[b]),Qb[b]))
    allc.extend(polished)
    allc.sort(key=lambda t:t[0])
    bestX=allc[0][1].copy()
    bestX/=np.linalg.norm(bestX,axis=1,keepdims=True)
    return bestX
