import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def adam_quench(x, steps=500, a0=40.0, a1=320.0, lr=0.003, seed=0):
    n, d = x.shape
    m = np.zeros_like(x)
    vv = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(steps):
        frac = t / max(steps-1, 1)
        alpha = a0 * ((a1/a0)**frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        E = np.exp(alpha*(G-M))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum()*0.5)
        if S<=0 or not np.isfinite(S):
            break
        Grad = (E @ x)/S
        dot = np.sum(Grad*x, axis=1, keepdims=True)
        Gt = Grad - dot*x
        m = b1*m + (1-b1)*Gt
        vv = b2*vv + (1-b2)*(Gt*Gt)
        mhat = m/(1-b1**(t+1))
        vhat = vv/(1-b2**(t+1))
        step = mhat/(np.sqrt(vhat)+eps)
        np.clip(step, -3.0, 3.0, out=step)
        lrt = lr*(1.0-0.5*frac)
        x = x - lrt*step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr==0]=1.0
        x/=nr
        if not np.all(np.isfinite(x)):
            break
    nr = np.linalg.norm(x, axis=1, keepdims=True)
    nr[nr==0]=1.0
    return x/nr

def lp_polish(X, seconds, radius_init=3e-3, cap=2800):
    x = X/np.linalg.norm(X, axis=1, keepdims=True)
    n,d = x.shape
    best = max_cos(x)
    radius = radius_init
    start = time.time()
    iu0 = np.triu_indices(n,1)
    while time.time()-start < seconds and radius>1e-14:
        Gfull = x@x.T
        c_all = Gfull[iu0]
        mval = float(c_all.max())
        thresh = mval - max(20*radius, 1e-9)
        sel = c_all>thresh
        cnt = int(np.count_nonzero(sel))
        if cnt==0:
            break
        if cnt>cap:
            idx_all = np.where(sel)[0]
            order = np.argpartition(-c_all[idx_all], cap)[:cap]
            keep = np.zeros_like(sel)
            keep[idx_all[order]]=True
            sel=keep
            cnt=cap
        I = iu0[0][sel]; J=iu0[1][sel]
        c = c_all[sel]
        p=cnt; nv=n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1),n*d)],1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
        try:
            A_ub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        except Exception:
            radius*=0.5
            continue
        obj=np.zeros(nv); obj[-1]=1.0
        try:
            res=linprog(obj,A_ub=A_ub,b_ub=-c,A_eq=A_eq,b_eq=np.zeros(n),bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5
            continue
        if res.status!=0:
            radius*=0.5
            continue
        y = x + res.x[:-1].reshape(n,d)
        nr=np.linalg.norm(y,axis=1,keepdims=True); nr[nr==0]=1.0; y/=nr
        v=max_cos(y)
        if v<best:
            x,best=y,v
            radius=min(radius*1.5,0.05)
        else:
            radius*=0.5
    return x

def build_modes_and_communities(X):
    n,d = X.shape
    G = X@X.T
    np.fill_diagonal(G,-2.0)
    mval=float(np.max(G))
    slack = mval - G
    iu=np.triu_indices(n,1)
    sl = slack[iu]
    order=np.argsort(sl)
    # active threshold 1e-3, cap 2500, at least 800
    thresh=1e-3
    selmask = sl<thresh
    cnt=int(np.count_nonzero(selmask))
    if cnt<800:
        cnt= min(800, len(sl))
        selmask=np.zeros_like(sl,dtype=bool)
        selmask[order[:cnt]]=True
    if cnt>2500:
        selmask=np.zeros_like(sl,dtype=bool)
        selmask[order[:2500]]=True
        cnt=2500
    I=iu[0][selmask]; J=iu[1][selmask]
    c=(X[I]*X[J]).sum(axis=1)
    p=cnt
    # adjacency & degrees
    adj=np.zeros((n,n),dtype=bool)
    adj[I,J]=True; adj[J,I]=True
    deg=adj.sum(axis=1)
    hubs=np.argsort(-deg)[:8]
    # tangent bases T (n,9,8)
    rng0=np.random.default_rng(7)
    R0=rng0.standard_normal((9,8))
    T=np.empty((n,9,8))
    for i in range(n):
        xi=X[i]
        coeff=xi@R0
        Y=R0 - xi[:,None]*coeff[None,:]
        Q,Rm=np.linalg.qr(Y)
        T[i]=Q
    # rigidity reduced matrix p x (n*8)
    N=n*8
    Ared=np.zeros((p,N),dtype=float)
    # vectorized per-contact projection
    # a_i = xJ - c xI, a_j = xI - c xJ
    AI = X[J]-c[:,None]*X[I]
    AJ = X[I]-c[:,None]*X[J]
    # project: AI @ T[I], AJ @ T[J]
    # loop (p up to 2500 fine)
    for k in range(p):
        i=I[k]; j=J[k]
        Ared[k, i*8:(i+1)*8]=AI[k]@T[i]
        Ared[k, j*8:(j+1)*8]=AJ[k]@T[j]
    # Gram + eigh
    try:
        Gt = Ared.T@Ared
        # symmetrize
        Gt=(Gt+Gt.T)*0.5
        w,V=np.linalg.eigh(Gt)
    except Exception:
        return None, hubs, adj, deg, I, J
    # zero threshold for rotations
    # rotations = 36 zero modes expected
    # find first index with w > 1e-8
    nz=np.searchsorted(w,1e-8)
    # flex modes after rotations (skip rotations)
    # take 20 modes starting at nz
    nmodes=20
    avail=V.shape[1]-nz
    if avail<5:
        nz=max(0,V.shape[1]-20)
    modes=[]
    for k in range(nmodes):
        idx=nz+k
        if idx>=V.shape[1]:
            break
        v=V[:,idx]  # N
        Ym=v.reshape(n,8)
        # lift to 9D: U_i = T_i @ y_i
        U=np.einsum('nij,nj->ni',T,Ym)
        # ensure tangent (project)
        dot=np.sum(U*X,axis=1,keepdims=True)
        U-=dot*X
        modes.append(U)
    # communities: BFS around each hub
    communities=[]
    for h in hubs:
        # BFS distances
        dist=np.full(n,999,dtype=int); dist[h]=0
        frontier=[int(h)]
        visited=set([int(h)])
        dd=0
        # expand to distance 2 first
        cur_set=set([int(h)])
        for depth in [1,2,3]:
            nxt=set()
            for u in cur_set:
                nbrs=np.where(adj[u])[0]
                for v in nbrs:
                    if int(v) not in visited:
                        visited.add(int(v)); nxt.add(int(v)); dist[int(v)]=depth
            if len(visited)>=35 and depth>=2:
                cur_set=nxt
                break
            cur_set=nxt
            if len(cur_set)==0:
                break
        comm=np.array(sorted(visited),dtype=int)
        # cap size 45, if larger keep closest + random subset? keep all if <=45 else truncate by distance then degree
        if len(comm)>45:
            # sort by dist then keep
            srt=np.argsort(dist[comm],kind='stable')
            comm=comm[srt[:45]]
        if len(comm)<12:
            # expand one more ring greedily by degree
            pass
        communities.append(comm)
    # spectral bipartition community as extra
    try:
        degf=adj.astype(float).sum(axis=1)
        L=np.diag(degf)-adj.astype(float)
        ww,VV=np.linalg.eigh(L)
        fied=VV[:,1]
        part0=np.where(fied<=0)[0]
        part1=np.where(fied>0)[0]
        if len(part0)>=12 and len(part0)<=160:
            communities.append(part0)
        if len(part1)>=12 and len(part1)<=160:
            communities.append(part1)
    except Exception:
        pass
    return modes, hubs, adj, deg, I, J, communities

def solve():
    t0=time.time()
    BUDGET=168.0
    rng=np.random.default_rng(0)
    try:
        xb=np.load("best.npy")
    except Exception:
        try:
            xb=np.load("record.npy")
        except Exception:
            xb=rng.standard_normal((176,9))
    xb=np.asarray(xb,dtype=float)
    if xb.shape!=(176,9):
        xb=rng.standard_normal((176,9))
    xb=xb/np.linalg.norm(xb,axis=1,keepdims=True)
    best_val=max_cos(xb)
    best_x=xb.copy()
    try:
        xr=np.load("record.npy")
        xr=np.asarray(xr,dtype=float)
        if xr.shape==(176,9):
            xr=xr/np.linalg.norm(xr,axis=1,keepdims=True)
            vr=max_cos(xr)
            if vr<best_val:
                best_val, best_x=vr, xr.copy()
    except Exception:
        pass
    # build rigidity info (fast, ~1s)
    modes=None; communities=None; deg=None
    try:
        out=build_modes_and_communities(best_x)
        if len(out)==7:
            modes,hubs,adj,deg,CI,CJ,communities=out
        else:
            modes,hubs,adj,deg,CI,CJ=out
            communities=None
    except Exception:
        modes=None
    cands=[]
    # if rigidity failed, fallback to random global jitters as communities
    if modes is None or len(modes)==0:
        modes=[rng.standard_normal((176,9))]
        # project
        for U in modes:
            dot=np.sum(U*best_x,axis=1,keepdims=True)
            U-=dot*best_x
        communities=[np.arange(176)]
    if communities is None or len(communities)==0:
        communities=[np.arange(176)]
    amps=np.array([0.01,0.03,0.06,0.1])
    Ncand=110
    scored=[]
    nm=len(modes)
    nc=len(communities)
    for t in range(Ncand):
        if time.time()-t0>BUDGET-70:
            break
        ci=t%nc
        comm=communities[ci]
        # random mode mixture of 1-3 modes
        nmix=1+(t%3==2)+(t%5==0)  # 1-3
        midx=rng.choice(nm,size=min(nmix,nm),replace=False)
        coeff=rng.standard_normal(len(midx))
        coeff/= (np.linalg.norm(coeff)+1e-12)
        if rng.random()<0.5:
            coeff*=-1.0
        U=np.zeros_like(best_x)
        for k,mi in enumerate(midx):
            U+=coeff[k]*modes[int(mi)]
        amp=amps[rng.integers(0,4)]
        # sometimes global move, sometimes community-masked
        if t%4==3:
            mask_all=True
            disp=U
        else:
            mask_all=False
            disp=np.zeros_like(best_x)
            disp[comm]=U[comm]
        # normalize to amplitude: mean norm over moved set = amp
        if mask_all:
            s=np.sqrt(np.mean(np.sum(disp*disp,axis=1)))+1e-12
        else:
            s=np.sqrt(np.mean(np.sum(disp[comm]*disp[comm],axis=1)))+1e-12
        disp=(amp/s)*disp
        # project to tangent
        dot=np.sum(disp*best_x,axis=1,keepdims=True)
        disp-=dot*best_x
        cand=best_x+disp
        cand/=np.linalg.norm(cand,axis=1,keepdims=True)
        v=max_cos(cand)
        scored.append((v,cand,ci,tuple(int(x) for x in midx),float(amp)))
    if len(scored)==0:
        return best_x
    scored.sort(key=lambda x: x[0])
    # keep top K for quench (also require diversity)
    K=min(7,len(scored))
    top=scored[:K]
    # quench each with 500 Adam steps
    best_q_val=best_val
    best_q_x=best_x.copy()
    # also consider unquenched best
    if scored[0][0]<best_q_val:
        # will be updated after quench anyway
        pass
    screened=[]
    for (v,cand,ci,mi,amp) in top:
        # skip if too bad vs best (+0.015) to save time, but keep at least 2
        if v>best_val+0.015 and len(screened)>=2:
            continue
        q=adam_quench(cand.copy(),steps=500,a0=40.0,a1=320.0,lr=0.003)
        qv=max_cos(q)
        screened.append((qv,q))
        if qv<best_q_val:
            best_q_val=qv
            best_q_x=q.copy()
    if len(screened)==0:
        return best_x
    screened.sort(key=lambda x: x[0])
    # LP-polish only single best screened candidate under hard deadline
    rem=BUDGET-(time.time()-t0)
    # reserve 1s for return
    lp_time=min(max(rem-1.0,0.0),55.0)
    if lp_time>3.0:
        cand_best=screened[0][1]
        # only polish if promising (within 0.005 of best or better)
        if screened[0][0] < best_val+0.005:
            try:
                pol=lp_polish(cand_best,seconds=lp_time,radius_init=3e-3)
                pv=max_cos(pol)
                if pv<best_q_val:
                    best_q_val=pv
                    best_q_x=pol.copy()
            except Exception:
                pass
    # update incumbent if improved, else return original best (never worse)
    if best_q_val < best_val - 1e-12:
        return best_q_x/np.linalg.norm(best_q_x,axis=1,keepdims=True)
    else:
        # if no improvement, return best_x LP-squeezed briefly if time left? avoid extra LP (timeout cause). Just return best_x.
        return best_x
