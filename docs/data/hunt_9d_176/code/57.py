import time
import itertools
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def adam_quench(x, steps=700, a0=40.0, a1=320.0, lr=0.003):
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

def build_pool():
    P=[]
    # 1-sparse
    for i in range(9):
        for s in (1.0,-1.0):
            v=np.zeros(9); v[i]=s; P.append(v)
    # 2-sparse
    sc=1.0/np.sqrt(2.0)
    for i,j in itertools.combinations(range(9),2):
        for si in (1.0,-1.0):
            for sj in (1.0,-1.0):
                v=np.zeros(9); v[i]=si*sc; v[j]=sj*sc; P.append(v)
    # 4-sparse +-1/2
    for comb in itertools.combinations(range(9),4):
        for mask in range(16):
            v=np.zeros(9)
            for k,idx in enumerate(comb):
                s=1.0 if ((mask>>k)&1)==0 else -1.0
                v[idx]=0.5*s
            P.append(v)
    # sign vectors +-1/3, fix first coord + to get 256
    for mask in range(256):
        v=np.full(9,1.0/3.0)
        v[0]=1.0/3.0
        for k in range(1,9):
            # use bits of mask
            b=(mask>>(k-1))&1
            if b:
                v[k]=-1.0/3.0
        # mask bit 8? need 8 bits for coords 1..8 exactly 256 combos, done
        P.append(v)
    Pool=np.array(P,dtype=float)
    # normalize (already unit, but ensure)
    Pool/=np.linalg.norm(Pool,axis=1,keepdims=True)
    return Pool

def solve():
    t0=time.time()
    BUDGET=172.0
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
    # ---- build pool ----
    Pool=build_pool()
    Np=Pool.shape[0]
    Gpool=Pool@Pool.T
    # clip diagonal to 1
    # ---- greedy select 176 minimizing (max, cnt46, cnt44, cnt40) ----
    nsel=176
    cur_max=np.full(Np,-2.0)
    cur_c46=np.zeros(Np,dtype=int)
    cur_c44=np.zeros(Np,dtype=int)
    cur_c40=np.zeros(Np,dtype=int)
    selected=[]
    in_sel=np.zeros(Np,dtype=bool)
    # start with index 0
    selected.append(0); in_sel[0]=True
    # update stats with s=0
    g0=Gpool[:,0]
    cur_max=np.maximum(cur_max,g0)
    cur_c46=(g0>0.46).astype(int)
    cur_c44=(g0>0.44).astype(int)
    cur_c40=(g0>0.40).astype(int)
    # second: most negative (min dot) to spread
    # pick argmin g0 among not selected
    second=int(np.argmin(g0+in_sel*10.0))
    selected.append(second); in_sel[second]=True
    gs=Gpool[:,second]
    cur_max=np.maximum(cur_max,gs)
    cur_c46+= (gs>0.46).astype(int)
    cur_c44+= (gs>0.44).astype(int)
    cur_c40+= (gs>0.40).astype(int)
    # mask selected to exclude
    for k in range(2,nsel):
        if time.time()-t0>BUDGET-100:
            # fill randomly if out of time
            rem=np.where(~in_sel)[0]
            need=nsel-len(selected)
            pick=rng.choice(rem,size=need,replace=False)
            for q in pick:
                selected.append(int(q)); in_sel[int(q)]=True
            break
        # find best among not selected lexicographically
        # vectorized lexicographic: sort by max, then c46, c44, c40
        # do argmin via iterative filtering to avoid full sort
        cand=np.where(~in_sel)[0]
        m=cur_max[cand]
        # primary: minimal m
        mmin=np.min(m)
        # allow tiny tol? strict
        f1=cand[m<=mmin+1e-12]
        if len(f1)>1:
            c46=cur_c46[f1]; cmin=np.min(c46); f2=f1[c46<=cmin]
            if len(f2)>1:
                c44=cur_c44[f2]; cmin2=np.min(c44); f3=f2[c44<=cmin2]
                if len(f3)>1:
                    c40=cur_c40[f3]; cmin3=np.min(c40); f4=f3[c40<=cmin3]
                    # tie break random among f4 (deterministic via rng)
                    choice=int(rng.choice(f4))
                else:
                    choice=int(f3[0]) if len(f3)==1 else int(rng.choice(f3))
                    # actually f3 len 1
                    if len(f3)==1:
                        choice=int(f3[0])
                    else:
                        choice=int(rng.choice(f3))
            else:
                # len f2 ==1? already filtered
                if len(f2)==1:
                    choice=int(f2[0])
                else:
                    choice=int(f2[0])
        else:
            choice=int(f1[0])
        selected.append(choice); in_sel[choice]=True
        gs=Gpool[:,choice]
        cur_max=np.maximum(cur_max,gs)
        cur_c46+=(gs>0.46).astype(int)
        cur_c44+=(gs>0.44).astype(int)
        cur_c40+=(gs>0.40).astype(int)
        # fix stats for selected themselves (self dot=1 should not count)
        # cur_max for selected is 1 due to self; ignore since we only query ~in_sel
    S=np.array(selected,dtype=int)
    # compute discrete objective of S
    # GramSel
    GramSel=Gpool[np.ix_(S,S)].copy()
    np.fill_diagonal(GramSel,-2.0)
    Mcur=float(np.max(GramSel))
    # counts (upper triangle)
    iu=np.triu_indices(nsel,1)
    vals=GramSel[iu]
    c46cur=int(np.count_nonzero(vals>0.46))
    c44cur=int(np.count_nonzero(vals>0.44))
    c40cur=int(np.count_nonzero(vals>0.40))
    bestS=S.copy()
    bestM=Mcur; bestc46=c46cur; bestc44=c44cur; bestc40=c40cur
    # per-element counts for fast updates? compute dots vs thresholds on fly
    # ---- swap-based simulated annealing, at most 80 s ----
    SA_TIME=80.0
    sa_start=time.time()
    sa_deadline=sa_start+SA_TIME
    # ensure GramSel diagonal -2 for max, but for updates need 1? keep -2 diag, handle separately
    # temperature schedule
    T0=0.01; T1=0.0002
    it=0
    # preallocate
    rng2=np.random.default_rng(1)
    # estimate total iterations from time: run until deadline
    # current GramSel with diag -2
    while time.time()<sa_deadline and time.time()-t0<BUDGET-55:
        frac=(time.time()-sa_start)/SA_TIME
        if frac<0: frac=0
        if frac>1: frac=1
        T=T0*((T1/T0)**frac)
        # propose swap: random out pos, random in idx
        p=int(rng2.integers(0,nsel))
        q=int(rng2.integers(0,Np))
        if in_sel[q]:
            it+=1
            continue
        out_idx=int(S[p])
        # dots
        dots_out=GramSel[p,:].copy()  # includes -2 diag at p
        # max involving out (exclude self)
        # compute max_out
        max_out=float(np.max(dots_out[np.arange(nsel)!=p])) if nsel>1 else -2.0
        # counts involving out
        cnt_out46=int(np.count_nonzero(dots_out>0.46))
        cnt_out44=int(np.count_nonzero(dots_out>0.44))
        cnt_out40=int(np.count_nonzero(dots_out>0.40))
        # incoming dots vs current S
        row=Gpool[q,S]  # size nsel, includes dot with out_idx at p
        # exclude p for remaining comparison
        # max_in
        # build mask exclude p
        # fast max without allocation: compute max over row with p ignored
        if nsel>1:
            max_in=float(row[np.arange(nsel)!=p].max())
        else:
            max_in=-2.0
        cnt_in46=int(np.count_nonzero(row[np.arange(nsel)!=p]>0.46))
        cnt_in44=int(np.count_nonzero(row[np.arange(nsel)!=p]>0.44))
        cnt_in40=int(np.count_nonzero(row[np.arange(nsel)!=p]>0.40))
        c46new=c46cur-cnt_out46+cnt_in46
        c44new=c44cur-cnt_out44+cnt_in44
        c40new=c40cur-cnt_out40+cnt_in40
        # M_rem
        if max_out < Mcur-1e-12:
            Mrem=Mcur
        else:
            # recompute max excluding p: temp set row/col p to -2
            saved_row=GramSel[p,:].copy()
            saved_col=GramSel[:,p].copy()
            GramSel[p,:]=-2.0
            GramSel[:,p]=-2.0
            Mrem=float(np.max(GramSel))
            GramSel[p,:]=saved_row
            GramSel[:,p]=saved_col
        Mnew=Mrem if Mrem>max_in else max_in
        dM=Mnew-Mcur
        accept=False
        if dM < -1e-12:
            accept=True
        elif dM > 1e-12:
            if rng2.random() < np.exp(-dM/max(T,1e-12)):
                accept=True
        else:
            # equal M: lexicographic on counts
            if c46new< c46cur or (c46new==c46cur and c44new < c44cur) or (c46new==c46cur and c44new==c44cur and c40new < c40cur):
                accept=True
            elif c46new==c46cur and c44new==c44cur and c40new==c40cur:
                # neutral: accept with 0.5 to diffuse
                if rng2.random()<0.2:
                    accept=True
            else:
                dc=(c46new-c46cur)+0.3*(c44new-c44cur)+0.1*(c40new-c40cur)
                dc_scaled=dc*0.001
                if dc_scaled<=0:
                    accept=True
                else:
                    if rng2.random() < np.exp(-dc_scaled/max(T,1e-12)):
                        accept=True
        if accept:
            # apply update to GramSel, S, in_sel, counters
            # new row/col values: row (size nsel) with diag -2 at p
            new_vals=row.copy()
            new_vals[p]=-2.0
            GramSel[p,:]=new_vals
            GramSel[:,p]=new_vals  # symmetric (row same)
            # GramSel symmetric: Gpool symmetric so row==col
            S[p]=q
            in_sel[out_idx]=False
            in_sel[q]=True
            Mcur=Mnew; c46cur=c46new; c44cur=c44new; c40cur=c40new
            # track best lexicographically
            better=False
            if Mcur < bestM-1e-12:
                better=True
            elif abs(Mcur-bestM)<=1e-12:
                if c46cur<bestc46 or (c46cur==bestc46 and c44cur<bestc44) or (c46cur==bestc46 and c44cur==bestc44 and c40cur<bestc40):
                    better=True
            if better:
                bestM=Mcur; bestc46=c46cur; bestc44=c44cur; bestc40=c40cur
                bestS=S.copy()
        it+=1
        # occasional early exit if time heavy? loop checks time each iter (costly). check every 512 iters
        if (it & 511)==0 and time.time()>=sa_deadline:
            break
    # ---- quench best discrete ----
    Xd=Pool[bestS].copy()
    # continuous quench
    # quick quench first to evaluate
    Xq=adam_quench(Xd.copy(),steps=700,a0=40.0,a1=320.0,lr=0.003)
    vq=max_cos(Xq)
    # one short LP polish if time permits
    rem=BUDGET-(time.time()-t0)
    if rem>8.0:
        lp_t=min(rem-2.0,25.0)
        try:
            Xp=lp_polish(Xq,seconds=lp_t,radius_init=3e-3)
            vp=max_cos(Xp)
            if vp<vq:
                Xq=Xp; vq=vp
        except Exception:
            pass
    if vq < best_val-1e-12:
        return Xq/np.linalg.norm(Xq,axis=1,keepdims=True)
    else:
        return best_x
