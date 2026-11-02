import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.sqrt(np.sum(X*X, axis=1, keepdims=True))
    n[n==0]=1.0
    return X/n

def _maxcos(X):
    G=X@X.T
    np.fill_diagonal(G,-2.0)
    return float(np.max(G))

def _bases_all(X):
    n,d=X.shape
    k4=d-1
    B=np.empty((n,d,k4),dtype=np.float64)
    for i in range(n):
        x=X[i]
        k=int(np.argmin(np.abs(x)))
        T=np.empty((d,k4),dtype=np.float64)
        c=0
        for j in range(d):
            if j==k: continue
            v=-x[j]*x
            v[j]+=1.0
            T[:,c]=v
            c+=1
        Q,_=np.linalg.qr(T)
        B[i]=Q[:,:k4]
    return B

def _bases_sel(Xsel):
    n=Xsel.shape[0]; d=Xsel.shape[1]; k4=d-1
    B=np.empty((n,d,k4),dtype=np.float64)
    for i in range(n):
        x=Xsel[i]
        k=int(np.argmin(np.abs(x)))
        T=np.empty((d,k4),dtype=np.float64)
        c=0
        for j in range(d):
            if j==k: continue
            v=-x[j]*x
            v[j]+=1.0
            T[:,c]=v
            c+=1
        Q,_=np.linalg.qr(T)
        B[i]=Q[:,:k4]
    return B

def solve():
    t0=time.time()
    T_END=t0+172.0
    LP_CUTOFF=t0+150.0
    rng=np.random.default_rng(0)
    n,d=453,5
    k4=4
    cands=[]
    for fn in ("best.npy","record.npy"):
        try:
            a=np.load(fn)
            cands.append(np.asarray(a,dtype=np.float64))
        except Exception:
            pass
    bestX=None; bestV=float("inf")
    for cc in cands:
        try:
            if cc.shape!=(453,5): continue
            cc=_norm(cc)
            if not np.all(np.isfinite(cc)): continue
            vv=_maxcos(cc)
            if vv<bestV:
                bestV=vv; bestX=cc.copy()
        except Exception:
            continue
    if bestX is None:
        bestX=rng.standard_normal((n,d)); bestX=_norm(bestX)
        bestV=_maxcos(bestX)
    iu_r,iu_c=np.triu_indices(n,1)

    # ---- diagnostics on bestX ----
    try:
        Gb=bestX@bestX.T
        cb=Gb[iu_r,iu_c]
        mb=float(np.max(cb))
        cSlack=mb-cb
        contact=(cSlack<1e-9)
        deg=np.zeros(n,dtype=int)
        np.add.at(deg,iu_r[contact],1)
        np.add.at(deg,iu_c[contact],1)
        Gt=Gb.copy(); np.fill_diagonal(Gt,-2.0)
        row_max=np.max(Gt,axis=1)
    except Exception:
        deg=np.zeros(n,dtype=int); row_max=np.zeros(n); mb=bestV
    ratt=np.where(deg==0)[0].tolist()
    # ensure 6 rattlers; pad if needed
    if len(ratt)<6:
        extra=np.argsort(deg)[:6-len(ratt)].tolist()
        ratt=list(ratt)+[e for e in extra if e not in ratt]
    ratt=ratt[:6]
    order_hub=np.argsort(-deg)
    hubs=[int(v) for v in order_hub if int(v) not in ratt][:15]

    subsets=[]
    # S0: rattlers + top 8 hubs
    s0=list(dict.fromkeys(ratt+hubs[:8]))
    subsets.append(s0[:14])
    # S1..S3: rattlers + one hub neighbourhood at a time
    for h in hubs[:3]:
        try:
            dots=bestX[h]@bestX.T
            dots[h]=-2.0
            nn=np.argsort(-dots)[:6].tolist()
            s=list(dict.fromkeys(ratt+[int(h)]+[int(v) for v in nn]))
            # cap 14
            subsets.append(s[:14])
        except Exception:
            pass
    # dedup subsets
    uniq=[]
    for s in subsets:
        if s not in uniq:
            uniq.append(s)
    subsets=uniq[:5]

    def lp_global(X, curBX, curBV, r_init, max_solves):
        X=X.copy(); bX=curBX.copy(); bV=float(curBV)
        radius=float(r_init)
        Nvar=n*4
        c_obj=np.zeros(Nvar+1); c_obj[-1]=1.0
        ar4=np.arange(4)
        for _ in range(max_solves):
            if time.time()>LP_CUTOFF or time.time()>T_END-8.0: break
            if radius<1e-12: break
            G=X@X.T
            c_all=G[iu_r,iu_c]
            m=float(np.max(c_all))
            if m<bV:
                bV=m; bX=X.copy()
            slack=m-c_all
            tight=np.where(slack<2e-5)[0]
            if tight.size==0:
                tight=np.argsort(slack)[:500]
            if tight.size>2500:
                part=np.argpartition(slack[tight],2500)[:2500]
                tight=tight[part]
            I=iu_r[tight]; J=iu_c[tight]; p=int(I.size)
            b_ub=slack[tight].astype(np.float64)
            B=_bases_all(X)
            aI=np.einsum('pdk,pd->pk',B[I],X[J])
            aJ=np.einsum('pdk,pd->pk',B[J],X[I])
            cols2=np.empty((p,9),dtype=np.int64)
            cols2[:,0:4]=I[:,None]*4+ar4[None,:]
            cols2[:,4:8]=J[:,None]*4+ar4[None,:]
            cols2[:,8]=Nvar
            vals2=np.empty((p,9),dtype=np.float64)
            vals2[:,0:4]=aI; vals2[:,4:8]=aJ; vals2[:,8]=-1.0
            rows=np.repeat(np.arange(p),9)
            try:
                Aub=coo_matrix((vals2.ravel(),(rows,cols2.ravel())),shape=(p,Nvar+1)).tocsr()
            except Exception:
                radius*=0.5; continue
            rem=min(LP_CUTOFF-time.time(),T_END-8.0-time.time(),9.0)
            if rem<0.5: break
            bounds=[(-radius,radius)]*Nvar+[(None,None)]
            try:
                res=linprog(c_obj,A_ub=Aub,b_ub=b_ub,bounds=bounds,method="highs",options={"time_limit":max(0.5,rem)})
            except Exception:
                radius*=0.5; continue
            if res.status!=0 or res.x is None:
                radius*=0.5; continue
            try:
                U=np.asarray(res.x[:Nvar]).reshape(n,4)
            except Exception:
                radius*=0.5; continue
            if not np.all(np.isfinite(U)):
                radius*=0.5; continue
            DX=np.einsum('ndk,nk->nd',B,U)
            Xt=_norm(X+DX)
            mt=_maxcos(Xt)
            if np.isfinite(mt) and mt<m-1e-13:
                X=Xt
                if mt<bV:
                    bV=mt; bX=Xt.copy()
                radius=min(radius*1.5,5e-4)
            else:
                radius*=0.5
        return bX,bV,X

    def softmax_local(X, active, iters, tau0, tau1, lr0, lr1):
        nonlocal_bestX=None
        # returns updated X
        Na=len(active)
        act=np.asarray(active,dtype=np.int64)
        Xa=X[act]
        V=np.zeros((Na,d),dtype=np.float64)
        ltau=np.log(tau1/tau0); llr=np.log(lr1/lr0)
        # frozen-frozen max constant
        # compute once full max for reference; update periodically
        for it in range(iters):
            if time.time()>T_END-8.0: break
            frac=it/max(iters-1,1)
            tau=tau0*np.exp(ltau*frac); lr=lr0*np.exp(llr*frac)
            # dots active vs all
            D=X[act]@X.T  # Na x n
            # mask self: set diag entries to -2
            for a_i,g in enumerate(act):
                D[a_i,int(g)]=-2.0
            m=float(np.max(D))
            # also consider frozen-frozen? approximate with global every 100 iters
            if it%100==0:
                try:
                    G=X@X.T; np.fill_diagonal(G,-2.0)
                    mf=float(np.max(G))
                    if mf>m: m=mf
                except Exception:
                    pass
            P=np.exp((D-m)/tau)  # Na x n
            Grad=P@X  # Na x d
            dot=np.sum(Grad*X[act],axis=1,keepdims=True)
            Gt=Grad-dot*X[act]
            # normalize grad scale
            gn=np.sqrt(np.sum(Gt*Gt,axis=1,keepdims=True))+1e-30
            Gt=Gt/gn*min(1.0,float(np.mean(gn))*0.0+1.0)  # keep direction
            V=0.8*V+0.2*Gt
            Xn=X.copy()
            Xn[act]=_norm(X[act]-lr*V)
            X=Xn
        return X

    def lp_local(X, active, r_init, max_solves, deadline):
        X=X.copy()
        act=np.asarray(active,dtype=np.int64)
        Na=len(act)
        if Na*4>120:  # keep under ~100 vars; trim neighbours if needed
            act=act[:25]
            Na=len(act)
        g2l={int(g):i for i,g in enumerate(act)}
        Nvar=Na*4
        c_obj=np.zeros(Nvar+1); c_obj[-1]=1.0
        radius=float(r_init)
        for _ in range(max_solves):
            if time.time()>deadline or time.time()>T_END-8.0: break
            if radius<1e-12: break
            G=X@X.T; np.fill_diagonal(G,-2.0)
            m=float(np.max(G))
            # collect pairs with one end active and slack<2e-5
            # vectorized: slack matrix rows active
            slack=m-G
            pairs=[]
            for a_i,g in enumerate(act):
                g=int(g)
                row=slack[g]
                # tight partners
                idx=np.where(row<2e-5)[0]
                for b in idx:
                    b=int(b)
                    if b==g: continue
                    if b<g:
                        I=b; J=g
                    else:
                        I=g; J=b
                    pairs.append((I,J))
            if len(pairs)==0:
                # take 200 nearest to active
                for a_i,g in enumerate(act):
                    g=int(g)
                    row=slack[g]
                    idx=np.argsort(row)[:30]
                    for b in idx:
                        b=int(b)
                        if b==g: continue
                        I=min(g,b); J=max(g,b)
                        pairs.append((I,J))
            # dedup
            pairs=list(dict.fromkeys(pairs))
            if len(pairs)>2500:
                # keep tightest
                sl=np.array([slack[i,j] for i,j in pairs])
                part=np.argpartition(sl,2500)[:2500]
                pairs=[pairs[k] for k in part]
            p=len(pairs)
            if p==0:
                break
            I=np.array([a for a,b in pairs],dtype=np.int64)
            J=np.array([b for a,b in pairs],dtype=np.int64)
            b_ub=np.array([slack[i,j] for i,j in pairs],dtype=np.float64)
            Bs=_bases_sel(X[act])
            # map to local base index
            # need X[J], X[I]
            rows=[]; cols=[]; vals=[]
            for r in range(p):
                i=int(I[r]); j=int(J[r])
                # coeff for i if active
                if i in g2l:
                    li=g2l[i]
                    ai=Bs[li].T@X[j]  # 4
                    base=li*4
                    for k_ in range(4):
                        rows.append(r); cols.append(base+k_); vals.append(float(ai[k_]))
                if j in g2l:
                    lj=g2l[j]
                    aj=Bs[lj].T@X[i]
                    base=lj*4
                    for k_ in range(4):
                        rows.append(r); cols.append(base+k_); vals.append(float(aj[k_]))
                rows.append(r); cols.append(Nvar); vals.append(-1.0)
            try:
                Aub=coo_matrix((np.array(vals),(np.array(rows),np.array(cols))),shape=(p,Nvar+1)).tocsr()
            except Exception:
                radius*=0.5; continue
            rem=min(deadline-time.time(),T_END-8.0-time.time(),6.0)
            if rem<0.5: break
            bounds=[(-radius,radius)]*Nvar+[(None,None)]
            try:
                res=linprog(c_obj,A_ub=Aub,b_ub=b_ub,bounds=bounds,method="highs",options={"time_limit":max(0.5,rem)})
            except Exception:
                radius*=0.5; continue
            if res.status!=0 or res.x is None:
                radius*=0.5; continue
            try:
                U=np.asarray(res.x[:Nvar]).reshape(Na,4)
            except Exception:
                radius*=0.5; continue
            if not np.all(np.isfinite(U)): radius*=0.5; continue
            DXs=np.einsum('ndk,nk->nd',Bs,U)
            Xt=X.copy()
            Xt[act]=_norm(X[act]+DXs)
            mt=_maxcos(Xt)
            if np.isfinite(mt) and mt<m-1e-13:
                X=Xt
                radius=min(radius*1.5,3e-4)
            else:
                radius*=0.5
        return X

    # initial quick global polish (2 solves) to stabilize
    try:
        bestX,bestV,_=lp_global(bestX,bestX,bestV,8e-5,2)
    except Exception:
        pass

    NS=22000  # >=20000 sampled holes
    # pre-sample base directions (regenerate per subset with different seed offset for diversity)
    per_subset_budget=22.0
    global_bestX=bestX.copy(); global_bestV=float(bestV)

    for si, victims in enumerate(subsets):
        subset_deadline=min(T_END-35.0, time.time()+per_subset_budget)
        if time.time()>T_END-40.0: break
        if time.time()>LP_CUTOFF-5.0: break
        victims=[int(v) for v in victims]
        # order victims worst-first by row_max
        try:
            Gtmp=global_bestX@global_bestX.T; np.fill_diagonal(Gtmp,-2.0)
            rm=np.max(Gtmp,axis=1)
            victims=sorted(victims,key=lambda v:-rm[int(v)])
        except Exception:
            pass
        # sample holes
        try:
            S=rng.standard_normal((NS,d)); S=_norm(S)
        except Exception:
            continue
        Xnew=global_bestX.copy()
        # sequential best-hole insertion scoring vs frozen+already reinserted
        victim_set=set(victims)
        done=[]
        for v in victims:
            if time.time()>subset_deadline-12.0: break
            # allowed reference = indices not in (victims not yet done)
            not_done=set(victims)-set(done)-{int(v)}
            # build Xe of frozen+done
            mask=np.ones(n,dtype=bool)
            for u in not_done:
                mask[int(u)]=False
            mask[int(v)]=False  # exclude self
            Xe=Xnew[mask]
            # score samples
            D=S@Xe.T
            md=np.max(D,axis=1)
            bi=int(np.argmin(md))
            h=S[bi].copy()
            # local hole refinement: push away from nearest
            for _ in range(30):
                dots=Xe@h
                ci=int(np.argmax(dots))
                cc_=float(dots[ci])
                xc=Xe[ci]
                tt=xc-cc_*h
                tn=float(np.linalg.norm(tt))
                if tn<1e-12: break
                tt/=tn
                step=0.02
                h=h-step*tt
                nh=float(np.linalg.norm(h))
                if nh==0 or not np.isfinite(nh): break
                h/=nh
            Xnew[int(v)]=h
            done.append(int(v))
        try:
            v0=_maxcos(Xnew)
        except Exception:
            continue
        if not np.isfinite(v0) or v0>global_bestV+0.005:
            # still try local opt only if mildly worse; else skip to next subset
            if v0>global_bestV+0.002:
                continue
        # build neighbour-extended active set (<100 vars => <=25 points)
        try:
            Gn=Xnew@Xnew.T; np.fill_diagonal(Gn,-2.0)
            active_set=list(victims)
            for v in victims:
                row=Gn[int(v)]
                nn=np.argsort(-row)[:4].tolist()
                for b in nn:
                    if int(b) not in active_set:
                        active_set.append(int(b))
                    if len(active_set)>=24: break
                if len(active_set)>=24: break
            active_set=active_set[:24]
        except Exception:
            active_set=victims
        # optimize only active set: annealed softmax
        try:
            if time.time()<subset_deadline-8.0:
                Xnew=softmax_local(Xnew,active_set,iters=500,tau0=2e-3,tau1=2e-4,lr0=4e-3,lr1=4e-4)
        except Exception:
            pass
        # small LP on active set
        try:
            if time.time()<subset_deadline-2.0:
                Xnew=lp_local(Xnew,active_set,r_init=8e-5,max_solves=3,deadline=subset_deadline)
        except Exception:
            pass
        try:
            v1=_maxcos(Xnew)
        except Exception:
            continue
        if np.isfinite(v1) and v1<global_bestV:
            global_bestV=v1; global_bestX=Xnew.copy()
            bestX=global_bestX.copy(); bestV=global_bestV

    # final global LP polish on best found
    try:
        if time.time()<T_END-10.0:
            bestX=global_bestX.copy(); bestV=global_bestV
            bestX,bestV,_=lp_global(bestX,bestX,bestV,8e-5,6)
            global_bestX,global_bestV=bestX.copy(),float(bestV)
    except Exception:
        pass
    # verify
    try:
        Xf=_norm(global_bestX)
        vv=_maxcos(Xf)
        if not np.isfinite(vv):
            Xf=_norm(bestX)
        return _norm(Xf)
    except Exception:
        return _norm(bestX)
