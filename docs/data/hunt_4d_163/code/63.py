import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

def _contact_pairs(x, tol=1e-7, cap=800):
    n = x.shape[0]
    iu0, iu1 = np.triu_indices(n, 1)
    g = (x @ x.T)[iu0, iu1]
    m = float(g.max())
    order = np.argsort(-g)
    sel = g > m - tol
    cnt = int(sel.sum())
    if cnt > cap:
        keep = order[:cap]
        I = iu0[keep]; J = iu1[keep]
        c = g[keep]
    else:
        if cnt == 0:
            k = int(np.argmax(g))
            I = np.array([iu0[k]]); J = np.array([iu1[k]]); c = np.array([g[k]])
        else:
            I = iu0[sel]; J = iu1[sel]; c = g[sel]
    return I, J, c, m

def _self_stress(x, tol=1e-7):
    # returns I,J,lam sorted desc, adjacency
    n = x.shape[0]
    I, J, c, m = _contact_pairs(x, tol=tol, cap=800)
    p = len(I)
    # adjacency for fallback
    adj = [set() for _ in range(n)]
    for a, b in zip(I, J):
        adj[int(a)].add(int(b)); adj[int(b)].add(int(a))
    if p < 4:
        return I, J, np.ones(p)/max(p,1), adj
    try:
        # tangent bases Ti (4x3)
        T = np.zeros((n, 4, 3))
        for i in range(n):
            xi = x[i]
            # project eye
            M = np.eye(4) - np.outer(xi, xi)
            # SVD to get basis
            u, s, _ = np.linalg.svd(M)
            # largest 3 singular vectors (s~1)
            T[i] = u[:, :3]
        mm = 3*n
        Mmat = np.zeros((mm, p))
        for k in range(p):
            i = int(I[k]); j = int(J[k])
            vi = T[i].T @ x[j]  # 3
            vj = T[j].T @ x[i]
            Mmat[3*i:3*i+3, k] = vi
            Mmat[3*j:3*j+3, k] = vj
        # LP: min s s.t. -s<=M lam<=s, sum lam=1, lam>=0
        # A_ub (2mm x (p+1))
        A_ub = np.zeros((2*mm, p+1))
        A_ub[:mm, :p] = Mmat
        A_ub[:mm, p] = -1.0
        A_ub[mm:, :p] = -Mmat
        A_ub[mm:, p] = -1.0
        b_ub = np.zeros(2*mm)
        A_eq = np.zeros((1, p+1)); A_eq[0, :p] = 1.0
        b_eq = np.array([1.0])
        cobj = np.zeros(p+1); cobj[p] = 1.0
        bounds = [(0, None)]*p + [(0, None)]
        res = linprog(cobj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs",
                      options={"time_limit": 5.0})
        if res.status == 0 and np.all(np.isfinite(res.x[:p])):
            lam = np.maximum(res.x[:p], 0)
            s = float(lam.sum())
            if s > 1e-12:
                lam /= s
                return I, J, lam, adj
    except Exception:
        pass
    # fallback: degree product stress proxy
    deg = np.zeros(n)
    for a, b in zip(I, J):
        deg[int(a)] += 1; deg[int(b)] += 1
    lam = np.array([deg[int(a)]+deg[int(b)] for a, b in zip(I, J)], dtype=float)
    lam = lam / max(lam.sum(), 1e-12)
    return I, J, lam, adj

def _bfs_patch(adj, seeds, k):
    n = len(adj)
    seen = set(int(s) for s in seeds)
    q = list(seen)
    qi = 0
    # BFS
    while len(seen) < k and qi < len(q):
        u = q[qi]; qi += 1
        for v in adj[u]:
            if v not in seen:
                seen.add(int(v))
                q.append(int(v))
                if len(seen) >= k:
                    break
    # if still short (disconnected), add highest-degree nodes
    if len(seen) < k:
        degs = sorted(range(n), key=lambda i: len(adj[i]), reverse=True)
        for v in degs:
            if v not in seen:
                seen.add(int(v))
                if len(seen) >= k:
                    break
    return np.array(sorted(seen), dtype=int)

def _refine_hole(y0, Xf, iters=180):
    y = y0 / max(float(np.linalg.norm(y0)), 1e-12)
    dots = Xf @ y
    vy = float(dots.max())
    step = 0.05
    for _ in range(iters):
        dots = Xf @ y
        j = int(np.argmax(dots))
        dj = float(dots[j])
        g = Xf[j] - dj * y
        ng = float(np.linalg.norm(g))
        if ng < 1e-13:
            break
        g /= ng
        improved = False
        for s in (step, step*0.3):
            yn = y - s*g
            yn /= max(float(np.linalg.norm(yn)), 1e-12)
            vn = float((Xf @ yn).max())
            if vn < vy:
                y, vy = yn, vn
                improved = True
                break
        if not improved:
            step *= 0.92
            if step < 1e-5:
                break
        else:
            step = min(step*1.05, 0.05)
    return y, vy

def _gd(x0, steps, T0, T1, lr0, mu, rng, active=None, deadline=None, noise0=0.0):
    x = x0.copy()
    n, d = x.shape
    if active is None:
        act = np.ones(n, dtype=bool)
    else:
        act = np.asarray(active, dtype=bool)
    v = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    best_val = float(G.max())
    best = x.copy()
    lratio = np.log(T1/T0) if T0>0 and T1>0 else 0.0
    for k in range(steps):
        if deadline is not None and (k & 255)==0 and time.time()>deadline:
            break
        frac = k/max(1, steps-1)
        T = T0*np.exp(lratio*frac)
        if T < 1e-12: T = 1e-12
        lr = lr0*(1.0-0.55*frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G-m)/T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum()*0.5)
        if S<1e-300 or not np.isfinite(S): S=1e-300
        grad = (E @ x)/S
        grad -= (grad*x).sum(axis=1, keepdims=True)*x
        if not np.all(act):
            grad[~act]=0.0; v[~act]=0.0
        v = mu*v + (1.0-mu)*grad
        x[act] = x[act]-lr*v[act]
        if noise0>0:
            amp = noise0*(1.0-frac)
            if amp>0:
                nz = rng.randn(n,d)*amp
                nz -= (nz*x).sum(axis=1, keepdims=True)*x
                nz[~act]=0.0
                x[act]=x[act]+nz[act]
        x[act] /= np.maximum(np.linalg.norm(x[act],axis=1,keepdims=True),1e-12)
        if m<best_val:
            best_val=m; best=x.copy()
    return best, best_val

def _lp_polish(x, deadline, r0=0.003, cap=1400):
    x = _norm(x.copy())
    n,d = x.shape
    try: best=_max_cos(x)
    except Exception: return x,1.0
    radius=float(r0)
    iu0,iu1=np.triu_indices(n,1)
    while time.time()<deadline and radius>1e-14:
        try:
            G=x@x.T; c=G[iu0,iu1]; m=float(c.max())
            sel=c>m-max(20.0*radius,1e-9)
            if int(sel.sum())==0: sel[int(np.argmax(c))]=True
            if int(sel.sum())>cap:
                idx=np.argsort(-c)[:cap]
                s2=np.zeros_like(sel,dtype=bool); s2[idx]=True; sel=s2
            I=iu0[sel]; J=iu1[sel]; csel=c[sel]; p=len(I)
            nv=n*d+1
            rows=np.repeat(np.arange(p),2*d+1)
            cols=np.empty((p,2*d+1),dtype=int)
            ar=np.arange(d)
            cols[:,:d]=I[:,None]*d+ar; cols[:,d:2*d]=J[:,None]*d+ar; cols[:,2*d]=n*d
            cols=cols.ravel()
            vals=np.empty((p,2*d+1),dtype=float)
            vals[:,:d]=x[J]; vals[:,d:2*d]=x[I]; vals[:,2*d]=-1.0
            vals=vals.ravel()
            A_ub=coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
            obj=np.zeros(nv); obj[-1]=1.0
            res=linprog(obj,A_ub=A_ub,b_ub=-csel,A_eq=A_eq,b_eq=np.zeros(n),
                        bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
            if res.status!=0:
                radius*=0.5; continue
            y=x+res.x[:-1].reshape(n,d); y=_norm(y)
            vv=_max_cos(y)
            if vv<best: x,best=y,vv; radius=min(radius*1.5,0.05)
            else: radius*=0.5
        except Exception:
            radius*=0.5; continue
    return x,best

def _jitter(x, rng, rad=0.15, m=200):
    # jitter each row m times
    n,d=x.shape
    out=np.empty((n*m,d))
    k=0
    for i in range(n):
        xi=x[i]
        for _ in range(m):
            r=rng.randn(d); r-=float(r@xi)*xi
            rn=float(np.linalg.norm(r))
            if rn<1e-12: r=rng.randn(d); r-=float(r@xi)*xi; rn=float(np.linalg.norm(r))
            if rn<1e-12: out[k]=xi; k+=1; continue
            r/=rn
            th=float(abs(rng.randn())*rad)
            if th>0.6: th=0.6
            y=np.cos(th)*xi+np.sin(th)*r
            y/=max(float(np.linalg.norm(y)),1e-12)
            out[k]=y; k+=1
    return out

def solve():
    t0=time.time()
    DEADLINE=t0+172.0
    SEARCH_END=t0+124.0
    rng=np.random.RandomState(7)
    n=163
    try: bx=np.load("best.npy")
    except Exception:
        try: bx=np.load("record.npy")
        except Exception: bx=rng.randn(n,4)
    bx=np.asarray(bx,dtype=float)
    if bx.shape!=(n,4) or not np.all(np.isfinite(bx)): bx=rng.randn(n,4)
    bx=_norm(bx)
    best_x=bx.copy(); best_val=_max_cos(best_x)
    # stress on incumbent
    try:
        I,J,lam,adj=_self_stress(best_x, tol=1e-7)
        eorder=np.argsort(-lam)
    except Exception:
        I=J=None; eorder=None; adj=None
    trials=[]
    if eorder is not None:
        radii=[12,18,24,18]
        for ei in range(min(4,len(eorder))):
            k=radii[ei % len(radii)]
            trials.append((int(I[eorder[ei]]),int(J[eorder[ei]]),int(k)))
        # extra diversified radii if time
        trials.append((int(I[eorder[0]]),int(J[eorder[0]]),18))
        trials.append((int(I[eorder[1]]),int(J[eorder[1]]),24))
        trials.append((int(I[eorder[2]]),int(J[eorder[2]]),12))
        trials.append((int(I[eorder[0]]),int(J[eorder[0]]),24))
    else:
        trials=[(0,1,12),(0,1,18),(0,1,24),(0,1,18)]
    # recompute adjacency from tight graph each trial? use current best adjacency
    ti=0
    while time.time()<SEARCH_END and ti<len(trials):
        if time.time()>DEADLINE-45: break
        si,sj,k = trials[ti]; ti+=1
        # rebuild stress/adj from current best every 2 trials to track topology
        if ti%2==1:
            try:
                I2,J2,lam2,adj2=_self_stress(best_x, tol=1e-7)
                # keep trials aligned: replace remaining with fresh top edges
                # but don't discard loop; just update adj
                adj=adj2
                # refresh seeds for future trials
                eo2=np.argsort(-lam2)
                # append one fresh trial
                if len(eo2)>0 and time.time()<SEARCH_END-20:
                    trials.append((int(I2[eo2[0]]),int(J2[eo2[1 % len(eo2)]]),int([12,18,24][ti%3])))
            except Exception:
                pass
        if adj is None:
            # build simple adjacency from tight pairs
            try:
                Ia,Ja,_,_= _contact_pairs(best_x, tol=2e-7, cap=800)
                adj=[set() for _ in range(n)]
                for a,b in zip(Ia,Ja):
                    adj[int(a)].add(int(b)); adj[int(b)].add(int(a))
            except Exception:
                continue
        try:
            patch=_bfs_patch(adj,[si,sj],k)
        except Exception:
            continue
        keep=np.ones(n,dtype=bool); keep[patch]=False
        removed_pts=best_x[patch].copy()
        x_rem=best_x[keep].copy()
        nr=x_rem.shape[0]
        # freeze remainder: light anneal (no LP yet) to keep shape
        # two-stage: patch-only later, so keep frozen fixed now
        # greedy reinsertion
        cur_fixed=x_rem.copy()
        # pre-jittered pool from removed points
        try:
            Yjit=_jitter(removed_pts, rng, rad=0.15, m=250)
        except Exception:
            Yjit=np.empty((0,4))
        ok=True
        for ins in range(k):
            if time.time()>SEARCH_END-1.0: ok=False; break
            Y=rng.randn(20000,4); Y=_norm(Y)
            if Yjit.shape[0]>0:
                Y=np.vstack([Y,Yjit])
            D=Y@cur_fixed.T
            vals=D.max(axis=1)
            # top 20
            if len(vals)<=20: cand=np.arange(len(vals))
            else: cand=np.argpartition(vals,20)[:20]
            by=None; bvy=1e9
            for ci in cand:
                try:
                    yr,vr=_refine_hole(Y[ci],cur_fixed,iters=180)
                except Exception: continue
                if vr<bvy: bvy=vr; by=yr
            if by is None: ok=False; break
            cur_fixed=np.vstack([cur_fixed,by.reshape(1,4)])
            # remove near-duplicate jitter rows to diversify? keep as is
        if not ok: continue
        cur_fixed=_norm(cur_fixed)
        # stage 1: patch-only softmax anneal T 0.004->0.0004 1500 steps
        act=np.zeros(n,dtype=bool); act[nr:]=True
        try:
            cur_p,_=_gd(cur_fixed,steps=1500,T0=0.004,T1=0.0004,lr0=0.03,mu=0.86,
                        rng=rng,active=act,deadline=SEARCH_END,noise0=0.0005)
        except Exception:
            cur_p=cur_fixed
        # stage 2: all-point anneal
        try:
            cur_b,cur_v=_gd(cur_p,steps=2200,T0=0.008,T1=0.0004,lr0=0.035,mu=0.87,
                            rng=rng,deadline=SEARCH_END,noise0=0.001)
        except Exception:
            cur_b,cur_v=cur_p,_max_cos(cur_p)
        # LP polish if promising
        if cur_v<best_val+0.003 and time.time()<DEADLINE-8:
            try:
                lx,lv=_lp_polish(cur_b,deadline=min(DEADLINE-2.0,time.time()+4.5),r0=0.003)
                if lv<cur_v: cur_b,cur_v=lx,lv
            except Exception: pass
        if cur_v<best_val-1e-12:
            best_val=cur_v; best_x=cur_b.copy()
            # update adjacency to new best
            try:
                _,_,_,adj=_self_stress(best_x, tol=1e-7)
            except Exception: pass
            # save elite immediately
            try: np.save("best.npy",best_x)
            except Exception: pass
    # final polish of best
    if time.time()<DEADLINE-6:
        try:
            best_x,best_val=_lp_polish(best_x,deadline=DEADLINE-1.0,r0=0.004)
        except Exception: pass
    best_x=_norm(best_x)
    return best_x
