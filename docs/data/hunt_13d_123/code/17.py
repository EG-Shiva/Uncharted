import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, p_cap=1400):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 1.2e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    itc = 0
    while time.time() - start < seconds and radius > 1e-14 and itc < 40:
        itc += 1
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = c_all.max()
        window = max(25 * radius, 2e-9)
        sel = c_all > m - window
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > p_cap:
            idx = np.where(sel)[0]
            part = np.argpartition(-c_all[idx], p_cap - 1)[:p_cap]
            keep = np.zeros_like(sel)
            keep[idx[part]] = True
            sel = keep
        I = iu0[sel]; J = iu1[sel]
        csel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([ (I[:,None]*d+np.arange(d)), (J[:,None]*d+np.arange(d)), np.full((p,1), n*d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d)+[(None,None)], method="highs",
                          options={"time_limit": max(0.3, seconds - (time.time()-start))})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn==0]=1.0
        y = y/yn
        v = _max_cos(y)
        if v < best:
            best = v; best_x = y.copy(); x = y
            radius = min(radius*1.6, 0.03)
        else:
            radius *= 0.5
        if time.time()-start >= seconds:
            break
    return best_x

def _smooth_subset(y, frozen_mask, steps, T0, T1, lr0, lr1):
    # frozen_mask True = frozen; only update free points but repulsion from all
    n, d = y.shape
    free = ~frozen_mask
    if not np.any(free):
        return y
    for s in range(steps):
        frac = s/max(steps-1,1)
        T = T0+(T1-T0)*frac
        lr = lr0+(lr1-lr0)*frac
        G = y @ y.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        E = np.exp((G-m)/T)
        np.fill_diagonal(E, 0.0)
        row = E.sum(axis=1, keepdims=True)+1e-20
        Avg = (E @ y)/row
        dot = (Avg*y).sum(axis=1, keepdims=True)
        Gt = Avg - dot*y
        y[free] = y[free] - lr*Gt[free]
        y[free] /= np.linalg.norm(y[free], axis=1, keepdims=True)
    return y

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    LIM = 175.0
    rng = _np.random.default_rng(7)
    n, d = 123, 13
    X0=None; bv=1e9
    for fn in ("best.npy","record.npy"):
        try:
            a=_np.load(fn); a=_np.asarray(a,float)
            if a.shape==(n,d):
                a/= _np.linalg.norm(a,axis=1,keepdims=True)
                v=_max_cos(a)
                if v<bv: bv=v; X0=a.copy()
        except Exception:
            pass
    if X0 is None:
        X0=rng.normal(size=(n,d)); X0/=_np.linalg.norm(X0,axis=1,keepdims=True); bv=_max_cos(X0)
    cur=X0.copy(); cur_val=float(bv)
    best=X0.copy(); best_val=float(bv)
    # quick initial tightening
    try:
        if _time.time()-t0 < LIM-40:
            cand=_lp_polish(best, 2.0)
            v=_max_cos(cand)
            if v<best_val: best=cand.copy(); best_val=v; cur=cand.copy(); cur_val=v
    except Exception:
        pass
    it=0
    while _time.time()-t0 < LIM-16:
        it+=1
        remain = LIM-16-(_time.time()-t0)
        if remain<2.5: break
        # diagnostics on cur
        Gc = cur @ cur.T
        _np.fill_diagonal(Gc,-2.0)
        wmax = Gc.max(axis=1)
        thr = cur_val-0.008
        counts=(Gc>thr).sum(axis=1)
        worst=_np.argsort(-wmax)
        ratt=_np.argsort(counts)
        trial=cur.copy()
        mode=it%6
        # choose free subset size
        if mode==0 or mode==3:
            # repack worst neighbourhood: pick worst hub + its top neighbours
            hub=int(worst[0])
            gh=cur@cur[hub]
            gh[hub]=-2
            nbr=_np.argsort(-gh)[:10]
            free_idx=set([hub]+nbr.tolist())
            free_idx.add(int(ratt[0]))
            for k in range(2): free_idx.add(int(worst[k+1]))
            for k in range(2): free_idx.add(int(rng.integers(0,n)))
            free_idx=list(free_idx)
            frozen=_np.ones(n,bool); frozen[free_idx]=False
            # reinsert one of them into hole vs rest
            # pick the rattler/low contact inside subset
            r=int(ratt[0])
            mask=_np.ones(n,bool); mask[r]=False
            Xr=trial[mask]
            Y=rng.normal(size=(6000,d)); Y/= _np.linalg.norm(Y,axis=1,keepdims=True)
            md=(Y@Xr.T).max(axis=1)
            order=_np.argpartition(md,4)[:4]
            pick=order[int(rng.integers(0,4))]
            v=Y[pick]+0.003*rng.normal(size=(d,)); v/= _np.linalg.norm(v)
            trial[r]=v
            trial[frozen]+=0.002*rng.normal(size=(frozen.sum(),d))
            # renormalize jittered frozen slightly (keep approx frozen)
            trial/= _np.linalg.norm(trial,axis=1,keepdims=True)
            # smooth mainly subset
            if it%2==0:
                trial=_smooth_subset(trial,frozen,steps=220,T0=0.018,T1=0.003,lr0=0.4,lr1=0.04)
            else:
                trial=_smooth_subset(trial,frozen,steps=180,T0=0.010,T1=0.002,lr0=0.3,lr1=0.03)
            # tiny global smooth
            trial=_smooth_subset(trial,_np.zeros(n,bool),steps=30,T0=0.004,T1=0.002,lr0=0.08,lr1=0.02)
        elif mode==1 or mode==4:
            # targeted kick on worst + rattlers
            targets=set()
            for k in range(min(3,n)): targets.add(int(worst[k]))
            for k in range(min(2,n)): targets.add(int(ratt[k]))
            for k in range(2): targets.add(int(rng.integers(0,n)))
            trial+=0.003*rng.normal(size=trial.shape)
            for idx in targets:
                trial[idx]+=0.16*rng.normal(size=(d,))
            trial/=_np.linalg.norm(trial,axis=1,keepdims=True)
            frozen=_np.zeros(n,bool)  # all free but short run
            trial=_smooth_subset(trial,frozen,steps=200,T0=0.022,T1=0.003,lr0=0.4,lr1=0.04)
        else:
            # hole reinsertion of 1-2 worst
            nrep=2 if mode==5 else 1
            repl=[int(worst[k]) for k in range(nrep)]
            for r in repl:
                mask=_np.ones(n,bool); mask[r]=False
                Xr=trial[mask]
                Y=rng.normal(size=(8000,d)); Y/=_np.linalg.norm(Y,axis=1,keepdims=True)
                md=(Y@Xr.T).max(axis=1)
                order=_np.argpartition(md,4)[:4]
                pick=order[int(rng.integers(0,4))]
                v=Y[pick]+0.004*rng.normal(size=(d,)); v/=_np.linalg.norm(v)
                trial[r]=v
            trial+=0.0015*rng.normal(size=trial.shape)
            trial/=_np.linalg.norm(trial,axis=1,keepdims=True)
            trial=_smooth_subset(trial,_np.zeros(n,bool),steps=220,T0=0.016,T1=0.003,lr0=0.35,lr1=0.04)
        # LP snap
        try:
            lp_sec=0.9 if remain>25 else 0.6
            trial=_lp_polish(trial, lp_sec)
        except Exception:
            pass
        v=_max_cos(trial)
        if v<best_val-1e-14:
            best_val=v; best=trial.copy(); cur=trial.copy(); cur_val=v
        else:
            tol=0.003*_np.exp(-it/30.0)+0.00015
            if v<cur_val+tol:
                cur=trial.copy(); cur_val=v
            if it%25==0:
                cur=best+0.008*rng.normal(size=best.shape)
                cur/=_np.linalg.norm(cur,axis=1,keepdims=True)
                cur_val=_max_cos(cur)
        if _time.time()-t0>=LIM-16: break
    fsec=max(1.0, LIM-(_time.time()-t0)-0.8)
    fsec=min(fsec,15.0)
    try:
        pol=_lp_polish(best,fsec)
        pv=_max_cos(pol)
        if pv<best_val: best=pol; best_val=pv
    except Exception:
        pass
    best/= _np.linalg.norm(best,axis=1,keepdims=True)
    return best
