import time
import hashlib
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _topo_hash(x, m=None, thr=1e-6):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    if m is None:
        m = float(G.max())
    iu0 = np.triu_indices(n, 1)
    c = G[iu0]
    sel = c > m - thr
    I = iu0[0][sel]; J = iu0[1][sel]
    ids = (I.astype(np.int64)*n + J.astype(np.int64))
    ids.sort()
    h = hashlib.md5(ids.tobytes()).hexdigest()
    return h, m, G, ids, I, J

def _soft_cheap(x, steps=300):
    n, d = x.shape
    v = np.zeros_like(x)
    beta0, beta1 = 30.0, 500.0
    lb = np.log(beta1/beta0)
    mu = 0.85
    for t in range(steps):
        frac = t/max(1, steps-1)
        beta = beta0*np.exp(lb*frac)
        lr = 0.05*(1.0-0.5*frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        W = np.exp(beta*(G-gmax))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True)+1e-300
        A = (W @ x)/S
        dot = (A*x).sum(axis=1, keepdims=True)
        Gt = A - dot*x
        v = mu*v + Gt
        vd = (v*x).sum(axis=1, keepdims=True)
        v = v - vd*x
        x = x - lr*v
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(nr, 1e-12, out=nr)
        x /= nr
    return x

def _lp_polish(x, seconds):
    x = x/np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    if seconds <= 0:
        return xb
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time()-start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = c.max()
        sel = c > m - max(20*radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        ci = I[:,None]*d + np.arange(d)
        cj = J[:,None]*d + np.arange(d)
        cols = np.concatenate([ci, cj, np.full((p,1), n*d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d), np.arange(n*d))), shape=(n,nv)).tocsr()
            obj = np.zeros(nv); obj[-1]=1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
        except Exception:
            radius*=0.5
            continue
        if res.status!=0:
            radius*=0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n,d)
        except Exception:
            radius*=0.5
            continue
        y /= np.maximum(np.linalg.norm(y,axis=1,keepdims=True),1e-12)
        vv = _max_cos(y)
        if vv < best:
            best=vv; xb=y.copy(); x=y
            radius=min(radius*1.5,0.05)
        else:
            radius*=0.5
    return xb

def _refine_hole(z0, X, steps=150):
    z = z0.copy()/max(np.linalg.norm(z0),1e-12)
    for s in range(steps):
        frac=s/max(1,steps-1)
        beta=500.0*(10000.0/500.0)**frac
        lr=0.06*(0.008/0.06)**frac
        D = X @ z
        dm = D.max()
        W = np.exp(beta*(D-dm))
        S = W.sum()+1e-300
        g = (W @ X)/S
        gt = g - float(g @ z)*z
        z = z - lr*gt
        nrm=np.linalg.norm(z)
        if nrm<1e-12:
            break
        z/=nrm
    return z

def _find_hole(X_keep, rng, n_cand=50000):
    d = X_keep.shape[1]
    Z = rng.standard_normal((n_cand,d))
    Z/=np.maximum(np.linalg.norm(Z,axis=1,keepdims=True),1e-12)
    D = Z @ X_keep.T
    worst = D.max(axis=1)
    order = np.argsort(worst)
    # take top 5 and refine
    best_z=None; best_v=2.0
    for k in range(min(5,len(order))):
        z0 = Z[order[k]]
        z = _refine_hole(z0, X_keep, steps=120)
        v = float(X_keep @ z .max() if False else (X_keep @ z).max())
        if v < best_v:
            best_v=v; best_z=z
    if best_z is None:
        best_z=Z[order[0]]
    return best_z

def solve():
    t0=time.time()
    TIME_LIMIT=172.0
    try:
        xb=np.load("best.npy").astype(np.float64)
    except Exception:
        try:
            xb=np.load("record.npy").astype(np.float64)
        except Exception:
            rng0=np.random.default_rng(0)
            xb=rng0.standard_normal((292,4))
    if xb.shape!=(292,4):
        rng0=np.random.default_rng(0)
        xb=rng0.standard_normal((292,4))
    xb=xb/np.maximum(np.linalg.norm(xb,axis=1,keepdims=True),1e-12)
    best_score=_max_cos(xb)
    x_best=xb.copy()
    cur_X=xb.copy()
    cur_score=best_score
    h0,_,_,_,_,_= _topo_hash(xb)
    archive={h0:best_score}
    rng=np.random.default_rng(987654321)
    n,d=292,4
    FINAL_RESERVE=26.0
    explore_end=t0+TIME_LIMIT-FINAL_RESERVE
    it=0
    full_polishes=0
    # temperature schedule endpoints
    T0=2e-6; T1=1e-8
    while time.time()<explore_end:
        it+=1
        remain=explore_end-time.time()
        if remain<2.5:
            break
        # restart from best only every 10 iters
        if (it%10)==0:
            base=x_best.copy()
            base_score=best_score
        else:
            base=cur_X.copy()
            base_score=cur_score
        # --- analyse stress / degree of base ---
        G=base@base.T
        np.fill_diagonal(G,-2.0)
        mval=float(G.max())
        rowmax=G.max(axis=1)
        deg_loose=(G>mval-1e-4).sum(axis=1)
        deg_tight=(G>mval-1e-6).sum(axis=1)
        # pick target
        r=rng.random()
        if r<0.5:
            # high-stress edge: among tightest 30 pairs
            iu=np.triu_indices(n,1)
            c=G[iu]
            kTop=min(30,len(c))
            part=np.argpartition(-c,kTop-1)[:kTop]
            # pick one at random weighted to larger c
            pick=int(rng.choice(part))
            a=int(iu[0][pick]); b=int(iu[1][pick])
            # neighbourhood: nearest to a and b
            na=np.argsort(-G[a])[:12]
            nb=np.argsort(-G[b])[:12]
            neigh=np.unique(np.concatenate([[a,b],na,nb]))
        else:
            # over-coordinated point by loose degree + rowmax
            jitter=rng.random(n)*1e-6
            score_p=rowmax+0.001*deg_loose.astype(float)+jitter
            p=int(np.argmax(score_p))
            nn=np.argsort(-G[p])[:14]
            neigh=np.unique(np.concatenate([[p],nn]))
        kdel=int(rng.integers(1,4))  # 1-3
        kdel=min(kdel,len(neigh),3)
        # choose kdel points from neighbourhood biased to high rowmax/degree
        w=rowmax[neigh]+0.001*deg_loose[neigh].astype(float)+rng.random(len(neigh))*1e-6
        idx_local=np.argsort(-w)[:kdel]
        # add randomness: sometimes replace one with random
        if rng.random()<0.25 and len(neigh)>kdel:
            idx_local[0]=int(rng.integers(len(neigh)))
        del_idx=np.unique(neigh[idx_local])
        kdel=len(del_idx)
        mask=np.ones(n,dtype=bool); mask[del_idx]=False
        X_keep=base[mask]
        # --- sequential largest-hole reinsertion ---
        new_pts=[]
        Xcur=X_keep.copy()
        for _ in range(kdel):
            if time.time()>explore_end-1.0:
                break
            z=_find_hole(Xcur,rng,n_cand=50000)
            # tiny jitter
            z=z+1e-4*rng.standard_normal(4)
            z/=max(np.linalg.norm(z),1e-12)
            new_pts.append(z)
            Xcur=np.vstack([Xcur,z.reshape(1,4)])
        if len(new_pts)<kdel:
            break
        Xnew=np.vstack([X_keep,np.stack(new_pts,axis=0)])
        # cheap polish: 300 softmax steps + LP capped 2s
        try:
            Xc=_soft_cheap(Xnew.copy(),steps=300)
        except Exception:
            continue
        if not np.all(np.isfinite(Xc)):
            continue
        # quick score before LP
        try:
            Xc2=_lp_polish(Xc,min(1.5,max(0.3,remain-0.8)))
        except Exception:
            Xc2=Xc
        if not np.all(np.isfinite(Xc2)):
            continue
        cheap_score=_max_cos(Xc2)
        h,_,_,_,_,_= _topo_hash(Xc2)
        # archive rejection: hash seen and not better
        if h in archive and cheap_score>=archive[h]-1e-12 and cheap_score>=best_score-1e-12:
            # record but reject move (keep chain)
            # still update archive min
            if cheap_score<archive[h]:
                archive[h]=cheap_score
            # SA still could accept? spec says reject immediately
            continue
        # update archive
        if h not in archive or cheap_score<archive[h]:
            archive[h]=cheap_score
        is_new = h not in archive or True  # we already updated; track novelty
        # novelty: was hash unseen before this iter?
        # (we lost info) recompute: use flag
        # simpler: consider new if cheap within bound and not previously seen count>1
        # we approximate by checking if h was just added; keep set size logic
        # For decision use: hash not seen before update
        # To avoid complexity, treat as new if cheap_score < best_score+2e-5 and archive size grew
        # SA acceptance
        # temperature decay by elapsed fraction
        frac_el=(time.time()-t0)/max(1e-9,(explore_end-t0))
        frac_el=min(max(frac_el,0.0),1.0)
        T=T0*((T1/T0)**frac_el)
        delta=cheap_score-cur_score
        accept=False
        if cheap_score<cur_score-1e-15:
            accept=True
        elif cheap_score<best_score+5e-4:
            # allow worse with SA prob
            if delta<=0:
                accept=True
            else:
                p_acc=np.exp(-delta/max(T,1e-12))
                if rng.random()<p_acc:
                    accept=True
        else:
            # far worse: only tiny chance
            if delta>0 and rng.random()<np.exp(-delta/max(T,1e-12))*0.1:
                accept=True
        if accept:
            cur_X=Xc2.copy(); cur_score=cheap_score
        if cheap_score<best_score:
            best_score=cheap_score; x_best=Xc2.copy()
            cur_X=Xc2.copy(); cur_score=cheap_score
        # full LP polish only if hash new and within 2e-5 of incumbent
        # determine novelty: count occurrences -> if first time we saw h this run before update?
        # We already inserted, so check if archive[h]==cheap_score and cheap_score<=best_score+2e-5
        # and limit number of full polishes and time
        need_full=False
        if cheap_score < best_score+2e-5+1e-12 and full_polishes<3:
            # require that h was unseen prior to this iteration: approximate by
            # cheap_score <= archive second best? Use: if hash count ==1
            # Since we overwrote, use heuristic: if accept or near-best, do full once per ~15 iters
            if h not in [h0] and (it%3==0 or cheap_score<best_score+2e-6):
                need_full=True
        if need_full and (explore_end-time.time())>14.0:
            try:
                Xf=_lp_polish(Xc2.copy(),min(10.0,explore_end-time.time()-1.0))
                vf=_max_cos(Xf)
                hf,_,_,_,_,_=_topo_hash(Xf)
                if hf not in archive or vf<archive[hf]:
                    archive[hf]=vf
                full_polishes+=1
                if vf<best_score:
                    best_score=vf; x_best=Xf.copy()
                if vf<cur_score:
                    cur_X=Xf.copy(); cur_score=vf
                elif rng.random()<np.exp(-max(0.0,vf-cur_score)/max(T,1e-12)):
                    cur_X=Xf.copy(); cur_score=vf
            except Exception:
                pass
    # final full polish
    time_left=TIME_LIMIT-(time.time()-t0)-1.0
    time_left=max(5.0,min(30.0,time_left))
    try:
        x_best=_lp_polish(x_best,time_left)
    except Exception:
        pass
    x_best=x_best/np.maximum(np.linalg.norm(x_best,axis=1,keepdims=True),1e-12)
    if not np.all(np.isfinite(x_best)):
        x_best=xb/np.maximum(np.linalg.norm(xb,axis=1,keepdims=True),1e-12)
    return x_best
