import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _row_max_cnt(x, m=None):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    rm = G.max(axis=1)
    if m is None:
        m = float(rm.max())
    cnt = (G >= m - 1e-9).sum(axis=1)
    return rm, cnt, G, float(m)

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    if seconds <= 0.05:
        return x
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c = G[iu0, iu1]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        ns = int(sel.sum())
        if ns > 2500:
            idx = np.where(sel)[0]
            order = np.argsort(c[idx])[-2500:]
            m2 = np.zeros_like(sel)
            m2[idx[order]] = True
            sel = m2
        I = iu0[sel]; J = iu1[sel]; cv = c[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        cols = np.concatenate([(I[:, None]*d+np.arange(d)), (J[:, None]*d+np.arange(d)), np.full((p,1), n*d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2*d+1)
        try:
            A_ub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
            obj = np.zeros(nv); obj[-1]=1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n,d)
        yn = np.linalg.norm(y,axis=1,keepdims=True); yn[yn==0]=1.0
        y = y/yn
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius*1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _smooth(X, steps, K0, K1, lr0):
    n, d = X.shape
    m1 = np.zeros_like(X); m2 = np.zeros_like(X)
    b1=0.9; b2=0.999; eps=1e-8
    l0=np.log(K0); l1=np.log(K1)
    for s in range(steps):
        frac = s/max(1,steps-1) if steps>1 else 0.0
        K = float(np.exp(l0+(l1-l0)*frac))
        lr = lr0*(0.2+0.8*0.5*(1.0+np.cos(np.pi*frac)))
        G = X @ X.T
        np.fill_diagonal(G,-2.0)
        m = float(G.max())
        W = np.exp(K*(G-m))
        np.fill_diagonal(W,0.0)
        Gr = W @ X
        dot = np.sum(Gr*X,axis=1,keepdims=True)
        Gt = Gr - dot*X
        rs = W.sum(axis=1,keepdims=True)+1e-12
        Gt = Gt/rs
        m1 = b1*m1+(1.0-b1)*Gt
        m2 = b2*m2+(1.0-b2)*(Gt*Gt)
        c1 = 1.0-b1**(s+1); c2 = 1.0-b2**(s+1)
        step = (m1/c1)/(np.sqrt(m2/c2)+eps)
        X = X - lr*step
        nr=np.linalg.norm(X,axis=1,keepdims=True); nr[nr==0]=1.0
        X = X/nr
    return X

def _void_fill(rem, k, rng, n_cand=20000):
    d = rem.shape[1]
    cand = rng.standard_normal((n_cand,d))
    cand /= np.linalg.norm(cand,axis=1,keepdims=True)
    s = (cand @ rem.T).max(axis=1)
    chosen=[]
    for _ in range(k):
        idx=int(np.argmin(s))
        pt=cand[idx].copy()
        chosen.append(pt)
        upd=cand @ pt
        np.maximum(s,upd,out=s)
        s[idx]=2.0
    return np.stack(chosen,axis=0)

def solve():
    t0=time.time(); BUDGET=172.0; deadline=t0+BUDGET
    rng=np.random.default_rng(20261101)
    n,d=173,5
    best=None; best_val=1e9
    for fn in ("best.npy","record.npy"):
        try:
            a=np.load(fn).astype(np.float64)
            if a.shape==(n,d):
                an=a/np.linalg.norm(a,axis=1,keepdims=True)
                if np.all(np.isfinite(an)):
                    v=_max_cos(an)
                    if v<best_val: best_val=v; best=an.copy()
        except Exception:
            pass
    if best is None:
        best=rng.normal(size=(n,d)); best/=np.linalg.norm(best,axis=1,keepdims=True)
        best_val=_max_cos(best)
    archive=[best.copy()]; arch_val=[best_val]
    trial=0
    while True:
        rem=deadline-time.time()
        if rem<9.0: break
        lp_t=1.0
        steps=900
        if rem<30:
            lp_t=0.6; steps=500
        # pick base: mostly best, sometimes archive diverse, rarely fresh
        r0=rng.random()
        if r0<0.68:
            base=best
        elif r0<0.90 and len(archive)>1:
            base=archive[rng.integers(len(archive))]
        elif r0<0.95:
            # fresh random
            base=None
        else:
            base=best
        if base is None:
            y=rng.normal(size=(n,d)); y/=np.linalg.norm(y,axis=1,keepdims=True)
            K0,K1,lr0=8.0,250.0,0.04
            sm_steps=1200 if rem>40 else steps
        else:
            y=base.copy()
            rm,cnt,G,m=_row_max_cnt(y)
            mode=trial%8
            if mode==0:
                # cap deletion: random center, delete worst cap
                u=rng.normal(size=(d,)); u/=np.linalg.norm(u)
                proj=y@u
                k=int(rng.integers(8,16))
                victims=np.argsort(-proj)[:k]
                keep=np.ones(n,bool); keep[victims]=False
                try: holes=_void_fill(y[keep],k,rng,n_cand=20000)
                except Exception:
                    holes=rng.normal(size=(k,d)); holes/=np.linalg.norm(holes,axis=1,keepdims=True)
                y[victims]=holes+0.01*rng.normal(size=(k,d))
                y+=0.008*rng.normal(size=(n,d))
                K0,K1,lr0=10.0,300.0,0.035
                sm_steps=steps
            elif mode==1:
                k=int(rng.integers(6,13))
                order=np.argsort(-cnt-0.3*(rm-m))
                victims=order[:k]
                keep=np.ones(n,bool); keep[victims]=False
                try: holes=_void_fill(y[keep],k,rng,n_cand=25000)
                except Exception:
                    holes=rng.normal(size=(k,d)); holes/=np.linalg.norm(holes,axis=1,keepdims=True)
                y[victims]=holes+0.008*rng.normal(size=(k,d))
                # shake neighbours of most overloaded
                nb=np.where(G[victims[0]]>m-0.05)[0]
                if len(nb)>0: y[nb]+=0.02*rng.normal(size=(len(nb),d))
                y+=0.004*rng.normal(size=(n,d))
                K0,K1,lr0=12.0,320.0,0.03
                sm_steps=steps
            elif mode==2:
                # big global shake to new basin
                sig=0.20+rng.random()*0.20
                y=y+sig*rng.normal(size=(n,d))
                y/=np.linalg.norm(y,axis=1,keepdims=True)
                K0,K1,lr0=7.0,220.0,0.045
                sm_steps=1100 if rem>40 else steps
            elif mode==3:
                # rattler-anchored: move overloaded into rattler void explicitly
                ratt=np.where(rm<m-1e-7)[0]
                k=int(rng.integers(5,11))
                order=np.argsort(-cnt-0.3*(rm-m))
                victims=order[:k]
                keep=np.ones(n,bool); keep[victims]=False
                # bias candidates near rattler if exists
                if len(ratt)>0:
                    rc=y[ratt[rng.integers(len(ratt))]]
                    cand=rng.normal(size=(20000,d)); cand/=np.linalg.norm(cand,axis=1,keepdims=True)
                    # mix: half near rattler cap
                    mix=cand.copy()
                    # pull half toward rattler
                    half=10000
                    mix[:half]=mix[:half]+1.2*rc
                    mix[:half]/=np.linalg.norm(mix[:half],axis=1,keepdims=True)
                    s=(mix@y[keep].T).max(axis=1)
                    chosen=[]
                    for _ in range(k):
                        idx=int(np.argmin(s)); chosen.append(mix[idx].copy())
                        np.maximum(s,mix@mix[idx],out=s); s[idx]=2.0
                    holes=np.stack(chosen)
                else:
                    try: holes=_void_fill(y[keep],k,rng,n_cand=20000)
                    except Exception:
                        holes=rng.normal(size=(k,d)); holes/=np.linalg.norm(holes,axis=1,keepdims=True)
                y[victims]=holes+0.008*rng.normal(size=(k,d))
                y+=0.005*rng.normal(size=(n,d))
                K0,K1,lr0=11.0,300.0,0.032
                sm_steps=steps
            elif mode==4:
                y+=0.03*rng.normal(size=(n,d))
                K0,K1,lr0=14.0,280.0,0.028
                sm_steps=700 if rem>30 else 450
            elif mode==5:
                # dense-pair neighbourhood excision
                iu0,iu1=np.triu_indices(n,1)
                c=G[iu0,iu1]
                # pick a tight pair at random among top 50
                top=np.argsort(-c)[:50]
                pick=top[rng.integers(len(top))]
                a0,b0=iu0[pick],iu1[pick]
                # victims = union of neighbours of a0,b0
                na=np.where(G[a0]>m-0.04)[0]; nb=np.where(G[b0]>m-0.04)[0]
                uni=np.unique(np.concatenate([na,nb,np.array([a0,b0])]))
                if len(uni)>14: uni=rng.choice(uni,size=14,replace=False)
                k=len(uni)
                keep=np.ones(n,bool); keep[uni]=False
                try: holes=_void_fill(y[keep],k,rng,n_cand=20000)
                except Exception:
                    holes=rng.normal(size=(k,d)); holes/=np.linalg.norm(holes,axis=1,keepdims=True)
                y[uni]=holes+0.01*rng.normal(size=(k,d))
                y+=0.005*rng.normal(size=(n,d))
                K0,K1,lr0=10.0,300.0,0.035
                sm_steps=steps
            elif mode==6:
                # subset large shake
                sub=rng.choice(n,30,replace=False)
                y[sub]+=0.12*rng.normal(size=(len(sub),d))
                y+=0.01*rng.normal(size=(n,d))
                K0,K1,lr0=12.0,260.0,0.03
                sm_steps=800 if rem>30 else 500
            else:
                y+=0.08*rng.normal(size=(n,d))
                K0,K1,lr0=9.0,240.0,0.035
                sm_steps=800 if rem>30 else 500
            y/=np.linalg.norm(y,axis=1,keepdims=True)
            sm_steps=int(sm_steps)
        try:
            y=_smooth(y,sm_steps,K0,K1,lr0)
        except Exception:
            trial+=1; continue
        pt=min(lp_t,rem-7.0)
        if pt>=0.35:
            try: y=_lp_polish(y,pt)
            except Exception: pass
        try: v=_max_cos(y)
        except Exception: trial+=1; continue
        if np.isfinite(v):
            if v<best_val-1e-12:
                best=y; best_val=v
                archive.append(y.copy()); arch_val.append(v)
                if len(archive)>8:
                    # drop worst
                    wi=int(np.argmax(arch_val)); archive.pop(wi); arch_val.pop(wi)
            elif v<best_val+0.004:
                # keep diverse near-best
                if len(archive)<8:
                    archive.append(y.copy()); arch_val.append(v)
                else:
                    # replace worst if better
                    wi=int(np.argmax(arch_val))
                    if v<arch_val[wi]:
                        archive[wi]=y.copy(); arch_val[wi]=v
        trial+=1
    rem=deadline-time.time()
    if rem>1.5:
        try: best=_lp_polish(best,rem-0.7)
        except Exception: pass
    best=best/np.linalg.norm(best,axis=1,keepdims=True)
    return best
