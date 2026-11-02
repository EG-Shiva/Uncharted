import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _personal_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1), g

def _lp_polish(x, deadline, r_init=0.002, r_max=0.02, cap=2000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r_init
    iu0 = np.triu_indices(n, 1)
    nv = n * d + 1
    obj = np.zeros(nv); obj[-1] = 1.0
    while time.time() < deadline and radius > 1e-13:
        if time.time() + 0.5 > deadline:
            break
        G = x @ x.T
        c_all = G[iu0]
        m = c_all.max()
        thresh = max(20.0 * radius, 1e-9)
        sel = c_all > m - thresh
        if np.count_nonzero(sel) > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        c_sel = c_all[sel]
        cols = np.concatenate([I[:, None]*d + np.arange(d), J[:, None]*d + np.arange(d), np.full((p,1), n*d)],1).ravel()
        rows = np.repeat(np.arange(p), 2*d+1)
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
        A_ub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
        A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        bounds = [(-radius,radius)]*(n*d)+[(None,None)]
        try:
            res = linprog(obj,A_ub=A_ub,b_ub=-c_sel,A_eq=A_eq,b_eq=np.zeros(n),bounds=bounds,method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n,d)
        yn = np.linalg.norm(y,axis=1,keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn<1e-12):
            radius *= 0.5
            continue
        y = y/yn
        v = _max_cos(y)
        if v < best:
            x=y; best=v
            radius=min(radius*1.5,r_max)
        else:
            radius*=0.5
    return x,best

def _quench(X0, steps, lr, tau0, tau1, iso_w, seed_off, rng):
    n,d = X0.shape
    X = X0.copy()
    X /= np.linalg.norm(X,axis=1,keepdims=True)
    m1 = np.zeros_like(X); m2 = np.zeros_like(X)
    b1,b2,eps = 0.9,0.999,1e-8
    best = _max_cos(X)
    bestX = X.copy()
    target = (n/d)
    for s in range(steps):
        frac = s/max(1,steps-1)
        tau = tau0*((tau1/tau0)**frac) if tau1>0 else tau0
        # Gram
        G = X @ X.T
        # personal max not needed; global max for shift
        # set diag to very low so it never dominates
        np.fill_diagonal(G, -1.0)
        m = G.max()
        # softmax weights
        # clip for stability
        E = np.exp((G-m)/tau)
        np.fill_diagonal(E, 0.0)
        sE = E.sum()
        if not np.isfinite(sE) or sE<=0:
            break
        W = E/sE
        Grad = W @ X  # smooth-max grad
        if iso_w>0:
            S = X.T @ X
            D = S - target*np.eye(d)
            Grad = Grad + iso_w*(X @ D)
        # tangential
        dot = (Grad*X).sum(axis=1,keepdims=True)
        Gt = Grad - dot*X
        # Adam
        m1 = b1*m1+(1-b1)*Gt
        m2 = b2*m2+(1-b2)*(Gt*Gt)
        c1 = 1-b1**(s+1); c2 = 1-b2**(s+1)
        mh = m1/c1; vh = m2/c2
        step = lr*(0.4+0.6*(1-frac)) / (np.sqrt(vh)+eps) * mh
        # clip step norm per row to avoid blowup
        sn = np.linalg.norm(step,axis=1,keepdims=True)+1e-12
        cap = 0.05
        scale = np.minimum(1.0, cap/(sn+1e-12))
        step = step*scale
        X = X - step
        X /= np.linalg.norm(X,axis=1,keepdims=True)
        if (s&15)==15 or s==steps-1:
            v = _max_cos(X)
            if v<best:
                best=v; bestX=X.copy()
    return bestX,best

def _holes(X, k, rng, n_probe=2500):
    n,d = X.shape
    P = rng.standard_normal((n_probe,d))
    P /= np.linalg.norm(P,axis=1,keepdims=True)
    D = P @ X.T
    md = D.max(axis=1)
    idx = np.argsort(md)[:k]
    return P[idx]

def solve():
    t0=time.time()
    HARD_END=t0+172.0
    rng=np.random.default_rng(12345)
    n,d=78,15
    cand=[]
    try:
        b=np.load("best.npy")
        if b.shape==(n,d): cand.append(b.astype(float))
    except Exception: pass
    try:
        r=np.load("record.npy")
        if r.shape==(n,d): cand.append(r.astype(float))
    except Exception: pass
    if len(cand)==0:
        x0=rng.standard_normal((n,d)); x0/=np.linalg.norm(x0,axis=1,keepdims=True)
        cand.append(x0)
    best_v=1e9; best_x=None
    for c in cand:
        c=c/np.linalg.norm(c,axis=1,keepdims=True)
        v=_max_cos(c)
        if v<best_v: best_v=v; best_x=c.copy()
    x_best=best_x.copy(); v_best=best_v

    # quick light quench of starting point to establish baseline (no LP yet)
    # main exploratory loop with fast quenches
    trial=0
    # reserve 28s for final LP
    while time.time() < HARD_END-28:
        trial+=1
        rem=HARD_END-28-time.time()
        if rem<1.0: break
        mode=trial%6
        x_try=x_best.copy()
        if mode==1:
            x_try = x_try + 0.02*rng.standard_normal((n,d))
        elif mode==2:
            x_try = x_try + 0.09*rng.standard_normal((n,d))
        elif mode==3:
            # subset shake: 12-20 points hard
            k=int(rng.integers(10,22))
            idx=rng.choice(n,k,replace=False)
            x_try = x_try + 0.008*rng.standard_normal((n,d))
            x_try[idx] += 0.35*rng.standard_normal((k,d))
        elif mode==4:
            # void reinsertion: replace worst 2-4 by deepest holes
            pm,_=_personal_max(x_try)
            order=np.argsort(-pm)
            k=2+(trial%3)  # 2..4
            victims=order[:k]
            H=_holes(x_try,k,rng)
            H+=0.01*rng.standard_normal((k,d))
            H/=np.linalg.norm(H,axis=1,keepdims=True)
            x_try[victims]=H
            # jitter rest lightly to break jamming
            x_try+=0.006*rng.standard_normal((n,d))
        elif mode==5:
            # cluster intermix: split by top PC, move points across
            try:
                U,Sv,Vt=np.linalg.svd(x_try-np.mean(x_try,axis=0,keepdims=True),full_matrices=False)
                pc=Vt[0]
                proj=x_try@pc
                med=np.median(proj)
                left=np.where(proj<=med)[0]; right=np.where(proj>med)[0]
                k=min(8,len(left),len(right))
                li=rng.choice(left,k,replace=False); ri=rng.choice(right,k,replace=False)
                # swap with noise: pull each toward opposite side centroid
                cl=np.mean(x_try[left],axis=0); cr=np.mean(x_try[right],axis=0)
                x_try[li]+=0.4*(cr-cl)+0.15*rng.standard_normal((k,d))
                x_try[ri]+=0.4*(cl-cr)+0.15*rng.standard_normal((k,d))
                x_try+=0.005*rng.standard_normal((n,d))
            except Exception:
                x_try+=0.05*rng.standard_normal((n,d))
        else: # mode 0: worst-point targeted shake
            pm,_=_personal_max(x_try)
            order=np.argsort(-pm)
            k=8
            worst=order[:k]
            x_try+=0.004*rng.standard_normal((n,d))
            x_try[worst]+=0.12*rng.standard_normal((k,d))
        x_try/=np.linalg.norm(x_try,axis=1,keepdims=True)
        # choose quench hyperparams by mode
        if mode in (2,3,5):
            steps,lr,tau0,tau1,iso=900,0.015,0.02,0.0025,2e-4
        elif mode==4:
            steps,lr,tau0,tau1,iso=1100,0.012,0.015,0.002,2e-4
        else:
            steps,lr,tau0,tau1,iso=700,0.01,0.012,0.002,1e-4
        # adapt steps to remaining time
        if rem<6:
            steps=min(steps,400)
        x_q,v_q=_quench(x_try,steps,lr,tau0,tau1,iso,0,rng)
        if v_q < v_best-1e-12:
            x_best=x_q; v_best=v_q
        # also accept near-equal with better isotropy? no, keep strictly best
        if time.time()>=HARD_END-28:
            break

    # final intensive: short LP squeeze
    if time.time()<HARD_END-1:
        # first a fine quench on best to smooth
        if time.time()<HARD_END-26:
            x_best,v2=_quench(x_best,800,0.008,0.008,0.0015,1e-4,0,rng)
            if v2<v_best: v_best=v2; x_best=x_q if False else v2 and x_best  # keep
            # _quench returns best; ensure x_best updated
            x_best=v2 if isinstance(v2,np.ndarray) else x_best
            # fix: _quench returns tuple, handle
            # recompute correctly (safety)
            pass
        # re-evaluate (in case above messed)
        x_best=x_best/np.linalg.norm(x_best,axis=1,keepdims=True) if np.all(np.finite(x_best)) else best_x
        # Actually re-run clean fine quench if time allows
        # (guard against variable shadowing above)
        try:
            v_best=_max_cos(x_best)
        except Exception:
            x_best=best_x.copy(); v_best=_max_cos(x_best)
        x_best,v_best=_lp_polish(x_best,min(HARD_END-0.4,time.time()+24),r_init=0.004,r_max=0.02,cap=2200)
        if time.time()<HARD_END-0.6:
            x_best,v_best=_lp_polish(x_best,HARD_END-0.3,r_init=0.0008,r_max=0.006,cap=2500)
    x_best=x_best/np.linalg.norm(x_best,axis=1,keepdims=True)
    if not np.all(np.isfinite(x_best)):
        x_best=best_x/np.linalg.norm(best_x,axis=1,keepdims=True)
    return x_best
