import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def smooth_opt(X, steps, b0, b1, lr0, lr1, mu=0.9):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n = X.shape[0]
    eye2 = 2.0 * np.eye(n)
    V = np.zeros_like(X)
    for k in range(steps):
        f = k / (steps - 1) if steps > 1 else 0.0
        beta = b0 + (b1 - b0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        m = float(np.max(G - eye2))
        A = beta * (G - m)
        np.clip(A, -12.0, 0.0, out=A)
        np.fill_diagonal(A, -30.0)
        np.exp(A, out=A)
        Grad = A @ X
        d = np.sum(Grad * X, axis=1, keepdims=True)
        T = Grad - d * X
        s = float(np.mean(np.sqrt(np.sum(T*T, axis=1)))) + 1e-12
        T /= s
        V *= mu
        V += (1.0 - mu) * T
        X -= lr * V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def lp_polish(x, budget):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < budget and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = c.max()
        sel = c > m - max(20*radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)],1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
        A_ub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
        A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        obj = np.zeros(nv); obj[-1]=1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
        except Exception:
            radius*=0.5
            continue
        if res.status!=0:
            radius*=0.5
            continue
        y = x + res.x[:-1].reshape(n,d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x,best,best_x = y,v,y.copy()
            radius = min(radius*1.5,0.05)
        else:
            radius*=0.5
        if time.time()-start>=budget:
            break
    return best_x

def contacts_of(X, thr=1e-7):
    G = X @ X.T
    np.fill_diagonal(G,-2.0)
    mx = float(G.max())
    cnt = np.sum(G > mx-thr, axis=1)
    return cnt, mx

def select_removal(best, cnt, k, mode, rng):
    n = best.shape[0]
    ratt = np.where(cnt==0)[0]
    order = np.argsort(-cnt, kind="stable")
    if mode=="hub":
        if k>=8:
            sel=set()
            for r in ratt:
                sel.add(int(r))
                if len(sel)>=k: break
            for h in order[:4]:
                sel.add(int(h))
                if len(sel)>=k: break
            if len(sel)<k:
                h0=int(order[0])
                dots=best@best[h0]
                nbr=np.argsort(-dots, kind="stable")
                for idx in nbr:
                    sel.add(int(idx))
                    if len(sel)>=k: break
            return list(sel)[:k]
        else:
            sel=[]
            for r in ratt:
                sel.append(int(r))
            for o in order:
                if len(sel)>=k: break
                if int(o) not in sel:
                    sel.append(int(o))
            return sel[:k]
    elif mode=="cap_hub":
        h0=int(order[0])
        center=best[h0]
        dots=best@center
        idx=np.argsort(-dots, kind="stable")[:k].tolist()
        if len(ratt)>0 and int(ratt[0]) not in idx:
            idx[-1]=int(ratt[0])
        return [int(v) for v in idx]
    elif mode=="cap_random":
        c=rng.standard_normal(4); c/=np.linalg.norm(c)
        dots=best@c
        idx=np.argsort(-dots, kind="stable")[:k]
        return [int(v) for v in idx]
    else:  # cluster
        if len(ratt)>0 and rng.random()<0.4:
            seed=int(ratt[0])
        elif rng.random()<0.7:
            seed=int(order[0])
        else:
            seed=int(rng.integers(0,n))
        dots=best@best[seed]
        idx=np.argsort(-dots, kind="stable")[:k]
        return [int(v) for v in idx]

def find_hole(Y, rng, n_cand=20000, n_best=8, local_steps=220):
    m=Y.shape[0]
    C=rng.standard_normal((n_cand,4))
    C/=np.linalg.norm(C,axis=1,keepdims=True)
    S=C@Y.T
    sm=S.max(axis=1)
    idx=np.argpartition(sm, n_best)[:n_best]
    Cb=C[idx].copy()
    # batched local optimisation of min max-dot via softmax repulsion
    for k in range(local_steps):
        f=k/(local_steps-1) if local_steps>1 else 0.0
        beta=30.0+(140.0-30.0)*f
        lr=0.08+(0.004-0.08)*f
        D=Cb@Y.T  # B x m
        dm=D.max(axis=1,keepdims=True)
        E=np.exp(beta*(D-dm))
        E/=E.sum(axis=1,keepdims=True)
        G=E@Y  # B x 4
        proj=np.sum(G*Cb,axis=1,keepdims=True)
        T=G-proj*Cb
        Cb-=lr*T
        Cb/=np.linalg.norm(Cb,axis=1,keepdims=True)
    S2=Cb@Y.T
    sm2=S2.max(axis=1)
    return Cb[int(np.argmin(sm2))].copy()

def solve():
    t0=time.time()
    HARD=176.0
    n,d=153,4
    rng=np.random.default_rng(20261031)
    best=None; best_val=1e9
    for fn in ("best.npy","record.npy"):
        try:
            v=np.load(fn).astype(float)
            if v.shape==(n,d):
                v/=np.linalg.norm(v,axis=1,keepdims=True)
                c=max_cos(v)
                if c<best_val:
                    best_val=c; best=v.copy()
        except Exception:
            pass
    if best is None:
        best=rng.standard_normal((n,d))
        best/=np.linalg.norm(best,axis=1,keepdims=True)
        best_val=max_cos(best)
    ks=[8,12,20,30]
    modes=["hub","cap_hub","cap_random","cluster"]
    # build task list cycling
    tasks=[]
    for rep in range(30):
        for k in ks:
            for md in modes:
                tasks.append((k,md))
    rng.shuffle(tasks)
    trial=0
    for (k,mode) in tasks:
        trial+=1
        elapsed=time.time()-t0
        if elapsed>HARD-22.0:
            break
        try:
            cnt,_=contacts_of(best)
        except Exception:
            cnt=np.zeros(n)
        rem=select_removal(best,cnt,k,mode,rng)
        mask=np.ones(n,dtype=bool); mask[np.array(rem,dtype=int)]=False
        Y=best[mask].copy()
        # optimise survivors: easier subconfiguration
        steps_rem=900 if k<=12 else 1300
        # reduce steps if time short
        if elapsed>HARD-60.0:
            steps_rem=min(steps_rem,700)
        Y=smooth_opt(Y,steps_rem,30.0,300.0,0.020,0.002,mu=0.9)
        # reinsert one at a time
        ok=True
        for t in range(k):
            if time.time()-t0>HARD-20.0:
                ok=False; break
            # hole search size
            time_left=HARD-(time.time()-t0)
            if time_left<25.0:
                n_cand=8000
                ls=120
            elif k>=20:
                n_cand=15000
                ls=180
            else:
                n_cand=22000
                ls=220
            cstar=find_hole(Y,rng,n_cand=n_cand,n_best=8,local_steps=ls)
            Y=np.vstack([Y,cstar.reshape(1,4)])
            Y=smooth_opt(Y,120,40.0,150.0,0.015,0.002,mu=0.9)
            if (t+1)%5==0 or t==k-1:
                # short LP polish after each block
                tl=HARD-(time.time()-t0)
                if tl>22.0:
                    Y=lp_polish(Y,min(1.2,tl-20.0))
        if not ok:
            continue
        v=max_cos(Y)
        if v<best_val+0.015:
            tl=HARD-(time.time()-t0)
            if tl>8.0:
                b=min(4.0 if v<best_val+0.002 else 1.5, tl-6.0)
                if b>0.6:
                    Yp=lp_polish(Y,b)
                    vp=max_cos(Yp)
                else:
                    Yp,vp=Y,v
            else:
                Yp,vp=Y,v
            if vp<best_val-1e-12:
                best,best_val=Yp,vp
        # opportunistic rattler randomisation if stuck: tiny extra trial
        if trial%7==0 and time.time()-t0<HARD-30.0:
            pass
    # final polish
    tl=HARD-(time.time()-t0)-1.0
    if tl>2.0:
        best=lp_polish(best,min(tl,20.0))
    best/=np.linalg.norm(best,axis=1,keepdims=True)
    return best
