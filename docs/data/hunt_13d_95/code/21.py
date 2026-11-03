import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def per_point_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)

def gd_refine(X, steps, T0, T1, lr0, lr1, mu=0.9):
    n, d = X.shape
    V = np.zeros_like(X)
    lT0 = np.log(T0); lT1 = np.log(T1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps-1)
        T = float(np.exp(lT0+(lT1-lT0)*f))
        lr = float(np.exp(ll0+(ll1-ll0)*f))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = float(G.max())
        W = np.exp((G-mx)/T)
        np.fill_diagonal(W, 0.0)
        sw = float(W.sum())
        if not np.isfinite(sw) or sw == 0:
            break
        W /= sw
        Grad = W @ X
        dot = np.sum(Grad*X, axis=1, keepdims=True)
        Gt = Grad - dot*X
        V = mu*V + Gt
        X = X - lr*V
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr==0]=1.0
        X /= nr
    return X

def deepest_hole_vec(others, rng, n_sample=4000, refine=200, lr0=0.15):
    d = others.shape[1]
    cand = rng.normal(size=(n_sample, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    md = dots.max(axis=1)
    j = int(np.argmin(md))
    y = cand[j].copy()
    lr = lr0
    for _ in range(refine):
        dts = others @ y
        k = int(np.argmax(dts))
        mval = float(dts[k])
        g = others[k] - mval*y
        gn = float(np.linalg.norm(g))
        if gn < 1e-12:
            break
        y = y - lr*g
        ny = float(np.linalg.norm(y))
        if ny==0 or not np.isfinite(ny):
            break
        y /= ny
        lr *= 0.996
    y /= np.linalg.norm(y)
    return y

def lp_polish(x, seconds, radius_init=1e-3, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = float(radius_init)
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time()-start < seconds and radius > 1e-14:
        if deadline is not None and time.time() > deadline:
            break
        if time.time()-start > seconds-0.05:
            break
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        thresh = m - max(20.0*radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 2200:
            idx = np.argpartition(c, -2200)[-2200:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx]=True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p==0:
            break
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1),n*d)],1).ravel()
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
        y /= np.linalg.norm(y,axis=1,keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius*1.5,0.05)
        else:
            radius*=0.5
    return x

def crossover(A,B,rng):
    n,d = A.shape
    try:
        cost = -(A @ B.T)
        _, col = linear_sum_assignment(cost)
        Bp = B[col]
    except Exception:
        Bp = B.copy()
    try:
        M = Bp.T @ A
        U,_,Vt = np.linalg.svd(M, full_matrices=False)
        R = U @ Vt
        Bp = Bp @ R
    except Exception:
        pass
    # align sign? already via rotation
    t = float(rng.uniform(0.35,0.65))
    C = (1-t)*A + t*Bp
    # add small noise to break symmetry
    C += 0.02*rng.normal(size=C.shape)
    C /= np.linalg.norm(C,axis=1,keepdims=True)
    return C

def solve():
    t0=time.time()
    BUDGET=170.0
    EXPL_T=126.0
    rng=np.random.default_rng(12345)
    np.random.seed(12345)
    try:
        xb=np.load("best.npy").astype(float)
    except Exception:
        xb=None
    try:
        xr=np.load("record.npy").astype(float)
    except Exception:
        xr=None
    cands=[]
    if xb is not None and xb.shape==(95,13):
        xb=xb/np.linalg.norm(xb,axis=1,keepdims=True)
        cands.append(xb)
    if xr is not None and xr.shape==(95,13):
        xr=xr/np.linalg.norm(xr,axis=1,keepdims=True)
        cands.append(xr)
    if len(cands)==0:
        x0=rng.normal(size=(95,13)); x0/=np.linalg.norm(x0,axis=1,keepdims=True)
        cands.append(x0)
    vals=[max_cos(c) for c in cands]
    bi=int(np.argmin(vals))
    x_best=cands[bi].copy(); best_val=float(vals[bi])
    elites=[(best_val,x_best.copy())]
    for c,v in zip(cands,vals):
        if v>best_val+1e-12:
            elites.append((float(v),c.copy()))
    elites=sorted(elites,key=lambda e:e[0])[:6]
    cur=x_best.copy(); cur_val=best_val
    n,d=95,13
    it=0
    sigmas=[0.008,0.02,0.05,0.10,0.20,0.35]
    while time.time()-t0 < EXPL_T:
        if time.time()-t0 > BUDGET-42:
            break
        it+=1
        u=rng.random()
        x_try=None
        T0,T1=0.016,0.0011
        lr0,lr1=0.10,0.008
        steps2=600
        # choose operator
        if u < 0.28:
            sigma=sigmas[int(rng.integers(0,len(sigmas)))]
            x_try=cur.copy()
            if rng.random()<0.30:
                ksub=int(rng.integers(4,14))
                idx=rng.choice(n,size=ksub,replace=False)
                x_try[idx]+=sigma*2.0*rng.normal(size=(ksub,d))
            else:
                x_try+=sigma*rng.normal(size=(n,d))
            x_try/=np.linalg.norm(x_try,axis=1,keepdims=True)
            steps2=int(rng.choice([400,600,800]))
        elif u < 0.44:
            centre=int(rng.integers(0,n))
            dots=cur[centre]@cur.T
            order=np.argsort(-dots)
            k=int(rng.integers(6,16))
            patch=order[1:1+k]
            x_try=cur.copy()
            s_big=float(rng.choice([0.08,0.15,0.25,0.40]))
            x_try[patch]+=s_big*rng.normal(size=(len(patch),d))
            x_try+=0.008*rng.normal(size=(n,d))
            x_try/=np.linalg.norm(x_try,axis=1,keepdims=True)
            steps2=int(rng.choice([500,700,900])); T0=0.018
        elif u < 0.62:
            ppm=per_point_max(cur)
            G=cur@cur.T; np.fill_diagonal(G,-2.0)
            cc=(G>cur_val-0.02).sum(axis=1)
            r=rng.random()
            if r<0.55:
                k=1 if rng.random()<0.5 else int(rng.integers(2,4))
                order=np.argsort(-ppm-0.001*cc)
                idx=order[:k].copy()
                if k>1 and rng.random()<0.35:
                    idx[-1]=int(rng.integers(0,n))
            elif r<0.75:
                k=int(rng.integers(1,4))
                idx=rng.choice(n,size=k,replace=False)
            else:
                # loose points (few contacts) or rattlers
                thr=cur_val-1e-4
                cnt=(G>thr).sum(axis=1)
                order=np.argsort(cnt)  # smallest first
                k=int(rng.integers(1,3))
                idx=order[:k].copy()
            x_try=cur.copy()
            for j in idx:
                mask=np.ones(n,dtype=bool); mask[int(j)]=False
                others=x_try[mask]
                ns=4000 if len(idx)==1 else 2200
                x_try[int(j)]=deepest_hole_vec(others,rng,n_sample=ns)
            x_try/=np.linalg.norm(x_try,axis=1,keepdims=True)
            steps2=int(rng.choice([500,700]))
        elif u < 0.74:
            G=cur@cur.T; np.fill_diagonal(G,-2.0)
            ij=np.unravel_index(int(np.argmax(G)),G.shape)
            a=int(ij[0])
            x_try=cur.copy()
            x_try[a]=x_try[a]-0.12*x_try[int(ij[1])]+0.05*rng.normal(size=(d,))
            x_try[a]/=np.linalg.norm(x_try[a])
            sigma=float(rng.choice([0.12,0.25,0.40]))
            x_try+=sigma*rng.normal(size=(n,d))
            x_try/=np.linalg.norm(x_try,axis=1,keepdims=True)
            steps2=800; T0,T1=0.020,0.0010; lr0,lr1=0.12,0.008
        elif u < 0.84:
            # crossover
            if len(elites)>=2:
                i1=int(rng.integers(0,len(elites))); i2=int(rng.integers(0,len(elites)))
                if i1==i2:
                    i2=(i2+1)%len(elites)
                A=elites[i1][1]; Bb=elites[i2][1]
                x_try=crossover(A,Bb,rng)
            else:
                x_try=cur.copy()+0.15*rng.normal(size=(n,d))
                x_try/=np.linalg.norm(x_try,axis=1,keepdims=True)
            steps2=int(rng.choice([500,700,800]))
        else:
            # cap re-opt: random direction cap shake
            ud=rng.normal(size=(d,)); ud/=np.linalg.norm(ud)
            proj=cur@ud
            m=int(rng.integers(6,20))
            order=np.argsort(-proj)
            cap=order[:m]
            x_try=cur.copy()
            s=float(rng.choice([0.10,0.20,0.35]))
            x_try[cap]+=s*rng.normal(size=(len(cap),d))
            x_try+=0.005*rng.normal(size=(n,d))
            x_try/=np.linalg.norm(x_try,axis=1,keepdims=True)
            steps2=int(rng.choice([500,700]))
            T0=0.017
        # staged GD: short screen then extend
        try:
            x_gd=gd_refine(x_try, 250, T0, 0.003, lr0, 0.01, mu=0.9)
        except Exception:
            continue
        v1=max_cos(x_gd)
        # discard clearly bad quickly
        if v1 > cur_val+0.006 and v1 > best_val+0.004 and rng.random()>0.02:
            # accept rarely as diversification
            if rng.random()<0.03:
                cur=x_gd.copy(); cur_val=v1
            # occasional restart
            if rng.random()<0.06:
                e=elites[int(rng.integers(0,len(elites)))]
                cur=e[1].copy(); cur_val=e[0]
            continue
        try:
            x_gd=gd_refine(x_gd, steps2, 0.006, T1, 0.06, lr1, mu=0.9)
        except Exception:
            continue
        v=max_cos(x_gd)
        do_lp=False; lp_sec=0.9
        if v < best_val-1e-12:
            do_lp=True; lp_sec=1.4
        elif v < best_val+0.0015 and (v < cur_val-0.0003 or rng.random()<0.07):
            do_lp=True; lp_sec=0.85
        if do_lp and time.time()-t0 < BUDGET-36:
            try:
                x_lp=lp_polish(x_gd, seconds=lp_sec, radius_init=0.006, deadline=t0+BUDGET-30)
            except Exception:
                x_lp=x_gd
            v_lp=max_cos(x_lp)
            if v_lp <= v+1e-9:
                x_gd,v=x_lp,v_lp
        if v < best_val-1e-12:
            x_best=x_gd.copy(); best_val=v
            cur=x_gd.copy(); cur_val=v
            elites.append((v,x_gd.copy()))
            elites=sorted(elites,key=lambda e:e[0])[:6]
        elif v < cur_val:
            cur=x_gd.copy(); cur_val=v
        else:
            if rng.random()<0.05 and v < cur_val+0.008:
                cur=x_gd.copy(); cur_val=v
            if rng.random()<0.10:
                if rng.random()<0.65 or len(elites)==1:
                    cur=x_best.copy(); cur_val=best_val
                else:
                    e=elites[int(rng.integers(0,len(elites)))]
                    cur=e[1].copy(); cur_val=e[0]
    # final squeeze
    elites=sorted(elites,key=lambda e:e[0])[:3]
    if len(elites)==0:
        elites=[(best_val,x_best.copy())]
    remain=BUDGET-(time.time()-t0)-1.0
    if remain < 3:
        xb2=elites[0][1]; xb2/=np.linalg.norm(xb2,axis=1,keepdims=True)
        return xb2
    total=float(np.clip(remain,4.0,38.0))
    best_final=elites[0][1].copy(); best_final_val=elites[0][0]
    for ei,(ev,ex) in enumerate(list(elites)):
        t_left=BUDGET-(time.time()-t0)-1.0
        if t_left < 2:
            break
        share=total*(0.6 if ei==0 else 0.4/max(1,len(elites)-1))
        share=min(share,t_left)
        if share < 1.5:
            continue
        try:
            if ei>0:
                ex=gd_refine(ex.copy(),400,0.008,0.0008,0.05,0.005)
            ex_p=lp_polish(ex,seconds=share*0.85,radius_init=8e-4,deadline=t0+BUDGET-1.0)
        except Exception:
            continue
        vp=max_cos(ex_p)
        if vp < best_final_val:
            best_final_val=vp; best_final=ex_p.copy()
    # micro taps around best
    t_end=t0+BUDGET-0.6
    rng2=rng
    curB=best_final.copy(); curBv=best_final_val
    while time.time() < t_end:
        amp=float(rng2.choice([3e-4,7e-4,1.5e-3,3e-3]))
        y=curB+amp*rng2.normal(size=curB.shape)
        y/=np.linalg.norm(y,axis=1,keepdims=True)
        try:
            y2=lp_polish(y,seconds=0.5,radius_init=amp*0.8,deadline=t_end)
        except Exception:
            continue
        v2=max_cos(y2)
        if v2 < curBv:
            curB,curBv=y2,v2
            if v2 < best_final_val:
                best_final, best_final_val=y2.copy(),v2
        if t_end-time.time()<1.2:
            break
    best_final/=np.linalg.norm(best_final,axis=1,keepdims=True)
    return best_final
