import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n < 1e-12] = 1.0
    return x / n

def _adam(x_init, steps, T0, T1, lr0, lr1, time_limit, it_off=0):
    t0 = time.time()
    X = _norm(x_init.copy())
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    bv = float(np.max(G0))
    bx = X.copy()
    m1 = np.zeros_like(X); m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lt0, lt1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for it in range(steps):
        if time.time() - t0 > time_limit:
            break
        frac = it / max(1, steps-1)
        T = float(np.exp(lt0 + (lt1-lt0)*frac))
        lr = float(np.exp(ll0 + (ll1-ll0)*frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        if m < bv:
            bv = m; bx = X.copy()
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        s = float(np.sum(E)*0.5)
        if not np.isfinite(s) or s <= 0:
            continue
        Gr = (E @ X) / s
        Gt = Gr - np.sum(Gr*X, axis=1, keepdims=True)*X
        gl = it+1+it_off
        m1 *= b1; m1 += (1-b1)*Gt
        m2 *= b2; m2 += (1-b2)*(Gt*Gt)
        m1h = m1/(1-b1**gl); m2h = m2/(1-b2**gl)
        step = m1h/(np.sqrt(m2h)+eps)
        X = X - lr*step
        X = _norm(X)
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    if m < bv:
        bv = m; bx = X.copy()
    return bx, bv

def _greedy(x, iters, t_end):
    x = _norm(x.copy())
    best = _max_cos(x)
    step = 4e-4
    n = x.shape[0]
    for _ in range(iters):
        if time.time() > t_end:
            break
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        k = int(np.argmax(G))
        i = k // n; j = k % n
        dot = float(G[i, j])
        vi = dot*x[i]-x[j]; vj = dot*x[j]-x[i]
        ni = float(np.linalg.norm(vi)); nj = float(np.linalg.norm(vj))
        if ni < 1e-14 or nj < 1e-14:
            step *= 0.5
            if step < 1e-13:
                step = 2e-4
            continue
        vi/=ni; vj/=nj
        y = x.copy()
        y[i]=x[i]+step*vi; y[j]=x[j]+step*vj
        y[i]/=np.linalg.norm(y[i]); y[j]/=np.linalg.norm(y[j])
        v = _max_cos(y)
        if v < best:
            x=y; best=v; step=min(step*1.25,5e-3)
            continue
        y2=x.copy(); y2[i]=x[i]+step*vi; y2[i]/=np.linalg.norm(y2[i])
        v2=_max_cos(y2)
        y3=x.copy(); y3[j]=x[j]+step*vj; y3[j]/=np.linalg.norm(y3[j])
        v3=_max_cos(y3)
        if v2<best and v2<=v3:
            x=y2; best=v2; step=min(step*1.15,5e-3)
        elif v3<best:
            x=y3; best=v3; step=min(step*1.15,5e-3)
        else:
            step*=0.5
            if step<1e-13:
                step=2e-4
    return x, best

def _relocate_worst(x, rng, t_end, max_try=35):
    x = _norm(x.copy())
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -10.0)
    rowmax = np.max(G, axis=1)
    order = np.argsort(-rowmax)
    cur_global = float(np.max(G))
    # pairs not involving i max, precompute? do per i
    for ii in range(min(max_try, n)):
        if time.time() > t_end:
            break
        i = int(order[ii])
        mi = float(rowmax[i])
        # only work on relatively bad points
        if mi < cur_global - 5e-06 and ii > 10:
            continue
        others = np.delete(x, i, axis=0)  # (n-1,d)
        # sample candidates
        S = 2500
        C = rng.randn(S, d); C = _norm(C)
        sc = C @ others.T
        smax = np.max(sc, axis=1)
        bi = int(np.argmin(smax))
        y = C[bi].copy()
        yv = float(smax[bi])
        # refine best few
        # take top 3 seeds
        idx = np.argpartition(smax, 3)[:3]
        best_y = y; best_v = yv
        for s0 in idx:
            yy = C[int(s0)].copy()
            lr = 0.06
            for st in range(350):
                dots = others @ yy
                j = int(np.argmax(dots))
                dv = float(dots[j])
                g = others[j] - dv*yy
                gn = float(np.linalg.norm(g))
                if gn < 1e-14:
                    break
                # move away from worst neighbor
                yy = yy - lr * (g/gn if gn>0 else g)
                yy = yy/np.linalg.norm(yy)
                lr *= 0.995
                if time.time() > t_end:
                    break
            dv2 = float(np.max(others @ yy))
            if dv2 < best_v:
                best_v = dv2; best_y = yy.copy()
        if best_v < mi - 1e-12:
            # compute new global
            others_max = cur_global
            # max among others (excluding i): compute quickly
            Go = others @ others.T
            np.fill_diagonal(Go, -2.0)
            others_max = float(np.max(Go))
            new_global = best_v if best_v > others_max else others_max
            if new_global < cur_global - 1e-12:
                xn = x.copy(); xn[i] = best_y
                x = xn
                cur_global = new_global
                # update G/rowmax for remaining
                G = x @ x.T
                np.fill_diagonal(G, -10.0)
                rowmax = np.max(G, axis=1)
    return _norm(x), cur_global

def _lp_polish(x, seconds, t_deadline):
    x = _norm(x.copy())
    try:
        best = _max_cos(x)
    except Exception:
        return x
    best_x = x.copy()
    n, d = x.shape
    radius = 9e-4
    iu = np.triu_indices(n, 1)
    t0 = time.time()
    while time.time() < t_deadline and (time.time()-t0) < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(np.max(c))
        sel = c > m - max(20*radius, 1e-9)
        cnt = int(np.sum(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 3200:
            idx = np.argpartition(-c, 3200)[:3200]
            mask = np.zeros_like(sel, dtype=bool); mask[idx]=True
            sel = sel & mask
            cnt = 3200
        I, J = iu[0][sel], iu[1][sel]
        p = cnt
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1),n*d)],1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
        try:
            A_ub = coo_matrix((vals,(rows,cols)),shape=(p,n*d+1)).tocsr()
            A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,n*d+1)).tocsr()
        except Exception:
            radius*=0.5; continue
        obj=np.zeros(n*d+1); obj[-1]=1.0
        try:
            res=linprog(obj,A_ub=A_ub,b_ub=-c[sel],A_eq=A_eq,b_eq=np.zeros(n),bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        y=x+res.x[:-1].reshape(n,d)
        y=_norm(y)
        v=_max_cos(y)
        if v<best:
            best=v; best_x=y.copy(); x=y
            radius=min(radius*1.5,0.05)
        else:
            radius*=0.5
    return best_x

def solve():
    np.random.seed(0)
    t_start=time.time()
    BUDGET=168.0
    def rem():
        return BUDGET-(time.time()-t_start)
    try:
        x0=np.load("best.npy")
    except Exception:
        try:
            x0=np.load("record.npy")
        except Exception:
            x0=np.random.RandomState(0).randn(182,9)
    x0=np.asarray(x0,dtype=np.float64)
    if x0.shape!=(182,9):
        rng0=np.random.RandomState(1)
        y=rng0.randn(182,9)
        if x0.ndim==2:
            mh=min(x0.shape[0],182); mw=min(x0.shape[1],9)
            y[:mh,:mw]=x0[:mh,:mw]
        x0=y
    x0=_norm(x0)
    best_val=_max_cos(x0)
    x_best=x0.copy()
    # Stage 1 broad anneal
    if rem()>25:
        xb,vb=_adam(x_best,steps=11000,T0=0.004,T1=0.00035,lr0=0.003,lr1=0.00025,time_limit=min(28.0,rem()-20))
        if vb<best_val:
            best_val=vb; x_best=xb
    # Stage 2 exploration: kicks + relocate + short anneal
    it_off=11000
    for trial in range(12):
        if rem()<55:
            break
        rng=np.random.RandomState(1000+trial*77)
        G=x_best@x_best.T
        np.fill_diagonal(G,-10.0)
        flat=G[np.triu_indices(182,1)]
        thr=np.sort(flat)[-500]
        iu0,iu1=np.triu_indices(182,1)
        msk=flat>thr-1e-12
        cnt=np.zeros(182)
        for a,b in zip(iu0[msk],iu1[msk]):
            cnt[a]+=1; cnt[b]+=1
        cnt=cnt/max(1.0,cnt.max())
        sigmas=[0.03,0.05,0.02,0.07,0.035,0.06,0.025,0.045,0.08,0.03,0.055,0.04]
        sb=sigmas[trial%len(sigmas)]
        noise=rng.randn(182,9)
        scale=(sb*(0.4+cnt))[:,None]
        Xp=x_best+scale*noise
        # push current worst pair apart strongly
        try:
            k=int(np.argmax(flat)); a=int(iu0[k]); b=int(iu1[k])
            v=rng.randn(9); v/=np.linalg.norm(v)+1e-12
            Xp[a]+=sb*2.0*v; Xp[b]-=sb*2.0*v
        except Exception:
            pass
        Xp=_norm(Xp)
        xb,vb=_adam(Xp,steps=3800,T0=0.0018,T1=0.00022,lr0=0.002,lr1=0.00018,time_limit=min(9.0,rem()-40),it_off=it_off)
        it_off+=3800
        # relocate worst points (hole filling)
        if rem()>40:
            t_end=time.time()+2.2
            xr,vr=_relocate_worst(xb,rng,t_end,maxtry=30)
            # short re-equalize after relocation
            if vr<vb+1e-09:
                xb2,vb2=_adam(xr,steps=2200,T0=0.0008,T1=0.0002,lr0=0.001,lr1=0.0002,time_limit=min(6.0,rem()-35),it_off=it_off)
                it_off+=2200
                # keep better of xb,xb2
                if vb2<vb:
                    xb,vb=xb2,vb2
                else:
                    # check relocated directly
                    if vr<vb:
                        xb,vb=xr,vr
        if vb<best_val-1e-12:
            best_val=vb; x_best=xb.copy()
    # Stage 3 sharp fine
    if rem()>14:
        xb,vb=_adam(x_best,steps=6000,T0=0.0006,T1=0.00012,lr0=0.0008,lr1=0.00012,time_limit=min(16.0,rem()-10),it_off=it_off)
        if vb<best_val:
            best_val=vb; x_best=xb
        # greedy on max pair
        if rem()>10:
            xg,vg=_greedy(x_best,iters=2500,t_end=time.time()+min(6.0,rem()-8))
            if vg<best_val:
                best_val=vg; x_best=xg
    # Stage 4 LP polish remainder
    r=rem()-1.5
    if r>4:
        r=min(r,38.0)
        try:
            t_dead=time.time()+r
            x_best=_lp_polish(x_best,r,t_dead)
            best_val=_max_cos(x_best)
        except Exception:
            pass
    x_best=_norm(x_best)
    x_best[~np.isfinite(x_best)]=0.1
    x_best=_norm(x_best)
    return x_best
