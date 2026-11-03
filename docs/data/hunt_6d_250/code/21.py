import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _smooth_full(X0, beta, maxiter, maxcor=10):
    n, d = X0.shape
    Y0 = X0.copy()
    def fg(Yflat):
        Y = Yflat.reshape(n, d)
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        X = Y / nrm
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        D = beta * (G - M)
        np.clip(D, -35.0, 0.0, out=D)
        E = np.empty_like(D)
        np.exp(D, out=E)
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E))
        if S <= 0 or not np.isfinite(S):
            S = 1e-300
        loss = M + np.log(S) / beta
        W = E / S
        GX = 2.0 * (W @ X)
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nrm
        return loss, GY.ravel()
    try:
        res = minimize(fg, Y0.ravel(), jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-20,
                                "gtol": 1e-12, "maxfun": maxiter*5,
                                "maxcor": maxcor})
        Y = res.x.reshape(n, d)
    except Exception:
        Y = Y0
    nrm = np.linalg.norm(Y, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    return Y / nrm

def _smooth_subset(X0, idx, beta, maxiter, maxcor=8):
    idx = np.asarray(idx, dtype=int)
    k = len(idx)
    if k == 0:
        return X0
    n, d = X0.shape
    Xfix = X0.copy()
    Y0 = X0[idx].copy()
    def fg(Yflat):
        Y = Yflat.reshape(k, d)
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        Xk = Y / nrm
        X = Xfix.copy()
        X[idx] = Xk
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        D = beta * (G - M)
        np.clip(D, -35.0, 0.0, out=D)
        E = np.empty_like(D)
        np.exp(D, out=E)
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E))
        if S <= 0 or not np.isfinite(S):
            S = 1e-300
        loss = M + np.log(S) / beta
        W = E / S
        GX = 2.0 * (W @ X)
        GXk = GX[idx]
        dot = np.sum(GXk * Xk, axis=1, keepdims=True)
        GY = (GXk - dot * Xk) / nrm
        return loss, GY.ravel()
    try:
        res = minimize(fg, Y0.ravel(), jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-20,
                                "gtol": 1e-12, "maxfun": maxiter*5,
                                "maxcor": maxcor})
        Y = res.x.reshape(k, d)
    except Exception:
        Y = Y0
    nrm = np.linalg.norm(Y, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    X = Xfix.copy()
    X[idx] = Y / nrm
    return X

def _lp_polish(x, deadline, radius_init=1.5e-3, cap=4500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius_init
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cmat = x @ x.T
        iu_c = cmat[iu]
        m = float(np.max(iu_c))
        sel = iu_c > m - max(20.0*radius, 6e-7)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            thr = np.sort(iu_c[sel])[-cap]
            sel = iu_c >= thr
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n*d+1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:, None]*d+np.arange(d),
                               J[:, None]*d+np.arange(d),
                               np.full((p,1), n*d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))], 1).ravel()
        A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
        A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        obj = np.zeros(nv); obj[-1]=1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-iu_c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n,d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius*1.5, 0.012)
        else:
            radius *= 0.5
    return x

def _void_pt(X0, exclude, rng, nc=2048):
    n, d = X0.shape
    cand = rng.standard_normal((nc, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    S = cand @ X0.T
    if np.ndim(exclude)==0:
        S[:, int(exclude)] = -2.0
    else:
        for w in exclude:
            S[:, int(w)] = -2.0
    mx = S.max(axis=1)
    return cand[int(np.argmin(mx))]

def solve():
    t0 = time.time()
    BUDGET = 168.0
    HOP_END = t0 + 128.0
    LP_END = t0 + 163.0
    rng = np.random.default_rng(3)
    n, d = 250, 6
    cands = []
    try:
        a = np.asarray(np.load("best.npy"), dtype=np.float64)
        if a.shape==(n,d):
            cands.append(a/np.linalg.norm(a,axis=1,keepdims=True))
    except Exception:
        pass
    try:
        b = np.asarray(np.load("record.npy"), dtype=np.float64)
        if b.shape==(n,d):
            cands.append(b/np.linalg.norm(b,axis=1,keepdims=True))
    except Exception:
        pass
    if not cands:
        t = rng.standard_normal((n,d)); t/=np.linalg.norm(t,axis=1,keepdims=True)
        cands.append(t)
    best = min(cands, key=_max_cos).copy()
    best_val = _max_cos(best)
    iu0, iu1 = np.triu_indices(n,1)
    # stage 1: quick global anneal
    cur = best.copy()
    for beta,it in [(3000.0,80),(12000.0,80)]:
        if time.time() > t0+28.0:
            break
        try:
            cur = _smooth_full(cur, beta, it)
        except Exception:
            break
        v = _max_cos(cur)
        if v < best_val:
            best_val=v; best=cur.copy()
    # stage 2: focused basin hopping
    trial=0
    while time.time() < HOP_END and trial < 90:
        trial+=1
        try:
            G = best @ best.T
            np.fill_diagonal(G,-2.0)
            rowmax = G.max(axis=1)
            order = np.argsort(-rowmax)
            flat = G[iu0,iu1]
            # bottleneck pair
            pkmax = int(np.argmax(flat))
            ib, jb = int(iu0[pkmax]), int(iu1[pkmax])
            K = min(600, flat.size-1)
            topk = np.argpartition(-flat, K)[:K]
            topk = topk[np.argsort(-flat[topk])]
        except Exception:
            order=np.arange(n); topk=np.array([0]); ib,jb=0,1
        X0 = best.copy()
        mode = trial % 8
        try:
            if mode==0:  # separate bottleneck pair
                step = 0.012 if (trial%2==0) else 0.006
                X0[ib] -= step*X0[jb]
                X0[jb] -= step*X0[ib]
                kk = order[:4]
                X0[kk] += 0.005*rng.standard_normal((len(kk),d))
                X0 += 0.0008*rng.standard_normal((n,d))
                # subset = bottleneck neighbourhood
                S = [ib,jb]+[int(x) for x in order[2:8]]
                S = sorted(set(S))[:10]
            elif mode==1:  # void reinsert worst
                w=int(order[0])
                X0[w]=_void_pt(X0,w,rng,nc=2048)
                X0+=0.001*rng.standard_normal((n,d))
                S=[w]+[int(x) for x in order[1:6]]
                S=sorted(set(S))[:8]
            elif mode==2:  # double void
                w1,w2=int(order[0]),int(order[1])
                X0[w1]=_void_pt(X0,[w1,w2],rng,nc=1536)
                X0[w2]=_void_pt(X0,[w1,w2],rng,nc=1024)
                X0+=0.0007*rng.standard_normal((n,d))
                S=[w1,w2]+[int(x) for x in order[2:7]]
                S=sorted(set(S))[:9]
            elif mode==3:  # kick worst K
                Kk=4+(trial%7)
                idxk=order[:Kk]
                sig=0.011 if (trial%2==0) else 0.006
                X0[idxk]+=sig*rng.standard_normal((Kk,d))
                X0+=0.0008*rng.standard_normal((n,d))
                S=[int(x) for x in idxk[:10]]
            elif mode==4:  # top-pair endpoints subset heavy shake
                # collect endpoints of top 6 pairs
                Sset=set()
                for t in range(6):
                    pk=int(topk[t % len(topk)])
                    Sset.add(int(iu0[pk])); Sset.add(int(iu1[pk]))
                S=sorted(Sset)[:12]
                X0[S]+=0.009*rng.standard_normal((len(S),d))
                X0+=0.0006*rng.standard_normal((n,d))
            elif mode==5:  # random subset + worst
                S=[int(x) for x in order[:5]]
                rsel=rng.integers(0,n,size=8)
                for r in rsel:
                    if int(r) not in S: S.append(int(r))
                S=S[:12]
                X0[S]+=0.008*rng.standard_normal((len(S),d))
            elif mode==6:  # heavy single bottleneck
                X0[ib]+=0.020*rng.standard_normal(d)
                X0[jb]+=0.009*rng.standard_normal(d)
                kk=order[2:6]
                X0[kk]+=0.004*rng.standard_normal((len(kk),d))
                S=[ib,jb]+[int(x) for x in kk]
                S=sorted(set(S))
            else:  # global small shake
                X0+=0.0022*rng.standard_normal((n,d))
                kk=order[:5]
                X0[kk]+=0.006*rng.standard_normal((len(kk),d))
                S=[int(x) for x in kk]+[ib,jb]
                S=sorted(set(S))
            X0/=np.linalg.norm(X0,axis=1,keepdims=True)
        except Exception:
            continue
        beta1 = 6000.0 if (trial%2==0) else 14000.0
        try:
            Y = _smooth_subset(X0, S, beta1, 70)
        except Exception:
            continue
        v = _max_cos(Y)
        # short global relax every other trial if promising
        if v < best_val+0.002 and (trial%2==0) and time.time()<HOP_END:
            try:
                Y2 = _smooth_full(Y, beta1, 35)
                v2 = _max_cos(Y2)
                if v2 < v:
                    Y,v = Y2,v2
            except Exception:
                pass
        # extra snap at high beta if close
        if v < best_val+0.0008 and time.time()<HOP_END:
            try:
                Y3 = _smooth_subset(Y, S, 24000.0, 30)
                v3 = _max_cos(Y3)
                if v3 < v:
                    Y,v = Y3,v3
            except Exception:
                pass
        if v < best_val:
            best_val=v; best=Y.copy()
        if t0+BUDGET-(time.time()) < 34.0:
            break
        # periodic LP squeeze
        if trial%12==0 and time.time()<LP_END-10.0:
            try:
                dl = min(time.time()+4.0, LP_END-6.0)
                if dl > time.time()+1.0:
                    p = _lp_polish(best, dl)
                    vp=_max_cos(p)
                    if vp < best_val:
                        best,best_val=p,vp
            except Exception:
                pass
    # stage 3: LP squeeze + snap
    if time.time() < LP_END:
        try:
            p=_lp_polish(best, min(LP_END, time.time()+12.0))
            vp=_max_cos(p)
            if vp<best_val:
                best,best_val=p,vp
        except Exception:
            pass
    if time.time() < t0+BUDGET-4.0:
        try:
            Yf=_smooth_full(best, 22000.0, 50)
            vf=_max_cos(Yf)
            if vf<best_val:
                best,best_val=Yf,vf
        except Exception:
            pass
        try:
            if time.time()<LP_END:
                p2=_lp_polish(best, time.time()+4.0, radius_init=6e-4)
                v2=_max_cos(p2)
                if v2<best_val:
                    best= p2
        except Exception:
            pass
    best=best/np.linalg.norm(best,axis=1,keepdims=True)
    best=np.where(np.isfinite(best),best,0.0)
    nrm=np.linalg.norm(best,axis=1,keepdims=True)
    nrm=np.maximum(nrm,1e-12)
    best=best/nrm
    return best
