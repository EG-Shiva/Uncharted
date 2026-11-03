import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _per_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _e8_roots():
    pts = []
    d = 8
    for i in range(d):
        for j in range(i+1, d):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = np.zeros(d)
                    v[i] = si; v[j] = sj
                    pts.append(v)
    for mask in range(256):
        v = np.ones(d)*0.5
        nminus = 0
        for k in range(d):
            if (mask >> k) & 1:
                v[k] = -0.5; nminus += 1
        if nminus % 2 == 0:
            pts.append(v)
    A = np.array(pts, dtype=np.float64)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    return A

def _balanced_quench(X, steps, b0, b1, lr0, lr1, lam=0.12, mu=0.85, t_lim=1e18):
    n, d = X.shape
    X = X.copy()
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    best_m = _max_cos(X)
    best = X.copy()
    V = np.zeros_like(X)
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    om = 1.0 - mu
    for s in range(steps):
        if (s & 127) == 0 and time.time() > t_lim:
            break
        frac = s / max(steps-1, 1)
        beta = np.exp(lb0 + (lb1-lb0)*frac)
        lr = np.exp(ll0 + (ll1-ll0)*frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        if m < best_m:
            best_m = m; best = X.copy()
        E = np.exp(beta*(G-m))
        np.fill_diagonal(E, 0.0)
        Sg = E.sum()/max(n,1) + 1e-30
        Gg = (E @ X)/Sg
        bl = beta*0.25
        if bl < 8.0: bl = 8.0
        mx = G.max(axis=1, keepdims=True)
        W = np.exp(bl*(G-mx))
        np.fill_diagonal(W, 0.0)
        Rs = W.sum(axis=1, keepdims=True)+1e-30
        Gl = (W @ X)/Rs
        Grad = Gg + lam*Gl
        dot = (Grad*X).sum(axis=1, keepdims=True)
        Gt = Grad - dot*X
        nn = np.sqrt((Gt*Gt).sum(axis=1, keepdims=True))+1e-12
        Gt = Gt*np.minimum(1.0, 3.0/(nn+1e-12))
        V = mu*V + om*Gt
        X = X - lr*V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return best, best_m

def _overlap_quench(X, T, steps, lr0, lr1, lam_bg=0.02, mu=0.85, t_lim=1e18):
    n, d = X.shape
    X = X.copy()
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    best_m = _max_cos(X)
    best = X.copy()
    V = np.zeros_like(X)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    om = 1.0 - mu
    for s in range(steps):
        if (s & 127) == 0 and time.time() > t_lim:
            break
        frac = s / max(steps-1, 1)
        lr = np.exp(ll0 + (ll1-ll0)*frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        if m < best_m:
            best_m = m; best = X.copy()
            if best_m < T:
                pass
        H = G - T
        np.maximum(H, 0.0, out=H)
        np.fill_diagonal(H, 0.0)
        Go = (H @ X)
        # weak background to keep rattlers moving
        Gb = G.copy()
        np.fill_diagonal(Gb, 0.0)
        np.maximum(Gb, 0.0, out=Gb)
        Gbg = (Gb @ X)*(1.0/n)
        Grad = Go + lam_bg*Gbg
        dot = (Grad*X).sum(axis=1, keepdims=True)
        Gt = Grad - dot*X
        nn = np.sqrt((Gt*Gt).sum(axis=1, keepdims=True))+1e-12
        Gt = Gt*np.minimum(1.0, 2.0/(nn+1e-12))
        V = mu*V + om*Gt
        X = X - lr*V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return best, best_m

def _hard_polish(X, steps=160, lr0=0.003, lr1=0.00012, tol0=3e-4, tol1=2e-6):
    X = X.copy()
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    for s in range(steps):
        frac = s/max(steps-1,1)
        tol = tol0+(tol1-tol0)*frac
        lr = lr0*(lr1/lr0)**frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        M = (G > (m-tol)).astype(np.float64)
        np.fill_diagonal(M, 0.0)
        cnt = M.sum(axis=1, keepdims=True)+1e-12
        act = (cnt[:,0] > 0.5)
        if not np.any(act):
            continue
        Grad = (M @ X)/cnt
        dot = (Grad*X).sum(axis=1, keepdims=True)
        Gt = Grad-dot*X
        nn = np.sqrt((Gt*Gt).sum(axis=1, keepdims=True))+1e-12
        step = np.zeros_like(Gt)
        step[act] = Gt[act]/nn[act]
        X = X - lr*step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _refine_hole(Y, p, iters=150, beta=90.0, lr0=0.12, lr1=0.004):
    p = p/np.linalg.norm(p)
    for it in range(iters):
        frac = it/max(iters-1,1)
        lr = lr0*(lr1/lr0)**frac
        g = Y@p
        gm = g.max()
        w = np.exp(beta*(g-gm))
        s = w.sum()+1e-18
        A = (w@Y)/s
        At = A-float(A@p)*p
        p = p-lr*At
        nr = np.linalg.norm(p)
        if nr == 0: break
        p /= nr
    return p

def _find_holes(Y, k, rng):
    d = Y.shape[1]
    n_probe = 8000
    P = rng.randn(n_probe, d)
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    bs = np.empty(n_probe)
    CH = 2000
    for a in range(0, n_probe, CH):
        b = min(a+CH, n_probe)
        bs[a:b] = (P[a:b] @ Y.T).max(axis=1)
    order = np.argsort(bs)
    holes = []
    for idx in order:
        p = P[idx]
        ok = True
        for q in holes:
            if abs(float(p@q)) > 0.88:
                ok = False; break
        if ok:
            Yext = Y if len(holes)==0 else np.vstack([Y, np.array(holes)])
            pr = _refine_hole(Yext, p, iters=160, beta=80.0+rng.rand()*50.0)
            holes.append(pr)
            if len(holes) >= k: break
    j = 0
    while len(holes) < k:
        p = P[order[j % len(order)]]; j += 1
        Yext = Y if len(holes)==0 else np.vstack([Y, np.array(holes)])
        holes.append(_refine_hole(Yext, p, iters=120, beta=90.0))
    return holes

def _lp_polish(x, seconds, r0=1.2e-3):
    x = x/np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time()-start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20*radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5; continue
        if p > 3000:
            idx = np.argsort(c[sel])[-3000:]
            I, J = I[idx], J[idx]
            cc = c[sel][idx]
        else:
            cc = c[sel]
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)],1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
        A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n),d), np.arange(n*d))), shape=(n,nv)).tocsr()
        obj = np.zeros(nv); obj[-1]=1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
        except Exception:
            radius *= 0.5; continue
        if res.status != 0:
            radius *= 0.5; continue
        y = x + res.x[:-1].reshape(n,d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius*1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    BUDGET = 174.0
    t_explore = t0 + 128.0
    n, d = 168, 8
    rng = np.random.RandomState(0)
    best = None; best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape != (n,d): continue
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)): continue
            va = _max_cos(a)
            if va < best_val:
                best_val = va; best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.randn(n,d); best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    E8 = _e8_roots()
    elite = [(best_val, best.copy())]
    def push(v, X):
        elite.append((v, X.copy()))
        elite.sort(key=lambda e: e[0])
        del elite[5:]
    it = 0
    while time.time() < t_explore:
        if time.time()+2.0 > t0+BUDGET: break
        it += 1
        r = rng.rand()
        cur = elite[0][1]
        if r < 0.32:
            X = cur.copy()
            G = X@X.T; np.fill_diagonal(G,-2.0)
            mcur = G.max()
            imax = G.max(axis=1)
            cnt = (G > mcur-5e-7).sum(axis=1)
            k = 7+int(rng.randint(10))
            victims = set()
            ow = np.argsort(imax); oc = np.argsort(cnt)
            for _ in range(k):
                u = rng.rand()
                if u < 0.5:
                    for idx in ow[::-1]:
                        if idx not in victims: victims.add(int(idx)); break
                elif u < 0.8:
                    for idx in oc[::-1]:
                        if idx not in victims: victims.add(int(idx)); break
                else:
                    vv = int(rng.randint(n))
                    while vv in victims: vv = int(rng.randint(n))
                    victims.add(vv)
            victims = sorted(victims)
            keep = np.ones(n,bool); keep[victims]=False
            Y = X[keep]
            holes = _find_holes(Y, k, rng)
            for vi,h in zip(victims, holes):
                X[vi]=h
            X += rng.randn(n,d)*0.0035
            X /= np.linalg.norm(X,axis=1,keepdims=True)
            Tt = best_val - (0.0005+rng.rand()*0.002)
            Xo, _ = _overlap_quench(X, Tt, 500+int(rng.randint(300)), 0.006, 0.0006, lam_bg=0.03+rng.rand()*0.04, t_lim=t_explore)
            b0 = 25.0+rng.rand()*25.0; b1 = 300.0+rng.rand()*350.0
            Xb, vb = _balanced_quench(Xo, 800+int(rng.randint(400)), b0, b1, 0.005, 0.0004, lam=0.05+rng.rand()*0.10, t_lim=t_explore)
            Xb = _hard_polish(Xb, steps=120, lr0=0.002, lr1=0.0002)
            vb = _max_cos(Xb)
            if vb < elite[-1][0]+1e-7:
                push(vb, Xb)
            if vb < best_val:
                best_val = vb; best = Xb.copy()
        elif r < 0.52:
            idx = rng.choice(240, n, replace=False)
            X = E8[idx].copy()
            X += rng.randn(n,d)*(0.008+rng.rand()*0.012)
            X /= np.linalg.norm(X,axis=1,keepdims=True)
            Tt = best_val - (0.0+rng.rand()*0.002)
            Xo, _ = _overlap_quench(X, Tt, 500, 0.006, 0.0006, lam_bg=0.04, t_lim=t_explore)
            Xb, vb = _balanced_quench(Xo, 900+int(rng.randint(400)), 20.0+rng.rand()*20.0, 320.0+rng.rand()*300.0, 0.005, 0.0004, lam=0.06+rng.rand()*0.08, t_lim=t_explore)
            Xb = _hard_polish(Xb, steps=120, lr0=0.002, lr1=0.0002)
            vb = _max_cos(Xb)
            if vb < elite[-1][0]+1e-7: push(vb, Xb)
            if vb < best_val: best_val=vb; best=Xb.copy()
        elif r < 0.67:
            X = rng.randn(n,d); X/=np.linalg.norm(X,axis=1,keepdims=True)
            Tt = best_val-(0.001+rng.rand()*0.003)
            Xo,_ = _overlap_quench(X, Tt, 600, 0.007, 0.0007, lam_bg=0.05, t_lim=t_explore)
            if time.time()>t_explore: continue
            Xb,vb = _balanced_quench(Xo, 900, 30.0, 500.0+rng.rand()*200.0, 0.005, 0.0004, lam=0.06, t_lim=t_explore)
            Xb=_hard_polish(Xb,steps=100,lr0=0.002,lr1=0.0002)
            vb=_max_cos(Xb)
            if vb < elite[-1][0]+1e-7: push(vb,Xb)
            if vb < best_val: best_val=vb; best=Xb.copy()
        elif r < 0.83:
            X = cur.copy()
            imax = _per_max(X)
            worst = np.argsort(imax)[-24:]
            X[worst] += rng.randn(len(worst),d)*(0.06+rng.rand()*0.08)
            mask = np.ones(n,bool); mask[worst]=False
            X[mask] += rng.randn(mask.sum(),d)*0.004
            X/=np.linalg.norm(X,axis=1,keepdims=True)
            Tt = best_val-(rng.rand()*0.002)
            Xo,_=_overlap_quench(X,Tt,450,0.006,0.0006,lam_bg=0.03,t_lim=t_explore)
            Xb,vb=_balanced_quench(Xo,850+int(rng.randint(350)),30.0+rng.rand()*20.0,280.0+rng.rand()*300.0,0.005,0.0004,lam=0.05+rng.rand()*0.07,t_lim=t_explore)
            Xb=_hard_polish(Xb,steps=100,lr0=0.002,lr1=0.0002)
            vb=_max_cos(Xb)
            if vb < elite[-1][0]+1e-7: push(vb,Xb)
            if vb < best_val: best_val=vb; best=Xb.copy()
        else:
            if len(elite) < 2: continue
            A = elite[0][1]; Bi = elite[1+rng.randint(len(elite)-1)][1]
            M = Bi.T@A
            U,S,Vt = np.linalg.svd(M)
            R = U@Vt
            Bal = Bi@R
            try:
                C = -(A@Bal.T)
                ri, ci = linear_sum_assignment(C)
                Bp = np.zeros_like(Bal); Bp[ri]=Bal[ci]
            except Exception:
                Bp = Bal
            mk = rng.rand(n) < 0.5
            X = A.copy(); X[~mk]=Bp[~mk]
            X += rng.randn(n,d)*0.004
            X/=np.linalg.norm(X,axis=1,keepdims=True)
            Xb,vb=_balanced_quench(X,850,25.0,400.0+rng.rand()*200.0,0.005,0.0004,lam=0.07,t_lim=t_explore)
            Xb=_hard_polish(Xb,steps=100,lr0=0.002,lr1=0.0002)
            vb=_max_cos(Xb)
            if vb < elite[-1][0]+1e-7: push(vb,Xb)
            if vb < best_val: best_val=vb; best=Xb.copy()
        elite.sort(key=lambda e:e[0])
        if elite[0][0] < best_val:
            best_val=elite[0][0]; best=elite[0][1].copy()
    remain = max(1.0, (t0+BUDGET)-time.time()-1.0)
    sec_lp = min(remain*0.7, 30.0)
    best = _lp_polish(best, seconds=max(2.0,sec_lp))
    best_val = _max_cos(best)
    if time.time() < t0+BUDGET-1.0:
        best2 = _hard_polish(best, steps=200, lr0=0.0015, lr1=0.0001, tol0=2e-4, tol1=1e-6)
        v2 = _max_cos(best2)
        if v2 < best_val:
            best, best_val = best2, v2
            best = _lp_polish(best, seconds=max(1.0,(t0+BUDGET)-time.time()-0.5))
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
