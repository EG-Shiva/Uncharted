import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _smooth_refine(x, steps, T0, T1, lr0, lr1, mom=0.85):
    n = x.shape[0]
    v = np.zeros_like(x)
    for k in range(steps):
        f = k / max(1, steps - 1)
        T = T0 * (1 - f) + T1 * f
        lr = lr0 * (1 - f) + lr1 * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp((G - m) / max(T, 1e-12))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        s = float(np.sum(W)) / max(n,1) + 1e-12
        grad = grad / s
        v = mom * v + (1.0 - mom) * grad
        x = x - lr * v
        x = _norm(x)
    return x

def _lp_polish(x, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline:
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu0]
        m = float(c_all.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 4000:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:4000]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I = iu0[0][sel]; J = iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)],1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
        try:
            A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))), shape=(n,nv)).tocsr()
            obj = np.zeros(nv); obj[-1]=1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n,d)
        y = _norm(y)
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius*1.5, 0.05)
        else:
            radius *= 0.5
        if deadline - time.time() < 0.25:
            break
    return x

def _random_orth(d, rng):
    A = rng.standard_normal((d,d))
    Q, R = np.linalg.qr(A)
    s = np.sign(np.diag(R))
    s[s==0]=1.0
    Q = Q * s[None,:]
    return Q

def _find_hole(Xk, rng, n_samples=10000):
    d = Xk.shape[1]
    Z = rng.standard_normal((n_samples,d))
    Z = _norm(Z)
    D = Z @ Xk.T
    worst = D.max(axis=1)
    idx = int(np.argmin(worst))
    return Z[idx].copy(), float(worst[idx])

def _refine_hole(z, Xk, steps=80, T=0.008, lr=0.03):
    for _ in range(steps):
        dots = Xk @ z
        m = float(dots.max())
        W = np.exp((dots-m)/T)
        W = W/(W.sum()+1e-12)
        g = W @ Xk
        g = g - float(np.dot(g,z))*z
        z = z - lr*g
        nz = np.linalg.norm(z)
        if nz < 1e-12:
            break
        z = z/nz
    return z

def _degrees(X, mv, tol=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    C = G >= (mv - tol)
    return C.sum(axis=1), G

def _edge_set(X, mv, tol=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    iu = np.triu_indices(X.shape[0],1)
    sel = G[iu] >= (mv-tol)
    return set(zip(iu[0][sel].tolist(), iu[1][sel].tolist()))

def solve():
    t0 = time.time()
    BUDGET = 172.0
    n, d = 131, 9
    rng = np.random.default_rng(0)
    rngD = np.random.default_rng(12345)
    # incumbent
    best_x = None; best_s = 1e9
    for fn in ("best.npy","record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape==(n,d):
                a=_norm(a); s=_max_cos(a)
                if s<best_s: best_s=s; best_x=a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x=_norm(rng.standard_normal((n,d))); best_s=_max_cos(best_x)
    incumb_x = best_x.copy(); incumb_s = best_s
    incumb_edges = _edge_set(incumb_x, incumb_s, tol=1e-7)

    # ---- build ternary pool P (19682 x 9) ----
    N3 = 3**9
    pows = 3**np.arange(9)
    idxa = np.arange(N3)
    P = ((idxa[:,None]//pows)%3)-1  # includes zero
    P = P.astype(float)
    # remove zero row
    masknz = np.any(P!=0, axis=1)
    P = P[masknz]
    Pn = np.linalg.norm(P,axis=1,keepdims=True)
    P = P/np.maximum(Pn,1e-12)
    M = P.shape[0]
    # subsampled 5-ary pool Q (for one seed): 15000 random
    Qr = rngD.integers(-2,3,size=(15000,9)).astype(float)
    qn = np.linalg.norm(Qr,axis=1)
    Qr = Qr[qn>1e-9]
    Qr = Qr/np.linalg.norm(Qr,axis=1,keepdims=True)
    # dedup Qr roughly against P: keep all, duplicates harmless for greedy

    def greedy_select(pool, start_idx, rngl):
        mloc = pool.shape[0]
        cur = pool @ pool[start_idx]
        cur[start_idx]=np.inf
        sel=[int(start_idx)]
        noise = rngl.standard_normal(n)*1e-12  # tie jitter per step not per element; use per step element noise below
        for k in range(1,n):
            # tiny random tie break
            jit = rngl.standard_normal(mloc)*1e-12
            cand = cur + jit
            # ensure selected stay inf (jit could make inf still inf)
            b = int(np.argmin(cand))
            sel.append(b)
            cur = np.maximum(cur, pool @ pool[b])
            cur[b]=np.inf
            # keep all previously selected inf
            # (maximum with inf stays inf)
        return sel

    def swap_improve(pool, sel_idx, max_iter=12):
        sel_idx = list(map(int, sel_idx))
        S = pool[np.array(sel_idx)]
        for it in range(max_iter):
            G = S @ S.T; np.fill_diagonal(G,-2.0)
            m = float(G.max())
            # m_excl per i
            m_excl = np.empty(n)
            for i in range(n):
                # max without i
                # fast: mask row/col
                # use submatrix max via copy of small matrix
                # 131x131 copy 131 times is fine
                G2 = G.copy()
                G2[i,:]=-2.0; G2[:,i]=-2.0
                m_excl[i]=float(G2.max())
            D = pool @ S.T  # Mloc x n
            max1 = D.max(axis=1)
            arg1 = D.argmax(axis=1)
            # second max
            Dtmp = D.copy()
            Dtmp[np.arange(D.shape[0]), arg1]=-2.0
            max2 = Dtmp.max(axis=1)
            best_gain = 0.0; best_i=-1; best_p=-1
            # for each i find best p
            for i in range(n):
                di = np.where(arg1!=i, max1, max2)
                # best p for this i
                pstar = int(np.argmin(di))
                newm = m_excl[i] if m_excl[i]>di[pstar] else float(di[pstar])
                gain = m - newm
                if gain > best_gain+1e-13:
                    best_gain=gain; best_i=i; best_p=pstar
            if best_i>=0 and best_gain>1e-12:
                sel_idx[best_i]=int(best_p)
                S = pool[np.array(sel_idx)]
                continue
            # no strict gain: try count reduction of near-max
            tol = 0.005
            thr = m - tol
            cnt = int((G>thr).sum()//2)
            # cnt excl per i (approx)
            improved=False
            # try limited random trials
            for _ in range(600):
                i = int(rngD.integers(0,n))
                p = int(rngD.integers(0,D.shape[0]))
                di_val = float(D[p,arg1[p]] if arg1[p]!=i else max2[p] if False else (max1[p] if arg1[p]!=i else max2[p]))
                # compute di_val correctly:
                # already: di = max1 or max2
                # new max
                newm = m_excl[i] if m_excl[i]>di_val else di_val
                if newm > m+1e-12:
                    continue
                # count new: cnt_excl + hits of p
                # cnt_excl
                # compute quickly: pairs not involving i above thr
                # use G copy count
                # hits of p: number of j!=i with D[p,j]>=thr
                hits = int(np.sum(D[p,:]>=thr) - (1 if D[p,i]>=thr else 0))
                # cnt_excl: total minus pairs involving i
                rowcnt = int(np.sum(G[i,:]>=thr) + np.sum(G[:,i]>=thr))  # double counts? G symmetric, row includes all
                # G[i,:] includes n entries (diag -2 not counted). pairs involving i = sum(G[i,:]>=thr)
                rowcnt2 = int(np.sum(G[i,:]>=thr))
                cnt_excl = cnt - rowcnt2
                newcnt = cnt_excl + hits
                if newcnt < cnt:
                    sel_idx[i]=p; S=pool[np.array(sel_idx)]
                    improved=True
                    break
            if improved:
                continue
            break
        return sel_idx, S

    # ---- generate discrete seeds ----
    discrete_cands=[]
    t_phaseC_end = t0 + 55.0
    # create ~5 seeds
    nseeds=5
    for s in range(nseeds):
        if time.time()>t_phaseC_end-30:
            break
        pool = P if s<4 else Qr
        if s==3:
            # mixed: concatenate P and Qr subsample
            pool = np.vstack([P, Qr[:5000]])
        start = int(rngD.integers(0, pool.shape[0]))
        # bias some starts to D9-type / coordinate for coverage
        if s==1:
            # coordinate vector e0
            diffs = np.sum((pool-np.eye(9)[0])**2,axis=1)
            start=int(np.argmin(diffs))
        if s==2:
            # D9-type (1,1,0...)/sqrt2
            tgt=np.zeros(9); tgt[0]=1/np.sqrt(2); tgt[1]=1/np.sqrt(2)
            start=int(np.argmin(np.sum((pool-tgt)**2,axis=1)))
        sel,_ = greedy_select(pool, start, rngD) if False else (None,None)
        # inline greedy to avoid func overhead with correct pool
        mloc=pool.shape[0]
        cur = pool @ pool[start]; cur[start]=np.inf
        sel=[start]
        for k in range(1,n):
            jit=rngD.standard_normal(mloc)*1e-12
            # keep infs inf: argmin of cur+jit where cur inf -> inf+jit still inf large
            b=int(np.argmin(cur+jit))
            sel.append(b)
            cur=np.maximum(cur, pool @ pool[b])
            cur[b]=np.inf
        sel_idx,S = swap_improve(pool, sel, max_iter=10)
        discrete_cands.append((pool, sel_idx, S))

    # ---- polish discrete seeds (40-60s total) ----
    polished=[]
    for (pool, sel_idx, S) in discrete_cands:
        if time.time()>t_phaseC_end:
            break
        Q = _random_orth(d, rngD)
        Y = S @ Q.T
        Y = Y + 0.002*rngD.standard_normal(Y.shape)
        Y=_norm(Y)
        # smooth repulsion ~500 steps
        try:
            Y=_smooth_refine(Y, 450, 0.02, 0.003, 0.014, 0.0025)
            Y=_smooth_refine(Y, 200, 0.005, 0.0012, 0.004, 0.001)
        except Exception:
            pass
        # short LP
        try:
            dl=min(time.time()+4.0, t_phaseC_end+2.0)
            Y=_lp_polish(Y, dl)
        except Exception:
            pass
        try:
            s=_max_cos(Y)
        except Exception:
            continue
        deg,_=_degrees(Y,s,tol=1e-7)
        nz=int(np.sum(deg==0))
        # edge overlap proxy: compare degree histograms? use edge overlap if same n but labels differ -> use sorted degree distance + gram sorted
        # compute overlap as fraction of shared degree values (rough). For strictness compute gram value histogram overlap.
        # We retain if no rattlers (nz==0) even if worse, per brief.
        polished.append((s,Y,nz))
        if s < best_s-1e-12:
            best_s=s; best_x=Y.copy()

    # choose diverse retained: those with nz==0 and s < incumb_s+0.02
    diverse_best=None; diverse_s=1e9
    for (s,Y,nz) in polished:
        if nz==0 and s<diverse_s and s<incumb_s+0.02:
            # edge overlap check: compute degree L1 vs incumb
            try:
                di,_=_degrees(incumb_x,incumb_s,tol=1e-7)
                dj,_=_degrees(Y,s,tol=1e-7)
                # overlap proxy: correlation of sorted degrees
                si=np.sort(di); sj=np.sort(dj)
                # if very similar, skip? require difference
                diff=float(np.mean(np.abs(si-sj)))
                # edge overlap below 70% proxy: diff>1.0 means different
                # retain regardless for hopping start, but record
                diverse_s=s; diverse_best=Y.copy()
            except Exception:
                diverse_s=s; diverse_best=Y.copy()

    # ---- elitist basin-hopping from best for remaining time ----
    HOP_END = t0 + 132.0
    def probs_from(X, mv):
        deg,_=_degrees(X,mv,tol=1e-7)
        w=(deg+1.0)**1.5
        return w/w.sum(), deg
    try:
        probs,degs=probs_from(best_x,best_s)
    except Exception:
        probs=np.full(n,1.0/n); degs=np.zeros(n)
    trial=0
    # if diverse basin exists and differs, alternate starts
    hop_starts=[best_x.copy()]
    if diverse_best is not None and diverse_s<9:
        # only add if truly different basin (check max diff)
        try:
            if _max_cos(diverse_best)<best_s+0.01:
                hop_starts.append(diverse_best.copy())
        except Exception:
            pass
    cur_chain = best_x.copy()
    while time.time()<HOP_END:
        trial+=1
        if trial%10==1:
            try:
                probs,degs=probs_from(best_x,best_s)
            except Exception:
                pass
            # occasionally restart chain from diverse basin to explore
            if len(hop_starts)>1 and trial%20==0:
                cur_chain=hop_starts[1].copy()
            else:
                cur_chain=best_x.copy()
        Y=cur_chain.copy()
        mode=trial%8
        try:
            if mode in (0,1):
                deg_thr,_=_degrees(Y,best_s,tol=5e-6)
                order=np.argsort(deg_thr)
                k=1 if mode==0 else 2
                victims=order[:k]
                for vi in victims:
                    msk=np.ones(n,dtype=bool); msk[vi]=False
                    Xk=Y[msk]
                    z,_=_find_hole(Xk,rng,n_samples=8000)
                    z=_refine_hole(z,Xk,steps=60,T=0.008,lr=0.03)
                    Y[vi]=z
                Y+=0.004*rng.standard_normal(Y.shape); Y=_norm(Y)
                Y=_smooth_refine(Y,300,0.016,0.003,0.016,0.003)
                Y=_smooth_refine(Y,200,0.005,0.0015,0.005,0.0012)
            elif mode in (2,3):
                k=1 if mode==2 else 2
                sig=0.18 if mode==2 else 0.28
                idx=rng.choice(n,k,replace=False,p=probs)
                Y[idx]+=sig*rng.standard_normal((k,d)); Y=_norm(Y)
                Y=_smooth_refine(Y,350,0.018,0.003,0.015,0.0025)
                Y=_smooth_refine(Y,150,0.004,0.0012,0.004,0.001)
            elif mode==4:
                k,sig=3,0.16
                idx=rng.choice(n,k,replace=False,p=probs)
                Y[idx]+=sig*rng.standard_normal((k,d)); Y=_norm(Y)
                Y=_smooth_refine(Y,320,0.02,0.004,0.014,0.0025)
            elif mode==5:
                Y+=0.010*rng.standard_normal(Y.shape)
                idx=rng.choice(n,1,replace=False,p=probs)
                Y[idx]+=0.14*rng.standard_normal((1,d)); Y=_norm(Y)
                Y=_smooth_refine(Y,350,0.016,0.0025,0.014,0.002)
            elif mode==6:
                Y+=0.025*rng.standard_normal(Y.shape); Y=_norm(Y)
                Y=_smooth_refine(Y,300,0.022,0.005,0.015,0.003)
                Y=_smooth_refine(Y,200,0.005,0.0015,0.005,0.0012)
            else:
                Y+=0.012*rng.standard_normal(Y.shape)
                idx=int(rng.choice(n,1,replace=False,p=probs)[0])
                rp=rng.standard_normal(d); rp/=np.linalg.norm(rp)
                ang=float(rng.uniform(0.15,0.5))
                v=Y[idx].copy()
                vp=rp-float(rp@v)*v
                nv_=np.linalg.norm(vp)
                if nv_>1e-9:
                    vp/=nv_
                    Y[idx]=np.cos(ang)*v+np.sin(ang)*vp
                Y=_norm(Y)
                Y=_smooth_refine(Y,350,0.018,0.003,0.013,0.002)
        except Exception:
            continue
        try:
            s_smooth=_max_cos(Y)
        except Exception:
            continue
        if s_smooth<best_s+0.002:
            dl=min(time.time()+1.0, HOP_END+4.0)
            try:
                Y2=_lp_polish(Y,dl)
                s2=_max_cos(Y2)
            except Exception:
                s2=s_smooth+1.0; Y2=Y
            if s2<best_s-1e-12:
                best_x=Y2.copy(); best_s=s2
                cur_chain=Y2.copy()
                try:
                    probs,degs=probs_from(best_x,best_s)
                except Exception:
                    pass
            elif s_smooth<best_s-1e-12:
                best_x=Y.copy(); best_s=s_smooth
                cur_chain=Y.copy()
                try:
                    probs,degs=probs_from(best_x,best_s)
                except Exception:
                    pass
        if time.time()>t0+BUDGET-35:
            break
    try:
        deadline=t0+BUDGET-1.0
        best_x=_lp_polish(best_x,deadline)
    except Exception:
        pass
    best_x=_norm(best_x)
    best_x=np.nan_to_num(best_x,nan=0.0,posinf=0.0,neginf=0.0)
    nr=np.linalg.norm(best_x,axis=1,keepdims=True); nr=np.maximum(nr,1e-12)
    best_x=best_x/nr
    return best_x
