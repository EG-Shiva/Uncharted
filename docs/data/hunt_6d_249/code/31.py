import time
import itertools
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    RESERVE_FINAL = 28.0
    rng = np.random.default_rng(1)
    np.random.seed(1)
    n, d = 249, 6

    # ---------- load incumbents ----------
    cands249 = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    cands249.append(a)
        except Exception:
            pass
    if cands249:
        vals0 = [_max_cos(a) for a in cands249]
        bestX = cands249[int(np.argmin(vals0))].copy()
        bestVal = float(np.min(vals0))
    else:
        bestX = _norm(rng.standard_normal(size=(n, d)))
        bestVal = _max_cos(bestX)

    # ---------- generic soft-max anneal ----------
    def anneal(Xinit, taus, iters, creg=0.08):
        nn = Xinit.shape[0]
        I0, J0 = np.triu_indices(nn, 1)
        xf = Xinit.ravel().copy()
        for tau, mi in zip(taus, iters):
            if time.time() > T_END - RESERVE_FINAL:
                break
            tau = float(tau)
            from scipy.optimize import minimize
            def fg(xf_, tau=tau):
                X = xf_.reshape(nn, d)
                r = np.sqrt(np.sum(X*X, axis=1))
                r = np.maximum(r, 1e-12)
                Xn = X / r[:, None]
                G = Xn @ Xn.T
                g = G[I0, J0]
                gm = np.max(g)
                e = np.exp((g-gm)/tau)
                S = np.sum(e)
                Lsoft = gm + tau*np.log(S)
                w = e / S
                W = np.zeros((nn, nn))
                W[I0, J0] = w
                W = W + W.T
                A = W @ Xn
                proj = np.sum(A*Xn, axis=1)
                G0 = (A - proj[:, None]*Xn) / r[:, None]
                diff = r - 1.0
                L = Lsoft + creg*np.sum(diff*diff)
                Gr = G0 + (2.0*creg*diff)[:, None]*(X/r[:, None])
                return float(L), Gr.ravel()
            try:
                res = minimize(fg, xf, jac=True, method='L-BFGS-B',
                               options={'maxiter': int(mi), 'maxfun': int(mi*4),
                                        'ftol': 1e-14, 'gtol': 1e-7, 'maxcor': 8})
                xf = res.x
            except Exception:
                break
            Xm = xf.reshape(nn, d)
            rn = np.maximum(np.linalg.norm(Xm, axis=1, keepdims=True), 1e-12)
            xf = (Xm/rn).ravel()
        Xo = xf.reshape(nn, d)
        Xo = _norm(Xo)
        return Xo

    def short_polish(X):
        return anneal(X, [0.02, 0.008], [45, 55], creg=0.08)

    def medium_polish(X):
        return anneal(X, [0.01, 0.003, 0.001], [70, 70, 70], creg=0.08)

    def sharp_polish(X):
        return anneal(X, [0.005, 0.0015, 0.0005, 0.0002], [140, 140, 130, 110], creg=0.05)

    # ---------- algebraic constructions ----------
    # D6 roots (60)
    D6 = []
    for i in range(6):
        for j in range(i+1, 6):
            for si in (-1.0, 1.0):
                for sj in (-1.0, 1.0):
                    v = np.zeros(6)
                    v[i] = si; v[j] = sj
                    v = v/np.linalg.norm(v)
                    D6.append(v)
    D6 = np.stack(D6, axis=0)  # 60x6

    # E8 roots (240 x 8)
    def build_E8():
        pts = []
        for i in range(8):
            for j in range(i+1, 8):
                for si in (-1.0, 1.0):
                    for sj in (-1.0, 1.0):
                        v = np.zeros(8); v[i]=si; v[j]=sj
                        pts.append(v)
        for mask in range(256):
            bits = [(mask>>k)&1 for k in range(8)]
            # even number of minus: sum bits even -> product +
            if sum(bits)%2==1:
                continue
            v = np.where(np.array(bits)==1, -0.5, 0.5)
            pts.append(v.astype(float))
        E = np.stack(pts, axis=0)
        return E  # should be 240
    try:
        E8 = build_E8()
    except Exception:
        E8 = None

    def random_rot6(r):
        A = r.standard_normal(size=(6,6))
        Q, R = np.linalg.qr(A)
        # fix sign for determinism
        s = np.sign(np.diag(R))
        s[s==0]=1
        Q = Q*s[None,:]
        return Q

    def greedy_extend(Xbase, need, pool):
        # pool (C,6) normalized; greedily pick need pts minimizing max to base+picked
        Xb = Xbase
        C = pool.shape[0]
        D = pool @ Xb.T
        cur = D.max(axis=1) if Xb.shape[0]>0 else np.full(C, -2.0)
        used = np.zeros(C, dtype=bool)
        picked = []
        for _ in range(need):
            cur[used] = np.inf
            bi = int(np.argmin(cur))
            used[bi]=True
            pt = pool[bi].copy()
            picked.append(pt)
            dd = pool @ pt
            np.maximum(cur, dd, out=cur)
            cur[used]=np.inf
        if picked:
            return np.stack(picked, axis=0)
        return np.zeros((0,6))

    def hole_best(Xkeep, n_cand=5000):
        Cc = _norm(rng.standard_normal(size=(n_cand, d)))
        D = Cc @ Xkeep.T
        cm = D.max(axis=1)
        bi = int(np.argmin(cm))
        return Cc[bi].copy()

    def refine_hole(pt0, Xkeep):
        # single-vector soft-max L-BFGS, 6 vars
        from scipy.optimize import minimize
        y = pt0.copy()
        # parametrize unconstrained then normalize inside
        yf = y.copy()
        for tau in (0.01, 0.003):
            def fg(yf_, tau=tau):
                r = max(np.linalg.norm(yf_),1e-12)
                yn = yf_/r
                dv = Xkeep @ yn  # m,
                gm = np.max(dv)
                e = np.exp((dv-gm)/tau)
                S = np.sum(e)
                L = gm + tau*np.log(S)
                w = e/S
                a = w @ Xkeep  # 6,
                proj = float(a @ yn)
                g0 = (a - proj*yn)/r
                return float(L), g0
            try:
                res = minimize(fg, yf, jac=True, method='L-BFGS-B',
                               options={'maxiter':60,'ftol':1e-14,'gtol':1e-8})
                yf = res.x
            except Exception:
                break
        yn = yf/max(np.linalg.norm(yf),1e-12)
        return yn

    def d6_seed(extra_need, seed_offset):
        r = np.random.default_rng(100+seed_offset)
        Q = random_rot6(r)
        drot = (D6 @ Q.T)
        drot = _norm(drot + 0.02*r.standard_normal(size=drot.shape))
        pool = _norm(r.standard_normal(size=(6000,6)))
        ext = greedy_extend(drot, extra_need, pool)
        X = np.vstack([drot, ext])
        # random global rotation + perturb
        X = X @ random_rot6(r).T
        X = _norm(X + 0.01*r.standard_normal(size=X.shape))
        return X

    def e8_seed(seed_offset):
        r = np.random.default_rng(200+seed_offset)
        if E8 is None:
            return _norm(r.standard_normal(size=(n,d)))
        A = r.standard_normal(size=(8,6))
        Q, _ = np.linalg.qr(A)  # 8x6 orthonormal cols
        Y = E8 @ Q  # 240x6
        # normalize, drop degenerate
        nr = np.linalg.norm(Y, axis=1)
        keep = nr > 1e-6
        Y = Y[keep]/nr[keep,None]
        # Y should be 240; if more/less adjust
        if Y.shape[0] >= 240:
            Y = Y[:240]
        # need to reach 249
        need = n - Y.shape[0]
        if need>0:
            pool = _norm(r.standard_normal(size=(6000,6)))
            ext = greedy_extend(Y, need, pool)
            Y = np.vstack([Y, ext])
        elif need<0:
            Y = Y[:n]
        Q6 = random_rot6(r)
        Y = Y @ Q6.T
        Y = _norm(Y + 0.015*r.standard_normal(size=Y.shape))
        return Y

    def pattern_seed(abc, seed_offset):
        r = np.random.default_rng(300+seed_offset)
        a,b,c = abc
        v = np.array([a,b,c],float); v/=np.linalg.norm(v)
        a,b,c = v
        perms = list(itertools.permutations([a,b,c]))
        combos = list(itertools.combinations(range(6),3))
        pool=[]
        for comb in combos:
            for perm in perms:
                for sm in range(8):
                    s = [1.0 if (sm>>k)&1 else -1.0 for k in range(3)]
                    vec=np.zeros(6)
                    for k,pos in enumerate(comb):
                        vec[pos]=perm[k]*s[k]
                    pool.append(vec)
        pool=np.stack(pool,axis=0)  # 960 x6, already unit
        # greedy pick 249 from pool
        # start from random element
        start = r.integers(0, pool.shape[0])
        chosen=[start]
        used=np.zeros(pool.shape[0],dtype=bool); used[start]=True
        Gp = pool @ pool.T
        cur = Gp[:,start].copy()
        cur[used]=-2
        for _ in range(n-1):
            # pick pool element minimizing max to chosen (cur)
            # cur holds max dot to chosen set
            bi=int(np.argmax(cur<-1.9)) if False else int(np.argmin(cur))
            # argmin over unused
            masked = np.where(used, np.inf, cur)
            bi=int(np.argmin(masked))
            used[bi]=True
            chosen.append(bi)
            np.maximum(cur, Gp[:,bi], out=cur)
            cur[used]=np.inf if False else cur
        X = pool[np.array(chosen)]
        Q6=random_rot6(r)
        X = X @ Q6.T
        X=_norm(X+0.01*r.standard_normal(size=X.shape))
        return X

    # ---------- build seed pool at N=249 ----------
    seeds = []
    # incumbents perturbed (ensure competitiveness)
    for sig, sd in [(0.03, 901),(0.08,902)]:
        if cands249:
            rr = np.random.default_rng(sd)
            Xp = _norm(bestX + sig*rr.standard_normal(size=(n,d)))
            seeds.append(('pert', Xp))
    # D6
    try:
        seeds.append(('d6a', d6_seed(n-60, 0)))
        seeds.append(('d6b', d6_seed(n-60, 1)))
    except Exception:
        pass
    # E8
    for k in range(3):
        try:
            seeds.append((f'e8{k}', e8_seed(k)))
        except Exception:
            pass
    # patterns
    for k, abc in enumerate([(0.9,0.4,0.2),(0.7,0.6,0.35),(0.6,0.55,0.55)]):
        try:
            seeds.append((f'pat{k}', pattern_seed(abc, k)))
        except Exception:
            pass
    # random
    for k in range(2):
        rr=np.random.default_rng(500+k)
        seeds.append((f'rnd{k}', _norm(rr.standard_normal(size=(n,d)))))

    # ---------- race with short polish ----------
    raced=[]
    for name, X in seeds:
        if time.time() > T_END-RESERVE_FINAL-70:
            break
        try:
            Y = short_polish(X)
            v = _max_cos(Y)
            raced.append((v, name, Y))
            if v < bestVal:
                bestVal=v; bestX=Y.copy()
        except Exception:
            continue
    raced.sort(key=lambda t: t[0])

    def jaccard_dist(Xa, Xb, K=1200):
        na=Xa.shape[0]; 
        I0,J0=np.triu_indices(na,1)
        def topk(X):
            G=X@X.T
            g=G[I0,J0]
            idx=np.argpartition(g,-K)[-K:]
            codes=I0[idx].astype(np.int64)*na+J0[idx].astype(np.int64)
            return set(codes.tolist())
        try:
            A=topk(Xa); B=topk(Xb)
            inter=len(A&B); union=len(A|B)
            return 1.0-inter/max(union,1)
        except Exception:
            return 1.0

    # ---------- cardinality homotopy: grow small, shrink big ----------
    def build_small(Ns, kind, sd):
        r=np.random.default_rng(sd)
        if kind=='e8' and E8 is not None:
            A=r.standard_normal(size=(8,6)); Q,_=np.linalg.qr(A)
            Y=E8@Q; nr=np.linalg.norm(Y,axis=1); keep=nr>1e-6
            Y=Y[keep]/nr[keep,None]
            if Y.shape[0]>=Ns:
                # subsample greedily to Ns
                idx=[r.integers(0,Y.shape[0])]
                used=np.zeros(Y.shape[0],dtype=bool); used[idx[0]]=True
                cur=(Y@Y[idx[0]]).copy(); cur[used]=np.inf
                for _ in range(Ns-1):
                    masked=np.where(used,np.inf,cur)
                    bi=int(np.argmin(masked)); used[bi]=True; idx.append(bi)
                    np.maximum(cur, Y@Y[bi], out=cur)
                Y=Y[np.array(idx)]
            else:
                need=Ns-Y.shape[0]
                pool=_norm(r.standard_normal(size=(6000,6)))
                ext=greedy_extend(Y,need,pool); Y=np.vstack([Y,ext])
            Y=Y@random_rot6(r).T; Y=_norm(Y+0.015*r.standard_normal(size=Y.shape))
            return Y
        elif kind=='d6':
            Q=random_rot6(r); drot=_norm(D6@Q.T)
            if Ns<=60:
                # subsample D6
                sel=r.choice(60,size=Ns,replace=False); return drot[sel]
            else:
                pool=_norm(r.standard_normal(size=(6000,6)))
                ext=greedy_extend(drot,Ns-60,pool)
                Y=np.vstack([drot,ext]); return _norm(Y)
        else:
            return _norm(r.standard_normal(size=(Ns,d)))

    def grow_to_249(Xs):
        X = short_polish(Xs) if Xs.shape[0]>30 else Xs
        while X.shape[0] < n:
            if time.time() > T_END-RESERVE_FINAL-25:
                break
            pt = hole_best(X, n_cand=4000)
            try:
                pt = refine_hole(pt, X)
            except Exception:
                pass
            X = np.vstack([X, pt[None,:]])
            # short re-jam
            X = anneal(X, [0.008], [45], creg=0.08)
        return X

    def shrink_to_249(Xb):
        X = short_polish(Xb)
        while X.shape[0] > n:
            if time.time() > T_END-RESERVE_FINAL-25:
                break
            m = X.shape[0]
            G = X@X.T; np.fill_diagonal(G,-2.0)
            # find removal minimizing max of remainder via top pairs
            # brute per-i max excluding i
            # use top ~20 pairs to decide quickly, fallback brute
            # brute: (m up to 260) loop
            best_i=0; best_m=np.inf
            # vectorized trick: row max etc not sufficient; do loop with numpy max
            for i in range(m):
                # max over pairs not involving i = max of G masked
                # compute quickly: max of full matrix, if arg pair doesn't involve i then same
                # for simplicity compute per i max of G without row/col i via precomputed?
                Gi = G.copy()  # small, okay for few iterations
                Gi[i,:]=-2; Gi[:,i]=-2
                mv=float(np.max(Gi))
                if mv < best_m:
                    best_m=mv; best_i=i
                    if best_m < _max_cos(X)-1e-9:
                        pass
            X = np.delete(X, best_i, axis=0)
            X = anneal(X, [0.008], [45], creg=0.08)
        return X

    # growth paths for easier sizes 235-248
    growth_Ns = [238, 245]
    for gi, Ns in enumerate(growth_Ns):
        if time.time() > T_END-RESERVE_FINAL-40:
            break
        try:
            kind = 'e8' if gi==0 else 'd6'
            Xs = build_small(Ns, kind, 700+gi)
            Y = grow_to_249(Xs)
            if Y.shape[0]!=n:
                continue
            Ym = medium_polish(Y) if time.time()<T_END-RESERVE_FINAL-20 else Y
            v=_max_cos(Ym)
            raced.append((v, f'grow{Ns}', Ym))
            if v<bestVal:
                bestVal=v; bestX=Ym.copy()
        except Exception:
            continue

    # shrink paths overfull 252-260
    for gi, Nb in enumerate([256, 253]):
        if time.time() > T_END-RESERVE_FINAL-30:
            break
        try:
            kind = 'd6' if gi==0 else 'rnd'
            Xb = build_small(Nb, kind, 800+gi)
            Y = shrink_to_249(Xb)
            if Y.shape[0]!=n:
                continue
            Ym = medium_polish(Y) if time.time()<T_END-RESERVE_FINAL-20 else Y
            v=_max_cos(Ym)
            raced.append((v, f'shrink{Nb}', Ym))
            if v<bestVal:
                bestVal=v; bestX=Ym.copy()
        except Exception:
            continue

    raced.sort(key=lambda t: t[0])
    # select finalists by score + dissimilarity
    finalists=[]
    if raced:
        finalists.append(raced[0])
        for v,nm,Y in raced[1:]:
            if len(finalists)>=4:
                break
            ok=True
            for _,_,Z in finalists:
                if jaccard_dist(Y,Z)<0.35:
                    ok=False; break
            if ok or v < finalists[0][0]+0.002:
                if ok:
                    finalists.append((v,nm,Y))
        # also ensure bestX included
        if bestX is not None and all(not np.allclose(bestX,F[2]) for F in finalists):
            finalists.append((bestVal,'best',bestX.copy()))
    else:
        finalists=[(bestVal,'best',bestX.copy())]

    # medium polish finalists, pick winner
    winner = finalists[0][2].copy(); wval=finalists[0][0]
    for v,nm,Y in finalists:
        if time.time() > T_END-RESERVE_FINAL:
            break
        try:
            Ym = medium_polish(Y.copy())
            vm=_max_cos(Ym)
            if vm<bestVal:
                bestVal=vm; bestX=Ym.copy()
            if vm<wval:
                wval=vm; winner=Ym.copy()
        except Exception:
            continue
    # ensure winner is global best
    if bestVal < wval:
        winner=bestX.copy(); wval=bestVal

    # ---------- final full sharp jam + LP equalization ----------
    if time.time() < T_END-20:
        try:
            winner = sharp_polish(winner)
            wval=_max_cos(winner)
            if wval<bestVal:
                bestVal=wval; bestX=winner.copy()
        except Exception:
            pass

    # trust-region LP equalization attempts
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
        trust=0.01
        for lp_it in range(6):
            if time.time() > T_END-6:
                break
            Xc=winner.copy()
            G=Xc@Xc.T; np.fill_diagonal(G,-2.0)
            m=float(np.max(G))
            I0,J0=np.triu_indices(n,1)
            g=G[I0,J0]
            K=min(1500, g.size)
            idx=np.argpartition(g,-K)[-K:]
            # keep only near-max for LP
            mask_active = g[idx] > (m-0.02)
            sel = idx[mask_active]
            if sel.size<50:
                sel=idx
            rows=sel.size
            nvar=n*d+1
            # Build sparse A_ub rows: x_j·dx_i + x_i·dx_j - t <= -dot
            # Use dense for simplicity (rows~1500, vars~1495 => 2.2M entries, okay)
            A_ub=np.zeros((rows, nvar))
            b_ub=np.zeros(rows)
            for r_,p in enumerate(sel):
                i=int(I0[p]); j=int(J0[p])
                A_ub[r_, i*d:(i+1)*d]=Xc[j]
                A_ub[r_, j*d:(j+1)*d]=Xc[i]
                A_ub[r_, nvar-1]=-1.0
                b_ub[r_]=-float(G[i,j])
            c=np.zeros(nvar); c[-1]=1.0
            A_eq=np.zeros((n, nvar))
            for i in range(n):
                A_eq[i, i*d:(i+1)*d]=Xc[i]
            b_eq=np.zeros(n)
            bounds=[(-trust,trust)]*(n*d)+[(-2.0, m)]
            res=linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                        bounds=bounds, method='highs',
                        options={'time_limit':4.0})
            if not res.success:
                trust*=0.5
                if trust<1e-5:
                    break
                continue
            dx=res.x[:n*d].reshape(n,d); tnew=float(res.x[-1])
            Xn=_norm(Xc+dx)
            vn=_max_cos(Xn)
            if vn < wval - 1e-12:
                winner=Xn; wval=vn
                if wval<bestVal:
                    bestVal=wval; bestX=winner.copy()
                trust=min(trust*1.3, 0.02)
            else:
                trust*=0.4
                if trust<2e-4:
                    break
    except Exception:
        pass

    # cheap subgradient squeeze
    try:
        deadline=T_END-1.0
        cur2=winner.copy(); cur2v=wval
        step=2e-3
        while time.time()<deadline and step>1e-13:
            G=cur2@cur2.T; np.fill_diagonal(G,-2.0)
            mm=float(np.max(G))
            window=max(10.0*step, 2e-4)
            mask=G>(mm-window)
            F=mask.astype(float)@cur2
            dot=np.sum(F*cur2,axis=1)
            Ft=F-dot[:,None]*cur2
            trial=_norm(cur2-step*Ft)
            vv=_max_cos(trial)
            if vv<cur2v:
                cur2,cur2v=trial,vv
                step=min(step*1.25,0.01)
            else:
                step*=0.5
                if step<1e-7:
                    break
        if cur2v<wval:
            winner, wval=cur2,cur2v
        if wval<bestVal:
            bestVal=wval; bestX=winner.copy()
    except Exception:
        pass

    bestX=_norm(bestX)
    return bestX
