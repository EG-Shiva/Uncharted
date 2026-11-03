import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-30)

def _anneal_exp(X, iters, a0, a1, s0, s1):
    n, d = X.shape
    for it in range(iters):
        frac = it / max(1, iters - 1)
        alpha = a0 * (a1 / a0) ** frac if iters > 1 else a0
        step = s0 * (s1 / s0) ** frac if iters > 1 else s0
        G = X @ X.T
        np.fill_diagonal(G, -np.inf)
        gmax = float(G.max())
        W = np.exp(alpha * (G - gmax))
        W[W < 1e-12] = 0.0
        np.fill_diagonal(W, 0.0)
        F = W @ X
        dot = np.sum(F * X, axis=1, keepdims=True)
        Ft = F - dot * X
        nrm = float(np.linalg.norm(Ft, axis=1).mean()) + 1e-12
        X = X - (step / nrm) * Ft
        X = _norm(X)
    return X

def _anneal_pow(X, iters, p0, p1, s0, s1):
    n, d = X.shape
    for it in range(iters):
        frac = it / max(1, iters - 1)
        p = p0 + (p1 - p0) * frac
        step = s0 * (s1 / s0) ** frac if iters > 1 else s0
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        D = 1.0 - G
        D[D < 1e-9] = 1e-9
        W = D ** (-p)
        np.fill_diagonal(W, 0.0)
        W[~np.isfinite(W)] = 0.0
        # cap to avoid overflow
        m = float(W.max())
        if m > 0:
            W = W / m
            W[W < 1e-12] = 0.0
        F = W @ X
        dot = np.sum(F * X, axis=1, keepdims=True)
        Ft = F - dot * X
        nrm = float(np.linalg.norm(Ft, axis=1).mean()) + 1e-12
        X = X - (step / nrm) * Ft
        X = _norm(X)
    return X

def _e8_roots():
    roots = []
    for i in range(8):
        for j in range(i+1, 8):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = np.zeros(8); v[i]=si; v[j]=sj
                    roots.append(v)
    for mask in range(256):
        if bin(mask).count('1') % 2 != 0:
            continue
        v = np.array([0.5 if ((mask >> k) & 1) else -0.5 for k in range(8)], dtype=float)
        roots.append(v)
    R = np.array(roots, dtype=float)
    R = R / np.linalg.norm(R, axis=1, keepdims=True)
    return R  # 240 x 8

def _halton(n, d, seed_off=0):
    primes = [2,3,5,7,11,13,17,19][:d]
    H = np.zeros((n, d))
    for j, p in enumerate(primes):
        for i in range(n):
            idx = i + 1 + seed_off*10007
            f = 1.0; r = 0.0
            base = p
            cur = idx
            denom = p
            while cur > 0:
                r += (cur % p) / denom
                cur //= p
                denom *= p
            H[i, j] = r
    # clip away 0/1
    H = np.clip(H, 1e-9, 1-1e-9)
    try:
        from scipy.stats import norm
        H = norm.ppf(H)
    except Exception:
        # fallback inverse approx via Box-Muller style: use uniform-> rough normal
        H = (H - 0.5) * 2.0
    H /= np.linalg.norm(H, axis=1, keepdims=True) + 1e-30
    return H

def _topvec(X, k=400):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    iu = np.triu_indices(X.shape[0], 1)
    c = G[iu]
    k = min(k, c.size)
    idx = np.argpartition(-c, k)[:k]
    v = np.sort(c[idx])[::-1]
    return v

def _is_dup(v1, m1, v2, m2):
    if abs(m1 - m2) > 2e-4:
        return False
    # scale-aware distance on top vectors
    d = float(np.linalg.norm(v1 - v2) / max(1, len(v1)))
    return d < 5e-5

def _find_hole_fast(X_keep, rng, n_samp=15000, iters=120):
    d = X_keep.shape[1]
    C = rng.standard_normal((n_samp, d))
    C = _norm(C)
    S = C @ X_keep.T
    sc = S.max(axis=1)
    bi = int(np.argmin(sc))
    v = C[bi].copy()
    lr = 0.05
    for it in range(iters):
        dts = X_keep @ v
        cur = float(dts.max())
        beta = 60.0 + 120.0*(it/max(1,iters))
        w = np.exp(beta*(dts-cur))
        g = w @ X_keep
        gt = g - float(g @ v)*v
        nn = float(np.linalg.norm(gt))+1e-12
        gt /= nn
        st = lr*(0.3**(it/iters))+0.002
        v = v - st*gt
        v /= np.linalg.norm(v)+1e-30
    return v

def _align_match(A, B):
    # ICP: Procrustes + Hungarian, returns B_rot, perm, R
    from scipy.optimize import linear_sum_assignment
    n, d = A.shape
    R = np.eye(d)
    perm = np.arange(n)
    for _ in range(4):
        Br = B @ R.T
        S = A @ R @ B.T  # n x n dots
        cost = -S
        try:
            ra, ca = linear_sum_assignment(cost)
        except Exception:
            break
        # ra should be 0..n-1
        perm_new = np.empty(n, dtype=int)
        perm_new[ra] = ca
        perm = perm_new
        Bp = B[perm]
        M = Bp.T @ A  # d x d
        try:
            U, _, Vt = np.linalg.svd(M)
            Q = U @ Vt  # maps Bp -> A
            RnewT = Q
            R = RnewT.T
        except Exception:
            break
    Br = B @ R.T
    return Br, perm, R

def _greedy_fix(C, target, rng, max_replace=6):
    C = C.copy()
    for _ in range(max_replace):
        G = C @ C.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m <= target + 2e-4:
            break
        # worst pair
        ij = int(np.argmax(G.max(axis=1)))
        # row max
        rowmax = G.max(axis=1)
        # find partner of ij
        j = int(np.argmax(G[ij]))
        # choose worse of the two by rowmax then by count
        cand = ij if rowmax[ij] >= rowmax[j] else j
        keep = np.delete(np.arange(C.shape[0]), cand)
        Xk = C[keep]
        v = _find_hole_fast(Xk, rng, n_samp=12000, iters=80)
        C[cand] = v
    return C

def _lp_polish(X, seconds):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    X = _norm(X.copy())
    n, d = X.shape
    best = _max_cos(X)
    radius = 2e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    iuI, iuJ = iu[0], iu[1]
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 1.2:
            break
        Gf = X @ X.T
        cfull = Gf[iuI, iuJ]
        m = float(cfull.max())
        if m < best:
            best = m
        sel = cfull > m - max(20*radius, 1e-9)
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5; continue
        if cnt > 2500:
            idx = np.argpartition(-cfull, 2500)[:2500]
            mask = np.zeros_like(sel); mask[idx]=True
            sel = mask
        I = iuI[sel]; J = iuJ[sel]; c = cfull[sel]
        p = len(I)
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)],1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p,1))],1).ravel()
        try:
            A_ub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq = coo_matrix((X.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        except Exception:
            radius*=0.5; continue
        obj = np.zeros(nv); obj[-1]=1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status != 0:
            radius*=0.5; continue
        y = X + res.x[:-1].reshape(n,d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best - 1e-15:
            X, best = y, v
            radius = min(radius*1.5, 0.03)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    BUDGET = 175.0
    inc = None; inc_val = 1e9
    for f in ("best.npy","record.npy"):
        try:
            A = np.load(f).astype(float)
            A = _norm(A)
            if A.shape==(162,8):
                v=_max_cos(A)
                if v<inc_val:
                    inc_val=v; inc=A.copy()
        except Exception:
            pass
    if inc is None:
        inc = _norm(np.random.default_rng(0).standard_normal((162,8)))
        inc_val = _max_cos(inc)
    try:
        E8 = _e8_roots()
    except Exception:
        E8 = None

    elites = []  # list of dict val,X,top
    def try_add(X):
        nonlocal elites
        try:
            X=_norm(X); v=_max_cos(X); tv=_topvec(X)
        except Exception:
            return False
        for e in elites:
            if _is_dup(tv,v,e['top'],e['val']):
                # keep better of duplicates
                if v < e['val']-1e-12:
                    e['val']=v; e['X']=X.copy(); e['top']=tv
                return False
        elites.append({'val':v,'X':X.copy(),'top':tv})
        elites.sort(key=lambda e:e['val'])
        if len(elites)>8:
            elites=elites[:8]
        return True

    # seed incumbent as elite
    try_add(inc)

    rng = np.random.default_rng(12345)
    # ---- Phase 1: diverse optima ----
    # build starts
    starts = []
    # random starts
    for s in [11,22,33]:
        starts.append(('rand',s))
    starts.append(('halton',0))
    starts.append(('halton',7))
    if E8 is not None:
        starts.append(('e8',101))
        starts.append(('e8',202))
    starts.append(('pert_big',301))
    starts.append(('pert_med',302))

    def make_start(kind, seed):
        r = np.random.default_rng(9000+seed)
        if kind=='rand':
            return _norm(r.standard_normal((162,8)))
        elif kind=='halton':
            H=_halton(162,8,seed_off=seed)
            # tiny jitter to break exact symmetry
            H = _norm(H + 0.01*r.standard_normal((162,8)))
            return H
        elif kind=='e8':
            idx = r.choice(240, size=162, replace=False)
            X = E8[idx].copy()
            sig = 0.03
            X = _norm(X + sig*r.standard_normal((162,8)))
            return X
        elif kind=='pert_big':
            return _norm(inc + 0.35*r.standard_normal((162,8)))
        elif kind=='pert_med':
            return _norm(inc + 0.12*r.standard_normal((162,8)))
        else:
            return _norm(r.standard_normal((162,8)))

    # anneal schedules per start index for diversity (Riesz exponents)
    scheds = [
        (2.0,6.0,0.03,0.003, 25.0,350.0,0.012,0.001),
        (1.0,4.0,0.03,0.003, 15.0,500.0,0.012,0.0008),
        (3.0,9.0,0.025,0.002, 40.0,450.0,0.01,0.0008),
        (2.0,5.0,0.03,0.003, 20.0,300.0,0.012,0.001),
        (1.5,7.0,0.03,0.002, 30.0,400.0,0.01,0.0008),
    ]
    si=0
    for (kind,sd) in starts:
        if time.time()-t0 > 75.0: break
        if len(elites)>=8 and si>=6: break
        try:
            X = make_start(kind, sd+si*17)
            p0,p1,sp0,sp1,a0,a1,sa0,sa1 = scheds[si % len(scheds)]
            X = _anneal_pow(X, 700, p0,p1,sp0,sp1)
            if time.time()-t0>80: 
                try_add(X); si+=1; continue
            X = _anneal_exp(X, 700, a0,a1,sa0,sa1)
            try_add(X)
        except Exception:
            pass
        si+=1

    # ensure at least 3 elites; if not, add perturbed quick ones
    extra_seed=999
    while len(elites)<3 and (time.time()-t0<85):
        try:
            r=np.random.default_rng(extra_seed); extra_seed+=1
            X=_norm(inc+0.2*r.standard_normal((162,8)))
            X=_anneal_exp(X,500,20.0,300.0,0.012,0.001)
            try_add(X)
            if len(elites)>=3: break
        except Exception:
            break

    elites.sort(key=lambda e:e['val'])
    # ---- Phase 2: recombination ----
    # iterate parent pairs
    pair_order=[]
    L=len(elites)
    for i in range(L):
        for j in range(i+1,L):
            pair_order.append((i,j))
    if not pair_order and len(elites)>=2:
        pair_order=[(0,1)]
    # deterministic rng for children
    crng = np.random.default_rng(777)
    child_count=0
    pi=0
    # time for recombination until BUDGET-28
    while time.time()-t0 < BUDGET-28.0:
        if len(elites)<2: break
        # refresh pair list occasionally
        L=len(elites)
        a_idx = pi % L
        b_idx = (pi*3+1) % L
        if a_idx==b_idx: b_idx=(b_idx+1)%L
        pi+=1
        A = elites[a_idx]['X']; B = elites[b_idx]['X']
        va = elites[a_idx]['val']; vb = elites[b_idx]['val']
        target = min(va,vb)
        try:
            Br, perm, R = _align_match(A,B)
        except Exception:
            continue
        try:
            Bm = Br[perm]  # aligned to A
            # per-point row max for better-pick
            GA=A@A.T; np.fill_diagonal(GA,-2.0); ra=GA.max(axis=1)
            GBm=Bm@Bm.T; np.fill_diagonal(GBm,-2.0); rb=GBm.max(axis=1)
        except Exception:
            continue
        children=[]
        try:
            # child 1: random matched subset
            mask = crng.random(162)<0.5
            C1 = np.where(mask[:,None], A, Bm)
            children.append(C1)
            # child 2: better per pair
            pickA = ra <= rb
            C2 = np.where(pickA[:,None], A, Bm)
            # add small jitter to avoid exact duplicates
            C2 = _norm(C2 + 0.002*crng.standard_normal((162,8)))
            children.append(C2)
            # child 3: complementary spatial halves
            u = crng.standard_normal(8); u/=np.linalg.norm(u)+1e-30
            pa = A@u; pb = Bm@u
            ma = np.median(pa); mb = np.median(pb)
            idxA = np.where(pa>ma)[0]; idxB = np.where(pb<=mb)[0]
            # adjust to 81/81
            # if sizes off, trim/pad deterministically
            if len(idxA)>81: 
                o=np.argsort(-pa[idxA]); idxA=idxA[o[:81]]
            if len(idxB)>81:
                o=np.argsort(pb[idxB]); idxB=idxB[o[:81]]
            # pad if short (rare)
            C3list=[A[idxA],Bm[idxB]]
            # if total <162, fill with remaining best-hole? just fill with leftover
            leftA=np.delete(np.arange(162),idxA) if len(idxA)>0 else np.arange(162)
            leftB=np.delete(np.arange(162),idxB) if len(idxB)>0 else np.arange(162)
            need = 162-len(idxA)-len(idxB)
            fill=[]
            k=0
            while need>0 and k<162:
                # alternate leftovers
                if k<len(leftA) and need>0:
                    fill.append(A[leftA[k]]); need-=1
                if k<len(leftB) and need>0:
                    fill.append(Bm[leftB[k]]); need-=1
                k+=1
            if fill:
                C3list.append(np.array(fill))
            C3=np.vstack(C3list)
            # if still not 162 due to trimming, fix
            if C3.shape[0]>162: C3=C3[:162]
            while C3.shape[0]<162:
                C3=np.vstack([C3,C3[:1]])
            children.append(C3)
        except Exception:
            continue
        for C in children:
            if time.time()-t0 > BUDGET-28.0: break
            try:
                C=_norm(C)
                # greedy repair above target
                C=_greedy_fix(C,target,crng,max_replace=5)
                # smooth repair: short power+exp
                C=_anneal_exp(C,450,30.0,400.0,0.01,0.0009)
                # extra sharper pass
                C=_anneal_exp(C,250,80.0,600.0,0.006,0.0006)
                try_add(C)
                child_count+=1
            except Exception:
                continue
        # stop if many children and time going
        if child_count>70:
            break
        # refresh elites order (best first) - keep indices valid by re-sorting
        elites.sort(key=lambda e:e['val'])

    elites.sort(key=lambda e:e['val'])
    # ---- Phase 3: LP polish best ----
    remain = BUDGET-(time.time()-t0)
    # reserve 4s
    lp_total = max(0.0, remain-4.0)
    if lp_total>3 and len(elites)>0:
        # polish top 2 with split 70/30 capped 25s each
        cands = elites[:2] if len(elites)>=2 else elites[:1]
        secs=[]
        if len(cands)==1:
            secs=[min(lp_total,25.0)]
        else:
            secs=[min(lp_total*0.7,25.0), min(lp_total*0.3,15.0)]
        for k,e in enumerate(cands):
            if secs[k]<3: continue
            if time.time()-t0>BUDGET-5: break
            try:
                Y=_lp_polish(e['X'],secs[k])
                try_add(Y)
            except Exception:
                continue
    elites.sort(key=lambda e:e['val'])
    bestX = elites[0]['X'] if len(elites)>0 else inc
    bestV = elites[0]['val'] if len(elites)>0 else inc_val
    if inc_val < bestV:
        return _norm(inc)
    return _norm(bestX)
