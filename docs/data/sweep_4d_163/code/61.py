import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _smooth(x, iters, T0, T1, lr0, lr1):
    n, d = x.shape
    for k in range(iters):
        frac = k / max(iters - 1, 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        rs = W.sum(axis=1, keepdims=True)
        step = gt / np.maximum(rs, 1e-12)
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        step = step / np.maximum(sn, 1.0)
        x = x - lr * step
        x = _norm(x)
    return x

def _contact_adj(x, thr_offset=0.008, topk=600):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    thr = m - thr_offset
    iu0, iu1 = np.triu_indices(n, 1)
    c = G[iu0, iu1]
    sel = np.where(c > thr)[0]
    if sel.size > topk:
        kth = np.sort(c[sel])[-topk]
        sel = np.where(c >= kth)[0][:topk*2]
    adj = set(zip(iu0[sel].tolist(), iu1[sel].tolist()))
    # normalize order
    adj = set((a,b) if a<b else (b,a) for a,b in adj)
    return adj, m

def _void_vec(x, rng, n_samp=1200):
    d = x.shape[1]
    cand = rng.normal(size=(n_samp, d))
    cand = _norm(cand)
    sc = cand @ x.T
    mx = sc.max(axis=1)
    i = int(np.argmin(mx))
    v = cand[i].copy()
    for _ in range(20):
        dots = x @ v
        w = np.exp(40.0 * (dots - dots.max()))
        g = w @ x
        gt = g - np.dot(g, v) * v
        nrm = np.linalg.norm(gt)
        if nrm < 1e-12:
            break
        gt /= nrm
        v = v - 0.06 * gt
        v /= np.linalg.norm(v) + 1e-12
    return v

def _kabsch_R(P, A):
    # R mapping P -> A : minimize ||P R - A||
    M = P.T @ A
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R

def _align_match(A, B, iters=3):
    # align B onto A, return B_aligned, perm where A[i] <-> B_aligned[perm[i]]? actually B_aligned reordered?
    # We keep B_aligned in original order rotated, plus perm array
    Bc = B.copy()
    n = A.shape[0]
    perm = np.arange(n)
    Rtot = np.eye(4)
    for _ in range(iters):
        C = 1.0 - A @ Bc.T
        r, cix = linear_sum_assignment(C)
        # r should be arange
        perm = cix[np.argsort(r)] if not np.array_equal(r, np.arange(n)) else cix
        P = Bc[perm]
        R = _kabsch_R(P, A)
        Bc = Bc @ R
        Rtot = Rtot @ R
    # final matching after last rotation
    C = 1.0 - A @ Bc.T
    r, cix = linear_sum_assignment(C)
    perm = cix[np.argsort(r)] if not np.array_equal(r, np.arange(n)) else cix
    return Bc, perm

def _clifford_code(n, n_layers, rng, jitter=0.0, seed_off=0.0):
    # allocate points over Clifford tori
    base = n // n_layers
    rem = n - base * n_layers
    counts = [base + (1 if l < rem else 0) for l in range(n_layers)]
    X = []
    phi = (np.sqrt(5)-1)/2*2*np.pi
    for l, nl in enumerate(counts):
        alpha = (l + 0.5) / n_layers * np.pi / 2
        # avoid degenerate tori
        alpha = min(max(alpha, 0.08), np.pi/2-0.08)
        r1 = np.cos(alpha); r2 = np.sin(alpha)
        ph1 = seed_off + l * 0.7
        ph2 = seed_off*1.3 + l * 1.1
        for i in range(nl):
            s = i / max(nl,1)
            a1 = 2*np.pi*s + ph1
            a2 = 2*np.pi*s*1.0 + s*phi*2 + ph2  # winding
            X.append([r1*np.cos(a1), r1*np.sin(a1), r2*np.cos(a2), r2*np.sin(a2)])
    X = np.array(X, dtype=float)
    if jitter > 0:
        X = X + rng.normal(scale=jitter, size=X.shape)
    return _norm(X)

def _lp_polish(x, seconds, r0=2e-3, cap=1500):
    x = _norm(x)
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    best = _max_cos(x)
    radius = r0
    start = time.time()
    cd = np.arange(d)
    while time.time() - start < seconds and radius > 1e-13:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thr = m - max(20.0*radius, 1e-9)
        sel = np.where(c_all > thr)[0]
        if sel.size == 0:
            break
        if sel.size > cap:
            v = c_all[sel]
            kth = np.sort(v)[-cap]
            sel = np.where(c_all >= kth)[0]
            if sel.size > cap + 400:
                sel = sel[::2][:cap]
        I = iu0[sel]; J = iu1[sel]
        c = c_all[sel]
        p = len(I)
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d+cd, J[:,None]*d+cd, np.full((p,1), n*d)],1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
        A_ub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
        A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        obj = np.zeros(nv); obj[-1]=1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        y = _norm(x + res.x[:-1].reshape(n,d))
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius*1.6, 0.02)
        else:
            radius*=0.5
    return x

def solve():
    t0 = time.time()
    DEAD = 168.0
    rng = np.random.default_rng(20261031)
    n, d = 163, 4
    inc = None; inc_val = 1e9
    for f in ("best.npy","record.npy"):
        try:
            c = np.asarray(np.load(f), dtype=float)
            if c.shape==(n,d):
                c = _norm(np.nan_to_num(c,nan=0.0,posinf=0.0,neginf=0.0))
                v = _max_cos(c)
                if v < inc_val:
                    inc_val=v; inc=c.copy()
        except Exception:
            pass
    if inc is None:
        inc = _norm(rng.normal(size=(n,d))); inc_val=_max_cos(inc)
    best = inc.copy(); best_val = inc_val

    # ---- Phase 1: diverse starts, cheap smooth-max only ----
    starts = []
    # perturbed incumbents at 0.01,0.03,0.1
    for amp, cnt in [(0.01,3),(0.03,3),(0.1,2)]:
        for _ in range(cnt):
            Y = _norm(inc + rng.normal(scale=amp,size=(n,d)))
            starts.append(Y)
    # Hopf/clifford layer codes
    starts.append(_clifford_code(n, 8, rng, 0.0, 0.0))
    starts.append(_clifford_code(n, 9, rng, 0.0, 2.1))
    starts.append(_clifford_code(n, 7, rng, 0.02, 4.2))
    # random codes
    starts.append(_norm(rng.normal(size=(n,d))))
    starts.append(_norm(rng.normal(size=(n,d))))
    starts.append(_norm(inc + rng.normal(scale=0.2,size=(n,d))))  # far kick
    # total 14
    pop = []
    for Y in starts:
        if time.time()-t0 > 55:
            break
        Yp = _smooth(Y.copy(), 220, 0.02, 0.0015, 0.07, 0.012)
        v = _max_cos(Yp)
        pop.append((v, Yp))
        if v < best_val:
            best_val=v; best=Yp.copy()
    pop.sort(key=lambda t:t[0])
    # keep top 8 as parents
    parents = [x for _,x in pop[:8]]
    pvals = sorted([v for v,_ in pop[:8]])
    # if too few, add incumbent
    if len(parents)<4:
        parents.append(inc)

    # ---- Phase 2: recombination with alignment+Hungarian ----
    # choose parent pairs: best vs others + diverse pairs
    def patch_from_contacts(X, rng):
        G = X @ X.T
        np.fill_diagonal(G,-2.0)
        m=float(G.max())
        thr = m-0.008
        # adjacency
        adj=[[] for _ in range(n)]
        iu0,iu1=np.triu_indices(n,1)
        mask = G[iu0,iu1]>thr
        I=iu0[mask]; J=iu1[mask]
        for a,b in zip(I,J):
            adj[a].append(b); adj[b].append(a)
        seed=int(rng.integers(n))
        S=set([seed]); frontier=[seed]
        target=n//2
        while len(S)<target and frontier:
            nxt=[]
            for u in frontier:
                for w in adj[u]:
                    if w not in S:
                        S.add(w)
                        nxt.append(w)
                        if len(S)>=target: break
                if len(S)>=target: break
            frontier=nxt
            if not frontier and len(S)<target:
                # add random unvisited
                cand=int(rng.integers(n))
                if cand not in S:
                    S.add(cand); frontier=[cand]
        mask_arr=np.zeros(n,dtype=bool); mask_arr[list(S)]=True
        return mask_arr

    children=[]
    # build pair list
    np.random.seed(0)
    pair_list=[]
    P=len(parents)
    # best with each other
    for j in range(1,P):
        pair_list.append((0,j,'half' if j%2==0 else 'patch'))
    # a few diverse non-best pairs
    for j in range(min(P,4)):
        for k in range(j+1,min(P,5)):
            if (j,k)!=(0,j) and len(pair_list)<10:
                pair_list.append((j,k,'patch' if (j+k)%2 else 'half'))
    for (ai,bi,kind) in pair_list:
        if time.time()-t0 > DEAD-32:
            break
        try:
            A=parents[ai]; B=parents[bi]
            Bb,perm=_align_match(A,B,iters=2)
            # Bb is B rotated; perm maps A[i]->Bb[perm[i]]
            if kind=='half':
                # principal-coordinate half of A
                C=np.cov(A,rowvar=False)
                w,V=np.linalg.eigh(C)
                pc=A@V[:,-1]
                med=np.median(pc)
                S=(pc>med)
                if S.sum()==0 or S.sum()==n:
                    S=np.zeros(n,dtype=bool); S[:n//2]=True
            else:
                S=patch_from_contacts(A,rng)
            # child: S from A, complement from matched B
            Ch=np.empty_like(A)
            # for i in S take A[i]; else take Bb[perm[i]]
            Ch[S]=A[S]
            comp=~S
            Ch[comp]=Bb[perm[comp]]
            # repair near-duplicates: if any pair >0.995 or >best_val+0.005, reinsert void
            G=Ch@Ch.T
            np.fill_diagonal(G,-2.0)
            # find duplicates to fix (limit 4)
            fixed=0
            for _ in range(4):
                G=Ch@Ch.T
                np.fill_diagonal(G,-2.0)
                mx=float(G.max())
                if mx<0.995:
                    break
                ij=np.unravel_index(int(np.argmax(G)),G.shape)
                # replace the one with larger row-max (or second)
                rm=G.max(axis=1)
                victim=ij[0] if rm[ij[0]]>=rm[ij[1]] else ij[1]
                Ch[victim]=_void_vec(np.delete(Ch,victim,axis=0),rng,800)
            Ch=_norm(Ch)
            # smooth polish cheap
            Ch=_smooth(Ch,180,0.015,0.0012,0.06,0.01)
            v=_max_cos(Ch)
            children.append((v,Ch))
            if v<best_val:
                best_val=v; best=Ch.copy()
        except Exception:
            continue

    # survivor selection by score + Hamming diversity (implicit: keep best few diverse)
    children.sort(key=lambda t:t[0])
    surv=[]
    kept_adjs=[]
    for v,Ch in children:
        try:
            adj,_=_contact_adj(Ch)
        except Exception:
            adj=set()
        div=True
        for ka in kept_adjs:
            ham=len(adj.symmetric_difference(ka))
            if ham<60:  # too similar
                div=False; break
        # always keep top2 regardless
        if len(surv)<2 or div:
            surv.append((v,Ch)); kept_adjs.append(adj)
        if len(surv)>=5:
            break
    # second generation: recombine survivors with best if time allows
    if time.time()-t0 < DEAD-40 and len(surv)>=2:
        try:
            for si in range(min(2,len(surv))):
                if time.time()-t0 > DEAD-35:
                    break
                A=surv[si][1]; B=inc
                Bb,perm=_align_match(A,B,iters=2)
                C=np.cov(A,rowvar=False)
                w,V=np.linalg.eigh(C)
                pc=A@V[:,-1]; med=np.median(pc)
                S=(pc>med)
                Ch=np.empty_like(A); Ch[S]=A[S]; Ch[~S]=Bb[perm[~S]]
                Ch=_norm(Ch)
                Ch=_smooth(Ch,150,0.012,0.001,0.05,0.01)
                v=_max_cos(Ch)
                children.append((v,Ch))
                if v<best_val:
                    best_val=v; best=Ch.copy()
        except Exception:
            pass

    # ---- Phase 3: LP only for best child + incumbent, final 25s ----
    remain = DEAD-(time.time()-t0)
    # reserve 1.5s for return
    lp_budget = min(25.0, max(6.0, remain-2.0))
    if lp_budget>4 and time.time()-t0 < DEAD-3:
        # split: 70% best, 30% incumbent-polished? Actually just polish best; if best is inc, polish once
        t_lp0=time.time()
        # polish best
        try:
            best=_lp_polish(best, lp_budget*0.7, r0=2e-3, cap=1500)
            best_val=_max_cos(best)
        except Exception:
            pass
        left = lp_budget-(time.time()-t_lp0)
        # micro shake of best + smooth as alternative basin, then LP remainder
        if left>5 and DEAD-(time.time()-t0)>6:
            try:
                Y=_norm(best+rng.normal(scale=0.003,size=best.shape))
                Y=_smooth(Y,100,0.008,0.0012,0.03,0.008)
                v=_max_cos(Y)
                if v<best_val:
                    best_val=v; best=Y
                # final LP with remaining
                rem2=min(left-1.0, DEAD-(time.time()-t0)-1.5)
                if rem2>2:
                    best=_lp_polish(best, rem2, r0=8e-4, cap=1500)
                    best_val=_max_cos(best)
            except Exception:
                pass
    if _max_cos(best)>inc_val:
        best=inc.copy()
    return _norm(best)
