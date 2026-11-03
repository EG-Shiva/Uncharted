import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contacts(x, tol=1e-9):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    gmax = float(G.max())
    rowmax = G.max(axis=1)
    deg = (G > (gmax - tol)).sum(axis=1)
    return G, gmax, rowmax, deg

def _adam_full(X, steps, beta0, beta1, lr0, lr1):
    n, d = X.shape
    m = np.zeros_like(X); v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lb0, lb1 = np.log(beta0), np.log(beta1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(1, steps+1):
        frac = t/steps
        beta = np.exp(lb0+(lb1-lb0)*frac)
        lr = np.exp(ll0+(ll1-ll0)*frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = G.max()
        W = np.exp(beta*(G-gmax))
        np.fill_diagonal(W, 0.0)
        Z = W.sum()
        if Z == 0:
            continue
        Grad = (W @ X)/Z
        dot = (Grad*X).sum(axis=1, keepdims=True)
        Gt = Grad - dot*X
        m *= b1; m += (1.0-b1)*Gt
        v *= b2; v += (1.0-b2)*(Gt*Gt)
        mhat = m/(1.0-b1**t); vhat = v/(1.0-b2**t)
        step = mhat/(np.sqrt(vhat)+eps)
        X = X - lr*step
        nrm = np.sqrt((X*X).sum(axis=1, keepdims=True)); nrm[nrm==0]=1.0
        X /= nrm
    return X

def _adam_masked(X, movable, steps, beta0, beta1, lr0, lr1):
    n, d = X.shape
    m = np.zeros_like(X); v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lb0, lb1 = np.log(beta0), np.log(beta1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    mov = movable.astype(bool)
    for t in range(1, steps+1):
        frac = t/steps
        beta = np.exp(lb0+(lb1-lb0)*frac)
        lr = np.exp(ll0+(ll1-ll0)*frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = G.max()
        W = np.exp(beta*(G-gmax))
        np.fill_diagonal(W, 0.0)
        Z = W.sum()
        if Z == 0:
            continue
        Grad = (W @ X)/Z
        dot = (Grad*X).sum(axis=1, keepdims=True)
        Gt = Grad - dot*X
        # freeze deep core: zero gradient there
        if not np.all(mov):
            Gt[~mov] = 0.0
            m[~mov] = 0.0; v[~mov] = 0.0
        m *= b1; m += (1.0-b1)*Gt
        v *= b2; v += (1.0-b2)*(Gt*Gt)
        bc1 = 1.0-b1**t; bc2 = 1.0-b2**t
        mhat = m/bc1; vhat = v/bc2
        step = mhat/(np.sqrt(vhat)+eps)
        step[~mov] = 0.0
        X = X - lr*step
        nrm = np.sqrt((X*X).sum(axis=1, keepdims=True)); nrm[nrm==0]=1.0
        X /= nrm
    return X

def _refine_hole(Y, p, iters=150, beta=90.0, lr0=0.10, lr1=0.003):
    p = p/np.linalg.norm(p)
    for it in range(iters):
        frac = it/max(iters-1,1)
        lr = lr0*(lr1/lr0)**frac
        g = Y @ p
        mx = g.max()
        w = np.exp(beta*(g-mx))
        s = w.sum()+1e-12
        A = (w @ Y)/s
        At = A - float(A @ p)*p
        p = p - lr*At
        nr = np.linalg.norm(p)
        if nr == 0:
            break
        p /= nr
    return p

def _best_probe(placed, rng, n_probe=50000):
    d = placed.shape[1]
    P = rng.randn(n_probe, d)
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    CH = 10000
    bs = np.empty(n_probe)
    for a in range(0, n_probe, CH):
        b = min(a+CH, n_probe)
        D = P[a:b] @ placed.T
        bs[a:b] = D.max(axis=1)
    i = int(np.argmin(bs))
    return P[i], float(bs[i])

def _lp_polish(x, seconds, r0=1e-3):
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
            break
        if p > 2500:
            idx = np.argsort(c[sel])[-2500:]
            I, J = I[idx], J[idx]
            cc = c[sel][idx]
            p = 2500
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
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n,d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius*1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    BUDGET = 174.0
    n, d = 168, 8
    best = None; best_val = 1e9
    for fn in ("best.npy","record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape != (n,d):
                continue
            a = a/np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            va = _max_cos(a)
            if va < best_val:
                best_val = va; best = a.copy()
        except Exception:
            pass
    if best is None:
        r0 = np.random.RandomState(0)
        best = r0.randn(n,d); best/=np.linalg.norm(best,axis=1,keepdims=True)
        best_val = _max_cos(best)
    cur = best.copy(); cur_val = best_val
    rng = np.random.RandomState(20261103)
    trial = 0
    Kchoices = [32,48,64]
    while True:
        trial += 1
        now = time.time()
        if now + 4.0 > t0 + BUDGET:
            break
        # time left guard: need at least ~3s per trial
        if now + 3.0 > t0 + 132.0 and trial > 5:
            # allow a few more quick trials but stop large blocks if late
            pass
        if now > t0 + 134.0:
            break
        X0 = cur.copy()
        try:
            G, gmax, rowmax, deg = _contacts(X0, tol=1e-9)
        except Exception:
            continue
        ratt = np.where(deg == 0)[0]
        # overloaded hubs deg>=8
        hubs_all = np.where(deg >= 8)[0]
        if len(hubs_all) == 0:
            hubs_all = np.argsort(deg)[-8:]
        # choose block size
        K = Kchoices[rng.randint(len(Kchoices))]
        # build block set to size K
        block_set = set()
        # how many rattlers to take
        if K == 64:
            # take all rattlers
            for i in ratt:
                block_set.add(int(i))
        elif K == 48:
            nr = min(len(ratt), 30 + int(rng.randint(6)))
            if len(ratt) > 0 and nr > 0:
                sel = rng.choice(ratt, min(nr,len(ratt)), replace=False)
                for i in sel:
                    block_set.add(int(i))
            else:
                sel = rng.choice(n, min(K//2,n), replace=False)
                for i in sel:
                    block_set.add(int(i))
        else:  # 32
            nr = min(len(ratt), 18 + int(rng.randint(6)))
            if len(ratt) > 0 and nr > 0:
                sel = rng.choice(ratt, min(nr,len(ratt)), replace=False)
                for i in sel:
                    block_set.add(int(i))
        # add 8-16 overloaded hubs
        nh = int(8 + rng.randint(9))  # 8..16
        # order hubs by degree
        order_h = hubs_all[np.argsort(deg[hubs_all])[::-1]]
        # if not enough hubs, add top-degree points
        if len(order_h) < nh:
            extra = np.argsort(deg)[::-1]
            lst = list(order_h)
            for e in extra:
                if int(e) not in block_set and int(e) not in lst:
                    lst.append(int(e))
                if len(lst) >= nh:
                    break
            order_h = np.array(lst, dtype=int)
        # random subset of hubs
        if len(order_h) > nh:
            hi = rng.choice(order_h, nh, replace=False)
        else:
            hi = order_h
        for h in hi:
            block_set.add(int(h))
        # add nearest neighbours of chosen hubs until size K
        # precompute nearest neighbour for each hub from G
        Gcopy = G.copy()
        for h in list(hi):
            if len(block_set) >= K:
                break
            row = Gcopy[int(h)].copy()
            row[int(h)] = -2
            # mask already in block
            for idx in np.argsort(row)[::-1]:
                if int(idx) not in block_set:
                    block_set.add(int(idx))
                    break
        # fill randomly if still short (high rowmax points)
        if len(block_set) < K:
            cand = np.argsort(rowmax)[::-1]
            for c in cand:
                if int(c) not in block_set:
                    block_set.add(int(c))
                if len(block_set) >= K:
                    break
        # if overfull (rattlers >K), trim: keep all hubs, trim rattlers randomly
        block = np.array(sorted(block_set), dtype=int)
        if len(block) > K and K != 64:
            # keep hubs, drop random others
            hubset = set(int(h) for h in hi)
            nonhub = [b for b in block if b not in hubset]
            keep_non = K - len(hubset)
            if keep_non < 0:
                block = np.array(sorted(list(hubset))[:K], dtype=int)
            else:
                sel = rng.choice(nonhub, keep_non, replace=False) if len(nonhub) > keep_non else nonhub
                block = np.array(sorted(list(hubset)+[int(s) for s in sel]), dtype=int)
        # for K==64 with rattlers 44 + hubs may exceed 64: allow up to ~80, trim to 72 max
        if len(block) > 80:
            # trim neighbours first
            block = block[rng.choice(len(block), 72, replace=False)]
            block = np.sort(block)
        core_mask = np.ones(n, dtype=bool); core_mask[block] = False
        core_idx = np.where(core_mask)[0]
        # sequential reinsertion: freeze core, grow placed set
        placed = X0[core_mask].copy()
        Xnew = X0.copy()
        perm = block[rng.permutation(len(block))]
        # use trial-specific rng for probes
        prng = np.random.RandomState(1000000+ trial*7919)
        beta_r = 70.0 + prng.rand()*60.0
        for jj, j in enumerate(perm):
            p0, _ = _best_probe(placed, prng, n_probe=50000)
            # refine hole against current placed
            try:
                p0 = _refine_hole(placed, p0, iters=150, beta=beta_r, lr0=0.10, lr1=0.003)
            except Exception:
                pass
            Xnew[int(j)] = p0
            placed = np.vstack([placed, p0[None,:]])
        # boundary: core points with dot>0.3 to any inserted
        try:
            Dcb = Xnew[core_idx] @ Xnew[block].T
            bmax = Dcb.max(axis=1)
            bnd_core = core_idx[bmax > 0.3]
        except Exception:
            bnd_core = np.array([], dtype=int)
        movable = np.zeros(n, dtype=bool)
        movable[block] = True
        movable[bnd_core] = True
        # jointly anneal inserted+boundary, deep core fixed, 1000 steps
        beta0 = 25.0 + rng.rand()*25.0
        beta1 = 250.0 + rng.rand()*350.0
        lr0 = 0.003 + rng.rand()*0.004
        try:
            Xann = _adam_masked(Xnew, movable, steps=1000, beta0=beta0, beta1=beta1, lr0=lr0, lr1=lr0*0.12)
        except Exception:
            continue
        # full thaw quench
        try:
            beta0b = 40.0 + rng.rand()*30.0
            beta1b = 300.0 + rng.rand()*400.0
            Xann = _adam_full(Xann, steps=700+int(rng.randint(400)), beta0=beta0b, beta1=beta1b, lr0=0.004, lr1=0.0004)
        except Exception:
            continue
        vann = _max_cos(Xann)
        if not np.isfinite(vann):
            continue
        # quick LP if promising
        if vann < best_val + 0.002 and (time.time()+3.0 < t0+BUDGET):
            try:
                Xp = _lp_polish(Xann, min(1.2, max(0.4, (t0+BUDGET-time.time())*0.05)))
                vp = _max_cos(Xp)
                if vp < vann and np.all(np.isfinite(Xp)):
                    Xann, vann = Xp, vp
            except Exception:
                pass
        if vann + 1e-12 < best_val:
            best = Xann.copy(); best_val = vann
            cur = Xann.copy(); cur_val = vann
        elif vann < cur_val or rng.rand() < np.exp((cur_val - vann)/0.0008):
            cur = Xann.copy(); cur_val = vann
        # restart wandering if too far
        if cur_val > best_val + 0.006 and rng.rand() < 0.6:
            cur = best.copy(); cur_val = best_val
    # final LP polish
    remaining = (t0+BUDGET)-time.time()
    if remaining > 7.0:
        try:
            secs = min(remaining-2.0, 40.0)
            polished = _lp_polish(best, secs)
            vp = _max_cos(polished)
            if vp < best_val and np.all(np.isfinite(polished)):
                best = polished; best_val = vp
        except Exception:
            pass
    best = best/np.linalg.norm(best, axis=1, keepdims=True)
    bad = ~np.all(np.isfinite(best), axis=1)
    if np.any(bad):
        rr = np.random.RandomState(1).randn(int(bad.sum()), d)
        rr /= np.linalg.norm(rr, axis=1, keepdims=True)
        best[bad] = rr
    best = best/np.linalg.norm(best, axis=1, keepdims=True)
    return best
