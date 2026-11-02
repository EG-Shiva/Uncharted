import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.linalg import orthogonal_procrustes

def solve():
    t0 = time.time()
    HARD = 170.0
    deadline = t0 + HARD
    rng = np.random.default_rng(12345)
    n, d = 153, 4
    iu0, iu1 = np.triu_indices(n, 1)

    def normed(x):
        x = np.asarray(x, dtype=float)
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        return x / nr

    def max_cos(x):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        return float(G.max())

    def contact_mask(x, v=None):
        if v is None:
            v = max_cos(x)
        G = x @ x.T
        c = G[iu0, iu1]
        return c > (v - 1e-6)

    try:
        xb0 = np.load("best.npy")
    except Exception:
        try:
            xb0 = np.load("record.npy")
        except Exception:
            xb0 = rng.normal(size=(n, d))
    xb0 = np.asarray(xb0, dtype=float)
    if xb0.shape != (n, d):
        xb0 = xb0.reshape(n, d)
    xb0 = normed(xb0)
    try:
        best_val0 = max_cos(xb0)
    except Exception:
        best_val0 = 1.0
    x_best = xb0.copy()
    best_val = best_val0

    # ---------- softmax anneal ----------
    def softmax_opt(x_init, steps, tau0, tau1, lr0, lr1, dl):
        x = x_init.copy()
        m1 = np.zeros_like(x)
        v1 = np.zeros_like(x)
        b1 = 0.9; b2 = 0.999; eps = 1e-8
        bv = max_cos(x)
        bx = x.copy()
        ltau = np.log(max(tau1,1e-12)/max(tau0,1e-12))/max(steps-1,1)
        llr = np.log(lr1/lr0)/max(steps-1,1)
        for s in range(steps):
            if (s & 127) == 0 and time.time() > dl:
                break
            tau = tau0*np.exp(ltau*s)
            lr = lr0*np.exp(llr*s)
            G = x @ x.T
            np.fill_diagonal(G, -10.0)
            gm = float(G.max())
            # weights
            W = np.exp((G-gm)/max(tau,1e-12))
            np.fill_diagonal(W, 0.0)
            sW = float(W.sum())
            if sW <= 0:
                break
            W /= sW
            grad = W @ x
            dot = (grad*x).sum(axis=1, keepdims=True)
            gt = grad - dot*x
            m1 = b1*m1 + (1-b1)*gt
            v1 = b2*v1 + (1-b2)*(gt*gt)
            c1 = 1-b1**(s+1); c2 = 1-b2**(s+1)
            step = (m1/c1)/(np.sqrt(v1/c2)+eps)
            x = x - lr*step
            x = normed(x)
            if (s & 31) == 0 or s == steps-1:
                v = max_cos(x)
                if v < bv:
                    bv = v; bx = x.copy()
        return bx, bv

    def lp_polish(x_init, budget):
        x = normed(x_init)
        try:
            best = max_cos(x)
        except Exception:
            best = 1.0
        radius = 1e-3
        start = time.time()
        while time.time()-start < budget and radius > 1e-14:
            if time.time() > deadline-2:
                break
            G = x @ x.T
            c_all = G[iu0, iu1]
            m = float(c_all.max())
            sel = c_all > m - max(20*radius, 1e-9)
            I = iu0[sel]; J = iu1[sel]
            csel = c_all[sel]
            p = len(I)
            if p == 0:
                break
            nv = n*d+1
            rows = np.repeat(np.arange(p), 2*d+1)
            cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1),n*d)],1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
            try:
                A_ub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
                A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
            except Exception:
                break
            obj = np.zeros(nv); obj[-1]=1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
            except Exception:
                radius*=0.5; continue
            if res.status!=0:
                radius*=0.5; continue
            y = x + res.x[:-1].reshape(n,d)
            y = normed(y)
            v = max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius*1.5, 0.05)
            else:
                radius*=0.5
        return x, best

    pool = []  # list of dicts
    graphs = []
    gvals = []

    def try_add(x, v=None):
        nonlocal x_best, best_val
        x = normed(x)
        if v is None:
            v = max_cos(x)
        if not np.isfinite(v) or v > 0.86:
            return False
        g = contact_mask(x, v)
        # hamming to pool
        if graphs:
            dists = [int(np.sum(g!=g0)) for g0 in graphs]
            md = min(dists)
            if md < 300:
                # allow if clearly better than similar one
                idx = int(np.argmin(dists))
                if v >= gvals[idx]-1e-6:
                    return False
                # replace similar
                # remove old
                # keep best
                pass
        graphs.append(g); gvals.append(v)
        pool.append((v, x.copy()))
        # track global best
        if v < best_val:
            best_val = float(v); x_best = x.copy()
        # sort pool by val, trim to 10 keeping diversity: simple sort+trim worst
        pool.sort(key=lambda t: t[0])
        # trim graphs correspondingly: rebuild alignment (pool and graphs may desync after replace logic)
        # For simplicity, if len>10 drop worst and its graph (need index). We appended; find worst index.
        if len(pool) > 10:
            # remove worst
            pool.pop()
            # remove graph of worst: find graph with max gval
            mi = int(np.argmax(gvals))
            graphs.pop(mi); gvals.pop(mi)
            # Note pool/graphs order not same; but diversity check only uses graphs set, ok
        return True

    # seed pool with incumbent
    try_add(xb0, best_val0)
    try:
        xr0 = np.load("record.npy")
        xr0 = normed(xr0.reshape(n,d))
        try_add(xr0)
    except Exception:
        pass

    # ---- Phase 1: diverse local optima ----
    # large perturbations of record
    sigmas = [0.15, 0.18, 0.22, 0.28, 0.35, 0.15, 0.22, 0.30]
    for si, sig in enumerate(sigmas):
        if time.time() > deadline-110:
            break
        if len(pool) >= 7 and si >= 4:
            break
        base = xb0
        y0 = base + rng.normal(0, sig, size=base.shape)
        y0 = normed(y0)
        dl = min(deadline-100, time.time()+7)
        steps = 1200
        y, _ = softmax_opt(y0, steps, 0.02, 0.004, 0.02, 0.0015, dl)
        if time.time() < deadline-100:
            y, v = lp_polish(y, 0.9)
        else:
            v = max_cos(y)
        try_add(y, v)

    # random starts
    nrand = 0
    while len(pool) < 8 and time.time() < deadline-100 and nrand < 6:
        nrand += 1
        y0 = normed(rng.normal(size=(n,d)))
        dl = min(deadline-100, time.time()+7)
        y, _ = softmax_opt(y0, 1500, 0.025, 0.003, 0.025, 0.0015, dl)
        if time.time() < deadline-98:
            y, v = lp_polish(y, 0.9)
        else:
            v = max_cos(y)
        try_add(y, v)

    # overload squeeze-out: 157-159 points then delete
    for nbig in [157, 159, 158]:
        if time.time() > deadline-95:
            break
        if len(pool) >= 9:
            break
        y0 = normed(rng.normal(size=(nbig,d)))
        # quick softmax on big code
        # inline big softmax (reuse logic with different n)
        def softmax_big(x_init, steps, dl):
            x = x_init.copy()
            m1=np.zeros_like(x); v1=np.zeros_like(x)
            b1=0.9;b2=0.999;eps=1e-8
            for s in range(steps):
                if (s&127)==0 and time.time()>dl:
                    break
                tau=0.02*np.exp(np.log(0.004/0.02)*s/max(steps-1,1))
                lr=0.02*np.exp(np.log(0.0015/0.02)*s/max(steps-1,1))
                G=x@x.T; np.fill_diagonal(G,-10.0)
                gm=float(G.max())
                W=np.exp((G-gm)/max(tau,1e-12)); np.fill_diagonal(W,0.0)
                sW=float(W.sum())
                if sW<=0: break
                W/=sW
                grad=W@x
                dot=(grad*x).sum(axis=1,keepdims=True)
                gt=grad-dot*x
                m1=b1*m1+(1-b1)*gt; v1=b2*v1+(1-b2)*(gt*gt)
                c1=1-b1**(s+1); c2=1-b2**(s+1)
                step=(m1/c1)/(np.sqrt(v1/c2)+eps)
                x=x-lr*step
                x=normed(x)
            return x
        dl = min(deadline-95, time.time()+6)
        yb = softmax_big(y0, 1000, dl)
        # contact load: count within 1e-3 of max + max value
        Gb = yb@yb.T; np.fill_diagonal(Gb,-2.0)
        mb = float(Gb.max())
        # per-point load: number of contacts within 2e-3 + sum
        loads = (Gb > mb-2e-3).sum(axis=1).astype(float) + Gb.max(axis=1)*0.1
        order = np.argsort(-loads)
        drop = order[:nbig-n]
        keep = np.array([i for i in range(nbig) if i not in set(drop.tolist())])
        y = normed(yb[keep])
        dl2 = min(deadline-93, time.time()+5)
        y,_ = softmax_opt(y, 900, 0.015, 0.003, 0.015, 0.001, dl2)
        if time.time() < deadline-92:
            y,v = lp_polish(y, 0.8)
        else:
            v = max_cos(y)
        try_add(y, v)

    # ensure pool sorted
    # rebuild pool sorted by value (graphs desync tolerated)
    pool.sort(key=lambda t: t[0])

    def crossover_child(A, B):
        try:
            R,_ = orthogonal_procrustes(A, B)
            Aal = A @ R
        except Exception:
            Aal = A.copy()
        u = rng.normal(size=4); u/=max(np.linalg.norm(u),1e-12)
        dA = Aal @ u; dB = B @ u
        mA = dA > 0; mB = dB <= 0
        pts = np.concatenate([Aal[mA], B[mB]], axis=0)
        k = pts.shape[0]
        # resolve length
        if k > n:
            # greedily remove most crowded
            pts_list = pts
            while pts_list.shape[0] > n:
                G = pts_list @ pts_list.T
                np.fill_diagonal(G, -2.0)
                per = G.max(axis=1)
                idx = int(np.argmax(per))
                pts_list = np.delete(pts_list, idx, axis=0)
            pts = pts_list
        elif k < n:
            need = n-k
            cur = pts
            for _ in range(need):
                cand = normed(rng.normal(size=(800,d)))
                # score by max dot to cur
                S = cur @ cand.T  # (kcur,800)
                mx = S.max(axis=0)
                bi = int(np.argmin(mx))
                cur = np.concatenate([cur, cand[bi:bi+1]], axis=0)
            pts = cur
        if pts.shape[0] != n:
            # fallback pad/trim
            if pts.shape[0] > n:
                pts = pts[:n]
            else:
                extra = normed(rng.normal(size=(n-pts.shape[0],d)))
                pts = np.concatenate([pts,extra],axis=0)
        # resolve near-duplicates: if any pair >0.985, jitter one
        G = pts @ pts.T; np.fill_diagonal(G,-2.0)
        if float(G.max()) > 0.985:
            # jitter the second of worst pair
            ij = np.unravel_index(int(G.argmax()), G.shape)
            pts[ij[1]] = normed(rng.normal(size=(1,4)))
        return normed(pts)

    # ---- Phase 2: selection-crossover-polish ----
    it = 0
    while time.time() < deadline-22:
        it += 1
        if len(pool) < 2:
            # make random child from best perturbation
            y0 = normed(x_best + rng.normal(0,0.2,size=x_best.shape))
            dl = min(deadline-20, time.time()+5)
            y,_ = softmax_opt(y0, 900, 0.015, 0.003, 0.015, 0.001, dl)
            y,v = lp_polish(y, 0.7)
            try_add(y,v)
            continue
        # pick two parents biased to best
        # pool sorted; pick idx via random geometric
        def pick():
            if rng.random()<0.6:
                # among top 3
                return int(rng.integers(0, min(3,len(pool))))
            else:
                return int(rng.integers(0, len(pool)))
        a = pick(); b = pick()
        if a==b:
            b = int(rng.integers(0,len(pool)))
        A = pool[a][1]; B = pool[b][1]
        # random swap order
        if rng.random()<0.5:
            A,B = B,A
        child0 = crossover_child(A,B)
        # anneal
        dl = min(deadline-20, time.time()+4.5)
        tau0 = float(rng.choice([0.015,0.02,0.01]))
        child,_ = softmax_opt(child0, 800, tau0, 0.0025, 0.015, 0.001, dl)
        # occasionally large void refill before polish: replace worst point with void sample
        if rng.random()<0.3:
            G = child@child.T; np.fill_diagonal(G,-2.0)
            per = G.max(axis=1)
            w = int(np.argmax(per))
            # find void: sample 1000, pick farthest
            cand = normed(rng.normal(size=(1000,d)))
            S = child @ cand.T
            mx = S.max(axis=0)
            bi = int(np.argmin(mx))
            child[w]=cand[bi]
            child=normed(child)
            dl2 = min(deadline-20, time.time()+2)
            child,_ = softmax_opt(child, 400, 0.008, 0.002, 0.008, 0.0008, dl2)
        if time.time() < deadline-20:
            child,v = lp_polish(child, 0.7)
        else:
            v = max_cos(child)
        try_add(child, v)
        # keep pool sorted
        pool.sort(key=lambda t: t[0])
        if it>60:
            # avoid endless loop, break if time low
            pass

    # final polish best
    remain = deadline - time.time() - 1.5
    if remain > 2.0:
        remain = min(remain, 18.0)
        try:
            yp, vp = lp_polish(x_best, remain)
            if vp < best_val:
                x_best, best_val = yp, vp
        except Exception:
            pass
    x_best = normed(x_best)
    return x_best
