import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def row_max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1), G

def fast_relax(X, iters, lr0=0.03, margin0=0.003):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    cur = X.copy()
    best_val = max_cos(cur)
    bestX = cur.copy()
    lr = lr0
    margin = margin0
    for it in range(iters):
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        tau = margin * 0.2 + 1e-12
        # weights
        W = np.exp((G - m) / tau)
        W[G < m - margin * 5.0] = 0.0
        np.fill_diagonal(W, 0.0)
        Grad = W @ cur
        proj = np.sum(Grad * cur, axis=1, keepdims=True)
        Gt = Grad - proj * cur
        gn = float(np.max(np.linalg.norm(Gt, axis=1)))
        if not np.isfinite(gn) or gn < 1e-14:
            margin *= 0.5
            if margin < 1e-7:
                margin = 1e-7
                lr = lr0 * 0.5
            continue
        Xn = cur - (lr / gn) * Gt
        Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
        # eval
        G2 = Xn @ Xn.T
        np.fill_diagonal(G2, -2.0)
        mn = float(np.max(G2))
        if mn < m:
            cur = Xn
            if mn < best_val:
                best_val = mn
                bestX = Xn.copy()
            if it % 200 == 0 and it > 0:
                margin *= 0.85
                # keep lr
        else:
            lr *= 0.6
            if lr < 1e-6:
                margin *= 0.7
                lr = lr0 * 0.3
                if margin < 5e-7:
                    # nudge to avoid stall
                    margin = 5e-7
                    if lr < 1e-6:
                        break
    return bestX, best_val

def lp_polish(x, seconds, deadline=1e18):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    bestX = x.copy()
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    iuI, iuJ = iu0
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c = Gfull[iuI, iuJ]
        m = float(np.max(c))
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        I = iuI[sel]; J = iuJ[sel]; csel = c[sel]
        p = len(I)
        if p == 0:
            break
        if p > 3500:
            k = 3500
            part = np.argpartition(-csel, k-1)[:k]
            I, J, csel = I[part], J[part], csel[part]
            p = k
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d + np.arange(d), J[:,None]*d + np.arange(d), np.full((p,1), n*d)],1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
        A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
        A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d), np.arange(n*d))), shape=(n,nv)).tocsr()
        obj = np.zeros(nv); obj[-1]=1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n,d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            bestX = y.copy()
            radius = min(radius*1.5, 0.05)
        else:
            radius *= 0.5
    return bestX, best

def refine_void(v, others, iters=250, lr0=0.05):
    # minimize max dot(v, others) by tangential descent
    v = v / np.linalg.norm(v)
    lr = lr0
    best = float(np.max(others @ v))
    bestv = v.copy()
    for it in range(iters):
        dots = others @ v
        m = float(np.max(dots))
        # soft weights near max
        tau = 0.002
        w = np.exp((dots - m)/tau)
        w[dots < m - 0.01] = 0.0
        g = w @ others  # gradient of weighted sum wrt v
        # project to tangent
        gt = g - np.dot(g, v)*v
        nrm = np.linalg.norm(gt)
        if nrm < 1e-14 or not np.finite(nrm):
            break
        vn = v - (lr/nrm)*gt
        vn /= np.linalg.norm(vn)
        mn = float(np.max(others @ vn))
        if mn < m:
            v = vn
            if mn < best:
                best = mn; bestv = vn.copy()
        else:
            lr *= 0.7
            if lr < 1e-5:
                break
    return bestv, best

def void_reinsert(X, idx_list, rng, n_cand=20000):
    n, d = X.shape
    Y = X.copy()
    for idx in idx_list:
        others = np.delete(Y, idx, axis=0)
        cand = rng.standard_normal((n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        scores = np.max(cand @ others.T, axis=1)
        # take a few best candidates and refine
        order = np.argpartition(scores, 8)[:8]
        bestv = None; bests = 1e9
        for o in order:
            rv, rs = refine_void(cand[o], others, iters=200, lr0=0.05)
            if rs < bests:
                bests = rs; bestv = rv
        Y[idx] = bestv / np.linalg.norm(bestv)
    return Y

def solve():
    t0 = time.time()
    LIMIT = 172.0
    HOP_END = t0 + 118.0
    FINAL_END = t0 + LIMIT - 2.0
    rng = np.random.default_rng(7)
    bestX = None; best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=np.float64)
            if v.shape == (219,6):
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                if np.all(np.isfinite(v)):
                    c = max_cos(v)
                    if c < best_val:
                        best_val = c; bestX = v.copy()
        except Exception:
            pass
    if bestX is None:
        bestX = rng.standard_normal((219,6))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        best_val = max_cos(bestX)
    n, d = bestX.shape
    # brief initial squeeze
    if time.time() - t0 < HOP_END - 60:
        try:
            bx, bv = lp_polish(bestX, seconds=3.0, deadline=HOP_END)
            if bv < best_val and np.all(np.isfinite(bx)):
                best_val, bestX = bv, bx
        except Exception:
            pass
    trial = 0
    # precompute for speed
    while time.time() < HOP_END:
        trial += 1
        mode = trial % 5
        try:
            rm, _ = row_max_cos(bestX)
            worst_order = np.argsort(-rm)
        except Exception:
            worst_order = np.arange(n)
        Xtry = None
        if mode == 0:
            k = int(rng.integers(2,6))
            sigma = 0.08 + float(rng.random())*0.08
            idx = worst_order[:k].copy()
            Xtry = bestX.copy()
            Xtry[idx] += sigma * rng.standard_normal((k,d))
            Xtry[idx] /= np.linalg.norm(Xtry[idx], axis=1, keepdims=True)
            iters = 900
            lr0 = 0.03; margin0 = 0.003
        elif mode == 1:
            k = int(rng.integers(5,14))
            sigma = 0.02 + float(rng.random())*0.06
            # mix worst + random
            idx = np.concatenate([worst_order[:k//2], rng.choice(n, k - k//2, replace=False)])
            # unique
            idx = np.unique(idx)
            k = len(idx)
            Xtry = bestX.copy()
            Xtry[idx] += sigma * rng.standard_normal((k,d))
            Xtry[idx] /= np.linalg.norm(Xtry[idx], axis=1, keepdims=True)
            iters = 1000
            lr0 = 0.025; margin0 = 0.0035
        elif mode == 2:
            k = int(rng.integers(1,4))
            idx = worst_order[:k].copy()
            # void reinsert
            try:
                Xtry = void_reinsert(bestX, idx.tolist(), rng, n_cand=15000)
            except Exception:
                continue
            iters = 900
            lr0 = 0.03; margin0 = 0.003
        elif mode == 3:
            sigma = 0.005 + float(rng.random())*0.010
            Xtry = bestX + sigma * rng.standard_normal((n,d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            iters = 700
            lr0 = 0.02; margin0 = 0.0025
        else:
            k = int(rng.integers(3,10))
            sigma = 0.05 + float(rng.random())*0.07
            idx = rng.choice(n, k, replace=False)
            Xtry = bestX.copy()
            Xtry[idx] += sigma * rng.standard_normal((k,d))
            Xtry[idx] /= np.linalg.norm(Xtry[idx], axis=1, keepdims=True)
            iters = 900
            lr0 = 0.03; margin0 = 0.004
        # cheap relax screen
        try:
            Xr, vr = fast_relax(Xtry, iters=iters, lr0=lr0, margin0=margin0)
        except Exception:
            continue
        # screen threshold: only LP if promising
        if vr > best_val + 0.0015:
            continue
        # short LP squeeze
        remain_hop = HOP_END - time.time()
        if remain_hop <= 0.5:
            # keep if improved without LP
            if vr < best_val:
                best_val, bestX = vr, Xr
            break
        lp_sec = min(2.0, max(0.6, remain_hop/8.0))
        # if clearly better, give a bit more
        if vr < best_val:
            lp_sec = min(3.0, max(1.0, remain_hop/6.0))
        try:
            Xp, vp = lp_polish(Xr, seconds=lp_sec, deadline=HOP_END)
        except Exception:
            continue
        if vp < best_val - 1e-12:
            best_val, bestX = vp, Xp
        elif vr < best_val - 1e-12:
            # LP failed to improve but relax did
            best_val, bestX = vr, Xr
    # final intensive polish: alternate relax + LP
    try:
        # one more relax with tighter margin
        if time.time() < FINAL_END - 5:
            Xr, vr = fast_relax(bestX, iters=1200, lr0=0.02, margin0=0.002)
            if vr < best_val:
                best_val, bestX = vr, Xr
    except Exception:
        pass
    try:
        remain = FINAL_END - time.time()
        if remain > 3:
            Xp, vp = lp_polish(bestX, seconds=remain, deadline=FINAL_END)
            if vp < best_val and np.all(np.isfinite(Xp)):
                best_val, bestX = vp, Xp
    except Exception:
        pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    # safety: replace any non-finite row
    bad = ~np.all(np.isfinite(bestX), axis=1)
    if np.any(bad):
        brng = np.random.default_rng(0)
        for i in np.where(bad)[0]:
            v = brng.standard_normal(6); v/=np.linalg.norm(v); bestX[i]=v
    return bestX
