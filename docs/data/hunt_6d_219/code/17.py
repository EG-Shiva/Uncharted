import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _rowmax(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1), float(np.max(G)), G

def _lp_polish(x, seconds, deadline, iu=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    if iu is None:
        iu = np.triu_indices(n, 1)
    iuI, iuJ = iu
    best = _max_cos(x)
    bestX = x.copy()
    radius = 1e-3
    start = time.time()
    nv = n * d + 1
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-14:
            break
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(np.max(c))
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        I = iuI[sel]; J = iuJ[sel]; csel = c[sel]
        p = len(I)
        if p == 0:
            break
        if p > 3500:
            k = 3500
            part = np.argpartition(-csel, k-1)[:k]
            I, J, csel = I[part], J[part], csel[part]
            p = k
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)],1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
        try:
            A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))), shape=(n,nv)).tocsr()
            obj = np.zeros(nv); obj[-1]=1.0
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
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            bestX = y.copy()
            radius = min(radius*1.5, 0.05)
        else:
            radius *= 0.5
    return bestX, best

def _anneal(X, steps, T0, T1, lr0):
    cur = X.copy()
    n, d = cur.shape
    best = _max_cos(cur)
    bestX = cur.copy()
    if steps <= 1:
        return bestX, best
    dec = (T1/T0)**(1.0/(steps-1))
    T = T0
    for it in range(steps):
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        if m < best:
            best = m
            bestX = cur.copy()
        S = (G - m) / T
        W = np.exp(S)
        W[G < m - 8.0*T] = 0.0
        np.fill_diagonal(W, 0.0)
        Grad = W @ cur
        proj = np.sum(Grad*cur, axis=1, keepdims=True)
        Gt = Grad - proj*cur
        gn = float(np.max(np.linalg.norm(Gt, axis=1)))
        if not np.isfinite(gn) or gn < 1e-14:
            T *= dec
            continue
        cur = cur - (lr0/gn)*Gt
        cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        T *= dec
    v = _max_cos(cur)
    if v < best:
        best = v
        bestX = cur.copy()
    return bestX, best

def _refine_single(v, others, iters=120, tau=0.002):
    v = v/np.linalg.norm(v)
    best = float(np.max(others @ v))
    bestv = v.copy()
    lr = 0.03
    for it in range(iters):
        dots = others @ v
        m = float(np.max(dots))
        w = np.exp((dots - m)/tau)
        w[dots < m - 6*tau] = 0.0
        g = w @ others
        gt = g - np.dot(g, v)*v
        nn = float(np.linalg.norm(gt))
        if nn < 1e-14 or not np.isfinite(nn):
            break
        st = lr/(1.0+it*0.04)/nn
        vn = v - st*gt
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

def _void_reinsert(Y, idx_list, rng, n_cand=20000):
    n, d = Y.shape
    for idx in idx_list:
        others = np.delete(Y, idx, axis=0)
        cand = rng.standard_normal((n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        sc = cand @ others.T
        cm = np.max(sc, axis=1)
        j = int(np.argmin(cm))
        bv, _ = _refine_single(cand[j], others, iters=150)
        Y[idx] = bv
    return Y

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    HOP_END = t0 + 138.0
    rng = np.random.default_rng(777)
    n, d = 219, 6
    best_val = 1e9
    bestX = None
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=np.float64)
            if v.shape == (n, d):
                v = v/np.linalg.norm(v, axis=1, keepdims=True)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    bestX = v.copy()
        except Exception:
            pass
    if bestX is None:
        bestX = rng.standard_normal((n,d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        best_val = _max_cos(bestX)
    iu = np.triu_indices(n, 1)
    # initial squeeze
    try:
        if time.time() < DEAD-130:
            Xp, vp = _lp_polish(bestX, seconds=4.0, deadline=HOP_END, iu=iu)
            if vp < best_val:
                best_val, bestX = vp, Xp
    except Exception:
        pass
    sigmas = [0.005, 0.012, 0.03, 0.07, 0.15, 0.008]
    ks = [5, 12, 30, 80, 219, 18]
    trial = 0
    while time.time() < HOP_END and time.time() < DEAD-32:
        trial += 1
        mode = trial % 4
        Xtry = bestX.copy()
        try:
            rm, mcur, G = _rowmax(bestX)
            # contact counts within 2e-4 of max to find jammed points
            Ccnt = np.sum(G > mcur - 2e-4, axis=1)
            order = np.argsort(-(rm*5.0 + Ccnt*0.002 + rng.random(n)*0.001))
            if mode == 0:
                # worst-point gaussian kick + anneal
                cyc = (trial//4) % len(sigmas)
                sigma = sigmas[cyc]; k = ks[cyc]
                sel = order[:k]
                noise = rng.standard_normal((k,d))*sigma
                Xtry[sel] += noise
                Xtry[sel] /= np.linalg.norm(Xtry[sel], axis=1, keepdims=True)
                Xa, va = _anneal(Xtry, steps=750, T0=0.004, T1=0.00035, lr0=0.025)
            elif mode == 1:
                # contact surgery: eject 1-3 most jammed into voids
                nj = 1 + (trial % 3)
                if trial % 8 == 0:
                    nj = 5
                victims = order[:nj].tolist()
                # also include rattler if exists
                ratt = np.where(rm < mcur - 5e-4)[0]
                for r in ratt[:1]:
                    if r not in victims:
                        victims.append(int(r))
                _void_reinsert(Xtry, victims, rng, n_cand=25000)
                Xa, va = _anneal(Xtry, steps=800, T0=0.003, T1=0.0003, lr0=0.022)
            elif mode == 2:
                # large-hole reinsert of random jammed subset
                topk = 10 + (trial % 5)*4
                cands_idx = order[:topk]
                nv = 2 + (trial % 3)
                victims = rng.choice(cands_idx, size=min(nv, len(cands_idx)), replace=False).tolist()
                _void_reinsert(Xtry, victims, rng, n_cand=18000)
                Xa, va = _anneal(Xtry, steps=700, T0=0.005, T1=0.0004, lr0=0.028)
            else:
                # full random subset kick with hotter anneal
                k = 24 + (trial % 4)*16
                sel = np.zeros(n, dtype=bool)
                hw = min(k//2+2, n)
                sel[order[:hw]] = True
                extra = rng.choice(n, size=min(k, n), replace=False)
                cnt = 0
                for e in extra:
                    if not sel[e]:
                        sel[e] = True
                        cnt += 1
                        if np.sum(sel) >= k:
                            break
                sigma = 0.02 if (trial % 8 < 4) else 0.06
                noise = rng.standard_normal((n,d))*sigma
                Xtry[sel] += noise[sel]
                Xtry[sel] /= np.linalg.norm(Xtry[sel], axis=1, keepdims=True)
                Xa, va = _anneal(Xtry, steps=650, T0=0.006, T1=0.00035, lr0=0.03)
            if va < best_val:
                best_val, bestX = va, Xa.copy()
                Xp, vp = _lp_polish(bestX, seconds=2.5, deadline=DEAD-28, iu=iu)
                if vp < best_val:
                    best_val, bestX = vp, Xp
            elif va < best_val + 0.0012 and (trial % 3 == 0):
                Xp, vp = _lp_polish(Xa, seconds=1.0, deadline=DEAD-28, iu=iu)
                if vp < best_val:
                    best_val, bestX = vp, Xp
        except Exception:
            continue
    try:
        left = DEAD - time.time()
        if left > 4:
            Xp, vp = _lp_polish(bestX, seconds=min(left-1.0, 28.0), deadline=DEAD-0.5, iu=iu)
            if vp < best_val:
                best_val, bestX = vp, Xp
    except Exception:
        pass
    bestX = bestX/np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
