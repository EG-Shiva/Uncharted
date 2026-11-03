import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _per_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return np.max(g, axis=1), float(np.max(g))

def _lp_polish(x, seconds, deadline, iu=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    if iu is None:
        iu = np.triu_indices(n, 1)
    best = _max_cos(x)
    bestX = x.copy()
    radius = 1e-3
    start = time.time()
    nv = n * d + 1
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-14:
            break
        G = x @ x.T
        c = G[iu]
        m = float(np.max(c))
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        I = iu[0][sel]; J = iu[1][sel]; csel = c[sel]
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
    G0 = cur @ cur.T
    np.fill_diagonal(G0, -2.0)
    m0 = float(np.max(G0))
    best = m0
    bestX = cur.copy()
    T = T0
    dec = (T1/T0)**(1.0/max(1,steps-1)) if steps>1 else 1.0
    lr = lr0
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
        cur = cur - (lr/gn)*Gt
        cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        T *= dec
    v = _max_cos(cur)
    if v < best:
        best = v
        bestX = cur.copy()
    return bestX, best

def _hole_improve(X, rng, n_try, topk):
    n, d = X.shape
    for _ in range(2):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        rowmax = np.max(G, axis=1)
        order = np.argsort(-rowmax)
        anyc = False
        for idx in order[:topk]:
            cur_v = float(rowmax[idx])
            C = rng.standard_normal((n_try, d))
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            D = C @ X.T
            D[:, idx] = -2.0
            cm = np.max(D, axis=1)
            j = int(np.argmin(cm))
            if float(cm[j]) < cur_v - 1e-12:
                X[idx] = C[j]
                anyc = True
                # quick single-point tangential refine
                for __ in range(60):
                    dots = X[idx] @ X.T
                    dots[idx] = -2.0
                    mm = float(np.max(dots))
                    w = np.exp((dots - mm)/0.0015)
                    w[idx] = 0.0
                    w[dots < mm - 0.012] = 0.0
                    g = w @ X
                    g = g - np.dot(g, X[idx])*X[idx]
                    nn = float(np.linalg.norm(g))
                    if nn < 1e-14 or not np.isfinite(nn):
                        break
                    st = 0.02 / (1.0 + __*0.05)
                    nx = X[idx] - (st/nn)*g
                    nx /= np.linalg.norm(nx)
                    nd = nx @ X.T
                    nd[idx] = -2.0
                    nm = float(np.max(nd))
                    if nm < mm:
                        X[idx] = nx
                    else:
                        break
        if not anyc:
            break
    return X

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    rng = np.random.default_rng(12345)
    best_val = 1e9
    bestX = None
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=np.float64)
            if v.shape == (219, 6):
                v = v/np.linalg.norm(v, axis=1, keepdims=True)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    bestX = v.copy()
        except Exception:
            pass
    if bestX is None:
        bestX = rng.standard_normal((219,6))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        best_val = _max_cos(bestX)
    n, d = bestX.shape
    iu = np.triu_indices(n,1)
    # initial hole + anneal
    try:
        Xt = bestX.copy()
        _hole_improve(Xt, rng, n_try=6000, topk=15)
        v = _max_cos(Xt)
        Xa, va = _anneal(Xt, steps=900, T0=0.004, T1=0.0004, lr0=0.025)
        if va < best_val:
            best_val, bestX = va, Xa
        elif v < best_val:
            best_val, bestX = v, Xt
    except Exception:
        pass
    # short squeeze
    try:
        if time.time() < DEAD-120:
            Xp, vp = _lp_polish(bestX, seconds=4.0, deadline=DEAD-115, iu=iu)
            if vp < best_val:
                best_val, bestX = vp, Xp
    except Exception:
        pass
    sigmas = [0.004, 0.008, 0.015, 0.03, 0.06, 0.12]
    ks = [6, 12, 24, 48, 90, 219]
    T0s = [0.005, 0.004, 0.003, 0.005, 0.006, 0.008]
    trial = 0
    HOP_END = t0 + 135.0
    while time.time() < HOP_END and time.time() < DEAD-35:
        trial += 1
        cyc = trial % 6
        sigma = sigmas[cyc]
        k = ks[cyc]
        T0c = T0s[cyc]
        Xtry = bestX.copy()
        try:
            if trial % 5 == 0:
                _hole_improve(Xtry, rng, n_try=4000, topk=12)
                Xa, va = _anneal(Xtry, steps=700, T0=0.003, T1=0.0003, lr0=0.02)
            else:
                _, pm = _per_max(bestX)
                # reuse rowmax ordering: compute again
                G = bestX @ bestX.T
                np.fill_diagonal(G, -2.0)
                rowmax = np.max(G, axis=1)
                order = np.argsort(-rowmax)
                sel = np.zeros(n, dtype=bool)
                hw = min(k//2+2, n)
                sel[order[:hw]] = True
                rem = k - hw
                if rem > 0:
                    cand = rng.choice(n, size=min(rem*2, n), replace=False)
                    cnt = 0
                    for cc in cand:
                        if not sel[cc]:
                            sel[cc] = True
                            cnt += 1
                            if cnt >= rem:
                                break
                noise = rng.standard_normal((n,d))*sigma
                Xtry[sel] += noise[sel]
                Xtry[sel] /= np.linalg.norm(Xtry[sel], axis=1, keepdims=True)
                Xa, va = _anneal(Xtry, steps=800, T0=T0c, T1=0.00035, lr0=0.025)
            if va < best_val:
                best_val, bestX = va, Xa.copy()
                Xp, vp = _lp_polish(bestX, seconds=2.5, deadline=DEAD-30, iu=iu)
                if vp < best_val:
                    best_val, bestX = vp, Xp
            elif va < best_val + 0.0015 and (trial % 3 == 0):
                Xp, vp = _lp_polish(Xa, seconds=1.2, deadline=DEAD-30, iu=iu)
                if vp < best_val:
                    best_val, bestX = vp, Xp
        except Exception:
            continue
    # final squeeze with remaining time
    try:
        left = DEAD - time.time()
        if left > 3:
            Xp, vp = _lp_polish(bestX, seconds=min(left-1.0, 28.0), deadline=DEAD-0.5, iu=iu)
            if vp < best_val:
                best_val, bestX = vp, Xp
    except Exception:
        pass
    bestX = bestX/np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
