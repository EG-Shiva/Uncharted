import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, seconds, deadline=None):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if deadline is not None and time.time() >= deadline:
            break
        if radius < 1e-14:
            break
        G = X @ X.T
        c = G[iu0]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        if p == 0:
            break
        # cap constraints for speed
        if p > 3000:
            # keep most-tight p
            idx = np.argpartition(-c[sel], 3000)[:3000]
            I = I[idx]; J = J[idx]
            csel = c[sel][idx]
            p = 3000
        else:
            csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None]*d+np.arange(d), J[:, None]*d+np.arange(d), np.full((p,1), n*d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p,1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
            A_eq = coo_matrix((X.ravel(),(np.repeat(np.arange(n),d), np.arange(n*d))), shape=(n,nv)).tocsr()
            obj = np.zeros(nv); obj[-1]=1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n,d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            X, best, radius = y, v, min(radius*1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _smooth(X, steps, k0=80.0, k1=420.0, lr0=0.02, lr1=0.002):
    n, d = X.shape
    for s in range(steps):
        frac = s/max(steps-1,1)
        k = k0+(k1-k0)*frac
        lr = lr0+(lr1-lr0)*frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(k*(G-gmax))
        np.fill_diagonal(W, 0.0)
        Gr = W @ X
        dot = np.einsum('ij,ij->i', Gr, X)[:,None]
        Gt = Gr - dot*X
        nm = np.linalg.norm(Gt, axis=1).max()
        if not np.isfinite(nm) or nm < 1e-12:
            break
        X = X - (lr/nm)*Gt
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _void(Xkeep, rng, n_cand=3000):
    d = Xkeep.shape[1]
    C = rng.standard_normal((n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    S = C @ Xkeep.T
    m = S.max(axis=1)
    v = C[int(np.argmin(m))].copy()
    # quick repel refine
    for _ in range(25):
        dv = Xkeep @ v
        mm = float(dv.max())
        w = np.exp(np.clip((dv-mm)/0.02, -50, 0))
        g = w @ Xkeep
        g = g - np.dot(g, v)*v
        ng = np.linalg.norm(g)
        if ng < 1e-12:
            break
        g /= ng
        v = v - 0.02*g
        v /= np.linalg.norm(v)
    return v

def solve():
    rng = np.random.default_rng(7)
    t0 = time.time()
    TDEAD = t0 + 171.0
    n, d = 181, 9
    cands = []
    for fn in ("best.npy","record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=float)
            if a.shape == (n,d):
                a /= np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal((n,d)); a/=np.linalg.norm(a,axis=1,keepdims=True)
        cands.append(a)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    Xb = cands[bi].copy()
    best = float(vals[bi])
    # short initial squeeze if time allows (in case best.npy not polished)
    # main basin hopping
    it = 0
    while True:
        if time.time() > TDEAD - 22.0:
            break
        if it > 120:
            break
        mode = it % 5
        X = Xb.copy()
        if mode == 0:
            sig = 0.012 if it < 40 else 0.007
            X = X + sig*rng.standard_normal(X.shape)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            X = _smooth(X, steps=260)
        elif mode == 1:
            G = X @ X.T; np.fill_diagonal(G,-2.0)
            per = G.max(axis=1)
            victim = int(np.argmax(per))
            keep = [i for i in range(n) if i!=victim]
            X[victim] = _void(X[keep], rng, n_cand=3000)
            X += 0.003*rng.standard_normal(X.shape)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            X = _smooth(X, steps=340)
        elif mode == 2:
            G = X @ X.T; np.fill_diagonal(G,-2.0)
            gmax = float(G.max())
            cnt = (G > gmax-0.015).sum(axis=1)
            # 3 most overloaded distinct
            order = np.argsort(-cnt - 0.001*np.arange(n)[::-1]*0)
            # deterministic tie break by index
            victims = order[:3].tolist()
            Xkeep_base = np.delete(X, victims, axis=0)
            placed = []
            for vv in victims:
                ref = Xkeep_base if not placed else np.vstack([Xkeep_base]+placed)
                nv_ = _void(ref, rng, n_cand=2500)
                placed.append(nv_[None,:])
            for k_, vv in enumerate(victims):
                X[vv] = placed[k_][0]
            X += 0.002*rng.standard_normal(X.shape)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            X = _smooth(X, steps=380, k0=100.0, k1=500.0)
        elif mode == 3:
            G = X @ X.T; np.fill_diagonal(G,-2.0)
            gmax = float(G.max())
            cnt = (G > gmax-0.02).sum(axis=1)
            center = int(np.argmax(cnt))
            order = np.argsort(-G[center])
            kne = 12
            neigh = order[:kne]
            X[neigh] += 0.055*rng.standard_normal((kne,d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            X = _smooth(X, steps=280)
        else:
            X = X + 0.032*rng.standard_normal(X.shape)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            X = _smooth(X, steps=380, k0=60.0, k1=350.0, lr0=0.025, lr1=0.002)
        v = _max_cos(X)
        if v < best + 0.010:
            if time.time() > TDEAD - 21.0:
                if v < best:
                    Xb, best = X, v
                it += 1
                continue
            X2 = _lp_polish(X, seconds=0.8, deadline=TDEAD-19.0)
            v2 = _max_cos(X2)
            if v2 < best:
                Xb, best = X2, v2
            elif v < best:
                Xb, best = X, v
        else:
            if v < best:
                Xb, best = X, v
        it += 1
    # final polish
    rem = TDEAD - time.time() - 0.5
    if rem > 2.0:
        Xb = _lp_polish(Xb, seconds=min(14.0, rem-1.0), deadline=TDEAD-0.3)
        if TDEAD - time.time() > 4.0:
            Xb = _smooth(Xb, steps=250, k0=300.0, k1=600.0, lr0=0.005, lr1=0.0005)
            Xb = _lp_polish(Xb, seconds=max(1.0, TDEAD-time.time()-0.2), deadline=TDEAD-0.1)
    Xb /= np.linalg.norm(Xb, axis=1, keepdims=True)
    if not np.all(np.isfinite(Xb)):
        Xb = cands[bi]
    return Xb
