import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _rowmax(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _adam(X, steps, beta0, beta1, lr0, lr1, rng=None, noise0=0.0, noise1=0.0):
    n, d = X.shape
    m = np.zeros_like(X); v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lb0, lb1 = np.log(beta0), np.log(beta1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(1, steps+1):
        frac = t / steps
        beta = np.exp(lb0 + (lb1-lb0)*frac)
        lr = np.exp(ll0 + (ll1-ll0)*frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = G.max()
        W = np.exp(beta*(G-gmax))
        np.fill_diagonal(W, 0.0)
        Z = W.sum()
        if Z == 0 or not np.isfinite(Z):
            continue
        Grad = (W @ X) / Z
        dot = (Grad*X).sum(axis=1, keepdims=True)
        Gt = Grad - dot*X
        # optional tangential noise to change topology
        if noise0 > 0 and rng is not None:
            nz = noise0*(noise1/noise0)**frac if noise1>0 else noise0
            if nz > 0:
                N = rng.randn(n, d)
                N -= (N*X).sum(axis=1, keepdims=True)*X
                Gt = Gt + nz * N
        m *= b1; m += (1.0-b1)*Gt
        v *= b2; v += (1.0-b2)*(Gt*Gt)
        mhat = m/(1.0-b1**t); vhat = v/(1.0-b2**t)
        step = mhat/(np.sqrt(vhat)+eps)
        X = X - lr*step
        nrm = np.sqrt((X*X).sum(axis=1, keepdims=True)); nrm[nrm==0]=1.0
        X /= nrm
    return X

def _best_void(placed, rng, n_probe=20000):
    d = placed.shape[1]
    P = rng.randn(n_probe, d)
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    best_i = -1; best_v = 1e9
    CH = 5000
    for a in range(0, n_probe, CH):
        b = min(a+CH, n_probe)
        D = P[a:b] @ placed.T
        mx = D.max(axis=1)
        i = int(np.argmin(mx))
        if float(mx[i]) < best_v:
            best_v = float(mx[i]); best_i = a+i
    p = P[best_i].copy()
    # refine: minimize softmax max-dot
    beta = 80.0
    for it in range(120):
        lr = 0.08 * (0.004/0.08)**(it/119)
        g = placed @ p
        mx = g.max()
        w = np.exp(beta*(g-mx)); w /= (w.sum()+1e-18)
        A = w @ placed
        At = A - float(A @ p)*p
        p = p - lr*At
        nr = np.linalg.norm(p)
        if nr == 0 or not np.isfinite(nr):
            break
        p /= nr
    return p

def _lp_polish(x, seconds, r0=1.1e-3):
    x = x/np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
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
            radius *= 0.5
            continue
        if p > 2500:
            idx = np.argsort(c[sel])[-2500:]
            I, J = I[idx], J[idx]
            cc = c[sel][idx]
        else:
            cc = c[sel]
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)],1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
        try:
            A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n),d), np.arange(n*d))), shape=(n,nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
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
    rng = np.random.RandomState(777)
    # quick LP on starting point is cheap and may shave a bit
    # diverse global trials with hub ejection + rejam
    trial = 0
    # reserve last ~22s for final polish
    while time.time() - t0 < BUDGET - 24.0:
        trial += 1
        if time.time() - t0 > 148.0:
            break
        X = cur.copy()
        # large perturbation: gaussian shake to leave basin
        sig = [0.02, 0.05, 0.09, 0.14, 0.20][rng.randint(5)]
        X = X + sig * rng.randn(n,d)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        # stage A: moderate-beta jam (spreads load, not just top pair)
        try:
            b0 = 18.0 + rng.rand()*30.0
            b1 = 180.0 + rng.rand()*420.0
            lr0 = 0.004 + rng.rand()*0.004
            X = _adam(X, steps=1400, beta0=b0, beta1=b1, lr0=lr0, lr1=lr0*0.1,
                      rng=rng, noise0=0.15, noise1=0.0)
        except Exception:
            continue
        # hub ejection rounds: move worst rowmax points into voids
        try:
            for rnd in range(2):
                if time.time()-t0 > BUDGET-26.0:
                    break
                rm = _rowmax(X)
                k = 8 + int(rng.randint(8))  # 8..15 hubs
                hubs = np.argsort(rm)[-k:]
                # keep core fixed conceptually: find voids w.r.t. non-hub points
                mask = np.ones(n, dtype=bool); mask[hubs] = False
                placed = X[mask].copy()
                prng = np.random.RandomState(9000+trial*1000+rnd*77)
                order = hubs[prng.permutation(len(hubs))]
                for h in order:
                    # void against all currently placed (including reinserted)
                    p = _best_void(placed, prng, n_probe=20000)
                    X[int(h)] = p
                    placed = np.vstack([placed, p[None,:]])
                # short re-jam after surgery
                b0b = 25.0 + prng.rand()*25.0
                b1b = 250.0 + prng.rand()*350.0
                X = _adam(X, steps=800, beta0=b0b, beta1=b1b, lr0=0.004, lr1=0.0004,
                          rng=None, noise0=0.0, noise1=0.0)
        except Exception:
            continue
        # stage B: high-beta quench to squeeze max
        try:
            b0c = 40.0 + rng.rand()*40.0
            b1c = 400.0 + rng.rand()*600.0
            X = _adam(X, steps=900, beta0=b0c, beta1=b1c, lr0=0.0035, lr1=0.0003,
                      rng=None, noise0=0.0, noise1=0.0)
        except Exception:
            continue
        try:
            v = _max_cos(X)
        except Exception:
            continue
        if v < cur_val and np.all(np.isfinite(X)):
            cur_val = v; cur = X.copy()
            if v < best_val:
                best_val = v; best = X.copy()
        # also accept near-miss with different ordering to diversify basin?
        # if stuck, restart from best with bigger shake occasionally
        if trial % 6 == 0 and best is not None:
            # inject best as restart seed with huge shake next trial
            cur = best.copy()
    # final polish on best found
    try:
        left = BUDGET - (time.time()-t0) - 1.0
        left = max(6.0, min(22.0, left))
        best = _adam(best, steps=1200, beta0=50.0, beta1=700.0, lr0=0.0025, lr1=0.00025)
        best = _lp_polish(best, seconds=left)
        best_val = _max_cos(best)
    except Exception:
        pass
    best = best/np.linalg.norm(best, axis=1, keepdims=True)
    return best
