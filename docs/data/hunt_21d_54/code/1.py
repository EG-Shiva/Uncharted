import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def proj_unit(x):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    # guard nan
    x = np.nan_to_num(x)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n==0]=1
    return x/n

def lp_polish(x, seconds, verbose=False):
    x = proj_unit(x)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = c.max()
        sel = c > m - max(20*radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n*d+1
        # cap constraints for speed
        if p > 3000:
            # keep tightest 3000
            idx = np.argpartition(-c[sel], 3000)[:3000]
            I, J = I[idx], J[idx]
            p = 3000
            csv = c[sel][idx]
        else:
            csv = c[sel]
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)],1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
        A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n),d), np.arange(n*d))), shape=(n,nv)).tocsr()
        obj = np.zeros(nv); obj[-1]=1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs",
                          options={"time_limit": 5.0})
        except Exception:
            radius*=0.5
            continue
        if res.status!=0:
            radius*=0.5
            continue
        y = x + res.x[:-1].reshape(n,d)
        y = proj_unit(y)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius*1.5, 0.05)
        else:
            radius*=0.5
    return x

def contact_counts(x, thresh=1e-9):
    G = x@x.T
    np.fill_diagonal(G,-2)
    m = G.max()
    C = (G > m-thresh)
    return C.sum(axis=1), m, G

def shake_and_repair(x, rng, strength, n_kick=None):
    n,d = x.shape
    cnts, m, G = contact_counts(x)
    y = x.copy()
    # choose points to kick: lowest contacts + highest contacts + random
    order_low = np.argsort(cnts)
    order_high = np.argsort(-cnts)
    k = n_kick if n_kick is not None else 8
    # take mix
    targets = set()
    targets.update(order_low[:k//3+1].tolist())
    targets.update(order_high[:k//3+1].tolist())
    while len(targets)<k:
        targets.add(int(rng.integers(0,n)))
    targets = list(targets)
    for i in targets:
        # tangent noise
        v = rng.normal(size=d)
        v -= (v @ y[i])*y[i]
        nv = np.linalg.norm(v)
        if nv<1e-12: continue
        v/=nv
        ang = strength * (0.5+rng.random())  # radians approx
        # rotate: y[i]*cos + v*sin
        y[i] = np.cos(ang)*y[i] + np.sin(ang)*v
    # also global small jitter
    noise = rng.normal(size=(n,d))* (strength*0.2)
    # make tangent
    noise -= (noise*y).sum(axis=1,keepdims=True)*y
    y = y+noise
    y = proj_unit(y)
    return y

def repulsion_smooth(x, steps, lr, rng=None, repulsion_pow=1.0):
    # minimize soft-max / logsumexp of inner products to spread
    n,d = x.shape
    y = x.copy()
    for s in range(steps):
        G = y @ y.T
        np.fill_diagonal(G, -10)
        m = G.max()
        # weights focused on near-max: exp((G-m)/T)
        T = max(0.005, 0.02*(1-s/steps))
        W = np.exp((G-m)/T)
        np.fill_diagonal(W, 0)
        # gradient: sum_j W_ij * y_j projected tangent
        grad = W @ y
        # tangent projection
        grad -= (grad*y).sum(axis=1,keepdims=True)*y
        gn = np.linalg.norm(grad,axis=1,keepdims=True)+1e-12
        grad/=gn
        # adaptive step
        step = lr*(1-0.5*s/steps)
        y = y - step*grad
        y = proj_unit(y)
    return y

def solve():
    import time as _t
    T0=_t.time()
    LIMIT=170
    rng = np.random.default_rng(12345)
    try:
        best0 = np.load("best.npy")
    except Exception:
        best0 = np.load("record.npy")
    best0 = proj_unit(best0)
    best_val = max_cos(best0)
    best = best0.copy()
    # quick polish remaining budget? first polish baseline briefly
    # Spend time in basin-hopping
    # Use multiple restarts with increasing shake
    # Keep time for final polish
    elapsed = lambda: _t.time()-T0
    # initial fast LP to ensure at optimum
    if elapsed()<LIMIT-10:
        best = lp_polish(best, min(8, LIMIT-elapsed()-60))
        best_val = max_cos(best)
    # basin hopping loop
    it=0
    strengths = [0.02, 0.05, 0.10, 0.20, 0.35, 0.03, 0.07, 0.15]
    while elapsed() < LIMIT-25:
        it+=1
        s = strengths[(it-1)%len(strengths)]
        nk = int(rng.integers(3,12))
        cand = shake_and_repair(best, rng, strength=s, n_kick=nk)
        # short repulsion to separate kicked points
        cand = repulsion_smooth(cand, steps=80, lr=0.02)
        # short LP polish
        budget = min(6, (LIMIT-25-elapsed())/2)
        if budget<1: break
        cand = lp_polish(cand, budget)
        v = max_cos(cand)
        if v < best_val - 1e-12:
            best, best_val = cand, v
            # print improvement
        # also try repulsion-only from best with different seed occasionally
        if it%6==0 and elapsed()<LIMIT-30:
            cand2 = repulsion_smooth(best + rng.normal(size=best.shape)*0.01, steps=150, lr=0.03)
            cand2 = proj_unit(cand2)
            cand2 = lp_polish(cand2, min(5, LIMIT-25-elapsed()))
            v2 = max_cos(cand2)
            if v2 < best_val-1e-12:
                best, best_val = cand2, v2
    # final intensive polish
    if elapsed()<LIMIT-2:
        best = lp_polish(best, LIMIT-elapsed()-0.5)
        best_val = max_cos(best)
    best = proj_unit(best)
    return best
