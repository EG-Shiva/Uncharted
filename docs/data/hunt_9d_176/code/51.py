import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _build_base_pool():
    vecs = []
    # type (±1,±1,0^7)
    for i in range(9):
        for j in range(i+1, 9):
            for s1 in (-1.0, 1.0):
                for s2 in (-1.0, 1.0):
                    v = np.zeros(9)
                    v[i] = s1; v[j] = s2
                    vecs.append(v / np.sqrt(2.0))
    # type (±2,±1,0^7)/sqrt5
    for i in range(9):
        for j in range(9):
            if i == j: continue
            for s1 in (-1.0, 1.0):
                for s2 in (-1.0, 1.0):
                    v = np.zeros(9)
                    v[i] = 2.0*s1; v[j] = 1.0*s2
                    vecs.append(v / np.sqrt(5.0))
    # type (±1,±1,±1,0^6)/sqrt3
    for i in range(9):
        for j in range(i+1, 9):
            for k in range(j+1, 9):
                for a in range(8):
                    s1 = 1.0 if (a & 1) else -1.0
                    s2 = 1.0 if (a & 2) else -1.0
                    s3 = 1.0 if (a & 4) else -1.0
                    v = np.zeros(9)
                    v[i]=s1; v[j]=s2; v[k]=s3
                    vecs.append(v / np.sqrt(3.0))
    # E8 x R : E8 roots in first 8 coords + ninth in {-1,0,1}
    e8 = []
    for i in range(8):
        for j in range(i+1, 8):
            for s1 in (-1.0, 1.0):
                for s2 in (-1.0, 1.0):
                    v = np.zeros(8); v[i]=s1; v[j]=s2
                    e8.append(v)
    for bits in range(256):
        nneg = bin(bits).count("1")
        if nneg % 2 == 1:
            continue
        v = np.ones(8)*0.5
        for b in range(8):
            if (bits >> b) & 1:
                v[b] = -0.5
        e8.append(v)
    # e8 should be 112+128=240
    for r in e8:
        for c in (-1.0, 0.0, 1.0):
            v9 = np.zeros(9); v9[:8]=r; v9[8]=c
            n = np.linalg.norm(v9)
            vecs.append(v9/n)
    P = np.stack(vecs, axis=0)
    return P

def _rand_Q(rng):
    A = rng.standard_normal((9,9))
    Q, R = np.linalg.qr(A)
    d = np.sign(np.diag(R))
    d[d==0]=1.0
    Q = Q * d[None,:]
    return Q

def _greedy_select(P, rng):
    N = P.shape[0]
    s0 = int(rng.integers(N))
    best_dot = P @ P[s0]
    best_dot[s0] = 1e9
    sel = [s0]
    selset = np.zeros(N, dtype=bool); selset[s0]=True
    for _ in range(1, 176):
        idx = int(np.argmin(best_dot))
        sel.append(idx); selset[idx]=True
        d = P @ P[idx]
        # max update only for unselected
        np.maximum(best_dot, d, out=best_dot)
        best_dot[idx]=1e9
        # keep selected at inf
        # (already)
    X = P[np.array(sel)]
    # renormalize (already unit)
    return X / np.linalg.norm(X, axis=1, keepdims=True)

def _quench(X, steps, beta0, beta1, lr0, lr1, deadline):
    n, d = X.shape
    m = np.zeros_like(X); vv = np.zeros_like(X)
    b1,b2,eps = 0.9,0.999,1e-8
    for t in range(steps):
        if (t % 50)==0 and time.time() > deadline:
            break
        frac = t / max(1, steps-1)
        beta = beta0 * ((beta1/beta0) ** frac)
        lr = lr0 * ((lr1/lr0) ** frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        E = np.exp(beta*(G-M))
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E)*0.5)
        if not np.isfinite(S) or S<=0:
            continue
        Grad = (E @ X) / S
        dot = np.sum(Grad*X, axis=1, keepdims=True)
        Gt = Grad - dot*X
        m = b1*m + (1-b1)*Gt
        vv = b2*vv + (1-b2)*(Gt*Gt)
        mhat = m/(1-b1**(t+1))
        vhat = vv/(1-b2**(t+1))
        step = mhat/(np.sqrt(vhat)+eps)
        np.clip(step, -3.0, 3.0, out=step)
        X = X - lr*step
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr==0]=1.0
        X = X/nr
        if not np.all(np.isfinite(X)):
            break
    nr = np.linalg.norm(X, axis=1, keepdims=True); nr[nr==0]=1.0
    return X/nr

def _lp_polish_deadline(x, deadline, r_init=0.003, cap=2200):
    x = x/np.linalg.norm(x,axis=1,keepdims=True)
    n,d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = r_init
    iu0 = np.triu_indices(n,1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = m - max(20.0*radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt==0:
            break
        if cnt > cap:
            idx_all = np.where(sel)[0]
            order = np.argpartition(-c_all[idx_all], cap)[:cap]
            keep = np.zeros_like(sel); keep[idx_all[order]]=True
            sel = keep
        I = iu0[0][sel]; J = iu0[1][sel]; c = c_all[sel]
        p = len(I); nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1),n*d)],1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
        try:
            A_ub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        except Exception:
            radius*=0.5; continue
        obj = np.zeros(nv); obj[-1]=1.0
        try:
            res = linprog(obj,A_ub=A_ub,b_ub=-c,A_eq=A_eq,b_eq=np.zeros(n),bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        y = x + res.x[:-1].reshape(n,d)
        yn = np.linalg.norm(y,axis=1,keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn<1e-12):
            radius*=0.5; continue
        y = y/yn
        v = _max_cos(y)
        if v < best:
            x,best = y,v
            radius = min(radius*1.5,0.03)
        else:
            radius*=0.5
    return x

def solve():
    t0 = time.time()
    SEARCH_END = t0 + 150.0
    HARD_END = t0 + 176.0
    rng = np.random.default_rng(20261103)
    n,d = 176,9
    base = None
    base_v = 1e9
    for fn in ("best.npy","record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a,dtype=float)
            if a.shape==(n,d) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a,axis=1,keepdims=True)
                if np.all(nr>1e-12):
                    a=a/nr
                    v=_max_cos(a)
                    if v<base_v:
                        base_v=v; base=a.copy()
        except Exception:
            pass
    if base is None:
        base = rng.standard_normal((n,d)); base/=np.linalg.norm(base,axis=1,keepdims=True)
        base_v=_max_cos(base)
    best_X = base.copy(); best_v = float(base_v)

    try:
        P0 = _build_base_pool()
    except Exception:
        P0 = None

    trial=0
    # alternate lattice-pool and pure random starts
    while time.time() < SEARCH_END:
        trial+=1
        # time check already
        is_lattice = (trial % 3 != 0) and (P0 is not None)  # 2/3 lattice, 1/3 random
        try:
            if is_lattice:
                Q1 = _rand_Q(rng); Q2 = _rand_Q(rng)
                P1 = P0 @ Q1.T
                P2 = P0 @ Q2.T
                Pool = np.vstack([P0, P1, P2])
                X0 = _greedy_select(Pool, rng)
            else:
                X0 = rng.standard_normal((n,d))
                X0 /= np.linalg.norm(X0,axis=1,keepdims=True)
        except Exception:
            continue
        # randomize schedule slightly
        steps = int(rng.integers(1200, 2200))
        beta0 = 20.0
        beta1 = float(rng.uniform(140.0, 190.0))
        lr0 = float(rng.uniform(0.006, 0.012))
        lr1 = float(rng.uniform(0.0012, 0.003))
        deadline = min(SEARCH_END, HARD_END-18.0)
        try:
            X1 = _quench(X0, steps, beta0, beta1, lr0, lr1, deadline)
            v = _max_cos(X1)
        except Exception:
            continue
        if np.isfinite(v) and v < best_v:
            best_v=float(v); best_X=X1.copy()
            # early exit if big win? continue to use time
        if time.time() >= SEARCH_END:
            break
    # if nothing beats baseline, return baseline unchanged (safety fallback)
    if best_v >= base_v - 1e-12:
        return base / np.linalg.norm(base,axis=1,keepdims=True)
    # one final LP polish <=15s
    try:
        fin_dead = min(HARD_END-1.0, time.time()+13.0)
        if time.time() < fin_dead:
            best_X = _lp_polish_deadline(best_X, fin_dead, r_init=0.0025, cap=2200)
    except Exception:
        pass
    best_X = best_X/np.linalg.norm(best_X,axis=1,keepdims=True)
    # safety: if polish worsened, revert
    try:
        v2 = _max_cos(best_X)
        if not np.isfinite(v2) or v2 > best_v + 1e-12:
            # revert to best known (base or pre-polish?) return base if best was pre-polish?
            # we kept best_X overwritten; if worsened vs base, return base
            if best_v >= base_v - 1e-12:
                return base
        if v2 > base_v and base_v < 1e9:
            return base
    except Exception:
        return base
    return best_X
