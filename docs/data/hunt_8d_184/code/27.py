import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _per_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1)

def _E8():
    pts = []
    # D8: (+-1,+-1,0...)
    for i in range(8):
        for j in range(i+1, 8):
            for s1 in (-1.0, 1.0):
                for s2 in (-1.0, 1.0):
                    v = np.zeros(8)
                    v[i] = s1; v[j] = s2
                    pts.append(v)
    # spinor: (+-0.5)^8 even minus
    for mask in range(256):
        bits = [(mask >> k) & 1 for k in range(8)]
        if sum(bits) % 2 == 1:
            continue
        v = np.where(np.array(bits) == 0, 0.5, -0.5)
        pts.append(v.astype(float))
    A = np.stack(pts, axis=0)  # 240x8
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    return A

def _smooth(x, steps, a0, a1, lr0, lr1):
    la0, la1 = np.log(a0), np.log(a1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(steps-1, 1)
        alpha = np.exp(la0 + (la1-la0)*f)
        lr = np.exp(ll0 + (ll1-ll0)*f)
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        mx = np.max(G)
        W = np.exp(alpha*(G-mx))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-30
        grad = W @ x
        dot = np.sum(grad*x, axis=1, keepdims=True)
        gt = grad - dot*x
        x = x - lr*(gt/S)
        x = _norm(x)
    return x

def _greedy(x, steps, m0, m1, lr):
    for t in range(steps):
        f = t / max(steps-1, 1)
        margin = m0 + (m1-m0)*f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = np.max(G)
        M = (G > (m-margin)).astype(np.float64)
        deg = M.sum(axis=1, keepdims=True)
        grad = M @ x
        dot = np.sum(grad*x, axis=1, keepdims=True)
        gt = grad - dot*x
        x = x - lr*(gt/np.maximum(deg, 1.0))
        x = _norm(x)
    return x

def _hole_sample(xfix, rng, n_sample=4000):
    d = xfix.shape[1]
    Y = rng.standard_normal(size=(n_sample, d))
    Y = _norm(Y)
    S = Y @ xfix.T
    m = S.max(axis=1)
    idx = int(np.argmin(m))
    return Y[idx], float(m[idx])

def _refine_hole(xfix, y, steps=80, a0=60.0, a1=300.0, lr0=0.06, lr1=0.005):
    la0, la1 = np.log(a0), np.log(a1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(steps-1, 1)
        alpha = np.exp(la0+(la1-la0)*f)
        lr = np.exp(ll0+(ll1-ll0)*f)
        s = xfix @ y
        mx = np.max(s)
        w = np.exp(alpha*(s-mx))
        sw = np.sum(w)+1e-30
        g = w @ xfix
        gt = g - np.dot(g, y)*y
        y = y - lr*gt/sw
        nrm = np.linalg.norm(y)
        if not np.isfinite(nrm) or nrm == 0:
            break
        y = y/nrm
    return y

def _lp_polish_cap(x, seconds, deadline, pcap=1600):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = 8e-4
    t0 = time.time()
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time()-t0 < seconds and time.time() < deadline:
        if radius < 1e-13:
            break
        G = x @ x.T
        cf = G[iu0]
        m = float(cf.max())
        thr = m - max(20.0*radius, 1e-9)
        sel = cf > thr
        I = iu0[0][sel]; J = iu0[1][sel]; cs = cf[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > pcap:
            idx = np.argpartition(-cs, pcap)[:pcap]
            I, J, cs = I[idx], J[idx], cs[idx]
            p = pcap
        nv = n*d+1
        rows = np.concatenate([np.repeat(np.arange(p), d), np.repeat(np.arange(p), d), np.arange(p)])
        cols = np.concatenate([np.repeat(I*d, d)+np.tile(ar, p), np.repeat(J*d, d)+np.tile(ar, p), np.full(p, n*d)])
        vals = np.concatenate([x[J].ravel(), x[I].ravel(), -np.ones(p)])
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm(x + res.x[:-1].reshape(n, d))
        v = _max_cos(y)
        if v < best:
            best = v; xb = y.copy(); x = y
            radius = min(radius*1.5, 0.02)
        else:
            radius *= 0.5
            if radius < 5e-7:
                break
    return xb

def solve():
    t0 = time.time()
    DEADLINE = t0 + 174.0
    rng = np.random.default_rng(1)
    n, d = 184, 8
    try:
        xb = np.asarray(np.load("best.npy"), dtype=np.float64)
    except Exception:
        xb = np.asarray(np.load("record.npy"), dtype=np.float64)
    if xb.shape != (n, d):
        xb = xb.reshape(n, d)
    xb = _norm(xb)
    gbest = xb.copy()
    gval = _max_cos(gbest)
    try:
        xr = _norm(np.asarray(np.load("record.npy"), dtype=np.float64).reshape(n, d))
        _ = _max_cos(xr)
    except Exception:
        xr = xb
    E8 = _E8()
    # quick contact stats for surgery
    def freq_deg(base, mcur, thr_off=0.005):
        G = base @ base.T
        np.fill_diagonal(G, -2.0)
        iu = np.triu_indices(n, 1)
        cf = G[iu]
        mask = cf >= (mcur - thr_off)
        freq = np.zeros(n, dtype=np.int64)
        if np.any(mask):
            np.add.at(freq, iu[0][mask], 1)
            np.add.at(freq, iu[1][mask], 1)
        return freq, G
    trial = 0
    # reserve ~12s for final squeeze
    while time.time() < DEADLINE - 14.0:
        trial += 1
        if time.time() > DEADLINE - 14.0:
            break
        # choose base: mostly gbest, sometimes perturb wandering is implicit
        base = gbest
        G0 = base @ base.T
        np.fill_diagonal(G0, -2.0)
        mcur = float(G0.max())
        pm = np.max(G0, axis=1)
        freq, _ = freq_deg(base, mcur, 0.005)
        order_hi = np.argsort(-freq, kind="stable")
        order_lo = np.argsort(pm, kind="stable")  # smallest per-max = rattlers/holes
        xt = base.copy()
        mode = trial % 10
        if trial <= 3:
            # E8 restart basin
            idx = rng.choice(240, size=n, replace=False)
            xt = E8[idx].copy() + rng.standard_normal(size=(n, d))*0.015
            xt = _norm(xt)
            xt = _smooth(xt, steps=220, a0=45.0, a1=320.0, lr0=0.030, lr1=0.0025)
            xt = _greedy(xt, steps=100, m0=0.007, m1=0.0008, lr=0.007)
            v = _max_cos(xt)
            if v < gval:
                gval = v; gbest = xt.copy()
            continue
        elif mode in (0, 5):
            # transplant low-contact points into largest holes
            k = 4 if mode == 0 else 3
            victims = order_lo[:k].tolist()
            keep = [i for i in range(n) if i not in set(victims)]
            xfix = xt[keep].copy()
            for li in victims:
                y0, _ = _hole_sample(xfix, rng, n_sample=4000)
                y0 = _refine_hole(xfix, y0, steps=70)
                xt[li] = y0
                xfix = np.vstack([xfix, y0[None, :]])
            # tiny jitter backbone
            xt[keep] = _norm(xt[keep] + rng.standard_normal(size=(len(keep), d))*0.003)
            xt = _smooth(xt, steps=240, a0=40.0, a1=340.0, lr0=0.032, lr1=0.0025)
            xt = _greedy(xt, steps=110, m0=0.008, m1=0.0007, lr=0.008)
        elif mode in (1, 6):
            # kick overloaded hubs hard
            k = 7 if mode == 1 else 5
            victims = order_hi[:k]
            xt[victims] = _norm(xt[victims] + rng.standard_normal(size=(k, d))*0.075)
            xt = _smooth(xt, steps=200, a0=45.0, a1=300.0, lr0=0.028, lr1=0.0025)
            xt = _greedy(xt, steps=90, m0=0.007, m1=0.0008, lr=0.007)
        elif mode in (2, 7):
            # block shake around random hub: nearest neighbours
            c = int(order_hi[int(rng.integers(0, 6))])
            Gr = xt @ xt.T
            np.fill_diagonal(Gr, -2.0)
            nearest = np.argsort(-Gr[c])[:12]
            xt[nearest] = _norm(xt[nearest] + rng.standard_normal(size=(12, d))*0.060)
            # also relocate single worst rattler to hole
            li = int(order_lo[0])
            keep = [i for i in range(n) if i != li]
            xfix = xt[keep]
            y0, _ = _hole_sample(xfix, rng, n_sample=3000)
            y0 = _refine_hole(xfix, y0, steps=60)
            xt[li] = y0
            xt = _smooth(xt, steps=210, a0=42.0, a1=320.0, lr0=0.030, lr1=0.0025)
            xt = _greedy(xt, steps=100, m0=0.007, m1=0.0007, lr=0.0075)
        elif mode in (3, 8):
            # global medium jitter (basin hop)
            xt = _norm(xt + rng.standard_normal(size=(n, d))*0.012)
            xt = _smooth(xt, steps=190, a0=50.0, a1=300.0, lr0=0.026, lr1=0.0022)
            xt = _greedy(xt, steps=80, m0=0.006, m1=0.0008, lr=0.007)
        else:
            # random 10 shake medium-large
            idx = rng.choice(n, size=10, replace=False)
            xt[idx] = _norm(xt[idx] + rng.standard_normal(size=(10, d))*0.11)
            xt = _smooth(xt, steps=200, a0=45.0, a1=310.0, lr0=0.028, lr1=0.0025)
            xt = _greedy(xt, steps=90, m0=0.007, m1=0.0008, lr=0.007)
        v = _max_cos(xt)
        if v < gval:
            gval = float(v); gbest = xt.copy()
        # early exit if time low
        if trial > 300:
            break
    # final squeeze: longer anneal then capped LP
    if time.time() < DEADLINE - 2.0:
        gbest = _smooth(gbest, steps=160, a0=60.0, a1=400.0, lr0=0.020, lr1=0.0015)
        gbest = _greedy(gbest, steps=90, m0=0.005, m1=0.0005, lr=0.006)
    remain = DEADLINE - time.time() - 0.5
    if remain > 4.0:
        gbest = _lp_polish_cap(gbest, seconds=min(remain, 9.0), deadline=DEADLINE-0.3)
    gbest = _norm(gbest)
    if not np.all(np.isfinite(gbest)):
        gbest = xb
    return gbest
