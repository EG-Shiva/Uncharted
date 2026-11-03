import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-300)

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _smooth(x, steps, a0, a1, lr0, lr1):
    n, d = x.shape
    la0, la1 = np.log(a0), np.log(a1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(steps - 1, 1)
        alpha = np.exp(la0 + (la1 - la0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        mx = np.max(G)
        W = np.exp(alpha * (G - mx))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-30
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        x = x - lr * gt / S
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _greedy(x, steps, m0, m1, lr):
    for t in range(steps):
        f = t / max(steps - 1, 1)
        margin = m0 + (m1 - m0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = np.max(G)
        M = (G > (m - margin)).astype(np.float64)
        np.fill_diagonal(M, 0.0)
        deg = M.sum(axis=1, keepdims=True)
        grad = M @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        x = x - lr * gt / np.maximum(deg, 1.0)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _hole_large(xfix, rng, ns=50000, batch=10000):
    d = xfix.shape[1]
    best_y = None
    best_m = 2.0
    nb = ns // batch
    for _ in range(nb):
        Y = rng.standard_normal(size=(batch, d))
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        S = Y @ xfix.T
        m = S.max(axis=1)
        i = int(np.argmin(m))
        if m[i] < best_m:
            best_m = float(m[i])
            best_y = Y[i].copy()
    return best_y

def _refine_one(xfix, y, steps=60, alpha=250.0, lr=0.02):
    for _ in range(steps):
        s = xfix @ y
        mx = np.max(s)
        w = np.exp(alpha * (s - mx))
        sw = np.sum(w) + 1e-30
        g = w @ xfix
        gt = g - np.dot(g, y) * y
        y = y - lr * gt / sw
        y /= np.linalg.norm(y) + 1e-300
    return y

def _local_opt(x, active, steps=50, alpha=250.0, lr=0.02):
    # optimize only active rows, repulsion vs all rows
    active = np.asarray(active, dtype=int)
    ka = len(active)
    if ka == 0:
        return x
    # map global col index -> position in active for self-mask
    # build col position lookup
    pos = -np.ones(x.shape[0], dtype=int)
    pos[active] = np.arange(ka)
    for _ in range(steps):
        Xa = x[active]  # ka x d
        S = Xa @ x.T  # ka x n
        # zero self interaction
        # S[a, active[a]] = -inf equivalent -> set W=0 there
        mx = np.max(S, axis=1, keepdims=True)
        # subtract max for stability; but self dot=1 may be max -> mask after exp
        W = np.exp(alpha * (S - mx))
        # mask self
        W[np.arange(ka), active] = 0.0
        Sw = W.sum(axis=1, keepdims=True) + 1e-30
        grad = W @ x
        dot = np.sum(grad * Xa, axis=1, keepdims=True)
        gt = grad - dot * Xa
        Xa2 = Xa - lr * gt / Sw
        Xa2 /= np.linalg.norm(Xa2, axis=1, keepdims=True) + 1e-300
        x[active] = Xa2
    return x

def _lp_polish(x, seconds, deadline):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = 1.0e-3
    t0 = time.time()
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - t0 < seconds and time.time() < deadline:
        if radius < 1e-14:
            break
        G = x @ x.T
        cf = G[iu0]
        m = float(cf.max())
        thr = m - max(20.0 * radius, 2e-9)
        sel = cf > thr
        I = iu0[0][sel]; J = iu0[1][sel]; cs = cf[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 2000:
            idx = np.argpartition(-cs, 2000)[:2000]
            I, J, cs = I[idx], J[idx], cs[idx]
            p = 2000
        nv = n * d + 1
        rows = np.concatenate([np.repeat(np.arange(p), d), np.repeat(np.arange(p), d), np.arange(p)])
        cols = np.concatenate([np.repeat(I * d, d) + np.tile(ar, p),
                               np.repeat(J * d, d) + np.tile(ar, p),
                               np.full(p, n * d)])
        vals = np.concatenate([x[J].ravel(), x[I].ravel(), -np.ones(p)])
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
            radius = min(radius * 1.5, 0.015)
        else:
            radius *= 0.5
            if radius < 4e-7:
                break
    return xb, best

def solve():
    t0 = time.time()
    DEADLINE = t0 + 168.0
    rng = np.random.default_rng(0)
    n, d = 184, 8
    cands = []
    for f in ("best.npy", "record.npy"):
        try:
            c = np.asarray(np.load(f), dtype=np.float64)
            if c.shape == (n, d):
                c = _norm(c)
                cands.append((_max_cos(c), c))
        except Exception:
            pass
    if cands:
        cands.sort(key=lambda t: t[0])
        gval, gbest = cands[0][0], cands[0][1].copy()
    else:
        gbest = _norm(rng.standard_normal(size=(n, d)))
        gval = _max_cos(gbest)
    gbest = np.ascontiguousarray(gbest)

    # contact degrees with tight threshold to find jam structure
    G0 = gbest @ gbest.T
    np.fill_diagonal(G0, -2.0)
    m0 = float(np.max(G0))
    contact = G0 > (m0 - 1e-9)
    deg = contact.sum(axis=1)
    rattler = int(np.argmin(deg))
    # high-degree: >=22, else top
    order_hi = np.argsort(-deg, kind="stable")
    order_lo = np.argsort(deg, kind="stable")
    high_pool = [i for i in order_hi if deg[i] >= 22]
    if len(high_pool) < 10:
        high_pool = list(order_hi[:12])
    low_set = set([i for i in range(n) if deg[i] <= 4])
    # greedy cluster: pick highest-degree not yet picked (simple)
    # build victim lists for each k
    ks = [1, 2, 3, 4, 6, 8]
    surg_results = []  # (val, array)
    # keep global best as fallback candidate
    surg_results.append((gval, gbest.copy()))

    for k in ks:
        if time.time() > DEADLINE - 35.0:
            break
        # victims: rattler + top (k-1) from high_pool maximizing contacts
        victims = []
        victims.append(rattler)
        for h in high_pool:
            if len(victims) >= k:
                break
            if h != rattler and h not in victims:
                victims.append(h)
        # if still short, fill from order_hi
        for h in order_hi:
            if len(victims) >= k:
                break
            if h not in victims:
                victims.append(h)
        victims = victims[:k]
        vset = set(victims)
        keep = [i for i in range(n) if i not in vset]
        cur = gbest[keep].copy()  # (n-k) x d
        # reinsert one at a time
        for s in range(k):
            if time.time() > DEADLINE - 30.0:
                break
            y = _hole_large(cur, rng, ns=50000, batch=10000)
            y = _refine_one(cur, y, steps=70, alpha=250.0, lr=0.02)
            cur = np.vstack([cur, y[None, :]])
            # local optimize: reinserted + neighbors with cos>0.35
            ncur = cur.shape[0]
            n_re = s + 1
            re_idx = np.arange(ncur - n_re, ncur)
            # neighbors among earlier points
            S = cur[re_idx] @ cur.T  # n_re x ncur
            neigh = set(re_idx.tolist())
            # add points with cos > 0.35 to any reinserted
            mask = (S > 0.35)
            # for each re row, add cols
            cols = np.where(mask.any(axis=0))[0]
            for c in cols.tolist():
                neigh.add(int(c))
            active = sorted(neigh)
            # limit active size to avoid blowup (take at most 60)
            if len(active) > 60:
                # keep all reinserted + closest neighbors
                # score neighbors by max dot to reinserted
                scores = S.max(axis=0)
                # sort candidate non-re by score desc
                cand = [a for a in active if a not in set(re_idx.tolist())]
                cand.sort(key=lambda a: -scores[a])
                active = sorted(re_idx.tolist() + cand[:60 - n_re])
            cur = _local_opt(cur, active, steps=50, alpha=250.0, lr=0.02)
        # cur should be n x d
        if cur.shape[0] != n:
            continue
        xt = _norm(cur)
        # short global anneal
        xt = _smooth(xt, steps=170, a0=50.0, a1=320.0, lr0=0.024, lr1=0.0025)
        # check time before greedy
        if time.time() > DEADLINE - 28.0:
            v = _max_cos(xt)
            surg_results.append((float(v), xt.copy()))
            break
        xt = _greedy(xt, steps=60, m0=0.006, m1=0.0008, lr=0.007)
        v = _max_cos(xt)
        surg_results.append((float(v), xt.copy()))
        if v < gval:
            gval = float(v)
            gbest = xt.copy()

    # pick best two surgical candidates for LP polishing
    surg_results.sort(key=lambda t: t[0])
    top2 = surg_results[:2]
    # polish each briefly
    for idx, (v, xc) in enumerate(top2):
        remain = DEADLINE - time.time()
        if remain < 8.0:
            break
        secs = min(12.0, remain - 2.0)
        if secs < 3.0:
            break
        xp, vp = _lp_polish(xc, secs, DEADLINE - 1.0)
        if vp < gval:
            gval = float(vp)
            gbest = xp.copy()

    gbest = _norm(gbest)
    if not np.all(np.isfinite(gbest)):
        try:
            gbest = _norm(np.asarray(np.load("best.npy"), dtype=np.float64))
        except Exception:
            pass
    return gbest
