import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _smooth_refine(x, steps=250, a0=40.0, a1=320.0, lr0=0.03, lr1=0.003):
    n, d = x.shape
    la0 = np.log(a0); la1 = np.log(a1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
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
        direction = gt / S
        x = x - lr * direction
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _refine_hole_single(x_fixed, y, steps=120, a0=60.0, a1=300.0, lr0=0.06, lr1=0.005):
    la0 = np.log(a0); la1 = np.log(a1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for t in range(steps):
        f = t / max(steps - 1, 1)
        alpha = np.exp(la0 + (la1 - la0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        s = x_fixed @ y
        mx = np.max(s)
        w = np.exp(alpha * (s - mx))
        sw = np.sum(w) + 1e-30
        g = w @ x_fixed
        gt = g - np.dot(g, y) * y
        y = y - lr * gt / sw
        nrm = np.linalg.norm(y)
        if nrm == 0 or not np.isfinite(nrm):
            break
        y = y / nrm
    return y

def _best_hole_sample(x_fixed, rng, n_sample=8192):
    d = x_fixed.shape[1]
    Y = rng.standard_normal(size=(n_sample, d))
    Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-300
    S = Y @ x_fixed.T
    m = S.max(axis=1)
    idx = int(np.argmin(m))
    return Y[idx].copy(), float(m[idx])

def _lp_polish_capped(x, seconds, deadline, pcap=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds:
        if time.time() > deadline:
            break
        if radius < 1e-14:
            break
        if time.time() > deadline - 0.5:
            break
        G = x @ x.T
        cf = G[iu0]
        m = float(cf.max())
        thr = m - max(20.0 * radius, 1e-9)
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
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            best = v; xb = y.copy(); x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return xb

def solve():
    t0 = time.time()
    DEADLINE = t0 + 165.0
    SURGERY_END = t0 + 100.0
    rng = np.random.default_rng(0)
    n, d = 184, 8
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.standard_normal(size=(n, d))
    xb = np.asarray(xb, dtype=np.float64)
    if xb.shape != (n, d):
        try:
            xb = xb.reshape(n, d)
        except Exception:
            xb = rng.standard_normal(size=(n, d))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    gbest = xb.copy()
    gval = _max_cos(gbest)

    ks = [12, 16, 20, 24]
    cyc = 0
    iu = np.triu_indices(n, 1)
    while time.time() < SURGERY_END and time.time() < DEADLINE - 45.0:
        k = ks[cyc % len(ks)]
        cyc += 1
        # --- contact graph at 1e-7 ---
        G = gbest @ gbest.T
        np.fill_diagonal(G, -2.0)
        mcur = float(np.max(G))
        tol = 1e-7
        mask = G > (mcur - tol)
        deg = mask.sum(axis=1).astype(np.int64)
        # worst pair
        w = int(np.argmax(G[iu]))
        wi, wj = int(iu[0][w]), int(iu[1][w])
        # victim selection
        victims = set()
        ratt = np.where(deg == 0)[0].tolist()
        rng.shuffle(ratt)
        for r in ratt:
            if len(victims) >= k:
                break
            victims.add(int(r))
        # hubs: top-degree
        order_hi = np.argsort(-deg, kind="stable")[:8]
        # random subset of hubs
        nh = min(4, k - len(victims))
        if nh > 0:
            perm = rng.permutation(order_hi)[:nh]
            for v in perm:
                victims.add(int(v))
        for v in (wi, wj):
            if len(victims) < k:
                victims.add(int(v))
            else:
                break
        # low-contact pool: bottom 40 + static block 85-105
        order_lo = np.argsort(deg, kind="stable")[:40]
        static_block = np.arange(85, 106)
        pool = np.unique(np.concatenate([order_lo, static_block]))
        pool = [int(v) for v in pool if int(v) not in victims]
        if pool and len(victims) < k:
            rng.shuffle(pool)
            need = k - len(victims)
            # take up to need, but leave at least some random fill for diversity
            take = min(need, max(need - 2, need // 2 + 1)) if need > 4 else need
            # vary take randomly
            if need > 4:
                take = int(rng.integers(need // 2, need))
                take = max(1, take)
            for v in pool[:take]:
                victims.add(int(v))
                if len(victims) >= k:
                    break
        while len(victims) < k:
            v = int(rng.integers(0, n))
            victims.add(v)
        victims = list(victims)
        rng.shuffle(victims)
        # --- remove ---
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[victims] = False
        base = gbest[keep_mask].copy()
        cur_pts = base.copy()
        # --- reinsert one at a time ---
        for ii in range(len(victims)):
            if time.time() > SURGERY_END or time.time() > DEADLINE - 45.0:
                break
            y0, _ = _best_hole_sample(cur_pts, rng, n_sample=8192)
            y0 = _refine_hole_single(cur_pts, y0, steps=120)
            cur_pts = np.vstack([cur_pts, y0[None, :]])
            if ((ii + 1) % 4 == 0) and ((ii + 1) < len(victims)):
                cur_pts = _smooth_refine(cur_pts, steps=250)
        if cur_pts.shape[0] != n:
            continue
        cur_pts = _smooth_refine(cur_pts, steps=250)
        v = _max_cos(cur_pts)
        if v < gval - 1e-7:
            gval = float(v)
            gbest = cur_pts.copy()
    # final LP polish, at most 40s
    remain = DEADLINE - time.time() - 1.0
    if remain > 5:
        lp_t = min(remain, 40.0)
        try:
            gbest = _lp_polish_capped(gbest, lp_t, DEADLINE, pcap=2500)
        except Exception:
            pass
    gbest = gbest / np.linalg.norm(gbest, axis=1, keepdims=True)
    if not np.all(np.isfinite(gbest)):
        gbest = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    return gbest
