import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-300
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _softmax_refine(x, steps=250, a0=40.0, a1=320.0, lr0=0.03, lr1=0.003):
    n, d = x.shape
    la0 = np.log(a0); la1 = np.log(a1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for t in range(steps):
        f = t / max(steps - 1, 1)
        alpha = np.exp(la0 + (la1 - la0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        mx = float(np.max(G))
        W = np.exp(alpha * (G - mx))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-30
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        direction = gt / S
        x = x - lr * direction
        x /= (np.linalg.norm(x, axis=1, keepdims=True) + 1e-300)
    return x

def _refine_hole_single(Xf, y, steps=120, a0=60.0, a1=300.0, lr0=0.06, lr1=0.005):
    la0 = np.log(a0); la1 = np.log(a1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for t in range(steps):
        f = t / max(steps - 1, 1)
        alpha = np.exp(la0 + (la1 - la0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        s = Xf @ y
        mx = float(np.max(s))
        w = np.exp(alpha * (s - mx))
        sw = float(np.sum(w)) + 1e-30
        g = w @ Xf
        g = g / sw
        d = float(np.dot(g, y))
        gt = g - d * y
        y = y - lr * gt
        nrm = float(np.linalg.norm(y))
        if nrm == 0 or not np.isfinite(nrm):
            break
        y = y / nrm
    return y

def _best_hole(Xf, rng, n_sample=8192):
    d = Xf.shape[1]
    Y = rng.standard_normal(size=(n_sample, d))
    Y /= (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-300)
    S = Y @ Xf.T
    m = S.max(axis=1)
    idx = int(np.argmin(m))
    return Y[idx]

def _lp_polish_capped(x, seconds, pcap=2500, r0=1e-3):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = r0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.5:
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
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
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
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return xb

def solve():
    t0 = time.time()
    SURGERY_END = t0 + 100.0
    HARD_CUT = t0 + 165.0
    rng = np.random.RandomState(20261103)
    n, d = 184, 8
    try:
        x0 = np.asarray(np.load("best.npy"), dtype=np.float64)
    except Exception:
        x0 = np.asarray(np.load("record.npy"), dtype=np.float64)
    if x0.shape != (n, d):
        x0 = x0.reshape(n, d)
    x0 = _norm(x0)
    best_val = _max_cos(x0)
    x_best = x0.copy()
    iu_all = np.triu_indices(n, 1)
    Ks = [12, 16, 20, 24]
    cyc = 0
    # surgery loop
    while time.time() < SURGERY_END and time.time() < HARD_CUT - 45.0:
        K = Ks[cyc % len(Ks)]
        cyc += 1
        # contacts at 1e-7 on incumbent best
        G = x_best @ x_best.T
        np.fill_diagonal(G, -2.0)
        iu = iu_all
        cf = G[iu]
        mcur = float(cf.max())
        contact_mask = cf > mcur - 1e-7
        deg = np.zeros(n, dtype=np.int64)
        if np.any(contact_mask):
            np.add.at(deg, iu[0][contact_mask], 1)
            np.add.at(deg, iu[1][contact_mask], 1)
        order_hi = np.argsort(-deg, kind="stable")
        tight_order = np.argsort(-cf, kind="stable")
        victims = set()
        # rattlers (degree 0, else lowest)
        rattle = np.where(deg == 0)[0]
        if len(rattle) == 0:
            rattle = np.argsort(deg, kind="stable")[:2]
        for v in rattle:
            if len(victims) < K:
                victims.add(int(v))
        # highest-degree hubs: K//4
        nh = max(2, K // 4)
        for v in order_hi[:nh * 2]:
            if len(victims) >= min(K, len(rattle) + nh):
                break
            victims.add(int(v))
        # endpoints of tightest pairs
        need_t = max(2, K // 4)
        cnt_t = 0
        for k in tight_order[:60]:
            if cnt_t >= need_t:
                break
            a = int(iu[0][k]); b = int(iu[1][k])
            added = False
            if a not in victims and len(victims) < K:
                victims.add(a); added = True
            if b not in victims and len(victims) < K:
                victims.add(b); added = True
            if added:
                cnt_t += 1
            if len(victims) >= K:
                break
        # random members of low-contact block 85..105 (use 80..110)
        low_block = np.arange(80, 111)
        # also consider global lowest-degree pool
        low_pool = np.argsort(deg, kind="stable")[:60]
        perm_b = rng.permutation(len(low_block))
        for k in perm_b:
            if len(victims) >= K:
                break
            victims.add(int(low_block[k]))
        perm_p = rng.permutation(len(low_pool))
        for k in perm_p:
            if len(victims) >= K:
                break
            victims.add(int(low_pool[k]))
        while len(victims) < K:
            victims.add(int(rng.randint(0, n)))
        vlist = list(victims)[:K]
        # random reinsert order
        order = rng.permutation(K)
        vlist = [vlist[o] for o in order]
        vset = set(vlist)
        keep = np.array([i for i in range(n) if i not in vset], dtype=np.int64)
        x_work = x_best.copy()
        # scramble victim rows to random (will be overwritten)
        R = rng.standard_normal(size=(K, d))
        R /= (np.linalg.norm(R, axis=1, keepdims=True) + 1e-300)
        for ii, vi in enumerate(vlist):
            x_work[vi] = R[ii]
        placed = []
        # incremental insertion
        for ii, vi in enumerate(vlist):
            if time.time() > SURGERY_END or time.time() > HARD_CUT - 45.0:
                break
            # fixed = keep + already placed
            if len(placed) == 0:
                Xf = x_work[keep]
            else:
                Xf = np.concatenate([x_work[keep], x_work[np.array(placed, dtype=np.int64)]], axis=0)
            y0 = _best_hole(Xf, rng, n_sample=8192)
            y0 = _refine_hole_single(Xf, y0, steps=120)
            x_work[vi] = y0
            placed.append(vi)
            if (len(placed) % 4 == 0) or (len(placed) == K):
                # 250 steps all-point softmax refinement
                x_work = _softmax_refine(x_work, steps=250)
        v = _max_cos(x_work)
        if v < best_val - 1e-7:
            best_val = v
            x_best = x_work.copy()
    # final LP polish, at most 40s and absolute cutoff 165s
    remain = HARD_CUT - time.time()
    lp_time = min(40.0, max(1.0, remain - 1.0))
    if lp_time > 0.5 and time.time() < HARD_CUT - 1.0:
        try:
            x_pol = _lp_polish_capped(x_best, seconds=lp_time, pcap=2500)
            vp = _max_cos(x_pol)
            if vp < best_val:
                x_best = x_pol
                best_val = vp
        except Exception:
            pass
    x_best = _norm(x_best)
    return x_best
