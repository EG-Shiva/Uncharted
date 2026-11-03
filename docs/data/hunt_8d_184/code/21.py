import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(x):
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-300
    return x / n

def _anneal(x, steps_soft=300, steps_hard=120, a0=50.0, a1=350.0, lr0=0.025, lr1=0.004, h_lr=0.008, h_m0=0.006, h_m1=0.0006):
    n, d = x.shape
    la0 = np.log(a0); la1 = np.log(a1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for t in range(steps_soft):
        f = t / max(steps_soft - 1, 1)
        alpha = np.exp(la0 + (la1 - la0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        # avoid overflow: exp(alpha*(G-m)), G-m<=0
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-30
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        x = x - lr * (gt / S)
        x /= (np.linalg.norm(x, axis=1, keepdims=True) + 1e-300)
    # hard top-pair phase
    for t in range(steps_hard):
        f = t / max(steps_hard - 1, 1)
        margin = h_m0 + (h_m1 - h_m0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        M = (G > (m - margin))
        deg = M.sum(axis=1, keepdims=True).astype(np.float64)
        # skip if no pairs (should not happen)
        grad = M.astype(np.float64) @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        denom = np.maximum(deg, 1.0)
        x = x - h_lr * (gt / denom)
        x /= (np.linalg.norm(x, axis=1, keepdims=True) + 1e-300)
    return x

def _refine_single(x_fixed, y, steps=120, a0=80.0, a1=350.0, lr0=0.06, lr1=0.005):
    la0 = np.log(a0); la1 = np.log(a1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for t in range(steps):
        f = t / max(steps - 1, 1)
        alpha = np.exp(la0 + (la1 - la0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        s = x_fixed @ y
        m = float(np.max(s))
        w = np.exp(alpha * (s - m))
        sw = float(np.sum(w)) + 1e-30
        g = w @ x_fixed
        gt = g - float(np.dot(g, y)) * y
        y = y - lr * gt / sw
        nrm = float(np.linalg.norm(y))
        if not np.isfinite(nrm) or nrm == 0:
            break
        y = y / nrm
    return y

def _sample_hole(x_fixed, rng, n_sample=5000):
    d = x_fixed.shape[1]
    Y = rng.standard_normal(size=(n_sample, d))
    Y /= (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-300)
    S = Y @ x_fixed.T
    m = S.max(axis=1)
    idx = int(np.argmin(m))
    return Y[idx], float(m[idx])

def _lp_polish(x, seconds, t_end):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = 1.2e-3
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() < t_end and (t_end - time.time()) > 0.5 and radius > 1e-14:
        # time guard for seconds
        G = x @ x.T
        cf = G[iu]
        m = float(cf.max())
        thr = m - max(20.0 * radius, 2e-9)
        sel = cf > thr
        I = iu[0][sel]; J = iu[1][sel]; cs = cf[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 2600:
            idx = np.argpartition(-cs, 2600)[:2600]
            I, J, cs = I[idx], J[idx], cs[idx]
            p = 2600
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
            if radius < 4e-7:
                # stuck, recenter on best with smaller radius once
                x = xb.copy()
                if radius < 1e-9:
                    break
    return xb

def solve():
    t0 = time.time()
    T_END = t0 + 175.0
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
    # hopping phase until LP budget
    # reserve ~28s for final LP
    HOP_END = T_END - 30.0
    trial = 0
    iu_full = np.triu_indices(n, 1)
    while time.time() < HOP_END:
        trial += 1
        if trial > 300:
            break
        # contact frequency with margin 0.003
        G = gbest @ gbest.T
        np.fill_diagonal(G, -2.0)
        mcur = float(G.max())
        thr_c = mcur - 0.003
        mask = G > thr_c
        # upper triangle freq
        freq = mask.sum(axis=1).astype(np.float64)
        order_hi = np.argsort(-freq)
        order_lo = np.argsort(freq)
        # worst pair
        flat = int(np.argmax(G))
        wi, wj = flat // n, flat % n
        # choose kick size
        r = rng.random()
        if trial % 25 == 0:
            k = 5 + int(rng.integers(0, 4))  # 5-8 big jump
        elif trial % 7 == 0:
            k = 3 + int(rng.integers(0, 2))  # 3-4
        else:
            k = 1 + int(rng.integers(0, 2))  # 1-2
        victims = set()
        victims.add(wi if rng.random() < 0.5 else wj)
        if rng.random() < 0.55:
            victims.add(wi); victims.add(wj)
        # add overloaded
        if rng.random() < 0.75:
            victims.add(int(order_hi[int(rng.integers(0, min(15, n)))]))
        # add loose / rattler
        if rng.random() < 0.65:
            victims.add(int(order_lo[int(rng.integers(0, min(25, n)))]))
        while len(victims) < k:
            if rng.random() < 0.5:
                victims.add(int(rng.integers(0, n)))
            else:
                victims.add(int(order_hi[int(rng.integers(0, min(30, n)))]))
        victims = list(victims)[:max(k, 6) if k > 5 else k]
        # cap k to avoid destroying
        x = gbest.copy()
        # apply kicks sequentially
        # for hole sampling, maintain fixed set
        for v in victims:
            if time.time() > HOP_END:
                break
            do_hole = rng.random() < 0.55
            # bias: loose points -> hole, hubs -> shake sometimes
            if freq[v] <= 3:
                do_hole = rng.random() < 0.85
            if do_hole:
                others = np.delete(x, v, axis=0)
                y0, _ = _sample_hole(others, rng, n_sample=4000)
                y0 = _refine_single(others, y0, steps=80)
                x[v] = y0
            else:
                sigma = 0.12 + 0.25 * rng.random()
                if trial % 25 == 0:
                    sigma = 0.35 + 0.25 * rng.random()
                noise = rng.standard_normal(size=(d,))
                # tangential
                noise = noise - float(np.dot(noise, x[v])) * x[v]
                nn = float(np.linalg.norm(noise))
                if nn > 0:
                    noise = noise / nn
                    x[v] = x[v] + sigma * noise
                    x[v] /= (float(np.linalg.norm(x[v])) + 1e-300)
        # anneal whole code
        # vary schedule slightly for diversity
        a0 = 40.0 + 20.0 * rng.random()
        a1 = 300.0 + 100.0 * rng.random()
        lr0 = 0.02 + 0.01 * rng.random()
        # fewer steps for big kicks? keep fixed for speed
        steps_soft = 260 if (time.time() < HOP_END - 10) else 160
        steps_hard = 90
        # quick time check: if little time, reduce steps
        if T_END - time.time() < 50:
            steps_soft = 160
            steps_hard = 60
        x = _anneal(x, steps_soft=steps_soft, steps_hard=steps_hard, a0=a0, a1=a1, lr0=lr0, lr1=0.004)
        v = _max_cos(x)
        if v < gval:
            gval = v
            gbest = x.copy()
            # early stop if clearly record-breaking by margin? continue to push
        # else discard (monotonic)
    # final LP polish on global best
    try:
        if time.time() < T_END - 2.0:
            gbest = _lp_polish(gbest, seconds=(T_END - time.time() - 1.0), t_end=T_END - 1.0)
    except Exception:
        pass
    gbest = _norm(gbest)
    return gbest
