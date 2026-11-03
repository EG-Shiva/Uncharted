import time
import numpy as np

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-300
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _per_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1)

def _smooth_refine(x, steps, a0, a1, lr0, lr1):
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
        x = x - lr * (gt / S)
        x = _norm(x)
    return x

def _greedy_refine(x, steps, m0, m1, lr):
    for t in range(steps):
        f = t / max(steps - 1, 1)
        margin = m0 + (m1 - m0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        thr = m - margin
        M = (G > thr).astype(np.float64)
        deg = M.sum(axis=1, keepdims=True)
        grad = M @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        x = x - lr * (gt / np.maximum(deg, 1.0))
        x = _norm(x)
    return x

def _hole_pick(xfix, rng, n_sample=2000):
    d = xfix.shape[1]
    Y = rng.standard_normal(size=(n_sample, d))
    Y = _norm(Y)
    S = Y @ xfix.T
    m = S.max(axis=1)
    idx = int(np.argmin(m))
    return Y[idx]

def _hole_refine(xfix, y, steps=80, a0=60.0, a1=250.0, lr0=0.05, lr1=0.005):
    la0 = np.log(a0); la1 = np.log(a1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for t in range(steps):
        f = t / max(steps - 1, 1)
        alpha = np.exp(la0 + (la1 - la0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        s = xfix @ y
        mx = float(np.max(s))
        w = np.exp(alpha * (s - mx))
        sw = float(np.sum(w)) + 1e-30
        g = w @ xfix
        gt = g - float(np.dot(g, y)) * y
        y = y - lr * gt / sw
        nrm = float(np.linalg.norm(y))
        if not np.isfinite(nrm) or nrm == 0:
            break
        y = y / nrm
    return y

def solve():
    t0 = time.time()
    HOP_BUDGET = 132.0  # leave large safety margin, no LP
    rng = np.random.default_rng(0)
    n, d = 184, 8
    cands = []
    for f in ("best.npy", "record.npy"):
        try:
            c = np.asarray(np.load(f), dtype=np.float64)
            if c.shape == (n, d):
                c = _norm(c)
                # guard nan
                if np.all(np.isfinite(c)):
                    cands.append((_max_cos(c), c))
        except Exception:
            pass
    if cands:
        cands.sort(key=lambda t: t[0])
        gval, gbest = cands[0][0], cands[0][1].copy()
    else:
        gbest = _norm(rng.standard_normal(size=(n, d)))
        gval = _max_cos(gbest)
    x_cur = gbest.copy()
    cur = float(gval)

    trial = 0
    # precompute nothing; loop
    while True:
        trial += 1
        el = time.time() - t0
        if el > HOP_BUDGET:
            break
        if trial > 2000:
            break
        # temperature for Metropolis (allow sideways)
        prog = el / HOP_BUDGET
        Temp = 0.0016 * (1.0 - prog) + 0.00008
        # periodic restart from best if wandering too bad
        if cur > gval + 0.012 or (trial % 25 == 0 and rng.random() < 0.35):
            base = gbest.copy()
            base_val = gval
        else:
            base = x_cur
            base_val = cur
        xt = base.copy()
        mode = trial % 7
        if mode == 0:
            # global small noise
            xt = xt + rng.standard_normal(size=xt.shape) * 0.012
            xt = _norm(xt)
        elif mode == 1:
            # kick worst points
            pm = _per_max(xt)
            worst = np.argsort(-pm, kind="stable")[:4]
            xt[worst] += rng.standard_normal(size=(4, d)) * 0.085
            xt[worst] = _norm(xt[worst])
        elif mode == 2:
            # kick hubs defined by near-max frequency
            G = xt @ xt.T
            np.fill_diagonal(G, -2.0)
            mcur = float(np.max(G))
            cnt = np.sum(G > mcur - 0.004, axis=1)
            hubs = np.argsort(-cnt, kind="stable")[:6]
            xt[hubs] += rng.standard_normal(size=(6, d)) * 0.10
            xt[hubs] = _norm(xt[hubs])
            # also tiny global jitter
            xt += rng.standard_normal(size=xt.shape) * 0.002
            xt = _norm(xt)
        elif mode == 3:
            # rattler / low-point relocation to voids
            pm = _per_max(xt)
            lowest = np.argsort(pm, kind="stable")[:2]
            keep_idx = [i for i in range(n) if i not in set(lowest.tolist())]
            xfix = xt[keep_idx].copy()
            for li in lowest:
                y0 = _hole_pick(xfix, rng, n_sample=2000)
                y0 = _hole_refine(xfix, y0, steps=80)
                xt[li] = y0
                xfix = np.vstack([xfix, y0[None, :]])
            # small jitter on rest to allow topology change
            xt[keep_idx] += rng.standard_normal(size=(len(keep_idx), d)) * 0.004
            xt = _norm(xt)
        elif mode == 4:
            # neighbourhood shake around random center
            G = xt @ xt.T
            np.fill_diagonal(G, -2.0)
            c = int(rng.integers(0, n))
            nearest = np.argsort(-G[c], kind="stable")[:12]
            xt[nearest] += rng.standard_normal(size=(12, d)) * 0.06
            xt[nearest] = _norm(xt[nearest])
        elif mode == 5:
            # medium global shake
            xt = xt + rng.standard_normal(size=xt.shape) * 0.028
            xt = _norm(xt)
        else:
            # shake low-contact block + one hub
            G = xt @ xt.T
            np.fill_diagonal(G, -2.0)
            mcur = float(np.max(G))
            cnt = np.sum(G > mcur - 0.004, axis=1)
            hubs = np.argsort(-cnt, kind="stable")[:3]
            pm = _per_max(xt)
            lows = np.argsort(pm, kind="stable")[:5]
            idx = np.unique(np.concatenate([hubs, lows]))
            xt[idx] += rng.standard_normal(size=(len(idx), d)) * 0.075
            xt[idx] = _norm(xt[idx])

        # anneal: parameters slightly randomized for diversity
        a0 = 40.0 + rng.random() * 15.0
        a1 = 260.0 + rng.random() * 80.0
        lr0 = 0.026 + rng.random() * 0.010
        lr1 = 0.0022 + rng.random() * 0.0012
        if mode == 3:
            xt = _smooth_refine(xt, steps=260, a0=a0, a1=a1, lr0=lr0, lr1=lr1)
            xt = _greedy_refine(xt, steps=90, m0=0.007, m1=0.0007, lr=0.007)
        else:
            xt = _smooth_refine(xt, steps=220, a0=a0, a1=a1, lr0=lr0, lr1=lr1)
            xt = _greedy_refine(xt, steps=70, m0=0.006, m1=0.0007, lr=0.0065)

        v = _max_cos(xt)
        if not np.isfinite(v):
            continue
        if v < gval:
            gval = float(v)
            gbest = xt.copy()
            x_cur = xt.copy()
            cur = float(v)
        else:
            # Metropolis acceptance for wandering state
            if v <= cur or rng.random() < np.exp(-(v - cur) / max(Temp, 1e-9)):
                x_cur = xt.copy()
                cur = float(v)
        # early time check
        if time.time() - t0 > HOP_BUDGET:
            break

    # short safe final tightening on best only (few ms each), no LP
    try:
        if time.time() - t0 < 150.0:
            gbest = _greedy_refine(gbest.copy(), steps=60, m0=0.004, m1=0.0004, lr=0.004)
            v = _max_cos(gbest)
            if np.isfinite(v) and v < gval:
                gval = float(v)
            else:
                # revert if worse (keep tightest)
                pass
    except Exception:
        pass

    gbest = _norm(gbest)
    if not np.all(np.isfinite(gbest)):
        # fallback to loaded best
        gbest = _norm(cands[0][1].copy())
    return gbest
