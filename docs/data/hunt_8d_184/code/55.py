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

def _per_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1)

def _smooth_e(Gup_max, Gup, p=64.0):
    # Gup: upper-tri values, Gup_max: max
    # log-mean-exp / p stable
    e = np.exp(p * (Gup - Gup_max))
    m = np.mean(e) + 1e-300
    return float(Gup_max + np.log(m) / p)

def _smooth_val(x, p=64.0):
    iu = np.triu_indices(x.shape[0], 1)
    c = (x @ x.T)[iu]
    m = float(np.max(c))
    e = np.exp(p * (c - m))
    return float(m + np.log(np.mean(e) + 1e-300) / p)

def _contact_hash(x, mc=None, tol=1e-4):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    if mc is None:
        mc = float(np.max(G))
    B = G > (mc - tol)
    deg = np.sum(B, axis=1).astype(int)
    return tuple(sorted(deg.tolist())), deg

def _procrustes_dist(A, B):
    # min_R ||A-BR||_F / sqrt(n), R orthogonal 8x8
    M = B.T @ A
    try:
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        R = U @ Vt
        D = A - B @ R
        return float(np.linalg.norm(D) / np.sqrt(A.shape[0]))
    except Exception:
        return float(np.linalg.norm(A - B) / np.sqrt(A.shape[0]))

def _softmax_adam(x, steps=200, a0=50.0, a1=260.0, lr0=0.015, lr1=0.004,
                  b1=0.9, b2=0.999, eps=1e-8):
    n, d = x.shape
    m1 = np.zeros_like(x)
    m2 = np.zeros_like(x)
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
        S = W.sum(axis=1, keepdims=True) + 1e-300
        grad = W @ x / S
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        # Adam
        m1 = b1 * m1 + (1.0 - b1) * gt
        m2 = b2 * m2 + (1.0 - b2) * (gt * gt)
        bc1 = 1.0 - b1 ** (t + 1)
        bc2 = 1.0 - b2 ** (t + 1)
        mh = m1 / bc1
        vh = m2 / bc2
        step = mh / (np.sqrt(vh) + eps)
        # keep tangential
        dot2 = np.sum(step * x, axis=1, keepdims=True)
        step = step - dot2 * x
        x = x - lr * step
        x = _norm(x)
    return x

def _random_hole(xfix, rng, ns=3000):
    d = xfix.shape[1]
    Y = rng.standard_normal(size=(ns, d))
    Y = _norm(Y)
    S = Y @ xfix.T
    m = S.max(axis=1)
    i = int(np.argmin(m))
    return Y[i].copy()

def _lp_polish_single(x, deadline):
    # at most ~20s, cap pairs, radius trust region
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while True:
        if time.time() > deadline - 1.5:
            break
        if radius < 1e-13:
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
        if p > 1500:
            idx = np.argpartition(-cs, 1500)[:1500]
            I, J, cs = I[idx], J[idx], cs[idx]
            p = 1500
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
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            tlim = max(0.5, deadline - time.time() - 0.4)
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": tlim})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = _norm(x + res.x[:-1].reshape(n, d))
        except Exception:
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best:
            best = v; xb = y.copy(); x = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return xb, best

def solve():
    t0 = time.time()
    HARD = t0 + 150.0
    HOP_END = t0 + 128.0
    rng = np.random.default_rng(7)
    n, d = 184, 8
    # load incumbent
    try:
        xb0 = np.asarray(np.load("best.npy"), dtype=np.float64)
    except Exception:
        xb0 = None
    try:
        xr = np.asarray(np.load("record.npy"), dtype=np.float64)
    except Exception:
        xr = None
    cands = []
    for c in (xb0, xr):
        if c is not None:
            try:
                c = np.asarray(c, dtype=np.float64).reshape(n, d)
                c = _norm(c)
                cands.append((c, _max_cos(c)))
            except Exception:
                pass
    if not cands:
        c = _norm(rng.standard_normal(size=(n, d)))
        cands.append((c, _max_cos(c)))
    cands.sort(key=lambda t: t[1])
    base, base_val = cands[0]
    gbest = base.copy()
    gval = float(base_val)
    gsmooth = _smooth_val(gbest)

    POP = 5
    pop_x = []
    pop_v = []
    pop_s = []
    # diverse init
    sigmas = [0.0, 0.012, 0.035, 0.07, 0.10]
    for k in range(POP):
        if time.time() > HOP_END - 5:
            x = base.copy()
        else:
            x = base.copy()
            s = sigmas[k]
            if s > 0:
                x = x + rng.standard_normal(size=x.shape) * s
                x = _norm(x)
            # for two members relocate lowest point to random hole
            if k >= 3:
                pm = _per_max(x)
                low = int(np.argmin(pm))
                keep = [i for i in range(n) if i != low]
                xf = x[keep]
                y = _random_hole(xf, rng, ns=2000)
                x[low] = y
                x = _norm(x)
            # short adam to separate basins quickly
            x = _softmax_adam(x, steps=120, a0=45.0, a1=220.0, lr0=0.012, lr1=0.004)
        v = _max_cos(x)
        s_ = _smooth_val(x)
        pop_x.append(x); pop_v.append(float(v)); pop_s.append(float(s_))
        if v < gval:
            gval = float(v); gbest = x.copy(); gsmooth = float(s_)
    # tabu of contact hashes
    tabu = set()
    for x in pop_x:
        try:
            h, _ = _contact_hash(x, tol=1e-4)
            tabu.add(h)
        except Exception:
            pass

    iu = np.triu_indices(n, 1)
    trial = 0
    # main non-monotonic search
    while time.time() < HOP_END and time.time() < HARD - 25:
        trial += 1
        frac = (time.time() - t0) / max(HOP_END - t0, 1e-9)
        frac = min(max(frac, 0.0), 1.0)
        Tsa = 8e-06 * (1.0 - frac) + 5e-07  # Metropolis temperature on smooth energy
        # pick parent round-robin with random
        pi = int(rng.integers(0, POP))
        parent = pop_x[pi]
        pv = pop_v[pi]
        ps = pop_s[pi]
        child = parent.copy()
        # mutation type
        u = rng.random()
        Gtmp = child @ child.T
        np.fill_diagonal(Gtmp, -2.0)
        mcur = float(np.max(Gtmp))
        permax = np.max(Gtmp, axis=1)
        # contact degrees at loose tol for hub detection
        cf = Gtmp[iu]
        thr_hub = mcur - 0.004
        freq = np.zeros(n, dtype=np.int64)
        mk = cf >= thr_hub
        if np.any(mk):
            np.add.at(freq, iu[0][mk], 1)
            np.add.at(freq, iu[1][mk], 1)
        if u < 0.22:
            # global small jitter
            sig = 0.006 + rng.random() * 0.010
            child = child + rng.standard_normal(size=child.shape) * sig
            child = _norm(child)
        elif u < 0.45:
            # shake worst k
            k = 3 + int(rng.integers(0, 5))
            worst = np.argsort(-permax)[:k]
            sig = 0.04 + rng.random() * 0.06
            child[worst] += rng.standard_normal(size=(k, d)) * sig
            child[worst] = _norm(child[worst])
        elif u < 0.65:
            # relocate 1-2 rattlers/low to holes
            nl = 1 + int(rng.integers(0, 2))
            if rng.random() < 0.5:
                lows = np.argsort(permax)[:nl]
            else:
                lows = np.argsort(freq)[:nl]
            # build fixed set incrementally
            for li in lows:
                mask = np.ones(n, dtype=bool); mask[li] = False
                # for second low, use updated child
                xf = child[mask]
                y = _random_hole(xf, rng, ns=3000)
                child[li] = y
            # tiny jitter rest
            child = child + rng.standard_normal(size=child.shape) * 0.002
            child = _norm(child)
        elif u < 0.85:
            # block shake around random center
            c0 = int(rng.integers(0, n))
            grow = child @ child[c0]
            nn = 10 + int(rng.integers(0, 6))
            near = np.argsort(-grow)[:nn]
            sig = 0.04 + rng.random() * 0.05
            child[near] += rng.standard_normal(size=(nn, d)) * sig
            child[near] = _norm(child[near])
        else:
            # tangential noise on overloaded hubs
            k = 4 + int(rng.integers(0, 4))
            hubs = np.argsort(-freq)[:k]
            sig = 0.05 + rng.random() * 0.05
            nz = rng.standard_normal(size=(k, d)) * sig
            # make tangential
            for ii, h in enumerate(hubs):
                nz[ii] -= np.dot(nz[ii], child[h]) * child[h]
            child[hubs] += nz
            child = _norm(child)

        # fast softmax-Adam reconvergence
        steps = 170 + int(rng.integers(0, 80))
        # vary annealing slightly
        child = _softmax_adam(child, steps=steps, a0=48.0, a1=250.0, lr0=0.013, lr1=0.0035)
        try:
            vnew = _max_cos(child)
        except Exception:
            continue
        snew = _smooth_val(child)
        # checkpoint global best immediately
        if vnew < gval:
            gval = float(vnew); gbest = child.copy(); gsmooth = float(snew)
        # topology tabu: degree sequence at 1e-4
        try:
            hnew, _ = _contact_hash(child, mc=vnew, tol=1e-4)
        except Exception:
            hnew = None
        is_tabu = (hnew in tabu) if hnew is not None else False
        if hnew is not None and hnew not in tabu:
            tabu.add(hnew)
        # Procrustes separation: min dist to pop
        try:
            dmin = min(_procrustes_dist(child, q) for q in pop_x)
        except Exception:
            dmin = 1.0
        # penalize tabu / too-close duplicates unless clearly better than global
        if is_tabu and not (vnew < gval + 1e-12):
            # allow with tiny prob only if smooth improves parent a lot
            if rng.random() > 0.02:
                continue
        if dmin < 0.0015 and vnew >= min(pop_v) - 1e-9:
            if rng.random() > 0.05:
                continue
        # Metropolis acceptance vs parent on smooth energy (p=64)
        delta = float(snew - ps)
        accept = False
        if delta < 0:
            accept = True
        else:
            prob = float(np.exp(-delta / max(Tsa, 1e-12)))
            # cap prob small for worse
            if rng.random() < min(prob, 0.15):
                accept = True
        # also always keep diversity: if vnew better than parent max, accept
        if vnew < pv:
            accept = True
        if accept:
            pop_x[pi] = child.copy()
            pop_v[pi] = float(vnew)
            pop_s[pi] = float(snew)
        else:
            # replace worst with small prob if child is good globally but not parent
            if vnew < max(pop_v) - 2e-07 and rng.random() < 0.10:
                wi = int(np.argmax(pop_v))
                # ensure separation
                try:
                    dd = min(_procrustes_dist(child, q) for j, q in enumerate(pop_x) if j != wi)
                except Exception:
                    dd = 1.0
                if dd > 0.001 or vnew < gval + 5e-07:
                    pop_x[wi] = child.copy(); pop_v[wi] = float(vnew); pop_s[wi] = float(snew)
        # occasional time check for checkpoint safety
        if trial % 25 == 0 and time.time() > HOP_END:
            break

    # final LP polish of single best distinct candidate, at most ~20s
    now = time.time()
    lp_deadline = min(HARD, now + 20.0)
    # ensure we return by HARD; leave 1s margin
    if now < HARD - 3.0 and gbest is not None:
        try:
            # polish copy, keep incumbent safe
            pb, pv2 = _lp_polish_single(gbest, lp_deadline - 0.5)
            if pv2 < gval:
                gval = float(pv2); gbest = pb.copy()
        except Exception:
            pass
    gbest = _norm(np.asarray(gbest, dtype=np.float64))
    # safety finite
    if not np.all(np.isfinite(gbest)):
        gbest = _norm(base.copy())
    return gbest
