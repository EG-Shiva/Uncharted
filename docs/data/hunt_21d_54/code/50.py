import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, r0=5e-3):
    x = _norm(np.asarray(x, dtype=float))
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        if np.count_nonzero(sel) > 800:
            idx = np.argsort(c)[-800:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        Ccol = np.empty((p, 2 * d + 1), dtype=int)
        Ccol[:, :d] = I[:, None] * d + np.arange(d)
        Ccol[:, d:2*d] = J[:, None] * d + np.arange(d)
        Ccol[:, 2*d] = n * d
        cols = Ccol.ravel()
        Arow = np.empty((p, 2 * d + 1))
        Arow[:, :d] = x[J]
        Arow[:, d:2*d] = x[I]
        Arow[:, 2*d] = -1.0
        vals = Arow.ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

# ---- symmetric orbit machinery: R21 = 10 planes + 1 axis ----
def _angles(m, mults):
    mults = np.asarray(mults, dtype=float)
    j = np.arange(m)[:, None]
    A = 2.0 * np.pi / m * j * mults[None, :]
    return np.cos(A), np.sin(A)  # (m,10)

def _gen_points(seeds, mults, m, cosA, sinA):
    # seeds (ns,21) assumed unit
    ns = seeds.shape[0]
    sp = seeds[:, :20].reshape(ns, 10, 2)
    sz = seeds[:, 20]
    N = ns * m
    P = np.empty((N, 21))
    idx = 0
    for s in range(ns):
        a0 = sp[s, :, 0]; a1 = sp[s, :, 1]
        z = sz[s]
        for jj in range(m):
            c = cosA[jj]; si = sinA[jj]
            P[idx, 0::2][:10] = a0 * c - a1 * si
            P[idx, 1::2][:10] = a0 * si + a1 * c
            # above slicing trick: P[idx,:20] interleaved
            # do it explicitly for safety
            # (overwrite with explicit)
            P[idx, :20:2] = a0 * c - a1 * si
            P[idx, 1:20:2] = a0 * si + a1 * c
            P[idx, 20] = z
            idx += 1
    return P

def _sym_loss_grad(seeds, mults, m, cosA, sinA, T):
    ns = seeds.shape[0]
    P = _gen_points(seeds, mults, m, cosA, sinA)
    N = P.shape[0]
    G = P @ P.T
    # exclude diag
    np.fill_diagonal(G, -1e9)
    m0 = float(G.max())
    E = (G - m0) / T
    np.clip(E, -30, 0, out=E)
    W = np.exp(E)
    ssum = float(W.sum())
    if ssum <= 0:
        ssum = 1.0
    W /= ssum
    loss = m0 + T * np.log(ssum)  # ordered-pair LSE (offset T*log2 vs unordered, irrelevant)
    # grad wrt P
    gradP = 2.0 * (W @ P)
    # reshape to (ns,m,21)
    g = gradP.reshape(ns, m, 21)
    gp0 = g[:, :, 0:20:2]  # (ns,m,10)
    gp1 = g[:, :, 1:20:2]
    gz = g[:, :, 20]  # (ns,m)
    # adjoint rotation sum
    # cosA,sinA (m,10)
    # grad_sp0 = sum_j c*g0 + s*g1 ; grad_sp1 = sum_j -s*g0 + c*g1
    c = cosA[None, :, :]  # (1,m,10)
    s = sinA[None, :, :]
    gs0 = np.sum(c * gp0 + s * gp1, axis=1)  # (ns,10)
    gs1 = np.sum(-s * gp0 + c * gp1, axis=1)
    gsz = np.sum(gz, axis=1)  # (ns,)
    gradS = np.empty_like(seeds)
    gradS[:, 0:20:2] = gs0
    gradS[:, 1:20:2] = gs1
    gradS[:, 20] = gsz
    # true max (with diag excluded properly)
    G2 = P @ P.T
    np.fill_diagonal(G2, -2.0)
    tmax = float(G2.max())
    return loss, gradS, tmax, P

def _adam_seeds(free_init, map_to_full, mults, m, steps, T0, T1, lr0, lr1, rng):
    # free_init (nf,21) raw; map_to_full(free (nf,21)) -> seeds (ns,21) normalized inside
    # we do projected Adam on free normalized vars
    free = _norm(np.asarray(free_init, dtype=float).copy())
    nf = free.shape[0]
    cosA, sinA = _angles(m, mults)
    mt = np.zeros_like(free)
    vt = np.zeros_like(free)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for step in range(steps):
        frac = step / max(1, steps - 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        full, free_to_full_grad = map_to_full(free)  # full (ns,21); also return info for chain
        # map_to_full returns (full_seeds_normalized, mode string)
        loss, gradS, tmax, P = _sym_loss_grad(full, mults, m, cosA, sinA, T)
        # chain to free
        gradF = _chain_grad(gradS, free, map_to_full)
        # tangent projection
        dot = np.sum(gradF * free, axis=1, keepdims=True)
        gt = gradF - dot * free
        mt = b1 * mt + (1 - b1) * gt
        vt = b2 * vt + (1 - b2) * gt * gt
        mhat = mt / (1 - b1 ** (step + 1))
        vhat = vt / (1 - b2 ** (step + 1))
        free = free - lr * mhat / (np.sqrt(vhat) + eps)
        # special: for antipodal18 self-antipodal seed force z=0
        free = _renorm_free(free, map_to_full)
        # safety
        if not np.all(np.isfinite(free)):
            free = _norm(rng.randn(*free.shape))
            mt[:] = 0; vt[:] = 0
    full, _ = map_to_full(free)
    _, _, tmax, P = _sym_loss_grad(full, mults, m, cosA, sinA, T1)
    return P, tmax

def _renorm_free(free, map_fn):
    # detect mode by attribute
    mode = getattr(map_fn, "_mode", "free")
    free = np.asarray(free, dtype=float)
    if mode == "anti18":
        # free[0] is self-antipodal: force z=0 then normalize
        free[0, 20] = 0.0
        return _norm(free)
    else:
        return _norm(free)

def _chain_grad(gradS, free, map_fn):
    mode = getattr(map_fn, "_mode", "free")
    if mode == "free":
        # full == free (up to normalization already handled by projection, approx identity)
        return gradS.copy()
    elif mode == "anti27":
        # full = [s, -s], s=free[0]; gradF0 = g0 - g1
        g0 = gradS[0]; g1 = gradS[1]
        return np.expand_dims(g0 - g1, axis=0)
    elif mode == "anti18":
        # full = [sA, sB, -sB]; free=[sA,sB]
        gA = gradS[0]; gB = gradS[1]; gC = gradS[2]
        gradF = np.empty_like(free)
        gradF[0] = gA
        gradF[1] = gB - gC
        # z of sA forced 0 -> zero out grad z
        gradF[0, 20] = 0.0
        return gradF
    else:
        return gradS.copy()

def solve():
    t0 = time.time()
    DEADLINE = 178.0
    CAP = 150.0
    TARGET = 0.0666634002447
    rng0 = np.random.RandomState(12345)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng0.randn(54, 21)
    xb = _norm(np.asarray(xb, dtype=float))
    if xb.shape != (54, 21):
        xb = _norm(rng0.randn(54, 21))
    best_val = _max_cos(xb)
    best = xb.copy()
    TARGET_EFF = min(best_val, TARGET)

    # multiplier choices
    m27_a = [1,2,3,4,5,6,7,8,9,10]
    m27_b = [1,2,4,5,7,8,10,11,13,14]
    m27_c = [1,3,5,7,9,11,13,15,17,19]
    m18_a = [1,2,3,4,5,6,7,8,9,10]
    m18_b = [1,3,5,7,9,11,13,15,17,1]
    m18_c = [1,5,7,11,13,17,3,9,15,2]

    def make_map(mode):
        if mode == "free":
            def f(fr):
                return _norm(fr), None
            f._mode = "free"
            return f
        elif mode == "anti27":
            def f(fr):
                s = _norm(fr)  # (1,21)
                full = np.vstack([s[0], -s[0]])
                # full rows already unit
                return full, None
            f._mode = "anti27"
            return f
        elif mode == "anti18":
            def f(fr):
                # fr (2,21), fr[0] z forced 0
                a = fr[0].copy(); a[20] = 0.0
                na = np.linalg.norm(a)
                if na == 0: a[0] = 1.0; na = 1.0
                a = a / na
                b = fr[1] / (np.linalg.norm(fr[1]) + 1e-12)
                full = np.vstack([a, b, -b])
                return full, None
            f._mode = "anti18"
            return f
        else:
            raise ValueError

    trials = []
    # (m, mults, mode, ns_free, steps, seed)
    trials.append((27, m27_a, "free", 2, 2200, 101))
    trials.append((27, m27_b, "free", 2, 2200, 102))
    trials.append((27, m27_c, "anti27", 1, 2200, 103))
    trials.append((27, m27_a, "anti27", 1, 2200, 104))
    trials.append((18, m18_a, "free", 3, 2000, 201))
    trials.append((18, m18_b, "free", 3, 2000, 202))
    trials.append((18, m18_b, "anti18", 2, 2200, 203))
    trials.append((18, m18_c, "free", 3, 2000, 204))
    trials.append((27, m27_b, "anti27", 1, 2000, 105))
    trials.append((18, m18_c, "anti18", 2, 2000, 205))
    trials.append((27, m27_c, "free", 2, 2000, 106))
    trials.append((18, m18_a, "anti18", 2, 2000, 206))
    # extra random mult trials if time remains
    rngM = np.random.RandomState(999)
    for k in range(6):
        m = 27 if k % 2 == 0 else 18
        if m == 27:
            mults = [int(x) for x in rngM.randint(1, 27, size=10)]
            if all(x % 3 == 0 for x in mults): mults[0] = 1
            mode = "free" if k % 4 == 0 else "anti27"
            nf = 2 if mode == "free" else 1
        else:
            mults = [int(x) for x in rngM.randint(1, 18, size=10)]
            if all(x % 2 == 0 for x in mults): mults[0] = 1
            mode = "free" if k % 4 == 0 else "anti18"
            nf = 3 if mode == "free" else 2
        trials.append((m, mults, mode, nf, 1800, 300 + k))

    for ti, (m, mults, mode, nf, steps, sd) in enumerate(trials):
        if time.time() - t0 > CAP:
            break
        if time.time() - t0 > DEADLINE - 12:
            break
        r = np.random.RandomState(sd)
        fmap = make_map(mode)
        free_init = r.randn(nf, 21)
        if mode == "anti18":
            free_init[0, 20] = 0.0
        try:
            Psym, vsym = _adam_seeds(free_init, fmap, np.array(mults, dtype=float), m,
                                     steps=steps, T0=0.0125, T1=0.001,
                                     lr0=0.02, lr1=0.003, rng=r)
        except Exception:
            continue
        # check valid orbit size
        if Psym.shape != (54, 21):
            continue
        # break symmetry with 1e-3 tangent noise
        noise = 1e-3 * r.randn(*Psym.shape)
        dot = np.sum(noise * Psym, axis=1, keepdims=True)
        noise = noise - dot * Psym
        Pb = _norm(Psym + noise)
        # quick screen: skip LP if symmetric+broken is hopelessly bad? still polish a couple
        # short LP polish
        remain = DEADLINE - 10 - (time.time() - t0)
        if remain < 1.0:
            break
        budget = min(2.0, remain)
        try:
            Pc = _lp_polish(Pb, seconds=budget, r0=5e-3)
        except Exception:
            continue
        v = _max_cos(Pc)
        if v < best_val - 1e-12 and np.all(np.isfinite(Pc)):
            best_val = v
            best = Pc.copy()
            TARGET_EFF = best_val
            # if we beat published target, give a bit more polish immediately
            if best_val < TARGET - 1e-12:
                rb = min(3.0, DEADLINE - 6 - (time.time() - t0))
                if rb > 1.0:
                    try:
                        best = _lp_polish(best, seconds=rb, r0=2e-3)
                        best_val = _max_cos(best)
                    except Exception:
                        pass
    # final polish of global best with remaining time
    remain = DEADLINE - (time.time() - t0)
    if remain > 2.0:
        try:
            cand = _lp_polish(best, seconds=max(2.0, remain - 1.0), r0=2e-3)
            vc = _max_cos(cand)
            if vc < best_val:
                best, best_val = cand, vc
        except Exception:
            pass
    best = _norm(best)
    return best
