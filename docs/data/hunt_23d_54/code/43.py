import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _build_orbits(a, b, c, sel, IDX=None):
    # circulant orbits: A[k,j]=a[(j-k)%23]
    if IDX is None:
        r = np.arange(23)
        IDX = (r[None, :] - r[:, None]) % 23
    A = a[IDX]
    B = b[IDX]
    Cfull = c[IDX]
    C = Cfull[sel]
    X = np.concatenate([A, B, C], axis=0)
    return X, A, B, Cfull

def _select8(A, B, Cfull):
    # minimise max dot involving C: AC, BC, CC
    AC = A @ Cfull.T  # 23x23
    BC = B @ Cfull.T
    CC = Cfull @ Cfull.T
    np.fill_diagonal(CC, -2.0)
    # greedy
    rem = list(range(23))
    sel = []
    for _ in range(8):
        best_c = None
        best_v = 1e9
        for cand in rem:
            m = float(AC[:, cand].max())
            m2 = float(BC[:, cand].max())
            if m2 > m:
                m = m2
            if sel:
                m3 = float(CC[np.ix_(sel, [cand])].max()) if len(sel) > 0 else -2
                # also CC[cand, sel]
                if m3 > m:
                    m = m3
            if m < best_v:
                best_v = m
                best_c = cand
        sel.append(best_c)
        rem.remove(best_c)
    # hill swap improvement
    def sel_cost(s):
        s = list(s)
        m = float(AC[:, s].max())
        m = max(m, float(BC[:, s].max()))
        if len(s) > 1:
            m = max(m, float(CC[np.ix_(s, s)].max()))
        return m
    cur = sel_cost(sel)
    improved = True
    for _ in range(10):
        improved = False
        for i_in in range(8):
            for o in rem:
                trial = sel.copy()
                trial[i_in] = o
                v = sel_cost(trial)
                if v < cur - 1e-12:
                    sel = trial
                    # update rem
                    rem = [x for x in range(23) if x not in sel]
                    cur = v
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return np.array(sorted(sel), dtype=int)

def _optimize_triple(a0, b0, c0, sel0, steps=2000, lr=0.006, b1=0.9, b2=0.999, eps=1e-8):
    a = a0.copy(); b = b0.copy(); c = c0.copy()
    sel = sel0.copy()
    r = np.arange(23)
    IDX = (r[None, :] - r[:, None]) % 23
    GIDX = (r[None, :] + r[:, None]) % 23  # for grad gather: S[k,t]=grad[k,(t+k)%23]
    # Adam states
    ma = np.zeros(23); va = np.zeros(23)
    mb = np.zeros(23); vb = np.zeros(23)
    mc = np.zeros(23); vc = np.zeros(23)
    # initial select refine
    _, A0, B0, C0 = _build_orbits(a, b, c, sel, IDX)
    # keep best
    Xbest, _, _, _ = _build_orbits(a, b, c, sel, IDX)
    best = _max_cos(Xbest)
    best_a, best_b, best_c, best_sel = a.copy(), b.copy(), c.copy(), sel.copy()
    # temperature schedule log
    lb0 = np.log(50.0); lb1 = np.log(800.0)
    for t in range(1, steps + 1):
        if (t % 250 == 0) or (t == 1):
            # refresh selection every 250
            _, Acur, Bcur, Ccur = _build_orbits(a, b, c, sel, IDX)
            sel = _select8(Acur, Bcur, Ccur)
        frac = (t - 1) / max(steps - 1, 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * frac))
        X, A, B, Cf = _build_orbits(a, b, c, sel, IDX)
        # Cf is full 23x23, X uses subset; need grad only via X
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        Z = float(E.sum()) / 2.0
        if Z <= 0:
            Z = 1.0
        W = E / Z
        Gr = W @ X  # 54x23
        gA = Gr[0:23]
        gB = Gr[23:46]
        gC = Gr[46:54]  # 8 rows
        # map to seed grads
        # grad_a[t]=sum_k gA[k,(t+k)%23]
        Sa = np.take_along_axis(gA, GIDX, axis=1)
        ga = Sa.sum(axis=0)
        Sb = np.take_along_axis(gB, GIDX, axis=1)
        gb = Sb.sum(axis=0)
        # for c: scatter 8 rows into 23 with inverse rolls
        gc = np.zeros(23)
        # need per selected k: roll(gC_row, -k)
        for ii, k in enumerate(sel):
            row = gC[ii]
            # roll by -k: out[t]=row[(t+k)%23]
            gc += row[(GIDX[k])]
        # tangent projection
        ga = ga - np.dot(ga, a) * a
        gb = gb - np.dot(gb, b) * b
        gc = gc - np.dot(gc, c) * c
        # Adam updates
        ma = b1 * ma + (1 - b1) * ga
        va = b2 * va + (1 - b2) * (ga * ga)
        mb = b1 * mb + (1 - b1) * gb
        vb = b2 * vb + (1 - b2) * (gb * gb)
        mc = b1 * mc + (1 - b1) * gc
        vc = b2 * vc + (1 - b2) * (gc * gc)
        c1 = 1 - b1 ** t
        c2 = 1 - b2 ** t
        sa = (ma / c1) / (np.sqrt(va / c2) + eps)
        sb = (mb / c1) / (np.sqrt(vb / c2) + eps)
        sc = (mc / c1) / (np.sqrt(vc / c2) + eps)
        a = a - lr * sa
        b = b - lr * sb
        c = c - lr * sc
        na = np.linalg.norm(a); nb = np.linalg.norm(b); nc = np.linalg.norm(c)
        if na == 0 or nb == 0 or nc == 0:
            break
        a /= na; b /= nb; c /= nc
        if t % 100 == 0 or t == steps:
            Xc, _, _, _ = _build_orbits(a, b, c, sel, IDX)
            v = _max_cos(Xc)
            if v < best:
                best = v
                best_a, best_b, best_c, best_sel = a.copy(), b.copy(), c.copy(), sel.copy()
    Xbest, _, _, _ = _build_orbits(best_a, best_b, best_c, best_sel, IDX)
    return Xbest, best, (best_a, best_b, best_c, best_sel)

def _smooth(Xinit, steps, k0, k1, lr0, lr1):
    X = Xinit.copy()
    n = X.shape[0]
    lk0 = np.log(k0); lk1 = np.log(k1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        k = float(np.exp(lk0 + (lk1 - lk0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp(k * (G - m))
        np.fill_diagonal(E, 0.0)
        Z = float(E.sum()) / 2.0 + 1e-30
        W = E / Z
        F = W @ X
        dot = np.sum(F * X, axis=1, keepdims=True)
        Ft = F - dot * X
        # normalize step by rms
        rms = float(np.sqrt(np.sum(Ft * Ft) / n)) + 1e-12
        X = X - (lr / rms) * Ft * 0.05
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _lp_polish(X, seconds, t_end=None):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t_end is not None and time.time() > t_end:
            break
        if radius < 1e-14:
            break
        G = X @ X.T
        c_all = G[iu]
        m = float(c_all.max())
        sel = c_all > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) > 900:
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[::-1][:900]
            nsel = np.zeros_like(sel, dtype=bool)
            nsel[idx[order]] = True
            sel = nsel
        I, J = iu[0][sel], iu[1][sel]
        c = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel() if False else np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        Y = X + res.x[:-1].reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        v = _max_cos(Y)
        if v < best:
            X, best = Y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.RandomState(0)
    n, d = 54, 23
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(n, d)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    if xb.shape != (n, d):
        xb = rng.randn(n, d)
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        if xr.shape == (n, d) and _max_cos(xr) < _max_cos(xb):
            xb = xr.copy()
    except Exception:
        pass
    base_best = _max_cos(xb)
    x_best_overall = xb.copy()
    best_overall = base_best

    # Phase 1: symmetric orbit search, strict ~110s
    PHASE1_END = t0 + 110.0
    cands = []  # list of (val, X)
    ntrials = 0
    # at least 30 triples attempted
    while True:
        if ntrials >= 30 and time.time() > PHASE1_END:
            break
        if time.time() > PHASE1_END and ntrials >= 10:
            break
        if time.time() > BUDGET - 45.0:
            break
        if ntrials >= 40:
            break
        # random seed triple on sphere
        a0 = rng.randn(d); a0 /= np.linalg.norm(a0)
        b0 = rng.randn(d); b0 /= np.linalg.norm(b0)
        c0 = rng.randn(d); c0 /= np.linalg.norm(c0)
        sel0 = np.array(sorted(rng.choice(23, 8, replace=False)), dtype=int)
        # adaptive steps: full 2000 if time allows else shorter
        remain = PHASE1_END - (time.time())
        steps = 2000
        if remain < 30 and ntrials > 10:
            steps = 800
        try:
            Xb, vb, _ = _optimize_triple(a0, b0, c0, sel0, steps=steps)
        except Exception:
            ntrials += 1
            continue
        ntrials += 1
        cands.append((vb, Xb.copy()))
        if vb < best_overall:
            best_overall = vb
            x_best_overall = Xb.copy()
        # keep only top few to save memory
        cands = sorted(cands, key=lambda z: z[0])[:5]
    # Phase 2: gradual symmetry release with tangent noise 1e-3,3e-3,1e-2 + cheap smooth
    # build pool: top orbit cands + loaded best
    pool = []
    for vb, Xb in cands[:3]:
        pool.append(Xb.copy())
    if len(pool) == 0:
        pool.append(xb.copy())
    LP_RESERVE = 20.0
    noises = [1e-3, 3e-3, 1e-2]
    # cheap smooth after each release
    idx_rng = np.random.RandomState(1)
    for Xc in list(pool):
        if time.time() > BUDGET - LP_RESERVE - 2.0:
            break
        for nz in noises:
            if time.time() > BUDGET - LP_RESERVE - 2.0:
                break
            Y = Xc.copy()
            # tangent noise
            N = idx_rng.randn(n, d) * nz
            # project to tangent
            N = N - np.sum(N * Y, axis=1, keepdims=True) * Y
            Y = Y + N
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            try:
                Y2 = _smooth(Y, steps=350, k0=80, k1=2500, lr0=0.012, lr1=0.0008)
            except Exception:
                continue
            v2 = _max_cos(Y2)
            cands.append((v2, Y2.copy()))
            if v2 < best_overall:
                best_overall = v2
                x_best_overall = Y2.copy()
    # ensure we never lose the loaded record
    if base_best < best_overall:
        best_overall = base_best
        x_best_overall = xb.copy()
    # keep best distinct: sort
    cands = sorted(cands, key=lambda z: z[0])
    # Phase 3: final trust-region LP polish (one polish, ~20s)
    t_left = BUDGET - (time.time())
    lp_time = min(19.0, max(5.0, t_left - 1.0))
    if lp_time < 2.0:
        lp_time = 2.0
    try:
        Xpol = _lp_polish(x_best_overall.copy(), seconds=lp_time, t_end=t0 + BUDGET)
        vpol = _max_cos(Xpol)
        if vpol < best_overall:
            x_best_overall = Xpol.copy()
            best_overall = vpol
        else:
            # if polish failed to improve, keep smoothed best but ensure LP tried on record too briefly?
            pass
    except Exception:
        pass
    # final safety: if somehow worse than loaded best, return loaded best polished lightly
    if _max_cos(x_best_overall) > base_best + 1e-12:
        x_best_overall = xb.copy()
    x_best_overall = x_best_overall / np.linalg.norm(x_best_overall, axis=1, keepdims=True)
    return x_best_overall
