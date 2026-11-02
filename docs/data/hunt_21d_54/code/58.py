import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _proj(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, cap=1500):
    if seconds <= 0.1:
        return x
    x = _proj(np.asarray(x, dtype=np.float64))
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start >= seconds:
            break
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        I = iu0[0][sel]; J = iu0[1][sel]
        csv = c[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > cap:
            idx = np.argpartition(-csv, cap)[:cap]
            I, J = I[idx], J[idx]
            csv = csv[idx]
            p = cap
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        tlim = max(0.5, min(2.0, seconds))
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": tlim})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _proj(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def _feas_run(X0, tau, steps, lr0, T0, deadline):
    Y = _proj(np.array(X0, dtype=np.float64, copy=True))
    n, d = Y.shape
    m = np.zeros((n, d), dtype=np.float64)
    v = np.zeros((n, d), dtype=np.float64)
    b1 = 0.9; b2 = 0.999; ead = 1e-8
    b1p = 1.0; b2p = 1.0
    best_max = 1.0
    best_Y = None
    # initial max
    G0 = Y @ Y.T
    np.fill_diagonal(G0, -2.0)
    try:
        m0 = float(G0.max())
    except Exception:
        m0 = 1.0
    best_max = m0
    best_Y = Y.copy()
    inv_eps_factor = 0.0
    for s in range(steps):
        if (s & 511) == 0 and time.time() > deadline:
            break
        frac = s / float(steps)
        eps = 0.01 * (0.1 ** frac)  # 1e-2 -> 1e-3
        lr = lr0 * (1.0 - 0.6 * frac)
        # Gram
        G = Y @ Y.T
        np.fill_diagonal(G, -2.0)
        cur_max = float(G.max())
        if cur_max < best_max:
            best_max = cur_max
            best_Y = Y.copy()
        # D = (G - tau)/eps, clip in place to [-20,20]
        # reuse G buffer as D
        G -= tau
        G /= eps
        np.clip(G, -20.0, 20.0, out=G)
        E = np.exp(G)  # 54x54
        # SP = log1p(E), SIG = E/(1+E)
        # W = 2/eps * SP * SIG
        # compute in place to save allocs
        SP = np.log1p(E)
        SIG = E / (1.0 + E)
        # free E
        W = SP * SIG
        W *= (2.0 / eps)
        np.fill_diagonal(W, 0.0)
        Grad = W @ Y
        # Riemannian projection
        dot = np.einsum('ij,ij->i', Grad, Y)
        Grad -= dot[:, None] * Y
        # Adam
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1.0 - b1) * Grad
        v *= b2
        v += (1.0 - b2) * (Grad * Grad)
        mh = m / (1.0 - b1p)
        vh = v / (1.0 - b2p)
        upd = lr * mh / (np.sqrt(vh) + ead)
        Y = Y - upd
        # retract
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        Y /= nr
        # transport momentum: project m onto new tangent
        dotm = np.einsum('ij,ij->i', m, Y)
        m -= dotm[:, None] * Y
    return best_Y, float(best_max)

def solve():
    T0 = time.time()
    HARD = 150.0
    deadline = T0 + HARD
    rng = np.random.RandomState(0)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.randn(54, 21)
    xb = np.asarray(xb, dtype=np.float64)
    if xb.shape != (54, 21):
        flat = xb.ravel()
        need = 54 * 21
        if flat.size < need:
            extra = rng.randn(need - flat.size)
            flat = np.concatenate([flat, extra])
        xb = flat[:need].reshape(54, 21)
    xb = _proj(xb)
    try:
        rec = np.load("record.npy")
        rec = np.asarray(rec, dtype=np.float64)
        if rec.shape == (54, 21):
            rec = _proj(rec)
        else:
            rec = xb
    except Exception:
        rec = xb.copy()
    try:
        bv = _max_cos(xb)
    except Exception:
        bv = 1.0
    try:
        rv = _max_cos(rec)
    except Exception:
        rv = 1.0
    if rv < bv:
        best = rec.copy(); best_val = rv
    else:
        best = xb.copy(); best_val = bv
    incumbent = best.copy()
    inc_val = best_val

    taus = [0.062, 0.058, 0.054, 0.05]
    sigmas = [0.01, 0.03, 0.08, 0.20, 0.50]
    lrs = [0.02, 0.012, 0.03]

    # build seed plan: per tau: 5 perturbed + 5 random = 10 seeds -> 40 total
    # each ~3000 steps (<1s). Sample until ~110s.
    STEPS = 3000
    subrecords = []  # candidates below inc
    seed_idx = 0
    # allow two passes if time remains
    for rep in range(3):
        if time.time() > deadline - 30:
            break
        for ti, tau in enumerate(taus):
            if time.time() > deadline - 30:
                break
            for k in range(10):
                if time.time() > deadline - 30:
                    break
                # time check for remaining sampling budget
                # X0 construction deterministically
                if k < 5:
                    sig = sigmas[(k + rep) % len(sigmas)]
                    # perturbed record: use rec as base
                    noise = rng.randn(54, 21)
                    X0 = rec + sig * noise
                    X0 = _proj(X0)
                else:
                    X0 = _proj(rng.randn(54, 21))
                lr0 = lrs[(seed_idx + ti + rep) % len(lrs)]
                # small variation of steps on second reps
                st = STEPS if rep == 0 else 2000
                bY, bmax = _feas_run(X0, tau, st, lr0, T0, deadline - 30)
                seed_idx += 1
                if bY is None:
                    continue
                if bmax < inc_val:
                    # new basin (maybe below record)
                    inc_val = bmax
                    incumbent = _proj(bY)
                    subrecords.append((bmax, incumbent.copy(), tau))
                    # keep only best few
                    subrecords.sort(key=lambda t: t[0])
                    subrecords = subrecords[:4]
                elif bmax < 0.075:
                    # keep near-misses for possible polish if we have nothing better?
                    pass
                # early exit if clearly infeasible and time low? continue sampling
    # polish only genuinely new basins below 0.0666 (or below inc if inc already below)
    # per spec: seed that reaches max <0.0666 -> spend 10-20s LP
    cands = [t for t in subrecords if t[0] < 0.0666]
    if len(cands) == 0 and len(subrecords) > 0 and subrecords[0][0] < inc_val + 1e-12 and subrecords[0][0] < best_val:
        # we improved over published best even if not below 0.0666 threshold due to numeric? polish best one
        cands = subrecords[:1]
    # limit to top 2
    cands = cands[:2]
    for ci, (cv, cx, tau) in enumerate(cands):
        left = deadline - time.time()
        if left < 6:
            break
        budget = min(18.0, max(6.0, left - 4.0))
        # first candidate gets more time
        if ci == 0:
            budget = min(20.0, max(8.0, left - 4.0))
        else:
            budget = min(12.0, max(5.0, left - 3.0))
        px = _lp_polish(cx, budget, cap=1800)
        try:
            pv = _max_cos(px)
        except Exception:
            continue
        if pv < inc_val:
            inc_val = pv
            incumbent = _proj(px)
        if time.time() > deadline - 2:
            break
    # if no candidate polished (no sub-record basin), ensure we return lab best (already incumbent)
    # use any tiny remaining time to lightly squeeze incumbent? avoid LP attractor burn: only 2s max
    # do not burn budget: skip unless time ample and no polish done
    if len(cands) == 0:
        left = deadline - time.time()
        if left > 8:
            # brief safety polish of incumbent (2s) - cheap
            px = _lp_polish(incumbent, min(2.0, left - 3.0), cap=1500)
            try:
                pv = _max_cos(px)
                if pv < inc_val:
                    inc_val = pv
                    incumbent = _proj(px)
            except Exception:
                pass
    incumbent = _proj(np.nan_to_num(incumbent, nan=0.0, posinf=0.0, neginf=0.0))
    # final safety: no zero rows
    nr = np.linalg.norm(incumbent, axis=1, keepdims=True)
    bad = (~np.isfinite(nr.reshape(-1))) | (nr.reshape(-1) < 1e-12)
    if np.any(bad):
        rr = rng.randn(int(bad.sum()), 21)
        incumbent[bad] = _proj(rr)
    incumbent = _proj(incumbent)
    return incumbent
