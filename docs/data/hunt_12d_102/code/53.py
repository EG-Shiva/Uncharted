import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, t_abs_end):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_abs_end:
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        mcur = float(c.max())
        sel = c > mcur - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y /= nr
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _fingerprints(Xb, K=250):
    # Xb: (B,N,D)
    B, N, D = Xb.shape
    G = np.matmul(Xb, Xb.transpose(0, 2, 1))
    iu = np.triu_indices(N, 1)
    vals = G[:, iu[0], iu[1]]  # (B,P)
    mc = np.empty(B)
    for b in range(B):
        v = vals[b]
        mc[b] = float(v.max())
    # sorted descending fingerprints
    fp = np.sort(vals, axis=1)[:, ::-1][:, :K]
    return mc, fp

def solve():
    t0 = time.time()
    HARD_END = t0 + 165.0
    BATCH_END = t0 + 142.0
    LP_END = HARD_END - 1.0
    rng = np.random.default_rng(0)
    N, D, B = 102, 12, 24
    # load incumbents
    try:
        xb = np.array(np.load("best.npy"), dtype=float)
    except Exception:
        xb = None
    try:
        xr = np.array(np.load("record.npy"), dtype=float)
    except Exception:
        xr = None
    def _norm(a):
        a = np.array(a, dtype=float)
        nr = np.linalg.norm(a, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        return a / nr
    cands = []
    for c in (xb, xr):
        if c is not None and c.shape == (N, D):
            try:
                a = _norm(c)
                cands.append((_max_cos(a), a))
            except Exception:
                pass
    if len(cands) == 0:
        a = rng.standard_normal((N, D))
        a = _norm(a)
        x_inc = a
        best_inc = _max_cos(a)
    else:
        cands.sort(key=lambda t: t[0])
        best_inc, x_inc = cands[0][0], cands[0][1].copy()
    # also keep second distinct if available
    x_rec = None
    if len(cands) > 1:
        x_rec = cands[1][1]
    # ---- build (B,N,D) population ----
    X = np.empty((B, N, D))
    X[0] = x_inc
    if x_rec is not None:
        X[1] = x_rec
    else:
        X[1] = _norm(x_inc + 0.01 * rng.standard_normal((N, D)))
    sigmas = [0.005, 0.008, 0.012, 0.02, 0.03, 0.045, 0.06, 0.08, 0.10, 0.12, 0.15, 0.02, 0.04, 0.07, 0.09, 0.11, 0.06, 0.025, 0.035]
    # fill 2..B-1: mostly perturbed, last 4 random
    for b in range(2, B):
        if b >= B - 4:
            a = rng.standard_normal((N, D))
            X[b] = _norm(a)
        else:
            s = sigmas[(b - 2) % len(sigmas)]
            X[b] = _norm(x_inc + s * rng.standard_normal((N, D)))
    # vectorised Riemannian Adam state
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-12
    p_sched = [4, 8, 16, 32, 64]
    steps_sched = [400, 400, 500, 600, 800]
    lr_sched = [0.008, 0.008, 0.006, 0.005, 0.004]
    gt = 0
    idxN = np.arange(N)
    abort_batch = False
    for si, p in enumerate(p_sched):
        steps = steps_sched[si]
        lr = lr_sched[si]
        for s in range(steps):
            if (s % 20 == 0) and (time.time() > BATCH_END):
                abort_batch = True
                break
            gt += 1
            # continuation objective sum ((1+G)/2)^p
            G = np.matmul(X, X.transpose(0, 2, 1))
            # clip for stability
            # G in [-1,1]; A in [0,1]
            A = (1.0 + G) * 0.5
            # avoid tiny negative due to numerics
            # A**(p-1)
            # use power directly
            Fp = (p * 0.5) * np.power(A, p - 1)
            Fp[:, idxN, idxN] = 0.0
            Grad = np.matmul(Fp, X)  # (B,N,D)
            # Riemannian projection
            dot = np.sum(Grad * X, axis=2, keepdims=True)
            Gt = Grad - dot * X
            m = b1 * m + (1.0 - b1) * Gt
            v = b2 * v + (1.0 - b2) * (Gt * Gt)
            c1 = 1.0 - b1 ** gt
            c2 = 1.0 - b2 ** gt
            mh = m / c1
            vh = v / c2
            upd = mh / (np.sqrt(vh) + eps)
            # project update to tangent
            ud = np.sum(upd * X, axis=2, keepdims=True)
            upd_t = upd - ud * X
            X = X - lr * upd_t
            nr = np.linalg.norm(X, axis=2, keepdims=True)
            nr[nr == 0] = 1.0
            X /= nr
            # safety: no NaN
            # (rare) if NaN appears, reset that member
            # check infrequently
            if (s % 100 == 0) and (not np.all(np.isfinite(X))):
                bad = ~np.all(np.isfinite(X), axis=(1, 2))
                for b in np.where(bad)[0]:
                    X[b] = _norm(x_inc + 0.05 * rng.standard_normal((N, D)))
                    m[b] = 0.0
                    v[b] = 0.0
        if abort_batch or time.time() > BATCH_END:
            break
        # ---- periodic basin fingerprint dedup ----
        try:
            mc, fp = _fingerprints(X, K=250)
        except Exception:
            continue
        order = np.argsort(mc)  # ascending: better first
        tol = 8e-4 if si < 3 else 4e-4
        kept = []
        dups = []
        for bi in order:
            is_dup = False
            for k in kept:
                if float(np.max(np.abs(fp[bi] - fp[k]))) < tol:
                    is_dup = True
                    break
            if is_dup:
                dups.append(int(bi))
            else:
                kept.append(int(bi))
        # reinitialise duplicates with fresh scaled perturbations of distinct elites
        # retain distinct elites untouched
        if len(dups) > 0:
            # best elite reference
            # use random kept elite as parent for each dup
            for di in dups:
                if time.time() > BATCH_END:
                    break
                parent = kept[int(rng.integers(len(kept)))] if len(kept) > 0 else int(order[0])
                sigma = float(rng.uniform(0.05, 0.15))
                X[di] = _norm(X[parent] + sigma * rng.standard_normal((N, D)))
                m[di] = 0.0
                v[di] = 0.0
    # ---- select best distinct candidates by true max-cos ----
    try:
        mc_all, fp_all = _fingerprints(X, K=300)
    except Exception:
        mc_all = np.array([_max_cos(X[b]) for b in range(B)])
        fp_all = None
    # ensure incumbent included
    # evaluate exact max cos (already in mc_all)
    order = np.argsort(mc_all)
    distinct = []
    tol_f = 4e-4
    for bi in order:
        dup = False
        for k in distinct:
            if fp_all is not None:
                if float(np.max(np.abs(fp_all[bi] - fp_all[k]))) < tol_f:
                    dup = True
                    break
            else:
                if abs(float(mc_all[bi]) - float(mc_all[k])) < 1e-12:
                    # fallback: check Gram directly
                    pass
        if not dup:
            distinct.append(int(bi))
        if len(distinct) >= 6:
            # keep top 6 distinct, but LP only top few
            pass
    # best verified overall (include file incumbent)
    best_val = best_inc
    best_x = x_inc.copy()
    for bi in order:
        vv = float(mc_all[bi])
        if vv < best_val:
            best_val = vv
            best_x = (X[bi] / np.linalg.norm(X[bi], axis=1, keepdims=True)).copy()
    # candidates for LP: top distinct (up to 3) + best_x already
    lp_cands = []
    for k in distinct[:3]:
        lp_cands.append((float(mc_all[k]), (X[k] / np.linalg.norm(X[k], axis=1, keepdims=True)).copy()))
    # dedup against best_x: keep anyway, LP will decide
    # sort
    lp_cands.sort(key=lambda t: t[0])
    # ---- final LP polishing only, within final ~20s ----
    remain = LP_END - time.time()
    if remain > 1.5 and len(lp_cands) > 0:
        per = remain / len(lp_cands)
        per = min(max(per, 1.0), 8.0)
        for (vv, xc) in lp_cands:
            if time.time() > LP_END - 0.5:
                break
            budget = min(per, LP_END - time.time() - 0.3)
            if budget < 0.6:
                break
            try:
                yp = _lp_polish(xc, budget, LP_END)
                vp = _max_cos(yp)
                if vp < best_val:
                    best_val = vp
                    best_x = yp
            except Exception:
                continue
    # also if no improvement, ensure we return best verified (file or batch)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    # final safety: replace NaN / zero rows
    if not np.all(np.isfinite(best_x)):
        best_x = x_inc.copy()
    nr = np.linalg.norm(best_x, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    best_x = best_x / nr
    return best_x
