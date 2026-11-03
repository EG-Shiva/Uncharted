import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

REC = 0.244097856927

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, seconds, radius_init=1e-3, cap=1600):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = float(radius_init)
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.05:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = c > thresh
        if int(sel.sum()) > cap:
            idx = np.argpartition(c, -cap)[-cap:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def get_duals(x, radius=1e-3):
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    c = (x @ x.T)[iu]
    m = float(c.max())
    sel = c > m - 1e-9
    I, J = iu[0][sel], iu[1][sel]
    p = len(I)
    if p == 0:
        return I, J, c[sel], None
    nv = n * d + 1
    ar = np.arange(d)
    rows = np.repeat(np.arange(p), 2 * d + 1)
    cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], 1).ravel()
    vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
    A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
    A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
    obj = np.zeros(nv); obj[-1] = 1.0
    try:
        res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                      bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
    except Exception:
        return I, J, c[sel], None
    if res.status != 0:
        return I, J, c[sel], None
    marg = None
    try:
        # scipy HiGHS marginals
        if hasattr(res, 'ineqlin') and res.ineqlin is not None:
            if hasattr(res.ineqlin, 'marginals'):
                marg = np.asarray(res.ineqlin.marginals, dtype=float).ravel()
            elif hasattr(res.ineqlin, 'dual'):
                marg = np.asarray(res.ineqlin.dual, dtype=float).ravel()
        if marg is None and hasattr(res, 'marginals'):
            marg = np.asarray(res.marginals, dtype=float).ravel()
    except Exception:
        marg = None
    if marg is not None and len(marg) != p:
        # may include eq marginals; truncate to first p
        try:
            marg = marg[:p]
        except Exception:
            marg = None
    return I, J, c[sel], marg

def adam_block(Xfull, S, target, steps, lr, scale=1e-4, seed_noise=0.02, rng=None):
    n, d = Xfull.shape
    k = len(S)
    Xs = Xfull[S].copy()
    if rng is not None and seed_noise > 0:
        Xs = Xs + seed_noise * rng.standard_normal(Xs.shape)
        Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
    mt = np.zeros_like(Xs)
    vt = np.zeros_like(Xs)
    b1, b2, eps = 0.9, 0.999, 1e-8
    # precompute fixed part for speed (Xfull changes only in S rows)
    for t in range(1, steps + 1):
        # current full matrix with updated block
        # D = Xs @ Xfull.T where Xfull's S rows = Xs
        # build temp full
        Xf = Xfull.copy()
        Xf[S] = Xs
        D = Xs @ Xf.T  # k x n
        # zero self pairs
        for a, gi in enumerate(S):
            D[a, gi] = -10.0
        Z = (D - target) / scale
        # stable sigmoid + softplus not needed for grad
        Zc = np.clip(Z, -20, 50)
        Sig = 1.0 / (1.0 + np.exp(-Zc))
        # ignore very negative for speed/sparsity
        Sig[Z < -12] = 0.0
        W = Sig / scale  # k x n
        Grad = W @ Xf  # k x d
        # tangent projection
        dot = np.sum(Grad * Xs, axis=1, keepdims=True)
        Gt = Grad - dot * Xs
        # Adam
        mt = b1 * mt + (1 - b1) * Gt
        vt = b2 * vt + (1 - b2) * (Gt * Gt)
        mhat = mt / (1 - b1 ** t)
        vhat = vt / (1 - b2 ** t)
        step = mhat / (np.sqrt(vhat) + eps)
        Xs = Xs - lr * step
        Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
    Xout = Xfull.copy()
    Xout[S] = Xs
    Xout /= np.linalg.norm(Xout, axis=1, keepdims=True)
    return Xout

def solve():
    t0 = time.time()
    DEADLINE = t0 + 150.0
    HARD = t0 + 172.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    base = None
    if xb is not None and xb.shape == (95, 13):
        base = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    elif xr is not None and xr.shape == (95, 13):
        base = xr / np.linalg.norm(xr, axis=1, keepdims=True)
    else:
        tmp = rng.standard_normal((95, 13))
        base = tmp / np.linalg.norm(tmp, axis=1, keepdims=True)
    # fallback copy
    fallback = base.copy()
    best_val = max_cos(base)
    # one LP polish
    remain = DEADLINE - (time.time() - t0) - 60.0
    init_sec = min(8.0, max(3.0, remain)) if remain > 0 else 2.0
    try:
        base = lp_polish(base, init_sec)
    except Exception:
        pass
    bv = max_cos(base)
    if bv < best_val:
        best_val = bv
        fallback = base.copy()
    # keep global best (only accept < REC per spec, but track best anyway)
    x_best = fallback.copy()
    x_best_val = max_cos(x_best)
    # extract dual-stress marginals for slack<1e-9
    try:
        CI, CJ, Cc, marg = get_duals(base)
    except Exception:
        CI, CJ, Cc, marg = None, None, None, None
    n = 95
    if CI is None or len(CI) == 0:
        iu = np.triu_indices(n, 1)
        g = (base @ base.T)[iu]
        m = float(g.max())
        sel = g > m - 1e-9
        CI, CJ = iu[0][sel], iu[1][sel]
        Cc = g[sel]
        marg = None
    p = len(CI)
    if marg is None or len(marg) != p or not np.all(np.isfinite(marg)):
        # proxy stress: smaller slack => higher stress; low-stress = larger slack
        # compute slack
        m0 = float(Cc.max())
        slack = m0 - Cc
        stress = -slack  # low slack = high stress; rank ascending for low stress? invert
        # we want low-stress = large slack => sort by slack descending
        order = np.argsort(-slack)  # large slack first? actually -slack ascending = large slack first
        # use uniform pseudo-marg
        absm = np.abs(slack) + 1e-18
        # rank: low stress first
        rank_idx = np.argsort(-slack)
    else:
        absm = np.abs(marg)
        rank_idx = np.argsort(absm)  # small |dual| first = low stress
    # sorted contact list low-stress first
    sI = CI[rank_idx]; sJ = CJ[rank_idx]
    targets = [0.2438, 0.2433, 0.2425]
    surg = 0
    # precompute contact counts for diversity
    while time.time() < DEADLINE - 8.0 and time.time() < HARD - 12.0:
        surg += 1
        # choose 10-30 low-stress contacts involving 8-20 points
        # sample from bottom pool (low stress). pool size grows with surg to diversify
        pool = min(p, 120 + (surg % 5) * 40)
        # low-stress pool = first `pool` in sorted order
        pI = sI[:pool]; pJ = sJ[:pool]
        # random choose nc contacts
        nc = int(rng.integers(10, 31))
        if pool < nc:
            nc = pool
        pick = rng.choice(pool, size=nc, replace=False)
        qI = pI[pick]; qJ = pJ[pick]
        pts = np.unique(np.concatenate([qI, qJ]))
        # adjust to 8-20 points
        if len(pts) < 8:
            # add random points from pool neighbours
            extra_pool = np.unique(np.concatenate([pI, pJ]))
            cand = np.setdiff1d(extra_pool, pts)
            rng.shuffle(cand)
            need = min(8 - len(pts), len(cand))
            pts = np.concatenate([pts, cand[:need]])
        if len(pts) > 20:
            # keep those with fewest global contacts? trim randomly
            rng.shuffle(pts)
            pts = pts[:20]
        # occasionally include rattler (0-contact point)
        if surg % 4 == 0:
            # find current rattler w.r.t base
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            mc = G.max()
            cnt = (G > mc - 1e-9).sum(axis=1)
            ratt = np.where(cnt == 0)[0]
            if len(ratt) > 0 and len(pts) < 20:
                r = int(ratt[rng.integers(0, len(ratt))])
                if r not in pts:
                    pts = np.append(pts, r)
        S = np.sort(pts.astype(int))
        # feasibility with frozen remainder
        Xtry = base.copy()
        # step through targets with Adam
        # adapt steps to remaining time
        tleft = DEADLINE - (time.time() - t0) - 5.0
        if tleft < 4.0:
            break
        # steps per target: smaller for speed
        steps_per = 500 if tleft > 40 else 350
        lr = 0.004
        noise = 0.02 + 0.01 * (surg % 3)
        try:
            for tt in targets:
                if time.time() > DEADLINE - 5.0:
                    break
                Xtry = adam_block(Xtry, S, tt, steps_per, lr, scale=1e-4, seed_noise=noise if tt == targets[0] else 0.0, rng=rng)
        except Exception:
            continue
        # unfreeze all, LP-polish short
        polish_sec = 2.5
        if tleft < 15:
            polish_sec = 1.2
        try:
            Xpol = lp_polish(Xtry, polish_sec)
        except Exception:
            continue
        v = max_cos(Xpol)
        # keep only if beats record
        if v < REC and v < x_best_val:
            x_best = Xpol.copy()
            x_best_val = v
            base = Xpol.copy()  # move base to new basin
            # refresh duals
            try:
                CI2, CJ2, Cc2, marg2 = get_duals(base)
                if CI2 is not None and len(CI2) > 0:
                    CI, CJ, Cc, marg = CI2, CJ2, Cc2, marg2
                    p = len(CI)
                    if marg is None or len(marg) != p or not np.all(np.isfinite(marg)):
                        m0 = float(Cc.max()); slack = m0 - Cc
                        rank_idx = np.argsort(-slack)
                    else:
                        absm = np.abs(marg)
                        rank_idx = np.argsort(absm)
                    sI = CI[rank_idx]; sJ = CJ[rank_idx]
            except Exception:
                pass
        # else discard (keep base unchanged) to stay in original basin per spec
        if surg > 40:
            break
    # final safety: return best beating record else fallback/record
    if x_best_val < REC:
        out = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    else:
        out = fallback / np.linalg.norm(fallback, axis=1, keepdims=True)
    # strict validity
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(out, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    out = out / nr
    return out
