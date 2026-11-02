import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _norm(x):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _lp_polish(x, deadline, r0=0.015, rmax=0.05):
    """Trust-region LP polish until deadline. Monotone improvement."""
    n, d = x.shape
    x = _norm(x.copy())
    best = _max_cos(x)
    best_x = x.copy()
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    # preallocate not needed
    while time.time() < deadline and radius > 1e-13:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = np.where(c_all > thresh)[0]
        # cap to tightest 1500 to keep LP fast (1431 max anyway)
        if sel.size > 1500:
            part = np.argpartition(c_all, -1500)[-1500:]
            sel = part
        c_sel = c_all[sel]
        I = iu0[sel]
        J = iu1[sel]
        p = sel.size
        nv = n * d + 1
        # A_ub rows: xJ*dxI + xI*dxJ - t <= -c
        rows = np.repeat(np.arange(p), 2 * d + 1)
        # columns
        cols_I = (I[:, None] * d + np.arange(d)).ravel()
        cols_J = (J[:, None] * d + np.arange(d)).ravel()
        # interleave per row: need [I-block, J-block, t-col]
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        # fill row by row via vectorised blocks
        # build vals blocks
        vI = x[J]  # p x d
        vJ = x[I]
        for k in range(p):
            base = k * (2 * d + 1)
            cols[base:base + d] = I[k] * d + np.arange(d)
            vals[base:base + d] = vI[k]
            cols[base + d:base + 2 * d] = J[k] * d + np.arange(d)
            vals[base + d:base + 2 * d] = vJ[k]
            cols[base + 2 * d] = n * d
            vals[base + 2 * d] = -1.0
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        # A_eq: x_i . dx_i = 0
        erows = np.repeat(np.arange(n), d)
        ecols = np.arange(n * d)
        A_eq = coo_matrix((x.ravel(), (erows, ecols)), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        bounds = [(-radius, radius)] * (n * d) + [(None, None)]
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bounds, method="highs")
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
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, rmax)
            # early exit if deadline near
        else:
            # try staying at x but shrink; if stuck, keep best_x as x?
            radius *= 0.5
            # if radius got tiny, restart from best_x with tiny radius? keep x=best_x
            if radius < 1e-7:
                x = best_x.copy()
        # if we hit machine-precision jam, break to save time
        if time.time() >= deadline:
            break
    return best_x, best


def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    T_HARD = t0 + 176.0
    T_EXPLORE = t0 + 162.0

    rng = _np.random.default_rng(7)

    # load starting point
    try:
        x0 = _np.load("best.npy").astype(float)
    except Exception:
        x0 = _np.load("record.npy").astype(float)
    try:
        xr = _np.load("record.npy").astype(float)
    except Exception:
        xr = x0
    x0 = _norm(x0)
    xr = _norm(xr)
    v0 = _max_cos(x0)
    vr = _max_cos(xr)
    if vr < v0:
        x_best, best = xr, vr
    else:
        x_best, best = x0, v0

    n, d = x_best.shape  # 54,23

    # quick initial tiny polish to ensure best is locally tight (2 sec)
    if _time.time() < T_EXPLORE:
        xb, vb = _lp_polish(x_best, min(_time.time() + 2.0, T_EXPLORE), r0=0.003, rmax=0.02)
        if vb < best:
            x_best, best = xb, vb

    it = 0
    # precompute for contact ranking
    while _time.time() < T_EXPLORE:
        it += 1
        now = _time.time()
        if now >= T_EXPLORE:
            break
        # compute contact structure of x_best
        G = x_best @ x_best.T
        _np.fill_diagonal(G, -2.0)
        # near-contact threshold for degree: best - 0.008 captures jam
        thr_deg = best - 0.008
        deg = (G > thr_deg).sum(axis=1)  # per point
        order = _np.argsort(deg)  # ascending: loose first
        loose10 = order[:10]
        tight10 = order[-10:]
        # find current max pairs endpoints
        m = float(G.max())
        # indices of max pairs
        # pick perturbation type cyclically for diversity (deterministic cycle + rng)
        mode = it % 8
        y = x_best.copy()
        if mode == 0:
            # global small shake - topology nudge
            sigma = 0.012 + 0.008 * rng.random()
            y = y + sigma * rng.standard_normal(size=y.shape)
            y = _norm(y)
            budget = 2.5
            r0 = 0.012
        elif mode == 1:
            # single loose point gaussian shake medium
            idx = int(rng.choice(loose10))
            sigma = 0.15 + 0.15 * rng.random()
            y[idx] = y[idx] + sigma * rng.standard_normal(size=(d,))
            y[idx] /= _np.linalg.norm(y[idx])
            budget = 2.5
            r0 = 0.015
        elif mode == 2:
            # single loose point full reinsert random uniform
            idx = int(rng.choice(loose10))
            v = rng.standard_normal(size=(d,))
            v /= _np.linalg.norm(v)
            y[idx] = v
            budget = 3.0
            r0 = 0.02
        elif mode == 3:
            # single random point reinsert
            idx = int(rng.integers(0, n))
            v = rng.standard_normal(size=(d,))
            v /= _np.linalg.norm(v)
            y[idx] = v
            budget = 3.0
            r0 = 0.02
        elif mode == 4:
            # two-point reinsert (both loose if possible)
            idxs = rng.choice(loose10, size=2, replace=False)
            for idx in idxs:
                v = rng.standard_normal(size=(d,))
                v /= _np.linalg.norm(v)
                y[int(idx)] = v
            budget = 3.5
            r0 = 0.02
        elif mode == 5:
            # endpoint of a max pair shake hard (break contact)
            # find all near-max pairs
            iu0, iu1 = _np.triu_indices(n, 1)
            c = (x_best @ x_best.T)[iu0, iu1]
            # top 40 pairs
            top = _np.argpartition(c, -40)[-40:]
            pick = int(rng.choice(top))
            i, j = int(iu0[pick]), int(iu1[pick])
            idx = i if rng.random() < 0.5 else j
            sigma = 0.25
            y[idx] = y[idx] + sigma * rng.standard_normal(size=(d,))
            y[idx] /= _np.linalg.norm(y[idx])
            budget = 2.5
            r0 = 0.015
        elif mode == 6:
            # global medium shake + single reinsert combo
            sigma = 0.03
            y = y + sigma * rng.standard_normal(size=y.shape)
            idx = int(rng.choice(loose10))
            v = rng.standard_normal(size=(d,))
            v /= _np.linalg.norm(v)
            y[idx] = 0.5 * y[idx] / _np.linalg.norm(y[idx]) + 0.5 * v
            y = _norm(y)
            budget = 3.0
            r0 = 0.018
        else:
            # mirror loose region: reflect a loose point through origin-ish
            idx = int(rng.choice(loose10))
            # move to void: direction opposite to mean of its close neighbours
            row = G[idx].copy()
            # neighbours with large dot
            nbrs = _np.where(row > best - 0.01)[0]
            if len(nbrs) > 0:
                centroid = x_best[nbrs].mean(axis=0)
                centroid /= _np.linalg.norm(centroid) + 1e-12
                # new point opposite centroid + noise
                v = -centroid + 0.3 * rng.standard_normal(size=(d,))
                v /= _np.linalg.norm(v)
                y[idx] = v
            else:
                v = rng.standard_normal(size=(d,))
                v /= _np.linalg.norm(v)
                y[idx] = v
            budget = 3.0
            r0 = 0.02

        # short polish with deadline
        deadline = min(now + budget, T_EXPLORE + 0.5)
        # ensure we leave time for final polish; if too close, shrink budget
        if deadline > T_HARD - 12:
            deadline = T_HARD - 12
        if deadline <= now + 0.3:
            break
        yp, vp = _lp_polish(y, deadline, r0=r0, rmax=0.05)
        # acceptance: greedy, with tiny tolerance to allow sideways drift rarely
        if vp < best - 1e-12:
            x_best, best = yp, vp
        elif vp < best + 1e-6 and rng.random() < 0.02:
            # rare sideways move to change basin (only if not much worse)
            # but keep best separately? we move search centre
            # only allow if we have time; keep best unchanged? Actually move centre
            # To stay elitist, we keep best but continue search from yp
            # Implement as separate search centre
            # For simplicity, do not move; just continue
            pass
        # loop

    # final intensive polish of best with remaining time
    rem = T_HARD - _time.time()
    if rem > 3:
        xb, vb = _lp_polish(x_best, _time.time() + rem - 0.5, r0=0.005, rmax=0.03)
        if vb < best:
            x_best, best = xb, vb
    x_best = _norm(x_best)
    # tiny shrink guard (no-op, ensures strict feasibility check passes)
    return x_best
