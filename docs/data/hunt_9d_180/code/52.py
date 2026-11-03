import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _polish_minmax(x, budget, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu0, iu1 = np.triu_indices(n, 1)
    t0 = time.time()
    while time.time() - t0 < budget and time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu0, iu1]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu0[sel], iu1[sel]
        cs = c[sel]
        p = len(I)
        if p == 0:
            break
        # cap to keep LP cheap
        if p > 2500:
            idx = np.argpartition(-cs, 2500)[:2500]
            I, J, cs = I[idx], J[idx], cs[idx]
            p = 2500
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def solve():
    import time as _tm
    import numpy as _np
    from scipy.optimize import linprog as _lp
    from scipy.sparse import coo_matrix as _coo
    t0 = _tm.time()
    HARD_SEARCH = t0 + 150.0
    HARD_END = t0 + 176.0
    rng = _np.random.RandomState(0)
    # load start
    try:
        xb = _np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = _np.load("record.npy")
    except Exception:
        xr = None
    cand = None
    bv = 1e9
    for c in (xb, xr):
        try:
            if c is None:
                continue
            c = _np.asarray(c, dtype=float)
            if c.shape != (180, 9):
                continue
            c = c / _np.linalg.norm(c, axis=1, keepdims=True)
            if not _np.all(_np.isfinite(c)):
                continue
            v = _max_cos(c)
            if v < bv:
                bv = v
                cand = c.copy()
        except Exception:
            continue
    if cand is None:
        cand = rng.randn(180, 9)
        cand /= _np.linalg.norm(cand, axis=1, keepdims=True)
        bv = _max_cos(cand)
    n, d = 180, 9
    x_cur = cand.copy()
    x_best = cand.copy()
    b_best = bv
    N = n * d
    iu0, iu1 = _np.triu_indices(n, 1)
    targets = [0.4620, 0.4615, 0.4610]
    radii = [0.01, 0.02, 0.03]
    # rotating exclusion: pairs excluded for next iteration
    excluded_next = set()
    noimp = 0
    # cycle targets
    outer = 0
    while _tm.time() < HARD_SEARCH and _tm.time() < HARD_END - 25:
        outer += 1
        for ti, tgt in enumerate(targets):
            for ri, rad in enumerate(radii):
                # 2 inner feasibility iterations per (t,r)
                for inner in range(2):
                    if _tm.time() >= HARD_SEARCH or _tm.time() >= HARD_END - 25:
                        break
                    try:
                        G = x_cur @ x_cur.T
                        vals = G[iu0, iu1]
                        thr = tgt - 0.01
                        mask = vals > thr
                        # apply rotating exclusion
                        if len(excluded_next) > 0:
                            # build boolean to drop excluded pairs: need index lookup
                            # excluded_next holds (i,j) tuples; convert to mask filtering
                            # for speed, only check candidates
                            cI = iu0[mask]
                            cJ = iu1[mask]
                            keep = _np.ones(len(cI), dtype=bool)
                            # small set, linear scan via set
                            for k in range(len(cI)):
                                if (int(cI[k]), int(cJ[k])) in excluded_next:
                                    keep[k] = False
                            # if everything excluded, ignore exclusion
                            if _np.any(keep):
                                # rebuild mask indices
                                idx_all = _np.where(mask)[0][keep]
                                nmask = _np.zeros_like(mask)
                                nmask[idx_all] = True
                                mask = nmask
                        cnt = int(_np.count_nonzero(mask))
                        if cnt == 0:
                            # feasible for this threshold window: try lower target directly
                            excluded_next = set()
                            continue
                        sI = iu0[mask]
                        sJ = iu1[mask]
                        sV = vals[mask]
                        # cap at 800 tightest (largest cosine)
                        if cnt > 800:
                            part = _np.argpartition(-sV, 800)[:800]
                            sI = sI[part]
                            sJ = sJ[part]
                            sV = sV[part]
                            cnt = 800
                        # random excluded-contact subset: drop ~5% randomly to diversify
                        # deterministic via rng
                        if cnt > 50:
                            nd = min(30, cnt // 10)
                            drop = rng.choice(cnt, size=nd, replace=False)
                            keepm = _np.ones(cnt, dtype=bool)
                            keepm[drop] = False
                            sI = sI[keepm]
                            sJ = sJ[keepm]
                            sV = sV[keepm]
                            cnt = len(sI)
                        p = cnt
                        nv = N + p
                        rows = _np.repeat(_np.arange(p), 2 * d + 1)
                        cIcol = (sI[:, None] * d + _np.arange(d)).ravel()
                        cJcol = (sJ[:, None] * d + _np.arange(d)).ravel()
                        cols = _np.empty(p * (2 * d + 1), dtype=_np.int64)
                        cols[0::(2 * d + 1)] = 0  # placeholder
                        # build cols vectorised
                        base = _np.empty((p, 2 * d + 1), dtype=_np.int64)
                        base[:, :d] = sI[:, None] * d + _np.arange(d)
                        base[:, d:2 * d] = sJ[:, None] * d + _np.arange(d)
                        base[:, 2 * d] = N + _np.arange(p)
                        cols = base.ravel()
                        vI = x_cur[sJ].ravel()
                        vJ = x_cur[sI].ravel()
                        vvals = _np.empty((p, 2 * d + 1), dtype=float)
                        vvals[:, :d] = x_cur[sJ]
                        vvals[:, d:2 * d] = x_cur[sI]
                        vvals[:, 2 * d] = -1.0
                        vflat = vvals.ravel()
                        A_ub = _coo((vflat, (rows, cols)), shape=(nv if False else (p, nv))).tocsr()
                        A_eq = _coo((x_cur.ravel(), (_np.repeat(_np.arange(n), d), _np.arange(N))), shape=(n, nv)).tocsr()
                        bub = tgt - sV
                        cobj = _np.zeros(nv)
                        cobj[N:] = 1.0
                        bnds = [(-rad, rad)] * N + [(0, None)] * p
                        res = _lp(cobj, A_ub=A_ub, b_ub=bub, A_eq=A_eq, b_eq=_np.zeros(n), bounds=bnds, method="highs")
                        if res.status != 0:
                            excluded_next = set()
                            # small random kick to escape
                            x_cur = x_cur + 0.003 * rng.randn(n, d)
                            x_cur /= _np.linalg.norm(x_cur, axis=1, keepdims=True)
                            continue
                        dx = res.x[:N].reshape(n, d)
                        svals = res.x[N:]
                        y = x_cur + dx
                        y /= _np.linalg.norm(y, axis=1, keepdims=True)
                        vy = _max_cos(y)
                        if vy < b_best:
                            b_best = vy
                            x_best = y.copy()
                            noimp = 0
                        else:
                            noimp += 1
                        x_cur = y
                        # identify most stressed contacts via primal violation s and duals if available
                        try:
                            marg = None
                            if hasattr(res, "ineqlin") and res.ineqlin is not None and hasattr(res.ineqlin, "marginals"):
                                marg = _np.asarray(res.ineqlin.marginals, dtype=float)
                        except Exception:
                            marg = None
                        # choose up to 25 stressed to exclude next few iterations (rotating)
                        kex = min(25, p)
                        if marg is not None and len(marg) == p and _np.all(_np.isfinite(marg)):
                            # duals negative for active; most stressed = most negative
                            o = _np.argsort(marg)[:kex]
                        else:
                            o = _np.argsort(-svals)[:kex]
                        # only exclude truly stressed (s>1e-12 or active)
                        new_ex = set()
                        for k in o:
                            # include if violation significant or always include top few
                            new_ex.add((int(sI[k]), int(sJ[k])))
                        # rotate: keep only new set for next iteration (deliberately break them)
                        excluded_next = new_ex
                        # occasional kick if stuck: perturb rattler-like points
                        if noimp > 12:
                            Gc = x_cur @ x_cur.T
                            _np.fill_diagonal(Gc, -2.0)
                            mx = _np.max(Gc, axis=1)
                            # points with small max (loose) get kicked
                            ordp = _np.argsort(mx)[:10]
                            x_cur[ordp] += 0.01 * rng.randn(10, d)
                            x_cur /= _np.linalg.norm(x_cur, axis=1, keepdims=True)
                            noimp = 0
                            excluded_next = set()
                    except Exception:
                        try:
                            excluded_next = set()
                        except Exception:
                            pass
                        continue
                if _tm.time() >= HARD_SEARCH:
                    break
            if _tm.time() >= HARD_SEARCH:
                break
        # safety: if time low break
        if outer > 200:
            break
    # final ordinary LP polish on best
    remain = min(HARD_END - _tm.time() - 1.0, 22.0)
    if remain > 2.0:
        try:
            xp, vp = _polish_minmax(x_best, remain, HARD_END - 0.5)
            if vp < b_best:
                x_best = xp
                b_best = vp
        except Exception:
            pass
    x_best = x_best / _np.linalg.norm(x_best, axis=1, keepdims=True)
    # tiny shrink guard not needed but ensure finite
    x_best = _np.where(_np.isfinite(x_best), x_best, 0.0)
    nn = _np.linalg.norm(x_best, axis=1, keepdims=True)
    nn[nn == 0] = 1.0
    x_best = x_best / nn
    return x_best
