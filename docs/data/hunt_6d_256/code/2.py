import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, seconds, radius_init=0.003, verbose=False):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    bx = x.copy()
    radius = radius_init
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    # preallocate? just loop
    while True:
        if time.time() - start >= seconds:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c_all = G[iu0]
        m = c_all.max()
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        # cap size for speed
        cnt = int(np.count_nonzero(sel))
        if cnt > 2500:
            # keep top 2500
            idx = np.argpartition(-c_all, 2500)[:2500]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
            cnt = 2500
        if cnt == 0:
            radius *= 0.5
            continue
        I = iu0[0][sel]
        J = iu0[1][sel]
        csel = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        # columns
        ar = np.arange(d)
        cols_I = (I[:, None] * d + ar).ravel()
        cols_J = (J[:, None] * d + ar).ravel()
        # Build cols array
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        cols[0::(2*d+1)] = cols_I[0::d]  # placeholder, easier to build directly
        # simpler: build with concatenate
        cols = np.concatenate([ (I[:, None]*d + ar).reshape(-1),
                                (J[:, None]*d + ar).reshape(-1),
                                np.full(p, n*d)], axis=0)
        # rows need matching order: first p*d for I-block? Let's build block-wise
        # Rebuild rows block-wise to match cols order above:
        # vals order: x[J] block, x[I] block, -1
        rows_I = np.repeat(np.arange(p), d)
        rows_J = np.repeat(np.arange(p), d)
        rows_T = np.arange(p)
        rows = np.concatenate([rows_I, rows_J, rows_T], axis=0)
        vals = np.concatenate([x[J].reshape(-1), x[I].reshape(-1), -np.ones(p)], axis=0)
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if np.any(~np.isfinite(yn)) or np.any(yn[:,0] < 1e-12):
            radius *= 0.5
            continue
        y = y / yn
        v = max_cos(y)
        if v < best:
            best = v
            bx = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            # accept move if it doesn't worsen too much and allows progress? keep x at bx? 
            # For polishing, move x even if equal to explore plateau
            if v <= max_cos(x) + 1e-15:
                x = y
                radius *= 0.5
            else:
                radius *= 0.5
        # early stop if time exceeded
    # return best found
    return bx, best

def contacts_info(x, best_val=None, tol=1e-9):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = G.max() if best_val is None else best_val
    C = G >= (m - tol)
    # per point counts (each contact counted once per endpoint)
    per = C.sum(axis=1)
    return G, m, per

def solve():
    t0 = time.time()
    TIME_TOTAL = 178.0
    # load best
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            rng0 = np.random.default_rng(0)
            xb = rng0.standard_normal((256,6))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (256,6):
        # try reshape / truncate
        rng0 = np.random.default_rng(0)
        xb = rng0.standard_normal((256,6))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    xb = np.nan_to_num(xb)
    # ensure finite
    best_val = max_cos(xb)
    rng = np.random.default_rng(20261103)

    # initial short polish to lock local optimum (5s)
    remain = TIME_TOTAL - (time.time()-t0)
    init_t = min(6.0, max(1.0, remain - 150.0))
    # if plenty time, do 5s
    if remain > 60:
        init_t = 5.0
    bx, bv = lp_polish(xb, seconds=init_t, radius_init=0.002)
    if bv < best_val:
        xb, best_val = bx, bv
    else:
        xb, best_val = bx, bv  # bx is polished
    x_best = xb.copy()

    it = 0
    # precompute for reinsert sampling pool? generate on fly
    while time.time() - t0 < TIME_TOTAL - 22.0:
        it += 1
        remain = TIME_TOTAL - 22.0 - (time.time()-t0)
        if remain <= 0:
            break
        r = rng.random()
        y = None
        if r < 0.50:
            # global shake
            sd = float(rng.choice(np.array([0.008,0.015,0.025,0.04,0.07,0.12])))
            y = x_best + sd * rng.standard_normal((256,6))
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            budget = 1.4
        elif r < 0.75:
            # subset shake
            k = int(rng.integers(3, 30))
            idx = rng.choice(256, size=k, replace=False)
            sd = float(rng.uniform(0.08, 0.45))
            y = x_best.copy()
            y[idx] = y[idx] + sd * rng.standard_normal((k,6))
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            budget = 1.4
        else:
            # delete worst / rattler and reinsert via far-point sampling
            G, m, per = contacts_info(x_best, best_val, tol=1e-9)
            # pick 1-2 points to replace: prefer max contacts, sometimes rattler
            if rng.random() < 0.3:
                # rattlers
                ratt = np.where(per == 0)[0]
                if len(ratt) > 0:
                    k = min(len(ratt), int(rng.integers(1,3)))
                    rep = rng.choice(ratt, size=k, replace=False)
                else:
                    k = int(rng.integers(1,3))
                    rep = np.argsort(-per)[:k]
            else:
                k = int(rng.integers(1,3))
                # worst points
                order = np.argsort(-per)
                # add randomness: pick among top 20
                pool = order[:20]
                rep = rng.choice(pool, size=k, replace=False)
            y = x_best.copy()
            # for each rep, find far point w.r.t. remaining
            for w in rep:
                mask = np.ones(256, dtype=bool); mask[w]=False
                rest = y[mask]  # 255x6
                # sample candidates
                ncand = 12000
                cand = rng.standard_normal((ncand,6))
                cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
                dots = cand @ rest.T  # ncand x 255
                md = dots.max(axis=1)
                # also consider antipodal? score is max dot, so minimize md
                # pick best of candidates, with small random tie-break among top few
                bi = int(np.argmin(md))
                # optionally refine best candidate by local search: few projected gradient steps to minimize max dot
                bc = cand[bi].copy()
                # quick local refine: 30 steps
                for _ in range(40):
                    d = rest @ bc  # 255
                    j = int(np.argmax(d))
                    g = rest[j] - (rest[j] @ bc)*bc  # tangent of max
                    step = 0.05
                    bc = bc - step*g
                    bc = bc/np.linalg.norm(bc)
                y[w] = bc
            # add tiny noise to others to break symmetry
            y = y + 0.003*rng.standard_normal((256,6))
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            budget = 1.6
        # polish
        # adapt budget to remaining
        budget = min(budget, max(0.4, TIME_TOTAL-22.0-(time.time()-t0)))
        if budget < 0.35:
            break
        yp, vp = lp_polish(y, seconds=budget, radius_init=0.003)
        if vp < best_val - 1e-12:
            x_best = yp
            best_val = vp
            # print progress (allowed)
            # print(f"iter {it} new best {best_val}")
        # else discard
        # occasional time check
        if time.time()-t0 > TIME_TOTAL-22.0:
            break

    # final long polish on best
    remain = TIME_TOTAL - (time.time()-t0) - 1.0
    if remain > 2.0:
        fp, fv = lp_polish(x_best, seconds=min(remain, 20.0), radius_init=0.002)
        if fv < best_val:
            x_best, best_val = fp, fv
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
