import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contacts(X, tol=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cnt = (G >= m - tol).sum(axis=1)
    return m, cnt, G

def _smooth_anneal(X, iters=1200, beta0=25.0, beta1=350.0, step0=0.03, step1=0.002):
    n, d = X.shape
    Y = X.copy()
    # precompute ratio logs
    lb0, lb1 = np.log(beta0), np.log(beta1)
    ls0, ls1 = np.log(step0), np.log(step1)
    for it in range(iters):
        frac = it / max(1, iters - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * frac))
        step = float(np.exp(ls0 + (ls1 - ls0) * frac))
        G = Y @ Y.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        # weights focused on near-max; cut far pairs for stability
        # shift by m to avoid overflow
        W = np.exp(beta * (G - m))
        # sparsify very small weights
        W[W < 1e-6] = 0.0
        np.fill_diagonal(W, 0.0)
        # tangent gradient: sum_j w_ij (Xj - dot*Xi)
        WX = W @ Y
        s = (W * G).sum(axis=1, keepdims=True)
        grad = WX - s * Y
        gn = np.linalg.norm(grad, axis=1, keepdims=True).max()
        if not np.isfinite(gn) or gn == 0:
            continue
        Y = Y - (step / gn) * grad
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def _find_hole(X_keep, rng, n_cand=20000, refine=250):
    d = X_keep.shape[1]
    C = rng.normal(size=(n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = C @ X_keep.T
    md = dots.max(axis=1)
    idx = int(np.argmin(md))
    c = C[idx].copy()
    # local refinement: move away from nearest neighbours
    beta = 120.0
    step = 0.02
    for _ in range(refine):
        v = X_keep @ c  # (m,)
        m = float(v.max())
        w = np.exp(beta * (v - m))
        w[w < 1e-6] = 0.0
        # gradient of weighted sum wrt c = X_keep.T @ w
        g = w @ X_keep  # (d,)
        # project to tangent at c
        g = g - float(g @ c) * c
        gn = float(np.linalg.norm(g))
        if gn == 0 or not np.isfinite(gn):
            break
        c = c - (step / gn) * g
        nrm = float(np.linalg.norm(c))
        if nrm == 0:
            break
        c = c / nrm
        step *= 0.995
    nrm = float(np.linalg.norm(c))
    if nrm == 0 or not np.isfinite(nrm):
        v = rng.normal(size=(d,))
        v /= np.linalg.norm(v)
        return v
    return c / np.linalg.norm(c)

def _lp_polish(X, seconds, radius0=1e-3):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = radius0
    start = time.time()
    iu = np.triu_indices(n, 1)
    iu0, iu1 = iu
    while time.time() - start < seconds and radius > 1e-14:
        G = X @ X.T
        cfull = G[iu0, iu1]
        m = float(cfull.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = cfull > thresh
        if int(sel.sum()) > 3000:
            idx = np.where(sel)[0]
            # keep tightest 3000
            part = np.argpartition(cfull[idx], len(idx) - 3000)[-3000:]
            mask = np.zeros_like(sel)
            mask[idx[part]] = True
            sel = mask
        I = iu0[sel]; J = iu1[sel]; c = cfull[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        cols = np.concatenate([(I[:, None] * d + np.arange(d)),
                               (J[:, None] * d + np.arange(d)),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline = t0 + BUDGET
    rng = np.random.default_rng(12345)
    n, d = 173, 5
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (n, d):
                an = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(an)
                if v < best_val:
                    best_val = v
                    best = an.copy()
        except Exception:
            pass
    if best is None:
        best = rng.normal(size=(n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)

    def time_left():
        return deadline - time.time()

    trial = 0
    # reserve ~10s for final polish
    while time_left() > 12.0:
        trial += 1
        _, cnt, Gfull = _contacts(best)
        order_desc = np.argsort(-cnt)
        order_asc = np.argsort(cnt)
        Y = best.copy()
        mode = trial % 9
        if mode == 0:
            # reinsert single worst into global hole
            victim = int(order_desc[0])
            keep = np.delete(np.arange(n), victim)
            hole = _find_hole(best[keep], rng, n_cand=22000, refine=200)
            Y[victim] = hole
            # small shake of neighbours
            nbrs = np.argsort(-Gfull[victim])[:4]
            Y[nbrs] += 0.01 * rng.normal(size=(len(nbrs), d))
        elif mode == 1:
            # reinsert 2 worst sequentially
            victims = [int(order_desc[0]), int(order_desc[1])]
            keep = best.copy()
            # temporarily remove victims for hole search
            mask_keep = np.ones(n, dtype=bool); mask_keep[victims] = False
            Xk = best[mask_keep]
            for vi in victims:
                hole = _find_hole(Xk, rng, n_cand=16000, refine=150)
                Y[vi] = hole
                Xk = np.vstack([Xk, hole[None, :]])
        elif mode == 2:
            # reinsert 3: worst + 2 random among top-15
            cand = order_desc[:15]
            victims = [int(order_desc[0])] + [int(x) for x in rng.choice(cand[1:], 2, replace=False)]
            mask_keep = np.ones(n, dtype=bool); mask_keep[victims] = False
            Xk = best[mask_keep]
            for vi in victims:
                hole = _find_hole(Xk, rng, n_cand=14000, refine=120)
                Y[vi] = hole
                Xk = np.vstack([Xk, hole[None, :]])
        elif mode == 3:
            # kick jammed cluster medium
            k = 6
            subset = order_desc[:k]
            Y[subset] += 0.06 * rng.normal(size=(k, d))
        elif mode == 4:
            # global small shake
            Y += 0.018 * rng.normal(size=(n, d))
        elif mode == 5:
            # random 8 points medium kick
            idx = rng.choice(n, 8, replace=False)
            Y[idx] += 0.10 * rng.normal(size=(8, d))
        elif mode == 6:
            # randomize 2 points fully + shake rattler neighbourhood
            idx = rng.choice(n, 2, replace=False)
            Y[idx] = rng.normal(size=(2, d))
            # also nudge rattlers (0-contact points) slightly
            rats = np.where(cnt == 0)[0]
            if len(rats) > 0:
                Y[rats] += 0.02 * rng.normal(size=(len(rats), d))
        elif mode == 7:
            # larger global shake
            Y += 0.045 * rng.normal(size=(n, d))
        else:
            # move worst to near rattler void + small global noise
            rats = np.where(cnt <= 1)[0]
            victim = int(order_desc[0])
            if len(rats) > 0:
                r = best[int(rng.choice(rats))]
                v = r + 0.15 * rng.normal(size=(d,))
                v /= np.linalg.norm(v)
                Y[victim] = v
            else:
                Y[victim] = rng.normal(size=(d,))
            Y += 0.008 * rng.normal(size=(n, d))
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)

        # cheap smooth anneal to new basin
        # adapt iters to time left
        rem = time_left()
        if rem < 12.5:
            break
        iters = 1100 if (trial % 3) else 1800
        # vary beta range a bit for diversity
        if trial % 4 == 0:
            Ys = _smooth_anneal(Y, iters=iters, beta0=15.0, beta1=250.0, step0=0.04, step1=0.002)
        else:
            Ys = _smooth_anneal(Y, iters=iters, beta0=30.0, beta1=400.0, step0=0.025, step1=0.0015)
        v_smooth = _max_cos(Ys)
        # decide whether to spend LP time
        # always LP if better than best after smoothing, else LP occasionally if close
        do_lp = False
        lp_time = 0.7
        if v_smooth < best_val:
            do_lp = True
            lp_time = 1.2 if rem > 30 else 0.7
        elif v_smooth < best_val + 0.004 and (trial % 3 == 0):
            do_lp = True
            lp_time = 0.6
        elif v_smooth < best_val + 0.010 and (trial % 7 == 0):
            do_lp = True
            lp_time = 0.5
        if do_lp:
            if time_left() < lp_time + 11.0:
                # not enough time, keep smooth if it is best
                if v_smooth < best_val:
                    best, best_val = Ys, v_smooth
                continue
            Yp = _lp_polish(Ys, seconds=min(lp_time, time_left() - 11.0))
            vp = _max_cos(Yp)
            if vp < best_val:
                best, best_val = Yp, vp
            elif v_smooth < best_val:
                best, best_val = Ys, v_smooth
            # elif accept slightly worse smooth as diverse restart? no, keep elitist
        else:
            if v_smooth < best_val:
                # polish lightly to confirm
                Yp = _lp_polish(Ys, seconds=min(0.8, time_left() - 11.0))
                vp = _max_cos(Yp)
                if vp < best_val:
                    best, best_val = Yp, vp
                else:
                    best, best_val = Ys, v_smooth
        # occasional random-restart exploration with smoothing only (no LP) to avoid wasting time
        if trial % 11 == 0 and time_left() > 40:
            R = rng.normal(size=(n, d)); R /= np.linalg.norm(R, axis=1, keepdims=True)
            Rs = _smooth_anneal(R, iters=1500, beta0=20.0, beta1=300.0, step0=0.04, step1=0.002)
            vs = _max_cos(Rs)
            if vs < best_val + 0.005:
                Rp = _lp_polish(Rs, seconds=min(0.9, time_left() - 11.0))
                vp2 = _max_cos(Rp)
                if vp2 < best_val:
                    best, best_val = Rp, vp2

    # final squeeze on best
    rem = deadline - time.time()
    if rem > 2.0:
        best = _lp_polish(best, seconds=max(1.0, rem - 1.0))
        best_val = _max_cos(best)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
