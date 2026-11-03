import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _p_continuation(X0, ps, steps_list, lrs, shift=0.15, track_best=True):
    X = _norm(X0.copy())
    n, d = X.shape
    best_val = _max_cos(X)
    best_X = X.copy()
    # Adam state, reset per stage
    for si, (p, steps, lr) in enumerate(zip(ps, steps_list, lrs)):
        m = np.zeros_like(X)
        v = np.zeros_like(X)
        t = 0
        b1, b2, eps = 0.9, 0.999, 1e-8
        for s in range(steps):
            G = X @ X.T
            np.fill_diagonal(G, 0.0)  # exclude self before shift
            S = G - shift
            S[S < 0] = 0.0
            np.fill_diagonal(S, 0.0)
            smax = float(S.max())
            if (not np.isfinite(smax)) or smax <= 1e-300:
                break
            R = S / smax
            # powers; underflow to 0 is fine (inactive)
            with np.errstate(over='ignore', under='ignore'):
                Rp1 = np.power(R, p - 1.0, out=np.zeros_like(R), where=(R > 0))
                Rp = Rp1 * R
            # sum over i<j
            Ssum = float(np.triu(Rp, 1).sum())
            if (not np.isfinite(Ssum)) or Ssum <= 0:
                break
            scale = Ssum ** (1.0 / p - 1.0)
            A = Rp1 * scale
            # gradient: grad_i = sum_j A_ij x_j
            grad = A @ X
            dot = np.einsum('ij,ij->i', grad, X)[:, None]
            gt = grad - dot * X
            # Riemannian Adam
            t += 1
            m = b1 * m + (1.0 - b1) * gt
            v = b2 * v + (1.0 - b2) * (gt * gt)
            mh = m / (1.0 - b1 ** t)
            vh = v / (1.0 - b2 ** t)
            upd = lr * mh / (np.sqrt(vh) + eps)
            # project update to tangent before retract (keeps sphere better)
            d2 = np.einsum('ij,ij->i', upd, X)[:, None]
            updt = upd - d2 * X
            X = X - updt
            X = _norm(X)
            # re-project momentum to new tangent to avoid normal drift
            dm = np.einsum('ij,ij->i', m, X)[:, None]
            m = m - dm * X
            if track_best and (s % 10 == 0 or s == steps - 1):
                vv = _max_cos(X)
                if vv < best_val:
                    best_val = vv
                    best_X = X.copy()
        # rescale implicitly by resetting momentum; continue from current X
        # monitor effective active number (for adaptivity, not selection)
        # keep X as is for next p
    # final check
    fv = _max_cos(X)
    if fv < best_val:
        best_val = fv
        best_X = X.copy()
    return best_X, best_val, X, _max_cos(X)

def _smooth_lowT(X, steps, T0, T1, lr0, lr1, mu=0.85):
    n, d = X.shape
    iu0, iu1 = np.triu_indices(n, 1)
    V = np.zeros_like(X)
    W = np.empty((n, n))
    for k in range(steps):
        f = k / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** f if T0 != T1 else T0
        lr = lr0 * (lr1 / lr0) ** f if lr0 != lr1 else lr0
        G = X @ X.T
        c = G[iu0, iu1]
        m = c.max()
        w = np.exp((c - m) / max(T, 1e-15))
        s = w.sum()
        if s == 0 or not np.isfinite(s):
            w = np.zeros_like(w)
            w[int(np.argmax(c))] = 1.0
            s = 1.0
        w /= s
        W.fill(0.0)
        W[iu0, iu1] = w
        W[iu1, iu0] = w
        grad = W @ X
        dot = np.einsum('ij,ij->i', grad, X)[:, None]
        gt = grad - dot * X
        V = mu * V + (1.0 - mu) * gt
        dotv = np.einsum('ij,ij->i', V, X)[:, None]
        Vt = V - dotv * X
        X = X - lr * Vt
        X = _norm(X)
    return X

def _hard_topK(X, iters, step0, K):
    best = _max_cos(X)
    step = step0
    n, d = X.shape
    iu0, iu1 = np.triu_indices(n, 1)
    K = min(K, len(iu0))
    grad = np.empty_like(X)
    for _ in range(iters):
        G = X @ X.T
        c = G[iu0, iu1]
        if K < len(c):
            idx = np.argpartition(-c, K)[:K]
        else:
            idx = np.arange(len(c))
        I = iu0[idx]; J = iu1[idx]
        grad.fill(0.0)
        np.add.at(grad, I, X[J])
        np.add.at(grad, J, X[I])
        grad /= float(len(idx))
        dot = np.einsum('ij,ij->i', grad, X)[:, None]
        gt = grad - dot * X
        ng = float(np.linalg.norm(gt))
        if ng == 0 or not np.isfinite(ng):
            step *= 0.5
            if step < 1e-13:
                break
            continue
        Xt = X - step * gt
        Xt = _norm(Xt)
        vv = _max_cos(Xt)
        if vv < best:
            X = Xt
            best = vv
            step = min(step * 1.3, 0.02)
        else:
            step *= 0.5
            if step < 1e-13:
                break
    return X, best

def solve():
    t0 = time.time()
    BUDGET = 170.0
    rng = np.random.default_rng(0)
    n, d = 93, 13
    xb = None; xr = None
    try:
        a = np.load("best.npy")
        a = np.asarray(a, dtype=np.float64)
        if a.shape == (93, 13):
            xb = _norm(a)
    except Exception:
        xb = None
    try:
        a = np.load("record.npy")
        a = np.asarray(a, dtype=np.float64)
        if a.shape == (93, 13):
            xr = _norm(a)
    except Exception:
        xr = None
    if xb is None and xr is None:
        xb = _norm(rng.standard_normal((n, d)))
    base_best = xb if xb is not None else xr
    base_rec = xr if xr is not None else base_best

    # build initial pool
    inits = []
    inits.append(("best", base_best.copy()))
    if base_rec is not base_best:
        inits.append(("record", base_rec.copy()))
    else:
        # duplicate with tiny shake to diversify
        inits.append(("record_dup", _norm(base_best + 0.01 * rng.standard_normal((n, d)))))
    sigmas = [0.15, 0.3, 0.6, 0.3, 0.15, 0.6]
    bases = [base_best, base_best, base_best, base_rec, base_rec, base_best]
    for sg, bs in zip(sigmas, bases):
        inits.append((f"pert_{sg}", _norm(bs + sg * rng.standard_normal((n, d)))))
    # two fresh random
    inits.append(("rand1", _norm(rng.standard_normal((n, d)))))
    inits.append(("rand2", _norm(rng.standard_normal((n, d)))))

    ps = [8, 16, 32, 64, 128, 256]
    steps_list = [350, 350, 350, 350, 350, 350]
    lrs = [3.0e-3, 2.0e-3, 1.2e-3, 7.0e-4, 4.5e-4, 3.0e-4]

    results = []  # (val, bestX, finalX)
    # quick evaluate inits to order? run continuation for each with time check
    for name, X0 in inits:
        if time.time() - t0 > BUDGET - 60:
            break
        bc, bv, fin, fv = _p_continuation(X0, ps, steps_list, lrs, shift=0.15, track_best=True)
        # keep best-tracked version
        results.append((bv, bc, fv, fin, name))
    if len(results) == 0:
        return _norm(base_best)
    # sort by tracked best max
    results.sort(key=lambda t: t[0])
    # promote best 3
    Kpromo = min(3, len(results))
    promo = results[:Kpromo]
    # global best
    gbest_val = results[0][0]
    gbest = results[0][1].copy()

    # hard-max / active-set polish for promoted
    polished = []
    for k in range(Kpromo):
        if time.time() - t0 > BUDGET - 25:
            break
        _, Xc, _, _, _ = promo[k]
        X = Xc.copy()
        # low-T softmax to equalize then greedy
        remain = BUDGET - (time.time() - t0)
        # allocate roughly equal
        if remain < 10:
            break
        X = _smooth_lowT(X, steps=900, T0=4e-4, T1=4e-5, lr0=0.012, lr1=0.002, mu=0.85)
        X, v = _hard_topK(X, iters=600, step0=0.002, K=400)
        # second round tighter
        X = _smooth_lowT(X, steps=500, T0=1.2e-4, T1=2e-5, lr0=0.006, lr1=0.001, mu=0.85)
        X, v = _hard_topK(X, iters=600, step0=0.001, K=200)
        X, v = _hard_topK(X, iters=400, step0=0.0005, K=120)
        polished.append((v, X))
        if v < gbest_val:
            gbest_val = v
            gbest = X.copy()
    # if time remains, try one more p-continuation seeded from gbest with higher p tail
    if time.time() - t0 < BUDGET - 15:
        try:
            ps2 = [32, 64, 128, 256, 512]
            steps2 = [250, 250, 250, 300, 300]
            lrs2 = [1.2e-3, 7e-4, 4.5e-4, 3e-4, 2e-4]
            bc2, bv2, _, _ = _p_continuation(gbest, ps2, steps2, lrs2, shift=0.15)
            if bv2 < gbest_val:
                gbest_val = bv2
                gbest = bc2.copy()
            # quick final squeeze
            gbest = _smooth_lowT(gbest, steps=400, T0=1e-4, T1=2e-5, lr0=0.005, lr1=0.001)
            gbest, gbest_val = _hard_topK(gbest, iters=500, step0=0.001, K=200)
        except Exception:
            pass
    # also consider all polished + unpolished best
    for v, X in polished:
        if v < gbest_val:
            gbest_val = v
            gbest = X.copy()
    gbest = _norm(gbest)
    return gbest
