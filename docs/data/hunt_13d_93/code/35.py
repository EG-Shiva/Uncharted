import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _p_continuation(X0, ps=(8, 16, 32, 64, 128, 256), steps_per=300,
                     lrs=None, shift=0.15, seed_t=0, deadline=1e18):
    n, d = X0.shape
    if lrs is None:
        lrs = [3e-3, 2e-3, 1.2e-3, 7e-4, 5e-4, 3e-4]
    X = X0.copy()
    best_X = X.copy()
    best_v = _maxcos(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for pi, p in enumerate(ps):
        lr = lrs[pi] if pi < len(lrs) else 3e-4
        m = np.zeros_like(X)
        v = np.zeros_like(X)
        tt = 0
        for s in range(steps_per):
            if time.time() > deadline:
                break
            # Gram + shifted slacks
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            C = G - shift
            cmax = float(C.max())
            if cmax <= 1e-12 or not np.isfinite(cmax):
                break
            R = C / cmax
            # clip negatives, zero diag
            np.fill_diagonal(R, 0.0)
            R[R < 0] = 0.0
            # cap >1 due to numerics
            # powers
            # use exp form for stability at large p? direct power is fine
            with np.errstate(under='ignore', over='ignore'):
                Rp = np.power(R, p, out=np.zeros_like(R), where=(R > 0))
                W = np.power(R, p - 1, out=np.zeros_like(R), where=(R > 0))
            np.fill_diagonal(Rp, 0.0)
            np.fill_diagonal(W, 0.0)
            S = float(Rp.sum() * 0.5)
            if S <= 0 or not np.isfinite(S):
                break
            scale = S ** (1.0 / p - 1.0)
            if not np.isfinite(scale):
                scale = 1.0
            WX = W @ X
            Grad = scale * WX
            dot = np.sum(Grad * X, axis=1, keepdims=True)
            Gt = Grad - dot * X
            # Riemannian Adam
            tt += 1
            m = b1 * m + (1.0 - b1) * Gt
            v = b2 * v + (1.0 - b2) * (Gt * Gt)
            mh = m / (1.0 - b1 ** tt)
            vh = v / (1.0 - b2 ** tt)
            step = lr * mh / (np.sqrt(vh) + eps)
            # project step to tangent
            d2 = np.sum(step * X, axis=1, keepdims=True)
            step_t = step - d2 * X
            # optional clip
            X = X - step_t
            X = _norm(X)
            if (s % 50 == 49) or (s == steps_per - 1):
                vv = _maxcos(X)
                if vv < best_v and np.all(np.isfinite(X)):
                    best_v = vv
                    best_X = X.copy()
        # end stage: rescale implicitly by resetting moments (done)
        # monitor Neff could be computed but just continue
        if time.time() > deadline:
            break
    return best_X, best_v, X

def _adam_softmax(X, steps, tau0, tau1, lr0, lr1, deadline=1e18):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    ltau = np.log(tau1 / tau0) if tau0 != tau1 and tau0 > 0 else 0.0
    llr = np.log(lr1 / lr0) if lr0 != lr1 and lr0 > 0 else 0.0
    tt = 0
    for s in range(steps):
        if (s % 100 == 0) and time.time() > deadline:
            break
        f = s / max(1, steps - 1)
        tau = tau0 * np.exp(ltau * f)
        lr = lr0 * np.exp(llr * f)
        if tau < 1e-12:
            tau = 1e-12
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mg = float(G.max())
        W = np.exp((G - mg) / tau)
        np.fill_diagonal(W, 0.0)
        grad = W @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        tt += 1
        m = b1 * m + (1.0 - b1) * gt
        vv = b2 * v + (1.0 - b2) * (gt * gt)
        mh = m / (1.0 - b1 ** tt)
        vh = vv / (1.0 - b2 ** tt)
        step = lr * mh / (np.sqrt(vh) + eps)
        d2 = np.sum(step * X, axis=1, keepdims=True)
        step_t = step - d2 * X
        X = X - step_t
        X = _norm(X)
    return X

def _topK_descent(X, iters, K, step0, deadline=1e18):
    n, d = X.shape
    iu0, iu1 = np.triu_indices(n, 1)
    L = len(iu0)
    K = int(min(K, L))
    best = _maxcos(X)
    bestX = X.copy()
    step = step0
    grad = np.empty_like(X)
    for it in range(iters):
        if (it % 50 == 0) and time.time() > deadline:
            break
        G = X @ X.T
        c = G[iu0, iu1]
        if K < L:
            idx = np.argpartition(-c, K)[:K]
        else:
            idx = np.arange(L)
        I = iu0[idx]
        J = iu1[idx]
        grad.fill(0.0)
        np.add.at(grad, I, X[J])
        np.add.at(grad, J, X[I])
        grad /= float(len(idx))
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        ng = float(np.linalg.norm(gt))
        if ng == 0 or not np.isfinite(ng):
            step *= 0.5
            if step < 1e-12:
                break
            continue
        Xt = X - step * gt
        Xt = _norm(Xt)
        vv = _maxcos(Xt)
        if vv < best:
            best = vv
            bestX = Xt.copy()
            X = Xt
            step = min(step * 1.2, 0.01)
        else:
            step *= 0.5
            if step < 1e-12:
                break
    return bestX, best

def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline_all = t0 + BUDGET
    # time allocation: continuation until ~120s, polish rest
    cont_deadline = t0 + 115.0
    rng = np.random.default_rng(12345)
    n, d = 93, 13
    loaded = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (93, 13):
                a = _norm(a)
                loaded.append(a)
        except Exception:
            pass
    if len(loaded) == 0:
        loaded = [_norm(rng.standard_normal((n, d)))]
    base_best = None
    base_val = 1e9
    for a in loaded:
        va = _maxcos(a)
        if va < base_val:
            base_val = va
            base_best = a.copy()
    # build initial configs: best, record, perturbed, random with sigma in {0.15,0.3,0.6}
    inits = []
    inits.append(base_best.copy())
    # add record separately if exists and different
    for a in loaded:
        if np.mean((a - base_best) ** 2) > 1e-12:
            inits.append(a.copy())
            break
    # perturbed versions
    sigmas = [0.15, 0.3, 0.6, 0.15, 0.3]
    for si, sig in enumerate(sigmas):
        pert = base_best + sig * rng.standard_normal((n, d))
        pert = _norm(pert)
        inits.append(pert)
        if len(inits) >= 7:
            break
    # one pure random
    inits.append(_norm(rng.standard_normal((n, d))))
    # limit to 7 inits
    inits = inits[:7]

    results = []  # (val, bestX, finalX)
    # give each init roughly equal time; fewer steps if many inits
    for ii, X0 in enumerate(inits):
        if time.time() > cont_deadline - 2.0:
            break
        # vary steps slightly for diversity, deterministic
        spp = 300 if ii < 4 else 250
        try:
            bX, bv, fX = _p_continuation(X0, steps_per=spp, deadline=cont_deadline)
        except Exception:
            continue
        # also consider final state value
        try:
            fv = _maxcos(fX)
        except Exception:
            fv = 1e9
        # keep the better of best-tracked and final
        if fv < bv:
            bv = fv
            bX = fX.copy()
        results.append((bv, bX.copy(), fX.copy()))
        # effective active pairs for info (not used to prune yet)
        # keep going

    if len(results) == 0:
        X = _norm(base_best.copy())
        return X
    # sort by value, promote best 3
    results.sort(key=lambda t: t[0])
    # global best so far
    gbest_val = results[0][0]
    gbest_X = results[0][1].copy()
    # include base if better (should not happen)
    if base_val < gbest_val:
        gbest_val = base_val
        gbest_X = base_best.copy()

    top = results[:3]
    # polish each of top 3 with softmax + topK, keep improvements
    for ti, (val, bX, fX) in enumerate(top):
        if time.time() > deadline_all - 8.0:
            break
        X = bX.copy()
        # stage 1: softmax anneal
        try:
            X = _adam_softmax(X, steps=700, tau0=0.004, tau1=0.0004,
                              lr0=0.002, lr1=0.0004,
                              deadline=deadline_all - 6.0)
        except Exception:
            pass
        # stage 2: topK sequence to tie plateau
        try:
            for K, it_, st in [(500, 120, 0.003), (200, 120, 0.002), (60, 150, 0.0015), (25, 100, 0.001)]:
                if time.time() > deadline_all - 5.0:
                    break
                X, _ = _topK_descent(X, iters=it_, K=K, step0=st, deadline=deadline_all - 4.0)
        except Exception:
            pass
        # stage 3: tiny softmax finish
        try:
            X = _adam_softmax(X, steps=250, tau0=0.0006, tau1=0.0002,
                              lr0=0.0006, lr1=0.0002,
                              deadline=deadline_all - 2.0)
        except Exception:
            pass
        try:
            vv = _maxcos(X)
        except Exception:
            continue
        if vv < gbest_val and np.all(np.isfinite(X)):
            gbest_val = vv
            gbest_X = X.copy()

    # final safety: if polish failed, return best continuation
    gbest_X = _norm(gbest_X)
    gbest_X = np.nan_to_num(gbest_X, nan=0.0, posinf=0.0, neginf=0.0)
    # avoid zero rows
    nr = np.linalg.norm(gbest_X, axis=1, keepdims=True)
    bad = (nr[:, 0] < 1e-12)
    if np.any(bad):
        gbest_X[bad] = _norm(rng.standard_normal((int(bad.sum()), d)))
    gbest_X = _norm(gbest_X)
    return gbest_X
