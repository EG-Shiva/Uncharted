import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _smooth_refine(x0, iters, a0, a1, lr, deadline=None):
    x = x0.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    # initial best
    G0 = x @ x.T
    np.fill_diagonal(G0, -2.0)
    best_v = float(G0.max())
    best_x = x.copy()
    b1p = 1.0
    b2p = 1.0
    # precompute ratio for alpha exp interp
    logr = np.log(a1 / a0) if a0 > 0 and a1 > 0 else 0.0
    for it in range(iters):
        frac = it / (iters - 1) if iters > 1 else 0.0
        alpha = a0 * np.exp(logr * frac)
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        mval = float(G.max())
        # weights, max-subtracted for stability
        W = np.exp(alpha * (G - mval))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad -= dot * x
        b1p *= b1
        b2p *= b2
        # adam
        m *= b1
        m += (1.0 - b1) * grad
        v *= b2
        v += (1.0 - b2) * (grad * grad)
        mhat_denom = 1.0 - b1p
        vhat_denom = 1.0 - b2p
        # step = lr * (m/mhat_denom) / (sqrt(v/vhat_denom)+eps)
        # compute in place to save allocs
        mh = m / mhat_denom
        vh = v / vhat_denom
        step = mh / (np.sqrt(vh) + eps)
        step *= lr
        x -= step
        # renormalize
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        # guard against zero
        nr[nr == 0] = 1.0
        x /= nr
        if (it + 1) % 200 == 0 or it == iters - 1:
            Gc = x @ x.T
            np.fill_diagonal(Gc, -2.0)
            cur = float(Gc.max())
            if cur < best_v:
                best_v = cur
                best_x = x.copy()
            if deadline is not None and (it + 1) % 400 == 0:
                if time.time() > deadline:
                    break
    return best_x, best_v


def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    nv = n * d + 1
    while time.time() - start < seconds and radius > 1e-14:
        xxT = x @ x.T
        cfull = xxT[iu]
        m = float(cfull.max())
        thr = m - max(20.0 * radius, 1e-9)
        selmask = cfull > thr
        # truncate to keep linprog fast
        p0 = int(np.count_nonzero(selmask))
        if p0 == 0:
            radius *= 0.5
            continue
        if p0 > 3500:
            # keep top 3500
            idx_all = np.where(selmask)[0]
            vals = cfull[idx_all]
            part = np.argpartition(vals, -3500)[-3500:]
            keep = np.zeros_like(cfull, dtype=bool)
            keep[idx_all[part]] = True
            selmask = keep
        I = iu[0][selmask]
        J = iu[1][selmask]
        csel = cfull[selmask]
        p = len(I)
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols_I = I[:, None] * d + np.arange(d)
        cols_J = J[:, None] * d + np.arange(d)
        cols = np.concatenate([cols_I, cols_J, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x = y
            best = vv
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
        if time.time() - start > seconds:
            break
    return x


def solve():
    t0 = time.time()
    hard_deadline = t0 + 177.0
    explore_deadline = t0 + 148.0
    rng = np.random.RandomState(7)
    np.random.seed(7)

    # load incumbents
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (182, 9):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                nr[nr == 0] = 1.0
                a = a / nr
                # remove nan
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        # random fallback
        a = rng.randn(182, 9)
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    bestX = None
    bestV = 1e9
    for a in cands:
        vv = _max_cos(a)
        if vv < bestV:
            bestV = vv
            bestX = a.copy()

    sigmas = [0.008, 0.02, 0.045, 0.09, 0.15, 0.03, 0.012, 0.06]
    trial = 0
    # quick initial smooth on incumbent to get local best fast
    # (helps if best.npy not fully polished)
    try:
        if time.time() < explore_deadline:
            bx, bv = _smooth_refine(bestX, 800, 30.0, 300.0, 0.003, deadline=explore_deadline)
            if bv < bestV and np.all(np.isfinite(bx)):
                bestV, bestX = bv, bx.copy()
    except Exception:
        pass

    while time.time() < explore_deadline:
        trial += 1
        if time.time() > hard_deadline - 25:
            break
        sigma = sigmas[(trial - 1) % len(sigmas)]
        # vary strategy
        if trial % 5 == 0:
            # delete + reinsert k worst points
            k = 1 + ((trial // 5) % 3)  # 1..3
            x_try = bestX.copy()
            G = bestX @ bestX.T
            np.fill_diagonal(G, -2.0)
            per_max = G.max(axis=1)
            idx = np.argsort(per_max)[-k:]
            for ii in idx:
                cand = rng.randn(2000, 9)
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                D = cand @ x_try.T
                D[:, ii] = -2.0
                worst = D.max(axis=1)
                bi = int(np.argmin(worst))
                x_try[ii] = cand[bi]
        else:
            noise = rng.randn(*bestX.shape)
            # focus noise a bit on currently bad points
            if trial % 2 == 0:
                G = bestX @ bestX.T
                np.fill_diagonal(G, -2.0)
                per_max = G.max(axis=1)
                w = (per_max - per_max.min() + 0.02)
                w = w / w.mean()
                noise = noise * w[:, None]
            x_try = bestX + sigma * noise
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)

        # adapt iters to remaining time
        rem = explore_deadline - time.time()
        if rem < 5:
            break
        iters = 1000
        if rem < 30:
            iters = 600
        # vary lr/alpha slightly for diversity
        lr = 0.003 if (trial % 3 != 0) else 0.005
        a0 = 25.0 if (trial % 2 == 0) else 35.0
        a1 = 280.0
        try:
            x_opt, v_opt = _smooth_refine(x_try, iters, a0, a1, lr, deadline=explore_deadline)
        except Exception:
            continue
        if v_opt < bestV + 0.004 and time.time() < explore_deadline - 2:
            # second-stage fine tune with high alpha small lr
            try:
                x_opt2, v_opt2 = _smooth_refine(x_opt, 500, 150.0, 450.0, 0.0012, deadline=explore_deadline)
                if v_opt2 < v_opt:
                    x_opt, v_opt = x_opt2, v_opt2
            except Exception:
                pass
        if v_opt < bestV and np.all(np.isfinite(x_opt)):
            bestV = v_opt
            bestX = x_opt.copy()

    # final polish: a bit more smoothing then LP
    try:
        rem = hard_deadline - time.time() - 1.0
        if rem > 4:
            # high-alpha settle
            st = min(800, int(rem * 60))
            # use ~ part of time for smoothing, rest for LP
            smooth_sec = min(8.0, rem * 0.35)
            lp_sec = rem - smooth_sec - 0.5
            # do smoothing with deadline
            dl = time.time() + smooth_sec
            bx, bv = _smooth_refine(bestX, st, 120.0, 500.0, 0.001, deadline=dl)
            if bv < bestV:
                bestV, bestX = bv, bx.copy()
            if lp_sec > 3:
                lpX = _lp_polish(bestX, lp_sec)
                lv = _max_cos(lpX)
                if lv < bestV:
                    bestV, bestX = lv, lpX.copy()
                else:
                    # keep LP result if it is at least as good within tiny tol
                    # (it ties contacts); prefer lower
                    if lv <= bestV + 1e-12:
                        bestX = lpX.copy()
                        bestV = lv
    except Exception:
        pass

    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    # strict finiteness
    bestX[~np.isfinite(bestX)] = 0.0
    # fix any accidental zero rows
    for i in range(bestX.shape[0]):
        n = np.linalg.norm(bestX[i])
        if n < 1e-12 or not np.isfinite(n):
            v = rng.randn(9)
            v /= np.linalg.norm(v)
            bestX[i] = v
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
