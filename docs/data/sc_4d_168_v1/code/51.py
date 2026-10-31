import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _lp_polish_deadline(x, deadline):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        iu_c = G[iu]
        m = float(np.max(iu_c))
        sel = iu_c > m - max(20 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        csel = iu_c[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _adam_homotopy_run(X0, t, steps, lr, rng, deadline, b1=0.9, b2=0.999, eps=1e-8):
    n, d = X0.shape
    X = X0.copy()
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1p = 1.0
    b2p = 1.0
    best_loss = np.inf
    X_best = X.copy()
    best_best_max = _max_cos(X)
    cur_lr = lr
    for s in range(1, steps + 1):
        if time.time() > deadline:
            break
        # linear lr decay within stage
        f = s / float(steps)
        lr_s = lr * (1.0 - 0.6 * f)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cmax = float(np.max(G))
        D1 = G - t
        np.maximum(D1, 0.0, out=D1)
        np.fill_diagonal(D1, 0.0)
        D2 = G - t + 0.002
        np.maximum(D2, 0.0, out=D2)
        np.fill_diagonal(D2, 0.0)
        # loss over i<j (half of full sum)
        loss = (np.sum(D1 * D1) + 0.1 * np.sum(D2 * D2)) * 0.5
        viol = int(np.sum(G > t + 1e-12) // 2)
        if loss < best_loss:
            best_loss = float(loss)
            X_best = X.copy()
            best_best_max = cmax
        # early feasible exit: keep a bit to push second shell
        if viol == 0 and cmax <= t + 1e-12:
            # do extra 150 steps to minimize second term if time allows
            if s > 300:
                # check if second term also tiny -> break
                if float(np.sum(D2 * D2) * 0.5) < 1e-14:
                    break
            if s > 800:
                break
        W = 2.0 * D1 + 0.2 * D2
        grad = W @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        Gt = grad - dot * X
        m *= b1
        m += (1.0 - b1) * Gt
        v *= b2
        v += (1.0 - b2) * (Gt * Gt)
        b1p *= b1
        b2p *= b2
        mh = m / (1.0 - b1p + 1e-30)
        vh = v / (1.0 - b2p + 1e-30)
        step = mh / (np.sqrt(vh) + eps)
        X = X - lr_s * step
        X = _norm(X)
    return X_best, best_loss, best_best_max, X

def _perturb_hubs(X, t, k, rng, lo=0.05, hi=0.15):
    Y = X.copy()
    n = X.shape[0]
    G = Y @ Y.T
    np.fill_diagonal(G, -2.0)
    D = np.maximum(G - t, 0.0)
    np.fill_diagonal(D, 0.0)
    score = np.sum(D, axis=1) + 1e-12 * rng.rand(n)
    # also consider count of violations to catch hubs
    order = np.argsort(-score, kind="stable")
    hubs = order[:k]
    for idx in hubs:
        ang = float(rng.uniform(lo, hi))
        r = rng.randn(4)
        xi = Y[idx]
        u = r - np.dot(r, xi) * xi
        nu = np.linalg.norm(u)
        if nu < 1e-12:
            r = rng.randn(4)
            u = r - np.dot(r, xi) * xi
            nu = np.linalg.norm(u) + 1e-30
        u = u / nu
        Y[idx] = np.cos(ang) * xi + np.sin(ang) * u
    return _norm(Y)

def solve():
    t0 = time.time()
    HARD_END = t0 + 170.0
    SEARCH_END = t0 + 138.0
    rng = np.random.RandomState(1234567)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(168, 4)
    xb = _norm(np.asarray(xb, dtype=float))
    try:
        xr = np.load("record.npy").astype(float)
        xr = _norm(xr)
        if xr.shape == (168, 4) and _max_cos(xr) < _max_cos(xb):
            xb = xr.copy()
    except Exception:
        pass
    n, d = 168, 4
    if xb.shape != (168, 4):
        xb = _norm(rng.randn(168, 4))
    xb = _norm(xb)
    best_val = _max_cos(xb)
    x_best = xb.copy()

    targets = [0.8592, 0.8590, 0.8587, 0.8584, 0.8581, 0.8578, 0.8575, 0.8570, 0.8565]
    # if best already below first target, keep only lower ones
    x_stage = x_best.copy()

    for ti, t in enumerate(targets):
        if time.time() > SEARCH_END - 2.0:
            break
        if best_val <= t + 1e-12:
            # already feasible, carry forward with current best as init
            x_stage = x_best.copy()
            continue
        # attempts for this target
        n_attempts = 3
        if ti == 0:
            n_attempts = 4
        for att in range(n_attempts):
            if time.time() > SEARCH_END - 2.0:
                break
            # steps budget shrinks for lower targets to save time
            steps = 2600 if ti < 2 else 2200
            # vary lr a bit per attempt
            lr = 0.006 if att == 0 else (0.004 if att == 1 else 0.008)
            # deadline for this run: leave time for LP
            run_deadline = min(SEARCH_END - 1.0, time.time() + 12.0)
            Xbpen, loss, bmax, Xlast = _adam_homotopy_run(
                x_stage, t, steps, lr, rng, run_deadline)
            vmax_pen = _max_cos(Xbpen)
            # record any global improvement directly (before LP)
            if vmax_pen < best_val:
                # opportunistic LP if close to best
                if vmax_pen < best_val + 0.0005 and time.time() < SEARCH_END - 2.0:
                    dl = min(time.time() + 1.3, SEARCH_END)
                    xs, vs = _lp_polish_deadline(Xbpen, dl)
                    if vs < best_val:
                        x_best, best_val = xs.copy(), vs
                        x_stage = xs.copy()
                        # if we beat target, break attempts -> next target
                        if vs <= t + 1e-9:
                            break
                        else:
                            # continue from polished point
                            continue
                    else:
                        # LP didn't beat best, but Adam point may still be best walk state
                        if vmax_pen < best_val:
                            x_best, best_val = Xbpen.copy(), vmax_pen
                            x_stage = Xbpen.copy()
                            if vmax_pen <= t + 1e-9:
                                break
                        # perturb for next attempt
                        x_stage = _perturb_hubs(Xbpen, t, k=int(rng.choice([3, 4, 5])), rng=rng)
                else:
                    x_best, best_val = Xbpen.copy(), vmax_pen
                    x_stage = Xbpen.copy()
                    if vmax_pen <= t + 1e-9:
                        break
                    x_stage = _perturb_hubs(Xbpen, t, k=4, rng=rng)
            else:
                # no direct improvement: check near-zero violation -> LP anyway
                if vmax_pen <= t + 2e-4 and time.time() < SEARCH_END - 2.0:
                    dl = min(time.time() + 1.0, SEARCH_END)
                    xs, vs = _lp_polish_deadline(Xbpen, dl)
                    if vs < best_val:
                        x_best, best_val = xs.copy(), vs
                        x_stage = xs.copy()
                        if vs <= t + 1e-9:
                            break
                        continue
                # plateau -> restart from best-penalty with hub rotation
                # use Xbpen as base
                k = int(rng.choice([3, 4, 5, 6]))
                lo, hi = (0.05, 0.15) if att < 2 else (0.08, 0.20)
                x_stage = _perturb_hubs(Xbpen, t, k=k, rng=rng, lo=lo, hi=hi)
                # also track best max seen in trajectory for walk state?
                vlast = _max_cos(Xlast)
                if vlast < best_val:
                    # rare: last better than pen-best due to tracking lag
                    if vlast < best_val + 0.0005 and time.time() < SEARCH_END - 2.0:
                        dl = min(time.time() + 1.0, SEARCH_END)
                        xs, vs = _lp_polish_deadline(Xlast, dl)
                        if vs < best_val:
                            x_best, best_val = xs.copy(), vs
                            x_stage = xs.copy()
                            if vs <= t + 1e-9:
                                break
            # plateau detection for lowering t: if loss tiny -> feasible
            if loss < 1e-14 or bmax <= t + 1e-12:
                # feasible stage
                if time.time() < SEARCH_END - 1.5:
                    dl = min(time.time() + 1.2, SEARCH_END)
                    xs, vs = _lp_polish_deadline(Xbpen, dl)
                    if vs < best_val:
                        x_best, best_val = xs.copy(), vs
                    x_stage = x_best.copy()
                break
        # after attempts, if stage not feasible (best still > t), do not go to much lower t
        # but try next target only if we made progress close to t
        if best_val > t + 3e-4:
            # too far, stop descending; do a final diversification and retry first target?
            # instead break to save time for final polish
            # try one more diversified homotopy at current best with same t but bigger kick
            if time.time() < SEARCH_END - 8.0 and ti < 2:
                # one big hub kick and short run will be handled in next target loop;
                # just continue to next (slightly lower demands more, so break)
                break
            break
        # else feasible/near-feasible: proceed to next lower target with polished best
        x_stage = x_best.copy()
        if time.time() > SEARCH_END - 2.0:
            break

    try:
        x_best, best_val = _lp_polish_deadline(x_best, HARD_END - 1.0)
    except Exception:
        pass
    x_best = _norm(x_best)
    return x_best
