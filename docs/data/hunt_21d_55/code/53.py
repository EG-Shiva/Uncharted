import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _norm(x):
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n


def _smooth_descent(x_init, steps, lr0, beta0, beta1, rng, noise=0.0):
    x = x_init.copy()
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = beta0 + (beta1 - beta0) * f
        lr = lr0 * (1.0 - 0.6 * f)
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Gw = W @ x
        dot = np.sum(Gw * x, axis=1, keepdims=True)
        Gt = Gw - dot * x
        nrm = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        Gt = Gt / nrm
        if noise > 0:
            nz = rng.standard_normal(x.shape)
            nz -= np.sum(nz * x, axis=1, keepdims=True) * x
            nn = np.linalg.norm(nz, axis=1, keepdims=True) + 1e-12
            nz = nz / nn
            Gt = (1.0 - noise) * Gt + noise * nz
            # renormalize direction not critical
        x = x - lr * Gt
        x = _norm(x)
    return x


def _hole(Y, rng, n_cand=12000):
    d = Y.shape[1]
    cand = rng.standard_normal((n_cand, d))
    cand = _norm(cand)
    dots = cand @ Y.T
    md = dots.max(axis=1)
    return cand[np.argmin(md)]


def _lp_polish(x, deadline, radius_init=1e-3, max_iter=60, cap=900):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius_init
    iu0 = np.triu_indices(n, 1)
    it = 0
    # preallocate nothing heavy
    while time.time() < deadline - 0.4 and it < max_iter and radius > 1e-14:
        it += 1
        G = x @ x.T
        cfull = G[iu0]
        m = float(cfull.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = cfull > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-cfull[idx])[:cap]
            nsel = np.zeros_like(sel)
            nsel[idx[order]] = True
            sel = nsel
            cnt = cap
        csel = cfull[sel]
        I = iu0[0][sel]
        J = iu0[1][sel]
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(cnt), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((cnt, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((cnt, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(cnt, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
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
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() >= deadline - 0.3:
            break
    return x


def solve():
    rng = np.random.default_rng(42)
    t0 = time.time()
    LIMIT = 168.0
    T_SEARCH_END = t0 + 138.0
    T_FINAL = t0 + LIMIT - 1.0

    xb = None
    best_score = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (55, 21):
                a = _norm(a)
                s = _max_cos(a)
                if s < best_score:
                    best_score = s
                    xb = a.copy()
        except Exception:
            pass
    if xb is None:
        xb = _norm(rng.standard_normal((55, 21)))
        best_score = _max_cos(xb)
    n, d = 55, 21
    x_best = xb.copy()
    best = best_score

    it = 0
    try:
        while time.time() < T_SEARCH_END:
            it += 1
            remain = T_FINAL - time.time()
            if remain < 4.0:
                break
            mode = it % 6
            x_try = x_best.copy()
            if mode == 1:
                sigma = 0.012 + 0.012 * rng.random()
                x_try = _norm(x_try + sigma * rng.standard_normal(x_try.shape))
                x_try = _smooth_descent(x_try, steps=350, lr0=0.005,
                                        beta0=200.0, beta1=700.0, rng=rng, noise=0.02)
            elif mode == 2:
                G = x_try @ x_try.T
                np.fill_diagonal(G, -2.0)
                rowmax = G.max(axis=1)
                order = np.argsort(-rowmax)
                k = 3 + int(rng.integers(0, 3))
                worst = order[:k]
                sig = np.full(n, 0.003)
                sig[worst] = 0.15 + 0.10 * rng.random()
                neigh = np.argsort(-G[worst[0]])[:6]
                for j in neigh:
                    if sig[j] < 0.04:
                        sig[j] = 0.04
                x_try = _norm(x_try + sig[:, None] * rng.standard_normal(x_try.shape))
                x_try = _smooth_descent(x_try, steps=450, lr0=0.006,
                                        beta0=250.0, beta1=800.0, rng=rng, noise=0.03)
            elif mode == 3:
                G = x_try @ x_try.T
                np.fill_diagonal(G, -2.0)
                rowmax = G.max(axis=1)
                order = np.argsort(-rowmax)
                nr = 1 + int(rng.integers(0, 2))
                for kk in range(nr):
                    w = int(order[kk])
                    mask = np.ones(n, dtype=bool)
                    mask[w] = False
                    h = _hole(x_try[mask], rng, n_cand=12000)
                    h = h + 0.008 * rng.standard_normal(d)
                    h = h / (np.linalg.norm(h) + 1e-12)
                    x_try[w] = h
                x_try = _smooth_descent(x_try, steps=400, lr0=0.006,
                                        beta0=200.0, beta1=800.0, rng=rng, noise=0.02)
            elif mode == 4:
                sigma = 0.05 + 0.06 * rng.random()
                x_try = _norm(x_try + sigma * rng.standard_normal(x_try.shape))
                x_try = _smooth_descent(x_try, steps=500, lr0=0.007,
                                        beta0=150.0, beta1=850.0, rng=rng, noise=0.04)
            else:
                frac = 0.25 + 0.3 * rng.random()
                k = int(n * frac)
                subset = rng.choice(n, size=k, replace=False)
                sigma = 0.06 + 0.08 * rng.random()
                x_try[subset] += sigma * rng.standard_normal((k, d))
                x_try = _norm(x_try)
                x_try = _smooth_descent(x_try, steps=450, lr0=0.006,
                                        beta0=180.0, beta1=850.0, rng=rng, noise=0.03)

            v_smooth = _max_cos(x_try)
            if v_smooth < best - 1e-12:
                # immediate improvement, keep without LP if no time, else polish briefly
                if v_smooth < best - 5e-6:
                    # worth LP
                    dl = min(time.time() + 2.0, T_FINAL - 8.0)
                    if dl > time.time() + 0.6:
                        x_pol = _lp_polish(x_try, deadline=dl, radius_init=1.0e-3)
                        v_pol = _max_cos(x_pol)
                        if v_pol < best:
                            x_best, best = x_pol, v_pol
                        else:
                            x_best, best = x_try, v_smooth
                    else:
                        x_best, best = x_try, v_smooth
                else:
                    x_best, best = x_try, v_smooth
                continue
            # not better yet: only LP if close
            if v_smooth > best + 0.008:
                continue
            # gated LP: smaller budget if far
            if v_smooth > best + 0.001:
                budget = 1.0
            else:
                budget = 2.0
            dl = min(time.time() + budget, T_FINAL - 8.0)
            if dl < time.time() + 0.5:
                continue
            x_pol = _lp_polish(x_try, deadline=dl,
                               radius_init=8e-4 if v_smooth > best else 1.2e-3)
            v_pol = _max_cos(x_pol)
            if v_pol < best - 1e-12:
                x_best, best = x_pol, v_pol

        # final polish of global best with remaining time
        remain = T_FINAL - time.time()
        if remain > 1.5:
            # cap final polish to ~22s to stay safe
            fend = min(time.time() + min(remain - 0.5, 22.0), T_FINAL)
            x_best = _lp_polish(x_best, deadline=fend, radius_init=8e-4)
    except Exception:
        pass

    x_best = _norm(x_best)
    # safety: if somehow worse than loaded best, reload best
    try:
        ab = np.load("best.npy")
        ab = _norm(np.asarray(ab, dtype=np.float64))
        if ab.shape == (55, 21):
            sb = _max_cos(ab)
            s_cur = _max_cos(x_best)
            if sb < s_cur:
                x_best = ab
    except Exception:
        pass
    x_best = _norm(x_best)
    return x_best
