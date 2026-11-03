import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def contacts_count(x, thresh):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return np.sum(g > thresh, axis=1)


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    iu0r = iu0[0]
    iu0c = iu0[1]
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        cfull = G[iu0r, iu0c]
        m = cfull.max()
        window = max(20 * radius, 1e-9)
        # limit constraints for speed
        sel = cfull > m - window
        # if too many, keep tightest 4000
        if np.count_nonzero(sel) > 4000:
            thr = np.sort(cfull[sel])[-4000]
            sel = cfull >= thr
        I = iu0r[sel]
        J = iu0c[sel]
        csel = cfull[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
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
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def smooth_refine(x_init, Ts, maxiter_each):
    n, d = x_init.shape
    iu_r, iu_c = np.triu_indices(n, 1)
    y = x_init.copy()
    for T in Ts:
        def fg(Yflat, T=T):
            Y = Yflat.reshape(n, d)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Y / rn
            G = X @ X.T
            c = G[iu_r, iu_c]
            cmax = c.max()
            e = np.exp((c - cmax) / T)
            s = e.sum()
            f = cmax + T * np.log(s)
            w = e / s
            W = np.zeros((n, n))
            W[iu_r, iu_c] = w
            W = W + W.T
            GX = W @ X
            dot = np.sum(GX * X, axis=1, keepdims=True)
            GY = (GX - dot * X) / rn
            return float(f), GY.ravel()
        try:
            res = minimize(fg, y.ravel(), jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter_each, "ftol": 1e-14, "gtol": 1e-7, "maxfun": 20000})
            y = res.x.reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        except Exception:
            break
    return y / np.linalg.norm(y, axis=1, keepdims=True)


def find_hole(X, exclude_idx, rng, n_samples=20000):
    n, d = X.shape
    mask = np.ones(n, dtype=bool)
    mask[exclude_idx] = False
    others = X[mask]
    cand = rng.normal(size=(n_samples, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    md = dots.max(axis=1)
    bi = int(np.argmin(md))
    return cand[bi]


def solve():
    np.random.seed(0)
    t0 = time.time()
    TIME_TOTAL = 168.0
    FINAL_POLISH = 28.0
    n_target, d_target = 123, 13
    # load best
    x_best = None
    best = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            if a.shape == (n_target, d_target):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = max_cos(a)
                if v < best:
                    best = v
                    x_best = a.copy()
        except Exception:
            pass
    if x_best is None:
        rng0 = np.random.RandomState(0)
        x_best = rng0.normal(size=(n_target, d_target))
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
        best = max_cos(x_best)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    best = max_cos(x_best)

    it = 0
    # main basin-hopping loop
    while time.time() - t0 < TIME_TOTAL - FINAL_POLISH:
        it += 1
        rng = np.random.RandomState(1000 + it * 7919)
        # contact stats with loose threshold to find hubs/rattlers
        G = x_best @ x_best.T
        np.fill_diagonal(G, -2.0)
        # adaptive contact threshold
        loose = best - 0.004
        cc = np.sum(G > loose, axis=1)
        order_low = np.argsort(cc)  # rattlers first
        order_high = np.argsort(-cc)  # hubs first

        x_try = x_best.copy()
        mode = it % 6
        if mode == 5:
            # hole re-insertion: replace 1-2 rattlers by farthest-point holes
            krep = 2 if (it % 12 == 5) else 1
            for k in range(krep):
                # recompute cc each replacement? use order_low
                idx = int(order_low[k % len(order_low)])
                h = find_hole(x_try, idx, rng, n_samples=25000)
                # small jitter around hole
                h = h + rng.normal(scale=0.01, size=d_target)
                h /= np.linalg.norm(h)
                x_try[idx] = h
            # global tiny shake
            x_try += rng.normal(scale=0.008, size=x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            Ts = [0.012, 0.006, 0.003]
            maxiter_each = 120
        elif mode == 4:
            # big shake of a random block to change topology
            k = 8
            subset = rng.choice(n_target, size=k, replace=False)
            sigma = 0.25
            x_try[subset] += rng.normal(scale=sigma, size=(k, d_target))
            x_try += rng.normal(scale=0.01, size=x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            Ts = [0.02, 0.008, 0.0035]
            maxiter_each = 150
        else:
            # targeted hub kick + small global noise
            sigmas = [0.06, 0.10, 0.16, 0.22]
            sigma = sigmas[mode % len(sigmas)]
            k = 3 + (it % 4)  # 3..6
            # bias half from hubs, half random
            nh = min(k, 4)
            hub_pick = rng.choice(order_high[:35], size=nh, replace=False)
            rand_pick = rng.choice(n_target, size=k - nh, replace=False)
            subset = np.concatenate([hub_pick, rand_pick])
            x_try += rng.normal(scale=0.008 + 0.002 * (it % 3), size=x_try.shape)
            x_try[subset] += rng.normal(scale=sigma, size=(len(subset), d_target))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            # vary temperature schedule deterministically
            if it % 2 == 0:
                Ts = [0.015, 0.007, 0.003]
            else:
                Ts = [0.010, 0.004, 0.002]
            maxiter_each = 130

        # smooth anneal
        try:
            x_sm = smooth_refine(x_try, Ts, maxiter_each)
        except Exception:
            continue
        v_sm = max_cos(x_sm)
        # only LP-polish promising basins (within 0.004 of best) to save time
        if v_sm > best + 0.004:
            # not promising; but occasionally keep diversity: if far worse skip
            # check time
            if time.time() - t0 > TIME_TOTAL - FINAL_POLISH - 5:
                break
            continue
        # quick LP polish to get true local optimum
        remain = TIME_TOTAL - FINAL_POLISH - (time.time() - t0)
        if remain < 2:
            break
        lp_time = min(4.0, max(1.5, remain / 8.0))
        # for very promising, give a bit more
        if v_sm < best + 1e-4:
            lp_time = min(6.0, max(2.5, remain / 6.0))
        try:
            x_lp = lp_polish(x_sm, lp_time)
        except Exception:
            continue
        v_lp = max_cos(x_lp)
        if v_lp < best - 1e-12:
            x_best = x_lp
            best = v_lp
        # safety time check
        if time.time() - t0 >= TIME_TOTAL - FINAL_POLISH:
            break
        # stop if we already improved a lot? keep going to squeeze more
        if it > 80:
            break

    # final thorough polish of best
    remain = TIME_TOTAL - (time.time() - t0)
    # leave 2 sec margin
    fp = max(5.0, min(FINAL_POLISH + 5.0, remain - 2.0))
    if fp > 3:
        try:
            x_best = lp_polish(x_best, fp)
            best = max_cos(x_best)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
