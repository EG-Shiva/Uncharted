import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 249, 6

    best_x = None
    best_v = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best_v:
                    best_v = v
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = rng.standard_normal(size=(n, d))
        best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
        best_v = _max_cos(best_x)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)

    iu0, iu1 = np.triu_indices(n, 1)

    def make_fun(alpha):
        def fun(yflat):
            Y = yflat.reshape(n, d)
            nrm = np.sqrt(np.sum(Y * Y, axis=1))
            nrm = np.maximum(nrm, 1e-12)
            X = Y / nrm[:, None]
            G = X @ X.T
            g = G[iu0, iu1]
            m = float(np.max(g))
            e = np.exp(alpha * (g - m))
            S = float(np.sum(e))
            f = m + np.log(S) / alpha
            w = e / S
            W = np.zeros((n, n), dtype=float)
            W[iu0, iu1] = w
            W[iu1, iu0] = w
            Gx = W @ X
            s = np.sum(Gx * X, axis=1)
            Gy = (Gx - X * s[:, None]) / nrm[:, None]
            diff = nrm - 1.0
            f = f + 1e-4 * float(np.mean(diff * diff))
            Gy = Gy + (X * (2e-4 * diff / float(n))[:, None])
            return float(f), Gy.ravel()
        return fun

    def lbfgs_run(Xinit, alphas, iters):
        cur = Xinit.copy()
        cur_v = _max_cos(cur)
        for alpha, mi in zip(alphas, iters):
            if time.time() > T_END - 28:
                break
            fun = make_fun(alpha)
            try:
                res = minimize(fun, cur.ravel(), jac=True, method="L-BFGS-B",
                               options={"maxiter": int(mi), "maxfun": int(mi*3),
                                        "ftol": 1e-14, "gtol": 1e-7, "maxcor": 10})
                y = res.x.reshape(n, d)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            except Exception:
                continue
            v = _max_cos(y)
            # always continue from better of cur/y for this anneal chain
            if v < cur_v:
                cur, cur_v = y, v
            else:
                # still continue from y if close (avoid stuck flat)
                if v < cur_v + 5e-5:
                    cur, cur_v = y, v
        return cur, cur_v

    def hole_candidate(X, n_cand=2500):
        # find best hole position w.r.t. current set
        C = rng.standard_normal(size=(n_cand, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        D = C @ X.T
        cmax = np.max(D, axis=1)
        bi = int(np.argmin(cmax))
        return C[bi], float(cmax[bi])

    # initial deep anneal of incumbent
    cur, cur_v = lbfgs_run(best_x, [80.0, 200.0, 500.0, 1200.0, 3000.0],
                           [150, 150, 200, 250, 300])
    if cur_v < best_v:
        best_x, best_v = cur.copy(), cur_v

    # basin hopping focused on jammed core
    # temperature for sideways moves
    Ttemp = 2.0e-4
    it_count = 0
    # pre-allocate quick schedule
    quick_alphas = [150.0, 600.0, 2000.0]
    quick_iters = [90, 110, 170]
    mid_alphas = [120.0, 500.0, 1500.0, 3500.0]
    mid_iters = [80, 100, 140, 180]

    while time.time() < T_END - 30:
        it_count += 1
        if time.time() > T_END - 30:
            break
        # decay temperature
        Ttemp *= 0.995
        Ttemp = max(Ttemp, 2.0e-5)
        # analyse current best-of-chain `cur`
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        row_max = np.max(G, axis=1)
        glob = float(np.max(G))
        order = np.argsort(-row_max)
        mode = it_count % 5
        trial = cur.copy()
        if mode == 0:
            # break the single worst pair: kick one endpoint hard
            ij = np.unravel_index(int(np.argmax(G)), G.shape)
            i = int(ij[0])
            sig = 0.10 + 0.05 * float(rng.random())
            noise = rng.standard_normal(size=(d,))
            trial[i] = trial[i] + sig * noise
            trial[i] /= np.linalg.norm(trial[i])
        elif mode == 1:
            # kick 3 of top-12 worst points medium
            k = 3
            idx = order[:12]
            pick = rng.choice(idx, size=k, replace=False)
            sig = 0.08 + 0.04 * float(rng.random())
            for i in pick:
                noise = rng.standard_normal(size=(d,))
                trial[int(i)] = trial[int(i)] + sig * noise
                trial[int(i)] /= np.linalg.norm(trial[int(i)])
        elif mode == 2:
            # largest-hole reinsertion of worst point
            worst = int(order[0])
            mask = np.ones(n, dtype=bool)
            mask[worst] = False
            Xo = trial[mask]
            hp, _ = hole_candidate(Xo, n_cand=2500)
            trial[worst] = hp
            # also small shake of second worst
            if rng.random() < 0.5:
                w2 = int(order[1])
                noise = rng.standard_normal(size=(d,)) * 0.03
                trial[w2] = trial[w2] + noise
                trial[w2] /= np.linalg.norm(trial[w2])
        elif mode == 3:
            # kick 2 random rattlers fully + 1 worst slightly
            # rattlers = low row_max
            rattler_order = np.argsort(row_max)  # ascending
            pick_r = rattler_order[:50]
            rr = rng.choice(pick_r, size=2, replace=False)
            for i in rr:
                r = rng.standard_normal(size=(d,))
                r /= np.linalg.norm(r)
                trial[int(i)] = r
            worst = int(order[0])
            noise = rng.standard_normal(size=(d,)) * 0.06
            trial[worst] = trial[worst] + noise
            trial[worst] /= np.linalg.norm(trial[worst])
        else:
            # broad shake of active set (points within 0.01 of max)
            active = np.where(row_max > glob - 0.01)[0]
            if len(active) < 4:
                active = order[:10]
            k = min(len(active), 4)
            pick = rng.choice(active, size=k, replace=False)
            sig = 0.05 + 0.06 * float(rng.random())
            for i in pick:
                noise = rng.standard_normal(size=(d,))
                trial[int(i)] = trial[int(i)] + sig * noise
                trial[int(i)] /= np.linalg.norm(trial[int(i)])

        # quick re-anneal
        if mode == 3:
            new_x, new_v = lbfgs_run(trial, mid_alphas, mid_iters)
        else:
            new_x, new_v = lbfgs_run(trial, quick_alphas, quick_iters)

        if new_v < best_v - 1e-12:
            best_x, best_v = new_x.copy(), new_v
            cur, cur_v = new_x.copy(), new_v
            Ttemp *= 0.9
        elif new_v < cur_v:
            cur, cur_v = new_x.copy(), new_v
        else:
            # simulated-annealing sideways acceptance to change basin
            if rng.random() < np.exp(-(new_v - cur_v) / Ttemp):
                cur, cur_v = new_x.copy(), new_v
            # occasionally reset cur to best if wandering too high
            if cur_v > best_v + 0.002 and rng.random() < 0.3:
                cur, cur_v = best_x.copy(), best_v
                # small kick off best
                idx = rng.choice(n, size=2, replace=False)
                for i in idx:
                    noise = rng.standard_normal(size=(d,)) * 0.04
                    cur[int(i)] = cur[int(i)] + noise
                    cur[int(i)] /= np.linalg.norm(cur[int(i)])
                cur_v = _max_cos(cur)

    # final trust-region LP polish on best
    def lp_polish(x, t_end):
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        nn, dd = x.shape
        best = _max_cos(x)
        radius = 5e-4
        iu_0, iu_1 = np.triu_indices(nn, 1)
        while time.time() < t_end - 0.5 and radius > 1e-14:
            c = (x @ x.T)[iu_0, iu_1]
            m = float(np.max(c))
            sel = c > m - max(20 * radius, 5e-7)
            cnt = int(np.count_nonzero(sel))
            if cnt > 6000:
                idx = np.argsort(c)[-6000:]
                msel = np.zeros_like(sel)
                msel[idx] = True
                sel = msel
            I = iu_0[sel]; J = iu_1[sel]
            p = len(I)
            if p == 0:
                break
            nv = nn * dd + 1
            rows = np.repeat(np.arange(p), 2 * dd + 1)
            cols = np.concatenate([I[:, None] * dd + np.arange(dd),
                                   J[:, None] * dd + np.arange(dd),
                                   np.full((p, 1), nn * dd)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(nn), dd), np.arange(nn * dd))),
                              shape=(nn, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(nn),
                              bounds=[(-radius, radius)] * (nn * dd) + [(None, None)],
                              method="highs",
                              options={"time_limit": max(1.0, t_end - time.time() - 0.5)})
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(nn, dd)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius * 1.5, 0.02)
            else:
                radius *= 0.5
        return x

    t_lp_end = T_END - 0.5
    # give at least ~15s to LP if possible
    if time.time() < t_lp_end - 2:
        best_x = lp_polish(best_x, t_lp_end)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
