import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _personal_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1), g


def _contacts(x, tol=2e-7):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = float(g.max())
    tight = g > m - tol
    cnt = tight.sum(axis=1).astype(int)
    return cnt, m, g


def _lp_polish(x, deadline, r_init=0.01, r_max=0.05, cap=2000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r_init
    iu0 = np.triu_indices(n, 1)
    nv = n * d + 1
    obj = np.zeros(nv)
    obj[-1] = 1.0
    while time.time() < deadline and radius > 1e-13:
        if time.time() + 0.35 > deadline:
            break
        G = x @ x.T
        c_all = G[iu0]
        m = c_all.max()
        thresh = max(20.0 * radius, 1e-9)
        sel = c_all > m - thresh
        nnz = int(np.count_nonzero(sel))
        if nnz == 0:
            radius *= 0.5
            continue
        if nnz > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        c_sel = c_all[sel]
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        bounds = [(-radius, radius)] * (n * d) + [(None, None)]
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y = y / yn
        v = _max_cos(y)
        if v < best:
            x = y
            best = v
            radius = min(radius * 1.5, r_max)
        else:
            radius *= 0.5
    return x, best


def _find_void(Xf, rng, n_samp=15000, iters=250):
    # Xf: (m,d) unit rows, fixed points. Find unit y minimizing max_i <y, Xf_i>
    d = Xf.shape[1]
    Y = rng.standard_normal((n_samp, d))
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    scores = Y @ Xf.T
    mx = scores.max(axis=1)
    bi = int(np.argmin(mx))
    y = Y[bi].copy()
    best = float(mx[bi])
    # also try opposite of mean / random best of few
    eta = 0.08
    for it in range(iters):
        g = Xf @ y
        j = int(np.argmax(g))
        # tangential gradient of <x_j, y>
        t = Xf[j] - (Xf[j] @ y) * y
        nrm = float(np.linalg.norm(t))
        if nrm < 1e-12:
            # random tangent kick
            r = rng.standard_normal(d)
            r -= (r @ y) * y
            r /= (np.linalg.norm(r) + 1e-12)
            t = r
            nrm = 1.0
        else:
            t = t / nrm
        # try step decreasing max
        step = eta / (1.0 + it * 0.02)
        y2 = y - step * t
        y2 /= (np.linalg.norm(y2) + 1e-18)
        v2 = float((Xf @ y2).max())
        if v2 < best:
            y = y2
            best = v2
        else:
            eta *= 0.97
            # small random tangent move to escape
            if it % 20 == 0:
                r = rng.standard_normal(d)
                r -= (r @ y) * y
                r /= (np.linalg.norm(r) + 1e-12)
                y2 = y + 0.01 * r
                y2 /= np.linalg.norm(y2)
                v2 = float((Xf @ y2).max())
                if v2 < best + 0.005:
                    y = y2
                    best = min(best, v2)
    return y


def solve():
    t0 = time.time()
    HARD_END = t0 + 171.0
    rng = np.random.default_rng(42)
    n, d = 78, 15
    cand = []
    for fn in ("best.npy", "record.npy"):
        try:
            b = np.load(fn)
            if b.shape == (n, d):
                b = b.astype(float)
                b /= np.linalg.norm(b, axis=1, keepdims=True)
                if np.all(np.isfinite(b)):
                    cand.append(b)
        except Exception:
            pass
    if not cand:
        x0 = rng.standard_normal((n, d))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        cand.append(x0)
    best_v = 1e9
    best_x = None
    for c in cand:
        v = _max_cos(c)
        if v < best_v:
            best_v = v
            best_x = c.copy()
    x_best = best_x.copy()
    v_best = best_v

    # quick squeeze of start
    if time.time() < HARD_END - 30:
        x_best, v_best = _lp_polish(x_best, min(time.time() + 6.0, HARD_END - 26),
                                    r_init=0.002, r_max=0.015, cap=2000)

    trial = 0
    while time.time() < HARD_END - 22:
        trial += 1
        remaining = HARD_END - 22 - time.time()
        if remaining < 3.0:
            break
        budget = min(7.0, max(2.5, remaining / 2.0))
        # don't overspend last bit
        if remaining - budget < 1.0:
            budget = remaining - 0.5
            if budget < 2.0:
                budget = remaining
        deadline = time.time() + budget
        mode = trial % 6
        x_try = x_best.copy()
        if mode == 1:
            x_try = x_try + 0.008 * rng.standard_normal((n, d))
            r_init = 0.008
        elif mode == 2:
            x_try = x_try + 0.022 * rng.standard_normal((n, d))
            r_init = 0.015
        elif mode == 3:
            # worst personal-max kick
            pm, _ = _personal_max(x_try)
            order = np.argsort(-pm)
            k = 6
            worst = order[:k]
            noise = 0.004 * rng.standard_normal((n, d))
            noise[worst] += 0.07 * rng.standard_normal((k, d))
            x_try = x_try + noise
            r_init = 0.015
        elif mode == 4:
            # void re-insertion for low-contact / rattlers
            cnt, m, G = _contacts(x_try, tol=2e-7)
            order = np.argsort(cnt)  # ascending
            k = 2
            if trial % 12 == 4:
                k = 3
            repl = order[:k].copy()
            # keep set fixed for void search (exclude all repl at once iteratively)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[repl] = False
            Xf = x_try[keep_mask]
            for r_idx in repl:
                # void w.r.t. current fixed set
                y = _find_void(Xf, rng, n_samp=12000, iters=200)
                x_try[r_idx] = y
                # add it to fixed set for next repl (to spread voids)
                Xf = np.vstack([Xf, y.reshape(1, d)])
            # jitter rest lightly
            mask = np.ones(n, dtype=bool); mask[repl] = False
            x_try[mask] += 0.003 * rng.standard_normal((np.count_nonzero(mask), d))
            r_init = 0.015
        elif mode == 5:
            # overloaded hub + neighbours shake
            cnt, m, G = _contacts(x_try, tol=1e-7)
            hub = int(np.argmax(cnt))
            # neighbours: points tied to hub
            row = G[hub].copy()
            row[hub] = -2
            neigh = np.where(row > m - 1e-7)[0]
            grp = np.concatenate([[hub], neigh])
            # limit group size
            if len(grp) > 20:
                # keep hub + closest 19
                o = np.argsort(-row[grp[1:]])
                grp = np.concatenate([[hub], grp[1:][o[:19]]])
            noise = 0.003 * rng.standard_normal((n, d))
            noise[grp] += 0.035 * rng.standard_normal((len(grp), d))
            x_try = x_try + noise
            r_init = 0.012
        else:  # mode 0: tail cluster shake (lowest contacts)
            cnt, _, _ = _contacts(x_try, tol=2e-7)
            order = np.argsort(cnt)
            tail = order[:16]
            noise = 0.003 * rng.standard_normal((n, d))
            noise[tail] += 0.045 * rng.standard_normal((len(tail), d))
            x_try = x_try + noise
            r_init = 0.015
        x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        if not np.all(np.isfinite(x_try)):
            continue
        x_try, v_try = _lp_polish(x_try, deadline, r_init=r_init, r_max=0.05, cap=2000)
        if v_try < v_best - 1e-13:
            x_best = x_try
            v_best = v_try
        if time.time() >= HARD_END - 22:
            break

    if time.time() < HARD_END - 0.8:
        x_best, v_best = _lp_polish(x_best, HARD_END - 0.4, r_init=0.002, r_max=0.015, cap=2500)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    if not np.all(np.isfinite(x_best)):
        x_best = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return x_best
