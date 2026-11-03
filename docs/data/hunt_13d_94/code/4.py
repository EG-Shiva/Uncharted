import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish_deadline(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 5e-3
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu0, iu1]
        m = c.max()
        sel_thresh = m - max(20 * radius, 1e-9)
        sel = c > sel_thresh
        if np.count_nonzero(sel) == 0:
            break
        I = iu0[sel]
        J = iu1[sel]
        csel = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y /= yn
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best


def solve():
    import time
    import numpy as np
    from scipy.optimize import minimize
    t0 = time.time()
    HARD_END = t0 + 178.0
    EXPL_END = t0 + 155.0

    rng = np.random.RandomState(7)
    n_d, d_d = 94, 13

    def load_norm(p):
        try:
            a = np.load(p)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (94, 13):
                return None
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                return None
            return a
        except Exception:
            return None

    xb = load_norm("best.npy")
    xr = load_norm("record.npy")
    cands = [a for a in [xb, xr] if a is not None]
    if len(cands) == 0:
        cands = [rng.randn(94, 13)]
        cands[0] /= np.linalg.norm(cands[0], axis=1, keepdims=True)
    best_val = 1e9
    X_best = cands[0]
    for c in cands:
        v = _max_cos(c)
        if v < best_val:
            best_val = v
            X_best = c.copy()

    n, d = 94, 13
    iu0, iu1 = np.triu_indices(n, 1)

    def worst_per_point(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return G.max(axis=1), G

    def smooth_refine(X_init, betas, iters):
        x_flat = X_init.ravel().copy()
        for beta, mi in zip(betas, iters):
            if time.time() > HARD_END - 2:
                break
            def fun(xf, _beta=beta):
                X = xf.reshape(n, d)
                norms = np.linalg.norm(X, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-12)
                Y = X / norms
                G = Y @ Y.T
                g = G[iu0, iu1]
                m = g.max()
                e = np.exp(_beta * (g - m))
                Z = e.sum()
                L = m + np.log(Z) / _beta
                w = e / Z
                P = np.zeros((n, n))
                P[iu0, iu1] = w
                P = P + P.T
                grad_Y = P @ Y
                dot = (grad_Y * Y).sum(axis=1, keepdims=True)
                grad_X = (grad_Y - dot * Y) / norms
                # tiny norm penalty to keep scale ~1
                pen = float(((norms - 1.0) ** 2).mean())
                L = L + 0.05 * pen
                grad_pen = (2.0 * 0.05 / n) * (norms - 1.0) * Y / norms
                # grad_pen w.r.t X: d/dX of mean((|X_i|-1)^2)
                # Y = X/norm, d norm/dX = Y, so grad = 2*(norm-1)/n * Y
                grad_X = grad_X + (2.0 * 0.05 / n) * (norms - 1.0) * Y / np.maximum(norms, 1e-12) * 0.0 + grad_pen * 0.0
                # simpler: use grad_pen approx as above scaled
                # recompute correctly:
                # Lpen = 0.05*mean((norm-1)^2), dLpen/dX_i = 0.05*2/n*(norm_i-1)*Y_i
                grad_X = grad_X + (0.1 / n) * (norms - 1.0) * Y
                return float(L), grad_X.ravel()
            try:
                res = minimize(fun, x_flat, jac=True, method="L-BFGS-B",
                               options={"maxiter": mi, "maxfun": mi * 6, "ftol": 1e-14, "gtol": 1e-7})
                x_flat = res.x
            except Exception:
                break
        Xf = x_flat.reshape(n, d)
        Xf /= np.linalg.norm(Xf, axis=1, keepdims=True)
        return Xf

    def hole_reinsert(X, kill_idx, S=4000):
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[kill_idx] = False
        Keep = X[keep_mask]
        R = rng.randn(S, d)
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        dots = R @ Keep.T
        worst = dots.max(axis=1)
        order = np.argsort(worst)
        Y = X.copy()
        for k, idx in enumerate(np.atleast_1d(kill_idx)):
            Y[idx] = R[order[k % len(order)]]
        return Y

    trial = 0
    # quick initial short polish already done upstream; skip
    while time.time() < EXPL_END:
        trial += 1
        base = X_best
        # analyse contacts of base every ~10 trials
        if trial % 8 == 1:
            Gtmp = base @ base.T
            np.fill_diagonal(Gtmp, -2.0)
            cmax = Gtmp.max()
            contacts = np.count_nonzero(Gtmp > cmax - 5e-4, axis=1)
            worst, _ = worst_per_point(base)
            order_worst = np.argsort(-worst)
            order_cont = np.argsort(contacts)
        mode = trial % 6
        X0 = base.copy()
        if mode == 0:
            # small global shake
            X0 = X0 + 0.015 * rng.randn(n, d)
        elif mode == 1:
            # shake high-contact overloaded points hard
            Gtmp = base @ base.T
            np.fill_diagonal(Gtmp, -2.0)
            cmax = Gtmp.max()
            contacts = np.count_nonzero(Gtmp > cmax - 5e-4, axis=1)
            hi = np.argsort(-contacts)[:10]
            X0[hi] += 0.12 * rng.randn(len(hi), d)
        elif mode == 2:
            # shake low-contact points + medium noise
            Gtmp = base @ base.T
            np.fill_diagonal(Gtmp, -2.0)
            cmax = Gtmp.max()
            contacts = np.count_nonzero(Gtmp > cmax - 5e-4, axis=1)
            lo = np.argsort(contacts)[:8]
            X0[lo] += 0.25 * rng.randn(len(lo), d)
            X0 += 0.01 * rng.randn(n, d)
        elif mode == 3:
            # kill worst 1-2 and refill in hole
            worst, _ = worst_per_point(base)
            k = 1 if (trial % 2 == 0) else 2
            kill = np.argsort(-worst)[:k]
            # randomize among top 8 sometimes
            if rng.rand() < 0.5:
                pool = np.argsort(-worst)[:8]
                kill = rng.choice(pool, size=k, replace=False)
            X0 = hole_reinsert(X0, kill, S=4000)
            X0 += 0.02 * rng.randn(n, d)
        elif mode == 4:
            # medium global hop
            X0 = X0 + 0.08 * rng.randn(n, d)
        else:
            # large hop + occasional full restart piece
            X0 = X0 + 0.20 * rng.randn(n, d)
            if rng.rand() < 0.2:
                # replace 3 random points by random far points
                kill = rng.choice(n, size=3, replace=False)
                X0 = hole_reinsert(X0, kill, S=2000)

        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        # annealed smoothing: two stages, beta grows
        if mode in (0, 1):
            betas, iters = [60, 180], [50, 60]
        elif mode == 5:
            betas, iters = [50, 120, 300], [40, 40, 40]
        else:
            betas, iters = [80, 220], [55, 55]
        X1 = smooth_refine(X0, betas, iters)
        v1 = _max_cos(X1)
        # decide to LP polish
        margin = 0.012 if mode == 5 else 0.008
        if v1 < best_val + margin and time.time() < EXPL_END:
            # short LP polish 1.2-2s
            budget = 1.5
            if v1 < best_val + 0.001:
                budget = 2.5
            if time.time() + budget > EXPL_END + 2 and time.time() < EXPL_END:
                budget = max(0.6, EXPL_END - time.time())
            X2, v2 = _lp_polish_deadline(X1, min(time.time() + budget, HARD_END - 1))
            if v2 < best_val - 1e-12:
                X_best = X2
                best_val = v2
        else:
            if v1 < best_val - 1e-12:
                # improbable without polish, but accept
                X_best = X1
                best_val = v1
        # opportunistic: if time very low break
        if time.time() > HARD_END - 3:
            break

    # final squeeze with remaining time
    if time.time() < HARD_END - 0.5:
        X_best, best_val = _lp_polish_deadline(X_best, HARD_END - 0.2)
    X_best = X_best / np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best
