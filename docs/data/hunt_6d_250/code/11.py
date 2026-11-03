import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _smooth_opt(X0, beta, maxiter):
    n, d = X0.shape
    Y0 = X0.copy()

    def fg(Yflat):
        Y = Yflat.reshape(n, d)
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        X = Y / nrm
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(G.max())
        D = beta * (G - M)
        E = np.empty_like(D)
        np.exp(D, out=E)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if S <= 0:
            S = 1e-300
        loss = M + np.log(S) / beta
        W = E / S
        GX = 2.0 * (W @ X)
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nrm
        return loss, GY.ravel()

    res = minimize(fg, Y0.ravel(), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": 1e-20, "gtol": 1e-12,
                            "maxfun": maxiter * 5, "maxcor": 20})
    Y = res.x.reshape(n, d)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y


def _lp_polish(x, seconds, t0):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 2e-3
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < seconds and radius > 1e-14:
        cmat = x @ x.T
        iu_c = cmat[iu]
        m = float(iu_c.max())
        sel = iu_c > m - max(25.0 * radius, 5e-7)
        if np.count_nonzero(sel) > 5000:
            thr = np.sort(iu_c[sel])[-5000]
            sel = iu_c >= thr
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-iu_c[sel], A_eq=A_eq,
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x


def solve():
    t_start = time.time()
    TIME_BUDGET = 168.0
    np.random.seed(0)
    rng = np.random.default_rng(1)
    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            x0 = rng.standard_normal((250, 6))
    x0 = np.asarray(x0, dtype=np.float64)
    if x0.shape != (250, 6):
        tmp = rng.standard_normal((250, 6))
        try:
            m = min(x0.shape[0], 250)
            dd = min(x0.shape[1], 6)
            tmp[:m, :dd] = x0[:m, :dd]
        except Exception:
            pass
        x0 = tmp
    x0 = x0 / np.linalg.norm(x0, axis=1, keepdims=True)
    try:
        rec = np.load("record.npy").astype(float)
        rec = rec / np.linalg.norm(rec, axis=1, keepdims=True)
        if rec.shape == (250, 6) and _max_cos(rec) < _max_cos(x0):
            x0 = rec.copy()
    except Exception:
        pass

    best = x0.copy()
    best_val = _max_cos(best)

    def remaining():
        return TIME_BUDGET - (time.time() - t_start)

    n, d = 250, 6
    iu0, iu1 = np.triu_indices(n, 1)

    # Stage 1: short annealed smoothing
    cur = best.copy()
    for beta, iters in [(800.0, 200), (4000.0, 200), (15000.0, 200)]:
        if remaining() < 40:
            break
        try:
            cur = _smooth_opt(cur, beta, iters)
        except Exception:
            break
        v = _max_cos(cur)
        if v < best_val:
            best_val = v
            best = cur.copy()

    # Stage 2: quick LP squeeze to true local optimum
    if remaining() > 35:
        try:
            sq = _lp_polish(best, min(8.0, remaining() - 28.0), time.time())
            v = _max_cos(sq)
            if v < best_val:
                best = sq
                best_val = v
        except Exception:
            pass

    def worst_order(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        rm = G.max(axis=1)
        return np.argsort(-rm), rm, G

    trial = 0
    # Stage 3: basin hopping with void reinsertion + small shakes
    while remaining() > 28 and trial < 80:
        trial += 1
        try:
            order, rm, G = worst_order(best)
        except Exception:
            order = np.arange(n)
        improved_this = False
        # --- void reinsertion for worst 2 points (cheap) ---
        if trial % 2 == 1 and remaining() > 20:
            try:
                for wi in range(min(2, n)):
                    w = int(order[wi])
                    # max among pairs not involving w
                    G2 = best @ best.T
                    np.fill_diagonal(G2, -2.0)
                    # mask row/col w
                    G2[w, :] = -2.0
                    G2[:, w] = -2.0
                    m_others = float(G2.max())
                    if m_others >= best_val:
                        continue
                    # sample voids
                    Nc = 4096
                    cand = rng.standard_normal((Nc, d))
                    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                    # score = max dot to others
                    sc = cand @ best.T
                    # ignore self column by setting to -2 (column w corresponds to old position)
                    sc[:, w] = -2.0
                    mx = sc.max(axis=1)
                    bi = int(np.argmin(mx))
                    if float(mx[bi]) < best_val - 1e-12:
                        Y = best.copy()
                        Y[w] = cand[bi]
                        # local single-point polish: few projected subgradient steps on w only
                        y = Y[w].copy()
                        others = Y[np.arange(n) != w]
                        for _ in range(200):
                            dots = others @ y
                            k = int(np.argmax(dots))
                            g = others[k] - (others[k] @ y) * y
                            gn = np.linalg.norm(g)
                            if gn < 1e-12:
                                break
                            step = 0.005 / (1.0 + _ * 0.02)
                            y = y - step * g / gn
                            y /= np.linalg.norm(y)
                        Y[w] = y
                        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                        v = _max_cos(Y)
                        if v < best_val:
                            # short smooth to settle
                            try:
                                Y2 = _smooth_opt(Y, 8000.0, 80)
                                v2 = _max_cos(Y2)
                                if v2 < v:
                                    Y, v = Y2, v2
                            except Exception:
                                pass
                            best = Y.copy()
                            best_val = v
                            improved_this = True
                            break
            except Exception:
                pass
            if improved_this:
                # squeeze immediately
                try:
                    sq = _lp_polish(best, min(3.0, remaining() - 25.0), time.time())
                    v = _max_cos(sq)
                    if v < best_val:
                        best = sq
                        best_val = v
                except Exception:
                    pass
                continue

        # --- small shake and re-optimize ---
        sigmas = [0.0025, 0.0012, 0.0006, 0.004, 0.0003]
        sigma = sigmas[trial % len(sigmas)]
        K = 3 + (trial % 6)
        victims = list(order[:K])
        if trial % 3 == 0:
            victims += list(rng.integers(0, n, size=3))
        victims = list(set(victims))
        Y = best.copy()
        Y[victims] += rng.standard_normal((len(victims), d)) * sigma
        if trial % 5 == 0:
            Y += rng.standard_normal(Y.shape) * 0.0003
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        beta = 6000.0 if (trial % 2 == 0) else 12000.0
        try:
            Y2 = _smooth_opt(Y, beta, 120)
        except Exception:
            continue
        v2 = _max_cos(Y2)
        if v2 < best_val:
            # optional second higher-beta polish
            if remaining() > 32:
                try:
                    Y3 = _smooth_opt(Y2, 25000.0, 60)
                    v3 = _max_cos(Y3)
                    if v3 < v2:
                        Y2, v2 = Y3, v3
                except Exception:
                    pass
            best = Y2.copy()
            best_val = v2
            # quick LP squeeze on improvement
            try:
                sq = _lp_polish(best, min(3.0, remaining() - 25.0), time.time())
                v = _max_cos(sq)
                if v < best_val:
                    best = sq
                    best_val = v
            except Exception:
                pass

    # Stage 4: final LP squeeze
    if remaining() > 6:
        try:
            polished = _lp_polish(best, remaining() - 2.0, time.time())
            vp = _max_cos(polished)
            if vp < best_val:
                best = polished
                best_val = vp
        except Exception:
            pass

    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    best = np.where(np.isfinite(best), best, 0.0)
    nrm = np.linalg.norm(best, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    best = best / nrm
    return best
