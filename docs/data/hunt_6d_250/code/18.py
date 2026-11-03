import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _smooth_opt(X0, beta, maxiter, mcor=12):
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
        np.clip(D, -30.0, 0.0, out=D)
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
                            "maxfun": maxiter * 5, "maxcor": mcor})
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
        sel = iu_c > m - max(20.0 * radius, 3e-7)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 4000:
            thr = np.sort(iu_c[sel])[-4000]
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
            x, best, radius = y, v, min(radius * 1.5, 0.015)
        else:
            radius *= 0.5
    return x

def solve():
    t_start = time.time()
    TIME_BUDGET = 168.0
    rng = np.random.default_rng(7)
    np.random.seed(7)
    n, d = 250, 6
    def remaining():
        return TIME_BUDGET - (time.time() - t_start)
    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            x0 = rng.standard_normal((n, d))
    x0 = np.asarray(x0, dtype=np.float64)
    if x0.shape != (n, d):
        tmp = rng.standard_normal((n, d))
        try:
            m = min(x0.shape[0], n); dd = min(x0.shape[1], d)
            tmp[:m, :dd] = x0[:m, :dd]
        except Exception:
            pass
        x0 = tmp
    x0 = x0 / np.linalg.norm(x0, axis=1, keepdims=True)
    try:
        rec = np.load("record.npy").astype(float)
        rec = rec / np.linalg.norm(rec, axis=1, keepdims=True)
        if rec.shape == (n, d) and _max_cos(rec) < _max_cos(x0):
            x0 = rec.copy()
    except Exception:
        pass
    best = x0.copy()
    best_val = _max_cos(best)
    # quick escalating smooth to settle plateau
    cur = best.copy()
    cur_val = best_val
    for beta, it in [(4000.0, 150), (15000.0, 150)]:
        if remaining() < 45:
            break
        try:
            c = _smooth_opt(cur, beta, it)
            v = _max_cos(c)
            if v < best_val:
                best = c.copy(); best_val = v
            if v < cur_val + 1e-9:
                cur = c.copy(); cur_val = v
        except Exception:
            break
    if remaining() > 35:
        try:
            sq = _lp_polish(best, min(6.0, remaining() - 28.0), time.time())
            v = _max_cos(sq)
            if v < best_val:
                best = sq.copy(); best_val = v
                cur = sq.copy(); cur_val = v
        except Exception:
            pass
    iu0, iu1 = np.triu_indices(n, 1)
    trial = 0
    # basin walking with bottleneck surgery
    while remaining() > 26 and trial < 120:
        trial += 1
        # choose base: mostly best, sometimes cur for uphill walk
        base = best if (trial % 4 != 0) else cur
        try:
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            order = np.argsort(-rowmax)
            # worst pair
            flat = np.argmax(G)
            wi, wj = flat // n, flat % n
        except Exception:
            order = np.arange(n); wi, wj = 0, 1
        Y = base.copy()
        mode = trial % 7
        try:
            if mode == 0:
                # worst-pair split: push apart along tangent
                sigma = 0.008 + 0.004 * ((trial // 7) % 3)
                # tangent directions
                a = Y[wi].copy(); b = Y[wj].copy()
                # move a away from b, b away from a plus random orthogonal
                da = a - (a @ b) * b
                db = b - (b @ a) * a
                na = np.linalg.norm(da); nb = np.linalg.norm(db)
                if na > 1e-12:
                    da /= na
                else:
                    da = rng.standard_normal(d); da /= np.linalg.norm(da)
                if nb > 1e-12:
                    db /= nb
                else:
                    db = rng.standard_normal(d); db /= np.linalg.norm(db)
                Y[wi] = Y[wi] + da * sigma + rng.standard_normal(d) * sigma * 0.4
                Y[wj] = Y[wj] + db * sigma + rng.standard_normal(d) * sigma * 0.4
                # also shake top neighbours of wi,wj
                nbr = np.argsort(-G[wi])[:4]
                for k in nbr[1:]:
                    Y[k] += rng.standard_normal(d) * sigma * 0.3
                nbr2 = np.argsort(-G[wj])[:4]
                for k in nbr2[1:]:
                    if k != wi:
                        Y[k] += rng.standard_normal(d) * sigma * 0.3
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                beta, iters = 12000.0, 120
            elif mode == 1:
                K = 4 + (trial % 5)
                victims = list(order[:K])
                sigma = 0.006 if (trial % 2 == 0) else 0.0025
                Y[victims] += rng.standard_normal((len(victims), d)) * sigma
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                beta, iters = 8000.0, 110
            elif mode == 2:
                # void reinsertion of worst point
                w = int(order[0])
                G2 = base @ base.T
                np.fill_diagonal(G2, -2.0)
                G2[w, :] = -2.0; G2[:, w] = -2.0
                # sample voids
                Nc = 8192
                cand = rng.standard_normal((Nc, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                sc = cand @ base.T
                sc[:, w] = -2.0
                mx = sc.max(axis=1)
                bi = int(np.argmin(mx))
                y = cand[bi].copy()
                others = base[np.arange(n) != w]
                for s in range(250):
                    dots = others @ y
                    k = int(np.argmax(dots))
                    gvec = others[k] - (others[k] @ y) * y
                    gn = np.linalg.norm(gvec)
                    if gn < 1e-12:
                        break
                    step = 0.004 / (1.0 + s * 0.02)
                    y = y - step * gvec / gn
                    y /= np.linalg.norm(y)
                Y[w] = y
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                beta, iters = 8000.0, 100
            elif mode == 3:
                # random block shake
                K = 25
                idx = rng.integers(0, n, size=K)
                Y[idx] += rng.standard_normal((K, d)) * 0.004
                Y += rng.standard_normal(Y.shape) * 0.0004
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                beta, iters = 15000.0, 110
            elif mode == 4:
                # targeted top-pairs shake
                flat_idx = np.argpartition(G[iu0, iu1] if False else (base @ base.T)[iu0, iu1], -30)[-30:]
                # simpler: get top 30 pairs
                gv = (base @ base.T)[iu0, iu1]
                top = np.argpartition(gv, -30)[-30:]
                pts = set()
                for t in top:
                    pts.add(int(iu0[t])); pts.add(int(iu1[t]))
                pts = list(pts)[:20]
                Y[pts] += rng.standard_normal((len(pts), d)) * 0.0035
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                beta, iters = 20000.0, 110
            elif mode == 5:
                Y += rng.standard_normal(Y.shape) * 0.0012
                # extra kick to worst pair
                Y[wi] += rng.standard_normal(d) * 0.004
                Y[wj] += rng.standard_normal(d) * 0.004
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                beta, iters = 6000.0, 110
            else:
                # large jump of 2 worst points to voids + keep rest
                for w in [int(order[0]), int(order[1])][:1]:
                    Nc = 4096
                    cand = rng.standard_normal((Nc, d))
                    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                    sc = cand @ base.T
                    sc[:, w] = -2.0
                    mx = sc.max(axis=1)
                    bi = int(np.argmin(mx))
                    Y[w] = cand[bi]
                Y += rng.standard_normal(Y.shape) * 0.0006
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                beta, iters = 10000.0, 120
        except Exception:
            continue
        try:
            Y2 = _smooth_opt(Y, beta, iters)
        except Exception:
            continue
        v2 = _max_cos(Y2)
        # uphill-tolerant walk for cur
        if v2 < cur_val or rng.random() < 0.15 and v2 < cur_val + 8e-6:
            cur = Y2.copy(); cur_val = v2
        if v2 < best_val - 1e-12:
            # second-stage high-beta tie
            if remaining() > 30:
                try:
                    Y3 = _smooth_opt(Y2, 30000.0, 60)
                    v3 = _max_cos(Y3)
                    if v3 < v2:
                        Y2, v2 = Y3, v3
                except Exception:
                    pass
            best = Y2.copy(); best_val = v2
            cur = Y2.copy(); cur_val = v2
            try:
                sq = _lp_polish(best, min(2.5, remaining() - 24.0), time.time())
                v = _max_cos(sq)
                if v < best_val:
                    best = sq.copy(); best_val = v
                    cur = sq.copy(); cur_val = v
            except Exception:
                pass
    # final tie + squeeze
    if remaining() > 8:
        try:
            f = _smooth_opt(best, 35000.0, 100)
            if _max_cos(f) < best_val:
                best = f; best_val = _max_cos(f)
        except Exception:
            pass
    if remaining() > 5:
        try:
            polished = _lp_polish(best, remaining() - 1.5, time.time())
            if _max_cos(polished) < best_val:
                best = polished; best_val = _max_cos(polished)
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    best = np.where(np.isfinite(best), best, 0.0)
    nrm = np.linalg.norm(best, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    best = best / nrm
    return best
