import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _rowmax(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G

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
                            "maxfun": maxiter * 5, "maxcor": 15})
    Y = res.x.reshape(n, d)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def _lp_polish(x, seconds, t0):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 2.5e-3
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < seconds and radius > 1e-14:
        cmat = x @ x.T
        iu_c = cmat[iu]
        m = float(iu_c.max())
        sel = iu_c > m - max(25.0 * radius, 8e-7)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 5000:
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
    def remaining():
        return TIME_BUDGET - (time.time() - t_start)
    rng = np.random.default_rng(7)
    n, d = 250, 6
    try:
        x0 = np.load("best.npy")
    except Exception:
        x0 = None
    try:
        rec = np.load("record.npy")
    except Exception:
        rec = None
    cands = []
    for z in (x0, rec):
        try:
            z = np.asarray(z, dtype=np.float64)
            if z.shape == (n, d):
                z = z / np.linalg.norm(z, axis=1, keepdims=True)
                cands.append(z)
        except Exception:
            pass
    if not cands:
        z = rng.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    best = min(cands, key=_max_cos).copy()
    best_val = _max_cos(best)

    # Stage 1: settle to nearby tied optimum at moderate beta
    cur = best.copy()
    if remaining() > 60:
        for beta, it in [(3000.0, 150), (12000.0, 150)]:
            if remaining() < 45:
                break
            try:
                cur = _smooth_opt(cur, beta, it)
            except Exception:
                break
            v = _max_cos(cur)
            if v < best_val:
                best = cur.copy(); best_val = v
    else:
        cur = best.copy()
    # quick squeeze
    if remaining() > 40:
        try:
            sq = _lp_polish(best, min(5.0, remaining() - 32.0), time.time())
            v = _max_cos(sq)
            if v < best_val:
                best = sq; best_val = v; cur = sq.copy()
        except Exception:
            pass

    trial = 0
    # Stage 2: bottleneck-breaking basin hopping
    while remaining() > 26 and trial < 100:
        trial += 1
        mode = trial % 4
        Y = None
        try:
            rm, G = _rowmax(best)
            # worst pair
            iu = np.triu_indices(n, 1)
            gv = G[iu]
            k = int(np.argmax(gv))
            a, b = int(iu[0][k]), int(iu[1][k])
            order = np.argsort(-rm)
        except Exception:
            a, b = 0, 1
            order = np.arange(n)
        try:
            if mode == 0 or mode == 1:
                # pair-splitting: push a,b apart symmetrically + shake neighbourhood
                Y = best.copy()
                eps = 0.008 + 0.012 * rng.random() if mode == 0 else 0.003 + 0.006 * rng.random()
                # move a away from b and vice versa
                na = Y[a] - eps * Y[b]
                nb = Y[b] - eps * Y[a]
                na /= np.linalg.norm(na); nb /= np.linalg.norm(nb)
                Y[a] = na; Y[b] = nb
                # neighbours: points close to a or b
                dots_a = Y @ Y[a]
                dots_b = Y @ Y[b]
                neigh = np.where((dots_a > best_val - 0.02) | (dots_b > best_val - 0.02))[0]
                neigh = [int(v) for v in neigh if v != a and v != b]
                sig = 0.008 if mode == 0 else 0.003
                for v in neigh[:12]:
                    Y[v] += rng.standard_normal(d) * sig * 0.5
                # a few random rattlers shaken too
                if mode == 0:
                    extra = rng.integers(0, n, size=4)
                    for v in extra:
                        Y[int(v)] += rng.standard_normal(d) * 0.004
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                beta = 8000.0 if mode == 0 else 15000.0
                Y2 = _smooth_opt(Y, beta, 110)
                v2 = _max_cos(Y2)
                if v2 < best_val - 1e-12:
                    best = Y2.copy(); best_val = v2
                    sq = _lp_polish(best, min(2.5, remaining() - 24.0), time.time())
                    v = _max_cos(sq)
                    if v < best_val:
                        best = sq; best_val = v
                elif v2 < best_val + 2e-7 and rng.random() < 0.15 and remaining() > 60:
                    # sideways walk to new basin
                    best = Y2.copy(); best_val = v2
            elif mode == 2:
                # cluster shake of top pairs
                try:
                    kk = 25
                    idx = np.argpartition(-gv, kk)[:kk]
                    pts = set()
                    for t in idx:
                        pts.add(int(iu[0][t])); pts.add(int(iu[1][t]))
                    pts = list(pts)[:25]
                except Exception:
                    pts = list(order[:10])
                Y = best.copy()
                sig = 0.006 + 0.008 * rng.random()
                for v in pts:
                    Y[int(v)] += rng.standard_normal(d) * sig
                Y += rng.standard_normal(Y.shape) * 0.0004
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                Y2 = _smooth_opt(Y, 10000.0, 110)
                v2 = _max_cos(Y2)
                if v2 < best_val:
                    best = Y2.copy(); best_val = v2
                    sq = _lp_polish(best, min(2.5, remaining() - 24.0), time.time())
                    v = _max_cos(sq)
                    if v < best_val:
                        best = sq; best_val = v
            else:
                # void reinsertion of worst point
                w = int(order[0])
                Nc = 6000
                cand = rng.standard_normal((Nc, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                sc = cand @ best.T
                sc[:, w] = -2.0
                mx = sc.max(axis=1)
                bi = int(np.argmin(mx))
                if float(mx[bi]) < best_val - 2e-7:
                    Y = best.copy()
                    Y[w] = cand[bi]
                    # local polish of w
                    y = Y[w].copy()
                    others = Y[np.arange(n) != w]
                    for it in range(150):
                        dots = others @ y
                        k2 = int(np.argmax(dots))
                        gvec = others[k2] - (others[k2] @ y) * y
                        gn = np.linalg.norm(gvec)
                        if gn < 1e-12:
                            break
                        step = 0.004 / (1.0 + it * 0.02)
                        y = y - step * gvec / gn
                        y /= np.linalg.norm(y)
                    Y[w] = y
                    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                    Y2 = _smooth_opt(Y, 9000.0, 100)
                    v2 = _max_cos(Y2)
                    if v2 < best_val:
                        best = Y2.copy(); best_val = v2
                        sq = _lp_polish(best, min(2.5, remaining() - 24.0), time.time())
                        v = _max_cos(sq)
                        if v < best_val:
                            best = sq; best_val = v
        except Exception:
            continue

    if remaining() > 5:
        try:
            polished = _lp_polish(best, remaining() - 1.5, time.time())
            vp = _max_cos(polished)
            if vp < best_val:
                best = polished; best_val = vp
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    best = np.where(np.isfinite(best), best, 0.0)
    nrm = np.linalg.norm(best, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    best = best / nrm
    return best
