import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _smooth_opt(X0, beta, maxiter, mcor=10):
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
        if S <= 0 or not np.isfinite(S):
            S = 1e-300
        loss = M + np.log(S) / beta
        W = E / S
        GX = 2.0 * (W @ X)
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nrm
        return loss, GY.ravel()
    try:
        res = minimize(fg, Y0.ravel(), jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-20, "gtol": 1e-12,
                                "maxfun": maxiter * 5, "maxcor": mcor})
        Y = res.x.reshape(n, d)
    except Exception:
        Y = Y0
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def _lp_polish(x, seconds, t0):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 2.0e-3
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < seconds and radius > 1e-14:
        cmat = x @ x.T
        iu_c = cmat[iu]
        m = float(iu_c.max())
        sel = iu_c > m - max(20.0 * radius, 5e-7)
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

def _void_point(base, skip, rng, Nc=12000, refine=150):
    n, d = base.shape
    cand = rng.standard_normal((Nc, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    sc = cand @ base.T
    if isinstance(skip, (list, tuple, np.ndarray)):
        for w in skip:
            sc[:, int(w)] = -2.0
    else:
        sc[:, int(skip)] = -2.0
    mx = sc.max(axis=1)
    bi = int(np.argmin(mx))
    y = cand[bi].copy()
    mask = np.ones(n, dtype=bool)
    try:
        if isinstance(skip, (list, tuple, np.ndarray)):
            for w in skip:
                mask[int(w)] = False
        else:
            mask[int(skip)] = False
    except Exception:
        pass
    others = base[mask]
    for s in range(refine):
        dots = others @ y
        k = int(np.argmax(dots))
        gv = others[k] - (others[k] @ y) * y
        gn = np.linalg.norm(gv)
        if gn < 1e-12:
            break
        step = 0.01 / (1.0 + s * 0.03)
        y = y - step * gv / gn
        y /= np.linalg.norm(y)
    return y

def solve():
    t_start = time.time()
    TIME_BUDGET = 168.0
    def remaining():
        return TIME_BUDGET - (time.time() - t_start)
    rng = np.random.default_rng(12345)
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
                cands.append((z, _max_cos(z)))
        except Exception:
            pass
    if not cands:
        z = rng.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append((z, _max_cos(z)))
    cands.sort(key=lambda t: t[1])
    best = cands[0][0].copy()
    best_val = float(cands[0][1])
    cur = best.copy()
    cur_val = best_val
    iu0, iu1 = np.triu_indices(n, 1)

    # Stage 1: collective plateau descent at moderate beta
    for beta, it in [(2500.0, 180), (5000.0, 180)]:
        if remaining() < 50:
            break
        try:
            c = _smooth_opt(cur, beta, it)
            v = _max_cos(c)
            if v < best_val:
                best = c.copy(); best_val = v
            if v < cur_val + 1e-12:
                cur = c.copy(); cur_val = v
        except Exception:
            break
    if remaining() > 40:
        try:
            sq = _lp_polish(best, min(4.0, remaining() - 33.0), time.time())
            v = _max_cos(sq)
            if v < best_val:
                best = sq.copy(); best_val = v
                cur = sq.copy(); cur_val = v
        except Exception:
            pass

    trial = 0
    while remaining() > 24 and trial < 90:
        trial += 1
        base = cur if (trial % 3 == 0) else best
        try:
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            order = np.argsort(-rowmax)
            flat = int(np.argmax(G))
            wi, wj = flat // n, flat % n
        except Exception:
            order = np.arange(n); wi, wj = 0, 1
            G = base @ base.T
        Y = base.copy()
        mode = trial % 5
        try:
            if mode == 0:
                # plateau shake: shake points involved in top ~1300 pairs
                gv = G[iu0, iu1]
                K = 1300
                top = np.argpartition(gv, -K)[-K:]
                pts = np.unique(np.concatenate([iu0[top], iu1[top]]))
                # shake all plateau points moderately + global jitter
                Y[pts] += rng.standard_normal((len(pts), d)) * 0.006
                Y += rng.standard_normal(Y.shape) * 0.0008
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                beta, iters = 3500.0, 110
            elif mode == 1:
                # worst-pair split with orthogonal noise
                sig = 0.015 + 0.01 * ((trial // 5) % 3) / 2.0
                a = Y[wi].copy(); b = Y[wj].copy()
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
                Y[wi] = Y[wi] + da * sig + rng.standard_normal(d) * sig * 0.5
                Y[wj] = Y[wj] + db * sig + rng.standard_normal(d) * sig * 0.5
                # shake their near neighbours
                for kk in np.argsort(-G[wi])[:5]:
                    if kk != wi and kk != wj:
                        Y[kk] += rng.standard_normal(d) * sig * 0.35
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                beta, iters = 5000.0, 110
            elif mode == 2:
                # multi-point void reinsertion (topology change)
                K = 3 + (trial % 3)
                victims = [int(v) for v in order[:K]]
                for w in victims:
                    skip = [v for v in victims if v != w] + [w]
                    # void wrt current Y (with already replaced points)
                    ynew = _void_point(Y, skip, rng, Nc=10000, refine=120)
                    Y[w] = ynew
                Y += rng.standard_normal(Y.shape) * 0.0005
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                beta, iters = 4000.0, 110
            elif mode == 3:
                # random block + worst focus
                K = 30
                idx = rng.integers(0, n, size=K)
                Y[idx] += rng.standard_normal((K, d)) * 0.008
                Y[int(order[0])] += rng.standard_normal(d) * 0.008
                Y[int(order[1])] += rng.standard_normal(d) * 0.008
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                beta, iters = 7000.0, 100
            else:
                # global annealed jitter
                sig = 0.004 + 0.008 * (rng.random())
                Y += rng.standard_normal(Y.shape) * sig
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                beta, iters = 3000.0, 100
        except Exception:
            continue
        try:
            Y2 = _smooth_opt(Y, beta, iters)
        except Exception:
            continue
        v2 = _max_cos(Y2)
        if v2 < best_val - 1e-12:
            # tie with higher beta once
            if remaining() > 30:
                try:
                    Y3 = _smooth_opt(Y2, 18000.0, 50)
                    v3 = _max_cos(Y3)
                    if v3 < v2:
                        Y2, v2 = Y3, v3
                except Exception:
                    pass
            best = Y2.copy(); best_val = v2
            cur = Y2.copy(); cur_val = v2
            if remaining() > 27:
                try:
                    sq = _lp_polish(best, min(1.8, remaining() - 25.0), time.time())
                    v = _max_cos(sq)
                    if v < best_val:
                        best = sq.copy(); best_val = v
                        cur = sq.copy(); cur_val = v
                except Exception:
                    pass
        elif v2 < cur_val + 6e-6 and rng.random() < 0.35:
            cur = Y2.copy(); cur_val = v2
        elif v2 < cur_val:
            cur = Y2.copy(); cur_val = v2

    if remaining() > 9:
        try:
            f = _smooth_opt(best, 12000.0, 90)
            v = _max_cos(f)
            if v < best_val:
                best = f; best_val = v
        except Exception:
            pass
    if remaining() > 5:
        try:
            polished = _lp_polish(best, min(6.0, remaining() - 1.5), time.time())
            v = _max_cos(polished)
            if v < best_val:
                best = polished; best_val = v
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
