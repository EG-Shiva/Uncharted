import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _smooth_opt(X0, beta, maxiter, maxcor=12):
    n, d = X0.shape
    Y0 = X0.copy()
    def fg(Yflat):
        Y = Yflat.reshape(n, d)
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        X = Y / nrm
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        D = beta * (G - M)
        np.clip(D, -35.0, 0.0, out=D)
        E = np.empty_like(D)
        np.exp(D, out=E)
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E))
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
                       options={"maxiter": maxiter, "ftol": 1e-20,
                                "gtol": 1e-12, "maxfun": maxiter * 5,
                                "maxcor": maxcor})
        Y = res.x.reshape(n, d)
    except Exception:
        Y = Y0
    nrm = np.linalg.norm(Y, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    return Y / nrm

def _pair_refine(X, a, b, beta=8000.0, maxiter=80):
    n, d = X.shape
    F = X.copy()
    Ya0 = X[a].copy()
    Yb0 = X[b].copy()
    mask = np.ones(n, dtype=bool)
    mask[a] = False
    mask[b] = False
    Fo = F[mask]  # (n-2,d)
    def fg(yflat):
        Ya = yflat[:d]; Yb = yflat[d:]
        na = max(np.linalg.norm(Ya), 1e-12)
        nb = max(np.linalg.norm(Yb), 1e-12)
        Xa = Ya / na; Xb = Yb / nb
        dab = float(Xa @ Xb)
        da = Fo @ Xa  # (m,)
        db = Fo @ Xb
        c = np.concatenate(([dab], da, db))
        M = float(np.max(c))
        D = beta * (c - M)
        np.clip(D, -35.0, 0.0, out=D)
        E = np.exp(D)
        S = float(np.sum(E))
        if S <= 0:
            S = 1e-300
        loss = M + np.log(S) / beta
        W = E / S
        w0 = W[0]
        m = Fo.shape[0]
        Wa = W[1:1+m]; Wb = W[1+m:]
        gXa = w0 * Xb + (Wa[:, None] * Fo).sum(axis=0)
        gXb = w0 * Xa + (Wb[:, None] * Fo).sum(axis=0)
        gYa = (gXa - float(gXa @ Xa) * Xa) / na
        gYb = (gXb - float(gXb @ Xb) * Xb) / nb
        return loss, np.concatenate((gYa, gYb))
    try:
        y0 = np.concatenate((Ya0, Yb0))
        res = minimize(fg, y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-20, "gtol": 1e-12,
                                "maxfun": maxiter*5, "maxcor": 10})
        y = res.x
        Xa = y[:d] / max(np.linalg.norm(y[:d]), 1e-12)
        Xb = y[d:] / max(np.linalg.norm(y[d:]), 1e-12)
        Xn = X.copy()
        Xn[a] = Xa; Xn[b] = Xb
        return Xn
    except Exception:
        return X

def _single_refine(X, w, beta=8000.0, maxiter=60):
    n, d = X.shape
    mask = np.ones(n, dtype=bool); mask[w] = False
    Fo = X[mask]
    Y0 = X[w].copy()
    def fg(yflat):
        na = max(np.linalg.norm(yflat), 1e-12)
        Xw = yflat / na
        c = Fo @ Xw
        M = float(np.max(c))
        D = beta * (c - M)
        np.clip(D, -35.0, 0.0, out=D)
        E = np.exp(D)
        S = float(np.sum(E))
        if S <= 0:
            S = 1e-300
        loss = M + np.log(S) / beta
        W = E / S
        gX = (W[:, None] * Fo).sum(axis=0)
        gY = (gX - float(gX @ Xw) * Xw) / na
        return loss, gY
    try:
        res = minimize(fg, Y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-20, "gtol": 1e-12,
                                "maxfun": maxiter*5, "maxcor": 8})
        Xw = res.x / max(np.linalg.norm(res.x), 1e-12)
        Xn = X.copy(); Xn[w] = Xw
        return Xn
    except Exception:
        return X

def _lp_polish(x, deadline, k_top=4500, radius_init=2e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius_init
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline - 0.8 and radius > 1e-14:
        Gall = x @ x.T
        c_all = Gall[iu0, iu1]
        m = float(np.max(c_all))
        sel = c_all > m - max(20.0 * radius, 8e-7)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > k_top:
            thr = np.sort(c_all[sel])[-k_top]
            sel = c_all >= thr
        I = iu0[sel]; J = iu1[sel]
        c_sel = c_all[sel]
        p = I.shape[0]
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            bounds = [(-radius, radius)] * (n * d) + [(None, None)]
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n), bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best - 1e-12:
            x, best = y, v
            radius = min(radius * 1.5, 0.012)
        else:
            radius *= 0.5
        if time.time() > deadline - 1.5:
            break
    return x

def _deep_hole(X, exclude, rng, nc=2048):
    n, d = X.shape
    cand = rng.standard_normal((nc, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    S = cand @ X.T
    if np.ndim(exclude) == 0:
        S[:, int(exclude)] = -2.0
    else:
        for w in exclude:
            S[:, int(w)] = -2.0
    mx = S.max(axis=1)
    return cand[int(np.argmin(mx))]

def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    HOP_END = t0 + 122.0
    rng = np.random.default_rng(7)
    n, d = 250, 6
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=float)
            if a.shape == (n, d):
                a = np.nan_to_num(a)
                a /= np.linalg.norm(a, axis=1, keepdims=True)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        tmp = rng.standard_normal((n, d))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        cands.append(tmp)
    best = min(cands, key=_max_cos).copy()
    best_val = _max_cos(best)
    cur = best.copy()
    # Stage 1: annealed global smoothing
    for beta, it in [(800.0, 100), (4000.0, 100), (12000.0, 90)]:
        if time.time() > t0 + 30.0 or time.time() > HARD - 60:
            break
        try:
            cur = _smooth_opt(cur, beta, it)
        except Exception:
            break
        v = _max_cos(cur)
        if v < best_val:
            best, best_val = cur.copy(), v
        # micro pair squeeze on current bottleneck
        try:
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            ij = int(np.argmax(G))
            a0, b0 = divmod(ij, n)
            cur2 = _pair_refine(cur, int(a0), int(b0), beta=8000.0, maxiter=60)
            v2 = _max_cos(cur2)
            if v2 < v:
                cur = cur2; v = v2
                if v < best_val:
                    best, best_val = cur.copy(), v
        except Exception:
            pass
    cur = best.copy()
    trial = 0
    sigmas = [0.003, 0.006, 0.011, 0.002, 0.008, 0.016]
    while time.time() < HOP_END and trial < 70:
        trial += 1
        if time.time() > HARD - 55:
            break
        try:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            rm = G.max(axis=1)
            order = np.argsort(-rm)
            ij = int(np.argmax(G))
            pa, pb = divmod(ij, n)
        except Exception:
            order = np.arange(n); pa, pb = 0, 1
        sigma = sigmas[trial % len(sigmas)]
        mode = trial % 6
        X0 = best.copy()
        if mode == 0:
            X0 += sigma * rng.standard_normal((n, d))
        elif mode == 1:
            K = 6 + (trial % 8)
            idx = order[:K]
            X0[idx] += (sigma * 3.0) * rng.standard_normal((K, d))
            msk = np.ones(n, dtype=bool); msk[idx] = False
            X0[msk] += (sigma * 0.3) * rng.standard_normal((n - K, d))
        elif mode == 2:
            X0[int(pa)] += sigma * 2.5 * rng.standard_normal(d)
            X0[int(pb)] += sigma * 2.5 * rng.standard_normal(d)
            X0 += (sigma * 0.3) * rng.standard_normal((n, d))
        elif mode == 3:
            w = int(order[0])
            try:
                hole = _deep_hole(best, w, rng, nc=2048)
                X0[w] = hole
            except Exception:
                X0[w] += sigma * 3.0 * rng.standard_normal(d)
            X0 += (sigma * 0.5) * rng.standard_normal((n, d))
        elif mode == 4:
            # deterministic pair separation + noise
            try:
                a = best[int(pa)]; b = best[int(pb)]
                # tangent components pushing apart
                dab = float(a @ b)
                at = b - dab * a
                bt = a - dab * b
                na = np.linalg.norm(at); nb = np.linalg.norm(bt)
                step = 0.012 + 0.008 * rng.random()
                if na > 1e-12:
                    X0[int(pa)] = a - step * at / na
                if nb > 1e-12:
                    X0[int(pb)] = b - step * bt / nb
            except Exception:
                pass
            X0 += sigma * rng.standard_normal((n, d))
        else:
            X0 += (sigma * 1.5) * rng.standard_normal((n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        try:
            if trial % 2 == 0:
                Xc = _smooth_opt(X0, 3500.0, 80)
            else:
                Xc = _smooth_opt(X0, 12000.0, 70)
        except Exception:
            continue
        # micro-refine bottleneck pair and worst single
        try:
            Gc = Xc @ Xc.T
            np.fill_diagonal(Gc, -2.0)
            vals = Gc[np.triu_indices(n, 1)]
            # top pair
            ij2 = int(np.argmax(Gc)); aa, bb = divmod(ij2, n)
            Xc2 = _pair_refine(Xc, int(aa), int(bb), beta=9000.0, maxiter=50)
            if _max_cos(Xc2) < _max_cos(Xc):
                Xc = Xc2
            # if still single-pair dominated, refine worst single too
            Gc2 = Xc @ Xc.T
            np.fill_diagonal(Gc2, -2.0)
            rm2 = Gc2.max(axis=1)
            w2 = int(np.argmax(rm2))
            # check gap between top1 and top2
            srt = np.sort(vals)[::-1]
            if len(srt) > 1 and (srt[0] - srt[1]) > 3e-7:
                Xc3 = _single_refine(Xc, w2, beta=9000.0, maxiter=40)
                if _max_cos(Xc3) < _max_cos(Xc):
                    Xc = Xc3
        except Exception:
            pass
        try:
            vc = _max_cos(Xc)
        except Exception:
            continue
        if vc < best_val - 1e-12:
            best, best_val = Xc, vc
            cur = Xc.copy()
        if time.time() >= HOP_END:
            break
    try:
        if time.time() < HARD - 8:
            best = _lp_polish(best, min(HARD - 1.0, time.time() + 26.0), k_top=4500, radius_init=2e-3)
            best_val = _max_cos(best)
    except Exception:
        pass
    try:
        if time.time() < HARD - 10:
            Xc = _smooth_opt(best, 18000.0, 60)
            if _max_cos(Xc) < best_val:
                best, best_val = Xc, _max_cos(Xc)
            # final pair squeeze
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            ij = int(np.argmax(G)); aa, bb = divmod(ij, n)
            Xp = _pair_refine(best, int(aa), int(bb), beta=12000.0, maxiter=60)
            vp = _max_cos(Xp)
            if vp < best_val:
                best, best_val = Xp, vp
    except Exception:
        pass
    try:
        if time.time() < HARD - 4:
            best = _lp_polish(best, HARD - 1.0, k_top=5000, radius_init=3e-4)
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    best = np.nan_to_num(best)
    return best
