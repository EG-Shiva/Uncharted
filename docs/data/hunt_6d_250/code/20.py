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
        np.clip(D, -30.0, 0.0, out=D)
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

def _hill_climb(X, rng, steps, sigma0, t_deadline):
    # greedy single-point moves on true max; very cheap O(n*d) per step
    n, d = X.shape
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    cur = float(np.max(G))
    rowmax = G.max(axis=1)
    for s in range(steps):
        if time.time() > t_deadline:
            break
        # pick victim biased to worst points
        if s % 3 == 0:
            # worst pair endpoint
            k = int(np.argmax(G.reshape(-1)))
            a = k // n; b = k % n
            v = a if rng.random() < 0.5 else b
        else:
            # sample among top-K by rowmax
            # recompute rowmax cheaply every 50 steps
            if s % 50 == 0:
                rowmax = G.max(axis=1)
            # top 20
            # use partition via argmax of rowmax + noise
            if rng.random() < 0.8:
                # pick from worst 25
                idx = np.argpartition(-rowmax, 25)[:25]
                v = int(idx[rng.integers(0, len(idx))])
            else:
                v = int(rng.integers(0, n))
        sig = sigma0 / (1.0 + s * 0.002)
        prop = X[v] + rng.standard_normal(d) * sig
        nn = np.linalg.norm(prop)
        if nn < 1e-12:
            continue
        prop /= nn
        dots = X @ prop
        dots[v] = -2.0
        nm = float(np.max(dots))
        # max among pairs not involving v
        # compute quickly: max of G with row/col v masked
        # G is current; rest max = max over G excluding v
        # maintain by full max scan with mask (O(n^2) occasionally) else approx:
        # we can compute rest as cur if worst pair doesn't involve v else recompute
        # simple: compute rest via precomputed full matrix scan every accept + when needed
        # here do scan of G (62k entries) - cheap enough
        # temporarily mask
        # To avoid O(n^2) every step, only compute rest if nm < cur
        if nm >= cur:
            continue
        # need rest max
        # max of G excluding v:
        # G row v / col v are old dots of X[v]; ignore them
        rest = cur
        # check if current max involves v
        # rowmax[v] >= cur - 1e-15 ?
        if rowmax[v] >= cur - 1e-15:
            # recompute rest
            tmp = G.copy()
            tmp[v, :] = -2.0
            tmp[:, v] = -2.0
            rest = float(np.max(tmp))
        newmax = nm if nm > rest else rest
        if newmax < cur:
            X[v] = prop
            # update G row/col v
            G[v, :] = X @ prop
            G[:, v] = G[v, :]
            G[v, v] = -2.0
            cur = newmax
            rowmax = G.max(axis=1)
    return X, cur

def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1.5e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cmat = x @ x.T
        iu_c = cmat[iu]
        m = float(np.max(iu_c))
        sel = iu_c > m - max(15.0 * radius, 1e-6)
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
            x, best, radius = y, v, min(radius * 1.5, 0.01)
        else:
            radius *= 0.5
    return x

def solve():
    t_start = time.time()
    BUDGET = 168.0
    HOP_END = t_start + 140.0
    HILL_END = t_start + 158.0
    LP_END = t_start + 165.5
    rng = np.random.default_rng(12345)
    n, d = 250, 6
    cands = []
    try:
        a = np.asarray(np.load("best.npy"), dtype=np.float64)
        if a.shape == (n, d):
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            cands.append(a)
    except Exception:
        pass
    try:
        b = np.asarray(np.load("record.npy"), dtype=np.float64)
        if b.shape == (n, d):
            b = b / np.linalg.norm(b, axis=1, keepdims=True)
            cands.append(b)
    except Exception:
        pass
    if not cands:
        t = rng.standard_normal((n, d)); t /= np.linalg.norm(t, axis=1, keepdims=True)
        cands.append(t)
    best = min(cands, key=_max_cos).copy()
    best_val = _max_cos(best)

    # stage 1: quick anneal
    cur = best.copy()
    for beta, it in [(1500.0, 100), (8000.0, 100), (30000.0, 80)]:
        if time.time() > t_start + 30:
            break
        try:
            cur = _smooth_opt(cur, beta, it)
        except Exception:
            break
        v = _max_cos(cur)
        if v < best_val:
            best_val = v; best = cur.copy()
    # immediate hill climb to kill single-pair spike
    if time.time() < t_start + 40:
        try:
            Xh, vh = _hill_climb(best.copy(), rng, 6000, 0.004, t_start + 40)
            if vh < best_val:
                best = Xh; best_val = vh
        except Exception:
            pass

    trial = 0
    betas = [4000.0, 12000.0, 30000.0, 60000.0]
    sigmas = [0.0015, 0.004, 0.012, 0.03, 0.06, 0.002]
    while time.time() < HOP_END and trial < 200:
        trial += 1
        try:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            order = np.argsort(-rowmax)
        except Exception:
            order = np.arange(n)
        Y = best.copy()
        mode = trial % 7
        if mode == 6:
            # large topology change: reinit 12-20 random worst points
            K = 12 + int(rng.integers(0, 8))
            victims = list(order[:6]) + list(rng.integers(0, n, size=K))
            victims = list(set(victims))[:K+6]
            # replace half by deep-hole candidates
            for v in victims:
                if rng.random() < 0.5:
                    cand = rng.standard_normal((512, d))
                    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                    # score vs current best (excluding v)
                    S = cand @ best.T
                    S[:, v] = -2.0
                    mx = S.max(axis=1)
                    Y[v] = cand[int(np.argmin(mx))]
                else:
                    Y[v] = rng.standard_normal(d)
                    Y[v] /= np.linalg.norm(Y[v])
            sigma_extra = 0.0
        else:
            K = 3 + (trial % 7)
            victims = list(order[:K])
            if trial % 2 == 0:
                victims += list(rng.integers(0, n, size=2))
            victims = list(set(victims))
            sigma = sigmas[trial % len(sigmas)]
            Y[victims] += rng.standard_normal((len(victims), d)) * sigma
            if trial % 4 == 0:
                Y += rng.standard_normal(Y.shape) * 0.0008
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        beta = betas[trial % len(betas)]
        iters = 90 if mode != 6 else 120
        try:
            Y2 = _smooth_opt(Y, beta, iters)
        except Exception:
            continue
        # escalate beta once for plateau
        if time.time() < HOP_END - 5:
            try:
                Y2 = _smooth_opt(Y2, beta * 2.5, 40, maxcor=8)
            except Exception:
                pass
        v2 = _max_cos(Y2)
        # hill climb squeeze (cheap, kills single spike)
        if v2 < best_val + 2e-4:
            try:
                Y2, v2h = _hill_climb(Y2, rng, 1500, 0.002, time.time() + 1.2)
                v2 = v2h
            except Exception:
                v2 = _max_cos(Y2)
        if v2 < best_val:
            best = Y2.copy(); best_val = v2

    # stage: intensive hill climbing on best
    if time.time() < HILL_END:
        try:
            Xh, vh = _hill_climb(best.copy(), rng, 30000, 0.003, HILL_END)
            if vh < best_val:
                best = Xh; best_val = vh
        except Exception:
            pass

    # stage: final high-beta polish + LP
    if time.time() < LP_END - 8:
        try:
            cur = _smooth_opt(best, 80000.0, 60, maxcor=8)
            v = _max_cos(cur)
            if v < best_val:
                best = cur; best_val = v
        except Exception:
            pass
    if time.time() < LP_END:
        try:
            p = _lp_polish(best, LP_END)
            vp = _max_cos(p)
            if vp < best_val:
                best = p; best_val = vp
        except Exception:
            pass

    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    best = np.where(np.isfinite(best), best, 0.0)
    nrm = np.linalg.norm(best, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    best = best / nrm
    return best
