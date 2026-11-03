import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G

def _descent(x, beta, steps, lr, mu):
    n, d = x.shape
    v = np.zeros_like(x)
    for it in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = np.max(G)
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        mean = (W @ x) / rs
        dot = np.sum(mean * x, axis=1, keepdims=True)
        tang = mean - dot * x
        v = mu * v + tang
        cur_lr = lr * (1.0 - 0.5 * it / steps)
        x = x - cur_lr * v
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        x = x / nrm
    return x

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
                   options={"maxiter": maxiter, "ftol": 1e-20,
                            "gtol": 1e-12, "maxfun": maxiter*5, "maxcor": 20})
    Y = res.x.reshape(n, d)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def _polish_single(y, X, idx, beta=150.0, steps=180, lr=0.12):
    for _ in range(steps):
        c = X @ y
        c[idx] = -2.0
        m = float(np.max(c))
        w = np.exp(beta * (c - m))
        w[idx] = 0.0
        rs = float(np.sum(w)) + 1e-12
        mean = (w @ X) / rs
        dot = float(mean @ y)
        tang = mean - dot * y
        y = y - lr * tang
        n = float(np.linalg.norm(y))
        if n == 0 or not np.isfinite(n):
            break
        y = y / n
        lr *= 0.996
    return y

def _void_for_idx(X, idx, rng, C=8000):
    n, d = X.shape
    cand = rng.standard_normal((C, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    S = cand @ X.T
    S[:, idx] = -2.0
    srow = S.max(axis=1)
    j = int(np.argmin(srow))
    return cand[j]

def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 2.5e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cmat = x @ x.T
        iuc = cmat[iu]
        m = float(iuc.max())
        sel = iuc > m - max(12.0 * radius, 4e-7)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 6000:
            thr = np.sort(iuc[sel])[-6000]
            sel = iuc >= thr
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None]*d + np.arange(d),
                               J[:, None]*d + np.arange(d),
                               np.full((p,1), n*d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n*d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-iuc[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d)+[(None, None)],
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
            x, best, radius = y, v, min(radius*1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    HARD = t0 + 175.0
    rng = np.random.default_rng(0)
    n, d = 250, 6
    cands = []
    for f in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(f), dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        tmp = rng.standard_normal((n, d))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        cands.append(tmp)
    best = min(cands, key=_max_cos).copy()
    best_val = _max_cos(best)
    # Stage 1: annealed L-BFGS squeeze
    for beta, it in [(500.0, 80), (2000.0, 80), (8000.0, 90), (20000.0, 70)]:
        if time.time() > t0 + 38.0 or time.time() > HARD - 135.0:
            break
        try:
            cur = _smooth_opt(best, beta, it)
        except Exception:
            break
        v = _max_cos(cur)
        if v < best_val:
            best, best_val = cur, v
    # Stage 2: basin hopping with fast descent + selective refine
    trial = 0
    betas = [400.0, 1200.0, 3000.0]
    while time.time() < min(t0 + 118.0, HARD - 55.0) and trial < 200:
        trial += 1
        if trial % 7 == 0:
            sigma = 0.11
        elif trial % 4 == 0:
            sigma = 0.045
        elif trial % 2 == 0:
            sigma = 0.022
        else:
            sigma = 0.012
        beta = betas[trial % 3]
        try:
            rm, G = _row_max(best)
        except Exception:
            rm = None
        x_try = best + sigma * rng.standard_normal((n, d))
        if rm is not None:
            worst = np.argsort(-rm)[:10]
            x_try[worst] += sigma * 1.2 * rng.standard_normal((len(worst), d))
        x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        try:
            if beta < 1000:
                x_try = _descent(x_try, beta, 300, 0.008, 0.85)
            else:
                x_try = _descent(x_try, beta, 380, 0.007, 0.86)
        except Exception:
            continue
        s = _max_cos(x_try)
        # refine promising basins with L-BFGS
        if s < best_val + 0.004 and time.time() < min(t0+118.0, HARD-55.0)-2.0:
            try:
                rb = 4000.0 if s < best_val + 0.001 else 1500.0
                x_try = _smooth_opt(x_try, rb, 60)
                s = _max_cos(x_try)
            except Exception:
                pass
        if s < best_val - 1e-12:
            best, best_val = x_try, s
        # intermittent void reinsertion directly on best
        if trial % 5 == 0 and time.time() < min(t0+118.0, HARD-55.0)-2.0:
            try:
                rm2, G2 = _row_max(best)
                order = np.argsort(-rm2)
                for k in range(3):
                    idx = int(order[k])
                    other_max = float(np.max(np.delete(np.delete(G2, idx, 0), idx, 1)))
                    y0 = _void_for_idx(best, idx, rng, C=6000)
                    y0 = _polish_single(y0, best, idx, beta=160.0, steps=160, lr=0.11)
                    cn = best @ y0; cn[idx] = -2.0
                    npm = float(np.max(cn))
                    ng = npm if npm > other_max else other_max
                    if ng < best_val - 1e-12:
                        best[idx] = y0 / np.linalg.norm(y0)
                        best_val = ng
                        G2 = best @ best.T
                        np.fill_diagonal(G2, -2.0)
                        break
            except Exception:
                pass
        if time.time() >= min(t0+118.0, HARD-55.0):
            break
    # Stage 3: coordinate single-point sweeps over worst points
    while time.time() < min(t0 + 148.0, HARD - 28.0):
        try:
            rm, G = _row_max(best)
        except Exception:
            break
        order = np.argsort(-rm)
        improved = False
        for k in range(min(40, n)):
            idx = int(order[k])
            if time.time() >= min(t0+148.0, HARD-28.0):
                break
            mask = np.ones(n, dtype=bool); mask[idx] = False
            other_max = float(np.max(G[mask][:, mask]))
            if rm[idx] < other_max - 1e-12:
                continue
            y0 = _void_for_idx(best, idx, rng, C=5000)
            y0 = _polish_single(y0, best, idx, beta=200.0, steps=180, lr=0.10)
            # also try local kick of current point
            y1 = best[idx] + 0.05 * rng.standard_normal(d)
            y1 /= np.linalg.norm(y1)
            y1 = _polish_single(y1, best, idx, beta=200.0, steps=150, lr=0.10)
            best_y = y0
            cn0 = best @ y0; cn0[idx] = -2.0; m0 = float(np.max(cn0))
            cn1 = best @ y1; cn1[idx] = -2.0; m1 = float(np.max(cn1))
            if m1 < m0:
                best_y, m0 = y1, m1
            ng = m0 if m0 > other_max else other_max
            if ng < best_val - 1e-12:
                best[idx] = best_y / np.linalg.norm(best_y)
                best_val = ng
                _, G = _row_max(best)
                # refresh G for next k
                G = best @ best.T
                np.fill_diagonal(G, -2.0)
                improved = True
                break
        if not improved:
            # small global nudge + descent to change basin
            if time.time() < min(t0+148.0, HARD-28.0)-3.0:
                xt = best + 0.010 * rng.standard_normal((n, d))
                xt /= np.linalg.norm(xt, axis=1, keepdims=True)
                try:
                    xt = _descent(xt, 1200.0, 220, 0.006, 0.86)
                    s = _max_cos(xt)
                    if s < best_val - 1e-12:
                        best, best_val = xt, s
                        continue
                except Exception:
                    pass
            break
    # Stage 4: final squeeze - L-BFGS high beta then LP
    if time.time() < HARD - 6.0:
        try:
            cur = _smooth_opt(best, 12000.0, 70)
            v = _max_cos(cur)
            if v < best_val:
                best, best_val = cur, v
        except Exception:
            pass
    if time.time() < HARD - 4.0:
        try:
            best = _lp_polish(best, HARD - 1.5)
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    best = np.nan_to_num(best)
    return best
