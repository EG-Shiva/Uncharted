import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _adam_lse(X0, beta0, beta1, lr0, lr1, steps, track_best=None):
    X = X0.copy()
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lb0, lb1 = np.log(beta0), np.log(beta1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    best = None
    bestv = None
    if track_best is not None:
        best, bestv = track_best[0].copy(), float(track_best[1])
    for t in range(1, steps + 1):
        frac = t / steps
        beta = float(np.exp(lb0 + (lb1 - lb0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = float(np.max(G))
        # stable softmax weights
        W = np.exp(beta * (G - mx))
        np.fill_diagonal(W, 0.0)
        S = float(np.sum(W))
        if not np.isfinite(S) or S <= 0:
            break
        Grad = (W @ X) / S
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        m1 = b1 * m1 + (1.0 - b1) * Gt
        m2 = b2 * m2 + (1.0 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** t
        c2 = 1.0 - b2 ** t
        step = (m1 / c1) / (np.sqrt(m2 / c2) + eps)
        # project step to tangent to reduce normal drift
        dn = np.sum(step * X, axis=1, keepdims=True)
        step = step - dn * X
        X = X - lr * step
        X = _norm(X)
        if track_best is not None and (t % 200 == 0):
            v = _max_cos(X)
            if v < bestv:
                bestv = v
                best = X.copy()
    if track_best is not None:
        v = _max_cos(X)
        if v < bestv:
            bestv = v
            best = X.copy()
        return X, best, bestv
    return X

def _jam_contacts(X0, iters, tol0, tol1, lr0, track_best=None):
    X = X0.copy()
    best = None
    bestv = None
    if track_best is not None:
        best, bestv = track_best[0].copy(), float(track_best[1])
    lr = lr0
    for it in range(iters):
        frac = it / max(iters - 1, 1)
        tol = tol0 * (tol1 / tol0) ** frac if tol0 > 0 and tol1 > 0 else tol0
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        Mask = G > (m - tol)
        # degree
        deg = np.sum(Mask, axis=1)
        # weighted sum: use uniform over contacts
        AX = Mask @ X
        wsum = np.sum(Mask * G, axis=1)[:, None]
        F = AX - wsum * X
        sc = np.maximum(deg, 1.0)[:, None]
        F = F / sc
        Y = _norm(X - lr * F)
        v = _max_cos(Y)
        if track_best is not None:
            if v < bestv:
                bestv = v
                best = Y.copy()
                X = Y
                lr = min(lr * 1.07, 5e-3)
            else:
                lr *= 0.65
                if lr < 1e-7:
                    lr = 4e-4
                # stay at X (don't move) to anneal
        else:
            X = Y
    if track_best is not None:
        return X, best, bestv
    return X

def _refine_single(y0, Others, beta=500.0, iters=250, lr0=0.05):
    y = y0 / max(float(np.linalg.norm(y0)), 1e-16)
    for it in range(iters):
        dots = Others @ y
        m0 = float(np.max(dots))
        W = np.exp(beta * (dots - m0))
        S = float(np.sum(W))
        if not np.isfinite(S) or S == 0:
            break
        grad = (W @ Others) / S
        gt = grad - float(grad @ y) * y
        lr = lr0 / (1.0 + it / 200.0)
        y = y - lr * gt
        nn = float(np.linalg.norm(y))
        if nn == 0 or not np.isfinite(nn):
            break
        y = y / nn
    s = float(np.max(Others @ y))
    return y, s

def _try_reinsert(X, bestv, rng, K=8000, nworst=12):
    n, d = X.shape
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    rowmax = np.max(G, axis=1)
    order = np.argsort(-rowmax)[:nworst]
    # pre-generate random pool
    R = rng.standard_normal(size=(K, d))
    R = _norm(R)
    # scores vs all: R @ X.T  (K x n)
    Sfull = R @ X.T
    improved_X = None
    improved_v = bestv
    for idx in order:
        idx = int(idx)
        # max without idx
        # Sfull column idx is dot with X[idx]; mask it
        # compute max over j != idx
        S = Sfull.copy()
        S[:, idx] = -2.0
        smax = np.max(S, axis=1)
        o = np.argsort(smax)
        # m_without for reference
        G2max = float(np.max(np.delete(np.delete(G, idx, axis=0), idx, axis=1))) if n > 2 else -2.0
        # try top 3 candidates with refinement
        Others = np.delete(X, idx, axis=0)
        for k in range(min(3, K)):
            y0 = R[o[k]]
            # quick check: if even random already worse than best, refinement may still help
            y, s = _refine_single(y0, Others, beta=400.0, iters=250, lr0=0.05)
            overall = s if s > G2max else G2max
            # overall max of new code = max(s, G2max)
            if overall < improved_v - 1e-12:
                Xnew = X.copy()
                Xnew[idx] = y
                vnew = _max_cos(Xnew)
                if vnew < improved_v:
                    improved_v = vnew
                    improved_X = Xnew.copy()
        if improved_X is not None and improved_v < bestv - 1e-9:
            break
    return improved_X, improved_v

def _lp_polish(X0, time_budget, t_deadline):
    X = _norm(X0.copy())
    n, d = X.shape
    best = X.copy()
    bestv = _max_cos(X)
    radius = 2e-3
    iu = np.triu_indices(n, 1)
    t_end = min(t_deadline, time.time() + time_budget)
    while time.time() < t_end and radius > 1e-13:
        Gfull = (X @ X.T)[iu]
        m = float(np.max(Gfull))
        thr = m - max(18.0 * radius, 2e-9)
        sel = Gfull > thr
        p = int(np.sum(sel))
        if p == 0:
            # take top 800
            kk = min(800, Gfull.size)
            idx = np.argpartition(-Gfull, kk - 1)[:kk]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx] = True
            p = kk
        if p > 2800:
            order = np.argpartition(-Gfull, 2800 - 1)[:2800]
            sel = np.zeros_like(sel, dtype=bool)
            sel[order] = True
            p = 2800
        I = iu[0][sel]
        J = iu[1][sel]
        c = Gfull[sel]
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], axis=1
        ).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix(
                (X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, n * d + 1),
            ).tocsr()
            obj = np.zeros(n * d + 1)
            obj[-1] = 1.0
            res = linprog(
                obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                method="highs",
            )
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        Y = X + res.x[:-1].reshape(n, d)
        Y = _norm(Y)
        v = _max_cos(Y)
        if v < bestv:
            X, best, bestv = Y, Y.copy(), v
            radius = min(radius * 1.4, 0.015)
        else:
            # check if it improves current X even if not best (X==best here)
            if v < m:
                X = Y
                radius *= 0.7
            else:
                radius *= 0.5
        if time.time() >= t_end:
            break
    return best, bestv

def solve():
    t_start = time.time()
    T_LIMIT = 172.0
    t_deadline = t_start + T_LIMIT
    rng = np.random.default_rng(0)
    np.random.seed(0)
    n, d = 182, 9
    cands = []
    try:
        xb = np.load("best.npy")
        if np.shape(xb) == (n, d):
            cands.append(_norm(np.array(xb, dtype=float)))
    except Exception:
        pass
    try:
        xr = np.load("record.npy")
        if np.shape(xr) == (n, d):
            cands.append(_norm(np.array(xr, dtype=float)))
    except Exception:
        pass
    if len(cands) == 0:
        cands.append(_norm(rng.standard_normal(size=(n, d))))
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best = cands[bi].copy()
    bestv = float(vals[bi])
    X = best.copy()

    def remain():
        return t_deadline - time.time()

    # quick squeeze of starting point
    if remain() > 50:
        X1, b1, v1 = _adam_lse(X, 120.0, 900.0, 0.006, 0.0015, 900, track_best=(best, bestv))
        if v1 < bestv:
            best, bestv = b1.copy(), v1
        X = best.copy()
        X, best, bestv = _jam_contacts(X, 250, 3e-3, 2e-5, 1.5e-3, track_best=(best, bestv))
        X = best.copy()

    # basin hopping
    sigmas = [0.012, 0.006, 0.02, 0.003, 0.009, 0.015, 0.004]
    si = 0
    while remain() > 48:
        si += 1
        sigma = sigmas[(si - 1) % len(sigmas)]
        # perturb: stronger on worst points
        G = X @ X.T if False else best @ best.T
        np.fill_diagonal(G, -2.0)
        rowmax = np.max(G, axis=1)
        order = np.argsort(-rowmax)
        noise = rng.standard_normal(size=(n, d))
        w = np.ones((n, 1))
        # boost worst 20
        w[order[:20]] = 2.0
        Xt = best + sigma * w * noise
        Xt = _norm(Xt)
        # anneal
        beta1 = 600.0 + (si % 3) * 400.0
        Xt_a, b_a, v_a = _adam_lse(Xt, 90.0, beta1, 0.007, 0.0012, 1000, track_best=(best, bestv))
        if v_a < bestv:
            best, bestv = b_a.copy(), v_a
            X = best.copy()
            # jam it
            if remain() > 45:
                Xj, bj, vj = _jam_contacts(best, 200, 2e-3, 1e-5, 1.2e-3, track_best=(best, bestv))
                if vj < bestv:
                    best, bestv = bj.copy(), vj
                X = best.copy()
            continue
        # else try jamming the annealed point anyway then reinsert
        if remain() < 48:
            break
        _, b_j, v_j = _jam_contacts(Xt_a, 150, 2e-3, 2e-5, 1.0e-3, track_best=(Xt_a, _max_cos(Xt_a)))
        # reinsertion attempt from b_j or best
        src = b_j if v_j < _max_cos(Xt_a) + 0.002 else Xt_a
        # only attempt reinsert if promising or periodically
        if (si % 2 == 0) or (v_j < bestv + 0.001):
            K = 6000 if remain() > 80 else 3500
            try:
                impX, impv = _try_reinsert(src, bestv, rng, K=K, nworst=14)
            except Exception:
                impX, impv = None, bestv
            if impX is not None and impv < bestv:
                best, bestv = impX.copy(), impv
                X = best.copy()
                # polish the new topology quickly
                Xp, bp, vp = _adam_lse(best, 200.0, 1200.0, 0.004, 0.001, 600, track_best=(best, bestv))
                if vp < bestv:
                    best, bestv = bp.copy(), vp
                X = best.copy()
                continue
        # occasional LP on best to re-jam after failed hop (cheap, rare)
        if si % 5 == 0 and remain() > 60:
            try:
                bp, vp = _lp_polish(best, 4.0, t_deadline)
                if vp < bestv:
                    best, bestv = bp.copy(), vp
                    X = best.copy()
            except Exception:
                pass
        X = best.copy()
        if remain() <= 48:
            break

    # final squeeze: high-beta adam + LP
    if remain() > 6:
        try:
            steps = 900 if remain() > 30 else 400
            Xf, bf, vf = _adam_lse(best, 500.0, 2500.0, 0.002, 0.0006, steps, track_best=(best, bestv))
            if vf < bestv:
                best, bestv = bf.copy(), vf
        except Exception:
            pass
    if remain() > 5:
        try:
            budget = max(remain() - 1.0, 2.0)
            bp, vp = _lp_polish(best, budget, t_deadline - 0.5)
            if vp < bestv:
                best, bestv = bp.copy(), vp
        except Exception:
            pass

    best = _norm(best)
    best = np.where(np.isfinite(best), best, 0.0)
    nr = np.linalg.norm(best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    best = best / nr
    return best
