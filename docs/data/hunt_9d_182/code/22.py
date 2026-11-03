import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n < 1e-14] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _adam_run(X0, beta, lr, steps, t_dead):
    X = X0.copy()
    n, d = X.shape
    b1, b2, eps = 0.9, 0.999, 1e-8
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    best = None
    bestv = np.inf
    b1p = 1.0
    b2p = 1.0
    for t in range(1, steps + 1):
        if (t & 63) == 0 and time.time() > t_dead:
            break
        b1p *= b1
        b2p *= b2
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        if m < bestv:
            bestv = m
            best = X.copy()
        # stable softmax weights
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E))
        if not np.isfinite(S) or S <= 0:
            break
        Grad = (E @ X) / S
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        m1 = b1 * m1 + (1.0 - b1) * Gt
        m2 = b2 * m2 + (1.0 - b2) * (Gt * Gt)
        mh = m1 / (1.0 - b1p)
        vh = m2 / (1.0 - b2p)
        step = mh / (np.sqrt(vh) + eps)
        # project step to tangent
        dn = np.sum(step * X, axis=1, keepdims=True)
        step = step - dn * X
        X = X - lr * step
        X = _norm(X)
    if best is None:
        best = X.copy()
        bestv = _max_cos(X)
    return best, bestv

def _greedy(X0, iters, t_dead, init_step=3e-4):
    X = X0.copy()
    n = X.shape[0]
    best = _max_cos(X)
    bestX = X.copy()
    step = init_step
    for it in range(iters):
        if (it & 127) == 0 and time.time() > t_dead:
            break
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        k = int(np.argmax(G))
        i = k // n
        j = k % n
        dot = float(G[i, j])
        vi = dot * X[i] - X[j]
        vj = dot * X[j] - X[i]
        ni = float(np.dot(vi, vi)) ** 0.5
        nj = float(np.dot(vj, vj)) ** 0.5
        if ni < 1e-14 or nj < 1e-14:
            step *= 0.5
            if step < 1e-12:
                step = init_step * 0.3
            continue
        vi /= ni
        vj /= nj
        # try symmetric
        Y = X.copy()
        Y[i] = X[i] + step * vi
        Y[j] = X[j] + step * vj
        Y[i] /= np.linalg.norm(Y[i])
        Y[j] /= np.linalg.norm(Y[j])
        v = _max_cos(Y)
        if v < best:
            X = Y
            best = v
            bestX = Y.copy()
            step = min(step * 1.2, 4e-3)
            continue
        # single sided
        Y2 = X.copy()
        Y2[i] = X[i] + step * vi
        Y2[i] /= np.linalg.norm(Y2[i])
        v2 = _max_cos(Y2)
        Y3 = X.copy()
        Y3[j] = X[j] + step * vj
        Y3[j] /= np.linalg.norm(Y3[j])
        v3 = _max_cos(Y3)
        if v2 < best and v2 <= v3:
            X = Y2; best = v2; bestX = Y2.copy()
            step = min(step * 1.15, 4e-3)
        elif v3 < best:
            X = Y3; best = v3; bestX = Y3.copy()
            step = min(step * 1.15, 4e-3)
        else:
            step *= 0.5
            if step < 5e-13:
                step = init_step * 0.5
    return bestX, best

def _lp_polish(X0, seconds, t_hard):
    X = _norm(X0.copy())
    n, d = X.shape
    bestv = _max_cos(X)
    best = X.copy()
    radius = 8e-4
    iu0 = np.triu_indices(n, 1)
    t_end = min(time.time() + seconds, t_hard)
    bnd = None
    while time.time() < t_end and radius > 1e-13:
        Gfull = X @ X.T
        c_all = Gfull[iu0]
        m = float(np.max(c_all))
        thr = m - max(20.0 * radius, 2e-9)
        sel = c_all > thr
        p = int(np.count_nonzero(sel))
        if p == 0:
            radius *= 0.5
            continue
        if p > 3500:
            idx = np.argpartition(-c_all, 3500)[:3500]
            msk = np.zeros_like(sel)
            msk[idx] = True
            sel = sel & msk
            p = 3500
        I = iu0[0][sel]; J = iu0[1][sel]
        cs = c_all[sel]
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None]*d + np.arange(d), J[:, None]*d + np.arange(d), np.full((p,1), n*d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p,1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n*d+1)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, n*d+1)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(n*d+1); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        dx = res.x[:-1].reshape(n, d)
        ok = False
        for alpha in (1.0, 0.5, 0.25):
            Y = _norm(X + alpha*dx)
            v = _max_cos(Y)
            if v < bestv - 1e-13:
                bestv = v; best = Y.copy(); X = Y; ok = True
                radius = min(radius*1.5, 0.01)
                break
        if not ok:
            # accept sideways move if it helps current even when not best? keep X==best so shrink
            radius *= 0.5
        if time.time() >= t_end:
            break
    return best

def solve():
    t_start = time.time()
    T_HARD = t_start + 172.0
    rng = np.random.default_rng(0)
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
    if not cands:
        cands.append(_norm(rng.standard_normal((n, d))))
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best = cands[bi].copy()
    bestv = float(vals[bi])
    cur = best.copy()

    # quick initial squeeze of current best
    if T_HARD - time.time() > 10:
        g0, v0 = _greedy(cur, 1500, T_HARD - 25)
        if v0 < bestv:
            bestv = v0; best = g0.copy()
        cur = best.copy()

    # basin hops targeted at worst pair/points
    cyc = 0
    # reserve ~25s for final LP
    hop_dead = T_HARD - 25.0
    sigmas = [0.015, 0.008, 0.004, 0.002, 0.001]
    betas = [250.0, 500.0, 900.0, 1600.0]
    while time.time() < hop_dead:
        cyc += 1
        if time.time() > hop_dead:
            break
        # find worst indices
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        rowmax = np.max(G, axis=1)
        order = np.argsort(-rowmax)
        k = 2 if (cyc % 3) else 4
        worst = order[:k]
        sigma = sigmas[cyc % len(sigmas)]
        # occasionally global shake
        Xp = cur.copy()
        Xp[worst] += sigma * rng.standard_normal((k, d))
        if cyc % 5 == 0:
            Xp += 0.0008 * rng.standard_normal((n, d))
        Xp = _norm(Xp)
        beta = betas[cyc % len(betas)]
        lr = 0.004 if beta < 600 else 0.002
        inner_dead = min(hop_dead, time.time() + 4.0)
        # short adam
        Xr, vr = _adam_run(Xp, beta, lr, 500, inner_dead)
        if time.time() > hop_dead - 1:
            if vr < bestv:
                bestv = vr; best = Xr.copy()
            break
        # greedy jam
        Xg, vg = _greedy(Xr, 1200, inner_dead, init_step=3e-4)
        if vg < bestv - 1e-12:
            bestv = vg; best = Xg.copy(); cur = Xg.copy()
        else:
            # diversify: move cur to Xg if close
            if vg < bestv + 5e-4 and rng.random() < 0.35:
                cur = Xg.copy()
            elif vr < bestv + 5e-4 and rng.random() < 0.2:
                cur = Xr.copy()
            else:
                # small kick from best to avoid stagnation
                if cyc % 7 == 0:
                    cur = best.copy()
                    cur[worst] += (sigma*0.5) * rng.standard_normal((k, d))
                    cur = _norm(cur)
        # early keep best
        if time.time() > hop_dead:
            break
    # final LP polish with remaining time
    remain = T_HARD - time.time() - 1.0
    if remain > 4.0:
        try:
            best = _lp_polish(best, min(remain, 22.0), T_HARD - 0.5)
        except Exception:
            pass
    best = _norm(best)
    best = np.where(np.isfinite(best), best, 0.0)
    best = _norm(best)
    return best
