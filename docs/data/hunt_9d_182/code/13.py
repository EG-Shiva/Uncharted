import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _gram_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    return G, m

def _anneal_adam(X0, steps, beta0, beta1, lr0, lr1, seed_m=True):
    X = _norm(X0.copy())
    n, d = X.shape
    m_ad = np.zeros_like(X)
    v_ad = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best = X.copy()
    bestv = _max_cos(X)
    for t in range(1, steps + 1):
        frac = (t - 1) / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        S = float(np.sum(W))
        if not np.isfinite(S) or S <= 0:
            continue
        Grad = (W @ X) / S
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        m_ad = b1 * m_ad + (1.0 - b1) * Gt
        v_ad = b2 * v_ad + (1.0 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** t
        c2 = 1.0 - b2 ** t
        upd = (m_ad / c1) / (np.sqrt(v_ad / c2) + eps)
        dn = np.sum(upd * X, axis=1, keepdims=True)
        upd_t = upd - dn * X
        X = _norm(X - lr * upd_t)
        if t % 100 == 0:
            v = _max_cos(X)
            if v < bestv:
                bestv = v
                best = X.copy()
    v = _max_cos(X)
    if v < bestv:
        bestv = v
        best = X.copy()
    return X, best, bestv

def _refine_single(y0, Others, beta=500.0, iters=250, lr0=0.05):
    y = y0 / (np.linalg.norm(y0) + 1e-30)
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

def _reinsert_worst(X, cur_best, rng, n_sample=2500):
    n, d = X.shape
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    row_max = np.max(G, axis=1)
    order = np.argsort(-row_max)
    # try 3 worst points
    for w in order[:3]:
        Others = np.delete(X, w, axis=0)
        # internal max of Others
        Go = Others @ Others.T
        np.fill_diagonal(Go, -2.0)
        others_max = float(np.max(Go))
        Y = rng.standard_normal(size=(n_sample, d))
        Y = _norm(Y)
        scores = np.max(Y @ Others.T, axis=1)
        kbest = np.argsort(scores)[:5]
        tried = False
        for k in kbest:
            y0 = Y[k]
            # vary beta a bit
            y, s = _refine_single(y0, Others, beta=500.0, iters=250, lr0=0.05)
            cand_max = s if s > others_max else others_max
            if cand_max < cur_best - 1e-12:
                X2 = Others.copy()
                # insert at position w to keep shape; simplest: replace row w
                Xnew = X.copy()
                Xnew[w] = y
                # verify
                v = _max_cos(Xnew)
                if v < cur_best - 1e-12:
                    return Xnew, v, True
        # if no improvement, continue to next worst
    return X, cur_best, False

def _greedy_pairs(X, iters, step0=2e-4):
    X = _norm(X.copy())
    best = _max_cos(X)
    step = step0
    n = X.shape[0]
    for _ in range(iters):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        k = int(np.argmax(G))
        i = k // n
        j = k % n
        dot = float(G[i, j])
        vi = dot * X[i] - X[j]
        vj = dot * X[j] - X[i]
        ni = float(np.linalg.norm(vi))
        nj = float(np.linalg.norm(vj))
        if ni < 1e-15 or nj < 1e-15:
            step *= 0.5
            if step < 1e-12:
                step = step0 * 0.5
            continue
        vi /= ni
        vj /= nj
        Y = X.copy()
        Y[i] = X[i] + step * vi
        Y[j] = X[j] + step * vj
        Y[i] /= np.linalg.norm(Y[i])
        Y[j] /= np.linalg.norm(Y[j])
        v = _max_cos(Y)
        if v < best:
            X = Y
            best = v
            step = min(step * 1.2, 3e-3)
        else:
            # try single sided
            Y2 = X.copy()
            Y2[i] = X[i] + step * vi
            Y2[i] /= np.linalg.norm(Y2[i])
            v2 = _max_cos(Y2)
            Y3 = X.copy()
            Y3[j] = X[j] + step * vj
            Y3[j] /= np.linalg.norm(Y3[j])
            v3 = _max_cos(Y3)
            if v2 < best and v2 <= v3:
                X = Y2
                best = v2
                step = min(step * 1.1, 3e-3)
            elif v3 < best:
                X = Y3
                best = v3
                step = min(step * 1.1, 3e-3)
            else:
                step *= 0.6
                if step < 5e-7:
                    step = step0 * 0.5
    return X, best

def _lp_jam(X0, t_end, K=2200, r0=2e-3):
    X = _norm(X0.copy())
    n, d = X.shape
    best = X.copy()
    bestv = _max_cos(X)
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < t_end and radius > 1e-13:
        if time.time() >= t_end:
            break
        Gfull = (X @ X.T)[iu]
        # top K pairs always constrained (wide window)
        p_all = Gfull.size
        Kcur = K if K < p_all else p_all
        idx = np.argpartition(-Gfull, Kcur - 1)[:Kcur]
        I = iu[0][idx]
        J = iu[1][idx]
        c = Gfull[idx]
        p = Kcur
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols_inter = np.concatenate(
            [I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], axis=1
        ).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols_inter)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix(
                (X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, n * d + 1),
            ).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(n * d + 1)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            Y = X + res.x[:-1].reshape(n, d)
            Y = _norm(Y)
            v = _max_cos(Y)
        except Exception:
            radius *= 0.5
            continue
        if v < bestv - 1e-12:
            X = Y
            best = Y.copy()
            bestv = v
            radius = min(radius * 1.4, 5e-3)
        else:
            # accept tiny sideways moves that increase contacts even if not improving?
            # only shrink
            radius *= 0.5
        # early stop if time
        if time.time() >= t_end:
            break
    return best, bestv

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.default_rng(0)
    np.random.seed(0)
    x_best = None
    best_val = 1e9
    for fname in ("best.npy", "record.npy"):
        try:
            y = np.load(fname)
            y = np.asarray(y, dtype=float)
            if y.shape != (182, 9):
                continue
            y = _norm(y)
            if np.any(~np.isfinite(y)):
                continue
            v = _max_cos(y)
            if v < best_val:
                best_val = v
                x_best = y.copy()
        except Exception:
            continue
    if x_best is None:
        x_best = _norm(rng.standard_normal(size=(182, 9)))
        best_val = _max_cos(x_best)
    # light initial jam to assess
    try:
        if time.time() < deadline - 60:
            x_best, best_val = _greedy_pairs(x_best, iters=150, step0=2e-4)
    except Exception:
        pass
    sigmas = [0.0, 0.008, 0.02, 0.035, 0.012, 0.05, 0.005, 0.025]
    trial = 0
    # exploratory phase until deadline-28
    while time.time() < deadline - 28.0 and trial < 25:
        sigma = sigmas[trial % len(sigmas)]
        trial += 1
        if time.time() >= deadline - 28.0:
            break
        try:
            if sigma == 0.0:
                x0 = x_best.copy()
            else:
                x0 = x_best + sigma * rng.standard_normal(size=x_best.shape)
                x0 = _norm(x0)
            # vary schedule
            if trial % 3 == 1:
                b0, b1, l0, l1, st = 60, 700, 0.006, 0.0006, 1400
            elif trial % 3 == 2:
                b0, b1, l0, l1, st = 40, 500, 0.007, 0.0008, 1200
            else:
                b0, b1, l0, l1, st = 100, 900, 0.004, 0.0004, 1500
            Xraw, Xb, vb = _anneal_adam(x0, steps=st, beta0=b0, beta1=b1, lr0=l0, lr1=l1)
            # greedy squeeze
            Xg, vg = _greedy_pairs(Xb, iters=120, step0=1.5e-4)
            # reinsertion attempts (topology change)
            curX, curV = Xg, vg
            # also try raw best
            if vb < curV:
                curX, curV = Xb, vb
            # two rounds of reinsertion
            for _ in range(2):
                if time.time() >= deadline - 28.0:
                    break
                curX, curV, imp = _reinsert_worst(curX, curV, rng, n_sample=2000)
                if imp:
                    # refine reinserted with short anneal
                    curX2, curXb, curVb = _anneal_adam(curX, steps=500, beta0=150, beta1=800, lr0=0.003, lr1=0.0005)
                    curXg2, curVg2 = _greedy_pairs(curXb, iters=80, step0=1e-4)
                    if curVg2 < curV:
                        curX, curV = curXg2, curVg2
                    elif curVb < curV:
                        curX, curV = curXb, curVb
                    else:
                        break
                else:
                    break
            # short LP burst to jam this trial (4 sec max)
            if curV < best_val + 5e-4 and time.time() < deadline - 30.0:
                tburst = min(4.0, (deadline - 28.0 - time.time()) * 0.3)
                if tburst > 1.0:
                    curXj, curVj = _lp_jam(curX, time.time() + tburst, K=1800, r0=1.5e-3)
                    if curVj < curV:
                        curX, curV = curXj, curVj
            if curV < best_val - 1e-12:
                x_best = curX.copy()
                best_val = curV
        except Exception:
            continue
    # final long jam
    remaining = deadline - time.time() - 1.5
    if remaining > 5.0:
        lp_time = min(remaining * 0.8, 60.0)
        try:
            x_jam, v_jam = _lp_jam(x_best, time.time() + lp_time, K=2200, r0=2e-3)
            if v_jam < best_val:
                x_best = x_jam
                best_val = v_jam
        except Exception:
            pass
    if time.time() < deadline - 1.0:
        try:
            x_best, best_val = _greedy_pairs(x_best, iters=150, step0=8e-5)
        except Exception:
            pass
    x_best = _norm(x_best)
    return x_best
