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


def _adam_smooth(X0, stages, rng, t_start, t_limit, best, bestv):
    X = X0.copy()
    n, d = X.shape
    b1, b2, eps = 0.9, 0.999, 1e-8
    for beta, lr, steps in stages:
        v1 = np.zeros_like(X)
        v2 = np.zeros_like(X)
        for t in range(1, steps + 1):
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            # weights exp(beta*(G-m))
            # reuse G buffer for W
            W = np.exp(beta * (G - m))
            np.fill_diagonal(W, 0.0)
            S = float(np.sum(W))
            if not np.isfinite(S) or S == 0:
                break
            Grad = (W @ X) / S
            dot = np.sum(Grad * X, axis=1, keepdims=True)
            Gt = Grad - dot * X
            v1 = b1 * v1 + (1.0 - b1) * Gt
            v2 = b2 * v2 + (1.0 - b2) * (Gt * Gt)
            c1 = 1.0 - b1 ** t
            c2 = 1.0 - b2 ** t
            step = (v1 / c1) / (np.sqrt(v2 / c2) + eps)
            X = X - lr * step
            X = _norm(X)
            if t % 100 == 0:
                cur = _max_cos(X)
                if cur < bestv:
                    bestv = cur
                    best = X.copy()
                if time.time() - t_start > t_limit:
                    return X, best, bestv
        # end stage check
        cur = _max_cos(X)
        if cur < bestv:
            bestv = cur
            best = X.copy()
        if time.time() - t_start > t_limit:
            break
    return X, best, bestv


def _refine_single(y0, Others, beta=400.0, iters=350, lr0=0.05):
    y = y0 / np.linalg.norm(y0)
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
        n = float(np.linalg.norm(y))
        if n == 0 or not np.isfinite(n):
            break
        y = y / n
    s = float(np.max(Others @ y))
    return y, s


def _try_replace(X, idx, rng, n_sample=4000):
    n, d = X.shape
    Others = np.delete(X, idx, axis=0)
    Y = rng.standard_normal(size=(n_sample, d))
    Y = _norm(Y)
    scores = np.max(Y @ Others.T, axis=1)
    order = np.argsort(scores)
    best_y = None
    best_s = float("inf")
    for k in range(min(5, n_sample)):
        y0 = Y[order[k]]
        y, s = _refine_single(y0, Others)
        if s < best_s:
            best_s = s
            best_y = y
    return best_y, best_s, Others


def _lp_polish(X0, seconds, t_start, t_global_limit):
    X = _norm(X0.copy())
    n, d = X.shape
    best = X.copy()
    bestv = _max_cos(X)
    radius = 3e-3
    iu = np.triu_indices(n, 1)
    t_end = min(t_start + t_global_limit, time.time() + seconds)
    while time.time() < t_end and radius > 1e-13:
        Gfull = (X @ X.T)[iu]
        m = float(np.max(Gfull))
        thr = m - max(12.0 * radius, 5e-9)
        sel = Gfull > thr
        cnt = int(np.sum(sel))
        if cnt == 0:
            break
        if cnt > 1500:
            # keep top 1500
            idx = np.argpartition(-Gfull, 1500)[:1500]
            sel = np.zeros_like(sel)
            sel[idx] = True
        I = iu[0][sel]
        J = iu[1][sel]
        c = Gfull[sel]
        p = len(I)
        rows = np.repeat(np.arange(p), 2 * d + 1)
        ci = (I[:, None] * d + np.arange(d)).ravel()
        cj = (J[:, None] * d + np.arange(d)).ravel()
        ct = np.full(p, n * d)
        cols = np.concatenate([ci, cj, ct])
        # interleave per row: need order [xJ, xI, -1] per row
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        # cols above is grouped by block, not interleaved; rebuild correctly:
        cols2 = np.empty(p * (2 * d + 1), dtype=np.int64)
        cols2[0::(2 * d + 1)] = I * d + 0  # placeholder, fix below with loop-free construction
        # Build correctly with stacking per row
        base = np.arange(p)[:, None]
        # recompute cols in row-major interleaved order:
        cols_inter = np.concatenate(
            [I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], axis=1
        ).ravel()
        A_ub = coo_matrix((vals, (rows, cols_inter)), shape=(p, n * d + 1)).tocsr()
        A_eq = coo_matrix(
            (X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, n * d + 1),
        ).tocsr()
        obj = np.zeros(n * d + 1)
        obj[-1] = 1.0
        try:
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
            radius = min(radius * 1.4, 0.02)
        else:
            # accept move if it improves current even if not best? X==best here so same
            radius *= 0.5
        if time.time() >= t_end:
            break
    return best


def solve():
    t_start = time.time()
    T_LIMIT = 172.0  # global budget
    rng = np.random.default_rng(0)
    np.random.seed(0)
    n, d = 182, 9
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    if xb is not None and np.shape(xb) == (n, d):
        cands.append(_norm(np.array(xb, dtype=float)))
    if xr is not None and np.shape(xr) == (n, d):
        cands.append(_norm(np.array(xr, dtype=float)))
    if len(cands) == 0:
        cands.append(_norm(rng.standard_normal(size=(n, d))))
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    X = cands[bi].copy()
    best = X.copy()
    bestv = float(vals[bi])

    # Stage 1: annealed softmax Adam
    stages1 = [(80.0, 0.008, 1200), (250.0, 0.004, 1200), (800.0, 0.002, 1200)]
    X, best, bestv = _adam_smooth(X, stages1, rng, t_start, T_LIMIT - 90.0, best, bestv)
    if time.time() - t_start > T_LIMIT - 90.0:
        return _norm(best)
    # keep working from best
    X = best.copy()

    # Stage 2: greedy single-point reinsertion (topology changes)
    # time box ~35s
    single_end = time.time() + 35.0
    # compute per-point worst
    for _round in range(8):
        if time.time() > single_end or time.time() - t_start > T_LIMIT - 60.0:
            break
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        rowmax = np.max(G, axis=1)
        order = np.argsort(-rowmax)  # worst first
        improved = False
        for idx in order[:30]:
            if time.time() > single_end or time.time() - t_start > T_LIMIT - 60.0:
                break
            # max without idx
            G2 = X @ X.T
            # mask idx row/col
            G2[idx, :] = -2.0
            G2[:, idx] = -2.0
            m_without = float(np.max(G2))
            if m_without >= bestv - 1e-12:
                # moving single point cannot beat best
                continue
            by, bs, _ = _try_replace(X, int(idx), rng, n_sample=4000)
            if by is None:
                continue
            Xnew = X.copy()
            Xnew[int(idx)] = by / np.linalg.norm(by)
            vnew = _max_cos(Xnew)
            if vnew < bestv - 1e-12:
                X = Xnew
                best = Xnew.copy()
                bestv = vnew
                improved = True
                break  # recompute ordering
        if not improved:
            # try small random kicks to worst pair
            if time.time() > single_end:
                break
            # perturb worst 2 points slightly and re-optimise locally
            w = order[:2]
            Xt = X.copy()
            Xt[w] += 0.002 * rng.standard_normal(size=(2, d))
            Xt = _norm(Xt)
            # quick local adam
            Xt, best2, bestv2 = _adam_smooth(
                Xt, [(500.0, 0.002, 400)], rng, t_start, T_LIMIT - 60.0, best, bestv
            )
            if bestv2 < bestv:
                best, bestv = best2.copy(), bestv2
                X = best.copy()
                improved = True
            else:
                break
    X = best.copy()
    if time.time() - t_start > T_LIMIT - 55.0:
        return _norm(best)

    # Stage 3: second Adam at high beta to squeeze
    stages2 = [(1500.0, 0.0015, 1000), (3000.0, 0.0008, 800)]
    X, best, bestv = _adam_smooth(X, stages2, rng, t_start, T_LIMIT - 40.0, best, bestv)
    X = best.copy()
    # Stage 4: LP polish with remaining time
    remaining = T_LIMIT - (time.time() - t_start) - 1.0
    if remaining > 5.0:
        best = _lp_polish(best, remaining, time.time(), T_LIMIT - (time.time() - t_start) - 0.5)
    best = _norm(best)
    # safety: remove NaN/zero
    best = np.where(np.isfinite(best), best, 0.0)
    nr = np.linalg.norm(best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    best = best / nr
    return best
