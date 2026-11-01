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


def _lp_polish(X, seconds):
    X = _norm(np.asarray(X, dtype=np.float64).copy())
    n, d = X.shape
    best = _max_cos(X)
    if seconds <= 0:
        return X
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        # time check for remaining
        if time.time() - start >= seconds:
            break
        G = X @ X.T
        c = G[iu0, iu1]
        m = float(np.max(c))
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        if not np.any(sel):
            break
        I = iu0[sel]
        J = iu1[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        # cols: I*d .. , J*d .. , last
        cI = (I[:, None] * d + np.arange(d)).ravel()
        cJ = (J[:, None] * d + np.arange(d)).ravel()
        cT = np.full(p, n * d)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        cols[0::2 * d + 1] = cI[0::d] if False else 0  # placeholder, replaced below
        # build cols vectorised
        base = np.concatenate(
            [np.repeat(I * d, d) + np.tile(np.arange(d), p),
             np.repeat(J * d, d) + np.tile(np.arange(d), p),
             np.full(p, n * d)]
        )
        # rows vectorised: each pair contributes 2*d+1 entries
        # order must match vals order: x[J], x[I], -1
        rows2 = np.repeat(np.arange(p), 2 * d + 1)
        vals = np.concatenate([X[J].ravel(), X[I].ravel(), -np.ones(p)])
        # Need cols in same order: I-block then J-block then t
        # I-block: for pair k, cols = I[k]*d + a
        # J-block: cols = J[k]*d + a
        cols_I = (I[:, None] * d + np.arange(d)).ravel()
        cols_J = (J[:, None] * d + np.arange(d)).ravel()
        # interleave per row: first d are I? Actually vals = [X[J], X[I]] so first d correspond to I-vars with X[J] coeffs
        # Constraint: <x_i, y_j> + <x_j, y_i> - t <= -c ? Let's follow parent: vals = [x[J], x[I]]
        # cols for first block should be I-indices, second J-indices
        cols_all = np.concatenate([cols_I, cols_J, np.full(p, n * d)])
        # But rows2 expects per-row grouping, while cols_all is block-grouped. Reorder to per-row.
        # Build per-row ordering:
        cols_per = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals_per = np.empty(p * (2 * d + 1), dtype=np.float64)
        # reshape helpers
        cI_mat = I[:, None] * d + np.arange(d)
        cJ_mat = J[:, None] * d + np.arange(d)
        vI_mat = X[J]  # coeff for dI is x_J
        vJ_mat = X[I]  # coeff for dJ is x_I
        for k in range(p):
            s = k * (2 * d + 1)
            cols_per[s:s + d] = cI_mat[k]
            vals_per[s:s + d] = vI_mat[k]
            cols_per[s + d:s + 2 * d] = cJ_mat[k]
            vals_per[s + d:s + 2 * d] = vJ_mat[k]
            cols_per[s + 2 * d] = n * d
            vals_per[s + 2 * d] = -1.0
        A_ub = coo_matrix((vals_per, (rows2, cols_per)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X


def _smooth_opt(X0, beta0, beta1, iters, lr):
    X = np.asarray(X0, dtype=np.float64).copy()
    n, d = X.shape
    iu0, iu1 = np.triu_indices(n, 1)
    M = np.zeros_like(X)
    V = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    for t in range(1, iters + 1):
        frac = (t - 1) / max(iters - 1, 1)
        beta = beta0 + (beta1 - beta0) * frac
        cur_lr = lr * (1.0 - 0.5 * frac)
        G = X @ X.T
        gmax = float(np.max(G[iu0, iu1]))
        # weights, shift by gmax for stability
        B = np.exp(beta * (G - gmax))
        np.fill_diagonal(B, 0.0)
        Grad = B @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        M = b1 * M + (1.0 - b1) * Gt
        V = b2 * V + (1.0 - b2) * (Gt * Gt)
        bc1 = 1.0 - (b1 ** t)
        bc2 = 1.0 - (b2 ** t)
        mhat = M / bc1
        vhat = V / bc2
        step = cur_lr * mhat / (np.sqrt(vhat) + eps)
        X = X - step
        X = _norm(X)
    return X


def _contacts(X, thr=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    cnt = np.sum(G > m - thr, axis=1)
    return m, cnt, G


def solve():
    np.random.seed(0)
    rng = np.random.default_rng(0)
    t0 = time.time()
    HARD = 175.0

    # load starting point
    X_best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            A = np.load(fn)
            A = np.asarray(A, dtype=np.float64)
            if A.shape != (173, 5):
                continue
            A = _norm(A)
            v = _max_cos(A)
            if v < best_val:
                best_val = v
                X_best = A.copy()
        except Exception:
            continue
    if X_best is None:
        X_best = _norm(rng.standard_normal((173, 5)))
        best_val = _max_cos(X_best)

    n, d = 173, 5

    # quick contact info
    def greedy_reinsert(X_rem, k):
        # X_rem: (m,d), return (m+k,d)
        cur = X_rem.copy()
        P = 20000
        C = rng.standard_normal((P, d))
        C = _norm(C)
        for _ in range(k):
            S = C @ cur.T
            smax = np.max(S, axis=1)
            # diversify: pick among top 30
            part = np.argpartition(smax, 30)[:30]
            # choose randomly among them weighted to best
            pick = part[rng.integers(0, len(part))]
            # slight jitter
            v = C[pick] + 0.01 * rng.standard_normal(d)
            v = v / np.linalg.norm(v)
            cur = np.vstack([cur, v[None, :]])
        return cur

    explore_end = t0 + HARD - 28.0
    it = 0
    # cycle modes
    while time.time() < explore_end:
        it += 1
        mode = it % 6
        # time left check
        if time.time() >= explore_end:
            break
        X0 = X_best.copy()
        # compute contacts for targeted modes
        if mode in (3, 4, 5):
            try:
                m0, cnt, _ = _contacts(X0, thr=1e-7)
            except Exception:
                cnt = np.zeros(n, dtype=float)
        if mode == 0:
            sigma = 0.015
            Xp = _norm(X0 + sigma * rng.standard_normal((n, d)))
            Xs = _smooth_opt(Xp, 40.0, 120.0, 900, 0.012)
        elif mode == 1:
            sigma = 0.05
            Xp = _norm(X0 + sigma * rng.standard_normal((n, d)))
            Xs = _smooth_opt(Xp, 50.0, 150.0, 1200, 0.015)
        elif mode == 2:
            sigma = 0.12
            Xp = _norm(X0 + sigma * rng.standard_normal((n, d)))
            Xs = _smooth_opt(Xp, 30.0, 100.0, 1500, 0.018)
        elif mode == 3:
            # targeted shake proportional to contacts
            mx = float(np.max(cnt)) if float(np.max(cnt)) > 0 else 1.0
            scale = 0.02 + 0.08 * (cnt / mx)
            noise = rng.standard_normal((n, d)) * scale[:, None]
            Xp = _norm(X0 + noise)
            Xs = _smooth_opt(Xp, 50.0, 160.0, 1200, 0.014)
        elif mode == 4:
            k = 2
            order = np.argsort(-cnt)
            drop = set(order[:k].tolist())
            # always also consider rattler (0 contacts) for diversity every other time
            if (it // 6) % 2 == 1:
                # drop lowest contact instead of one highest
                drop = set([int(order[0]), int(np.argmin(cnt))])
                drop = list(drop)[:k]
                if len(drop) < k:
                    drop = order[:k]
            keep = [i for i in range(n) if i not in set(np.asarray(drop).tolist())]
            X_rem = X0[np.array(keep)]
            Xp = greedy_reinsert(X_rem, k)
            # small global jitter
            Xp = _norm(Xp + 0.01 * rng.standard_normal((n, d)))
            Xs = _smooth_opt(Xp, 40.0, 140.0, 1300, 0.015)
        else:
            k = 5
            order = np.argsort(-cnt)
            drop = order[:k]
            keep = [i for i in range(n) if i not in set(drop.tolist())]
            X_rem = X0[np.array(keep)]
            Xp = greedy_reinsert(X_rem, k)
            Xp = _norm(Xp + 0.015 * rng.standard_normal((n, d)))
            Xs = _smooth_opt(Xp, 30.0, 120.0, 1400, 0.016)

        try:
            v_sm = _max_cos(Xs)
        except Exception:
            continue
        # promising? allow small regression for LP to fix
        if v_sm < best_val + 0.004:
            # short LP polish
            remain = explore_end - time.time()
            if remain < 1.0:
                # just update if strictly better without polish
                if v_sm < best_val:
                    X_best = Xs
                    best_val = v_sm
                continue
            budget = min(4.0, remain)
            # use less budget if far
            if v_sm > best_val + 0.002:
                budget = min(2.0, remain)
            Xlp = _lp_polish(Xs, budget)
            try:
                v_lp = _max_cos(Xlp)
            except Exception:
                continue
            if v_lp < best_val:
                X_best = Xlp
                best_val = v_lp
        else:
            if v_sm < best_val:
                X_best = Xs
                best_val = v_sm
        # safety: stop if time over
        if time.time() >= explore_end:
            break

    # final intensive polish with remaining time
    rem = HARD - (time.time() - t0)
    rem = float(np.clip(rem, 3.0, 30.0))
    # ensure we leave margin for 180s limit
    elapsed = time.time() - t0
    if elapsed < 178.0:
        try:
            X_best = _lp_polish(X_best, min(rem, 178.0 - elapsed))
        except Exception:
            pass
    X_best = _norm(X_best)
    return X_best
