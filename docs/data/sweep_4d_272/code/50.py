import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _norm(X):
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def _counts(X):
    n = X.shape[0]
    iu0 = np.triu_indices(n, 1)
    c = (X @ X.T)[iu0]
    m = float(np.max(c))
    cnt1e4 = int(np.sum(c > m - 1e-4))
    cnt1e6 = int(np.sum(c > m - 1e-6))
    cnt8980 = int(np.sum(c > 0.8980))
    return m, cnt1e4, cnt1e6, cnt8980, c


def _hinge_adam(X, t, tau, steps, lr, seed_off=0):
    n, d = X.shape
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        G = X @ X.T
        S = (G - t) / tau
        Sc = np.clip(S, -60.0, 60.0)
        sig = 1.0 / (1.0 + np.exp(-Sc))
        # softplus stable
        sp = np.where(Sc > 20.0, Sc, np.log1p(np.exp(Sc)))
        W = (2.0 * sp * sig) / tau
        np.fill_diagonal(W, 0.0)
        grad = W @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        m1 = b1 * m1 + (1.0 - b1) * gt
        v1 = b2 * v1 + (1.0 - b2) * (gt * gt)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        mh = m1 / bc1
        vh = v1 / bc2
        step = mh / (np.sqrt(vh) + eps)
        np.clip(step, -2.0, 2.0, out=step)
        X = X - lr * step
        X = _norm(X)
    return X


def _lp_polish(X, budget, deadline):
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    start = time.time()
    while time.time() < deadline and (time.time() - start) < budget and radius > 1e-14:
        if deadline - time.time() < 0.8 and radius < 5e-4:
            break
        C = (X @ X.T)[iu]
        mcur = float(C.max())
        sel = C > mcur - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-C[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        Y = X + res.x[:-1].reshape(n, d)
        Y = _norm(Y)
        v = _max_cos(Y)
        if v < best:
            X, best = Y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X


def solve():
    t0 = time.time()
    T_HARD = t0 + 175.0
    T_SEARCH_END = t0 + 148.0
    rng = np.random.default_rng(0)
    try:
        base = np.load("best.npy")
    except Exception:
        try:
            base = np.load("record.npy")
        except Exception:
            base = rng.standard_normal((272, 4))
    base = _norm(np.asarray(base, dtype=np.float64))
    if base.shape != (272, 4):
        base = _norm(rng.standard_normal((272, 4)))
    # remove NaN safety
    base[~np.isfinite(base)] = 0.0
    base = _norm(base)

    best_max_X = base.copy()
    best_max = _max_cos(best_max_X)
    m0, c1e4_0, _, _c8980, _ = _counts(best_max_X)

    # sparse pool: keep max<0.8985 with smallest near-contact count
    sparse_X = base.copy()
    sparse_cnt = c1e4_0
    sparse_max = m0

    thresholds = [0.8980, 0.8975, 0.8970, 0.8965]
    taus = [1e-4, 1e-4, 1e-4, 1e-4]
    # log histograms (for internal selection, cheap)
    hist_log = []

    it = 0
    sigmas = [0.008, 0.015, 0.025, 0.012, 0.020, 0.006]
    while time.time() < T_SEARCH_END:
        ti = it % len(thresholds)
        t_thr = thresholds[ti]
        tau = taus[ti]
        # choose base: mostly sparse, sometimes best max
        if it % 4 == 3:
            startX = best_max_X.copy()
        else:
            startX = sparse_X.copy()
        sigma = sigmas[it % len(sigmas)]
        # kick to change basin topology
        startX = startX + sigma * rng.standard_normal(startX.shape)
        startX = _norm(startX)

        # surrogate stage: Riemannian Adam on smooth hinge
        # adapt steps/lr to remaining time
        remain = T_SEARCH_END - time.time()
        if remain < 3.0:
            break
        steps = 350
        if remain < 12.0:
            steps = 150
        lr = 0.0025 if ti < 2 else 0.0020
        # slightly anneal lr within stage via two calls
        Xsur = _hinge_adam(startX, t_thr, tau, steps, lr)
        # extra sharpening at higher threshold for last 100 steps
        if time.time() < T_SEARCH_END - 2.0 and it % 2 == 0:
            Xsur = _hinge_adam(Xsur, 0.8980, 8e-5, 80, 0.0012)

        ms, cnt1e4, cnt1e6, cntT, _ = _counts(Xsur)
        hist_log.append((float(ms), int(cnt1e4), float(t_thr)))

        # retain sparse if max below 0.8985 and sparser
        if ms < 0.8985:
            if cnt1e4 < sparse_cnt or (cnt1e4 == sparse_cnt and ms < sparse_max):
                sparse_X = Xsur.copy()
                sparse_cnt = cnt1e4
                sparse_max = ms
            # also keep very good max even if not sparser for best_max
            if ms < best_max:
                best_max_X = Xsur.copy()
                best_max = ms

        # LP polish short burst to snap to local optimum
        budget = 3.5
        if remain < 10.0:
            budget = 1.5
        Xlp = _lp_polish(Xsur, budget, T_SEARCH_END + 1.0)
        mlp, cnt1e4lp, _, _, _ = _counts(Xlp)
        if mlp < best_max:
            best_max_X = Xlp.copy()
            best_max = mlp
        if mlp < 0.8985 and cnt1e4lp < sparse_cnt:
            sparse_X = Xlp.copy()
            sparse_cnt = cnt1e4lp
            sparse_max = mlp
        # if LP made progress on sparse max, also consider it sparse
        if mlp < 0.8985 and cnt1e4lp == sparse_cnt and mlp < sparse_max:
            sparse_X = Xlp.copy()
            sparse_max = mlp

        it += 1
        if it > 60:
            break

    # final polish of the true best with remaining time
    if time.time() < T_HARD - 2.0:
        budget = max(1.0, T_HARD - 1.0 - time.time())
        best_max_X = _lp_polish(best_max_X, budget, T_HARD - 0.5)
        best_max = _max_cos(best_max_X)
        # if sparse is actually better max, prefer it
        if sparse_max < best_max and sparse_max < 0.8985:
            # polish sparse too briefly if time left
            if T_HARD - time.time() > 3.0:
                sparse_X = _lp_polish(sparse_X, min(4.0, T_HARD - 1.0 - time.time()), T_HARD - 0.5)
                sparse_max = _max_cos(sparse_X)
            if sparse_max < best_max:
                best_max_X = sparse_X.copy()
                best_max = sparse_max

    best_max_X = _norm(best_max_X)
    return best_max_X
