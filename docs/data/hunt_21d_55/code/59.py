import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _counts(x, thr=None, m=None):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    if m is None:
        m = float(np.max(g))
    if thr is None:
        return m, int(np.sum(g > m - 1e-6) // 1) if False else (m, 0)
    return m, int(np.count_nonzero(g > thr))

def _quench(X0, steps, beta0, beta1, lr0, lr1):
    X = X0.copy()
    n, d = X.shape
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = beta0 + (beta1 - beta0) * f
        lr = lr0 * (lr1 / lr0) ** f if lr1 > 0 else lr0
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        # avoid overflow: beta*(G-m) <= 0
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Gw = W @ X
        dot = np.sum(Gw * X, axis=1, keepdims=True)
        Gt = Gw - dot * X
        nrm = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        Gt = Gt / nrm
        X = X - lr * Gt
        X = _norm_rows(X)
    return X

def _lp_polish(x, deadline, radius_init=1e-3):
    x = _norm_rows(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius_init
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        m = float(cfull.max())
        sel = cfull > m - max(20 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 1100:
            idx = np.where(sel)[0]
            order = np.argsort(-cfull[idx])[:1100]
            nsel = np.zeros_like(sel, dtype=bool)
            nsel[idx[order]] = True
            sel = nsel
        csel = cfull[sel]
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
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
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    LIMIT = 170.0
    # load best
    X0 = None
    best_load = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (55, 21):
                a = _norm_rows(a)
                s = _max_cos(a)
                if s < best_load:
                    best_load = s
                    X0 = a.copy()
        except Exception:
            pass
    if X0 is None:
        rng0 = np.random.default_rng(0)
        X0 = _norm_rows(rng0.standard_normal((55, 21)))
        best_load = _max_cos(X0)
    n, d = 55, 21
    X = X0.copy()
    m0 = _max_cos(X)
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    cnt0 = int(np.count_nonzero(G0 > m0 - 1e-6) // 2)  # unique pairs
    # active pairs with cosine > max-1e-6
    iu = np.triu_indices(n, 1)
    cfull = (X @ X.T)[iu]
    act = np.where(cfull > m0 - 1e-6)[0]
    # if too many, keep tightest 600 for Jacobian (still representative)
    if len(act) > 800:
        order = np.argsort(-cfull[act])[:800]
        act = act[order]
    p = len(act)
    N = n * d
    # Build Jacobian p x N
    J = np.zeros((p, N), dtype=np.float64)
    s2 = 1.0 / np.sqrt(2.0)
    for k, aidx in enumerate(act):
        i = int(iu[0][aidx]); j = int(iu[1][aidx])
        cij = float(cfull[aidx])
        vi = (X[j] - cij * X[i]) * s2
        vj = (X[i] - cij * X[j]) * s2
        J[k, i * d:(i + 1) * d] = vi
        J[k, j * d:(j + 1) * d] = vj
    # nullspace via eig of J^T J (N x N)
    try:
        JTJ = J.T @ J
        # symmetrize
        JTJ = 0.5 * (JTJ + JTJ.T)
        w, V = np.linalg.eigh(JTJ)  # ascending, columns
    except Exception:
        V = None
        w = None
    dirs = []
    if V is not None:
        Ktake = min(80, N)
        for k in range(Ktake):
            vec = V[:, k]
            M = vec.reshape(n, d)
            dot = np.sum(M * X, axis=1, keepdims=True)
            Mt = M - dot * X
            tn = float(np.linalg.norm(Mt))
            if tn < 0.25:  # mostly radial -> skip
                continue
            Mt = Mt / tn
            dirs.append(Mt)
            if len(dirs) >= 32:
                break
    # fallback random tangent dirs if eig failed
    if len(dirs) < 8:
        rng = np.random.default_rng(0)
        while len(dirs) < 12:
            M = rng.standard_normal((n, d))
            dot = np.sum(M * X, axis=1, keepdims=True)
            M = M - dot * X
            M = M / (np.linalg.norm(M) + 1e-12)
            dirs.append(M)
    amps = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16]
    # line search: NO LP inside, score by (active count, max)
    # active count = #pairs within 1e-6 of new max (unique)
    best_cands = []  # (cnt_new, m_new, amp, dir_idx, Y)
    for di, D in enumerate(dirs):
        if time.time() - t0 > 60:
            break
        for amp in amps:
            for sgn in (+1.0, -1.0):
                Y = _norm_rows(X + (sgn * amp) * D)
                G = Y @ Y.T
                np.fill_diagonal(G, -2.0)
                m = float(G.max())
                if m > 0.075:
                    continue
                cnt = int(np.count_nonzero(G > m - 1e-6) // 2)
                best_cands.append((cnt, m, amp, di, sgn, Y))
    # sort by cnt then m
    best_cands.sort(key=lambda t: (t[0], t[1]))
    # keep candidates that materially reduce active set or improve max
    filtered = []
    for (cnt, m, amp, di, sgn, Y) in best_cands:
        if cnt < cnt0 - 5 or m < m0 - 1e-9:
            filtered.append((cnt, m, amp, di, sgn, Y))
        if len(filtered) >= 40:
            break
    # if nothing reduces, take top few anyway with smallest cnt
    if len(filtered) == 0:
        filtered = best_cands[:6]
    else:
        filtered = filtered[:6]
    x_best = X.copy()
    best_score = m0
    # quench each candidate with soft-max descent (no LP yet)
    for (cnt, m, amp, di, sgn, Y) in filtered:
        if time.time() - t0 > 120:
            break
        # two quench settings: short then longer if promising
        Yq = _quench(Y, steps=700, beta0=60.0, beta1=700.0, lr0=0.015, lr1=0.002)
        mq = _max_cos(Yq)
        # second anneal if improved over start or cnt was much smaller
        if mq < m + 0.002:
            Yq2 = _quench(Yq, steps=500, beta0=200.0, beta1=1200.0, lr0=0.006, lr1=0.001)
            mq2 = _max_cos(Yq2)
            if mq2 < mq:
                Yq, mq = Yq2, mq2
        if mq < best_score:
            best_score = mq
            x_best = Yq.copy()
    # only then short LP polish on the best quenched (and on original as fallback)
    remain = LIMIT - (time.time() - t0)
    if remain > 12:
        # polish the best found with limited budget
        deadline = time.time() + min(remain - 2, 25.0)
        x_best = _lp_polish(x_best, deadline)
        best_score = _max_cos(x_best)
    # ensure we never return worse than loaded best
    if best_score > m0:
        return _norm_rows(X0)
    return _norm_rows(x_best)
