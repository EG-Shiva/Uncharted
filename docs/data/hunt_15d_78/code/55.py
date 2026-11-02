import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _top200(X):
    n = X.shape[0]
    G = X @ X.T
    iu0, iu1 = np.triu_indices(n, 1)
    gv = G[iu0, iu1]
    s = np.sort(gv)[::-1]
    return s[:200].copy(), float(gv.max())

def _lp_polish(X, budget, cap=1500, max_it=30):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    best_x = X.copy()
    radius = 4e-3
    iu0, iu1 = np.triu_indices(n, 1)
    nv = n * d + 1
    obj = np.zeros(nv); obj[-1] = 1.0
    start = time.time()
    it = 0
    while time.time() - start < budget and radius > 1e-13 and it < max_it:
        it += 1
        G = X @ X.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thr = max(20.0 * radius, 1e-9)
        sel = c_all > m - thr
        nnz = int(np.count_nonzero(sel))
        if nnz == 0:
            radius *= 0.5
            continue
        if nnz > cap:
            idx = np.where(sel)[0]
            part = np.argpartition(-c_all[idx], cap)[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[part]] = True
            sel = mask
        I = iu0[sel]; J = iu1[sel]
        p = len(I)
        c_sel = c_all[sel]
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        bounds = [(-radius, radius)] * (n * d) + [(None, None)]
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y = y / yn
        v = _max_cos(y)
        if v < best:
            X = y; best = v; best_x = y.copy()
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return best_x, best

def solve():
    t0 = time.time()
    HARD = t0 + 165.0
    n, d = 78, 15
    bestX = None; bestV = 1e9
    recX = None
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape != (78, 15):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            v = _max_cos(a)
            if fn == "record.npy":
                recX = a.copy()
            if v < bestV:
                bestV = v; bestX = a.copy()
        except Exception:
            continue
    if bestX is None:
        rng0 = np.random.default_rng(0)
        bestX = rng0.standard_normal((78, 15))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)
    if recX is None:
        recX = bestX.copy()
    try:
        rec_top, rec_max = _top200(recX)
    except Exception:
        rec_top = None

    # batched fresh search
    B = 32
    STEPS = 800
    rng = np.random.default_rng(20261101)
    X = rng.standard_normal((B, n, d))
    X /= np.linalg.norm(X, axis=2, keepdims=True)
    M = np.zeros_like(X)
    V = np.zeros_like(X)
    b1, b2 = 0.9, 0.999
    eps = 1e-8
    eyeD = np.eye(d, dtype=np.float64) / float(d)
    arB = np.arange(B)
    arN = np.arange(n)
    # anneal params precomputed
    for s in range(STEPS):
        if time.time() > HARD - 50:
            break
        frac = s / max(STEPS - 1, 1)
        beta = 20.0 * ((300.0 / 20.0) ** frac)
        lr = 0.03 * ((0.005 / 0.03) ** frac)
        lam1 = 2.0 * (1.0 - frac)
        lam2 = 0.5 * (1.0 - frac)
        G = np.matmul(X, X.transpose(0, 2, 1))
        G[:, arN, arN] = -2.0
        m = G.max(axis=(1, 2))
        # LSE weights
        Eexp = np.exp(beta * (G - m[:, None, None]))
        Eexp[:, arN, arN] = 0.0
        Ssum = Eexp.sum(axis=(1, 2))
        Ssum = np.maximum(Ssum, 1e-300)
        W = Eexp / Ssum[:, None, None]
        Gr = np.matmul(W, X)
        # isotropy grad
        C = np.matmul(X.transpose(0, 2, 1), X) / float(n)
        Emat = C - eyeD[None, :, :]
        Giso = (4.0 / float(n)) * np.matmul(X, Emat)
        # personal-max variance grad (simplified, gather only)
        srow = G.max(axis=2)
        sbar = srow.mean(axis=1)
        coeff = 2.0 * (srow - sbar[:, None]) / float(n)
        idx = G.argmax(axis=2)
        try:
            Xstar = X[arB[:, None], idx]
        except Exception:
            Xstar = np.zeros_like(X)
            for bb in range(B):
                Xstar[bb] = X[bb][idx[bb]]
        Gv = coeff[:, :, None] * Xstar
        Raw = Gr + lam1 * Giso + lam2 * Gv
        dot = (Raw * X).sum(axis=2, keepdims=True)
        Gt = Raw - dot * X
        if not np.all(np.isfinite(Gt)):
            break
        M = b1 * M + (1.0 - b1) * Gt
        V = b2 * V + (1.0 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** (s + 1)
        c2 = 1.0 - b2 ** (s + 1)
        Mh = M / c1
        Vh = V / c2
        step = lr * Mh / (np.sqrt(Vh) + eps)
        sn = np.sqrt((step * step).sum(axis=2, keepdims=True))
        scale = np.minimum(1.0, 0.05 / (sn + 1e-12))
        step = step * scale
        X = X - step
        nn = np.linalg.norm(X, axis=2, keepdims=True)
        nn = np.maximum(nn, 1e-12)
        X = X / nn

    # evaluate replicas
    vals = np.empty(B)
    tops = []
    eig_spread = np.empty(B)
    deg_var = np.empty(B)
    for bb in range(B):
        Xb = X[bb] / np.linalg.norm(X[bb], axis=1, keepdims=True)
        X[bb] = Xb
        Gbb = Xb @ Xb.T
        np.fill_diagonal(Gbb, -2.0)
        vals[bb] = float(Gbb.max())
        gv = Gbb[np.triu_indices(n, 1)]
        srt = np.sort(gv)[::-1][:200]
        tops.append(srt)
        Cb = (Xb.T @ Xb) / float(n)
        try:
            ev = np.linalg.eigvalsh(Cb)
            eig_spread[bb] = float(ev.max() - ev.min())
        except Exception:
            eig_spread[bb] = 1e9
        try:
            thr = vals[bb] - 1e-3
            deg = (Gbb > thr).sum(axis=1)
            deg_var[bb] = float(deg.var())
        except Exception:
            deg_var[bb] = 1e9
    tops = np.stack(tops, axis=0) if len(tops) > 0 else np.zeros((B, 200))
    # fingerprint distance to record
    dup = np.zeros(B, dtype=bool)
    if rec_top is not None:
        for bb in range(B):
            dist = float(np.linalg.norm(tops[bb] - rec_top))
            if dist < 2e-3:
                dup[bb] = True
    order = np.argsort(vals)
    # pick distinct candidates
    cand = []
    for bb in order:
        if vals[bb] > 0.20:
            continue
        if dup[bb]:
            continue
        ok = True
        for cc in cand:
            if float(np.linalg.norm(tops[bb] - tops[cc])) < 1e-3:
                ok = False
                break
        if ok:
            cand.append(int(bb))
        if len(cand) >= 3:
            break
    # if no non-duplicate good candidate, try best distinct regardless of dup for polishing (but keep fallback)
    if len(cand) == 0:
        # take best 2 with small eig spread as balanced
        bal = order[np.argsort(eig_spread[order][:8])]
        for bb in bal[:2]:
            if vals[int(bb)] < 0.20:
                cand.append(int(bb))
        if len(cand) == 0:
            for bb in order[:2]:
                cand.append(int(bb))
    # LP polish best 2-3 distinct
    for k, bb in enumerate(cand[:3]):
        if time.time() > HARD - 8:
            break
        remaining = HARD - 5 - time.time()
        if remaining < 2:
            break
        budget = min(12.0, remaining / (len(cand[:3]) - k))
        if budget < 1.0:
            break
        Yb = X[bb].copy()
        try:
            Yp, vp = _lp_polish(Yb, budget=budget)
        except Exception:
            continue
        if vp < bestV - 1e-12:
            bestV = vp
            bestX = Yp.copy()
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
