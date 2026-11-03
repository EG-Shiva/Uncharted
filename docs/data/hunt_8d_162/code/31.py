import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _sym_score(Y):
    G = Y @ Y.T
    np.fill_diagonal(G, 0.0)
    return float(np.abs(G).max())

def _greedy_sym(X):
    n = X.shape[0]
    G = X @ X.T
    rem = set(range(n))
    Ylist = []
    while rem:
        # global minimal dot among remaining
        best = None
        bestv = 1e9
        rl = list(rem)
        # brute force over remaining pairs
        for ii in range(len(rl)):
            i = rl[ii]
            row = G[i, rl]
            # mask self
            for jj in range(len(rl)):
                if jj == ii:
                    continue
                v = float(row[jj])
                if v < bestv:
                    bestv = v
                    best = (i, rl[jj])
        i, j = best
        rem.remove(i); rem.remove(j)
        v = X[i] - X[j]
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            v = X[i].copy()
            nv = np.linalg.norm(v)
        Ylist.append(v / nv)
    return np.array(Ylist)

def _sym_smooth_opt(Y, iters, a0, a1, lr0, lr1, noise0, noise1, rng, mu=0.8):
    n, d = Y.shape
    Y = _norm(Y)
    V = np.zeros_like(Y)
    for t in range(iters):
        f = t / max(1, iters - 1)
        if a1 != a0:
            a = a0 * ((a1 / a0) ** f)
        else:
            a = a0
        if lr1 != lr0:
            lr = lr0 * ((lr1 / lr0) ** f)
        else:
            lr = lr0
        if noise0 <= 0:
            nm = 0.0
        else:
            if noise1 is None or noise1 <= 0:
                nm = noise0 * (1.0 - f)
            else:
                nm = noise0 * ((noise1 / noise0) ** f)
        G = Y @ Y.T
        np.fill_diagonal(G, 0.0)
        A = np.abs(G)
        m = float(A.max())
        # avoid overflow: exp(a*(A-m)), A-m<=0
        W = np.exp(a * (A - m))
        np.fill_diagonal(W, 0.0)
        Sgn = np.sign(G)
        np.fill_diagonal(Sgn, 0.0)
        Ws = W * Sgn
        S = float(W.sum())
        if not np.isfinite(S) or S == 0:
            break
        H = Ws @ Y
        WG = (Ws * G).sum(axis=1)
        Grad = (H - WG[:, None] * Y) / S
        V = mu * V + Grad
        Y = Y - lr * V
        if nm > 0:
            N = rng.randn(n, d)
            nd = (N * Y).sum(axis=1, keepdims=True)
            N = N - nd * Y
            Y = Y + nm * N
        Y = _norm(Y)
    return Y

def _smooth_opt(X, iters, a0, a1, lr0, lr1, noise0, noise1, rng, mu=0.8):
    n, d = X.shape
    V = np.zeros_like(X)
    for t in range(iters):
        f = t / max(1, iters - 1)
        a = a0 * ((a1 / a0) ** f) if a1 != a0 else a0
        lr = lr0 * ((lr1 / lr0) ** f) if lr1 != lr0 else lr0
        if noise0 <= 0:
            nm = 0.0
        else:
            nm = noise0 * ((noise1 / noise0) ** f) if (noise1 is not None and noise1 > 0) else noise0 * (1.0 - f)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        Gm = float(G.max())
        W = np.exp(a * (G - Gm))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum())
        if not np.isfinite(S) or S == 0:
            break
        H = W @ X
        WG = (W * G).sum(axis=1)
        Grad = (H - WG[:, None] * X) / S
        V = mu * V + Grad
        X = X - lr * V
        if nm > 0:
            N = rng.randn(n, d)
            nd = (N * X).sum(axis=1, keepdims=True)
            N = N - nd * X
            X = X + nm * N
        X = _norm(X)
    return X

def _lp_polish(X, seconds, t_deadline=None):
    X = _norm(np.array(X, dtype=float))
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t_deadline is not None and time.time() > t_deadline:
            break
        if radius < 1e-14:
            break
        Gfull = X @ X.T
        cfull = Gfull[iu0]
        m = float(cfull.max())
        sel = cfull > m - max(20 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 4000:
            idx = np.argpartition(-cfull[sel], 4000)[:4000]
            I = I[idx]; J = J[idx]
            c = (X @ X.T)[I, J]
            p = len(I)
        else:
            c = cfull[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c,
                          A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
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

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.RandomState(12345)
    # load incumbent
    Xb = None; sb = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (162, 8):
                v = _norm(v)
                s = _max_cos(v)
                if s < sb:
                    sb = s; Xb = v.copy()
        except Exception:
            pass
    if Xb is None:
        Xb = _norm(rng.randn(162, 8)); sb = _max_cos(Xb)
    X_inc = Xb.copy(); s_inc = float(sb)
    n81 = 81; d = 8
    # greedy symmetric init from best
    try:
        Y0 = _greedy_sym(X_inc)
    except Exception:
        Y0 = _norm(rng.randn(n81, d))
    Y0 = _norm(Y0)
    s0 = _sym_score(Y0)
    # quick staged polish of greedy init
    Y_best = _sym_smooth_opt(Y0.copy(), 500, 30, 350, 0.012, 0.0015, 0.0025, 0.0, rng)
    Y_best = _sym_smooth_opt(Y_best, 350, 80, 900, 0.006, 0.0008, 0.0, 0.0, rng)
    y_best_score = _sym_score(Y_best)
    # also try random inits quickly to diversify
    # time limit for symmetric phase: reserve ~34s for final
    sym_deadline = deadline - 34.0
    # if greedy is very bad, random may be better; evaluate one random polished
    Yr = _norm(rng.randn(n81, d))
    Yr = _sym_smooth_opt(Yr, 500, 30, 350, 0.012, 0.0015, 0.0025, 0.0, rng)
    Yr = _sym_smooth_opt(Yr, 350, 80, 900, 0.006, 0.0008, 0.0, 0.0, rng)
    sr = _sym_score(Yr)
    if sr < y_best_score:
        Y_best = Yr; y_best_score = sr
    trial = 0
    # basin hopping in symmetric space
    while time.time() < sym_deadline and time.time() < t0 + 135:
        if time.time() > deadline - 40:
            break
        trial += 1
        mode = trial % 5
        Yt = Y_best.copy()
        if mode == 0:
            Yt = _norm(Yt + rng.randn(n81, d) * 0.010)
            Yt = _sym_smooth_opt(Yt, 450, 30, 350, 0.012, 0.0015, 0.002, 0.0, rng)
            Yt = _sym_smooth_opt(Yt, 300, 80, 900, 0.006, 0.0008, 0.0, 0.0, rng)
        elif mode == 1:
            Yt = _norm(Yt + rng.randn(n81, d) * 0.028)
            Yt = _sym_smooth_opt(Yt, 550, 25, 300, 0.014, 0.0012, 0.003, 0.0, rng)
            Yt = _sym_smooth_opt(Yt, 300, 80, 900, 0.006, 0.0008, 0.0, 0.0, rng)
        elif mode == 2:
            k = 6
            idx = rng.choice(n81, k, replace=False)
            Yt[idx] = _norm(Yt[idx] + rng.randn(k, d) * 0.18)
            Yt = _sym_smooth_opt(Yt, 550, 25, 350, 0.014, 0.0012, 0.003, 0.0, rng)
            Yt = _sym_smooth_opt(Yt, 300, 80, 900, 0.006, 0.0008, 0.0, 0.0, rng)
        elif mode == 3:
            # fresh random start
            Yt = _norm(rng.randn(n81, d))
            Yt = _sym_smooth_opt(Yt, 500, 25, 350, 0.013, 0.0012, 0.0025, 0.0, rng)
            Yt = _sym_smooth_opt(Yt, 300, 80, 900, 0.006, 0.0008, 0.0, 0.0, rng)
        else:
            # sign-flip + rotate cap: coordinated mirrored move
            u = rng.randn(d); u /= np.linalg.norm(u)
            proj = (Y_best @ u)
            # flip signs of points with negative proj randomly to explore projective equivalence
            # actually sign of y is irrelevant, so rotate cap instead
            thr = np.quantile(np.abs(proj), 0.80)
            sel = np.where(np.abs(proj) >= thr)[0]
            R = rng.randn(d, d) * 0.05; R = R - R.T
            for i in sel:
                Yt[i] = Yt[i] + (R @ Yt[i]) + rng.randn(d) * 0.008
            Yt = _norm(Yt)
            Yt = _sym_smooth_opt(Yt, 500, 25, 350, 0.013, 0.0012, 0.002, 0.0, rng)
            Yt = _sym_smooth_opt(Yt, 300, 80, 900, 0.006, 0.0008, 0.0, 0.0, rng)
        try:
            st = _sym_score(Yt)
        except Exception:
            continue
        if st < y_best_score:
            Y_best = Yt; y_best_score = st
    # unfold best symmetric candidate: X = (y,-y)
    X_sym = np.vstack([Y_best, -Y_best])
    X_sym = _norm(X_sym)
    s_sym = _max_cos(X_sym)  # equals y_best_score in theory
    # final unconstrained phase: allow symmetry breaking (25-35s)
    # start from slightly broken symmetry + smooth anneal then LP
    X_cur = _norm(X_sym + rng.randn(162, 8) * 0.004)
    # short symmetry-breaking smooth polish
    # use remaining time: smooth takes ~ few seconds
    if time.time() < deadline - 30:
        X_cur = _smooth_opt(X_cur, 500, 30, 400, 0.012, 0.0012, 0.0015, 0.0, rng)
    if time.time() < deadline - 28:
        X_cur = _smooth_opt(X_cur, 350, 80, 900, 0.006, 0.0008, 0.0, 0.0, rng)
    s_cur = _max_cos(X_cur)
    # LP polish with remaining budget
    remain = deadline - time.time() - 1.0
    if remain > 3:
        X_cur = _lp_polish(X_cur, remain, t_deadline=deadline - 0.5)
        s_cur = _max_cos(X_cur)
    # choose best among symmetric-polished and incumbent (also quick polish incumbent if sym failed badly?)
    X_best = X_inc; s_best = s_inc
    if s_sym < s_best:
        # unfolded exact symmetric (keep if better)
        pass
    if s_cur < s_best:
        X_best = X_cur; s_best = s_cur
    else:
        # if symmetric route did not beat incumbent, spend any leftover tiny time polishing incumbent?
        # only if we have time and sym route was worse by a lot, return incumbent to avoid regression
        pass
    # safety: if s_cur is within 0.003 of incumbent but incumbent is better, still return incumbent
    X_best = _norm(X_best)
    return X_best
