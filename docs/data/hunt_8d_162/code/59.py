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

def _contacts(X, tol=1e-9):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cnt = np.zeros(n, dtype=int)
    iu = np.triu_indices(n, 1)
    c = G[iu]
    sel = c >= m - tol
    I = iu[0][sel]; J = iu[1][sel]
    for i, j in zip(I, J):
        cnt[i] += 1; cnt[j] += 1
    return cnt, float(m)

def _smooth_opt(X, iters, a0, a1, lr0, lr1, rng, noise0=0.0, noise1=0.0, deadline=None):
    X = _norm(X)
    n, d = X.shape
    for t in range(iters):
        if deadline is not None and time.time() > deadline:
            break
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
            if noise1 <= 0:
                nm = noise0 * (1.0 - f)
            else:
                nm = noise0 * ((noise1 / noise0) ** f)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        Gm = float(G.max())
        # avoid overflow: (G-Gm)<=0
        W = np.exp(a * (G - Gm))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum())
        if not np.isfinite(S) or S <= 0:
            break
        H = W @ X
        WG = (W * G).sum(axis=1)
        Grad = (H - WG[:, None] * X) / S
        gn = float(np.linalg.norm(Grad, axis=1).max())
        if gn < 1e-16:
            # still add noise maybe
            pass
        else:
            Grad = Grad / gn
        X = X - lr * Grad
        if nm > 0:
            N = rng.randn(n, d)
            nd = (N * X).sum(axis=1, keepdims=True)
            N = N - nd * X
            nn = np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
            N = N / nn
            X = X + nm * N
        X = _norm(X)
    return X

def _lp_polish(X, seconds, deadline=None, r0=0.005, rmax=0.03):
    X = _norm(np.array(X, dtype=float))
    n, d = X.shape
    best = _max_cos(X)
    radius = r0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    iuI = iu0[0]; iuJ = iu0[1]
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        G = X @ X.T
        cf = G[iuI, iuJ]
        m = float(cf.max())
        if m < best:
            best = m
        thresh = m - max(20.0 * radius, 1e-9)
        sel = cf > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 3000:
            part = np.argpartition(-cf, 3000)[:3000]
            mask = np.zeros_like(sel)
            mask[part] = True
            # intersect with sel? keep tightest overall but ensure they are near max;
            # if tightest are all within thresh then fine
            sel = mask & sel
            # if too few, just take top 3000
            if int(sel.sum()) < 500:
                sel = mask
                cnt = 3000
            else:
                cnt = int(sel.sum())
            I = iuI[sel]; J = iuJ[sel]; c = cf[sel]
        else:
            I = iuI[sel]; J = iuJ[sel]; c = cf[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
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
        if v < best - 1e-15:
            X, best = y, v
            radius = min(radius * 1.5, rmax)
        else:
            radius *= 0.5
    return X

def _largest_hole(Xf, rng, n_cand=50000):
    d = Xf.shape[1]
    C = rng.randn(n_cand, d)
    C = _norm(C)
    dots = C @ Xf.T
    m = dots.max(axis=1)
    idx = int(np.argmin(m))
    return C[idx].copy(), float(m[idx])

def _refine_hole(v, Xf, iters=250, beta0=20.0, beta1=400.0, lr0=0.05, lr1=0.002):
    v = v / (np.linalg.norm(v) + 1e-12)
    for t in range(iters):
        f = t / max(1, iters - 1)
        beta = beta0 * ((beta1 / beta0) ** f) if beta1 != beta0 else beta0
        lr = lr0 * ((lr1 / lr0) ** f) if lr1 != lr0 else lr0
        dots = Xf @ v
        dm = float(dots.max())
        w = np.exp(beta * (dots - dm))
        s = float(w.sum())
        if not np.finite(s) or s <= 0:
            break
        g = (w[:, None] * Xf).sum(axis=0) / s
        # move away from weighted neighbours, projected
        gt = g - float(g @ v) * v
        gn = float(np.linalg.norm(gt))
        if gn < 1e-14:
            break
        gt = gt / gn
        v = v - lr * gt
        v = v / (np.linalg.norm(v) + 1e-12)
    return v

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    # load incumbent
    X_best = None
    s_best = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (162, 8):
                v = _norm(v)
                s = _max_cos(v)
                if s < s_best:
                    s_best = s
                    X_best = v.copy()
        except Exception:
            pass
    if X_best is None:
        rng0 = np.random.RandomState(0)
        X_best = _norm(rng0.randn(162, 8))
        s_best = _max_cos(X_best)
    X_best = _norm(X_best)
    # quick safety: ensure finite
    try:
        # branch loop: deletion-reinsertion size continuation
        k_cycle = [8, 12, 4, 20, 8, 12, 20, 4, 12, 8]
        branch = 0
        # precompute contacts of incumbent for first branch; recompute each branch from current best
        while time.time() < deadline - 22 and branch < 12:
            if time.time() > deadline - 22:
                break
            branch_deadline = min(deadline - 4, time.time() + 26)
            k = k_cycle[branch % len(k_cycle)]
            rng = np.random.RandomState(1000 + branch * 7919)
            try:
                cnt, m0 = _contacts(X_best, tol=1e-9)
                order = np.argsort(-cnt, kind="stable")
                rattlers = np.where(cnt == 0)[0].tolist()
                # build removal set: rattlers first then highest contacts
                rem = []
                for r in rattlers:
                    if len(rem) < k:
                        rem.append(int(r))
                for idx in order:
                    if len(rem) >= k:
                        break
                    if int(idx) not in rem:
                        rem.append(int(idx))
                rem = np.array(rem[:k], dtype=int)
                keep_mask = np.ones(162, dtype=bool)
                keep_mask[rem] = False
                X_keep = X_best[keep_mask].copy()
                # small shake to leave basin
                sig = 0.008 if (branch % 2 == 0) else 0.015
                X_keep = _norm(X_keep + rng.randn(*X_keep.shape) * sig)
                # staged exponential repulsion on smaller problem (easier)
                # stage 1 broad
                X_keep = _smooth_opt(X_keep, 350, 25, 250, 0.015, 0.002, rng, 0.002, 0.0,
                                     deadline=branch_deadline)
                if time.time() > branch_deadline:
                    raise TimeoutError()
                # stage 2 focused
                X_keep = _smooth_opt(X_keep, 300, 150, 900, 0.007, 0.001, rng, 0.0, 0.0,
                                     deadline=branch_deadline)
                if time.time() > branch_deadline:
                    raise TimeoutError()
                # LP relax smaller code well below target
                X_keep = _lp_polish(X_keep, seconds=2.0, deadline=branch_deadline, r0=0.008, rmax=0.05)
                if time.time() > branch_deadline:
                    raise TimeoutError()
                # reinsert one at a time at largest hole (50000+ candidates + refine)
                X_cur = X_keep.copy()
                # randomize insertion loop slightly per branch
                for ii in range(k):
                    if time.time() > branch_deadline - 6:
                        break
                    # at least 50000 candidates
                    nc = 50000 if (branch % 3 != 2) else 60000
                    hv, hs = _largest_hole(X_cur, rng, n_cand=nc)
                    hv = _refine_hole(hv, X_cur, iters=250)
                    X_cur = np.vstack([X_cur, hv.reshape(1, -1)])
                    X_cur = _norm(X_cur)
                    # short smooth repair (allow temporary increase)
                    X_cur = _smooth_opt(X_cur, 220, 60, 600, 0.006, 0.0008, rng, 0.0, 0.0,
                                        deadline=branch_deadline)
                    # short LP polish
                    X_cur = _lp_polish(X_cur, seconds=1.0, deadline=branch_deadline, r0=0.004, rmax=0.03)
                # final polish of full 162
                if X_cur.shape[0] == 162 and time.time() < branch_deadline - 2:
                    X_cur = _smooth_opt(X_cur, 250, 120, 1000, 0.005, 0.0006, rng, 0.0, 0.0,
                                        deadline=branch_deadline)
                    if time.time() < branch_deadline - 1.5:
                        X_cur = _lp_polish(X_cur, seconds=min(3.0, max(0.5, branch_deadline - time.time() - 0.5)),
                                           deadline=branch_deadline, r0=0.003, rmax=0.03)
                    s_cur = _max_cos(X_cur)
                    if np.isfinite(s_cur) and s_cur < s_best and X_cur.shape == (162, 8):
                        X_best = _norm(X_cur).copy()
                        s_best = float(s_cur)
            except Exception:
                pass
            branch += 1
        # final safety polish of best if time remains (very short)
        try:
            if time.time() < deadline - 6:
                X_try = _lp_polish(X_best, seconds=min(4.0, deadline - time.time() - 1.0), deadline=deadline)
                s_try = _max_cos(X_try)
                if np.isfinite(s_try) and s_try < s_best:
                    X_best = X_try
                    s_best = s_try
        except Exception:
            pass
    except Exception:
        pass
    X_best = _norm(np.asarray(X_best, dtype=float))
    # remove any NaN
    X_best = np.where(np.isfinite(X_best), X_best, 0.0)
    nr = np.linalg.norm(X_best, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-12)
    X_best = X_best / nr
    return X_best
