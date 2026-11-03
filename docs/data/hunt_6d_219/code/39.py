import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def adam_anneal(X0, alpha, steps, lr, cap=0.05):
    X = X0.copy()
    m1 = np.zeros_like(X); m2 = np.zeros_like(X)
    b1, b2 = 0.9, 0.999
    eps = 1e-8
    t = 0
    for _ in range(steps):
        t += 1
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mmax = float(G.max())
        W = np.exp(alpha * (G - mmax))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            break
        grad = (W @ X) / max(S, 1e-300)
        dot = (grad * X).sum(axis=1, keepdims=True)
        gt = grad - dot * X
        m1 = b1 * m1 + (1 - b1) * gt
        m2 = b2 * m2 + (1 - b2) * (gt * gt)
        c1 = 1.0 - b1 ** t; c2 = 1.0 - b2 ** t
        step = (lr * (m1 / c1)) / (np.sqrt(m2 / c2) + eps)
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        fac = np.minimum(1.0, cap / np.maximum(sn, 1e-12))
        X = _norm_rows(X - step * fac)
    return X

def subset_anneal(X0, subset, alpha, steps, lr, cap=0.03):
    X = X0.copy()
    subset = np.asarray(subset, dtype=int)
    m1 = np.zeros((len(subset), X.shape[1])); m2 = np.zeros((len(subset), X.shape[1]))
    b1, b2 = 0.9, 0.999
    eps = 1e-8
    t = 0
    for _ in range(steps):
        t += 1
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mmax = float(G.max())
        W = np.exp(alpha * (G - mmax))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            break
        grad = (W @ X) / max(S, 1e-300)
        dot = (grad[subset] * X[subset]).sum(axis=1, keepdims=True)
        gt = grad[subset] - dot * X[subset]
        m1 = b1 * m1 + (1 - b1) * gt
        m2 = b2 * m2 + (1 - b2) * (gt * gt)
        c1 = 1.0 - b1 ** t; c2 = 1.0 - b2 ** t
        step = (lr * (m1 / c1)) / (np.sqrt(m2 / c2) + eps)
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        fac = np.minimum(1.0, cap / np.maximum(sn, 1e-12))
        X[subset] = _norm_rows(X[subset] - step * fac)
    return X

def refine_single(v0, Xcur, beta=600.0, steps=250, lr=0.05):
    v = v0 / max(np.linalg.norm(v0), 1e-12)
    for it in range(steps):
        d = Xcur @ v
        m = float(d.max())
        e = np.exp(beta * (d - m))
        S = float(e.sum())
        if not np.isfinite(S) or S <= 0:
            break
        w = e / max(S, 1e-300)
        g = w @ Xcur
        gt = g - float(g @ v) * v
        # decaying lr
        cur_lr = lr * (1.0 - 0.7 * it / max(steps, 1))
        v = v - cur_lr * gt
        n = np.linalg.norm(v)
        if not np.finite(n) or n < 1e-12:
            break
        v = v / n
    return v

def lp_polish(x, seconds, deadline=1e18):
    x = _norm_rows(x.copy())
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    bestX = x.copy()
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c = Gfull[iu0]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        if int(sel.sum()) < 1:
            break
        I = iu0[0][sel]; J = iu0[1][sel]; cs = c[sel]
        p = len(I)
        if p > 4000:
            k = 4000
            part = np.argpartition(-cs, k - 1)[:k]
            I, J, cs = I[part], J[part], cs[part]
            p = k
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
            obj = np.zeros(n * d + 1); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm_rows(x + res.x[:-1].reshape(n, d))
        v = max_cos(y)
        if v < best:
            x, best, bestX = y, v, y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return bestX

def solve():
    np.random.seed(0)
    rng = np.random.default_rng(0)
    t0 = time.time()
    LIMIT = 178.0
    SURG_END = t0 + 120.0
    n0, d = 219, 6
    best = None; bestv = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n0, d):
                a = _norm_rows(a)
                if np.all(np.isfinite(a)):
                    v = max_cos(a)
                    if v < bestv:
                        bestv = v; best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm_rows(rng.standard_normal((n0, d)))
        bestv = max_cos(best)
    incumbent = best.copy(); incv = bestv

    trial = 0
    # surgery loop
    while time.time() < SURG_END:
        remain_total = LIMIT - (time.time() - t0)
        if remain_total < 45:
            break
        trial += 1
        try:
            base = incumbent.copy()
            n = base.shape[0]
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            m0 = float(G.max())
            contact = G > m0 - 1e-9
            deg = contact.sum(axis=1)
            per = G.max(axis=1)
            order_deg = np.argsort(-deg)
            order_per = np.argsort(-per)
            rattlers = np.where(deg == 0)[0]
            k = int(rng.integers(12, 25))  # 12..24
            # pick seed among top-degree
            top_pool = order_deg[:max(20, n // 10)]
            seed = int(rng.choice(top_pool))
            # neighbours of seed sorted
            nbr_order = np.argsort(-G[seed])
            Dset = set()
            # include 1-2 rattlers if exist
            if len(rattlers) > 0:
                nr = min(len(rattlers), 1 + (trial % 2))
                for rr in rng.choice(rattlers, size=nr, replace=False):
                    Dset.add(int(rr))
            Dset.add(seed)
            for cand in nbr_order:
                if len(Dset) >= k:
                    break
                c = int(cand)
                if c == seed:
                    continue
                # prefer high-degree / high per-row max
                Dset.add(c)
                if len(Dset) >= k:
                    break
            # if still short (e.g. seed==rattler overlap), fill random from top-per
            while len(Dset) < k:
                c = int(rng.choice(order_per[:40]))
                Dset.add(c)
            # clustered variant every 3rd trial: take contiguous block in per-order
            if trial % 3 == 0:
                s0 = int(rng.integers(0, n - k))
                # mix: half cluster, half random already; force cluster block
                Dset = set(int(x) for x in order_per[s0:s0 + k])
                if len(rattlers) > 0 and trial % 2 == 0:
                    # ensure a rattler included
                    Dset.pop()
                    Dset.add(int(rattlers[0]))
            D = np.array(sorted(Dset), dtype=int)
            if len(D) != k:
                # adjust
                k = len(D)
            mask = np.ones(n, dtype=bool); mask[D] = False
            Xrem = base[mask].copy()  # (n-k, 6)
            # LP/soft-max re-optimise remaining until jam at lower cosine
            # two-stage anneal on core
            Xrem = adam_anneal(Xrem, 150, 200, 0.012)
            Xrem = adam_anneal(Xrem, 500, 200, 0.006)
            # quick check: if core already worse than incumbent, still continue (reinsert may fix)
            Xcur = Xrem.copy()
            # reinsert one at a time in random order
            ins_order = rng.permutation(len(D))
            ok = True
            for ii in ins_order:
                if time.time() > SURG_END + 8:
                    ok = False
                    break
                # sample ~50000 candidates
                C = rng.standard_normal((50000, d))
                C = _norm_rows(C)
                # scores = max dot with current set
                # chunk to avoid huge memory? single matmul 50000 x m (~200) is fine
                S = C @ Xcur.T
                # max over columns
                mx = S.max(axis=1)
                bi = int(np.argmin(mx))
                # take top-3 distinct best for refinement
                # get indices of 3 smallest mx
                if True:
                    part = np.argpartition(mx, 3)[:3]
                    best_v = None; best_s = 1e9
                    for pi in part:
                        v_init = C[int(pi)]
                        v_ref = refine_single(v_init, Xcur, beta=600.0, steps=200, lr=0.05)
                        s = float((Xcur @ v_ref).max())
                        if s < best_s:
                            best_s = s; best_v = v_ref
                    vnew = best_v
                else:
                    vnew = refine_single(C[bi], Xcur)
                # append
                Xcur = np.vstack([Xcur, vnew[None, :]])
                new_idx = Xcur.shape[0] - 1
                # 20 nearest neighbours (largest dots to new point)
                dots = Xcur[:-1] @ Xcur[-1]
                nn = min(20, Xcur.shape[0] - 1)
                nbrs = np.argsort(-dots)[:nn]
                subset = np.concatenate([[new_idx], nbrs])
                # L-BFGS only on inserted + neighbours (subset anneal surrogate)
                Xcur = subset_anneal(Xcur, subset, 500, 120, 0.008, cap=0.03)
                # short global soft-max polish (cheap surrogate for short global LP)
                # very short: 40 steps low lr
                Xcur = adam_anneal(Xcur, 700, 40, 0.003, cap=0.015)
                # no full LP per insertion to save time
            if not ok:
                continue
            # after all insertions, short global LP polish
            # ensure shape 219
            if Xcur.shape[0] != n0:
                continue
            # peak + anneal sharpen
            Xcur = adam_anneal(Xcur, 900, 150, 0.004, cap=0.02)
            vcur = max_cos(Xcur)
            # LP polish with small budget
            tb = min(6.0, max(1.5, LIMIT - (time.time() - t0) - 30.0))
            if tb > 1.0:
                Xp = lp_polish(Xcur, tb, deadline=t0 + LIMIT - 1.5)
                vp = max_cos(Xp)
                if vp < vcur:
                    Xcur, vcur = Xp, vp
            if vcur < incv - 1e-12 and np.all(np.isfinite(Xcur)):
                incumbent, incv = _norm_rows(Xcur), vcur
        except Exception:
            continue
    # final global polish of incumbent within remaining time
    remain = LIMIT - (time.time() - t0)
    if remain > 8:
        try:
            Xp = lp_polish(incumbent, min(remain - 2.0, 25.0), deadline=t0 + LIMIT - 1.5)
            vp = max_cos(Xp)
            if vp < incv:
                incumbent, incv = Xp, vp
        except Exception:
            pass
    # fallback: never worse than best loaded
    if incv > bestv + 1e-12:
        incumbent = best
    incumbent = _norm_rows(incumbent)
    return incumbent
