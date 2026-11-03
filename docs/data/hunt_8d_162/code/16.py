import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n


def _smooth(X, steps, b0, b1, lr0, lr1, noise0, noise1, rng, mu=0.8):
    n, d = X.shape
    V = np.zeros_like(X)
    for t in range(steps):
        f = t / max(1, steps - 1)
        if b1 == b0:
            beta = b0
        else:
            beta = b0 * ((b1 / b0) ** f)
        if lr1 == lr0:
            lr = lr0
        else:
            lr = lr0 * ((lr1 / lr0) ** f)
        if noise0 <= 0:
            nm = 0.0
        else:
            if noise1 <= 0:
                nm = noise0 * (1.0 - f)
            else:
                nm = noise0 * ((noise1 / noise0) ** f) if noise1 > 0 else noise0 * (1.0 - f)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum())
        if not np.isfinite(S) or S <= 0:
            break
        H = W @ X
        WG = (W * G).sum(axis=1)
        Grad = (H - WG[:, None] * X) / S
        V = mu * V + Grad
        vn = float(np.linalg.norm(V, axis=1).max())
        if vn > 1e-14:
            X = X - (lr / vn) * V
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
    radius = 2e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    iuI = iu0[0]
    iuJ = iu0[1]
    while time.time() - start < seconds:
        if t_deadline is not None and time.time() > t_deadline:
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
            # keep tightest 3000
            idx = np.argpartition(-cf, 3000)[:3000]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
            cnt = 3000
        I = iuI[sel]
        J = iuJ[sel]
        c = cf[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
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
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best - 1e-15:
            X = y
            best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return X


def _find_hole(X_keep, rng, n_cand=20000):
    d = X_keep.shape[1]
    C = _norm(rng.randn(n_cand, d))
    dots = C @ X_keep.T
    m = dots.max(axis=1)
    idx = int(np.argmin(m))
    return C[idx]


def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.RandomState(0)
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
        X_best = _norm(rng.randn(162, 8))
        s_best = _max_cos(X_best)
    n, d = (162, 8)
    # Phase A: smooth basin hopping, collect candidates
    cands = []  # list of (score, X)
    tA_end = t0 + 105.0
    trial = 0
    while time.time() < min(tA_end, deadline - 60):
        trial += 1
        op = trial % 6
        Xt = X_best.copy()
        try:
            if op == 0:  # small global shake
                sigma = 0.012 + 0.008 * rng.rand()
                Xt = _norm(Xt + rng.randn(n, d) * sigma)
                Xt = _smooth(Xt, 450, 40, 400, 0.014, 0.002, 0.002, 0.0, rng)
                Xt = _smooth(Xt, 300, 150, 900, 0.005, 0.0008, 0.0, 0.0, rng)
            elif op == 1:  # medium global shake
                sigma = 0.030 + 0.020 * rng.rand()
                Xt = _norm(Xt + rng.randn(n, d) * sigma)
                Xt = _smooth(Xt, 500, 30, 350, 0.016, 0.002, 0.003, 0.0, rng)
                Xt = _smooth(Xt, 300, 150, 900, 0.005, 0.0008, 0.0, 0.0, rng)
            elif op == 2:  # remove worst k and reinsert at holes
                G = Xt @ Xt.T
                np.fill_diagonal(G, -2.0)
                rowmax = G.max(axis=1)
                k = int(rng.choice([1, 2, 2, 3]))
                worst = np.argsort(-rowmax)[:k]
                keep_mask = np.ones(n, dtype=bool)
                keep_mask[worst] = False
                X_keep = Xt[keep_mask]
                new_pts = []
                cur_keep = X_keep
                for _ in range(k):
                    h = _find_hole(cur_keep, rng, n_cand=20000)
                    # add jitter to avoid exact duplicate sampling bias
                    h = _norm(h + rng.randn(d) * 0.002)
                    new_pts.append(h)
                    cur_keep = np.vstack([cur_keep, h[None, :]])
                Xt[worst] = np.array(new_pts)
                Xt = _norm(Xt)
                Xt = _smooth(Xt, 500, 40, 500, 0.014, 0.0015, 0.0015, 0.0, rng)
                Xt = _smooth(Xt, 300, 200, 1000, 0.004, 0.0007, 0.0, 0.0, rng)
            elif op == 3:  # relocate low-contact / rattler points to holes
                G = Xt @ Xt.T
                np.fill_diagonal(G, -2.0)
                m = float(G.max())
                rowmax = G.max(axis=1)
                # contacts with tight tol for rattler sense
                low = np.where(rowmax < m - 0.004)[0]
                if len(low) == 0:
                    low = np.argsort(rowmax)[:2]
                k = min(len(low), int(rng.choice([1, 1, 2])))
                picks = rng.choice(low, size=k, replace=False) if len(low) >= k else low
                keep_mask = np.ones(n, dtype=bool)
                keep_mask[picks] = False
                X_keep = Xt[keep_mask]
                cur_keep = X_keep
                new_pts = []
                for _ in range(len(picks)):
                    h = _find_hole(cur_keep, rng, n_cand=20000)
                    h = _norm(h + rng.randn(d) * 0.002)
                    new_pts.append(h)
                    cur_keep = np.vstack([cur_keep, h[None, :]])
                for pi, h in zip(picks, new_pts):
                    Xt[pi] = h
                Xt = _norm(Xt)
                Xt = _smooth(Xt, 450, 40, 500, 0.013, 0.0015, 0.001, 0.0, rng)
                Xt = _smooth(Xt, 300, 200, 1000, 0.004, 0.0007, 0.0, 0.0, rng)
            elif op == 4:  # overload cluster shake
                G = Xt @ Xt.T
                np.fill_diagonal(G, -2.0)
                m = float(G.max())
                cnt = (G > m - 0.01).sum(axis=1)
                order = np.argsort(-cnt)
                center = int(order[rng.randint(0, 3)])
                dists = G[center]
                neigh = np.argsort(-dists)[:12]
                sigma_c = 0.05 + 0.03 * rng.rand()
                sigma_n = 0.03 + 0.02 * rng.rand()
                Xt[center] = _norm((Xt[center] + rng.randn(d) * sigma_c)[None, :])[0]
                for nb in neigh[1:]:
                    if rng.rand() < 0.7:
                        Xt[nb] = Xt[nb] + rng.randn(d) * sigma_n
                Xt = _norm(Xt)
                Xt = _smooth(Xt, 500, 35, 450, 0.015, 0.0018, 0.002, 0.0, rng)
                Xt = _smooth(Xt, 300, 180, 950, 0.0045, 0.0008, 0.0, 0.0, rng)
            else:  # contact-weighted shake
                G = Xt @ Xt.T
                np.fill_diagonal(G, -2.0)
                m = float(G.max())
                cnt = (G > m - 0.008).sum(axis=1).astype(float)
                sig = 0.008 + 0.05 * (cnt / max(1.0, cnt.max()))
                Xt = _norm(Xt + rng.randn(n, d) * sig[:, None])
                Xt = _smooth(Xt, 450, 45, 450, 0.014, 0.0018, 0.0015, 0.0, rng)
                Xt = _smooth(Xt, 280, 200, 1000, 0.004, 0.0007, 0.0, 0.0, rng)
        except Exception:
            continue
        try:
            s = _max_cos(Xt)
        except Exception:
            continue
        # keep top candidates
        if len(cands) < 8:
            cands.append((s, Xt.copy()))
            cands.sort(key=lambda z: z[0])
        else:
            if s < cands[-1][0] - 1e-12:
                cands[-1] = (s, Xt.copy())
                cands.sort(key=lambda z: z[0])
        # immediate promotion if smooth already beats best -> quick LP to lock it
        if s < s_best - 2e-6:
            Xt2 = _lp_polish(Xt, seconds=2.0, t_deadline=deadline - 25)
            s2 = _max_cos(Xt2)
            if s2 < s_best:
                s_best = s2
                X_best = Xt2.copy()
                cands.insert(0, (s2, Xt2.copy()))
                cands = sorted(cands, key=lambda z: z[0])[:8]
        if time.time() > min(tA_end, deadline - 60):
            break
    # Phase B: LP polish most promising smooth candidates
    cands = sorted(cands, key=lambda z: z[0])[:6]
    # always include current best
    cands.append((s_best, X_best.copy()))
    # dedup sort
    cands.sort(key=lambda z: z[0])
    seen_best = s_best
    for si, Xi in list(cands):
        if time.time() > deadline - 22:
            break
        # only polish if within window of best
        if si > seen_best + 0.006:
            continue
        Xi2 = _lp_polish(Xi, seconds=6.0, t_deadline=deadline - 18)
        s2 = _max_cos(Xi2)
        if s2 < s_best:
            s_best = s2
            X_best = Xi2.copy()
            seen_best = min(seen_best, s2)
    # Phase C: final long polish
    remain = deadline - time.time() - 1.0
    if remain > 3:
        X_best = _lp_polish(X_best, seconds=min(remain, 18.0), t_deadline=deadline - 0.5)
        s_best = _max_cos(X_best)
    X_best = _norm(X_best)
    return X_best
