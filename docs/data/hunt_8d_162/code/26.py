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


def _smooth(X, rng, steps, beta0, beta1, lr0, lr1, noise0, noise1, mu=0.85):
    n, d = X.shape
    V = np.zeros_like(X)
    for t in range(steps):
        f = t / max(1, steps - 1)
        if beta1 != beta0:
            beta = beta0 * ((beta1 / beta0) ** f)
        else:
            beta = beta0
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
        m = float(G.max())
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum())
        if not np.isfinite(S) or S <= 0:
            break
        H = W @ X
        WG = (W * G).sum(axis=1)
        Grad = (H - WG[:, None] * X) / S
        mx = float(np.linalg.norm(Grad, axis=1).max())
        if mx > 1e-14:
            Gn = Grad / mx
            V = mu * V + (1.0 - mu) * Gn
            dot = (V * X).sum(axis=1, keepdims=True)
            V = V - dot * X
            X = X - lr * V
        if nm > 0:
            N = rng.randn(n, d)
            nd = (N * X).sum(axis=1, keepdims=True)
            N = N - nd * X
            X = X + nm * N
        X = _norm(X)
    return X


def _lp_polish(X, seconds, r0=0.004):
    X = _norm(np.array(X, dtype=float))
    n, d = X.shape
    best = _max_cos(X)
    Xb = X.copy()
    radius = r0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    iuI = iu0[0]
    iuJ = iu0[1]
    while time.time() - start < seconds:
        if radius < 1e-13:
            break
        G = X @ X.T
        cfull = G[iuI, iuJ]
        m = float(cfull.max())
        if m < best:
            best = m
            Xb = X.copy()
        thresh = m - max(20.0 * radius, 1e-9)
        # select
        if cfull.size > 2500:
            # fast path: check count first via threshold
            sel = cfull > thresh
            cnt = int(sel.sum())
            if cnt == 0:
                radius *= 0.5
                continue
            if cnt > 2500:
                idx = np.argpartition(-cfull, 2500)[:2500]
                I = iuI[idx]
                J = iuJ[idx]
                c = cfull[idx]
            else:
                I = iuI[sel]
                J = iuJ[sel]
                c = cfull[sel]
        else:
            sel = cfull > thresh
            if not np.any(sel):
                radius *= 0.5
                continue
            I = iuI[sel]
            J = iuJ[sel]
            c = cfull[sel]
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
            obj = np.zeros(nv)
            obj[-1] = 1.0
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
            Xb = y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return Xb


def _random_hole(X_keep, rng, n_samp=15000):
    d = X_keep.shape[1]
    C = rng.randn(n_samp, d)
    C = _norm(C)
    dots = C @ X_keep.T
    m = dots.max(axis=1)
    idx = int(np.argmin(m))
    return C[idx]


def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline = t0 + BUDGET
    rng = np.random.RandomState(12345)
    n, d = 162, 8
    X_best = None
    s_best = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = _norm(v)
                s = _max_cos(v)
                if s < s_best:
                    s_best = s
                    X_best = v.copy()
        except Exception:
            pass
    if X_best is None:
        X_best = _norm(rng.randn(n, d))
        s_best = _max_cos(X_best)
    elites = [(s_best, X_best.copy())]
    trial = 0
    # exploration phase until ~135s
    explore_end = t0 + 138.0
    while time.time() < explore_end and time.time() < deadline - 30:
        trial += 1
        # choose base
        if len(elites) > 1 and rng.rand() < 0.3:
            bi = rng.randint(len(elites))
            Xb = elites[bi][1]
        else:
            Xb = X_best
        Xt = Xb.copy()
        mode = trial % 5
        # randomize a bit with rng to diversify
        if rng.rand() < 0.25:
            mode = int(rng.randint(5))
        if mode == 0:
            # single worst relocation
            G = Xt @ Xt.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            w = int(np.argmax(rowmax))
            X_keep = np.delete(Xt, w, axis=0)
            nv = _random_hole(X_keep, rng, n_samp=20000)
            Xt = np.vstack([X_keep, nv[None, :]])
            Xt = _smooth(Xt, rng, steps=550, beta0=40, beta1=800, lr0=0.014, lr1=0.002, noise0=0.0022, noise1=0.0)
        elif mode == 1:
            # multi worst relocation
            k = int(rng.choice(np.array([3, 4, 5])))
            G = Xt @ Xt.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            worst = np.argsort(-rowmax)[:k]
            mask = np.ones(n, dtype=bool)
            mask[worst] = False
            X_keep = Xt[mask]
            cur = X_keep.copy()
            news = []
            for _ in range(k):
                nv = _random_hole(cur, rng, n_samp=15000)
                news.append(nv)
                cur = np.vstack([cur, nv[None, :]])
            Xt = np.vstack([X_keep, np.array(news)])
            # shuffle order to avoid bias (deterministic via rng)
            perm = rng.permutation(n)
            Xt = Xt[perm]
            Xt = _smooth(Xt, rng, steps=700, beta0=30, beta1=900, lr0=0.016, lr1=0.0018, noise0=0.0028, noise1=0.0)
        elif mode == 2:
            sigma = float(rng.uniform(0.012, 0.038))
            Xt = _norm(Xt + rng.randn(n, d) * sigma)
            Xt = _smooth(Xt, rng, steps=600, beta0=50, beta1=1000, lr0=0.013, lr1=0.0015, noise0=0.0018, noise1=0.0)
        elif mode == 3:
            u = rng.randn(d)
            u = u / max(1e-12, np.linalg.norm(u))
            proj = Xt @ u
            msz = int(rng.randint(8, 26))
            if rng.rand() < 0.5:
                idx = np.argsort(-proj)[:msz]
            else:
                idx = np.argsort(proj)[:msz]
            sigma = float(rng.uniform(0.05, 0.16))
            Xt[idx] = Xt[idx] + rng.randn(msz, d) * sigma
            Xt = _norm(Xt)
            Xt = _smooth(Xt, rng, steps=600, beta0=35, beta1=850, lr0=0.015, lr1=0.0018, noise0=0.0022, noise1=0.0)
        else:
            sigma = 0.006
            Xt = _norm(Xt + rng.randn(n, d) * sigma)
            Xt = _smooth(Xt, rng, steps=450, beta0=80, beta1=1200, lr0=0.008, lr1=0.0012, noise0=0.0008, noise1=0.0)
            Xt = _smooth(Xt, rng, steps=300, beta0=200, beta1=1500, lr0=0.005, lr1=0.0008, noise0=0.0, noise1=0.0)
        try:
            s = _max_cos(Xt)
        except Exception:
            continue
        if not np.isfinite(s):
            continue
        if s < s_best - 1e-12:
            # direct improvement without LP (rare)
            X_best = Xt.copy()
            s_best = s
            elites.append((s, Xt.copy()))
            elites = sorted(elites, key=lambda z: z[0])[:5]
            continue
        # short LP if promising
        if s < s_best + 0.007:
            remain = deadline - time.time()
            if remain < 8:
                break
            secs = 1.1 if (s_best - s) < 0.002 else 1.8
            secs = min(secs, max(0.4, remain - 28))
            try:
                Xp = _lp_polish(Xt, seconds=secs, r0=0.004)
                sp = _max_cos(Xp)
            except Exception:
                continue
            if sp < s_best - 1e-12:
                X_best = Xp.copy()
                s_best = sp
                elites.append((sp, Xp.copy()))
                elites = sorted(elites, key=lambda z: z[0])[:5]
            elif sp < elites[-1][0] - 1e-9 and sp < s_best + 0.002:
                elites.append((sp, Xp.copy()))
                elites = sorted(elites, key=lambda z: z[0])[:5]
    # final polish of best and elites
    try:
        remain = deadline - time.time() - 1.0
        if remain > 3:
            # polish best longest
            share = min(22.0, remain * 0.6)
            X_best = _lp_polish(X_best, seconds=share, r0=0.003)
            s_best = _max_cos(X_best)
            # polish runner-ups briefly if time left
            remain2 = deadline - time.time() - 1.0
            for k in range(1, min(len(elites), 3)):
                if remain2 < 4:
                    break
                try:
                    Xc = _lp_polish(elites[k][1], seconds=min(4.0, remain2 / 2.0), r0=0.003)
                    sc = _max_cos(Xc)
                    if sc < s_best:
                        X_best = Xc
                        s_best = sc
                    remain2 = deadline - time.time() - 1.0
                except Exception:
                    continue
    except Exception:
        pass
    X_best = _norm(np.array(X_best, dtype=float))
    # safety: replace any bad rows
    norms = np.linalg.norm(X_best, axis=1)
    for i in range(n):
        if not np.isfinite(norms[i]) or norms[i] < 1e-9:
            v = rng.randn(d)
            X_best[i] = v / np.linalg.norm(v)
    X_best = _norm(X_best)
    return X_best
