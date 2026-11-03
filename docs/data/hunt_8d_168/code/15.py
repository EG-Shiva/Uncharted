import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _quench(X, steps, beta0, beta1, lr0, lr1):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    lb0 = np.log(beta0)
    lb1 = np.log(beta1)
    ll0 = np.log(lr0)
    ll1 = np.log(lr1)
    for t in range(1, steps + 1):
        frac = t / steps
        beta = np.exp(lb0 + (lb1 - lb0) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        Z = W.sum()
        if Z == 0:
            continue
        Grad = (W @ X) / Z
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        m *= b1
        m += (1.0 - b1) * Gt
        v *= b2
        v += (1.0 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        mhat = m / bc1
        vhat = v / bc2
        step = mhat / (np.sqrt(vhat) + eps)
        X = X - lr * step
        nrm = np.sqrt((X * X).sum(axis=1, keepdims=True))
        nrm[nrm == 0] = 1.0
        X /= nrm
    return X

def _refine_hole(Y, p, iters=300, beta=120.0, lr0=0.12, lr1=0.004):
    p = p / np.linalg.norm(p)
    m = np.zeros_like(p)
    vv = np.zeros_like(p)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    for t in range(1, iters + 1):
        frac = t / iters
        lr = lr0 * (lr1 / lr0) ** frac if lr0 > 0 and lr1 > 0 else lr0
        g = Y @ p
        gm = g.max()
        w = np.exp(beta * (g - gm))
        s = w.sum() + 1e-15
        A = (w @ Y) / s
        At = A - float(A @ p) * p
        m = b1 * m + (1 - b1) * At
        vv = b2 * vv + (1 - b2) * (At * At)
        mh = m / (1 - b1 ** t)
        vh = vv / (1 - b2 ** t)
        st = mh / (np.sqrt(vh) + eps)
        p = p - lr * st
        n = np.linalg.norm(p)
        if n == 0:
            break
        p /= n
    return p

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        mm = c.max()
        sel = c > mm - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    BUDGET = 174.0
    PHASE1 = 130.0
    n, d = 168, 8
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape != (n, d):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            va = _max_cos(a)
            if va < best_val:
                best_val = va
                best = a.copy()
        except Exception:
            pass
    if best is None:
        r0 = np.random.RandomState(0)
        best = r0.randn(n, d)
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val
    rng = np.random.RandomState(12345)
    it = 0
    deadline = t0 + PHASE1
    # temperature schedule for wandering
    T0, T1 = 0.0015, 0.00012
    est_total = 90.0
    while time.time() < deadline:
        if time.time() + 2.5 > t0 + BUDGET:
            break
        it += 1
        el = time.time() - t0
        T = T0 * (T1 / T0) ** min(el / est_total, 1.0)
        X = cur.copy()
        u = rng.rand()
        if u < 0.55:
            # jammed shake + rattler void reinsertion
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            gmax = rowmax.max()
            # rattlers: far from max
            ratt = np.where(rowmax < gmax - 5e-4)[0]
            # near-contact counts for jammed ranking
            cnt = (G > gmax - 0.004).sum(axis=1)
            # number to move
            k_r = int(rng.randint(1, 4))  # 1..3 rattlers
            k_j = int(rng.randint(2, 7))  # 2..6 jammed shake
            # pick jammed indices by count then rowmax
            order = np.argsort(cnt * 10.0 + rowmax)
            jam = order[-k_j:]
            sig = [0.02, 0.04, 0.08][rng.randint(3)]
            noise = np.zeros_like(X)
            noise[jam] = rng.randn(len(jam), d) * sig
            X = X + noise
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            # choose rattlers to reinsert
            if len(ratt) > 0:
                k_r = min(k_r, len(ratt))
                ridx = ratt[rng.choice(len(ratt), k_r, replace=False)] if len(ratt) > k_r else ratt
            else:
                ridx = rng.choice(n, k_r, replace=False)
                # avoid overlapping jam
                ridx = np.array([i for i in ridx if i not in jam][:k_r])
                if len(ridx) == 0:
                    ridx = rng.choice(n, 1, replace=False)
            fixed_mask = np.ones(n, dtype=bool)
            fixed_mask[ridx] = False
            # find voids w.r.t fixed + already placed
            placed = X[fixed_mask]
            for jj, j in enumerate(ridx):
                # sample probes
                P = rng.randn(16000, d)
                P /= np.linalg.norm(P, axis=1, keepdims=True)
                # chunk
                bs = np.empty(16000)
                CH = 4000
                for a in range(0, 16000, CH):
                    b = min(a + CH, 16000)
                    D = P[a:b] @ placed.T
                    bs[a:b] = D.max(axis=1)
                # pick diverse best among top
                top = np.argpartition(bs, 40)[:40]
                # choose one at random among top 8 to diversify
                cand_idx = top[np.argsort(bs[top])[:8]]
                p0 = P[cand_idx[rng.randint(len(cand_idx))]]
                p0 = _refine_hole(placed, p0, iters=250, beta=90.0 + rng.rand() * 60.0)
                X[j] = p0
                placed = np.vstack([placed, p0[None, :]])
            beta0 = 30.0 + rng.rand() * 30.0
            beta1 = 250.0 + rng.rand() * 350.0
            steps = 1100 + int(rng.randint(800))
            lr0 = 0.003 + rng.rand() * 0.005
            lr1 = lr0 * 0.1
            Xq = _quench(X, steps, beta0, beta1, lr0, lr1)
        elif u < 0.80:
            sig = [0.008, 0.015, 0.03, 0.06][rng.randint(4)]
            X = X + rng.randn(n, d) * sig
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            beta0 = 30.0 + rng.rand() * 30.0
            beta1 = 200.0 + rng.rand() * 400.0
            steps = 1100 + int(rng.randint(800))
            lr0 = 0.003 + rng.rand() * 0.005
            lr1 = lr0 * 0.1
            Xq = _quench(X, steps, beta0, beta1, lr0, lr1)
        else:
            c = rng.randn(d)
            c /= np.linalg.norm(c)
            dots = X @ c
            k = 8 + int(rng.randint(10))
            idx = np.argsort(dots)[-k:]
            sigma = 0.04 + rng.rand() * 0.06
            noise = np.zeros_like(X)
            noise[idx] = rng.randn(k, d) * sigma
            X = X + noise
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            Xq = _quench(X, 1400, 40.0, 400.0, 0.005, 0.0005)
        vq = _max_cos(Xq)
        if not np.all(np.isfinite(Xq)):
            continue
        # update best (with quick LP squeeze if promising new best)
        if vq + 1e-12 < best_val:
            # quick squeeze to true basin value if time allows
            if time.time() + 5.0 < t0 + BUDGET and (best_val - vq) > 2e-6:
                try:
                    Xqs = _lp_polish(Xq, 1.5)
                    vs = _max_cos(Xqs)
                    if vs < vq and np.all(np.isfinite(Xqs)):
                        Xq, vq = Xqs, vs
                except Exception:
                    pass
            best = Xq.copy()
            best_val = vq
            cur = Xq.copy()
            cur_val = vq
        elif vq < cur_val or rng.rand() < np.exp((cur_val - vq) / max(T, 1e-6)):
            cur = Xq.copy()
            cur_val = vq
        # occasional restart from best if wandering too far
        if cur_val > best_val + 0.008 and rng.rand() < 0.5:
            cur = best.copy()
            cur_val = best_val
    remaining = (t0 + BUDGET) - time.time()
    if remaining > 7.0:
        secs = min(remaining - 2.0, 40.0)
        try:
            polished = _lp_polish(best, secs)
            vp = _max_cos(polished)
            if vp < best_val and np.all(np.isfinite(polished)):
                best = polished
                best_val = vp
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    bad = ~np.all(np.isfinite(best), axis=1)
    if np.any(bad):
        rr = np.random.RandomState(1).randn(bad.sum(), d)
        rr /= np.linalg.norm(rr, axis=1, keepdims=True)
        best[bad] = rr
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
