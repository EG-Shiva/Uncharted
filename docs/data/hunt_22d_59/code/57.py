import time
import numpy as np

def _normed(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return A / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _greedy_select(pool, k, start):
    P = pool.shape[0]
    start = int(start % P)
    selected = [start]
    selmask = np.zeros(P, dtype=bool)
    selmask[start] = True
    cur = pool @ pool[start]
    cur[start] = np.inf
    for _ in range(1, k):
        best = -1
        bestv = np.inf
        # argmin over unselected
        # manual loop via numpy
        idx = int(np.argmin(np.where(selmask, np.inf, cur)))
        selected.append(idx)
        selmask[idx] = True
        if len(selected) >= k:
            break
        v = pool @ pool[idx]
        # update max
        cur = np.maximum(cur, v)
        cur[selmask] = np.inf
    return pool[np.array(selected, dtype=int)].copy()

def _adam_steps(X0, steps, lr0, lr1, kind, p0=4.0, p1=4.0, b0=30.0, b1=250.0):
    X = X0.copy()
    n, d = X.shape
    M = np.zeros_like(X)
    V = np.zeros_like(X)
    b1a, b2a, eps = 0.9, 0.999, 1e-8
    best_val = _max_cos(X)
    best_X = X.copy()
    ll0, ll1 = np.log(lr0), np.log(lr1)
    lb0, lb1 = np.log(b0), np.log(b1)
    lp0, lp1 = np.log(p0), np.log(p1)
    bt1 = 1.0
    bt2 = 1.0
    for s in range(steps):
        frac = s / float(max(1, steps - 1))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        if kind == 'riesz':
            pp = float(np.exp(lp0 + (lp1 - lp0) * frac))
            # weights ((1+G)/2)^(p-1), clip G
            Q = (1.0 + G) * 0.5
            np.clip(Q, 0.0, 1.0, out=Q)
            np.fill_diagonal(Q, 0.0)
            # Q^(p-1); for large p avoid underflow: fine
            W = np.power(Q, pp - 1.0, out=Q)
            np.fill_diagonal(W, 0.0)
            Gr = (W @ X) / float(n)
        else:
            beta = float(np.exp(lb0 + (lb1 - lb0) * frac))
            m = float(G.max())
            W = np.exp(beta * (G - m))
            np.fill_diagonal(W, 0.0)
            S = float(W.sum() * 0.5)
            if S < 1e-30:
                continue
            Gr = (W @ X) / S
        dot = (Gr * X).sum(axis=1, keepdims=True)
        Gt = Gr - dot * X
        bt1 *= b1a
        bt2 *= b2a
        M *= b1a
        M += (1.0 - b1a) * Gt
        V *= b2a
        V += (1.0 - b2a) * (Gt * Gt)
        Mh = M / (1.0 - bt1)
        Vh = V / (1.0 - bt2)
        X = X - lr * Mh / (np.sqrt(Vh) + eps)
        X = _normed(X)
        if (s % 100 == 99) or (s == steps - 1):
            v = _max_cos(X)
            if v < best_val:
                best_val = v
                best_X = X.copy()
    return best_X, best_val

def _hard_polish(X0, steps, lr0, lr1, m0, m1):
    X = X0.copy()
    vel = np.zeros_like(X)
    mu = 0.85
    best_val = _max_cos(X)
    best_X = X.copy()
    ll0, ll1 = np.log(lr0), np.log(lr1)
    lm0, lm1 = np.log(m0), np.log(m1)
    for s in range(steps):
        frac = s / float(max(1, steps - 1))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        marg = float(np.exp(lm0 + (lm1 - lm0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = (G >= (m - marg)).astype(float)
        deg = W.sum(axis=1, keepdims=True)
        Gr = (W @ X) / np.maximum(deg, 1.0)
        zm = (deg[:, 0] == 0)
        if np.any(zm):
            Gr[zm, :] = 0.0
        dot = (Gr * X).sum(axis=1, keepdims=True)
        Gt = Gr - dot * X
        vel = mu * vel + (1.0 - mu) * Gt
        X = X - lr * vel
        X = _normed(X)
        if (s % 200 == 199) or (s == steps - 1):
            v = _max_cos(X)
            if v < best_val:
                best_val = v
                best_X = X.copy()
    return best_X, best_val

def solve():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    # fallback record
    fb_X = None
    fb_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (59, 22) and np.all(np.isfinite(a)):
                a = _normed(a)
                v = _max_cos(a)
                if v < fb_val:
                    fb_val = v
                    fb_X = a.copy()
        except Exception:
            pass
    if fb_X is None:
        fb_X = _normed(rng.standard_normal((59, 22)))
        fb_val = _max_cos(fb_X)

    def tl(limit=158.0):
        return (time.time() - t0) < limit

    # ---- build pools ----
    # D22 minimal vectors (924)
    D = np.zeros((924, 22), dtype=float)
    k = 0
    s2 = 1.0 / np.sqrt(2.0)
    for i in range(22):
        for j in range(i + 1, 22):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    D[k, i] = si * s2
                    D[k, j] = sj * s2
                    k += 1
    # random spherical pool
    R = _normed(rng.standard_normal((4000, 22)))
    # signed coords + pair sums pool
    C = np.zeros((44, 22), dtype=float)
    for i in range(22):
        C[2 * i, i] = 1.0
        C[2 * i + 1, i] = -1.0
    # pair sums (random subset of pairs)
    P2 = _normed(rng.standard_normal((1500, 22)))
    Mpool = np.vstack([C, P2])

    seeds = []
    try:
        seeds.append(_greedy_select(D, 59, 0))
        seeds.append(_greedy_select(D, 59, 311))
        seeds.append(_greedy_select(R, 59, 0))
        seeds.append(_greedy_select(R, 59, 1723))
        seeds.append(_greedy_select(Mpool, 59, 5))
        seeds.append(_normed(rng.standard_normal((59, 22))))
    except Exception:
        pass

    results = []
    # optimise each seed with Riesz annealing then LSE Adam
    for si, S0 in enumerate(seeds):
        if not tl(120.0):
            break
        try:
            X = S0.copy()
            X, _ = _adam_steps(X, steps=500, lr0=0.03, lr1=0.008, kind='riesz', p0=4, p1=4)
            X, _ = _adam_steps(X, steps=500, lr0=0.03, lr1=0.008, kind='riesz', p0=16, p1=16)
            X, _ = _adam_steps(X, steps=600, lr0=0.02, lr1=0.004, kind='riesz', p0=64, p1=64)
            X, v = _adam_steps(X, steps=1200, lr0=0.012, lr1=0.001, kind='lse', b0=30, b1=250)
            # contact sparsity metric
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            nc = int(((G >= m - 1e-3).sum()) // 2)
            results.append((v, nc, X))
        except Exception:
            continue

    # pick best 2-3 structurally distinct (sort by value, enforce diversity in nc)
    results.sort(key=lambda t: t[0])
    picked = []
    for v, nc, X in results:
        ok = True
        for _, nc2, _ in picked:
            if abs(nc - nc2) < 5 and abs(v - picked[0][0]) < 2e-4 and len(picked) >= 1:
                # allow at most similar; skip if too similar and we already have 2
                pass
        picked.append((v, nc, X))
        if len(picked) >= 3:
            break

    best_X = fb_X.copy()
    best_val = fb_val

    # polish picked seeds harder
    for _, _, X0p in picked:
        if not tl(140.0):
            break
        try:
            Xp, vp = _adam_steps(X0p, steps=1200, lr0=0.006, lr1=0.0006, kind='lse', b0=150, b1=500)
            Xp, vp2 = _hard_polish(Xp, steps=800, lr0=0.003, lr1=0.0002, m0=0.004, m1=0.0005)
            vfh = min(vp, vp2)
            # vp2 corresponds to Xp after hard; recompute
            vh = _max_cos(Xp)
            if vh < best_val:
                best_val = vh
                best_X = Xp.copy()
        except Exception:
            continue

    # final fallback polish of record to guarantee competitiveness
    if tl(158.0):
        try:
            Xf, vf = _adam_steps(best_X, steps=1500, lr0=0.006, lr1=0.0005, kind='lse', b0=60, b1=450)
            if vf < best_val:
                best_val = vf
                best_X = Xf.copy()
            Xf2, vf2 = _hard_polish(best_X, steps=1000, lr0=0.0015, lr1=0.00008, m0=0.002, m1=0.00035)
            vh2 = _max_cos(Xf2)
            if vh2 < best_val:
                best_val = vh2
                best_X = Xf2.copy()
        except Exception:
            pass

    best_X = _normed(best_X)
    return best_X
