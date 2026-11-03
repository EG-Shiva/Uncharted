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

def _contacts(X, thr=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    C = (G > m - thr).sum(axis=1)
    return m, C, G

def _anneal(X, rng, iters=700, b0=40.0, b1=500.0, lr0=0.008, lr1=0.0006,
            n0=0.0025, n1=0.0, b1m=0.9, b2m=0.999, eps=1e-8):
    n, d = X.shape
    Mt = np.zeros_like(X)
    Vt = np.zeros_like(X)
    for t in range(iters):
        f = t / max(1, iters - 1)
        beta = b0 * ((b1 / b0) ** f)
        lr = lr0 * ((lr1 / lr0) ** f)
        if n0 <= 0:
            nm = 0.0
        else:
            if n1 <= 0:
                nm = n0 * (1.0 - f)
            else:
                nm = n0 * ((n1 / n0) ** f)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        Gm = float(G.max())
        W = np.exp(beta * (G - Gm))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum())
        if not np.isfinite(S) or S <= 0:
            break
        H = W @ X
        WG = (W * G).sum(axis=1)
        Grad = (H - WG[:, None] * X) / S
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        Mt = b1m * Mt + (1 - b1m) * Gt
        Vt = b2m * Vt + (1 - b2m) * Gt * Gt
        mhat = Mt / (1 - b1m ** (t + 1))
        vhat = Vt / (1 - b2m ** (t + 1))
        step = lr * mhat / (np.sqrt(vhat) + eps)
        X = X - step
        if nm > 0:
            N = rng.randn(n, d)
            nd = (N * X).sum(axis=1, keepdims=True)
            N = N - nd * X
            X = X + nm * N
        X = _norm(X)
    return X

def _lp_polish(X, seconds, deadline=None):
    X = _norm(np.array(X, dtype=float))
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    iuI = iu0[0]; iuJ = iu0[1]
    ar = np.arange(d)
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
        if cnt > 3200:
            idx = np.argpartition(-cf, 3200)[:3200]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
        I = iuI[sel]; J = iuJ[sel]
        c = cf[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar,
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
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
            X = y; best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _holes(X_keep, k, rng, n_samp=16000, div=0.55):
    d = X_keep.shape[1]
    C = _norm(rng.randn(n_samp, d))
    S = C @ X_keep.T
    sc = S.max(axis=1)
    order = np.argsort(sc)
    picked = []
    for idx in order:
        if len(picked) >= k:
            break
        v = C[idx]
        ok = True
        for q in picked:
            if float(v @ q) > div:
                ok = False; break
        if ok:
            picked.append(v.copy())
    while len(picked) < k:
        v = rng.randn(d); v /= np.linalg.norm(v)
        picked.append(v)
    return np.array(picked)

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.RandomState(12345)
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
    X_best = Xb.copy(); s_best = float(sb)
    n, d = X_best.shape
    trial = 0
    # quick record of bases for diversity
    archive = [X_best.copy()]
    while time.time() < deadline - 30.0 and trial < 2000:
        trial += 1
        # pick base: mostly best, sometimes archive
        if trial % 11 == 0 and len(archive) > 1:
            base = archive[rng.randint(len(archive))].copy()
        else:
            base = X_best.copy()
        m0, C0, G0 = _contacts(base)
        order_worst = np.argsort(-C0)
        order_best = np.argsort(C0)
        mode = trial % 7
        Xt = base.copy()
        if mode == 0:
            Xt = _norm(Xt + rng.randn(n, d) * 0.008)
            Xt = _anneal(Xt, rng, iters=500, b0=30, b1=400, lr0=0.008, lr1=0.0006, n0=0.002, n1=0.0)
            Xt = _anneal(Xt, rng, iters=300, b0=150, b1=900, lr0=0.004, lr1=0.0004, n0=0.0, n1=0.0)
        elif mode == 1:
            Xt = _norm(Xt + rng.randn(n, d) * 0.028)
            Xt = _anneal(Xt, rng, iters=550, b0=25, b1=350, lr0=0.010, lr1=0.0008, n0=0.003, n1=0.0)
            Xt = _anneal(Xt, rng, iters=300, b0=150, b1=900, lr0=0.005, lr1=0.0004, n0=0.0, n1=0.0)
        elif mode == 2:
            # overloaded cluster shake
            ci = int(order_worst[0])
            dists = (base @ base[ci])
            nn = np.argsort(-dists)[:16]
            sig = np.full(n, 0.005)
            sig[nn] = 0.055
            sig[ci] = 0.08
            Xt = _norm(Xt + rng.randn(n, d) * sig[:, None])
            Xt = _anneal(Xt, rng, iters=550, b0=30, b1=400, lr0=0.009, lr1=0.0007, n0=0.0025, n1=0.0)
            Xt = _anneal(Xt, rng, iters=300, b0=150, b1=1000, lr0=0.004, lr1=0.0004, n0=0.0, n1=0.0)
        elif mode == 3:
            # remove 1 worst + move rattlers to holes
            ratt = np.where(C0 <= 3)[0]
            rem = [int(order_worst[0])]
            for r in ratt[:2]:
                if int(r) not in rem:
                    rem.append(int(r))
            rem = np.array(rem, dtype=int)
            keep = np.ones(n, dtype=bool); keep[rem] = False
            Xk = base[keep]
            nh = _holes(Xk, len(rem), rng)
            Xt[keep] = Xk
            Xt[rem] = nh
            # small shake rest
            Xt = _norm(Xt + rng.randn(n, d) * 0.004)
            Xt = _anneal(Xt, rng, iters=550, b0=30, b1=450, lr0=0.009, lr1=0.0007, n0=0.002, n1=0.0)
            Xt = _anneal(Xt, rng, iters=350, b0=150, b1=1000, lr0=0.004, lr1=0.0004, n0=0.0, n1=0.0)
        elif mode == 4:
            k = 2 + (trial % 2)
            rem = np.array([int(x) for x in order_worst[:k]], dtype=int)
            keep = np.ones(n, dtype=bool); keep[rem] = False
            Xk = base[keep]
            nh = _holes(Xk, k, rng, n_samp=20000, div=0.5)
            Xt[keep] = Xk
            Xt[rem] = nh
            Xt = _anneal(Xt, rng, iters=600, b0=25, b1=400, lr0=0.009, lr1=0.0007, n0=0.0025, n1=0.0)
            Xt = _anneal(Xt, rng, iters=350, b0=150, b1=1000, lr0=0.004, lr1=0.0004, n0=0.0, n1=0.0)
        elif mode == 5:
            ratt = np.where(C0 <= 4)[0]
            if len(ratt) == 0:
                ratt = order_best[:1]
            keep = np.ones(n, dtype=bool); keep[ratt[:1]] = False
            Xk = base[keep]
            nh = _holes(Xk, 1, rng)
            Xt = base.copy(); Xt[ratt[0]] = nh[0]
            Xt = _norm(Xt + rng.randn(n, d) * 0.003)
            Xt = _anneal(Xt, rng, iters=450, b0=40, b1=500, lr0=0.007, lr1=0.0005, n0=0.0015, n1=0.0)
            Xt = _anneal(Xt, rng, iters=300, b0=180, b1=1100, lr0=0.003, lr1=0.0003, n0=0.0, n1=0.0)
        else:
            # random-subspace rotation jitter + global shake
            A = rng.randn(d, d) * 0.02
            A = A - A.T
            # exp approx I+A
            R = np.eye(d) + A
            # orthonormalize via QR
            Q, _ = np.linalg.qr(R)
            Xt = _norm((Xt @ Q.T) + rng.randn(n, d) * 0.012)
            Xt = _anneal(Xt, rng, iters=550, b0=25, b1=380, lr0=0.010, lr1=0.0007, n0=0.003, n1=0.0)
            Xt = _anneal(Xt, rng, iters=300, b0=150, b1=900, lr0=0.004, lr1=0.0004, n0=0.0, n1=0.0)
        s_smooth = _max_cos(Xt)
        # decide LP budget
        if s_smooth < s_best + 0.004 and time.time() < deadline - 28:
            # short LP
            Xt = _lp_polish(Xt, seconds=0.9, deadline=deadline - 26)
            s_lp = _max_cos(Xt)
        else:
            s_lp = s_smooth
        if s_lp < s_best - 1e-12:
            X_best = _norm(Xt.copy()); s_best = float(s_lp)
            archive.append(X_best.copy())
            if len(archive) > 6:
                archive.pop(0)
        elif s_lp < s_best + 0.0015 and rng.rand() < 0.05:
            # keep diverse near-best for later hopping
            if len(archive) < 6:
                archive.append(_norm(Xt.copy()))
            else:
                archive[rng.randint(len(archive))] = _norm(Xt.copy())
        if time.time() > deadline - 30:
            break
    # final polish of best with remaining time
    remain = deadline - time.time() - 1.0
    if remain > 2:
        X_best = _lp_polish(X_best, seconds=min(remain, 25.0), deadline=deadline)
    X_best = _norm(X_best)
    return X_best
