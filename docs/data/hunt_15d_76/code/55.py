import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _contacts(X, thr=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    C = (G >= m - thr).sum(axis=1)
    per = np.max(G, axis=1)
    return C, per, m, G

def _anneal(X, steps, T0, T1, lr0, lr1, t0, t_end, check=64):
    n, d = X.shape
    for k in range(steps):
        if (k & (check - 1)) == 0 and (time.time() - t0) > t_end:
            break
        f = k / max(1, steps - 1)
        T = T0 * (T1 / T0) ** f if T1 > 0 else T0
        lr = lr0 * (lr1 / lr0) ** f if lr1 > 0 else lr0
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E))
        if not np.isfinite(S) or S <= 0:
            break
        Gr = (E @ X) / S
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gr -= dot * X
        # limit per-point step
        gn = np.linalg.norm(Gr, axis=1, keepdims=True) + 1e-12
        # clip extreme
        scale = np.minimum(1.0, 1.0 / (gn * 5.0 + 1e-12) + 0.5)
        # simple: just lr step, with safety
        X = X - lr * Gr
        X = _norm(X)
    return X

def _exact_descent(X, steps, lr0, lr1, topk, t0, t_end):
    n, d = X.shape
    for k in range(steps):
        if (k & 31) == 0 and (time.time() - t0) > t_end:
            break
        f = k / max(1, steps - 1)
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        # find topk pairs
        # partial: flatten argpartition
        flat = G.ravel()
        # topk indices
        idx = np.argpartition(-flat, topk)[:topk]
        ii, jj = np.unravel_index(idx, G.shape)
        # gradient accumulation
        Gr = np.zeros_like(X)
        for a, b in zip(ii, jj):
            if a == b:
                continue
            # move a away from b, b away from a (tangent)
            dab = float(np.dot(X[a], X[b]))
            Gr[a] += X[b] - dab * X[a]
            Gr[b] += X[a] - dab * X[b]
        Gr /= max(1, topk)
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gr -= dot * X
        X = X - lr * Gr
        X = _norm(X)
    return X

def _refine_hole(v, Y, iters, T0, lr0, rng=None):
    v = v / (np.linalg.norm(v) + 1e-12)
    for k in range(iters):
        f = k / max(1, iters - 1)
        T = T0 * (0.15 ** f)
        lr = lr0 * (0.15 ** f)
        if T < 1e-5:
            T = 1e-5
        s = Y @ v
        m = float(np.max(s))
        w = np.exp((s - m) / T)
        sw = float(np.sum(w))
        if not np.isfinite(sw) or sw <= 0:
            break
        w /= sw
        g = w @ Y
        g = g - float(np.dot(g, v)) * v
        ng = float(np.linalg.norm(g))
        if ng < 1e-12:
            break
        # normalized direction scaled
        g = g / (ng + 1e-12) * min(1.0, ng * 4.0)
        v = v - lr * g
        nv = float(np.linalg.norm(v))
        if not np.isfinite(nv) or nv == 0:
            break
        v = v / nv
    return v

def _hole_insert(Xt, idx, rng, NS=2500, refine_iters=200):
    n, d = Xt.shape
    mask = np.ones(n, dtype=bool)
    mask[idx] = False
    Y = Xt[mask]
    C = rng.standard_normal(size=(NS, d))
    C = _norm(C)
    scores = (C @ Y.T).max(axis=1)
    bi = int(np.argmin(scores))
    v0 = C[bi].copy()
    v0 = _refine_hole(v0, Y, iters=refine_iters, T0=0.008, lr0=0.12)
    return v0

def solve():
    rng = np.random.default_rng(0)
    t0 = time.time()
    BUDGET = 168.0
    T_END = 160.0  # leave margin for final return
    try:
        Xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            Xb = np.load("record.npy").astype(float)
        except Exception:
            Xb = rng.standard_normal(size=(76, 15))
    Xb = np.asarray(Xb, dtype=float)
    if Xb.shape != (76, 15):
        Y = rng.standard_normal(size=(76, 15))
        Y = _norm(Y)
        try:
            m0 = min(76, Xb.shape[0]); m1 = min(15, Xb.shape[1])
            Y[:m0, :m1] = Xb[:m0, :m1]
            Y = _norm(Y)
        except Exception:
            pass
        Xb = Y
    Xb = _norm(Xb)
    bestX = Xb.copy()
    bestV = _maxcos(bestX)
    n, d = 76, 15

    # quick exact squeeze of incumbent (cheap)
    # trial 0: light anneal to ensure local squeeze
    if time.time() - t0 < T_END - 20:
        Y = bestX.copy()
        Y = _anneal(Y, steps=900, T0=0.004, T1=0.0006, lr0=1.2, lr1=0.10, t0=t0, t_end=T_END)
        Y = _exact_descent(Y, steps=400, lr0=0.02, lr1=0.002, topk=60, t0=t0, t_end=T_END)
        v = _maxcos(Y)
        if v < bestV:
            bestV, bestX = v, Y.copy()

    trial = 0
    sigmas = [0.006, 0.015, 0.035, 0.08, 0.16]
    while True:
        if time.time() - t0 > T_END - 18:
            break
        trial += 1
        if trial > 60:
            break
        try:
            Ccnt, per, mcur, Gcur = _contacts(bestX, thr=1e-7)
        except Exception:
            Ccnt = np.zeros(n); per = np.zeros(n)
        # decide k and mode
        mode = trial % 4
        # k cycles 1..3, occasionally 4-5 for large jumps
        if trial % 11 == 0:
            k_re = 5
        elif trial % 5 == 0:
            k_re = 4
        else:
            k_re = 1 + ((trial // 4) % 3)  # 1,2,3
        sigma = sigmas[(trial - 1) % len(sigmas)]
        # select indices to replace
        cand = []
        if mode == 0:
            # worst-pair surgery: take top pairs, collect endpoints
            try:
                Gu = Gcur.copy()
                # get worst pairs greedily
                for _ in range(k_re):
                    a, b = np.unravel_index(int(np.argmax(Gu)), Gu.shape)
                    # prefer the one with larger per-point max
                    if per[a] >= per[b]:
                        cand.append(int(a))
                    else:
                        cand.append(int(b))
                    Gu[a, :] = -2; Gu[:, a] = -2
                    Gu[b, :] = -2; Gu[:, b] = -2
            except Exception:
                cand = [int(rng.integers(0, n)) for _ in range(k_re)]
        elif mode == 1:
            # overloaded (high contacts, high per-point max)
            order = np.argsort(-(Ccnt.astype(float) * 2.0 + per * 10.0))
            for j in range(k_re):
                cand.append(int(order[(trial + j) % n]))
        elif mode == 2:
            # mix: one overloaded + random
            order = np.argsort(-Ccnt)
            cand.append(int(order[(trial) % n]))
            for _ in range(k_re - 1):
                cand.append(int(rng.integers(0, n)))
        else:
            # random + occasional rattler
            for _ in range(k_re):
                cand.append(int(rng.integers(0, n)))
        # dedup
        cand = list(dict.fromkeys(cand))
        Xt = bestX.copy()
        # sequential hole-optimized reinsertion
        NS = 2500 if sigma < 0.1 else 3500
        for idx in cand:
            if time.time() - t0 > T_END - 16:
                break
            try:
                v0 = _hole_insert(Xt, idx, rng, NS=NS, refine_iters=220)
                Xt[idx] = v0
            except Exception:
                Xt[idx] = _norm(rng.standard_normal(size=(d,)))
        # gaussian shake on the rest
        try:
            noise = rng.standard_normal(size=(n, d))
            # don't shake freshly inserted as much
            shake = np.ones(n) * sigma
            for idx in cand:
                shake[idx] = sigma * 0.15
            Xt = Xt + shake[:, None] * noise
            Xt = _norm(Xt)
        except Exception:
            pass
        # anneal schedule depends on sigma
        if time.time() - t0 > T_END - 15:
            v = _maxcos(Xt)
            if v < bestV:
                bestV, bestX = v, Xt.copy()
            break
        if sigma < 0.02:
            Xt = _anneal(Xt, steps=2200, T0=0.008, T1=0.0007, lr0=2.2, lr1=0.12, t0=t0, t_end=T_END)
            Xt = _exact_descent(Xt, steps=300, lr0=0.015, lr1=0.002, topk=80, t0=t0, t_end=T_END)
        elif sigma < 0.06:
            Xt = _anneal(Xt, steps=2800, T0=0.012, T1=0.0007, lr0=3.0, lr1=0.10, t0=t0, t_end=T_END)
            Xt = _exact_descent(Xt, steps=300, lr0=0.015, lr1=0.002, topk=80, t0=t0, t_end=T_END)
        else:
            Xt = _anneal(Xt, steps=3200, T0=0.018, T1=0.0010, lr0=4.0, lr1=0.12, t0=t0, t_end=T_END)
            Xt = _exact_descent(Xt, steps=400, lr0=0.02, lr1=0.002, topk=100, t0=t0, t_end=T_END)
        v = _maxcos(Xt)
        if v < bestV - 1e-12:
            bestV = v
            bestX = Xt.copy()
            # greedy extra polish on improvement if time allows
            if time.time() - t0 < T_END - 20:
                Xp = bestX.copy()
                Xp = _anneal(Xp, steps=1800, T0=0.0009, T1=0.00025, lr0=0.7, lr1=0.05, t0=t0, t_end=T_END)
                vp = _maxcos(Xp)
                if vp < bestV:
                    bestV, bestX = vp, Xp.copy()
        # also try a fresh random restart occasionally (different basin)
        if trial % 12 == 0 and (time.time() - t0) < T_END - 40:
            try:
                Xr = _norm(rng.standard_normal(size=(n, d)))
                Xr = _anneal(Xr, steps=3000, T0=0.02, T1=0.001, lr0=4.0, lr1=0.12, t0=t0, t_end=T_END)
                vr = _maxcos(Xr)
                if vr < bestV:
                    bestV, bestX = vr, Xr.copy()
            except Exception:
                pass

    # final squeeze on best with small steps
    if time.time() - t0 < T_END - 5:
        try:
            Z = bestX.copy()
            Z = _anneal(Z, steps=1000, T0=0.0008, T1=0.00025, lr0=0.5, lr1=0.05, t0=t0, t_end=T_END + 6)
            Z = _exact_descent(Z, steps=500, lr0=0.008, lr1=0.0008, topk=120, t0=t0, t_end=T_END + 6)
            v = _maxcos(Z)
            if v < bestV:
                bestV, bestX = v, Z.copy()
        except Exception:
            pass
    bestX = _norm(bestX)
    return bestX
