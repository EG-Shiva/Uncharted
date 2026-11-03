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

def solve():
    t0 = time.time()
    DEADLINE = 172.0
    rng = np.random.default_rng(12345)
    n, d = 93, 13
    iu0, iu1 = np.triu_indices(n, 1)

    # load starts
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (n, d):
                continue
            a = _norm(a)
            v = _maxcos(a)
            if v < best_val:
                best_val = v
                best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_val = _maxcos(best)

    def smooth_opt(X, steps, T0, T1, lr0, lr1, mu):
        V = np.zeros_like(X)
        grad = np.empty_like(X)
        for k in range(steps):
            f = k / max(1, steps - 1)
            T = T0 * (T1 / T0) ** f if T1 != T0 else T0
            lr = lr0 * (lr1 / lr0) ** f if lr1 != lr0 else lr0
            G = X @ X.T
            c = G[iu0, iu1]
            m = float(c.max())
            # global softmax
            w = np.exp((c - m) / T)
            s = float(w.sum())
            if not np.isfinite(s) or s == 0:
                w = np.zeros_like(w)
                w[int(np.argmax(c))] = 1.0
                s = 1.0
            w /= s
            grad.fill(0.0)
            # grad_i = sum_j w_ij X_j
            np.add.at(grad, iu0, w[:, None] * X[iu1])
            np.add.at(grad, iu1, w[:, None] * X[iu0])
            dot = np.einsum('ij,ij->i', grad, X)[:, None]
            gt = grad - dot * X
            V = mu * V + (1.0 - mu) * gt
            dotv = np.einsum('ij,ij->i', V, X)[:, None]
            Vt = V - dotv * X
            X = X - lr * Vt
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            if k % 500 == 499 and time.time() - t0 > DEADLINE:
                break
        return X

    def hard_polish(X, iters, step0, K):
        bv = _maxcos(X)
        step = step0
        grad = np.empty_like(X)
        K = min(K, len(iu0))
        for _ in range(iters):
            if time.time() - t0 > DEADLINE:
                break
            G = X @ X.T
            c = G[iu0, iu1]
            if K < len(c):
                idx = np.argpartition(-c, K)[:K]
            else:
                idx = np.arange(len(c))
            I = iu0[idx]
            J = iu1[idx]
            grad.fill(0.0)
            np.add.at(grad, I, X[J])
            np.add.at(grad, J, X[I])
            grad /= float(len(idx))
            dot = np.einsum('ij,ij->i', grad, X)[:, None]
            gt = grad - dot * X
            if not np.all(np.isfinite(gt)):
                step *= 0.5
                if step < 1e-13:
                    break
                continue
            Xt = X - step * gt
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            v = _maxcos(Xt)
            if v < bv:
                X = Xt
                bv = v
                step = min(step * 1.3, 0.02)
            else:
                step *= 0.5
                if step < 1e-13:
                    break
        return X, bv

    # initial squeeze of incumbent
    try:
        best = smooth_opt(best, steps=1200, T0=1e-3, T1=5e-5, lr0=0.04, lr1=0.005, mu=0.85)
        best, bv = hard_polish(best, iters=300, step0=0.004, K=300)
        if bv < best_val:
            best_val = bv
    except Exception:
        pass

    def void_reinsert(X, nbad=1, Nc=6000, jitter=0.002):
        Xn = X.copy()
        G = Xn @ Xn.T
        np.fill_diagonal(G, -2.0)
        rowmax = G.max(axis=1)
        order = np.argsort(-rowmax)
        for t in range(nbad):
            ibad = int(order[t])
            cand = _norm(rng.standard_normal((Nc, d)))
            D = cand @ Xn.T
            D[:, ibad] = -2.0
            worst = D.max(axis=1)
            ibest = int(np.argmin(worst))
            Xn[ibad] = cand[ibest]
        if jitter > 0:
            Xn += jitter * rng.standard_normal(Xn.shape)
            Xn = _norm(Xn)
        return Xn

    trial = 0
    sigmas_all = [0.02, 0.008, 0.005, 0.03, 0.012]
    while time.time() - t0 < 150.0:
        if time.time() - t0 > DEADLINE - 12:
            break
        mode = trial % 6
        sigma = sigmas_all[trial % len(sigmas_all)]
        try:
            if mode == 0:
                prop = _norm(best + sigma * rng.standard_normal(best.shape))
                T0, T1 = 0.004, 0.00015
                lr0, lr1 = 0.06, 0.005
                steps = 2000
            elif mode == 1:
                prop = best.copy() + 0.002 * rng.standard_normal(best.shape)
                G = best @ best.T
                np.fill_diagonal(G, -2.0)
                rm = G.max(axis=1)
                worst = np.argsort(-rm)[:10]
                prop[worst] += 0.025 * rng.standard_normal((len(worst), d))
                prop = _norm(prop)
                T0, T1 = 0.003, 0.00012
                lr0, lr1 = 0.05, 0.004
                steps = 2000
            elif mode == 2:
                prop = void_reinsert(best, nbad=1, Nc=8000, jitter=0.0015)
                T0, T1 = 0.0035, 0.00012
                lr0, lr1 = 0.055, 0.004
                steps = 2200
            elif mode == 3:
                prop = void_reinsert(best, nbad=2, Nc=6000, jitter=0.0015)
                T0, T1 = 0.004, 0.00015
                lr0, lr1 = 0.06, 0.005
                steps = 2200
            elif mode == 4:
                prop = best.copy()
                idx = rng.choice(n, size=18, replace=False)
                prop[idx] += 0.02 * rng.standard_normal((len(idx), d))
                prop = _norm(prop)
                T0, T1 = 0.005, 0.0002
                lr0, lr1 = 0.06, 0.005
                steps = 1800
            else:
                prop = _norm(best + 0.006 * rng.standard_normal(best.shape))
                T0, T1 = 0.002, 8e-05
                lr0, lr1 = 0.04, 0.003
                steps = 1800
            prop = smooth_opt(prop, steps=steps, T0=T0, T1=T1, lr0=lr0, lr1=lr1, mu=0.85)
            prop, pv = hard_polish(prop, iters=260, step0=0.0035, K=280)
            # second micro squeeze with larger K
            if pv < best_val + 0.0005:
                prop2, pv2 = hard_polish(prop, iters=120, step0=0.0012, K=500)
                if pv2 < pv:
                    prop, pv = prop2, pv2
            if pv < best_val:
                best = prop.copy()
                best_val = pv
        except Exception:
            pass
        trial += 1

    # final micro squeeze on champion
    try:
        if time.time() - t0 < DEADLINE - 2:
            best, bv = hard_polish(best, iters=400, step0=0.002, K=400)
            if bv < best_val:
                best_val = bv
        if time.time() - t0 < DEADLINE - 2:
            best = smooth_opt(best, steps=600, T0=3e-4, T1=3e-5, lr0=0.015, lr1=0.002, mu=0.8)
            best2, bv2 = hard_polish(best, iters=400, step0=0.0015, K=350)
            if bv2 < best_val:
                best = best2
                best_val = bv2
    except Exception:
        pass

    best = _norm(np.asarray(best, dtype=np.float64))
    return best
