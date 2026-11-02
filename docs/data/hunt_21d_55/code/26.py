import time
import numpy as np

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _maxcos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _quench(X0, steps, beta0, beta1, lr0, lr1, mu=0.85, seed_off=0):
    X = X0.copy()
    n, d = X.shape
    V = np.zeros_like(X)
    for t in range(1, steps + 1):
        frac = t / steps
        beta = beta0 * (beta1 / beta0) ** frac if beta1 != beta0 else beta0
        lr = lr0 * (lr1 / lr0) ** frac if lr1 != lr0 else lr0
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(G.max())
        # soft-max weights focused on max
        E = np.exp(beta * (G - M))
        np.fill_diagonal(E, 0.0)
        # gradient of log-sum-exp ~ (E @ X)/sum(E)
        Gr = E @ X
        # tangent projection
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        # row-normalize for scale invariance
        rn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        Gn = Gt / rn
        # heavy ball
        V = mu * V + (1.0 - mu) * Gn
        X = X - lr * V
        # renorm
        nn = np.linalg.norm(X, axis=1, keepdims=True)
        nn[nn == 0] = 1.0
        X = X / nn
        if not np.all(np.isfinite(X)):
            return X0
    return X

def _maxpush(X0, steps, lr0, tol=2e-3, mu=0.5):
    # cheap greedy squeeze on near-max pairs only
    X = X0.copy()
    n, d = X.shape
    V = np.zeros_like(X)
    lr = lr0
    best = _maxcos(X)
    Xb = X.copy()
    for t in range(steps):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(G.max())
        W = (G > M - tol).astype(float)
        np.fill_diagonal(W, 0.0)
        # anneal tol
        if t % 100 == 99:
            tol = max(tol * 0.7, 2e-4)
            lr *= 0.85
        Gr = W @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        rn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        # only move points that have contacts
        has = (W.sum(axis=1, keepdims=True) > 0).astype(float)
        Gn = Gt / rn * has
        V = mu * V + (1.0 - mu) * Gn
        # scale step by typical magnitude
        Y = X - lr * V
        Y = _norm(Y)
        v = _maxcos(Y)
        # greedy accept / small uphill allowed early
        if v < best or (t < steps // 3 and v < best + 1e-4):
            X = Y
            if v < best:
                best = v
                Xb = X.copy()
        else:
            # shrink and reset momentum on failure
            lr *= 0.97
            V *= 0.5
            if lr < 1e-5:
                break
    return Xb

def _hole(Y, rng, n_cand=3000):
    d = Y.shape[1]
    C = rng.randn(n_cand, d)
    C = _norm(C)
    D = C @ Y.T
    md = D.max(axis=1)
    return C[int(np.argmin(md))]

def solve():
    t0 = time.time()
    DEADLINE = 168.0
    rng = np.random.RandomState(0)
    n, d = 55, 21
    # load incumbents
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=float)
            if a.shape == (n, d):
                a = _norm(a)
                v = _maxcos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.randn(n, d))
        best_val = _maxcos(best)
    elites = [(best_val, best.copy())]

    def consider(X):
        nonlocal best_val, best
        v = _maxcos(X)
        if v < best_val - 1e-12:
            best_val = v
            best = X.copy()
            elites.append((v, X.copy()))
            elites.sort(key=lambda e: e[0])
            del elites[4:]
            return True
        return False

    # quick initial max-push on incumbent (cheap, no LP)
    if time.time() - t0 < DEADLINE - 10:
        try:
            sq = _maxpush(best, steps=400, lr0=0.02)
            consider(sq)
        except Exception:
            pass

    trial = 0
    # precompute a binary seed pool for diversity
    while time.time() - t0 < DEADLINE - 25:
        trial += 1
        mode = trial % 10
        if mode == 0:
            # fresh random gaussian
            X0 = _norm(rng.randn(n, d))
        elif mode == 1:
            # fresh binary +-1 (symmetric basin)
            X0 = _norm(rng.choice(np.array([-1.0, 1.0]), size=(n, d)).astype(float) + 0.2 * rng.randn(n, d))
        elif mode == 2:
            # fresh uniform sphere via normal + large quench
            X0 = _norm(rng.randn(n, d) * 1.0)
        elif mode == 3:
            # perturb elite
            e = elites[int(rng.rand() * len(elites))][1]
            sigma = 0.05 + rng.rand() * 0.25
            X0 = _norm(e + sigma * rng.randn(n, d))
        elif mode == 4:
            # hole reinsertion of worst points
            e = elites[0][1].copy()
            G = e @ e.T
            np.fill_diagonal(G, -2.0)
            rm = G.max(axis=1)
            k = 1 + int(rng.rand() * 3)  # 1-3
            worst = np.argsort(-rm)[:k]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[worst] = False
            Y = e[keep_mask]
            for w in worst:
                if rng.rand() < 0.7:
                    h = _hole(Y, rng, n_cand=2500)
                else:
                    h = _norm(rng.randn(d)[None, :])[0]
                e[w] = h
                Y = np.vstack([Y, h[None, :]])
            X0 = _norm(e + 0.01 * rng.randn(n, d))
        elif mode == 5:
            # subset shake: shake 5-12 random points strongly
            e = elites[int(rng.rand() * len(elites))][1].copy()
            k = 5 + int(rng.rand() * 8)
            idx = rng.choice(n, size=k, replace=False)
            e[idx] = _norm(rng.randn(k, d))
            X0 = _norm(e + 0.02 * rng.randn(n, d))
        elif mode == 6:
            # crossover of two elites
            if len(elites) >= 2:
                a = elites[0][1]; b = elites[1][1]
                m = rng.rand(n, 1) < 0.5
                X0 = np.where(m, a, b)
                X0 = _norm(X0 + 0.03 * rng.randn(n, d))
            else:
                X0 = _norm(elites[0][1] + 0.1 * rng.randn(n, d))
        elif mode == 7:
            # overloaded shake based on contact counts
            e = elites[0][1].copy()
            G = e @ e.T
            np.fill_diagonal(G, -2.0)
            thr = G.max() - 2e-3
            cnt = (G > thr).sum(axis=1)
            over = np.argsort(-cnt)[:8]
            e[over] += 0.12 * rng.randn(len(over), d)
            X0 = _norm(e)
        elif mode == 8:
            # sparse seed: random coordinate spikes
            X0 = np.zeros((n, d))
            for i in range(n):
                j = rng.randint(0, d)
                X0[i, j] = 1.0
            X0 = _norm(X0 + 0.5 * rng.randn(n, d))
        else:
            e = elites[0][1]
            X0 = _norm(e + 0.15 * rng.randn(n, d))

        # random quench hyperparams (annealed beta -> sharp max)
        r = rng.rand
        beta0 = 15.0 + r() * 30.0
        beta1 = 120.0 + r() * 280.0
        lr0 = 0.03 + r() * 0.05
        lr1 = 0.002 + r() * 0.006
        steps = 350 + int(r() * 450)
        # smaller steps for fresh starts need more; cap by time
        try:
            Xq = _quench(X0, steps=steps, beta0=beta0, beta1=beta1, lr0=lr0, lr1=lr1)
        except Exception:
            continue
        vq = _maxcos(Xq)
        improved = consider(Xq)
        # if promising, short max-push + second quench with holes
        if vq < best_val + 0.005 and time.time() - t0 < DEADLINE - 26:
            try:
                Xp = _maxpush(Xq, steps=150, lr0=0.01)
                consider(Xp)
                # one hole-reinsert + re-quench if close to best
                if _maxcos(Xp) < best_val + 0.002 and (rng.rand() < 0.4 or improved):
                    G = Xp @ Xp.T
                    np.fill_diagonal(G, -2.0)
                    rm = G.max(axis=1)
                    w = int(np.argmax(rm))
                    Y = np.delete(Xp, w, axis=0)
                    h = _hole(Y, rng, n_cand=2000)
                    Xh = Xp.copy(); Xh[w] = h
                    Xh = _quench(Xh, steps=250, beta0=60.0, beta1=300.0, lr0=0.02, lr1=0.002)
                    consider(Xh)
            except Exception:
                pass
        if trial > 800:
            break

    # final intensive polish of global best with remaining time
    rem = DEADLINE - (time.time() - t0)
    if rem > 2:
        try:
            # long gentle quench
            Xf = _quench(best, steps=min(2500, int(400 * rem)), beta0=80.0, beta1=400.0, lr0=0.015, lr1=0.0015, mu=0.9)
            consider(Xf)
        except Exception:
            pass
        try:
            rem2 = DEADLINE - (time.time() - t0)
            if rem2 > 2:
                Xf2 = _maxpush(best, steps=min(1200, int(150 * rem2)), lr0=0.008)
                consider(Xf2)
        except Exception:
            pass

    best = _norm(best)
    best[~np.isfinite(best)] = 0.01
    best = _norm(best)
    return best
