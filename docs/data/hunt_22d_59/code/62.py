import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def solve():
    t0 = time.time()
    BUDGET = 164.0
    rng = np.random.default_rng(0)
    n, d = 59, 22

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                a = _norm(a)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_norm(rng.standard_normal((n, d))))
    vals = [_max_cos(c) for c in cands]
    gbest = cands[int(np.argmin(vals))].copy()
    gval = float(min(vals))

    b1, b2, eps = 0.9, 0.999, 1e-8

    def adam_polish(X0, steps, a0, a1, lr0, lr1):
        X = X0.copy()
        m = np.zeros_like(X)
        v = np.zeros_like(X)
        cur_best = X.copy()
        cur_val = _max_cos(X)
        la0, la1 = np.log(a0), np.log(a1)
        ll0, ll1 = np.log(lr0), np.log(lr1)
        b1t, b2t = 1.0, 1.0
        for t in range(1, steps + 1):
            if (t & 63) == 0 and (time.time() - t0) > BUDGET:
                break
            frac = t / steps
            alpha = float(np.exp(la0 + (la1 - la0) * frac))
            lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            M = float(np.max(G))
            G -= M
            G *= alpha
            np.exp(G, out=G)
            np.fill_diagonal(G, 0.0)
            S = float(np.sum(G))
            if not np.isfinite(S) or S <= 0:
                continue
            Gr = (G @ X) / S
            dot = np.sum(Gr * X, axis=1, keepdims=True)
            Gr -= dot * X
            b1t *= b1; b2t *= b2
            m *= b1; m += (1.0 - b1) * Gr
            v *= b2; v += (1.0 - b2) * (Gr * Gr)
            mh = m / (1.0 - b1t)
            vh = v / (1.0 - b2t)
            X -= lr * mh / (np.sqrt(vh) + eps)
            rn = np.linalg.norm(X, axis=1, keepdims=True)
            rn[rn == 0] = 1.0
            X /= rn
            if (t % 400 == 0) or (t == steps):
                cv = _max_cos(X)
                if cv < cur_val:
                    cur_val = cv
                    cur_best = X.copy()
        fv = _max_cos(X)
        if fv < cur_val:
            cur_val = fv
            cur_best = X.copy()
        return cur_best, cur_val

    # Stage 0: secure baseline polish
    gbest, gval = adam_polish(gbest, steps=6000, a0=60.0, a1=900.0, lr0=0.006, lr1=0.0004)
    if (time.time() - t0) > BUDGET - 10:
        return _norm(gbest)

    cur = gbest.copy()
    cval = gval

    # hole-filling helper
    cand_pool = _norm(rng.standard_normal((4000, d)))

    def ruin_recreate(base, k, hole_samples=2500):
        G = base @ base.T
        np.fill_diagonal(G, -2.0)
        rowmax = np.max(G, axis=1)
        w = int(np.argmax(rowmax))
        # k-1 nearest neighbours of w (largest dots)
        row = (base @ base[w])
        row[w] = -2.0
        idx = [w]
        if k > 1:
            nn = np.argsort(-row)[:k-1]
            idx.extend([int(i) for i in nn])
        idx = np.array(idx, dtype=int)
        keep = np.ones(n, dtype=bool)
        keep[idx] = False
        F = base[keep]  # (n-k, d)
        # sample fresh candidates
        pool = _norm(rng.standard_normal((hole_samples, d)))
        dots_pool_fixed = pool @ F.T  # (S, n-k)
        max_fixed = np.max(dots_pool_fixed, axis=1)
        new_pts = []
        # iterative greedy insertion
        chosen = np.zeros(len(pool), dtype=bool)
        for _ in range(k):
            if new_pts:
                NP = np.stack(new_pts)  # (j,d)
                dp = pool @ NP.T
                mp = np.max(dp, axis=1)
                score = np.maximum(max_fixed, mp)
            else:
                score = max_fixed
            score = score.copy()
            score[chosen] = 2.0
            bi = int(np.argmin(score))
            chosen[bi] = True
            new_pts.append(pool[bi].copy())
        Y = base.copy()
        for j, ii in enumerate(idx):
            Y[ii] = new_pts[j]
        # tiny jitter on kept points to unlock
        Y[keep] += 0.003 * rng.standard_normal((n - k, d))
        Y = _norm(Y)
        return Y

    it = 0
    # temperature schedule for Metropolis walker
    while (time.time() - t0) < BUDGET - 12:
        it += 1
        remain = BUDGET - (time.time() - t0)
        if remain < 5:
            break
        if remain > 80:
            steps = 4000
        elif remain > 30:
            steps = 3000
        else:
            steps = 2000
        # alternate move types
        mode = it % 5
        if mode in (0, 1, 2):
            k = 3 + (it % 4)  # 3..6
            if (it % 11) == 0:
                k = 8
            Y0 = ruin_recreate(cur, k=k)
            a0, a1 = 30.0, 800.0
            lr0, lr1 = 0.006, 0.00035
        elif mode == 3:
            sigma = 0.05 if (it % 2 == 0) else 0.15
            Y0 = cur + sigma * rng.standard_normal((n, d))
            # one-point jump occasionally
            if (it % 3) == 0:
                kk = int(rng.integers(0, n))
                Y0[kk] = rng.standard_normal(d)
            Y0 = _norm(Y0)
            a0, a1 = 50.0, 800.0
            lr0, lr1 = 0.005, 0.00035
        else:
            sigma = 0.012
            Y0 = cur + sigma * rng.standard_normal((n, d))
            Y0 = _norm(Y0)
            a0, a1 = 150.0, 1200.0
            lr0, lr1 = 0.003, 0.00025
        Yp, Yv = adam_polish(Y0, steps=steps, a0=a0, a1=a1, lr0=lr0, lr1=lr1)
        if Yv < gval - 1e-12:
            gbest = Yp.copy()
            gval = Yv
        # Metropolis move for walker (allows uphill to change basin)
        # T decays with iteration
        T = 0.0015 * (0.94 ** (it // 2)) + 2e-05
        if Yv <= cval or rng.random() < np.exp(-(Yv - cval) / T):
            cur = Yp.copy()
            cval = Yv
        # restart walker from global best occasionally if too bad
        if cval > gval + 0.004 and (it % 6) == 0:
            cur = gbest.copy()
            cval = gval
        if (time.time() - t0) > BUDGET - 12:
            break

    # final squeeze of global best
    if (time.time() - t0) < BUDGET - 2:
        gbest, gval = adam_polish(gbest, steps=3000, a0=900.0, a1=2500.0, lr0=0.0012, lr1=0.00012)
    gbest = _norm(gbest)
    return gbest
