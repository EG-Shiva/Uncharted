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
    BUDGET = 165.0
    rng = np.random.default_rng(0)
    # load
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (59, 22) and np.all(np.isfinite(a)):
                a = _norm(a)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_norm(rng.standard_normal((59, 22))))
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    best_val = float(vals[bi])

    n, d = 59, 22

    def polish(Y0, steps, a0, a1, lr0, lr1):
        Y = Y0.copy()
        m = np.zeros_like(Y)
        v = np.zeros_like(Y)
        b1, b2, eps = 0.9, 0.999, 1e-8
        G0 = Y @ Y.T
        np.fill_diagonal(G0, -10.0)
        cur_best_val = float(np.max(G0))
        cur_best = Y.copy()
        la0, la1 = np.log(a0), np.log(a1)
        ll0, ll1 = np.log(lr0), np.log(lr1)
        b1t_pow = 1.0
        b2t_pow = 1.0
        # precompute decay per step for bias correction without pow
        for t in range(1, steps + 1):
            if (t & 511) == 0:
                if (time.time() - t0) > BUDGET - 3.0:
                    break
            frac = t / steps
            alpha = float(np.exp(la0 + (la1 - la0) * frac))
            lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
            G = Y @ Y.T
            np.fill_diagonal(G, -10.0)
            gm = float(np.max(G))
            G -= gm
            G *= alpha
            np.exp(G, out=G)
            np.fill_diagonal(G, 0.0)
            s = float(np.sum(G))
            if not np.isfinite(s) or s <= 0:
                continue
            Gr = (G @ Y) / s
            dot = np.sum(Gr * Y, axis=1, keepdims=True)
            Gr -= Y * dot
            m *= b1
            m += (1.0 - b1) * Gr
            v *= b2
            v += (1.0 - b2) * (Gr * Gr)
            b1t_pow *= b1
            b2t_pow *= b2
            mh = m / (1.0 - b1t_pow)
            vh = v / (1.0 - b2t_pow)
            Y -= lr * mh / (np.sqrt(vh) + eps)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn[rn == 0] = 1.0
            Y /= rn
            if (t % 500 == 0) or (t == steps):
                cv = _max_cos(Y)
                if cv < cur_best_val:
                    cur_best_val = cv
                    cur_best = Y.copy()
        fv = _max_cos(Y)
        if fv < cur_best_val:
            cur_best_val = fv
            cur_best = Y.copy()
        return cur_best, cur_best_val

    def void_vector(X_others, ntry=8000):
        # find unit vector minimizing max dot to X_others
        C = rng.standard_normal((ntry, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        D = C @ X_others.T  # ntry x m
        mx = np.max(D, axis=1)
        i = int(np.argmin(mx))
        # local refine: sample around best
        best_c = C[i].copy()
        best_m = float(mx[i])
        for _ in range(2):
            Cc = best_c + 0.15 * rng.standard_normal((2000, d))
            Cc /= np.linalg.norm(Cc, axis=1, keepdims=True)
            D2 = Cc @ X_others.T
            mx2 = np.max(D2, axis=1)
            j = int(np.argmin(mx2))
            if float(mx2[j]) < best_m:
                best_m = float(mx2[j])
                best_c = Cc[j].copy()
        return best_c

    # initial polish
    if (time.time() - t0) < BUDGET - 5:
        x_best, best_val = polish(x_best, steps=6000, a0=80, a1=800, lr0=0.005, lr1=0.0004)

    cur = x_best.copy()
    cur_val = best_val

    it = 0
    while (time.time() - t0) < BUDGET - 8:
        it += 1
        elapsed = time.time() - t0
        remain = BUDGET - elapsed
        if remain < 6:
            break
        steps = 3000 if remain > 40 else (2200 if remain > 20 else 1500)
        mode = it % 10

        # temperature for wandering acceptance
        Temp = 0.003 * (1.0 - elapsed / BUDGET) + 0.0002

        Y = cur.copy()
        if mode == 0:
            Y = Y + 0.012 * rng.standard_normal((n, d))
            Y = _norm(Y)
            a0, a1, lr0, lr1 = 150.0, 1200.0, 0.003, 0.00025
        elif mode == 1:
            Y = Y + 0.035 * rng.standard_normal((n, d))
            Y = _norm(Y)
            a0, a1, lr0, lr1 = 80.0, 800.0, 0.004, 0.0003
        elif mode == 2:
            Y = Y + 0.09 * rng.standard_normal((n, d))
            Y = _norm(Y)
            a0, a1, lr0, lr1 = 60.0, 700.0, 0.005, 0.0004
        elif mode == 3:
            # subset shake 4-8 points hard
            k = int(rng.integers(4, 9))
            idx = rng.choice(n, size=k, replace=False)
            Y[idx] += 0.35 * rng.standard_normal((k, d))
            Y = _norm(Y)
            a0, a1, lr0, lr1 = 60.0, 700.0, 0.005, 0.0004
        elif mode == 4:
            # kill 1 worst -> void
            G = Y @ Y.T
            np.fill_diagonal(G, -2.0)
            rowmax = np.max(G, axis=1)
            k = int(np.argmax(rowmax))
            mask = np.ones(n, dtype=bool); mask[k] = False
            Y[k] = void_vector(Y[mask], ntry=8000)
            # small jitter others
            Y += 0.005 * rng.standard_normal((n, d))
            Y = _norm(Y)
            a0, a1, lr0, lr1 = 100.0, 900.0, 0.004, 0.0003
        elif mode == 5:
            # kill 3 worst -> void sequentially
            G = Y @ Y.T
            np.fill_diagonal(G, -2.0)
            rowmax = np.max(G, axis=1)
            order = np.argsort(-rowmax)[:3]
            for k in order:
                mask = np.ones(n, dtype=bool); mask[k] = False
                # use current Y for others (with already replaced)
                Y[k] = void_vector(Y[mask], ntry=6000)
            Y += 0.005 * rng.standard_normal((n, d))
            Y = _norm(Y)
            a0, a1, lr0, lr1 = 70.0, 800.0, 0.0045, 0.00035
        elif mode == 6:
            # kill 5 random + void
            idx = rng.choice(n, size=5, replace=False)
            for k in idx:
                mask = np.ones(n, dtype=bool); mask[k] = False
                Y[k] = void_vector(Y[mask], ntry=5000)
            Y = _norm(Y)
            a0, a1, lr0, lr1 = 60.0, 700.0, 0.005, 0.0004
        elif mode == 7:
            # single random jump to random dir
            k = int(rng.integers(n))
            Y[k] = rng.standard_normal(d)
            Y = _norm(Y)
            a0, a1, lr0, lr1 = 80.0, 800.0, 0.004, 0.0003
        elif mode == 8:
            # partial random restart: 15 random points
            idx = rng.choice(n, size=15, replace=False)
            Y[idx] = rng.standard_normal((15, d))
            Y = _norm(Y)
            a0, a1, lr0, lr1 = 50.0, 600.0, 0.005, 0.0004
        else:
            Y = Y + 0.16 * rng.standard_normal((n, d))
            Y = _norm(Y)
            a0, a1, lr0, lr1 = 50.0, 600.0, 0.005, 0.0004

        Yp, Yv = polish(Y, steps=steps, a0=a0, a1=a1, lr0=lr0, lr1=lr1)

        if Yv < best_val - 1e-12:
            x_best = Yp.copy()
            best_val = Yv
            cur = Yp.copy()
            cur_val = Yv
        else:
            # wandering acceptance (Metropolis-like) to change basin
            if Yv < cur_val or rng.random() < float(np.exp(-(Yv - cur_val) / Temp)):
                cur = Yp.copy()
                cur_val = Yv
            # occasionally restart wandering from best with shake
            if (it % 25) == 0 and rng.random() < 0.5:
                cur = x_best + 0.05 * rng.standard_normal((n, d))
                cur = _norm(cur)
                cur_val = _max_cos(cur)

    # final squeeze
    if (time.time() - t0) < BUDGET - 2:
        y, v = polish(x_best, steps=3000, a0=800, a1=2500, lr0=0.0012, lr1=0.00012)
        if v < best_val:
            x_best, best_val = y, v

    x_best = _norm(x_best)
    return x_best
