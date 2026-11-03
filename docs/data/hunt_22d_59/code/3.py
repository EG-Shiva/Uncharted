import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    BUDGET = 165.0  # leave margin for 180s limit
    np.random.seed(0)

    # load starting points
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (59, 22):
                nrm = np.linalg.norm(a, axis=1, keepdims=True)
                nrm[nrm == 0] = 1.0
                a = a / nrm
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        X0 = np.random.randn(59, 22)
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands.append(X0)
    # also add a random candidate for diversity (deterministic)
    # keep best of loaded as base
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best = cands[bi].copy()
    best_val = float(vals[bi])

    n, d = best.shape

    def adam_polish(X, steps, a0, a1, lr0, lr1):
        # X is (n,d) normalized, will be copied
        X = X.copy()
        m = np.zeros_like(X)
        v = np.zeros_like(X)
        b1 = 0.9
        b2 = 0.999
        eps = 1e-8
        cur_best = X.copy()
        cur_val = _max_cos(X)
        la0 = np.log(a0)
        la1 = np.log(a1)
        llr0 = np.log(lr0)
        llr1 = np.log(lr1)
        for t in range(1, steps + 1):
            if (t & 127) == 0 and (time.time() - t0) > BUDGET:
                break
            frac = t / steps
            alpha = float(np.exp(la0 + (la1 - la0) * frac))
            lr = float(np.exp(llr0 + (llr1 - llr0) * frac))
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            M = float(np.max(G))
            # E = exp(alpha*(G-M)), reuse G buffer
            # in-place: G = G-M then *alpha then exp
            G -= M
            G *= alpha
            # exp in place
            np.exp(G, out=G)
            np.fill_diagonal(G, 0.0)
            S = float(np.sum(G))
            if not np.isfinite(S) or S <= 0:
                continue
            Grad = (G @ X) / S
            # tangent projection
            dot = np.sum(Grad * X, axis=1, keepdims=True)
            Gt = Grad - dot * X
            # adam
            m *= b1
            m += (1.0 - b1) * Gt
            v *= b2
            v += (1.0 - b2) * (Gt * Gt)
            # bias correction
            # pow is fine
            c1 = 1.0 - (b1 ** t)
            c2 = 1.0 - (b2 ** t)
            if c1 < 1e-12:
                c1 = 1.0
            if c2 < 1e-12:
                c2 = 1.0
            mc = m / c1
            vc = v / c2
            upd = mc / (np.sqrt(vc) + eps)
            # clip extreme updates to stabilize
            # adam update is ~[-1,1], no need
            X -= lr * upd
            nrm = np.linalg.norm(X, axis=1, keepdims=True)
            nrm[nrm == 0] = 1.0
            X /= nrm
            if (t % 250 == 0) or (t == steps):
                cv = _max_cos(X)
                if cv < cur_val:
                    cur_val = cv
                    cur_best = X.copy()
        # final check
        fv = _max_cos(X)
        if fv < cur_val:
            cur_val = fv
            cur_best = X.copy()
        return cur_best, cur_val

    # Stage 1: long polish from record
    if (time.time() - t0) < BUDGET:
        best, best_val = adam_polish(best, steps=12000, a0=50, a1=800, lr0=0.006, lr1=0.0004)

    # basin hopping
    sigmas = [0.004, 0.008, 0.015, 0.030, 0.060]
    it = 0
    while (time.time() - t0) < BUDGET - 6:
        it += 1
        remain = BUDGET - (time.time() - t0)
        if remain < 4:
            break
        # choose steps adaptively
        if remain > 60:
            steps = 5000
        elif remain > 25:
            steps = 3500
        else:
            steps = 2000
        sigma = sigmas[it % len(sigmas)]
        # per-row sigma boost for rattlers
        G = best @ best.T
        np.fill_diagonal(G, -2.0)
        rowmax = np.max(G, axis=1)
        # rattlers have clearly smaller row max
        is_rattler = rowmax < (best_val - 0.008)
        noise = np.random.randn(n, d)
        scale = np.full((n, 1), sigma, dtype=np.float64)
        scale[is_rattler[:, None].repeat(d, axis=1).any(axis=1) if False else is_rattler, :] *= 3.0
        # simpler: scale rows
        # scale is (n,1)
        Y = best + scale * noise
        # occasional single-point big jump
        if (it % 7) == 0:
            k = int(np.random.randint(n))
            Y[k] = np.random.randn(d)
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        Y /= nrm
        # anneal range varies with sigma: big shake -> lower alpha start
        if sigma >= 0.03:
            a0, a1 = 60.0, 700.0
            lr0, lr1 = 0.005, 0.0004
        else:
            a0, a1 = 150.0, 1200.0
            lr0, lr1 = 0.003, 0.00025
        Yp, Yv = adam_polish(Y, steps=steps, a0=a0, a1=a1, lr0=lr0, lr1=lr1)
        if Yv < best_val - 1e-12:
            best = Yp
            best_val = Yv

    # final fine squeeze with high alpha small lr
    if (time.time() - t0) < BUDGET - 2:
        best, best_val = adam_polish(best, steps=3000, a0=800, a1=2000, lr0=0.0015, lr1=0.00015)

    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
