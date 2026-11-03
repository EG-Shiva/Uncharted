import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _smooth_opt(X, steps, T0, T1, lr0, lr1, mu, iu0, iu1):
    n, d = X.shape
    V = np.zeros_like(X)
    W = np.empty((n, n), dtype=X.dtype)
    for k in range(steps):
        f = k / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = X @ X.T
        c = G[iu0, iu1]
        m = c.max()
        # weights exp((c-m)/T)
        w = np.exp((c - m) / T)
        s = w.sum()
        if s == 0 or not np.isfinite(s):
            w = np.zeros_like(w)
            w[np.argmax(c)] = 1.0
            s = 1.0
        w /= s
        W.fill(0.0)
        W[iu0, iu1] = w
        W[iu1, iu0] = w
        grad = W @ X
        dot = np.einsum('ij,ij->i', grad, X)[:, None]
        grad_t = grad - dot * X
        V = mu * V + (1.0 - mu) * grad_t
        # project velocity to current tangent
        dotv = np.einsum('ij,ij->i', V, X)[:, None]
        Vt = V - dotv * X
        X = X - lr * Vt
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _hard_polish(X, iters, step0, K, iu0, iu1):
    best = _max_cos(X)
    step = step0
    n, d = X.shape
    K = min(K, len(iu0))
    grad = np.empty_like(X)
    for _ in range(iters):
        G = X @ X.T
        c = G[iu0, iu1]
        m = c.max()
        # top-K active pairs
        if K < len(c):
            idx = np.argpartition(-c, K)[:K]
        else:
            idx = np.arange(len(c))
        I = iu0[idx]
        J = iu1[idx]
        grad.fill(0.0)
        # grad_i = sum X_j over active pairs containing i
        np.add.at(grad, I, X[J])
        np.add.at(grad, J, X[I])
        grad /= float(len(idx))
        dot = np.einsum('ij,ij->i', grad, X)[:, None]
        gt = grad - dot * X
        ng = np.linalg.norm(gt)
        if ng == 0 or not np.isfinite(ng):
            step *= 0.5
            if step < 1e-13:
                break
            continue
        Xt = X - step * gt
        Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
        v = _max_cos(Xt)
        if v < best:
            X = Xt
            best = v
            step = min(step * 1.3, 0.02)
        else:
            step *= 0.5
            if step < 1e-13:
                break
    return X, best

def solve():
    t0 = time.time()
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    for a in (xb, xr):
        if a is not None:
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (93, 13):
                a = _norm(a)
                cands.append(( _max_cos(a), a))
    n, d = 93, 13
    if len(cands) == 0:
        best = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(best)
    else:
        cands.sort(key=lambda t: t[0])
        best_val, best = cands[0]
        best = best.copy()

    iu0, iu1 = np.triu_indices(n, 1)

    # immediate squeeze of the single active pair
    best = _smooth_opt(best, steps=1500, T0=1e-3, T1=3e-5, lr0=0.03, lr1=0.004, mu=0.8, iu0=iu0, iu1=iu1)
    best, best_val_new = _hard_polish(best, iters=400, step0=0.004, K=250, iu0=iu0, iu1=iu1)
    best_val = float(min(best_val, best_val_new))

    sigma = 0.03
    n_trials = 75
    for trial in range(n_trials):
        if time.time() - t0 > 160:
            break
        r = rng.random()
        if r < 0.62:
            # gaussian kick of best
            prop = best + sigma * rng.standard_normal(best.shape)
            prop = _norm(prop)
            prop = _smooth_opt(prop, steps=2200, T0=0.004, T1=0.00015, lr0=0.05, lr1=0.004, mu=0.8, iu0=iu0, iu1=iu1)
            prop, pv = _hard_polish(prop, iters=250, step0=0.003, K=250, iu0=iu0, iu1=iu1)
        elif r < 0.87:
            # worst-point reinsertion into largest void
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            ibad = int(np.argmax(rowmax))
            # tie: also consider counts in top pairs
            # generate candidates, pick most isolated
            Nc = 2000
            cand = rng.standard_normal((Nc, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            D = cand @ best.T  # Nc x 93
            D[:, ibad] = -2.0
            worst = D.max(axis=1)
            ibest = int(np.argmin(worst))
            prop = best.copy()
            prop[ibad] = cand[ibest]
            # small shake of neighbours then anneal
            prop = _smooth_opt(prop, steps=2200, T0=0.003, T1=0.00012, lr0=0.05, lr1=0.003, mu=0.8, iu0=iu0, iu1=iu1)
            prop, pv = _hard_polish(prop, iters=250, step0=0.003, K=250, iu0=iu0, iu1=iu1)
        else:
            # fresh random global anneal
            prop = _norm(rng.standard_normal((n, d)))
            prop = _smooth_opt(prop, steps=3500, T0=0.02, T1=0.0002, lr0=0.06, lr1=0.004, mu=0.85, iu0=iu0, iu1=iu1)
            prop, pv = _hard_polish(prop, iters=300, step0=0.004, K=300, iu0=iu0, iu1=iu1)
        if pv < best_val - 1e-12:
            best = prop
            best_val = float(pv)
            sigma = max(0.008, sigma * 0.72)
        else:
            sigma = min(0.30, sigma * 1.07)
            # occasional sideways drift: keep near-best if close
            if pv < best_val + 0.002 and rng.random() < 0.08:
                # short chain from prop to diversify without losing time
                pass

    # final polish with tight focus
    best, best_val = _hard_polish(best, iters=800, step0=0.002, K=200, iu0=iu0, iu1=iu1)
    best = _smooth_opt(best, steps=800, T0=3e-4, T1=3e-5, lr0=0.01, lr1=0.0015, mu=0.8, iu0=iu0, iu1=iu1)
    best, best_val = _hard_polish(best, iters=800, step0=0.001, K=150, iu0=iu0, iu1=iu1)
    best = _norm(best)
    return best
