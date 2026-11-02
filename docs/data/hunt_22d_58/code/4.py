import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def lp_polish(x, seconds):
    """Trust-region LP: move points tangentially to lower the largest cosine."""
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def quench(X0, steps=1200, beta0=150.0, beta1=5000.0, lr0=0.02, lr1=0.001, mu=0.85):
    X = X0.copy()
    V = np.zeros_like(X)
    for s in range(steps):
        frac = s / (steps - 1) if steps > 1 else 1.0
        beta = beta0 * (beta1 / beta0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = np.max(G)
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        s_half = np.sum(E) * 0.5
        if not np.isfinite(s_half) or s_half <= 0:
            continue
        W = E / s_half
        grad = W @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        # clip extreme tangential grads for stability
        nrm = np.linalg.norm(gt, axis=1, keepdims=True)
        # scale down huge grads (rare)
        scale = np.maximum(1.0, nrm / 0.5)
        gt = gt / scale
        V = mu * V - lr * gt
        X = X + V
        nn = np.linalg.norm(X, axis=1, keepdims=True)
        nn = np.maximum(nn, 1e-12)
        X = X / nn
    return X


def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.default_rng(0)

    # load starting points
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None

    cands = []
    if xb is not None:
        cands.append(np.array(xb, dtype=float))
    if xr is not None:
        cands.append(np.array(xr, dtype=float))
    if len(cands) == 0:
        cands.append(rng.normal(size=(58, 22)))
    # also random init as diversity
    # pick best among loads
    best = None
    best_val = 1e9
    for c in cands:
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        v = max_cos(c)
        if v < best_val:
            best_val = v
            best = c.copy()

    n, d = best.shape  # expect 58,22
    # contact counts to find overloaded hubs
    def contact_counts(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = np.max(G)
        C = (G > m - 1e-9).astype(int)
        # C symmetric, sum rows counts contacts (each pair counted once per row)
        return np.sum(C, axis=1), m

    counts, _ = contact_counts(best)
    order = np.argsort(-counts)
    hubs = order[:8].tolist()

    sigmas = [0.006, 0.012, 0.022, 0.035, 0.06]
    trial = 0
    # reserve ~28s for final polish
    explore_end = deadline - 28.0

    # quick initial quench of best to ensure smooth baseline (short)
    # skip to save time; go directly to hopping

    while time.time() < explore_end:
        trial += 1
        sigma = sigmas[(trial - 1) % len(sigmas)]
        Xp = best.copy()
        # global shake
        noise = rng.normal(size=Xp.shape)
        # boost hubs
        boost = np.ones((n, 1))
        # pick 2-3 hubs to boost this trial
        k = 1 + (trial % 3)
        idx = rng.choice(hubs, size=k, replace=False) if len(hubs) >= k else np.arange(k)
        for ii in idx:
            boost[ii, 0] = 3.0
        # occasionally kill-and-replace 1-2 points fully
        if trial % 11 == 0:
            kill = rng.choice(n, size=2, replace=False)
            for kk in kill:
                v = rng.normal(size=(d,))
                v /= np.linalg.norm(v)
                Xp[kk] = v
            # add small noise to rest
            Xp = Xp + (sigma * 0.5) * noise * boost
        elif trial % 17 == 0:
            kill = rng.choice(n, size=1, replace=False)
            for kk in kill:
                v = rng.normal(size=(d,))
                v /= np.linalg.norm(v)
                Xp[kk] = v
            Xp = Xp + (sigma * 0.5) * noise * boost
        else:
            Xp = Xp + sigma * noise * boost
        Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)

        # vary quench length / beta slightly for diversity
        steps = 900 if (trial % 2 == 0) else 1300
        if sigma >= 0.035:
            steps = 1500
        b1 = 4000.0 if (trial % 3 == 0) else 6000.0
        Xq = quench(Xp, steps=steps, beta0=150.0, beta1=b1, lr0=0.02, lr1=0.0008, mu=0.85)
        vq = max_cos(Xq)
        if vq < best_val:
            # promising: short LP refine (2s) to verify true local optimum
            now = time.time()
            budget = min(2.5, max(0.5, explore_end - now - 0.5))
            if budget > 0.4:
                Xp2 = lp_polish(Xq, budget)
                v2 = max_cos(Xp2)
            else:
                Xp2, v2 = Xq, vq
            if v2 < best_val - 1e-12:
                best, best_val = Xp2, v2
                counts, _ = contact_counts(best)
                order = np.argsort(-counts)
                hubs = order[:8].tolist()
        # safety: stop if time nearly up
        if time.time() >= explore_end:
            break
        # every 40 trials, also try pure exploitation: tiny shake + quench best
        # (covered by sigma cycle)

    # final polish with remaining time
    rem = deadline - time.time() - 1.0
    if rem > 2.0:
        rem = min(rem, 27.0)
        best = lp_polish(best, rem)
        best_val = max_cos(best)

    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    # shrink tiny margin not needed; ensure finite
    best = np.where(np.isfinite(best), best, 0.0)
    # renormalize after fix
    nn = np.linalg.norm(best, axis=1, keepdims=True)
    nn = np.maximum(nn, 1e-12)
    best = best / nn
    return best
