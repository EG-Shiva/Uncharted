import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def lp_polish(x, seconds, t_deadline=None):
    """Trust-region LP: move points tangentially to lower largest cosine."""
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if t_deadline is not None and time.time() >= t_deadline:
            break
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
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
        obj = np.zeros(nv)
        obj[-1] = 1.0
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


def smooth_quench(X, steps, lr0, T0, T1, noise0, noise1, rng, max_step=0.02):
    n, d = X.shape
    for s in range(steps):
        frac = s / max(1, steps - 1) if steps > 1 else 1.0
        T = T0 * (1.0 - frac) + T1 * frac
        noise_amp = noise0 * (1.0 - frac) + noise1 * frac
        lr = lr0 * (1.0 - 0.8 * frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(np.max(G))
        # weights focused on large cosines
        W = np.exp((G - gmax) / T)
        np.fill_diagonal(W, 0.0)
        S = float(np.sum(W) * 0.5)
        if S < 1e-30:
            gt = np.zeros_like(X)
        else:
            W /= S
            grad = W @ X
            dot = np.sum(grad * X, axis=1, keepdims=True)
            gt = grad - dot * X
        step = lr * gt
        norms = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        scale = np.minimum(1.0, max_step / norms)
        step = step * scale
        X = X - step
        if noise_amp > 0:
            N = rng.standard_normal(size=X.shape)
            ndot = np.sum(N * X, axis=1, keepdims=True)
            Nt = N - ndot * X
            X = X + (noise_amp * 0.5) * Nt
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X


def best_hole_candidates(X, exclude_idx, rng, n_cand=8000):
    n, d = X.shape
    cand = rng.standard_normal(size=(n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    S = cand @ X.T
    if exclude_idx is not None:
        # allow list or single
        S[:, exclude_idx] = -2.0
    vals = np.max(S, axis=1)
    return cand[np.argmin(vals)]


def solve():
    t0 = time.time()
    HARD_DEADLINE = t0 + 172.0
    EXPL_END = t0 + 122.0
    rng = np.random.default_rng(12345)
    np.random.seed(0)

    # load starting points
    X0 = None
    best_val0 = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (272, 4):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            v = max_cos(a)
            if v < best_val0:
                best_val0 = v
                X0 = a.copy()
        except Exception:
            continue
    if X0 is None:
        X0 = rng.standard_normal(size=(272, 4))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        best_val0 = max_cos(X0)

    Xbest = X0.copy()
    best_val = float(best_val0)

    # initial light smooth to have a clean reference (short)
    # no heavy work here; go directly to basin hopping

    trial = 0
    sigmas = [0.05, 0.09, 0.14, 0.22]
    while time.time() < EXPL_END and time.time() < HARD_DEADLINE - 45:
        trial += 1
        mode = trial % 6
        X = Xbest.copy()
        n = X.shape[0]
        if mode == 0:
            # relieve most jammed points: replace worst row-max with holes
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            rowmax = np.max(G, axis=1)
            worst = np.argsort(rowmax)[-3:]
            for idx in worst:
                h = best_hole_candidates(X, idx, rng, n_cand=8000)
                h = h + 0.002 * rng.standard_normal(4)
                h /= np.linalg.norm(h)
                X[idx] = h
            X = X + 0.02 * rng.standard_normal(size=X.shape)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        elif mode == 1:
            # relocate rattlers / low-contact points to largest holes
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            gmax = float(np.max(G))
            contacts = np.sum(G > gmax - 1e-7, axis=1)
            ratt = np.where(contacts <= 1)[0]
            if len(ratt) > 0:
                tod = ratt[:4]
                for idx in tod:
                    h = best_hole_candidates(X, idx, rng, n_cand=8000)
                    h = h + 0.002 * rng.standard_normal(4)
                    h /= np.linalg.norm(h)
                    X[idx] = h
                X = X + 0.015 * rng.standard_normal(size=X.shape)
                X /= np.linalg.norm(X, axis=1, keepdims=True)
            else:
                sigma = sigmas[trial % len(sigmas)]
                X = X + sigma * rng.standard_normal(size=X.shape)
                X /= np.linalg.norm(X, axis=1, keepdims=True)
        elif mode == 2:
            # strong subset kick: topology change in a patch
            sel = rng.choice(n, size=40, replace=False)
            X[sel] = X[sel] + 0.28 * rng.standard_normal(size=(40, 4))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        else:
            sigma = sigmas[trial % len(sigmas)]
            # alternate global vs single-pair kick
            if mode == 5:
                G = X @ X.T
                np.fill_diagonal(G, -2.0)
                ij = int(np.argmax(G))
                ii, jj = ij // n, ij % n
                # move one endpoint of max pair to a hole
                h = best_hole_candidates(X, jj, rng, n_cand=8000)
                X[jj] = h
                X = X + 0.03 * rng.standard_normal(size=X.shape)
                X /= np.linalg.norm(X, axis=1, keepdims=True)
            else:
                X = X + sigma * rng.standard_normal(size=X.shape)
                X /= np.linalg.norm(X, axis=1, keepdims=True)

        # quench with Langevin LogSumExp
        # adapt steps to remaining time
        remain = min(EXPL_END - time.time(), HARD_DEADLINE - 45 - time.time())
        if remain < 1.0:
            break
        steps = 800
        if remain < 8:
            steps = 350
        elif remain < 20:
            steps = 550
        Xq = smooth_quench(X, steps=steps, lr0=12.0, T0=0.015, T1=0.002,
                           noise0=0.014, noise1=0.0, rng=rng, max_step=0.02)
        v = max_cos(Xq)
        if v < best_val:
            Xbest = Xq
            best_val = v
            # early exit if big win? continue to polish further
            # opportunistic short LP if promising and time allows
            if best_val < 0.8975 and time.time() < EXPL_END - 10:
                Xlp = lp_polish(Xbest, seconds=3.0, t_deadline=HARD_DEADLINE - 30)
                vlp = max_cos(Xlp)
                if vlp < best_val:
                    Xbest, best_val = Xlp, vlp

    # final trust-region LP polish with remaining budget
    remain = HARD_DEADLINE - time.time() - 2.0
    if remain > 5:
        secs = min(remain, 45.0)
        Xp = lp_polish(Xbest, seconds=secs, t_deadline=HARD_DEADLINE - 1.0)
        vp = max_cos(Xp)
        if vp < best_val and np.all(np.isfinite(Xp)):
            Xbest, best_val = Xp, vp

    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    # safety: replace any non-finite row
    bad = ~np.all(np.isfinite(Xbest), axis=1)
    if np.any(bad):
        for i in np.where(bad)[0]:
            v = rng.standard_normal(4)
            v /= np.linalg.norm(v)
            Xbest[i] = v
    return Xbest
