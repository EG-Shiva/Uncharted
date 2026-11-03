import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _anneal_lse(x, steps, beta0, beta1, lr0, lr1, rng=None, jitter=0.0):
    n, d = x.shape
    # optional tiny symmetry-breaking jitter already applied outside
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = beta0 + (beta1 - beta0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        # softmax weights, symmetry-free (no symmetrisation of X)
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ x
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Tan = Grad - dot * x
        # normalise step by max norm to stabilise
        tn = np.sqrt((Tan * Tan).sum(axis=1, keepdims=True)) + 1e-30
        # clip extreme
        x = x - lr * Tan / np.maximum(1.0, tn * 5.0)
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-30)
    return x


def _lp_polish(x, deadline, radius0=8e-3, max_pairs=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        c = Gfull[iu]
        m = c.max()
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > max_pairs:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:max_pairs]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1
        ).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(
                obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                b_eq=np.zeros(n),
                bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                method="highs", options={"presolve": True})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-30)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _lp_restarted(x, deadline, max_pairs=2500):
    # two restarts of trust region from same basin
    x = _lp_polish(x, deadline, radius0=8e-3, max_pairs=max_pairs)
    if time.time() >= deadline:
        return x
    x = _lp_polish(x, deadline, radius0=2e-3, max_pairs=max_pairs)
    return x


def _fiedler_split(X, thr=1e-5, rng=None):
    n, d = X.shape
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    A = (G > (m - thr)).astype(float)
    np.fill_diagonal(A, 0.0)
    # symmetrise
    A = np.maximum(A, A.T)
    deg = A.sum(axis=1)
    # isolated nodes: connect weakly to avoid trivial split failure
    # Laplacian
    L = np.diag(deg) - A
    try:
        w, V = np.linalg.eigh(L)
    except Exception:
        # fallback random split
        s = np.ones(n)
        s[n // 2:] = -1.0
        if rng is not None:
            rng.shuffle(s)
        u = rng.normal(size=(d,)) if rng is not None else np.ones(d)
        u = u / (np.linalg.norm(u) + 1e-30)
        return s, u, None
    # second smallest eigenvector (Fiedler)
    # w already ascending for eigh
    if n >= 2:
        f = V[:, 1]
    else:
        f = V[:, 0]
    med = np.median(f)
    s = np.where(f > med, 1.0, -1.0)
    # enforce balance if degenerate (all same side)
    if np.all(s > 0) or np.all(s < 0):
        idx = np.argsort(f)
        s = np.ones(n) * -1.0
        s[idx[n // 2:]] = 1.0
    # separation direction = mean(A)-mean(B)
    try:
        mA = X[s > 0].mean(axis=0)
        mB = X[s < 0].mean(axis=0)
        u = mA - mB
        nu = np.linalg.norm(u)
        if nu < 1e-12 or not np.isfinite(nu):
            raise ValueError("degenerate u")
        u = u / nu
    except Exception:
        if rng is None:
            u = np.zeros(d)
            u[0] = 1.0
        else:
            u = rng.normal(size=(d,))
            u = u / (np.linalg.norm(u) + 1e-30)
    return s, u, f


def _displace_sides(X, s, u, amp, rng, noise=0.0015):
    n, d = X.shape
    dots = X @ u  # (n,)
    T = u[None, :] - dots[:, None] * X  # projection of u onto tangent
    tn = np.linalg.norm(T, axis=1, keepdims=True) + 1e-30
    That = T / tn
    # where tangent tiny, use random tangent
    tiny = (tn[:, 0] < 1e-9)
    if np.any(tiny):
        R = rng.normal(size=(int(np.count_nonzero(tiny)), d))
        # project
        Xi = X[tiny]
        R = R - (R * Xi).sum(axis=1, keepdims=True) * Xi
        rn = np.linalg.norm(R, axis=1, keepdims=True) + 1e-30
        That[tiny] = R / rn
    step = amp * s[:, None] * That
    # small symmetry-free noise
    if noise > 0:
        Rn = rng.normal(size=(n, d))
        Rn = Rn - (Rn * X).sum(axis=1, keepdims=True) * X
        step = step + noise * Rn / (np.linalg.norm(Rn, axis=1, keepdims=True) + 1e-30)
    Y = X + step
    Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-30)
    return Y


def solve():
    rng = np.random.default_rng(12345)
    t0 = time.time()
    deadline = t0 + 168.0
    n, d = 138, 8
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-30)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = rng.normal(size=(n, d))
        best = best / (np.linalg.norm(best, axis=1, keepdims=True) + 1e-30)
        best_val = _max_cos(best)

    # quick initial tighten so Fiedler graph is meaningful
    try:
        if time.time() < deadline - 60:
            best = _lp_polish(best, min(deadline - 55, time.time() + 2.0), radius0=5e-3)
            best_val = _max_cos(best)
    except Exception:
        pass

    amps = [0.01, 0.03, 0.08, 0.15, 0.25]
    outer = 0
    # alternate thresholds to handle varying jam width
    thrs = [1e-5, 3e-5, 1e-4]

    while time.time() < deadline - 10.0 and outer < 60:
        outer += 1
        thr = thrs[(outer - 1) % len(thrs)]
        try:
            s, u, f = _fiedler_split(best, thr=thr, rng=rng)
        except Exception:
            continue
        base = best.copy()
        # for diversity, on alternate outers randomise u slightly
        # (still opposite tangent push, but different direction)
        # keep first outer exactly as prescribed
        u_list = [u]
        # we keep prescribed u for all amps; randomness comes from noise
        for amp in amps:
            if time.time() > deadline - 10.0:
                break
            try:
                y = _displace_sides(base, s, u, amp, rng, noise=0.0015)
                # short symmetry-free LSE re-jam
                y1 = _anneal_lse(y, steps=450, beta0=15.0, beta1=350.0,
                                 lr0=0.020, lr1=0.0012)
                v1 = _max_cos(y1)
                if v1 < best_val - 1e-12:
                    best_val = v1
                    best = y1.copy()
                # one restarted-trust-region LP polish from re-jammed point
                # give each amp ~0.9s, share remaining time
                lp_dead = min(deadline - 9.0, time.time() + 0.9)
                if lp_dead > time.time() + 0.05:
                    y2 = _lp_restarted(y1, lp_dead, max_pairs=2500)
                    v2 = _max_cos(y2)
                    if v2 < best_val - 1e-12:
                        best_val = v2
                        best = y2.copy()
                # also evaluate the raw displaced point (rarely better but cheap)
                # no extra cost: already have y
            except Exception:
                continue
        # if we improved, immediately recompute Fiedler (new basin topology)
        # else add a mild global shake to the Fiedler direction for next round:
        # rotate u with small random component to try nearby cut
        if time.time() < deadline - 10.0 and outer % 3 == 0:
            # occasional diversification: small random tangent shake of best
            # without destroying core, to get a different Fiedler cut next time
            try:
                y = best + 0.008 * rng.normal(size=(n, d))
                y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-30)
                y = _anneal_lse(y, steps=250, beta0=30.0, beta1=300.0,
                                lr0=0.012, lr1=0.001)
                v = _max_cos(y)
                if v < best_val + 0.005:
                    y = _lp_polish(y, min(deadline - 9.0, time.time() + 0.6),
                                   radius0=5e-3, max_pairs=2200)
                    v = _max_cos(y)
                    if v < best_val - 1e-12:
                        best_val = v
                        best = y.copy()
            except Exception:
                pass

    try:
        best = _lp_restarted(best, min(deadline - 0.5, time.time() + 7.0), max_pairs=3500)
    except Exception:
        pass
    best = best / (np.linalg.norm(best, axis=1, keepdims=True) + 1e-30)
    # strict finiteness guard
    best[~np.isfinite(best)] = 0.125
    best = best / (np.linalg.norm(best, axis=1, keepdims=True) + 1e-30)
    return best
