import time
import numpy as np

try:
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    _HAS_LP = True
except Exception:
    _HAS_LP = False


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _per_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G


def _smooth_polish(X, steps, beta0, beta1, lr0, lr1, mu=0.88):
    n, d = X.shape
    V = np.zeros_like(X)
    lbeta0 = np.log(beta0)
    lbeta1 = np.log(beta1)
    llr0 = np.log(lr0)
    llr1 = np.log(lr1)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = float(np.exp(lbeta0 + (lbeta1 - lbeta0) * f))
        lr = float(np.exp(llr0 + (llr1 - llr0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True) + 1e-12
        Gr = E @ X
        dot = (Gr * X).sum(axis=1, keepdims=True)
        Gt = (Gr - dot * X) / rs
        # clip per-row to avoid spikes
        gn = np.sqrt((Gt * Gt).sum(axis=1, keepdims=True) + 1e-12)
        clip = np.minimum(1.0, 1.5 / (gn + 1e-12))
        Gt = Gt * clip
        V = mu * V + (1.0 - mu) * Gt
        bc = 1.0 - mu ** (s + 1)
        Vhat = V / bc
        X = X - lr * Vhat
        nr = np.sqrt((X * X).sum(axis=1, keepdims=True))
        nr[nr == 0] = 1.0
        X = X / nr
    return X


def _lp_polish(x, seconds):
    if not _HAS_LP:
        return x
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape

    def _mc(a):
        g = a @ a.T
        np.fill_diagonal(g, -2.0)
        return float(g.max())

    best = _mc(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        xxT = x @ x.T
        cfull = xxT[iu]
        m = float(cfull.max())
        sel = cfull > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cfull[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y = y / nr
        v = _mc(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _greedy_hole_vec(X_rem, rng, ncand=4000):
    d = X_rem.shape[1]
    R = rng.randn(ncand, d)
    R /= np.linalg.norm(R, axis=1, keepdims=True) + 1e-12
    S = R @ X_rem.T
    scores = S.max(axis=1)
    return R[int(np.argmin(scores))]


def solve():
    t0 = time.time()
    deadline = t0 + 175.0
    loop_end = t0 + 142.0
    rng = np.random.RandomState(0)
    np.random.seed(0)
    n, d = 82, 14

    Xinit = None
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d):
                Xinit = a.astype(float)
                break
        except Exception:
            continue
    if Xinit is None:
        Xinit = rng.randn(n, d)
    Xinit = Xinit / np.linalg.norm(Xinit, axis=1, keepdims=True)

    best = Xinit.copy()
    best_val = _max_cos(best)

    # quick refine of loaded point is already done; keep it
    trial = 0
    while time.time() < loop_end:
        remain = deadline - time.time()
        if remain < 8:
            break
        Y = best.copy()
        # diagnostics on current best
        try:
            per, Gfull = _per_max(best)
            m0 = float(per.max())
            cnt = (Gfull > m0 - 0.002).sum(axis=1)
            hubs = np.argsort(-cnt)[:8]
            rattlers = np.where(per < m0 - 1e-4)[0]
            worst = int(np.argmax(per))
            worst2 = np.argsort(-per)[:2]
        except Exception:
            hubs = np.array([0, 1, 2, 3, 4, 5, 6, 7])
            rattlers = np.array([], dtype=int)
            worst = 0
            worst2 = np.array([0, 1])

        mode = trial % 9
        # default polish params
        if mode == 0:
            Y = Y + 0.0025 * rng.randn(n, d)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Y = _smooth_polish(Y, 2200, 100.0, 3000.0, 0.012, 0.0003)
        elif mode == 1:
            Y = Y + 0.010 * rng.randn(n, d)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Y = _smooth_polish(Y, 2000, 80.0, 2200.0, 0.018, 0.0005)
        elif mode == 2:
            Y = Y + 0.035 * rng.randn(n, d)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Y = _smooth_polish(Y, 2000, 60.0, 1600.0, 0.025, 0.0006)
        elif mode == 3:
            # hub shake
            noise = 0.004 * rng.randn(n, d)
            noise[hubs] = 0.060 * rng.randn(len(hubs), d)
            Y = Y + noise
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Y = _smooth_polish(Y, 2000, 80.0, 2500.0, 0.020, 0.0005)
        elif mode == 4:
            # rattler teleport + small shake
            if len(rattlers) > 0:
                for idx in rattlers[:8]:
                    v = rng.randn(d)
                    v /= np.linalg.norm(v)
                    Y[idx] = v
            else:
                # teleport loosest 5
                order = np.argsort(per)
                for idx in order[:5]:
                    v = rng.randn(d)
                    v /= np.linalg.norm(v)
                    Y[idx] = v
            Y = Y + 0.005 * rng.randn(n, d)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Y = _smooth_polish(Y, 2000, 70.0, 2200.0, 0.020, 0.0005)
        elif mode == 5:
            # worst hole reinsertion (greedy best-of-many)
            for w in worst2:
                Xrem = np.delete(Y, w, axis=0)
                Y[w] = _greedy_hole_vec(Xrem, rng, ncand=4000)
            Y = Y + 0.003 * rng.randn(n, d)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Y = _smooth_polish(Y, 1800, 90.0, 2600.0, 0.014, 0.0004)
        elif mode == 6:
            # random subset reinit
            k = 7
            sel = rng.choice(n, size=k, replace=False)
            for idx in sel:
                v = rng.randn(d)
                v /= np.linalg.norm(v)
                Y[idx] = v
            Y = _smooth_polish(Y, 2200, 60.0, 1800.0, 0.025, 0.0006)
        elif mode == 7:
            Y = Y + 0.10 * rng.randn(n, d)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Y = _smooth_polish(Y, 2400, 50.0, 1200.0, 0.030, 0.0008)
        else:  # mode 8
            if trial % 36 == 8:
                # rare fresh start
                Y = rng.randn(n, d)
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                Y = _smooth_polish(Y, 2600, 50.0, 1500.0, 0.030, 0.0007)
            else:
                # hub + rattler combo
                noise = 0.006 * rng.randn(n, d)
                noise[hubs] = 0.035 * rng.randn(len(hubs), d)
                Y = Y + noise
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                if len(rattlers) > 0:
                    # relocate one rattler greedily
                    r0 = int(rattlers[rng.randint(len(rattlers))]) if len(rattlers) > 0 else int(rng.randint(n))
                    Xrem = np.delete(Y, r0, axis=0)
                    Y[r0] = _greedy_hole_vec(Xrem, rng, ncand=3000)
                Y = _smooth_polish(Y, 2000, 80.0, 2400.0, 0.018, 0.0004)

        v = _max_cos(Y)
        # snap promising improvements with short LP
        if v < best_val:
            # short LP snap if time allows
            if deadline - time.time() > 10:
                try:
                    Y2 = _lp_polish(Y, min(3.0, max(0.8, (deadline - time.time()) * 0.05)))
                    v2 = _max_cos(Y2)
                    if v2 < v:
                        Y, v = Y2, v2
                except Exception:
                    pass
            if v < best_val:
                best, best_val = Y, v
        trial += 1
        # safety: if many trials, continue until time
        if trial > 500:
            break

    # final LP polish with remaining time
    rem = deadline - time.time() - 1.0
    if rem > 2 and _HAS_LP:
        try:
            Yf = _lp_polish(best, min(rem, 30.0))
            vf = _max_cos(Yf)
            if vf < best_val:
                best, best_val = Yf, vf
        except Exception:
            pass

    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
