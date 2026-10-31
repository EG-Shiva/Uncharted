import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def _contact_counts(x, mc=None, tol=1e-9):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    if mc is None:
        mc = float(np.max(G))
    thr = mc - tol
    # count per point
    C = (G > thr)
    cnt = C.sum(axis=1)
    # tight pair indices
    iu0 = np.triu_indices(n, 1)
    selm = G[iu0] > thr
    return cnt, mc, iu0, selm


def _void_positions(x_fixed, k, rng, n_samples=15000, sep=0.90):
    # sample candidates on S^3, score by max dot to fixed set (lower is better)
    C = rng.standard_normal((n_samples, 4))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    G = C @ x_fixed.T
    m = G.max(axis=1)
    order = np.argsort(m)
    chosen = []
    for idx in order:
        if len(chosen) >= k:
            break
        c = C[idx]
        ok = True
        for j in chosen:
            if float(c @ C[j]) > sep:
                ok = False
                break
        if ok:
            chosen.append(idx)
        # safety: if too many rejections, relax
        if idx > 5000 and len(chosen) == 0:
            chosen.append(order[0])
            break
    if len(chosen) < k:
        # fill with best remaining
        for idx in order:
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) >= k:
                break
    return C[np.array(chosen[:k])]


def _soft_anneal(x, steps, beta0, beta1, lr0, mu):
    n, d = x.shape
    v = np.zeros_like(x)
    lb = np.log(beta1 / beta0) if beta1 > beta0 else 0.0
    for t in range(steps):
        frac = t / max(1, steps - 1)
        beta = beta0 * np.exp(lb * frac)
        lr = lr0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-300
        A = (W @ x) / S
        dot = (A * x).sum(axis=1, keepdims=True)
        Gt = A - dot * x
        v = mu * v + Gt
        vd = (v * x).sum(axis=1, keepdims=True)
        v = v - vd * x
        x = x - lr * v
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(nrm, 1e-12, out=nrm)
        x /= nrm
    return x


def _lp_polish(x, seconds, r0=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        m = cfull.max()
        sel = cfull > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]
        J = iu[1][sel]
        csel = cfull[sel]
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
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            # accept, tiny margin not needed
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
            # early exit if time almost up
            if time.time() - start >= seconds:
                break
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    BUDGET = 172.0
    try:
        xb = np.load("best.npy").astype(np.float64)
    except Exception:
        try:
            xb = np.load("record.npy").astype(np.float64)
        except Exception:
            rg0 = np.random.default_rng(0)
            xb = rg0.standard_normal((292, 4))
    if xb.shape != (292, 4):
        rg0 = np.random.default_rng(0)
        xb = rg0.standard_normal((292, 4))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best = _max_cos(xb)
    x_best = xb.copy()

    rng = np.random.default_rng(20261031)
    n, d = 292, 4

    explore_deadline = t0 + 112.0

    # presets for anneal diversity
    presets = [
        (900, 30.0, 400.0, 0.07, 0.85),
        (450, 80.0, 800.0, 0.045, 0.90),
        (1100, 20.0, 250.0, 0.09, 0.80),
        (600, 50.0, 600.0, 0.06, 0.88),
        (300, 120.0, 1000.0, 0.035, 0.92),
    ]
    sigmas = [0.004, 0.008, 0.015, 0.03, 0.06, 0.12]

    it = 0
    # quick evaluate contacts of best
    while True:
        if time.time() > explore_deadline:
            break
        if it >= 300:
            break
        # need at least ~2s for one iteration + final polish reserve
        if time.time() + 2.0 > t0 + BUDGET - 45.0 and it > 8:
            break
        it += 1

        cnt, mc, _, _ = _contact_counts(x_best)
        order = np.argsort(cnt)  # low first
        # choose victims
        mode = it % 5
        if mode == 0:
            # relocate rattlers + lows
            ratt = np.where(cnt == 0)[0].tolist()
            k = int(rng.integers(2, 6))
            # start with rattlers
            victims = ratt[:k]
            # fill from lowest contact points
            for o in order:
                if len(victims) >= k:
                    break
                if o not in victims:
                    victims.append(int(o))
            victims = np.array(victims[:k], dtype=int)
            sigma = float(sigmas[int(rng.integers(0, 3))])
        elif mode == 1:
            # attack hubs + random lows
            hubs = np.where(cnt >= 7)[0]
            k = int(rng.integers(2, 5))
            victims = []
            if len(hubs) > 0:
                nh = min(len(hubs), int(rng.integers(1, 3)))
                ch = rng.choice(hubs, size=nh, replace=False)
                victims.extend([int(c) for c in ch])
            while len(victims) < k:
                c = int(rng.integers(n))
                if c not in victims:
                    victims.append(c)
            victims = np.array(victims, dtype=int)
            sigma = float(sigmas[int(rng.integers(1, 4))])
        elif mode == 2:
            # random kill-reinsert
            k = int(rng.integers(1, 7))
            victims = rng.choice(n, size=k, replace=False)
            sigma = float(sigmas[int(rng.integers(0, 5))])
        elif mode == 3:
            # global jitter only, no kill
            k = 0
            victims = np.array([], dtype=int)
            sigma = float(sigmas[int(rng.integers(2, 6))])
        else:
            # low-contact focused
            k = int(rng.integers(1, 4))
            pool = order[:40]
            victims = rng.choice(pool, size=k, replace=False)
            sigma = float(sigmas[int(rng.integers(0, 4))])

        X = x_best.copy()
        if k > 0:
            mask = np.ones(n, dtype=bool)
            mask[victims] = False
            x_fixed = X[mask]
            # find voids
            try:
                V = _void_positions(x_fixed, k, rng, n_samples=15000)
            except Exception:
                V = rng.standard_normal((k, d))
                V /= np.linalg.norm(V, axis=1, keepdims=True)
            X[victims] = V
        # global tangential-ish jitter
        X = X + sigma * rng.standard_normal((n, d))
        X /= np.linalg.norm(X, axis=1, keepdims=True)

        # occasional full random restart
        if it % 17 == 0:
            R = rng.standard_normal((n, d))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            if rng.random() < 0.5:
                # heavy mix
                X = R
            else:
                # keep best but large shake already done; use R for half points
                idxh = rng.choice(n, size=n // 4, replace=False)
                X[idxh] = R[idxh]

        pre = presets[it % len(presets)]
        # vary steps slightly deterministically
        steps = pre[0] + int(rng.integers(-100, 101))
        steps = max(150, steps)
        try:
            X = _soft_anneal(X, steps, pre[1], pre[2], pre[3], pre[4])
        except Exception:
            pass
        # short LP evaluation
        remain = explore_deadline - time.time()
        short_t = min(1.6, max(0.6, remain / max(1, (30 - it % 30))))
        if short_t < 0.4:
            short_t = 0.4
        try:
            X = _lp_polish(X, short_t, r0=8e-4)
        except Exception:
            pass
        v = _max_cos(X)
        if v < best - 1e-13:
            best = v
            x_best = X.copy()
        # also accept equal-but-different to move across plateau rarely
        elif v < best + 1e-12 and rng.random() < 0.02:
            # check topology difference: different contact set size
            try:
                cnt2, _, _, _ = _contact_counts(X)
                if int(cnt2.sum()) != int(cnt.sum()):
                    x_best = X.copy()
                    best = v if v < best else best
            except Exception:
                pass

    # final long polish of best
    time_left = t0 + BUDGET - time.time() - 1.5
    if time_left > 3.0:
        try:
            # two-stage: moderate radius then tiny radius for precision
            half = time_left * 0.6
            x_best = _lp_polish(x_best, half, r0=1e-3)
            time_left2 = t0 + BUDGET - time.time() - 1.0
            if time_left2 > 2.0:
                x_best = _lp_polish(x_best, time_left2, r0=5e-4)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
