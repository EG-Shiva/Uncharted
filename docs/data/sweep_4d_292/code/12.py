import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _soft_anneal(x, steps, beta0, beta1, lr0, mu, rng=None, noise=0.0):
    n, d = x.shape
    v = np.zeros_like(x)
    lb = np.log(beta1 / beta0) if beta1 > beta0 else 0.0
    for t in range(steps):
        frac = t / max(1, steps - 1)
        beta = beta0 * np.exp(lb * frac)
        lr = lr0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(np.max(G))
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-300
        A = (W @ x) / S
        dot = (A * x).sum(axis=1, keepdims=True)
        Gt = A - dot * x
        v = mu * v + Gt
        vd = (v * x).sum(axis=1, keepdims=True)
        v -= vd * x
        x = x - lr * v
        if noise > 0.0 and rng is not None:
            # decaying tangential noise
            amp = noise * (1.0 - frac)
            N = rng.standard_normal((n, d))
            Nd = (N * x).sum(axis=1, keepdims=True)
            N -= Nd * x
            x = x + amp * N
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(nrm, 1e-12, out=nrm)
        x /= nrm
    return x


def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
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
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _best_void_point(Xkeep, rng, n_cand=6000, refine=60, lr0=0.2):
    # sample candidates, pick most distant, then hill-climb single point
    d = Xkeep.shape[1]
    cand = rng.standard_normal((n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True) + 1e-12
    dots = cand @ Xkeep.T
    md = dots.max(axis=1)
    y = cand[int(np.argmin(md))].copy()
    lr = lr0
    for it in range(refine):
        dv = Xkeep @ y
        j = int(np.argmax(dv))
        dj = float(dv[j])
        g = Xkeep[j] - dj * y
        y = y - lr * g
        nrm = float(np.linalg.norm(y))
        if nrm < 1e-12:
            y = rng.standard_normal(d)
            nrm = float(np.linalg.norm(y))
        y /= nrm
        lr *= 0.97
    return y


def _contact_degrees(x, thresh=1e-9):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    B = G > m - thresh
    deg = B.sum(axis=1)
    return deg, m


def solve():
    t0 = time.time()
    TIME_LIMIT = 172.0
    try:
        xb = np.load("best.npy").astype(np.float64)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(np.float64)
    except Exception:
        xr = None
    cands = []
    for z in (xb, xr):
        if z is not None and z.shape == (292, 4):
            zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
            if np.all(np.isfinite(zn)):
                cands.append(zn)
    if not cands:
        rng0 = np.random.default_rng(0)
        z = rng0.standard_normal((292, 4))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    vals = [_max_cos(z) for z in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    best = float(vals[bi])

    rng = np.random.default_rng(20261031)
    n, d = 292, 4
    explore_deadline = t0 + TIME_LIMIT - 32.0
    it = 0
    sigmas = [0.008, 0.02, 0.05, 0.12, 0.30]
    while time.time() < explore_deadline:
        it += 1
        if time.time() + 0.4 > explore_deadline and it > 5:
            break
        X = x_best.copy()
        rtype = rng.random()
        if rtype < 0.38:
            # shake + random kill-reinsert
            sigma = sigmas[int(rng.integers(len(sigmas)))]
            X = X + sigma * rng.standard_normal((n, d))
            if rng.random() < 0.7:
                k = int(rng.integers(1, 9))
                idx = rng.choice(n, size=k, replace=False)
                R = rng.standard_normal((k, d))
                R /= np.linalg.norm(R, axis=1, keepdims=True) + 1e-12
                X[idx] = R
            X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        elif rtype < 0.72:
            # region wipe + greedy void refill
            center = int(rng.integers(n))
            dots_c = X @ X[center]
            mwipe = int(rng.integers(6, 15))
            order = np.argsort(-dots_c)
            wipe = order[:mwipe]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[wipe] = False
            Xkeep = X[keep_mask]
            # greedily refill one by one with void search
            newpts = np.empty((mwipe, d))
            cur = Xkeep
            for s in range(mwipe):
                y = _best_void_point(cur, rng, n_cand=5000, refine=50)
                newpts[s] = y
                cur = np.vstack([cur, y[None, :]])
            X[wipe] = newpts
            # small shake to break symmetry
            X = X + 0.012 * rng.standard_normal((n, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        elif rtype < 0.88:
            # rattler / low-degree relocation to voids
            try:
                deg, mcur = _contact_degrees(X, thresh=1e-7)
            except Exception:
                deg = np.full(n, 5)
            # pick lowest degree points
            k = int(rng.integers(1, 5))
            order = np.argsort(deg)
            # only relocate if degree <= 2, else pick random few
            cand_idx = order[:8]
            # choose k among lowest
            sel = rng.choice(cand_idx, size=min(k, len(cand_idx)), replace=False)
            # if degrees not low, still relocate one random rattler-like
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[sel] = False
            Xkeep = X[keep_mask]
            cur = Xkeep
            for ii, idx in enumerate(sel):
                y = _best_void_point(cur, rng, n_cand=6000, refine=60)
                X[idx] = y
                cur = np.vstack([cur, y[None, :]])
            X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        else:
            # correlated patch jitter: move a cap coherently + individual noise
            center = int(rng.integers(n))
            dots_c = X @ X[center]
            cap_size = int(rng.integers(12, 30))
            order = np.argsort(-dots_c)
            cap = order[:cap_size]
            shift = rng.standard_normal(d)
            shift = shift / (np.linalg.norm(shift) + 1e-12) * float(rng.uniform(0.02, 0.10))
            X[cap] = X[cap] + shift[None, :] + 0.015 * rng.standard_normal((cap_size, d))
            # also global tiny shake
            X = X + 0.005 * rng.standard_normal((n, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12

        # anneal hyperparams
        rr = rng.random()
        if rr < 0.35:
            steps, b0, b1, lr0, mu, nz = 1200, 25.0, 350.0, 0.06, 0.85, 0.0
        elif rr < 0.65:
            steps, b0, b1, lr0, mu, nz = 1700, 15.0, 250.0, 0.10, 0.88, 0.002
        elif rr < 0.85:
            steps, b0, b1, lr0, mu, nz = 2200, 40.0, 500.0, 0.04, 0.80, 0.0
        else:
            steps, b0, b1, lr0, mu, nz = 900, 10.0, 150.0, 0.15, 0.90, 0.005
        # shorten if little time
        t_rem = explore_deadline - time.time()
        est = steps * 0.00025
        if est > t_rem:
            steps = max(400, int(t_rem / 0.00025))
            if steps <= 120:
                break
        Xo = _soft_anneal(X, steps, b0, b1, lr0, mu, rng=rng, noise=nz)
        if not np.all(np.isfinite(Xo)):
            continue
        val_raw = _max_cos(Xo)
        # short LP probe on promising or alternating iterations
        do_short = (val_raw < best + 0.004) or (it % 2 == 0)
        if do_short:
            # budget ~0.7s
            try:
                Xp = _lp_polish(Xo, 0.7)
            except Exception:
                Xp = Xo
            if not np.all(np.isfinite(Xp)):
                Xp = Xo
            val = _max_cos(Xp)
            # keep polished version if better than raw
            if val > val_raw and val_raw < best:
                Xp, val = Xo, val_raw
        else:
            Xp, val = Xo, val_raw
        if val < best - 1e-12:
            best = float(val)
            x_best = (Xp / (np.linalg.norm(Xp, axis=1, keepdims=True) + 1e-12)).copy()

    time_left = TIME_LIMIT - (time.time() - t0) - 1.0
    time_left = max(5.0, min(34.0, time_left))
    try:
        x_best = _lp_polish(x_best, time_left)
    except Exception:
        pass
    x_best = x_best / (np.linalg.norm(x_best, axis=1, keepdims=True) + 1e-12)
    if not np.all(np.isfinite(x_best)):
        x_best = cands[bi] / (np.linalg.norm(cands[bi], axis=1, keepdims=True) + 1e-12)
    return x_best
