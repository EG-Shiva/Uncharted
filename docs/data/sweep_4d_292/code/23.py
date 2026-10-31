import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _anneal_repel(x, steps, beta0, beta1, lr0, mu, noise0, rng):
    n, d = x.shape
    v = np.zeros_like(x)
    # preallocate to avoid allocs? simple loop
    for t in range(steps):
        frac = t / max(1, steps - 1)
        beta = beta0 * ((beta1 / beta0) ** frac)
        lr = lr0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        # exp in safe range (<=0)
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-300
        A = (W @ x) / S
        dot = (A * x).sum(axis=1, keepdims=True)
        Gt = A - dot * x
        v = mu * v + (1.0 - mu) * Gt
        # keep momentum tangential
        vd = (v * x).sum(axis=1, keepdims=True)
        v -= vd * x
        x = x - lr * v
        if noise0 > 0:
            amp = noise0 * (1.0 - frac)
            z = rng.standard_normal((n, d))
            zd = (z * x).sum(axis=1, keepdims=True)
            z -= zd * x
            x = x + amp * z
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(nrm, 1e-12, out=nrm)
        x /= nrm
    return x


def _cheap_tighten(x, steps, lr0):
    n, d = x.shape
    lr = lr0
    for t in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        band = 0.004 * (1.0 - 0.8 * t / max(1, steps - 1))
        Mask = G > m - band
        W = np.zeros_like(G)
        W[Mask] = 1.0
        np.fill_diagonal(W, 0.0)
        deg = W.sum(axis=1, keepdims=True)
        has = deg.ravel() > 0.5
        if not np.any(has):
            break
        deg = deg + 1e-300
        A = (W @ x) / deg
        dot = (A * x).sum(axis=1, keepdims=True)
        Gt = A - dot * x
        Gt[~has] = 0.0
        x = x - lr * Gt
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        lr *= 0.998
    return x


def _find_holes(x_keep, k, rng, nsamp=30000, sep=0.88):
    d = x_keep.shape[1]
    # sample in chunks
    nc = 3
    per = nsamp // nc
    all_c = np.empty((nsamp, d))
    all_s = np.empty(nsamp)
    pos = 0
    for _ in range(nc):
        c = rng.standard_normal((per, d))
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        dots = c @ x_keep.T
        sc = dots.max(axis=1)
        all_c[pos:pos+per] = c
        all_s[pos:pos+per] = sc
        pos += per
    # leftover if nsamp not divisible
    if pos < nsamp:
        m = nsamp - pos
        c = rng.standard_normal((m, d))
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        dots = c @ x_keep.T
        sc = dots.max(axis=1)
        all_c[pos:] = c
        all_s[pos:] = sc
    order = np.argsort(all_s)
    picked = []
    for idx in order:
        c = all_c[idx]
        ok = True
        for p in picked:
            if float(p @ c) > sep:
                ok = False
                break
        if ok:
            picked.append(c)
            if len(picked) >= k:
                break
    if len(picked) < k:
        for idx in order:
            if len(picked) >= k:
                break
            picked.append(all_c[idx])
    return np.array(picked)


def _lp_polish(x, seconds, t0=None, tl=178.0):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t0 is not None and time.time() - t0 > tl:
            break
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
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
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    TL = 178.0
    try:
        xb = np.load("best.npy").astype(np.float64)
    except Exception:
        try:
            xb = np.load("record.npy").astype(np.float64)
        except Exception:
            xb = None
    if xb is None or xb.shape != (292, 4):
        rng0 = np.random.default_rng(1)
        xb = rng0.standard_normal((292, 4))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = 292, 4
    best = _max_cos(xb)
    x_best = xb.copy()

    rng = np.random.default_rng(20261031)

    explore_end = t0 + TL - 32.0
    it = 0
    sigmas_global = [0.005, 0.012, 0.03, 0.07, 0.15, 0.30]

    while time.time() < explore_end:
        it += 1
        if time.time() + 0.4 > explore_end:
            break
        X = x_best.copy()
        mode = rng.random()
        if mode < 0.32:
            # global shake
            sigma = sigmas_global[int(rng.integers(len(sigmas_global)))]
            X = X + sigma * rng.standard_normal((n, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            k_kill = 0
        elif mode < 0.62:
            # patch shake
            knear = int(rng.choice(np.array([12, 20, 30, 45])))
            sig_big = float(rng.choice(np.array([0.08, 0.15, 0.25, 0.40])))
            sig_small = 0.004
            Xn = X + sig_small * rng.standard_normal((n, d))
            cen = int(rng.integers(n))
            dots = X @ X[cen]
            nn = np.argsort(-dots)[:knear]
            Xn[nn] += sig_big * rng.standard_normal((knear, d))
            Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
            X = Xn
            k_kill = 0
        else:
            # kill and fill into holes
            if rng.random() < 0.35:
                k_kill = int(rng.integers(1, 4))
            elif rng.random() < 0.7:
                k_kill = int(rng.integers(4, 8))
            else:
                k_kill = int(rng.integers(8, 13))
            k_kill = min(k_kill, 12)
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            glo = G.max()
            mx = G.max(axis=1)
            slack = glo - mx
            deg = (G > glo - 2e-4).sum(axis=1)
            order_loose = np.argsort(-slack)
            order_jam = np.argsort(-deg)
            victims = set()
            # always consider including most jammed
            if rng.random() < 0.6:
                victims.add(int(order_jam[0]))
                if rng.random() < 0.4 and len(order_jam) > 1:
                    victims.add(int(order_jam[int(rng.integers(1, min(10, n)))]))
            while len(victims) < k_kill:
                r = rng.random()
                if r < 0.45:
                    pool = order_loose[:20]
                    victims.add(int(pool[int(rng.integers(len(pool)))]))
                elif r < 0.70:
                    pool = order_jam[:12]
                    victims.add(int(pool[int(rng.integers(len(pool)))]))
                else:
                    victims.add(int(rng.integers(n)))
            victims = list(victims)[:k_kill]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[victims] = False
            x_keep = X[keep_mask]
            sep = float(rng.choice(np.array([0.82, 0.88, 0.93])))
            nsamp = int(rng.choice(np.array([20000, 30000, 45000])))
            holes = _find_holes(x_keep, len(victims), rng, nsamp=nsamp, sep=sep)
            X[victims] = holes[:len(victims)]
            X[victims] += 0.008 * rng.standard_normal((len(victims), d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)

        # random anneal hyperparams
        beta0 = float(rng.choice(np.array([15.0, 25.0, 40.0])))
        beta1 = float(rng.choice(np.array([200.0, 400.0, 700.0, 1000.0])))
        lr0 = float(rng.choice(np.array([0.04, 0.08, 0.14, 0.22])))
        mu = float(rng.choice(np.array([0.80, 0.86, 0.90, 0.93])))
        noise0 = float(rng.choice(np.array([0.0, 0.0, 0.001, 0.0025])))
        steps = int(rng.choice(np.array([180, 260, 380, 520])))

        # short screen if steps large: do half then check
        if steps > 300 and (mode < 0.62):
            # two chunk screening
            h = steps // 2
            X = _anneal_repel(X, h, beta0, beta1, lr0, mu, noise0, rng)
            vmid = _max_cos(X)
            # if far worse than best, abort early (allow 0.005 slack)
            if vmid > best + 0.006:
                # discard, continue with small chance to keep exploring
                continue
            X = _anneal_repel(X, steps - h, beta0, beta1, max(lr0 * 0.5, 0.02), mu, 0.0, rng)
        else:
            X = _anneal_repel(X, steps, beta0, beta1, lr0, mu, noise0, rng)

        # cheap tighten for final focusing
        if rng.random() < 0.7:
            X = _cheap_tighten(X, 40, 0.02)
        v = _max_cos(X)
        if v < best - 1e-13:
            best = v
            x_best = X.copy()
        # occasional extra sharpening of new best with high-beta short run
        if v < best + 1e-7 and rng.random() < 0.05 and time.time() + 2.0 < explore_end:
            Xs = _anneal_repel(X.copy(), 120, 60.0, 1200.0, 0.03, 0.88, 0.0, rng)
            Xs = _cheap_tighten(Xs, 60, 0.012)
            vs = _max_cos(Xs)
            if vs < best - 1e-13:
                best = vs
                x_best = Xs.copy()

    # final LP polish with remaining time
    rem = (t0 + TL) - time.time() - 1.5
    if rem > 4:
        lp_time = min(rem, 30.0)
        try:
            x_best = _lp_polish(x_best, lp_time, t0=t0, tl=TL - 1.0)
        except Exception:
            pass
        best = _max_cos(x_best)

    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
