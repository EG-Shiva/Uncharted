import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _normalize(x):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _smooth_anneal(X0, steps, a0, a1, lr0, lr1, mu, rng, noise0=0.0):
    X = X0.copy()
    V = np.zeros_like(X)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        alpha = a0 * (a1 / a0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        # exp(alpha*(G-m)), underflow to 0 is fine
        E = np.exp(alpha * (G - m))
        np.fill_diagonal(E, 0.0)
        S = E.sum()
        if S == 0 or not np.isfinite(S):
            S = 1.0
        Grad = (E @ X) / S
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        V = mu * V + (1.0 - mu) * Gt if s == 0 else mu * V + Gt
        # Langevin noise early
        X = X - lr * V
        if noise0 > 0:
            nz = noise0 * (1.0 - frac)
            if nz > 1e-12:
                R = rng.standard_normal(size=X.shape)
                R -= np.sum(R * X, axis=1, keepdims=True) * X
                X = X + nz * R
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        X = X / nrm
    return X

def _hard_polish(X0, steps, lr0, decay=0.997):
    cur = X0.copy()
    best = cur.copy()
    bv = _max_cos(cur)
    n = cur.shape[0]
    for s in range(steps):
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        delta = 2e-3 * (0.985 ** s) + 1e-7
        mask = G > (m - delta)
        deg = mask.sum(axis=1, keepdims=True).astype(float) + 1.0
        Grad = mask.astype(float) @ cur
        dot = np.sum(Grad * cur, axis=1, keepdims=True)
        Gt = Grad - dot * cur
        Gt = Gt / deg
        lr = lr0 * (decay ** s)
        cur = cur - lr * Gt
        cur = _normalize(cur)
        if (s % 10 == 0) or s == steps - 1:
            v = _max_cos(cur)
            if v < bv:
                bv = v
                best = cur.copy()
    return best, bv

def _reinsert_worst(X, rng, k, n_cand=20000):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    rowmax = G.max(axis=1)
    idx = np.argsort(rowmax)[-k:]
    Xn = X.copy()
    for ii in idx:
        cand = rng.standard_normal(size=(n_cand, 4))
        cand = _normalize(cand)
        D = cand @ Xn.T
        D[:, ii] = -2.0
        cm = D.max(axis=1)
        bi = int(np.argmin(cm))
        y = cand[bi].copy()
        # local refine y against fixed others
        for t in range(250):
            d = Xn @ y
            d[ii] = -2.0
            j = int(np.argmax(d))
            # move away from Xn[j]
            gvec = Xn[j] - float(Xn[j] @ y) * y
            step = 0.08 * (0.99 ** t)
            y = y - step * gvec
            ny = np.linalg.norm(y)
            if ny == 0 or not np.isfinite(ny):
                y = cand[bi].copy()
                break
            y = y / ny
        Xn[ii] = y
    return Xn

def solve():
    t0 = time.time()
    TIME_BUDGET = 168.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        cands.append(_normalize(xb))
    if xr is not None:
        cands.append(_normalize(xr))
    if not cands:
        r = rng.standard_normal(size=(163, 4))
        cands.append(_normalize(r))
    best = None
    best_val = 2.0
    for c in cands:
        v = _max_cos(c)
        if v < best_val:
            best_val = v
            best = c.copy()
    # quick hard polish of starting point to set baseline
    try:
        pb, pv = _hard_polish(best, 200, 0.002)
        if pv < best_val:
            best, best_val = pb, pv
    except Exception:
        pass

    sigmas = [0.008, 0.02, 0.05, 0.10, 0.015, 0.035, 0.07, 0.005]
    restart = 0
    # main basin-hopping
    while True:
        if time.time() - t0 > TIME_BUDGET - 8.0:
            break
        restart += 1
        sigma = sigmas[(restart - 1) % len(sigmas)]
        mode = restart % 6
        Xt = best.copy()
        if mode == 0:
            # heavy: reinsert 3 worst + perturb
            try:
                Xt = _reinsert_worst(Xt, rng, k=3, n_cand=15000)
            except Exception:
                pass
            Xt = Xt + sigma * rng.standard_normal(size=Xt.shape)
            Xt = _normalize(Xt)
            steps = 1400
            a0, a1 = 8.0, 350.0
            lr0, lr1 = 0.6, 0.02
            mu = 0.85
        elif mode == 1:
            try:
                Xt = _reinsert_worst(Xt, rng, k=2, n_cand=15000)
            except Exception:
                pass
            Xt = Xt + sigma * rng.standard_normal(size=Xt.shape)
            Xt = _normalize(Xt)
            steps = 1200
            a0, a1 = 12.0, 300.0
            lr0, lr1 = 0.5, 0.015
            mu = 0.9
        elif mode == 2:
            # pure jitter exploration
            Xt = Xt + sigma * rng.standard_normal(size=Xt.shape)
            # also randomize rattlers / low-contact points occasionally
            if restart % 12 == 5:
                G = Xt @ Xt.T
                np.fill_diagonal(G, -2.0)
                rm = G.max(axis=1)
                low = np.argsort(rm)[:3]
                R = rng.standard_normal(size=(3, 4))
                Xt[low] = _normalize(R)
            Xt = _normalize(Xt)
            steps = 1100
            a0, a1 = 15.0, 400.0
            lr0, lr1 = 0.4, 0.01
            mu = 0.88
        elif mode == 3:
            Xt = Xt + sigma * rng.standard_normal(size=Xt.shape)
            Xt = _normalize(Xt)
            steps = 1600
            a0, a1 = 6.0, 250.0
            lr0, lr1 = 0.7, 0.02
            mu = 0.82
        elif mode == 4:
            # relocate 1 worst to void without jitter
            try:
                Xt = _reinsert_worst(Xt, rng, k=1, n_cand=25000)
            except Exception:
                pass
            Xt = Xt + 0.01 * rng.standard_normal(size=Xt.shape)
            Xt = _normalize(Xt)
            steps = 1200
            a0, a1 = 10.0, 350.0
            lr0, lr1 = 0.5, 0.012
            mu = 0.9
        else:
            Xt = Xt + sigma * rng.standard_normal(size=Xt.shape)
            Xt = _normalize(Xt)
            steps = 900
            a0, a1 = 20.0, 500.0
            lr0, lr1 = 0.3, 0.008
            mu = 0.9

        # adapt steps to remaining time
        rem = TIME_BUDGET - 8.0 - (time.time() - t0)
        if rem < 12.0:
            steps = min(steps, 400)
        # smooth anneal
        try:
            Xo = _smooth_anneal(Xt, steps, a0, a1, lr0, lr1, mu, rng,
                                noise0=0.002 if mode == 3 else 0.0)
        except Exception:
            continue
        # hard polish
        try:
            Xp, vp = _hard_polish(Xo, 250, 0.0025)
        except Exception:
            continue
        # second finer hard polish if promising
        if vp < best_val + 2e-4:
            try:
                Xp2, vp2 = _hard_polish(Xp, 250, 0.0008, decay=0.998)
                Xp, vp = Xp2, vp2
            except Exception:
                pass
        if vp < best_val - 1e-12 and np.all(np.isfinite(Xp)):
            best = Xp.copy()
            best_val = vp
            # early exit if big win
            if best_val < 0.856415430312 - 5e-5:
                # spend remaining time polishing winner
                try:
                    Xf, vf = _hard_polish(best, 600, 0.0015)
                    if vf < best_val:
                        best, best_val = Xf, vf
                except Exception:
                    pass
        # opportunistic time check
        if time.time() - t0 > TIME_BUDGET - 8.0:
            break

    # final intensive polish on winner with remaining time
    try:
        rem = TIME_BUDGET - (time.time() - t0) - 1.0
        if rem > 4.0:
            s = min(800, int(rem * 120))
            Xf, vf = _hard_polish(best, s, 0.0012, decay=0.9985)
            if vf < best_val:
                best, best_val = Xf, vf
    except Exception:
        pass
    best = _normalize(best)
    return best
