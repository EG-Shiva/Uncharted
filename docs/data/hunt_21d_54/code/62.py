import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _proj(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    y = x / n
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    n2 = np.linalg.norm(y, axis=1, keepdims=True)
    n2[n2 == 0] = 1.0
    return y / n2

def _gram_alt_proj(tau_target, iters, rng, tau_start=0.12):
    n, d = 54, 21
    # random initial Gram from random points (topologically fresh)
    X0 = rng.normal(size=(n, d))
    X0 = _proj(X0)
    G = X0 @ X0.T
    # ensure diag 1
    np.fill_diagonal(G, 1.0)
    for it in range(iters):
        frac = it / max(1, iters - 1)
        cur_tau = tau_start + (tau_target - tau_start) * frac
        # PSD rank<=21 projection
        w, V = np.linalg.eigh(G)
        Vt = V[:, -d:]
        wt = w[-d:]
        wt = np.maximum(wt, 0.0)
        # reconstruct: (Vt * wt) @ Vt.T
        Gpsd = (Vt * wt) @ Vt.T
        # box projection: off-diag <= cur_tau, in [-1,1], diag=1
        # clip in place
        np.clip(Gpsd, -1.0, cur_tau, out=Gpsd)
        np.fill_diagonal(Gpsd, 1.0)
        # symmetrize (clip preserves symmetry)
        G = Gpsd
    # final PSD factor from last G
    w, V = np.linalg.eigh(G)
    Vt = V[:, -d:]
    wt = w[-d:]
    wt = np.maximum(wt, 1e-12)
    X = Vt * np.sqrt(wt)  # n x d
    X = _proj(X)
    # diagnostics: spectrum + histogram (for exploration, no effect)
    try:
        ev = np.linalg.eigvalsh(G)
        _ = ev  # spectrum logged implicitly
        Gu = X @ X.T
        iu = np.triu_indices(n, 1)
        _ = np.histogram(Gu[iu], bins=20)
    except Exception:
        pass
    return X, w

def _quench(y, steps):
    n, d = y.shape
    vel = np.zeros((n, d))
    mu = 0.85
    lr0 = 0.32
    for s in range(steps):
        frac = s / max(1, steps)
        T = 0.02 * (1.0 - frac) + 0.003
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = (G - m) / T
        np.clip(E, -20.0, 0.0, out=E)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = W @ y
        dot = np.einsum('ij,ij->i', grad, y)
        grad -= dot[:, None] * y
        grad /= rs
        vel = mu * vel + grad
        lr = lr0 * (1.0 - 0.6 * frac)
        y = y - lr * vel
        y = _proj(y)
    return y

def _lp_polish(x, seconds):
    x = _proj(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 2500:
            idx = np.argpartition(-c[sel], 2500)[:2500]
            I, J = I[idx], J[idx]
            csv = c[sel][idx]
            p = 2500
        else:
            csv = c[sel]
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
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs",
                          options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _proj(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    T0 = time.time()
    LIMIT = 150.0  # strict inner deadline, verifier allows 180
    rng = np.random.default_rng(0)
    try:
        b0 = np.load("best.npy")
    except Exception:
        b0 = np.load("record.npy")
    b0 = _proj(np.nan_to_num(np.asarray(b0, dtype=float)))
    if b0.shape != (54, 21):
        b0 = _proj(rng.normal(size=(54, 21)))
    try:
        r0 = _proj(np.asarray(np.load("record.npy"), dtype=float))
        if r0.shape != (54, 21):
            r0 = b0
    except Exception:
        r0 = b0
    best = b0.copy()
    bv = _max_cos(best)
    rv = _max_cos(r0)
    if rv < bv:
        best, bv = r0.copy(), rv

    def tl():
        return LIMIT - (time.time() - T0)

    # ---- Gram alternating-projection exploration ----
    taus = [0.0666, 0.064, 0.061, 0.058, 0.055, 0.05]
    cands = []  # (val, X)
    # iters per seed: few thousand would be heavy for 12 seeds; use ~800-1200
    for ti, tau in enumerate(taus):
        if tl() < 45:
            break
        for rep in range(2):
            if tl() < 40:
                break
            iters = 1000 if tau > 0.058 else 1400
            # vary tau_start slightly for diversity
            ts = 0.12 if (rep == 0) else 0.09
            Xg, spec = _gram_alt_proj(tau, iters, rng, tau_start=max(ts, tau + 0.01))
            v0 = _max_cos(Xg)
            # short soft-max quench
            steps = 900
            # adapt steps to time left
            if tl() < 50:
                steps = 400
            Xq = _quench(Xg.copy(), steps=steps)
            vq = _max_cos(Xq)
            cands.append((vq, Xq))
            # keep global best if quench already beats (unlikely but keep)
            if vq < bv - 1e-12:
                best, bv = Xq.copy(), vq
            if tl() < 30:
                break

    # pick most promising distinct seeds for capped LP polish
    if cands:
        cands.sort(key=lambda t: t[0])
        # take top-3 with diversity (just top 3)
        K = min(3, len(cands))
        for k in range(K):
            if tl() < 15:
                break
            v_init, Xc = cands[k]
            # skip hopeless seeds far above record to save budget
            # still polish the best one even if hopeless, to try topology change
            if k > 0 and v_init > bv + 0.06:
                continue
            budget = min(10.0, tl() - 8.0)
            if budget < 2.0:
                break
            # brief extra quench before LP for stability
            Xc = _quench(Xc.copy(), steps=300)
            Xp = _lp_polish(Xc, seconds=budget)
            vp = _max_cos(Xp)
            if vp < bv:
                best, bv = Xp.copy(), vp

    # final safety: ensure best in hand is returned; tiny final squeeze if time
    if tl() > 6:
        # only squeeze best if it came from Gram seeds (different basin);
        # squeezing the jammed record wastes time, so skip if best is initial jam
        # detect: if best matches b0/r0 closely, skip
        try:
            d0 = min(float(np.abs(best - b0).max()), float(np.abs(best - r0).max()))
        except Exception:
            d0 = 1.0
        if d0 > 1e-6:
            best = _lp_polish(best, seconds=min(6.0, tl() - 1.0))

    best = _proj(best)
    return best
