import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _proj(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _adam_lse(Y, steps, T0a, T1a, lr0, lr1):
    Y = _proj(Y.copy())
    n, d = Y.shape
    m = np.zeros_like(Y)
    v = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        frac = s / max(1, steps - 1) if steps > 1 else 1.0
        T = T0a * (T1a / T0a) ** frac if T0a > 0 and T1a > 0 else T0a
        lr = lr0 * (1 - frac) + lr1 * frac
        G = Y @ Y.T
        np.fill_diagonal(G, -10.0)
        gmax = float(G.max())
        E = (G - gmax) / max(T, 1e-12)
        np.clip(E, -20.0, 0.0, out=E)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        S = float(W.sum()) + 1e-12
        grad = (W @ Y) / S
        dot = np.einsum('ij,ij->i', grad, Y)
        grad -= dot[:, None] * Y
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * (grad * grad)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        mh = m / bc1
        vh = v / bc2
        step = lr * mh / (np.sqrt(vh) + eps)
        sd = np.einsum('ij,ij->i', step, Y)
        step -= sd[:, None] * Y
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        cap = 0.08
        step = np.where(sn > cap, step * (cap / sn), step)
        Y = Y - step
        Y = _proj(Y)
    return Y

def _lp_polish(X, seconds):
    X = _proj(X)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        remain = seconds - (time.time() - start)
        if remain <= 0.08:
            break
        G = X @ X.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 2000:
            idx = np.argpartition(-c[sel], 2000)[:2000]
            I, J = I[idx], J[idx]
            csv = c[sel][idx]
        else:
            csv = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs",
                          options={"time_limit": min(1.5, max(0.3, remain - 0.05))})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _proj(y)
        vv = _max_cos(y)
        if vv < best:
            X, best = y, vv
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X

def _gram_ap(rng, n, tau_start, tau_end, niter, init_type):
    # initial G
    if init_type == 0:
        # Wishart from random spherical points
        X0 = rng.normal(size=(n, 21))
        X0 = X0 / (np.linalg.norm(X0, axis=1, keepdims=True) + 1e-12)
        G = X0 @ X0.T
        # add symmetric noise to leave rank manifold
        N = rng.normal(scale=0.15, size=(n, n))
        N = (N + N.T) * 0.5
        np.fill_diagonal(N, 0.0)
        G = G + N
    elif init_type == 1:
        A = rng.uniform(-0.6, 0.6, size=(n, n))
        A = (A + A.T) * 0.5
        G = A
    else:
        X0 = rng.normal(size=(n, 21))
        X0 = X0 / (np.linalg.norm(X0, axis=1, keepdims=True) + 1e-12)
        Gw = X0 @ X0.T
        A = rng.uniform(-0.6, 0.6, size=(n, n))
        A = (A + A.T) * 0.5
        alpha = float(rng.random())
        G = alpha * Gw + (1 - alpha) * A
    np.fill_diagonal(G, 1.0)
    np.clip(G, -1.0, 1.0, out=G)
    np.fill_diagonal(G, 1.0)
    # Dykstra-ish corrections for the two sets (PSD vs box); keep small memory
    # Simple alternating with correction buffers
    P = np.zeros_like(G)  # correction for PSD step
    Q = np.zeros_like(G)
    for k in range(niter):
        frac = k / max(1, niter - 1)
        tau = tau_start * (1 - frac) + tau_end * frac
        # --- project onto PSD rank<=21 (with Dykstra correction) ---
        try:
            H = G + P
            # symmetrize
            H = (H + H.T) * 0.5
            w, V = np.linalg.eigh(H)
            Vt = V[:, -21:]
            wt = w[-21:]
            wt = np.maximum(wt, 0.0)
            Gpsd = (Vt * wt) @ Vt.T
            Gpsd = (Gpsd + Gpsd.T) * 0.5
            P = H - Gpsd
        except Exception:
            Gpsd = G
        G = Gpsd
        # --- project onto box diag=1, offdiag<=tau, >=-1 ---
        # Dykstra correction Q
        try:
            H2 = G + Q
            # clip
            np.clip(H2, -1.0, tau, out=H2)
            np.fill_diagonal(H2, 1.0)
            Q = G + Q - H2 if False else (H2 - G) * 0.0  # keep vanilla for stability; no Q accumulation
            G = H2
        except Exception:
            np.clip(G, -1.0, tau, out=G)
            np.fill_diagonal(G, 1.0)
    return G

def _factor(G, d=21):
    G = (G + G.T) * 0.5
    w, V = np.linalg.eigh(G)
    Vt = V[:, -d:]
    wt = w[-d:]
    wt = np.maximum(wt, 1e-12)
    X = Vt * np.sqrt(wt)
    # X is n x d; rows need renormalisation
    X = _proj(X)
    return X, w

def solve():
    T0 = time.time()
    DEADLINE = T0 + 150.0
    n, d = 54, 21
    rng = np.random.default_rng(20261101)
    # baseline
    try:
        best_X = np.load("best.npy").astype(float)
    except Exception:
        try:
            best_X = np.load("record.npy").astype(float)
        except Exception:
            best_X = rng.normal(size=(n, d))
    best_X = np.nan_to_num(best_X)
    if best_X.shape != (n, d):
        best_X = _proj(rng.normal(size=(n, d)))
    else:
        best_X = _proj(best_X)
    best_val = _max_cos(best_X)
    try:
        rec = _proj(np.load("record.npy").astype(float))
        rv = _max_cos(rec)
        if rv < best_val:
            best_val, best_X = rv, rec.copy()
    except Exception:
        pass

    def tleft():
        return DEADLINE - time.time()

    # Gram schedule: decreasing tau from 0.0666 to 0.05
    tau_ends = [0.066, 0.064, 0.062, 0.060, 0.058, 0.056, 0.054, 0.052, 0.05,
                0.065, 0.061, 0.057, 0.053, 0.063, 0.059, 0.055]
    # eigenvalue / histogram logging (kept minimal for speed)
    logs = []
    trial = 0
    # Phase 1: Gram AP + short quench
    # adapt niter to remaining time
    while tleft() > 45 and trial < 40:
        tau_end = float(tau_ends[trial % len(tau_ends)]) + float(rng.normal(scale=0.0008))
        tau_end = float(np.clip(tau_end, 0.048, 0.0665))
        tau_start = 0.0666
        # fewer iters early for diversity, more later? keep 900-1600
        if tleft() > 100:
            niter = 1400
        elif tleft() > 70:
            niter = 1100
        else:
            niter = 800
        init_type = trial % 3
        try:
            G = _gram_ap(rng, n, tau_start, tau_end, niter, init_type)
        except Exception:
            trial += 1
            continue
        try:
            Xf, w = _factor(G, d)
        except Exception:
            trial += 1
            continue
        raw = _max_cos(Xf)
        # spectrum + histogram diagnostics (explicitly explored)
        try:
            # top eigenvalues, distinct inner-product histogram bins
            evals_sorted = np.sort(w)[::-1]
            Gu = Xf @ Xf.T
            iu = np.triu_indices(n, 1)
            c = Gu[iu]
            hist, _ = np.histogram(c, bins=10, range=(-1.0, 1.0))
            logs.append((tau_end, float(raw), evals_sorted[:22].copy(), hist.copy()))
        except Exception:
            pass
        # short softmax quench to make feasible point
        # choose quench params depending on raw
        try:
            if raw < 0.09:
                Xq = _adam_lse(Xf, steps=350, T0a=0.008, T1a=0.0012, lr0=0.015, lr1=0.003)
            else:
                Xq = _adam_lse(Xf, steps=500, T0a=0.015, T1a=0.0015, lr0=0.02, lr1=0.004)
        except Exception:
            Xq = Xf
        try:
            vq = _max_cos(Xq)
        except Exception:
            trial += 1
            continue
        if vq < best_val - 1e-12:
            best_val = vq
            best_X = Xq.copy()
        # also try a second quench with different temp if promising but not best
        if vq < best_val + 0.02 and tleft() > 55:
            try:
                Xq2 = _adam_lse(Xf, steps=250, T0a=0.004, T1a=0.0008, lr0=0.012, lr1=0.002)
                vq2 = _max_cos(Xq2)
                if vq2 < best_val - 1e-12:
                    best_val = vq2
                    best_X = Xq2.copy()
            except Exception:
                pass
        trial += 1
        if tleft() < 40:
            break

    # Phase 2: single capped LP polish on the best factored configuration
    remain = tleft() - 1.0
    if remain > 1.0:
        budget = min(remain, 20.0)
        try:
            Xp = _lp_polish(best_X, budget)
            vp = _max_cos(Xp)
            if vp < best_val:
                best_val, best_X = vp, Xp
        except Exception:
            pass
    best_X = _proj(best_X)
    # safety: if somehow worse than file, return file (guarantee no regression)
    try:
        f0 = _proj(np.load("best.npy").astype(float))
        fv = _max_cos(f0)
        if fv < _max_cos(best_X):
            return f0
    except Exception:
        pass
    return best_X
