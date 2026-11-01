import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
import itertools

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contact_stats(X, tol=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cnt = (G >= m - tol).sum(axis=1)
    return m, cnt, float(cnt.std()), int((cnt == 0).sum())

def _greedy_select(pool, start_idx):
    P = pool.shape[0]
    s = pool @ pool[start_idx]
    s = s.copy()
    s[start_idx] = 2.0
    chosen = [int(start_idx)]
    for _ in range(1, 173):
        idx = int(np.argmin(s))
        chosen.append(idx)
        upd = pool @ pool[idx]
        np.maximum(s, upd, out=s)
        s[idx] = 2.0
    return pool[np.array(chosen, dtype=int)].copy()

def _smooth(Y, betas, iters_list, lrs):
    X = Y.copy()
    n = X.shape[0]
    for beta, iters, lr0 in zip(betas, iters_list, lrs):
        for it in range(int(iters)):
            frac = it / max(1, int(iters) - 1)
            step = lr0 * (1.0 - 0.8 * frac)
            G = X @ X.T
            np.fill_diagonal(G, -10.0)
            m = float(G.max())
            W = np.exp(beta * (G - m))
            np.fill_diagonal(W, 0.0)
            S = float(W.sum() * 0.5) + 1e-300
            WX = W @ X
            grad = WX / S
            dot = (grad * X).sum(axis=1, keepdims=True)
            Gt = grad - dot * X
            gn = float(np.linalg.norm(Gt, axis=1).max()) + 1e-300
            if not np.isfinite(gn) or gn == 0:
                continue
            X = X - (step / gn) * Gt
            X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _lp_polish(X, seconds, radius0=1e-3, fix=None):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = radius0
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    if fix is None:
        fix = np.zeros(n, dtype=bool)
    while time.time() - start < seconds and radius > 1e-14:
        G = X @ X.T
        cfull = G[iu0, iu1]
        m = float(cfull.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = cfull > thresh
        if int(sel.sum()) > 3000:
            idx = np.where(sel)[0]
            part = np.argpartition(cfull[idx], len(idx) - 3000)[-3000:]
            mask = np.zeros_like(sel)
            mask[idx[part]] = True
            sel = mask
        I = iu0[sel]; J = iu1[sel]; c = cfull[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        cols = np.concatenate([(I[:, None] * d + np.arange(d)),
                               (J[:, None] * d + np.arange(d)),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        bnds = []
        for i in range(n):
            if fix[i]:
                bnds.extend([(0.0, 0.0)] * d)
            else:
                bnds.extend([(-radius, radius)] * d)
        bnds.append((None, None))
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bnds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _build_pools(rng):
    d = 5
    lat = []
    # integer lattice -2..2
    for tup in itertools.product([-2, -1, 0, 1, 2], repeat=5):
        if all(v == 0 for v in tup):
            continue
        v = np.array(tup, dtype=float)
        v /= np.linalg.norm(v)
        lat.append(v)
    lat = np.stack(lat, axis=0)
    # dedup directions (keep one of each antipodal pair half? keep both for pool diversity is fine, but reduce size)
    # keep all lat (3124)
    # Fourier torus: (1,cos t,sin t,cos2t,sin2t)/sqrt3
    T = 2000
    thetas = np.linspace(0, 2 * np.pi, T, endpoint=False)
    F = np.stack([np.ones(T), np.cos(thetas), np.sin(thetas), np.cos(2 * thetas), np.sin(2 * thetas)], axis=1)
    F /= np.linalg.norm(F, axis=1, keepdims=True)
    # cyclic shifts of random prototypes
    C = []
    for _ in range(300):
        p = rng.normal(size=d)
        p /= np.linalg.norm(p)
        for s in range(5):
            C.append(np.roll(p, s))
    C = np.stack(C, axis=0)
    # random uniform
    R = rng.normal(size=(15000, d))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    struct = np.concatenate([lat, F, C], axis=0)
    full = np.concatenate([struct, R], axis=0)
    # normalize just in case
    full /= np.linalg.norm(full, axis=1, keepdims=True)
    struct /= np.linalg.norm(struct, axis=1, keepdims=True)
    return full, struct, R

def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline = t0 + BUDGET
    def left():
        return deadline - time.time()
    n, d = 173, 5
    rng = np.random.default_rng(20260601)
    # fallback record
    fb = None
    fbv = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (n, d):
                an = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(an)
                if v < fbv:
                    fbv = v
                    fb = an.copy()
        except Exception:
            pass
    if fb is None:
        fb = rng.normal(size=(n, d))
        fb /= np.linalg.norm(fb, axis=1, keepdims=True)
        fbv = _max_cos(fb)
    full, struct, _rand = _build_pools(rng)
    Pf = full.shape[0]
    Ps = struct.shape[0]
    # diverse start indices: fixed spread + a couple random
    starts_full = [0, Pf // 8, Pf // 4, Pf // 2, 3 * Pf // 4]
    starts_struct = [0, Ps // 4, Ps // 2]
    # add deterministic pseudo-random starts
    starts_full.append(int(rng.integers(Pf)))
    starts_struct.append(int(rng.integers(Ps)))
    cands = []
    betas = [8.0, 20.0, 50.0, 120.0, 250.0]
    iters_list = [250, 250, 300, 300, 350]
    lrs = [0.15, 0.12, 0.08, 0.05, 0.03]
    # milder variant for some seeds
    betas2 = [5.0, 15.0, 40.0, 100.0, 200.0]
    iters2 = [300, 300, 300, 350, 400]
    lrs2 = [0.18, 0.14, 0.09, 0.05, 0.025]
    jobs = []
    for s in starts_full:
        jobs.append((full, int(s % Pf), betas, iters_list, lrs))
    for s in starts_struct:
        jobs.append((struct, int(s % Ps), betas2, iters2, lrs2))
    # one pure-random seed smoothed directly
    try:
        Xr = rng.normal(size=(n, d))
        Xr /= np.linalg.norm(Xr, axis=1, keepdims=True)
        if left() > 20:
            Xs = _smooth(Xr, betas, [200, 200, 250, 250, 300], lrs)
            v = _max_cos(Xs)
            _, _, std, _ = _contact_stats(Xs)
            cands.append((v, std, Xs))
    except Exception:
        pass
    for (pool, sidx, bb, ii, ll) in jobs:
        if left() < 25.0:
            break
        try:
            X0 = _greedy_select(pool, sidx)
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            Xs = _smooth(X0, bb, ii, ll)
            v = _max_cos(Xs)
            _, _, std, _ = _contact_stats(Xs)
            cands.append((v, std, Xs.copy()))
        except Exception:
            continue
    if len(cands) == 0:
        # fallback polish only
        rem = left()
        if rem > 2.0:
            fb = _lp_polish(fb, seconds=max(1.0, rem - 1.0))
        fb /= np.linalg.norm(fb, axis=1, keepdims=True)
        return fb
    # sort by max cos then regularity (std)
    cands = sorted(cands, key=lambda z: (z[0], z[1]))
    # short LP polish for top up to 6
    polished = []
    K = min(6, len(cands))
    for k in range(K):
        if left() < 14.0:
            break
        v, std, Xc = cands[k]
        sec = 1.4 if left() > 40 else 0.8
        if left() < sec + 11.0:
            polished.append((v, std, Xc))
            continue
        try:
            Xp = _lp_polish(Xc, seconds=sec, radius0=1e-3)
            vp = _max_cos(Xp)
            _, _, stdp, _ = _contact_stats(Xp)
            polished.append((vp, stdp, Xp))
        except Exception:
            polished.append((v, std, Xc))
    if len(polished) == 0:
        polished = cands[:1]
    polished = sorted(polished, key=lambda z: (z[0], z[1]))
    # extended polish for top 2
    final = polished
    for k in range(min(2, len(polished))):
        if left() < 12.0:
            break
        v, std, Xc = final[k]
        sec = 4.0 if left() > 30 else 1.5
        if left() < sec + 8.0:
            break
        try:
            Xp = _lp_polish(Xc, seconds=sec, radius0=8e-4)
            vp = _max_cos(Xp)
            _, _, stdp, _ = _contact_stats(Xp)
            final[k] = (vp, stdp, Xp)
        except Exception:
            pass
    final = sorted(final, key=lambda z: (z[0], z[1]))
    bv, _, bx = final[0]
    # compare to fallback; keep best strictly
    if fbv <= bv:
        rem = left()
        if rem > 3.0:
            try:
                fb = _lp_polish(fb, seconds=min(6.0, rem - 1.5), radius0=1e-3)
                fbv = _max_cos(fb)
            except Exception:
                pass
        # return whichever is better after final polish
        if fbv <= bv:
            fb /= np.linalg.norm(fb, axis=1, keepdims=True)
            return fb
    # else de novo wins; give it remaining time
    rem = left()
    if rem > 3.0:
        try:
            bx = _lp_polish(bx, seconds=min(8.0, rem - 1.0), radius0=6e-4)
        except Exception:
            pass
    # final safety: pick best among bx and fb
    try:
        vbx = _max_cos(bx)
    except Exception:
        fb /= np.linalg.norm(fb, axis=1, keepdims=True)
        return fb
    if fbv < vbx:
        fb /= np.linalg.norm(fb, axis=1, keepdims=True)
        return fb
    bx /= np.linalg.norm(bx, axis=1, keepdims=True)
    return bx
