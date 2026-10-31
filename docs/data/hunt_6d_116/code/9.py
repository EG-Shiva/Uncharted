import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contact_counts(X, tol=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cnt = (G > m - tol).sum(axis=1)
    return m, cnt, G

def _smooth(X, iters, a0, a1, lr0, lr1):
    n, d = X.shape
    m = np.zeros_like(X)
    vv = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    b1p = 1.0
    b2p = 1.0
    for it in range(1, iters + 1):
        frac = it / iters
        alpha = a0 + (a1 - a0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cmax = float(G.max())
        W = np.exp(alpha * (G - cmax))
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        s = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - s * X
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1.0 - b1) * Gt
        vv *= b2
        vv += (1.0 - b2) * Gt * Gt
        mhat = m / (1.0 - b1p)
        vhat = vv / (1.0 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        X = X - step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _lp_polish(X, seconds):
    X = (X / np.linalg.norm(X, axis=1, keepdims=True)).copy()
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.05:
            break
        Gf = X @ X.T
        c = Gf[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
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
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _micro_hill(X, seconds, rng):
    X = (X / np.linalg.norm(X, axis=1, keepdims=True)).copy()
    n, d = X.shape
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    best = float(G.max())
    t_end = time.time() + seconds
    sigmas = [3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7]
    si = 0
    sigma = sigmas[0]
    it = 0
    while time.time() < t_end:
        it += 1
        if it % 700 == 0:
            si = (si + 1) % len(sigmas)
            sigma = sigmas[si]
        i = int(rng.integers(0, n))
        r = rng.normal(size=d)
        r -= float(r @ X[i]) * X[i]
        rn = float(np.linalg.norm(r))
        if rn < 1e-15:
            continue
        r /= rn
        y = X[i] + sigma * r
        y /= np.linalg.norm(y)
        dv = y @ X.T
        dv[i] = -2.0
        dmax = float(dv.max())
        if dmax >= best:
            continue
        # max among pairs not involving i
        # quick check: if row i max < best, other max == best
        if float(G[i].max()) < best - 1e-12:
            m_other = best
        else:
            m_other = float(np.max(np.delete(np.delete(G, i, axis=0), i, axis=1)))
            if m_other >= best:
                continue
        new_best = dmax if dmax > m_other else m_other
        if new_best < best - 1e-15:
            X[i] = y
            nr = X[i] @ X.T
            nr[i] = -2.0
            G[i, :] = nr
            G[:, i] = nr
            best = new_best
    return X

def _largest_holes(Y, exclude, nh, rng, ncand=6000):
    n, d = Y.shape
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[np.asarray(exclude, dtype=int)] = False
    Yk = Y[keep_mask]
    cand = rng.normal(size=(ncand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    S = cand @ Yk.T
    s max per cand
    scores = S.max(axis=1)
    holes = []
    # greedy diverse holes
    chosen = []
    order = np.argsort(scores)
    # pick best, then enforce separation
    picked = []
    for idx in order:
        c = cand[idx]
        ok = True
        for p in picked:
            if float(c @ p) > 0.9:
                ok = False
                break
        if ok:
            picked.append(c)
            if len(picked) >= nh:
                break
    if len(picked) < nh:
        for idx in order:
            if len(picked) >= nh:
                break
            picked.append(cand[idx])
    return np.array(picked)

def solve():
    t0 = time.time()
    BUDGET = 172.0
    np.random.seed(0)
    rng = np.random.default_rng(0)
    cands = []
    for f in ("best.npy", "record.npy"):
        try:
            a = np.load(f)
            a = np.asarray(a, dtype=float)
            if a.shape == (116, 6):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        a = rng.normal(size=(116, 6))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    best_x = min(cands, key=_max_cos)
    best_v = _max_cos(best_x)
    n, d = best_x.shape
    # light initial squeeze to establish true value
    if time.time() - t0 < BUDGET - 10:
        best_x = _lp_polish(best_x, 2.0)
        best_v = _max_cos(best_x)
    if time.time() - t0 < BUDGET - 10:
        best_x = _micro_hill(best_x, 1.5, rng)
        best_v = _max_cos(best_x)
    trial = 0
    while time.time() - t0 < 118.0 and time.time() - t0 < BUDGET - 35:
        trial += 1
        mode = trial % 6
        _, cnt, Gcur = _contact_counts(best_x, tol=1e-7)
        Y = best_x.copy()
        if mode == 0:
            sigma = 0.012 + 0.015 * float(rng.random())
            Y = Y + sigma * rng.normal(size=Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        elif mode == 1:
            order = np.argsort(-cnt)
            k = int(rng.integers(6, 11))
            idx = order[:k]
            Y[idx] = Y[idx] + 0.12 * rng.normal(size=(k, d))
            Y = Y + 0.008 * rng.normal(size=Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        elif mode == 2:
            k = 2 if (trial // 6) % 2 == 0 else 3
            worst = np.argsort(-cnt)[:k]
            holes = _largest_holes(Y, worst, k, rng, ncand=6000)
            Y[worst] = holes
            Y = Y + 0.004 * rng.normal(size=Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        elif mode == 3:
            ratt = np.where(cnt <= 1)[0]
            if len(ratt) == 0:
                ratt = np.argsort(cnt)[:6]
            # randomize rattlers fully, kick jammed a bit
            Y[ratt] = rng.normal(size=(len(ratt), d))
            jam = np.argsort(-cnt)[:6]
            Y[jam] = Y[jam] + 0.05 * rng.normal(size=(6, d))
            Y = Y + 0.003 * rng.normal(size=Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        elif mode == 4:
            # kick one side of a random max pair
            m = float(Gcur.max())
            # find pairs near max
            iu0 = np.where(Gcur > m - 2e-9)
            pairs = [(int(iu0[0][k]), int(iu0[1][k])) for k in range(len(iu0[0])) if iu0[0][k] < iu0[1][k]]
            if len(pairs) == 0:
                pairs = [(int(rng.integers(0, n)), int(rng.integers(0, n)))]
            pi, pj = pairs[int(rng.integers(0, len(pairs)))]
            kick = pi if bool(rng.integers(0, 2)) else pj
            Y[kick] = Y[kick] + rng.normal(0, 0.25, size=d)
            Y = Y + rng.normal(0, 0.008, size=Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        else:
            # single worst to hole + small noise
            worst = np.argsort(-cnt)[:1]
            holes = _largest_holes(Y, worst, 1, rng, ncand=8000)
            Y[worst[0]] = holes[0]
            Y = Y + rng.normal(0, 0.006, size=Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        # fast smooth reconverge with randomized schedule
        a0 = 60 + 40 * float(rng.random())
        a1 = 400 + 400 * float(rng.random())
        lr0 = 0.015 + 0.01 * float(rng.random())
        lr1 = 0.001 + 0.002 * float(rng.random())
        iters = 1800 + int(rng.integers(0, 1400))
        Ys = _smooth(Y, iters, a0, a1, lr0, lr1)
        v = _max_cos(Ys)
        if v < best_v + 0.0025:
            remain = BUDGET - (time.time() - t0) - 22
            if remain < 0.6:
                break
            secs = min(0.9, remain)
            Yp = _lp_polish(Ys, secs)
            vp = _max_cos(Yp)
            if vp < best_v + 0.0012:
                Yp = _micro_hill(Yp, 0.3, rng)
                vp = _max_cos(Yp)
            if vp < best_v - 5e-13:
                best_x, best_v = Yp, vp
        # occasional opportunistic keep of near-best diverse basin
        # (no-op to stay deterministic)
    # final intensive polish
    while time.time() - t0 < BUDGET - 1.0:
        remain = BUDGET - (time.time() - t0) - 0.6
        if remain < 0.6:
            break
        s1 = min(5.0, remain * 0.5)
        best_x = _lp_polish(best_x, s1)
        best_v = _max_cos(best_x)
        remain2 = BUDGET - (time.time() - t0) - 0.3
        if remain2 < 0.4:
            break
        s2 = min(5.0, remain2)
        best_x = _micro_hill(best_x, s2, rng)
        best_v = _max_cos(best_x)
        if BUDGET - (time.time() - t0) < 1.5:
            break
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
