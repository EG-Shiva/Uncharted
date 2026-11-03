import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _smooth_refine(X, iters, alpha0, alpha1, step0, step1, rng=None):
    n, d = X.shape
    for it in range(iters):
        frac = it / max(1, iters - 1)
        alpha = alpha0 * (alpha1 / alpha0) ** frac if iters > 1 else alpha0
        step = step0 * (step1 / step0) ** frac if iters > 1 else step0
        G = X @ X.T
        np.fill_diagonal(G, -np.inf)
        gmax = float(G.max())
        # weights: exp(alpha*(G-gmax))
        W = np.exp(alpha * (G - gmax))
        # sparsify tiny weights
        # (keeps gradient focused, saves nothing but stabilizes)
        W[W < 1e-9] = 0.0
        F = W @ X
        dot = np.sum(F * X, axis=1, keepdims=True)
        Ft = F - dot * X
        nrm = float(np.linalg.norm(Ft, axis=1).mean()) + 1e-12
        X = X - (step / nrm) * Ft
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-30
    return X

def _hole_vector(X_rem, rng, n_sample=8000):
    # find vector on S^7 minimizing max dot to X_rem
    n, d = X_rem.shape
    # random candidates
    C = rng.standard_normal((n_sample, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-30
    dots = C @ X_rem.T  # C x n
    m = dots.max(axis=1)
    idx = np.argpartition(m, 4)[:4]
    best_v = None
    best_s = 1e9
    for k in idx:
        v = C[k].copy()
        # hill climb to push away from closest points
        lr = 0.05
        for it in range(250):
            dts = X_rem @ v
            cur = float(dts.max())
            act = dts > cur - 0.03
            g = X_rem[act].sum(axis=0)
            # tangent component
            gt = g - float(g @ v) * v
            nrm = float(np.linalg.norm(gt)) + 1e-12
            gt /= nrm
            step = lr * (0.2 ** (it / 250))
            v = v - step * gt
            v /= np.linalg.norm(v) + 1e-30
        s = float((X_rem @ v).max())
        if s < best_s:
            best_s = s
            best_v = v
    return best_v

def _lp_polish(X, seconds):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    try:
        best = _max_cos(X)
    except Exception:
        return X
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 1.0:
            break
        Gfull = X @ X.T
        c = Gfull[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        # cap size to keep linprog tractable
        if int(sel.sum()) > 4000:
            thr = np.sort(c[sel])[-4000]
            sel = c >= thr
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True) + 1e-30
        v = _max_cos(y)
        if v < best:
            X, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    BUDGET = 175.0
    rng = np.random.default_rng(0)
    # load start
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        cands.append(np.array(xb, dtype=float))
    if xr is not None:
        cands.append(np.array(xr, dtype=float))
    if len(cands) == 0:
        X0 = rng.standard_normal((162, 8))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
    else:
        # pick better
        best_pick = None
        best_v = 1e9
        for c in cands:
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            v = _max_cos(c)
            if v < best_v:
                best_v = v
                best_pick = c
        X0 = best_pick
    X0 = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
    best_X = X0.copy()
    best_v = _max_cos(best_X)

    # quick initial smooth to ensure local tightness
    # (short, to leave time for hopping)
    best_X = _smooth_refine(best_X.copy(), 400, 30.0, 250.0, 0.02, 0.002)
    v = _max_cos(best_X)
    if v < best_v:
        best_v = v
    else:
        # keep original if worse (avoid drift)
        pass

    # basin hopping; reserve ~30s for final polish
    hopping_deadline = t0 + BUDGET - 32.0
    trial = 0
    n = 162
    while time.time() < hopping_deadline and trial < 400:
        trial += 1
        r = rng.random()
        cand = best_X.copy()
        if r < 0.45:
            # kill worst 1-2 points and refill holes
            G = cand @ cand.T
            np.fill_diagonal(G, -np.inf)
            per = G.max(axis=1)
            k = 1 if rng.random() < 0.6 else 2
            worst = np.argsort(-per)[:k]
            # also occasionally include random high-degree point
            if rng.random() < 0.3:
                worst[0] = int(rng.integers(0, n))
            mask = np.ones(n, dtype=bool)
            mask[worst] = False
            X_rem = cand[mask]
            # refill sequentially (update X_rem each time)
            new_vs = []
            tmp_rem = X_rem
            for _ in range(k):
                hv = _hole_vector(tmp_rem, rng, n_sample=6000)
                new_vs.append(hv)
                tmp_rem = np.vstack([tmp_rem, hv[None, :]])
            for wi, hv in zip(worst, new_vs):
                cand[wi] = hv
            cand = _smooth_refine(cand, 600, 25.0, 300.0, 0.025, 0.002)
        elif r < 0.75:
            # subset strong shake
            s = int(rng.integers(4, 12))
            idx = rng.choice(n, size=s, replace=False)
            sigma = float(rng.uniform(0.08, 0.30))
            cand[idx] += sigma * rng.standard_normal((s, 8))
            cand[idx] /= np.linalg.norm(cand[idx], axis=1, keepdims=True) + 1e-30
            cand = _smooth_refine(cand, 600, 25.0, 300.0, 0.02, 0.0015)
        else:
            # global light jitter
            sigma = float(rng.uniform(0.005, 0.03))
            cand = cand + sigma * rng.standard_normal(cand.shape)
            cand /= np.linalg.norm(cand, axis=1, keepdims=True) + 1e-30
            cand = _smooth_refine(cand, 450, 30.0, 250.0, 0.015, 0.0015)

        v = _max_cos(cand)
        if v < best_v - 1e-12:
            best_v = v
            best_X = cand
            # extra exploitation: longer refine of new best
            if time.time() < hopping_deadline - 2:
                refined = _smooth_refine(best_X.copy(), 800, 40.0, 400.0, 0.012, 0.001)
                vr = _max_cos(refined)
                if vr < best_v:
                    best_v = vr
                    best_X = refined
        # early stop if clearly improved a lot? continue to push
        if best_v < 0.4935:
            # keep pushing but allow more time
            pass

    # final intensification
    remain = BUDGET - (time.time() - t0)
    # allocate: smooth long then LP
    if remain > 8:
        # long smooth
        try:
            tmp = _smooth_refine(best_X.copy(), 1200, 30.0, 500.0, 0.01, 0.0008)
            vt = _max_cos(tmp)
            if vt < best_v:
                best_v = vt
                best_X = tmp
        except Exception:
            pass
    remain = BUDGET - (time.time() - t0)
    lp_sec = min(28.0, max(5.0, remain - 2.0))
    if lp_sec > 3:
        try:
            polished = _lp_polish(best_X.copy(), lp_sec)
            vp = _max_cos(polished)
            if vp < best_v:
                best_v = vp
                best_X = polished
        except Exception:
            pass
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    # shrink guard not needed; just return
    return best_X
