import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _adam_refine(X, iters, a0, a1, s0, s1, beta1=0.9, beta2=0.999, eps=1e-8):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1p = 1.0
    b2p = 1.0
    for it in range(iters):
        frac = it / max(1, iters - 1)
        alpha = a0 * (a1 / a0) ** frac if iters > 1 else a0
        step = s0 * (s1 / s0) ** frac if iters > 1 else s0
        G = X @ X.T
        np.fill_diagonal(G, -np.inf)
        gmax = float(G.max())
        # cutoff far pairs to focus gradient
        cut = gmax - 15.0 / max(1.0, alpha)
        Gm = G - gmax
        # exp, zero out far
        W = np.exp(alpha * Gm, out=np.empty_like(G))
        W[G < cut] = 0.0
        np.fill_diagonal(W, 0.0)
        F = W @ X
        dot = np.sum(F * X, axis=1, keepdims=True)
        Gt = F - dot * X
        # adam update
        b1p *= beta1
        b2p *= beta2
        m *= beta1
        m += (1.0 - beta1) * Gt
        v *= beta2
        v += (1.0 - beta2) * (Gt * Gt)
        mhat = m / (1.0 - b1p)
        vhat = v / (1.0 - b2p)
        upd = mhat / (np.sqrt(vhat) + eps)
        # project update to tangent
        ud = np.sum(upd * X, axis=1, keepdims=True)
        upd = upd - ud * X
        # clip extreme
        X = X - step * upd
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-30
    return X

def _hole_vector(X_rem, rng, n_sample=7000):
    n, d = X_rem.shape
    C = rng.standard_normal((n_sample, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-30
    dots = C @ X_rem.T
    m = dots.max(axis=1)
    idx = np.argpartition(m, 4)[:4]
    best_v = None
    best_s = 1e9
    for k in idx:
        vv = C[k].copy()
        lr0 = 0.08
        for it in range(300):
            dts = X_rem @ vv
            cur = float(dts.max())
            act = dts > cur - 0.04
            g = X_rem[act].sum(axis=0)
            gt = g - float(g @ vv) * vv
            nn = float(np.linalg.norm(gt)) + 1e-12
            gt /= nn
            st = lr0 * (0.15 ** (it / 300.0))
            vv = vv - st * gt
            vv /= np.linalg.norm(vv) + 1e-30
        s = float((X_rem @ vv).max())
        if s < best_s:
            best_s = s
            best_v = vv
    return best_v

def _lp_polish(X, seconds):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-30)
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
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True) + 1e-30
        vv = _max_cos(y)
        if vv < best:
            X, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    BUDGET = 173.0
    rng = np.random.default_rng(12345)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.array(a, dtype=float)
            a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-30)
            if a.shape == (162, 8):
                cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        X0 = rng.standard_normal((162, 8))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True) + 1e-30
    else:
        X0 = min(cands, key=lambda c: _max_cos(c))
    X0 = X0 / (np.linalg.norm(X0, axis=1, keepdims=True) + 1e-30)
    best_X = X0.copy()
    best_v = _max_cos(best_X)
    # quick jam of starting point
    try:
        tmp = _adam_refine(best_X.copy(), 700, 25.0, 350.0, 0.012, 0.0012)
        vt = _max_cos(tmp)
        if vt < best_v:
            best_v = vt
            best_X = tmp
    except Exception:
        pass
    n = 162
    hopping_deadline = t0 + BUDGET - 32.0
    trial = 0
    while time.time() < hopping_deadline and trial < 300:
        trial += 1
        r = rng.random()
        cand = best_X.copy()
        if r < 0.45:
            G = cand @ cand.T
            np.fill_diagonal(G, -np.inf)
            per = G.max(axis=1)
            k = 1 if rng.random() < 0.55 else (2 if rng.random() < 0.8 else 3)
            worst = np.argsort(-per)[:k]
            if rng.random() < 0.25:
                worst[0] = int(rng.integers(0, n))
            mask = np.ones(n, dtype=bool)
            mask[worst] = False
            tmp_rem = cand[mask]
            new_vs = []
            for _ in range(k):
                hv = _hole_vector(tmp_rem, rng, n_sample=6000)
                new_vs.append(hv)
                tmp_rem = np.vstack([tmp_rem, hv[None, :]])
            for wi, hv in zip(worst, new_vs):
                cand[wi] = hv
            a0 = float(rng.uniform(18.0, 35.0))
            a1 = float(rng.uniform(250.0, 450.0))
            cand = _adam_refine(cand, 650, a0, a1, 0.016, 0.0012)
        elif r < 0.75:
            # shake high-contact cluster + rattlers
            G = cand @ cand.T
            np.fill_diagonal(G, -np.inf)
            per = G.max(axis=1)
            order = np.argsort(-per)
            s = int(rng.integers(4, 14))
            idx = order[:s] if rng.random() < 0.6 else rng.choice(n, size=s, replace=False)
            sigma = float(rng.uniform(0.06, 0.28))
            cand[idx] += sigma * rng.standard_normal((s, 8))
            cand[idx] /= np.linalg.norm(cand[idx], axis=1, keepdims=True) + 1e-30
            cand = _adam_refine(cand, 650, 22.0, 350.0, 0.014, 0.001)
        else:
            sigma = float(rng.uniform(0.004, 0.028))
            cand = cand + sigma * rng.standard_normal(cand.shape)
            cand /= np.linalg.norm(cand, axis=1, keepdims=True) + 1e-30
            cand = _adam_refine(cand, 500, 28.0, 300.0, 0.010, 0.001)
        v = _max_cos(cand)
        if v < best_v - 1e-12:
            best_v = v
            best_X = cand
            if time.time() < hopping_deadline - 2:
                try:
                    ref = _adam_refine(best_X.copy(), 700, 40.0, 450.0, 0.008, 0.0006)
                    vr = _max_cos(ref)
                    if vr < best_v:
                        best_v = vr
                        best_X = ref
                except Exception:
                    pass
    remain = BUDGET - (time.time() - t0)
    if remain > 8:
        try:
            it2 = 1200 if remain > 25 else 600
            tmp = _adam_refine(best_X.copy(), it2, 30.0, 550.0, 0.008, 0.0006)
            vt = _max_cos(tmp)
            if vt < best_v:
                best_v = vt
                best_X = tmp
        except Exception:
            pass
    remain = BUDGET - (time.time() - t0)
    lp_sec = min(27.0, max(4.0, remain - 2.0))
    if lp_sec > 3:
        try:
            pol = _lp_polish(best_X.copy(), lp_sec)
            vp = _max_cos(pol)
            if vp < best_v:
                best_v = vp
                best_X = pol
        except Exception:
            pass
    best_X = best_X / (np.linalg.norm(best_X, axis=1, keepdims=True) + 1e-30)
    return best_X
