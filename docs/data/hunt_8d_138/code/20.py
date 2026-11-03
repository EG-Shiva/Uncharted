import time
import numpy as np

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _soft_adam(x, steps, beta0, beta1, lr, noise=0.0, rng=None):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        f = (t - 1) / max(steps - 1, 1)
        beta = beta0 + (beta1 - beta0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mv = G.max()
        W = np.exp(beta * (G - mv))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        nd = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - nd * x
        if noise > 0 and rng is not None:
            gt = gt + noise * rng.normal(size=(n, d))
            # project noise tangentially
            ndn = (gt * x).sum(axis=1, keepdims=True)
            # gt already tangential approx; re-project
            pass
        m *= b1
        m += (1.0 - b1) * gt
        v *= b2
        v += (1.0 - b2) * (gt * gt)
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        upd = (m / bc1) / (np.sqrt(v / bc2) + eps)
        nd2 = (upd * x).sum(axis=1, keepdims=True)
        upd = upd - nd2 * x
        x = x - lr * upd
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _lp_polish(x, deadline):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        c = Gfull[iu]
        mv = c.max()
        thr = mv - max(20.0 * radius, 1e-9)
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 3000:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:3000]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv)).tocsr()
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
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _void_refine(v0, Xrem, beta=120.0, steps=120, lr=0.08):
    v = v0 / (np.linalg.norm(v0) + 1e-30)
    for t in range(steps):
        dots = Xrem @ v
        mv = dots.max()
        W = np.exp(beta * (dots - mv))
        S = W.sum() + 1e-300
        g = (W @ Xrem) / S
        gt = g - np.dot(g, v) * v
        st = lr / (1.0 + t * 0.02)
        v = v - st * gt
        nrm = np.linalg.norm(v) + 1e-30
        v = v / nrm
    return v

def _void_replace(src, rng, k, n_cand=6000):
    n, d = src.shape
    G = src @ src.T
    np.fill_diagonal(G, -2.0)
    mv = G.max()
    cnt = (G > mv - 0.015).sum(axis=1)
    mx = G.max(axis=1)
    score = cnt.astype(float) * 2.0 + mx
    victims = np.argsort(-score)[:k]
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[victims] = False
    kept = src[keep_mask]
    new_pts = []
    cur_set = kept
    for _ in range(k):
        cand = rng.normal(size=(n_cand, d))
        cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
        S = cand @ cur_set.T
        worst = S.max(axis=1)
        bi = int(np.argmin(worst))
        bp = cand[bi]
        bp = _void_refine(bp, cur_set, beta=120.0, steps=80, lr=0.08)
        new_pts.append(bp)
        cur_set = np.vstack([cur_set, bp[None, :]])
    y = src.copy()
    for vi, bp in zip(victims, new_pts):
        y[vi] = bp
    return y

def solve():
    import time
    rng = np.random.default_rng(1)
    t0 = time.time()
    deadline = t0 + 168.0
    n, d = 138, 8
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = rng.normal(size=(n, d))
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    try:
        best = _lp_polish(best, min(deadline - 10, time.time() + 2.5))
        best_val = _max_cos(best)
    except Exception:
        pass
    cur = best.copy()
    cur_val = best_val
    elites = [best.copy()]
    elite_vals = [best_val]
    it = 0
    while time.time() < deadline - 9.0 and it < 400:
        it += 1
        if time.time() > deadline - 9.0:
            break
        mode = it % 12
        y = None
        if mode in (0, 6):
            k = int(rng.choice([6, 8, 10, 12, 14]))
            src = cur if rng.random() < 0.6 else best
            if len(elites) >= 2 and rng.random() < 0.25:
                src = elites[rng.integers(len(elites))]
            y = _void_replace(src, rng, k, n_cand=6000)
            y = y + 0.006 * rng.normal(size=(n, d))
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            steps, b0, b1, lr = 450, 12.0, 220.0, 0.007
        elif mode in (1, 7):
            src = cur if rng.random() < 0.5 else best
            G = src @ src.T
            np.fill_diagonal(G, -2.0)
            mv = G.max()
            cnt = (G > mv - 0.015).sum(axis=1)
            order = np.argsort(-cnt)
            pool = order[:20]
            y = src.copy()
            y[pool] += 0.35 * rng.normal(size=(len(pool), d))
            y += 0.012 * rng.normal(size=(n, d))
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            steps, b0, b1, lr = 450, 10.0, 200.0, 0.007
        elif mode in (2, 8, 10):
            sig = float(rng.choice([0.05, 0.10, 0.18, 0.30, 0.45]))
            src = best if (it % 2 == 0) else cur
            y = src + sig * rng.normal(size=(n, d))
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            if rng.random() < 0.4:
                G = y @ y.T
                np.fill_diagonal(G, -2.0)
                mx = G.max(axis=1)
                victims = np.argsort(-mx)[:int(rng.integers(2, 7))]
                rp = rng.normal(size=(len(victims), d))
                rp = rp / np.linalg.norm(rp, axis=1, keepdims=True)
                y[victims] = rp
            steps, b0, b1, lr = 400, 15.0, 260.0, 0.006
        elif mode == 4:
            # rattler rebirth: replace low-contact points by voids
            src = cur if rng.random() < 0.5 else best
            G = src @ src.T
            np.fill_diagonal(G, -2.0)
            mv = G.max()
            cnt = (G > mv - 0.01).sum(axis=1)
            # lowest contact points
            order = np.argsort(cnt)
            k = int(rng.integers(3, 9))
            victims = order[:k]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[victims] = False
            kept = src[keep_mask]
            cur_set = kept
            new_pts = []
            for _ in range(k):
                cand = rng.normal(size=(6000, d))
                cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
                worst = (cand @ cur_set.T).max(axis=1)
                bi = int(np.argmin(worst))
                bp = _void_refine(cand[bi], cur_set)
                new_pts.append(bp)
                cur_set = np.vstack([cur_set, bp[None, :]])
            y = src.copy()
            for vi, bp in zip(victims, new_pts):
                y[vi] = bp
            y = y + 0.004 * rng.normal(size=(n, d))
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            steps, b0, b1, lr = 450, 12.0, 220.0, 0.007
        else:
            # fresh random basin
            y = rng.normal(size=(n, d))
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            steps, b0, b1, lr = 700, 8.0, 320.0, 0.008
        # anneal
        y = _soft_adam(y, steps=steps, beta0=b0, beta1=b1, lr=lr)
        v_anneal = _max_cos(y)
        # filter hopeless to save LP time (allow fresh basins more slack)
        thresh = best_val + (0.015 if mode == 3 else 0.004)
        if v_anneal > thresh:
            if v_anneal < cur_val:
                cur, cur_val = y.copy(), v_anneal
            continue
        try:
            y = _lp_polish(y, min(deadline - 9.0, time.time() + 1.4))
        except Exception:
            pass
        v = _max_cos(y)
        if v < best_val - 1e-12:
            best_val = v
            best = y.copy()
            cur, cur_val = y.copy(), v
            elites.append(y.copy())
            elite_vals.append(v)
            o = np.argsort(elite_vals)
            elites = [elites[i] for i in o[:5]]
            elite_vals = [elite_vals[i] for i in o[:5]]
        elif v < cur_val:
            cur, cur_val = y.copy(), v
            if len(elites) < 5 or v < max(elite_vals) - 5e-7:
                elites.append(y.copy())
                elite_vals.append(v)
                o = np.argsort(elite_vals)
                elites = [elites[i] for i in o[:5]]
                elite_vals = [elite_vals[i] for i in o[:5]]
        else:
            if rng.random() < 0.07:
                cur, cur_val = y.copy(), v
    try:
        best = _lp_polish(best, min(deadline - 1.0, time.time() + 6.0))
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
