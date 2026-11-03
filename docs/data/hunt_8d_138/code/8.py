import time
import numpy as np

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _soft_adam(x, steps, beta0, beta1, lr, seed_off=0):
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
        # avoid overflow: exp(beta*(G-mv)), max=1
        W = np.exp(beta * (G - mv))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        nd = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - nd * x
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
        if cnt > 2800:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:2800]
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

def _degrees(x, m, tol):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    cnt = (G > m - tol).sum(axis=1)
    mx = G.max(axis=1)
    return cnt, mx, G

def _void_replace(x, rng, k, n_cand=8000):
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    mv = G.max()
    cnt = (G > mv - 0.015).sum(axis=1)
    mx = G.max(axis=1)
    # victim score: high degree + high max
    score = cnt.astype(float) * 2.0 + mx
    victims = np.argsort(-score)[:k]
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[victims] = False
    kept = x[keep_mask]
    new_pts = []
    cur_set = kept
    for _ in range(k):
        cand = rng.normal(size=(n_cand, d))
        cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
        # best void = minimize max dot to cur_set
        S = cand @ cur_set.T
        worst = S.max(axis=1)
        best_i = int(np.argmin(worst))
        bp = cand[best_i]
        new_pts.append(bp)
        cur_set = np.vstack([cur_set, bp[None, :]])
    y = x.copy()
    for vi, bp in zip(victims, new_pts):
        y[vi] = bp
    return y

def solve():
    import time
    rng = np.random.default_rng(0)
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
    # archive of diverse elites for splicing
    elites = [best.copy()]
    elite_vals = [best_val]
    cur = best.copy()
    cur_val = best_val

    # quick tighten of incumbent if time allows
    try:
        best = _lp_polish(best, min(deadline - 10, time.time() + 3.0))
        best_val = _max_cos(best)
        elites[0] = best.copy(); elite_vals[0] = best_val
        cur = best.copy(); cur_val = best_val
    except Exception:
        pass

    it = 0
    while time.time() < deadline - 9.0 and it < 300:
        it += 1
        now = time.time()
        budget_left = deadline - 9.0 - now
        if budget_left <= 0:
            break
        mode = it % 10
        y = None
        if mode == 9 and len(elites) >= 2 and it > 15:
            # hemispheric splice crossover between two elites
            a = elites[rng.integers(len(elites))]
            b = elites[rng.integers(len(elites))]
            u = rng.normal(size=(d,)); u /= np.linalg.norm(u)
            pa = a @ u; pb = b @ u
            # align: take hemisphere from each
            y = a.copy()
            msk = pb < 0  # take negative side from b? need same count; just blend
            # build mixed set: half from a (pa>median), half from b
            ia = np.argsort(-pa); ib = np.argsort(pb)
            h = n // 2
            y = np.vstack([a[ia[:h]], b[ib[:n - h]]])
            # if duplicate count mismatch, y has n rows
            y = y[rng.permutation(n)]
            y = y + 0.02 * rng.normal(size=(n, d))
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            steps, b0, b1, lr = 700, 15.0, 220.0, 0.006
        elif mode in (0, 5):
            # kill worst k into voids
            k = int(rng.choice([2, 3, 4, 5, 6]))
            src = cur if rng.random() < 0.7 else best
            y = _void_replace(src, rng, k, n_cand=6000)
            y = y + 0.008 * rng.normal(size=(n, d))
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            steps, b0, b1, lr = 650, 15.0, 200.0, 0.007
        elif mode in (1, 6):
            # targeted shake of overloaded points
            src = cur if rng.random() < 0.5 else best
            G = src @ src.T; np.fill_diagonal(G, -2.0)
            mv = G.max()
            cnt = (G > mv - 0.015).sum(axis=1)
            order = np.argsort(-cnt)
            pool = order[:12]
            y = src.copy()
            sig_big = 0.28
            sig_small = 0.015
            y[pool] += sig_big * rng.normal(size=(len(pool), d))
            y += sig_small * rng.normal(size=(n, d))
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            steps, b0, b1, lr = 650, 12.0, 180.0, 0.007
        elif mode == 3:
            # fresh random basin (rare)
            y = rng.normal(size=(n, d))
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            steps, b0, b1, lr = 1100, 8.0, 300.0, 0.008
        else:
            # global gaussian kick around best/cur
            sig = float(rng.choice([0.02, 0.05, 0.09, 0.18, 0.30]))
            src = best if (it % 2 == 0) else cur
            y = src + sig * rng.normal(size=(n, d))
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            if rng.random() < 0.35:
                # random rebirth of few rattlers/worst
                G = y @ y.T; np.fill_diagonal(G, -2.0)
                mv = G.max()
                mx = G.max(axis=1)
                victims = np.argsort(-mx)[:int(rng.integers(2, 6))]
                rp = rng.normal(size=(len(victims), d))
                rp = rp / np.linalg.norm(rp, axis=1, keepdims=True)
                y[victims] = rp
            steps, b0, b1, lr = 600, 18.0, 250.0, 0.006

        # anneal
        y = _soft_adam(y, steps=steps, beta0=b0, beta1=b1, lr=lr)
        v_anneal = _max_cos(y)
        # filter: only LP-polish if promising enough (save time)
        # always polish fresh or void moves; skip hopeless kicks
        if v_anneal > best_val + 0.008 and mode != 3:
            # accept as cur with small prob to diversify
            if v_anneal < cur_val:
                cur, cur_val = y.copy(), v_anneal
            continue
        lp_budget = min(deadline - 9.0, time.time() + 1.6)
        try:
            y = _lp_polish(y, lp_budget)
        except Exception:
            pass
        v = _max_cos(y)
        if v < best_val - 1e-12:
            best_val = v
            best = y.copy()
            cur, cur_val = y.copy(), v
            elites.append(y.copy()); elite_vals.append(v)
            # keep top 5
            o = np.argsort(elite_vals)
            elites = [elites[i] for i in o[:5]]
            elite_vals = [elite_vals[i] for i in o[:5]]
        elif v < cur_val:
            cur, cur_val = y.copy(), v
            # add diversity to elites
            if len(elites) < 5 or v < max(elite_vals) - 1e-6:
                elites.append(y.copy()); elite_vals.append(v)
                o = np.argsort(elite_vals)
                elites = [elites[i] for i in o[:5]]
                elite_vals = [elite_vals[i] for i in o[:5]]
        else:
            # occasional sideways move
            if rng.random() < 0.08:
                cur, cur_val = y.copy(), v

    try:
        best = _lp_polish(best, min(deadline - 1.0, time.time() + 6.0))
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
