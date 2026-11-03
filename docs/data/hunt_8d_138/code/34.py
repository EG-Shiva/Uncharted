import time
import numpy as np

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _soft_adam(x, steps, beta0, beta1, lr):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        f = (t - 1) / max(steps - 1, 1)
        beta = beta0 + (beta1 - beta0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mv = float(G.max())
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

def _lp_polish(x, deadline, cap=2500):
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
        mv = float(c.max())
        thr = mv - max(20.0 * radius, 1e-9)
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:cap]
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

def _void_refine(v0, Xrem, beta=150.0, steps=150, lr=0.08):
    v = v0 / (np.linalg.norm(v0) + 1e-30)
    for t in range(steps):
        dots = Xrem @ v
        mv = float(dots.max())
        W = np.exp(beta * (dots - mv))
        S = float(W.sum()) + 1e-300
        g = (W @ Xrem) / S
        gt = g - float(np.dot(g, v)) * v
        st = lr / (1.0 + t * 0.02)
        v = v - st * gt
        v = v / (np.linalg.norm(v) + 1e-30)
    return v

def _prune_to_138(X142):
    X = X142.copy()
    while X.shape[0] > 138:
        n = X.shape[0]
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        best_i = -1
        best_v = 1e9
        for i in range(n):
            # max over pairs not involving i
            # mask row/col i
            m = -2.0
            # brute but n small
            for j in range(n):
                if j == i:
                    continue
                row = G[j].copy()
                row[i] = -2.0
                # need only j<k or all; max over row
                rmax = float(row.max())
                if rmax > m:
                    m = rmax
                    if m >= best_v:
                        break
            if m < best_v:
                best_v = m
                best_i = i
        X = np.delete(X, best_i, axis=0)
    return X

def solve():
    t0 = time.time()
    deadline = t0 + 168.0
    rng = np.random.default_rng(7)
    n, d = 138, 8
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            vv = np.load(fn).astype(float)
            if vv.shape == (n, d):
                vv = vv / np.linalg.norm(vv, axis=1, keepdims=True)
                c = _max_cos(vv)
                if c < best_val:
                    best_val = c
                    best = vv.copy()
        except Exception:
            pass
    if best is None:
        best = rng.normal(size=(n, d))
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    # light initial polish
    try:
        best = _lp_polish(best, min(deadline - 10, time.time() + 2.0), cap=3000)
        best_val = _max_cos(best)
    except Exception:
        pass
    base_best = best.copy()
    base_val = best_val

    # identify 30 rattlers (lowest contacts at 1e-9)
    try:
        G = base_best @ base_best.T
        np.fill_diagonal(G, -2.0)
        mv0 = float(G.max())
        cnt0 = (G > mv0 - 1e-9).sum(axis=1)
        order0 = np.argsort(cnt0)  # ascending
        victims0 = order0[:30]
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[victims0] = False
        core_init = base_best[keep_mask].copy()
    except Exception:
        core_init = base_best[:108].copy()
    # LP-polish the 108-point core
    try:
        core = _lp_polish(core_init, min(deadline - 10, time.time() + 2.0), cap=2500)
    except Exception:
        core = core_init.copy()
    core_val = _max_cos(core)

    glob_best = base_best.copy()
    glob_val = base_val

    run = 0
    # N-homotopy runs until time runs out
    while time.time() < deadline - 12.0 and run < 8:
        run += 1
        X = core.copy()
        # small random tangential kick to core to diversify runs
        if run > 1:
            amp = 0.003 * run
            X = X + amp * rng.normal(size=X.shape)
            X = X / np.linalg.norm(X, axis=1, keepdims=True)
            try:
                X = _lp_polish(X, min(deadline - 10, time.time() + 0.4), cap=1500)
            except Exception:
                pass
        # grow 108 -> 142
        cand138 = None
        cand138_val = 1e9
        failed = False
        for target in range(X.shape[0], 142):
            if time.time() > deadline - 11.0:
                failed = True
                break
            m_cur = X.shape[0]
            # sample 50000 random directions, find 5 best holes
            Ncand = 50000
            cand = rng.normal(size=(Ncand, d))
            cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
            worst = (cand @ X.T).max(axis=1)
            part = np.argpartition(worst, 5)[:5]
            o = part[np.argsort(worst[part])]
            pick = o[int(rng.integers(5))]
            v0 = cand[int(pick)]
            # locally maximize min cosine by projected gradient ascent
            try:
                vstar = _void_refine(v0, X, beta=150.0, steps=150, lr=0.08)
            except Exception:
                vstar = v0
            Y = np.vstack([X, vstar[None, :]])
            # short global soft relaxation
            try:
                Y = _soft_adam(Y, steps=150, beta0=8.0, beta1=150.0, lr=0.006)
            except Exception:
                pass
            # LP-polish after every addition
            try:
                budget = 0.35 if m_cur < 135 else 0.5
                Y = _lp_polish(Y, min(deadline - 10, time.time() + budget), cap=1500)
            except Exception:
                pass
            X = Y
            if X.shape[0] == 138:
                vv = _max_cos(X)
                cand138 = X.copy()
                cand138_val = vv
                if vv < glob_val - 1e-12:
                    glob_val = vv
                    glob_best = X.copy()
        if failed:
            # if we at least reached 138, prune/keep
            if cand138 is not None and cand138_val < glob_val + 0.004:
                pass
            continue
        # X is now 142; prune back to 138 by greedy deletion
        try:
            Xp = _prune_to_138(X)
            try:
                Xp = _lp_polish(Xp, min(deadline - 10, time.time() + 0.8), cap=2500)
            except Exception:
                pass
            vp = _max_cos(Xp)
            if vp < glob_val - 1e-12:
                glob_val = vp
                glob_best = Xp.copy()
            # also consider intermediate 138
            if cand138 is not None:
                try:
                    c2 = _lp_polish(cand138, min(deadline - 10, time.time() + 0.5), cap=2000)
                    v2 = _max_cos(c2)
                    if v2 < glob_val - 1e-12:
                        glob_val = v2
                        glob_best = c2.copy()
                except Exception:
                    pass
        except Exception:
            pass
        # also try pruning 142 -> keep diversity: random alternative prune (remove 4 with best void score)
        if time.time() > deadline - 12.0:
            break

    # final choice
    final = glob_best if glob_val <= base_val else base_best
    try:
        final = _lp_polish(final, min(deadline - 1.0, time.time() + 5.0), cap=3000)
    except Exception:
        pass
    final = final / np.linalg.norm(final, axis=1, keepdims=True)
    return final
