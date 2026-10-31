import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _norm(x):
    x = np.asarray(x, dtype=float)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n < 1e-12] = 1.0
    x = x / n
    bad = ~np.isfinite(x).all(axis=1)
    if np.any(bad):
        x[bad] = np.array([1.0, 0.0, 0.0, 0.0])
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _personal_worst(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)


def _find_void(others, rng, n_cand=6000, iters=160):
    cand = rng.standard_normal((n_cand, 4))
    cand = _norm(cand)
    sims = cand @ others.T
    scores = sims.max(axis=1)
    bi = int(np.argmin(scores))
    b = cand[bi].copy()
    bs = float(scores[bi])
    step = 0.06
    for _ in range(iters):
        prop = b + step * rng.standard_normal((32, 4))
        prop = _norm(prop)
        sc = (prop @ others.T).max(axis=1)
        j = int(np.argmin(sc))
        if float(sc[j]) < bs:
            b = prop[j].copy()
            bs = float(sc[j])
            step *= 1.06
        else:
            step *= 0.96
        if step < 2e-5:
            break
    return b


def _soft_run(y, steps, beta0, beta1, lr0, deadline=None):
    y = _norm(y)
    n = y.shape[0]
    for k in range(steps):
        if deadline is not None and time.time() > deadline:
            break
        frac = k / (steps - 1) if steps > 1 else 0.0
        beta = beta0 * (beta1 / beta0) ** frac if beta0 > 0 else beta1
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # weights focused near max
        D = G - m
        # beta*D <= 0, safe
        W = np.exp(beta * D)
        np.fill_diagonal(W, 0.0)
        # sparsify negligible weights
        W[W < 1e-12] = 0.0
        grad = W @ y
        proj = (grad * y).sum(axis=1, keepdims=True)
        gt = grad - proj * y
        nrm = float(np.mean(np.linalg.norm(gt, axis=1)))
        if not np.isfinite(nrm) or nrm < 1e-14:
            break
        lr = lr0 * (1.0 - 0.7 * frac)
        y = y - (lr / (nrm + 1e-12)) * gt
        y = _norm(y)
    return y


def _lp_polish(x, seconds, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        mcur = float(c.max())
        sel = c > mcur - max(20.0 * radius, 1e-9)
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
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
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _burden_delete_idx(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    pw = G.max(axis=1)
    deg = (G > m - 5e-4).sum(axis=1).astype(float)
    burden = pw + 2e-4 * deg
    return int(np.argmax(burden))


def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    try:
        x0 = np.load("best.npy")
    except Exception:
        x0 = np.load("record.npy")
    x0 = _norm(np.asarray(x0, dtype=float))
    if x0.shape != (272, 4):
        rng0 = np.random.default_rng(0)
        flat = x0.ravel()
        need = 272 * 4
        if flat.size < need:
            flat = np.concatenate([flat, rng0.standard_normal((need - flat.size,))])
        x0 = _norm(flat[:need].reshape(272, 4))
    best = x0.copy()
    best_val = _max_cos(best)

    def consider(cand):
        nonlocal best, best_val
        cand = _norm(cand)
        if cand.shape != (272, 4):
            return False
        v = _max_cos(cand)
        if v < best_val:
            best = cand.copy()
            best_val = v
            return True
        return False

    # quick polish of record to lock baseline
    if time.time() < deadline - 100:
        try:
            c0 = _lp_polish(best, 4.0, deadline)
            consider(c0)
        except Exception:
            pass

    # ---- upward continuation: 272 -> Nup -> 272 ----
    for (Nup, seed) in [(284, 11), (280, 22)]:
        if time.time() > deadline - 95:
            break
        try:
            rng = np.random.default_rng(seed)
            x_up = best.copy()
            # insert into largest voids
            cur = x_up.copy()
            for _ in range(Nup - 272):
                v = _find_void(cur, rng, n_cand=6000, iters=140)
                cur = np.vstack([cur, v])
            x_up = cur
            # fully reoptimize at Nup (forces core redistribution)
            x_up = _soft_run(x_up, 350, 25.0, 140.0, 0.06, deadline)
            x_up = _soft_run(x_up, 200, 80.0, 200.0, 0.03, deadline)
            # prune back one at a time with re-polish
            for _ in range(Nup - 272):
                if time.time() > deadline - 60:
                    break
                di = _burden_delete_idx(x_up)
                x_up = np.delete(x_up, di, axis=0)
                x_up = _soft_run(x_up, 90, 50.0, 150.0, 0.03, deadline)
            # final tighten at 272
            x_up = _soft_run(x_up, 150, 60.0, 220.0, 0.02, deadline)
            x_up = _lp_polish(x_up, 5.0, deadline)
            consider(x_up)
        except Exception:
            continue

    # ---- downward continuation: 272 -> Nlow -> 272 ----
    for (Nlow, seed, width) in [(252, 33, 40), (256, 44, 36)]:
        if time.time() > deadline - 55:
            break
        try:
            rng = np.random.default_rng(seed)
            x = best.copy()
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            pw = G.max(axis=1)
            order = np.argsort(-pw)
            pool = order[:width]
            kdel = 272 - Nlow
            victims = rng.choice(pool, size=kdel, replace=False)
            keep = np.ones(272, dtype=bool)
            keep[victims] = False
            x_low = x[keep].copy()
            # thoroughly reoptimize looser code
            x_low = _soft_run(x_low, 350, 25.0, 140.0, 0.06, deadline)
            x_low = _soft_run(x_low, 200, 80.0, 200.0, 0.03, deadline)
            # reinsert one at a time via largest void + global polish
            for _ in range(kdel):
                if time.time() > deadline - 35:
                    break
                v = _find_void(x_low, rng, n_cand=6000, iters=140)
                x_low = np.vstack([x_low, v])
                x_low = _soft_run(x_low, 110, 50.0, 160.0, 0.03, deadline)
            x_low = _soft_run(x_low, 150, 60.0, 220.0, 0.02, deadline)
            x_low = _lp_polish(x_low, 5.0, deadline)
            consider(x_low)
        except Exception:
            continue

    # ---- final polish of best with remaining time ----
    try:
        remain = deadline - time.time()
        if remain > 6:
            polished = _lp_polish(best, min(remain - 1.0, 18.0), deadline)
            consider(polished)
    except Exception:
        pass

    best = _norm(best)
    return best
