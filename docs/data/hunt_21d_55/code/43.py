import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _refine_single(pt, Y, steps=300, a0=40.0, a1=600.0, lr0=0.03, lr1=0.002):
    # annealed LogSumExp Adam for one point against fixed Y
    x = pt.copy()
    d = x.shape[0]
    m = np.zeros(d)
    vv = np.zeros(d)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        frac = t / steps
        alpha = a0 * (a1 / a0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        dots = Y @ x
        mx = float(np.max(dots))
        e = np.exp(alpha * (dots - mx))
        s = float(np.sum(e))
        if not np.isfinite(s) or s <= 0:
            continue
        w = e / s
        grad = w @ Y
        gt = grad - float(np.dot(grad, x)) * x
        m = b1 * m + (1.0 - b1) * gt
        vv = b2 * vv + (1.0 - b2) * (gt * gt)
        mhat = m / (1.0 - b1 ** t)
        vhat = vv / (1.0 - b2 ** t)
        x = x - lr * mhat / (np.sqrt(vhat) + eps)
        nrm = float(np.linalg.norm(x))
        if not np.isfinite(nrm) or nrm < 1e-12:
            x = pt.copy()
            m[:] = 0.0
            vv[:] = 0.0
            continue
        x /= nrm
    return x


def _best_hole(Y, rng, n_cand=20000):
    d = Y.shape[1]
    cand = rng.standard_normal((n_cand, d))
    cand = _norm_rows(cand)
    dots = cand @ Y.T
    md = np.max(dots, axis=1)
    return cand[np.argmin(md)]


def _lp_polish_deadline(x, deadline, radius_init=1e-3):
    x = _norm_rows(np.asarray(x, dtype=np.float64))
    n, d = x.shape
    best = _max_cos(x)
    radius = radius_init
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        m = float(np.max(cfull))
        sel = cfull > m - max(20.0 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 1100:
            idx = np.where(sel)[0]
            order = np.argsort(-cfull[idx])[:1100]
            nsel = np.zeros_like(sel)
            nsel[idx[order]] = True
            sel = nsel
        csel = cfull[sel]
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        if time.time() >= deadline:
            break
        y = x + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    T_SEARCH_END = t0 + 150.0
    T_HARD = t0 + 165.0
    rng = np.random.default_rng(20260601)
    n, d = 55, 21
    x_base = None
    best_score = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = _norm_rows(a)
                s = _max_cos(a)
                if s < best_score:
                    best_score = s
                    x_base = a.copy()
        except Exception:
            pass
    if x_base is None:
        x_base = _norm_rows(rng.standard_normal((n, d)))
        best_score = _max_cos(x_base)
    x_best = x_base.copy()
    best = best_score

    it = 0
    while time.time() < T_SEARCH_END:
        it += 1
        if time.time() >= T_SEARCH_END - 0.2:
            break
        # contact graph from current best
        G = x_best @ x_best.T
        np.fill_diagonal(G, -2.0)
        thr = best - 1e-6
        # contact counts
        counts = np.sum(G > thr, axis=1)
        # pick hub among top-5 to diversify, strictly highest-contact biased
        order = np.argsort(-counts)
        topk = order[:5]
        hub = int(topk[rng.integers(0, len(topk))])
        k = int(rng.integers(6, 10))  # 6..9
        # BFS cluster
        visited = set([hub])
        queue = [hub]
        # precompute neighbour order for speed
        while len(visited) < k and len(queue) > 0:
            cur = queue.pop(0)
            row = G[cur]
            neigh = np.where(row > thr)[0]
            if len(neigh) > 0:
                o = neigh[np.argsort(-row[neigh])]
            else:
                o = neigh
            for nb in o:
                nb = int(nb)
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
                    if len(visited) >= k:
                        break
            if len(visited) >= k:
                break
            if len(queue) == 0 and len(visited) < k:
                # add random outside to keep connectivity attempt
                cand = int(rng.integers(0, n))
                if cand not in visited:
                    visited.add(cand)
                    queue.append(cand)
        cluster = list(visited)
        mask_keep = np.ones(n, dtype=bool)
        mask_keep[cluster] = False
        X_keep = x_best[mask_keep].copy()
        # reinsert one at a time
        Y = X_keep.copy()
        ok = True
        for _ in range(len(cluster)):
            if time.time() >= T_SEARCH_END:
                ok = False
                break
            hole = _best_hole(Y, rng, n_cand=20000)
            hole = _refine_single(hole, Y, steps=300)
            Y = np.vstack([Y, hole[None, :]])
        if not ok:
            break
        # Y is now (55,21) but order differs; evaluate
        x_cand = _norm_rows(Y)
        v_raw = _max_cos(x_cand)
        if v_raw < best:
            # immediate improvement even before LP
            x_best, best = x_cand, v_raw
            continue
        # screen and LP only if promising
        if v_raw < 0.0675:
            now = time.time()
            if now >= T_SEARCH_END - 0.5:
                continue
            dl = min(now + 3.0, T_SEARCH_END)
            # only polish if we have at least ~0.6s
            if dl - now < 0.5:
                continue
            x_pol = _lp_polish_deadline(x_cand, dl, radius_init=8e-4)
            v_pol = _max_cos(x_pol)
            if v_pol < best:
                x_best, best = x_pol, v_pol
        # else discard cheaply
        if time.time() >= T_HARD - 16.0:
            # leave time for final polish; stop search a bit early if needed
            if time.time() >= T_SEARCH_END:
                break

    # reserve final 15 seconds for polishing best candidate
    now = time.time()
    if now < T_HARD - 0.5:
        dl = min(T_HARD - 0.3, now + 15.0)
        if dl > now + 0.5:
            try:
                x_best = _lp_polish_deadline(x_best, dl, radius_init=1e-3)
            except Exception:
                pass
    x_best = _norm_rows(x_best)
    return x_best
