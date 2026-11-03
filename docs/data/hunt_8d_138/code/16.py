import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _anneal_softmax(x, steps, beta0, beta1, lr0, lr1):
    n, d = x.shape
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = beta0 + (beta1 - beta0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-12
        Avg = (W @ x) / S
        dot = (Avg * x).sum(axis=1, keepdims=True)
        Tan = Avg - dot * x
        x = x - lr * Tan
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _contacts(x, thr=1e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    cnt = (G > m - thr).sum(axis=1)
    return G, m, cnt


def _best_void(existing, n_cand, rng):
    # existing (m,d) unit rows, find unit vector minimizing max dot
    d = existing.shape[1]
    cand = rng.normal(size=(n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ existing.T
    vals = dots.max(axis=1)
    return cand[np.argmin(vals)]


def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        c = Gfull[iu]
        m = c.max()
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        if np.count_nonzero(sel) > 3000:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:3000]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
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
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    rng = np.random.default_rng(0)
    np.random.seed(0)
    t0 = time.time()
    deadline = t0 + 170.0
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
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    # incumbent for kicks (start at record optimum)
    cur = best.copy()
    cur_val = best_val
    # coarse best (may be unpolished but guides search)
    coarse_best = best.copy()
    coarse_val = best_val

    it = 0
    sigmas = [0.02, 0.08, 0.25, 0.4, 0.12]
    while time.time() < deadline - 12.0 and it < 2000:
        it += 1
        mode = it % 6
        try:
            if mode == 0:
                # fresh random basin
                y = rng.normal(size=(n, d))
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            elif mode == 1:
                sigma = sigmas[(it // 6) % len(sigmas)]
                y = cur + sigma * rng.normal(size=(n, d))
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                if it % 12 == 0:
                    k = 8
                    idx = rng.choice(n, size=k, replace=False)
                    rp = rng.normal(size=(k, d))
                    rp /= np.linalg.norm(rp, axis=1, keepdims=True)
                    y[idx] = rp
            elif mode == 2:
                # kill worst k and greedy void reinsert
                G = cur @ cur.T
                np.fill_diagonal(G, -2.0)
                row_max = G.max(axis=1)
                k = 5 + (it % 4)  # 5..8
                worst = np.argsort(-row_max)[:k]
                keep = np.ones(n, dtype=bool)
                keep[worst] = False
                rem = cur[keep]
                y = cur.copy()
                # sequential greedy voids
                cur_rem = rem.copy()
                new_pts = []
                for _ in range(k):
                    nv = _best_void(cur_rem, 2500, rng)
                    new_pts.append(nv)
                    cur_rem = np.vstack([cur_rem, nv[None, :]])
                y[worst] = np.array(new_pts)
                y += 0.01 * rng.normal(size=(n, d))
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            elif mode == 3:
                # cap scramble
                u = rng.normal(size=(d,))
                u /= np.linalg.norm(u)
                proj = cur @ u
                nc = 18 + (it % 10)
                cap = np.argsort(-proj)[:nc]
                y = cur.copy()
                y[cap] += 0.35 * rng.normal(size=(nc, d))
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            elif mode == 4:
                # shake overloaded + rebirth rattlers
                _, _, cnt = _contacts(cur, thr=0.01)
                order = np.argsort(-cnt)
                pool = order[:12]
                k = 4
                chosen = pool[rng.choice(len(pool), size=k, replace=False)]
                y = cur.copy()
                y[chosen] += 0.30 * rng.normal(size=(k, d))
                # rattlers = zero contacts at tight tol
                G2 = y @ y.T
                np.fill_diagonal(G2, -2.0)
                m2 = G2.max()
                cnt2 = (G2 > m2 - 1e-7).sum(axis=1)
                ratt = np.where(cnt2 == 0)[0]
                if len(ratt) > 0:
                    nr = min(len(ratt), 4)
                    sel = rng.choice(ratt, size=nr, replace=False)
                    keepm = np.ones(n, dtype=bool)
                    keepm[sel] = False
                    rem2 = y[keepm]
                    cur_rem = rem2.copy()
                    for j, idxr in enumerate(sel):
                        nv = _best_void(cur_rem, 2000, rng)
                        y[idxr] = nv
                        cur_rem = np.vstack([cur_rem, nv[None, :]])
                else:
                    y += 0.01 * rng.normal(size=(n, d))
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            else:
                # large kick of best
                sigma = 0.35
                y = best + sigma * rng.normal(size=(n, d))
                y /= np.linalg.norm(y, axis=1, keepdims=True)

            # coarse + fine anneal (fast, no LP)
            y = _anneal_softmax(y, steps=550, beta0=12.0, beta1=160.0, lr0=0.04, lr1=0.006)
            y = _anneal_softmax(y, steps=250, beta0=200.0, beta1=500.0, lr0=0.006, lr1=0.001)
            v = _max_cos(y)
            if v < coarse_val:
                coarse_val = v
                coarse_best = y.copy()
            # basin-hopping acceptance for cur
            if v < cur_val or (it % 20 == 0 and v < cur_val + 0.005):
                cur = y.copy()
                cur_val = v
            # if clearly better than polished record, update polished best immediately via quick LP later
            # keep best (polished) fallback; coarse_best tracks unpolished
            if v < best_val - 2e-6:
                # promising: keep as new reference for kicks
                best = y.copy()
                best_val = v  # unpolished value, will be polished at end to go lower
                cur = y.copy()
                cur_val = v
        except Exception:
            continue

    # final selection: polish top candidates (original record + coarse best + cur)
    cands = []
    try:
        rb = np.load("best.npy").astype(float)
        rb /= np.linalg.norm(rb, axis=1, keepdims=True)
        cands.append(rb)
    except Exception:
        pass
    cands.append(coarse_best)
    cands.append(cur)
    # deduplicate by value order: polish best coarse first
    # sort by current max
    cands = sorted(cands, key=lambda z: _max_cos(z))[:3]
    final = cands[0].copy()
    final_val = _max_cos(final)
    remain = deadline - time.time()
    # give most time to best candidate, some to others
    for ci, cand in enumerate(cands):
        if time.time() >= deadline - 0.5:
            break
        budget = time.time() + max(2.0, remain / (len(cands) - ci + 1))
        if ci == 0:
            budget = deadline - 0.2
        try:
            p = _lp_polish(cand.copy(), min(deadline - 0.2, budget))
            v = _max_cos(p)
            if v < final_val:
                final_val = v
                final = p
        except Exception:
            continue
    final = final / np.linalg.norm(final, axis=1, keepdims=True)
    return final
