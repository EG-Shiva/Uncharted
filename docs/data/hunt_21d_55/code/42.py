import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _refine_point(p0, Y, steps=300, beta0=25.0, beta1=500.0, lr0=0.03, lr1=0.003):
    p = p0.astype(float).copy()
    nn = np.linalg.norm(p)
    if nn == 0 or not np.isfinite(nn):
        return p0
    p = p / nn
    d = p.shape[0]
    m = np.zeros(d)
    v = np.zeros(d)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lb = np.log(beta1 / beta0) / steps if steps > 0 else 0.0
    ll = np.log(lr1 / lr0) / steps if steps > 0 else 0.0
    for t in range(1, steps + 1):
        beta = beta0 * np.exp(lb * t)
        lr = lr0 * np.exp(ll * t)
        s = Y @ p
        Mx = np.max(s)
        E = np.exp(beta * (s - Mx))
        S = np.sum(E)
        if not np.isfinite(S) or S <= 0:
            break
        grad = (E @ Y) / S
        gt = grad - np.dot(grad, p) * p
        m = b1 * m + (1.0 - b1) * gt
        vv = b2 * v + (1.0 - b2) * (gt * gt)
        # bias correction
        mhat = m / (1.0 - b1 ** t)
        vhat = vv / (1.0 - b2 ** t)
        p = p - lr * mhat / (np.sqrt(vhat) + eps)
        n2 = np.linalg.norm(p)
        if not np.isfinite(n2) or n2 == 0:
            return p0
        p = p / n2
    return p

def _lp_polish_budget(x, budget, hard_end):
    t_s = time.time()
    x = _norm_rows(x.copy())
    n, d = x.shape
    best = _maxcos(x)
    xb = x.copy()
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() - t_s < budget and time.time() < hard_end and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        mxx = float(np.max(c))
        thr = mxx - max(20.0 * radius, 1e-9)
        sel = c > thr
        cntp = int(np.sum(sel))
        if cntp == 0:
            radius *= 0.5
            continue
        if cntp > 2500:
            # keep top 2500 pairs
            kth = np.partition(c, -2500)[-2500]
            sel = c >= kth
            cntp = int(np.sum(sel))
            if cntp > 2600:
                idx = np.argpartition(-c, 2500)[:2500]
                mask = np.zeros_like(c, dtype=bool)
                mask[idx] = True
                sel = mask
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if getattr(res, "status", 1) != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y = _norm_rows(y)
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        vv = _maxcos(y)
        if vv < best:
            x, best = y, vv
            xb = y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        # stop if budget nearly exhausted
        if time.time() - t_s >= budget:
            break
    return xb, best

def solve():
    t0 = time.time()
    HARD_END = t0 + 165.0
    SEARCH_END = t0 + 150.0  # reserve final 15 s
    rng = np.random.RandomState(0)
    n, d = 55, 21
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=float)
            if a.shape == (n, d):
                a = _norm_rows(a)
                v = _maxcos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm_rows(rng.randn(n, d))
        best_val = _maxcos(best)
    # main cheap surgery loop
    trial = 0
    try:
        while time.time() < SEARCH_END:
            trial += 1
            # need at least ~1s for a candidate + possible LP
            if time.time() > SEARCH_END - 0.5:
                break
            # contact graph with threshold best-1e-6
            try:
                G = best @ best.T
                np.fill_diagonal(G, -10.0)
                thr = best_val - 1e-6
                adj = G > thr
                cnt = np.sum(adj, axis=1)
                hub = int(np.argmax(cnt))
            except Exception:
                hub = int(rng.randint(0, n))
                cnt = np.zeros(n)
            k = 6 + int(rng.randint(0, 4))  # 6-9
            # BFS cluster from hub
            visited = set([hub])
            queue = [hub]
            # precompute neighbour lists
            try:
                while len(visited) < k and len(queue) > 0:
                    u = queue.pop(0)
                    neigh = np.where(adj[u])[0]
                    if len(neigh) > 1:
                        perm = rng.permutation(len(neigh))
                        neigh = neigh[perm]
                    for vv in neigh:
                        vv = int(vv)
                        if vv not in visited:
                            visited.add(vv)
                            queue.append(vv)
                            if len(visited) >= k:
                                break
                # if disconnected, fill with highest-contact nodes
                if len(visited) < k:
                    order = np.argsort(-cnt)
                    for vv in order:
                        vv = int(vv)
                        if vv not in visited:
                            visited.add(vv)
                            if len(visited) >= k:
                                break
            except Exception:
                while len(visited) < k:
                    visited.add(int(rng.randint(0, n)))
                    if len(visited) >= k:
                        break
            del_idx = sorted(list(visited))[:k]
            mask = np.ones(n, dtype=bool)
            mask[del_idx] = False
            try:
                retained = best[mask]
            except Exception:
                continue
            curY = retained.copy()
            ok = True
            for _j in range(k):
                if time.time() >= SEARCH_END:
                    ok = False
                    break
                # sample 20000 random sphere points
                try:
                    C = rng.randn(20000, d)
                    C = _norm_rows(C)
                    D = C @ curY.T
                    md = np.max(D, axis=1)
                    bi = int(np.argmin(md))
                    p0 = C[bi]
                except Exception:
                    p0 = _norm_rows(rng.randn(1, d))[0]
                try:
                    p = _refine_point(p0, curY, steps=300)
                except Exception:
                    p = p0
                # append
                try:
                    curY = np.vstack([curY, p[None, :]])
                except Exception:
                    ok = False
                    break
            if not ok:
                continue
            if curY.shape != (n, d):
                continue
            curY = _norm_rows(curY)
            if not np.all(np.isfinite(curY)):
                continue
            raw = _maxcos(curY)
            # screen raw
            if raw < best_val - 1e-12:
                # direct improvement even before LP
                best_val = raw
                best = curY.copy()
                thr_update = True
            else:
                thr_update = False
            # LP only for candidates below 0.0675, at most 3 s each
            if raw < 0.0675:
                remain_search = SEARCH_END - time.time()
                remain_hard = HARD_END - 15.0 - time.time()  # keep reserve? during search keep reserve
                # allow LP only if we still have time and keep final reserve
                if remain_search < 0.5:
                    continue
                bud = min(3.0, remain_search)
                # also don't exceed hard end reserve
                if bud <= 0.2:
                    continue
                try:
                    xp, vp = _lp_polish_budget(curY, bud, SEARCH_END)
                except Exception:
                    continue
                if vp < best_val - 1e-12:
                    best_val = vp
                    best = xp.copy()
                # check wall clock in every loop (already at top)
            # safety: never lose incumbent; continue
            if trial > 5000:
                break
    except Exception:
        pass
    # final 15 s polishing of best candidate
    try:
        now = time.time()
        rem = HARD_END - now
        if rem > 0.5 and best is not None:
            # use at most rem (<=15s) for LP polish
            bud = min(rem - 0.2, 14.0)
            if bud > 0.5:
                try:
                    xp, vp = _lp_polish_budget(best, bud, HARD_END - 0.2)
                    if vp < best_val:
                        best_val = vp
                        best = xp.copy()
                except Exception:
                    pass
    except Exception:
        pass
    try:
        best = _norm_rows(np.asarray(best, dtype=float))
    except Exception:
        best = _norm_rows(rng.randn(n, d))
    # strict finiteness
    bad = ~np.isfinite(best)
    if np.any(bad):
        best[bad] = 0.01
        best = _norm_rows(best)
    # clip to time: ensure return
    return best
