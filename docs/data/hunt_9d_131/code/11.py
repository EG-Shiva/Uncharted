import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def rowmaxs(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1), g

def lp_polish(x, seconds, r0=1e-3):
    if seconds <= 0:
        return x
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = r0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start >= seconds:
            break
        try:
            Gfull = x @ x.T
            c = Gfull[iu0]
        except Exception:
            break
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) == 0:
            break
        I, J = iu0[0][sel], iu0[1][sel]
        p = len(I)
        nv = n * d + 1
        try:
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
            obj = np.zeros(nv); obj[-1] = 1.0
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
        try:
            y = x + res.x[:-1].reshape(n, d)
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            if not np.all(np.isfinite(y)):
                radius *= 0.5
                continue
            v = max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def smooth_descent(x, steps=100, beta0=20.0, beta1=80.0, lr0=0.002, mask=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    if mask is None:
        free = np.ones(n, dtype=bool)
    else:
        free = np.asarray(mask, dtype=bool)
        if not np.any(free):
            return x
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = beta0 + (beta1 - beta0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        if not np.all(np.isfinite(grad)):
            break
        nrm = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-12
        step = lr0 * (1.0 - 0.5 * frac)
        upd = np.zeros_like(grad)
        upd[free] = (grad[free] / nrm[free]) * step
        x = x - upd
        n2 = np.linalg.norm(x, axis=1, keepdims=True)
        n2[n2 == 0] = 1.0
        x = x / n2
    return x

def best_hole(Xo, rng, n_sample=25000, refine=120, beta0=30.0, beta1=200.0):
    # Xo: (m,d) fixed points, find y on sphere minimizing max_j Xo_j . y
    m, d = Xo.shape
    S = rng.standard_normal((n_sample, d))
    S /= np.linalg.norm(S, axis=1, keepdims=True) + 1e-30
    scores = (S @ Xo.T).max(axis=1)
    k = min(4, n_sample)
    idx = np.argpartition(scores, k - 1)[:k]
    best_y = S[idx[0]]
    best_v = float(scores[idx[0]])
    for t in idx:
        y = S[t].copy()
        lr = 0.02
        for s in range(refine):
            frac = s / max(1, refine - 1)
            beta = beta0 + (beta1 - beta0) * frac
            g = Xo @ y
            mx = float(g.max())
            w = np.exp(beta * (g - mx))
            sw = w.sum() + 1e-30
            w /= sw
            grad = w @ Xo
            grad = grad - np.dot(grad, y) * y
            nrm = np.linalg.norm(grad) + 1e-12
            y = y - lr * grad / nrm
            y /= np.linalg.norm(y) + 1e-30
            lr *= 0.995
        v = float((Xo @ y).max())
        if v < best_v:
            best_v = v
            best_y = y
    return best_y, best_v

def contacts_info(x, tol=1e-9):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iu = np.triu_indices(n, 1)
    c = G[iu]
    sel = c > m - tol
    I = iu[0][sel]; J = iu[1][sel]
    rm = G.max(axis=1)
    ratt = np.where(rm < m - tol)[0]
    # adjacency for BFS
    adj = [[] for _ in range(n)]
    for a, b in zip(I.tolist(), J.tolist()):
        adj[a].append(b); adj[b].append(a)
    return m, I, J, ratt, rm, adj

def bfs_block(adj, seed, size, rng):
    n = len(adj)
    vis = {int(seed)}
    frontier = [int(seed)]
    while frontier and len(vis) < size:
        u = frontier.pop(rng.integers(0, len(frontier)) if len(frontier) > 1 else 0)
        nbrs = adj[u]
        if len(nbrs) == 0:
            # jump to random contact node
            v = int(rng.integers(0, n))
            if v not in vis:
                vis.add(v); frontier.append(v)
            continue
        rng.shuffle(nbrs)
        for v in nbrs:
            if v not in vis and len(vis) < size:
                vis.add(v); frontier.append(v)
            if len(vis) >= size:
                break
    while len(vis) < size:
        v = int(rng.integers(0, n))
        if v not in vis:
            vis.add(v)
    return np.array(sorted(vis), dtype=int)

def solve():
    t0 = time.time()
    HOP_END = t0 + 148.0
    HARD_END = t0 + 174.0
    rng = np.random.default_rng(7)
    n, d = 131, 9
    xb = None
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=float)
            if v.shape == (n, d) and np.all(np.isfinite(v)):
                xb = v; break
        except Exception:
            continue
    if xb is None:
        xb = rng.standard_normal((n, d))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    v_best = max_cos(xb)
    x_cur = xb.copy(); v_cur = v_best
    try:
        _, cI, cJ, rattlers, rm_cur, adj_cur = contacts_info(x_cur, tol=2e-7)
    except Exception:
        cI = np.array([], dtype=int); cJ = np.array([], dtype=int)
        rattlers = np.array([], dtype=int); rm_cur = np.full(n, v_cur); adj_cur = [[] for _ in range(n)]

    trial = 0
    while time.time() < HOP_END:
        if HOP_END - time.time() < 1.0:
            break
        mod = trial % 8
        y = None
        use_smooth = False
        smooth_kw = dict(steps=100, beta0=20.0, beta1=80.0, lr0=0.0018)
        mask = None
        try:
            if mod == 0:
                # worst kill-and-fill (single optimal reinsertion)
                order = np.argsort(-rm_cur)
                cand_top = order[:12]
                vi = int(cand_top[rng.integers(0, len(cand_top))])
                Xo = np.delete(x_cur, vi, axis=0)
                yopt, _ = best_hole(Xo, rng, n_sample=22000, refine=100)
                y = x_cur.copy(); y[vi] = yopt
                # tiny jitter others to break symmetry
                y = y + 0.0015 * rng.standard_normal((n, d))
                y = y / np.linalg.norm(y, axis=1, keepdims=True)
            elif mod == 1:
                # rattler hole centering (move up to 3 rattlers optimally)
                y = x_cur.copy()
                if len(rattlers) > 0:
                    k = min(len(rattlers), 3)
                    picks = rng.choice(rattlers, size=k, replace=False)
                    for vi in picks:
                        Xo = np.delete(y, int(vi), axis=0)
                        yopt, _ = best_hole(Xo, rng, n_sample=15000, refine=80)
                        y[int(vi)] = yopt
                else:
                    # no rattlers: pick low-contact points
                    order = np.argsort(rm_cur)
                    picks = order[:3]
                    for vi in picks:
                        Xo = np.delete(y, int(vi), axis=0)
                        yopt, _ = best_hole(Xo, rng, n_sample=15000, refine=60)
                        y[int(vi)] = yopt
                y = y / np.linalg.norm(y, axis=1, keepdims=True)
            elif mod == 2:
                # contact-cluster block shake + masked smooth
                if len(cI) > 0:
                    seed = int(rng.integers(0, n))
                    # prefer high-degree seed
                    if rng.random() < 0.7:
                        degs = np.array([len(a) for a in adj_cur])
                        seed = int(np.argmax(degs + rng.random(n) * 2.0))
                    blk = bfs_block(adj_cur, seed, 14, rng)
                    y = x_cur.copy()
                    y[blk] = y[blk] + 0.05 * rng.standard_normal((len(blk), d))
                    y = y / np.linalg.norm(y, axis=1, keepdims=True)
                    mask = np.zeros(n, dtype=bool); mask[blk] = True
                    use_smooth = True
                    smooth_kw = dict(steps=140, beta0=15.0, beta1=100.0, lr0=0.003)
                else:
                    y = x_cur + 0.01 * rng.standard_normal((n, d))
                    y = y / np.linalg.norm(y, axis=1, keepdims=True)
                    use_smooth = True
            elif mod == 3:
                # contact edge breaking kick
                y = x_cur.copy() + 0.003 * rng.standard_normal((n, d))
                if len(cI) > 0:
                    e = int(rng.integers(0, len(cI)))
                    i, j = int(cI[e]), int(cJ[e])
                    diff = y[i] - y[j]
                    nd = np.linalg.norm(diff) + 1e-12
                    diff /= nd
                    push = 0.04 + 0.03 * rng.random()
                    y[i] += push * diff; y[j] -= push * diff
                y = y / np.linalg.norm(y, axis=1, keepdims=True)
                use_smooth = True
                smooth_kw = dict(steps=130, beta0=18.0, beta1=95.0, lr0=0.0022)
            elif mod == 4:
                # double kill: two worst -> two holes sequentially
                order = np.argsort(-rm_cur)
                v1 = int(order[rng.integers(0, 8)])
                v2 = int(order[rng.integers(0, 8)])
                if v2 == v1:
                    v2 = int(order[(np.where(order == v1)[0][0] + 1) % n])
                y = x_cur.copy()
                Xo1 = np.delete(y, v1, axis=0)
                yopt1, _ = best_hole(Xo1, rng, n_sample=18000, refine=80)
                y[v1] = yopt1
                Xo2 = np.delete(y, v2, axis=0)
                yopt2, _ = best_hole(Xo2, rng, n_sample=18000, refine=80)
                y[v2] = yopt2
                y = y / np.linalg.norm(y, axis=1, keepdims=True)
            elif mod == 5:
                # subset shake broad
                y = x_cur.copy() + 0.0025 * rng.standard_normal((n, d))
                k = 10
                idx = rng.choice(n, size=k, replace=False)
                y[idx] += 0.07 * rng.standard_normal((k, d))
                y = y / np.linalg.norm(y, axis=1, keepdims=True)
                use_smooth = True
                smooth_kw = dict(steps=110, beta0=20.0, beta1=85.0, lr0=0.002)
            elif mod == 6:
                # global medium shake
                y = x_cur.copy() + 0.018 * rng.standard_normal((n, d))
                y = y / np.linalg.norm(y, axis=1, keepdims=True)
                use_smooth = True
                smooth_kw = dict(steps=90, beta0=22.0, beta1=75.0, lr0=0.0016)
            else:
                # large shake escape
                y = x_cur.copy() + 0.05 * rng.standard_normal((n, d))
                y = y / np.linalg.norm(y, axis=1, keepdims=True)
                use_smooth = True
                smooth_kw = dict(steps=120, beta0=15.0, beta1=90.0, lr0=0.0022)
        except Exception:
            trial += 1
            continue
        if y is None:
            trial += 1
            continue
        try:
            if use_smooth:
                if mask is not None:
                    y = smooth_descent(y, mask=mask, **smooth_kw)
                else:
                    y = smooth_descent(y, **smooth_kw)
        except Exception:
            pass
        budget = min(1.4, HOP_END - time.time() - 0.05)
        if budget < 0.25:
            trial += 1
            continue
        try:
            y = lp_polish(y, seconds=budget)
            v = max_cos(y)
        except Exception:
            trial += 1
            continue
        if not np.all(np.isfinite(y)):
            trial += 1
            continue
        # acceptance: walk + best tracking
        if v < v_best - 1e-12:
            xb = y.copy(); v_best = v
            x_cur = y.copy(); v_cur = v
            try:
                _, cI, cJ, rattlers, rm_cur, adj_cur = contacts_info(x_cur, tol=2e-7)
            except Exception:
                pass
            if time.time() < HOP_END - 3.0:
                try:
                    y2 = lp_polish(xb, seconds=min(2.5, HOP_END - time.time()))
                    v2 = max_cos(y2)
                    if v2 < v_best:
                        xb = y2; v_best = v2; x_cur = y2.copy(); v_cur = v2
                        _, cI, cJ, rattlers, rm_cur, adj_cur = contacts_info(x_cur, tol=2e-7)
                except Exception:
                    pass
        elif v < v_cur + 3e-4:
            # neutral / slightly worse walk move (keeps topology changing)
            # keep best unchanged, move cursor
            x_cur = y.copy(); v_cur = v
            try:
                _, cI, cJ, rattlers, rm_cur, adj_cur = contacts_info(x_cur, tol=2e-7)
            except Exception:
                pass
            # if cursor drifted too far, pull back to best occasionally
            if v_cur > v_best + 2e-3 and (trial % 5 == 0):
                x_cur = xb.copy(); v_cur = v_best
                try:
                    _, cI, cJ, rattlers, rm_cur, adj_cur = contacts_info(x_cur, tol=2e-7)
                except Exception:
                    pass
        else:
            # reject; occasionally restart cursor at best
            if trial % 11 == 10:
                x_cur = xb.copy(); v_cur = v_best
                try:
                    _, cI, cJ, rattlers, rm_cur, adj_cur = contacts_info(x_cur, tol=2e-7)
                except Exception:
                    pass
        trial += 1
    try:
        remain = HARD_END - time.time() - 0.5
        if remain > 1.0:
            xb = lp_polish(xb, seconds=remain)
    except Exception:
        pass
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    return xb
