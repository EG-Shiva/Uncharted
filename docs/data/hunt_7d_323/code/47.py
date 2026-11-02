import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _lp_polish(X, seconds, init_radius=3e-3):
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    radius = float(init_radius)
    t0 = time.time()
    iuI, iuJ = np.triu_indices(n, 1)
    while True:
        if time.time() - t0 >= seconds or radius < 1e-14:
            break
        G = X @ X.T
        c = G[iuI, iuJ]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 2500:
            idx = np.where(sel)[0]
            cidx = c[idx]
            part = np.argpartition(cidx, -2500)[-2500:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
            cnt = 2500
        I = iuI[sel]; J = iuJ[sel]
        csel = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        Cb = cols.reshape(p, 2 * d + 1)
        Vb = vals.reshape(p, 2 * d + 1)
        ar = np.arange(d)
        Cb[:, :d] = I[:, None] * d + ar
        Cb[:, d:2 * d] = J[:, None] * d + ar
        Cb[:, 2 * d] = n * d
        Vb[:, :d] = X[J]
        Vb[:, d:2 * d] = X[I]
        Vb[:, 2 * d] = -1.0
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
        if time.time() - t0 >= seconds:
            break
    return X

def _smooth_relax(X, iters=120, step0=0.015, alpha=180.0, gap=0.08):
    n, d = X.shape
    step = step0
    for it in range(iters):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # weights focused near max
        W = np.exp(alpha * (G - m))
        W[G < m - gap] = 0.0
        np.fill_diagonal(W, 0.0)
        WS = (W * G).sum(axis=1)
        Grad = W @ X - WS[:, None] * X
        # normalize grad scale
        gn = np.linalg.norm(Grad, axis=1, keepdims=True) + 1e-12
        # clip large
        Grad = Grad / np.maximum(gn, 1.0)
        X = X - step * Grad
        X = _norm(X)
        step *= 0.995
    return X

def _build_adj(X, m, tol=1e-7):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    I, J = np.where(np.triu(G > m - tol, 1))
    adj = [[] for _ in range(n)]
    for a, b in zip(I, J):
        adj[a].append(int(b))
        adj[b].append(int(a))
    return adj, G

def _bfs_cluster(adj, G, start, size):
    n = len(adj)
    vis = set([int(start)])
    q = [int(start)]
    qi = 0
    while qi < len(q) and len(vis) < size:
        u = q[qi]; qi += 1
        # neighbours sorted by dot descending for determinism
        nbrs = adj[u]
        if len(nbrs) > 1:
            dots = [(G[u, v], v) for v in nbrs if v not in vis]
            dots.sort(reverse=True)
            ordered = [v for _, v in dots]
        else:
            ordered = [v for v in nbrs if v not in vis]
        for v in ordered:
            if len(vis) >= size:
                break
            if v not in vis:
                vis.add(v); q.append(v)
        if qi >= len(q) and len(vis) < size:
            # component exhausted: add nearest by cosine to start
            row = G[start].copy()
            for v in vis:
                row[v] = -2
            # also exclude visited
            order = np.argsort(-row)
            for v in order:
                if len(vis) >= size:
                    break
                v = int(v)
                if v not in vis:
                    vis.add(v); q.append(v)
            break
    # if still short, fill globally nearest
    if len(vis) < size:
        row = G[start].copy()
        order = np.argsort(-row)
        for v in order:
            if len(vis) >= size:
                break
            v = int(v)
            if v not in vis:
                vis.add(v)
    return sorted(list(vis))[:size]

def solve():
    t0 = time.time()
    DEADLINE = 165.0
    rng = np.random.default_rng(0)
    # load best
    bestX = None
    bestV = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = _norm(a)
                v = _max_cos(a)
                if v < bestV:
                    bestV = v
                    bestX = a.copy()
        except Exception:
            pass
    if bestX is None:
        a = rng.standard_normal(size=(323, 7))
        bestX = _norm(a)
        bestV = _max_cos(bestX)
    n, d = bestX.shape
    curX = bestX.copy()
    curV = bestV
    lp_count = 0
    max_lp = 7
    # precompute hub cycling
    cycle = 0
    # candidate count for reinsert
    NCAND = 15000
    while time.time() - t0 < DEADLINE - 8:
        cycle += 1
        # rebuild contact graph from current best (or cur)
        refX = bestX
        Gtmp = refX @ refX.T
        np.fill_diagonal(Gtmp, -2.0)
        m0 = float(Gtmp.max())
        adj, Gfull = _build_adj(refX, m0, tol=1e-7)
        counts = np.array([len(a) for a in adj])
        # select hub: cycle through top hubs + random
        order = np.argsort(-counts, kind="stable")
        if cycle % 3 == 0:
            hub = int(rng.integers(0, n))
            # bias to higher counts: pick from top 60 randomly sometimes
            if rng.random() < 0.5:
                hub = int(order[rng.integers(0, 60)])
        else:
            # cycle through top 8
            hub = int(order[(cycle - 1) % 8])
        csize = int(12 + (cycle % 7))  # 12..18
        if csize > 18:
            csize = 18
        cluster = _bfs_cluster(adj, Gfull, hub, csize)
        # surgery on copy of bestX
        y = bestX.copy()
        fixed_mask = np.ones(n, dtype=bool)
        fixed_mask[cluster] = False
        Xfixed = y[fixed_mask]  # (n-c, d)
        # sequential max-min reinsertion from random candidates
        new_pts = []
        # random order of insertion
        ins_order = cluster.copy()
        # shuffle deterministically via rng
        perm = rng.permutation(len(ins_order))
        ins_order = [ins_order[i] for i in perm]
        for k in ins_order:
            if time.time() - t0 > DEADLINE - 6:
                break
            cand = rng.standard_normal(size=(NCAND, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            if new_pts:
                Xa = np.vstack([Xfixed] + new_pts)
            else:
                Xa = Xfixed
            dots = cand @ Xa.T
            worst = dots.max(axis=1)
            bi = int(np.argmin(worst))
            bp = cand[bi]
            new_pts.append(bp[None, :])
            y[k] = bp
        # frozen-only hill climb on inserted points
        for sweep in range(2):
            for k in ins_order:
                if time.time() - t0 > DEADLINE - 6:
                    break
                others = np.delete(y, k, axis=0)
                cur = y[k].copy()
                cur_w = float((others @ cur).max())
                sig = 0.04 if sweep == 0 else 0.015
                trial = rng.standard_normal(size=(1200, d)) * sig + cur[None, :]
                trial /= np.linalg.norm(trial, axis=1, keepdims=True)
                dw = (trial @ others.T).max(axis=1)
                bi = int(np.argmin(dw))
                if dw[bi] < cur_w:
                    y[k] = trial[bi]
        v_pre = _max_cos(y)
        # short all-point smooth relaxation
        # fewer iters to stay fast
        y = _smooth_relax(y, iters=90, step0=0.012, alpha=200.0, gap=0.07)
        v_rel = _max_cos(y)
        if v_rel < bestV - 1e-12:
            bestV = v_rel
            bestX = y.copy()
        # selective LP polish
        if v_rel < 0.670 and v_rel < bestV + 0.002 and lp_count < max_lp:
            if time.time() - t0 > DEADLINE - 7:
                break
            # budget 3s per polish, less near deadline
            remain = DEADLINE - (time.time() - t0)
            budget = min(3.5, max(1.0, remain - 4))
            yp = _lp_polish(y, seconds=budget, init_radius=2e-3)
            vp = _max_cos(yp)
            lp_count += 1
            if vp < bestV:
                bestV = vp
                bestX = yp.copy()
        # also track cur for diversity: occasionally keep slightly worse to seed next? we always base on bestX so fine
        if time.time() - t0 > DEADLINE - 6:
            break
    # final quick polish of best if time allows
    if time.time() - t0 < DEADLINE - 3:
        remain = DEADLINE - (time.time() - t0)
        try:
            bp = _lp_polish(bestX, seconds=min(4.0, max(0.5, remain - 1)), init_radius=1e-3)
            vp = _max_cos(bp)
            if vp < bestV:
                bestV = vp
                bestX = bp
        except Exception:
            pass
    bestX = _norm(bestX)
    return bestX
