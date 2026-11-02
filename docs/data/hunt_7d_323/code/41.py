import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x, iu=None):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, deadline, init_radius=8e-3, iu=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(init_radius)
    start = time.time()
    if iu is None:
        iu = np.triu_indices(n, 1)
    iuI, iuJ = iu
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        if time.time() - start >= seconds or time.time() >= deadline:
            break
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 3000:
            idx = np.where(sel)[0]
            cidx = c[idx]
            part = np.argpartition(cidx, -3000)[-3000:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
            cnt = 3000
        I = iuI[sel]
        J = iuJ[sel]
        csel = c[sel]
        p = cnt
        nv = n * d + 1
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        Cb = cols.reshape(p, 2 * d + 1)
        Vb = vals.reshape(p, 2 * d + 1)
        ar = np.arange(d)
        Cb[:, :d] = I[:, None] * d + ar
        Cb[:, d:2 * d] = J[:, None] * d + ar
        Cb[:, 2 * d] = n * d
        Vb[:, :d] = x[J]
        Vb[:, d:2 * d] = x[I]
        Vb[:, 2 * d] = -1.0
        rows = np.repeat(np.arange(p), 2 * d + 1)
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            x = y
            best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _relax_all(x, steps=120, step0=0.15, window=0.04):
    n, d = x.shape
    y = x.copy()
    for it in range(steps):
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        thr = m - window
        W = G - thr
        W[W < 0] = 0.0
        np.fill_diagonal(W, 0.0)
        Wx = W @ y
        s = (W * G).sum(axis=1)
        force = Wx - s[:, None] * y
        st = step0 * (1.0 - 0.5 * it / max(1, steps))
        y = y - st * force
        y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y

def solve():
    t0 = time.time()
    HARD = t0 + 165.0
    rng = np.random.default_rng(12345)
    n, d = 323, 7
    x0 = None
    b0 = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                g = a @ a.T
                np.fill_diagonal(g, -2.0)
                v = float(g.max())
                if v < b0:
                    b0 = v
                    x0 = a.copy()
        except Exception:
            pass
    if x0 is None:
        x0 = rng.standard_normal(size=(n, d))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        b0 = _max_cos(x0)
    x_best = x0.copy()
    best = float(b0)
    iu = np.triu_indices(n, 1)
    iuI, iuJ = iu

    def contact_order(xc, m):
        G = xc @ xc.T
        np.fill_diagonal(G, -2.0)
        cnt = np.sum(G > m - 1e-7, axis=1)
        order = np.argsort(-cnt)
        return cnt, order, G

    G0 = x_best @ x_best.T
    np.fill_diagonal(G0, -2.0)
    m0 = float(G0.max())
    _, hub_order, _ = contact_order(x_best, m0)

    attempt = 0
    # pre-allocate candidate buffer reuse via regeneration each attempt
    while True:
        now = time.time()
        # reserve 8s for final polish
        if now > HARD - 8.0:
            break
        remain = HARD - 8.0 - now
        if remain < 2.0:
            break
        # pick hub: cycle hubs, every 3rd pick random start for diversity
        if attempt % 3 == 2:
            start_node = int(rng.integers(0, n))
        else:
            start_node = int(hub_order[attempt % n])
        # current max for graph threshold
        Gc = x_best @ x_best.T
        np.fill_diagonal(Gc, -2.0)
        mc = float(Gc.max())
        # adjacency with tol 1e-7
        adj = [np.where(Gc[i] > mc - 1e-7)[0].tolist() for i in range(n)]
        # BFS neighbourhood 12-18
        target = 12 + (attempt % 7)
        seen = set([start_node])
        queue = [start_node]
        qi = 0
        # expand BFS; if disconnected component smaller, add random connectors
        while len(seen) < target and qi < len(queue):
            u = queue[qi]
            qi += 1
            nbrs = adj[u]
            # shuffle neighbours deterministically via rng for variety
            if len(nbrs) > 1:
                perm = rng.permutation(len(nbrs))
                nbrs = [nbrs[i] for i in perm]
            for v in nbrs:
                if v not in seen:
                    seen.add(v)
                    queue.append(v)
                    if len(seen) >= target:
                        break
        # if still small (isolated rattler), add random nodes / worst nodes
        while len(seen) < target:
            # add a random neighbour of current set or random node
            cand_node = int(rng.integers(0, n))
            if cand_node not in seen:
                seen.add(cand_node)
        cluster = sorted(list(seen))[:target]
        k = len(cluster)
        mask = np.ones(n, dtype=bool)
        mask[cluster] = False
        fixed = x_best[mask]
        # 50k candidates
        Nc = 50000
        cand = rng.standard_normal(size=(Nc, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        aug = fixed.copy()
        new_pts = []
        # sequential max-min
        for s in range(k):
            if time.time() > HARD - 8.0:
                break
            best_val = np.inf
            best_pt = None
            # chunk to save memory
            for s0 in range(0, Nc, 10000):
                chunk = cand[s0:s0 + 10000]
                dots = chunk @ aug.T
                worst = dots.max(axis=1)
                li = int(np.argmin(worst))
                if worst[li] < best_val:
                    best_val = float(worst[li])
                    best_pt = chunk[li].copy()
            new_pts.append(best_pt)
            aug = np.vstack([aug, best_pt[None, :]])
        if len(new_pts) != k:
            attempt += 1
            continue
        y = x_best.copy()
        for ii, cc in enumerate(cluster):
            y[cc] = new_pts[ii]
        # frozen optimisation: only inserted points, core frozen
        # 2 rounds, 1500 trials each with sigma 0.03
        for rnd in range(2):
            order_c = rng.permutation(k)
            for oi in order_c:
                cc = cluster[int(oi)]
                others_mask = np.ones(n, dtype=bool)
                others_mask[cc] = False
                others = y[others_mask]
                cur = y[cc].copy()
                dots0 = others @ cur
                cur_w = float(dots0.max())
                sig = 0.035 if rnd == 0 else 0.015
                Nt = 1500
                trial = rng.standard_normal(size=(Nt, d)) * sig + cur[None, :]
                trial /= np.linalg.norm(trial, axis=1, keepdims=True)
                dw = (trial @ others.T).max(axis=1)
                bi = int(np.argmin(dw))
                if dw[bi] < cur_w:
                    y[cc] = trial[bi]
            if time.time() > HARD - 8.0:
                break
        # short all-point smooth relaxation
        try:
            y = _relax_all(y, steps=100, step0=0.15, window=0.04)
        except Exception:
            pass
        g = y @ y.T
        np.fill_diagonal(g, -2.0)
        v_pre = float(g.max())
        # keep candidates whose pre-LP max below 0.670
        if v_pre < 0.670:
            budget = min(4.0, (HARD - 8.0) - (time.time()))
            if budget >= 1.0:
                try:
                    y2 = _lp_polish(y, seconds=budget, deadline=HARD - 8.0, init_radius=8e-3, iu=iu)
                except Exception:
                    y2 = y
                g2 = y2 @ y2.T
                np.fill_diagonal(g2, -2.0)
                v_post = float(g2.max())
                if v_post < best - 1e-12:
                    best = v_post
                    x_best = y2 / np.linalg.norm(y2, axis=1, keepdims=True)
                    # refresh hub order
                    _, hub_order, _ = contact_order(x_best, best)
        attempt += 1
        # occasional hub refresh even without improvement to avoid stale graph
        if attempt % 10 == 0:
            _, hub_order, _ = contact_order(x_best, best)

    # final squeeze
    rem = HARD - (time.time())
    if rem > 2.0:
        try:
            x_best = _lp_polish(x_best, seconds=min(rem - 0.5, 7.0), deadline=HARD, init_radius=2e-3, iu=iu)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
