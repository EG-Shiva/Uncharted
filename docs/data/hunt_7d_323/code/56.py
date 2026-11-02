import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, deadline, init_radius=6e-3, cap=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(init_radius)
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            cidx = c[idx]
            part = np.argpartition(cidx, -cap)[-cap:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
            cnt = cap
        I = iuI[sel]
        J = iuJ[sel]
        csel = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        Cm = cols.reshape(p, 2 * d + 1)
        Vm = vals.reshape(p, 2 * d + 1)
        ar = np.arange(d)
        Cm[:, :d] = I[:, None] * d + ar
        Cm[:, d:2 * d] = J[:, None] * d + ar
        Cm[:, 2 * d] = n * d
        Vm[:, :d] = x[J]
        Vm[:, d:2 * d] = x[I]
        Vm[:, 2 * d] = -1.0
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
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
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best:
            x = y
            best = v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
        if time.time() >= deadline:
            break
    return x

def _relax(x, move_idx, steps, tau0, tau1, lr0, lr1, mu, deadline, iuI, iuJ):
    n, d = x.shape
    if move_idx is None:
        frozen = None
        move_mask = None
    else:
        is_move = np.zeros(n, dtype=bool)
        is_move[np.asarray(move_idx, dtype=int)] = True
        frozen = ~is_move
    vel = np.zeros_like(x)
    best = _max_cos(x)
    best_x = x.copy()
    for s in range(steps):
        if s % 10 == 0 and time.time() >= deadline:
            break
        frac = s / max(1, steps - 1)
        tau = tau0 + (tau1 - tau0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        e = np.exp((c - m) / tau)
        S = float(e.sum())
        if S == 0 or not np.isfinite(S):
            break
        w = e / S
        Wx = np.zeros_like(x)
        np.add.at(Wx, iuI, w[:, None] * x[iuJ])
        np.add.at(Wx, iuJ, w[:, None] * x[iuI])
        dot = np.sum(Wx * x, axis=1, keepdims=True)
        Gt = Wx - dot * x
        if frozen is not None:
            Gt[frozen] = 0.0
        vel = mu * vel + Gt
        if frozen is not None:
            vel[frozen] = 0.0
            x_move = x[is_move] - lr * vel[is_move]
            xn = np.linalg.norm(x_move, axis=1, keepdims=True)
            xn[xn == 0] = 1.0
            x[is_move] = x_move / xn
        else:
            x = x - lr * vel
            xn = np.linalg.norm(x, axis=1, keepdims=True)
            xn[xn == 0] = 1.0
            x = x / xn
        if (s % 50 == 49) or (s == steps - 1):
            v = _max_cos(x)
            if v < best:
                best = v
                best_x = x.copy()
    return best_x

def solve():
    t0 = time.time()
    HARD = t0 + 165.0
    EXPLORE_END = t0 + 138.0
    rng = np.random.default_rng(20260707)
    n, d = 323, 7
    iuI, iuJ = np.triu_indices(n, 1)
    x0 = None
    b0 = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < b0:
                        b0 = v
                        x0 = a.copy()
        except Exception:
            pass
    if x0 is None:
        a = rng.standard_normal(size=(323, 7))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        x0 = a
        b0 = _max_cos(a)
    x_best = x0.copy()
    best = float(b0)
    it = 0
    while time.time() < EXPLORE_END and time.time() < HARD - 10:
        if time.time() >= HARD - 10:
            break
        x = x_best.copy()
        # contact graph at max-1e-7
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        cfull = G[iuI, iuJ]
        mcur = float(cfull.max())
        thr = mcur - 1e-7
        selc = cfull > thr
        Ic = iuI[selc]
        Jc = iuJ[selc]
        adj = [[] for _ in range(n)]
        for a_, b_ in zip(Ic.tolist(), Jc.tolist()):
            adj[a_].append(b_)
            adj[b_].append(a_)
        counts = np.zeros(n, dtype=int)
        for i in range(n):
            counts[i] = len(adj[i])
        # choose hub: cycle top hubs and random seeds
        if it % 3 == 0 or it % 3 == 1:
            order = np.argsort(-counts, kind="stable")
            top = order[:24]
            hub = int(top[it % len(top)])
        else:
            hub = int(rng.integers(0, n))
        ksize = int(rng.integers(12, 19))  # 12..18
        # BFS neighbourhood
        visited = set([hub])
        queue = [hub]
        # shuffle adjacency for variety
        qpos = 0
        while len(visited) < ksize and qpos < len(queue):
            u = queue[qpos]
            qpos += 1
            neigh = adj[u][:]
            if len(neigh) > 1:
                perm = rng.permutation(len(neigh))
                neigh = [neigh[i] for i in perm]
            for v in neigh:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
                    if len(visited) >= ksize:
                        break
        # if still short, add nearest neighbours by cosine
        if len(visited) < ksize:
            # worst dots to hub
            dots_h = x @ x[hub]
            dots_h[list(visited)] = -2.0
            dots_h[hub] = -2.0
            need = ksize - len(visited)
            extra = np.argpartition(dots_h, -need)[-need:]
            for v in extra.tolist():
                visited.add(int(v))
        cluster = sorted([int(v) for v in visited])[:ksize]
        keep = np.ones(n, dtype=bool)
        keep[np.array(cluster, dtype=int)] = False
        core = x[keep]  # (n-k, d)
        # sequential max-min reinsertion from 50000 candidates
        NC = 50000
        try:
            C = rng.standard_normal(size=(NC, d))
            C /= np.linalg.norm(C, axis=1, keepdims=True)
        except Exception:
            it += 1
            continue
        if time.time() >= HARD - 10:
            break
        # initial worst to core
        D = C @ core.T
        worst = D.max(axis=1)
        y = x.copy()
        # order cluster insertion: start from hub first for stability
        # move hub to front
        if hub in cluster:
            cluster = [hub] + [v for v in cluster if v != hub]
        else:
            permc = rng.permutation(len(cluster))
            cluster = [cluster[i] for i in permc]
        for kk, idx in enumerate(cluster):
            if time.time() >= HARD - 9:
                break
            bi = int(np.argmin(worst))
            y[idx] = C[bi]
            # update worst with new point
            dn = C @ y[idx]
            # in-place maximum
            np.maximum(worst, dn, out=worst)
        # frozen optimisation of inserted points
        dl1 = min(time.time() + 1.2, HARD - 8)
        try:
            y = _relax(y, cluster, steps=220, tau0=0.012, tau1=0.0025, lr0=3.0, lr1=0.5, mu=0.8, deadline=dl1, iuI=iuI, iuJ=iuJ)
        except Exception:
            pass
        if time.time() >= HARD - 8:
            # evaluate anyway
            try:
                vpre = _max_cos(y)
                if vpre < best - 1e-12 and np.all(np.isfinite(y)):
                    best = vpre
                    x_best = y.copy()
            except Exception:
                pass
            it += 1
            continue
        # short all-point smooth relaxation
        dl2 = min(time.time() + 1.0, HARD - 7)
        try:
            y = _relax(y, None, steps=180, tau0=0.010, tau1=0.002, lr0=2.5, lr1=0.4, mu=0.8, deadline=dl2, iuI=iuI, iuJ=iuJ)
        except Exception:
            pass
        try:
            vpre = _max_cos(y)
        except Exception:
            it += 1
            continue
        # keep candidates whose pre-LP max cosine is below ~0.670
        if vpre < 0.670 and time.time() < EXPLORE_END and time.time() < HARD - 6:
            lp_dead = min(time.time() + 2.8, EXPLORE_END + 4.0, HARD - 4.0)
            # don't exceed explore too much; allow a bit for polishing
            if lp_dead - time.time() > 0.6:
                try:
                    y = _lp_polish(y, deadline=lp_dead, init_radius=6e-3)
                except Exception:
                    pass
                try:
                    vpost = _max_cos(y)
                except Exception:
                    vpost = np.inf
            else:
                vpost = vpre
        else:
            vpost = vpre
        if np.all(np.isfinite(y)) and vpost < best - 1e-12:
            best = float(vpost)
            x_best = y.copy()
        it += 1
    # final short squeeze if time remains
    if time.time() < HARD - 2.5:
        try:
            dl = min(HARD - 1.0, time.time() + max(1.0, HARD - 1.5 - time.time()))
            x_best = _lp_polish(x_best, deadline=dl, init_radius=2e-3, cap=2500)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    x_best[~np.isfinite(x_best)] = 0.0
    nn = np.linalg.norm(x_best, axis=1, keepdims=True)
    for i in range(n):
        if not np.isfinite(nn[i, 0]) or nn[i, 0] < 1e-12:
            v = rng.standard_normal(size=(d,))
            nv = np.linalg.norm(v)
            if nv == 0 or not np.isfinite(nv):
                v = np.ones(d)
                nv = np.linalg.norm(v)
            x_best[i] = v / nv
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
