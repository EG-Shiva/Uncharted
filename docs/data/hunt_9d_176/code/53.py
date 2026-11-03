import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _fast_quench(x, steps):
    n, d = x.shape
    v = np.zeros_like(x)
    mu = 0.82
    for s in range(steps):
        frac = s / max(1, steps - 1)
        alpha = 18.0 + 110.0 * frac
        lr = 0.02 * (1.0 - frac) + 0.002
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        # shifted exp, all <=1
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        tang = grad - dot * x
        tn = np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12
        direction = tang / tn
        v = mu * v - lr * direction
        vn = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        maxv = 3.0 * lr
        sc = np.minimum(1.0, maxv / vn)
        v *= sc
        x = x + v
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x

def _find_hole(Xr, rng, ntrials=8000, ref_steps=120):
    n_f, d = Xr.shape
    S = rng.standard_normal((ntrials, d))
    S /= np.linalg.norm(S, axis=1, keepdims=True) + 1e-12
    dots = S @ Xr.T
    rowm = np.max(dots, axis=1)
    bi = int(np.argmin(rowm))
    h = S[bi].copy()
    hn = np.linalg.norm(h)
    if hn == 0 or not np.isfinite(hn):
        h = S[0]
    else:
        h /= hn
    for s in range(ref_steps):
        lr = 0.08 * (1.0 - s / max(1, ref_steps)) + 0.005
        dv = Xr @ h
        j = int(np.argmax(dv))
        dm = float(dv[j])
        gj = Xr[j] - dm * h
        h = h - lr * gj
        nh = np.linalg.norm(h)
        if nh == 0 or not np.isfinite(nh):
            break
        h /= nh
    return h

def _contacts_hubs(x, topk=12):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    row_max = np.max(G, axis=1)
    cnts = np.sum(G > m - 1e-9, axis=1)
    order = np.lexsort((row_max, cnts))
    hubs = order[-topk:]
    return G, m, cnts, row_max, hubs

def _bfs_set(adj_list, start, K, rng):
    n = len(adj_list)
    seen = set([int(start)])
    queue = [int(start)]
    # randomize neighbour order for diversity
    qi = 0
    while len(seen) < K and qi < len(queue):
        u = queue[qi]; qi += 1
        nbrs = adj_list[u]
        if len(nbrs) > 1:
            perm = rng.permutation(len(nbrs))
            nbrs = [nbrs[i] for i in perm]
        for w in nbrs:
            if w not in seen:
                seen.add(int(w))
                queue.append(int(w))
                if len(seen) >= K:
                    break
    # if contact component too small, fill randomly
    while len(seen) < K:
        c = int(rng.integers(0, n))
        if c not in seen:
            seen.add(c)
    return sorted(seen)

def _lp_polish(x, seconds, r0=2e-3, cap=3000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    t_end = time.time() + seconds
    radius = float(r0)
    iu0 = np.triu_indices(n, 1)
    while time.time() < t_end and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(np.max(c_all))
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > cap:
            idx_all = np.where(sel)[0]
            part = np.argpartition(-c_all[idx_all], cap)[:cap]
            keep = np.zeros_like(sel)
            keep[idx_all[part]] = True
            sel = keep
            cnt = cap
        I = iu0[0][sel]
        J = iu0[1][sel]
        c = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        base_I = I[:, None] * d + np.arange(d)
        base_J = J[:, None] * d + np.arange(d)
        cols = np.concatenate([base_I, base_J, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y /= nr
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    np.random.seed(0)
    rng = np.random.default_rng(7)
    t0 = time.time()
    BUDGET = 168.0
    FINAL_RESERVE = 22.0
    EXPLORE_END = t0 + BUDGET - FINAL_RESERVE
    try:
        x_best = np.load("best.npy").astype(float)
    except Exception:
        x_best = np.load("record.npy").astype(float)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    n, d = x_best.shape
    best_val = _max_cos(x_best)
    # cache contact structure, refresh periodically
    Gc, mc, cnts, rowm, hubs = _contacts_hubs(x_best)
    # build adjacency for BFS
    def build_adj():
        nonlocal Gc, mc
        thr = mc - 5e-8
        A = Gc > thr
        np.fill_diagonal(A, False)
        return [list(np.where(A[i])[0]) for i in range(n)]
    adj = build_adj()
    K_cycle = [10, 16, 22, 30, 12, 20, 26, 14]
    it = 0
    refresh = 0
    while time.time() < EXPLORE_END:
        it += 1
        # periodic refresh of hubs/adj from best
        if refresh >= 12 or it == 1:
            Gc, mc, cnts, rowm, hubs = _contacts_hubs(x_best)
            adj = build_adj()
            refresh = 0
        refresh += 1
        xc = x_best.copy()
        mode = it % 10
        if mode == 0:
            # pure random restart quench (different basin)
            xc = rng.standard_normal((n, d))
            xc /= np.linalg.norm(xc, axis=1, keepdims=True)
            xc = _fast_quench(xc, 350)
        elif mode == 9:
            # global jitter
            sig = 0.06
            xc = xc + sig * rng.standard_normal(xc.shape)
            xc /= np.linalg.norm(xc, axis=1, keepdims=True)
            xc = _fast_quench(xc, 350)
        elif mode == 8:
            # subset shake of 12
            sel = rng.choice(n, size=12, replace=False)
            xc[sel] = xc[sel] + 0.08 * rng.standard_normal((12, d))
            xc[sel] /= np.linalg.norm(xc[sel], axis=1, keepdims=True)
            xc = _fast_quench(xc, 350)
        else:
            K = K_cycle[(it - 1) % len(K_cycle)]
            hub = int(rng.choice(hubs))
            Sdel = _bfs_set(adj, hub, K, rng)
            mask = np.ones(n, dtype=bool)
            mask[Sdel] = False
            cur = xc[mask].copy()
            order = list(Sdel)
            rng.shuffle(order)
            for j in order:
                h = _find_hole(cur, rng, ntrials=7000, ref_steps=100)
                xc[j] = h
                cur = np.vstack([cur, h[None, :]])
                if time.time() >= EXPLORE_END:
                    break
            # small jitter on reinserted to break symmetry
            xc[Sdel] = xc[Sdel] + 0.005 * rng.standard_normal((len(Sdel), d))
            xc[Sdel] /= np.linalg.norm(xc[Sdel], axis=1, keepdims=True)
            xc = _fast_quench(xc, 380)
        v = _max_cos(xc)
        if v < best_val - 1e-12:
            x_best = xc / np.linalg.norm(xc, axis=1, keepdims=True)
            best_val = float(v)
            refresh = 100  # force rebuild next iter
        # safety: if very close to timeout break
        if time.time() >= EXPLORE_END:
            break
    remain = BUDGET - (time.time() - t0)
    if remain > 6:
        # longer smooth quench then LP
        x_best = _fast_quench(x_best, 900)
        best_val = _max_cos(x_best)
        remain2 = BUDGET - (time.time() - t0)
        if remain2 > 4:
            x_best = _lp_polish(x_best, seconds=remain2 - 1.5, r0=1.2e-3)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
