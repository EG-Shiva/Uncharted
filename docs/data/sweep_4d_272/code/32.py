import time
import numpy as np
from collections import deque
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _anneal(x, steps, tau0, tau1, lr0, lr1):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for k in range(steps):
        frac = k / (steps - 1) if steps > 1 else 0.0
        tau = tau0 * (tau1 / tau0) ** frac
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gm = G.max()
        np.subtract(G, gm, out=G)
        np.divide(G, tau, out=G)
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        S = G.sum()
        if not np.isfinite(S) or S == 0:
            break
        G /= S
        grad = G @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        g = grad - dot * x
        m *= b1
        m += (1 - b1) * g
        v *= b2
        v += (1 - b2) * (g * g)
        bc1 = 1.0 - b1 ** (k + 1)
        bc2 = 1.0 - b2 ** (k + 1)
        mh = m / bc1
        vh = v / bc2
        step = mh / (np.sqrt(vh) + eps)
        np.clip(step, -2.0, 2.0, out=step)
        x = x - lr * step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _lp_polish(x, seconds, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        mcur = c.max()
        sel = c > mcur - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _contact_adj(x, slack):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    adj = G > (m - slack)
    np.fill_diagonal(adj, False)
    return adj, G, m

def _bfs_community(adj, degrees, seed, target):
    n = adj.shape[0]
    visited = np.zeros(n, dtype=bool)
    q = deque([int(seed)])
    visited[int(seed)] = True
    order = []
    # precompute neighbor lists
    neigh = [np.where(adj[i])[0] for i in range(n)]
    # sort neighbors by degree descending for determinism
    while q and len(order) < target:
        u = q.popleft()
        order.append(u)
        if len(order) >= target:
            break
        # add neighbors sorted by degree
        nb = neigh[u]
        if len(nb) > 0:
            nb = nb[np.argsort(-degrees[nb])]
            for w in nb:
                if not visited[int(w)]:
                    visited[int(w)] = True
                    q.append(int(w))
    if len(order) < target:
        # fill with highest-degree unvisited to reach size
        rest = np.argsort(-degrees)
        for w in rest:
            if not visited[int(w)]:
                visited[int(w)] = True
                order.append(int(w))
                if len(order) >= target:
                    break
    return np.array(order[:target], dtype=int)

def solve():
    rng = np.random.default_rng(7)
    t0 = time.time()
    deadline = t0 + 172.0
    try:
        x0 = np.load("best.npy")
    except Exception:
        x0 = np.load("record.npy")
    x0 = np.asarray(x0, dtype=float)
    if x0.shape != (272, 4):
        flat = x0.ravel()
        need = 272 * 4
        if flat.size < need:
            extra = rng.standard_normal((need - flat.size,))
            flat = np.concatenate([flat, extra])
        x0 = flat[:need].reshape(272, 4)
    x0 = np.where(np.isfinite(x0), x0, 0.0)
    xn = np.linalg.norm(x0, axis=1, keepdims=True)
    xn[xn == 0] = 1.0
    x0 = x0 / xn

    x_best = x0.copy()
    best_val = _max_cos(x_best)
    x_cur = x_best.copy()
    cur_val = best_val

    # contact set helper for overlap
    iu0, iu1 = np.triu_indices(272, 1)
    def contact_mask(x, slack=1e-9):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        c = G[iu0, iu1]
        return (c > m - slack), m

    cur_mask, _ = contact_mask(x_cur, 1e-9)

    patch_sizes = [28, 36, 24, 40, 32, 26, 34, 30]
    slacks = [5e-7, 1e-6, 2e-7, 8e-7, 1e-7]
    cycle = 0

    while time.time() < deadline - 38.0:
        # stop if too little time for a full cycle (~10s)
        if time.time() > deadline - 45:
            break
        target = patch_sizes[cycle % len(patch_sizes)]
        slack = slacks[cycle % len(slacks)]
        adj, _, mcur = _contact_adj(x_cur, slack)
        degrees = adj.sum(axis=1)
        # tight degrees for rattlers/seeds
        adj_t, _, _ = _contact_adj(x_cur, 1e-9)
        deg_t = adj_t.sum(axis=1)
        order = np.argsort(-deg_t, kind="stable")
        # pick distinct seed: cycle through top-12
        top = order[:12]
        seed = int(top[cycle % len(top)])
        comm = _bfs_community(adj, deg_t, seed, target)
        rattlers = np.where(deg_t == 0)[0]
        del_set = set(comm.tolist()) | set(rattlers.tolist())
        # ensure size: if rattlers added beyond target, keep all (patch+3)
        del_idx = np.array(sorted(del_set), dtype=int)
        # safety: cap at 45
        if len(del_idx) > 45:
            # keep community + up to 3 rattlers
            keep_r = [r for r in rattlers if r not in set(comm.tolist())][:3]
            del_idx = np.array(sorted(set(comm.tolist()) | set(keep_r)), dtype=int)
        k = len(del_idx)
        core_mask = np.ones(272, dtype=bool)
        core_mask[del_idx] = False
        x_core = x_cur[core_mask]  # (272-k, 4)
        # sequential refill against frozen core + inserted
        combined = x_core.copy()
        new_pts = []
        for t in range(k):
            if time.time() > deadline - 38.0:
                break
            n_samples = 20000
            cand = rng.standard_normal((n_samples, 4))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            sims = cand @ combined.T
            scores = sims.max(axis=1)
            bi = int(scores.argmin())
            b = cand[bi].copy()
            bs = float(scores[bi])
            sigma = 0.03
            # local optimize insertion
            for it in range(110):
                prop = b + sigma * rng.standard_normal((64, 4))
                prop /= np.linalg.norm(prop, axis=1, keepdims=True)
                sc = (prop @ combined.T).max(axis=1)
                j = int(sc.argmin())
                if sc[j] < bs - 1e-12:
                    b = prop[j].copy()
                    bs = float(sc[j])
                    sigma *= 1.03
                else:
                    sigma *= 0.97
                if sigma < 2e-4:
                    sigma = 2e-4
            new_pts.append(b)
            combined = np.vstack([combined, b[None, :]])
        if len(new_pts) < k:
            cycle += 1
            continue
        new_pts = np.array(new_pts)
        x_new = x_cur.copy()
        x_new[del_idx] = new_pts
        # unfreeze: short high-temp log-sum-exp phase
        # vary temperature per cycle
        if cycle % 2 == 0:
            x_new = _anneal(x_new, 450, 0.022, 0.0025, 0.011, 0.0011)
        else:
            x_new = _anneal(x_new, 380, 0.014, 0.0012, 0.008, 0.0008)
        # light LP polish to re-jam (short)
        remain_lp = min(4.0, max(0.5, (deadline - 32.0 - time.time()) / 3.0))
        if remain_lp > 0.8:
            x_new = _lp_polish(x_new, remain_lp, deadline - 30.0)
        new_val = _max_cos(x_new)
        new_mask, _ = contact_mask(x_new, 1e-9)
        inter = float(np.logical_and(cur_mask, new_mask).sum())
        union = float(np.logical_or(cur_mask, new_mask).sum())
        overlap = inter / union if union > 0 else 1.0
        # acceptance based partly on graph change
        accept = False
        if new_val < cur_val - 1e-12:
            accept = True
        elif new_val < cur_val + 0.004 and overlap < 0.88:
            accept = True
        elif new_val < cur_val + 0.0015 and overlap < 0.95:
            accept = True
        elif new_val < cur_val + 0.0005 and rng.random() < 0.15:
            accept = True
        if new_val < best_val:
            x_best = x_new.copy()
            best_val = new_val
            # also move cur to best
            x_cur = x_new.copy()
            cur_val = new_val
            cur_mask = new_mask
        elif accept:
            x_cur = x_new.copy()
            cur_val = new_val
            cur_mask = new_mask
        # if stuck (no accept for a while), force jump to a different seed anyway
        cycle += 1
        # occasional restart from best with different patch if cur drifted too high
        if cur_val > best_val + 0.005:
            x_cur = x_best.copy()
            cur_val = best_val
            cur_mask, _ = contact_mask(x_cur, 1e-9)

    # final polish on best
    remain = deadline - time.time() - 1.5
    if remain > 4:
        x_pol = _lp_polish(x_best, min(remain, 30.0), deadline - 0.5)
        vpol = _max_cos(x_pol)
        if vpol <= best_val and np.all(np.isfinite(x_pol)):
            x_best = x_pol
            best_val = vpol
    if not np.all(np.isfinite(x_best)):
        x_best = x0.copy()
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
