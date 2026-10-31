import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _smooth_descent(x, iters, T0, T1, lr0, lr1):
    n, d = x.shape
    for k in range(iters):
        frac = k / max(iters - 1, 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        rs = W.sum(axis=1, keepdims=True)
        step = gt / np.maximum(rs, 1e-12)
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        step = step / np.maximum(sn, 1.0)
        x = x - lr * step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _hard_descent(x, iters=200, lr0=0.006, lr1=0.001, topk=2):
    n, d = x.shape
    for k in range(iters):
        frac = k / max(iters - 1, 1)
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        # top-k worst neighbours per row
        # n=163 small, full argsort is fine
        idx = np.argpartition(-G, topk, axis=1)[:, :topk]
        grad = np.zeros_like(x)
        for t in range(topk):
            w = 1.0 / (t + 1.0)
            nb = x[idx[:, t]]
            dprod = np.sum(nb * x, axis=1, keepdims=True)
            grad += w * (nb - dprod * x)
        # grad points toward neighbours; move opposite
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        gn = np.linalg.norm(gt, axis=1, keepdims=True)
        gt = gt / np.maximum(gn, 1e-12)
        # only move points that are near the global max to avoid spoiling
        m = float(G.max())
        rm = G.max(axis=1)
        mask = (rm > m - 0.01).astype(float)[:, None]
        x = x - lr * gt * (0.3 + 0.7 * mask)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

_IU = None
def _lp_polish(x, seconds, r0=2e-3):
    global _IU
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    if _IU is None or _IU[0].shape[0] != n * (n - 1) // 2:
        _IU = np.triu_indices(n, 1)
    iu = _IU
    best = _max_cos(x)
    radius = r0
    start = time.time()
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 2200:
            # keep top 2200
            vals_all = c_all[sel]
            kth = np.sort(vals_all)[-2200]
            sel = c_all >= kth
            # if still too many due to ties, subsample deterministically
            cnt2 = int(np.count_nonzero(sel))
            if cnt2 > 3000:
                ids = np.where(sel)[0][::2]
                ns = np.zeros_like(sel)
                ns[ids[:3000]] = True
                sel = ns
        I = iu[0][sel]; J = iu[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cd = np.arange(d)
        cols = np.concatenate([I[:, None] * d + cd, J[:, None] * d + cd, np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.6, 0.05)
        else:
            radius *= 0.5
    return x

def _find_voids(x, rng, n_voids=3, n_samples=4000):
    n, d = x.shape
    cand = rng.normal(size=(n_samples, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = (cand @ x.T).max(axis=1)
    order = np.argsort(scores)[:n_voids * 4]
    picks = cand[order]
    voids = []
    for v0 in picks:
        v = v0.copy()
        for _ in range(40):
            dots = x @ v
            # soft-max weights to nearest points
            w = np.exp(60.0 * (dots - dots.max()))
            g = w @ x
            gt = g - np.dot(g, v) * v
            nrm = np.linalg.norm(gt)
            if nrm < 1e-12:
                break
            gt /= nrm
            v = v - 0.08 * gt
            v /= np.linalg.norm(v)
        voids.append(v)
    voids = np.array(voids)
    # rank refined voids by max dot (lower better), deduplicate
    sc = np.array([float((x @ v).max()) for v in voids])
    o = np.argsort(sc)
    voids = voids[o]
    out = []
    for v in voids:
        if all(float(np.dot(v, u)) < 0.98 for u in out):
            out.append(v)
        if len(out) >= n_voids:
            break
    while len(out) < n_voids:
        z = rng.normal(size=d); z /= np.linalg.norm(z)
        out.append(z)
    return np.array(out)

def solve():
    rng = np.random.default_rng(20261031)
    t_start = time.time()
    DEADLINE = t_start + 172.0
    HOP_END = t_start + 142.0
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        cands.append(np.asarray(xb, dtype=float))
    if xr is not None:
        cands.append(np.asarray(xr, dtype=float))
    if not cands:
        z = rng.normal(size=(163, 4))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    best = None; best_val = 1e9
    for c in cands:
        if c.shape != (163, 4):
            continue
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        v = _max_cos(c)
        if v < best_val:
            best_val = v; best = c.copy()
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    # quick initial squeeze in case best.npy not polished
    if time.time() - t_start < 20:
        best = _lp_polish(best, seconds=min(4.0, max(1.0, HOP_END - time.time()) * 0.1))
        best_val = _max_cos(best)
    cur = best.copy(); cur_val = best_val
    n, d = 163, 4
    trial = 0
    stall = 0
    sigmas = [0.012, 0.03, 0.06, 0.02, 0.045, 0.008]
    while time.time() < HOP_END and trial < 200:
        sigma = sigmas[trial % len(sigmas)]
        # occasional big kick
        if trial % 13 == 12:
            move = 5
        else:
            move = trial % 5
        y = cur.copy()
        if move == 0:
            y = y + rng.normal(scale=sigma, size=y.shape)
        elif move == 1:
            u = rng.normal(size=4); u /= np.linalg.norm(u)
            dots = y @ u
            mask = dots > 0.70
            if int(np.count_nonzero(mask)) < 3:
                idx = np.argsort(dots)[-12:]
                mask = np.zeros(n, dtype=bool); mask[idx] = True
            noise = np.zeros_like(y)
            nm = int(np.count_nonzero(mask))
            noise[mask] = rng.normal(scale=sigma * 2.0, size=(nm, d))
            noise[~mask] = rng.normal(scale=sigma * 0.25, size=(n - nm, d))
            y = y + noise
        elif move == 2:
            G = y @ y.T; np.fill_diagonal(G, -2.0)
            rm = G.max(axis=1)
            k = 12
            idx = np.argsort(rm)[-k:]
            noise = rng.normal(scale=sigma * 0.3, size=y.shape)
            noise[idx] = rng.normal(scale=sigma * 2.5, size=(k, d))
            y = y + noise
        elif move == 3:
            # void reinsertion for rattlers / worst
            G = y @ y.T; np.fill_diagonal(G, -2.0)
            rm = G.max(axis=1); m = float(rm.max())
            ratt = np.where(rm < m - 3e-4)[0]
            worst = np.argsort(rm)[-6:]
            targets = list(ratt[:4]) + [w for w in worst if w not in ratt]
            targets = targets[:4]
            if len(targets) == 0:
                targets = list(np.argsort(rm)[-2:])
            y = y + rng.normal(scale=sigma * 0.4, size=y.shape)
            voids = _find_voids(y, rng, n_voids=len(targets), n_samples=3500)
            for r, v in zip(targets, voids):
                y[r] = v + rng.normal(scale=0.01, size=d)
            # jam relief: move one hub slightly too
            G2 = y @ y.T; np.fill_diagonal(G2, -2.0)
            rm2 = (G2 > m - 1e-9).sum(axis=1)
            hub = int(np.argmax(rm2))
            if rm2[hub] >= 8 and hub not in targets:
                y[hub] = y[hub] + rng.normal(scale=sigma, size=d)
        elif move == 4:
            # edge-split: push max pair apart
            G = y @ y.T; np.fill_diagonal(G, -2.0)
            a, b = np.unravel_index(int(np.argmax(G)), G.shape)
            y = y + rng.normal(scale=sigma * 0.4, size=y.shape)
            diff_a = y[a] - y[b]; diff_a -= np.dot(diff_a, y[a]) * y[a]
            diff_b = y[b] - y[a]; diff_b -= np.dot(diff_b, y[b]) * y[b]
            na = np.linalg.norm(diff_a); nb = np.linalg.norm(diff_b)
            if na > 1e-12:
                y[a] = y[a] + 0.03 * diff_a / na
            if nb > 1e-12:
                y[b] = y[b] + 0.03 * diff_b / nb
            # shake their common neighbours
            neigh = np.where((G[a] > float(G.max()) - 0.02) | (G[b] > float(G.max()) - 0.02))[0]
            for q in neigh[:8]:
                y[q] = y[q] + rng.normal(scale=sigma, size=d)
        else:
            y = y + rng.normal(scale=0.09, size=y.shape)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        # annealed smoothing with randomized temps
        T0 = float(rng.uniform(0.015, 0.035))
        T1 = float(rng.uniform(0.001, 0.003))
        it = 550 if sigma > 0.03 else 650
        y = _smooth_descent(y, iters=it, T0=T0, T1=T1, lr0=0.009, lr1=0.0012)
        y = _hard_descent(y, iters=120, lr0=0.004, lr1=0.0008, topk=2)
        # screen before expensive LP
        vs = _max_cos(y)
        if vs > cur_val + 0.025 and vs > best_val + 0.01:
            trial += 1
            continue
        remain_hop = HOP_END - time.time()
        if remain_hop < 0.5:
            break
        sec = 1.1 if trial < 30 else 0.9
        sec = min(sec, max(0.4, remain_hop / 3.0))
        if sec > 0.35:
            y = _lp_polish(y, seconds=sec)
        v = _max_cos(y)
        improved_best = v < best_val - 1e-12
        if improved_best:
            best = y.copy(); best_val = v
            cur = y.copy(); cur_val = v
            stall = 0
            # squeeze extra immediately
            if HOP_END - time.time() > 3.0:
                best = _lp_polish(best, seconds=1.2, r0=1e-3)
                best_val = _max_cos(best)
                cur = best.copy(); cur_val = best_val
        else:
            # Metropolis sideways move on cur
            Temp = 3e-6 * (1.0 - (time.time() - t_start) / (HOP_END - t_start + 1e-9))
            Temp = max(Temp, 5e-7)
            if v < cur_val or rng.random() < np.exp((cur_val - v) / Temp):
                cur = y.copy(); cur_val = v
                stall = 0
            else:
                stall += 1
            if stall >= 12:
                # restart from best with kick
                cur = best.copy(); cur_val = best_val
                stall = 0
        trial += 1
    # final long polish
    elapsed = time.time() - t_start
    budget = max(4.0, min(28.0, DEADLINE - time.time() - 2.0))
    if budget > 1.0:
        # alternate smooth + LP twice
        best = _smooth_descent(best, iters=400, T0=0.012, T1=0.0008, lr0=0.005, lr1=0.0006)
        best = _lp_polish(best, seconds=budget * 0.6)
        if DEADLINE - time.time() > 3.0:
            best = _hard_descent(best, iters=80, lr0=0.002, lr1=0.0005, topk=3)
            best = _lp_polish(best, seconds=max(1.0, DEADLINE - time.time() - 1.5))
    best = np.nan_to_num(best)
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
