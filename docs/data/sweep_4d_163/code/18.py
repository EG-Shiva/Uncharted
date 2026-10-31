import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _row_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1), g

def _smooth(x, iters, T0, T1, lr0, lr1):
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

def _hard(x, iters=150, lr0=0.005, lr1=0.0008):
    n, d = x.shape
    for k in range(iters):
        frac = k / max(iters - 1, 1)
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # nearest neighbour per row
        nb_idx = np.argmax(G, axis=1)
        nb = x[nb_idx]
        dprod = np.sum(nb * x, axis=1, keepdims=True)
        grad = nb - dprod * x
        gn = np.linalg.norm(grad, axis=1, keepdims=True)
        grad = grad / np.maximum(gn, 1e-12)
        rm = G.max(axis=1)
        mask = (rm > m - 0.01).astype(float)[:, None]
        x = x - lr * grad * (0.3 + 0.7 * mask)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

_IU = None
def _lp(x, seconds, r0=2e-3):
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
            vals = c_all[sel]
            kth = np.sort(vals)[-2200]
            sel = c_all >= kth
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

def _void_against(ref, rng, n_samples=3000):
    d = ref.shape[1]
    cand = rng.normal(size=(n_samples, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = (cand @ ref.T).max(axis=1)
    v = cand[int(np.argmin(scores))].copy()
    for _ in range(50):
        dots = ref @ v
        mx = dots.max()
        w = np.exp(60.0 * (dots - mx))
        g = w @ ref
        gt = g - np.dot(g, v) * v
        nrm = np.linalg.norm(gt)
        if nrm < 1e-12:
            break
        gt /= nrm
        v = v - 0.07 * gt
        v /= np.linalg.norm(v)
    return v

def solve():
    rng = np.random.default_rng(12345)
    t0 = time.time()
    BUDGET = 165.0
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        if c is None:
            continue
        c = np.asarray(c, dtype=float)
        if c.shape != (163, 4):
            continue
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        cands.append(c)
    if not cands:
        z = rng.normal(size=(163, 4))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    best = min(cands, key=_max_cos).copy()
    best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val
    n, d = 163, 4
    sigmas = [0.015, 0.03, 0.06, 0.09, 0.02, 0.045]
    TEMP = 0.0015
    trial = 0
    lp_count = 0
    # pre-allocate iu
    global _IU
    _IU = np.triu_indices(n, 1)
    while time.time() - t0 < BUDGET and trial < 250:
        if time.time() - t0 > BUDGET - 12 and lp_count > 0:
            break
        move = trial % 6
        sigma = sigmas[trial % len(sigmas)]
        y = cur.copy()
        rm, Gcur = _row_max(cur)
        mcur = float(Gcur.max())
        if move == 0:
            y = y + rng.normal(scale=sigma, size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif move == 1:
            u = rng.normal(size=4); u /= np.linalg.norm(u)
            dots = y @ u
            mask = dots > 0.65
            if np.count_nonzero(mask) < 5:
                idx = np.argsort(dots)[-12:]
                mask = np.zeros(n, dtype=bool); mask[idx] = True
            nm = int(np.count_nonzero(mask))
            y[mask] += rng.normal(scale=sigma * 2.5, size=(nm, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif move == 2:
            k = 8
            worst = np.argsort(rm)[-k:]
            y[worst] += rng.normal(scale=0.09, size=(k, d))
            # neighbours of worst
            neigh_mask = np.zeros(n, dtype=bool)
            for wi in worst:
                neigh_mask = neigh_mask | (Gcur[wi] > mcur - 0.02)
            neigh_mask[worst] = False
            ni = np.where(neigh_mask)[0]
            if len(ni) > 0:
                y[ni] += rng.normal(scale=0.035, size=(len(ni), d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif move == 3:
            # rattler / coldest re-insertion to voids
            order = np.argsort(rm)
            nr = 3 if (trial % 12 < 6) else 2
            # include true rattlers first
            ratt = np.where(rm < mcur - 0.002)[0]
            if len(ratt) >= nr:
                rep = rng.choice(ratt, size=nr, replace=False)
            else:
                rep = order[:nr]
                # add random choice among coldest 15 for diversity
                if rng.random() < 0.5:
                    rep[-1] = rng.choice(order[:15])
            keep = np.ones(n, dtype=bool); keep[rep] = False
            ref = y[keep]
            newpts = []
            for _ in range(nr):
                v = _void_against(ref if len(newpts) == 0 else np.vstack([ref] + newpts), rng)
                newpts.append(v.reshape(1, d))
            y[rep] = np.vstack(newpts)
            # small shake of rest to break symmetry
            y[keep] += rng.normal(scale=0.005, size=(n - nr, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif move == 4:
            # jam relief: remove hub + its worst neighbour
            thr = mcur - 1e-4
            deg = (Gcur >= thr).sum(axis=1)
            hub = int(np.argmax(deg + rng.normal(scale=0.3, size=n) * 0.1))
            nb = int(np.argmax(Gcur[hub]))
            rep = np.array([hub, nb])
            keep = np.ones(n, dtype=bool); keep[rep] = False
            ref = y[keep]
            v1 = _void_against(ref, rng)
            v2 = _void_against(np.vstack([ref, v1.reshape(1, d)]), rng)
            y[hub] = v1; y[nb] = v2
            y[keep] += rng.normal(scale=0.008, size=(n - 2, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        else:
            y = y + rng.normal(scale=0.05, size=y.shape)
            # random global rotation to diversify
            A = rng.normal(size=(d, d))
            Q, _ = np.linalg.qr(A)
            y = y @ Q
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        # relax with annealed soft-max
        T0 = 0.018 if (trial % 3 == 0) else 0.028
        y = _smooth(y, iters=450, T0=T0, T1=0.0015, lr0=0.009, lr1=0.001)
        y = _hard(y, iters=120, lr0=0.004, lr1=0.0008)
        y_val = _max_cos(y)
        # Metropolis acceptance on smoothed value
        if y_val < cur_val or rng.random() < np.exp((cur_val - y_val) / TEMP):
            cur, cur_val = y, y_val
        # selective LP polish
        do_lp = False
        if y_val < best_val + 0.004 and (y_val < cur_val + 0.001 or y_val < best_val + 0.0008):
            # limit frequency: every 3rd trial or very promising
            if (trial % 3 == 0) or (y_val < best_val):
                do_lp = True
        # ensure at least occasional LP on current best region
        if trial % 20 == 19:
            do_lp = True
            y = cur.copy()
            y_val = cur_val
        if do_lp and time.time() - t0 < BUDGET - 10:
            secs = 5.0 if y_val < best_val else 3.5
            # shrink secs near end
            remain = BUDGET - (time.time() - t0)
            secs = min(secs, max(remain - 2, 1.5))
            z = _lp(y, seconds=secs)
            z_val = _max_cos(z)
            lp_count += 1
            if z_val < best_val:
                best, best_val = z, z_val
                cur, cur_val = z.copy(), z_val
            elif z_val < cur_val and rng.random() < 0.7:
                cur, cur_val = z.copy(), z_val
        trial += 1
    # final quick LP on best if time remains
    if time.time() - t0 < BUDGET - 3:
        secs = min(8.0, max(BUDGET - (time.time() - t0) - 1, 1.0))
        best = _lp(best, seconds=secs, r0=1e-3)
        best_val = _max_cos(best)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
