import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _adam_smooth(x, iters, T0, T1, lr0, lr1, b1=0.9, b2=0.999, eps=1e-8):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1p = 1.0
    b2p = 1.0
    for k in range(iters):
        frac = k / max(iters - 1, 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mx = float(G.max())
        # softmax weights
        W = np.exp((G - mx) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        rs = W.sum(axis=1, keepdims=True)
        gt = gt / np.maximum(rs, 1e-12)
        b1p *= b1
        b2p *= b2
        m = b1 * m + (1.0 - b1) * gt
        v = b2 * v + (1.0 - b2) * (gt * gt)
        mh = m / (1.0 - b1p)
        vh = v / (1.0 - b2p)
        step = mh / (np.sqrt(vh) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        st = step - sdot * x
        # clip large tangent steps
        sn = np.linalg.norm(st, axis=1, keepdims=True)
        st = st / np.maximum(sn, 1.0)
        x = x - lr * st
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _greedy_init(rng, N, pool_n=4000):
    pool = rng.normal(size=(pool_n, 4))
    pool /= np.linalg.norm(pool, axis=1, keepdims=True)
    first = int(rng.integers(pool_n))
    chosen = [first]
    is_chosen = np.zeros(pool_n, dtype=bool)
    is_chosen[first] = True
    # iterative farthest (min max-dot)
    for _ in range(1, N):
        c = pool[chosen]  # (k,4)
        dots = pool @ c.T
        md = dots.max(axis=1)
        md[is_chosen] = 1e9
        j = int(np.argmin(md))
        chosen.append(j)
        is_chosen[j] = True
    return pool[np.array(chosen)].copy()

def _best_void(x, rng, n_samp=4000, beta=60.0):
    base = x
    cand = rng.normal(size=(n_samp, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    s = cand @ base.T
    sc = s.max(axis=1)
    order = np.argsort(sc)[:8]
    best_v = None
    best_s = 1e9
    for oi in order:
        v = cand[int(oi)].copy()
        for _ in range(30):
            dots = base @ v
            dm = float(dots.max())
            w = np.exp(beta * (dots - dm))
            g = w @ base
            gt = g - np.dot(g, v) * v
            nrm = float(np.linalg.norm(gt))
            if nrm < 1e-12:
                break
            gt /= nrm
            v = v - 0.08 * gt
            v /= np.linalg.norm(v)
        sc2 = float(np.max(base @ v))
        if sc2 < best_s:
            best_s = sc2
            best_v = v.copy()
    return best_v

def _delete_choice(x):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    iu0 = np.triu_indices(n, 1)
    c = G[iu0]
    K = min(40, c.shape[0])
    order = np.argpartition(-c, K - 1)[:K]
    # sort top K descending
    ov = c[order]
    o = np.argsort(-ov)
    order = order[o]
    TI = iu0[0][order]
    TJ = iu0[1][order]
    best_i = 0
    best_v = 1e9
    for i in range(n):
        val = None
        for k in range(K):
            if TI[k] != i and TJ[k] != i:
                val = float(c[order[k]])
                break
        if val is None:
            # fallback brute
            G2 = G.copy()
            G2[i, :] = -2.0
            G2[:, i] = -2.0
            val = float(G2.max())
        if val < best_v:
            best_v = val
            best_i = i
    return int(best_i)

def _lp_polish(x, seconds, r0=2e-3, cap=2200):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            v = c_all[sel]
            kth = np.sort(v)[-cap]
            sel = c_all >= kth
            cnt2 = int(np.count_nonzero(sel))
            if cnt2 > cap + 800:
                ids = np.where(sel)[0]
                keep = ids[::2][:cap]
                ns = np.zeros_like(sel)
                ns[keep] = True
                sel = ns
        I = iu0[0][sel]
        J = iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cd = np.arange(d)
        cols = np.concatenate([I[:, None] * d + cd, J[:, None] * d + cd,
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
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
            x, best = y, vv
            radius = min(radius * 1.6, 0.05)
        else:
            radius *= 0.5
    return x

def _grow_track(rng, N0, t0, cutoff):
    x = _greedy_init(rng, N0)
    x = _adam_smooth(x, iters=500, T0=0.022, T1=0.0018, lr0=0.016, lr1=0.002)
    n = N0
    while n < 163:
        if time.time() - t0 > cutoff:
            break
        v = _best_void(x, rng)
        x = np.vstack([x, v.reshape(1, 4)])
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        # polish after insertion, shorter as n grows
        x = _adam_smooth(x, iters=280, T0=0.016, T1=0.0015, lr0=0.012, lr1=0.0015)
        n += 1
    # final tune at 163
    if x.shape[0] == 163 and time.time() - t0 < cutoff:
        x = _adam_smooth(x, iters=250, T0=0.010, T1=0.0012, lr0=0.008, lr1=0.001)
    return x

def _shrink_track(rng, N1, t0, cutoff):
    z = rng.normal(size=(N1, 4))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    z = _adam_smooth(z, iters=550, T0=0.022, T1=0.0018, lr0=0.016, lr1=0.002)
    while z.shape[0] > 163:
        if time.time() - t0 > cutoff:
            break
        idx = _delete_choice(z)
        z = np.delete(z, idx, axis=0)
        z = _adam_smooth(z, iters=280, T0=0.016, T1=0.0015, lr0=0.012, lr1=0.0015)
    if z.shape[0] == 163 and time.time() - t0 < cutoff:
        z = _adam_smooth(z, iters=250, T0=0.010, T1=0.0012, lr0=0.008, lr1=0.001)
    return z

def solve():
    t0 = time.time()
    HARD = 165.0
    # homotopy must finish early to leave 15s for LP
    def time_left():
        return HARD - (time.time() - t0)
    # load fallback record
    fb = None
    fb_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            c = np.load(fn)
            c = np.asarray(c, dtype=float)
            if c.shape == (163, 4):
                c = c / np.linalg.norm(c, axis=1, keepdims=True)
                v = _max_cos(c)
                if v < fb_val:
                    fb_val = v
                    fb = c.copy()
        except Exception:
            continue
    if fb is None:
        rng0 = np.random.default_rng(0)
        fb = rng0.normal(size=(163, 4))
        fb /= np.linalg.norm(fb, axis=1, keepdims=True)
        fb_val = _max_cos(fb)

    homo_cutoff = 148.0  # seconds after t0 to stop homotopy
    best_h = None
    best_h_val = 1e9
    # independent seeds: 3 grow + 3 shrink
    grow_N0 = [150, 154, 157]
    shrink_N1 = [168, 172, 176]
    # interleave tracks
    plan = [("g", grow_N0[0], 101), ("s", shrink_N1[0], 201),
            ("g", grow_N0[1], 102), ("s", shrink_N1[1], 202),
            ("g", grow_N0[2], 103), ("s", shrink_N1[2], 203)]
    for kind, N, seed in plan:
        elapsed = time.time() - t0
        if elapsed > homo_cutoff:
            break
        if time_left() < 20:
            break
        rng = np.random.default_rng(seed)
        try:
            if kind == "g":
                x = _grow_track(rng, N, t0, homo_cutoff)
            else:
                x = _shrink_track(rng, N, t0, homo_cutoff)
        except Exception:
            continue
        if x is None or x.shape != (163, 4):
            continue
        if not np.all(np.isfinite(x)):
            continue
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        v = _max_cos(x)
        if v < best_h_val:
            best_h_val = v
            best_h = x.copy()

    # reserve final 15 seconds for LP polishing
    elapsed = time.time() - t0
    lp_budget = min(15.0, max(0.5, HARD - elapsed - 0.5))
    if best_h is not None and time_left() > 1.0:
        try:
            best_h = _lp_polish(best_h, seconds=lp_budget)
            best_h = best_h / np.linalg.norm(best_h, axis=1, keepdims=True)
            best_h_val = _max_cos(best_h)
        except Exception:
            pass

    # fallback if no homotopy beats record
    if best_h is None or not np.all(np.isfinite(best_h)) or best_h_val >= fb_val:
        out = fb.copy()
    else:
        out = best_h.copy()
    out = out / np.linalg.norm(out, axis=1, keepdims=True)
    out = np.nan_to_num(out)
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out
