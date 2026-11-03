import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _lp_polish(X, budget):
    t0 = time.time()
    x = _norm(X.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < budget and radius > 1e-14:
        if time.time() - t0 + 0.25 > budget and radius < 8e-4:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu]
        m = float(c_all.max())
        sel = c_all > m - max(20.0 * radius, 1e-9)
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 3000:
            kth = np.partition(c_all, -3000)[-3000]
            sel = c_all >= kth
            cnt = int(sel.sum())
            if cnt > 3500:
                idx = np.where(sel)[0]
                order = np.argsort(-c_all[idx])[:3500]
                mask = np.zeros_like(c_all, dtype=bool)
                mask[idx[order]] = True
                sel = mask
        I = iu[0][sel]; J = iu[1][sel]
        csel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            y = _norm(y)
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _relax(Y, steps, T0, T1, lr0, lr1):
    Y = _norm(Y.copy())
    lT0, lT1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = Y @ Y.T
        np.fill_diagonal(G, -1.0)
        m = float(np.max(G))
        W = (G - m) / T
        np.clip(W, -50.0, 0.0, out=W)
        np.exp(W, out=W)
        np.fill_diagonal(W, 0.0)
        rs = np.sum(W, axis=1)
        WY = W @ Y
        r = np.sum(W * G, axis=1)
        inv = np.zeros_like(rs)
        mk = rs > 1e-12
        inv[mk] = 1.0 / rs[mk]
        avgWY = WY * inv[:, None]
        avgR = r * inv
        Dir = avgWY - avgR[:, None] * Y
        Dir[~mk] = 0.0
        nm = float(np.mean(np.linalg.norm(Dir, axis=1))) + 1e-12
        Y = Y - (lr / nm) * Dir
        Y = _norm(Y)
    return Y

def _best_hole(Xk, rng, ncand=20000):
    d = Xk.shape[1]
    cand = rng.standard_normal((ncand, d))
    cand = _norm(cand)
    dots = cand @ Xk.T
    vals = np.max(dots, axis=1)
    bi = int(np.argmin(vals))
    v = cand[bi].copy()
    for _ in range(25):
        dv = Xk @ v
        m = float(np.max(dv))
        w = np.exp(np.clip((dv - m) / 0.02, -50, 0))
        g = w @ Xk
        g = g - float(np.dot(g, v)) * v
        ng = float(np.linalg.norm(g))
        if ng < 1e-12:
            break
        g /= ng
        v = v - 0.03 * g
        nv = float(np.linalg.norm(v))
        if nv == 0:
            break
        v /= nv
    return v / float(np.linalg.norm(v))

def _greedy(pool, k, rng):
    N = pool.shape[0]
    s0 = int(rng.integers(N))
    sel = [s0]
    selset = np.zeros(N, dtype=bool); selset[s0] = True
    cur = pool @ pool[s0]
    cur[s0] = np.inf
    for _ in range(1, k):
        idx = int(np.argmin(cur))
        sel.append(idx); selset[idx] = True
        cur[idx] = np.inf
        d = pool @ pool[idx]
        np.maximum(cur, d, out=cur)
        cur[selset] = np.inf
    return pool[np.array(sel)].copy()

def _d9_pool(rng, n_extra=4500):
    d = 9
    roots = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in (-1.0, 1.0):
                for sj in (-1.0, 1.0):
                    v = np.zeros(d); v[i] = si; v[j] = sj
                    roots.append(v / np.sqrt(2.0))
    roots = np.array(roots)
    extra = np.zeros((n_extra, d))
    for k in range(n_extra):
        idx = rng.choice(d, size=4, replace=False)
        s = rng.choice(np.array([-1.0, 1.0]), size=4)
        v = np.zeros(d); v[idx] = s
        extra[k] = v / 2.0
    axes = np.eye(d)
    pool = np.vstack([roots, axes, -axes, extra])
    return _norm(pool)

def _e8xr_pool(rng, M=5000):
    d8 = 8
    rts = []
    for i in range(d8):
        for j in range(i + 1, d8):
            for si in (-1.0, 1.0):
                for sj in (-1.0, 1.0):
                    v = np.zeros(d8); v[i] = si; v[j] = sj
                    rts.append(v)
    for mask in range(256):
        if bin(mask).count("1") % 2 != 0:
            continue
        v = np.empty(d8)
        for i in range(d8):
            v[i] = 0.5 if ((mask >> i) & 1) else -0.5
        # even parity already; keep
        rts.append(v)
    rts = np.array(rts)  # ~240
    pool = np.empty((M, 9))
    for k in range(M):
        r = rts[int(rng.integers(len(rts)))]
        b = float(rng.uniform(-0.75, 0.75))
        a = np.sqrt(max((1 - b * b) / 2.0, 1e-12))
        pool[k, :8] = r * a
        pool[k, 8] = b
    return _norm(pool)

def _sorted_gram(X):
    n = X.shape[0]
    G = X @ X.T
    iu = np.triu_indices(n, 1)
    return np.sort(G[iu])

def _grow(X0, target, rng, t0, deadline, ncand=20000):
    X = _norm(X0.copy())
    while X.shape[0] < target:
        if time.time() > deadline - 2.0:
            break
        k = min(5, target - X.shape[0])
        for _ in range(k):
            v = _best_hole(X, rng, ncand=ncand)
            X = np.vstack([X, v[None, :]])
            if time.time() > deadline - 2.0:
                break
        X = _relax(X, steps=80, T0=0.05, T1=0.006, lr0=0.18, lr1=0.03)
    X = _relax(X, steps=120, T0=0.03, T1=0.003, lr0=0.12, lr1=0.02)
    return _norm(X)

def solve():
    t0 = time.time()
    DEADLINE = t0 + 155.0
    rng = np.random.default_rng(0)
    n, d = 181, 9
    ref = None; ref_val = np.inf; ref_sort = None
    best_loaded = None; best_loaded_val = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < best_loaded_val:
                        best_loaded_val = v; best_loaded = a.copy()
        except Exception:
            pass
    if best_loaded is not None:
        ref_sort = _sorted_gram(best_loaded)
        ref_val = best_loaded_val
    # build pools
    d9pool = _d9_pool(rng)
    e8pool = _e8xr_pool(rng, M=5000)
    big_rand = _norm(rng.standard_normal((6000, d)))
    seeds150 = []
    # 1 random
    seeds150.append(_relax(_norm(rng.standard_normal((150, d))), steps=80, T0=0.05, T1=0.006, lr0=0.18, lr1=0.03))
    # 2 D9 greedy
    seeds150.append(_relax(_greedy(d9pool, 150, rng), steps=80, T0=0.05, T1=0.006, lr0=0.18, lr1=0.03))
    # 3 E8xR greedy
    seeds150.append(_relax(_greedy(e8pool, 150, rng), steps=80, T0=0.05, T1=0.006, lr0=0.18, lr1=0.03))
    # 4 big-random greedy low-correlation subset
    seeds150.append(_relax(_greedy(big_rand, 150, rng), steps=80, T0=0.05, T1=0.006, lr0=0.18, lr1=0.03))
    # 5 mixed pool greedy
    mix = _norm(np.vstack([d9pool[rng.choice(len(d9pool), 2500, replace=False)], e8pool[rng.choice(len(e8pool), 2500, replace=False)]]))
    seeds150.append(_relax(_greedy(mix, 150, rng), steps=80, T0=0.05, T1=0.006, lr0=0.18, lr1=0.03))
    grown = []
    snap165 = []
    for s in seeds150:
        if time.time() > DEADLINE - 35.0:
            break
        X = _norm(s.copy())
        # grow to 165 snapshot then to 181, keep snapshot for delete-regrow variant
        X165 = _grow(X, 165, rng, t0, DEADLINE, ncand=20000)
        snap165.append(X165.copy())
        X181 = _grow(X165, 181, rng, t0, DEADLINE, ncand=22000)
        grown.append(X181)
    # delete-and-regrow variants (15-25 deletions) from snapshots / best grown
    if grown and time.time() < DEADLINE - 30.0:
        vals = [_max_cos(g) for g in grown]
        order = np.argsort(vals)
        # variant from best snapshot: delete 20 from 165 optimum and regrow to 181
        for oi in order[:2]:
            if time.time() > DEADLINE - 30.0:
                break
            base = snap165[oi] if oi < len(snap165) else grown[oi][:165]
            # per-point worst cosine
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            pw = np.max(G, axis=1)
            kdel = 20
            del_idx = np.argsort(-pw)[:kdel]
            keep = np.delete(base, del_idx, axis=0)
            # regrow to 181
            Xv = _grow(keep, 181, rng, t0, DEADLINE, ncand=22000)
            grown.append(Xv)
    # evaluate and distinct-filter via sorted-gram distance (rotation invariant)
    if not grown:
        return best_loaded if best_loaded is not None else _norm(rng.standard_normal((n, d)))
    vals = [_max_cos(g) for g in grown]
    sorts = [_sorted_gram(g) for g in grown]
    idx_sorted = np.argsort(vals)
    kept = []
    kept_vals = []
    DUP_TH = 5e-4
    for ii in idx_sorted:
        s = sorts[ii]
        dup = False
        for k2 in kept:
            if float(np.linalg.norm(s - sorts[k2])) < DUP_TH:
                dup = True
                break
        if ref_sort is not None and float(np.linalg.norm(s - ref_sort)) < DUP_TH:
            # near-duplicate of known record: discard per direction (unless nothing else)
            dup = True
        if not dup:
            kept.append(ii)
    if len(kept) == 0:
        # all duplicates: keep best grown anyway
        kept = [int(idx_sorted[0])]
    # two most promising distinct
    cand_idx = kept[:2]
    # short LP polish only for these two
    polished = []
    for ii in cand_idx:
        if time.time() > DEADLINE - 4.0:
            polished.append((vals[ii], grown[ii]))
            continue
        remain = DEADLINE - time.time()
        secs = min(7.0, max(2.0, remain - 2.0 if len(cand_idx) > 1 else remain - 1.0))
        try:
            xp = _lp_polish(grown[ii], budget=secs)
            vp = _max_cos(xp)
        except Exception:
            xp, vp = grown[ii], vals[ii]
        polished.append((vp, xp))
    polished.sort(key=lambda t: t[0])
    best_v, best_x = polished[0]
    # also consider unpolished kept beyond top2 (in case polish hurt)
    for ii in kept[2:]:
        if vals[ii] < best_v:
            best_v, best_x = vals[ii], grown[ii]
    # never regress vs loaded best
    if best_loaded is not None and best_loaded_val <= best_v:
        return _norm(best_loaded)
    return _norm(best_x)
