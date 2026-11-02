import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def contact_counts(x, tol=1.5e-3):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = float(g.max())
    cnt = np.sum(g > m - tol, axis=1)
    return cnt, m

def lp_polish(x, deadline, radius_init=2.5e-3, cap=1500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = radius_init
    iu = np.triu_indices(n, 1)
    while radius > 1e-13:
        if time.time() >= deadline:
            break
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        ns = int(np.sum(sel))
        if ns == 0:
            break
        if ns > cap:
            idx = np.argpartition(-c, cap)[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = sel & mask
        I, J = iu[0][sel], iu[1][sel]
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
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def quench_lse(X_init, steps, lr0, lr1, b0, b1):
    X = X_init.copy()
    n, d = X.shape
    m1 = np.zeros_like(X); m2 = np.zeros_like(X)
    be1, be2, eps = 0.9, 0.999, 1e-8
    p1 = 1.0; p2 = 1.0
    l0, l1 = np.log(lr0), np.log(lr1)
    e0, e1 = np.log(b0), np.log(b1)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        lr = float(np.exp(l0 + (l1 - l0) * frac))
        beta = float(np.exp(e0 + (e1 - e0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gm = float(G.max())
        E = np.exp(np.clip(beta * (G - gm), -30.0, 0.0))
        np.fill_diagonal(E, 0.0)
        Gr = E @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        p1 *= be1; p2 *= be2
        m1 = be1 * m1 + (1.0 - be1) * Gt
        m2 = be2 * m2 + (1.0 - be2) * (Gt * Gt)
        mh = m1 / (1.0 - p1); vh = m2 / (1.0 - p2)
        X = X - lr * mh / (np.sqrt(vh) + eps)
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X

def greedy_init(n, d, rng, cand=4000):
    X = []
    v = rng.standard_normal(d); v /= np.linalg.norm(v)
    X.append(v)
    for _ in range(1, n):
        Z = rng.standard_normal((cand, d))
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        F = np.array(X)
        S = Z @ F.T
        m = S.max(axis=1)
        X.append(Z[int(np.argmin(m))])
    return np.array(X)

def best_hole_insertion(fixed, rng, n_cand=200000, chunk=25000):
    d = fixed.shape[1]
    best_v = None; best_m = 1e9
    nch = n_cand // chunk
    for _ in range(nch):
        Z = rng.standard_normal((chunk, d))
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        S = Z @ fixed.T
        m = S.max(axis=1)
        i = int(np.argmin(m))
        if m[i] < best_m:
            best_m = float(m[i]); best_v = Z[i].copy()
    # local refine on sphere with smooth LSE objective
    v = best_v.copy()
    for _ in range(80):
        dots = fixed @ v
        beta = 120.0
        dm = float(dots.max())
        w = np.exp(np.clip(beta * (dots - dm), -30, 0))
        w = w / (w.sum() + 1e-300)
        g = w @ fixed
        g = g - np.dot(g, v) * v
        v = v - 0.03 * g
        nv = np.linalg.norm(v)
        if nv == 0 or not np.isfinite(nv):
            v = best_v.copy(); break
        v = v / nv
    return v

def deletion_pick(X):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    iu = np.triu_indices(n, 1)
    c = G[iu]
    order = np.argsort(-c)
    topk = min(30, len(order))
    topI = iu[0][order[:topk]]; topJ = iu[1][order[:topk]]; topC = c[order[:topk]]
    best_r = -1; best_score = 1e9
    for r in range(n):
        # remaining max after removing r
        mask = (topI != r) & (topJ != r)
        if np.any(mask):
            rem = float(topC[mask].max())
        else:
            # all top pairs involve r: fallback to global second best
            m2 = float(c[(iu[0] != r) & (iu[1] != r)].max())
            rem = m2
        cnt_r = int(np.sum(G[r] > rem - 1.5e-3))
        score = rem + 1e-6 * cnt_r
        if score < best_score:
            best_score = score; best_r = r
    return best_r

def fingerprint(X):
    G = X @ X.T
    w = np.linalg.eigvalsh(G)
    w = np.sort(w)[::-1][:6]
    cnt, m = contact_counts(X, tol=1.5e-3)
    hist = np.bincount(np.clip(cnt, 0, 8))[:(9)]
    if len(hist) < 9:
        hist = np.pad(hist, (0, 9 - len(hist)))
    return w.astype(float), hist.astype(float), float(m)

def spec_dist(a, b):
    return float(np.linalg.norm(a - b))

def align_matrix(A, B, rng, iters=3):
    # find orthogonal Q aligning B to A with greedy assignment
    d = A.shape[1]; n = A.shape[0]
    Qr = rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(Qr)
    for _ in range(iters):
        Bq = B @ Q
        S = A @ Bq.T  # n x n cosine
        # greedy assignment: sort pairs desc
        flat = np.argsort(-S.ravel(), kind='stable')
        used_a = np.zeros(n, bool); used_b = np.zeros(n, bool)
        pa = []; pb = []
        for f in flat:
            a = f // n; b = f % n
            if (not used_a[a]) and (not used_b[b]):
                used_a[a] = True; used_b[b] = True
                pa.append(a); pb.append(b)
                if len(pa) >= n:
                    break
        Am = A[np.array(pa)]; Bm = B[np.array(pb)]
        M = Bm.T @ Am
        U, _, Vt = np.linalg.svd(M)
        Q = U @ Vt
    return Q

def recombine(A, B, rng):
    n, d = A.shape
    Q = align_matrix(A, B, rng, iters=3)
    Bal = B @ Q
    pool = np.vstack([A, Bal])
    # greedy farthest selection of n from pool starting from random seed
    start = int(rng.integers(0, 2 * n))
    sel = [start]
    selset = {start}
    # precompute pool gram
    Gp = pool @ pool.T
    curmax = Gp[start].copy()
    curmax[start] = -2
    for _ in range(1, n):
        # pick candidate minimizing max-sim to selected, among unselected; add randomness: sample 8 best
        order = np.argsort(curmax)
        # order ascending: smallest curmax first; skip selected
        pick = None
        for o in order:
            if o not in selset:
                pick = o; break
        # occasional random among top-8 to diversify
        if rng.random() < 0.35:
            cands = [o for o in order if o not in selset][:8]
            pick = int(rng.choice(np.array(cands)))
        sel.append(pick); selset.add(pick)
        curmax = np.maximum(curmax, Gp[pick])
    child = pool[np.array(sel)]
    # small shake
    child = child + 0.006 * rng.standard_normal(child.shape)
    child /= np.linalg.norm(child, axis=1, keepdims=True)
    return child

def solve():
    t0 = time.time()
    LIMIT = 168.0
    rng = np.random.default_rng(7)
    n, d = 82, 14
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal((n, d))
    if xb.shape != (n, d):
        xb = rng.standard_normal((n, d))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    if np.any(~np.isfinite(xb)):
        xb = rng.standard_normal((n, d)); xb /= np.linalg.norm(xb, axis=1, keepdims=True)

    archive = []  # list of (val, X, spec)
    def archive_add(X):
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        v = max_cos(X)
        if not np.isfinite(v):
            return None
        sp, hist, _ = fingerprint(X)
        # diversity: reject near-duplicates (val close and spec close)
        for (vv, XX, ss) in archive:
            if abs(vv - v) < 8e-5 and spec_dist(ss, sp) < 0.02:
                if v >= vv:
                    return None
        archive.append((v, X.copy(), sp))
        archive.sort(key=lambda t: t[0])
        while len(archive) > 6:
            archive.pop()
        return v

    # initial jam of incumbent briefly
    if time.time() - t0 < LIMIT - 40:
        xb = lp_polish(xb, deadline=time.time() + 1.5)
    best_val = max_cos(xb)
    X_best = xb.copy()
    archive_add(X_best)

    # ---- bottom-up chains from smaller N ----
    small_seeds = [(76, 'greedy'), (79, 'greedy'), (81, 'greedy'), (78, 'random')]
    for (N0, kind) in small_seeds:
        if time.time() - t0 > LIMIT - 45:
            break
        t_seed = time.time()
        if kind == 'greedy':
            X0 = greedy_init(N0, d, rng, cand=4000)
        else:
            X0 = rng.standard_normal((N0, d)); X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        # short smooth quench (cap ~20s per seed enforced by small steps)
        X0 = quench_lse(X0, steps=180, lr0=0.04, lr1=0.004, b0=25.0, b1=250.0)
        if time.time() - t0 >= LIMIT - 40:
            break
        X0 = lp_polish(X0, deadline=min(time.time() + 1.2, t0 + LIMIT - 35))
        # grow to 82
        Xg = X0
        while Xg.shape[0] < n:
            if time.time() - t0 > LIMIT - 35:
                break
            if time.time() - t_seed > 20:
                break  # no more than ~20s on unpromising seed
            vnew = best_hole_insertion(Xg, rng, n_cand=200000, chunk=25000)
            Xg = np.vstack([Xg, vnew[None, :]])
            Xg = quench_lse(Xg, steps=200, lr0=0.03, lr1=0.003, b0=30.0, b1=300.0)
            if time.time() - t0 > LIMIT - 33:
                break
            Xg = lp_polish(Xg, deadline=min(time.time() + 1.0, t0 + LIMIT - 30))
            if time.time() - t_seed > 20:
                # quick check: if far from best, abandon
                vg = max_cos(Xg)
                if vg > best_val + 0.004:
                    break
        if Xg.shape[0] == n:
            vg = max_cos(Xg)
            archive_add(Xg)
            if vg < best_val:
                best_val = vg; X_best = Xg.copy()
        # keep time for other phases

    # ---- top-down chains from larger N ----
    for Nbig in [84, 83]:
        if time.time() - t0 > LIMIT - 30:
            break
        t_seed = time.time()
        base = X_best.copy()
        # add points greedily
        need = Nbig - n
        ext = base
        for _ in range(need):
            vnew = best_hole_insertion(ext, rng, n_cand=60000, chunk=20000)
            ext = np.vstack([ext, vnew[None, :]])
        ext = quench_lse(ext, steps=200, lr0=0.03, lr1=0.003, b0=30.0, b1=300.0)
        ext = lp_polish(ext, deadline=min(time.time() + 1.0, t0 + LIMIT - 25))
        # shrink one by one with best-deletion
        while ext.shape[0] > n:
            if time.time() - t0 > LIMIT - 25 or time.time() - t_seed > 20:
                break
            r = deletion_pick(ext)
            ext = np.delete(ext, r, axis=0)
            ext = quench_lse(ext, steps=200, lr0=0.03, lr1=0.003, b0=30.0, b1=300.0)
            ext = lp_polish(ext, deadline=min(time.time() + 1.0, t0 + LIMIT - 22))
        if ext.shape[0] == n:
            ve = max_cos(ext)
            archive_add(ext)
            if ve < best_val:
                best_val = ve; X_best = ext.copy()

    # ---- crossover of distant survivors ----
    if len(archive) >= 2 and time.time() - t0 < LIMIT - 20:
        # pick most distant pair by spec
        bi = bj = -1; bd = -1
        for i in range(len(archive)):
            for j in range(i + 1, len(archive)):
                dd = spec_dist(archive[i][2], archive[j][2]) + abs(archive[i][0] - archive[j][0]) * 50.0
                if dd > bd:
                    bd = dd; bi = i; bj = j
        for _ in range(2):
            if time.time() - t0 > LIMIT - 15:
                break
            A = archive[bi][1]; B = archive[bj][1]
            child = recombine(A, B, rng)
            child = quench_lse(child, steps=250, lr0=0.03, lr1=0.002, b0=30.0, b1=350.0)
            child = lp_polish(child, deadline=min(time.time() + 1.5, t0 + LIMIT - 10))
            vc = max_cos(child)
            archive_add(child)
            if vc < best_val:
                best_val = vc; X_best = child.copy()

    # ---- final basin-hop squeeze on best with hole reinsert of rattlers ----
    it = 0
    while time.time() - t0 < LIMIT - 6:
        it += 1
        if time.time() - t0 > LIMIT - 6:
            break
        cnt, _ = contact_counts(X_best, tol=1.5e-3)
        order_low = np.argsort(cnt, kind="stable")
        X = X_best.copy()
        if it % 3 == 0 and len(archive) >= 2:
            # recombine best with random archive member
            j = int(rng.integers(0, len(archive)))
            X = recombine(X_best, archive[j][1], rng)
        else:
            nv = int(rng.integers(2, 5))
            victims = order_low[:nv]
            keep = np.ones(n, bool); keep[victims] = False
            fixed = X[keep]
            for vix in victims:
                X[vix] = best_hole_insertion(fixed, rng, n_cand=30000, chunk=15000)
                fixed = np.vstack([fixed, X[vix][None, :]])
            X += (0.008 if it % 2 == 0 else 0.02) * rng.standard_normal((n, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        Xq = quench_lse(X, steps=220, lr0=0.03, lr1=0.0025, b0=35.0, b1=320.0)
        vq = max_cos(Xq)
        if vq < best_val + 0.0015:
            Xp = lp_polish(Xq, deadline=min(time.time() + 1.2, t0 + LIMIT - 3))
            vp = max_cos(Xp)
            archive_add(Xp)
            if vp < best_val - 1e-12:
                X_best = Xp.copy(); best_val = vp
            elif vq < best_val - 1e-12:
                X_best = Xq.copy(); best_val = vq
        if it > 12:
            break

    if time.time() - t0 < LIMIT - 1.5:
        X_best = lp_polish(X_best, deadline=t0 + LIMIT - 0.8)
    X_best = X_best / np.linalg.norm(X_best, axis=1, keepdims=True)
    # tiny shrink guard (no-op, ensures finite)
    X_best = np.where(np.isfinite(X_best), X_best, rng.standard_normal((n, d)))
    X_best = X_best / np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best
