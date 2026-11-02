import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def lp_polish(x, seconds, radius_init=2e-3, cap=2600):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = radius_init
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        ns = int(np.sum(sel))
        if ns == 0:
            break
        if ns > cap:
            k = cap
            idx = np.argpartition(-c, k)[:k]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = sel & mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
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
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def smooth_quench(x, steps=250, beta0=10.0, beta1=110.0, lr0=0.03, lr1=0.003, mu=0.75, seed_off=0):
    n, d = x.shape
    vel = np.zeros_like(x)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        scale = float(W.sum() / n)
        if scale < 1e-12:
            scale = 1.0
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        grad = grad / scale
        rn = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-12
        grad = grad * np.minimum(1.0, 1.0 / rn)
        vel = mu * vel - lr * grad
        vdot = np.sum(vel * x, axis=1, keepdims=True)
        vel = vel - vdot * x
        x = x + vel
        xn = np.linalg.norm(x, axis=1, keepdims=True)
        xn[xn == 0] = 1.0
        x = x / xn
    return x

def greedy_farthest(n_target, d, rng, n_cand=5000):
    X = []
    v = rng.standard_normal(d); v /= np.linalg.norm(v)
    X.append(v)
    while len(X) < n_target:
        cand = rng.standard_normal((n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        M = np.array(X)  # k x d
        dots = cand @ M.T
        s = dots.max(axis=1)
        b = int(np.argmin(s))
        X.append(cand[b])
    return np.array(X)

def refine_candidate(X, v, steps=150, beta=80.0, lr=0.05):
    # locally minimise max dot to fixed set X
    for _ in range(steps):
        dots = X @ v
        m = float(dots.max())
        w = np.exp(beta * (dots - m))
        w = w / (w.sum() + 1e-16)
        g = w @ X  # d,
        g = g - np.dot(g, v) * v
        v = v - lr * g
        nv = np.linalg.norm(v)
        if nv == 0:
            break
        v = v / nv
    return v

def best_insertion(X, rng, total=200000, batch=50000, refine_top=3):
    n, d = X.shape
    best_v = None
    best_s = 1e9
    cands_best = []
    nb = total // batch
    for _ in range(nb):
        cand = rng.standard_normal((batch, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ X.T
        s = dots.max(axis=1)
        # keep top few of this batch
        k = min(refine_top + 2, batch)
        idx = np.argpartition(s, k)[:k]
        for j in idx:
            cands_best.append((float(s[j]), cand[j].copy()))
    cands_best.sort(key=lambda t: t[0])
    # refine top few
    top = cands_best[:refine_top]
    best_v = top[0][1]
    best_s = top[0][0]
    for sc, v in top:
        vr = refine_candidate(X, v.copy(), steps=120, beta=60.0, lr=0.05)
        sr = float(np.max(X @ vr))
        if sr < best_s:
            best_s = sr
            best_v = vr
    return best_v

def worst_removal_index(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    # count contacts near max and per-point max
    per_max = G.max(axis=1)
    cnt = np.sum(G > m - 2e-3, axis=1)
    # score: higher per_max and higher cnt is worse
    score = per_max + 0.002 * cnt
    return int(np.argmax(score))

def gspectrum(X):
    G = X @ X.T
    w = np.linalg.eigvalsh(G)
    w = np.sort(w)
    # normalise
    return w / (np.linalg.norm(w) + 1e-16)

def contact_fingerprint(X, tol=2e-3):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cnt = np.sum(G > m - tol, axis=1)
    return np.sort(cnt.astype(float))

def spec_dist(a, b):
    return float(np.linalg.norm(a - b))

def procrustes_align(A, B):
    # optimal orthogonal Q minimising ||A - B Q|| (same ordering)
    M = B.T @ A
    U, _, Vt = np.linalg.svd(M)
    Q = U @ Vt
    return B @ Q

def crossover(A, B, rng):
    # align B to A then exchange random half
    Ba = procrustes_align(A, B)
    n = A.shape[0]
    # greedy matching after alignment to improve correspondence:
    # match each A point to nearest unused Ba point
    used = np.zeros(n, dtype=bool)
    perm = np.zeros(n, dtype=int)
    S = A @ Ba.T  # higher = closer
    for i in range(n):
        row = S[i].copy()
        row[used] = -2.0
        j = int(np.argmax(row))
        perm[i] = j
        used[j] = True
    Ba_matched = Ba[perm]
    # random mask exchange
    mask = rng.random(n) < 0.5
    if mask.sum() == 0:
        mask[0] = True
    if mask.sum() == n:
        mask[0] = False
    C = A.copy()
    C[mask] = Ba_matched[mask]
    # small jitter to break symmetry
    C = C + rng.standard_normal(C.shape) * 1e-3
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    return C

def solve():
    t0 = time.time()
    LIMIT = 168.0
    rng = np.random.default_rng(12345)
    n, d = 82, 14
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = np.load("record.npy").astype(float)
    if xb.shape != (n, d):
        z = rng.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        xb = z
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    if time.time() - t0 < LIMIT - 30:
        xb = lp_polish(xb, seconds=2.0)
    best_val = max_cos(xb)
    X_best = xb.copy()

    archive = [(best_val, X_best.copy(), gspectrum(X_best), contact_fingerprint(X_best))]

    def try_add_archive(X):
        nonlocal archive
        v = max_cos(X)
        sp = gspectrum(X)
        fp = contact_fingerprint(X)
        # diversity: if spectrum very close to existing, keep only better
        for i, (vv, XX, ss, ff) in enumerate(archive):
            if spec_dist(sp, ss) < 0.02 and float(np.linalg.norm(fp - ff)) < 3.0:
                if v < vv:
                    archive[i] = (v, X.copy(), sp, fp)
                return
        archive.append((v, X.copy(), sp, fp))
        # keep at most 6, prefer low max-cos but keep diversity: sort by val, drop worst if >6
        archive.sort(key=lambda t: t[0])
        if len(archive) > 6:
            archive = archive[:6]

    # ---- bottom-up continuation N=76..81 -> 82 (two seeds, each < ~20s) ----
    for chain, Nstart in enumerate([78, 80, 76]):
        if time.time() - t0 > LIMIT - 55:
            break
        c0 = time.time()
        rs = np.random.default_rng(1000 + chain)
        if chain == 2 and time.time() - t0 > LIMIT - 90:
            break
        X = greedy_farthest(Nstart, d, rs, n_cand=4000)
        X = smooth_quench(X, steps=250, beta0=8.0, beta1=90.0)
        X = lp_polish(X, seconds=0.7)
        cur = Nstart
        aborted = False
        while cur < 82:
            if time.time() - t0 > LIMIT - 50:
                aborted = True
                break
            if time.time() - c0 > 20.0:  # no more than ~20s on unpromising seed
                # only continue if already promising (within 0.01 of best)
                if max_cos(X) > best_val + 0.01:
                    aborted = True
                    break
                if time.time() - c0 > 26.0:
                    aborted = True
                    break
            v = best_insertion(X, rs, total=200000, batch=50000, refine_top=3)
            X = np.vstack([X, v.reshape(1, d)])
            X = smooth_quench(X, steps=220, beta0=8.0, beta1=100.0)
            X = lp_polish(X, seconds=0.7)
            cur += 1
        if aborted:
            continue
        if X.shape[0] == 82:
            vv = max_cos(X)
            try_add_archive(X)
            if vv < best_val:
                best_val = vv
                X_best = X.copy()

    # ---- top-down continuation N=83..86 -> 82 ----
    for chain, Nbig in enumerate([85, 83]):
        if time.time() - t0 > LIMIT - 35:
            break
        c0 = time.time()
        rs = np.random.default_rng(2000 + chain)
        X = greedy_farthest(Nbig, d, rs, n_cand=4000)
        X = smooth_quench(X, steps=250, beta0=8.0, beta1=90.0)
        X = lp_polish(X, seconds=0.7)
        while X.shape[0] > 82:
            if time.time() - t0 > LIMIT - 30 or time.time() - c0 > 20.0:
                break
            j = worst_removal_index(X)
            X = np.delete(X, j, axis=0)
            X = smooth_quench(X, steps=220, beta0=8.0, beta1=100.0)
            X = lp_polish(X, seconds=0.7)
        if X.shape[0] == 82:
            vv = max_cos(X)
            try_add_archive(X)
            if vv < best_val:
                best_val = vv
                X_best = X.copy()

    # ---- crossover distant survivors ----
    archive.sort(key=lambda t: t[0])
    # find distant pairs by spectrum distance
    tries = 0
    while time.time() - t0 < LIMIT - 12 and len(archive) >= 2 and tries < 6:
        # pick pair with max spectrum distance among top survivors
        bi, bj, bd = 0, 1, -1.0
        for a in range(len(archive)):
            for b in range(a + 1, len(archive)):
                dd = spec_dist(archive[a][2], archive[b][2])
                if dd > bd:
                    bd, bi, bj = dd, a, b
        A = archive[bi][1]
        B = archive[bj][1]
        rs = np.random.default_rng(3000 + tries)
        C = crossover(A, B, rs)
        C = smooth_quench(C, steps=300, beta0=9.0, beta1=110.0)
        C = lp_polish(C, seconds=0.9)
        vv = max_cos(C)
        try_add_archive(C)
        if vv < best_val:
            best_val = vv
            X_best = C.copy()
        tries += 1
        # also random global kick of best for extra basin move if time remains
        if time.time() - t0 < LIMIT - 25 and tries < 5:
            rs2 = np.random.default_rng(4000 + tries)
            Y = X_best + rs2.standard_normal((n, d)) * 0.05
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Y = smooth_quench(Y, steps=250, beta0=8.0, beta1=100.0)
            Y = lp_polish(Y, seconds=0.7)
            vy = max_cos(Y)
            try_add_archive(Y)
            if vy < best_val:
                best_val = vy
                X_best = Y.copy()

    rem = LIMIT - (time.time() - t0) - 1.0
    if rem > 1.0:
        X_best = lp_polish(X_best, seconds=min(rem, 8.0), radius_init=7e-4)
    X_best = X_best / np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best
