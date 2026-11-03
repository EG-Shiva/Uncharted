import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _adam_riesz(x0, steps, p, lr, deadline=1e18):
    x = x0.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    expc = float(p * 0.5 - 1.0)
    best_val = _max_cos(x)
    best_x = x.copy()
    for t in range(1, steps + 1):
        if time.time() > deadline and t > 10:
            break
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        umax = (1.0 + gmax) * 0.5
        if umax <= 1e-12:
            umax = 0.5
        logUmax = np.log(umax)
        U = (1.0 + G) * 0.5
        np.fill_diagonal(U, 1.0)
        logU = np.log(np.maximum(U, 1e-300))
        E = expc * (logU - logUmax)
        np.fill_diagonal(E, -100.0)
        np.clip(E, -50.0, 0.0, out=E)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (W @ x) / S
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1 - b1) * gt
        v *= b2
        v += (1 - b2) * (gt * gt)
        mhat = m / (1 - b1p)
        vhat = v / (1 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if t % 100 == 0 or t == steps:
            val = _max_cos(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
    return best_x, best_val

def _adam_lse(x0, steps, K0, K1, lr, deadline=1e18):
    x = x0.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    la0, la1 = np.log(K0), np.log(K1)
    best_val = _max_cos(x)
    best_x = x.copy()
    for t in range(1, steps + 1):
        if time.time() > deadline and t > 10:
            break
        K = float(np.exp(la0 + (la1 - la0) * t / steps))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(K * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (W @ x) / S
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1 - b1) * gt
        v *= b2
        v += (1 - b2) * (gt * gt)
        mhat = m / (1 - b1p)
        vhat = v / (1 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if t % 100 == 0 or t == steps:
            val = _max_cos(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
    return best_x, best_val

def _lp_polish(x, deadline, r0=2.0e-3, cap=3000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iuI, iuJ]
        mval = float(c.max())
        margin = max(20.0 * radius, 8e-5)
        sel = c > (mval - margin)
        I = iuI[sel]
        J = iuJ[sel]
        csel = c[sel]
        pnum = len(I)
        if pnum == 0:
            radius *= 0.5
            continue
        if pnum > cap:
            part = np.argpartition(-csel, cap)[:cap]
            I = I[part]; J = J[part]; csel = csel[part]
            pnum = cap
        rows = np.repeat(np.arange(pnum), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((pnum, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((pnum, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(pnum, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
        except Exception:
            break
        obj = np.zeros(n * d + 1)
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        val = _max_cos(y)
        if val < best:
            x, best = y, val
            radius = min(radius * 1.4, 0.02)
        else:
            radius *= 0.5
    return x

def _void_point(x, rng, exclude=-1, trials=4000):
    n, d = x.shape
    Z = rng.normal(size=(trials, d))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    S = Z @ x.T
    if exclude >= 0:
        S[:, exclude] = -2.0
    cm = S.max(axis=1)
    return Z[int(np.argmin(cm))]

def _dedupe_refill(child, rng, thresh=0.92):
    child = child / np.linalg.norm(child, axis=1, keepdims=True)
    n = child.shape[0]
    for _ in range(10):
        G = child @ child.T
        np.fill_diagonal(G, -2.0)
        mv = float(G.max())
        if mv < thresh:
            break
        fi = int(np.argmax(G))
        i = fi // n; j = fi % n
        ri = float(G[i].max()); rj = float(G[j].max())
        k = j if rj >= ri else i
        child[k] = _void_point(child, rng, exclude=k, trials=4000)
    return child

def _row_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G

def _align_procrustes(A, B):
    # align B to A via alternating Hungarian + Procrustes
    Q = np.eye(A.shape[1])
    perm = np.arange(A.shape[0])
    for _ in range(3):
        Bq = B @ Q
        S = A @ Bq.T
        r, c = linear_sum_assignment(-S)
        perm = np.empty(A.shape[0], dtype=int)
        perm[r] = c
        Bp = B[perm]
        M = Bp.T @ A
        U, _, Vt = np.linalg.svd(M)
        Q = U @ Vt
    return B @ Q

def _fingerprint(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    tri = G[np.triu_indices(x.shape[0], 1)]
    tri = np.sort(tri)[::-1]
    # decimated spectrum + contact counts
    spec = tri[::512].copy()
    m = float(tri.max())
    c1 = int(np.sum(tri > m - 1e-7))
    c4 = int(np.sum(tri > m - 1e-4))
    return spec, m, c1, c4

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    CENSUS_END = t0 + 88.0
    CROSS_END = t0 + 140.0
    rng0 = np.random.default_rng(0)
    # load incumbent fallback
    x_best = None
    best = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (256, 6) and np.all(np.isfinite(a)):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best:
                    best = v
                    x_best = a.copy()
        except Exception:
            continue
    if x_best is None:
        x_best = rng0.normal(size=(256, 6))
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
        best = _max_cos(x_best)
    incumbent = x_best.copy()
    inc_best = best

    # ---- de novo multi-basin census ----
    N_BASIN = 12
    parents = []
    p_stages = [(8, 220), (32, 220), (128, 220), (512, 220)]
    for b in range(N_BASIN):
        if time.time() > CENSUS_END:
            break
        rng = np.random.default_rng(1000 + b)
        x = rng.normal(size=(256, 6))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        # vary lr slightly for diversity
        lr_r = 0.012 if (b % 3 == 0) else (0.010 if (b % 3 == 1) else 0.014)
        for (p, steps) in p_stages:
            if time.time() > CENSUS_END:
                break
            # fewer steps if late
            st = steps
            if b >= 8:
                st = 160
            try:
                x, _ = _adam_riesz(x, st, p, lr_r, deadline=CENSUS_END)
            except Exception:
                break
        if time.time() > CENSUS_END:
            # quick eval and store
            try:
                v = _max_cos(x)
                parents.append((v, x.copy()))
            except Exception:
                pass
            break
        try:
            x, v = _adam_lse(x, 280, 200.0, 2500.0, 0.008, deadline=CENSUS_END)
        except Exception:
            v = _max_cos(x)
        parents.append((v, x.copy()))
        if v < best:
            best = v
            x_best = x.copy()

    # archive distinct minima (sorted spectra + contact counts) and pick best few
    parents_sorted = sorted(parents, key=lambda t: t[0])
    # dedupe by fingerprint: keep genetically different
    kept = []
    seen = []
    for v, x in parents_sorted:
        try:
            spec, m, c1, c4 = _fingerprint(x)
        except Exception:
            kept.append((v, x))
            continue
        dup = False
        for (sv, ss, sm, s1, s4) in seen:
            if abs(sm - m) < 5e-6 and abs(s1 - c1) < 3 and np.linalg.norm(ss - spec) < 5e-4:
                dup = True
                break
        if not dup:
            kept.append((v, x))
            seen.append((v, spec, m, c1, c4))
        if len(kept) >= 6:
            break
    if len(kept) == 0 and len(parents_sorted) > 0:
        kept = parents_sorted[:4]
    top_census = [x for (v, x) in kept[:4]]

    # ---- crossover: incumbent + top census ----
    cross_rng = np.random.default_rng(777)
    pool = []
    # build pool of distinct parents: incumbent + top census (avoid near-dup of incumbent)
    pool.append(incumbent)
    for x in top_census:
        # skip if essentially identical to incumbent (aligned max diff small)
        try:
            d = _max_cos(np.vstack([np.zeros((1, 6)), np.zeros((1, 6))]))  # dummy
        except Exception:
            pass
        pool.append(x)
        if len(pool) >= 4:
            break
    # pairwise children
    pairs = []
    for a in range(len(pool)):
        for bb in range(a + 1, len(pool)):
            pairs.append((a, bb))
    # limit to 6 pairs
    pairs = pairs[:6]
    for (ai, bi) in pairs:
        if time.time() > CROSS_END:
            break
        A = pool[ai]
        B = pool[bi]
        try:
            Bq = _align_procrustes(A, B)
        except Exception:
            Bq = B
        try:
            ra, _ = _row_max(A)
            rb, _ = _row_max(Bq)
            ia = np.argsort(ra)[:128]
            ib = np.argsort(rb)[:128]
            child = np.vstack([A[ia], Bq[ib]])
            child = _dedupe_refill(child, cross_rng, thresh=0.92)
            # refill check: if child has huge max, void refill already handled
            # repolish: short Riesz + sharp LSE
            child, _ = _adam_riesz(child, 120, 64, 0.010, deadline=CROSS_END)
            if time.time() > CROSS_END:
                v = _max_cos(child)
                if v < best:
                    best = v
                    x_best = child.copy()
                break
            child, v = _adam_lse(child, 320, 300.0, 2000.0, 0.008, deadline=CROSS_END)
            if v < best and np.all(np.isfinite(child)):
                best = v
                x_best = child.copy()
        except Exception:
            continue

    # extra void-refill mutants of best for diversity if time remains
    if time.time() < CROSS_END - 5 and x_best is not None:
        try:
            for _ in range(2):
                if time.time() > CROSS_END:
                    break
                y = x_best.copy()
                G = y @ y.T
                np.fill_diagonal(G, -2.0)
                prow = G.max(axis=1)
                # replace 2 worst with voids then repolish briefly
                order = np.argsort(-prow)[:2]
                for k in order:
                    y[int(k)] = _void_point(y, cross_rng, exclude=int(k), trials=5000)
                y, v = _adam_lse(y, 250, 300.0, 2000.0, 0.008, deadline=CROSS_END)
                if v < best:
                    best = v
                    x_best = y.copy()
        except Exception:
            pass

    # ---- final LP polish of global best ----
    try:
        if time.time() < T_END - 2:
            x_best = _lp_polish(x_best, T_END, r0=2.0e-3, cap=3000)
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    x_best = np.where(np.isfinite(x_best), x_best, 0.0)
    nr = np.linalg.norm(x_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    x_best = x_best / nr
    return x_best
