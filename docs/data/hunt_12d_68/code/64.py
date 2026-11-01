import time
import numpy as np
from scipy.optimize import minimize, linprog, linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.special import ndtri

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _fp_top(X, k=200):
    n = X.shape[0]
    G = X @ X.T
    iu = np.triu_indices(n, 1)
    g = G[iu]
    idx = np.argsort(-g)[:k]
    w = np.sort(g[idx])[::-1]
    # per-point max sorted
    G2 = X @ X.T
    np.fill_diagonal(G2, -2.0)
    pm = np.sort(G2.max(axis=1))[::-1]
    return w, pm

def _fp_dist(a, b):
    wa, pa = a
    wb, pb = b
    d1 = float(np.mean(np.abs(wa - wb)))
    d2 = float(np.mean(np.abs(pa - pb)))
    return 0.7 * d1 + 0.3 * d2

def _make_fg_pos(n, d, IU0, p, scale):
    M = float(len(IU0[0]))
    def fg(yf):
        Y = yf.reshape(n, d)
        r = np.linalg.norm(Y, axis=1, keepdims=True)
        r = np.maximum(r, 1e-12)
        X = Y / r
        G = X @ X.T
        g = G[IU0]
        pos = g > 0
        u = np.zeros_like(g)
        u[pos] = g[pos] / scale
        # power, guard overflow: cap exponent
        # u<=~2 normally; p<=16 so safe. use plain power
        with np.errstate(over='ignore', invalid='ignore'):
            up = np.zeros_like(g)
            up[pos] = np.power(u[pos], p)
            F = float(np.sum(up) / M)
            # grad weights
            w = np.zeros_like(g)
            # p*u^(p-1)/scale/M
            upp = np.power(u[pos], p - 1) if p >= 1 else np.ones_like(u[pos])
            w[pos] = (p / scale / M) * upp
            w[~np.isfinite(w)] = 0.0
            if not np.isfinite(F):
                F = 1e100
        P = np.zeros((n, n))
        P[IU0] = w
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / r
        GY[~np.isfinite(GY)] = 0.0
        return F, GY.ravel()
    return fg

def _make_fg_lse(n, d, IU0, beta):
    def fg(yf):
        Y = yf.reshape(n, d)
        r = np.linalg.norm(Y, axis=1, keepdims=True)
        r = np.maximum(r, 1e-12)
        X = Y / r
        G = X @ X.T
        g = G[IU0]
        m = float(np.max(g))
        e = np.exp(beta * (g - m))
        S = float(np.sum(e)) + 1e-300
        F = m + np.log(S) / beta
        w = e / S
        P = np.zeros((n, n))
        P[IU0] = w
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / r
        return F, GY.ravel()
    return fg

def _run_fg(X0, fg, maxiter):
    n, d = X0.shape
    y0 = (X0 / np.linalg.norm(X0, axis=1, keepdims=True)).ravel().copy()
    try:
        res = minimize(fg, y0, jac=True, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-7,
                                'maxfun': maxiter * 2, 'maxcor': 12})
        Y = res.x.reshape(n, d)
        Y /= (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-300)
        return Y
    except Exception:
        return X0.copy()

def _vdc(i, base):
    f = 0.0
    inv = 1.0 / base
    b = inv
    while i > 0:
        f += (i % base) * b
        i //= base
        b *= inv
    return f

def _halton_init(n, d, offset, rng):
    primes = [2,3,5,7,11,13,17,19,23,29,31,37][:d]
    U = np.zeros((n, d))
    for r in range(n):
        idx = offset + r + 1
        for c, p in enumerate(primes):
            U[r, c] = _vdc(idx, p)
    # jitter to avoid exact 0/1
    U = np.clip(U, 1e-9, 1 - 1e-9)
    Z = ndtri(U)
    # random rotation to diversify
    A = rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(A)
    Z = Z @ Q
    Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-300)
    return Z

def _align_rotate(A, B):
    # return B rotated to best align to A (orthogonal Procrustes, no perm)
    M = B.T @ A
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return B @ R

def _recombine(A, B, rng):
    n, d = A.shape
    Bal = B.copy()
    perm = np.arange(n)
    # 2 iterations of match + rotate
    for _ in range(3):
        D = A @ Bal.T
        _, perm = linear_sum_assignment(-D)
        Bm = Bal[perm]
        M = Bm.T @ A
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        Bal = Bal @ R
    D = A @ Bal.T
    _, perm = linear_sum_assignment(-D)
    Bm = Bal[perm]
    # two children
    mask = rng.random(n) < 0.5
    C1 = A.copy()
    C1[mask] = Bm[mask]
    # average + noise
    C2 = (A + Bm) * 0.5
    # small tangent noise
    N = rng.standard_normal((n, d)) * 0.01
    C2 = C2 + N
    for C in (C1, C2):
        C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-300)
    return C1, C2

def _lp_polish(X, deadline, r0=8e-4, cap_p=1600, max_solves=120):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = r0
    iu = np.triu_indices(n, 1)
    solves = 0
    while time.time() < deadline and radius > 1e-14 and solves < max_solves:
        if time.time() > deadline - 0.4:
            break
        c = (X @ X.T)[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        csel = c[sel]
        if p > cap_p:
            od = np.argsort(-csel)[:cap_p]
            I, J = I[od], J[od]
            csel = csel[od]
            p = cap_p
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        solves += 1
        if res.status != 0:
            radius *= 0.5
            continue
        step = res.x[:-1].reshape(n, d)
        y = X + step
        y /= (np.linalg.norm(y, axis=1, keepdims=True) + 1e-300)
        v = _max_cos(y)
        if v < best - 1e-15:
            X, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            improved = False
            for sh in (0.3, 0.1):
                yy = X + step * sh
                yy /= (np.linalg.norm(yy, axis=1, keepdims=True) + 1e-300)
                vv = _max_cos(yy)
                if vv < best:
                    X, best = yy, vv
                    improved = True
                    break
            if not improved:
                radius *= 0.5
    return X

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    rng = np.random.default_rng(20261031)
    n, d = 68, 12
    IU0 = np.triu_indices(n, 1)

    def quick_anneal(X, budget=True):
        # p=8, p=16, lse 120
        scale = max(0.25, _max_cos(X))
        X = _run_fg(X, _make_fg_pos(n, d, IU0, 8, scale), 60)
        if time.time() > T_END - 30:
            return X
        scale = max(0.22, _max_cos(X))
        X = _run_fg(X, _make_fg_pos(n, d, IU0, 16, scale), 60)
        if time.time() > T_END - 25:
            return X
        X = _run_fg(X, _make_fg_lse(n, d, IU0, 120.0), 80)
        return X

    def long_anneal(X):
        # emulate p=32,64,128 via increasing beta + one more pos power
        scale = max(0.22, _max_cos(X))
        # p=32 with guard (may be heavy) -> use LSE 200 instead for stability
        X = _run_fg(X, _make_fg_lse(n, d, IU0, 220.0), 100)
        if time.time() > T_END - 22:
            return X
        X = _run_fg(X, _make_fg_lse(n, d, IU0, 450.0), 120)
        if time.time() > T_END - 18:
            return X
        X = _run_fg(X, _make_fg_lse(n, d, IU0, 800.0), 120)
        return X

    # ---- incumbent-free population ----
    pop = []
    # 3 pure random
    for _ in range(3):
        X = rng.standard_normal((n, d))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        pop.append(X)
    # 2 antipodal (34 + opposites)
    for _ in range(2):
        H = rng.standard_normal((34, d))
        H /= np.linalg.norm(H, axis=1, keepdims=True)
        X = np.vstack([H, -H])
        # shuffle
        perm = rng.permutation(n)
        pop.append(X[perm])
    # 3 Halton low-discrepancy
    for k, off in enumerate([0, 5000, 12000]):
        pop.append(_halton_init(n, d, off, rng))
    # 2 perturbed lattice (D12 roots sample)
    # build D12 roots: two nonzeros +-1
    roots = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = np.zeros(d)
                    v[i] = si; v[j] = sj
                    v /= np.linalg.norm(v)
                    roots.append(v)
    roots = np.array(roots)
    for _ in range(2):
        idx = rng.choice(len(roots), size=n, replace=False)
        X = roots[idx] + 0.06 * rng.standard_normal((n, d))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        pop.append(X)
    # 2 cube-vertex samples perturbed
    for _ in range(2):
        X = np.sign(rng.standard_normal((n, d)))
        # avoid zeros
        X[X == 0] = 1.0
        X = X + 0.15 * rng.standard_normal((n, d))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        pop.append(X)

    archive = []  # list of dicts
    for X0 in pop:
        if time.time() > T_END - 55:
            break
        X = quick_anneal(X0.copy())
        s = _max_cos(X)
        fp = _fp_top(X)
        archive.append((s, X, fp))

    # dedup by histogram fingerprint
    archive.sort(key=lambda t: t[0])
    kept = []
    for s, X, fp in archive:
        dup = False
        for _, _, kfp in kept:
            if _fp_dist(fp, kfp) < 4e-4:
                dup = True
                break
        if not dup:
            kept.append((s, X, fp))
        if len(kept) >= 8:
            # keep only distinct, but continue to allow better? already sorted so first wins
            pass
    # promote top distinct basins to longer anneals
    kept = kept[:6]
    promoted = []
    for s, X, fp in kept:
        if time.time() > T_END - 40:
            promoted.append((s, X, fp))
            continue
        Xl = long_anneal(X.copy())
        sl = _max_cos(Xl)
        fpl = _fp_top(Xl)
        promoted.append((sl, Xl, fpl))
    promoted.sort(key=lambda t: t[0])

    # re-dedup after long anneal
    kept2 = []
    for s, X, fp in promoted:
        dup = False
        for _, _, kfp in kept2:
            if _fp_dist(fp, kfp) < 4e-4:
                dup = True
                break
        if not dup:
            kept2.append((s, X, fp))

    # recombine distinct basins after optimal alignment
    children = []
    if len(kept2) >= 2 and time.time() < T_END - 30:
        pairs = [(0, 1), (0, 2), (1, 2), (0, 3)] if len(kept2) >= 4 else [(0, 1)]
        for a, b in pairs:
            if a >= len(kept2) or b >= len(kept2):
                continue
            if time.time() > T_END - 28:
                break
            A = kept2[a][1]; B = kept2[b][1]
            try:
                C1, C2 = _recombine(A, B, rng)
            except Exception:
                continue
            for C in (C1, C2):
                if time.time() > T_END - 26:
                    break
                Cc = quick_anneal(C)
                # short extra high-beta
                Cc = _run_fg(Cc, _make_fg_lse(n, d, IU0, 400.0), 60)
                sc = _max_cos(Cc)
                fpc = _fp_top(Cc)
                # discard near-duplicates even if good
                dup = False
                for _, _, kfp in kept2:
                    if _fp_dist(fpc, kfp) < 4e-4:
                        dup = True
                        break
                if not dup:
                    children.append((sc, Cc, fpc))
                else:
                    # keep if strictly better than all
                    best_so = min(t[0] for t in kept2)
                    if sc < best_so - 1e-9:
                        children.append((sc, Cc, fpc))

    all_cand = kept2 + children
    # also include pre-promotion quick versions that might be diverse
    for s, X, fp in kept:
        all_cand.append((s, X, fp))
    all_cand.sort(key=lambda t: t[0])
    best_found = None
    best_s = 1e9
    if len(all_cand) > 0:
        best_s, best_found, _ = all_cand[0]

    # final comparison only: load records (not used as seeds)
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (n, d):
                a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-300)
                if np.all(np.isfinite(a)):
                    s = _max_cos(a)
                    if s < best_s:
                        best_s = s
                        best_found = a.copy()
        except Exception:
            pass
    if best_found is None:
        best_found = rng.standard_normal((n, d))
        best_found /= np.linalg.norm(best_found, axis=1, keepdims=True)
        best_s = _max_cos(best_found)

    # final squeeze of overall best with remaining time
    rem = T_END - time.time()
    if rem > 4.0:
        best_found = _lp_polish(best_found, T_END - 0.5, r0=8e-4, cap_p=1600, max_solves=150)
        best_s = _max_cos(best_found)
    best_found = best_found / (np.linalg.norm(best_found, axis=1, keepdims=True) + 1e-300)
    # tiny shrink guard
    return best_found
