import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _anneal_riesz(X, s, steps, lr, mu, track_best=True):
    n, d = X.shape
    V = np.zeros_like(X)
    best_m = _max_cos(X)
    best_X = X.copy()
    for it in range(steps):
        G = X @ X.T
        np.fill_diagonal(G, -1.0)
        D = 2.0 - 2.0 * G
        np.clip(D, 1e-12, None, out=D)
        logD = np.log(D)
        logE = (-0.5 * float(s)) * logD
        np.fill_diagonal(logE, -1e100)
        ml = float(logE.max())
        E = np.exp(np.clip(logE - ml, -50, 0.0))
        np.fill_diagonal(E, 0.0)
        W = (float(s) * E) / D
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        rms = float(np.sqrt(np.mean(Gt * Gt))) + 1e-30
        Gn = Gt / rms
        V = mu * V + (1.0 - mu) * Gn
        X = X - lr * V
        X = _norm_rows(X)
        if track_best and (it % 25 == 0 or it == steps - 1):
            m = _max_cos(X)
            if m < best_m:
                best_m = m
                best_X = X.copy()
    if track_best:
        m = _max_cos(X)
        if m < best_m:
            best_m = m
            best_X = X.copy()
        return best_X, best_m
    return X, _max_cos(X)

def _anneal_lse(X, T0, T1, steps, lr, mu, track_best=True):
    n, d = X.shape
    V = np.zeros_like(X)
    lt0 = np.log(T0); lt1 = np.log(T1)
    best_m = _max_cos(X)
    best_X = X.copy()
    for it in range(steps):
        frac = it / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -1e100)
        m = float(G.max())
        Q = (G - m) / T
        np.clip(Q, -50, 0, out=Q)
        E = np.exp(Q)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum()) + 1e-30
        Grad = (E @ X) / S
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        rms = float(np.sqrt(np.mean(Gt * Gt))) + 1e-30
        Gn = Gt / rms
        V = mu * V + (1.0 - mu) * Gn
        X = X - lr * V
        X = _norm_rows(X)
        if track_best and (it % 50 == 0 or it == steps - 1):
            # m from before step approx; recompute
            mm = _max_cos(X)
            if mm < best_m:
                best_m = mm
                best_X = X.copy()
    if track_best:
        mm = _max_cos(X)
        if mm < best_m:
            best_X = X.copy()
            best_m = mm
        return best_X, best_m
    return X, _max_cos(X)

def _refine_single(s, R, steps, T, lr, mu=0.8):
    s = s / (np.linalg.norm(s) + 1e-30)
    v = np.zeros_like(s)
    for _ in range(steps):
        g = R @ s
        m = float(g.max())
        e = np.exp(np.clip((g - m) / T, -50, 0))
        den = float(e.sum()) + 1e-30
        Grad = (R.T @ e) / den
        Gt = Grad - float(Grad @ s) * s
        v = mu * v + Gt
        s = s - lr * v
        s = s / (np.linalg.norm(s) + 1e-30)
    return s

def _find_hole(R, rng, n_sample=4000):
    d = R.shape[1]
    Z = _norm_rows(rng.standard_normal(size=(n_sample, d)))
    S = Z @ R.T
    sc = S.max(axis=1)
    idx = np.argpartition(sc, 3)[:3]
    best_s = None
    best_v = 1e100
    for k in idx:
        s0 = Z[k].copy()
        s1 = _refine_single(s0, R, 120, 0.005, 0.12)
        s1 = _refine_single(s1, R, 120, 0.0015, 0.07)
        v = float((R @ s1).max())
        if v < best_v:
            best_v = v
            best_s = s1
    return best_s / (np.linalg.norm(best_s) + 1e-30)

def _prune_to(X, target, rng=None):
    X = _norm_rows(X.copy())
    while X.shape[0] > target:
        n = X.shape[0]
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        flat = int(np.argmax(G))
        a = flat // n; b = flat % n
        # remaining max without a / without b
        # compute row maxes
        # mask approach
        Gm = G.copy()
        # max without a: max over pairs with neither index == a
        # brute per candidate (only 2 candidates)
        def rem_max(rem):
            M = G.copy()
            M[rem, :] = -2.0
            M[:, rem] = -2.0
            return float(M.max())
        ma = rem_max(a); mb = rem_max(b)
        rem = a if ma < mb else b
        keep = [i for i in range(n) if i != rem]
        X = X[keep]
    return _norm_rows(X)

def _fingerprint(X, k=300):
    n = X.shape[0]
    iu = np.triu_indices(n, 1)
    c = (X @ X.T)[iu]
    s = np.sort(c)[::-1]
    return s[:k].copy()

def _fp_dist(f1, f2):
    return float(np.linalg.norm(f1 - f2))

def _crossover(A, B, rng, noise=2e-3):
    n = A.shape[0]
    C = A @ B.T
    # greedy matching: sort flat descending
    flat = np.argsort(-C.ravel())
    used_r = np.zeros(n, dtype=bool); used_c = np.zeros(n, dtype=bool)
    perm = np.zeros(n, dtype=int)  # row i of A <- col perm[i] of B
    cnt = 0
    for f in flat:
        i = f // n; j = f % n
        if (not used_r[i]) and (not used_c[j]):
            perm[i] = j
            used_r[i] = True; used_c[j] = True
            cnt += 1
            if cnt >= n:
                break
    Bp = B[perm]
    # orthogonal procrustes: Bp @ Q ~= A
    try:
        M = Bp.T @ A
        U, _, Vt = np.linalg.svd(M)
        Q = U @ Vt
        Ba = Bp @ Q
    except Exception:
        Ba = Bp
    mask = np.zeros(n, dtype=bool)
    idx = rng.permutation(n)
    mask[idx[:n // 2]] = True
    Ch = np.empty_like(A)
    Ch[mask] = A[mask]
    Ch[~mask] = Ba[~mask]
    if noise > 0:
        N = rng.standard_normal(size=Ch.shape) * noise
        N -= Ch * np.sum(N * Ch, axis=1, keepdims=True)
        Ch = _norm_rows(Ch + N)
    else:
        Ch = _norm_rows(Ch)
    return Ch

def _lp_polish(X0, deadline, max_it=8, cap_p=900):
    X = _norm_rows(X0.copy())
    n, d = X.shape
    iu = np.triu_indices(n, 1)
    best = _max_cos(X)
    bestX = X.copy()
    radius = 1.0e-3
    nv = n * d + 1
    N = n * d
    for _ in range(max_it):
        if time.time() > deadline:
            break
        G = X @ X.T
        cuv = G[iu]
        m = float(cuv.max())
        thr = max(20.0 * radius, 1e-9)
        sel = cuv > (m - thr)
        if int(np.count_nonzero(sel)) == 0:
            break
        I = iu[0][sel]; J = iu[1][sel]; cs = cuv[sel]
        if len(I) > cap_p:
            kk = np.argpartition(-cs, cap_p)[:cap_p]
            I = I[kk]; J = J[kk]; cs = cs[kk]
        p = len(I)
        cmat = np.empty((p, 2 * d + 1), dtype=np.int64)
        cmat[:, :d] = I[:, None] * d + np.arange(d)
        cmat[:, d:2 * d] = J[:, None] * d + np.arange(d)
        cmat[:, 2 * d] = N
        vmat = np.empty((p, 2 * d + 1), dtype=float)
        vmat[:, :d] = X[J]
        vmat[:, d:2 * d] = X[I]
        vmat[:, 2 * d] = -1.0
        try:
            A_ub = coo_matrix((vmat.ravel(), (np.repeat(np.arange(p), 2 * d + 1), cmat.ravel())), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(N))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * N + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        dx = res.x[:-1].reshape(n, d)
        improved = False
        for alpha in (1.0, 0.5, 0.25):
            Y = _norm_rows(X + alpha * dx)
            v = _max_cos(Y)
            if v < m - 1e-14:
                X = Y
                if v < best:
                    best = v; bestX = Y.copy()
                radius = min(radius * 1.4, 0.01)
                improved = True
                break
        if not improved:
            radius *= 0.5
            if radius < 2e-6:
                break
    return bestX, best

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    LP_START = HARD_END - 26.0
    rng = np.random.default_rng(0)
    n, d = 71, 12
    bench = None
    bench_score = 1e100
    try:
        bench = _norm_rows(np.asarray(np.load("best.npy"), dtype=float))
        if bench.shape == (71, 12):
            bench_score = _max_cos(bench)
        else:
            bench = None
    except Exception:
        bench = None
    if bench is None:
        try:
            bench = _norm_rows(np.asarray(np.load("record.npy"), dtype=float))
            bench_score = _max_cos(bench)
        except Exception:
            bench = _norm_rows(rng.standard_normal((n, d)))
            bench_score = _max_cos(bench)
    bestX = bench.copy()
    best_score = float(bench_score)
    elites = []  # list of dict(x, s, fp)
    def add_elite(X, s=None):
        nonlocal bestX, best_score
        X = _norm_rows(X)
        if s is None:
            s = _max_cos(X)
        if s < best_score - 1e-15:
            best_score = float(s)
            bestX = X.copy()
        fp = _fingerprint(X)
        elites.append({"x": X.copy(), "s": float(s), "fp": fp})
    # seed elites with bench lightly kicked? keep bench as fallback but not for evolution
    # Phase A: grow 55 -> 71
    n_traj = 7
    for traj in range(n_traj):
        if time.time() > LP_START - 45.0:
            break
        rs = np.random.default_rng(1000 + traj)
        m0 = 55
        X = _norm_rows(rs.standard_normal((m0, d)))
        X, _ = _anneal_riesz(X, 4, 250, 0.08, 0.9, track_best=False)
        X, _ = _anneal_riesz(X, 8, 250, 0.06, 0.9, track_best=False)
        X, _ = _anneal_lse(X, 0.02, 0.005, 250, 0.15, 0.85, track_best=False)
        # grow
        for target in range(m0 + 1, 72):
            if time.time() > LP_START - 40.0:
                break
            h = _find_hole(X, rs, n_sample=3500)
            X = np.vstack([X, h[None, :]])
            X = _norm_rows(X)
            # short quench after insertion
            X, _ = _anneal_lse(X, 0.008, 0.002, 160, 0.12, 0.85, track_best=False)
            if target % 4 == 0:
                X, _ = _anneal_riesz(X, 8 + (traj % 3) * 8, 120, 0.05, 0.9, track_best=False)
        # final chain with varying s/T per traj
        s_pick = [8, 16, 32][traj % 3]
        X, _ = _anneal_riesz(X, s_pick, 250, 0.05, 0.9, track_best=False)
        X, _ = _anneal_lse(X, 0.01, 0.002, 300, 0.12, 0.85, track_best=False)
        X, sm = _anneal_lse(X, 0.002, 0.0005, 300, 0.08, 0.85, track_best=True)
        add_elite(X, sm)
        # large random-rotation kick + requench to create sibling basin
        if time.time() < LP_START - 30.0:
            for ks, sigma in ((0, 0.08), (1, 0.18)):
                Xk = elites[-1]["x"].copy()
                Nz = rs.standard_normal(size=Xk.shape) * sigma
                Nz -= Xk * np.sum(Nz * Xk, axis=1, keepdims=True)
                # random global rotation to diversify orientation
                try:
                    Qr, _ = np.linalg.qr(rs.standard_normal((d, d)))
                    Xk = (Xk + Nz)
                    Xk = _norm_rows(Xk) @ Qr
                except Exception:
                    Xk = _norm_rows(Xk + Nz)
                Xk, _ = _anneal_lse(Xk, 0.005, 0.0008, 300, 0.10, 0.85, track_best=False)
                Xk, sk = _anneal_lse(Xk, 0.001, 0.0005, 200, 0.06, 0.85, track_best=True)
                add_elite(Xk, sk)
                if time.time() > LP_START - 30.0:
                    break
    # Phase B: oversized then prune
    for traj in range(3):
        if time.time() > LP_START - 20.0:
            break
        rs = np.random.default_rng(2000 + traj)
        no = 78 if traj < 2 else 76
        X = _norm_rows(rs.standard_normal((no, d)))
        X, _ = _anneal_riesz(X, 4, 300, 0.08, 0.9, track_best=False)
        X, _ = _anneal_lse(X, 0.015, 0.004, 350, 0.12, 0.85, track_best=False)
        X = _prune_to(X, 71)
        X, _ = _anneal_riesz(X, 16, 250, 0.05, 0.9, track_best=False)
        X, sm = _anneal_lse(X, 0.004, 0.0005, 400, 0.09, 0.85, track_best=True)
        add_elite(X, sm)
    # dedup + rank, keep diverse 16-24
    def dedup_keep(kmax=24, thr=8e-5):
        elites.sort(key=lambda e: e["s"])
        kept = []
        for e in elites:
            ok = True
            for q in kept:
                if _fp_dist(e["fp"], q["fp"]) < thr:
                    ok = False
                    break
            if ok:
                kept.append(e)
            if len(kept) >= kmax:
                break
        # if too few, relax thr
        if len(kept) < 6:
            kept = elites[:min(kmax, len(elites))]
        return kept
    kept = dedup_keep(20, 8e-5)
    # Phase C: crossover of elites
    if len(kept) >= 2 and time.time() < LP_START - 12.0:
        rs = np.random.default_rng(999)
        order = np.argsort([e["s"] for e in kept])
        nchild = 6
        for c in range(nchild):
            if time.time() > LP_START - 10.0:
                break
            i = int(order[c % len(order)])
            j = int(order[(c * 3 + 1) % len(order)])
            if i == j:
                j = (j + 1) % len(order)
            A = kept[i]["x"]; B = kept[j]["x"]
            Ch = _crossover(A, B, rs, noise=3e-3)
            Ch, _ = _anneal_lse(Ch, 0.004, 0.001, 280, 0.10, 0.85, track_best=False)
            Ch, sc = _anneal_lse(Ch, 0.001, 0.0005, 220, 0.06, 0.85, track_best=True)
            add_elite(Ch, sc)
        kept = dedup_keep(22, 5e-5)
    else:
        kept = dedup_keep(22, 5e-5)
    # ensure best overall included
    # Phase D: reserve LP polish only for final distinct elites
    kept.sort(key=lambda e: e["s"])
    # pick up to 5 distinct with good scores
    finalists = kept[:5]
    lp_deadline = HARD_END - 2.0
    per = max(4.0, (lp_deadline - time.time()) / max(1, len(finalists)))
    for e in finalists:
        if time.time() > lp_deadline - 3.0:
            break
        Xp, sp = _lp_polish(e["x"], min(time.time() + per, lp_deadline), max_it=10)
        # light LSE touch after LP
        Xp, _ = _anneal_lse(Xp, 0.0008, 0.0004, 120, 0.04, 0.85, track_best=False)
        sp2 = _max_cos(Xp)
        if sp2 < sp:
            sp = sp2
        else:
            Xp, sp = _lp_polish(e["x"], min(time.time() + 2.0, lp_deadline), max_it=6)
        add_elite(Xp, sp)
        if time.time() > lp_deadline - 3.0:
            break
    # return global best (fallback to bench if nothing better)
    elites.sort(key=lambda e: e["s"])
    if elites and elites[0]["s"] < best_score:
        bestX = elites[0]["x"].copy()
        best_score = elites[0]["s"]
    bestX = _norm_rows(bestX)
    return bestX
