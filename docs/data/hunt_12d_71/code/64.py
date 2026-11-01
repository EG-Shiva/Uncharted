import time
import numpy as np

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lse_quench(X, T0, T1, steps, lr, mu, track_best=True, deadline=None):
    n, d = X.shape
    V = np.zeros_like(X)
    lt0 = np.log(T0); lt1 = np.log(T1)
    best_m = _max_cos(X)
    best_X = X.copy()
    for it in range(steps):
        if deadline is not None and (it % 200 == 0) and time.time() > deadline:
            break
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
            mm = _max_cos(X)
            if mm < best_m:
                best_m = mm
                best_X = X.copy()
    if track_best:
        mm = _max_cos(X)
        if mm < best_m:
            best_m = mm
            best_X = X.copy()
        return best_X, best_m
    return X, _max_cos(X)

def _refine_single(s, R, steps, T, lr, mu=0.8):
    s = s / (np.linalg.norm(s) + 1e-30)
    v = np.zeros_like(s)
    for _ in range(steps):
        g = R @ s
        m = float(g.max())
        e = np.exp(np.clip((g - m) / T, -50, 0.0))
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
    k = min(4, n_sample)
    idx = np.argpartition(sc, k - 1)[:k]
    best_s = None
    best_v = 1e100
    for j in idx:
        s0 = Z[j].copy()
        s1 = _refine_single(s0, R, 120, 0.005, 0.12)
        s1 = _refine_single(s1, R, 120, 0.0015, 0.07)
        v = float((R @ s1).max())
        if v < best_v:
            best_v = v
            best_s = s1
    return best_s / (np.linalg.norm(best_s) + 1e-30)

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G

def _tight_polish(X, steps, T, lr, mu=0.85, deadline=None):
    n, d = X.shape
    V = np.zeros_like(X)
    best_m = _max_cos(X)
    best_X = X.copy()
    for it in range(steps):
        if deadline is not None and (it % 200 == 0) and time.time() > deadline:
            break
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
        if it % 40 == 0 or it == steps - 1:
            mm = _max_cos(X)
            if mm < best_m:
                best_m = mm
                best_X = X.copy()
    mm = _max_cos(X)
    if mm < best_m:
        return X, mm
    return best_X, best_m

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    SEARCH_END = HARD_END - 22.0
    rng = np.random.default_rng(7)
    n, d = 71, 12
    bench = None
    bench_score = 1e100
    for fn in ("best.npy", "record.npy"):
        try:
            A = _norm_rows(np.asarray(np.load(fn), dtype=float))
            if A.shape == (n, d):
                s = _max_cos(A)
                if s < bench_score:
                    bench_score = float(s)
                    bench = A.copy()
                if fn == "best.npy" and bench is not None:
                    break
        except Exception:
            continue
    if bench is None:
        bench = _norm_rows(rng.standard_normal((n, d)))
        bench_score = _max_cos(bench)
    bestX = bench.copy()
    best_score = float(bench_score)
    elites = []
    def add_elite(X, s=None):
        nonlocal bestX, best_score
        X = _norm_rows(X)
        if s is None:
            s = _max_cos(X)
        # keep archive small
        elites.append((float(s), X.copy()))
        elites.sort(key=lambda e: e[0])
        del elites[8:]
        if s < best_score:
            best_score = float(s)
            bestX = X.copy()
    add_elite(bestX, best_score)

    # Phase A: clean-slate quenches with varied schedules
    trial = 0
    while time.time() < SEARCH_END - 45.0 and trial < 60:
        rs = np.random.default_rng(1000 + trial)
        # every 6th trial: lattice-flavoured start for new basin
        if trial % 6 == 5:
            # random D12-like integer shell sample
            pool = []
            tries = 0
            while len(pool) < n and tries < 8000:
                tries += 1
                v = rs.integers(-2, 3, size=d)
                if int(v.sum()) % 2 != 0:
                    continue
                n2 = int((v * v).sum())
                if n2 not in (4, 6, 8):
                    continue
                pool.append(v.astype(float) / np.sqrt(n2))
            if len(pool) >= n:
                X = _norm_rows(np.stack(pool[:n]) + 0.15 * rs.standard_normal((n, d)))
            else:
                X = _norm_rows(rs.standard_normal((n, d)))
        else:
            X = _norm_rows(rs.standard_normal((n, d)))
        # stage 1 spread
        T0 = float(rs.uniform(0.02, 0.05))
        X, _ = _lse_quench(X, T0, 0.006, 300, float(rs.uniform(0.10, 0.18)), 0.85, track_best=False, deadline=SEARCH_END)
        # stage 2 mid
        X, _ = _lse_quench(X, 0.008, 0.002, 350, float(rs.uniform(0.08, 0.13)), 0.87, track_best=False, deadline=SEARCH_END)
        # stage 3 tight with tracking
        T1 = float(rs.uniform(0.0004, 0.0009))
        X, sm = _lse_quench(X, 0.002, T1, 450, float(rs.uniform(0.05, 0.09)), 0.88, track_best=True, deadline=SEARCH_END)
        add_elite(X, sm)
        trial += 1
        if time.time() > SEARCH_END - 45.0:
            break

    # Phase B: defect-driven basin hopping
    it = 0
    while time.time() < SEARCH_END and it < 400:
        it += 1
        # pick elite: bias to best
        if len(elites) == 0:
            base = bestX
        elif rng.random() < 0.55:
            base = elites[0][1]
        else:
            base = elites[int(rng.integers(0, len(elites)))][1]
        Xc = base.copy()
        r = float(rng.random())
        rs = np.random.default_rng(5000 + it)
        if r < 0.35:
            # global kick
            sigma = float(rs.uniform(0.03, 0.12))
            Nz = rs.standard_normal(size=Xc.shape) * sigma
            Nz -= Xc * np.sum(Nz * Xc, axis=1, keepdims=True)
            try:
                Qr, _ = np.linalg.qr(rs.standard_normal((d, d)))
                Xc = _norm_rows(Xc + Nz) @ Qr
            except Exception:
                Xc = _norm_rows(Xc + Nz)
            Xc, sm = _lse_quench(Xc, 0.004, 0.0006, 400, 0.08, 0.87, track_best=True, deadline=SEARCH_END)
            add_elite(Xc, sm)
        elif r < 0.68:
            # targeted kick on worst points
            rm, _ = _row_max(Xc)
            k = int(rs.integers(4, 11))
            worst = np.argsort(-rm)[:k]
            sigma = float(rs.uniform(0.15, 0.38))
            for idx_ in worst:
                z = rs.standard_normal(d) * sigma
                z -= Xc[idx_] * float(Xc[idx_] @ z)
                Xc[idx_] = Xc[idx_] + z
            Xc = _norm_rows(Xc)
            Xc, sm = _lse_quench(Xc, 0.003, 0.0005, 400, 0.08, 0.87, track_best=True, deadline=SEARCH_END)
            add_elite(Xc, sm)
        else:
            # delete worst 1-2 and reinsert into holes
            rm, _ = _row_max(Xc)
            kdel = int(rs.integers(1, 3))
            worst = np.argsort(-rm)[:kdel]
            keep = [i for i in range(n) if i not in set(worst.tolist())]
            R = Xc[keep]
            for _k in range(kdel):
                h = _find_hole(R, rs, n_sample=3500)
                R = np.vstack([R, h[None, :]])
            Xc = _norm_rows(R)
            Xc, sm = _lse_quench(Xc, 0.004, 0.0006, 350, 0.09, 0.86, track_best=True, deadline=SEARCH_END)
            # extra tight
            Xc, sm = _lse_quench(Xc, 0.001, 0.0004, 250, 0.06, 0.88, track_best=True, deadline=SEARCH_END)
            add_elite(Xc, sm)
        # occasional crossover between two elites to mix basins
        if it % 12 == 0 and len(elites) >= 2 and time.time() < SEARCH_END - 5:
            A = elites[0][1]; B = elites[1][1]
            try:
                C = A @ B.T
                flat = np.argsort(-C.ravel())
                used_r = np.zeros(n, dtype=bool); used_c = np.zeros(n, dtype=bool)
                perm = np.zeros(n, dtype=int)
                cnt = 0
                for f in flat:
                    ii = f // n; jj = f % n
                    if (not used_r[ii]) and (not used_c[jj]):
                        perm[ii] = jj
                        used_r[ii] = True; used_c[jj] = True
                        cnt += 1
                        if cnt >= n:
                            break
                Bp = B[perm]
                M = Bp.T @ A
                U, _, Vt = np.linalg.svd(M)
                Qp = U @ Vt
                Ba = Bp @ Qp
                mask = rs.random(n) < 0.5
                Ch = np.empty_like(A)
                Ch[mask] = A[mask]; Ch[~mask] = Ba[~mask]
                Nz = rs.standard_normal(size=Ch.shape) * 0.02
                Nz -= Ch * np.sum(Nz * Ch, axis=1, keepdims=True)
                Ch = _norm_rows(Ch + Nz)
                Ch, sm = _lse_quench(Ch, 0.003, 0.0005, 350, 0.08, 0.87, track_best=True, deadline=SEARCH_END)
                add_elite(Ch, sm)
            except Exception:
                pass

    # Phase C: final polish of top elites
    elites_sorted = sorted(elites, key=lambda e: e[0])[:3]
    # ensure bestX included
    cands = []
    for s, X in elites_sorted:
        cands.append(X.copy())
    if len(cands) == 0:
        cands = [bestX.copy()]
    for Xc in cands:
        if time.time() > HARD_END - 4.0:
            break
        Xp, sp = _tight_polish(Xc.copy(), 700, 0.0006, 0.06, 0.88, deadline=HARD_END - 2.0)
        add_elite(Xp, sp)
        if time.time() > HARD_END - 4.0:
            break
        Xp2, sp2 = _tight_polish(Xp.copy(), 700, 0.0003, 0.04, 0.90, deadline=HARD_END - 2.0)
        add_elite(Xp2, sp2)
    # safety: never regress
    if _max_cos(bestX) > bench_score and time.time() < HARD_END:
        bestX = bench.copy()
    return _norm_rows(bestX)
