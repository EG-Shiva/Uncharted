import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return X / n

def solve():
    t0 = time.time()
    deadline = t0 + 170.0
    rng = np.random.default_rng(12345)
    n, d = 138, 8
    iu_all = np.triu_indices(n, 1)

    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = _norm(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.normal(size=(n, d)))
        best_val = _max_cos(best)
    incumb_val = best_val

    # ---------- helpers (closure for speed) ----------
    from scipy.optimize import minimize

    def lse_opt(X0, betas, iters):
        Y = _norm(X0.copy())
        for beta, mi in zip(betas, iters):
            if time.time() > deadline - 15.0:
                break
            def fun(Yf, _b=beta):
                Ym = Yf.reshape(n, d)
                nr = np.sqrt(np.sum(Ym * Ym, axis=1, keepdims=True)) + 1e-30
                X = Ym / nr
                G = X @ X.T
                c = G[iu_all]
                mv = c.max()
                E = np.exp(_b * (c - mv))
                S = E.sum() + 1e-300
                f = float(mv + np.log(S) / _b)
                W = np.zeros((n, n))
                W[iu_all] = E
                W = W + W.T
                GX = (W @ X) / S
                dot = np.sum(GX * X, axis=1, keepdims=True)
                Gt = GX - dot * X
                GY = Gt / nr
                return f, GY.ravel()
            try:
                res = minimize(fun, Y.ravel(), jac=True, method='L-BFGS-B',
                               options={'maxiter': int(mi), 'ftol': 1e-14, 'gtol': 1e-5, 'maxcor': 10})
                Y = _norm(res.x.reshape(n, d))
            except Exception:
                break
        return _norm(Y)

    def lp_polish(X, dl, radius0=2e-3, cap=2500, max_lp=25):
        try:
            from scipy.optimize import linprog
            from scipy.sparse import coo_matrix
        except Exception:
            return X
        X = _norm(X)
        bv = _max_cos(X)
        radius = radius0
        it = 0
        while time.time() < dl and radius > 1e-14 and it < max_lp:
            it += 1
            Gf = X @ X.T
            c = Gf[iu_all]
            m = c.max()
            thr = m - max(20.0 * radius, 5e-4)
            sel = c > thr
            cnt = int(np.count_nonzero(sel))
            if cnt == 0:
                break
            if cnt > cap:
                idx = np.where(sel)[0]
                o = np.argsort(-c[idx])[:cap]
                mk = np.zeros_like(sel, dtype=bool)
                mk[idx[o]] = True
                sel = mk
            I, J = iu_all[0][sel], iu_all[1][sel]
            p = len(I)
            nv = n * d + 1
            try:
                rows = np.repeat(np.arange(p), 2 * d + 1)
                cols = np.concatenate([I[:, None] * d + np.arange(d),
                                       J[:, None] * d + np.arange(d),
                                       np.full((p, 1), n * d)], 1).ravel()
                vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            obj = np.zeros(nv)
            obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = X + res.x[:-1].reshape(n, d)
            y = _norm(y)
            v = _max_cos(y)
            if v < bv:
                X, bv = y, v
                radius = min(radius * 1.5, 0.02)
            else:
                radius *= 0.5
        return X

    def find_voids(X_fixed, k, n_samp=25000):
        # largest voids = sample points minimizing max dot to fixed set
        Y = rng.normal(size=(n_samp, d))
        Y = _norm(Y)
        S = Y @ X_fixed.T
        mc = S.max(axis=1)
        order = np.argsort(mc)
        picked = []
        for idx in order:
            y = Y[idx]
            ok = True
            for q in picked:
                if float(y @ q) > 0.85:
                    ok = False
                    break
            if ok:
                picked.append(y.copy())
                if len(picked) >= k:
                    break
        if len(picked) < k:
            for idx in order:
                if len(picked) >= k:
                    break
                picked.append(Y[idx].copy())
        return np.array(picked)

    def degrees(X, gap=1e-3):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        return np.count_nonzero(G > m - gap, axis=1), m, G

    # ---------- exploration: hub -> void jumps ----------
    explore_end = t0 + 135.0
    trial = 0
    # quick anneal schedules (fast)
    sched_fast = ([120.0, 350.0], [90, 140])
    sched_med = ([100.0, 250.0, 500.0], [80, 120, 140])

    while time.time() < explore_end and trial < 120:
        trial += 1
        deg, mv, G = degrees(best, gap=1e-3)
        # choose k hubs
        if trial % 5 == 0:
            k = 3
        elif trial % 2 == 0:
            k = 2
        else:
            k = 1
        hubs = np.argsort(-deg)[:k]
        # occasionally also move a rattler / low-contact point
        if trial % 7 == 0:
            low = np.argsort(deg)[:1]
            hubs = np.unique(np.concatenate([hubs, low]))
            k = len(hubs)
        mask = np.ones(n, dtype=bool)
        mask[hubs] = False
        X_fixed = best[mask]
        try:
            voids = find_voids(X_fixed, k, n_samp=22000)
        except Exception:
            voids = _norm(rng.normal(size=(k, d)))
        sigma = 0.006 + 0.004 * ((trial * 7) % 5)
        if trial % 11 == 0:
            sigma = 0.025
        X0 = best + sigma * rng.normal(size=(n, d))
        # place hubs at voids with tiny jitter
        X0[hubs] = voids + 0.002 * rng.normal(size=(k, d))
        X0 = _norm(X0)
        if trial % 3 == 0:
            bts, mits = sched_med
        else:
            bts, mits = sched_fast
        try:
            Xj = lse_opt(X0, bts, mits)
            vj = _max_cos(Xj)
        except Exception:
            continue
        if vj < best_val - 2e-9:
            # squeeze promising basin with short LP
            try:
                Xp = lp_polish(Xj, min(deadline - 10.0, time.time() + 4.0), radius0=1.5e-3)
                vp = _max_cos(Xp)
                if vp <= vj:
                    Xj, vj = Xp, vp
            except Exception:
                pass
            if vj < best_val:
                best_val = vj
                best = Xj.copy()
        elif vj < best_val:
            best_val = vj
            best = Xj.copy()
        # every ~20 trials try pure jitter basin without relocation
        if trial % 20 == 0 and time.time() < explore_end - 10:
            try:
                X0b = _norm(best + 0.008 * rng.normal(size=(n, d)))
                Xjb = lse_opt(X0b, [150.0, 400.0], [100, 140])
                vjb = _max_cos(Xjb)
                if vjb < best_val:
                    best_val = vjb
                    best = Xjb.copy()
            except Exception:
                pass

    # ---------- final squeeze ----------
    if time.time() < deadline - 3.0:
        try:
            # one more medium jam from best with tiny kick
            if time.time() < deadline - 20.0:
                X0 = _norm(best + 0.003 * rng.normal(size=(n, d)))
                Xj = lse_opt(X0, [150.0, 400.0, 600.0], [100, 150, 100])
                if _max_cos(Xj) < best_val:
                    best = Xj.copy()
                    best_val = _max_cos(best)
        except Exception:
            pass
        try:
            Xp = lp_polish(best, deadline - 1.0, radius0=1.5e-3, cap=2800, max_lp=40)
            if _max_cos(Xp) <= _max_cos(best):
                best = Xp.copy()
        except Exception:
            pass

    if best is None:
        best = _norm(rng.normal(size=(n, d)))
    # safety: if we somehow got worse than incumbent file, reload it
    try:
        vb = np.load("best.npy").astype(float)
        if vb.shape == (n, d):
            vb = _norm(vb)
            if _max_cos(vb) < _max_cos(best):
                best = vb.copy()
    except Exception:
        pass
    best = _norm(best)
    return best
