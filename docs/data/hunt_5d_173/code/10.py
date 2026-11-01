import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return X / n


def _lp_polish(X, deadline, r0=2e-3, cap=3500):
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    radius = r0
    iu0 = np.triu_indices(n, 1)
    try:
        while time.time() < deadline and radius > 1e-14:
            Gfull = X @ X.T
            c = Gfull[iu0]
            m = float(c.max())
            thr = m - max(20.0 * radius, 2e-9)
            sel = c > thr
            if np.count_nonzero(sel) > cap:
                idx = np.where(sel)[0]
                order = np.argsort(-c[idx])[:cap]
                mask = np.zeros_like(sel, dtype=bool)
                mask[idx[order]] = True
                sel = mask
            I = iu0[0][sel]
            J = iu0[1][sel]
            p = len(I)
            if p == 0:
                break
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(),
                               (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            b_ub = -c[sel]
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq,
                              b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                              method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = X + res.x[:-1].reshape(n, d)
            y = _norm(y)
            v = _max_cos(y)
            if v < best:
                X, best = y, v
                radius = min(radius * 1.5, 0.02)
            else:
                radius *= 0.5
    except Exception:
        pass
    return X


def _lbfgs_anneal(Xinit, s_seq, iters_seq, deadline):
    X = _norm(Xinit).copy()
    n, d = X.shape
    lam = 0.05
    for s, maxit in zip(s_seq, iters_seq):
        if time.time() > deadline:
            break
        def fun(flat, s=s):
            Xv = flat.reshape(n, d)
            nr = np.sqrt((Xv * Xv).sum(axis=1, keepdims=True)) + 1e-12
            U = Xv / nr
            G = U @ U.T
            np.fill_diagonal(G, -10.0)
            m = float(G.max())
            E = np.exp(s * (G - m))
            np.fill_diagonal(E, 0.0)
            Z = float(E.sum() * 0.5)
            if Z <= 0:
                Z = 1.0
            val = m + np.log(Z) / s
            P = E / Z * 0.5  # each unordered pair weight sums to 1; full matrix sums to 1? adjust
            # Correction: E.sum()*0.5=Z, P_full=E/(2Z) gives sum 1 over i<j
            # Grad_U = P_full @ U *? need sum_j P_ij U_j with P_ij=E_ij/Z for i!=j
            # Since P above = E/(2Z), multiply by 2:
            GradU = (E / Z) @ U * 0.5  # = (P_pair) @ U where P_pair=E/Z*? check
            # Actually (E/Z)@U gives sum_j E_ij/Z U_j, but E symmetric includes both
            # triangles: for fixed i, sum_j E_ij/Z = 2*sum_{j>i}... so divide by 2 to get pair-normalized
            # The above *0.5 achieves pair normalization. Good.
            # radial projection
            dot = (GradU * U).sum(axis=1, keepdims=True)
            Gt = GradU - dot * U
            Gx = Gt / nr
            # penalty to keep norms ~1
            pen = float(((nr - 1.0) ** 2).mean() * lam)
            Gpen = (2.0 * lam / n * (nr - 1.0) * U) / 1.0
            # note Gpen already in X-space (d/dX of pen)
            Gtot = Gx + Gpen
            return val + pen, Gtot.ravel()
        try:
            res = minimize(fun, X.ravel(), jac=True, method="L-BFGS-B",
                           options={"maxiter": maxit, "ftol": 1e-12, "gtol": 1e-6, "maxcor": 12})
            X = res.x.reshape(n, d)
            X = _norm(X)
        except Exception:
            break
    return X


def _refine_void(Y, v0, s=160.0, iters=200, lr=0.08):
    v = v0.copy()
    nrm = np.linalg.norm(v)
    if nrm < 1e-12:
        return v0
    v /= nrm
    for _ in range(iters):
        dd = Y @ v
        dm = float(dd.max())
        w = np.exp(s * (dd - dm))
        sw = float(w.sum())
        if sw <= 0:
            break
        w /= sw
        g = Y.T @ w
        g = g - (g @ v) * v
        gn = float(np.linalg.norm(g))
        if gn < 1e-12:
            break
        v = v - lr * g
        nrm = float(np.linalg.norm(v))
        if nrm < 1e-12:
            break
        v /= nrm
    return v


def _find_void(Y, rng, ntry=12000):
    S = rng.randn(ntry, 5)
    S = _norm(S)
    dots = S @ Y.T
    md = dots.max(axis=1)
    i0 = int(np.argmin(md))
    v0 = S[i0]
    return _refine_void(Y, v0)


def solve():
    t0 = time.time()
    t_end = t0 + 170.0
    rng = np.random.RandomState(12345)
    try:
        best = np.load("best.npy").astype(float)
    except Exception:
        try:
            best = np.load("record.npy").astype(float)
        except Exception:
            best = rng.randn(173, 5)
    best = _norm(best)
    # in case files have wrong shape, fix
    if best.shape != (173, 5):
        tmp = rng.randn(173, 5)
        m = min(best.shape[0], 173)
        tmp[:m] = best[:m, :5] if best.shape[1] >= 5 else best[:m]
        best = _norm(tmp)
    best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val
    n = 173

    trial = 0
    Ttemp = 0.0015
    while time.time() < t_end - 12.0:
        trial += 1
        X = cur.copy()
        r = rng.rand()
        # periodic fresh random basin for topology change
        if trial % 28 == 0 and rng.rand() < 0.6:
            X = _norm(rng.randn(n, 5))
            kind = "fresh"
        elif r < 0.18:
            sigma = 0.008 + rng.rand() * 0.022
            X = X + rng.randn(n, 5) * sigma
            kind = "small"
        elif r < 0.38:
            k = int(rng.choice([2, 3, 5, 8, 15, 30]))
            idx = rng.choice(n, size=k, replace=False)
            sigma = 0.06 + rng.rand() * 0.18
            X[idx] = X[idx] + rng.randn(k, 5) * sigma
            kind = "subset"
        elif r < 0.54:
            # rattler / low-contact relocation to voids
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            # adaptive threshold: before polish use 1e-3 band
            cnt = (G > m - 1e-3).sum(axis=1)
            order = np.argsort(cnt)
            take = order[:2]
            for li in take:
                mask = np.ones(n, dtype=bool)
                mask[li] = False
                Y = X[mask]
                v = _find_void(Y, rng, ntry=8000)
                X[li] = v
            X = X + rng.randn(n, 5) * 0.006
            kind = "void"
        elif r < 0.70:
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cnt = (G > m - 1e-3).sum(axis=1)
            hi = int(np.argmax(cnt))
            dots = X @ X[hi]
            nn = np.argsort(-dots)[:12]
            X[nn] = X[nn] + rng.randn(len(nn), 5) * (0.06 + rng.rand() * 0.10)
            X = X + rng.randn(n, 5) * 0.004
            kind = "jam"
        elif r < 0.86:
            # contact pair breaking: push max pair apart
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            # pick among top pairs
            iu = np.triu_indices(n, 1)
            c = G[iu]
            topk = np.argsort(-c)[:20]
            pick = topk[rng.randint(len(topk))]
            a = int(iu[0][pick]); b = int(iu[1][pick])
            eps = 0.06 + rng.rand() * 0.08
            da = X[a] - X[b]
            db = -da
            # project to tangent
            da = da - (da @ X[a]) * X[a]
            db = db - (db @ X[b]) * X[b]
            dna = np.linalg.norm(da) + 1e-12
            dnb = np.linalg.norm(db) + 1e-12
            X[a] = X[a] + (da / dna) * eps + rng.randn(5) * 0.01
            X[b] = X[b] + (db / dnb) * eps + rng.randn(5) * 0.01
            # jiggle common neighbours
            com = np.where((G[a] > m - 2e-3) | (G[b] > m - 2e-3))[0]
            if len(com) > 0:
                X[com] = X[com] + rng.randn(len(com), 5) * 0.02
            kind = "break"
        else:
            k = int(rng.randint(1, 4))
            idx = rng.choice(n, size=k, replace=False)
            R = _norm(rng.randn(k, 5))
            X[idx] = R
            X = X + rng.randn(n, 5) * 0.005
            kind = "reinit"

        X = _norm(X)

        # choose annealing schedule
        if kind in ("small",):
            s_seq = [80, 200, 380]
            it_seq = [60, 80, 100]
        elif kind in ("fresh",):
            s_seq = [40, 100, 200, 320, 480]
            it_seq = [80, 80, 100, 120, 140]
        else:
            s_seq = [60, 150, 280, 420]
            it_seq = [70, 80, 100, 110]

        # limit anneal time so many trials fit
        adead = min(time.time() + 4.0, t_end - 12.0)
        X2 = _lbfgs_anneal(X, s_seq, it_seq, adead)
        v2 = _max_cos(X2)

        thresh = min(best_val + 0.004, cur_val + 0.006)
        if v2 < thresh and time.time() < t_end - 12.0:
            X3 = _lp_polish(X2, deadline=min(time.time() + 1.8, t_end - 12.0), r0=2e-3)
            v3 = _max_cos(X3)
        else:
            X3, v3 = X2, v2

        if v3 < best_val - 1e-12:
            best, best_val = X3.copy(), float(v3)
            cur, cur_val = X3.copy(), float(v3)
        elif v3 < cur_val:
            cur, cur_val = X3.copy(), float(v3)
        else:
            # Metropolis acceptance for basin hopping
            if v3 < 0.90 and rng.rand() < np.exp((cur_val - v3) / Ttemp):
                cur, cur_val = X3.copy(), float(v3)
        if trial % 25 == 0 and rng.rand() < 0.5:
            cur, cur_val = best.copy(), best_val
        if best_val < 0.70:
            break

    # final intensive polish
    try:
        best = _lp_polish(best, deadline=t_end - 1.0, r0=1.5e-3)
    except Exception:
        pass
    best = _norm(best)
    best = np.nan_to_num(best, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(best, axis=1, keepdims=True)
    nr[nr < 1e-12] = 1.0
    best = best / nr
    return best
