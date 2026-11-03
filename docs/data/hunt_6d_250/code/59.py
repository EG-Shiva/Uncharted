import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n


def _block_opt(Yb0, Xf, beta, maxiter):
    b = Yb0.shape[0]
    Y0 = Yb0.copy()

    def fg(Yflat):
        Y = Yflat.reshape(b, 6)
        nrm = np.sqrt(np.sum(Y * Y, axis=1, keepdims=True))
        nrm = np.maximum(nrm, 1e-12)
        Xb = Y / nrm
        Gbf = Xb @ Xf.T
        Gbb = Xb @ Xb.T
        # max
        m1 = float(np.max(Gbf)) if Gbf.size else -2.0
        if b > 1:
            Gbb2 = Gbb.copy()
            np.fill_diagonal(Gbb2, -2.0)
            m2 = float(np.max(Gbb2))
        else:
            m2 = -2.0
        M = m1 if m1 > m2 else m2
        Dbf = beta * (Gbf - M)
        np.clip(Dbf, -30.0, 0.0, out=Dbf)
        Ebf = np.exp(Dbf)
        if b > 1:
            Dbb = beta * (Gbb - M)
            np.clip(Dbb, -30.0, 0.0, out=Dbb)
            Ebb = np.exp(Dbb)
            np.fill_diagonal(Ebb, 0.0)
            S = float(np.sum(Ebf) + np.sum(Ebb))
        else:
            Ebb = None
            S = float(np.sum(Ebf))
        if S <= 0 or not np.finite(S):
            S = 1e-300
        loss = M + np.log(S) / beta
        Wbf = Ebf / S
        if Ebb is not None:
            Wbb = Ebb / S
            GXb = Wbf @ Xf + 2.0 * (Wbb @ Xb)
        else:
            GXb = Wbf @ Xf
        dot = np.sum(GXb * Xb, axis=1, keepdims=True)
        GY = (GXb - dot * Xb) / nrm
        return loss, GY.ravel()

    try:
        res = minimize(fg, Y0.ravel(), jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-20,
                                "gtol": 1e-9, "maxfun": maxiter * 6,
                                "maxcor": 8})
        Y = res.x.reshape(b, 6)
    except Exception:
        Y = Y0
    return _norm(Y)


def _greedy_hole(Xf, b, rng, ncand=4000):
    # random candidates on S^5
    C = rng.standard_normal((ncand, 6))
    C = _norm(C)
    # score by max dot to frozen set (want small)
    S = C @ Xf.T
    m = np.max(S, axis=1)
    order = np.argsort(m)
    picked = []
    # greedy with mutual separation
    cap = 0.72
    for idx in order:
        if len(picked) >= b:
            break
        c = C[idx]
        if picked:
            P = np.stack(picked, axis=0)
            md = float(np.max(P @ c))
            if md > cap:
                continue
        picked.append(c)
    # if not enough due to cap, relax
    if len(picked) < b:
        for idx in order:
            c = C[idx]
            # skip already picked (by index)
            dup = False
            for p in picked:
                if np.allclose(p, c):
                    dup = True
                    break
            if dup:
                continue
            picked.append(c)
            if len(picked) >= b:
                break
    return np.stack(picked[:b], axis=0)


def _lp_polish(x, deadline, k_top=4000, r0=2e-3):
    x = _norm(x)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    ar = np.arange(d)
    it = 0
    while time.time() < deadline - 1.5 and radius > 1e-13 and it < 80:
        it += 1
        Gall = x @ x.T
        c_all = Gall[iu0, iu1]
        m = float(np.max(c_all))
        window = max(20.0 * radius, 5e-7)
        sel = c_all > m - window
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > k_top:
            idx = np.argpartition(-c_all, k_top)[:k_top]
            I = iu0[idx]; J = iu1[idx]; c_sel = c_all[idx]
        else:
            I = iu0[sel]; J = iu1[sel]; c_sel = c_all[sel]
        p = I.shape[0]
        nv = n * d + 1
        try:
            cols = np.concatenate([I[:, None] * d + ar,
                                   J[:, None] * d + ar,
                                   np.full((p, 1), n * d)], axis=1).ravel()
            rows = np.repeat(np.arange(p), 2 * d + 1)
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            bounds = [(-radius, radius)] * (n * d) + [(None, float(m))]
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq,
                          b_eq=np.zeros(n), bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.008)
        else:
            radius *= 0.5
    return x


def _smooth_full(X0, beta, maxiter):
    n, d = X0.shape
    Y0 = X0.copy()

    def fg(Yflat):
        Y = Yflat.reshape(n, d)
        nrm = np.sqrt(np.sum(Y * Y, axis=1, keepdims=True))
        nrm = np.maximum(nrm, 1e-12)
        X = Y / nrm
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        D = beta * (G - M)
        np.clip(D, -30.0, 0.0, out=D)
        E = np.exp(D)
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E))
        if S <= 0 or not np.finite(S):
            S = 1e-300
        loss = M + np.log(S) / beta
        W = E / S
        GX = 2.0 * (W @ X)
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nrm
        return loss, GY.ravel()

    try:
        res = minimize(fg, Y0.ravel(), jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-20,
                                "gtol": 1e-10, "maxfun": maxiter * 5,
                                "maxcor": 8})
        Y = res.x.reshape(n, d)
    except Exception:
        Y = Y0
    return _norm(Y)


def solve():
    t_start = time.time()
    BUDGET = 168.0
    deadline = t_start + BUDGET
    rng = np.random.default_rng(12345)
    n, d = 250, 6
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            z = np.asarray(np.load(fn), dtype=np.float64)
            if z.shape == (n, d):
                z = _norm(z)
                cands.append(z)
        except Exception:
            pass
    if not cands:
        z = _norm(rng.standard_normal((n, d)))
        cands.append(z)
    # elite
    vals = [_max_cos(z) for z in cands]
    X = cands[int(np.argmin(vals))].copy()
    best_val = float(np.min(vals))
    best = X.copy()

    # ---- large neighbourhood basin hopping ----
    # time for LNS: until ~ deadline-45
    lns_end = t_start + 108.0
    it = 0
    # precompute triu for contact degrees quickly each iter
    while time.time() < lns_end - 2.0:
        it += 1
        frac = (time.time() - t_start) / max(lns_end - t_start, 1e-9)
        frac = min(max(frac, 0.0), 1.0)
        # temperature for uphill acceptance
        Temp = 1.5e-3 * (1.0 - frac) ** 2
        # block size schedule: larger early, smaller late
        if rng.random() < 0.5:
            b = int(rng.integers(12, 26))
        else:
            b = int(rng.integers(26, 42))
        # contact degrees with loose threshold to find rattlers
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        Mcur = float(np.max(G))
        thr = Mcur - 1e-3
        deg = np.sum(G >= thr, axis=1)  # symmetric count (x2 minus diag ok)
        order = np.argsort(deg, kind="stable")
        # build moving set: half low-degree, half random, plus max-pair sometimes
        nlow = b // 2
        Mset = set(order[:nlow].tolist())
        while len(Mset) < b:
            Mset.add(int(rng.integers(0, n)))
        Mset = np.array(sorted(Mset), dtype=np.int64)
        # force inclusion of current bottleneck pair with prob 0.6
        if rng.random() < 0.6:
            flat = int(np.argmax(G))
            ii, jj = divmod(flat, n)
            # replace two random entries in Mset with ii,jj
            Mset[0] = ii
            if b > 1:
                Mset[1] = jj
            Mset = np.unique(Mset)
            # pad back if duplicate collapsed
            while Mset.shape[0] < b:
                c = int(rng.integers(0, n))
                if c not in Mset:
                    Mset = np.append(Mset, c)
        mask = np.ones(n, dtype=bool)
        mask[Mset] = False
        Xf = X[mask]
        # propose initial block
        r = rng.random()
        if r < 0.55:
            # deep-hole teleport
            try:
                Yb0 = _greedy_hole(Xf, Mset.shape[0], rng, ncand=3500)
            except Exception:
                Yb0 = X[Mset] + 0.3 * rng.standard_normal((Mset.shape[0], d))
                Yb0 = _norm(Yb0)
        elif r < 0.8:
            sig = 0.25 * (1.0 - frac) + 0.05
            Yb0 = X[Mset] + sig * rng.standard_normal((Mset.shape[0], d))
            Yb0 = _norm(Yb0)
        else:
            # random orthogonal kick of subset: reflect through random direction
            sig = 0.4
            Yb0 = X[Mset] + sig * rng.standard_normal((Mset.shape[0], d))
            Yb0 = _norm(Yb0)
        beta = float(rng.choice([600.0, 900.0, 1400.0, 2200.0, 3500.0]))
        maxiter = int(rng.choice([60, 90, 120]))
        Yb = _block_opt(Yb0, Xf, beta, maxiter)
        Xt = X.copy()
        Xt[Mset] = Yb
        v = _max_cos(Xt)
        if v < best_val - 1e-12:
            X, best_val, best = Xt, v, Xt.copy()
        elif v < Mcur + Temp and rng.random() < 0.35:
            # uphill move to change basin
            X = Xt
            # Mcur updated implicitly next iter
        # occasional opportunistic keep of X even if not best but improved over current
        elif v < Mcur:
            X = Xt
        if time.time() > lns_end - 2.0:
            break

    # keep best
    X = best.copy()

    # short global smoothing to equalise shell
    if time.time() < deadline - 35.0:
        try:
            Xs = _smooth_full(X, 1500.0, 120)
            vs = _max_cos(Xs)
            if vs < best_val:
                X, best_val = Xs, vs
                best = X.copy()
            else:
                # try higher beta
                Xs2 = _smooth_full(X, 3000.0, 80)
                vs2 = _max_cos(Xs2)
                if vs2 < best_val:
                    X, best_val = Xs2, vs2
                    best = X.copy()
        except Exception:
            pass

    # final LP jamming
    X = _lp_polish(best, deadline, k_top=4000, r0=1.5e-3)
    v = _max_cos(X)
    if v < best_val:
        best = X
    else:
        # if LP did not improve (numerics), return previous best
        best = best if _max_cos(best) <= v else X
    best = _norm(best)
    # strict feasibility: tiny shrink not needed (unit rows already), just ensure finite
    best[~np.isfinite(best)] = 0.0
    # renormalise after fix
    nr = np.linalg.norm(best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    best = best / nr
    return best
