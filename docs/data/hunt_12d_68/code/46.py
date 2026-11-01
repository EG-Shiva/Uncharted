import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _per_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _lbfgs_refine(X0, beta, maxiter, ftol=1e-14, gtol=1e-6):
    n, d = X0.shape
    IU = np.triu_indices(n, 1)
    def fg(yf):
        Y = yf.reshape(n, d)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = Y / nr
        g = (X @ X.T)[IU]
        m = float(g.max())
        e = np.exp(beta * (g - m))
        S = float(e.sum()) + 1e-300
        F = m + np.log(S) / beta
        w = e / S
        P = np.zeros((n, n))
        P[IU] = w
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return F, GY.ravel()
    y0 = (X0 / np.linalg.norm(X0, axis=1, keepdims=True)).ravel().copy()
    try:
        r = minimize(fg, y0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxfun': maxiter * 3, 'maxcor': 20})
        Y = r.x.reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    except Exception:
        return X0

def _lp_polish(X, deadline, r0=8e-4, cap_p=2500):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() > deadline - 0.4:
            break
        C = (X @ X.T)[iu]
        m = float(C.max())
        sel = C > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        csel = C[sel]
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
        if res.status != 0:
            radius *= 0.5
            continue
        step = res.x[:-1].reshape(n, d)
        Y = X + step
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        v = _max_cos(Y)
        if v < best - 1e-15:
            X, best = Y, v
            radius = min(radius * 1.5, 0.02)
        else:
            ok = False
            for sh in (0.3, 0.1):
                YY = X + step * sh
                YY /= np.linalg.norm(YY, axis=1, keepdims=True)
                vv = _max_cos(YY)
                if vv < best:
                    X, best = YY, vv
                    ok = True
                    break
            if not ok:
                radius *= 0.5
    return X

def _best_hole(Xr, rng, ncand=20000):
    d = Xr.shape[1]
    cand = rng.standard_normal((ncand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    sc = (cand @ Xr.T).max(axis=1)
    return cand[int(np.argmin(sc))]

def _opt_single(Xr, z0, beta, maxiter=80):
    def fg(z):
        r = max(float(np.linalg.norm(z)), 1e-12)
        y = z / r
        s = Xr @ y
        m = float(s.max())
        e = np.exp(beta * (s - m))
        S = float(e.sum()) + 1e-300
        F = m + np.log(S) / beta
        w = e / S
        gy = Xr.T @ w
        gz = (gy - float(gy @ y) * y) / r
        return F, gz
    try:
        r = minimize(fg, z0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-9, 'maxfun': maxiter * 2})
        z = r.x
        z /= (np.linalg.norm(z) + 1e-300)
        return z
    except Exception:
        z = z0 / (np.linalg.norm(z0) + 1e-300)
        return z

def solve():
    t0 = time.time()
    TL = 172.0
    deadline_final = t0 + TL
    rng = np.random.default_rng(987654321)
    cands = []
    try:
        xb = np.load("best.npy").astype(float)
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
        cands.append(xb)
    except Exception:
        pass
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        cands.append(xr)
    except Exception:
        pass
    if not cands:
        x0 = rng.standard_normal((68, 12))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        cands.append(x0)
    x_best = cands[0].copy()
    best_val = _max_cos(x_best)
    for c in cands[1:]:
        v = _max_cos(c)
        if v < best_val:
            best_val = v
            x_best = c.copy()
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)

    pool = [x_best.copy()]

    # Phase A: diverse random restarts with moderate LSE (different basins)
    phaseA_end = t0 + 42.0
    nA = 0
    while time.time() < phaseA_end and time.time() < deadline_final - 130.0:
        nA += 1
        u = rng.random()
        if u < 0.5 or len(pool) < 2:
            X = rng.standard_normal((68, 12))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        else:
            # large shake of incumbent to seed new basin
            sig = float(rng.choice([0.25, 0.45, 0.7]))
            X = x_best + sig * rng.standard_normal((68, 12))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            # random block replacement
            k = int(rng.integers(4, 12))
            idx = rng.choice(68, size=k, replace=False)
            R = rng.standard_normal((k, 12))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            X[idx] = R
        beta = float(rng.choice([60.0, 90.0, 140.0]))
        X = _lbfgs_refine(X, beta, 70)
        if time.time() > phaseA_end:
            v = _max_cos(X)
            if v < best_val:
                best_val = v; x_best = X.copy()
            if v < best_val + 0.02:
                pool.append(X.copy())
            break
        X = _lbfgs_refine(X, 250.0, 70)
        v = _max_cos(X)
        if v < best_val:
            best_val = v; x_best = X.copy()
        if v < best_val + 0.015:
            pool.append(X.copy())
            if len(pool) > 12:
                # keep most diverse / best
                pool = sorted(pool, key=_max_cos)[:8]
        if nA > 14:
            break

    # Phase B: temperature basin-hopping with surgery, NO LP in loop
    cur = x_best.copy()
    cur_val = best_val
    explore_end = t0 + TL - 24.0
    trial = 0
    while time.time() < explore_end:
        trial += 1
        frac = (time.time() - t0) / TL
        temp = 0.0035 * (1.0 - frac) + 0.0003
        # choose base: mostly cur (annealing chain), sometimes best, sometimes pool
        u0 = rng.random()
        if u0 < 0.6:
            base = cur
        elif u0 < 0.85:
            base = x_best
        else:
            base = pool[int(rng.integers(len(pool)))]
        X = base.copy()
        kick = rng.random()
        if kick < 0.22:
            # global gaussian shake, variable scale
            sig = float(rng.choice([0.03, 0.08, 0.18, 0.35]))
            X = X + sig * rng.standard_normal(X.shape)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        elif kick < 0.50:
            # k-point surgery on worst points (overloaded hubs)
            pm = _per_max(X)
            k = int(rng.integers(2, 8))
            # mix worst + random to avoid deterministic cycle
            worst = np.argsort(-pm)[:k + 3]
            sel = rng.choice(worst, size=k, replace=False)
            for idx in sel:
                mask = np.ones(68, dtype=bool); mask[idx] = False
                Xr = X[mask]
                hole = _best_hole(Xr, rng, ncand=15000)
                # jitter hole then single-point LSE opt
                hole = hole + 0.03 * rng.standard_normal(12)
                hole /= (np.linalg.norm(hole) + 1e-300)
                hole = _opt_single(Xr, hole, 200.0, maxiter=60)
                X[idx] = hole
        elif kick < 0.72:
            # max-pair split: push top pairs apart hard
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            # get top few pairs
            for _ in range(int(rng.integers(1, 4))):
                ii, jj = np.unravel_index(int(np.argmax(G)), G.shape)
                if G[ii, jj] < -1.5:
                    break
                amp = float(rng.uniform(0.15, 0.5))
                # move i away from j in tangent space
                d1 = X[ii] - float(X[ii] @ X[jj]) * X[jj]
                d2 = X[jj] - float(X[jj] @ X[ii]) * X[ii]
                n1 = np.linalg.norm(d1) + 1e-12; n2 = np.linalg.norm(d2) + 1e-12
                X[ii] = X[ii] + amp * (d1 / n1) + 0.02 * rng.standard_normal(12)
                X[jj] = X[jj] + amp * (d2 / n2) + 0.02 * rng.standard_normal(12)
                X[ii] /= np.linalg.norm(X[ii]); X[jj] /= np.linalg.norm(X[jj])
                G = X @ X.T
                np.fill_diagonal(G, -2.0)
        else:
            # random subspace rotation of a subset (coherent block move)
            m = int(rng.integers(6, 18))
            sel = rng.choice(68, size=m, replace=False)
            # random 2-plane rotation by large angle
            A = rng.standard_normal((12, 12))
            Q, _ = np.linalg.qr(A)
            # random orthogonal with strength: interpolate
            alpha = float(rng.uniform(0.2, 0.8))
            # X_sel = (1-alpha)*X_sel + alpha*(X_sel @ Q) then renormalize -> new basin
            Xs = X[sel] @ Q
            X[sel] = (1 - alpha) * X[sel] + alpha * Xs + 0.03 * rng.standard_normal((m, 12))
            X[sel] /= np.linalg.norm(X[sel], axis=1, keepdims=True)
        # quench at moderate beta only (allow topology change)
        b1 = float(rng.choice([90.0, 150.0, 220.0]))
        X = _lbfgs_refine(X, b1, 80)
        if trial % 3 == 0 and time.time() < explore_end - 5:
            X = _lbfgs_refine(X, 320.0, 50)
        v = _max_cos(X)
        if v < best_val:
            best_val = v; x_best = X.copy()
            cur = X.copy(); cur_val = v
            pool.append(X.copy())
            if len(pool) > 10:
                pool = sorted(pool, key=_max_cos)[:8]
        else:
            if v < cur_val or rng.random() < np.exp(-(v - cur_val) / max(temp, 1e-6)):
                cur = X.copy(); cur_val = v
            elif v < best_val + 0.006 and rng.random() < 0.15:
                pool.append(X.copy())
                if len(pool) > 10:
                    pool = sorted(pool, key=_max_cos)[:8]
        # occasional reheat from best
        if trial % 25 == 0:
            cur = x_best.copy(); cur_val = best_val

    # Phase C: final squeeze of top candidates with high beta + LP
    uniq = sorted(pool, key=_max_cos)[:4]
    # ensure best included
    cand_list = [x_best.copy()] + [u for u in uniq if _max_cos(u) < best_val + 0.008][:3]
    # dedup roughly
    final_best = x_best.copy()
    final_val = best_val
    time_left = deadline_final - time.time()
    per = max(time_left / max(len(cand_list), 1), 2.0)
    for ci, C0 in enumerate(cand_list):
        if time.time() > deadline_final - 1.5:
            break
        C = C0.copy()
        # high-beta LSE chain short
        for beta, mi in [(350.0, 90), (600.0, 90)]:
            if time.time() > deadline_final - 2.0:
                break
            C = _lbfgs_refine(C, beta, mi, ftol=1e-16, gtol=1e-8)
        dl = min(time.time() + per * 0.55, deadline_final - 0.5)
        C = _lp_polish(C, dl)
        v = _max_cos(C)
        if v < final_val:
            final_val = v; final_best = C.copy()
    final_best /= np.linalg.norm(final_best, axis=1, keepdims=True)
    return final_best
