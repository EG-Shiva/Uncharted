import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _personal_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)

def _quench(x, steps=380, T0=0.022, T1=0.0022, lr0=0.05, lr1=0.005, mu=0.75):
    n, d = x.shape
    x = x.copy()
    best = _max_cos(x)
    best_x = x.copy()
    vel = np.zeros_like(x)
    lT0 = np.log(T0); lT1 = np.log(T1)
    llr0 = np.log(lr0); llr1 = np.log(lr1)
    G = np.empty((n, n))
    W = np.empty((n, n))
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * frac))
        lr = float(np.exp(llr0 + (llr1 - llr0) * frac))
        np.dot(x, x.T, out=G)
        np.fill_diagonal(G, -1.0)
        m = G.max()
        # W = exp((G-m)/T)
        np.subtract(G, m, out=W)
        W /= T
        # clip lower to avoid underflow (exp(-20)~2e-9)
        np.clip(W, -20.0, 0.0, out=W)
        np.exp(W, out=W)
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-12
        C = (W @ x) / S
        dot = (C * x).sum(axis=1, keepdims=True)
        Ct = C - dot * x
        vel = mu * vel + Ct
        x = x - lr * vel
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm[nrm < 1e-12] = 1.0
        x /= nrm
        if (s & 63) == 63 or s == steps - 1:
            v = _max_cos(x)
            if v < best:
                best = v
                best_x = x.copy()
    return best_x, best

def _hole_vec(Xkeep, rng, S=2500):
    d = Xkeep.shape[1]
    R = rng.standard_normal((S, d))
    R /= np.linalg.norm(R, axis=1, keepdims=True) + 1e-16
    D = R @ Xkeep.T
    m = D.max(axis=1)
    i = int(np.argmin(m))
    return R[i].copy(), float(m[i])

def _half_whiten(X, alpha=0.6):
    n, d = X.shape
    C = (X.T @ X) / n
    Cd = C * d
    # symmetrize
    Cd = 0.5 * (Cd + Cd.T) + 1e-12 * np.eye(d)
    try:
        w, V = np.linalg.eigh(Cd)
    except Exception:
        return X
    w = np.clip(w, 1e-3, 1e3)
    # M = (Cd)^(-alpha/2)
    lam = w ** (-alpha * 0.5)
    M = (V * lam) @ V.T
    Y = X @ M
    nrm = np.linalg.norm(Y, axis=1, keepdims=True)
    nrm[nrm < 1e-12] = 1.0
    Y /= nrm
    return Y

def _lp_polish(x, deadline, r_init=0.01, r_max=0.04, cap=1200, max_it=40):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = r_init
    iu0 = np.triu_indices(n, 1)
    nv = n * d + 1
    obj = np.zeros(nv); obj[-1] = 1.0
    it = 0
    while time.time() + 0.4 < deadline and radius > 1e-13 and it < max_it:
        it += 1
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = max(20.0 * radius, 1e-9)
        sel = c_all > m - thresh
        nnz = int(np.count_nonzero(sel))
        if nnz == 0:
            radius *= 0.5
            continue
        if nnz > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        c_sel = c_all[sel]
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            bounds = [(-radius, radius)] * (n * d) + [(None, None)]
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y = y / yn
        v = _max_cos(y)
        if v < best:
            x = y; best = v; best_x = y.copy()
            radius = min(radius * 1.5, r_max)
        else:
            radius *= 0.5
            if radius < 1e-12:
                break
    return best_x, best

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    rng = np.random.default_rng(12345)
    n, d = 78, 15
    cand = []
    try:
        b = np.load("best.npy")
        if b.shape == (n, d):
            cand.append(b.astype(float))
    except Exception:
        pass
    try:
        r = np.load("record.npy")
        if r.shape == (n, d):
            cand.append(r.astype(float))
    except Exception:
        pass
    if len(cand) == 0:
        x0 = rng.standard_normal((n, d))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        cand.append(x0)
    best_v = 1e9; best_x0 = None
    for c in cand:
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        v = _max_cos(c)
        if v < best_v:
            best_v = v; best_x0 = c.copy()
    x_best = best_x0.copy(); v_best = best_v

    trial = 0
    # reserve ~16s for final LP
    while time.time() < HARD_END - 16:
        trial += 1
        if time.time() > HARD_END - 17:
            break
        mode = trial % 8
        x_try = x_best.copy()
        try:
            pm = _personal_max(x_try)
        except Exception:
            pm = np.zeros(n)
        if mode == 0:
            x_try = x_try + 0.16 * rng.standard_normal((n, d))
        elif mode == 1:
            x_try = x_try + 0.34 * rng.standard_normal((n, d))
        elif mode == 2:
            order = np.argsort(-pm)
            k = 6
            worst = order[:k]
            x_try = x_try + 0.012 * rng.standard_normal((n, d))
            x_try[worst] += 0.30 * rng.standard_normal((k, d))
        elif mode == 3:
            order = np.argsort(-pm)
            k = 3
            repl = order[:k]
            keep = np.ones(n, dtype=bool); keep[repl] = False
            Xkeep = x_try[keep]
            for idx in repl:
                hv, _ = _hole_vec(Xkeep, rng, S=2500)
                x_try[idx] = hv
                keep2 = np.ones(n, dtype=bool); keep2[repl] = False
                # update keep set with new point for next hole (approx)
                Xkeep = x_try[keep2 | np.isin(np.arange(n), [idx])]
                # rebuild correctly: all except remaining repl
                # simple rebuild:
                tmp = np.ones(n, dtype=bool); tmp[repl] = False
                # put back already-done idx
                # find position: easier rebuild incrementally
                # just use current x_try excluding not-yet-done: approximate by full keep+done
                # recompute:
                done = True
                # fallback: use all current except future repl
                fut = repl[np.where(np.array([rr != idx for rr in repl]))[0]] if len(repl) > 1 else np.array([], dtype=int)
                # Actually simplest: Xkeep = x_try with future repl removed
                m2 = np.ones(n, dtype=bool); m2[fut] = False
                # exclude future only, include current
                # need to exclude current? no include it
                # so Xkeep for next = x_try[m2 without future]
                # store for next iter
                Xkeep = x_try[m2]
                # remove current duplicate? keep it (it is included), but hole search should include it
                # to avoid trivial duplicate, keep as is
                pass
            others = np.where(np.array([i not in set(repl.tolist()) for i in range(n)]))[0]
            x_try[others] += 0.006 * rng.standard_normal((len(others), d))
        elif mode == 4:
            k = 5
            repl = rng.choice(n, size=k, replace=False)
            keep = np.ones(n, dtype=bool); keep[repl] = False
            Xkeep = x_try[keep]
            for idx in repl:
                hv, _ = _hole_vec(Xkeep, rng, S=2000)
                # mix with noise half the time
                if (trial % 2) == 0:
                    hv = hv + 0.15 * rng.standard_normal(d)
                    hv /= np.linalg.norm(hv) + 1e-16
                x_try[idx] = hv
                m2 = np.ones(n, dtype=bool)
                # include idx for next search
                rem = [r2 for r2 in repl.tolist() if r2 != idx]
                # crude: use all except unprocessed
                # track processed: just use full minus remaining unprocessed (excluding current)
                # rebuild: exclude repl not yet processed -> approximate by excluding all repl except idx
                # simpler: Xkeep = x_try[mask excluding future]
                # we don't track order, just use current full minus a random subset -> use all current points except idx duplicates not issue
                Xkeep = np.concatenate([Xkeep, hv[None, :]], axis=0) if len(rem) > 0 else Xkeep
            others = np.where(keep)[0]
            x_try[others] += 0.008 * rng.standard_normal((len(others), d))
        elif mode == 5:
            # cluster merge: bring two PC clusters together
            try:
                mu = x_try.mean(axis=0)
                Xc = x_try - mu
                _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
                v1 = Vt[0]
                s = Xc @ v1
                med = np.median(s)
                L = s <= med; R = s > med
                cL = x_try[L].mean(axis=0); cR = x_try[R].mean(axis=0)
                # move each cluster toward global mean
                gm = x_try.mean(axis=0)
                beta = 0.35
                x_try[L] += beta * (gm - cL)
                x_try[R] += beta * (gm - cR)
                x_try += 0.05 * rng.standard_normal((n, d))
            except Exception:
                x_try += 0.12 * rng.standard_normal((n, d))
        elif mode == 6:
            try:
                x_try = _half_whiten(x_try, alpha=0.6)
                x_try += 0.08 * rng.standard_normal((n, d))
            except Exception:
                x_try += 0.12 * rng.standard_normal((n, d))
        else:
            # mode 7: relocate overloaded (high pm) to holes
            order = np.argsort(-pm)
            k = 2
            repl = order[:k]
            keep = np.ones(n, dtype=bool); keep[repl] = False
            Xkeep = x_try[keep]
            for idx in repl:
                hv, _ = _hole_vec(Xkeep, rng, S=2500)
                x_try[idx] = hv
                m2 = np.ones(n, dtype=bool); m2[repl] = False
                # include new point
                Xkeep = x_try[np.array([True]*n)]
                # exclude other repl still old: remove them
                mask2 = np.ones(n, dtype=bool)
                for r2 in repl:
                    if r2 != idx:
                        # if not yet replaced, exclude
                        pass
                Xkeep = x_try[keep]  # approx keep fixed
                Xkeep = np.vstack([Xkeep, hv])
            x_try[keep] += 0.006 * rng.standard_normal((int(keep.sum()), d))

        x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        # quench screening
        if mode in (1, 5, 6):
            steps = 420
        elif mode in (3, 4, 7):
            steps = 400
        else:
            steps = 340
        try:
            x_s, v_s = _quench(x_try, steps=steps)
        except Exception:
            continue
        if v_s < v_best - 1e-13:
            # immediate take; LP-refine only if clearly better to save time
            if v_s < v_best - 5e-6 and time.time() < HARD_END - 30:
                try:
                    x_p, v_p = _lp_polish(x_s, min(time.time() + 3.5, HARD_END - 16), r_init=0.012, r_max=0.04, cap=1200, max_it=20)
                    if v_p < v_s:
                        x_s, v_s = x_p, v_p
                except Exception:
                    pass
            x_best = x_s; v_best = v_s

    # final intensive polish
    try:
        if time.time() < HARD_END - 1.0:
            x_best, v_best = _lp_polish(x_best, HARD_END - 0.4, r_init=0.003, r_max=0.015, cap=1500, max_it=60)
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    if not np.all(np.isfinite(x_best)):
        x_best = best_x0 / np.linalg.norm(best_x0, axis=1, keepdims=True)
    return x_best
