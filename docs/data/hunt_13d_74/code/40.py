import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def solve():
    t0 = time.time()
    LIM = 172.0
    rng = np.random.default_rng(7)
    n, d = 74, 13
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = _normalize(a)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_normalize(rng.standard_normal((n, d))))
    vals = [_max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    best = cands[bi].copy()
    best_val = float(vals[bi])

    IU = np.triu_indices(n, 1)
    I0 = IU[0].astype(np.int64); J0 = IU[1].astype(np.int64)

    def lse_polish(X0, tau, maxiter):
        y0 = X0.ravel().copy()
        def fun(yf):
            Y = yf.reshape(n, d)
            r2 = (Y * Y).sum(axis=1, keepdims=True)
            r = np.sqrt(np.maximum(r2, 1e-24))
            r = np.maximum(r, 1e-12)
            X = Y / r
            G = X @ X.T
            g = G[I0, J0]
            mmax = float(g.max())
            e = np.exp((g - mmax) / tau)
            S = float(e.sum())
            if not np.isfinite(S) or S <= 0:
                return float(mmax), np.zeros_like(yf)
            F = mmax + tau * np.log(S + 1e-300)
            w = e / (S + 1e-300)
            Wx = np.zeros_like(X)
            np.add.at(Wx, I0, w[:, None] * X[J0])
            np.add.at(Wx, J0, w[:, None] * X[I0])
            dot = (Wx * X).sum(axis=1, keepdims=True)
            gradY = (Wx - dot * X) / r
            return float(F), gradY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-20,
                                    "gtol": 1e-7, "maxfun": int(maxiter * 8)})
            Y = res.x.reshape(n, d)
        except Exception:
            return X0
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        X = Y / rn
        if not np.all(np.isfinite(X)):
            return X0
        return _normalize(X)

    def lp_polish(X, budget):
        # trust-region linear program on tangent displacements
        cur = _max_cos(X)
        radius = 1e-3
        start = time.time()
        Nv = n * d + 1
        obj = np.zeros(Nv); obj[-1] = 1.0
        while time.time() - start < budget:
            if time.time() - t0 > LIM - 1.5:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            Cfull = G[I0, J0]
            m = float(Cfull.max())
            thr = m - max(20.0 * radius, 5e-7)
            sel = Cfull > thr
            if np.count_nonzero(sel) > 900:
                # keep top 900
                idx = np.argpartition(-Cfull, 900)[:900]
                mask = np.zeros_like(sel)
                mask[idx] = True
                sel = mask
            I = I0[sel]; J = J0[sel]; c = Cfull[sel]
            p = len(I)
            if p == 0:
                break
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            try:
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, Nv)).tocsr()
                # equality Xi.di=0
                er = np.repeat(np.arange(n), d)
                ec = np.arange(n * d)
                ev = X.ravel()
                A_eq = coo_matrix((ev, (er, ec)), shape=(n, Nv)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            b_ub = -c
            b_eq = np.zeros(n)
            bounds = [(-radius, radius)] * (n * d) + [(None, None)]
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                              bounds=bounds, method="highs",
                              options={"time_limit": 5.0})
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                if radius < 1e-13:
                    break
                continue
            D = res.x[:-1].reshape(n, d)
            Y = _normalize(X + D)
            v = _max_cos(Y)
            if v < cur - 1e-13:
                X[:] = Y
                cur = v
                radius = min(radius * 1.6, 0.01)
            else:
                radius *= 0.5
                if radius < 1e-13:
                    break
        return X, cur

    def hole_scores(cand, X, exclude=-1):
        S = cand @ X.T
        if 0 <= exclude < n:
            S[:, exclude] = -10.0
        return S.max(axis=1)

    def largest_hole_vec(X, exclude=-1, n_cand=15000):
        cand = _normalize(rng.standard_normal((n_cand, d)))
        sc = hole_scores(cand, X, exclude)
        j = int(np.argmin(sc))
        return cand[j].copy(), float(sc[j])

    def tightest_feasible_vec(X, exclude=-1, n_cand=20000, cap=None):
        cand = _normalize(rng.standard_normal((n_cand, d)))
        sc = hole_scores(cand, X, exclude)
        if cap is None:
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            cap = float(G.max()) - 1e-4
        feas = np.where(sc < cap)[0]
        if len(feas) == 0:
            j = int(np.argmin(sc))
            return cand[j].copy(), float(sc[j])
        # tightest feasible = maximal sc below cap
        jf = feas[int(np.argmax(sc[feas]))]
        return cand[jf].copy(), float(sc[jf])

    def contact_info(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        rowmax = G.max(axis=1)
        cnt tight = (G > gmax - 2e-4).sum(axis=1)
        return G, gmax, rowmax, tight

    # initial exact squeeze of starting basin
    if time.time() - t0 < LIM - 30:
        Xc = best.copy()
        Xc = lse_polish(Xc, 3e-4, 60)
        Xc, vc = lp_polish(Xc, 5.0)
        if vc < best_val:
            best_val, best = float(vc), Xc.copy()
        else:
            # keep lp-polished anyway if very close (better basin for moves)
            pass

    trial = 0
    while time.time() - t0 < LIM - 22:
        trial += 1
        if trial > 30:
            break
        G, gmax, rowmax, cnt = contact_info(best)
        ratt = np.where(rowmax < gmax - 1e-4)[0]
        hubs = np.argsort(-cnt)
        mode = trial % 5
        Xt = best.copy()
        if mode == 1:
            # recruit up to 3 rattlers into tight feasible holes
            nr = min(3, len(ratt))
            if nr == 0:
                # no rattlers: transplant worst hub to tight hole
                h = int(hubs[0])
                if cnt[h] <= 1:
                    h = int(np.argmax(rowmax))
                nv, _ = tightest_feasible_vec(Xt, exclude=h, n_cand=20000, cap=gmax - 2e-4)
                Xt[h] = nv
            else:
                order = ratt[np.argsort(rowmax[ratt])]
                for r_ in order[:nr]:
                    nv, _ = tightest_feasible_vec(Xt, exclude=int(r_), n_cand=20000, cap=gmax - 2e-4)
                    Xt[int(r_)] = nv
            Xt = _normalize(Xt + 0.002 * rng.standard_normal(Xt.shape))
            Xt = lse_polish(Xt, 0.012, 60)
            Xt = lse_polish(Xt, 0.003, 60)
        elif mode == 2:
            # hub to largest hole + neighbor repulsion
            h = int(hubs[0])
            if cnt[h] <= 1:
                h = int(np.argmax(rowmax))
            nv, _ = largest_hole_vec(Xt, exclude=h, n_cand=12000)
            Xt[h] = nv
            # push neighbors of second hub slightly away
            h2 = int(hubs[1 % n])
            G2 = Xt @ Xt.T
            np.fill_diagonal(G2, -2.0)
            nbrs = np.where(G2[h2] > gmax - 5e-4)[0]
            for nb in nbrs[:4]:
                Xt[nb] = _normalize((Xt[nb] - 0.02 * Xt[h2]).reshape(1, -1))[0]
            Xt = _normalize(Xt + 0.003 * rng.standard_normal(Xt.shape))
            Xt = lse_polish(Xt, 0.015, 55)
            Xt = lse_polish(Xt, 0.0035, 55)
        elif mode == 3:
            # coordinated shake biased to hubs, isotropize center
            w = (cnt.astype(float) + 1.0)
            w = w / w.mean()
            noise = rng.standard_normal(Xt.shape) * (0.008 * w[:, None])
            Xt = _normalize(Xt + noise)
            # recenter: subtract mean component orthogonal? just recentre on sphere
            Xt = _normalize(Xt - 0.05 * Xt.mean(axis=0, keepdims=True))
            Xt = lse_polish(Xt, 0.01, 60)
            Xt = lse_polish(Xt, 0.0025, 60)
        elif mode == 4:
            # double transplant: one hub to largest hole, one rattler to tight hole
            h = int(hubs[0])
            nv, _ = largest_hole_vec(Xt, exclude=h, n_cand=10000)
            Xt[h] = nv
            G3 = Xt @ Xt.T
            np.fill_diagonal(G3, -2.0)
            rm3 = G3.max(axis=1)
            gm3 = float(G3.max())
            r2 = np.where(rm3 < gm3 - 1e-4)[0]
            if len(r2) > 0:
                r_ = int(r2[np.argmin(rm3[r2])])
                nv2, _ = tightest_feasible_vec(Xt, exclude=r_, n_cand=15000, cap=gm3 - 2e-4)
                Xt[r_] = nv2
            Xt = lse_polish(Xt, 0.014, 55)
            Xt = lse_polish(Xt, 0.003, 55)
        else:
            Xt = _normalize(best + 0.010 * rng.standard_normal(best.shape))
            Xt = lse_polish(Xt, 0.012, 55)
            Xt = lse_polish(Xt, 0.004, 55)
        if time.time() - t0 > LIM - 24:
            break
        Xt = lse_polish(Xt, 5e-4, 60)
        Xt, vt = lp_polish(Xt, 3.0)
        # extra fine LSE
        Xt2 = lse_polish(Xt, 8e-5, 70)
        v2 = _max_cos(Xt2)
        if v2 < vt:
            Xt, vt = Xt2, v2
        if vt < best_val - 1e-13:
            best = Xt.copy(); best_val = float(vt)
            # intensify around new basin immediately
            Xt3, vt3 = lp_polish(best.copy(), 2.0)
            if vt3 < best_val:
                best_val, best = float(vt3), Xt3.copy()

    # final intensive squeeze
    Xf = best.copy()
    remain = (t0 + LIM - 0.7) - time.time()
    if remain > 2:
        Xf = lse_polish(Xf, 6e-5, 120)
        Xf, vf = lp_polish(Xf, max(1.0, remain - 2.0))
        Xf2 = lse_polish(Xf, 3e-5, 120)
        v2 = _max_cos(Xf2)
        if v2 < vf:
            Xf, vf = Xf2, v2
        if vf < best_val:
            best_val, best = float(vf), Xf.copy()
    best = _normalize(best)
    return best
