import time
import numpy as np
from scipy.optimize import minimize

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
    rng = np.random.default_rng(0)
    n, d = 74, 13
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (74, 13):
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
    I0 = IU[0].astype(np.int64)
    J0 = IU[1].astype(np.int64)

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

    def largest_hole(Y, exclude=-1, n_cand=4000):
        cand = _normalize(rng.standard_normal((n_cand, d)))
        S = cand @ Y.T
        if 0 <= exclude < n:
            S[:, exclude] = -10.0
        sc = S.max(axis=1)
        j = int(np.argmin(sc))
        return cand[j]

    def micro_split(X, cur, steps=300):
        step = 4e-6
        for _ in range(steps):
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            k = int(np.argmax(G))
            i = k // n; j = k % n
            c = float(G[i, j])
            if c < cur - 1e-12:
                cur = c
            xi = X[i]; xj = X[j]
            gi = xj - c * xi
            gj = xi - c * xj
            ni = float(np.linalg.norm(gi)); nj = float(np.linalg.norm(gj))
            if ni < 1e-12 or nj < 1e-12:
                break
            gi /= ni; gj /= nj
            improved = False
            # try both, i-only, j-only
            for mode in (0, 1, 2):
                Y = X.copy()
                if mode == 0:
                    Y[i] = xi - step * gi
                    Y[j] = xj - step * gj
                elif mode == 1:
                    Y[i] = xi - step * 2.0 * gi
                else:
                    Y[j] = xj - step * 2.0 * gj
                Y[i] /= np.linalg.norm(Y[i]); Y[j] /= np.linalg.norm(Y[j])
                v = _max_cos(Y)
                if v < cur:
                    X[:] = Y; cur = v; improved = True
                    break
            if improved:
                step = min(step * 1.2, 5e-5)
            else:
                step *= 0.5
                if step < 5e-13:
                    break
        return X, cur

    # baseline quick squeeze
    if time.time() - t0 < LIM - 20:
        Xb = lse_polish(best, 2e-4, 80)
        vb = _max_cos(Xb)
        if vb < best_val:
            best_val, best = float(vb), Xb.copy()
        Xb, vb = micro_split(Xb, vb, steps=250)
        if vb < best_val:
            best_val, best = float(vb), Xb.copy()

    trial = 0
    while time.time() - t0 < LIM - 26:
        trial += 1
        remain = LIM - 26 - (time.time() - t0)
        if remain < 8:
            break
        mode = trial % 5
        Xt = best.copy()
        if mode == 1:
            # hub transplant 1 to largest hole
            G = Xt @ Xt.T; np.fill_diagonal(G, -2.0)
            gmax = float(G.max())
            cnt = (G > (gmax - 2e-4)).sum(axis=1)
            hub = int(np.argmax(cnt))
            if cnt.max() <= 1:
                hub = int(np.argmax(G.max(axis=1)))
            Xt[hub] = largest_hole(Xt, exclude=hub, n_cand=5000)
            Xt = _normalize(Xt + 0.003 * rng.standard_normal(Xt.shape))
            Xt = lse_polish(Xt, 0.012, 60)
            Xt = lse_polish(Xt, 0.003, 60)
        elif mode == 2:
            # rattler recruitment: move up to 4 loose points to holes
            G = Xt @ Xt.T; np.fill_diagonal(G, -2.0)
            gmax = float(G.max())
            prow = G.max(axis=1)
            order = np.argsort(prow)
            nrep = 0
            for r_ in order[:8]:
                if prow[r_] < gmax - 2e-4 and nrep < 4:
                    Xt[r_] = largest_hole(Xt, exclude=int(r_), n_cand=4000)
                    nrep += 1
                elif prow[r_] < gmax - 1e-4 and nrep < 2:
                    Xt[r_] = largest_hole(Xt, exclude=int(r_), n_cand=4000)
                    nrep += 1
            if nrep == 0:
                # no clear rattler: transplant hub
                cnt = (G > (gmax - 2e-4)).sum(axis=1)
                hub = int(np.argmax(cnt))
                Xt[hub] = largest_hole(Xt, exclude=hub, n_cand=5000)
            Xt = _normalize(Xt + 0.002 * rng.standard_normal(Xt.shape))
            Xt = lse_polish(Xt, 0.015, 55)
            Xt = lse_polish(Xt, 0.004, 55)
        elif mode == 3:
            # double hub transplant
            for _ in range(2):
                G = Xt @ Xt.T; np.fill_diagonal(G, -2.0)
                gmax = float(G.max())
                cnt = (G > (gmax - 2e-4)).sum(axis=1)
                hub = int(np.argmax(cnt))
                Xt[hub] = largest_hole(Xt, exclude=hub, n_cand=3000)
            Xt = _normalize(Xt + 0.004 * rng.standard_normal(Xt.shape))
            Xt = lse_polish(Xt, 0.018, 50)
            Xt = lse_polish(Xt, 0.003, 55)
        elif mode == 4:
            # moderate global kick + medium anneal
            Xt = _normalize(Xt + 0.010 * rng.standard_normal(Xt.shape))
            # pull loosest toward neighbour to recruit before spreading
            G = Xt @ Xt.T; np.fill_diagonal(G, -2.0)
            gmax = float(G.max()); prow = G.max(axis=1)
            ratt = np.where(prow < gmax - 3e-4)[0]
            for r_ in ratt[:3]:
                nn = int(np.argmax(G[r_]))
                Xt[r_] = _normalize((Xt[r_] + 0.10 * Xt[nn]).reshape(1, -1))[0]
            Xt = lse_polish(Xt, 0.010, 55)
            Xt = lse_polish(Xt, 0.004, 55)
        else:
            # small kick + fine anneal
            Xt = _normalize(Xt + 0.005 * rng.standard_normal(Xt.shape))
            Xt = lse_polish(Xt, 0.008, 55)
            Xt = lse_polish(Xt, 0.002, 55)
        if time.time() - t0 > LIM - 28:
            break
        Xt = lse_polish(Xt, 4e-4, 60)
        Xt = lse_polish(Xt, 1e-4, 80)
        vq = _max_cos(Xt)
        Xt, vq = micro_split(Xt, vq, steps=200)
        if vq < best_val - 1e-13:
            best = Xt.copy(); best_val = float(vq)
        if trial > 22:
            break

    if time.time() - t0 < LIM - 1.0:
        Xf = best.copy()
        Xf = lse_polish(Xf, 6e-5, 120)
        vf = _max_cos(Xf)
        if vf < best_val:
            best_val, best = float(vf), Xf.copy()
        else:
            Xf = best.copy()
        Xf, vf = micro_split(Xf, _max_cos(Xf), steps=400)
        if vf < best_val:
            best = Xf.copy(); best_val = float(vf)
    best = _normalize(best)
    return best
