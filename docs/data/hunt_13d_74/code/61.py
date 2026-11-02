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
    rng = np.random.default_rng(1234567)
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

    cur = best.copy()
    cur_val = float(best_val)

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
                                    "gtol": 1e-7, "maxfun": int(maxiter * 6)})
            Y = res.x.reshape(n, d)
        except Exception:
            return X0.copy()
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        X = Y / rn
        if not np.all(np.isfinite(X)):
            return X0.copy()
        return _normalize(X)

    def micro_split(X, curc, steps=200):
        step = 5e-6
        for _ in range(int(steps)):
            if time.time() - t0 > LIM - 3:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            k = int(np.argmax(G))
            i = k // n; j = k % n
            c = float(G[i, j])
            if c < curc - 1e-12:
                curc = c
            xi = X[i]; xj = X[j]
            gi = xj - c * xi
            gj = xi - c * xj
            ni = float(np.linalg.norm(gi)); nj = float(np.linalg.norm(gj))
            if ni < 1e-12 or nj < 1e-12:
                break
            gi /= ni; gj /= nj
            improved = False
            for mode in (0, 1, 2):
                Y = X.copy()
                if mode == 0:
                    Y[i] = xi - step * gi
                    Y[j] = xj - step * gj
                elif mode == 1:
                    Y[i] = xi - step * 2.0 * gi
                else:
                    Y[j] = xj - step * 2.0 * gj
                Y[i] /= (np.linalg.norm(Y[i]) + 1e-15)
                Y[j] /= (np.linalg.norm(Y[j]) + 1e-15)
                v = _max_cos(Y)
                if v < curc:
                    X[:] = Y; curc = v; improved = True
                    break
            if improved:
                step = min(step * 1.2, 5e-5)
            else:
                step *= 0.5
                if step < 5e-13:
                    break
        return X, curc

    def tangent_kick(x, theta):
        v = rng.standard_normal(d)
        v = v - float(v @ x) * x
        nv = float(np.linalg.norm(v))
        if nv < 1e-12 or not np.isfinite(nv):
            return x
        v /= nv
        return np.cos(theta) * x + np.sin(theta) * v

    def largest_holes(Xkeep, m, n_cand=8000):
        cand = _normalize(rng.standard_normal((n_cand, d)))
        S = cand @ Xkeep.T
        sc = S.max(axis=1)
        if m == 1:
            k = int(np.argmin(sc))
            return cand[k:k+1]
        idx = np.argpartition(sc, m - 1)[:m]
        # order by sc ascending
        idx = idx[np.argsort(sc[idx])]
        return cand[idx]

    # quick initial squeeze of incumbent
    try:
        if time.time() - t0 < LIM - 30:
            Xb = lse_polish(best, 3e-4, 60)
            vb = _max_cos(Xb)
            if vb < best_val:
                best, best_val = Xb, float(vb)
                cur, cur_val = Xb.copy(), float(vb)
    except Exception:
        pass

    trial = 0
    while time.time() - t0 < LIM - 12:
        trial += 1
        elapsed = time.time() - t0
        remain = LIM - 12 - elapsed
        if remain < 5:
            break
        # temperature: allow worse moves early
        frac = remain / (LIM - 12.0)
        temp = 0.004 * frac + 0.0002
        # reset walker if drifted too far
        if cur_val > best_val + 0.015:
            cur = best.copy(); cur_val = best_val
        # base: mostly from walker, sometimes from best
        if rng.random() < 0.30:
            base = best.copy()
        else:
            base = cur.copy()
        mode = trial % 6
        Xt = base.copy()
        try:
            if mode == 0:
                # kick 2 points medium
                idx = rng.choice(n, size=2, replace=False)
                for ii in idx:
                    th = float(rng.uniform(0.15, 0.45))
                    Xt[ii] = tangent_kick(Xt[ii], th)
            elif mode == 1:
                # kick 5-8 points small-medium
                k = int(rng.integers(4, 9))
                idx = rng.choice(n, size=k, replace=False)
                for ii in idx:
                    th = float(rng.uniform(0.08, 0.30))
                    Xt[ii] = tangent_kick(Xt[ii], th)
            elif mode == 2:
                # single hole transplant: worst point -> hole
                G = Xt @ Xt.T; np.fill_diagonal(G, -2.0)
                gmax = float(G.max())
                cnt = (G > gmax - 2e-4).sum(axis=1)
                hub = int(np.argmax(cnt))
                keep = [i for i in range(n) if i != hub]
                Xkeep = Xt[keep]
                h = largest_holes(Xkeep, 1, n_cand=8000)[0]
                th = float(rng.uniform(0.0, 0.10))
                if th > 0:
                    h = tangent_kick(h / (np.linalg.norm(h)+1e-15), th)
                Xt[hub] = h / (np.linalg.norm(h)+1e-15)
            elif mode == 3:
                # double hole transplant
                G = Xt @ Xt.T; np.fill_diagonal(G, -2.0)
                gmax = float(G.max())
                cnt = (G > gmax - 2e-4).sum(axis=1)
                o = np.argsort(-cnt)
                h1 = int(o[0]); h2 = int(o[1])
                keep = [i for i in range(n) if i != h1 and i != h2]
                Xkeep = Xt[keep]
                hs = largest_holes(Xkeep, 2, n_cand=8000)
                for hh, tgt in zip(hs, (h1, h2)):
                    th = float(rng.uniform(0.0, 0.12))
                    v = hh
                    if th > 0:
                        v = tangent_kick(v/(np.linalg.norm(v)+1e-15), th)
                    Xt[tgt] = v/(np.linalg.norm(v)+1e-15)
            elif mode == 4:
                # global jitter
                sig = float(rng.uniform(0.01, 0.05))
                Xt = Xt + sig * rng.standard_normal((n, d))
                Xt = _normalize(Xt)
            else:
                # random subset replacement by fresh random points
                k = int(rng.integers(1, 4))
                idx = rng.choice(n, size=k, replace=False)
                for ii in idx:
                    v = rng.standard_normal(d); v /= (np.linalg.norm(v)+1e-15)
                    Xt[ii] = v
            Xt = _normalize(Xt)
        except Exception:
            continue
        # staged polish: coarse then fine
        try:
            # adapt maxiter to remaining time
            if remain < 15:
                m1, m2 = 40, 40
            else:
                m1, m2 = 60, 60
            Xa = lse_polish(Xt, 2e-3, m1)
            if time.time() - t0 > LIM - 10:
                va = _max_cos(Xa)
                if va < best_val:
                    best, best_val = Xa, float(va)
                continue
            va_coarse = _max_cos(Xa)
            # prune hopeless: if far worse than walker, skip fine polish half the time
            if va_coarse > cur_val + 0.02 and va_coarse > best_val + 0.02:
                # still allow walker move occasionally without fine polish
                if va_coarse < cur_val + temp and rng.random() < 0.2:
                    cur, cur_val = Xa, float(va_coarse)
                continue
            Xb2 = lse_polish(Xa, 3e-4, m2)
            vb2 = _max_cos(Xb2)
            # extra fine polish only if competitive
            if vb2 < best_val + 0.002:
                Xc = lse_polish(Xb2, 8e-5, 50)
                vc = _max_cos(Xc)
                Xc, vc = micro_split(Xc, vc, steps=150)
            else:
                Xc, vc = Xb2, vb2
            if vc < best_val - 1e-12:
                best = _normalize(Xc).copy(); best_val = float(vc)
            if vc < cur_val + temp or vc < best_val + 1e-12:
                # accept as walker (sideways allowed)
                cur = _normalize(Xc).copy(); cur_val = float(vc)
            elif va_coarse < cur_val + temp and rng.random() < 0.1:
                cur = Xa.copy(); cur_val = float(va_coarse)
        except Exception:
            continue

    # final polish on best
    try:
        if time.time() - t0 < LIM - 5:
            Xf = lse_polish(best, 3e-4, 60)
            vf = _max_cos(Xf)
            if vf < best_val:
                best, best_val = Xf, float(vf)
            if time.time() - t0 < LIM - 3:
                Xf2 = lse_polish(best, 8e-5, 80)
                vf2 = _max_cos(Xf2)
                Xf2, vf2 = micro_split(Xf2, vf2, steps=250)
                if vf2 < best_val:
                    best, best_val = Xf2, float(vf2)
    except Exception:
        pass

    best = _normalize(best)
    best[~np.isfinite(best)] = 0.0
    nr = np.linalg.norm(best, axis=1)
    for i in range(n):
        if nr[i] < 1e-12 or not np.isfinite(nr[i]):
            v = rng.standard_normal(d); v /= (np.linalg.norm(v)+1e-15)
            best[i] = v
    best = _normalize(best)
    return best
