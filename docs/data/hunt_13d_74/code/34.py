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
    rng = np.random.default_rng(1)
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
        cands.append(_normalize(rng.standard_normal((74, 13))))
    vals = [_max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    best = cands[bi].copy()
    best_val = float(vals[bi])

    IU = np.triu_indices(n, 1)
    I0 = IU[0].astype(np.int64); J0 = IU[1].astype(np.int64)
    P = len(I0)

    def general_polish(X0, tau, w_frame, lam, maxiter, tau2=None, tau3=0.02):
        if tau2 is None:
            tau2 = tau
        pure = (w_frame == 0.0 and lam == 0.0)
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
            Wpair = w
            Ffp = 0.0
            v = None
            if w_frame != 0.0:
                fp_mean = float((g * g).mean())
                Ffp = w_frame * fp_mean
                v = (2.0 * g / P) * w_frame
                Wpair = Wpair + v
            m_min_smooth = 0.0
            z = None
            if lam != 0.0:
                G0 = G.copy()
                np.fill_diagonal(G0, -10.0)
                row_max = G0.max(axis=1)
                E = np.exp((G0 - row_max[:, None]) / tau2)
                np.fill_diagonal(E, 0.0)
                Srow = E.sum(axis=1)
                Srow = np.maximum(Srow, 1e-300)
                m_i = row_max + tau2 * np.log(Srow)
                mn = float(m_i.min())
                ex = np.exp((mn - m_i) / tau3)
                Qs = float(ex.sum())
                if not np.isfinite(Qs) or Qs <= 0:
                    Qs = 1.0
                    q = np.ones(n) / n
                else:
                    q = ex / Qs
                m_min_smooth = mn - tau3 * np.log(max(Qs, 1e-300))
                Ei = E[I0, J0] / Srow[I0]
                Ej = E[J0, I0] / Srow[J0]
                z = -lam * (q[I0] * Ei + q[J0] * Ej)
                Wpair = Wpair + z
                F = F + Ffp - lam * m_min_smooth
            else:
                F = F + Ffp
            Wx = np.zeros_like(X)
            np.add.at(Wx, I0, Wpair[:, None] * X[J0])
            np.add.at(Wx, J0, Wpair[:, None] * X[I0])
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

    def min_norm_g(Grad, iters=120):
        g = np.mean(Grad, axis=0)
        for _ in range(iters):
            dots = Grad @ g
            kmin = int(np.argmin(dots))
            gn2 = float(g @ g)
            dg = float(dots[kmin]) - gn2
            if dg >= -1e-16:
                break
            dk = Grad[kmin] - g
            dd = float(dk @ dk)
            if dd <= 1e-24:
                break
            gamma = -dg / dd
            if gamma > 1.0:
                gamma = 1.0
            if gamma < 1e-14:
                break
            g = g + gamma * dk
        return g

    def bundle_polish(X, deadline, K=500, lr_init=2e-5):
        cur = _max_cos(X)
        lr = float(lr_init)
        N = n * d
        outer = 0
        fails = 0
        while time.time() < deadline:
            outer += 1
            if outer > 5000:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            gfull = G[I0, J0]
            if K < len(gfull):
                part = np.argpartition(-gfull, K)[:K]
                Ik = I0[part]; Jk = J0[part]; ck = gfull[part]
            else:
                Ik, Jk, ck = I0, J0, gfull
            p = len(ck)
            Grad = np.zeros((p, N), dtype=np.float64)
            for k in range(p):
                i = int(Ik[k]); j = int(Jk[k]); c = float(ck[k])
                xi = X[i]; xj = X[j]
                gi = xj - c * xi
                gj = xi - c * xj
                Grad[k, i * d:(i + 1) * d] = gi
                Grad[k, j * d:(j + 1) * d] = gj
            gvec = min_norm_g(Grad, iters=120)
            ng = float(np.sqrt(gvec @ gvec))
            if not np.isfinite(ng) or ng < 1e-10:
                if K < 900 and fails % 3 == 0:
                    K = min(900, K + 150)
                    continue
                lr *= 0.5
                if lr < 1e-12:
                    break
                fails += 1
                if fails > 12:
                    break
                continue
            Gd = gvec.reshape(n, d)
            accepted = False
            for _ in range(12):
                if time.time() >= deadline:
                    break
                Xt = _normalize(X - lr * Gd)
                nv = _max_cos(Xt)
                if nv < cur - 1e-14:
                    X[:] = Xt
                    cur = nv
                    lr *= 1.35
                    if lr > 5e-4:
                        lr = 5e-4
                    accepted = True
                    fails = 0
                    break
                lr *= 0.5
                if lr < 1e-13:
                    break
            if not accepted:
                fails += 1
                if fails > 14:
                    if K < 900:
                        K = min(900, K + 200)
                        fails = 0
                        lr = lr_init * 0.5
                        continue
                    break
        return X, cur

    def hole_transplant(X, n_move=1, n_cand=4000):
        Y = X.copy()
        for _ in range(n_move):
            G = Y @ Y.T
            np.fill_diagonal(G, -2.0)
            gmax = float(G.max())
            cnt = (G > (gmax - 2e-4)).sum(axis=1)
            hub = int(np.argmax(cnt))
            # if all equal, pick worst row max
            if cnt.max() <= 1:
                hub = int(np.argmax(G.max(axis=1)))
            cand = _normalize(rng.standard_normal((n_cand, d)))
            S = cand @ Y.T
            S[:, hub] = -10.0
            sc = S.max(axis=1)
            j = int(np.argmin(sc))
            Y[hub] = cand[j]
        return _normalize(Y)

    # quick baseline squeeze with corrected gradients
    if time.time() - t0 < LIM - 20:
        Xb = general_polish(best, 2e-4, 0.0, 0.0, 80)
        vb = _max_cos(Xb)
        if vb < best_val:
            best_val, best = float(vb), Xb.copy()
        Xb, vb = bundle_polish(Xb if vb < best_val + 1e-6 else best.copy(),
                               time.time() + 6, K=550, lr_init=1.2e-5)
        if vb < best_val:
            best_val, best = float(vb), Xb.copy()

    trial = 0
    while time.time() - t0 < LIM - 28:
        trial += 1
        remain = LIM - 28 - (time.time() - t0)
        if remain < 10:
            break
        mode = trial % 4
        if mode == 1:
            Xt = hole_transplant(best, n_move=1, n_cand=4000)
            Xt = _normalize(Xt + 0.004 * rng.standard_normal(Xt.shape))
            Xt = general_polish(Xt, 0.015, 0.4, 0.25, 60, tau2=0.015)
            Xt = general_polish(Xt, 0.004, 0.08, 0.05, 60)
        elif mode == 2:
            Xt = hole_transplant(best, n_move=2, n_cand=3000)
            Xt = general_polish(Xt, 0.02, 0.5, 0.3, 50, tau2=0.02)
            Xt = general_polish(Xt, 0.003, 0.05, 0.03, 60)
        elif mode == 3:
            Xt = _normalize(best + 0.012 * rng.standard_normal(best.shape))
            Xt = general_polish(Xt, 0.012, 0.35, 0.2, 60, tau2=0.012)
            Xt = general_polish(Xt, 0.005, 0.1, 0.05, 50)
        else:
            Xt = _normalize(best + 0.006 * rng.standard_normal(best.shape))
            # pull rattlers toward nearest neighbour to recruit them
            G = Xt @ Xt.T
            np.fill_diagonal(G, -2.0)
            prow = G.max(axis=1)
            gmax = float(G.max())
            ratt = np.where(prow < gmax - 1e-4)[0]
            for r_ in ratt[:4]:
                nn = int(np.argmax(G[r_]))
                Xt[r_] = _normalize((Xt[r_] + 0.15 * Xt[nn]).reshape(1, -1))[0]
            Xt = _normalize(Xt + 0.002 * rng.standard_normal(Xt.shape))
            Xt = general_polish(Xt, 0.01, 0.3, 0.15, 60, tau2=0.01)
            Xt = general_polish(Xt, 0.004, 0.08, 0.05, 50)
        # anneal pure
        if time.time() - t0 > LIM - 30:
            break
        Xt = general_polish(Xt, 5e-4, 0.0, 0.0, 70)
        Xt = general_polish(Xt, 1e-4, 0.0, 0.0, 90)
        vq = _max_cos(Xt)
        Xt, vq2 = bundle_polish(Xt, min(time.time() + 5, t0 + LIM - 24), K=450, lr_init=1.2e-5)
        if vq2 < best_val - 1e-13:
            best = Xt.copy(); best_val = float(vq2)
        elif vq2 < best_val + 3e-4 and trial % 5 == 0:
            # occasional diversification: keep topology-different base for next transplant
            pass
        if trial > 14:
            break

    # final intensive squeeze
    if time.time() - t0 < LIM - 1.0:
        Xf = best.copy()
        # two fine LSE steps
        Xf = general_polish(Xf, 6e-5, 0.0, 0.0, 120)
        vf = _max_cos(Xf)
        if vf < best_val:
            best_val, best = float(vf), Xf.copy()
        else:
            Xf = best.copy()
        Xf, vf = bundle_polish(Xf, t0 + LIM - 0.8, K=650, lr_init=7e-6)
        if vf < best_val:
            best = Xf.copy(); best_val = float(vf)
    best = _normalize(best)
    return best
