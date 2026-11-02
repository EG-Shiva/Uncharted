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
            if a.shape == (n, d):
                cands.append(_normalize(a))
        except Exception:
            pass
    if not cands:
        cands.append(_normalize(rng.standard_normal((n, d))))
    vals = [_max_cos(a) for a in cands]
    best = cands[int(np.argmin(vals))].copy()
    best_val = float(min(vals))

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
            Wx[I0] += w[:, None] * X[J0]
            Wx[J0] += w[:, None] * X[I0]
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

    def greedy(X, steps, lr0, deadline):
        cur = _max_cos(X)
        lr = float(lr0)
        for _ in range(int(steps)):
            if time.time() >= deadline:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            gmax = float(G.max())
            delta = max(5.0 * lr, 2e-6)
            mask = G > (gmax - delta)
            np.fill_diagonal(mask, False)
            Gr = mask.astype(np.float64) @ X
            # frame drift to move rattlers (isotropic spreading)
            H = X @ X.T
            np.fill_diagonal(H, 0.0)
            Gf = H @ X
            Dir = Gr + 0.03 * Gf
            dd = (Dir * X).sum(axis=1, keepdims=True)
            Gt = Dir - dd * X
            nr = np.linalg.norm(Gt, axis=1, keepdims=True)
            act = (nr[:, 0] > 1e-14)
            if not np.any(act):
                break
            Step = np.zeros_like(Gt)
            Step[act] = Gt[act] / (nr[act] + 1e-12) * lr
            Xn = _normalize(X - Step)
            nv = _max_cos(Xn)
            if nv < cur - 1e-15:
                X[:] = Xn
                cur = nv
                lr *= 1.08
                if lr > 8e-5:
                    lr = 8e-5
            else:
                lr *= 0.5
                if lr < 2e-11:
                    break
        return X, cur

    def deepest_hole(Xkeep, n_cand=20000, n_keep=5, refine=120):
        cand = _normalize(rng.standard_normal((n_cand, d)))
        S = cand @ Xkeep.T
        sc = S.max(axis=1)
        idx = np.argpartition(sc, n_keep)[:n_keep]
        best_y = cand[idx[0]].copy()
        best_s = float(sc[idx[0]])
        for k in idx:
            y = cand[k].copy()
            lr = 0.12
            for it in range(refine):
                dots = Xkeep @ y
                j = int(np.argmax(dots))
                c = float(dots[j])
                gv = Xkeep[j] - c * y
                ng = float(np.linalg.norm(gv))
                if ng < 1e-12 or not np.isfinite(ng):
                    break
                y = y - lr * gv
                nn = float(np.linalg.norm(y))
                if nn == 0 or not np.isfinite(nn):
                    break
                y = y / nn
                lr *= 0.996
            f = float(np.max(Xkeep @ y))
            if f < best_s:
                best_s = f
                best_y = y.copy()
        return best_y, best_s

    def transplant(X, victims):
        Y = X.copy()
        for v in victims:
            keep_idx = [i for i in range(n) if i != v]
            # if we already replaced earlier victims, use updated Y
            Xkeep = Y[keep_idx]
            y, _ = deepest_hole(Xkeep, n_cand=16000, refine=100)
            # small jitter to break symmetry
            y = _normalize((y + 0.005 * rng.standard_normal(d).reshape(1, -1))[0])
            Y[v] = y
        return _normalize(Y)

    def row_stats(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        prow = G.max(axis=1)
        deg = (G > (gmax - 2e-4)).sum(axis=1)
        return G, gmax, prow, deg

    # baseline squeeze
    if time.time() - t0 < LIM - 20:
        Xb = lse_polish(best, 3e-4, 60)
        vb = _max_cos(Xb)
        if vb < best_val:
            best_val, best = float(vb), Xb.copy()
        Xb = lse_polish(best, 8e-5, 80)
        vb = _max_cos(Xb)
        if vb < best_val:
            best_val, best = float(vb), Xb.copy()
        else:
            Xb = best.copy()
        Xb, vb = greedy(Xb, 150, 1.2e-5, time.time() + 6)
        if vb < best_val:
            best_val, best = float(vb), Xb.copy()

    trial = 0
    while time.time() - t0 < LIM - 30:
        trial += 1
        remain = LIM - 30 - (time.time() - t0)
        if remain < 9:
            break
        _, gmax, prow, deg = row_stats(best)
        order_low = np.argsort(prow)  # rattlers first
        order_hub = np.argsort(-deg)
        mode = trial % 5
        try:
            if mode == 1:
                k = 1 if (trial % 10 != 1) else 2
                victims = order_low[:max(8, k * 2)][:k].tolist()
                # randomize among 10 lowest for diversity
                pool = order_low[:10].tolist()
                victims = [pool[int(rng.integers(len(pool)))] for _ in range(k)]
                Xt = transplant(best, victims)
                Xt = lse_polish(Xt, 0.02, 40)
                Xt = lse_polish(Xt, 0.005, 45)
            elif mode == 2:
                pool = order_hub[:8].tolist()
                k = 1
                victims = [pool[int(rng.integers(len(pool)))]]
                # also move one rattler together to recruit
                victims.append(int(order_low[int(rng.integers(min(8, n)))]))
                victims = list(dict.fromkeys(victims))[:2]
                Xt = transplant(best, victims)
                Xt = _normalize(Xt + 0.003 * rng.standard_normal(Xt.shape))
                Xt = lse_polish(Xt, 0.025, 40)
                Xt = lse_polish(Xt, 0.006, 45)
            elif mode == 3:
                # worst-pair split: move one endpoint to hole
                G, _, _, _ = row_stats(best)
                flat = int(np.argmax(G))
                wi, wj = flat // n, flat % n
                # pick the higher-degree endpoint
                victim = wi if deg[wi] >= deg[wj] else wj
                Xt = transplant(best, [int(victim)])
                Xt = _normalize(Xt + 0.004 * rng.standard_normal(Xt.shape))
                Xt = lse_polish(Xt, 0.015, 45)
                Xt = lse_polish(Xt, 0.004, 45)
            elif mode == 4:
                # democratic restart: big kick + high-temp isotropize
                Xt = _normalize(best + 0.10 * rng.standard_normal(best.shape))
                Xt = lse_polish(Xt, 0.05, 40)
                Xt = lse_polish(Xt, 0.012, 45)
            else:
                Xt = _normalize(best + 0.02 * rng.standard_normal(best.shape))
                Xt = lse_polish(Xt, 0.018, 40)
                Xt = lse_polish(Xt, 0.005, 40)
        except Exception:
            continue
        if time.time() - t0 > LIM - 28:
            break
        try:
            Xt = lse_polish(Xt, 5e-4, 55)
            Xt = lse_polish(Xt, 1e-4, 70)
            vq = _max_cos(Xt)
            # short greedy only if promising
            if vq < best_val + 2e-3:
                Xt, vq = greedy(Xt, 90, 1.0e-5, min(time.time() + 4, t0 + LIM - 24))
            else:
                vq = _max_cos(Xt)
        except Exception:
            continue
        if vq < best_val - 1e-13:
            best = Xt.copy(); best_val = float(vq)
        if trial > 30:
            break

    if time.time() - t0 < LIM - 1.0:
        try:
            Xf = best.copy()
            Xf = lse_polish(Xf, 6e-5, 110)
            vf = _max_cos(Xf)
            if vf < best_val:
                best_val, best = float(vf), Xf.copy()
            else:
                Xf = best.copy()
            Xf, vf = greedy(Xf, 800, 7e-6, t0 + LIM - 0.7)
            if vf < best_val:
                best = Xf.copy(); best_val = float(vf)
        except Exception:
            pass
    best = _normalize(best)
    return best
