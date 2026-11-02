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
    I0 = IU[0].astype(np.int64)
    J0 = IU[1].astype(np.int64)

    def lse_polish(X0, tau, maxiter):
        X0 = _normalize(X0)
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
                                    "gtol": 1e-7, "maxfun": int(maxiter * 10)})
            Y = res.x.reshape(n, d)
        except Exception:
            return X0
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        X = Y / rn
        if not np.all(np.isfinite(X)):
            return X0
        return _normalize(X)

    def per_point_info(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        prow = G.max(axis=1)
        return G, prow

    def hole_best(Xkeep, n_cand, rng_):
        cand = rng_.standard_normal((n_cand, d))
        cn = np.linalg.norm(cand, axis=1, keepdims=True)
        cn[cn == 0] = 1.0
        cand = cand / cn
        S = cand @ Xkeep.T
        sc = S.max(axis=1)
        bi = int(np.argmin(sc))
        return cand[bi], float(sc[bi]), cand, sc

    def perturb(y, lo=0.01, hi=0.08):
        th = float(rng.uniform(lo, hi))
        v = rng.standard_normal(d)
        v = v - float(v @ y) * y
        nv = float(np.linalg.norm(v))
        if nv < 1e-12 or not np.finite(nv):
            return y.copy()
        v = v / nv
        yn = np.cos(th) * y + np.sin(th) * v
        nn = float(np.linalg.norm(yn))
        if nn == 0 or not np.finite(nn):
            return y.copy()
        return yn / nn

    def minimax_squeeze(X, cur, seconds):
        start = time.time()
        lr = 2e-4
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        while time.time() - start < seconds and time.time() - t0 < LIM - 2:
            np.fill_diagonal(G, -2.0) if False else None
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            gmax = float(G.max())
            delta = max(4.0 * lr, 1e-6)
            mask = G > (gmax - delta)
            Gr = mask.astype(np.float64) @ X
            dd = (Gr * X).sum(axis=1, keepdims=True)
            Gt = Gr - dd * X
            norms = np.linalg.norm(Gt, axis=1, keepdims=True)
            act = norms[:, 0] > 1e-14
            if not np.any(act):
                lr *= 0.5
                if lr < 1e-12:
                    break
                continue
            step = np.zeros_like(Gt)
            step[act] = Gt[act] / (norms[act] + 1e-12) * lr
            Xn = _normalize(X - step)
            nv = _max_cos(Xn)
            if nv < cur:
                X[:] = Xn
                cur = nv
                lr *= 1.07
                if lr > 3e-3:
                    lr = 3e-3
            else:
                lr *= 0.5
                if lr < 1e-11:
                    break
        return X, cur

    # quick baseline tighten
    try:
        if time.time() - t0 < LIM - 140:
            Xb = lse_polish(best, 8e-5, 50)
            vb = _max_cos(Xb)
            if vb < best_val:
                best, best_val = Xb, float(vb)
    except Exception:
        pass

    current = best.copy()
    cur_val = float(best_val)

    trial = 0
    # main basin hopping until ~150s
    while time.time() - t0 < LIM - 22 and trial < 300:
        trial += 1
        if time.time() - t0 > LIM - 22:
            break
        try:
            _, prow = per_point_info(current)
        except Exception:
            continue
        order_hi = np.argsort(-prow)
        order_lo = np.argsort(prow)
        mode = trial % 5
        nvict = 2
        victims = []
        if mode == 0:
            # 1 hub + 1 rattler
            h = int(order_hi[int(rng.integers(0, min(10, n)))])
            r_ = int(order_lo[int(rng.integers(0, min(10, n)))])
            victims = [h, r_]
        elif mode == 1:
            # 2 hubs
            a = int(order_hi[int(rng.integers(0, min(12, n)))])
            b = int(order_hi[int(rng.integers(0, min(12, n)))])
            victims = [a, b]
        elif mode == 2:
            # 1 hub + 1 random
            a = int(order_hi[int(rng.integers(0, min(10, n)))])
            b = int(rng.integers(0, n))
            victims = [a, b]
        elif mode == 3:
            # 2 random
            victims = [int(rng.integers(0, n)), int(rng.integers(0, n))]
        else:
            # 3 points: 2 hubs + 1 rattler (stronger jump every 5th)
            a = int(order_hi[int(rng.integers(0, min(10, n)))])
            b = int(order_hi[int(rng.integers(0, min(10, n)))])
            c_ = int(order_lo[int(rng.integers(0, min(10, n)))])
            victims = [a, b, c_]
        victims = list(dict.fromkeys([int(v) for v in victims if 0 <= int(v) < n]))
        if len(victims) == 0:
            continue
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[victims] = False
        Xkeep = current[keep_mask]
        # temperature for acceptance
        frac = trial / 300.0
        temp = 0.004 * (1.0 - frac) + 0.0003
        # find holes sequentially
        Xt = current.copy()
        Xk = Xkeep.copy()
        for k, v in enumerate(victims):
            nc = 8000 if k == 0 else 5000
            try:
                ystar, _, cand_all, sc_all = hole_best(Xk, nc, rng)
            except Exception:
                ystar = _normalize(rng.standard_normal((1, d)))[0]
            # diversity: sometimes pick among top-20
            if (trial + k) % 3 == 0:
                try:
                    topk = np.argpartition(sc_all, 20)[:20]
                    ystar = cand_all[int(topk[int(rng.integers(len(topk)))] )].copy()
                except Exception:
                    pass
            ystar = perturb(ystar, 0.01, 0.10)
            Xt[v] = ystar
            # extend keep set for next hole (use unperturbed? use perturbed)
            Xk = np.vstack([Xk, ystar.reshape(1, -1)])
        Xt = _normalize(Xt)
        # staged polish: coarse then fine (cheap)
        try:
            Xp = lse_polish(Xt, 5e-4, 45)
        except Exception:
            continue
        if time.time() - t0 > LIM - 20:
            v = _max_cos(Xp)
            if v < best_val:
                best, best_val = Xp, float(v)
            continue
        try:
            Xp = lse_polish(Xp, 1.2e-4, 55)
        except Exception:
            continue
        v = _max_cos(Xp)
        if v < best_val - 1e-12:
            best, best_val = Xp.copy(), float(v)
            current, cur_val = Xp.copy(), float(v)
        elif v < cur_val + temp and v < best_val + 0.006:
            # random-walk move to new basin
            current, cur_val = Xp.copy(), float(v)
        # occasionally reset to best if wandering too high
        if cur_val > best_val + 0.004 and (trial % 10 == 0):
            current, cur_val = best.copy(), float(best_val)

    # final intensive polish of best
    try:
        if time.time() - t0 < LIM - 12:
            Xf = lse_polish(best, 6e-5, 120)
            vf = _max_cos(Xf)
            if vf < best_val:
                best, best_val = Xf, float(vf)
    except Exception:
        pass
    try:
        if time.time() - t0 < LIM - 4:
            rem = LIM - 2 - (time.time() - t0)
            if rem > 1.0:
                Xc = best.copy()
                Xc, vc = minimax_squeeze(Xc, float(best_val), min(rem, 8.0))
                if vc < best_val:
                    best, best_val = Xc, float(vc)
    except Exception:
        pass
    best = _normalize(best)
    return best
