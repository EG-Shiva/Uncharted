import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix


def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-16
    return X / n


def _max_cos(Y):
    G = Y @ Y.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _polish_softmax(X0, k_list, it_list):
    n, d = X0.shape
    X = _norm(X0.copy())
    Xf = X.ravel().copy()
    for k, maxiter in zip(k_list, it_list):
        def fun(xf, n=n, d=d, k=k):
            X_ = xf.reshape(n, d)
            r = np.linalg.norm(X_, axis=1) + 1e-16
            Y = X_ / r[:, None]
            G = Y @ Y.T
            np.fill_diagonal(G, -10.0)
            mg = G.max()
            # stable softmax over i<j
            # use full matrix then correct
            E = np.exp(k * (G - mg))
            np.fill_diagonal(E, 0.0)
            Z = E.sum() * 0.5
            if Z <= 0:
                Z = 1e-300
            loss = mg + np.log(Z) / k
            W = E / (2.0 * Z) + E.T / (2.0 * Z)
            # W symmetric, diag 0, row sum weight
            # grad_Y = W @ Y  (since W_ij = w_ij for i!=j)
            # E symmetric so W = E/Z
            gradY = (E / Z) @ Y
            proj = (gradY * Y).sum(axis=1, keepdims=True)
            gradX = (gradY - proj * Y) / r[:, None]
            return float(loss), gradX.ravel()
        try:
            res = minimize(fun, Xf, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-14, "gtol": 1e-8})
            Xf = res.x
        except Exception:
            break
    X = Xf.reshape(n, d)
    return _norm(X)


def _find_hole(Y, idx, rng, n_samples=15000, k=150):
    # Y assumed normalized
    others = np.delete(Y, idx, axis=0)
    d = Y.shape[1]
    S = rng.standard_normal((n_samples, d))
    S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-16)
    dots = S @ others.T
    mxs = dots.max(axis=1)
    v0 = S[np.argmin(mxs)]

    def fun(v):
        nv = float(np.linalg.norm(v)) + 1e-16
        y = v / nv
        g = others @ y
        mg = float(g.max())
        e = np.exp(k * (g - mg))
        Z = float(e.sum()) + 1e-300
        loss = mg + np.log(Z) / k
        w = e / Z
        grad_y = w @ others
        grad_v = (grad_y - float(grad_y @ y) * y) / nv
        return float(loss), grad_v
    try:
        res = minimize(fun, v0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 120, "ftol": 1e-14, "gtol": 1e-9})
        v = res.x
        nv = np.linalg.norm(v) + 1e-16
        return v / nv
    except Exception:
        return v0 / (np.linalg.norm(v0) + 1e-16)


def _lp_polish(x, seconds):
    x = _norm(x.copy())
    n, d = x.shape

    def maxc(a):
        G = a @ a.T
        np.fill_diagonal(G, -2.0)
        return float(G.max())

    best = maxc(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 1e-6:
            break
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                      np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = maxc(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    HARD = 168.0
    rng = np.random.default_rng(0)
    # load start
    bestX = None
    bestVal = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = _norm(np.asarray(a, dtype=float))
            if a.shape == (123, 13):
                v = _max_cos(a)
                if v < bestVal:
                    bestVal = v
                    bestX = a.copy()
        except Exception:
            pass
    if bestX is None:
        a = rng.standard_normal((123, 13))
        bestX = _norm(a)
        bestVal = _max_cos(bestX)
    n, d = bestX.shape

    # quick initial annealed polish (small budget)
    try:
        if HARD - (time.time() - t0) > 40:
            Xp = _polish_softmax(bestX, [80, 250, 600], [120, 120, 120])
            vp = _max_cos(Xp)
            if vp < bestVal:
                bestX, bestVal = Xp, vp
    except Exception:
        pass

    trial = 0
    # reserve ~35s for final LP
    while time.time() - t0 < (HARD - 35):
        trial += 1
        elapsed = time.time() - t0
        if elapsed > HARD - 40:
            break
        Y = bestX
        G = Y @ Y.T
        np.fill_diagonal(G, -2.0)
        mx = float(G.max())
        # contact counts with loose and tight thresholds
        loose_thr = mx - 0.004
        tight_thr = mx - 1e-7
        cnt_loose = (G > loose_thr).sum(axis=1)
        cnt_tight = (G > tight_thr).sum(axis=1)
        order = np.argsort(-cnt_loose)

        if trial % 2 == 1:
            # global / targeted shake
            if trial % 4 == 1:
                sigma_all, sigma_top = 0.055, 0.11
                m = 6
            elif trial % 4 == 3:
                sigma_all, sigma_top = 0.025, 0.07
                m = 4
            else:
                sigma_all, sigma_top = 0.035, 0.09
                m = 5
            # vary m deterministically
            m = int(3 + (trial % 5))
            top = order[:m]
            noise = rng.standard_normal((n, d))
            Xt = Y.copy()
            Xt += sigma_all * noise
            Xt[top] += (sigma_top - sigma_all) * rng.standard_normal((m, d))
            X_trial = Xt
        else:
            # hole reinsertion for rattlers / lowest contact points
            ratt = np.where(cnt_tight == 0)[0]
            if len(ratt) == 0:
                ratt = np.argsort(cnt_tight)[:2]
            ratt = ratt[:2]
            Ytmp = Y.copy()
            for idx in ratt:
                try:
                    v = _find_hole(_norm(Ytmp), int(idx), rng)
                    Ytmp[int(idx)] = v
                except Exception:
                    Ytmp[int(idx)] = rng.standard_normal(d)
                    Ytmp[int(idx)] /= np.linalg.norm(Ytmp[int(idx)]) + 1e-16
            # small shake of the rest to allow topology change
            mask = np.ones(n, dtype=bool)
            mask[ratt] = False
            Ytmp[mask] += 0.018 * rng.standard_normal((int(mask.sum()), d))
            # also shake one overloaded point strongly
            ov = int(order[0])
            if ov not in ratt:
                Ytmp[ov] += 0.06 * rng.standard_normal(d)
            X_trial = Ytmp

        # polish with schedule depending on trial
        try:
            if trial % 3 == 0:
                Xp = _polish_softmax(X_trial, [60, 180, 500], [80, 80, 100])
            else:
                Xp = _polish_softmax(X_trial, [50, 150, 400], [90, 90, 120])
        except Exception:
            continue
        vp = _max_cos(Xp)
        if vp < bestVal - 1e-12:
            bestX, bestVal = Xp, vp
        # early exit if time low
        if time.time() - t0 > HARD - 36:
            break

    # final LP squeeze with remaining time
    rem = HARD - (time.time() - t0) - 1.0
    if rem > 4:
        rem = min(rem, 32.0)
        try:
            Xf = _lp_polish(bestX, rem)
            vf = _max_cos(Xf)
            if vf <= bestVal:
                bestX, bestVal = Xf, vf
        except Exception:
            pass
    bestX = _norm(bestX)
    return bestX
