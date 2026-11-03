import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, r0=0.004):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    t0 = time.time()
    iu = np.triu_indices(n, 1)
    iuI = iu[0]; iuJ = iu[1]
    while time.time() - t0 < seconds and radius > 1e-14:
        cfull = (x @ x.T)[iuI, iuJ]
        m = float(cfull.max())
        if m < best:
            best = m
        sel = cfull > m - max(20.0 * radius, 1e-9)
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 2500:
            idx = np.argpartition(-cfull, 2500)[:2500]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
            cnt = 2500
        I = iuI[sel]; J = iuJ[sel]
        c = cfull[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c,
                          A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best - 1e-15:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - t0 >= seconds:
            break
    return x

def solve():
    t_start = time.time()
    BUDGET = 174.0
    n, d = 162, 8
    rng = np.random.default_rng(20261103)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and xb.shape == (162, 8):
        cands.append(xb / np.linalg.norm(xb, axis=1, keepdims=True))
    if xr is not None and xr.shape == (162, 8):
        cands.append(xr / np.linalg.norm(xr, axis=1, keepdims=True))
    if not cands:
        a = rng.standard_normal(size=(162, 8))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    X_best = cands[bi].copy()
    best_val = float(vals[bi])

    SHIFT = 0.5
    def eg_fn(Yf, alpha):
        Y = Yf.reshape(n, d)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = Y / nr
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = alpha * (G - SHIFT)
        np.clip(M, -700, 200, out=M)
        Emat = np.exp(M)
        E = float(Emat.sum() * 0.5)
        GradX = alpha * (Emat @ X)
        dot = np.sum(GradX * X, axis=1, keepdims=True)
        GradY = (GradX - dot * X) / nr
        return E, GradY.ravel()

    def staged_opt(X_init, alphas, maxiters):
        Yf = X_init.ravel().copy()
        for alpha, mi in zip(alphas, maxiters):
            if time.time() - t_start > BUDGET - 42:
                break
            try:
                res = minimize(lambda y: eg_fn(y, alpha), Yf, jac=True,
                               method="L-BFGS-B",
                               options={"maxiter": mi, "ftol": 1e-12,
                                        "gtol": 1e-5, "maxfun": mi * 20})
                Yf = res.x
            except Exception:
                break
        Y = Yf.reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y

    def hole_replace(Xcur, idx_list, rng):
        Xnew = Xcur.copy()
        k = len(idx_list)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[np.asarray(idx_list)] = False
        Xkeep = Xnew[keep_mask]
        nsamp = 8000
        C = rng.standard_normal(size=(nsamp, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        S = C @ Xkeep.T
        scores = S.max(axis=1)
        order = np.argsort(scores)
        picked = []
        for oi in order:
            if len(picked) >= k:
                break
            v = C[oi]
            ok = True
            for q in picked:
                if float(v @ q) > 0.60:
                    ok = False
                    break
            if ok:
                picked.append(v.copy())
        while len(picked) < k:
            v = rng.standard_normal(size=(d,))
            v /= np.linalg.norm(v)
            picked.append(v)
        P = np.array(picked)
        for t, w in enumerate(idx_list):
            Xnew[w] = P[t]
        return Xnew

    # quick diverse random basin to seed alternative topology
    # one short random anneal (cheap)
    try:
        Ar = rng.standard_normal(size=(n, d))
        Ar /= np.linalg.norm(Ar, axis=1, keepdims=True)
        Ar = staged_opt(Ar, [30, 90], [35, 35])
        vr = _max_cos(Ar)
    except Exception:
        Ar, vr = None, 1e9
    # Metropolis chain state: start from best but allow drift
    X_cur = X_best.copy()
    cur_val = best_val
    # if random basin is competitive (within 0.01), start chain there half the time
    # to force different topology; else stay at best
    if Ar is not None and vr < best_val + 0.008:
        # large kick mixing: keep best as elite but start chain from random basin
        # choose the better as elite, chain from the more balanced one
        # estimate balance: contact variance
        def balance_score(X, thr):
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            cc = (G > thr).sum(axis=1)
            return float(cc.std())
        try:
            b0 = balance_score(X_best, best_val - 0.006)
            b1 = balance_score(Ar, vr - 0.006)
            if b1 < b0 and vr < best_val + 0.004:
                X_cur = Ar.copy(); cur_val = vr
        except Exception:
            pass

    TEMP = 0.0006
    trial = 0
    # main hopping until ~128s
    while time.time() - t_start < 128:
        trial += 1
        mode = trial % 6
        X_try = X_cur.copy()
        long_anneal = False
        if mode == 0:
            sigma = 0.03
            X_try = X_try + sigma * rng.standard_normal(size=X_try.shape)
            X_try /= np.linalg.norm(X_try, axis=1, keepdims=True)
        elif mode == 1:
            sigma = 0.22
            X_try = X_try + sigma * rng.standard_normal(size=X_try.shape)
            X_try /= np.linalg.norm(X_try, axis=1, keepdims=True)
            long_anneal = True
        elif mode == 2:
            # cluster shake around overloaded points
            G = X_cur @ X_cur.T
            np.fill_diagonal(G, -2.0)
            per_max = G.max(axis=1)
            worst = np.argsort(per_max)[-2:]
            thresh = cur_val - 0.008
            neigh = set()
            for w in worst:
                nb = np.where(G[w] > thresh)[0]
                for u in nb:
                    neigh.add(int(u))
            neigh = list(neigh)[:24]
            X_try[worst] += 0.25 * rng.standard_normal(size=(len(worst), d))
            if len(neigh) > 0:
                X_try[neigh] += 0.12 * rng.standard_normal(size=(len(neigh), d))
            X_try /= np.linalg.norm(X_try, axis=1, keepdims=True)
            long_anneal = True
        elif mode == 3:
            G = X_cur @ X_cur.T
            np.fill_diagonal(G, -2.0)
            per_max = G.max(axis=1)
            worst = np.argsort(per_max)[-3:]
            X_try = hole_replace(X_try, worst, rng)
            long_anneal = True
        elif mode == 4:
            G = X_cur @ X_cur.T
            np.fill_diagonal(G, -2.0)
            thresh = cur_val - 0.006
            cont = (G > thresh).sum(axis=1)
            per_max = G.max(axis=1)
            low = np.argsort(cont)[:3]
            high = np.argsort(per_max)[-2:]
            idx = list(dict.fromkeys(list(low) + list(high)))[:5]
            X_try = hole_replace(X_try, idx, rng)
            long_anneal = True
        else:
            # medium global + subset twist
            sigma = 0.09
            X_try = X_try + sigma * rng.standard_normal(size=X_try.shape)
            X_try /= np.linalg.norm(X_try, axis=1, keepdims=True)

        if long_anneal:
            alphas, mis = [25, 60, 140], [45, 55, 55]
        else:
            if mode == 0:
                alphas, mis = [45, 130], [50, 50]
            else:
                alphas, mis = [40, 120], [50, 50]
        X_opt = staged_opt(X_try, alphas, mis)
        v = _max_cos(X_opt)
        # elite update
        if v < best_val - 1e-12:
            X_best = X_opt.copy()
            best_val = v
        # Metropolis acceptance for chain movement (allows sideways drift)
        if v < cur_val:
            X_cur, cur_val = X_opt, v
            # occasional quick LP lock-in when clear gain and early
            if v < best_val + 1e-12 and (best_val < cur_val + 1e-12) and time.time() - t_start < 95:
                try:
                    X_lp = _lp_polish(X_best, 1.5, r0=0.003)
                    v2 = _max_cos(X_lp)
                    if v2 < best_val:
                        X_best, best_val = X_lp, v2
                        X_cur, cur_val = X_lp.copy(), v2
                except Exception:
                    pass
        else:
            # accept worse with prob exp((cur-v)/TEMP)
            try:
                p_acc = float(np.exp((cur_val - v) / TEMP))
            except Exception:
                p_acc = 0.0
            if rng.random() < p_acc:
                X_cur, cur_val = X_opt, v
        # slow cooling
        elapsed = time.time() - t_start
        TEMP = max(0.00018, 0.0006 * (1.0 - elapsed / 140.0))
        if time.time() - t_start > BUDGET - 42:
            break

    remain = BUDGET - (time.time() - t_start) - 2.0
    if remain > 5:
        try:
            X_best = _lp_polish(X_best, remain, r0=0.005)
        except Exception:
            pass
    X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
    X_best = np.nan_to_num(X_best, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(X_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    X_best = X_best / nr
    return X_best
