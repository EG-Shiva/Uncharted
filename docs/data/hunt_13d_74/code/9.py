import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix


def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def smooth_opt(X_init, tau, maxiter, gtol=1e-7):
    n, d = X_init.shape
    iu0_0, iu0_1 = np.triu_indices(n, 1)

    def fun(y_flat):
        Y = y_flat.reshape(n, d)
        r = np.linalg.norm(Y, axis=1, keepdims=True)
        r = np.maximum(r, 1e-12)
        X = Y / r
        G = X @ X.T
        g = G[iu0_0, iu0_1]
        mmax = g.max()
        e = np.exp((g - mmax) / tau)
        S = e.sum()
        F = mmax + tau * np.log(S + 1e-300)
        w = e / (S + 1e-300)
        Wx = np.zeros_like(X)
        # correct accumulation for duplicates
        np.add.at(Wx, iu0_0, w[:, None] * X[iu0_1])
        np.add.at(Wx, iu0_1, w[:, None] * X[iu0_0])
        dot = np.sum(Wx * X, axis=1, keepdims=True)
        grad_Y = (Wx - X * dot) / r
        return float(F), grad_Y.ravel()

    y0 = X_init.ravel().copy()
    try:
        res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-14,
                                "gtol": gtol, "maxfun": 20000})
        Y = res.x.reshape(n, d)
    except Exception:
        return X_init
    nrm = np.linalg.norm(Y, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    X = Y / nrm
    if not np.all(np.isfinite(X)):
        return X_init
    return X


def lp_polish(X, deadline, r0=1.2e-3):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = max_cos(X)
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = X @ X.T
        c_all = G[iu0, iu1]
        m = c_all.max()
        sel_thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > sel_thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        # keep all (2701 max) but cap for speed if huge
        if cnt > 2701:
            idx = np.argpartition(c_all, -2701)[-2701:]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
            cnt = 2701
        I = iu0[sel]; J = iu1[sel]
        c = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        # build cols/vals
        # I-block, J-block, t-col
        base_I = I[:, None] * d + np.arange(d)
        base_J = J[:, None] * d + np.arange(d)
        cols = np.concatenate([base_I, base_J, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X


def worst_scores(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)


def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    EXPLORE_END = t0 + 148.0
    rng = np.random.default_rng(7)
    n, d = 74, 13
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.standard_normal((n, d))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (n, d):
        tmp = rng.standard_normal((n, d))
        m = min(xb.shape[0], n); k = min(xb.shape[1], d)
        tmp[:m, :k] = xb[:m, :k]
        xb = tmp
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    # also consider record.npy
    try:
        xr = np.load("record.npy")
        xr = np.asarray(xr, dtype=float)
        if xr.shape == (n, d):
            xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
            if max_cos(xr) < max_cos(xb):
                xb = xr
    except Exception:
        pass

    global_best = xb.copy()
    global_val = max_cos(global_best)

    # quick initial jam
    if DEADLINE - time.time() > 10:
        try:
            xj = lp_polish(global_best, min(time.time() + 6.0, EXPLORE_END))
            vj = max_cos(xj)
            if vj < global_val:
                global_best, global_val = xj, vj
        except Exception:
            pass

    it = 0
    sigmas = [0.004, 0.01, 0.02, 0.007, 0.03, 0.002, 0.05, 0.015]
    while time.time() < EXPLORE_END:
        it += 1
        remain = EXPLORE_END - time.time()
        if remain < 1.0:
            break
        mode = it % 12
        sigma = sigmas[it % len(sigmas)]
        Xb = global_best

        if mode == 5:
            # from-scratch random (rare)
            x0 = rng.standard_normal((n, d))
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        elif mode == 10:
            # single-point void reinsertion: replace worst point by best void
            ws = worst_scores(Xb)
            w = int(np.argmax(ws))
            # search void: best of random candidates minimizing max dot
            cand = rng.standard_normal((2500, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            dots = cand @ Xb.T
            # for replacing w, ignore column w
            dots[:, w] = -2.0
            md = dots.max(axis=1)
            b = int(np.argmin(md))
            x0 = Xb.copy()
            x0[w] = cand[b]
            # small jitter others
            x0 = x0 + 0.002 * rng.standard_normal((n, d))
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        elif mode == 7:
            # targeted worst-k heavy kick
            ws = worst_scores(Xb)
            order = np.argsort(-ws)
            k = 8
            x0 = Xb.copy()
            noise = rng.standard_normal((n, d))
            scale = np.full((n, 1), sigma * 0.3)
            scale[order[:k]] = sigma * 2.0
            x0 = x0 + scale * noise
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        elif mode == 11:
            # push max pair apart tangentially + jitter
            G = Xb @ Xb.T
            np.fill_diagonal(G, -2)
            ij = np.argmax(G)
            a, b = ij // n, ij % n
            x0 = Xb.copy()
            # tangent direction
            da = Xb[a] - G[a, b] * Xb[b]
            db = Xb[b] - G[a, b] * Xb[a]
            na = np.linalg.norm(da); nb = np.linalg.norm(db)
            if na > 1e-12:
                x0[a] = x0[a] - 0.02 * da / na
            if nb > 1e-12:
                x0[b] = x0[b] - 0.02 * db / nb
            x0 = x0 + sigma * rng.standard_normal((n, d))
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        else:
            x0 = Xb + sigma * rng.standard_normal((n, d))
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)

        # annealed smooth
        try:
            if it % 2 == 0:
                x = smooth_opt(x0, 0.008, 120, gtol=1e-6)
                if time.time() > EXPLORE_END:
                    v = max_cos(x)
                    if v < global_val:
                        global_best, global_val = x, v
                    break
                x = smooth_opt(x, 0.0025, 180, gtol=1e-7)
            else:
                x = smooth_opt(x0, 0.005, 150, gtol=1e-6)
                if time.time() > EXPLORE_END:
                    v = max_cos(x)
                    if v < global_val:
                        global_best, global_val = x, v
                    break
                x = smooth_opt(x, 0.0012, 200, gtol=1e-7)
        except Exception:
            continue
        v = max_cos(x)
        if v < global_val:
            global_best, global_val = x.copy(), v
            # immediate short LP to jam new basin
            try:
                xp = lp_polish(x, min(time.time() + 1.5, EXPLORE_END))
                vp = max_cos(xp)
                if vp < global_val:
                    global_best, global_val = xp, vp
            except Exception:
                pass
        else:
            # polish only if promising
            if (v < global_val + 0.004) and (it % 3 == 0):
                try:
                    xp = lp_polish(x, min(time.time() + 0.8, EXPLORE_END))
                    vp = max_cos(xp)
                    if vp < global_val:
                        global_best, global_val = xp, vp
                except Exception:
                    pass

    # final refine
    try:
        if time.time() < DEADLINE - 3:
            # fine smooth
            xf = smooth_opt(global_best, 0.0008, 350, gtol=1e-9)
            vf = max_cos(xf)
            if vf < global_val + 1e-6:
                # keep xf even if microscopically worse, LP may fix
                if vf < global_val:
                    global_best, global_val = xf, vf
                else:
                    # keep better but try LP from xf if close
                    if vf < global_val + 5e-6:
                        global_best_lp_seed = xf
                    else:
                        global_best_lp_seed = global_best
            else:
                global_best_lp_seed = global_best
            # final LP with remaining time
            rem = DEADLINE - time.time()
            if rem > 2:
                xp = lp_polish(global_best_lp_seed, DEADLINE - 1.0, r0=1e-3)
                vp = max_cos(xp)
                if vp < global_val:
                    global_best, global_val = xp, vp
                # try one more tiny-radius pass from global_best
                rem2 = DEADLINE - time.time()
                if rem2 > 2:
                    xp2 = lp_polish(global_best, DEADLINE - 0.5, r0=2e-4)
                    vp2 = max_cos(xp2)
                    if vp2 < global_val:
                        global_best, global_val = xp2, vp2
    except Exception:
        pass

    global_best = global_best / np.linalg.norm(global_best, axis=1, keepdims=True)
    return global_best
