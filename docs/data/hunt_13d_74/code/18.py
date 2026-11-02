import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _smooth_opt(X_init, tau, maxiter, gtol=1e-7):
    n, d = X_init.shape
    iu0, iu1 = np.triu_indices(n, 1)
    def fun(y_flat):
        Y = y_flat.reshape(n, d)
        r = np.linalg.norm(Y, axis=1, keepdims=True)
        r = np.maximum(r, 1e-12)
        X = Y / r
        G = X @ X.T
        g = G[iu0, iu1]
        mmax = g.max()
        e = np.exp((g - mmax) / tau)
        S = e.sum()
        F = mmax + tau * np.log(S + 1e-300)
        w = e / (S + 1e-300)
        Wx = np.zeros_like(X)
        np.add.at(Wx, iu0, w[:, None] * X[iu1])
        np.add.at(Wx, iu1, w[:, None] * X[iu0])
        dot = np.sum(Wx * X, axis=1, keepdims=True)
        grad_Y = (Wx - X * dot) / r
        return float(F), grad_Y.ravel()
    y0 = X_init.ravel().copy()
    try:
        res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-14, "gtol": gtol, "maxfun": 20000})
        Y = res.x.reshape(n, d)
    except Exception:
        return X_init
    nrm = np.linalg.norm(Y, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    X = Y / nrm
    if not np.all(np.isfinite(X)):
        return X_init
    return X

def _subset_opt(X_base, idx, tau, maxiter, rng):
    # optimize rows idx keeping others fixed, LogSumExp over pairs touching idx
    n, d = X_base.shape
    k = len(idx)
    idx = np.asarray(idx, dtype=int)
    iset = np.zeros(n, dtype=bool)
    iset[idx] = True
    iu0, iu1 = np.triu_indices(n, 1)
    pmask = iset[iu0] | iset[iu1]
    P0 = iu0[pmask]; P1 = iu1[pmask]
    # map global row -> position in subset (-1 if frozen)
    g2l = np.full(n, -1, dtype=int)
    g2l[idx] = np.arange(k)
    Xfrozen = X_base.copy()
    def fun(y_flat):
        Y = y_flat.reshape(k, d)
        r = np.linalg.norm(Y, axis=1, keepdims=True)
        r = np.maximum(r, 1e-12)
        Xs = Y / r
        Xc = Xfrozen.copy()
        Xc[idx] = Xs
        G = Xc @ Xc.T
        g = G[P0, P1]
        mmax = g.max()
        e = np.exp((g - mmax) / tau)
        S = e.sum()
        F = mmax + tau * np.log(S + 1e-300)
        w = e / (S + 1e-300)
        # gradient w.r.t Xc rows in idx
        Wx = np.zeros_like(Xc)
        np.add.at(Wx, P0, w[:, None] * Xc[P1])
        np.add.at(Wx, P1, w[:, None] * Xc[P0])
        # only subset rows matter
        Ws = Wx[idx]
        dot = np.sum(Ws * Xs, axis=1, keepdims=True)
        grad_Y = (Ws - Xs * dot) / r
        return float(F), grad_Y.ravel()
    y0 = X_base[idx].ravel().copy()
    try:
        res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-7, "maxfun": 10000})
        Y = res.x.reshape(k, d)
    except Exception:
        return X_base
    Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    Xn = X_base.copy()
    Xn[idx] = Y
    if not np.all(np.isfinite(Xn)):
        return X_base
    return _norm(Xn)

def _lp_polish(X, deadline, r0=1.1e-3):
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
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
        if cnt > 2701:
            idxp = np.argpartition(c_all, -2701)[-2701:]
            mask = np.zeros_like(sel)
            mask[idxp] = True
            sel = mask
            cnt = 2701
        I = iu0[sel]; J = iu1[sel]
        c = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        base_I = I[:, None] * d + np.arange(d)
        base_J = J[:, None] * d + np.arange(d)
        cols = np.concatenate([base_I, base_J, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
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
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _greedy(X, deadline, lr0=0.002):
    X = _norm(X)
    cur = _max_cos(X)
    lr = lr0
    n = X.shape[0]
    while time.time() < deadline:
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        delta = max(3.0 * lr, 5e-7)
        mask = G > (gmax - delta)
        A = mask.astype(np.float64)
        Gr = A @ X
        d = (Gr * X).sum(axis=1, keepdims=True)
        Gt = Gr - d * X
        norms = np.linalg.norm(Gt, axis=1, keepdims=True)
        step = np.zeros_like(Gt)
        act = norms[:, 0] > 1e-14
        if not np.any(act):
            break
        step[act] = Gt[act] / (norms[act] + 1e-12) * lr
        Xn = _norm(X - step)
        nv = _max_cos(Xn)
        if nv < cur:
            X[:] = Xn
            cur = nv
            lr *= 1.08
        else:
            lr *= 0.5
            if lr < 1e-11:
                break
    return X

def _worst_scores(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    EXPLORE_END = t0 + 146.0
    rng = np.random.default_rng(12345)
    n, d = 74, 13
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    for a in (xb, xr):
        if a is None:
            continue
        a = np.asarray(a, dtype=float)
        if a.shape == (n, d):
            a = _norm(a)
            cands.append(a)
    if not cands:
        cands.append(_norm(rng.standard_normal((n, d))))
    vals = [_max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    global_best = cands[bi].copy()
    global_val = float(vals[bi])
    # quick jam
    try:
        if time.time() < EXPLORE_END - 5:
            xj = _lp_polish(global_best, min(time.time() + 5.0, EXPLORE_END))
            vj = _max_cos(xj)
            if vj < global_val:
                global_best, global_val = xj, vj
    except Exception:
        pass
    sigmas = [0.004, 0.012, 0.022, 0.007, 0.035, 0.002, 0.05, 0.015]
    it = 0
    while time.time() < EXPLORE_END:
        it += 1
        remain = EXPLORE_END - time.time()
        if remain < 1.5:
            break
        mode = it % 10
        sigma = sigmas[it % len(sigmas)]
        Xb = global_best
        if mode == 8:
            x0 = _norm(rng.standard_normal((n, d)))
        elif mode in (3, 6):
            # multi-void reinsertion of worst k
            k = 5 if mode == 3 else 8
            ws = _worst_scores(Xb)
            order = np.argsort(-ws)
            victims = order[:k].copy()
            x0 = Xb.copy()
            # greedy farthest voids sequential
            cand = _norm(rng.standard_normal((3000, d)))
            taken = np.zeros(n, dtype=bool)
            taken[victims] = True
            # background without victims
            for t, widx in enumerate(victims):
                # dots to all non-victim + already placed victims
                dots = cand @ x0.T
                # ignore columns of not-yet-replaced victims (allow) -> set low for future victims, keep placed
                for v2 in victims[t+1:]:
                    dots[:, v2] = -10.0
                # also ignore self column? self will be replaced, set low
                dots[:, widx] = -10.0
                md = dots.max(axis=1)
                b = int(np.argmin(md))
                x0[widx] = cand[b]
            x0 = _norm(x0 + 0.003 * rng.standard_normal((n, d)))
            # subset refine the reinserted points with frozen background
            try:
                if time.time() < EXPLORE_END - 2:
                    x0 = _subset_opt(x0, victims, 0.004, 80, rng)
            except Exception:
                pass
        elif mode == 4:
            # single void best reinsertion
            ws = _worst_scores(Xb)
            w = int(np.argmax(ws))
            cand = _norm(rng.standard_normal((2500, d)))
            dots = cand @ Xb.T
            dots[:, w] = -2.0
            md = dots.max(axis=1)
            b = int(np.argmin(md))
            x0 = Xb.copy()
            x0[w] = cand[b]
            x0 = _norm(x0 + 0.002 * rng.standard_normal((n, d)))
        elif mode == 5:
            # worst-k heavy kick
            ws = _worst_scores(Xb)
            order = np.argsort(-ws)
            k = 8
            x0 = Xb.copy()
            noise = rng.standard_normal((n, d))
            scale = np.full((n, 1), sigma * 0.3)
            scale[order[:k]] = sigma * 2.2
            x0 = _norm(x0 + scale * noise)
        elif mode == 7:
            # push top-3 pairs apart tangentially + jitter
            G = Xb @ Xb.T
            np.fill_diagonal(G, -2.0)
            # get top 3 pairs
            flat = G[np.triu_indices(n, 1)]
            # find indices of largest 3
            if flat.size >= 3:
                top = np.argpartition(flat, -3)[-3:]
                iu0, iu1 = np.triu_indices(n, 1)
                x0 = Xb.copy()
                for tt in top:
                    a = int(iu0[tt]); b = int(iu1[tt])
                    c = float(G[a, b])
                    da = Xb[a] - c * Xb[b]
                    db = Xb[b] - c * Xb[a]
                    na = np.linalg.norm(da); nb = np.linalg.norm(db)
                    if na > 1e-12:
                        x0[a] = x0[a] - 0.025 * da / na
                    if nb > 1e-12:
                        x0[b] = x0[b] - 0.025 * db / nb
                x0 = _norm(x0 + sigma * rng.standard_normal((n, d)))
            else:
                x0 = _norm(Xb + sigma * rng.standard_normal((n, d)))
        else:
            x0 = _norm(Xb + sigma * rng.standard_normal((n, d)))
        # annealed smooth
        try:
            if it % 2 == 0:
                x = _smooth_opt(x0, 0.008, 110, gtol=1e-6)
                if time.time() >= EXPLORE_END:
                    v = _max_cos(x)
                    if v < global_val:
                        global_best, global_val = x, v
                    break
                x = _smooth_opt(x, 0.0022, 160, gtol=1e-7)
            else:
                x = _smooth_opt(x0, 0.005, 130, gtol=1e-6)
                if time.time() >= EXPLORE_END:
                    v = _max_cos(x)
                    if v < global_val:
                        global_best, global_val = x, v
                    break
                x = _smooth_opt(x, 0.0011, 180, gtol=1e-7)
        except Exception:
            continue
        v = _max_cos(x)
        if v < global_val:
            global_best, global_val = x.copy(), v
            try:
                xp = _lp_polish(x, min(time.time() + 1.2, EXPLORE_END))
                vp = _max_cos(xp)
                if vp < global_val:
                    global_best, global_val = xp, vp
            except Exception:
                pass
        else:
            if (v < global_val + 0.004) and (it % 3 == 0):
                try:
                    xp = _lp_polish(x, min(time.time() + 0.7, EXPLORE_END))
                    vp = _max_cos(xp)
                    if vp < global_val:
                        global_best, global_val = xp, vp
                except Exception:
                    pass
    # final refine
    try:
        if time.time() < DEADLINE - 4:
            xf = _smooth_opt(global_best, 0.0007, 300, gtol=1e-9)
            vf = _max_cos(xf)
            seed = global_best
            if vf < global_val:
                global_best, global_val = xf, vf
                seed = xf
            elif vf < global_val + 5e-6:
                seed = xf
            rem = DEADLINE - time.time()
            if rem > 3:
                xg = _greedy(seed, time.time() + min(3.0, rem - 2.0), lr0=0.001)
                vg = _max_cos(xg)
                if vg < global_val:
                    global_best, global_val = xg, vg
                    seed = xg
                rem2 = DEADLINE - time.time()
                if rem2 > 2:
                    xp = _lp_polish(seed, DEADLINE - 1.0, r0=8e-4)
                    vp = _max_cos(xp)
                    if vp < global_val:
                        global_best, global_val = xp, vp
                    rem3 = DEADLINE - time.time()
                    if rem3 > 1.2:
                        xp2 = _greedy(global_best, DEADLINE - 0.4, lr0=2e-4)
                        vp2 = _max_cos(xp2)
                        if vp2 < global_val:
                            global_best, global_val = xp2, vp2
    except Exception:
        pass
    global_best = _norm(global_best)
    return global_best
