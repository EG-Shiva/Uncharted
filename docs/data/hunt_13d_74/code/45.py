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

def _degrees(X, thr=1e-7):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    C = G > (m - thr)
    # exclude diagonal (already -2)
    deg = C.sum(axis=1).astype(int)
    return deg, float(m), G

def _smooth_opt_full(X_init, tau, maxiter, gtol=1e-7):
    n, d = X_init.shape
    iu0, iu1 = np.triu_indices(n, 1)
    def fun(yf):
        Y = yf.reshape(n, d)
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
        grad = (Wx - X * dot) / r
        return float(F), grad.ravel()
    y0 = X_init.ravel().copy()
    try:
        res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-14, "gtol": gtol, "maxfun": 20000})
        Y = res.x.reshape(n, d)
    except Exception:
        return X_init
    Y = _norm(Y)
    if not np.all(np.isfinite(Y)):
        return X_init
    return Y

def _smooth_opt_free(X_base, free_idx, tau, maxiter, gtol=1e-7):
    n, d = X_base.shape
    free_idx = np.asarray(free_idx, dtype=int)
    k = len(free_idx)
    if k == 0 or k == n:
        return _smooth_opt_full(X_base, tau, maxiter, gtol)
    g2l = np.full(n, -1, dtype=int)
    g2l[free_idx] = np.arange(k)
    iset = np.zeros(n, dtype=bool)
    iset[free_idx] = True
    iu0, iu1 = np.triu_indices(n, 1)
    pmask = iset[iu0] | iset[iu1]
    P0 = iu0[pmask]; P1 = iu1[pmask]
    Xf = X_base.copy()
    def fun(yf):
        Y = yf.reshape(k, d)
        r = np.linalg.norm(Y, axis=1, keepdims=True)
        r = np.maximum(r, 1e-12)
        Xs = Y / r
        Xc = Xf.copy()
        Xc[free_idx] = Xs
        G = Xc @ Xc.T
        g = G[P0, P1]
        mmax = g.max()
        e = np.exp((g - mmax) / tau)
        S = e.sum()
        F = mmax + tau * np.log(S + 1e-300)
        w = e / (S + 1e-300)
        Wx = np.zeros_like(Xc)
        np.add.at(Wx, P0, w[:, None] * Xc[P1])
        np.add.at(Wx, P1, w[:, None] * Xc[P0])
        Ws = Wx[free_idx]
        dot = np.sum(Ws * Xs, axis=1, keepdims=True)
        grad = (Ws - Xs * dot) / r
        return float(F), grad.ravel()
    y0 = X_base[free_idx].ravel().copy()
    try:
        res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-14, "gtol": gtol, "maxfun": 15000})
        Y = res.x.reshape(k, d)
    except Exception:
        return X_base
    Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    Xn = X_base.copy()
    Xn[free_idx] = Y
    Xn = _norm(Xn)
    if not np.all(np.isfinite(Xn)):
        return X_base
    return Xn

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
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
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
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    TRIAL_END = t0 + 120.0
    rng = np.random.default_rng(20261102)
    n, d = 74, 13
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=float)
            if a.shape == (n, d):
                cands.append(_norm(a))
        except Exception:
            pass
    if not cands:
        cands.append(_norm(rng.standard_normal((n, d))))
    vals = [_max_cos(a) for a in cands]
    best = cands[int(np.argmin(vals))].copy()
    best_val = float(min(vals))
    # light initial jam to get contact structure
    try:
        bj = _lp_polish(best, min(time.time() + 4.0, TRIAL_END))
        vj = _max_cos(bj)
        if vj < best_val:
            best, best_val = bj, vj
    except Exception:
        pass

    def best_void(frozen_X, n_samp=20000):
        cand = _norm(rng.standard_normal((n_samp, d)))
        dots = cand @ frozen_X.T
        md = dots.max(axis=1)
        b = int(np.argmin(md))
        return cand[b], float(md[b])

    def perturb_rad(x, lo=0.05, hi=0.20):
        ang = float(rng.uniform(lo, hi))
        t = rng.standard_normal(d)
        t = t - float(t @ x) * x
        nt = np.linalg.norm(t)
        if nt < 1e-12:
            t = rng.standard_normal(d)
            t = t - float(t @ x) * x
            nt = np.linalg.norm(t)
            if nt < 1e-12:
                return _norm(x[None, :])[0]
        t = t / nt
        y = np.cos(ang) * x + np.sin(ang) * t
        return y / max(np.linalg.norm(y), 1e-12)

    # build surgery list
    deg, mmax, G = _degrees(best, thr=1e-7)
    rattlers = list(np.where(deg == 0)[0])
    hubs = list(np.where(deg >= 7)[0])
    if len(rattlers) == 0:
        # fallback: 8 loosest points
        Gm = G.copy()
        rowmax = Gm.max(axis=1)
        rattlers = list(np.argsort(rowmax)[:8])
    if len(hubs) == 0:
        rowmax = G.max(axis=1)
        hubs = list(np.argsort(-rowmax)[:8])
    # hub neighbour with highest contact count to hub
    # precompute contact adjacency
    C = G > (mmax - 1e-7)
    np.fill_diagonal(C, False)
    trials = []
    for h in hubs:
        for r in rattlers:
            trials.append((int(h), int(r), False))
    # double moves: hub + its highest-contact neighbour
    for h in hubs:
        nbrs = np.where(C[h])[0]
        if len(nbrs) == 0:
            # nearest by cosine
            row = G[h].copy()
            row[h] = -2
            nb = int(np.argmax(row))
        else:
            # pick neighbour with max degree
            nb = int(nbrs[np.argmax(deg[nbrs])])
        for r in rattlers:
            trials.append((int(h), int(r), int(nb)))
    # shuffle deterministic
    order = rng.permutation(len(trials))
    trials = [trials[i] for i in order]
    # cap trials
    trials = trials[:60]

    archive = {}
    deg_seq_best = tuple(sorted(deg.tolist()))
    archive[deg_seq_best] = best_val

    for ti, tr in enumerate(trials):
        if time.time() > TRIAL_END - 3.0:
            break
        if len(tr) == 3 and tr[2] is False:
            h, r = tr[0], tr[1]
            dbl = None
        else:
            h, r, dbl = tr[0], tr[1], tr[2]
            if dbl == h or dbl == r:
                dbl = None
        moving = [h] if dbl is None else [h, dbl]
        # frozen background = all except moving
        mask = np.ones(n, dtype=bool)
        mask[moving] = False
        frozen = best[mask]
        try:
            v0, _ = best_void(frozen, n_samp=20000)
        except Exception:
            continue
        x0 = best.copy()
        # transplant hub to void perturbed
        y1 = perturb_rad(v0, 0.05, 0.20)
        x0[h] = y1
        if dbl is not None:
            # second void: score excluding new hub? use second best direction offset
            # create orthogonal offset from y1 within tangent of v0
            y2 = perturb_rad(v0, 0.05, 0.20)
            # ensure not too close to y1
            for _ in range(3):
                if float(y2 @ y1) < 0.95:
                    break
                y2 = perturb_rad(v0, 0.05, 0.20)
            x0[dbl] = y2
        x0 = _norm(x0)
        free_idx = [i for i in range(n) if i not in moving]
        try:
            # stage 1: frozen transplant, annealed
            x1 = _smooth_opt_free(x0, free_idx, tau=1e-3, maxiter=60, gtol=1e-6)
            if time.time() > TRIAL_END:
                v = _max_cos(x1)
                if v < best_val:
                    best, best_val = x1, v
                break
            # stage 2: unfreeze, polish
            x2 = _smooth_opt_full(x1, tau=3e-4, maxiter=70, gtol=1e-7)
            if time.time() > TRIAL_END:
                v = _max_cos(x2)
                if v < best_val:
                    best, best_val = x2, v
                break
            x3 = _smooth_opt_full(x2, tau=8e-5, maxiter=80, gtol=1e-7)
        except Exception:
            continue
        try:
            x3p = _lp_polish(x3, min(time.time() + 1.2, TRIAL_END + 2.0))
        except Exception:
            x3p = x3
        v3 = _max_cos(x3p)
        try:
            d2, _, _ = _degrees(x3p, thr=1e-7)
            seq = tuple(sorted(d2.tolist()))
        except Exception:
            seq = None
        is_new = (seq is not None and seq not in archive)
        near = (v3 < best_val + 1e-6)
        if seq is not None and seq not in archive:
            archive[seq] = v3
        # only spend final polish on novel degree sequence or near-best
        if is_new or near:
            try:
                # extra 1-2s LP + tiny LBFGS if time allows and promising
                if v3 < best_val + 1e-6 and time.time() < TRIAL_END:
                    x4 = _lp_polish(x3p, min(time.time() + 1.0, DEAD - 8.0))
                    v4 = _max_cos(x4)
                    if v4 < v3:
                        x3p, v3 = x4, v4
                        try:
                            d2b, _, _ = _degrees(x3p, thr=1e-7)
                            seqb = tuple(sorted(d2b.tolist()))
                            if seqb not in archive:
                                archive[seqb] = v3
                        except Exception:
                            pass
            except Exception:
                pass
        if v3 < best_val:
            best, best_val = x3p, v3
            # refresh hub/rattler lists occasionally from new incumbent
            if ti % 8 == 7:
                try:
                    degN, mN, GN = _degrees(best, thr=1e-7)
                    deg_seq_best = tuple(sorted(degN.tolist()))
                except Exception:
                    pass
    # final intensive polish on incumbent
    try:
        remain = DEAD - time.time()
        if remain > 4.0:
            best = _smooth_opt_full(best, tau=1.2e-4, maxiter=150, gtol=1e-9)
            best_val = _max_cos(best)
    except Exception:
        pass
    try:
        remain = DEAD - time.time()
        if remain > 3.0:
            best = _lp_polish(best, DEAD - 1.0)
            best_val = _max_cos(best)
    except Exception:
        pass
    return _norm(best)
