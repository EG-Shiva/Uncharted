import time
import numpy as np

_TRIU = None

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lbfgs_smooth(X0, beta, maxiter):
    from scipy.optimize import minimize
    n, d = X0.shape
    global _TRIU
    if _TRIU is None or _TRIU[0].shape[0] != n * (n - 1) // 2:
        _TRIU = np.triu_indices(n, 1)
    IU0, IU1 = _TRIU
    Y0 = X0.ravel().copy()

    def fun_grad(Yf):
        Y = Yf.reshape(n, d)
        nr = np.linalg.norm(Y, axis=1, keepdims=True) + 1e-30
        X = Y / nr
        G = X @ X.T
        g = G[IU0, IU1]
        m = float(g.max())
        e = np.exp(beta * (g - m))
        s = float(e.sum()) + 1e-300
        L = m + np.log(s) / beta
        P = e / s
        # symmetric weight matrix
        W = np.zeros((n, n))
        W[IU0, IU1] = P
        W = W + W.T
        GX = W @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return L, GY.ravel()

    try:
        res = minimize(fun_grad, Y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8})
        Y = res.x.reshape(n, d)
    except Exception:
        return X0 / (np.linalg.norm(X0, axis=1, keepdims=True) + 1e-30)
    Y /= (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-30)
    return Y

def _schedule(X, betas, iters):
    for b, it in zip(betas, iters):
        X = _lbfgs_smooth(X, b, it)
    return X

def _refine_hole(X_rem, v0, beta=120.0, maxiter=80):
    from scipy.optimize import minimize
    d = v0.shape[0]
    Y0 = v0.copy()
    def fg(Yf):
        nr = float(np.linalg.norm(Yf)) + 1e-30
        x = Yf / nr
        dots = X_rem @ x
        m = float(dots.max())
        e = np.exp(beta * (dots - m))
        s = float(e.sum()) + 1e-300
        L = m + np.log(s) / beta
        P = e / s
        g = X_rem.T @ P  # (d,)
        gd = float(g @ x)
        GY = (g - gd * x) / nr
        return L, GY
    try:
        res = minimize(fg, Y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8})
        v = res.x
    except Exception:
        v = Y0
    v /= (float(np.linalg.norm(v)) + 1e-30)
    return v

def _hole_vector(X_rem, rng, n_sample=6000):
    n, d = X_rem.shape
    C = rng.standard_normal((n_sample, d))
    C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-30)
    scores = (C @ X_rem.T).max(axis=1)
    order = np.argsort(scores)
    # take diverse top candidates
    cand_idx = []
    for idx in order:
        if len(cand_idx) >= 6:
            break
        v = C[idx]
        ok = True
        for j in cand_idx:
            if float(v @ C[j]) > 0.6:
                ok = False
                break
        if ok:
            cand_idx.append(int(idx))
    best_v = None
    best_s = 1e9
    for idx in cand_idx:
        v0 = C[idx].copy()
        for beta in (60.0, 150.0):
            v0 = _refine_hole(X_rem, v0, beta=beta, maxiter=60)
        s = float((X_rem @ v0).max())
        if s < best_s:
            best_s = s
            best_v = v0
    return best_v

def _lp_polish(X, seconds):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    try:
        best = _max_cos(X)
    except Exception:
        return X
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    iu0, iu1 = iu[0], iu[1]
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.8:
            break
        Gfull = X @ X.T
        c = Gfull[iu0, iu1]
        m = float(c.max())
        if m < best:
            best = m
        sel = c > m - max(20.0 * radius, 1e-9)
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 3000:
            thr = np.sort(c[sel])[-3000]
            sel = c >= thr
            cnt = int(sel.sum())
            if cnt > 3200:
                # subsample hardest
                idx = np.argpartition(-c, 3000)[:3000]
                mask = np.zeros_like(sel)
                mask[idx] = True
                sel = mask
        I = iu0[sel]; J = iu1[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        b_ub = -c[sel]
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= (np.linalg.norm(y, axis=1, keepdims=True) + 1e-30)
        v = _max_cos(y)
        if v < best - 1e-15:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    BUDGET = 175.0
    rng = np.random.default_rng(0)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            c = np.load(fn)
            c = np.array(c, dtype=float)
            c = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-30)
            if c.shape == (162, 8) and np.all(np.isfinite(c)):
                cands.append(c)
        except Exception:
            pass
    if len(cands) == 0:
        X0 = rng.standard_normal((162, 8))
        X0 /= (np.linalg.norm(X0, axis=1, keepdims=True) + 1e-30)
    else:
        X0 = min(cands, key=lambda c: _max_cos(c))
    n, d = 162, 8
    best_X = X0.copy()
    best_v = _max_cos(best_X)
    # initial jam: L-BFGS schedule
    try:
        tmp = _schedule(best_X.copy(), [60.0, 150.0], [120, 120])
        vt = _max_cos(tmp)
        if vt < best_v:
            best_v, best_X = vt, tmp
    except Exception:
        pass
    # quick LP to reach local optimum
    remain = BUDGET - (time.time() - t0)
    if remain > 40:
        try:
            tmp = _lp_polish(best_X.copy(), min(10.0, remain - 30.0))
            vt = _max_cos(tmp)
            if vt < best_v:
                best_v, best_X = vt, tmp
        except Exception:
            pass

    hopping_deadline = t0 + BUDGET - 28.0
    trial = 0
    while time.time() < hopping_deadline and trial < 200:
        trial += 1
        r = rng.random()
        cand = best_X.copy()
        if r < 0.42:
            # kill worst 1-3 and refill holes
            G = cand @ cand.T
            np.fill_diagonal(G, -np.inf)
            per = G.max(axis=1)
            k = 1 if rng.random() < 0.55 else (2 if rng.random() < 0.75 else 3)
            worst = np.argsort(-per)[:k].copy()
            if rng.random() < 0.25:
                worst[0] = int(rng.integers(0, n))
            mask = np.ones(n, dtype=bool)
            mask[worst] = False
            tmp_rem = cand[mask]
            new_vs = []
            for _ in range(k):
                hv = _hole_vector(tmp_rem, rng, n_sample=6000)
                new_vs.append(hv)
                tmp_rem = np.vstack([tmp_rem, hv[None, :]])
            for wi, hv in zip(worst, new_vs):
                cand[wi] = hv
            try:
                cand = _schedule(cand, [80.0, 160.0], [80, 80])
            except Exception:
                pass
        elif r < 0.72:
            # overload cluster shake around worst point
            G = cand @ cand.T
            np.fill_diagonal(G, -np.inf)
            per = G.max(axis=1)
            w = int(np.argmax(per))
            neigh = np.argsort(-G[w])[:10]
            s = int(rng.integers(6, 12))
            idx = neigh[:s] if s <= len(neigh) else rng.choice(n, size=s, replace=False)
            if rng.random() < 0.5 and w not in idx:
                idx = np.append(idx, w)
                s += 1
            sigma = float(rng.uniform(0.08, 0.28))
            cand[idx] += sigma * rng.standard_normal((len(idx), d))
            cand[idx] /= (np.linalg.norm(cand[idx], axis=1, keepdims=True) + 1e-30)
            try:
                cand = _schedule(cand, [70.0, 150.0], [90, 90])
            except Exception:
                pass
        else:
            # global light jitter
            sigma = float(rng.uniform(0.005, 0.035))
            cand = cand + sigma * rng.standard_normal(cand.shape)
            cand /= (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-30)
            try:
                cand = _schedule(cand, [80.0, 150.0], [70, 70])
            except Exception:
                pass
        v = _max_cos(cand)
        # direct improvement
        if v < best_v - 1e-12:
            # squeeze with short LP
            lp_t = 4.0 if (hopping_deadline - time.time()) > 12 else 2.0
            try:
                cand2 = _lp_polish(cand.copy(), lp_t)
                v2 = _max_cos(cand2)
            except Exception:
                cand2, v2 = cand, v
            if v2 < best_v:
                best_v, best_X = v2, cand2
            else:
                best_v, best_X = v, cand
        elif v < best_v + 0.002:
            # near miss: try very short LP, may dip below best
            if trial % 2 == 0 or v < best_v + 0.0006:
                if hopping_deadline - time.time() < 6:
                    continue
                try:
                    cand2 = _lp_polish(cand.copy(), 2.0)
                    v2 = _max_cos(cand2)
                    if v2 < best_v - 1e-12:
                        best_v, best_X = v2, cand2
                except Exception:
                    pass
        if best_v < 0.4920:
            pass

    remain = BUDGET - (time.time() - t0)
    if remain > 8:
        try:
            tmp = _schedule(best_X.copy(), [100.0, 250.0, 450.0], [120, 120, 100])
            vt = _max_cos(tmp)
            if vt < best_v:
                best_v, best_X = vt, tmp
        except Exception:
            pass
    remain = BUDGET - (time.time() - t0)
    lp_sec = min(26.0, max(4.0, remain - 2.0))
    if lp_sec > 3 and remain > 5:
        try:
            polished = _lp_polish(best_X.copy(), lp_sec)
            vp = _max_cos(polished)
            if vp < best_v:
                best_v, best_X = vp, polished
        except Exception:
            pass
    best_X = best_X / (np.linalg.norm(best_X, axis=1, keepdims=True) + 1e-30)
    return best_X
