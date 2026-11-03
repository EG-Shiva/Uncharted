import time
import numpy as np

_TRIU = None

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _per_point(X):
    G = X @ X.T
    np.fill_diagonal(G, -np.inf)
    per = G.max(axis=1)
    return G, per

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

def _refine_hole(X_rem, v0, beta=130.0, maxiter=60):
    from scipy.optimize import minimize
    def fg(Yf):
        nr = float(np.linalg.norm(Yf)) + 1e-30
        x = Yf / nr
        dots = X_rem @ x
        m = float(dots.max())
        e = np.exp(beta * (dots - m))
        s = float(e.sum()) + 1e-300
        L = m + np.log(s) / beta
        P = e / s
        g = X_rem.T @ P
        gd = float(g @ x)
        GY = (g - gd * x) / nr
        return L, GY
    try:
        res = minimize(fg, v0.copy(), jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8})
        v = res.x
    except Exception:
        v = v0.copy()
    v /= (float(np.linalg.norm(v)) + 1e-30)
    return v

def _hole_vector(X_rem, rng, n_sample=8000):
    n, d = X_rem.shape
    C = rng.standard_normal((n_sample, d))
    C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-30)
    scores = (C @ X_rem.T).max(axis=1)
    order = np.argsort(scores)
    cand_idx = []
    for idx in order:
        if len(cand_idx) >= 5:
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
    for idx in cand_idx[:3]:
        v0 = C[idx].copy()
        for beta in (70.0, 150.0):
            v0 = _refine_hole(X_rem, v0, beta=beta, maxiter=50)
        s = float((X_rem @ v0).max())
        if s < best_s:
            best_s = s
            best_v = v0
    if best_v is None:
        best_v = C[order[0]].copy()
        best_v /= np.linalg.norm(best_v)
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
    iu0, iu1 = np.triu_indices(n, 1)
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
        if cnt > 2800:
            idx = np.argpartition(-c, 2800)[:2800]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
        I = iu0[sel]; J = iu1[sel]
        p = int(sel.sum())
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
    rng = np.random.default_rng(12345)
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
    cur_X = best_X.copy()
    cur_v = best_v

    hopping_deadline = t0 + BUDGET - 30.0
    trial = 0
    temp = 0.0022
    while time.time() < hopping_deadline and trial < 400:
        trial += 1
        frac = (time.time() - t0) / max(1.0, (hopping_deadline - t0))
        cur_temp = temp * (1.0 - 0.7 * frac)
        r = rng.random()
        cand = cur_X.copy()
        if r < 0.34:
            # kill worst / overloaded 1-3 and refill holes sequentially
            G, per = _per_point(cand)
            # near-contact overload count
            thr = float(per.max()) - 0.004
            near_cnt = np.sum(G > thr, axis=1)
            tight_thr = best_v - 1e-7
            # score = per + overload bonus
            score = per + 0.0006 * near_cnt
            # sometimes target pure overload
            if rng.random() < 0.35:
                score = near_cnt.astype(float) + per * 2.0
            k = 1 if rng.random() < 0.5 else (2 if rng.random() < 0.7 else 3)
            worst = np.argsort(-score)[:k].copy()
            if rng.random() < 0.15:
                worst[0] = int(rng.integers(0, n))
            # include rattler occasionally: point with min per
            if rng.random() < 0.20:
                ratt = int(np.argmin(per))
                worst[-1] = ratt
            mask = np.ones(n, dtype=bool)
            mask[worst] = False
            tmp_rem = cand[mask]
            new_vs = []
            for _ in range(k):
                hv = _hole_vector(tmp_rem, rng, n_sample=8000)
                new_vs.append(hv)
                tmp_rem = np.vstack([tmp_rem, hv[None, :]])
            for wi, hv in zip(worst, new_vs):
                cand[wi] = hv
            try:
                cand = _schedule(cand, [70.0, 150.0], [60, 60])
            except Exception:
                pass
        elif r < 0.62:
            # contact-graph cluster shake around worst
            G, per = _per_point(cand)
            w = int(np.argmax(per))
            if rng.random() < 0.3:
                # pick overloaded hub
                thr = float(per.max()) - 0.004
                near_cnt = np.sum(G > thr, axis=1)
                w = int(np.argmax(near_cnt + rng.random(n) * 0.5))
            order = np.argsort(-G[w])
            s = int(rng.integers(7, 16))
            idx = order[:s].copy()
            if w not in idx and rng.random() < 0.7:
                idx = np.append(idx, w)
            sigma = float(rng.uniform(0.10, 0.32))
            cand[idx] += sigma * rng.standard_normal((len(idx), d))
            cand[idx] /= (np.linalg.norm(cand[idx], axis=1, keepdims=True) + 1e-30)
            try:
                cand = _schedule(cand, [60.0, 140.0], [60, 60])
            except Exception:
                pass
        elif r < 0.78:
            # rattler / low-contact relocation
            G, per = _per_point(cand)
            m = float(per.max())
            tight = G > (m - 1e-7)
            tight_cnt = np.sum(tight, axis=1)
            # points with few contacts
            low = np.argsort(tight_cnt)[:4]
            pick = int(low[int(rng.integers(0, len(low)))]) if len(low) > 0 else int(np.argmin(per))
            mask = np.ones(n, dtype=bool)
            mask[pick] = False
            hv = _hole_vector(cand[mask], rng, n_sample=8000)
            # small blend to keep some continuity 10% of time
            if rng.random() < 0.15:
                hv = hv + 0.15 * rng.standard_normal(d)
                hv /= np.linalg.norm(hv)
            cand[pick] = hv
            # jitter its new neighbours a bit
            G2 = cand @ cand.T
            np.fill_diagonal(G2, -np.inf)
            nn = np.argsort(-G2[pick])[:4]
            cand[nn] += 0.03 * rng.standard_normal((len(nn), d))
            cand[nn] /= (np.linalg.norm(cand[nn], axis=1, keepdims=True) + 1e-30)
            try:
                cand = _schedule(cand, [70.0, 150.0], [60, 70])
            except Exception:
                pass
        else:
            sigma = float(rng.uniform(0.006, 0.04))
            cand = cand + sigma * rng.standard_normal(cand.shape)
            cand /= (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-30)
            try:
                cand = _schedule(cand, [70.0, 140.0], [55, 55])
            except Exception:
                pass
        v = _max_cos(cand)
        if v < best_v - 1e-12:
            # polish improvement immediately with short LP
            lp_t = 3.0 if (hopping_deadline - time.time()) > 12 else 1.5
            try:
                cand2 = _lp_polish(cand.copy(), lp_t)
                v2 = _max_cos(cand2)
            except Exception:
                cand2, v2 = cand, v
            if v2 < best_v:
                best_v, best_X = v2, cand2
                cur_X, cur_v = cand2.copy(), v2
            else:
                best_v, best_X = v, cand
                cur_X, cur_v = cand.copy(), v
        else:
            # Metropolis move for exploration
            if v < cur_v + cur_temp and rng.random() < np.exp(-(v - cur_v) / max(cur_temp, 1e-9)):
                cur_X, cur_v = cand.copy(), v
            elif v < best_v + 0.0015 and trial % 3 == 0:
                if hopping_deadline - time.time() > 7:
                    try:
                        cand2 = _lp_polish(cand.copy(), 1.5)
                        v2 = _max_cos(cand2)
                        if v2 < best_v - 1e-12:
                            best_v, best_X = v2, cand2
                            cur_X, cur_v = cand2.copy(), v2
                        elif v2 < cur_v:
                            cur_X, cur_v = cand2.copy(), v2
                    except Exception:
                        pass
        if time.time() >= hopping_deadline:
            break

    remain = BUDGET - (time.time() - t0)
    if remain > 10:
        try:
            tmp = _schedule(best_X.copy(), [110.0, 250.0, 500.0], [100, 100, 80])
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
