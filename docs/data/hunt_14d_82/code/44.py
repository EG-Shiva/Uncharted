import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _adam_quench(X0, p_list, steps_list, lr_list, deadline, rng=None):
    X = _norm(X0.copy())
    n, d = X.shape
    b1, b2, eps = 0.9, 0.999, 1e-8
    for p, steps, lr in zip(p_list, steps_list, lr_list):
        m = np.zeros_like(X)
        v = np.zeros_like(X)
        t = 0
        for s in range(int(steps)):
            if time.time() >= deadline:
                return X
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            mval = float(np.max(G))
            S = p * (G - mval)
            np.clip(S, -40.0, 0.0, out=S)
            E = np.exp(S)
            np.fill_diagonal(E, 0.0)
            esum = float(E.sum())
            if not np.isfinite(esum) or esum <= 0:
                continue
            ssum = esum * 0.5
            W = E / ssum
            Gx = W @ X
            dot = np.sum(Gx * X, axis=1, keepdims=True)
            Gt = Gx - dot * X
            # clip huge grads
            t += 1
            m = b1 * m + (1 - b1) * Gt
            v = b2 * v + (1 - b2) * (Gt * Gt)
            mhat = m / (1 - b1 ** t)
            vhat = v / (1 - b2 ** t)
            step = lr * mhat / (np.sqrt(vhat) + eps)
            # limit step norm
            X = _norm(X - step)
    return X

def _lp_polish(X0, deadline, radius_init=2e-4, cap=900, max_it=10):
    X = _norm(X0.copy())
    n, d = X.shape
    best = _max_cos(X)
    Xbest = X.copy()
    radius = float(radius_init)
    iu = np.triu_indices(n, 1)
    it = 0
    while it < max_it and time.time() < deadline and radius > 1e-14:
        it += 1
        G = X @ X.T
        c_all = G[iu]
        m = float(np.max(c_all))
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            vals = c_all[idx]
            part = np.argpartition(-vals, cap - 1)[:cap]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[part]] = True
            sel = keep
        I = iu[0][sel]; J = iu[1][sel]
        c = c_all[sel]
        p_ = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p_), 2 * d + 1)
        cd = np.arange(d)
        cols = np.concatenate([I[:, None] * d + cd, J[:, None] * d + cd, np.full((p_, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p_, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p_, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        Y = _norm(X + res.x[:-1].reshape(n, d))
        vv = _max_cos(Y)
        if vv < best:
            X, best = Y, vv
            Xbest = Y.copy()
            radius = min(radius * 1.5, 8e-3)
        else:
            radius *= 0.5
    return Xbest, best

def solve():
    t0 = time.time()
    DEADLINE = t0 + 132.0
    rng = np.random.default_rng(1)
    n, d = 82, 14
    xb = None
    bv = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < bv:
                        bv = v; xb = a.copy()
        except Exception:
            continue
    if xb is None:
        xb = _norm(rng.standard_normal((n, d)))
        bv = _max_cos(xb)
    X_best = xb.copy()
    best_val = float(bv)
    iu0 = np.triu_indices(n, 1)
    I0 = iu0[0]; J0 = iu0[1]

    def remain_deadline():
        return time.time() < DEADLINE

    # Phase A: refine incumbent in place (no topology change yet)
    try:
        if remain_deadline():
            Xq = _adam_quench(X_best, [50.0, 150.0], [250, 250], [0.008, 0.004], min(DEADLINE, time.time() + 8.0))
            vq = _max_cos(Xq)
            if vq < best_val:
                best_val = vq; X_best = Xq.copy()
    except Exception:
        pass
    try:
        if time.time() + 4.0 < DEADLINE:
            Xp, vp = _lp_polish(X_best, min(DEADLINE, time.time() + 6.0), radius_init=2e-4, cap=1200, max_it=10)
            if vp < best_val:
                best_val = vp; X_best = Xp.copy()
    except Exception:
        pass

    sigmas = [5e-4, 1e-3, 2e-3, 5e-3, 1.2e-2, 2.5e-2]
    it = 0
    # Phase B: basin hopping
    while time.time() + 6.0 < DEADLINE and it < 300:
        it += 1
        try:
            Xt = X_best.copy()
            mode = it % 6
            if mode <= 3:
                sig = sigmas[(it - 1) % len(sigmas)]
                # every 12th: bigger global kick for topology change
                if it % 12 == 0:
                    sig = 3.0e-2
                Xt = _norm(Xt + sig * rng.standard_normal((n, d)))
                # occasionally explicitly separate current max pair
                if it % 4 == 0:
                    G = Xt @ Xt.T
                    np.fill_diagonal(G, -2.0)
                    k = int(np.argmax(G[I0, J0]))
                    # map flat upper index to pair: use argmax on full
                    # find pair via search
                    ii = int(I0[k]); jj = int(J0[k])
                    # push apart
                    push = 2e-3 + sig * 0.5
                    di = Xt[ii] - Xt[jj]
                    di = di - float(di @ Xt[ii]) * Xt[ii]
                    dj = Xt[jj] - Xt[ii]
                    dj = dj - float(dj @ Xt[jj]) * Xt[jj]
                    ni = float(np.linalg.norm(di)); nj = float(np.linalg.norm(dj))
                    if ni > 1e-12:
                        Xt[ii] = _norm((Xt[ii] + (push / ni) * di)[None, :])[0]
                    if nj > 1e-12:
                        Xt[jj] = _norm((Xt[jj] + (push / nj) * dj)[None, :])[0]
            else:
                # far-hole reinsertion of worst point(s)
                G = Xt @ Xt.T
                np.fill_diagonal(G, -9.0)
                rowmax = G.max(axis=1)
                iw = int(np.argmax(rowmax))
                # second worst occasionally
                cand_pool = rng.standard_normal((2500, d))
                cand_pool = cand_pool / np.linalg.norm(cand_pool, axis=1, keepdims=True)
                # score by max dot to others
                Xo = Xt[np.arange(n) != iw]
                dots = cand_pool @ Xo.T
                scores = dots.max(axis=1)
                ib = int(np.argmin(scores))
                Xt[iw] = cand_pool[ib]
                if mode == 5:
                    # reinsert second point too
                    G2 = Xt @ Xt.T
                    np.fill_diagonal(G2, -9.0)
                    rowmax2 = G2.max(axis=1)
                    rowmax2[iw] = -9.0
                    iw2 = int(np.argmax(rowmax2))
                    cand2 = rng.standard_normal((1500, d))
                    cand2 = cand2 / np.linalg.norm(cand2, axis=1, keepdims=True)
                    Xo2 = Xt[np.arange(n) != iw2]
                    s2 = (cand2 @ Xo2.T).max(axis=1)
                    Xt[iw2] = cand2[int(np.argmin(s2))]
                Xt = _norm(Xt)
            # quick quench
            qdead = min(DEADLINE - 4.0, time.time() + 1.2)
            if qdead <= time.time():
                break
            # anneal: moderate then sharp
            Xq = _adam_quench(Xt, [60.0, 220.0], [180, 180], [0.010, 0.004], qdead)
            vq = _max_cos(Xq)
            if vq < best_val - 1e-12:
                # squeeze new basin
                try:
                    pdead = min(DEADLINE - 2.0, time.time() + 2.5)
                    Xp, vp = _lp_polish(Xq, pdead, radius_init=1.5e-4, cap=900, max_it=7)
                    if vp < vq:
                        Xq, vq = Xp, vp
                except Exception:
                    pass
                if vq < best_val:
                    best_val = vq; X_best = Xq.copy()
            elif vq < best_val + 2e-4 and it % 20 == 0:
                # occasional polish of near-best to explore shell
                try:
                    pdead = min(DEADLINE - 2.0, time.time() + 1.0)
                    Xp, vp = _lp_polish(Xq, pdead, radius_init=1e-4, cap=700, max_it=4)
                    if vp < best_val:
                        best_val = vp; X_best = Xp.copy()
                except Exception:
                    pass
        except Exception:
            continue

    # Final polish of best
    try:
        if time.time() + 2.0 < DEADLINE:
            Xp, vp = _lp_polish(X_best, min(DEADLINE, time.time() + 7.0), radius_init=1.5e-4, cap=1500, max_it=12)
            if vp < best_val and np.all(np.isfinite(Xp)):
                best_val = vp; X_best = Xp.copy()
    except Exception:
        pass
    try:
        if time.time() + 1.0 < DEADLINE and best_val > 1e-9:
            # tiny Adam touch to remove numerical tie excess
            Xq = _adam_quench(X_best, [300.0], [120], [0.002], min(DEADLINE, time.time() + 2.0))
            vq = _max_cos(Xq)
            if vq < best_val:
                best_val = vq; X_best = Xq.copy()
    except Exception:
        pass

    X_best = _norm(np.asarray(X_best, dtype=float))
    # safety: replace any bad row
    nr = np.linalg.norm(X_best, axis=1)
    bad = ~(np.isfinite(nr)) | (nr < 1e-12)
    if np.any(bad):
        idx = np.where(bad)[0]
        for i in idx:
            v = rng.standard_normal(d); v /= np.linalg.norm(v)
            X_best[i] = v
    X_best = _norm(X_best)
    return X_best
