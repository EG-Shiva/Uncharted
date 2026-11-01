import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _per_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _quench(X, steps, beta0, beta1, lr0, lr1, mu=0.85):
    vel = np.zeros_like(X)
    for s in range(steps):
        t = s / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** t
        lr = lr0 + (lr1 - lr0) * t
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        Tg = Grad - np.sum(Grad * X, axis=1, keepdims=True) * X
        rms = float(np.sqrt(np.mean(Tg * Tg))) + 1e-12
        Tn = Tg / rms
        vel = mu * vel + (1.0 - mu) * Tn
        vc = vel / (1.0 - mu ** (s + 1))
        X = X - lr * vc
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _make_full_fg(n, d, IU, beta):
    def fg(yf):
        Y = yf.reshape(n, d)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = Y / nr
        g = (X @ X.T)[IU]
        m = float(g.max())
        e = np.exp(beta * (g - m))
        S = float(e.sum()) + 1e-300
        F = m + np.log(S) / beta
        w = e / S
        P = np.zeros((n, n))
        P[IU] = w
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return F, GY.ravel()
    return fg

def _run_full(X0, beta, maxiter):
    n, d = X0.shape
    IU = np.triu_indices(n, 1)
    fg = _make_full_fg(n, d, IU, beta)
    y0 = (X0 / np.linalg.norm(X0, axis=1, keepdims=True)).ravel().copy()
    try:
        r = minimize(fg, y0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-7, 'maxfun': maxiter * 2})
        Y = r.x.reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    except Exception:
        return X0

def _make_sub_fg(C, K, d, beta):
    m = C.shape[0]
    def fg(yf):
        Y = yf.reshape(K, d)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = Y / nr
        Gac = X @ C.T
        Gbb = X @ X.T
        iu = np.triu_indices(K, 1)
        gbb = Gbb[iu] if K > 1 else np.zeros(0)
        gac = Gac.ravel()
        m0 = float(gac.max())
        if gbb.size:
            m0 = max(m0, float(gbb.max()))
        eac = np.exp(beta * (gac - m0))
        ebb = np.exp(beta * (gbb - m0)) if gbb.size else np.zeros(0)
        S = float(eac.sum() + ebb.sum()) + 1e-300
        F = m0 + np.log(S) / beta
        wac = (eac / S).reshape(K, m)
        GX = wac @ C
        if gbb.size:
            wbb = np.zeros((K, K))
            wbb[iu] = ebb / S
            wbb = wbb + wbb.T
            GX = GX + wbb @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return F, GY.ravel()
    return fg

def _run_sub(Y0, C, beta, maxiter):
    K, d = Y0.shape
    fg = _make_sub_fg(C, K, d, beta)
    y0 = (Y0 / np.linalg.norm(Y0, axis=1, keepdims=True)).ravel().copy()
    try:
        r = minimize(fg, y0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-7, 'maxfun': maxiter * 2})
        Y = r.x.reshape(K, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    except Exception:
        return Y0

def _lp(X, seconds, deadline=None):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 7e-4
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start > seconds - 0.15:
            break
        if deadline is not None and time.time() > deadline - 0.4:
            break
        if radius < 1e-14:
            break
        c = (X @ X.T)[iu]
        mm = float(c.max())
        win = max(20.0 * radius, 1e-9)
        sel = c > mm - win
        if int(sel.sum()) == 0:
            radius *= 0.5
            continue
        if int(sel.sum()) > 2200:
            idx = np.where(sel)[0]
            od = np.argsort(-c[idx])[:2200]
            keep = np.zeros_like(sel)
            keep[idx[od]] = True
            sel = keep
        I, J = iu[0][sel], iu[1][sel]
        csel = c[sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best - 1e-15:
            X, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            ok = False
            for sh in (0.3, 0.1):
                yy = X + res.x[:-1].reshape(n, d) * sh
                yy /= np.linalg.norm(yy, axis=1, keepdims=True)
                vv = _max_cos(yy)
                if vv < best:
                    X, best = yy, vv
                    ok = True
                    break
            if not ok:
                radius *= 0.5
    return X

def _fingerprint(X, k=120):
    n = X.shape[0]
    G = X @ X.T
    iu = np.triu_indices(n, 1)
    g = G[iu]
    idx = np.argsort(-g)[:k]
    return np.sort(g[idx])[::-1].copy()

def solve():
    t0 = time.time()
    TLIM = 172.0
    deadline = t0 + TLIM
    n, d = 68, 12
    rng = np.random.default_rng(12345)
    try:
        xb = np.load("best.npy").astype(float)
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        if xb.shape != (68, 12):
            raise ValueError()
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
            xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        except Exception:
            xb = rng.standard_normal((n, d))
            xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    x_best = xb.copy()
    cur = xb.copy()
    cur_val = best_val
    # pool for refill
    MPOOL = 120000
    pool = rng.standard_normal((MPOOL, d))
    pool /= np.linalg.norm(pool, axis=1, keepdims=True)
    tabu = [_fingerprint(x_best)]
    # temperature schedule
    T0, T1 = 0.0035, 0.00012
    max_trials = 60
    trial = 0
    while time.time() < deadline - 13.0 and trial < max_trials:
        frac = trial / max_trials
        Temp = T0 * (T1 / T0) ** frac
        # choose K and mode
        u = rng.random()
        if u < 0.65:
            K = int(rng.choice([7, 8, 9, 10, 11, 12, 13]))
        elif u < 0.85:
            K = int(rng.choice([4, 5, 6]))
        else:
            K = int(rng.choice([1, 2, 3]))
        # dense-cluster selection on cur: start from top loaded
        try:
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            pm = G.max(axis=1)
            # randomize start among top 8 to diversify
            top = np.argsort(-pm)[:8]
            start_idx = int(top[rng.integers(0, len(top))])
            selected = [start_idx]
            selmask = np.zeros(n, dtype=bool)
            selmask[start_idx] = True
            # greedily add point maximizing sum-dot to selected
            for _ in range(K - 1):
                # sum of G to selected
                s = G[:, selected].sum(axis=1)
                s[selmask] = -1e9
                nxt = int(np.argmax(s))
                selected.append(nxt)
                selmask[nxt] = True
            Didx = np.array(selected, dtype=int)
        except Exception:
            Didx = rng.choice(n, size=K, replace=False)
        mask = np.ones(n, dtype=bool)
        mask[Didx] = False
        C = cur[mask]
        # filter pool: scores to core
        try:
            chunk = 20000
            scores = np.empty(MPOOL, dtype=float)
            for s in range(0, MPOOL, chunk):
                e = min(s + chunk, MPOOL)
                scores[s:e] = (pool[s:e] @ C.T).max(axis=1)
            # keep best NFAR farthest
            NFAR = 6000
            far_idx = np.argpartition(scores, NFAR)[:NFAR]
            far_scores = scores[far_idx]
            far_vecs = pool[far_idx]
            # farthest-point sampling among far_vecs for K seeds
            order0 = int(np.argmin(far_scores))
            picked = [order0]
            # precompute dots among far_vecs lazily
            # maintain best mutual value
            best_mut = (far_vecs @ far_vecs[order0])
            Y0_list = [far_vecs[order0]]
            for _ in range(K - 1):
                vals = np.maximum(far_scores, best_mut)
                vals[picked] = 2.0
                nxt = int(np.argmin(vals))
                picked.append(nxt)
                Y0_list.append(far_vecs[nxt])
                best_mut = np.maximum(best_mut, far_vecs @ far_vecs[nxt])
            Y0 = np.stack(Y0_list, axis=0).copy()
            # small random kick to break symmetry
            Y0 = Y0 + 1e-3 * rng.standard_normal(Y0.shape)
            Y0 /= np.linalg.norm(Y0, axis=1, keepdims=True)
        except Exception:
            Y0 = rng.standard_normal((K, d))
            Y0 /= np.linalg.norm(Y0, axis=1, keepdims=True)
        # joint subset optimization
        try:
            Y = _run_sub(Y0, C, 120.0, 50)
            Y = _run_sub(Y, C, 400.0, 50)
            Y = _run_sub(Y, C, 1000.0, 40)
        except Exception:
            Y = Y0
        Xt = np.empty_like(cur)
        Xt[mask] = C
        Xt[~mask] = Y
        Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
        # light global shake
        try:
            Xt = _quench(Xt, 120, 60.0, 300.0, 0.03, 0.003, mu=0.85)
        except Exception:
            pass
        if time.time() < deadline - 11.0:
            try:
                Xt = _run_full(Xt, 280.0, 25)
            except Exception:
                pass
        rem = deadline - time.time()
        if rem > 9.0:
            try:
                Xt = _lp(Xt, min(1.2, rem - 8.0), deadline=deadline - 7.0)
            except Exception:
                pass
        v = _max_cos(Xt)
        # tabu distance
        try:
            fp = _fingerprint(Xt)
            dmin = min(float(np.max(np.abs(fp - t))) for t in tabu)
        except Exception:
            dmin = 1.0
        accept = False
        if v < cur_val - 1e-12:
            accept = True
        else:
            # sideways SA, penalize revisits
            pen = 0.0
            if dmin < 2e-5:
                pen = 0.0015
            gap = (v - cur_val) + pen
            if gap < 0:
                accept = True
            else:
                prob = float(np.exp(-gap / max(Temp, 1e-9)))
                if rng.random() < prob:
                    accept = True
        if accept:
            cur = Xt.copy()
            cur_val = v
            tabu.append(fp.copy() if 'fp' in dir() else _fingerprint(cur))
            if len(tabu) > 25:
                tabu.pop(0)
            if v < best_val - 1e-12:
                x_best = Xt.copy()
                best_val = v
        trial += 1
        # refresh slice of pool for diversity
        if trial % 4 == 0 and time.time() < deadline - 13.0:
            try:
                ns = 15000
                ii = rng.choice(MPOOL, size=ns, replace=False)
                nw = rng.standard_normal((ns, d))
                nw /= np.linalg.norm(nw, axis=1, keepdims=True)
                pool[ii] = nw
            except Exception:
                pass
    # final polish winner
    remain = deadline - time.time()
    if remain > 5.0:
        try:
            x_best = _run_full(x_best, 700.0, 60)
        except Exception:
            pass
    remain = deadline - time.time()
    if remain > 4.0:
        try:
            x_best = _lp(x_best, min(remain - 0.6, 9.0), deadline=deadline - 0.3)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # safety: return stored best if we degraded
    try:
        xb0 = np.load("best.npy").astype(float)
        xb0 = xb0 / np.linalg.norm(xb0, axis=1, keepdims=True)
        if _max_cos(xb0) < _max_cos(x_best):
            x_best = xb0
    except Exception:
        pass
    return x_best
