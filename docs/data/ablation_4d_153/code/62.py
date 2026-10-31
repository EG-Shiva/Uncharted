import time
import numpy as np

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _row_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    rm = np.max(G, axis=1)
    return rm, float(np.max(rm))

def _smooth_opt(x, steps, tau0, tau1, lr0, lr1, mom=0.9):
    n, d = x.shape
    vel = np.zeros_like(x)
    lt0, lt1 = np.log(tau0), np.log(tau1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(1, steps - 1)
        tau = float(np.exp(lt0 + (lt1 - lt0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        row = np.sum(W, axis=1, keepdims=True)
        row[row < 1e-12] = 1e-12
        grad = (W @ x) / row
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        gn = np.linalg.norm(grad, axis=1, keepdims=True)
        big = gn > 2.0
        if np.any(big):
            grad = grad * np.where(big, 2.0 / np.maximum(gn, 1e-12), 1.0)
        vel = mom * vel + (1.0 - mom) * grad
        bc = 1.0 - mom ** (t + 1)
        step = vel / bc
        x = x - lr * step
        x = _norm(x)
    return x

def _lp_polish(x, deadline, cap=2000, r0=8e-4, maxit=150):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu0 = np.triu_indices(n, 1)
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < maxit:
        it += 1
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thresh)[0]
        if idx.size == 0:
            break
        if idx.size > cap:
            order = np.argpartition(c[idx], -cap)[-cap:]
            idx = idx[order]
        I = iu0[0][idx]; J = iu0[1][idx]
        csel = c[idx]
        p = idx.size
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = I[:, None] * d + np.arange(d)
        cJ = J[:, None] * d + np.arange(d)
        cols = np.concatenate([cI, cJ, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x = y; best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _partition(y, K, rng):
    n = y.shape[0]
    centers = rng.choice(n, size=K, replace=False)
    C = y[centers]
    S = y @ C.T
    labels = np.argmax(S, axis=1)
    return labels

def _cluster_rotate(y, labels, K, sigma, rng):
    z = y.copy()
    d = y.shape[1]
    I4 = np.eye(d)
    for k in range(K):
        mask = labels == k
        if not np.any(mask):
            continue
        Rn = rng.standard_normal((d, d))
        A = (Rn - Rn.T) * 0.5
        s = sigma * (0.5 + rng.random())
        try:
            Q = np.linalg.solve(I4 - s * A, I4 + s * A)
        except Exception:
            continue
        z[mask] = z[mask] @ Q.T
    return _norm(z)

def _cluster_translate(y, labels, K, sigma, rng):
    z = y.copy()
    for k in range(K):
        mask = labels == k
        if not np.any(mask):
            continue
        t = rng.standard_normal(4)
        t = t / max(np.linalg.norm(t), 1e-12) * sigma * (0.5 + rng.random())
        pts = z[mask]
        dot = np.sum(pts * t.reshape(1, -1), axis=1, keepdims=True)
        pts = pts + (t.reshape(1, -1) - dot * pts)
        z[mask] = pts
    return _norm(z)

def _tangent_kick(y, idx, sigma, rng):
    z = y.copy()
    k = len(idx)
    if k == 0:
        return z
    noise = rng.standard_normal(size=(k, y.shape[1])) * sigma
    base = z[idx]
    dot = np.sum(noise * base, axis=1, keepdims=True)
    noise = noise - dot * base
    z[idx] = _norm(base + noise)
    return z

def _greedy_fill(Y, k, rng, n_cand=6000):
    d = Y.shape[1]
    cur = Y.copy()
    for _ in range(k):
        C = rng.standard_normal(size=(n_cand, d))
        C = _norm(C)
        S = C @ cur.T
        sm = S.max(axis=1)
        bi = int(np.argmin(sm))
        cur = np.vstack([cur, C[bi:bi+1]])
    return cur

def solve():
    t0 = time.time()
    HARD_END = t0 + 173.0
    FINAL_RESERVE = 22.0
    EXPLORE_END = HARD_END - FINAL_RESERVE
    rng = np.random.default_rng(0)
    n, d = 153, 4
    best_x = None
    best_v = 1e9
    for fname in ("best.npy", "record.npy"):
        try:
            a = np.load(fname)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                a = _norm(a)
                v = _max_cos(a)
                if v < best_v:
                    best_v = v
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _norm(rng.standard_normal(size=(n, d)))
        best_v = _max_cos(best_x)
    try:
        best_x, best_v = _lp_polish(best_x, time.time() + 4.0, cap=2500, r0=1e-3, maxit=200)
    except Exception:
        pass
    global_best_x = best_x.copy()
    global_best_v = float(best_v)
    cur_x = global_best_x.copy()
    cur_v = global_best_v
    it = 0
    while time.time() < EXPLORE_END:
        it += 1
        if time.time() > EXPLORE_END - 1.0:
            break
        mode = it % 7
        base = cur_x if (it % 3 != 0) else global_best_x
        y = base.copy()
        try:
            if mode == 0:
                K = 2 + (it % 4)
                sigma = 0.06 + 0.03 * ((it // 7) % 3)
                labels = _partition(y, K, rng)
                y = _cluster_rotate(y, labels, K, sigma, rng)
                y = _smooth_opt(y, steps=320, tau0=0.012, tau1=0.0012, lr0=0.09, lr1=0.005)
            elif mode == 1:
                K = 3 + (it % 3)
                sigma = 0.05 + 0.02 * ((it // 7) % 3)
                labels = _partition(y, K, rng)
                y = _cluster_translate(y, labels, K, sigma, rng)
                y = _smooth_opt(y, steps=320, tau0=0.012, tau1=0.0012, lr0=0.09, lr1=0.005)
            elif mode == 2:
                rm, _ = _row_max(y)
                k = 4 + (it % 4)
                worst = np.argsort(-rm)[:max(k*3, 12)]
                sel = rng.choice(worst, size=k, replace=False)
                mask = np.ones(n, dtype=bool); mask[sel] = False
                Ym = y[mask]
                Ym = _smooth_opt(Ym, steps=120, tau0=0.012, tau1=0.002, lr0=0.08, lr1=0.006)
                y = _greedy_fill(Ym, k, rng, n_cand=6000)
                y = _smooth_opt(y, steps=380, tau0=0.015, tau1=0.0015, lr0=0.10, lr1=0.006)
            elif mode == 3:
                sigma = 0.05 + 0.03 * ((it // 7) % 2)
                noise = rng.standard_normal(size=(n, d)) * sigma
                dot = np.sum(noise * y, axis=1, keepdims=True)
                noise = noise - dot * y
                y = _norm(y + noise)
                y = _smooth_opt(y, steps=350, tau0=0.015, tau1=0.0015, lr0=0.10, lr1=0.006)
            elif mode == 4:
                rm, _ = _row_max(y)
                k = 8 + (it % 6)
                worst = np.argsort(-rm)[:k]
                if k // 2 > 0:
                    cand = rng.choice(n, size=k//2, replace=False)
                    worst[:k//2] = cand
                y = _tangent_kick(y, worst, 0.07, rng)
                y = _smooth_opt(y, steps=320, tau0=0.012, tau1=0.0012, lr0=0.09, lr1=0.005)
            elif mode == 5:
                K = 2
                labels = _partition(y, K, rng)
                y = _cluster_rotate(y, labels, K, 0.18, rng)
                y = _smooth_opt(y, steps=420, tau0=0.025, tau1=0.002, lr0=0.14, lr1=0.007)
            else:
                rm, _ = _row_max(y)
                rdel = int(np.argmax(rm))
                mask = np.ones(n, dtype=bool); mask[rdel] = False
                Ym = y[mask]
                C = _norm(rng.standard_normal(size=(8000, d)))
                S = C @ Ym.T
                sm = S.max(axis=1)
                bi = int(np.argmin(sm))
                y[rdel] = C[bi]
                y = _smooth_opt(y, steps=300, tau0=0.010, tau1=0.001, lr0=0.08, lr1=0.005)
        except Exception:
            continue
        try:
            v_smooth = _max_cos(y)
        except Exception:
            continue
        thresh = min(global_best_v + 0.004, cur_v + 0.0025)
        force = (it % 19 == 0)
        if (v_smooth < thresh) or force:
            if time.time() > EXPLORE_END - 1.0:
                if v_smooth < cur_v:
                    cur_x = y.copy(); cur_v = float(v_smooth)
                break
            try:
                budget = 0.9 if v_smooth < global_best_v + 0.001 else 0.5
                dl = min(time.time() + budget, EXPLORE_END)
                if dl <= time.time() + 0.05:
                    continue
                yp, vp = _lp_polish(y, dl, cap=1800, r0=7e-4, maxit=120)
            except Exception:
                continue
            if vp < global_best_v - 1e-12:
                global_best_v = float(vp)
                global_best_x = yp.copy()
                cur_x = yp.copy(); cur_v = float(vp)
            elif vp < cur_v:
                cur_x = yp.copy(); cur_v = float(vp)
            elif force and (vp < cur_v + 0.0015) and rng.random() < 0.3:
                cur_x = yp.copy(); cur_v = float(vp)
            if it % 25 == 0:
                cur_x = global_best_x.copy(); cur_v = global_best_v
        else:
            if v_smooth < cur_v:
                cur_x = y.copy(); cur_v = float(v_smooth)
    try:
        dl = HARD_END - 0.8
        if dl > time.time():
            global_best_x, global_best_v = _lp_polish(global_best_x, dl, cap=3000, r0=1e-3, maxit=300)
    except Exception:
        pass
    global_best_x = _norm(global_best_x)
    return global_best_x
