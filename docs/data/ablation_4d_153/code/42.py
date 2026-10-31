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

def _degrees_rowmax(x, m=None, tol=1e-6):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    if m is None:
        m = float(np.max(G))
    deg = np.sum(G > (m - tol), axis=1)
    rowmax = np.max(G, axis=1)
    return deg, rowmax, m, G

def _hash_key(x):
    deg, _, m, _ = _degrees_rowmax(x, tol=5e-7)
    return (tuple(sorted(deg.tolist())), round(float(m) * 1e7))

def _smooth_opt(x, steps, tau0, tau1, lr0, lr1, mom=0.9, active=None):
    n, d = x.shape
    vel = np.zeros_like(x)
    lt0, lt1 = np.log(tau0), np.log(tau1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    if active is not None:
        amask = np.zeros(n, dtype=bool)
        amask[np.asarray(active)] = True
    else:
        amask = None
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
        if amask is not None:
            grad[~amask] = 0.0
        vel = mom * vel + (1.0 - mom) * grad
        bc = 1.0 - mom ** (t + 1)
        step = vel / bc
        x = x - lr * step
        x = _norm(x)
    return x

def _lp_polish(x, deadline, cap=2500, r0=1e-3, maxit=200):
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

def _deepest_hole(Y, rng, n_cand=20000, batch=5000):
    n, d = Y.shape
    best_c = None
    best_s = np.inf
    nb = max(1, n_cand // batch)
    for _ in range(nb):
        C = rng.standard_normal(size=(batch, d))
        C = _norm(C)
        S = C @ Y.T
        sm = S.max(axis=1)
        i = int(np.argmin(sm))
        if float(sm[i]) < best_s:
            best_s = float(sm[i])
            best_c = C[i].copy()
    return best_c

def _shear_move(X, rng, eps):
    n, d = X.shape
    Q, _ = np.linalg.qr(rng.standard_normal(size=(4, 3)))
    u = Q[:, 0]; v = Q[:, 1]; w = Q[:, 2]
    a = X @ u
    b = X @ v
    s = X @ w
    ang = eps * s
    ca = np.cos(ang); sa = np.sin(ang)
    a2 = ca * a - sa * b
    b2 = sa * a + ca * b
    rest_a = np.outer(a, u)
    rest_b = np.outer(b, v)
    base = X - rest_a - rest_b
    Y = base + np.outer(a2, u) + np.outer(b2, v)
    return _norm(Y)

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    FINAL_RESERVE = 28.0
    rng = np.random.default_rng(777)
    n, d = 153, 4
    best_x = None
    best_v = 1e9
    for fname in ("best.npy", "record.npy"):
        try:
            a = np.load(fname)
            if a.shape == (n, d):
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
    global_best_x = best_x.copy()
    global_best_v = best_v
    cur_x = best_x.copy()
    cur_v = best_v
    seen = set([_hash_key(global_best_x)])
    ks = [18, 12, 24, 10, 20, 15]
    epss = [0.5, 0.9, 0.35, 0.7, 1.1]
    it = 0
    EXPLORE_END = HARD_END - FINAL_RESERVE
    while time.time() < EXPLORE_END and time.time() < HARD_END - 5:
        it += 1
        elapsed = (time.time() - t0) / max(1e-9, (EXPLORE_END - t0))
        elapsed = min(1.0, max(0.0, elapsed))
        Temp = 0.003 * (1.0 - elapsed) + 0.0002 * elapsed
        mode = it % 6
        # choose base: mostly cur, every 5th from global best
        if it % 5 == 0:
            base = global_best_x
        else:
            base = cur_x
        Y = None
        try:
            if mode in (0, 1, 2, 3):
                k = ks[(it // 2) % len(ks)]
                deg, rowmax, mcur, G = _degrees_rowmax(base, tol=1e-6)
                if mode in (0, 2):
                    center = int(np.argmax(rowmax))
                else:
                    # random center biased to high rowmax
                    order = np.argsort(-rowmax)
                    pool = order[:30]
                    center = int(pool[rng.integers(len(pool))])
                sims = base @ base[center]
                idx_sorted = np.argsort(-sims)
                del_idx = idx_sorted[:k]
                keep = np.ones(n, dtype=bool); keep[del_idx] = False
                Ykeep = base[keep]
                Yg = Ykeep.copy()
                for _ in range(k):
                    if time.time() > EXPLORE_END:
                        break
                    c = _deepest_hole(Yg, rng, n_cand=15000, batch=5000)
                    Yg = np.vstack([Yg, c.reshape(1, -1)])
                if Yg.shape[0] != n:
                    continue
                nn = Yg.shape[0]
                new_act = list(range(nn - k, nn))
                # masked pre-opt for new points only
                Yg = _smooth_opt(Yg, steps=180, tau0=0.015, tau1=0.002, lr0=0.10, lr1=0.008, active=new_act)
                # joint
                Yg = _smooth_opt(Yg, steps=260, tau0=0.012, tau1=0.002, lr0=0.08, lr1=0.006)
                Yg, vg = _lp_polish(Yg, time.time() + 1.4, cap=2200, r0=8e-4)
                Y, v = Yg, vg
            elif mode == 4:
                eps = epss[(it // 6) % len(epss)]
                Ys = _shear_move(base, rng, eps)
                Ys = _smooth_opt(Ys, steps=280, tau0=0.012, tau1=0.002, lr0=0.09, lr1=0.006)
                Ys, vs = _lp_polish(Ys, time.time() + 1.2, cap=2200, r0=8e-4)
                Y, v = Ys, vs
            else:  # mode 5: rattler transplant + small patch
                deg, rowmax, mcur, G = _degrees_rowmax(base, tol=1e-6)
                mindeg = deg.min()
                cand_r = np.where(deg == mindeg)[0]
                r = int(cand_r[int(np.argmin(rowmax[cand_r]))])
                mask = np.ones(n, dtype=bool); mask[r] = False
                Yex = base[mask]
                c = _deepest_hole(Yex, rng, n_cand=30000, batch=5000)
                Yt = base.copy(); Yt[r] = c
                # shake neighbours of r slightly to allow rewire
                sims = Yt @ Yt[r]
                nbr = np.argsort(-sims)[1:7]
                noise = rng.standard_normal(size=(len(nbr), d)) * 0.01
                Yt[nbr] = _norm(Yt[nbr] + noise - np.sum(noise * Yt[nbr], axis=1, keepdims=True) * Yt[nbr])
                Yt = _smooth_opt(Yt, steps=280, tau0=0.012, tau1=0.002, lr0=0.08, lr1=0.006)
                Yt, vt = _lp_polish(Yt, time.time() + 1.2, cap=2200, r0=8e-4)
                Y, v = Yt, vt
        except Exception:
            continue
        if Y is None:
            continue
        hk = _hash_key(Y)
        is_new = hk not in seen
        if v < global_best_v - 1e-12:
            global_best_v = v
            global_best_x = Y.copy()
            cur_x = Y.copy(); cur_v = v
            seen.add(hk)
        elif v < cur_v + Temp and (is_new or v < cur_v):
            cur_x = Y.copy(); cur_v = v
            seen.add(hk)
        # opportunistic intensify: if close to best, try extra polish
        if global_best_v < best_v - 5e-7:
            # found real progress, recenter
            cur_x = global_best_x.copy(); cur_v = global_best_v
    try:
        dl = HARD_END - 1.0
        if dl > time.time():
            global_best_x, global_best_v = _lp_polish(global_best_x, dl, cap=3000, r0=1e-3)
    except Exception:
        pass
    global_best_x = _norm(global_best_x)
    return global_best_x
