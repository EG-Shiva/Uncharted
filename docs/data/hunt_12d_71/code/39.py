import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _quench_full(X, steps, T0, T1, lr, mu=0.9, deadline=None):
    n, d = X.shape
    V = np.zeros_like(X)
    lt0 = np.log(T0); lt1 = np.log(T1)
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    bv = float(G0.max())
    bX = X.copy()
    for s in range(steps):
        if deadline is not None and (s % 250 == 0) and time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -1e100)
        m = float(G.max())
        Q = (G - m) / T
        np.clip(Q, -50, 0, out=Q)
        E = np.exp(Q)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        Grad = (E @ X) / rs
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        big = gn[:, 0] > 2.0
        if np.any(big):
            Gt[big] *= (2.0 / gn[big])
        V = mu * V + Gt
        X = X - lr * V
        X = _norm_rows(X)
        if (s % 200 == 0) or (s == steps - 1):
            Gc = X @ X.T
            np.fill_diagonal(Gc, -2.0)
            vc = float(Gc.max())
            if vc < bv:
                bv = vc
                bX = X.copy()
    return bX, bv

def _quench_active(X, active, steps, T0, T1, lr, mu=0.9, deadline=None):
    # X modified in place copy; active: array of indices
    X = X.copy()
    na = len(active)
    V = np.zeros((na, 12))
    lt0 = np.log(T0); lt1 = np.log(T1)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -1e100)
        m = float(G.max())
        Q = (G - m) / T
        np.clip(Q, -50, 0, out=Q)
        E = np.exp(Q)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        Grad = (E @ X) / rs
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        Ga = Gt[active]
        gn = np.linalg.norm(Ga, axis=1, keepdims=True) + 1e-12
        big = gn[:, 0] > 2.0
        if np.any(big):
            Ga[big] *= (2.0 / gn[big])
        V = mu * V + Ga
        X[active] = X[active] - lr * V
        X[active] = _norm_rows(X[active])
    return X

def _lp_polish(x, seconds, deadline=None, wide=False):
    x = _norm_rows(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = 1.2e-3 if wide else 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        cmat = (x @ x.T)[iu]
        m = float(cmat.max())
        band = max(25.0 * radius, 2e-3 if wide else 1e-9)
        sel = cmat > m - band
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 1500:
            order = np.argsort(-cmat[sel])[:1500]
            I, J = I[order], J[order]
            csel = cmat[sel][order]
            p = 1500
        else:
            csel = cmat[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cc = np.concatenate([I[:, None] * d + np.arange(d),
                             J[:, None] * d + np.arange(d),
                             np.full((p, 1), n * d)], 1).ravel()
        vv = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vv, (rows, cc)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
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
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _refine_single(s, R, steps, T, lr, mu=0.8):
    s = s / (np.linalg.norm(s) + 1e-12)
    v = np.zeros_like(s)
    for _ in range(steps):
        g = R @ s
        m = float(g.max())
        e = np.exp(np.clip((g - m) / T, -50, 0))
        den = float(e.sum()) + 1e-30
        Grad = (R.T @ e) / den
        Gt = Grad - float(Grad @ s) * s
        v = mu * v + Gt
        s = s - lr * v
        nn = float(np.linalg.norm(s))
        if not np.isfinite(nn) or nn < 1e-12:
            break
        s = s / nn
    return s

def _find_hole(R, rng, n_sample=40000, n_cand=3):
    d = R.shape[1]
    Z = rng.standard_normal(size=(n_sample, d))
    Z = _norm_rows(Z)
    S = Z @ R.T
    scores = S.max(axis=1)
    idx = np.argpartition(scores, n_cand)[:n_cand]
    best_s = None; best_v = 1e100
    for k in idx:
        s0 = Z[k]
        s1 = _refine_single(s0, R, steps=180, T=0.005, lr=0.12)
        s1 = _refine_single(s1, R, steps=180, T=0.0015, lr=0.07)
        v = float((R @ s1).max())
        if v < best_v:
            best_v = v; best_s = s1
    best_s = best_s / (np.linalg.norm(best_s) + 1e-12)
    return best_s

def _fingerprint(X):
    g = X @ X.T
    iu = np.triu_indices(g.shape[0], 1)
    vals = g[iu]
    top = np.sort(vals)[-100:]
    q = np.round(top / 2e-7).astype(np.int64)
    return tuple(q.tolist())

def solve():
    t0 = time.time()
    TIME_LIMIT = 173.0
    deadline = t0 + TIME_LIMIT
    rng = np.random.default_rng(12345)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = _norm_rows(np.asarray(np.load(fn), dtype=np.float64))
            if a.shape == (71, 12):
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_norm_rows(rng.standard_normal(size=(71, 12))))
    bestX = min(cands, key=_max_cos).copy()
    bestVal = _max_cos(bestX)
    curX = bestX.copy(); curVal = bestVal
    tabu = set()
    try:
        tabu.add(_fingerprint(bestX))
    except Exception:
        pass
    it = 0
    # short initial squeeze to get row-max/degree structure
    while time.time() < deadline - 5.0:
        it += 1
        Xc = curX.copy()
        n = 71
        G = Xc @ Xc.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        flat = int(np.argmax(G))
        a = flat // n; b = flat % n
        row_max = G.max(axis=1)
        # near-contact degree at slack 1e-6
        deg = np.count_nonzero(G > m - 1e-6, axis=1)
        # coordinated delete size 8-18
        s_size = int(rng.integers(8, 19))
        ep = a if rng.random() < 0.5 else b
        order = np.argsort(-row_max)
        med = float(np.median(deg))
        seed = -1
        for idx in order:
            if int(idx) == int(ep):
                continue
            if float(deg[int(idx)]) <= med:
                seed = int(idx); break
        if seed == -1:
            for idx in order:
                if int(idx) != int(ep):
                    seed = int(idx); break
        # spatial cluster: nearest in dot (PCA/rigidity proxy = Gram neighbours)
        neigh = np.argsort(-G[seed])
        cluster = []
        for j in neigh:
            j = int(j)
            if j == ep or j == seed:
                continue
            cluster.append(j)
            if len(cluster) >= s_size - 2:
                break
        Dset = set([int(ep), int(seed)] + cluster[:max(0, s_size-2)])
        while len(Dset) < s_size:
            Dset.add(int(rng.integers(0, 71)))
        Dlist = list(Dset)
        keep_mask = np.ones(71, dtype=bool); keep_mask[Dlist] = False
        keep_idx = np.where(keep_mask)[0]
        k = len(keep_idx)
        stack = Xc[keep_idx].copy()  # (k,12)
        # adapt sample size to time left
        tleft = deadline - time.time()
        if tleft < 20:
            nsamp = 20000
        elif tleft < 60:
            nsamp = 32000
        else:
            nsamp = 45000
        # greedy reinsert one at a time
        for tpos in range(s_size):
            if time.time() > deadline - 4.0:
                break
            try:
                hole = _find_hole(stack, rng, n_sample=nsamp, n_cand=3)
            except Exception:
                hole = _norm_rows(rng.standard_normal(size=(1, 12)))[0]
            stack = np.vstack([stack, hole[None, :]])
            # joint Adam quench of inserted + 10 nearest neighbours, core frozen
            new_pos = stack.shape[0] - 1
            dots = stack @ stack[new_pos]
            dots[new_pos] = -1e100
            nn_order = np.argsort(-dots)[:10]
            ins_start = k  # first inserted position in stack
            active = list(range(ins_start, stack.shape[0])) + [int(x) for x in nn_order if int(x) < ins_start]
            active = sorted(set(active))
            try:
                stack = _quench_active(stack, np.array(active, dtype=int), steps=280, T0=0.006, T1=0.0006, lr=0.05, deadline=deadline)
            except Exception:
                pass
        if stack.shape[0] != 71:
            continue
        Y = stack.copy()
        # unfreeze short quench
        try:
            Y, _ = _quench_full(Y, steps=700, T0=0.004, T1=0.0003, lr=0.04, deadline=deadline-2.0)
        except Exception:
            pass
        # wide-band LP
        if time.time() < deadline - 2.5:
            try:
                Y = _lp_polish(Y, seconds=min(1.2, max(0.3, deadline - time.time() - 0.5)), deadline=deadline-0.3, wide=True)
            except Exception:
                pass
        try:
            v = _max_cos(Y)
            fp = _fingerprint(Y)
        except Exception:
            continue
        is_tabu = fp in tabu
        if v < bestVal - 1e-12:
            bestVal = v; bestX = Y.copy()
            curX = Y.copy(); curVal = v
            tabu.add(fp)
            # keep tabu bounded
            if len(tabu) > 5000:
                tabu = set(list(tabu)[-3000:])
        elif (v < curVal + 2e-7) and (not is_tabu):
            curX = Y.copy(); curVal = v
            tabu.add(fp)
        elif (v < bestVal + 2e-7) and (not is_tabu) and rng.random() < 0.15:
            # occasional sideways to force topology change
            curX = Y.copy(); curVal = v
            tabu.add(fp)
        else:
            # tabu hit or worse: diversify current by small kick to escape basin
            if is_tabu and rng.random() < 0.5:
                try:
                    kk = int(rng.integers(2, 5))
                    idx = rng.choice(71, size=kk, replace=False)
                    curX[idx] = _norm_rows(curX[idx] + rng.standard_normal(size=(kk, 12)) * 0.05)
                except Exception:
                    pass
        # periodic return to best
        if it % 12 == 0 and rng.random() < 0.5:
            curX = bestX.copy(); curVal = bestVal
        if time.time() > deadline - 1.0:
            break
    return _norm_rows(bestX.copy())
