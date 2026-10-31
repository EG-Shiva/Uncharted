import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contact_info(X, tol=1e-9):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    mask = G > (m - tol)
    deg = mask.sum(axis=1).astype(int)
    iu0 = np.triu_indices(n, 1)
    sel = G[iu0] > (m - tol)
    edges = set(zip(iu0[0][sel].tolist(), iu0[1][sel].tolist()))
    return m, deg, edges, int(sel.sum())

def _anneal_full(X, steps, beta0, beta1, lr0, mu, rng=None):
    X = _norm(X)
    n = X.shape[0]
    vel = np.zeros_like(X)
    if beta1 > beta0:
        lb = np.log(beta1 / beta0)
    else:
        lb = 0.0
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = beta0 * np.exp(lb * frac)
        lr = lr0 * (1.0 - 0.5 * frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-300
        A = (W @ X) / rs
        dot = (A * X).sum(axis=1, keepdims=True)
        Gt = A - dot * X
        vel = mu * vel + Gt
        vd = (vel * X).sum(axis=1, keepdims=True)
        vel = vel - vd * X
        X = X - lr * vel
        X = _norm(X)
    return X

def _anneal_mask(X, mask, steps, beta0, beta1, lr0, mu):
    X = _norm(X)
    vel = np.zeros_like(X)
    if beta1 > beta0:
        lb = np.log(beta1 / beta0)
    else:
        lb = 0.0
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return X
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = beta0 * np.exp(lb * frac)
        lr = lr0 * (1.0 - 0.5 * frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        # avoid overflow: beta*(G-gmax) <=0
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-300
        A = (W @ X) / rs
        dot = (A * X).sum(axis=1, keepdims=True)
        Gt = A - dot * X
        vel[m] = mu * vel[m] + Gt[m]
        # re-tangentize
        vd = (vel[m] * X[m]).sum(axis=1, keepdims=True)
        vel[m] = vel[m] - vd * X[m]
        X[m] = X[m] - lr * vel[m]
        X[m] = _norm(X[m])
    return X

def _lp_polish(X, seconds, r0=8e-4):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return X
    X = _norm(np.array(X, dtype=float, copy=True))
    n, d = X.shape
    best = _max_cos(X)
    if seconds <= 0.05:
        return X
    start = time.time()
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = X @ X.T
        c = G[iu]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        if not np.any(sel):
            break
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            ci = I[:, None] * d + np.arange(d)
            cj = J[:, None] * d + np.arange(d)
            cols = np.concatenate([ci, cj, np.full((p, 1), n * d)], axis=1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if getattr(res, "status", 1) != 0:
            radius *= 0.5
            continue
        try:
            y = X + res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            best = v
            X = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _sample_holes(Xkeep, k, rng, n_cand=65536, n_refine=300):
    # sample n_cand random points, score by max dot to Xkeep
    cand = rng.standard_normal((n_cand, 4))
    cand = _norm(cand)
    m = Xkeep.shape[0]
    best = np.empty(n_cand, dtype=float)
    cs = 4096
    for a in range(0, n_cand, cs):
        b = min(a + cs, n_cand)
        D = cand[a:b] @ Xkeep.T
        best[a:b] = D.max(axis=1)
    order = np.argsort(best)
    H = cand[order[:n_refine]].copy()
    # refine best few hundred by minimizing softmax max vs Xkeep
    R = Xkeep
    beta = 150.0
    lr = 0.06
    for it in range(80):
        D = H @ R.T  # 300 x m
        mr = D.max(axis=1, keepdims=True)
        W = np.exp(beta * (D - mr))
        W /= (W.sum(axis=1, keepdims=True) + 1e-300)
        Gv = W @ R
        dot = (Gv * H).sum(axis=1, keepdims=True)
        Gt = Gv - dot * H
        H = H - lr * Gt
        H = _norm(H)
        lr *= 0.995
        beta = min(beta * 1.01, 400.0)
    # scores after refine
    Df = H @ R.T
    mf = Df.max(axis=1)
    rorder = np.argsort(mf)
    H = H[rorder]
    return H, cand[order[:8]]

def solve():
    t0 = time.time()
    TLIM = 170.0
    explore_end = t0 + 150.0
    rng = np.random.default_rng(292038)
    n, d = 292, 4
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if float(nr.min()) > 1e-9:
                    cands.append(_norm(a))
        except Exception:
            pass
    if not cands:
        z = rng.standard_normal((n, d))
        Xb = _norm(z)
    else:
        vals = [_max_cos(c) for c in cands]
        Xb = cands[int(np.argmin(vals))].copy()
    best_val = _max_cos(Xb)
    X_best = Xb.copy()
    _, _, inc_edges, inc_cnt = _contact_info(X_best)
    inc_edges0 = set(inc_edges)
    pool = [(best_val, X_best.copy())]
    ks = [4, 8, 12, 16, 20]
    it = 0
    TEMP_TARGET = 0.90347
    while time.time() < explore_end:
        it += 1
        rem = explore_end - time.time()
        if rem < 4.0:
            break
        # choose base: mostly best, sometimes diverse pool
        r = rng.random()
        if r < 0.7 or len(pool) == 1:
            base = X_best
        else:
            idxp = int(rng.integers(len(pool)))
            base = pool[idxp][1]
        try:
            _, deg, edges_base, _ = _contact_info(base)
        except Exception:
            base = X_best
            _, deg, edges_base, _ = _contact_info(base)
        ratt = np.where(deg == 0)[0].tolist()
        low = np.where(deg <= 4)[0].tolist()
        high = np.where(deg >= 8)[0].tolist()
        k = ks[int(rng.integers(len(ks)))]
        # build deletion set mixing classes
        del_set = []
        # always include rattlers first
        rng.shuffle(ratt)
        for v in ratt:
            if len(del_set) < k:
                del_set.append(int(v))
        # portion from low
        tmp = low.copy()
        rng.shuffle(tmp)
        # remove already taken
        tmp = [v for v in tmp if v not in del_set]
        n_low = min(len(tmp), max(1, k // 3))
        # randomize how many from each class a bit
        if rng.random() < 0.5:
            n_low = min(len(tmp), k // 2)
        for v in tmp[:n_low]:
            if len(del_set) < k:
                del_set.append(int(v))
        # portion from high clusters
        tmp2 = high.copy()
        rng.shuffle(tmp2)
        tmp2 = [v for v in tmp2 if v not in del_set]
        n_high = min(len(tmp2), max(1, k // 3))
        for v in tmp2[:n_high]:
            if len(del_set) < k:
                del_set.append(int(v))
        # fill rest randomly
        while len(del_set) < k:
            v = int(rng.integers(n))
            if v not in del_set:
                del_set.append(v)
        del_arr = np.array(del_set, dtype=int)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[del_arr] = False
        Xkeep = base[keep_mask]
        # hole sampling
        try:
            H, _ = _sample_holes(Xkeep, k, rng, n_cand=65536, n_refine=300)
        except Exception:
            continue
        # greedy insertion under temporary separation target
        # combined matrix grows as we insert
        combined = Xkeep.copy()  # (n-k,4)
        picked = []
        Hpool = H.copy()
        # available pool indices
        avail = list(range(Hpool.shape[0]))
        for s in range(k):
            if not avail:
                break
            # compute max dot of each avail to combined
            # chunk over avail to save time (avail<=300)
            Ha = Hpool[avail]  # (A,4)
            D = Ha @ combined.T
            mx = D.max(axis=1)
            # prefer below TEMP_TARGET; pick best (smallest mx)
            bi = int(np.argmin(mx))
            chosen_vec = Ha[bi].copy()
            picked.append(chosen_vec)
            combined = np.vstack([combined, chosen_vec.reshape(1, 4)])
            # remove chosen from avail
            avail.pop(bi)
            # enforce mutual separation: drop avail vectors too close to chosen (>0.98 sep -> too close)
            # keep diversity: remove any with dot > 0.995 to chosen (near duplicate)
            if avail:
                Ha2 = Hpool[avail]
                dd = Ha2 @ chosen_vec
                keep = [avail[i] for i in range(len(avail)) if dd[i] < 0.995]
                avail = keep
        if len(picked) < k:
            # fill with random far points
            while len(picked) < k:
                v = rng.standard_normal((4,))
                v /= max(np.linalg.norm(v), 1e-12)
                picked.append(v)
        Xnew = base.copy()
        for j, idx in enumerate(del_arr):
            Xnew[idx] = picked[j] / max(np.linalg.norm(picked[j]), 1e-12)
        Xnew = _norm(Xnew)
        # local optimize: inserted + neighbours with cos>0.75
        try:
            Dins = Xnew[del_arr] @ Xnew.T  # k x n
            neigh = np.where(Dins.max(axis=0) > 0.75)[0]
            active = np.zeros(n, dtype=bool)
            active[del_arr] = True
            active[neigh] = True
            # ensure not everything (cap): if too many, keep del + closest 60
            if int(active.sum()) > 120:
                # keep del plus top neighbours by max dot
                scores = Dins.max(axis=0)
                order_n = np.argsort(scores)[::-1][:120]
                active = np.zeros(n, dtype=bool)
                active[order_n] = True
                active[del_arr] = True
        except Exception:
            active = np.zeros(n, dtype=bool)
            active[del_arr] = True
        # local anneal
        try:
            Xnew = _anneal_mask(Xnew, active, steps=260, beta0=30.0, beta1=500.0, lr0=0.06, mu=0.86)
        except Exception:
            continue
        # global annealed softmax repulsion (short)
        try:
            # vary schedule a bit per iteration
            b0 = float(rng.choice([20.0, 30.0, 45.0]))
            b1 = float(rng.choice([350.0, 500.0, 700.0]))
            Xnew = _anneal_full(Xnew, steps=320, beta0=b0, beta1=b1, lr0=0.06, mu=0.86)
        except Exception:
            continue
        try:
            vmid = _max_cos(Xnew)
        except Exception:
            continue
        # only polish if reasonably close (avoid wasting time)
        if vmid > best_val + 0.0025 and rng.random() > 0.15:
            # still occasionally polish diverse ones? skip mostly
            continue
        # LP polish
        budget = min(1.6, max(0.6, rem - 2.0))
        # give a bit more if direct win candidate
        if vmid < best_val:
            budget = min(2.5, max(1.0, rem - 2.0))
        try:
            Y = _lp_polish(Xnew, seconds=budget)
            vy = _max_cos(Y)
        except Exception:
            continue
        try:
            _, _, edges_y, _ = _contact_info(Y)
            diff = len(edges_y ^ inc_edges0)
        except Exception:
            diff = 0
        # bank diverse basins
        if diff >= 20:
            # add to pool (keep sorted, cap 8)
            pool.append((vy, Y.copy()))
            pool.sort(key=lambda t: t[0])
            # dedupe pool by edge-set difference: keep only if distinct from others
            # simple cap
            if len(pool) > 8:
                pool = pool[:8]
        if vy < best_val:
            X_best = Y.copy()
            best_val = float(vy)
            pool.append((best_val, X_best.copy()))
            pool.sort(key=lambda t: t[0])
            if len(pool) > 8:
                pool = pool[:8]
            # update inc reference? keep original for diversity, but also track best
            # continue exploiting new best
    # final polish of best
    try:
        if time.time() < t0 + TLIM - 3.0:
            X_best = _lp_polish(X_best, seconds=max(0.5, (t0 + TLIM - 1.0) - time.time()))
    except Exception:
        pass
    X_best = _norm(X_best)
    return X_best
