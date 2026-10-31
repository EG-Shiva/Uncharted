import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _contact_set(X, tol=1e-7):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    iu = np.triu_indices(n, 1)
    sel = G[iu] > m - tol
    idx = np.where(sel)[0]
    # encode as flat ids
    I = iu[0][idx]; J = iu[1][idx]
    return set((int(i)*1000+int(j)) for i, j in zip(I, J)), m

def _degrees(X, tol=1e-9):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    B = G > m - tol
    deg = B.sum(axis=1)
    return deg, m

def _adam_softmax(X, steps, beta0, beta1, lr0, lr1, free=None, frozen_vals=None):
    X = _norm(X.copy())
    n, d = X.shape
    m1 = np.zeros_like(X); m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    if free is None:
        free = np.ones(n, dtype=bool)
    for it in range(1, steps+1):
        frac = it / float(steps)
        beta = beta0 + (beta1 - beta0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cmax = float(np.max(G))
        W = np.exp(beta * (G - cmax))
        np.fill_diagonal(W, 0.0)
        s = float(W.sum())
        if not np.isfinite(s) or s <= 0:
            break
        W /= s
        Gr = W @ X
        proj = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - proj * X
        Gt[~free] = 0.0
        b1p *= b1; b2p *= b2
        m1 *= b1; m1 += (1.0 - b1) * Gt
        m2 *= b2; m2 += (1.0 - b2) * Gt * Gt
        mh = m1 / (1.0 - b1p)
        vh = m2 / (1.0 - b2p)
        step = lr * mh / (np.sqrt(vh) + eps)
        # clip
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        # avoid huge moves
        cap = 0.08
        scale = np.ones_like(sn)
        over = sn > cap
        scale[over] = cap / (sn[over] + 1e-15)
        step = step * scale
        X = X - step
        if frozen_vals is not None:
            # restore frozen rows (keep exact)
            X[~free] = frozen_vals[~free]
        X = _norm(X)
    return _norm(X)

def _deepest_hole(fixed, rng, ntry):
    d = fixed.shape[1]
    C = rng.standard_normal((ntry, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = C @ fixed.T
    w = np.max(dots, axis=1)
    y = C[int(np.argmin(w))].copy()
    step = 0.06
    for _ in range(100):
        dd = fixed @ y
        j = int(np.argmax(dd))
        g = fixed[j] - float(fixed[j] @ y) * y
        nn = float(np.linalg.norm(g))
        if nn < 1e-12:
            break
        g /= nn
        y = y - step * g
        ny = np.linalg.norm(y)
        if ny == 0 or not np.isfinite(ny):
            break
        y /= ny
        step *= 0.97
    return y / np.linalg.norm(y)

def _lp_polish(X, deadline):
    X = _norm(X.copy())
    n, d = X.shape
    best = _maxcos(X)
    bx = X.copy()
    radius = 1.5e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() + 0.3 > deadline:
            break
        G = X @ X.T
        c = G[iu]
        m = float(np.max(c))
        thr = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thr)[0]
        if idx.size == 0:
            break
        if idx.size > 1400:
            o = np.argsort(c[idx])[-1400:]
            idx = idx[o]
        sel = np.zeros_like(c, dtype=bool)
        sel[idx] = True
        I = iu[0][sel]; J = iu[1][sel]
        cs = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _maxcos(y)
        if v < best:
            best = v; bx = y.copy(); X = y
            radius = min(radius * 1.5, 0.05)
        else:
            if v <= best + 1e-15:
                X = y
            radius *= 0.5
    return bx, best

def _pca_periphery(X):
    # center then SVD
    Xc = X - X.mean(axis=0, keepdims=True)
    try:
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        # project onto top-2
        P = Xc @ Vt[:2].T  # 116x2
    except Exception:
        P = X[:, :2]
    m = P.mean(axis=0, keepdims=True)
    dist = np.sum((P - m) ** 2, axis=1)
    return int(np.argmax(dist))

def solve():
    t0 = time.time()
    HARD_END = t0 + 178.0
    CYCLE_END = t0 + 157.0
    rng = np.random.default_rng(20261031)
    try:
        X0 = np.load("best.npy").astype(float)
    except Exception:
        try:
            X0 = np.load("record.npy").astype(float)
        except Exception:
            X0 = rng.standard_normal((116, 6))
    if X0.shape != (116, 6):
        X0 = rng.standard_normal((116, 6))
    X0 = _norm(X0)
    best_val = _maxcos(X0)
    best_X = X0.copy()
    try:
        inc_set, _ = _contact_set(best_X, tol=1e-7)
    except Exception:
        inc_set = set()

    n, d = 116, 6
    ncycle = 0
    # precompute alternating schedule
    while time.time() < CYCLE_END:
        ncycle += 1
        if time.time() + 8.0 > CYCLE_END:
            break
        mode = ncycle % 3  # 1,2,0 -> map to hub/random/periphery
        # choose centre from current best (elitist base) with small perturbation of base choice?
        base = best_X.copy()
        if mode == 1:
            # worst hub
            try:
                deg, _ = _degrees(base, tol=1e-9)
                centre = int(np.argmax(deg + rng.random(n) * 1e-6))
            except Exception:
                centre = int(rng.integers(n))
        elif mode == 2:
            centre = int(rng.integers(n))
        else:
            try:
                centre = _pca_periphery(base)
                # alternate slightly: sometimes second farthest
                if ncycle % 2 == 0:
                    # pick random among top-5 periphery? compute dist again
                    Xc = base - base.mean(axis=0, keepdims=True)
                    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
                    P = Xc @ Vt[:2].T
                    m = P.mean(axis=0, keepdims=True)
                    dist = np.sum((P - m) ** 2, axis=1)
                    top = np.argsort(dist)[-5:]
                    centre = int(top[rng.integers(5)])
            except Exception:
                centre = int(rng.integers(n))
        # target count 24-36
        K = 24 + ((ncycle * 7) % 13)  # 24..36
        cvec = base[centre]
        dots = base @ cvec
        order = np.argsort(-dots)
        del_idx = order[:K]
        # build working array: keep base values for frozen
        Xw = base.copy()
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[del_idx] = False
        frozen_vals = base.copy()
        # free mask evolves: initially only first inserted? We'll insert one by one
        # positions to fill in random order to avoid bias
        fill_order = del_idx[rng.permutation(K)]
        # current fixed set
        # ntry 20000-50000: vary per cycle
        ntry = 20000 + ((ncycle * 13000) % 30001)  # 20k..50k
        # insert sequentially
        filled = 0
        # free indicator: points already refilled are free-tunable during intermediate relax
        for k, pos in enumerate(fill_order):
            if time.time() + 4.0 > CYCLE_END and k < K - 1:
                # need to rush: reduce ntry for remaining
                ntry_small = 6000
                use_ntry = ntry_small
            else:
                use_ntry = ntry
            # fixed set = all except unfilled positions
            # collect current points: kept + already filled
            # mask of active rows
            active = keep_mask.copy()
            active[fill_order[:k]] = True  # already filled
            fixed = Xw[active]
            y = _deepest_hole(fixed, rng, ntry=use_ntry)
            Xw[pos] = y
            filled += 1
            # short tangential softmax relaxation after every 4-6 insertions
            if (filled % 5 == 0 and filled < K) or (filled == K and False):
                # relax only the inserted-so-far (free), freeze rest
                free = np.zeros(n, dtype=bool)
                free[fill_order[:filled]] = True
                # 300 steps beta 30->300
                # use frozen_vals = Xw for frozen rows? need to keep them fixed
                Xw = _adam_softmax(Xw, steps=300, beta0=30, beta1=300, lr0=0.05, lr1=0.005, free=free, frozen_vals=Xw.copy())
                if time.time() > CYCLE_END:
                    break
        if time.time() > CYCLE_END:
            # still need to evaluate what we have
            pass
        # unfreeze: annealed softmax 800-1200 steps beta 50->1000
        steps_ann = 800 + ((ncycle * 137) % 401)  # 800..1200
        # choose lr schedule modest
        Xw = _adam_softmax(Xw, steps=steps_ann, beta0=50, beta1=1000, lr0=0.06, lr1=0.004, free=np.ones(n, dtype=bool))
        pre_val = _maxcos(Xw)
        # decide LP polish
        do_lp = False
        try:
            cur_set, _ = _contact_set(Xw, tol=1e-7)
            inter = len(cur_set & inc_set)
            union = len(cur_set | inc_set)
            jacc = (inter / union) if union > 0 else 0.0
        except Exception:
            jacc = 0.0
        if jacc < 0.85:
            do_lp = True
        elif pre_val <= best_val + 0.002:
            do_lp = True
        if do_lp:
            # give limited time: min(remaining cycle budget, 6s)
            lp_dead = min(time.time() + 6.0, CYCLE_END + 2.0, HARD_END - 21.0)
            if lp_dead - time.time() > 1.0:
                Xp, vp = _lp_polish(Xw, lp_dead)
            else:
                Xp, vp = Xw, pre_val
        else:
            Xp, vp = Xw, pre_val
        if vp < best_val and np.all(np.isfinite(Xp)):
            # verify shape
            Xn = _norm(Xp)
            vv = _maxcos(Xn)
            if vv < best_val:
                best_val = vv
                best_X = Xn.copy()
                try:
                    inc_set, _ = _contact_set(best_X, tol=1e-7)
                except Exception:
                    pass
        # loop
        if ncycle > 30:
            break
    # final LP polish of overall best, reserve 20s
    final_dead = min(HARD_END - 1.0, time.time() + 19.0)
    if time.time() < final_dead - 1.0:
        try:
            Xf, vf = _lp_polish(best_X, final_dead)
            if vf < best_val and np.all(np.isfinite(Xf)):
                best_X = _norm(Xf); best_val = vf
        except Exception:
            pass
    best_X = _norm(best_X)
    # shrink tiny margin? ensure unit rows
    return best_X
