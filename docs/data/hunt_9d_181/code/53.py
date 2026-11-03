import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, budget, t_dead=None):
    t0 = time.time()
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    last_I = None
    last_J = None
    last_w = None
    it = 0
    while True:
        if time.time() - t0 >= budget:
            break
        if t_dead is not None and time.time() >= t_dead:
            break
        if radius < 1e-14 or it >= 30:
            break
        it += 1
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        csel = c[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 1500:
            idx = np.argpartition(-csel, 1500)[:1500]
            I = I[idx]; J = J[idx]; csel = csel[idx]
            p = 1500
        nv = n * d + 1
        cols_mat = np.empty((p, 2 * d + 1), dtype=np.int64)
        cols_mat[:, :d] = I[:, None] * d + ar
        cols_mat[:, d:2 * d] = J[:, None] * d + ar
        cols_mat[:, 2 * d] = n * d
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = cols_mat.ravel()
        vals_mat = np.empty((p, 2 * d + 1))
        vals_mat[:, :d] = x[J]
        vals_mat[:, d:2 * d] = x[I]
        vals_mat[:, 2 * d] = -1.0
        vals = vals_mat.ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        # save duals
        try:
            marg = np.asarray(res.ineqlin.marginals, dtype=np.float64).ravel()
            w = np.abs(marg)
            if w.size == p and np.all(np.isfinite(w)) and w.sum() > 0:
                last_I = I.copy(); last_J = J.copy(); last_w = w.copy()
        except Exception:
            pass
        try:
            y = x + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best, (last_I, last_J, last_w)

def _smooth_full(X, steps, k0=80.0, k1=300.0, lr0=0.02, lr1=0.002):
    n, d = X.shape
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        k = k0 + (k1 - k0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(np.clip(k * (G - gmax), -50, 0))
        np.fill_diagonal(W, 0.0)
        Gr = W @ X
        dot = np.einsum('ij,ij->i', Gr, X)[:, None]
        Gt = Gr - dot * X
        nrm = np.linalg.norm(Gt, axis=1).max()
        if not np.isfinite(nrm) or nrm < 1e-12:
            break
        X = X - (lr / nrm) * Gt
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _refine_insert(Y, Kaug, steps=110, k0=60.0, k1=350.0, lr0=0.15, lr1=0.02):
    # Y: (M,d) variable, Kaug: (Na,d) fixed
    M, d = Y.shape
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        k = k0 + (k1 - k0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        S = Y @ Kaug.T
        smax = S.max(axis=1, keepdims=True)
        E = np.exp(np.clip(k * (S - smax), -50, 0))
        Ws = E / (E.sum(axis=1, keepdims=True) + 1e-300)
        Gr = Ws @ Kaug
        dot = np.einsum('ij,ij->i', Gr, Y)[:, None]
        Gt = Gr - dot * Y
        nr = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        Gt = Gt / nr
        Y = Y - lr * Gt
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def solve():
    rng = np.random.default_rng(7)
    t0 = time.time()
    DEAD = t0 + 155.0
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = np.load("record.npy")
    xb = np.asarray(xb, dtype=np.float64)
    if xb.shape != (181, 9):
        flat = xb.ravel()
        need = 181 * 9
        if flat.size < need:
            extra = rng.standard_normal((need - flat.size,))
            flat = np.concatenate([flat, extra])
        xb = flat[:need].reshape(181, 9)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    X_best = xb.copy()

    n, d = 181, 9

    # 1) one LP polish to get near optimum + duals
    Xpol, vpol, dualt = _lp_polish(X_best, budget=7.0, t_dead=DEAD)
    if vpol < best_val:
        X_best, best_val = Xpol, vpol
    else:
        Xpol = X_best.copy()
    if time.time() >= DEAD - 8.0:
        return X_best / np.linalg.norm(X_best, axis=1, keepdims=True)

    # 2) dual-weighted stress
    LI, LJ, Lw = dualt
    Gfull = Xpol @ Xpol.T
    np.fill_diagonal(Gfull, -2.0)
    mfull = float(Gfull.max())
    cnt = (Gfull > mfull - 1e-9).sum(axis=1)
    rattlers = np.where(cnt == 0)[0].tolist()
    # ensure exactly 2 rattlers: pad with lowest-contact points
    order_cnt = np.argsort(cnt)
    while len(rattlers) < 2:
        for c in order_cnt:
            if c not in rattlers:
                rattlers.append(int(c))
                break
    rattlers = rattlers[:2]

    npts = n
    stress = np.zeros(npts)
    if LI is not None and Lw is not None and len(Lw) == len(LI):
        for a, b, w in zip(LI, LJ, Lw):
            stress[a] += w
            stress[b] += w
    # fallback if duals missing/degenerate
    if stress.sum() <= 0 or not np.all(np.isfinite(stress)):
        # use contact-count + slack stress
        stress = cnt.astype(float) + 0.1 * (Gfull - (mfull - 1e-9)).clip(min=0).sum(axis=1)
        stress[rattlers] = -1.0
    else:
        stress[np.array(rattlers)] = -1.0

    # contact graph for connectivity (1e-9 threshold, fallback to 1e-7 if disconnected)
    adj = [set() for _ in range(npts)]
    Cmask = Gfull > (mfull - 1e-9)
    # if too sparse, widen
    if Cmask.sum() < 500:
        Cmask = Gfull > (mfull - 1e-7)
    for i in range(npts):
        nbrs = np.where(Cmask[i])[0]
        for j in nbrs:
            if j != i:
                adj[i].add(int(j))

    # grow highest-stress connected cluster of size 14
    KCORE = 14
    seed = int(np.argmax(stress))
    Sset = {seed}
    while len(Sset) < KCORE:
        frontier = set()
        for s in Sset:
            frontier.update(adj[s])
        frontier -= Sset
        # exclude rattlers
        frontier = {f for f in frontier if f not in rattlers}
        if not frontier:
            # add next highest stress not yet in set
            for c in np.argsort(-stress):
                c = int(c)
                if c not in Sset and c not in rattlers:
                    Sset.add(c)
                    break
            continue
        # pick highest stress in frontier
        bf = max(frontier, key=lambda f: stress[f])
        Sset.add(bf)
    core = sorted(list(Sset))
    # total removed
    removed = sorted(set(rattlers) | set(core))
    # enforce 12..20 removed; adjust if needed
    if len(removed) > 20:
        # drop lowest-stress core members
        core_sorted = sorted(core, key=lambda c: stress[c])
        while len(removed) > 20:
            c = core_sorted.pop(0)
            removed.remove(c)
    keep_idx = [i for i in range(npts) if i not in set(removed)]
    K = Xpol[np.array(keep_idx)].copy()
    kdel = len(removed)

    if time.time() >= DEAD - 8.0:
        return X_best / np.linalg.norm(X_best, axis=1, keepdims=True)

    # 3) reinsert one at a time with frozen remainder
    D = np.empty((0, d))
    for j in range(kdel):
        if time.time() >= DEAD - 6.0:
            break
        if D.shape[0] == 0:
            Kaug = K
        else:
            Kaug = np.vstack([K, D])
        # 50k random sphere samples for holes
        C = rng.standard_normal((50000, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        S = C @ Kaug.T
        mx = S.max(axis=1)
        # deepest holes
        hole_idx = np.argpartition(mx, 6)[:6]
        # order holes by mx
        hole_idx = hole_idx[np.argsort(mx[hole_idx])]
        holes = C[hole_idx]
        # 220 random starts
        Y0 = rng.standard_normal((220, d))
        Y0 /= np.linalg.norm(Y0, axis=1, keepdims=True)
        Y = np.vstack([Y0, holes])
        Y = _refine_insert(Y, Kaug, steps=100, k0=60.0, k1=350.0, lr0=0.15, lr1=0.02)
        Sf = Y @ Kaug.T
        mxf = Sf.max(axis=1)
        bi = int(np.argmin(mxf))
        nxt = Y[bi:bi+1] / np.linalg.norm(Y[bi:bi+1], axis=1, keepdims=True)
        D = np.vstack([D, nxt]) if D.shape[0] > 0 else nxt.copy()

    if D.shape[0] != kdel:
        # incomplete due to deadline -> return incumbent
        return X_best / np.linalg.norm(X_best, axis=1, keepdims=True)

    Xnew = np.vstack([K, D])
    # 4) short exponential-repulsion smoothing unfrozen
    if time.time() < DEAD - 5.0:
        Xnew = _smooth_full(Xnew, steps=220, k0=80.0, k1=320.0, lr0=0.02, lr1=0.002)
    vnew = _max_cos(Xnew)
    if vnew < best_val:
        X_best, best_val = Xnew, vnew

    # 5) final LP polish with remaining time
    remain = DEAD - time.time()
    if remain > 3.0:
        Xf, vf, _ = _lp_polish(X_best, budget=min(remain - 0.5, 25.0), t_dead=DEAD)
        if vf < best_val:
            X_best, best_val = Xf, vf

    X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
    if not np.all(np.isfinite(X_best)):
        X_best = xb
    return X_best
