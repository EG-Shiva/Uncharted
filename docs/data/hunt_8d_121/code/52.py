import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contact_set(X, tol):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    gm = float(G.max())
    iu0 = np.triu_indices(X.shape[0], 1)
    sel = G[iu0] > gm - tol
    s = set(zip(iu0[0][sel].tolist(), iu0[1][sel].tolist()))
    return s, gm

def _kmeans(Y, k, rng, iters=60):
    n = Y.shape[0]
    idx = rng.choice(n, size=k, replace=False)
    C = Y[idx].copy()
    lab = np.zeros(n, dtype=int)
    for _ in range(iters):
        d2 = ((Y[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        nl = d2.argmin(axis=1)
        if np.array_equal(nl, lab) and _ > 0:
            break
        lab = nl
        for c in range(k):
            m = lab == c
            if m.sum() > 0:
                C[c] = Y[m].mean(axis=0)
            else:
                C[c] = Y[rng.integers(n)]
    return lab

def _spectral_labels(A, k, seed=0):
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    deg = A.sum(axis=1)
    # unnormalized laplacian
    L = np.diag(deg) - A
    try:
        w, V = np.linalg.eigh(L)
    except Exception:
        return rng.integers(0, k, size=n)
    # k smallest eigenvectors (skip first trivial, take next k)
    # include first for disconnected; just take k smallest
    X = V[:, :k].copy()
    # row normalize
    nr = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    X = X / nr
    lab = _kmeans(X, k, rng, iters=80)
    return lab

def _lp_polish(X, seconds, deadline):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if time.time() > deadline:
            break
        G = X @ X.T
        c = G[iu0]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            if radius < 1e-14:
                break
            continue
        if cnt > 2500:
            thr = np.sort(c)[-2500]
            sel = c >= thr
        I = iu0[0][sel]; J = iu0[1][sel]
        cc = c[sel]
        p = len(I)
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((X.ravel(),
                               (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, n * d + 1)).tocsr()
        except Exception:
            break
        obj = np.zeros(n * d + 1); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            if radius < 1e-14:
                break
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
            if radius < 1e-14:
                break
    return X

def _adam_subset(Xrep, Xfro, betas, steps_each, lr0, rng):
    # optimize Xrep (r,d) vs frozen + self
    r = Xrep.shape[0]
    Xrep = Xrep / np.linalg.norm(Xrep, axis=1, keepdims=True)
    m1 = np.zeros_like(Xrep); v1 = np.zeros_like(Xrep)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1t, b2t = 1.0, 1.0
    it = 0
    for beta in betas:
        for _ in range(steps_each):
            it += 1
            Xall = np.vstack([Xfro, Xrep])
            G = Xrep @ Xall.T  # r x n
            # mask self matches: self columns are last r columns diagonal
            # zero them by -10
            # indices: Xall[n-r+i] == Xrep[i]
            G[np.arange(r), Xfro.shape[0] + np.arange(r)] = -10.0
            gm = float(G.max())
            E = np.exp(beta * (G - gm))
            S = float(E.sum()) + 1e-30
            Gr = (E @ Xall) / S
            dot = (Gr * Xrep).sum(axis=1, keepdims=True)
            Gt = Gr - dot * Xrep
            frac = 0.5 + 0.5 * (1.0 - it / float(len(betas) * steps_each))
            lr = lr0 * frac
            b1t *= b1; b2t *= b2
            m1 = b1 * m1 + (1 - b1) * Gt
            v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
            mh = m1 / (1 - b1t); vh = v1 / (1 - b2t)
            step = lr * mh / (np.sqrt(vh) + eps)
            step = np.clip(step, -0.05, 0.05)
            Xrep = Xrep - step
            Xrep /= np.linalg.norm(Xrep, axis=1, keepdims=True)
    return Xrep

def _adam_all(X, beta, steps, lr0):
    n, d = X.shape
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    m1 = np.zeros_like(X); v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1t, b2t = 1.0, 1.0
    for it in range(1, steps + 1):
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gm = float(G.max())
        E = np.exp(beta * (G - gm))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum()) + 1e-30
        Gr = (E @ X) / S
        dot = (Gr * X).sum(axis=1, keepdims=True)
        Gt = Gr - dot * X
        lr = lr0 * (1.0 - 0.5 * it / steps)
        b1t *= b1; b2t *= b2
        m1 = b1 * m1 + (1 - b1) * Gt
        v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
        mh = m1 / (1 - b1t); vh = v1 / (1 - b2t)
        step = lr * mh / (np.sqrt(vh) + eps)
        step = np.clip(step, -0.06, 0.06)
        X = X - step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def solve():
    t0 = time.time()
    BUDGET = 168.0
    DEADLINE = t0 + BUDGET
    POLISH_TIME = 26.0
    n, d = 121, 8
    rng0 = np.random.default_rng(7)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        X0 = rng0.standard_normal((n, d)); X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands = [X0]
    vals = [_max_cos(a) for a in cands]
    Xinc = cands[int(np.argmin(vals))].copy()
    Vinc = float(min(vals))
    Xbest = Xinc.copy(); Vbest = Vinc
    # contact graph at 2e-5
    Ginc = Xinc @ Xinc.T
    np.fill_diagonal(Ginc, -2.0)
    gm_inc = float(Ginc.max())
    A = (Ginc > gm_inc - 2e-5).astype(float)
    np.fill_diagonal(A, 0.0)
    # tight degrees for outlier id
    tight = (Ginc > gm_inc - 1e-9).sum(axis=1)
    # communities
    K = 7
    try:
        labels = _spectral_labels(A, K, seed=1)
    except Exception:
        labels = np.arange(n) % K
    # incumbent contact set for Jaccard (use 2e-5 graph)
    Sinc, _ = _contact_set(Xinc, 2e-5)
    def jaccard(X):
        S, _ = _contact_set(X, 2e-5)
        if not S and not Sinc:
            return 1.0
        inter = len(S & Sinc); union = len(S | Sinc)
        return inter / max(union, 1)
    survivors = []  # (jacc, val, X)
    # time for trials
    trial_deadline = DEADLINE - POLISH_TIME - 3.0
    # build community lists
    comms = []
    for c in range(K):
        comms.append(np.where(labels == c)[0].tolist())
    # sort communities by size to prioritize 12-20
    # keep order but will cycle
    ntrials = 0
    max_trials = 30
    for trial in range(max_trials):
        if time.time() > trial_deadline:
            break
        ci = trial % K
        base = np.array(comms[ci], dtype=int)
        seed = 1000 + trial * 17
        rng = np.random.default_rng(seed)
        if len(base) == 0:
            continue
        if len(base) > 20:
            # subsample 18
            sel = rng.choice(len(base), size=18, replace=False)
            base_sub = base[sel]
        else:
            base_sub = base.copy()
        # adjacent low-contact outliers: neighbors of base_sub outside it
        is_in = np.zeros(n, dtype=bool); is_in[base_sub] = True
        neigh_mask = (A[base_sub].sum(axis=0) > 0) & (~is_in)
        neigh_idx = np.where(neigh_mask)[0]
        # sort by tight degree ascending (low contact first)
        if len(neigh_idx) > 0:
            order = np.argsort(tight[neigh_idx])
            neigh_idx = neigh_idx[order]
            # take up to 4 with tight<=7, or top 2
            add = []
            for j in neigh_idx:
                if tight[j] <= 7 and len(add) < 4:
                    add.append(j)
            if len(add) < 2 and len(neigh_idx) >= 2:
                add = neigh_idx[:2].tolist()
            base_sub = np.concatenate([base_sub, np.array(add, dtype=int)]) if len(add) > 0 else base_sub
        # ensure 12-20 (pad if too small)
        if len(base_sub) < 12:
            # pad with lowest tight-degree nodes not in set
            outside = np.where(~is_in)[0]
            # exclude already added
            remaining = [j for j in outside if j not in base_sub.tolist()]
            ro = np.argsort(tight[remaining])
            need = min(12 - len(base_sub), len(remaining))
            base_sub = np.concatenate([base_sub, np.array(remaining, dtype=int)[ro[:need]]])
        if len(base_sub) > 24:
            base_sub = rng.choice(base_sub, size=22, replace=False)
        R = base_sub
        r = len(R)
        keep_mask = np.ones(n, dtype=bool); keep_mask[R] = False
        Xfro = Xinc[keep_mask].copy()  # (m,d) frozen reference positions (will be slightly perturbed later in all-anneal)
        m = Xfro.shape[0]
        # generate 50000 candidates
        C = rng.standard_normal((50000, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        placed = []
        # greedy with mutual separation + tiny hill refine
        cur_all = Xfro.copy()
        for k in range(r):
            if time.time() > trial_deadline:
                break
            D = C @ cur_all.T
            cm = D.max(axis=1)
            bi = int(np.argmin(cm))
            bc = C[bi].copy()
            cur = float(cm[bi])
            # quick refine 120 steps
            for st in range(120):
                sig = 0.02 * (1.0 - st / 120.0) + 0.001
                prop = bc + sig * rng.standard_normal(d)
                prop /= np.linalg.norm(prop)
                md = float((prop @ cur_all.T).max())
                if md < cur:
                    bc = prop; cur = md
            placed.append(bc)
            cur_all = np.vstack([cur_all, bc[None, :]])
        if len(placed) != r:
            continue
        Xrep0 = np.stack(placed, axis=0)
        # jointly optimize only replacements vs frozen
        try:
            Xrep1 = _adam_subset(Xrep0, Xfro, betas=[80.0, 200.0, 400.0], steps_each=220, lr0=0.02, rng=rng)
        except Exception:
            Xrep1 = Xrep0
        # assemble full
        Xt = np.zeros((n, d))
        Xt[keep_mask] = Xfro
        Xt[~keep_mask] = Xrep1
        Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
        # short all-point anneal (two betas)
        try:
            Xt = _adam_all(Xt, 250.0, 350, 0.012)
            if time.time() < trial_deadline:
                Xt = _adam_all(Xt, 500.0, 250, 0.008)
        except Exception:
            pass
        Vt = _max_cos(Xt)
        ntrials += 1
        if Vt < Vbest - 1e-12:
            Vbest = Vt; Xbest = Xt.copy()
        # accept intermediates up to 0.470 with J<0.75
        if Vt < 0.470:
            try:
                J = jaccard(Xt)
            except Exception:
                J = 1.0
            if J < 0.75:
                survivors.append((J, Vt, Xt.copy()))
        # also keep track even if not distinct but good? no
    # select 2-3 most distinct survivors for LP polish
    survivors.sort(key=lambda t: (t[0], t[1]))
    # deduplicate roughly: keep those with J differing
    to_polish = survivors[:3]
    for (J, Vt, Xs) in to_polish:
        if time.time() > DEADLINE - 2.0:
            break
        # give each ~ POLISH_TIME/3
        secs = POLISH_TIME / max(len(to_polish), 1)
        # cap by remaining
        secs = min(secs, DEADLINE - time.time() - 1.0)
        if secs < 2.0:
            continue
        try:
            Xp = _lp_polish(Xs, secs, DEADLINE - 1.0)
            Vp = _max_cos(Xp)
            if Vp < Vbest:
                Vbest = Vp; Xbest = Xp.copy()
        except Exception:
            pass
    # final safety LP polish on best if time remains and best not yet polished much
    if time.time() < DEADLINE - 4.0:
        try:
            Xp = _lp_polish(Xbest, min(6.0, DEADLINE - time.time() - 1.0), DEADLINE - 0.5)
            Vp = _max_cos(Xp)
            if Vp < Vbest:
                Vbest = Vp; Xbest = Xp.copy()
        except Exception:
            pass
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    # shrink tiny margin guard (avoid boundary exact ties causing verifier diff): scale not needed; just return
    return Xbest
