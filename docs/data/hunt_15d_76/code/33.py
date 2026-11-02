import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _rowmax(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), float(G.max())

def _anneal(X, n_iter, T0, T1, lr0, lr1, t_end=None):
    n, d = X.shape
    for k in range(n_iter):
        if t_end is not None and (k & 255) == 0 and time.time() > t_end:
            break
        f = k / max(1, n_iter - 1)
        T = T0 * (T1 / T0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if S <= 0 or not np.finite(S):
            break
        Gr = (E @ X) / S
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        # limit step
        X = X - lr * Gt
        # renormalize
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X

def _refine_hole(v, Y, n_iter=200, T0=0.008, lr0=0.08, t_end=None):
    v = v / np.linalg.norm(v)
    for k in range(n_iter):
        if t_end is not None and (k & 63) == 0 and time.time() > t_end:
            break
        f = k / max(1, n_iter - 1)
        Tk = T0 * (0.15 ** f)
        lrk = lr0 * (0.15 ** f)
        s = Y @ v
        m = float(s.max())
        w = np.exp((s - m) / Tk)
        sw = float(w.sum())
        if sw == 0 or not np.finite(sw):
            break
        w /= sw
        g = w @ Y
        g = g - float(g @ v) * v
        v = v - lrk * g
        nv = float(np.linalg.norm(v))
        if nv == 0 or not np.finite(nv):
            break
        v /= nv
    return v

def _hill_max(X, iters=300, step0=0.002, t_end=None):
    # subgradient descent on true max
    X = _norm(X)
    cur = _maxcos(X)
    step = step0
    n, d = X.shape
    for it in range(iters):
        if t_end is not None and (it & 31) == 0 and time.time() > t_end:
            break
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # contacts within tightening band
        tol = max(1e-4, 5.0 * step)
        C = G > m - tol
        # force per point: sum of contact neighbours tangent components
        # F_i = sum_j C_ij * (Xj - (Xi.Xj) Xi)
        # compute efficiently
        # weight by (G - (m-tol))/tol to focus on tightest? use linear weight
        W = np.zeros_like(G)
        # only upper? need symmetric; use C
        W[C] = (G[C] - (m - tol)) / tol + 0.2
        # gradient
        F = W @ X  # (n,d)
        dot = np.sum(F * X, axis=1, keepdims=True)
        Ft = F - dot * X
        # normalize step per point by degree to keep balanced
        deg = C.sum(axis=1, keepdims=True).astype(float) + 1.0
        Y = X - (step / deg) * Ft
        Y = _norm(Y)
        v = _maxcos(Y)
        if v < cur:
            # accept, slightly grow step
            X = Y
            cur = v
            step *= 1.08
        else:
            step *= 0.5
            if step < 1e-12:
                step = step0 * 0.5
                # restart with tighter tol by continuing
                pass
    return X

def solve():
    t0 = time.time()
    BUDGET = 168.0
    t_end = t0 + BUDGET
    rng = np.random.default_rng(0)
    try:
        Xb = np.load("best.npy").astype(float)
    except Exception:
        Xb = None
    try:
        Xr = np.load("record.npy").astype(float)
    except Exception:
        Xr = None
    cands = []
    for c in (Xb, Xr):
        if c is not None and c.shape == (76, 15):
            c = _norm(c.astype(float))
            # filter bad
            if np.all(np.isfinite(c)):
                cands.append(c)
    if not cands:
        X = _norm(rng.normal(size=(76, 15)))
        return X
    bv = 1e9
    Xbest = cands[0].copy()
    for c in cands:
        v = _maxcos(c)
        if v < bv:
            bv = v
            Xbest = c.copy()
    n, d = 76, 15
    # quick exact-max squeeze on incumbent (cheap, keeps at least tie)
    if time.time() < t_end - 5:
        Xbest = _hill_max(Xbest, iters=250, step0=0.0015, t_end=t_end - 5)
        bv = _maxcos(Xbest)

    sigmas = [0.02, 0.05, 0.10, 0.20, 0.32, 0.07, 0.14, 0.03]
    trial = 0
    # hole candidate pool size adapts to remaining time
    while time.time() < t_end - 4:
        trial += 1
        if trial > 600:
            break
        sigma = sigmas[(trial - 1) % len(sigmas)]
        # vary number of replacements 1..5
        k_re = 1 + ((trial - 1) % 5)
        if trial % 7 == 0:
            k_re = 2
        # copy and shake source: mostly Xbest
        if trial % 11 == 0:
            # occasional fresh random restart for diversity (balanced basin search)
            Xt = _norm(rng.normal(size=(n, d)))
            # immediately anneal hard below; don't do hole step from Xbest
            # short-circuit to anneal
            remain = t_end - time.time()
            if remain < 6:
                break
            # choose anneal length by time
            Xt = _anneal(Xt, n_iter=1800, T0=0.025, T1=0.0012, lr0=6.0, lr1=0.12, t_end=t_end - 3)
            Xt = _hill_max(Xt, iters=250, step0=0.0015, t_end=t_end - 2)
            v = _maxcos(Xt)
            if v < bv - 1e-12:
                bv = v
                Xbest = Xt.copy()
            continue
        Xt = Xbest.copy()
        # pick victims: worst rowmax first, mixed with random for diversity
        rm, mcur = _rowmax(Xt)
        order = np.argsort(-rm)
        victims = []
        # always take the worst
        victims.append(int(order[0]))
        # add next worst with stride depending on trial
        for j in range(1, k_re):
            if (trial + j) % 2 == 0:
                victims.append(int(order[j % n]))
            else:
                victims.append(int(rng.integers(0, n)))
        # dedupe preserving order
        victims = list(dict.fromkeys(victims))
        # sequential hole reinsertion
        ok = True
        for idx in victims:
            if time.time() > t_end - 4:
                ok = False
                break
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            Y = Xt[mask]
            NS = 3000
            C = rng.normal(size=(NS, d))
            C = _norm(C)
            scores = (C @ Y.T).max(axis=1)
            # pick best few and choose randomly among top to diversify
            k_top = 5
            part = np.argpartition(scores, k_top - 1)[:k_top]
            bi = int(part[int(rng.integers(0, k_top))])
            v0 = C[bi]
            v0 = _refine_hole(v0, Y, n_iter=180, T0=0.008, lr0=0.08, t_end=t_end - 3)
            Xt[idx] = v0 / np.linalg.norm(v0)
        if not ok:
            break
        # gaussian shake on all (tangential-ish)
        if sigma > 0:
            Xt = Xt + sigma * rng.normal(size=(n, d))
            Xt = _norm(Xt)
        # anneal: params by sigma
        if sigma < 0.04:
            Xt = _anneal(Xt, n_iter=900, T0=0.010, T1=0.0009, lr0=3.5, lr1=0.12, t_end=t_end - 2)
        elif sigma < 0.12:
            Xt = _anneal(Xt, n_iter=1200, T0=0.016, T1=0.0009, lr0=5.0, lr1=0.10, t_end=t_end - 2)
        else:
            Xt = _anneal(Xt, n_iter=1500, T0=0.022, T1=0.0011, lr0=6.0, lr1=0.12, t_end=t_end - 2)
        if time.time() > t_end - 2:
            v = _maxcos(Xt)
            if v < bv:
                bv = v
                Xbest = Xt.copy()
            break
        Xt = _hill_max(Xt, iters=200, step0=0.0015, t_end=t_end - 1)
        v = _maxcos(Xt)
        if v < bv - 1e-12:
            bv = v
            Xbest = Xt.copy()
        # also accept tiny equal improvements that improve balance (variance of rowmax)
        elif abs(v - bv) <= 1e-12 and v < 0.2:
            # compare balance: prefer lower std of rowmax / fewer contacts
            rm_new, _ = _rowmax(Xt)
            rm_best, _ = _rowmax(Xbest)
            if float(rm_new.std()) < float(rm_best.std()) - 1e-13:
                Xbest = Xt.copy()
                bv = v
    Xbest = _norm(Xbest)
    return Xbest
