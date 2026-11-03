import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _adam_free(X, steps, a0, a1, lr0, lr1, seed=0):
    n, d = X.shape
    m = np.zeros_like(X); vv = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    mask = np.ones((n, n)); np.fill_diagonal(mask, 0.0)
    for t in range(1, steps + 1):
        f = t / steps
        alpha = a0 * (a1 / a0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cm = G.max()
        E = np.exp(alpha * (G - cm)) * mask
        S = E.sum()
        if not np.isfinite(S) or S <= 0:
            break
        W = E / S
        Grad = W @ X
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        m = b1 * m + (1 - b1) * Gt
        vv = b2 * vv + (1 - b2) * (Gt * Gt)
        mh = m / (1 - b1 ** t); vh = vv / (1 - b2 ** t)
        step = lr * mh / (np.sqrt(vh) + eps)
        sd = (step * X).sum(axis=1, keepdims=True)
        st = step - sd * X
        nm = np.linalg.norm(st, axis=1, keepdims=True) + 1e-12
        clip = np.minimum(1.0, 0.05 / nm)
        X = X - st * clip
        X = _norm(X)
    return X

def _lp_polish(X, seconds):
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.05:
            break
        C = (X @ X.T)[iu]
        m = float(C.max())
        sel = C > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-C[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        Y = X + res.x[:-1].reshape(n, d)
        Y = _norm(Y)
        v = _max_cos(Y)
        if v < best:
            X, best, radius = Y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _build_pool(rng):
    P = []
    # D10 roots (±1,±1,0^8)/sqrt2 : 180
    for i in range(10):
        for j in range(i + 1, 10):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = np.zeros(10); v[i] = si; v[j] = sj
                    P.append(v / np.sqrt(2.0))
    # coordinates ±e
    for i in range(10):
        for s in (1.0, -1.0):
            v = np.zeros(10); v[i] = s
            P.append(v)
    # signed all-ones type: random signs
    for _ in range(600):
        v = rng.choice(np.array([-1.0, 1.0]), size=10)
        P.append(v / np.sqrt(10.0))
    # Hadamard-ish: cyclic shifts of a binary pattern
    # parameterized seeds (a,a,a,b,b,0^5) with several theta
    thetas = [0.3, 0.7, 1.0, 1.2, 0.5, 0.9]
    for th in thetas:
        a = np.cos(th) / np.sqrt(3.0); b = np.sin(th) / np.sqrt(2.0)
        base = np.array([a, a, a, b, b, 0, 0, 0, 0, 0])
        for _ in range(220):
            perm = rng.permutation(10)
            sgn = rng.choice(np.array([-1.0, 1.0]), size=10)
            # keep zeros zero: apply perm then random signs only on nonzeros
            v = base.copy()
            v = v[perm] if False else base[perm]  # permute
            # random sign flips on the 5 nonzeros
            nz = np.abs(v) > 1e-12
            v[nz] *= sgn[nz]
            nrm = np.linalg.norm(v)
            if nrm > 0:
                P.append(v / nrm)
    # patterns (a,a,b,0^7), (a,b,c,0^7)
    for _ in range(300):
        t = rng.rand(3); t = t / np.linalg.norm(t)
        v = np.zeros(10)
        idx = rng.choice(10, size=3, replace=False)
        v[idx] = t * rng.choice(np.array([-1.0, 1.0]), size=3)
        P.append(v / (np.linalg.norm(v) + 1e-30))
    for _ in range(300):
        t = rng.rand(4); t = t / np.linalg.norm(t)
        v = np.zeros(10)
        idx = rng.choice(10, size=4, replace=False)
        v[idx] = t * rng.choice(np.array([-1.0, 1.0]), size=4)
        P.append(v / (np.linalg.norm(v) + 1e-30))
    P = np.array(P, dtype=float)
    P = _norm(P)
    # dedup: greedy keep if max dot < 0.9999 to kept? simple filter
    keep = np.ones(len(P), dtype=bool)
    # random order dedup via blocking
    order = rng.permutation(len(P))
    kept = []
    rep = []
    for oi in order:
        v = P[oi]
        if len(rep) == 0:
            kept.append(oi); rep.append(v)
        else:
            R = np.array(rep)
            # check in chunks to save time: only last 500
            tail = R[-600:]
            if np.abs(tail @ v).max() < 0.9999:
                kept.append(oi); rep.append(v)
            # else discard
        if len(kept) > 2600:
            break
    P2 = P[np.array(kept)]
    return P2

def _greedy_select(pool, rng):
    P = len(pool)
    # start from pool index 0 rotated? deterministic: pick vector with max sum? just 0
    sel = [int(rng.randint(P))]
    # precompute pool norms already 1
    while len(sel) < 154:
        S = pool[sel]  # k x d
        D = pool @ S.T  # P x k
        md = D.max(axis=1)
        md[sel] = 1e9  # exclude
        # mask already selected
        best = int(np.argmin(md))
        sel.append(best)
    return pool[np.array(sel)]

def _swap_sa(X0, pool, rng, iters=12000, alpha=60.0):
    n = X0.shape[0]
    # map pool: for fast membership, mark indices used via nearest match
    # Instead work with X directly and candidate pool vectors
    X = X0.copy()
    G = X @ X.T
    # objective sum exp(alpha*G) upper triangle
    iu = np.triu_indices(n, 1)
    obj = float(np.exp(alpha * (G[iu] - G[iu].max())).sum())  # shifted; rescale not needed for delta? use unshifted delta approx
    # use unshifted for delta: S = sum exp(alpha*g)
    S = float(np.exp(alpha * G[iu]).sum())
    T0, T1 = 0.02, 0.0005
    # precompute pool dots? just random picks
    P = len(pool)
    for it in range(iters):
        f = it / max(iters - 1, 1)
        T = T0 * (T1 / T0) ** f
        i = int(rng.randint(n))
        j = int(rng.randint(P))
        cand = pool[j]
        # skip if almost duplicate of existing (except i)
        dts = X @ cand
        dts[i] = -2
        if dts.max() > 0.998:
            continue
        old = np.exp(alpha * G[i, :])
        old[i] = 0
        sold = old.sum()
        nd = np.exp(alpha * dts)
        nd[i] = 0
        snew = nd.sum()
        delta = (snew - sold)  # proportional to change in S (factor, same)
        if delta < 0 or rng.rand() < np.exp(-delta / (T * n + 1e-12)):
            X[i] = cand
            G[i, :] = X @ X[i]
            G[:, i] = G[i, :]
            G[i, i] = 1.0
            S = S + float(snew - sold)
    return X

def _sym_orbit_opt(rng, steps=500, a0=30.0, a1=400.0, lr=0.02):
    # 15 cyclic orbits (150 pts) + 4 free
    n_orb = 15
    S = rng.normal(size=(n_orb, 10)); S = _norm(S)
    E = rng.normal(size=(4, 10)); E = _norm(E)
    mS = np.zeros_like(S); vS = np.zeros_like(S)
    mE = np.zeros_like(E); vE = np.zeros_like(E)
    b1, b2, eps = 0.9, 0.999, 1e-8
    tglob = 0
    for t in range(1, steps + 1):
        tglob = t
        f = t / steps
        alpha = a0 * (a1 / a0) ** f
        # expand
        blocks = [np.roll(S, k, axis=1) for k in range(10)]
        X = np.concatenate(blocks + [E], axis=0)  # 154
        Xn = _norm(X)
        n = Xn.shape[0]
        G = Xn @ Xn.T
        np.fill_diagonal(G, -10.0)
        cm = G.max()
        Ee = np.exp(alpha * (G - cm))
        np.fill_diagonal(Ee, 0.0)
        Sm = Ee.sum() + 1e-30
        W = Ee / Sm
        Grad = W @ Xn
        dot = (Grad * Xn).sum(axis=1, keepdims=True)
        Gt = Grad - dot * Xn
        # collapse to S,E
        gS = np.zeros_like(S); gE = Gt[-4:]
        for k in range(10):
            gb = Gt[k * n_orb:(k + 1) * n_orb]
            gS += np.roll(gb, -k, axis=1)
        gS /= 10.0
        # project to tangent of seeds (approx)
        gS = gS - (gS * S).sum(axis=1, keepdims=True) * S
        gE = gE - (gE * E).sum(axis=1, keepdims=True) * E
        mS = b1 * mS + (1 - b1) * gS; vS = b2 * vS + (1 - b2) * (gS * gS)
        mE = b1 * mE + (1 - b1) * gE; vE = b2 * vE + (1 - b2) * (gE * gE)
        mhS = mS / (1 - b1 ** t); vhS = vS / (1 - b2 ** t)
        mhE = mE / (1 - b1 ** t); vhE = vE / (1 - b2 ** t)
        stS = lr * mhS / (np.sqrt(vhS) + eps)
        stE = lr * mhE / (np.sqrt(vhE) + eps)
        # clip
        for arr in (stS, stE):
            nm = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
            arr[:] = arr * np.minimum(1.0, 0.05 / nm)
        S = _norm(S - stS); E = _norm(E - stE)
    blocks = [np.roll(S, k, axis=1) for k in range(10)]
    X = np.concatenate(blocks + [E], axis=0)
    return _norm(X)

def solve():
    t0 = time.time()
    BUDGET = 176.0
    rng = np.random.RandomState(20261103)
    try:
        xb = np.load("best.npy")
        xb = _norm(np.asarray(xb, dtype=float))
    except Exception:
        xb = None
    if xb is None or xb.shape != (154, 10):
        xb = _norm(rng.normal(size=(154, 10)))
    best = xb.copy(); best_val = _max_cos(best)
    # Branch 1: structured pool + greedy + swap SA + Adam
    try:
        pool = _build_pool(np.random.RandomState(7))
        Xg = _greedy_select(pool, np.random.RandomState(11))
        Xs = _swap_sa(Xg, pool, np.random.RandomState(13), iters=9000, alpha=60.0)
        v0 = _max_cos(Xs)
        X1 = _adam_free(Xs, steps=900, a0=40.0, a1=450.0, lr0=0.012, lr1=0.0015)
        v1 = _max_cos(X1)
        if v1 < best_val:
            best, best_val = X1.copy(), v1
        elif v0 < best_val:
            best, best_val = Xs.copy(), v0
        b1v = min(v0, v1)
    except Exception:
        b1v = 1e9
    if time.time() - t0 > BUDGET - 60:
        pass
    else:
        # Branch 2: cyclic-symmetric orbit optimization (2 restarts, short)
        try:
            for rs in (101, 202):
                if time.time() - t0 > BUDGET - 55:
                    break
                rng2 = np.random.RandomState(rs)
                # randomize inside via global rng sequence: use RandomState rs
                import random as _r
                # temporarily seed numpy global for _sym_orbit_opt rng pass
                Xs2 = _sym_orbit_opt(rng2, steps=350, a0=25.0, a1=350.0, lr=0.025)
                v2s = _max_cos(Xs2)
                # release symmetry: free Adam
                Xs2r = _adam_free(Xs2, steps=700, a0=50.0, a1=500.0, lr0=0.010, lr1=0.0012)
                v2r = _max_cos(Xs2r)
                if v2r < best_val:
                    best, best_val = Xs2r.copy(), v2r
                elif v2s < best_val:
                    best, best_val = Xs2.copy(), v2s
        except Exception:
            pass
    # final: short free refine of current best + LP polish with remaining time
    remain = BUDGET - (time.time() - t0)
    if remain > 8:
        try:
            # light Adam escape on best
            Xf = _adam_free(best, steps=400, a0=60.0, a1=600.0, lr0=0.006, lr1=0.001)
            vf = _max_cos(Xf)
            if vf < best_val:
                best, best_val = Xf, vf
        except Exception:
            pass
    remain = BUDGET - (time.time() - t0)
    sec = max(1.0, remain - 2.0)
    sec = min(sec, 28.0)
    try:
        Xp = _lp_polish(best, seconds=sec)
        vp = _max_cos(Xp)
        if vp < best_val and np.all(np.isfinite(Xp)):
            best, best_val = Xp, vp
    except Exception:
        pass
    best = _norm(best)
    best = np.where(np.isfinite(best), best, 0.0)
    best = _norm(best)
    return best
