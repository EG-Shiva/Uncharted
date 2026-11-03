import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _build_X(P):
    # P shape (12,9): 7 seeds + 5 specials, rows unit
    seeds = P[:7]
    specs = P[7:]
    n, d = 131, 9
    X = np.empty((n, d), dtype=float)
    for i in range(7):
        s = seeds[i]
        sf = s[::-1].copy()
        base = i * 18
        for k in range(9):
            X[base + k] = np.roll(s, k)
            X[base + 9 + k] = np.roll(sf, k)
    X[126:] = specs
    return X

def _grad_P_from_grad_X(gX, P):
    # gX (131,9) tangential, P (12,9)
    gP = np.zeros_like(P)
    for i in range(7):
        base = i * 18
        acc = np.zeros(9, dtype=float)
        for k in range(9):
            gc = gX[base + k]
            acc += np.roll(gc, -k)
        for k in range(9):
            gc = gX[base + 9 + k]
            # inverse: flip(roll(gc,-k))
            acc += np.flip(np.roll(gc, -k))
        gP[i] = acc
    gP[7:] = gX[126:]
    # project to tangent of P rows
    dot = np.sum(gP * P, axis=1, keepdims=True)
    gP = gP - dot * P
    return gP

def _symmetric_adam(P0, steps, rng):
    P = _norm(P0.copy())
    m1 = np.zeros_like(P)
    m2 = np.zeros_like(P)
    b1, b2, eps = 0.9, 0.999, 1e-8
    n = 131
    iu = np.triu_indices(n, 1)
    bestP = P.copy()
    # evaluate initial
    X0 = _build_X(P)
    bestv = _max_cos(X0)
    for k in range(steps):
        frac = k / max(1, steps - 1)
        alpha = 30.0 * (10.0 ** frac)  # 30 -> 300
        T = 1.0 / alpha
        lr = 0.01 * ((0.001 / 0.01) ** frac)
        X = _build_X(P)
        G = X @ X.T
        c = G[iu]
        mx = float(c.max())
        # softmax weights
        E = np.exp((c - mx) / max(T, 1e-12))
        S = float(E.sum())
        Wv = E / (S + 1e-300)
        # build symmetric W matrix
        W = np.zeros((n, n), dtype=float)
        W[iu] = Wv
        W = W + W.T
        gradX = W @ X
        # tangential projection
        dot = np.sum(gradX * X, axis=1, keepdims=True)
        gradX = gradX - dot * X
        # scale (normalize by mean weight) to keep lr stable
        # W sum = 2 (since symmetric counts twice?) use S scaling already; divide by ~1
        gP = _grad_P_from_grad_X(gradX, P)
        # Adam
        m1 = b1 * m1 + (1 - b1) * gP
        m2 = b2 * m2 + (1 - b2) * (gP * gP)
        c1 = 1 - b1 ** (k + 1)
        c2 = 1 - b2 ** (k + 1)
        mh = m1 / c1
        vh = m2 / c2
        P = P - lr * mh / (np.sqrt(vh) + eps)
        P = _norm(P)
        # track best with true max (cheap every 50 steps + last)
        if (k % 50 == 0) or (k == steps - 1):
            try:
                v = _max_cos(_build_X(P))
                if v < bestv:
                    bestv = v
                    bestP = P.copy()
            except Exception:
                pass
    return bestP, bestv

def _smooth_release(x, steps, T0, T1, lr0, lr1, mom=0.85):
    x = _norm(x)
    v = np.zeros_like(x)
    n = x.shape[0]
    for k in range(steps):
        frac = k / max(1, steps - 1)
        T = T0 + (T1 - T0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp((G - m) / max(T, 1e-9))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        s = float(W.sum()) / n + 1e-12
        grad = grad / s
        if not np.all(np.isfinite(grad)):
            break
        v = mom * v + (1.0 - mom) * grad
        x = x - lr * v
        x = _norm(x)
    return x

def _lp_polish_deadline(x, deadline):
    x = _norm(x)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline:
        if radius < 1e-14:
            break
        try:
            G = x @ x.T
            c_all = G[iu0]
        except Exception:
            break
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 3500:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:3500]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I = iu0[0][sel]
        J = iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate(
                [I[:, None] * d + np.arange(d),
                 J[:, None] * d + np.arange(d),
                 np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            y = _norm(y)
            if not np.all(np.isfinite(y)):
                radius *= 0.5
                continue
            vv = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if deadline - time.time() < 0.25:
            break
    return x

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    SYM_END = t0 + 60.0
    rng = np.random.default_rng(12345)
    n, d = 131, 9
    xb = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=float)
            if v.shape == (n, d) and np.all(np.isfinite(v)):
                v = _norm(v)
                s = _max_cos(v)
                if s < best_val:
                    best_val = s
                    xb = v.copy()
        except Exception:
            continue
    if xb is None:
        xb = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(xb)

    # symmetric search: several restarts
    best_sym_X = None
    best_sym_v = 1e9
    restarts = 0
    # steps per restart adapt to remaining time
    while time.time() < SYM_END - 1.0 and restarts < 20:
        if HARD_END - time.time() < 50:
            break
        try:
            P0 = _norm(rng.standard_normal((12, 9)))
            # remove near-duplicates within orbit: perturb seed if needed
            # check each seed's orbit internal max
            for si in range(7):
                s = P0[si]
                orb = np.empty((18, 9))
                for k in range(9):
                    orb[k] = np.roll(s, k)
                    orb[9 + k] = np.roll(s[::-1], k)
                orb = _norm(orb)
                G = orb @ orb.T
                np.fill_diagonal(G, -2.0)
                if float(G.max()) > 0.99:
                    P0[si] = _norm((s + 0.5 * rng.standard_normal(9)[None, :]).ravel())
            # steps: 900 for early restarts, fewer if time short
            remain = SYM_END - time.time()
            steps = 900
            if remain < 8:
                steps = 400
            elif remain < 20:
                steps = 700
            Popt, _v = _symmetric_adam(P0, steps, rng)
            Xsym = _build_X(Popt)
            Xsym = _norm(Xsym)
            v = _max_cos(Xsym)
            if v < best_sym_v:
                best_sym_v = v
                best_sym_X = Xsym.copy()
            # also keep global best if symmetric itself beats record
            if v < best_val - 1e-12:
                best_val = v
                xb = Xsym.copy()
        except Exception:
            pass
        restarts += 1

    # release symmetry from best symmetric config
    cand = None
    cand_v = 1e9
    if best_sym_X is not None:
        try:
            Xrel = _norm(best_sym_X + 1e-3 * rng.standard_normal((n, d)))
            # 1500 ordinary repulsion steps
            Xrel = _smooth_release(Xrel, 900, 0.020, 0.003, 0.014, 0.002, mom=0.85)
            if time.time() < HARD_END - 45:
                Xrel = _smooth_release(Xrel, 600, 0.004, 0.0012, 0.004, 0.001, mom=0.80)
            v = _max_cos(Xrel)
            cand, cand_v = Xrel, v
            if v < best_val - 1e-12:
                best_val = v
                xb = Xrel.copy()
            # second released variant with different noise
            try:
                Xrel2 = _norm(best_sym_X + 1e-3 * rng.standard_normal((n, d)))
                Xrel2 = _smooth_release(Xrel2, 700, 0.018, 0.002, 0.012, 0.002, mom=0.85)
                v2 = _max_cos(Xrel2)
                if v2 < cand_v:
                    cand, cand_v = Xrel2.copy(), v2
                if v2 < best_val - 1e-12:
                    best_val = v2
                    xb = Xrel2.copy()
            except Exception:
                pass
        except Exception:
            cand = None
    else:
        cand = None

    # LP polish released candidate (up to 40s) and also polish incumbent lightly
    # polish the better of cand / xb
    try:
        target = None
        if cand is not None:
            target = cand.copy()
        else:
            target = xb.copy() + 1e-4 * rng.standard_normal((n, d))
            target = _norm(target)
        deadline = min(HARD_END - 2.0, time.time() + 40.0)
        if deadline - time.time() > 3.0:
            target = _lp_polish_deadline(target, deadline)
            v = _max_cos(target)
            if v < best_val - 1e-12:
                best_val = v
                xb = target.copy()
            elif cand is None or v < cand_v:
                cand, cand_v = target, v
    except Exception:
        pass

    # final safety: return whichever is better among xb and cand/sym
    try:
        bestX = xb
        bestV = _max_cos(bestX)
        if cand is not None:
            try:
                vc = _max_cos(cand)
                if vc < bestV:
                    bestX, bestV = cand, vc
            except Exception:
                pass
        if best_sym_X is not None:
            try:
                vs = _max_cos(best_sym_X)
                if vs < bestV:
                    bestX, bestV = best_sym_X, vs
            except Exception:
                pass
        xb = _norm(bestX)
    except Exception:
        pass
    return _norm(xb)
