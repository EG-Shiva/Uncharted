import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _max_abs(Y):
    G = Y @ Y.T
    np.fill_diagonal(G, 0.0)
    return float(np.abs(G).max())

def _greedy_antipodal(X):
    n = X.shape[0]
    D = X @ X.T
    un = set(range(n))
    Y = []
    while un:
        i = min(un)
        un.remove(i)
        if not un:
            break
        # closest to antipode of i among remaining
        j = min(un, key=lambda k: D[i, k])
        un.remove(j)
        v = X[i] - X[j]
        nv = np.linalg.norm(v)
        if nv == 0 or not np.isfinite(nv):
            v = X[i].copy()
            nv = np.linalg.norm(v)
        Y.append(v / nv)
    return np.array(Y, dtype=float)

def _adam_abs(Y, steps, a0, a1, lr0, lr1, deadline=None):
    m1 = np.zeros_like(Y)
    m2 = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        if deadline is not None and (s & 127) == 0 and time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        alpha = a0 * (a1 / a0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = Y @ Y.T
        np.fill_diagonal(G, 0.0)
        mv = float(np.abs(G).max())
        # smooth abs minimax
        Wp = np.exp(alpha * (G - mv))
        Wn = np.exp(alpha * (-G - mv))
        np.fill_diagonal(Wp, 0.0)
        np.fill_diagonal(Wn, 0.0)
        S = float(Wp.sum() + Wn.sum())
        if not np.isfinite(S) or S == 0:
            break
        W = (Wp - Wn) / S
        grad = W @ Y
        dot = (grad * Y).sum(axis=1, keepdims=True)
        gt = grad - dot * Y
        m1 = b1 * m1 + (1 - b1) * gt
        m2 = b2 * m2 + (1 - b2) * (gt * gt)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        Y = Y - lr * (m1 / bc1) / (np.sqrt(m2 / bc2) + eps)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        Y = Y / nr
        if not np.all(np.isfinite(Y)):
            break
    return Y

def _adam_relax(X, steps, a0, a1, lr0, lr1, deadline=None, hinge=None):
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        if deadline is not None and (s & 127) == 0 and time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        alpha = a0 * (a1 / a0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        if hinge is not None:
            # below-record hinge: shift focus to pairs above hinge
            # use smooth max centered at max(hinge, current max) blend
            mv = float(G.max())
            # if mv below hinge, still optimize standard
            W = np.exp(alpha * (G - mv))
        else:
            mv = float(G.max())
            W = np.exp(alpha * (G - mv))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum())
        if not np.isfinite(S) or S == 0:
            break
        W = W / S
        grad = W @ X
        dot = (grad * X).sum(axis=1, keepdims=True)
        gt = grad - dot * X
        m1 = b1 * m1 + (1 - b1) * gt
        m2 = b2 * m2 + (1 - b2) * (gt * gt)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        X = X - lr * (m1 / bc1) / (np.sqrt(m2 / bc2) + eps)
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
        if not np.all(np.isfinite(X)):
            break
    return X

def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cmat = (x @ x.T)[iu]
        m = float(cmat.max())
        sel = cmat > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cmat[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def solve():
    T0 = time.time()
    BUDGET = 176.0
    SYM_END = T0 + 52.0
    BRK_END = T0 + 148.0
    rng = np.random.default_rng(12345)
    best = None
    best_val = 1e9
    rec = None
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn).astype(float)
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if a.shape == (94, 13) and np.all(np.isfinite(a)):
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
                if fn == "record.npy":
                    rec = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal(size=(94, 13))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    if rec is None:
        rec = best.copy()
    n, d = 94, 13
    global_best = best.copy()
    global_val = best_val
    record_val = float(min(best_val, _max_cos(rec)))

    # ---- Stage 1: antipodally symmetric ansatz, 47 reps ----
    Y_inits = []
    try:
        Yg1 = _greedy_antipodal(best)
        if Yg1.shape == (47, 13):
            Y_inits.append(Yg1)
    except Exception:
        pass
    try:
        Yg2 = _greedy_antipodal(rec)
        if Yg2.shape == (47, 13):
            # avoid duplicate
            Y_inits.append(Yg2)
    except Exception:
        pass
    # random inits
    for k in range(4):
        Yr = rng.standard_normal(size=(47, 13))
        Yr /= np.linalg.norm(Yr, axis=1, keepdims=True)
        Y_inits.append(Yr)
    # perturbed greedy for diversity
    if len(Y_inits) >= 1:
        base = Y_inits[0]
        for k in range(2):
            N = rng.standard_normal(size=(47, 13))
            # tangent
            N = N - (N * base).sum(axis=1, keepdims=True) * base
            sig = 0.15 + 0.15 * k
            Yp = base + sig * N
            Yp /= np.linalg.norm(Yp, axis=1, keepdims=True)
            Y_inits.append(Yp)

    Y_opts = []
    for yi, Y0 in enumerate(Y_inits):
        if time.time() > SYM_END - 3:
            break
        Y = Y0.copy()
        # anneal smooth abs minimax, two rounds coarse->fine
        try:
            Y = _adam_abs(Y, steps=900, a0=15.0, a1=200.0, lr0=0.02, lr1=0.003, deadline=SYM_END)
            if time.time() < SYM_END - 2:
                Y = _adam_abs(Y, steps=700, a0=60.0, a1=400.0, lr0=0.008, lr1=0.0015, deadline=SYM_END)
        except Exception:
            continue
        try:
            va = _max_abs(Y)
        except Exception:
            continue
        Y_opts.append((va, Y.copy()))
    Y_opts.sort(key=lambda t: t[0])
    # keep top 3 symmetric
    Y_top = [Y for _, Y in Y_opts[:3]] if Y_opts else []

    # ---- Stage 2: deliberately break symmetry ----
    # expand symmetric optima to 94 and shake tangentially at several magnitudes
    sigmas = [0.002, 0.006, 0.015, 0.035, 0.07]
    # also include unperturbed expansions for reference
    candidates = []
    for Y in Y_top:
        Xs = np.vstack([Y, -Y])
        Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
        candidates.append(Xs)
    # if symmetric stage failed, use record/best as fallback seeds
    if not candidates:
        candidates = [global_best.copy()]

    # hinge target: below record
    hinge_target = record_val - 5e-4

    cur_pool = candidates[:]
    # generate shaken replicas and anneal with full hinge/LSE objective
    trial = 0
    for Y in Y_top:
        Xs0 = np.vstack([Y, -Y])
        Xs0 /= np.linalg.norm(Xs0, axis=1, keepdims=True)
        for sig in sigmas:
            if time.time() > BRK_END - 5:
                break
            for rep in range(2):
                if time.time() > BRK_END - 5:
                    break
                trial += 1
                rs = np.random.default_rng(9000 + trial * 101 + int(sig * 1e5))
                N = rs.standard_normal(size=(94, 13))
                N = N - (N * Xs0).sum(axis=1, keepdims=True) * Xs0
                Xp = Xs0 + sig * N
                Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
                try:
                    # annealed smooth minimax (below-record hinge phase: start soft, end sharp)
                    Xr = _adam_relax(Xp.copy(), steps=650, a0=25.0, a1=300.0, lr0=0.01, lr1=0.0015, deadline=BRK_END, hinge=hinge_target)
                    if time.time() < BRK_END - 3:
                        Xr = _adam_relax(Xr, steps=350, a0=80.0, a1=380.0, lr0=0.004, lr1=0.0008, deadline=BRK_END, hinge=hinge_target)
                except Exception:
                    continue
                v = _max_cos(Xr)
                if v < global_val - 1e-12:
                    global_val = v
                    global_best = Xr.copy()
                cur_pool.append(Xr.copy())
                # keep pool bounded: retain 6 best distinct
                if len(cur_pool) > 12:
                    scored = [( _max_cos(X), X) for X in cur_pool]
                    scored.sort(key=lambda t: t[0])
                    cur_pool = [X for _, X in scored[:6]]
    # extra unstructured refinement of current global best with small shakes
    # (still symmetry-broken family, ensures competitiveness)
    for extra in range(6):
        if time.time() > BRK_END - 3:
            break
        rs = np.random.default_rng(5555 + extra * 77)
        sig = [0.004, 0.01, 0.025, 0.05, 0.008, 0.02][extra % 6]
        N = rs.standard_normal(size=(94, 13))
        N = N - (N * global_best).sum(axis=1, keepdims=True) * global_best
        Xp = global_best + sig * N
        Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        try:
            Xr = _adam_relax(Xp.copy(), steps=600, a0=30.0, a1=320.0, lr0=0.008, lr1=0.0012, deadline=BRK_END, hinge=hinge_target)
        except Exception:
            continue
        v = _max_cos(Xr)
        if v < global_val - 1e-12:
            global_val = v
            global_best = Xr.copy()
            cur_pool.append(Xr.copy())

    # ---- Stage 3: LP polish elites ----
    # gather elites: global best + best few from pool + original record
    elites = [(global_val, global_best.copy())]
    for X in cur_pool:
        try:
            v = _max_cos(X)
            elites.append((v, X.copy()))
        except Exception:
            pass
    elites.append((record_val, rec.copy()))
    # dedup sort
    elites.sort(key=lambda t: t[0])
    keep = elites[:3]
    total_end = T0 + BUDGET - 2.0
    remaining = max(0.0, total_end - time.time())
    per = remaining / max(1, len(keep)) if keep else 0
    per = min(max(per, 4.0), 16.0)
    final_best = global_best.copy()
    final_val = global_val
    for vv, XX in keep:
        if time.time() > total_end - 1.5:
            break
        dl = min(time.time() + per, total_end)
        try:
            P, pv = _lp_polish(XX.copy(), dl)
        except Exception:
            continue
        if pv < final_val:
            final_val = pv
            final_best = P.copy()
    final_best = final_best / np.linalg.norm(final_best, axis=1, keepdims=True)
    return final_best
