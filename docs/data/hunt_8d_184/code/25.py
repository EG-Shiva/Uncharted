import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-300
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _max_abs_y(Y):
    G = Y @ Y.T
    np.fill_diagonal(G, 0.0)
    return float(np.abs(G).max())

def _greedy_Y(X, m=92):
    n = X.shape[0]
    G = X @ X.T
    unpaired = list(range(n))
    # deterministic order: use set sorted
    unpaired_set = set(unpaired)
    reps = []
    # sort to be deterministic
    while len(unpaired_set) >= 2:
        i = min(unpaired_set)
        # find j with minimal inner product (most negative)
        best_j = None
        best_v = 1e9
        for j in unpaired_set:
            if j == i:
                continue
            v = G[i, j]
            if v < best_v:
                best_v = v
                best_j = j
        j = best_j
        v = X[i] - X[j]
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            # degenerate, use X[i]
            v = X[i].copy()
            nv = np.linalg.norm(v)
        v = v / nv
        reps.append(v)
        unpaired_set.remove(i)
        unpaired_set.remove(j)
        if len(reps) >= m:
            break
    reps = np.array(reps, dtype=np.float64)
    if len(reps) < m:
        # pad with random orthogonal-ish (should not happen for 184)
        rng = np.random.RandomState(0)
        extra = rng.randn(m - len(reps), X.shape[1])
        extra = _norm(extra)
        reps = np.vstack([reps, extra])
    if len(reps) > m:
        reps = reps[:m]
    return _norm(reps)

def _smooth_Y(Y0, steps, T0, T1, lr0, lr1):
    Y = _norm(Y0.copy())
    m, d = Y.shape
    for s in range(steps):
        f = s / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = Y @ Y.T
        np.fill_diagonal(G, 0.0)
        A = np.abs(G)
        am = float(A.max())
        E = np.exp((A - am) / T)
        np.fill_diagonal(E, 0.0)
        Sgn = np.sign(G)
        S = Sgn * E
        rs = E.sum(axis=1, keepdims=True) + 1e-300
        D = (S @ Y) / rs
        dot = (D * Y).sum(axis=1, keepdims=True)
        Dt = D - dot * Y
        Y = Y - lr * Dt
        Y = _norm(Y)
    return Y

def _smooth_X(X0, steps, T0, T1, lr0, lr1):
    X = _norm(X0.copy())
    for s in range(steps):
        f = s / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = X @ X.T
        np.fill_diagonal(G, -1.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True) + 1e-300
        D = (E @ X) / rs
        dot = (D * X).sum(axis=1, keepdims=True)
        Dt = D - dot * X
        X = X - lr * Dt
        X = _norm(X)
    return X

def _lp_polish(x, seconds, r0=1.2e-3, pcap=2600):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = r0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while True:
        if time.time() - start > seconds - 0.4:
            break
        if radius < 1e-13:
            break
        G = x @ x.T
        cf = G[iu0]
        m = float(cf.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = cf > thr
        I = iu0[0][sel]; J = iu0[1][sel]; cs = cf[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > pcap:
            idx = np.argpartition(-cs, pcap)[:pcap]
            I, J, cs = I[idx], J[idx], cs[idx]
            p = pcap
        nv = n * d + 1
        rows = np.concatenate([np.repeat(np.arange(p), d), np.repeat(np.arange(p), d), np.arange(p)])
        cols = np.concatenate([np.repeat(I * d, d) + np.tile(ar, p),
                               np.repeat(J * d, d) + np.tile(ar, p),
                               np.full(p, n * d)])
        vals = np.concatenate([x[J].ravel(), x[I].ravel(), -np.ones(p)])
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm(x + res.x[:-1].reshape(n, d))
        v = _max_cos(y)
        if v < best:
            best = v; xb = y.copy(); x = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return xb

def solve():
    t0 = time.time()
    LIM = 172.0
    rng = np.random.RandomState(12345)
    n, d, m = 184, 8, 92
    cands = []
    for f in ("best.npy", "record.npy"):
        try:
            c = np.asarray(np.load(f), dtype=np.float64)
            if c.shape == (n, d):
                c = _norm(c)
                cands.append((_max_cos(c), c))
        except Exception:
            pass
    if cands:
        cands.sort(key=lambda t: t[0])
        best_val, x_best_loaded = cands[0][0], cands[0][1].copy()
    else:
        x_best_loaded = _norm(rng.randn(n, d))
        best_val = _max_cos(x_best_loaded)

    # Build Y inits: greedy from best/record + random
    Y_inits = []
    # greedy pairings
    for _, X in cands:
        try:
            Yg = _greedy_Y(X, m=m)
            Y_inits.append(Yg)
        except Exception:
            pass
    # random inits
    Y_inits.append(_norm(rng.randn(m, d)))
    Y_inits.append(_norm(rng.randn(m, d)))
    # subset init: random 92 of best
    try:
        perm = rng.permutation(n)[:m]
        Ysub = _norm(x_best_loaded[perm].copy())
        Y_inits.append(Ysub)
    except Exception:
        pass

    # Screen each Y with short anneal
    screened = []
    for Y0 in Y_inits:
        if time.time() - t0 > LIM - 60:
            break
        Y1 = _smooth_Y(Y0, steps=1800, T0=0.05, T1=0.004, lr0=0.06, lr1=0.004)
        v = _max_abs_y(Y1)
        screened.append((v, Y1))
    if len(screened) == 0:
        Ystar = _norm(rng.randn(m, d))
    else:
        screened.sort(key=lambda t: t[0])
        # refine top 2
        top = screened[:2]
        refined = []
        for v, Y1 in top:
            if time.time() - t0 > LIM - 55:
                refined.append((v, Y1))
                continue
            Y2 = _smooth_Y(Y1, steps=3500, T0=0.02, T1=0.0018, lr0=0.03, lr1=0.003)
            v2 = _max_abs_y(Y2)
            refined.append((v2, Y2))
        refined.sort(key=lambda t: t[0])
        Ystar = refined[0][1].copy()

    # Duplicate antipodally
    Xsym = np.vstack([Ystar, -Ystar])
    Xsym = _norm(Xsym)
    vsym = _max_cos(Xsym)

    # Release symmetry: refine full 184
    remain = LIM - (time.time() - t0)
    # two-stage release refinement
    if remain > 40:
        Xrel = _smooth_X(Xsym, steps=2200, T0=0.025, T1=0.0022, lr0=0.04, lr1=0.003)
        Xrel = _smooth_X(Xrel, steps=1200, T0=0.010, T1=0.0015, lr0=0.012, lr1=0.002)
    else:
        Xrel = _smooth_X(Xsym, steps=1000, T0=0.02, T1=0.002, lr0=0.03, lr1=0.003)
    vrel = _max_cos(Xrel)

    # Choose starting point for final polish: best among released and loaded
    if vrel < best_val:
        x_init = Xrel
        cur_best = vrel
    else:
        # still allow small symmetry-breaking noise to enter different basin
        # perturb Xsym slightly and refine once more if it is close
        if vsym < best_val + 0.005 and (LIM - (time.time() - t0) > 35):
            Xp = _norm(Xsym + 0.01 * rng.randn(n, d))
            Xp = _smooth_X(Xp, steps=1200, T0=0.015, T1=0.002, lr0=0.02, lr1=0.002)
            vp = _max_cos(Xp)
            if vp < vrel:
                Xrel, vrel = Xp, vp
            if vrel < best_val:
                x_init, cur_best = Xrel, vrel
            else:
                # keep loaded best but inject symmetric info? just keep loaded
                x_init, cur_best = x_best_loaded.copy(), best_val
                # if symmetric released is not too bad, prefer it for novelty? no, keep best
        else:
            x_init, cur_best = x_best_loaded.copy(), best_val

    # Final LP polish once on chosen init
    remain = LIM - (time.time() - t0) - 1.0
    if remain > 2.0:
        budget = min(remain, 24.0)
        x_init = _lp_polish(x_init, seconds=budget, r0=1.2e-3, pcap=2600)

    x_init = _norm(x_init)
    x_init = np.where(np.isfinite(x_init), x_init, 0.0)
    nr = np.linalg.norm(x_init, axis=1)
    for i in range(n):
        if nr[i] < 1e-12 or not np.isfinite(nr[i]):
            v = rng.randn(d); v /= np.linalg.norm(v)
            x_init[i] = v
    x_init = _norm(x_init)
    return x_init
