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

def _max_abs(Y):
    G = Y @ Y.T
    np.fill_diagonal(G, 0.0)
    return float(np.abs(G).max())

def _greedy_Y(X):
    n, d = X.shape
    assert n % 2 == 0
    m = n // 2
    G = X @ X.T
    unmatched = set(range(n))
    Y = []
    while unmatched:
        i = min(unmatched)
        # most negative inner product among unmatched
        best_j = None
        best_v = 1e100
        for j in unmatched:
            if j == i:
                continue
            v = G[i, j]
            if v < best_v:
                best_v = v
                best_j = j
        if best_j is None:
            # odd leftover (should not happen)
            v = X[i].copy()
            Y.append(v / (np.linalg.norm(v) + 1e-300))
            unmatched.remove(i)
            break
        v = X[i] - X[best_j]
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            # degenerate, random orthogonal
            v = np.random.randn(d)
            nv = np.linalg.norm(v)
        Y.append(v / nv)
        unmatched.remove(i)
        unmatched.remove(best_j)
    Y = np.array(Y, dtype=np.float64)
    # in case pairing failed to give m rows, pad/truncate
    if Y.shape[0] != m:
        rng = np.random.RandomState(0)
        while Y.shape[0] < m:
            v = rng.randn(d); v /= np.linalg.norm(v)
            Y = np.vstack([Y, v[None, :]])
        Y = Y[:m]
    return _norm(Y)

def _smooth_abs(Y, steps, a0, a1, lr0, lr1):
    Y = _norm(Y.copy())
    n, d = Y.shape
    for s in range(steps):
        f = s / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = Y @ Y.T
        # max abs off-diagonal
        np.fill_diagonal(G, 0.0)
        m = float(np.abs(G).max())
        # stabilized exps
        Eplus = np.exp(alpha * (G - m))
        Eminus = np.exp(alpha * (-G - m))
        np.fill_diagonal(Eplus, 0.0)
        np.fill_diagonal(Eminus, 0.0)
        S = float(Eplus.sum() + Eminus.sum()) / 2.0 + 1e-300
        W = Eplus - Eminus  # symmetric
        D = (W @ Y) / S
        # tangent projection
        dot = (D * Y).sum(axis=1, keepdims=True)
        Dt = D - dot * Y
        # safety clip
        Y = Y - lr * Dt
        Y = _norm(Y)
    return Y

def _smooth(X, steps, T0, T1, lr0, lr1):
    X = _norm(X.copy())
    n, d = X.shape
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

def _lp(x, seconds, r0=1.2e-3, pcap=2500):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = r0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.4:
            break
        G = x @ x.T
        cf = G[iu0]
        m = float(cf.max())
        thr = m - max(20.0 * radius, 2e-9)
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
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
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
        try:
            y = _norm(x + res.x[:-1].reshape(n, d))
        except Exception:
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best:
            best = v; xb = y.copy(); x = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
            if radius < 4e-7:
                break
    return xb

def solve():
    t0 = time.time()
    LIM = 170.0
    n, d, m = 184, 8, 92
    rng = np.random.RandomState(20261103)
    # load incumbents
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
        best_val, best_X = cands[0][0], cands[0][1].copy()
    else:
        best_X = _norm(rng.randn(n, d)); best_val = _max_cos(best_X)

    # build Y initial pool
    Y_inits = []
    for _, X in cands[:2]:
        try:
            Y_inits.append(_greedy_Y(_norm(X)))
        except Exception:
            pass
    # random inits
    for k in range(6):
        Y_inits.append(_norm(rng.randn(m, d)))
    # small perturbed greedy for diversity
    if cands:
        baseX = cands[0][1]
        for k in range(2):
            Xn = _norm(baseX + 0.02 * rng.randn(n, d))
            try:
                Y_inits.append(_greedy_Y(Xn))
            except Exception:
                pass

    # optimize reduced Grassmann packing
    best_Y = None
    best_abs = 1e100
    # budget: keep symmetric phase < ~90s
    for yi, Y0 in enumerate(Y_inits):
        if time.time() - t0 > LIM - 70.0:
            break
        # two-stage increasing alpha
        Y = _smooth_abs(Y0, steps=900, a0=10.0, a1=45.0, lr0=0.08, lr1=0.015)
        Y = _smooth_abs(Y, steps=900, a0=45.0, a1=130.0, lr0=0.015, lr1=0.004)
        v = _max_abs(Y)
        if v < best_abs:
            best_abs = v
            best_Y = Y.copy()
        # extra polish of promising one with higher alpha
        if v < best_abs + 0.002 and (time.time() - t0 < LIM - 80.0):
            Y2 = _smooth_abs(Y, steps=600, a0=130.0, a1=220.0, lr0=0.003, lr1=0.0015)
            v2 = _max_abs(Y2)
            if v2 < best_abs:
                best_abs = v2
                best_Y = Y2.copy()

    if best_Y is None:
        best_Y = _norm(rng.randn(m, d))

    # duplicate antipodally
    X0 = np.vstack([best_Y, -best_Y])
    X0 = _norm(X0)
    v0 = _max_cos(X0)

    # release constraint: refine all 184 points (try 2 variants, keep best)
    Xrel = _smooth(X0, steps=700, T0=0.012, T1=0.0015, lr0=0.02, lr1=0.002)
    vrel = _max_cos(Xrel)
    # second variant with noise to break exact symmetry
    if time.time() - t0 < LIM - 45.0:
        Xn2 = _norm(X0 + 0.003 * rng.randn(n, d))
        Xrel2 = _smooth(Xn2, steps=700, T0=0.010, T1=0.0012, lr0=0.018, lr1=0.002)
        vrel2 = _max_cos(Xrel2)
        if vrel2 < vrel:
            Xrel, vrel = Xrel2, vrel2

    # choose release base
    Xbase = Xrel if vrel < v0 + 0.01 else X0

    # single LP polish
    rem = LIM - (time.time() - t0) - 1.0
    if rem > 3.0:
        Xpol = _lp(Xbase, seconds=min(rem, 28.0), r0=1.2e-3)
        vpol = _max_cos(Xpol)
    else:
        Xpol, vpol = Xbase, _max_cos(Xbase)

    # final selection vs incumbent
    if vpol < best_val:
        out = _norm(Xpol)
    else:
        # if symmetric path failed, still try quick smooth+short LP on incumbent? No extra LP (once rule) - just return incumbent refined by already-done work if better else incumbent
        # return incumbent to guarantee validity
        if vpol < best_val + 0.005 and vpol < _max_cos(best_X):
            out = _norm(Xpol)
        else:
            out = _norm(best_X)
    # sanitize
    out = np.where(np.isfinite(out), out, 0.0)
    out = _norm(out)
    nr = np.linalg.norm(out, axis=1)
    for i in range(n):
        if not np.isfinite(nr[i]) or nr[i] < 1e-12:
            v = rng.randn(d); v /= np.linalg.norm(v)
            out[i] = v
    return _norm(out)
