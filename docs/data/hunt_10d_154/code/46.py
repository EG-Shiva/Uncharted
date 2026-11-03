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
    return float(g.max())

def _max_abs(Y):
    G = Y @ Y.T
    np.fill_diagonal(G, 0.0)
    return float(np.abs(G).max())

def _relax_abs(Y, steps, lr, T0, T1):
    # minimize softmax of |<yi,yj>| over unit rows
    m_ad = np.zeros_like(Y)
    v_ad = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p = 1.0
    b2p = 1.0
    if steps > 1:
        logr = np.log(max(T1, 1e-12) / max(T0, 1e-12)) / (steps - 1)
    else:
        logr = 0.0
    for t in range(1, steps + 1):
        T = T0 * np.exp(logr * (t - 1))
        if T < 5e-5:
            T = 5e-5
        G = Y @ Y.T
        np.fill_diagonal(G, 0.0)
        A = np.abs(G)
        mv = float(A.max())
        E = np.exp((A - mv) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            break
        Sg = np.sign(G)  # 0 on diagonal already
        Es = E * Sg
        Grad = (Es @ Y) / S
        dot = (Grad * Y).sum(axis=1, keepdims=True)
        Gt = Grad - dot * Y
        b1p *= b1
        b2p *= b2
        m_ad = b1 * m_ad + (1.0 - b1) * Gt
        v_ad = b2 * v_ad + (1.0 - b2) * (Gt * Gt)
        mhat = m_ad / (1.0 - b1p)
        vhat = v_ad / (1.0 - b2p)
        step = mhat / (np.sqrt(vhat) + eps)
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        step = step / np.maximum(sn / 1.0, 1.0)
        Y = Y - lr * step
        Y = _norm(Y)
    return Y

def _relax_std(X, steps, lr, T0, T1, deadline=1e18):
    m_ad = np.zeros_like(X)
    v_ad = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p = 1.0
    b2p = 1.0
    if steps > 1:
        logr = np.log(max(T1, 1e-12) / max(T0, 1e-12)) / (steps - 1)
    else:
        logr = 0.0
    for t in range(1, steps + 1):
        if (t % 200) == 0 and time.time() > deadline:
            break
        T = T0 * np.exp(logr * (t - 1))
        if T < 5e-5:
            T = 5e-5
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mv = float(G.max())
        E = np.exp((G - mv) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            break
        Grad = (E @ X) / S
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        b1p *= b1
        b2p *= b2
        m_ad = b1 * m_ad + (1.0 - b1) * Gt
        v_ad = b2 * v_ad + (1.0 - b2) * (Gt * Gt)
        mhat = m_ad / (1.0 - b1p)
        vhat = v_ad / (1.0 - b2p)
        step = mhat / (np.sqrt(vhat) + eps)
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        step = step / np.maximum(sn / 1.0, 1.0)
        X = X - lr * step
        X = _norm(X)
    return X

def _lp_polish(x, seconds, deadline=1e18):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() > deadline:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
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
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    DEAD = t0 + 170.0
    rng = np.random.default_rng(20240)
    n_full, m, d = 154, 77, 10
    # load incumbent
    inc = None
    inc_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n_full, d):
                a = _norm(a.astype(float))
                v = _max_cos(a)
                if v < inc_val:
                    inc_val = v
                    inc = a.copy()
        except Exception:
            pass
    if inc is None:
        inc = _norm(rng.normal(size=(n_full, d)))
        inc_val = _max_cos(inc)
    thresh = 0.4012521675368

    # Phase A: >=16 antipodal starts on absolute-Gram objective
    NSTART = 22
    cands = []
    for s in range(NSTART):
        if time.time() > t0 + 95.0:
            break
        Y0 = _norm(rng.normal(size=(m, d)))
        # per-start anneal jitter for diversity (deterministic via rng)
        j1 = float(rng.random())
        j2 = float(rng.random())
        T0a = 0.016 + 0.010 * j1
        T1a = 0.0035 + 0.0015 * j2
        lra = 0.018 + 0.008 * (1.0 - j1)
        Y0 = _relax_abs(Y0, steps=650, lr=lra, T0=T0a, T1=T1a)
        if time.time() > t0 + 105.0:
            # skip second stage if late, still keep
            v = _max_abs(Y0)
            cands.append((v, Y0))
            continue
        T0b = 0.0045 + 0.0015 * j2
        T1b = 0.0007 + 0.0004 * j1
        lrb = 0.010 + 0.004 * j2
        Y0 = _relax_abs(Y0, steps=650, lr=lrb, T0=T0b, T1=T1b)
        v = _max_abs(Y0)
        cands.append((v, Y0.copy()))
    if len(cands) == 0:
        best = inc.copy()
        best_val = inc_val
        return _norm(best)
    cands.sort(key=lambda kv: kv[0])
    # retain several geometrically distinct finalists
    finals = []
    invs = []
    for v, Y in cands:
        G = Y @ Y.T
        np.fill_diagonal(G, 0.0)
        tri = np.sort(np.abs(G[np.triu_indices(m, 1)]))
        # subsample for distance to be cheap
        feat = tri[::37]  # ~76 values
        keep = True
        for f in invs:
            if np.linalg.norm(feat - f) < 2e-3:
                keep = False
                break
        if keep or len(finals) < 2:
            finals.append((v, Y))
            invs.append(feat)
        else:
            # still allow if we lack enough finalists
            if len(finals) < 6:
                # check looser threshold
                mind = min(np.linalg.norm(feat - f) for f in invs) if invs else 1e9
                if mind > 5e-4:
                    finals.append((v, Y))
                    invs.append(feat)
        if len(finals) >= 6:
            break
    if len(finals) == 0:
        finals = cands[:6]
    # Phase B: unfreeze each finalist, break symmetry, unrestricted relax
    unfrozen = []
    for k, (v, Y) in enumerate(finals):
        if time.time() > t0 + 130.0:
            break
        X = np.concatenate([Y, -Y], axis=0)
        # small symmetry-breaking jitter (deterministic)
        X = _norm(X + rng.normal(scale=0.0025, size=X.shape))
        # two-stage unrestricted relaxation
        X = _relax_std(X, steps=500, lr=0.014, T0=0.009, T1=0.0012, deadline=t0 + 135.0)
        X = _relax_std(X, steps=350, lr=0.009, T0=0.0035, T1=0.0007, deadline=t0 + 135.0)
        val = _max_cos(X)
        unfrozen.append((val, X))
    if len(unfrozen) == 0:
        best = inc.copy()
        best_val = inc_val
        return _norm(best)
    unfrozen.sort(key=lambda kv: kv[0])
    bval, bX = unfrozen[0]
    # keep only if beats threshold/incumbent; otherwise return incumbent polished lightly
    if bval < min(inc_val, thresh) - 1e-12 or bval < inc_val - 2e-7:
        # final capped LP polish on the antipodal-derived winner
        remain = DEAD - time.time() - 2.0
        if remain > 5.0:
            lp_t = min(max(5.0, remain - 2.0), 28.0)
            try:
                bX = _lp_polish(bX, lp_t, deadline=DEAD)
                bval2 = _max_cos(bX)
                if bval2 < bval:
                    bval = bval2
                else:
                    pass
            except Exception:
                pass
        if bval < inc_val:
            out = _norm(bX)
            out[~np.isfinite(out)] = 0.0
            return _norm(out)
        else:
            return _norm(inc)
    else:
        # antipodal path did not beat incumbent: return incumbent (guarantees no regression)
        # spend leftover on tiny LP snap of incumbent if it helps
        remain = DEAD - time.time() - 2.0
        if remain > 8.0 and bval < inc_val + 0.01:
            # try one more aggressive unfrozen refinement of best finalist with longer relax
            try:
                Yb = finals[0][1]
                X2 = np.concatenate([Yb, -Yb], axis=0)
                X2 = _norm(X2 + rng.normal(scale=0.006, size=X2.shape))
                X2 = _relax_std(X2, steps=600, lr=0.016, T0=0.012, T1=0.0009, deadline=DEAD - 8.0)
                v2 = _max_cos(X2)
                if v2 < inc_val:
                    X2 = _lp_polish(X2, min(10.0, DEAD - time.time() - 2.0), deadline=DEAD)
                    v3 = _max_cos(X2)
                    if v3 < inc_val:
                        return _norm(X2)
            except Exception:
                pass
        return _norm(inc)
