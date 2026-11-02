import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lse_anneal(X, steps, a0, a1, s0, s1):
    n, d = X.shape
    la0 = np.log(a0); la1 = np.log(a1)
    for k in range(steps):
        f = k / (steps - 1) if steps > 1 else 1.0
        alpha = float(np.exp(la0 + (la1 - la0) * f))
        step = s0 + (s1 - s0) * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ X
        proj = np.sum(grad * X, axis=1, keepdims=True)
        grad -= proj * X
        gn = float(np.max(np.sum(grad * grad, axis=1)))
        gn = np.sqrt(gn)
        if not np.isfinite(gn) or gn < 1e-14:
            continue
        grad *= (step / gn)
        X = X - grad
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X

def _lp_polish(X, best, t_deadline, r_init=4e-4):
    n, d = X.shape
    radius = r_init
    iu0, iu1 = np.triu_indices(n, 1)
    N = n * d
    ar_d = np.arange(d)
    rep_n = np.repeat(np.arange(n), d)
    it = 0
    while time.time() < t_deadline and radius > 1e-13 and it < 60:
        it += 1
        G = X @ X.T
        cfull = G[iu0, iu1]
        m = float(cfull.max())
        thresh = max(25.0 * radius, 1.5e-7)
        sel = cfull > (m - thresh)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 1700:
            idx = np.where(sel)[0]
            order = np.argsort(-cfull[idx])[:1700]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I = iu0[sel]; J = iu1[sel]
        csel = cfull[sel]
        p = len(I)
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = (I[:, None] * d + ar_d)
        cJ = (J[:, None] * d + ar_d)
        ct = np.full((p, 1), N)
        cols = np.concatenate([cI, cJ, ct], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, N + 1)).tocsr()
            A_eq = coo_matrix((X.ravel(), (rep_n, np.arange(N))), shape=(n, N + 1)).tocsr()
            obj = np.zeros(N + 1); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * N + [(None, None)],
                          method="highs", options={"presolve": True, "time_limit": 6.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0 or res.x is None:
            radius *= 0.5
            continue
        delta = res.x[:-1].reshape(n, d)
        improved = False
        for a in (1.0, 0.5, 0.25, 0.1):
            Y = X + a * delta
            nr = np.linalg.norm(Y, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            Y /= nr
            v = _max_cos(Y)
            if v < best - 1e-12:
                X = Y; best = v
                radius = min(radius * 1.6, 0.01)
                improved = True
                break
        if not improved:
            radius *= 0.5
    return X, best

def solve():
    t0 = time.time()
    DEADLINE = t0 + 174.0
    SCREEN_END = t0 + 122.0
    rng = np.random.RandomState(0)
    n, d = 73, 13
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(np.float64)
            if v.shape != (n, d):
                continue
            nr = np.linalg.norm(v, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            v = v / nr
            if not np.all(np.isfinite(v)):
                continue
            c = _max_cos(v)
            if c < best_val:
                best_val = c
                best = v.copy()
        except Exception:
            continue
    if best is None:
        best = rng.randn(n, d)
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    Xbest = best.copy()

    trial = 0
    sigmas = [0.08, 0.2, 0.4, 0.7]
    while time.time() < SCREEN_END:
        trial += 1
        r = rng.rand()
        X = None
        if r < 0.22:
            X = rng.randn(n, d)
            X = X / np.linalg.norm(X, axis=1, keepdims=True)
            steps = 900
            a0, a1 = 60.0, 2500.0
            s0, s1 = 0.08, 0.002
        elif r < 0.75:
            sig = sigmas[rng.randint(0, len(sigmas))]
            X = Xbest + sig * rng.randn(n, d)
            X = X / np.linalg.norm(X, axis=1, keepdims=True)
            steps = 650
            a0, a1 = 150.0, 2800.0
            s0, s1 = 0.05, 0.002
        else:
            X = Xbest.copy()
            k = int(4 + rng.randint(0, 7))  # 4..10
            idx = rng.choice(n, size=k, replace=False)
            R = rng.randn(k, d)
            R = R / np.linalg.norm(R, axis=1, keepdims=True)
            X[idx] = R
            # also jitter rest slightly
            X = X + 0.02 * rng.randn(n, d)
            X = X / np.linalg.norm(X, axis=1, keepdims=True)
            steps = 700
            a0, a1 = 120.0, 2800.0
            s0, s1 = 0.06, 0.002
        # adapt steps if little time left
        if time.time() + 0.6 > SCREEN_END:
            steps = min(steps, 300)
        try:
            X = _lse_anneal(X, steps=steps, a0=a0, a1=a1, s0=s0, s1=s1)
        except Exception:
            continue
        try:
            v = _max_cos(X)
        except Exception:
            continue
        if not np.isfinite(v):
            continue
        if v < best_val:
            # short high-alpha squeeze to equalize before accepting
            try:
                X = _lse_anneal(X, steps=150, a0=3000.0, a1=5000.0, s0=0.004, s1=0.0008)
                v2 = _max_cos(X)
            except Exception:
                v2 = v
            if np.isfinite(v2) and v2 < best_val:
                Xbest = X.copy()
                best_val = float(v2)
            elif v < best_val:
                # v was better but squeeze hurt; keep pre-squeeze
                pass
        # time guard
        if time.time() > DEADLINE - 45:
            break

    # intensive refine of incumbent basin
    try:
        # long equalizing anneal
        t_rem = DEADLINE - time.time()
        if t_rem > 50:
            Xbest = _lse_anneal(Xbest, steps=1200, a0=200.0, a1=4000.0, s0=0.02, s1=0.001)
            best_val = _max_cos(Xbest)
            Xbest = _lse_anneal(Xbest, steps=400, a0=4000.0, a1=6000.0, s0=0.003, s1=0.0006)
            best_val = _max_cos(Xbest)
        elif t_rem > 25:
            Xbest = _lse_anneal(Xbest, steps=600, a0=300.0, a1=4000.0, s0=0.015, s1=0.001)
            best_val = _max_cos(Xbest)
    except Exception:
        pass
    try:
        Xbest, best_val = _lp_polish(Xbest, best_val, DEADLINE - 3.0, r_init=3e-4)
    except Exception:
        pass
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
