import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _rowmax(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _adam_sm(x0, steps, K0, K1, lr):
    x = x0.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    la0, la1 = np.log(K0), np.log(K1)
    best_val = _max_cos(x)
    best_x = x.copy()
    for t in range(1, steps + 1):
        K = float(np.exp(la0 + (la1 - la0) * t / steps))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(np.clip(K * (G - gmax), -60.0, 0.0))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (W @ x) / S
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1 - b1) * gt
        v *= b2
        v += (1 - b2) * (gt * gt)
        mhat = m / (1 - b1p)
        vhat = v / (1 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        sc = np.minimum(1.0, 0.06 / (sn + 1e-12))
        step = step * sc
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if t % 200 == 0 or t == steps:
            val = _max_cos(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
    return best_x, best_val

def _lp_polish(x, deadline, r0=2.5e-3, cap=3500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = r0
    iuI, iuJ = np.triu_indices(n, 1)
    eq_rows = np.repeat(np.arange(n), d)
    eq_cols = np.arange(n * d)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iuI, iuJ]
        mm = float(c.max())
        margin = max(25.0 * radius, 8e-5)
        sel = c > (mm - margin)
        I = iuI[sel]; J = iuJ[sel]; csel = c[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > cap:
            part = np.argpartition(-csel, cap)[:cap]
            I = I[part]; J = J[part]; csel = csel[part]
            p = cap
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (eq_rows, eq_cols)), shape=(n, n * d + 1)).tocsr()
        except Exception:
            break
        obj = np.zeros(n * d + 1)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        val = _max_cos(y)
        if val < best:
            x, best = y, val
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def _pool_scores(pool, X, chunk=20000):
    M = pool.shape[0]
    scores = np.empty(M, dtype=float)
    for s in range(0, M, chunk):
        C = pool[s:s+chunk]
        m = (C @ X.T).max(axis=1)
        scores[s:s+chunk] = m
    return scores

def _add_four(X, rng, M=200000):
    n, d = X.shape
    pool = rng.normal(size=(M, d))
    pool /= np.linalg.norm(pool, axis=1, keepdims=True)
    scores = _pool_scores(pool, X)
    chosen = []
    for _ in range(4):
        idx = int(np.argmin(scores))
        v = pool[idx].copy()
        chosen.append(v)
        dots = pool @ v
        # update max vs augmented set
        np.maximum(scores, dots, out=scores)
        # avoid re-picking same (dots=1 at idx -> won't be min again)
    return np.vstack([X] + chosen)

def _best_removal_idx(X):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    iuI, iuJ = np.triu_indices(n, 1)
    c = G[iuI, iuJ]
    K = min(len(c), 3000)
    part = np.argpartition(-c, K-1)[:K]
    order = part[np.argsort(-c[part])]
    sI = iuI[order]; sJ = iuJ[order]; sC = c[order]
    best_i = 0
    best_v = 2.0
    # for each i find first pair not involving i
    for i in range(n):
        for k in range(K):
            if sI[k] != i and sJ[k] != i:
                v = float(sC[k])
                break
        else:
            v = -2.0
        if v < best_v:
            best_v = v
            best_i = i
    return best_i, best_v

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    T_END = t0 + 172.0
    rng = _np.random.default_rng(20261103)
    n, d = 256, 6
    x_best = None
    best = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = _np.load(fn)
            a = _np.asarray(a, dtype=float)
            if a.shape == (n, d) and _np.all(_np.isfinite(a)):
                a = a / _np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best:
                    best = v
                    x_best = a.copy()
        except Exception:
            continue
    if x_best is None:
        x_best = rng.normal(size=(n, d))
        x_best /= _np.linalg.norm(x_best, axis=1, keepdims=True)
        best = _max_cos(x_best)
    # helper to update
    def consider(X):
        nonlocal x_best, best
        try:
            v = _max_cos(X)
        except Exception:
            return False
        if v < best:
            best = v
            x_best = X.copy()
            return True
        return False

    # build homotopy starts
    try:
        rm = _rowmax(x_best)
    except Exception:
        rm = None

    starts = {}  # N -> X
    # N=240,248 by deleting smallest row-max
    if rm is not None:
        order = _np.argsort(rm)  # ascending: smallest first
        try:
            del16 = _np.sort(order[:16])
            mask = _np.ones(n, dtype=bool); mask[del16] = False
            starts[240] = x_best[mask].copy()
        except Exception:
            pass
        try:
            del8 = _np.sort(order[:8])
            mask = _np.ones(n, dtype=bool); mask[del8] = False
            starts[248] = x_best[mask].copy()
        except Exception:
            pass
    # N=264,272 by greedy void insertion from >=200k candidates
    # do sequentially to respect time
    try:
        if _time.time() < T_END - 100:
            cur = x_best.copy()
            cur = _add_four(cur, rng, M=200000)
            cur = _add_four(cur, rng, M=200000)
            starts[264] = cur.copy()
    except Exception:
        pass
    try:
        if _time.time() < T_END - 90:
            if 264 in starts:
                cur = starts[264].copy()
            else:
                cur = x_best.copy()
                cur = _add_four(cur, rng, M=200000)
                cur = _add_four(cur, rng, M=200000)
            cur = _add_four(cur, rng, M=200000)
            cur = _add_four(cur, rng, M=200000)
            starts[272] = cur.copy()
    except Exception:
        pass

    # optimize each size with annealed softmax Adam
    opt = {}
    for N in (240, 248, 264, 272):
        if N not in starts:
            continue
        if _time.time() > T_END - 80:
            break
        X0 = starts[N]
        try:
            Xb, vb = _adam_sm(X0, 700, 20.0, 3000.0, 0.015)
        except Exception:
            continue
        # one Direction-1 LP short polish
        try:
            if _time.time() < T_END - 70:
                Xb = _lp_polish(Xb, min(_time.time() + 4.0, T_END - 60), r0=2.0e-3, cap=3500)
        except Exception:
            pass
        opt[N] = Xb

    # convert back to 256
    # N<256: insert four at a time with rejamming
    for N in (240, 248):
        if N not in opt:
            continue
        cur = opt[N].copy()
        # number of 4-blocks needed
        while cur.shape[0] < 256 and _time.time() < T_END - 30:
            try:
                cur = _add_four(cur, rng, M=200000)
            except Exception:
                break
            try:
                cur, _ = _adam_sm(cur, 500, 80.0, 2500.0, 0.012)
            except Exception:
                pass
            try:
                if cur.shape[0] == 256 and _time.time() < T_END - 25:
                    cur = _lp_polish(cur, min(_time.time() + 5.0, T_END - 15), r0=2.0e-3, cap=3500)
            except Exception:
                pass
            if cur.shape[0] == 256:
                consider(cur)
            elif cur.shape[0] < 256:
                # intermediate <256: project to 256 by padding voids? skip but rejamm already
                pass
        # if we overshoot due to logic, trim
        if cur.shape[0] == 256:
            consider(cur)

    # N>256: delete by leave-one-out with rejamming
    for N in (264, 272):
        if N not in opt:
            continue
        if _time.time() > T_END - 30:
            break
        cur = opt[N].copy()
        while cur.shape[0] > 256 and _time.time() < T_END - 25:
            try:
                bi, _ = _best_removal_idx(cur)
            except Exception:
                break
            mask = _np.ones(cur.shape[0], dtype=bool); mask[bi] = False
            cur = cur[mask].copy()
            # rejamm after each deletion (light for intermediate, heavier at 256)
            try:
                steps = 450 if cur.shape[0] == 256 else 250
                cur, _ = _adam_sm(cur, steps, 80.0, 2500.0, 0.012)
            except Exception:
                pass
            if cur.shape[0] == 256:
                try:
                    cur = _lp_polish(cur, min(_time.time() + 6.0, T_END - 12), r0=2.0e-3, cap=3500)
                except Exception:
                    pass
                consider(cur)
        if cur.shape[0] == 256:
            consider(cur)

    # final polish of global best with remaining time
    try:
        if _time.time() < T_END - 12:
            y, _ = _adam_sm(x_best, 600, 100.0, 3000.0, 0.010)
            consider(y)
            # keep better of the two for LP
    except Exception:
        pass
    try:
        if _time.time() < T_END - 6:
            x_best = _lp_polish(x_best, T_END - 1.5, r0=2.5e-3, cap=4000)
            consider(x_best)
    except Exception:
        pass

    x_best = x_best / _np.linalg.norm(x_best, axis=1, keepdims=True)
    # tiny shrink guard not needed; return
    return x_best
