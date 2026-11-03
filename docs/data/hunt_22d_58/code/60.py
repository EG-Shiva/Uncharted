import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _quench(X, steps, b0, b1, lr0, lr1, mu=0.65):
    n, d = X.shape
    V = np.zeros_like(X)
    lb0, lb1 = np.log(b0), np.log(b1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(lr0 + (lr1 - lr0) * f)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # active window to sparsify far pairs
        thr = m - 0.03
        M = G > thr
        D = G - m
        # exp only where active
        W = np.zeros_like(G)
        # clip beta*D to avoid overflow (D<=0 so no overflow)
        W[M] = np.exp(beta * D[M])
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1)
        Gr = W @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        nr = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-18
        Gn = Gt / nr
        # freeze isolated points (no active contacts)
        iso = (rs < 1e-12)[:, None]
        Gn = np.where(iso, 0.0, Gn)
        V = mu * V + (1.0 - mu) * Gn
        # retangent momentum
        vd = np.sum(V * X, axis=1, keepdims=True)
        Vt = V - vd * X
        # clip per-point step
        sn = np.linalg.norm(Vt, axis=1, keepdims=True) + 1e-18
        # scale to unit-mean then lr
        step = (lr / (float(np.mean(sn)) + 1e-18)) * Vt
        # hard clip large moves
        sn2 = np.linalg.norm(step, axis=1, keepdims=True) + 1e-18
        over = sn2 > 0.03
        step = np.where(over, step * (0.03 / sn2), step)
        X = X - step
        X = _norm(X)
    return X

def _greedy_push(X, steps, e0=0.004, e1=0.0005):
    n = X.shape[0]
    for s in range(steps):
        f = s / max(1, steps - 1)
        eps = float(e0 * (e1 / e0) ** f) if e0 != e1 else float(e0)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        k = int(np.argmax(G))
        i = k // n
        j = k % n
        g = float(G[i, j])
        xi = X[i].copy()
        xj = X[j].copy()
        ti = xj - g * xi
        tj = xi - g * xj
        ni = float(np.linalg.norm(ti)) + 1e-12
        nj = float(np.linalg.norm(tj)) + 1e-12
        X[i] = xi - eps * ti / ni
        X[j] = xj - eps * tj / nj
        X[i] /= (np.linalg.norm(X[i]) + 1e-12)
        X[j] /= (np.linalg.norm(X[j]) + 1e-12)
    return X

def _hole_replace(X, victims, rng, n_cand=1500):
    n, d = X.shape
    Y = X.copy()
    for v in victims:
        C = rng.standard_normal((n_cand, d))
        C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        dots = C @ Y.T
        # ignore victim column
        dots[:, v] = -2.0
        worst = dots.max(axis=1)
        bi = int(np.argmin(worst))
        Y[v] = C[bi]
    return Y

def _degrees(X, gap=0.004):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    deg = (G > m - gap).sum(axis=1)
    return deg, m, G

def _lp_polish(X, seconds, deadline):
    X = _norm(X.copy())
    n, d = X.shape
    best = _max_cos(X)
    radius = 8e-4
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    itc = 0
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-13:
        itc += 1
        if itc > 40:
            break
        G = X @ X.T
        c = G[iu0]
        m = float(c.max())
        window = max(20.0 * radius, 4e-5)
        order = np.argsort(-c)
        sel = []
        for idx in order:
            if c[idx] > m - window and len(sel) < 600:
                sel.append(int(idx))
            elif c[idx] <= m - window:
                break
            if len(sel) >= 600:
                break
        if len(sel) == 0:
            break
        sel = np.array(sel, dtype=int)
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(sel)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best - 1e-15:
            X, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    DEAD = t0 + 165.0
    rng = np.random.default_rng(7)
    best = None
    best_val = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            v = np.load(f).astype(float)
            if v.shape != (58, 22):
                continue
            v = _norm(v)
            c = _max_cos(v)
            if c < best_val:
                best_val = c
                best = v.copy()
        except Exception:
            continue
    if best is None:
        best = _norm(rng.standard_normal((58, 22)))
        best_val = _max_cos(best)
    base = best.copy()
    base_val = best_val

    # Phase A: a few from-scratch diverse basins (cheap)
    for a in range(4):
        if time.time() > DEAD - 60:
            break
        X0 = _norm(rng.standard_normal((58, 22)))
        X0 = _quench(X0, steps=350, b0=50.0, b1=400.0, lr0=0.030, lr1=0.006, mu=0.65)
        X0 = _greedy_push(X0, steps=40)
        X0 = _quench(X0, steps=200, b0=150.0, b1=600.0, lr0=0.012, lr1=0.004, mu=0.7)
        v = _max_cos(X0)
        if v < best_val - 1e-12:
            best, best_val = X0.copy(), v
            base, base_val = X0.copy(), v
        elif v < base_val and v < 0.09:
            # adopt diverse good basin occasionally
            pass

    # Phase B: hub-rewiring basin hopping from incumbent
    ntrials = 300
    for it in range(ntrials):
        if time.time() >= DEAD - 25:
            break
        # temperature schedule
        frac = it / max(1, ntrials - 1)
        Temp = 0.003 * (1 - frac) + 0.0002 * frac
        # pick k victims
        r = rng.random()
        if r < 0.35:
            k = 1
        elif r < 0.65:
            k = 2
        elif r < 0.85:
            k = 3
        else:
            k = 4
        deg, m, G = _degrees(base, gap=0.004)
        order = np.argsort(-deg)
        # top pool
        pool = order[:10].copy()
        # also include worst-pair endpoints to target max
        iu_n = base.shape[0]
        flat = int(np.argmax(G))
        wi = flat // iu_n
        wj = flat % iu_n
        cand_pool = list(pool) + [wi, wj]
        victims = []
        # random sample k distinct from cand_pool + random
        while len(victims) < k:
            if rng.random() < 0.8:
                c = int(cand_pool[rng.integers(0, len(cand_pool))])
            else:
                c = int(rng.integers(0, 58))
            if c not in victims:
                victims.append(c)
        Xc = _hole_replace(base, victims, rng, n_cand=1500)
        # global tangent shake (small, scaled by k)
        sig = 0.004 + 0.004 * k + (0.008 if rng.random() < 0.2 else 0.0)
        N = rng.standard_normal(Xc.shape)
        N -= np.sum(N * Xc, axis=1, keepdims=True) * Xc
        nn = np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
        N /= nn
        Xc = _norm(Xc + sig * N)
        # quench: two-stage
        Xc = _quench(Xc, steps=320, b0=60.0, b1=450.0, lr0=0.028, lr1=0.006, mu=0.65)
        if time.time() >= DEAD - 25:
            # evaluate anyway
            v = _max_cos(Xc)
            if v < best_val:
                best, best_val = Xc.copy(), v
            break
        Xc = _greedy_push(Xc, steps=30)
        Xc = _quench(Xc, steps=180, b0=200.0, b1=700.0, lr0=0.010, lr1=0.003, mu=0.7)
        v = _max_cos(Xc)
        if v < best_val - 1e-12:
            best, best_val = Xc.copy(), v
            base, base_val = Xc.copy(), v
            continue
        # SA acceptance for base to change topology
        if v < base_val or rng.random() < np.exp((base_val - v) / max(Temp, 1e-9)):
            base, base_val = Xc.copy(), v
        # periodically pull base back to best to intensify
        if (it + 1) % 25 == 0:
            base, base_val = best.copy(), best_val
            # add medium kick to best to restart exploration
            N2 = rng.standard_normal(base.shape)
            N2 -= np.sum(N2 * base, axis=1, keepdims=True) * base
            N2 /= (np.linalg.norm(N2, axis=1, keepdims=True) + 1e-12)
            base = _norm(base + 0.012 * N2)
            base_val = _max_cos(base)

    # final polish of global best
    try:
        if time.time() < DEAD - 5:
            secs = max(2.0, min(16.0, DEAD - 3 - time.time()))
            best = _lp_polish(best, secs, DEAD - 1)
            best_val = _max_cos(best)
    except Exception:
        pass
    best = _norm(best)
    return best
