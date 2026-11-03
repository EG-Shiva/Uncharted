import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _row_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)

def _soft_adam(x, steps, beta0, beta1, lr):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        f = (t - 1) / max(steps - 1, 1)
        beta = beta0 + (beta1 - beta0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mv = G.max()
        W = np.exp(beta * (G - mv))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        nd = (grad * x).sum(axis=1, keepdims=True)
        grad = grad - nd * x
        m *= b1
        m += (1.0 - b1) * grad
        v *= b2
        v += (1.0 - b2) * (grad * grad)
        mh = m / (1.0 - b1 ** t)
        vh = v / (1.0 - b2 ** t)
        upd = mh / (np.sqrt(vh) + eps)
        upd -= (upd * x).sum(axis=1, keepdims=True) * x
        x = x - lr * upd
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _void_point(X, rng):
    # best void for one point w.r.t. set X (m,d)
    cand = rng.normal(size=(8000, X.shape[1]))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = cand @ X.T
    best = scores.max(axis=1).argmin()
    y = cand[best].copy()
    lr = 0.05
    for _ in range(200):
        dots = X @ y
        i = int(np.argmax(dots))
        g = X[i] - (X[i] @ y) * y
        y = y - lr * g
        n = np.linalg.norm(y)
        if n == 0 or not np.isfinite(n):
            y = cand[best].copy()
            lr *= 0.5
            continue
        y /= n
        lr *= 0.995
    return y

def _lp_polish(x, deadline):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        mval = c.max()
        thr = mval - max(20.0 * radius, 1e-9)
        sel = c > thr
        if np.count_nonzero(sel) > 2500:
            idx = np.where(sel)[0]
            o = np.argsort(-c[idx])[:2500]
            mk = np.zeros_like(sel, dtype=bool)
            mk[idx[o]] = True
            sel = mk
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    deadline = t0 + 170.0
    rng = np.random.default_rng(0)
    n, d = 138, 8
    best = None; best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            vv = np.load(fn).astype(float)
            if vv.shape == (n, d):
                vv /= np.linalg.norm(vv, axis=1, keepdims=True)
                c = _max_cos(vv)
                if c < best_val:
                    best_val = c; best = vv.copy()
        except Exception:
            pass
    if best is None:
        best = rng.normal(size=(n, d)); best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    cur = best.copy(); cur_val = best_val
    it = 0
    while time.time() < deadline - 8.0:
        it += 1
        now = time.time()
        r = rng.random()
        y = None
        if r < 0.30:
            sig = [0.02, 0.08, 0.25, 0.45][it % 4]
            base = cur if (it % 2 == 0) else best
            y = base + sig * rng.normal(size=(n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            if it % 5 == 0:
                k = int(rng.integers(4, 10))
                idx = rng.choice(n, size=k, replace=False)
                rp = rng.normal(size=(k, d)); rp /= np.linalg.norm(rp, axis=1, keepdims=True)
                y[idx] = rp
            y = _soft_adam(y, 500, 15.0, 160.0, 0.008)
        elif r < 0.50:
            # kill worst, rebirth in voids
            rm = _row_max(cur)
            k = int(rng.integers(3, 9))
            kill = np.argsort(-rm)[:k]
            y = cur.copy()
            tmp = np.delete(y, kill, axis=0)
            for j, idx in enumerate(kill):
                nv = _void_point(tmp, rng)
                tmp = np.vstack([tmp, nv])
                y[idx] = nv
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _soft_adam(y, 500, 12.0, 140.0, 0.008)
        elif r < 0.65:
            # rattler void optimisation
            rm = _row_max(cur)
            gm = rm.max()
            rats = np.where(rm < gm - 0.002)[0]
            if len(rats) == 0:
                rats = np.argsort(rm)[-8:]
            else:
                rats = rats[rng.permutation(len(rats))[:min(len(rats), 12)]]
            y = cur.copy()
            core = np.delete(y, rats, axis=0)
            for idx in rats:
                nv = _void_point(core, rng)
                core = np.vstack([core, nv])
                y[idx] = nv
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _soft_adam(y, 600, 10.0, 180.0, 0.006)
        elif r < 0.80:
            sig = 0.35
            y = best + sig * rng.normal(size=(n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _soft_adam(y, 700, 8.0, 120.0, 0.009)
        else:
            y = rng.normal(size=(n, d)); y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _soft_adam(y, 900, 8.0, 200.0, 0.008)
        v = _max_cos(y)
        if v < best_val - 1e-12:
            # squeeze immediately
            try:
                y2 = _lp_polish(y, min(deadline - 8.0, time.time() + 2.0))
                v2 = _max_cos(y2)
                if v2 <= v:
                    y, v = y2, v2
            except Exception:
                pass
            if v < best_val:
                best_val = v; best = y.copy()
                cur = y.copy(); cur_val = v
                continue
        # basin-hopping move for cur
        if v < cur_val or rng.random() < np.exp(-(v - cur_val) / 0.003):
            cur = y.copy(); cur_val = v
        # occasional LP on cur if close
        if it % 12 == 0 and cur_val < best_val + 0.004:
            try:
                yc = _lp_polish(cur, min(deadline - 8.0, time.time() + 1.0))
                vc = _max_cos(yc)
                if vc < cur_val:
                    cur, cur_val = yc, vc
                if vc < best_val:
                    best_val, best = vc, yc.copy()
            except Exception:
                pass
    try:
        best = _lp_polish(best, deadline)
    except Exception:
        pass
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
