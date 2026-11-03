import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _per_point_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)

def _contact_counts(x, tol=1e-9):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = g.max()
    return (g > m - tol).sum(axis=1), m

def _adam_softmax(x, steps, beta0, beta1, lr):
    n, d = x.shape
    m1 = np.zeros_like(x)
    m2 = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        f = (t - 1) / max(steps - 1, 1)
        beta = beta0 + (beta1 - beta0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mv = G.max()
        W = np.exp(beta * (G - mv))
        np.fill_diagonal(W, 0.0)
        S = W.sum() * 0.5 + 1e-30
        grad = (W @ x) / S
        # tangent projection
        nd = (grad * x).sum(axis=1, keepdims=True)
        grad = grad - nd * x
        m1 *= b1
        m1 += (1.0 - b1) * grad
        m2 *= b2
        m2 += (1.0 - b2) * (grad * grad)
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        mh = m1 / bc1
        vh = m2 / bc2
        upd = mh / (np.sqrt(vh) + eps)
        nd2 = (upd * x).sum(axis=1, keepdims=True)
        upd = upd - nd2 * x
        x = x - lr * upd
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-30)
    return x

def _best_void(X_exist, rng, n_samples=8000):
    d = X_exist.shape[1]
    cand = rng.standard_normal(size=(n_samples, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True) + 1e-30
    # max dot with existing
    md = (cand @ X_exist.T).max(axis=1)
    return cand[np.argmin(md)]

def _void_rebirth(x, k, rng):
    y = x.copy()
    n, d = y.shape
    pm = _per_point_max(y)
    worst = np.argsort(-pm)[:k]
    # iteratively replace
    for idx in worst:
        # existing without idx
        mask = np.ones(n, dtype=bool)
        mask[idx] = False
        X_exist = y[mask]
        v = _best_void(X_exist, rng)
        y[idx] = v
    return y

def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        c = Gfull[iu]
        m = c.max()
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 2500:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:2500]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    rng = np.random.default_rng(0)
    t0 = time.time()
    deadline = t0 + 174.0
    n, d = 138, 8
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal(size=(n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val

    it = 0
    # explore, reserve ~22s for final LP
    while time.time() < deadline - 22.0:
        it += 1
        mode = it % 6
        if mode == 5 and it % 30 == 0:
            # fresh random basin (rare)
            y = rng.standard_normal(size=(n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _adam_softmax(y, steps=1200, beta0=10.0, beta1=300.0, lr=0.02)
        elif mode == 0:
            # small shake all
            y = cur + 0.02 * rng.standard_normal(size=(n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _adam_softmax(y, steps=500, beta0=30.0, beta1=220.0, lr=0.006)
        elif mode == 1:
            # medium shake all
            y = cur + 0.12 * rng.standard_normal(size=(n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _adam_softmax(y, steps=700, beta0=20.0, beta1=260.0, lr=0.009)
        elif mode == 2:
            # void rebirth of worst k
            k = int(rng.integers(3, 9))
            y = _void_rebirth(cur, k, rng)
            # tiny jitter to others
            y = y + 0.01 * rng.standard_normal(size=(n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _adam_softmax(y, steps=800, beta0=20.0, beta1=280.0, lr=0.008)
        elif mode == 3:
            # overload-targeted dispersal
            try:
                cnts, _ = _contact_counts(cur, tol=0.01)
            except Exception:
                cnts = _per_point_max(cur)
                cnts = np.argsort(np.argsort(-cnts))
            order = np.argsort(-cnts)
            pool = order[:12]
            k = 4
            chosen = pool[rng.choice(len(pool), size=k, replace=False)]
            y = cur.copy()
            y[chosen] += 0.35 * rng.standard_normal(size=(k, d))
            y += 0.008 * rng.standard_normal(size=(n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _adam_softmax(y, steps=800, beta0=15.0, beta1=260.0, lr=0.009)
        else:
            # rattler / random-subset shake: shake 10 random points hard
            y = cur.copy()
            idx = rng.choice(n, size=10, replace=False)
            y[idx] += 0.25 * rng.standard_normal(size=(10, d))
            y += 0.005 * rng.standard_normal(size=(n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _adam_softmax(y, steps=700, beta0=25.0, beta1=300.0, lr=0.007)

        v = _max_cos(y)
        if v < best_val - 1e-12:
            best_val = v
            best = y.copy()
            cur = y.copy()
            cur_val = v
        elif v < cur_val:
            cur = y.copy()
            cur_val = v
        else:
            # random-walk acceptance to change basin (Metropolis-like)
            if v < cur_val + 0.008 and rng.random() < 0.15:
                cur = y.copy()
                cur_val = v
            # occasional restart cur from best
            if it % 25 == 0:
                cur = best.copy()
                cur_val = best_val
        # early exit if we beat record by margin -> go to polish
        if best_val < 0.4665:
            break

    # final tight polish of incumbent (and cur if different)
    try:
        best = _lp_polish(best, deadline - 1.0)
        best_val = _max_cos(best)
    except Exception:
        pass
    # if time remains, try polishing cur as alternative
    try:
        if time.time() < deadline - 3.0:
            alt = _lp_polish(cur, deadline - 1.0)
            av = _max_cos(alt)
            if av < best_val:
                best = alt
                best_val = av
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
