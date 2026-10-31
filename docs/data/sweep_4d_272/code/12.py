import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _normalize(x):
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(nrm, 1e-15)
    bad = ~np.all(np.isfinite(x), axis=1)
    if np.any(bad):
        x[bad] = np.array([1.0, 0.0, 0.0, 0.0])
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _lp_polish(x, seconds):
    x = _normalize(x)
    n, d = x.shape
    best = _max_cos(x)
    bx = x.copy()
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu0]
        m = float(np.max(c))
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        if not np.any(sel):
            break
        I = iu0[0][sel]
        J = iu0[1][sel]
        csel = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _normalize(y)
        v = _max_cos(y)
        if v < best:
            best = v
            bx = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            # stay at best-seen x for stability, shrink
            radius *= 0.5
            # if we shrank a lot, re-anchor to bx
            x = bx.copy() if radius < 1e-6 else x
        # early stop if time nearly out
        if time.time() - start >= seconds:
            break
    return _normalize(bx)

def _smooth_relax(y, beta, steps, lr):
    y = _normalize(y)
    for _ in range(steps):
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        W[W < 1e-12] = 0.0
        grad = W @ y
        proj = np.sum(grad * y, axis=1, keepdims=True)
        gt = grad - proj * y
        nrm = float(np.mean(np.linalg.norm(gt, axis=1)))
        if not np.isfinite(nrm) or nrm < 1e-12:
            break
        y = y - (lr / max(nrm, 1e-12)) * gt
        y = _normalize(y)
        if not np.all(np.isfinite(y)):
            break
    return _normalize(y)

def _best_void(others, rng, n_cand=4000):
    cand = rng.standard_normal((n_cand, 4))
    cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
    s = cand @ others.T
    scores = np.max(s, axis=1)
    bi = int(np.argmin(scores))
    b = cand[bi].copy()
    bs = float(scores[bi])
    step = 0.05
    for _ in range(250):
        prop = b + step * rng.standard_normal((32, 4))
        prop = prop / np.linalg.norm(prop, axis=1, keepdims=True)
        sc = np.max(prop @ others.T, axis=1)
        j = int(np.argmin(sc))
        if sc[j] < bs:
            b = prop[j].copy()
            bs = float(sc[j])
            step *= 1.05
        else:
            step *= 0.97
        if step < 1e-5:
            break
    return b / np.linalg.norm(b)

def solve():
    t0 = time.time()
    LIMIT = 172.0
    rng = np.random.default_rng(12345)
    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            x0 = rng.standard_normal((272, 4))
    x0 = _normalize(np.asarray(x0, dtype=float))
    if x0.shape != (272, 4):
        tmp = rng.standard_normal((272, 4))
        x0 = _normalize(tmp)
    # initial tightening
    best = _lp_polish(x0, 6.0)
    best_val = _max_cos(best)
    # also try loading record as alternative start
    try:
        xr = _normalize(np.asarray(np.load("record.npy"), dtype=float))
        if xr.shape == (272, 4):
            xr = _lp_polish(xr, 2.0)
            vr = _max_cos(xr)
            if vr < best_val:
                best, best_val = xr, vr
    except Exception:
        pass

    def personal_worst(x):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        return np.max(G, axis=1)

    it = 0
    while time.time() - t0 < LIMIT - 22.0:
        it += 1
        if time.time() - t0 > LIMIT - 22.0:
            break
        y = best.copy()
        mode = it % 5
        if mode == 0:
            # single worst-point void reinsertion
            pw = personal_worst(y)
            order = np.argsort(-pw)
            cand_idx = order[:8]
            wi = int(cand_idx[rng.integers(0, len(cand_idx))])
            others = np.delete(y, wi, axis=0)
            b = _best_void(others, rng, n_cand=5000)
            y[wi] = b
            y = _smooth_relax(y, 60.0, 120, 0.008)
            y = _smooth_relax(y, 130.0, 80, 0.004)
        elif mode == 1:
            # cluster shake of overloaded points
            pw = personal_worst(y)
            order = np.argsort(-pw)
            k = int(rng.integers(4, 11))
            idx = order[: max(k * 2, 8)]
            sel = rng.choice(idx, size=min(k, len(idx)), replace=False)
            sigma = float(rng.uniform(0.03, 0.10))
            y[sel] += sigma * rng.standard_normal((len(sel), 4))
            y = _normalize(y)
            y = _smooth_relax(y, 50.0, 120, 0.008)
        elif mode == 2:
            # global small shake
            sigma = float(rng.uniform(0.006, 0.025))
            y = y + sigma * rng.standard_normal(y.shape)
            y = _normalize(y)
            y = _smooth_relax(y, 70.0, 100, 0.006)
        elif mode == 3:
            # double reinsertion: worst + rattler
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            contacts = np.sum(G > m - 1e-9, axis=1)
            pw = np.max(G, axis=1)
            w1 = int(np.argmax(pw))
            low = np.where(contacts <= 3)[0]
            if len(low) > 0:
                w2 = int(low[rng.integers(0, len(low))])
            else:
                w2 = int(rng.integers(0, 272))
            if w2 == w1:
                w2 = (w1 + 137) % 272
            keep = [i for i in range(272) if i != w1 and i != w2]
            others = y[keep]
            b1 = _best_void(others, rng, n_cand=3000)
            tmp = np.vstack([others, b1.reshape(1, 4)])
            b2 = _best_void(tmp, rng, n_cand=3000)
            y[w1] = b1
            y[w2] = b2
            y = _smooth_relax(y, 60.0, 100, 0.007)
        else:
            # moderate global shake + sharper relax
            sigma = float(rng.uniform(0.015, 0.045))
            y = y + sigma * rng.standard_normal(y.shape)
            y = _normalize(y)
            y = _smooth_relax(y, 90.0, 120, 0.005)

        # LP tighten candidate (short budget) - guarded acceptance only
        remain = LIMIT - 22.0 - (time.time() - t0)
        if remain <= 0:
            break
        budget = min(2.2, max(0.8, remain / 8.0))
        z = _lp_polish(y, budget)
        v = _max_cos(z)
        if v < best_val - 1e-15:
            best = z.copy()
            best_val = v

    # final tightening of guarded best only
    rem = LIMIT - (time.time() - t0) - 1.5
    if rem > 3.0:
        rem = min(rem, 20.0)
        zb = _lp_polish(best, rem)
        vb = _max_cos(zb)
        if vb < best_val:
            best = zb
            best_val = vb
    return _normalize(best)
