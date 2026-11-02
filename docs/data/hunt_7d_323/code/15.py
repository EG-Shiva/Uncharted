import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _worst_vec(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)


def _contact_counts(x, m, tol=1e-9):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return np.sum(g > m - tol, axis=1)


def _fast_anneal(x, steps, eta0, eta1, m0, m1, mu=0.85):
    # hinge repulsion with momentum, tangent projected
    n, d = x.shape
    v = np.zeros_like(x)
    for t in range(steps):
        a = t / max(1, steps - 1)
        eta = eta0 * (1 - a) + eta1 * a
        margin = m0 * (1 - a) + m1 * a
        G = x @ x.T
        # diag exclude
        # max without diag
        # set diag low for max
        # faster: use triu max
        m = float(G.max() - 1.0)  # placeholder
        # need diag -2: fill quickly
        # G diagonal is 1.0, max is >=..., so max is off-diag anyway unless n=1
        # but to be safe:
        # G max off diag = max of G with diag ignored; since diag=1 and m~0.66, max is off-diag
        # so m = G.max() works (diag 1 would dominate!). must mask.
        # fix: compute m correctly
        # diag is 1, so G.max() would be 1. Need to handle.
        # redo with fill
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        thresh = m - margin
        # weights: linear hinge
        # W = max(G-thresh,0)
        # to save memory reuse G
        np.subtract(G, thresh, out=G)
        np.maximum(G, 0.0, out=G)
        # G now is W symmetric, diag 0 (since -2-thresh<0)
        grad = G @ x
        # tangent projection
        s = np.sum(grad * x, axis=1, keepdims=True)
        grad -= s * x
        v = mu * v + grad
        x = x - eta * v
        # renormalize
        nr = np.sqrt(np.sum(x * x, axis=1, keepdims=True))
        nr[nr == 0] = 1.0
        x /= nr
    return x


def _greedy_reinsert(y, to_replace, rng, n_cand=5000):
    n, d = y.shape
    fixed_mask = np.ones(n, dtype=bool)
    fixed_mask[np.array(to_replace, dtype=int)] = False
    X_fixed = y[fixed_mask]
    new_list = []
    # order worst first already given
    for k in to_replace:
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        if new_list:
            Xa = np.vstack([X_fixed] + new_list)
        else:
            Xa = X_fixed
        dots = cand @ Xa.T
        worst = dots.max(axis=1)
        bi = int(np.argmin(worst))
        bp = cand[bi]
        y[k] = bp
        new_list.append(bp[None, :])
    # tiny hill climb on replaced
    for k in to_replace:
        cur = y[k].copy()
        others = np.delete(y, k, axis=0)
        cw = float((others @ cur).max())
        tr = rng.standard_normal(size=(300, d)) * 0.05 + cur[None, :]
        tr /= np.linalg.norm(tr, axis=1, keepdims=True)
        dw = (tr @ others.T).max(axis=1)
        bi = int(np.argmin(dw))
        if dw[bi] < cw:
            y[k] = tr[bi]
    return y


def _lp_polish(x, seconds, init_radius=3e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(init_radius)
    start = time.time()
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start >= seconds:
            break
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 2500:
            idx = np.where(sel)[0]
            cidx = c[idx]
            part = np.argpartition(cidx, -2500)[-2500:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
            cnt = 2500
        I = iuI[sel]
        J = iuJ[sel]
        csel = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        Cb = cols.reshape(p, 2 * d + 1)
        Vb = vals.reshape(p, 2 * d + 1)
        ar = np.arange(d)
        Cb[:, :d] = I[:, None] * d + ar
        Cb[:, d:2 * d] = J[:, None] * d + ar
        Cb[:, 2 * d] = n * d
        Vb[:, :d] = x[J]
        Vb[:, d:2 * d] = x[I]
        Vb[:, 2 * d] = -1.0
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
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
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        vv = _max_cos(y)
        if vv < best:
            x = y
            best = vv
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    rng = np.random.default_rng(20261101)
    x0 = None
    b0 = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < b0:
                    b0 = v
                    x0 = a.copy()
        except Exception:
            pass
    if x0 is None:
        a = rng.standard_normal(size=(323, 7))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        x0 = a
        b0 = _max_cos(a)
    x_best = x0.copy()
    best = float(b0)
    n, d = x_best.shape

    # brief fast settle on best (helps if loaded record not fully squeezed by our metric)
    try:
        if time.time() - t0 < 150:
            y = _fast_anneal(x_best.copy(), steps=2500, eta0=0.012, eta1=0.002, m0=0.015, m1=0.004, mu=0.85)
            v = _max_cos(y)
            if v < best:
                best = v
                x_best = y
    except Exception:
        pass

    it = 0
    noimp = 0
    # hopping budget until 140s
    while True:
        if time.time() - t0 > 140:
            break
        remain = 140 - (time.time() - t0)
        if remain < 0.6:
            break
        x = x_best.copy()
        s = it % 7
        torep = None
        if s == 0:
            sig = 0.012 + min(noimp, 20) * 0.0008
            x = x + rng.standard_normal(size=x.shape) * sig
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, e0, e1, g0, g1 = 3500, 0.020, 0.002, 0.025, 0.004
        elif s == 1:
            w = _worst_vec(x)
            k = 25
            idx = np.argpartition(w, -k)[-k:]
            x[idx] += rng.standard_normal(size=(k, d)) * 0.07
            x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
            steps, e0, e1, g0, g1 = 3500, 0.022, 0.002, 0.025, 0.004
        elif s == 2:
            m = best
            cc = _contact_counts(x, m, tol=1e-9)
            w = _worst_vec(x)
            ratt = np.where(cc == 0)[0].tolist()
            order = np.argsort(-w).tolist()
            torep = []
            for r in ratt:
                torep.append(int(r))
            for o in order:
                if len(torep) >= 4:
                    break
                if int(o) not in torep:
                    torep.append(int(o))
            torep = torep[:4]
            x = _greedy_reinsert(x, torep, rng, n_cand=5000)
            x += rng.standard_normal(size=x.shape) * 0.003
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, e0, e1, g0, g1 = 4000, 0.018, 0.002, 0.022, 0.004
        elif s == 3:
            x = x + rng.standard_normal(size=x.shape) * 0.028
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, e0, e1, g0, g1 = 4000, 0.025, 0.002, 0.030, 0.005
        elif s == 4:
            k = 30
            idx = rng.choice(n, size=k, replace=False)
            x[idx] += rng.standard_normal(size=(k, d)) * 0.06
            x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
            steps, e0, e1, g0, g1 = 3500, 0.022, 0.002, 0.025, 0.004
        elif s == 5:
            w = _worst_vec(x)
            k = 8
            idx = np.argpartition(w, -k)[-k:].tolist()
            torep = [int(vv) for vv in idx[:6]]
            x = _greedy_reinsert(x, torep, rng, n_cand=4000)
            x += rng.standard_normal(size=x.shape) * 0.004
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, e0, e1, g0, g1 = 4000, 0.018, 0.002, 0.022, 0.004
        else:
            # cluster shake around random hub
            ci = int(rng.integers(0, n))
            dots = x @ x[ci]
            dots[ci] = -2
            nn = np.argpartition(dots, -15)[-15:]
            x[nn] += rng.standard_normal(size=(15, d)) * 0.06
            x[nn] /= np.linalg.norm(x[nn], axis=1, keepdims=True)
            x[ci] += rng.standard_normal(size=(d,)) * 0.04
            x[ci] /= np.linalg.norm(x[ci])
            steps, e0, e1, g0, g1 = 3500, 0.022, 0.002, 0.025, 0.004
        try:
            y = _fast_anneal(x, steps=steps, eta0=e0, eta1=e1, m0=g0, m1=g1, mu=0.85)
        except Exception:
            it += 1
            noimp += 1
            continue
        v = _max_cos(y)
        if v < best - 1e-12:
            best = v
            x_best = y
            noimp = 0
        else:
            noimp += 1
        it += 1

    remain = 178 - (time.time() - t0)
    if remain > 3:
        try:
            x_best = _lp_polish(x_best, seconds=min(remain, 30), init_radius=3e-3)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
