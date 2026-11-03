import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def adam_steps(x, alpha, steps, lr):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        gmax = G.max()
        E = np.exp(alpha * (G - gmax))
        np.fill_diagonal(E, 0.0)
        S = E.sum()
        if not np.isfinite(S) or S == 0:
            break
        grad = (E @ x) * (n / S)
        nd = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - nd * x
        m = b1 * m + (1.0 - b1) * gt
        v = b2 * v + (1.0 - b2) * gt * gt
        mhat = m / (1.0 - b1 ** t)
        vhat = v / (1.0 - b2 ** t)
        delta = lr * mhat / (np.sqrt(vhat) + eps)
        dn = (delta * x).sum(axis=1, keepdims=True)
        delta = delta - dn * x
        rn = np.linalg.norm(delta, axis=1, keepdims=True) + 1e-12
        scale = np.minimum(1.0, 0.05 / rn)
        delta = delta * scale
        x = x - delta
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def anneal300(x):
    x = adam_steps(x, 30.0, 120, 0.008)
    x = adam_steps(x, 140.0, 180, 0.003)
    return x

def lp_capped(x, seconds, radius_init=0.003, cap=4000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = radius_init
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c_full = (x @ x.T)[iu]
        m = c_full.max()
        sel = c_full > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-c_full[idx])[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        c = c_full[sel]
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
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def spectral_partition(contact_adj, K, rng):
    n = contact_adj.shape[0]
    try:
        deg = contact_adj.sum(axis=1)
        L = np.diag(deg) - contact_adj
        w, V = np.linalg.eigh(L)
        F = V[:, 1:K]
        if F.shape[1] == 0:
            return rng.randint(0, K, size=n)
        nr = np.linalg.norm(F, axis=1, keepdims=True) + 1e-12
        Xn = F / nr
        fdim = Xn.shape[1]
        init = rng.choice(n, K, replace=False)
        C = Xn[init].copy()
        labels = np.zeros(n, dtype=int)
        for _ in range(30):
            d2 = ((Xn[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
            nl = d2.argmin(axis=1)
            if _ > 0 and np.array_equal(nl, labels):
                break
            labels = nl
            for k in range(K):
                mk = labels == k
                if np.any(mk):
                    C[k] = Xn[mk].mean(axis=0)
                else:
                    C[k] = Xn[rng.randint(0, n)]
        # avoid empty clusters
        for k in range(K):
            if np.sum(labels == k) == 0:
                labels[rng.randint(0, n)] = k
        return labels
    except Exception:
        return rng.randint(0, K, size=n)

def solve():
    np.random.seed(0)
    t0 = time.time()
    search_end = t0 + 130.0
    deadline = t0 + 176.0
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (180, 9):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        r0 = np.random.RandomState(0)
        best = r0.randn(180, 9)
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = max_cos(best)
    n, d = 180, 9
    rng = np.random.RandomState(0)
    it = 0
    iu = np.triu_indices(n, 1)
    while time.time() < search_end:
        it += 1
        if time.time() > deadline - 8.0:
            break
        K = 6 + (it % 5)  # 6..10
        # build contact graph within 1e-6 of max
        try:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            Adj = (G > m - 1e-6).astype(float)
            np.fill_diagonal(Adj, 0.0)
            # symmetrize
            Adj = np.maximum(Adj, Adj.T)
        except Exception:
            Adj = np.zeros((n, n))
        labels = spectral_partition(Adj, K, rng)
        c = int(rng.randint(0, K))
        idx_c = np.where(labels == c)[0]
        if len(idx_c) == 0 or len(idx_c) == n:
            continue
        # random 2-plane
        a_ax = int(rng.randint(0, d))
        b_ax = int(rng.randint(0, d))
        while b_ax == a_ax:
            b_ax = int(rng.randint(0, d))
        theta = float(rng.uniform(0.03, 0.25)) * (1.0 if rng.rand() < 0.5 else -1.0)
        y = best.copy()
        co, si = np.cos(theta), np.sin(theta)
        xa = y[idx_c, a_ax].copy()
        xb = y[idx_c, b_ax].copy()
        y[idx_c, a_ax] = co * xa - si * xb
        y[idx_c, b_ax] = si * xa + co * xb
        # optionally counter-rotate neighbouring community
        if rng.rand() < 0.5:
            # neighbour with most inter edges
            best_nb = -1
            best_cnt = -1
            for k in range(K):
                if k == c:
                    continue
                idx_k = np.where(labels == k)[0]
                if len(idx_k) == 0:
                    continue
                try:
                    cnt = Adj[np.ix_(idx_c, idx_k)].sum()
                except Exception:
                    cnt = 0
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_nb = k
            if best_nb >= 0:
                idx_n = np.where(labels == best_nb)[0]
                th2 = -theta * 0.5
                co2, si2 = np.cos(th2), np.sin(th2)
                xa2 = y[idx_n, a_ax].copy()
                xb2 = y[idx_n, b_ax].copy()
                y[idx_n, a_ax] = co2 * xa2 - si2 * xb2
                y[idx_n, b_ax] = si2 * xa2 + co2 * xb2
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        # annealed LogSumExp Adam, at most 300 steps
        try:
            y2 = anneal300(y)
        except Exception:
            continue
        try:
            v2 = max_cos(y2)
        except Exception:
            continue
        # only spend LP if promising
        if v2 > best_val + 0.002:
            continue
        remain_search = search_end - time.time()
        if remain_search < 1.5:
            if v2 < best_val:
                best, best_val = y2, v2
            continue
        secs = min(1.0, max(0.5, remain_search - 0.5))
        # slightly longer if it already beats incumbent
        if v2 < best_val:
            secs = min(1.5, secs + 0.3)
        try:
            y3 = lp_capped(y2, secs, radius_init=0.003, cap=4000)
            v3 = max_cos(y3)
        except Exception:
            continue
        if v3 < best_val:
            best, best_val = y3, v3
        if time.time() >= search_end:
            break
    # short final capped polish if time left
    if time.time() < deadline - 3.0:
        try:
            secs = min(6.0, max(1.0, deadline - time.time() - 1.0))
            best = lp_capped(best, secs, radius_init=1e-3, cap=6000)
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
