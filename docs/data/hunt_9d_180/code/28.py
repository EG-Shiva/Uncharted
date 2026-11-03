import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def _kmeans(Xemb, k, rng, iters=40):
    n = Xemb.shape[0]
    idx = rng.choice(n, k, replace=False)
    C = Xemb[idx].copy()
    lab = np.zeros(n, dtype=int)
    for _ in range(iters):
        d2 = ((Xemb[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        nl = np.argmin(d2, axis=1)
        if np.array_equal(nl, lab) and _ > 0:
            break
        lab = nl
        for c in range(k):
            m = lab == c
            if np.any(m):
                C[c] = Xemb[m].mean(axis=0)
            else:
                C[c] = Xemb[rng.randint(n)]
    # fix empty clusters
    for c in range(k):
        if not np.any(lab == c):
            lab[rng.randint(n)] = c
    return lab


def spectral_partition(X, mc, k=8, rng=None):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -10.0)
    A = (G > mc - 1e-6).astype(float)
    # connect isolates to nearest neighbour to keep graph connected
    deg = A.sum(axis=1)
    for i in range(n):
        if deg[i] < 0.5:
            j = int(np.argmax(G[i]))
            A[i, j] = 1.0
            A[j, i] = 1.0
    D = A.sum(axis=1)
    L = np.diag(D) - A
    try:
        w, V = np.linalg.eigh(L)
    except Exception:
        # fallback: random partition
        return rng.randint(0, k, size=n)
    # first k eigenvectors (includes zero modes which mark components)
    kk = min(k, n)
    Xemb = V[:, :kk].copy()
    # row normalize
    nr = np.linalg.norm(Xemb, axis=1, keepdims=True) + 1e-12
    Xemb = Xemb / nr
    lab = _kmeans(Xemb, k, rng, iters=40)
    # enforce size sanity: merge tiny / split huge by random reassignment
    for c in range(k):
        s = int(np.sum(lab == c))
        if s == 0:
            # steal from largest
            sizes = [(np.sum(lab == cc), cc) for cc in range(k)]
            sizes.sort(reverse=True)
            big = sizes[0][1]
            members = np.where(lab == big)[0]
            lab[members[0]] = c
    return lab


def anneal_adam(X, steps, T0, T1, lr, deadline):
    n, d = X.shape
    x = X.copy()
    m1 = np.zeros_like(x)
    m2 = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        if time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        T = T0 * (T1 / T0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(np.max(G))
        E = np.exp((G - gmax) / T)
        np.fill_diagonal(E, 0.0)
        row = E.sum(axis=1, keepdims=True) + 1e-12
        grad = E @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        gtan = grad - dot * x
        step = gtan / row
        # clip extreme
        nm = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        step = step * np.minimum(1.0, 1.0 / nm)
        m1 = b1 * m1 + (1 - b1) * step
        m2 = b2 * m2 + (1 - b2) * (step * step)
        c1 = 1 - b1 ** (s + 1)
        c2 = 1 - b2 ** (s + 1)
        mh = m1 / c1
        vh = m2 / c2
        upd = mh / (np.sqrt(vh) + eps)
        # keep tangential
        dot2 = (upd * x).sum(axis=1, keepdims=True)
        upd = upd - dot2 * x
        # clip update norm
        un = np.linalg.norm(upd, axis=1, keepdims=True) + 1e-12
        upd = upd * np.minimum(1.0, 1.0 / un)
        lrs = lr * (1.0 - 0.4 * frac)
        x = x - lrs * upd
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def lp_snap(X, budget, deadline, max_iters=5):
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() - start < budget and time.time() < deadline and radius > 1e-14 and it < max_iters:
        it += 1
        Gf = x @ x.T
        c = Gf[iu]
        m = float(np.max(c))
        sel = c > m - max(20 * radius, 1e-9)
        I = iu[0][sel]
        J = iu[1][sel]
        cs = c[sel]
        p = len(I)
        if p == 0:
            break
        if p > 2500:
            # keep tightest 2500
            part = np.argpartition(cs, -2500)[-2500:]
            I, J, cs = I[part], J[part], cs[part]
            p = 2500
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best


def solve():
    t0 = time.time()
    DEAD = t0 + 175.0
    SEARCH_END = t0 + 115.0
    rng = np.random.RandomState(0)
    np.random.seed(0)
    # load incumbent
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            c = np.asarray(np.load(fn), dtype=float)
            if c.shape == (180, 9):
                c = c / np.linalg.norm(c, axis=1, keepdims=True)
                if np.all(np.isfinite(c)):
                    cands.append((max_cos(c), c))
        except Exception:
            pass
    if len(cands) == 0:
        x_best = rng.randn(180, 9)
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
        b_best = max_cos(x_best)
    else:
        cands.sort(key=lambda t: t[0])
        b_best, x_best = cands[0][0], cands[0][1].copy()
    n, d = x_best.shape
    K = 8
    labels = spectral_partition(x_best, b_best, k=K, rng=rng)
    # precompute contact adjacency for neighbour lookup (refresh on improvement)
    def adjacency(X, mc):
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        A = (G > mc - 1e-6).astype(float)
        for i in range(n):
            if A[i].sum() < 0.5:
                j = int(np.argmax(G[i]))
                A[i, j] = 1.0
                A[j, i] = 1.0
        return A
    Adj = adjacency(x_best, b_best)

    it = 0
    while time.time() < SEARCH_END and time.time() < DEAD - 35:
        it += 1
        # periodically rebuild communities to follow topology
        if it % 12 == 0:
            try:
                labels = spectral_partition(x_best, b_best, k=K, rng=rng)
                Adj = adjacency(x_best, b_best)
            except Exception:
                pass
        c = int(rng.randint(K))
        idx = np.where(labels == c)[0]
        if idx.size < 4 or idx.size > 70:
            # pick another community
            c = int(rng.randint(K))
            idx = np.where(labels == c)[0]
            if idx.size < 4 or idx.size > 70:
                continue
        a, b = rng.choice(d, 2, replace=False)
        ang = rng.uniform(0.03, 0.25) * (1 if rng.rand() < 0.5 else -1)
        ca, sa = float(np.cos(ang)), float(np.sin(ang))
        x_try = x_best.copy()
        xa = x_try[idx, a].copy()
        xb = x_try[idx, b].copy()
        x_try[idx, a] = ca * xa - sa * xb
        x_try[idx, b] = sa * xa + ca * xb
        # optionally counter-rotate neighbouring community
        if rng.rand() < 0.5:
            best_cut, nb = -1, -1
            for k2 in range(K):
                if k2 == c:
                    continue
                idx2 = np.where(labels == k2)[0]
                if idx2.size == 0:
                    continue
                cut = float(Adj[np.ix_(idx, idx2)].sum())
                # also consider geometric proximity if no contacts
                if cut > best_cut:
                    best_cut, nb = cut, k2
            if nb >= 0:
                idxN = np.where(labels == nb)[0]
                ang2 = -ang * 0.5
                ca2, sa2 = float(np.cos(ang2)), float(np.sin(ang2))
                xn_a = x_try[idxN, a].copy()
                xn_b = x_try[idxN, b].copy()
                x_try[idxN, a] = ca2 * xn_a - sa2 * xn_b
                x_try[idxN, b] = sa2 * xn_a + ca2 * xn_b
        # annealed LogSumExp Adam, at most 300 steps
        steps = 220 if idx.size < 30 else 260
        if steps > 300:
            steps = 300
        T0 = 0.008 if rng.rand() < 0.5 else 0.012
        T1 = 0.0004
        lr = 0.03 if rng.rand() < 0.5 else 0.05
        inner_dead = min(DEAD - 5, time.time() + 4.0)
        x_rel = anneal_adam(x_try, steps, T0, T1, lr, inner_dead)
        v = max_cos(x_rel)
        # quick reject if far worse (LP unlikely to recover)
        if v > b_best + 0.015:
            continue
        # short capped LP snap
        lp_budget = 1.5
        if v < b_best + 1e-4:
            lp_budget = 2.5
        if time.time() > SEARCH_END or time.time() > DEAD - 35:
            break
        try:
            x_pol, vp = lp_snap(x_rel, lp_budget, min(DEAD - 30, SEARCH_END + 5), max_iters=4)
        except Exception:
            continue
        if vp < b_best - 1e-12:
            x_best, b_best = x_pol, vp
            try:
                labels = spectral_partition(x_best, b_best, k=K, rng=rng)
                Adj = adjacency(x_best, b_best)
            except Exception:
                pass
    # final polish with remaining time
    remain = DEAD - time.time() - 1.0
    if remain > 2:
        try:
            budget = min(remain, 40.0)
            x_best, b_best = lp_snap(x_best, budget, DEAD - 0.5, max_iters=50)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
