import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def lp_polish_capped(x, budget, max_iters=6, r0=1e-3):
    t0 = time.time()
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = r0
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() - t0 < budget and radius > 1e-14 and it < max_iters:
        it += 1
        if budget - (time.time() - t0) < 0.3:
            break
        Gf = x @ x.T
        c = Gf[iu]
        m = float(np.max(c))
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        csub = c[sel]
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
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": max(0.5, budget - (time.time() - t0))})
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
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x, best

def smooth_relax(x, steps, T0, T1, lr0, mu=0.85):
    n, d = x.shape
    x = x.copy()
    bv = max_cos(x)
    best = x.copy()
    vel = np.zeros_like(x)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(np.max(G))
        E = np.exp((G - gmax) / T)
        np.fill_diagonal(E, 0.0)
        row = np.sum(E, axis=1, keepdims=True) + 1e-12
        grad = E @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gtan = grad - dot * x
        step = gtan / row
        nm = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        step = step * np.minimum(1.0, 0.8 / nm)
        vel = mu * vel + step
        x = x - lr * vel
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if s == steps - 1 or (s & 31) == 31:
            v = max_cos(x)
            if v < bv:
                bv = v
                best = x.copy()
    return best, bv

def spectral_partition(x, K=8, thresh=1e-6, rng=None):
    n = x.shape[0]
    if rng is None:
        rng = np.random.RandomState(0)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    A = (G > m - thresh).astype(float)
    np.fill_diagonal(A, 0.0)
    # symmetrize
    A = np.maximum(A, A.T)
    deg = np.sum(A, axis=1)
    L = np.diag(deg) - A
    try:
        w, V = np.linalg.eigh(L)
    except Exception:
        return np.array([i % K for i in range(n)], dtype=int), A
    # take K vectors skipping first (smallest)
    if V.shape[1] < K + 1:
        K = max(2, V.shape[1] - 1)
    F = V[:, 1:K + 1]
    # row normalize (Ng-Jordan-Weiss)
    nr = np.linalg.norm(F, axis=1, keepdims=True) + 1e-12
    F = F / nr
    # k-means with 2 restarts, 30 iters
    best_labels = None
    best_cost = np.inf
    for _ in range(2):
        idx = rng.choice(n, K, replace=False)
        C = F[idx].copy()
        labels = np.zeros(n, dtype=int)
        for _it in range(30):
            d2 = ((F[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
            nl = np.argmin(d2, axis=1)
            if np.array_equal(nl, labels) and _it > 0:
                labels = nl
                break
            labels = nl
            for k in range(K):
                msk = labels == k
                if np.any(msk):
                    C[k] = F[msk].mean(axis=0)
                else:
                    C[k] = F[rng.randint(n)]
        # cost
        cost = 0.0
        for k in range(K):
            msk = labels == k
            if np.any(msk):
                cost += float((((F[msk] - C[k]) ** 2).sum()))
        if cost < best_cost:
            best_cost = cost
            best_labels = labels.copy()
    # relabel empty safety: ensure 6-10 nonempty; if some empty, merge is fine
    return best_labels.astype(int), A

def solve():
    t0 = time.time()
    HARD_END = t0 + 168.0
    SEARCH_END = t0 + 115.0
    rng = np.random.RandomState(12345)
    # load
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
    try:
        labels, A = spectral_partition(x_best, K=K, thresh=1e-6, rng=rng)
    except Exception:
        labels = np.array([i % K for i in range(n)])
        A = np.zeros((n, n))
    # quick initial capped LP if time allows (improve seed slightly)
    # keep it very short to preserve search budget
    if time.time() < SEARCH_END - 20:
        try:
            x_best, b_best = lp_polish_capped(x_best, budget=4.0, max_iters=5)
            labels, A = spectral_partition(x_best, K=K, thresh=1e-6, rng=rng)
        except Exception:
            pass
    nmove = 0
    # precompute community members
    while time.time() < SEARCH_END and time.time() < HARD_END - 30 and nmove < 60:
        nmove += 1
        remain = SEARCH_END - time.time()
        if remain < 3.0:
            break
        # repartition periodically or if labels degenerate
        if nmove % 12 == 0:
            try:
                labels, A = spectral_partition(x_best, K=K, thresh=1e-6, rng=rng)
            except Exception:
                pass
        # pick community (prefer larger / high-contact? uniform is fine + deterministic via rng)
        c = int(rng.randint(K))
        idx = np.where(labels == c)[0]
        if len(idx) < 3 or len(idx) > 120:
            # pick another
            c = int(rng.randint(K))
            idx = np.where(labels == c)[0]
            if len(idx) < 3:
                continue
        a, b = rng.choice(d, 2, replace=False)
        ang = float(rng.uniform(0.03, 0.25)) * (1.0 if rng.rand() < 0.5 else -1.0)
        ca, sa = float(np.cos(ang)), float(np.sin(ang))
        # neighbour community with most inter edges for counter-rotation
        do_counter = rng.rand() < 0.5
        nidx = np.array([], dtype=int)
        nang = 0.0
        if do_counter:
            try:
                best_k, best_e = -1, -1
                for k in range(K):
                    if k == c:
                        continue
                    jdx = np.where(labels == k)[0]
                    if len(jdx) == 0:
                        continue
                    e = float(np.sum(A[np.ix_(idx, jdx)]))
                    if e > best_e:
                        best_e, best_k = e, k
                if best_k >= 0 and best_e > 0:
                    nidx = np.where(labels == best_k)[0]
                    nang = -ang * 0.5
                else:
                    do_counter = False
            except Exception:
                do_counter = False
        x_try = x_best.copy()
        # rigid rotation in plane (a,b): x' = x @ R.T where R is rotation
        # for community c
        xa = x_try[idx, a].copy()
        xb = x_try[idx, b].copy()
        x_try[idx, a] = ca * xa - sa * xb
        x_try[idx, b] = sa * xa + ca * xb
        if do_counter and len(nidx) > 0:
            cn, sn = float(np.cos(nang)), float(np.sin(nang))
            ya = x_try[nidx, a].copy()
            yb = x_try[nidx, b].copy()
            x_try[nidx, a] = cn * ya - sn * yb
            x_try[nidx, b] = sn * ya + cn * yb
        x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        # annealed LogSumExp descent (Adam-like momentum), at most 300 steps
        steps = 200 if remain > 40 else (120 if remain > 15 else 80)
        steps = min(300, steps)
        try:
            x_rel, v_rel = smooth_relax(x_try, steps=steps, T0=0.012, T1=0.0008, lr0=0.18, mu=0.85)
        except Exception:
            continue
        # gate LP: only polish if close to incumbent
        if v_rel > b_best + 0.003:
            continue
        if time.time() > SEARCH_END - 2.0 or time.time() > HARD_END - 28:
            # no time for LP; accept if raw beats best
            if v_rel < b_best - 1e-12:
                x_best, b_best = x_rel, v_rel
                try:
                    labels, A = spectral_partition(x_best, K=K, thresh=1e-6, rng=rng)
                except Exception:
                    pass
            continue
        try:
            x_pol, v_pol = lp_polish_capped(x_rel, budget=min(2.5, max(0.8, SEARCH_END - time.time(), HARD_END - 28 - (time.time()))), max_iters=4)
        except Exception:
            continue
        if v_pol < b_best - 1e-12:
            x_best, b_best = x_pol, v_pol
            try:
                labels, A = spectral_partition(x_best, K=K, thresh=1e-6, rng=rng)
            except Exception:
                pass
        if time.time() >= SEARCH_END:
            break
    # final polish with remaining time
    try:
        rem = HARD_END - time.time()
        if rem > 2.0:
            x_best, b_best = lp_polish_capped(x_best, budget=min(22.0, rem - 1.0), max_iters=30)
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # tiny shrink guard not needed; ensure finite
    x_best = np.where(np.isfinite(x_best), x_best, 0.0)
    nr = np.linalg.norm(x_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    x_best = x_best / nr
    return x_best
