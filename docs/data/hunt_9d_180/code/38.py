import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

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
        # track using already computed gmax next iter; cheap check every 25
        if (s % 25 == 24) or (s == steps - 1):
            # gmax is from before update; evaluate current
            G2 = x @ x.T
            np.fill_diagonal(G2, -2.0)
            v = float(np.max(G2))
            if v < bv:
                bv = v
                best = x.copy()
    return best, bv

def capped_lp(x, time_budget=2.0, max_iters=4, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    xb = x.copy()
    radius = 8e-4
    t0 = time.time()
    iu = np.triu_indices(n, 1)
    it = 0
    while it < max_iters and (time.time() - t0) < time_budget:
        if deadline is not None and time.time() > deadline - 0.5:
            break
        if radius < 1e-13:
            break
        Gfull = x @ x.T
        cfull = Gfull[iu]
        m = float(np.max(cfull))
        window = max(6.0 * radius, 3e-4)
        sel = cfull > m - window
        I0, J0 = iu[0][sel], iu[1][sel]
        csub0 = cfull[sel]
        p0 = len(I0)
        if p0 == 0:
            break
        # cap to closest 1800 pairs
        if p0 > 1800:
            idx = np.argpartition(-csub0, 1800)[:1800]
            I, J, csub = I0[idx], J0[idx], csub0[idx]
            p = 1800
        else:
            I, J, csub = I0, J0, csub0
            p = p0
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
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            it += 1
            continue
        if res.status != 0:
            radius *= 0.5
            it += 1
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        it += 1
        if v < best:
            x, best, xb = y, v, y.copy()
            radius = min(radius * 1.4, 0.02)
        else:
            radius *= 0.5
    return xb, best

def spectral_communities(x, K, thresh=1e-6, rng=None):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    A = (G > m - thresh).astype(float)
    np.fill_diagonal(A, 0.0)
    # symmetrize
    A = np.maximum(A, A.T)
    deg = np.sum(A, axis=1)
    dinv = np.zeros(n)
    nz = deg > 0
    dinv[nz] = 1.0 / np.sqrt(deg[nz])
    # normalized Laplacian Lsym = I - D^{-1/2} A D^{-1/2}
    S = (dinv[:, None] * A) * dinv[None, :]
    L = np.eye(n) - S
    try:
        w, V = np.linalg.eigh(L)
    except Exception:
        # fallback random
        return rng.randint(0, K, size=n)
    order = np.argsort(w)
    # take K smallest (including zero)
    take = order[:K]
    F = V[:, take]
    # row normalize (skip zero rows)
    nr = np.linalg.norm(F, axis=1, keepdims=True) + 1e-12
    F = F / nr
    # k-means
    if rng is None:
        rng = np.random.RandomState(0)
    # init centroids as K random distinct rows
    perm = rng.permutation(n)
    cent = F[perm[:K]].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(30):
        d2 = np.sum((F[:, None, :] - cent[None, :, :]) ** 2, axis=2)
        nl = np.argmin(d2, axis=1)
        if np.array_equal(nl, labels) and _ > 0:
            labels = nl
            break
        labels = nl
        for k in range(K):
            msk = labels == k
            if np.any(msk):
                cent[k] = np.mean(F[msk], axis=0)
            else:
                cent[k] = F[rng.randint(n)]
    # fix empty clusters
    return labels

def solve():
    t0 = time.time()
    deadline = t0 + 178.0
    search_end = t0 + 115.0
    rng = np.random.RandomState(0)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        if c is not None:
            try:
                c = np.asarray(c, dtype=float)
                if c.shape == (180, 9):
                    c = c / np.linalg.norm(c, axis=1, keepdims=True)
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
    n, d = 180, 9
    K = 8
    try:
        labels = spectral_communities(x_best, K, thresh=1e-6, rng=rng)
    except Exception:
        labels = rng.randint(0, K, size=n)
    it = 0
    # precompute inter-community helper recomputed periodically
    while time.time() < search_end and time.time() < deadline - 30:
        it += 1
        if it % 12 == 1:
            try:
                labels = spectral_communities(x_best, K, thresh=1e-6, rng=rng)
            except Exception:
                pass
        # pick one community
        ci = int(rng.randint(K))
        idx = np.where(labels == ci)[0]
        if len(idx) == 0 or len(idx) > 90:
            continue
        # random 2-plane (coordinate plane) and angle 0.03-0.25
        a, b = rng.choice(d, size=2, replace=False)
        ang = float(rng.uniform(0.03, 0.25)) * (1.0 if rng.rand() < 0.5 else -1.0)
        ca, sa = np.cos(ang), np.sin(ang)
        Q = np.eye(d)
        Q[a, a] = ca; Q[a, b] = -sa; Q[b, a] = sa; Q[b, b] = ca
        x_try = x_best.copy()
        x_try[idx] = x_try[idx] @ Q.T
        # optionally counter-rotate a neighbouring community in same plane
        if rng.rand() < 0.5:
            # neighbour = community with most cross edges else random
            try:
                G = x_best @ x_best.T
                m = float(np.max(G - 2*np.eye(n)))
                A = (G > m - 1e-6).astype(float)
                np.fill_diagonal(A, 0.0)
                # cross counts
                best_j, best_c = -1, -1
                for j in range(K):
                    if j == ci:
                        continue
                    jdx = np.where(labels == j)[0]
                    if len(jdx) == 0:
                        continue
                    csum = float(np.sum(A[np.ix_(idx, jdx)]))
                    if csum > best_c:
                        best_c = csum
                        best_j = j
                if best_j >= 0 and (best_c > 0 or rng.rand() < 0.3):
                    jdx = np.where(labels == best_j)[0]
                    ang2 = -ang * float(rng.uniform(0.3, 1.0))
                    c2, s2 = np.cos(ang2), np.sin(ang2)
                    Q2 = np.eye(d)
                    Q2[a, a] = c2; Q2[a, b] = -s2; Q2[b, a] = s2; Q2[b, b] = c2
                    x_try[jdx] = x_try[jdx] @ Q2.T
            except Exception:
                pass
        x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        # annealed LogSumExp Adam, at most 300 steps
        steps = 250 if (it % 3 != 0) else 300
        # vary temperature slightly for diversity
        T0 = 0.010 if (it % 2 == 0) else 0.014
        T1 = 0.0007
        lr0 = 0.012
        try:
            nb, vb = smooth_relax(x_try, steps=steps, T0=T0, T1=T1, lr0=lr0, mu=0.85)
        except Exception:
            continue
        # gate LP: only if close to incumbent
        if vb > b_best + 0.004:
            continue
        # short capped LP
        lbudget = 1.5
        if vb < b_best - 1e-12:
            lbudget = 3.0
        try:
            if time.time() > deadline - 30:
                break
            pb, pv = capped_lp(nb, time_budget=lbudget, max_iters=3, deadline=deadline)
        except Exception:
            continue
        if pv < b_best - 1e-12:
            x_best = pb.copy()
            b_best = pv
            try:
                labels = spectral_communities(x_best, K, thresh=1e-6, rng=rng)
            except Exception:
                pass
        if time.time() > search_end:
            break
    # final polish with remaining time
    try:
        rem = deadline - time.time() - 1.0
        if rem > 3:
            # light sharpen
            if rem > 18:
                try:
                    nb, vb = smooth_relax(x_best, steps=200, T0=0.004, T1=0.0004, lr0=0.006, mu=0.9)
                    if vb < b_best:
                        x_best, b_best = nb, vb
                except Exception:
                    pass
                rem = deadline - time.time() - 1.0
            if rem > 3:
                try:
                    x_best, _ = capped_lp(x_best, time_budget=min(rem - 0.5, 25.0), max_iters=12, deadline=deadline)
                except Exception:
                    pass
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
