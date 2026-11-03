import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def spectral_partition(adj, k, rng):
    n = adj.shape[0]
    try:
        deg = adj.sum(axis=1)
        L = np.diag(deg) - adj
        w, V = np.linalg.eigh(L)
        idx = np.argsort(w)
        # skip trivial constant vector if graph connected
        if w[idx[1]] > 1e-10:
            cols = idx[1:k + 1]
            if len(cols) < k:
                cols = idx[:k]
        else:
            cols = idx[:k]
        emb = V[:, cols]
        nr = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        emb = emb / nr
        centers = emb[rng.choice(n, k, replace=False)].copy()
        assign = np.zeros(n, dtype=int)
        for _ in range(40):
            d2 = ((emb[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            na = np.argmin(d2, axis=1)
            if _ > 0 and np.array_equal(na, assign):
                break
            assign = na
            for c in range(k):
                m = assign == c
                if np.any(m):
                    centers[c] = emb[m].mean(axis=0)
                else:
                    centers[c] = emb[rng.integers(0, n)]
        # fix empty clusters
        for c in range(k):
            if np.sum(assign == c) == 0:
                assign[rng.integers(0, n)] = c
        return assign
    except Exception:
        return rng.integers(0, k, size=n)


def rotate_community(x, mask, plane, theta):
    R = np.eye(x.shape[1])
    a, b = plane
    c = np.cos(theta)
    s = np.sin(theta)
    R[a, a] = c
    R[a, b] = -s
    R[b, a] = s
    R[b, b] = c
    y = x.copy()
    y[mask] = y[mask] @ R.T
    # renorm for safety
    y[mask] /= np.linalg.norm(y[mask], axis=1, keepdims=True)
    return y


def anneal_lse_adam(x, steps, k0, k1, lr0, lr1):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for it in range(steps):
        f = it / (steps - 1) if steps > 1 else 1.0
        k = k0 + (k1 - k0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(np.max(G))
        # avoid overflow: G-gmax <= ~1, k up to ~250 => exp ok
        W = np.exp(k * (G - gmax))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        m = b1 * m + (1.0 - b1) * grad
        v = b2 * v + (1.0 - b2) * grad * grad
        mhat = m / (1.0 - b1 ** (it + 1))
        vhat = v / (1.0 - b2 ** (it + 1))
        step = mhat / (np.sqrt(vhat) + eps)
        x = x - lr * step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def lp_polish_capped(x, budget, deadline):
    # short trust-region LP, at most a few solves, p capped
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    n_iter = 0
    while time.time() < deadline and time.time() < budget and radius > 1e-14 and n_iter < 4:
        n_iter += 1
        G = x @ x.T
        c_all = G[iu0]
        mcur = float(np.max(c_all))
        thr = mcur - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        p = int(np.count_nonzero(sel))
        if p == 0:
            break
        I_all = iu0[0][sel]
        J_all = iu0[1][sel]
        c_sel = c_all[sel]
        if p > 1500:
            idx = np.argpartition(-c_sel, 1500)[:1500]
            I_all = I_all[idx]
            J_all = J_all[idx]
            c_sel = c_sel[idx]
            p = 1500
        I = I_all
        J = J_all
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            ar = np.arange(d)
            cols = np.empty(p * (2 * d + 1), dtype=int)
            vals = np.empty(p * (2 * d + 1))
            # build row-major blocks
            XJ = x[J]
            XI = x[I]
            block = np.concatenate([XJ, XI, -np.ones((p, 1))], axis=1)
            vals[:] = block.ravel()
            # cols row-major
            base_I = I[:, None] * d + ar[None, :]
            base_J = J[:, None] * d + ar[None, :]
            cblock = np.concatenate([base_I, base_J, np.full((p, 1), n * d)], axis=1)
            cols[:] = cblock.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv),
            ).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(
                obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                method="highs",
            )
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


def solve():
    t0 = time.time()
    HARD_END = t0 + 168.0
    SEARCH_END = t0 + 130.0
    np.random.seed(0)
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
        if c is not None and np.shape(c) == (180, 9):
            c = np.array(c, dtype=float)
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            cands.append(c)
    if not cands:
        r0 = np.random.default_rng(0)
        c = r0.standard_normal((180, 9))
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        cands.append(c)
    vals = [max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    best_val = float(vals[bi])

    rng = np.random.default_rng(12345)
    trial = 0
    n, d = 180, 9

    while time.time() < SEARCH_END and time.time() < HARD_END - 20.0:
        trial += 1
        if HARD_END - time.time() < 12.0:
            break
        # build contact graph within 1e-6 of max on incumbent
        try:
            G = x_best @ x_best.T
            np.fill_diagonal(G, -2.0)
            m0 = float(np.max(G))
            adj = (G > m0 - 1e-6).astype(float)
            np.fill_diagonal(adj, 0.0)
        except Exception:
            m0 = best_val
            adj = np.zeros((n, n))
        k = int(rng.integers(6, 11))  # 6..10
        assign = spectral_partition(adj, k, rng)
        # pick one community (prefer larger ones sometimes)
        # choose random community weighted by size
        uniq, counts = np.unique(assign, return_counts=True)
        # pick uniformly among communities to ensure coverage
        c_pick = int(uniq[rng.integers(0, len(uniq))])
        mask = assign == c_pick
        sz = int(np.sum(mask))
        if sz == 0 or sz >= n - 1:
            continue
        # random coordinate 2-plane
        a = int(rng.integers(0, d))
        b = int(rng.integers(0, d))
        while b == a:
            b = int(rng.integers(0, d))
        theta = float(rng.uniform(0.03, 0.25))
        if rng.random() < 0.5:
            theta = -theta
        x_try = rotate_community(x_best, mask, (a, b), theta)
        # optionally counter-rotate neighbouring community
        if rng.random() < 0.6:
            # neighbour = community with most inter edges to mask
            inter = adj[mask][:, ~mask] if sz < n else None
            try:
                # compute inter-community edge counts
                best_nb = -1
                best_e = -1
                for cc in uniq:
                    if int(cc) == c_pick:
                        continue
                    m2 = assign == int(cc)
                    e = float(adj[np.ix_(mask, m2)].sum())
                    if e > best_e:
                        best_e = e
                        best_nb = int(cc)
                if best_nb >= 0:
                    mask2 = assign == best_nb
                    x_try = rotate_community(x_try, mask2, (a, b), -theta * 0.5)
            except Exception:
                pass
        # annealed LogSumExp Adam, at most 300 steps
        steps = int(rng.integers(180, 301))
        k0 = float(rng.uniform(15.0, 40.0))
        k1 = float(rng.uniform(120.0, 250.0))
        lr0 = float(rng.uniform(0.003, 0.007))
        lr1 = lr0 * 0.1
        try:
            x_sm = anneal_lse_adam(x_try, steps, k0, k1, lr0, lr1)
        except Exception:
            continue
        try:
            v_sm = max_cos(x_sm)
        except Exception:
            continue
        if v_sm > best_val + 0.012:
            continue
        # short capped LP
        lp_budget = time.time() + 2.0
        lp_deadline = min(HARD_END - 16.0, SEARCH_END + 6.0)
        # don't exceed search end too much; keep quick
        if time.time() > SEARCH_END - 1.0 and trial > 5:
            # after search end, skip LP except promising
            if v_sm >= best_val:
                continue
        try:
            # limit LP time to ~1.5-2s
            end_t = min(lp_budget, lp_deadline)
            if end_t - time.time() < 0.5:
                x_pol = x_sm
                v_pol = v_sm
            else:
                x_pol = lp_polish_capped(x_sm, end_t, end_t)
                v_pol = max_cos(x_pol)
        except Exception:
            continue
        if v_pol < best_val:
            x_best = x_pol
            best_val = v_pol

    # final polish with remaining time
    rem = HARD_END - time.time() - 1.0
    if rem > 2.0:
        try:
            x_best = lp_polish_capped(x_best, time.time() + min(rem, 12.0), HARD_END - 0.5)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    x_best = np.nan_to_num(x_best)
    nr = np.linalg.norm(x_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    x_best = x_best / nr
    return x_best
