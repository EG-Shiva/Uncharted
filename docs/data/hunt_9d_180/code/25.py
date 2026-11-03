import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds, cap=1800):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 2e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.05:
            break
        G = x @ x.T
        c_all = G[iu]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        I_all = iu[0][sel]
        J_all = iu[1][sel]
        c_sel = c_all[sel]
        if cnt > cap:
            idx = np.argpartition(-c_sel, cap)[:cap]
            I_all = I_all[idx]
            J_all = J_all[idx]
            c_sel = c_sel[idx]
            cnt = cap
        I = I_all
        J = J_all
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols2d = np.empty((p, 2 * d + 1), dtype=np.int64)
        cols2d[:, :d] = I[:, None] * d + ar
        cols2d[:, d:2 * d] = J[:, None] * d + ar
        cols2d[:, 2 * d] = n * d
        cols = cols2d.ravel()
        vals2d = np.empty((p, 2 * d + 1), dtype=np.float64)
        vals2d[:, :d] = x[J]
        vals2d[:, d:2 * d] = x[I]
        vals2d[:, 2 * d] = -1.0
        vals = vals2d.ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv),
            ).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(
                obj,
                A_ub=A_ub,
                b_ub=-c_sel,
                A_eq=A_eq,
                b_eq=np.zeros(n),
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
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def lse_adam(x, steps, k0, k1, lr0, lr1, b1=0.9, b2=0.999, eps=1e-8):
    n, d = x.shape
    mt = np.zeros_like(x)
    vt = np.zeros_like(x)
    for it in range(steps):
        f = it / (steps - 1) if steps > 1 else 1.0
        k = k0 + (k1 - k0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(k * (G - gmax))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        # tangent projection
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        mt = b1 * mt + (1.0 - b1) * grad
        vt = b2 * vt + (1.0 - b2) * (grad * grad)
        mhat = mt / (1.0 - b1 ** (it + 1))
        vhat = vt / (1.0 - b2 ** (it + 1))
        step = mhat / (np.sqrt(vhat) + eps)
        # project step to tangent
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        x = x - lr * step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def spectral_labels(A, K, rng):
    n = A.shape[0]
    try:
        deg = A.sum(axis=1)
        L = np.diag(deg) - A
        w, V = np.linalg.eigh(L)
        nz = int(np.count_nonzero(w < 1e-10))
        if nz >= 2:
            # disconnected: use K smallest (includes discriminating constants)
            kk = min(K, n)
            Emb = V[:, :kk]
        else:
            kk = min(K, n - 1)
            Emb = V[:, 1:kk + 1]
        # row normalize
        nr = np.linalg.norm(Emb, axis=1, keepdims=True) + 1e-12
        Emb = Emb / nr
        # k-means
        idx = rng.choice(n, size=K, replace=False)
        C = Emb[idx].copy()
        lab = np.zeros(n, dtype=int)
        for _ in range(25):
            d2 = ((Emb[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
            nlab = np.argmin(d2, axis=1)
            if np.array_equal(nlab, lab) and _ > 0:
                lab = nlab
                break
            lab = nlab
            for kk_i in range(K):
                m = lab == kk_i
                if np.any(m):
                    C[kk_i] = Emb[m].mean(axis=0)
                else:
                    C[kk_i] = Emb[rng.integers(0, n)]
        # fix empty clusters
        return lab
    except Exception:
        return rng.integers(0, K, size=n)


def solve():
    t0 = time.time()
    HARD = 168.0
    SEARCH_END = 130.0
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
        if c is not None and getattr(c, "shape", None) == (180, 9):
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            c = np.nan_to_num(c)
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

    trial = 0
    while time.time() - t0 < SEARCH_END and time.time() - t0 < HARD - 25.0:
        trial += 1
        elapsed = time.time() - t0
        if HARD - elapsed < 10.0:
            break
        rng = np.random.default_rng(1000 + trial * 12345)
        # contact graph within 1e-6 of max
        try:
            G0 = x_best @ x_best.T
            np.fill_diagonal(G0, -2.0)
            m0 = float(G0.max())
            A = (G0 > m0 - 1e-6).astype(np.float64)
            np.fill_diagonal(A, 0.0)
            A = np.maximum(A, A.T)
        except Exception:
            m0 = best_val
            A = np.zeros((180, 180))
        K = int(rng.integers(6, 11))  # 6..10
        lab = spectral_labels(A, K, rng)
        # pick one community (prefer non-trivial size)
        c = int(rng.integers(0, K))
        # avoid empty / full
        for _ in range(K):
            idx_c = np.where(lab == c)[0]
            if 3 <= len(idx_c) <= 170:
                break
            c = (c + 1) % K
        idx_c = np.where(lab == c)[0]
        if len(idx_c) < 3 or len(idx_c) > 175:
            continue
        # random 2-plane and angle 0.03-0.25 rad
        a = int(rng.integers(0, 9))
        b = int(rng.integers(0, 9))
        while b == a:
            b = int(rng.integers(0, 9))
        theta = float(rng.uniform(0.03, 0.25))
        if rng.random() < 0.5:
            theta = -theta
        co, si = float(np.cos(theta)), float(np.sin(theta))
        x_try = x_best.copy()
        xa = x_try[idx_c, a].copy()
        xb_ = x_try[idx_c, b].copy()
        x_try[idx_c, a] = co * xa - si * xb_
        x_try[idx_c, b] = si * xa + co * xb_
        # optionally counter-rotate a neighbouring community
        if rng.random() < 0.5:
            # neighbour = community with most inter edges to c
            best_nb = -1
            best_e = -1
            for kk in range(K):
                if kk == c:
                    continue
                idx_k = np.where(lab == kk)[0]
                if len(idx_k) == 0:
                    continue
                e = float(A[np.ix_(idx_c, idx_k)].sum())
                if e > best_e:
                    best_e = e
                    best_nb = kk
            if best_nb >= 0:
                idx_n = np.where(lab == best_nb)[0]
                if 2 <= len(idx_n) <= 175:
                    # scale counter angle to balance
                    th2 = -theta * min(1.0, len(idx_c) / max(1, len(idx_n))) * 0.7
                    co2, si2 = float(np.cos(th2)), float(np.sin(th2))
                    ya = x_try[idx_n, a].copy()
                    yb = x_try[idx_n, b].copy()
                    x_try[idx_n, a] = co2 * ya - si2 * yb
                    x_try[idx_n, b] = si2 * ya + co2 * yb
        x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        # annealed LogSumExp Adam, at most 300 steps
        steps = int(rng.integers(200, 301))
        k0 = float(rng.uniform(20.0, 40.0))
        k1 = float(rng.uniform(120.0, 200.0))
        lr0 = float(rng.uniform(0.008, 0.015))
        lr1 = lr0 * 0.15
        try:
            x_sm = lse_adam(x_try, steps, k0, k1, lr0, lr1)
            v_sm = max_cos(x_sm)
        except Exception:
            continue
        if v_sm > best_val + 0.008:
            continue
        sec = 2.0
        rem = HARD - (time.time() - t0) - 14.0
        if rem < 2.0:
            if v_sm < best_val:
                x_best, best_val = x_sm, v_sm
            break
        sec = min(sec, max(0.8, rem))
        # shorter LP when time is low
        if time.time() - t0 > 100.0:
            sec = min(sec, 1.2)
        try:
            x_pol = lp_polish(x_sm, sec)
            v_pol = max_cos(x_pol)
        except Exception:
            continue
        if v_pol < best_val - 1e-12:
            x_best = x_pol
            best_val = v_pol

    rem = HARD - (time.time() - t0) - 1.0
    if rem > 2.0:
        try:
            x_best = lp_polish(x_best, min(rem, 14.0))
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    x_best = np.nan_to_num(x_best)
    nr = np.linalg.norm(x_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    x_best = x_best / nr
    return x_best
