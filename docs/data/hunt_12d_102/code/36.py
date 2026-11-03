import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def adam_quench(x_init, steps, T0, T1, lr):
    x = x_init.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lT0 = np.log(T0)
    lT1 = np.log(T1)
    for s in range(1, steps + 1):
        lt = lT0 + (lT1 - lT0) * (s / steps)
        T = float(np.exp(lt))
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        M = G.max()
        W = np.exp((G - M) / T)
        np.fill_diagonal(W, 0.0)
        Grad = (W @ x) / T
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        m = b1 * m + (1.0 - b1) * Gt
        v = b2 * v + (1.0 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** s
        c2 = 1.0 - b2 ** s
        mh = m / c1
        vh = v / c2
        upd = mh / (np.sqrt(vh) + eps)
        x = x - lr * upd
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x


def lp_polish(x, seconds, t_end):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end:
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        mcur = c.max()
        sel = c > mcur - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y /= nr
        vv = max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def contact_set(X, delta=2e-3):
    G = X @ X.T
    iu = np.triu_indices(X.shape[0], 1)
    c = G[iu]
    m = c.max()
    sel = c > m - delta
    return set((iu[0][sel] * 102 + iu[1][sel]).tolist()), m


def edit_dist(s1, s2):
    if not s1 and not s2:
        return 0.0
    inter = len(s1 & s2)
    union = len(s1 | s2)
    if union == 0:
        return 0.0
    return 1.0 - inter / union


def soft_modes(X, rng, thresh=2e-3, sigma=5e-4, K=30, skip=168):
    n, d = X.shape
    ND = n * d
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    M = G.max()
    iu = np.triu_indices(n, 1)
    c = G[iu]
    sel = c > M - thresh
    I = iu[0][sel]
    J = iu[1][sel]
    cv = c[sel]
    # cap size for speed
    if len(I) > 2000:
        # keep tightest 2000
        o = np.argsort(-cv)[:2000]
        I, J, cv = I[o], J[o], cv[o]
    W = np.exp((cv - M) / sigma)
    H = np.zeros((ND, ND), dtype=float)
    for k in range(len(I)):
        i = int(I[k]); j = int(J[k]); w = float(W[k])
        xi = X[i]; xj = X[j]
        ii = slice(i * d, (i + 1) * d)
        jj = slice(j * d, (j + 1) * d)
        H[ii, ii] += w * np.outer(xj, xj)
        H[jj, jj] += w * np.outer(xi, xi)
        oij = w * np.outer(xj, xi)
        H[ii, jj] += oij
        H[jj, ii] += oij.T
    # tangent projector
    P = np.eye(ND)
    for i in range(n):
        xi = X[i]
        ii = slice(i * d, (i + 1) * d)
        P[ii, ii] -= np.outer(xi, xi)
    Hp = P @ H @ P
    Hp = 0.5 * (Hp + Hp.T)
    try:
        ev, U = np.linalg.eigh(Hp)
    except Exception:
        return None
    if U.shape[1] < skip + K:
        return None
    return U[:, skip:skip + K].copy()


def kmeans_labels(X, k, rng, iters=12):
    n = X.shape[0]
    idx = rng.choice(n, size=k, replace=False)
    C = X[idx].copy()
    lab = np.zeros(n, dtype=int)
    for _ in range(iters):
        D = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        lab = np.argmin(D, axis=1)
        for c in range(k):
            m = lab == c
            if np.any(m):
                C[c] = X[m].mean(axis=0)
            else:
                C[c] = X[int(rng.integers(n))]
    return lab


def solve():
    t0 = time.time()
    T_LIMIT = 172.0
    t_end = t0 + T_LIMIT
    rng = np.random.default_rng(12345)
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
                a = np.array(c, dtype=float)
                a /= np.linalg.norm(a, axis=1, keepdims=True)
                cands.append((max_cos(a), a))
            except Exception:
                pass
    if len(cands) == 0:
        a = rng.standard_normal((102, 12))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        x_best = a
        best = max_cos(a)
    else:
        cands.sort(key=lambda t: t[0])
        best, x_best = cands[0][0], cands[0][1].copy()
    n, d = 102, 12
    x_cur = x_best.copy()
    cur = best
    cs_best, _ = contact_set(x_best)
    cs_cur = set(cs_best)
    lp_calls = 0

    outer = 0
    # reserve final polish
    while time.time() < t_end - 18.0 and outer < 30:
        outer += 1
        # 1) spectral modes from current (once per cycle)
        modes = None
        try:
            if time.time() < t_end - 20.0:
                modes = soft_modes(x_cur, rng, thresh=2e-3, sigma=5e-4, K=30, skip=168)
        except Exception:
            modes = None
        # spectral trials
        n_spec = 7 if modes is not None else 0
        for s in range(n_spec):
            if time.time() >= t_end - 19.0:
                break
            K = modes.shape[1]
            kuse = int(rng.integers(20, min(41, K + 1)))
            # random subset of modes
            perm = rng.permutation(K)[:kuse]
            coeff = rng.standard_normal(kuse)
            nn = np.linalg.norm(coeff)
            if nn == 0:
                continue
            coeff /= nn
            amp = float(rng.uniform(0.02, 0.15))
            delta = (modes[:, perm] @ coeff) * amp
            x_try = x_cur + delta.reshape(n, d)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_q = adam_quench(x_try, 600, 0.02, 0.003, 0.0022)
            vq = max_cos(x_q)
            if vq < 0.269 + 1e-12:
                cs_q, _ = contact_set(x_q)
                ed = edit_dist(cs_cur, cs_q)
            else:
                ed = 0.0
            if vq < best - 1e-12:
                if vq < 0.269 and lp_calls < 12 and time.time() < t_end - 22.0:
                    b = min(2.0, max(0.8, (t_end - 18.0 - time.time()) / 6.0))
                    x_p = lp_polish(x_q, b, t_end - 18.0)
                    lp_calls += 1
                    vp = max_cos(x_p)
                    if vp < vq:
                        x_q, vq = x_p, vp
                        cs_q, _ = contact_set(x_q)
                        ed = edit_dist(cs_cur, cs_q)
                x_best, best = x_q.copy(), vq
                cs_best = set(cs_q) if 'cs_q' in locals() else contact_set(x_best)[0]
                x_cur, cur, cs_cur = x_q.copy(), vq, set(cs_best)
            elif vq < best + 0.002 and ed > 0.25:
                # diverse intermediate: wander
                if vq < cur or rng.random() < 0.4:
                    x_cur, cur, cs_cur = x_q.copy(), vq, set(cs_q)
            elif vq < cur - 1e-12:
                x_cur, cur = x_q.copy(), vq
                try:
                    cs_cur, _ = contact_set(x_q)
                except Exception:
                    pass
        # 2) cluster rotation trials
        for s in range(8):
            if time.time() >= t_end - 19.0:
                break
            k = int(rng.integers(4, 7))
            try:
                lab = kmeans_labels(x_cur, k, rng, iters=10)
            except Exception:
                continue
            # sometimes split by PC1 sign instead
            if rng.random() < 0.3:
                try:
                    _, _, Vt = np.linalg.svd(x_cur - x_cur.mean(axis=0), full_matrices=False)
                    scores = x_cur @ Vt[0]
                    lab = (scores > 0).astype(int)
                    k = 2
                except Exception:
                    pass
            cl = int(rng.integers(k))
            idx = np.where(lab == cl)[0]
            if len(idx) == 0 or len(idx) == n:
                continue
            a_, b_ = rng.choice(d, size=2, replace=False)
            th = float(rng.uniform(0.03, 0.2)) * (1.0 if rng.random() < 0.5 else -1.0)
            co, si = np.cos(th), np.sin(th)
            x_try = x_cur.copy()
            xa = x_try[idx, a_].copy()
            xb_ = x_try[idx, b_].copy()
            x_try[idx, a_] = co * xa - si * xb_
            x_try[idx, b_] = si * xa + co * xb_
            # tiny jitter to break symmetry
            x_try += 0.002 * rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_q = adam_quench(x_try, 650, 0.022, 0.003, 0.0022)
            vq = max_cos(x_q)
            do_lp = False
            if vq < best - 1e-12:
                do_lp = True
            elif vq < best + 0.0004 and (s % 3 == 0) and vq < 0.269:
                do_lp = True
            if do_lp and lp_calls < 14 and time.time() < t_end - 20.0:
                b = 1.2 if vq >= best else 2.0
                b = min(b, max(0.6, t_end - 18.0 - time.time()))
                if b > 0.55:
                    x_p = lp_polish(x_q, b, t_end - 18.0)
                    lp_calls += 1
                    vp = max_cos(x_p)
                    if vp < vq:
                        x_q, vq = x_p, vp
            if vq < best:
                x_best, best = x_q.copy(), vq
                try:
                    cs_best, _ = contact_set(x_best)
                except Exception:
                    pass
                x_cur, cur, cs_cur = x_q.copy(), vq, set(cs_best)
            else:
                if vq < best + 0.002:
                    try:
                        cs_q, _ = contact_set(x_q)
                        ed = edit_dist(cs_cur, cs_q)
                    except Exception:
                        ed = 0.0
                    if ed > 0.3 and (vq < cur or rng.random() < 0.35):
                        x_cur, cur, cs_cur = x_q.copy(), vq, set(cs_q)
                elif vq < cur:
                    x_cur, cur = x_q.copy(), vq
                    try:
                        cs_cur, _ = contact_set(x_q)
                    except Exception:
                        pass
        # occasional re-seed cur to best if cur drifted too high
        if cur > best + 0.0025:
            x_cur, cur = x_best.copy(), best
            cs_cur = set(cs_best)
    remain = t_end - time.time() - 1.5
    if remain > 2.0:
        x_best = lp_polish(x_best, min(remain, 14.0), t_end - 1.0)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
