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
        upd = (m / c1) / (np.sqrt(v / c2) + eps)
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
        cI = (I[:, None] * d + np.arange(d)).ravel()
        cJ = (J[:, None] * d + np.arange(d)).ravel()
        cols = np.concatenate([cI, cJ, np.full(p, n * d)])
        # interleave per row: need repeat structure matching rows
        # build correctly: for each pair k, 2d+1 entries
        cols2 = np.empty(p * (2 * d + 1), dtype=int)
        cols2[0::2 * d + 1] = 0  # placeholder
        # simpler: construct with loops via vectorized index
        # redo straightforward:
        cols_arr = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
        vals_arr = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        # rows already repeat
        try:
            A_ub = coo_matrix((vals_arr, (rows, cols_arr)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
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
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y /= nr
        vv = max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def tangent_bases(X):
    n, d = X.shape
    k = d - 1
    Qall = np.empty((n, d, k))
    for i in range(n):
        xi = X[i]
        B = np.eye(d) - np.outer(xi, xi)
        drop = int(np.argmax(np.abs(xi)))
        idx = [j for j in range(d) if j != drop]
        B2 = B[:, idx]
        Q, _ = np.linalg.qr(B2)
        # qr reduced gives d x k
        if Q.shape[1] != k:
            Q = Q[:, :k]
        Qall[i] = Q
    return Qall


def soft_hessian_eig(X, Th=0.015, K=30, gap=0.12):
    n, d = X.shape
    k = d - 1
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    M = G.max()
    W = np.exp(np.clip((G - M) / Th, -50, 0))
    np.fill_diagonal(W, 0.0)
    Qall = tangent_bases(X)
    Qt = Qall.transpose(0, 2, 1)  # n,k,d
    # A[i,j,p] = Q_i^T x_j
    Atmp = np.matmul(Qt, X.T)  # n,k,n
    A = Atmp.transpose(0, 2, 1)  # n,n,k
    N = n * k
    Hr = np.zeros((N, N))
    invT = 1.0 / Th
    invT2 = invT * invT
    # diagonal blocks
    for i in range(n):
        Ai = A[i]  # n,k
        wi = W[i]  # n
        WA = Ai * wi[:, None]
        Hii = (WA.T @ Ai) * invT2
        # ridge
        Hii += np.eye(k) * 1e-10
        Hr[i * k:(i + 1) * k, i * k:(i + 1) * k] = Hii
    iu = np.triu_indices(n, 1)
    Gv = G[iu]
    keep = Gv > (M - gap)
    Is = iu[0][keep]
    Js = iu[1][keep]
    for ii, jj in zip(Is, Js):
        w = float(W[ii, jj])
        if w < 1e-12:
            continue
        Qi = Qall[ii]  # d,k
        Qj = Qall[jj]
        QTQ = Qi.T @ Qj  # k,k
        a = A[ii, jj]  # k
        c = A[jj, ii]  # k
        Hij = w * invT * QTQ + w * invT2 * np.outer(a, c)
        Hr[ii * k:(ii + 1) * k, jj * k:(jj + 1) * k] = Hij
        Hr[jj * k:(jj + 1) * k, ii * k:(ii + 1) * k] = Hij.T
    try:
        vals, vecs = np.linalg.eigh(Hr)
    except Exception:
        return None, None, Qall
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]
    Vlow = vecs[:, :K]
    return Vlow, vals[:K], Qall


def kmeans_labels(X, k, rng, iters=10):
    n = X.shape[0]
    idx = rng.choice(n, size=k, replace=False)
    centers = X[idx].copy()
    for _ in range(iters):
        sims = X @ centers.T
        labels = np.argmax(sims, axis=1)
        for c in range(k):
            m = labels == c
            if np.any(m):
                cc = X[m].mean(axis=0)
                nr = np.linalg.norm(cc)
                if nr > 1e-12:
                    centers[c] = cc / nr
                else:
                    centers[c] = X[rng.integers(n)]
            else:
                centers[c] = X[rng.integers(n)]
                centers[c] /= np.linalg.norm(centers[c])
    sims = X @ centers.T
    labels = np.argmax(sims, axis=1)
    return labels


def contact_key(X, tol=2e-3):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    M = G.max()
    iu = np.triu_indices(X.shape[0], 1)
    sel = G[iu] > M - tol
    return set(zip(iu[0][sel].tolist(), iu[1][sel].tolist())), M


def solve():
    t0 = time.time()
    T_LIMIT = 172.0
    t_end = t0 + T_LIMIT
    rng = np.random.default_rng(0)
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
    k = d - 1
    x_cur = x_best.copy()
    cur_score = best
    best_key, _ = contact_key(x_best)
    cur_key = set(best_key)

    it = 0
    # main alternating loop, reserve ~14s final polish
    while time.time() < t_end - 14.0 and it < 200:
        it += 1
        spectral = (it % 2 == 1)
        if spectral:
            # build Hessian once, try 3 spectral kicks
            Th = 0.015 if (it % 4 == 1) else 0.010
            try:
                Vlow, _, Qall = soft_hessian_eig(x_cur, Th=Th, K=30, gap=0.12)
            except Exception:
                Vlow, Qall = None, None
            if Vlow is None:
                # fallback jitter
                x_try = x_cur + 0.01 * rng.standard_normal((n, d))
                x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
                x_q = adam_quench(x_try, 400, 0.02, 0.003, 0.002)
                vq = max_cos(x_q)
                if vq < best:
                    x_best, best = x_q, vq
                    x_cur, cur_score = x_q.copy(), vq
                    best_key, _ = contact_key(x_best)
                continue
            n_inner = 3
            for inner in range(n_inner):
                if time.time() > t_end - 14.0:
                    break
                amp = float(rng.uniform(0.02, 0.15))
                Kc = int(rng.integers(20, 31))
                Kc = min(Kc, Vlow.shape[1])
                coeff = rng.standard_normal(Kc)
                y = Vlow[:, :Kc] @ coeff  # 1122
                Ym = y.reshape(n, k)
                rms = float(np.sqrt(np.mean(np.sum(Ym * Ym, axis=1))))
                if rms < 1e-12 or not np.isfinite(rms):
                    continue
                Ym = Ym / rms * amp
                dx = np.einsum('iap,ip->ia', Qall, Ym)
                x_try = x_cur + dx
                x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
                x_q = adam_quench(x_try, 500, 0.02, 0.003, 0.0022)
                vq = max_cos(x_q)
                if vq < best - 1e-12:
                    budget = min(3.0, max(0.8, (t_end - 14.0 - time.time()) / 6.0))
                    if vq < 0.269 and budget > 0.7 and time.time() + budget < t_end - 13.0:
                        x_q = lp_polish(x_q, budget, t_end - 13.0)
                        vq = max_cos(x_q)
                    if vq < best:
                        x_best, best = x_q.copy(), vq
                        x_cur, cur_score = x_q.copy(), vq
                        best_key, _ = contact_key(x_best)
                        cur_key = set(best_key)
                        break
                elif vq < 0.269 and vq < best + 0.002:
                    # favour topology change
                    try:
                        qkey, _ = contact_key(x_q)
                        inter = len(qkey & best_key)
                        union = len(qkey | best_key) + 1e-9
                        jacc = inter / union
                    except Exception:
                        jacc = 1.0
                    if jacc < 0.95:
                        # accept as wanderer
                        x_cur, cur_score, cur_key = x_q.copy(), vq, qkey
                    elif rng.random() < 0.15 and vq < cur_score:
                        x_cur, cur_score = x_q.copy(), vq
                        try:
                            cur_key, _ = contact_key(x_cur)
                        except Exception:
                            pass
                elif vq < cur_score - 5e-5:
                    x_cur, cur_score = x_q.copy(), vq
                    try:
                        cur_key, _ = contact_key(x_cur)
                    except Exception:
                        pass
            # end inner
        else:
            # cluster rotation move
            nclus = int(rng.integers(4, 7))
            try:
                if (it // 2) % 2 == 0:
                    labels = kmeans_labels(x_cur, nclus, rng, iters=8)
                else:
                    # PC1 quantile split
                    U, S, Vt = np.linalg.svd(x_cur, full_matrices=False)
                    pc1 = Vt[0]
                    proj = x_cur @ pc1
                    order = np.argsort(proj)
                    labels = np.empty(n, dtype=int)
                    for r, idxv in enumerate(order):
                        labels[idxv] = (r * nclus) // n
            except Exception:
                labels = rng.integers(0, nclus, size=n)
            c = int(rng.integers(nclus))
            mask = labels == c
            if mask.sum() == 0 or mask.sum() == n:
                continue
            a_, b_ = rng.choice(d, size=2, replace=False)
            theta = float(rng.uniform(0.03, 0.2)) * (1.0 if rng.random() < 0.5 else -1.0)
            co, si = float(np.cos(theta)), float(np.sin(theta))
            x_try = x_cur.copy()
            xa = x_try[mask, a_].copy()
            xb_ = x_try[mask, b_].copy()
            x_try[mask, a_] = co * xa - si * xb_
            x_try[mask, b_] = si * xa + co * xb_
            # tiny jitter to break symmetry
            x_try += 0.002 * rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_q = adam_quench(x_try, 500, 0.02, 0.003, 0.002)
            vq = max_cos(x_q)
            if vq < best - 1e-12:
                budget = min(3.0, max(0.8, (t_end - 14.0 - time.time()) / 6.0))
                if vq < 0.269 and budget > 0.7 and time.time() + budget < t_end - 13.0:
                    x_q = lp_polish(x_q, budget, t_end - 13.0)
                    vq = max_cos(x_q)
                if vq < best:
                    x_best, best = x_q.copy(), vq
                    x_cur, cur_score = x_q.copy(), vq
                    best_key, _ = contact_key(x_best)
                    cur_key = set(best_key)
            elif vq < best + 0.002 and vq < 0.269:
                try:
                    qkey, _ = contact_key(x_q)
                    inter = len(qkey & best_key)
                    union = len(qkey | best_key) + 1e-9
                    jacc = inter / union
                except Exception:
                    jacc = 1.0
                if jacc < 0.93:
                    x_cur, cur_score, cur_key = x_q.copy(), vq, qkey
                elif vq < cur_score and rng.random() < 0.3:
                    x_cur, cur_score = x_q.copy(), vq
            elif vq < cur_score:
                # allow drift of current basin
                if rng.random() < 0.5:
                    x_cur, cur_score = x_q.copy(), vq
    remain = t_end - time.time() - 1.5
    if remain > 2.0:
        x_best = lp_polish(x_best, min(remain, 12.0), t_end - 1.0)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
