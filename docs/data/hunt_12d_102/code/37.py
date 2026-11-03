import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def contact_mask_flat(x, mval=None, slack=2e-3):
    g = x @ x.T
    iu = np.triu_indices(x.shape[0], 1)
    c = g[iu]
    if mval is None:
        mval = float(c.max())
    return (c > mval - slack), iu

def adam_quench(x_init, steps, T0, T1, lr):
    x = x_init.copy()
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    lT0 = np.log(T0); lT1 = np.log(T1)
    for s in range(1, steps + 1):
        lt = lT0 + (lT1 - lT0) * (s / steps)
        T = float(np.exp(lt))
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        M = float(G.max())
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
        mcur = float(c.max())
        sel = c > mcur - max(20 * radius, 1e-9)
        if int(sel.sum()) > 2200:
            idx = np.argpartition(-c, 2200)[:2200]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = sel & mask
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
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
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
    dm = d - 1
    B = np.empty((n, d, dm))
    eye = np.eye(d)
    for i in range(n):
        xi = X[i]
        am = int(np.argmax(np.abs(xi)))
        cols = [xi]
        for k in range(d):
            if k != am:
                cols.append(eye[:, k])
        M = np.column_stack(cols)  # d x d
        Q, _ = np.linalg.qr(M)
        # ensure first column aligned with xi (up to sign)
        if float(Q[:, 0] @ xi) < 0:
            Q[:, 0] = -Q[:, 0]
        B[i] = Q[:, 1:]
    return B

def soft_modes(X, sigma=0.007):
    n, d = X.shape
    N = n * d
    dm = d - 1
    Nt = n * dm
    G = X @ X.T
    np.fill_diagonal(G, -10.0)
    M = float(G.max())
    F = np.exp((G - M) / sigma)
    np.fill_diagonal(F, 0.0)
    Fs = F / sigma
    G2 = F / (sigma * sigma)
    lam = np.sum(Fs * G, axis=1)  # lambda_i
    # H1 = kron(Fs0, I)
    H1 = np.kron(Fs, np.eye(d))
    # H2 off-diagonal part
    T4 = np.einsum('ij,ja,ib->iajb', G2, X, X)
    H2 = T4.transpose(0, 1, 2, 3).reshape(N, N)
    D = np.einsum('ik,ka,kb->iab', G2, X, X)
    for i in range(n):
        H2[i*d:(i+1)*d, i*d:(i+1)*d] += D[i]
    HL = H1 + H2
    for i in range(n):
        HL[i*d:(i+1)*d, i*d:(i+1)*d] -= lam[i] * np.eye(d)
    HL = 0.5 * (HL + HL.T)
    B = tangent_bases(X)
    P = np.zeros((N, Nt))
    for i in range(n):
        P[i*d:(i+1)*d, i*dm:(i+1)*dm] = B[i]
    HP = HL @ P
    Ht = P.T @ HP
    Ht = 0.5 * (Ht + Ht.T)
    try:
        w, V = np.linalg.eigh(Ht)
    except Exception:
        w = np.zeros(Nt); V = np.eye(Nt)
    return B, w, V

def apply_soft_perturb(X, B, V, rng, amp, k_modes):
    n, d = X.shape
    dm = d - 1
    Nt = n * dm
    k_modes = int(min(max(k_modes, 4), Nt))
    # lowest k eigenvectors are first columns (eigh ascending)
    coeff = rng.standard_normal(k_modes)
    cn = np.linalg.norm(coeff)
    if cn == 0:
        cn = 1.0
    coeff /= cn
    # total tangent norm = amp*sqrt(n)
    scale = amp * np.sqrt(n)
    v = np.zeros(Nt)
    for k in range(k_modes):
        v += coeff[k] * V[:, k]
    vn = np.linalg.norm(v)
    if vn > 0:
        v = v / vn * scale
    dX = np.empty_like(X)
    for i in range(n):
        seg = v[i*dm:(i+1)*dm]
        dX[i] = B[i] @ seg
    Y = X + dX
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def kmeans_assign(X, k, rng, iters=6):
    n = X.shape[0]
    idx = rng.choice(n, size=k, replace=False)
    C = X[idx].copy()
    assign = np.zeros(n, dtype=int)
    for _ in range(iters):
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        assign = np.argmin(d2, axis=1)
        for c in range(k):
            m = assign == c
            if np.any(m):
                C[c] = X[m].mean(axis=0)
                nn = np.linalg.norm(C[c])
                if nn > 0:
                    C[c] /= nn
            else:
                C[c] = X[rng.integers(0, n)]
    return assign

def rotate_cluster(X, assign, rng, theta):
    Y = X.copy()
    d = X.shape[1]
    uniq = np.unique(assign)
    cl = int(uniq[rng.integers(0, len(uniq))])
    sel = np.where(assign == cl)[0]
    if len(sel) == 0:
        return Y
    a, b = rng.choice(d, size=2, replace=False)
    ca, sa = np.cos(theta), np.sin(theta)
    Q = np.eye(d)
    Q[a, a] = ca; Q[a, b] = -sa
    Q[b, a] = sa; Q[b, b] = ca
    Y[sel] = X[sel] @ Q.T
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def solve():
    t0 = time.time()
    T_LIMIT = 172.0
    t_end = t0 + T_LIMIT
    rng = np.random.default_rng(20261103)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.array(np.load(fn), dtype=float)
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            if a.shape == (102, 12) and np.all(np.isfinite(a)):
                cands.append((max_cos(a), a))
        except Exception:
            pass
    if len(cands) == 0:
        a = rng.standard_normal((102, 12))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        x_best = a; best = max_cos(a)
    else:
        cands.sort(key=lambda t: t[0])
        best, x_best = cands[0][0], cands[0][1].copy()
    n, d = 102, 12
    x_cur = x_best.copy()
    cur = best
    cur_mask, iu_all = contact_mask_flat(x_cur)
    # initial soft modes for incumbent
    try:
        Bcur, wcur, Vcur = soft_modes(x_cur, sigma=0.007)
        have_modes = True
    except Exception:
        have_modes = False
        Bcur = None; Vcur = None
    modes_age = 0
    it = 0
    lp_used = 0
    while time.time() < t_end - 18.0 and it < 300:
        it += 1
        # refresh modes periodically or after accepted move
        if (not have_modes) or (modes_age >= 7):
            if time.time() > t_end - 40.0:
                break
            try:
                Bcur, wcur, Vcur = soft_modes(x_cur, sigma=float(rng.choice([0.005, 0.007, 0.010])))
                have_modes = True
                modes_age = 0
            except Exception:
                have_modes = False
        use_spectral = (it % 2 == 1)
        if use_spectral and have_modes:
            amp = float(rng.uniform(0.02, 0.15))
            k_modes = int(rng.integers(20, 41))
            try:
                x_try = apply_soft_perturb(x_cur, Bcur, Vcur, rng, amp, k_modes)
            except Exception:
                x_try = x_cur + 0.03 * rng.standard_normal((n, d))
                x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            steps = int(rng.integers(380, 650))
            T0 = float(rng.uniform(0.015, 0.05)); T1 = float(rng.uniform(0.002, 0.0045))
            lr = float(rng.uniform(0.0018, 0.003))
        else:
            # rigid cluster rotation
            k = int(rng.integers(4, 7))
            # split by PC1 sign half the time for speed
            if (it % 4 == 0):
                try:
                    _, _, Vt = np.linalg.svd(x_cur - x_cur.mean(axis=0), full_matrices=False)
                    s = (x_cur - x_cur.mean(axis=0)) @ Vt[0]
                    assign = (s > 0).astype(int)
                    if len(np.unique(assign)) < 2:
                        assign = kmeans_assign(x_cur, k, rng, iters=5)
                except Exception:
                    assign = kmeans_assign(x_cur, k, rng, iters=5)
            else:
                assign = kmeans_assign(x_cur, k, rng, iters=5)
            theta = float(rng.uniform(0.03, 0.2)) * (1.0 if rng.random() < 0.5 else -1.0)
            x_try = rotate_cluster(x_cur, assign, rng, theta)
            # small extra jitter to break exact rigidity
            x_try += float(rng.uniform(0.0, 0.008)) * rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            steps = int(rng.integers(380, 700))
            T0 = float(rng.uniform(0.015, 0.045)); T1 = float(rng.uniform(0.002, 0.004))
            lr = float(rng.uniform(0.0018, 0.003))
        x_q = adam_quench(x_try, steps, T0, T1, lr)
        vq = max_cos(x_q)
        modes_age += 1
        if vq < best - 1e-12:
            # snap promising improver with short LP if below 0.269
            if vq < 0.269 and time.time() < t_end - 20.0:
                budget = min(3.0, max(1.0, (t_end - 18.0 - time.time()) / 5.0))
                if budget > 0.5:
                    x_q = lp_polish(x_q, budget, t_end - 16.0)
                    vq = max_cos(x_q)
                    lp_used += 1
            if vq < best:
                x_best, best = x_q.copy(), vq
                x_cur, cur = x_q.copy(), vq
                cur_mask, _ = contact_mask_flat(x_cur)
                modes_age = 99  # force refresh
                continue
        # accept diverse intermediate within 0.002 with large edit distance
        if vq < best + 0.002 and vq < 0.272:
            try:
                mval_q = vq
                Gq = x_q @ x_q.T
                cq = Gq[iu_all]
                mq = (cq > mval_q - 2e-3)
                inter = int(np.logical_and(mq, cur_mask).sum())
                union = int(np.logical_or(mq, cur_mask).sum())
                edit = int(np.logical_xor(mq, cur_mask).sum())
                div = (1.0 - inter / max(union, 1))
            except Exception:
                edit = 0; div = 0.0
            # favour substantial rewiring: edit > 250 or div > 0.25
            if edit > 250 or div > 0.25:
                # optional short LP for very close diverse states
                if vq < 0.269 and vq < best + 0.001 and (it % 5 == 0) and time.time() < t_end - 22.0:
                    budget = 1.0
                    try:
                        x_p = lp_polish(x_q, budget, t_end - 16.0)
                        vp = max_cos(x_p)
                        lp_used += 1
                    except Exception:
                        vp = vq; x_p = x_q
                    if vp < best:
                        x_best, best = x_p.copy(), vp
                        x_cur, cur = x_p.copy(), vp
                        cur_mask, _ = contact_mask_flat(x_cur)
                        modes_age = 99
                        continue
                # random walk to new basin
                if vq < cur or rng.random() < 0.6:
                    x_cur, cur = x_q.copy(), vq
                    cur_mask = mq
                    modes_age = 99 if div > 0.35 else 4
            elif vq < cur and rng.random() < 0.15:
                # small drift improvement of walker
                x_cur, cur = x_q.copy(), vq
                cur_mask = mq
        if time.time() >= t_end - 18.0:
            break
    remain = t_end - time.time() - 1.5
    if remain > 2.0:
        try:
            x_best = lp_polish(x_best, min(remain, 14.0), t_end - 1.0)
        except Exception:
            pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
