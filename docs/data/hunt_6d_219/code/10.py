import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def worst_pair(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    f = int(np.argmax(g))
    n = x.shape[0]
    return f // n, f % n, float(g[f // n, f % n])

def peak_shave(X, iters=500):
    X = X.copy()
    n = X.shape[0]
    for _ in range(iters):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        flat = int(np.argmax(G))
        i = flat // n; j = flat % n
        m = float(G[i, j])
        G[i, j] = -2.0; G[j, i] = -2.0
        m2 = float(G.max())
        d = m - m2
        if d < 1e-12:
            break
        eta = min(d * 0.4, 8e-4)
        ui = X[j] - m * X[i]
        vj = X[i] - m * X[j]
        ni = float(np.linalg.norm(ui)); nj = float(np.linalg.norm(vj))
        if ni < 1e-12 or nj < 1e-12:
            break
        ui /= ni; vj /= nj
        improved = False
        for scale in (1.0, 0.3, 0.1, 0.03):
            e = eta * scale
            Xn = X.copy()
            Xn[i] -= e * ui
            Xn[j] -= e * vj
            Xn[i] /= max(np.linalg.norm(Xn[i]), 1e-12)
            Xn[j] /= max(np.linalg.norm(Xn[j]), 1e-12)
            G2 = Xn @ Xn.T
            np.fill_diagonal(G2, -2.0)
            if float(G2.max()) < m - 1e-12:
                X = Xn
                improved = True
                break
        if not improved:
            break
    return X

def adam_anneal(X0, alpha, steps, lr, cap=0.05):
    X = X0.copy()
    m1 = np.zeros_like(X); m2 = np.zeros_like(X)
    b1, b2 = 0.9, 0.999
    eps = 1e-8
    t = 0
    for _ in range(steps):
        t += 1
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mmax = float(G.max())
        W = np.exp(alpha * (G - mmax))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            break
        grad = (W @ X) / max(S, 1e-300)
        dot = (grad * X).sum(axis=1, keepdims=True)
        gt = grad - dot * X
        m1 = b1 * m1 + (1 - b1) * gt
        m2 = b2 * m2 + (1 - b2) * (gt * gt)
        c1 = 1.0 - b1 ** t; c2 = 1.0 - b2 ** t
        step = (lr * (m1 / c1)) / (np.sqrt(m2 / c2) + eps)
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        fac = np.minimum(1.0, cap / np.maximum(sn, 1e-12))
        X = _norm_rows(X - step * fac)
    return X

def short_anneal(X, fast=True):
    # two-stage sharpening
    if fast:
        X = adam_anneal(X, 150, 250, 0.012)
        X = adam_anneal(X, 500, 250, 0.006)
    else:
        X = adam_anneal(X, 120, 300, 0.015)
        X = adam_anneal(X, 350, 300, 0.008)
        X = adam_anneal(X, 900, 300, 0.004)
    return X

def lp_polish(x, seconds, deadline=1e18):
    x = _norm_rows(x.copy())
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    bestX = x.copy()
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c = Gfull[iu0]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        if int(sel.sum()) < 1:
            break
        I = iu0[0][sel]; J = iu0[1][sel]; cs = c[sel]
        p = len(I)
        if p > 4000:
            k = 4000
            part = np.argpartition(-cs, k - 1)[:k]
            I, J, cs = I[part], J[part], cs[part]
            p = k
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
            obj = np.zeros(n * d + 1); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm_rows(x + res.x[:-1].reshape(n, d))
        v = max_cos(y)
        if v < best:
            x, best, bestX = y, v, y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return bestX

def solve():
    np.random.seed(0)
    rng = np.random.default_rng(0)
    t0 = time.time()
    LIMIT = 178.0
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (219, 6):
                a = _norm_rows(a)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_norm_rows(rng.standard_normal((219, 6))))
    best = None; bestv = 1e9
    for c in cands:
        v = max_cos(c)
        if v < bestv:
            bestv = v; best = c.copy()
    # initial shave + anneal
    try:
        Xs = peak_shave(best, 2000)
        vs = max_cos(Xs)
        if vs < bestv:
            best, bestv = Xs, vs
    except Exception:
        pass
    try:
        Xa = short_anneal(best, fast=False)
        Xa = peak_shave(Xa, 400)
        va = max_cos(Xa)
        if va < bestv:
            best, bestv = Xa, va
    except Exception:
        pass

    n, d = best.shape
    trial = 0
    # focused basin hopping
    while time.time() - t0 < 138.0:
        trial += 1
        remain = LIMIT - (time.time() - t0)
        if remain < 42:
            break
        try:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            order = np.argsort(-per)
            i0 = int(order[0]); j0 = int(np.argmax(G[i0]))
            # neighbourhood of worst pair: top dots to i0 or j0
            sim = np.maximum(G[i0], G[j0])
            # choose kick mode
            mode = trial % 6
            P = best.copy()
            noise = rng.standard_normal(P.shape)
            # project noise to tangent for stability
            noise = noise - (noise * P).sum(axis=1, keepdims=True) * P
            if mode == 0:
                # global small
                P = P + 0.008 * noise
            elif mode == 1:
                # worst-pair hard kick
                s = np.full(n, 0.003)
                s[i0] = 0.06; s[j0] = 0.06
                # neighbours medium
                nb = np.argsort(-sim)[:16]
                s[nb] = np.maximum(s[nb], 0.025)
                P = P + s[:, None] * noise
            elif mode == 2:
                s = np.full(n, 0.002)
                topk = order[:12]
                s[topk] = 0.035
                s[i0] = 0.05; s[j0] = 0.05
                P = P + s[:, None] * noise
            elif mode == 3:
                # reinsert a loose point into worst gap
                loose = int(order[-1])
                mid = best[i0] + best[j0]
                nm = np.linalg.norm(mid)
                if nm > 1e-9:
                    mid /= nm
                    # place loose point near orthogonal of mid to break symmetry
                    P[loose] = _norm_rows((mid + 0.15 * rng.standard_normal(6)[None, :]))[0]
                s = np.full(n, 0.004)
                s[i0] = 0.03; s[j0] = 0.03
                P = P + s[:, None] * noise * 0.5
            elif mode == 4:
                # medium global
                P = P + 0.02 * noise
            else:
                s = np.full(n, 0.005)
                topk = order[:25]
                # random subset of topk gets larger kick
                idx = rng.choice(topk, size=8, replace=False)
                s[idx] = 0.05
                P = P + s[:, None] * noise
            P = _norm_rows(P)
            # anneal: alternate fast/slow by mode
            if mode in (0, 4):
                Q = short_anneal(P, fast=True)
            else:
                # sharper focus on top pairs
                Q = adam_anneal(P, 200, 300, 0.010)
                Q = adam_anneal(Q, 700, 300, 0.005)
            Q = peak_shave(Q, 300)
            vq = max_cos(Q)
            if vq < bestv - 1e-12:
                best, bestv = Q, vq
            # occasional sideways move to escape: if within 2e-6 and random, keep as base for next trial
            # (do not update best, but perturb base via P?) -> implement tempering every 25 trials
            if trial % 25 == 0 and time.time() - t0 < 110:
                # extra long anneal from best with different sharpness
                try:
                    Q2 = adam_anneal(best, 1200, 400, 0.003, cap=0.03)
                    Q2 = peak_shave(Q2, 300)
                    v2 = max_cos(Q2)
                    if v2 < bestv:
                        best, bestv = Q2, v2
                except Exception:
                    pass
        except Exception:
            continue

    remain = LIMIT - (time.time() - t0)
    if remain > 8:
        try:
            Xp = lp_polish(best, min(remain - 2.0, 28.0), deadline=t0 + LIMIT - 1.5)
            vp = max_cos(Xp)
            if vp < bestv:
                best, bestv = Xp, vp
        except Exception:
            pass
    else:
        try:
            best = peak_shave(best, 500)
        except Exception:
            pass
    best = _norm_rows(best)
    return best
