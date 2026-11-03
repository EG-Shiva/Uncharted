import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n < 1e-16] = 1.0
    return X / n

def _adam_lse(X0, stages, deadline=1e18):
    X = _norm(X0.copy())
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    b1p = 1.0
    b2p = 1.0
    best = X.copy()
    try:
        bestv = _max_cos(X)
    except Exception:
        bestv = 1e9
    for beta, lr, steps in stages:
        for _ in range(steps):
            if (t & 127) == 0 and time.time() > deadline:
                return best, bestv
            t += 1
            b1p *= b1
            b2p *= b2
            G = X @ X.T
            np.fill_diagonal(G, -10.0)
            mx = float(np.max(G))
            E = np.exp(beta * (G - mx))
            np.fill_diagonal(E, 0.0)
            S = float(np.sum(E))
            if not np.isfinite(S) or S <= 0:
                continue
            Grad = (E @ X) / S
            dot = np.sum(Grad * X, axis=1, keepdims=True)
            Gt = Grad - dot * X
            m1 = b1 * m1 + (1.0 - b1) * Gt
            m2 = b2 * m2 + (1.0 - b2) * (Gt * Gt)
            mhat = m1 / (1.0 - b1p)
            vhat = m2 / (1.0 - b2p)
            upd = mhat / (np.sqrt(vhat) + eps)
            dn = np.sum(upd * X, axis=1, keepdims=True)
            upd = upd - dn * X
            X = X - lr * upd
            X = _norm(X)
            if (t % 150) == 0:
                v = _max_cos(X)
                if v < bestv:
                    bestv = v
                    best = X.copy()
    v = _max_cos(X)
    if v < bestv:
        bestv = v
        best = X.copy()
    return best, bestv

def _greedy(X0, iters, step0=2e-4):
    X = _norm(X0.copy())
    try:
        best = _max_cos(X)
    except Exception:
        return X, 1e9
    bestX = X.copy()
    step = step0
    n = X.shape[0]
    for _ in range(iters):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        k = int(np.argmax(G))
        i = k // n
        j = k % n
        dot = float(G[i, j])
        vi = dot * X[i] - X[j]
        vj = dot * X[j] - X[i]
        ni = float(np.linalg.norm(vi))
        nj = float(np.linalg.norm(vj))
        if ni < 1e-15 or nj < 1e-15:
            step *= 0.5
            if step < 1e-14:
                step = step0 * 0.5
            continue
        vi /= ni
        vj /= nj
        # symmetric
        Y = X.copy()
        Y[i] = X[i] + step * vi
        Y[j] = X[j] + step * vj
        Y[i] /= np.linalg.norm(Y[i])
        Y[j] /= np.linalg.norm(Y[j])
        v = _max_cos(Y)
        if v < best:
            X = Y
            best = v
            bestX = Y.copy()
            step = min(step * 1.2, 4e-3)
            continue
        Y2 = X.copy()
        Y2[i] = X[i] + step * vi
        Y2[i] /= np.linalg.norm(Y2[i])
        v2 = _max_cos(Y2)
        Y3 = X.copy()
        Y3[j] = X[j] + step * vj
        Y3[j] /= np.linalg.norm(Y3[j])
        v3 = _max_cos(Y3)
        if v2 < best and v2 <= v3:
            X = Y2
            best = v2
            bestX = Y2.copy()
            step = min(step * 1.12, 4e-3)
        elif v3 < best:
            X = Y3
            best = v3
            bestX = Y3.copy()
            step = min(step * 1.12, 4e-3)
        else:
            step *= 0.55
            if step < 1e-13:
                step = step0 * 0.4
    return bestX, best

def _single_refine(y0, Others, beta=450.0, iters=300, lr=0.03):
    y = np.asarray(y0, dtype=float).copy()
    nn = float(np.linalg.norm(y))
    if not np.finite(nn) or nn < 1e-16:
        y = Others[0].copy()
        nn = float(np.linalg.norm(y))
    y = y / nn
    m1 = np.zeros_like(y)
    m2 = np.zeros_like(y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best_y = y.copy()
    dots0 = Others @ y
    best_s = float(np.max(dots0))
    for t in range(1, iters + 1):
        dots = Others @ y
        m0 = float(np.max(dots))
        W = np.exp(beta * (dots - m0))
        S = float(np.sum(W))
        if not np.isfinite(S) or S <= 0:
            break
        grad = (W @ Others) / S
        gt = grad - float(grad @ y) * y
        m1 = b1 * m1 + (1.0 - b1) * gt
        m2 = b2 * m2 + (1.0 - b2) * (gt * gt)
        c1 = 1.0 - b1 ** t
        c2 = 1.0 - b2 ** t
        st = (m1 / c1) / (np.sqrt(m2 / c2) + eps)
        st = st - float(st @ y) * y
        lrt = lr / (1.0 + t / 250.0)
        y = y - lrt * st
        nn = float(np.linalg.norm(y))
        if not np.finite(nn) or nn < 1e-16:
            break
        y = y / nn
        if (t % 25) == 0 or t == iters:
            s = float(np.max(Others @ y))
            if s < best_s:
                best_s = s
                best_y = y.copy()
    s = float(np.max(Others @ best_y))
    # also check last y
    s2 = float(np.max(Others @ y))
    if s2 < s:
        return y, s2
    return best_y, s

def _reinsert_once(X, bestv, rng, n_cand=4000, n_worst=6):
    n, d = X.shape
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    rowmax = np.max(G, axis=1)
    order = np.argsort(-rowmax)[:n_worst]
    R = rng.standard_normal(size=(n_cand, d))
    R = _norm(R)
    for idx in order:
        idx = int(idx)
        Others = np.delete(X, idx, axis=0)
        sc = R @ Others.T
        smax = np.max(sc, axis=1)
        o = np.argsort(smax)
        for kk in range(min(3, n_cand)):
            y0 = R[o[kk]]
            y, _ = _single_refine(y0, Others, beta=450.0, iters=280, lr=0.04)
            Y = X.copy()
            Y[idx] = y
            v = _max_cos(Y)
            if v < bestv - 1e-12:
                return Y, v
        # also try heavy refine of current point
        y0 = X[idx].copy()
        y, _ = _single_refine(y0, Others, beta=550.0, iters=280, lr=0.02)
        Y = X.copy()
        Y[idx] = y
        v = _max_cos(Y)
        if v < bestv - 1e-12:
            return Y, v
    return None, bestv

def _lp_polish(X0, seconds, deadline):
    X = _norm(X0.copy())
    best = _max_cos(X)
    bestX = X.copy()
    n, d = X.shape
    radius = 8e-4
    iu = np.triu_indices(n, 1)
    start = time.time()
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-14:
            break
        G = X @ X.T
        c = G[iu]
        m = float(np.max(c))
        thr = m - max(25.0 * radius, 5e-7)
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 5000:
            idxp = np.argpartition(-c, 5000)[:5000]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idxp] = True
            sel = sel & mask
            cnt = 5000
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        rows = np.repeat(np.arange(p), 2 * d + 1)
        ci = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, ci)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(n * d + 1)
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
        dx = res.x[:-1].reshape(n, d)
        improved = False
        for alpha in (1.0, 0.5, 0.25, 0.125):
            Y = X + alpha * dx
            Y = _norm(Y)
            v = _max_cos(Y)
            if v < best - 1e-12:
                best = v
                bestX = Y.copy()
                X = Y
                radius = min(radius * 1.6, 0.02)
                improved = True
                break
        if not improved:
            radius *= 0.5
    return bestX, best

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.default_rng(7)
    np.random.seed(7)
    Xbest = None
    bestv = 1e9
    for fname in ("best.npy", "record.npy"):
        try:
            Y = np.load(fname)
            Y = np.asarray(Y, dtype=float)
            if Y.shape != (182, 9):
                continue
            Y = _norm(Y)
            if np.any(~np.isfinite(Y)):
                continue
            v = _max_cos(Y)
            if v < bestv:
                bestv = v
                Xbest = Y.copy()
        except Exception:
            continue
    if Xbest is None:
        Xbest = _norm(rng.standard_normal(size=(182, 9)))
        bestv = _max_cos(Xbest)
    # quick squeeze
    try:
        Xg, vg = _greedy(Xbest, iters=120, step0=2e-4)
        if vg < bestv:
            Xbest, bestv = Xg, vg
    except Exception:
        pass
    cfgs = [
        [(60, 0.006, 400), (150, 0.003, 400), (320, 0.0015, 500), (620, 0.0008, 500)],
        [(40, 0.007, 450), (120, 0.004, 450), (280, 0.002, 450), (550, 0.001, 450)],
        [(100, 0.004, 350), (250, 0.002, 450), (520, 0.001, 550)],
    ]
    sigmas = [0.006, 0.014, 0.028, 0.008, 0.045, 0.011, 0.02, 0.005]
    trial = 0
    while time.time() < deadline - 52.0:
        trial += 1
        if time.time() >= deadline - 52.0:
            break
        try:
            if trial % 6 == 0:
                Y, v = _reinsert_once(Xbest, bestv, rng, n_cand=3500, n_worst=8)
                if Y is not None and v < bestv:
                    # polish reinserted point a bit with greedy
                    Yg, vg = _greedy(Y, iters=80, step0=1.5e-4)
                    if vg < v:
                        Y, v = Yg, vg
                    Xbest, bestv = Y, v
                continue
            sigma = sigmas[(trial - 1) % len(sigmas)]
            cfg = cfgs[(trial - 1) % len(cfgs)]
            if trial % 4 == 0:
                # targeted shake of worst points
                G = Xbest @ Xbest.T
                np.fill_diagonal(G, -2.0)
                rowmax = np.max(G, axis=1)
                worst = np.argsort(-rowmax)[:10]
                pert = rng.standard_normal(size=Xbest.shape) * sigma * 0.4
                pert[worst] += rng.standard_normal(size=(10, 9)) * 0.06
                X0 = _norm(Xbest + pert)
            else:
                X0 = _norm(Xbest + sigma * rng.standard_normal(size=Xbest.shape))
            X1, v1 = _adam_lse(X0, cfg, deadline=deadline - 48.0)
            if v1 < bestv - 1e-12:
                Xbest, bestv = X1, v1
                continue
            # shallow greedy to gauge basin depth (cheap)
            X1g, v1g = _greedy(X1, iters=60, step0=1.5e-4)
            if v1g < bestv - 1e-12:
                Xbest, bestv = X1g, v1g
            if trial > 40:
                break
        except Exception:
            continue
    # final jamming
    if time.time() < deadline - 3.0:
        rem = deadline - time.time() - 1.5
        if rem > 8.0:
            lp_t = min(rem * 0.65, 45.0)
            try:
                Xlp, vlp = _lp_polish(Xbest, seconds=lp_t, deadline=time.time() + lp_t + 1.0)
                if vlp < bestv:
                    Xbest, bestv = Xlp, vlp
            except Exception:
                pass
    # single-point sweep over worst points
    if time.time() < deadline - 6.0:
        try:
            for _ in range(2):
                if time.time() > deadline - 4.0:
                    break
                G = Xbest @ Xbest.T
                np.fill_diagonal(G, -2.0)
                rowmax = np.max(G, axis=1)
                order = np.argsort(-rowmax)[:12]
                for idx in order:
                    if time.time() > deadline - 4.0:
                        break
                    Others = np.delete(Xbest, int(idx), axis=0)
                    y, _ = _single_refine(Xbest[int(idx)], Others, beta=550.0, iters=250, lr=0.02)
                    Y = Xbest.copy()
                    Y[int(idx)] = y
                    v = _max_cos(Y)
                    if v < bestv - 1e-12:
                        Xbest, bestv = Y, v
        except Exception:
            pass
    if time.time() < deadline - 1.0:
        try:
            Xg, vg = _greedy(Xbest, iters=150, step0=1e-4)
            if vg < bestv:
                Xbest, bestv = Xg, vg
        except Exception:
            pass
    Xbest = _norm(Xbest)
    return Xbest
