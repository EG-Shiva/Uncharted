import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _rand_rot(rng, d, sigma):
    A = rng.standard_normal((d, d))
    A = A - A.T
    M = np.eye(d) + sigma * A
    Q, R = np.linalg.qr(M)
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    Q = Q * s[None, :]
    return Q

def _riesz_quench(X, s, steps, lr0, lr1, mu):
    X = _norm(X.copy())
    vel = np.zeros_like(X)
    expv = -(s + 2.0) * 0.5
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.clip(G, -1.0, 1.0, out=G)
        R2 = 2.0 - 2.0 * G
        np.fill_diagonal(R2, 1.0)
        np.maximum(R2, 1e-12, out=R2)
        W = R2 ** expv
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1)
        H = S[:, None] * X - W @ X
        proj = (H * X).sum(axis=1, keepdims=True)
        Ht = H - proj * X
        rms = float(np.sqrt(np.mean(np.sum(Ht * Ht, axis=1)))) + 1e-15
        Ht = Ht / rms
        vel = mu * vel + (1.0 - mu) * Ht
        X = X + lr * vel
        X = _norm(X)
    return X

def _softmax_quench(X, steps, T0, T1, lr0, lr1, mu):
    X = _norm(X.copy())
    vel = np.zeros_like(X)
    for it in range(1, steps + 1):
        f = it / float(max(steps, 1))
        T = T0 * (T1 / T0) ** f if T0 > 0 and T1 > 0 else T0
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cmax = float(G.max())
        W = np.exp((G - cmax) / max(T, 1e-12))
        np.fill_diagonal(W, 0.0)
        s = float(W.sum())
        if not np.isfinite(s) or s <= 0:
            break
        W /= s
        Gr = W @ X
        proj = (Gr * X).sum(axis=1, keepdims=True)
        Gt = Gr - proj * X
        rms = float(np.sqrt(np.mean(np.sum(Gt * Gt, axis=1)))) + 1e-15
        Gt = Gt / rms
        vel = mu * vel + (1.0 - mu) * Gt
        X = X - lr * vel
        X = _norm(X)
    return X

def _hinge(X, iters=80, lr=0.02, band=0.004):
    X = _norm(X.copy())
    for _ in range(iters):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(G.max())
        Mk = G > (M - band)
        A = Mk @ X
        w = (Mk * G).sum(axis=1, keepdims=True)
        Gt = A - w * X
        gn = float(np.max(np.linalg.norm(Gt, axis=1)))
        if gn < 1e-14:
            break
        mx = 0.03
        step = lr
        if gn * step > mx:
            step = mx / gn
        X = X - step * Gt
        X = _norm(X)
        lr *= 0.997
    return X

def _lp_polish(X, deadline):
    X = _norm(X.copy())
    n, d = X.shape
    best = _maxcos(X)
    bx = X.copy()
    radius = 2.0e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() > deadline - 0.05:
            break
        G = X @ X.T
        c = G[iu]
        mm = float(c.max())
        thr = mm - max(20.0 * radius, 1e-9)
        idx = np.where(c > thr)[0]
        if idx.size == 0:
            break
        if idx.size > 1200:
            o = np.argsort(c[idx])[-1200:]
            idx = idx[o]
        sel = np.zeros_like(c, dtype=bool)
        sel[idx] = True
        I = iu[0][sel]; J = iu[1][sel]
        cs = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        vv = _maxcos(y)
        if vv < best:
            best = vv; bx = y.copy(); X = y
            radius = min(radius * 1.5, 0.05)
        else:
            if vv <= best + 1e-15:
                X = y
            radius *= 0.5
    return bx, best

def _refine_void(y, others):
    y = y / (np.linalg.norm(y) + 1e-15)
    step = 0.08
    for _ in range(40):
        dd = others @ y
        j = int(np.argmax(dd))
        g = others[j] - float(others[j] @ y) * y
        nn = float(np.linalg.norm(g))
        if nn < 1e-12:
            break
        g /= nn
        y = y - step * g
        y /= (np.linalg.norm(y) + 1e-15)
        step *= 0.97
    return y / (np.linalg.norm(y) + 1e-15)

def _greedy_fill(kept, k, rng, ntry=2000):
    d = kept.shape[1]
    cur = kept.copy()
    for _ in range(k):
        C = rng.standard_normal((ntry, d))
        C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-15)
        if len(cur) > 0:
            dots = C @ cur.T
            w = dots.max(axis=1)
            bi = int(np.argmin(w))
            y = C[bi]
            y = _refine_void(y, cur)
        else:
            y = C[0]
        cur = np.vstack([cur, y[None, :]])
    return cur

def solve():
    t0 = time.time()
    HARD = t0 + 168.0
    EXPL = t0 + 138.0
    rng = np.random.default_rng(1160606)
    def clean(a):
        try:
            a = np.asarray(a, dtype=float)
        except Exception:
            return None
        if a.shape != (116, 6) or not np.all(np.isfinite(a)):
            return None
        n = np.linalg.norm(a, axis=1, keepdims=True)
        if np.any(n < 1e-12):
            return None
        return a / n
    cands = []
    for f in ("best.npy", "record.npy"):
        try:
            c = clean(np.load(f))
            if c is not None:
                cands.append(c)
        except Exception:
            pass
    if not cands:
        cands.append(_norm(rng.standard_normal((116, 6))))
    vals = [_maxcos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_x = cands[bi].copy()
    best_v = float(vals[bi])
    cur_x = best_x.copy()
    cur_v = best_v
    s_choices = [1.0, 2.0, 4.0, 8.0, 16.0]
    it = 0
    temp0, temp1 = 0.0015, 0.00008
    while time.time() < EXPL:
        it += 1
        if time.time() > HARD - 30:
            break
        frac = (time.time() - t0) / max(EXPL - t0, 1e-9)
        frac = min(1.0, max(0.0, frac))
        temp = temp0 + (temp1 - temp0) * frac
        base = best_x.copy() if (it % 10 == 0 and rng.random() < 0.5) else cur_x.copy()
        if it % 25 == 0 and rng.random() < 0.15:
            base = _norm(rng.standard_normal((116, 6)))
        mode = rng.random()
        try:
            if mode < 0.30:
                sigma = float(rng.uniform(0.30, 0.70))
                kick = _norm(base + sigma * rng.standard_normal(base.shape))
            elif mode < 0.55:
                kick = base.copy()
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                mx = G.max(axis=1)
                m = float(G.max())
                deg = (G > m - 0.004).sum(axis=1)
                score = mx + 0.002 * deg
                order = np.argsort(-score)
                K = int(rng.integers(6, 14))
                victims = order[:K]
                mask = np.ones(116, dtype=bool)
                mask[victims] = False
                kept = kick[mask]
                filled = _greedy_fill(kept, K, rng, ntry=1500)
                newpts = filled[len(kept):]
                for ii, vi in enumerate(victims):
                    kick[vi] = newpts[ii]
                kick = kick + float(rng.uniform(0.03, 0.12)) * rng.standard_normal(kick.shape)
                kick = _norm(kick)
            elif mode < 0.73:
                kick = base.copy()
                u = rng.standard_normal(6); u /= (np.linalg.norm(u) + 1e-15)
                dots = base @ u
                K = int(rng.integers(8, 18))
                victims = np.argsort(-dots)[:K]
                mask = np.ones(116, dtype=bool); mask[victims] = False
                kept = kick[mask]
                filled = _greedy_fill(kept, K, rng, ntry=1500)
                newpts = filled[len(kept):]
                for ii, vi in enumerate(victims):
                    kick[vi] = newpts[ii]
                kick = _norm(kick + float(rng.uniform(0.02, 0.10)) * rng.standard_normal(kick.shape))
            elif mode < 0.88:
                kick = base.copy()
                K = int(rng.integers(25, 60))
                idx = rng.choice(116, size=K, replace=False)
                Q = _rand_rot(rng, 6, float(rng.uniform(0.4, 1.2)))
                kick[idx] = kick[idx] @ Q.T
                kick = _norm(kick + float(rng.uniform(0.02, 0.09)) * rng.standard_normal(kick.shape))
            else:
                kick = base.copy()
                K = int(rng.integers(10, 22))
                idx = rng.choice(116, size=K, replace=False)
                kick[idx] = _norm(rng.standard_normal((K, 6)))
        except Exception:
            kick = _norm(base + 0.5 * rng.standard_normal(base.shape))
        try:
            s = float(s_choices[int(rng.integers(0, len(s_choices)))])
            rsteps = int(rng.integers(200, 401))
            mu = float(rng.uniform(0.75, 0.90))
            lr0 = float(rng.uniform(0.04, 0.08))
            kick = _riesz_quench(kick, s, rsteps, lr0, 0.004, mu)
        except Exception:
            pass
        try:
            T0 = float(rng.uniform(0.004, 0.012))
            T1 = float(rng.uniform(0.0003, 0.001))
            if T1 >= T0:
                T1 = T0 * 0.1
            kick = _softmax_quench(kick, 260, T0, T1, 0.05, 0.005, 0.85)
            if rng.random() < 0.5:
                kick = _hinge(kick, iters=60, lr=0.02, band=0.004)
        except Exception:
            pass
        v = _maxcos(kick)
        if v < best_v - 1e-12:
            best_v = v; best_x = kick.copy()
            cur_x = kick.copy(); cur_v = v
        elif v < cur_v or rng.random() < np.exp(-(v - cur_v) / max(temp, 1e-12)):
            if v < cur_v + 0.025:
                cur_x = kick.copy(); cur_v = v
    # intensify around best with small noise and longer quenches
    try:
        while time.time() < HARD - 18:
            cand = _norm(best_x + rng.uniform(0.01, 0.06) * rng.standard_normal(best_x.shape))
            cand = _softmax_quench(cand, 500, 0.006, 0.0004, 0.04, 0.003, 0.88)
            cand = _hinge(cand, iters=80, lr=0.015, band=0.003)
            vv = _maxcos(cand)
            if vv < best_v:
                best_v = vv; best_x = cand.copy()
            if time.time() > HARD - 18:
                break
    except Exception:
        pass
    try:
        remain = HARD - time.time() - 1.0
        if remain > 2.0:
            bx, bv = _lp_polish(best_x, time.time() + min(remain, 14.0))
            if bv < best_v:
                best_x, best_v = bx, bv
    except Exception:
        pass
    return _norm(best_x)
