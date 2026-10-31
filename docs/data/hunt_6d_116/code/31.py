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

def _riesz_quench(X, s, steps, lr0, lr1, mu, rng, t_end):
    X = _norm(X.copy())
    n, d = X.shape
    vel = np.zeros_like(X)
    expv = -(s + 2.0) * 0.5
    for it in range(steps):
        if time.time() > t_end:
            break
        frac = it / max(steps - 1, 1)
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.clip(G, -1.0, 1.0, out=G)
        R2 = 2.0 - 2.0 * G
        np.fill_diagonal(R2, 1.0)
        np.maximum(R2, 1e-12, out=R2)
        # W = r^{-(s+2)}
        W = R2 ** expv
        np.fill_diagonal(W, 0.0)
        # guard overflow
        if not np.all(np.isfinite(W)):
            W[~np.isfinite(W)] = 1e12
            np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1)
        H = S[:, None] * X - W @ X
        proj = (H * X).sum(axis=1, keepdims=True)
        Ht = H - proj * X
        rms = float(np.sqrt(np.mean(np.sum(Ht * Ht, axis=1)))) + 1e-15
        Ht /= rms
        vel = mu * vel + (1.0 - mu) * Ht
        X = X + lr * vel
        X = _norm(X)
    return X

def _softmax_quench(X, steps, T0, T1, lr0, lr1, mu, t_end):
    X = _norm(X.copy())
    n, d = X.shape
    vel = np.zeros_like(X)
    for it in range(steps):
        if time.time() > t_end:
            break
        frac = it / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** frac if T0 > 0 and T1 > 0 else T0
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            break
        Gr = (E @ X) / S
        proj = (Gr * X).sum(axis=1, keepdims=True)
        Gt = Gr - proj * X
        rms = float(np.sqrt(np.mean(np.sum(Gt * Gt, axis=1)))) + 1e-15
        Gt /= rms
        vel = mu * vel + (1.0 - mu) * Gt
        X = X - lr * vel
        X = _norm(X)
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
        m = float(c.max())
        thr = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thr)[0]
        if idx.size == 0:
            break
        if idx.size > 1300:
            o = np.argsort(c[idx])[-1300:]
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
        v = _maxcos(y)
        if v < best:
            best = v; bx = y.copy(); X = y
            radius = min(radius * 1.5, 0.05)
        else:
            if v <= best + 1e-15:
                X = y
            radius *= 0.5
    return bx, best

def _void_point(others, rng):
    d = others.shape[1]
    C = rng.standard_normal((1500, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = C @ others.T
    w = dots.max(axis=1)
    y = C[int(np.argmin(w))].copy()
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
        y /= np.linalg.norm(y)
        step *= 0.97
    return y / np.linalg.norm(y)

def solve():
    t0 = time.time()
    HARD = t0 + 171.0
    EXPL = t0 + 140.0
    rng = np.random.default_rng(1162026)
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
    # brief initial squeeze
    try:
        if HARD - time.time() > 20:
            bx, bv = _lp_polish(best_x, min(time.time() + 2.5, HARD - 10))
            if bv < best_v:
                best_x, best_v = bx, bv
                cur_x, cur_v = bx.copy(), bv
    except Exception:
        pass
    s_choices = [0.5, 1.0, 2.0, 4.0, 7.0, 12.0, 20.0, 30.0]
    it = 0
    temp0, temp1 = 0.002, 0.00015
    while time.time() < EXPL:
        it += 1
        if time.time() > HARD - 25:
            break
        frac = (time.time() - t0) / max(EXPL - t0, 1e-9)
        frac = min(1.0, max(0.0, frac))
        temp = temp0 + (temp1 - temp0) * frac
        mode = rng.random()
        # base selection
        if it % 10 == 0 and rng.random() < 0.5:
            base = best_x.copy()
        else:
            base = cur_x.copy()
        # large perturbation
        if mode < 0.55:
            sigma = float(rng.uniform(0.35, 0.75))
            kick = base + sigma * rng.standard_normal(base.shape)
            kick = _norm(kick)
        elif mode < 0.8:
            # block surgery: replace K hubs/worst with voids
            kick = base.copy()
            try:
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                mx = G.max(axis=1)
                order = np.argsort(-mx)
                K = int(rng.integers(6, 15))
                victims = order[:K]
                keep_mask = np.ones(116, dtype=bool)
                keep_mask[victims] = False
                others = kick[keep_mask]
                for v_i in victims:
                    y = _void_point(others, rng)
                    kick[v_i] = y
                    others = np.vstack([others, y[None, :]])
                # jitter survivors a bit
                kick = kick + float(rng.uniform(0.05, 0.18)) * rng.standard_normal(kick.shape)
                kick = _norm(kick)
            except Exception:
                kick = _norm(base + 0.5 * rng.standard_normal(base.shape))
        else:
            # fresh random rebirth of subset
            kick = base.copy()
            K = int(rng.integers(10, 25))
            idx = rng.choice(116, size=K, replace=False)
            kick[idx] = _norm(rng.standard_normal((K, 6)))
        s = float(rng.choice(s_choices))
        # time guard for quenches
        q_end = min(time.time() + 2.2, EXPL + 2.0, HARD - 20)
        try:
            rsteps = int(rng.integers(350, 701))
            kick = _riesz_quench(kick, s, rsteps, 0.06, 0.004, 0.85, rng, q_end)
        except Exception:
            pass
        try:
            T0 = float(rng.uniform(0.004, 0.012))
            T1 = float(rng.uniform(0.0004, 0.0012))
            if T1 >= T0:
                T1 = T0 * 0.1
            kick = _softmax_quench(kick, 280, T0, T1, 0.06, 0.006, 0.85, q_end)
        except Exception:
            pass
        v = _maxcos(kick)
        # LP snap if promising and diverse enough
        if v < best_v + 0.004 and time.time() < HARD - 18:
            try:
                tp, vp = _lp_polish(kick, min(time.time() + 1.0, HARD - 15))
                kick, v = tp, vp
            except Exception:
                pass
        if v < best_v - 1e-12:
            best_v = v
            best_x = kick.copy()
            cur_x = kick.copy()
            cur_v = v
        elif v < cur_v or rng.random() < np.exp(-(v - cur_v) / max(temp, 1e-12)):
            if v < cur_v + 0.025:
                cur_x = kick.copy()
                cur_v = v
    # final polish of best
    try:
        remain = HARD - time.time() - 1.0
        if remain > 2.0:
            bx, bv = _lp_polish(best_x, time.time() + min(remain, 18.0))
            if bv < best_v:
                best_x, best_v = bx, bv
    except Exception:
        pass
    return _norm(best_x)
