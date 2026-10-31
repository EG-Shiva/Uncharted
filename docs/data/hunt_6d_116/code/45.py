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
    return float(np.max(G))

def _adam_quench(X, steps, b0, b1, lr0, lr1):
    X = _norm(X.copy())
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1p = 1.0
    b2p = 1.0
    be1, be2 = 0.9, 0.999
    eps = 1e-8
    for it in range(1, steps + 1):
        f = it / float(steps)
        beta = b0 * (b1 / b0) ** f if b0 > 0 and b1 > 0 else b0
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cmax = float(np.max(G))
        W = np.exp(beta * (G - cmax))
        np.fill_diagonal(W, 0.0)
        S = float(np.sum(W))
        if not np.isfinite(S) or S <= 0:
            break
        Gr = (W @ X) / S
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        b1p *= be1
        b2p *= be2
        m *= be1
        m += (1.0 - be1) * Gt
        v *= be2
        v += (1.0 - be2) * Gt * Gt
        mh = m / (1.0 - b1p)
        vh = v / (1.0 - b2p)
        step = lr * mh / (np.sqrt(vh) + eps)
        # cap max move
        sn = np.max(np.linalg.norm(step, axis=1))
        if sn > 0.08:
            step = step * (0.08 / sn)
        X = X - step
        X = _norm(X)
    return X

def _hinge(X, iters=80, lr=0.012, band=0.003):
    X = _norm(X.copy())
    for k in range(iters):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        Mk = G > (M - band)
        A = Mk @ X
        w = np.sum(Mk * G, axis=1, keepdims=True)
        Gt = A - w * X
        gn = float(np.max(np.linalg.norm(Gt, axis=1)))
        if gn < 1e-14 or not np.isfinite(gn):
            break
        st = lr * (1.0 - 0.5 * k / max(iters, 1))
        mx = 0.03
        if gn * st > mx:
            st = mx / gn
        X = X - st * Gt
        X = _norm(X)
    return X

def _fast_opt(X):
    X = _adam_quench(X, steps=220, b0=25.0, b1=130.0, lr0=0.025, lr1=0.010)
    X = _adam_quench(X, steps=140, b0=130.0, b1=550.0, lr0=0.010, lr1=0.003)
    X = _hinge(X, iters=70, lr=0.012, band=0.003)
    X = _adam_quench(X, steps=60, b0=550.0, b1=900.0, lr0=0.002, lr1=0.001)
    return X

def _hole_fill(fixed, rng, ntry=3000):
    d = fixed.shape[1]
    C = rng.standard_normal((ntry, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = C @ fixed.T
    w = np.max(dots, axis=1)
    y = C[int(np.argmin(w))].copy()
    y /= np.linalg.norm(y)
    step = 0.05
    beta = 60.0
    for _ in range(80):
        dd = fixed @ y
        m = float(np.max(dd))
        Wt = np.exp(beta * (dd - m))
        S = float(np.sum(Wt))
        if not np.isfinite(S) or S <= 0:
            break
        g = (Wt @ fixed) / S
        g = g - float(g @ y) * y
        nn = float(np.linalg.norm(g))
        if nn < 1e-12:
            break
        g /= nn
        # project scale by step
        y = y - step * g * 1.0
        y /= np.linalg.norm(y)
        step *= 0.98
        beta = min(beta * 1.03, 400.0)
    return y / np.linalg.norm(y)

def _subset_rotate(X, k, theta, rng):
    n, d = X.shape
    idx = rng.choice(n, size=k, replace=False)
    a = rng.standard_normal(d)
    a /= np.linalg.norm(a)
    b = rng.standard_normal(d)
    b = b - float(a @ b) * a
    nb = float(np.linalg.norm(b))
    if nb < 1e-12:
        return _norm(X)
    b /= nb
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    R = np.eye(d) + (c - 1.0) * (np.outer(a, a) + np.outer(b, b)) + s * (np.outer(a, b) - np.outer(b, a))
    Y = X.copy()
    Y[idx] = Y[idx] @ R.T
    return _norm(Y)

def _lp_polish(x, deadline):
    x = _norm(x.copy())
    n, d = x.shape
    best = _maxcos(x)
    bx = x.copy()
    radius = 2.0e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(np.max(c))
        thr = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thr)[0]
        if idx.size == 0:
            break
        if idx.size > 1300:
            o = np.argsort(c[idx])[-1300:]
            idx = idx[o]
        I = iu[0][idx]
        J = iu[1][idx]
        cs = c[idx]
        p = len(idx)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _maxcos(y)
        if v < best:
            best = v
            bx = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            if v <= best + 1e-15:
                x = y
            radius *= 0.5
    return bx, best

def solve():
    t0 = time.time()
    HARD_END = t0 + 171.0
    EXPL_END = t0 + 138.0
    rng = np.random.default_rng(12345)
    n, d = 116, 6

    def clean(a):
        try:
            a = np.asarray(a, dtype=float)
        except Exception:
            return None
        if a.shape != (116, 6) or not np.all(np.isfinite(a)):
            return None
        nn = np.linalg.norm(a, axis=1, keepdims=True)
        if np.any(nn < 1e-12):
            return None
        return a / nn

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

    it = 0
    while time.time() < EXPL_END:
        it += 1
        if time.time() > HARD_END - 20.0:
            break
        # pick base: wander from cur, often return to best
        if it % 17 == 0:
            base = best_x
            base_v = best_v
        else:
            if rng.random() < 0.32:
                base = best_x
                base_v = best_v
            else:
                base = cur_x
                base_v = cur_v
        try:
            G0 = base @ base.T
            np.fill_diagonal(G0, -2.0)
            m0 = float(np.max(G0))
            per_max = np.max(G0, axis=1)
            per_deg = np.sum(G0 > (m0 - 0.0012), axis=1).astype(float)
        except Exception:
            continue
        X = base.copy()
        r = rng.random()
        try:
            if r < 0.26:
                sig = float(rng.uniform(0.06, 0.34))
                X = _norm(X + rng.normal(scale=sig, size=(n, d)))
                X = _fast_opt(X)
            elif r < 0.42:
                k = int(rng.integers(15, 34))
                th = float(rng.uniform(0.3, 1.2))
                X = _subset_rotate(X, k, th, rng)
                X = _norm(X + rng.normal(scale=0.012, size=(n, d)))
                X = _fast_opt(X)
            elif r < 0.72:
                k = int(rng.integers(6, 16))
                score = per_max + 0.004 * per_deg + rng.normal(scale=0.0008, size=n)
                kill = np.argsort(-score)[:k]
                order = kill.copy()
                rng.shuffle(order)
                for j in order:
                    jj = int(j)
                    fixed = np.delete(X, jj, axis=0)
                    X[jj] = _hole_fill(fixed, rng, ntry=2500)
                X = _norm(X + rng.normal(scale=0.007, size=(n, d)))
                X = _fast_opt(X)
            else:
                vd = rng.standard_normal(d)
                vd /= np.linalg.norm(vd)
                proj = X @ vd
                k = int(rng.integers(8, 20))
                if rng.random() < 0.5:
                    kill = np.argsort(-proj)[:k]
                else:
                    kill = np.argsort(proj)[:k]
                # bias half to hubs
                if rng.random() < 0.5:
                    hub = np.argsort(-(per_max + 0.004 * per_deg))[:k // 2]
                    kill[:len(hub)] = hub
                rng.shuffle(kill)
                for j in kill:
                    jj = int(j)
                    fixed = np.delete(X, jj, axis=0)
                    X[jj] = _hole_fill(fixed, rng, ntry=2200)
                X = _norm(X + rng.normal(scale=0.009, size=(n, d)))
                X = _fast_opt(X)
        except Exception:
            continue
        try:
            vs = _maxcos(X)
        except Exception:
            continue
        if not np.isfinite(vs):
            continue
        if vs < best_v - 5e-13:
            best_v = vs
            best_x = X.copy()
            cur_x, cur_v = X.copy(), vs
            continue
        frac = (time.time() - t0) / max(EXPL_END - t0, 1.0)
        frac = min(max(frac, 0.0), 1.0)
        Temp = 0.0035 * (0.0002 / 0.0035) ** frac
        if vs < cur_v or rng.random() < np.exp(-(vs - cur_v) / max(Temp, 1e-12)):
            if vs < cur_v + 0.012:
                cur_x, cur_v = X.copy(), vs
        else:
            if cur_v > best_v + 0.008 and (it % 12 == 0):
                cur_x, cur_v = best_x.copy(), best_v

    rem = HARD_END - time.time()
    if rem > 6.0:
        fb = min(rem - 1.5, 27.0)
        try:
            Xf, vf = _lp_polish(best_x, time.time() + fb)
            if vf <= best_v:
                best_x, best_v = Xf, vf
        except Exception:
            pass
    best_x = _norm(best_x)
    return best_x
