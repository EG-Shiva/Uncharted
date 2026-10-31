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

def _quench(X, steps, T0, T1, lr0, lr1):
    X = _norm(X)
    n, d = X.shape
    for s in range(steps):
        f = s / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** f if T0 > 0 and T1 > 0 else T0
        lr = lr0 * (1 - f) + lr1 * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        E = np.exp((G - M) / T)
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E))
        if not np.isfinite(S) or S <= 0:
            break
        Grad = (E @ X) / S
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        gn = float(np.max(np.linalg.norm(Gt, axis=1)))
        if gn > 0:
            mx = 0.05
            if gn * lr > mx:
                Gt = Gt * (mx / (gn * lr))
        X = X - lr * Gt
        X = _norm(X)
    return X

def _hinge(X, iters=120, lr=0.015, band=0.004):
    X = _norm(X.copy())
    n = X.shape[0]
    for _ in range(iters):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        thr = M - band
        Mk = G > thr
        # gradient of sum of active dots
        A = Mk @ X
        w = np.sum(Mk * G, axis=1, keepdims=True)
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
        lr *= 0.995
    return X

def _fast_opt(X, steps=320):
    X = _quench(X, steps=steps, T0=0.028, T1=0.0032, lr0=2.0, lr1=0.25)
    X = _hinge(X, iters=100, lr=0.02, band=0.004)
    X = _quench(X, steps=80, T0=0.004, T1=0.0018, lr0=0.4, lr1=0.15)
    return X

def _lp_polish(x, deadline):
    x = _norm(x)
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
        if idx.size > 1400:
            o = np.argsort(c[idx])[-1400:]
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
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
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
            best = v; bx = y.copy(); x = y
            radius = min(radius * 1.5, 0.05)
        else:
            if v <= best + 1e-15:
                x = y
            radius *= 0.5
    return bx, best

def _hole_fill(fixed, rng, ntry=2500):
    d = fixed.shape[1]
    C = rng.standard_normal((ntry, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = C @ fixed.T
    w = np.max(dots, axis=1)
    y = C[int(np.argmin(w))].copy()
    step = 0.06
    for _ in range(70):
        dd = fixed @ y
        j = int(np.argmax(dd))
        g = fixed[j] - float(fixed[j] @ y) * y
        nn = float(np.linalg.norm(g))
        if nn < 1e-12:
            break
        g /= nn
        y = y - step * g
        y /= np.linalg.norm(y)
        step *= 0.97
    return y / np.linalg.norm(y)

def _subset_rotate(X, k, theta, rng):
    n, d = X.shape
    idx = rng.choice(n, size=k, replace=False)
    a = rng.standard_normal(d); a /= np.linalg.norm(a)
    b = rng.standard_normal(d); b = b - float(a @ b) * a
    nb = float(np.linalg.norm(b))
    if nb < 1e-12:
        return _norm(X)
    b /= nb
    c = float(np.cos(theta)); s = float(np.sin(theta))
    R = np.eye(d) + (c - 1.0) * (np.outer(a, a) + np.outer(b, b)) + s * (np.outer(a, b) - np.outer(b, a))
    Y = X.copy()
    Y[idx] = Y[idx] @ R.T
    return _norm(Y)

def solve():
    t0 = time.time()
    HARD_END = t0 + 171.0
    EXPL_END = t0 + 138.0
    rng = np.random.default_rng(7)
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
    n, d = 116, 6
    # initial quick LP squeeze if room
    if HARD_END - time.time() > 12:
        try:
            bx, bv = _lp_polish(best_x, min(time.time() + 3.0, HARD_END - 8.0))
            if bv < best_v:
                best_x, best_v = bx, bv
                cur_x, cur_v = bx.copy(), bv
        except Exception:
            pass
    it = 0
    while time.time() < EXPL_END:
        it += 1
        if time.time() > HARD_END - 18.0:
            break
        if it % 15 == 0:
            base = best_x
            base_v = best_v
        else:
            # mostly wander from cur, sometimes best
            if rng.random() < 0.3:
                base = best_x
                base_v = best_v
            else:
                base = cur_x
                base_v = cur_v
        G0 = base @ base.T
        np.fill_diagonal(G0, -2.0)
        m0 = float(np.max(G0))
        per = np.sum(G0 > (m0 - 5e-4), axis=1)
        mode = it % 6
        X = base.copy()
        try:
            if mode == 0:
                k = int(rng.integers(18, 31))
                th = float(rng.uniform(0.3, 0.75))
                X = _subset_rotate(X, k, th, rng)
                X = _norm(X + rng.normal(scale=0.012, size=(n, d)))
                X = _fast_opt(X, steps=300)
            elif mode == 1:
                k = int(rng.integers(25, 41))
                th = float(rng.uniform(0.6, 1.25))
                X = _subset_rotate(X, k, th, rng)
                X = _norm(X + rng.normal(scale=0.015, size=(n, d)))
                X = _fast_opt(X, steps=340)
            elif mode == 2:
                # hub regrow
                k = int(rng.integers(10, 19))
                kill = np.argsort(-per)[:k]
                # add randomness: replace 20% with random
                if rng.random() < 0.5:
                    kill[0] = int(rng.integers(0, n))
                order = kill.copy()
                rng.shuffle(order)
                for j in order:
                    fixed = np.delete(X, int(j), axis=0)
                    X[int(j)] = _hole_fill(fixed, rng, ntry=2500)
                X = _norm(X + rng.normal(scale=0.008, size=(n, d)))
                X = _fast_opt(X, steps=340)
            elif mode == 3:
                k = int(rng.integers(14, 26))
                kill = rng.choice(n, size=k, replace=False)
                # bias to hubs: mix
                top = np.argsort(-per)[:12]
                nmix = min(k // 2, 12)
                kill[:nmix] = top[rng.choice(12, size=nmix, replace=False)]
                rng.shuffle(kill)
                for j in kill:
                    fixed = np.delete(X, int(j), axis=0)
                    X[int(j)] = _hole_fill(fixed, rng, ntry=2000)
                X = _norm(X + rng.normal(scale=0.008, size=(n, d)))
                X = _fast_opt(X, steps=340)
            elif mode == 4:
                sig = float(rng.uniform(0.18, 0.32))
                X = _norm(X + rng.normal(scale=sig, size=(n, d)))
                X = _fast_opt(X, steps=360)
            else:
                # cap regrow: points most aligned with random dir
                v = rng.standard_normal(d); v /= np.linalg.norm(v)
                proj = X @ v
                k = int(rng.integers(14, 22))
                if rng.random() < 0.5:
                    kill = np.argsort(-proj)[:k]
                else:
                    kill = np.argsort(proj)[:k]
                rng.shuffle(kill)
                for j in kill:
                    fixed = np.delete(X, int(j), axis=0)
                    X[int(j)] = _hole_fill(fixed, rng, ntry=2000)
                X = _norm(X + rng.normal(scale=0.01, size=(n, d)))
                X = _fast_opt(X, steps=340)
        except Exception:
            continue
        try:
            vs = _maxcos(X)
        except Exception:
            continue
        if not np.isfinite(vs):
            continue
        # LP check if promising
        if vs < best_v + 0.0009 and vs < best_v + 0.025:
            # only LP if close to best to save time
            if vs < best_v - 2e-6 or (vs < best_v + 0.0004 and (it % 3 == 0)):
                dl = min(time.time() + 1.4, EXPL_END + 1.0, HARD_END - 12.0)
                if dl > time.time() + 0.2:
                    try:
                        Xp, vp = _lp_polish(X, dl)
                    except Exception:
                        Xp, vp = X, vs
                    if vp < best_v - 5e-13:
                        best_v = vp; best_x = Xp.copy()
                        cur_x, cur_v = Xp.copy(), vp
                        continue
                    # fall through: consider Xp for wandering
                    X, vs = Xp, vp
        if vs < best_v - 5e-13:
            best_v = vs; best_x = X.copy()
            cur_x, cur_v = X.copy(), vs
            continue
        # wandering Metropolis on cur
        frac = (time.time() - t0) / max(EXPL_END - t0, 1.0)
        frac = min(max(frac, 0.0), 1.0)
        Temp = 0.004 * (0.00025 / 0.004) ** frac
        if vs < cur_v or rng.random() < np.exp(-(vs - cur_v) / max(Temp, 1e-12)):
            if vs < cur_v + 0.012:
                cur_x, cur_v = X.copy(), vs
        else:
            if cur_v > best_v + 0.008 and (it % 10 == 0):
                cur_x, cur_v = best_x.copy(), best_v
    rem = HARD_END - time.time()
    if rem > 5.0:
        fb = min(rem - 1.5, 28.0)
        try:
            Xf, vf = _lp_polish(best_x, time.time() + fb)
            if vf <= best_v:
                best_x, best_v = Xf, vf
        except Exception:
            pass
    best_x = _norm(best_x)
    return best_x
