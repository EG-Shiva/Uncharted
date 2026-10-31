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
    X = _norm(X.copy())
    n = X.shape[0]
    for s in range(steps):
        f = s / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** f if T0 > 0 and T1 > 0 else T0
        lr = lr0 * (1.0 - f) + lr1 * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        E = np.exp((G - M) / max(T, 1e-12))
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E))
        if not np.isfinite(S) or S <= 0:
            break
        Gr = (E @ X) / S
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        gn = float(np.max(np.linalg.norm(Gt, axis=1)))
        if gn > 0:
            mx = 0.05
            if gn * lr > mx:
                Gt = Gt * (mx / (gn * lr))
        X = X - lr * Gt
        X = _norm(X)
    return X

def _hinge(X, iters=60, lr=0.02, band=0.004):
    X = _norm(X.copy())
    n = X.shape[0]
    for _ in range(iters):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        Mk = G > (M - band)
        A = Mk @ X
        w = np.sum(Mk * G, axis=1, keepdims=True)
        Gt = A - w * X
        gn = float(np.max(np.linalg.norm(Gt, axis=1)))
        if gn < 1e-14:
            break
        mx = 0.03
        st = lr
        if gn * st > mx:
            st = mx / gn
        X = X - st * Gt
        X = _norm(X)
        lr *= 0.997
    return X

def _fast_jam(X, steps=350):
    X = _quench(X, steps=steps, T0=0.022, T1=0.0028, lr0=1.6, lr1=0.25)
    X = _hinge(X, iters=60, lr=0.02, band=0.004)
    X = _quench(X, steps=70, T0=0.0035, T1=0.0015, lr0=0.35, lr1=0.12)
    return X

def _short_quench(X, steps=130):
    return _quench(X, steps=steps, T0=0.008, T1=0.0018, lr0=0.6, lr1=0.15)

def _lp_polish(X, deadline):
    X = _norm(X.copy())
    n, d = X.shape
    best = _maxcos(X)
    bx = X.copy()
    radius = 2.0e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() > deadline - 0.3:
            break
        G = X @ X.T
        c = G[iu]
        m = float(np.max(c))
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

def _deep_hole(fixed, rng, ntry=2000, refine=40):
    d = fixed.shape[1]
    C = rng.standard_normal((ntry, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = C @ fixed.T
    w = np.max(dots, axis=1)
    y = C[int(np.argmin(w))].copy()
    step = 0.05
    for _ in range(refine):
        dd = fixed @ y
        j = int(np.argmax(dd))
        g = fixed[j] - float(fixed[j] @ y) * y
        nn = float(np.linalg.norm(g))
        if nn < 1e-12:
            break
        g /= nn
        y = y - step * g
        ny = float(np.linalg.norm(y))
        if ny < 1e-12:
            break
        y /= ny
        step *= 0.97
    return y / np.linalg.norm(y)

def solve():
    t0 = time.time()
    HARD = t0 + 171.0
    FINAL = HARD - 20.0
    rng = np.random.default_rng(20261031)

    def _clean(a):
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

    best_X = None
    best_val = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            a = np.load(f)
            c = _clean(a)
            if c is not None:
                v = _maxcos(c)
                if v < best_val:
                    best_val = v; best_X = c.copy()
        except Exception:
            continue
    if best_X is None:
        best_X = _norm(rng.standard_normal((116, 6)))
        best_val = _maxcos(best_X)
    incumbent_X = best_X.copy()
    incumbent_val = best_val

    specs = [("high", 16), ("rand", 14), ("cap", 16), ("rand", 18), ("high", 12), ("rand", 20), ("cap", 14), ("high", 20)]
    si = 0
    while time.time() < FINAL - 5.0 and si < len(specs):
        kind, k = specs[si]
        si += 1
        if time.time() > FINAL - 4.0:
            break
        variant_deadline = min(FINAL, time.time() + 32.0)
        # compute deletion set from current incumbent
        try:
            X0 = incumbent_X.copy()
            n0 = X0.shape[0]
            G = X0 @ X0.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            if kind == "high":
                cnt = np.sum(G > (m - 5e-4), axis=1)
                order = np.argsort(-cnt, kind="stable")
                # tie-break by sum of near dots
                del_idx = np.sort(order[:k])
            elif kind == "cap":
                # cap around most loaded point
                cnt = np.sum(G > (m - 5e-4), axis=1)
                center = int(np.argmax(cnt))
                dots = X0 @ X0[center]
                order = np.argsort(-dots, kind="stable")
                del_idx = np.sort(order[:k])
            else:
                rs = np.random.default_rng(1000 + si * 77 + int(incumbent_val * 1e9) % 1009)
                del_idx = np.sort(rs.choice(n0, size=k, replace=False))
            keep = np.ones(n0, dtype=bool)
            keep[del_idx] = False
            Xs = X0[keep]
        except Exception:
            continue
        # re-jam smaller code, strict cap
        try:
            if time.time() > variant_deadline - 2.0:
                continue
            Xs = _fast_jam(Xs, steps=320)
            if time.time() > variant_deadline - 1.0:
                continue
            # quick LP on small code if time allows (1s)
            if variant_deadline - time.time() > 8.0:
                Xs, _ = _lp_polish(Xs, min(time.time() + 1.0, variant_deadline - 1.0))
        except Exception:
            continue
        # regrow one at a time
        cur = Xs
        ok = True
        n_ins = k
        for ii in range(n_ins):
            if time.time() > variant_deadline - 1.0:
                ok = False
                break
            try:
                y = _deep_hole(cur, rng, ntry=2000, refine=35)
                cur = np.vstack([cur, y.reshape(1, -1)])
                cur = _short_quench(cur, steps=120)
                # LP every 4 insertions, 1-2 s
                if (ii + 1) % 4 == 0 and (ii + 1) < n_ins:
                    if variant_deadline - time.time() > 4.0:
                        cur, _ = _lp_polish(cur, min(time.time() + 1.5, variant_deadline - 0.5))
                elif (ii + 1) == n_ins:
                    pass
            except Exception:
                ok = False
                break
        if not ok or cur.shape[0] != 116:
            continue
        try:
            v = _maxcos(cur)
            # short jam
            if time.time() < variant_deadline - 1.0:
                cur2 = _short_quench(cur, steps=80)
                v2 = _maxcos(cur2)
                if v2 < v:
                    cur, v = cur2, v2
            if v < incumbent_val:
                # verify then accept; quick LP to lock
                if variant_deadline - time.time() > 3.0:
                    curp, vp = _lp_polish(cur, min(time.time() + 1.5, variant_deadline))
                    if vp <= v + 1e-12:
                        cur, v = curp, vp
                if v < incumbent_val:
                    incumbent_val = v; incumbent_X = _norm(cur.copy())
            else:
                # even if not better, try one LP polish to find better basin if cheap
                if variant_deadline - time.time() > 6.0 and v < incumbent_val + 0.002:
                    curp, vp = _lp_polish(cur, min(time.time() + 1.2, variant_deadline))
                    if vp < incumbent_val:
                        incumbent_val = vp; incumbent_X = _norm(curp.copy())
        except Exception:
            continue

    # longer final polish reserved for last 20 seconds
    try:
        if time.time() < HARD - 2.0 and incumbent_X is not None:
            incumbent_X, incumbent_val = _lp_polish(incumbent_X, HARD - 1.0)
    except Exception:
        pass
    X = _norm(incumbent_X.copy())
    # tiny shrink guard not needed; just return normalized
    return X
