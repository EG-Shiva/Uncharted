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

def _soft_quench(X0, steps, T0, T1, lr0, lr1, rng=None, noise0=0.0, deadline=None):
    X = _norm(X0.copy())
    for it in range(1, steps + 1):
        if deadline is not None and (it & 31) == 0 and time.time() > deadline:
            break
        f = (it - 1) / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** f if T0 > 0 and T1 > 0 else T0
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        E = np.exp((G - M) / T)
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E))
        if not np.isfinite(S) or S <= 0:
            break
        Gr = (E @ X) / S
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        avg = float(np.mean(np.linalg.norm(Gt, axis=1))) + 1e-15
        Gt = Gt / avg
        mx = 0.08
        gn = float(np.max(np.linalg.norm(Gt, axis=1))) + 1e-15
        st = lr
        if gn * st > mx:
            st = mx / gn
        if noise0 > 0 and rng is not None:
            amp = noise0 * (1.0 - f)
            if amp > 1e-12:
                Z = rng.standard_normal(X.shape)
                Z = Z - np.sum(Z * X, axis=1, keepdims=True) * X
                zn = float(np.mean(np.linalg.norm(Z, axis=1))) + 1e-15
                Gt = Gt + (amp / zn) * Z
                # renormalize step cap again roughly
                gn2 = float(np.max(np.linalg.norm(Gt, axis=1))) + 1e-15
                if gn2 * st > mx:
                    st = mx / gn2
        X = X - st * Gt
        X = _norm(X)
    return _norm(X)

def _riesz_quench(X0, s, steps, lr0, lr1, deadline=None):
    X = _norm(X0.copy())
    for it in range(1, steps + 1):
        if deadline is not None and (it & 31) == 0 and time.time() > deadline:
            break
        f = (it - 1) / max(steps - 1, 1)
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        np.fill_diagonal(G, 1.0)
        r2 = 2.0 - 2.0 * G
        np.fill_diagonal(r2, 1.0)
        r2 = np.maximum(r2, 1e-12)
        r = np.sqrt(r2)
        W = r ** (-(s + 2.0))
        np.fill_diagonal(W, 0.0)
        rs = np.sum(W, axis=1)
        Gw = rs[:, None] * X - W @ X
        dot = np.sum(Gw * X, axis=1, keepdims=True)
        Gt = Gw - dot * X
        avg = float(np.mean(np.linalg.norm(Gt, axis=1))) + 1e-15
        Gt = Gt / avg
        mx = 0.08
        gn = float(np.max(np.linalg.norm(Gt, axis=1))) + 1e-15
        st = lr
        if gn * st > mx:
            st = mx / gn
        X = X + st * Gt
        X = _norm(X)
    return _norm(X)

def _hinge(X0, iters=100, lr=0.02, band=0.004, deadline=None):
    X = _norm(X0.copy())
    for it in range(iters):
        if deadline is not None and (it & 31) == 0 and time.time() > deadline:
            break
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
        step = lr
        if gn * step > mx:
            step = mx / gn
        X = X - step * Gt
        X = _norm(X)
        lr *= 0.995
    return _norm(X)

def _hole_fill(fixed, rng, ntry=1500):
    d = fixed.shape[1]
    C = rng.standard_normal((ntry, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-15
    dots = C @ fixed.T
    w = np.max(dots, axis=1)
    y = C[int(np.argmin(w))].copy()
    step = 0.06
    for _ in range(50):
        dd = fixed @ y
        j = int(np.argmax(dd))
        g = fixed[j] - float(fixed[j] @ y) * y
        nn = float(np.linalg.norm(g))
        if nn < 1e-12:
            break
        g /= nn
        y = y - step * g
        y /= np.linalg.norm(y) + 1e-15
        step *= 0.97
    return y / (np.linalg.norm(y) + 1e-15)

def _subset_rotate(X, k, theta, rng):
    n, d = X.shape
    idx = rng.choice(n, size=k, replace=False)
    a = rng.standard_normal(d); a /= np.linalg.norm(a) + 1e-15
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

def _lp_polish(X, deadline):
    X = _norm(X.copy())
    n, d = X.shape
    try:
        best = _maxcos(X)
    except Exception:
        return X, 1e9
    bx = X.copy()
    radius = 2.0e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline - 0.15 and radius > 1e-14:
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

def solve():
    t0 = time.time()
    HARD = t0 + 168.0
    HOP_END = t0 + 118.0
    DEEP_END = t0 + 142.0
    rng = np.random.default_rng(987654321)
    inc_best = None
    inc_val = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(f), dtype=float)
            if a.shape == (116, 6) and np.all(np.isfinite(a)):
                a = _norm(a)
                v = _maxcos(a)
                if v < inc_val:
                    inc_val = v
                    inc_best = a.copy()
        except Exception:
            pass
    if inc_best is None:
        inc_best = _norm(rng.standard_normal((116, 6)))
        inc_val = _maxcos(inc_best)
    best_val = float(inc_val)
    best_X = inc_best.copy()
    pool = [(best_val, best_X.copy())]
    # Phase A: diverse random basins (fast, no LP)
    n_init = 7
    for i in range(n_init):
        if time.time() > t0 + 35.0:
            break
        if time.time() > HOP_END:
            break
        X0 = _norm(rng.standard_normal((116, 6)))
        dl = min(HOP_END, HARD - 30)
        try:
            if i % 3 == 0:
                X1 = _riesz_quench(X0, 4.0, steps=220, lr0=0.12, lr1=0.015, deadline=dl)
                X2 = _soft_quench(X1, steps=140, T0=0.02, T1=0.003, lr0=0.08, lr1=0.02, deadline=dl)
            elif i % 3 == 1:
                X1 = _riesz_quench(X0, 1.0, steps=200, lr0=0.12, lr1=0.015, deadline=dl)
                X2 = _soft_quench(X1, steps=140, T0=0.025, T1=0.0025, lr0=0.09, lr1=0.02, deadline=dl)
            else:
                X2 = _soft_quench(X0, steps=300, T0=0.028, T1=0.0025, lr0=0.10, lr1=0.02, deadline=dl)
            v = _maxcos(X2)
            pool.append((v, X2.copy()))
            if v < best_val:
                best_val = v; best_X = X2.copy()
        except Exception:
            continue
    # antipodal doubled start for different topology
    try:
        if time.time() < HOP_END - 2:
            A0 = _norm(rng.standard_normal((58, 6)))
            A0 = _soft_quench(A0, steps=150, T0=0.02, T1=0.004, lr0=0.08, lr1=0.02, deadline=HOP_END)
            Xd = np.vstack([A0, -A0])
            Xd = _soft_quench(Xd, steps=160, T0=0.02, T1=0.003, lr0=0.08, lr1=0.02, deadline=HOP_END)
            v = _maxcos(Xd)
            pool.append((v, Xd.copy()))
            if v < best_val:
                best_val = v; best_X = Xd.copy()
    except Exception:
        pass
    pool = sorted(pool, key=lambda z: z[0])[:8]
    # Phase B: hub-fission + block-rotation hopping
    it = 0
    while time.time() < HOP_END and time.time() < HARD - 45:
        it += 1
        # pick base: mostly best, sometimes random
        try:
            r = rng.random()
            if r < 0.55:
                base = pool[0][1]
            elif len(pool) > 1 and r < 0.85:
                base = pool[int(rng.integers(0, len(pool)))][1]
            else:
                base = best_X
            Xp = base.copy()
            mode = it % 4
            if mode == 0 or mode == 1:
                # hub-fission: delete worst / high-contact points
                G = Xp @ Xp.T
                np.fill_diagonal(G, -2.0)
                wmax = np.max(G, axis=1)
                M = float(np.max(G))
                contacts = np.sum(G > (M - 0.003), axis=1)
                score = wmax + 0.002 * (contacts / (np.max(contacts) + 1e-9))
                # randomize k
                k = int(rng.integers(4, 11))
                if mode == 1:
                    k = int(rng.integers(8, 16))
                del_idx = np.argsort(-score)[:k]
                keep = np.ones(116, dtype=bool); keep[del_idx] = False
                rem = Xp[keep]
                news = []
                for _ in range(k):
                    cur = rem if not news else np.vstack([rem, np.stack(news)])
                    y = _hole_fill(cur, rng, ntry=1200)
                    news.append(y)
                Xp = np.vstack([rem, np.stack(news)])
                # light shuffle of new block positions
                perm = rng.permutation(116)
                # keep mostly ordered but move new points randomly: just permute all
                Xp = Xp[perm] if rng.random() < 0.5 else Xp
            elif mode == 2:
                k = int(rng.integers(12, 30))
                theta = float(rng.uniform(0.3, 0.95))
                Xp = _subset_rotate(Xp, k, theta, rng)
                # small kick to a few points to break symmetry
                if rng.random() < 0.5:
                    idx2 = rng.choice(116, size=6, replace=False)
                    Xp[idx2] += 0.25 * rng.standard_normal((6, 6))
                    Xp = _norm(Xp)
            else:
                # subset Gaussian kick + global jitter
                k = int(rng.integers(10, 26))
                idx = rng.choice(116, size=k, replace=False)
                sig = float(rng.uniform(0.25, 0.55))
                Xp[idx] += sig * rng.standard_normal((k, 6))
                Xp = _norm(Xp)
                if rng.random() < 0.4:
                    Xp = Xp + 0.03 * rng.standard_normal(Xp.shape)
                    Xp = _norm(Xp)
            # fast screen quench
            dl = min(HOP_END + 2.0, HARD - 30)
            # vary temperature schedule slightly
            T0 = float(rng.uniform(0.015, 0.030))
            T1 = float(rng.uniform(0.002, 0.004))
            Xq = _soft_quench(Xp, steps=210, T0=T0, T1=T1, lr0=0.085, lr1=0.018, rng=rng, noise0=0.0, deadline=dl)
            # short low-T tighten
            Xq = _soft_quench(Xq, steps=60, T0=0.004, T1=0.0018, lr0=0.04, lr1=0.015, deadline=dl)
            v = _maxcos(Xq)
            # pool update: keep best 8, allow slightly worse to preserve diversity
            pool.append((v, Xq.copy()))
            pool = sorted(pool, key=lambda z: z[0])[:8]
            if v < best_val - 1e-12:
                best_val = v; best_X = Xq.copy()
        except Exception:
            continue
    # Phase C: deep polish top candidates
    pool = sorted(pool, key=lambda z: z[0])[:4]
    deep_best_val = best_val
    deep_best_X = best_X.copy()
    for pi in range(min(3, len(pool))):
        if time.time() > DEEP_END:
            break
        try:
            _, Xc = pool[pi]
            Xc = Xc.copy()
            dl = DEEP_END
            # deeper anneal + hinge
            Xc = _soft_quench(Xc, steps=280, T0=0.012, T1=0.002, lr0=0.07, lr1=0.015, deadline=dl)
            Xc = _hinge(Xc, iters=90, lr=0.02, band=0.004, deadline=dl)
            Xc = _soft_quench(Xc, steps=90, T0=0.0035, T1=0.0015, lr0=0.035, lr1=0.012, deadline=dl)
            v = _maxcos(Xc)
            if v < deep_best_val:
                deep_best_val = v; deep_best_X = Xc.copy()
            # update pool entry
            pool[pi] = (v, Xc.copy())
        except Exception:
            continue
    if deep_best_val < best_val:
        best_val = deep_best_val; best_X = deep_best_X.copy()
    # ensure we never lose incumbent
    if best_val > inc_val:
        best_val = float(inc_val); best_X = inc_best.copy()
    # Phase D: LP re-jam best (and runner-up if time)
    try:
        cand_list = sorted(pool, key=lambda z: z[0])[:2]
        # ensure best_X is first
        cand_list = [(best_val, best_X.copy())] + [c for c in cand_list if c[0] > best_val + 1e-15][:1]
        final_val = best_val
        final_X = best_X.copy()
        for (vv, XX) in cand_list:
            if time.time() > HARD - 1.5:
                break
            try:
                Y, v = _lp_polish(XX.copy(), HARD - 0.5)
                if v < final_val:
                    final_val = v; final_X = Y.copy()
            except Exception:
                continue
        best_X = final_X.copy(); best_val = float(final_val)
    except Exception:
        pass
    # safety: if somehow worse than incumbent, return incumbent
    try:
        if _maxcos(best_X) > inc_val and _maxcos(inc_best) <= inc_val + 1e-15:
            best_X = inc_best.copy()
    except Exception:
        pass
    return _norm(np.asarray(best_X, dtype=float))
