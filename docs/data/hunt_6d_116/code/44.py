import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _riesz_quench(X0, s, steps, lr0, lr1, deadline=None):
    X = _norm(X0.copy())
    n = X.shape[0]
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
        # cap max move
        mx = 0.08
        gn = float(np.max(np.linalg.norm(Gt, axis=1))) + 1e-15
        st = lr
        if gn * st > mx:
            st = mx / gn
        X = X + st * Gt
        X = _norm(X)
    return _norm(X)

def _soft_quench(X0, steps, T0, T1, lr0, lr1, deadline=None):
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
        X = X - st * Gt
        X = _norm(X)
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

def _lp_polish(X, deadline):
    X = _norm(X.copy())
    n, d = X.shape
    try:
        best = _maxcos(X)
    except Exception:
        return X
    bx = X.copy()
    radius = 2.0e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() > deadline - 0.1:
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
    return bx

def _align_pair(A, B):
    n = A.shape[0]
    Bc = B.copy()
    for _ in range(3):
        C = A @ Bc.T
        try:
            _, col = linear_sum_assignment(-C)
        except Exception:
            break
        Bm = Bc[col]
        M = Bm.T @ A
        try:
            U, _, Vt = np.linalg.svd(M)
            R = U @ Vt
        except Exception:
            R = np.eye(A.shape[1])
        Bc = Bm @ R
        Bc = _norm(Bc)
    return Bc

def _signature(X, k=30):
    G = X @ X.T
    iu = np.triu_indices(X.shape[0], 1)
    c = G[iu]
    o = np.argsort(-c)[:k]
    return np.sort(c[o])[::-1].copy()

def solve():
    t0 = time.time()
    HARD = t0 + 170.0
    P1_END = t0 + 82.0
    rng = np.random.default_rng(20261116)
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
    parents = []
    # diverse random starts with Riesz s in {1,4,12} + softmax
    cfgs = []
    for i in range(12):
        if i % 4 == 0:
            cfgs.append(("riesz1", 1.0))
        elif i % 4 == 1:
            cfgs.append(("riesz4", 4.0))
        elif i % 4 == 2:
            cfgs.append(("riesz12", 12.0))
        else:
            cfgs.append(("soft", 0.0))
    for ci, (kind, s) in enumerate(cfgs):
        if time.time() > P1_END:
            break
        if time.time() > HARD - 80:
            break
        X0 = _norm(rng.standard_normal((116, 6)))
        dl = min(P1_END, HARD - 60)
        if kind.startswith("riesz"):
            X1 = _riesz_quench(X0, s, steps=260, lr0=0.12, lr1=0.015, deadline=dl)
            X2 = _soft_quench(X1, steps=150, T0=0.02, T1=0.003, lr0=0.08, lr1=0.02, deadline=dl)
        else:
            X2 = _soft_quench(X0, steps=320, T0=0.028, T1=0.0025, lr0=0.10, lr1=0.02, deadline=dl)
        v = _maxcos(X2)
        parents.append((v, X2.copy()))
    # heavily deleted/reinserted incumbent variant
    if time.time() < P1_END - 1 and time.time() < HARD - 70:
        try:
            Xb = inc_best.copy()
            G = Xb @ Xb.T
            np.fill_diagonal(G, -2.0)
            worst = np.max(G, axis=1)
            ndel = 22
            del_idx = np.argsort(-worst)[:ndel]
            keep = np.ones(116, dtype=bool); keep[del_idx] = False
            fixed = Xb[keep]
            newpts = []
            for _ in range(ndel):
                y = _hole_fill(np.vstack([fixed] + ([np.stack(newpts)] if newpts else [])), rng)
                newpts.append(y)
            Xd = np.vstack([fixed, np.stack(newpts)])
            # shuffle to avoid order bias
            perm = rng.permutation(116)
            Xd = Xd[perm]
            dl = min(P1_END, HARD - 60)
            Xd = _soft_quench(Xd, steps=220, T0=0.02, T1=0.003, lr0=0.08, lr1=0.02, deadline=dl)
            v = _maxcos(Xd)
            parents.append((v, Xd.copy()))
        except Exception:
            pass
    # always include incumbent itself as parent for crossover
    parents.append((inc_val, inc_best.copy()))
    # brief ranking, keep best ~10 diverse
    parents = sorted(parents, key=lambda z: z[0])
    # diversity filter by signature
    filt = []
    for v, X in parents:
        s = _signature(X)
        dup = False
        for _, X2 in filt:
            pass
        # keep signature distance check
        ok = True
        for vv, XX in filt:
            s2 = _signature(XX)
            if float(np.linalg.norm(s - s2)) < 5e-5 and abs(v - vv) < 2e-6:
                ok = False
                break
        if ok:
            filt.append((v, X))
        if len(filt) >= 11:
            break
    parents = filt
    if len(parents) < 2:
        parents = sorted(parents, key=lambda z: z[0])
    # genetic crossover
    P2_END = HARD - 22.0
    children = []
    # all pairs, but cap number
    pairs = []
    np_ = len(parents)
    for i in range(np_):
        for j in range(i + 1, np_):
            pairs.append((i, j))
    rng.shuffle(pairs)
    # limit to ~10 pairs to fit time
    pairs = pairs[:10]
    for (i, j) in pairs:
        if time.time() > P2_END:
            break
        vA, A = parents[i]
        vB, B = parents[j]
        try:
            Bal = _align_pair(A, B)
        except Exception:
            Bal = B.copy()
        thr = max(float(vA), float(vB))
        # two child types per pair
        for ctype in ("half", "hemi"):
            if time.time() > P2_END:
                break
            n = 116
            child = np.empty_like(A)
            if ctype == "half":
                S = rng.choice(n, size=n // 2, replace=False)
                mask = np.zeros(n, dtype=bool); mask[S] = True
                child[mask] = A[mask]
                child[~mask] = Bal[~mask]
            else:
                u = rng.standard_normal(6); u /= np.linalg.norm(u) + 1e-15
                mid = 0.5 * (A + Bal)
                sc = mid @ u
                med = float(np.median(sc))
                mask = sc > med
                # ensure both nonempty and balanced-ish
                if int(np.sum(mask)) < 20 or int(np.sum(mask)) > 96:
                    S = rng.choice(n, size=n // 2, replace=False)
                    mask = np.zeros(n, dtype=bool); mask[S] = True
                child[mask] = A[mask]
                child[~mask] = Bal[~mask]
            child = _norm(child)
            # delete points creating cosines above parents range
            try:
                Gc = child @ child.T
                np.fill_diagonal(Gc, -2.0)
                bad = Gc > thr + 1e-12
                if np.any(bad):
                    wmax = np.max(Gc, axis=1)
                    ndel = min(8, int(np.sum(wmax > thr)))
                    if ndel > 0:
                        del_idx = np.argsort(-wmax)[:ndel]
                        keep = np.ones(n, dtype=bool); keep[del_idx] = False
                        rem = child[keep]
                        news = []
                        for _ in range(ndel):
                            cur = rem if not news else np.vstack([rem] + [np.stack(news)])
                            y = _hole_fill(cur, rng, ntry=1200)
                            news.append(y)
                        child = np.vstack([rem, np.stack(news)])
                        # re-shuffle lightly to mix positions
                        child = _norm(child)
            except Exception:
                pass
            # quench child briefly
            try:
                child = _soft_quench(child, steps=200, T0=0.015, T1=0.0025, lr0=0.07, lr1=0.015, deadline=P2_END)
            except Exception:
                pass
            try:
                v = _maxcos(child)
            except Exception:
                continue
            children.append((v, child.copy()))
            # early stop if too many
            if len(children) >= 20:
                break
        if len(children) >= 20:
            break
    # rank all by exact max cosine
    pool = parents + children
    pool = sorted(pool, key=lambda z: z[0])
    # diversity-preserving selection for LP polish: take best + diverse
    sel = []
    for v, X in pool:
        if len(sel) >= 5:
            break
        s = _signature(X)
        dup = False
        for _, Xs in sel:
            s2 = _signature(Xs)
            if float(np.linalg.norm(s - s2)) < 1e-7 and abs(v - sel[0][0]) < 1e-9:
                dup = True
                break
        if not dup or len(sel) == 0:
            sel.append((v, X))
    # LP polish selected in order
    best_val = inc_val
    best_X = inc_best.copy()
    # include unpolished pool best
    if pool and pool[0][0] < best_val:
        best_val = pool[0][0]; best_X = pool[0][1].copy()
    for v, X in sel:
        if time.time() > HARD - 2:
            break
        # only polish promising (within 0.005 of best)
        if v > best_val + 0.005:
            continue
        try:
            Xp = _lp_polish(X, deadline=min(HARD - 1, time.time() + 8.0))
            vp = _maxcos(Xp)
        except Exception:
            continue
        if vp < best_val:
            best_val = vp; best_X = Xp.copy()
    best_X = _norm(best_X)
    return best_X
