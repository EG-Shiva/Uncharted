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

def _degrees(X, tol=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    C = G > (m - tol)
    deg = np.sum(C, axis=1)
    return deg, m, G

def _edges(X, tol=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    iu = np.triu_indices(X.shape[0], 1)
    sel = G[iu] > (m - tol)
    s = set(zip(iu[0][sel].tolist(), iu[1][sel].tolist()))
    return s, m

def _deep_hole(cur_fixed, rng, ntry=50000):
    d = cur_fixed.shape[1]
    C = rng.standard_normal((ntry, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-15
    dots = C @ cur_fixed.T
    w = np.max(dots, axis=1)
    y = C[int(np.argmin(w))].copy()
    step = 0.06
    for _ in range(50):
        dd = cur_fixed @ y
        j = int(np.argmax(dd))
        fj = cur_fixed[j]
        g = fj - float(fj @ y) * y
        nn = float(np.linalg.norm(g))
        if nn < 1e-12 or not np.isfinite(nn):
            break
        g /= nn
        y = y - step * g
        ny = float(np.linalg.norm(y))
        if ny < 1e-12:
            break
        y /= ny
        step *= 0.985
    return y / (np.linalg.norm(y) + 1e-15)

def _hinge_frozen(X0, free_idx, iters=200, band=0.01, lr0=0.02):
    X = _norm(X0.copy())
    n = X.shape[0]
    free = np.zeros(n, dtype=bool)
    free[free_idx] = True
    lr = lr0
    for _ in range(iters):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        thr = M - band
        Mk = G > thr
        A = Mk @ X
        w = np.sum(Mk * G, axis=1, keepdims=True)
        Gt = A - w * X
        Gt[~free] = 0.0
        gn = float(np.max(np.linalg.norm(Gt, axis=1)))
        if gn < 1e-14 or not np.isfinite(gn):
            break
        mx = 0.03
        step = lr
        if gn * step > mx:
            step = mx / gn
        X[free] = X[free] - step * Gt[free]
        X = _norm(X)
        lr *= 0.995
    return _norm(X)

def _softmax_all(X0, steps=300, T0=0.02, T1=0.002, lr0=2.0, lr1=0.25):
    X = _norm(X0.copy())
    for s in range(steps):
        f = s / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** f
        lr = lr0 * (1 - f) + lr1 * f
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
        gn = float(np.max(np.linalg.norm(Gt, axis=1)))
        if not np.isfinite(gn) or gn < 1e-14:
            continue
        mx = 0.05
        if gn * lr > mx:
            Gt = Gt * (mx / (gn * lr))
        X = X - lr * Gt
        X = _norm(X)
    return _norm(X)

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
        m = float(c.max())
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
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
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

def solve():
    t0 = time.time()
    HARD = t0 + 171.0
    CYCLE_END = t0 + 140.0
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

    incumbent = None
    incumbent_val = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            a = _clean(np.load(f))
        except Exception:
            a = None
        if a is not None:
            v = _maxcos(a)
            if v < incumbent_val:
                incumbent_val = v
                incumbent = a.copy()
    except_fn = None
    if incumbent is None:
        incumbent = _norm(rng.standard_normal((116, 6)))
        incumbent_val = _maxcos(incumbent)
    inc_edges, _ = _edges(incumbent, tol=1e-9)
    n_inc_edges = max(len(inc_edges), 1)

    best_X = incumbent.copy()
    best_val = incumbent_val
    bases = [incumbent.copy()]

    trial = 0
    ks = [3, 5, 8]
    while time.time() < CYCLE_END and time.time() < HARD - 25.0:
        trial += 1
        # pick base round-robin/random for diversity
        bi = int(rng.integers(0, len(bases)))
        base = bases[bi]
        deg, _, _ = _degrees(base, tol=1e-9)
        hubs = np.where(deg >= 14)[0]
        lows = np.where(deg <= 3)[0]
        k = ks[trial % 3]
        # select k hub deletions
        if len(hubs) >= k:
            del_h = rng.choice(hubs, size=k, replace=False)
        else:
            order = np.argsort(-deg)
            del_h = order[:k]
        # optionally 0-2 low-degree deletions
        n_low = int(rng.integers(0, 3))
        del_l = np.array([], dtype=int)
        if n_low > 0 and len(lows) > 0:
            n_low = min(n_low, len(lows), 2)
            # avoid overlap with del_h
            avail = np.setdiff1d(lows, del_h)
            if len(avail) > 0:
                n_low = min(n_low, len(avail))
                del_l = rng.choice(avail, size=n_low, replace=False)
            else:
                del_l = np.array([], dtype=int)
        del_idx = np.unique(np.concatenate([del_h, del_l]).astype(int))
        n_del = len(del_idx)
        if n_del == 0 or n_del > 12:
            continue
        keep_mask = np.ones(116, dtype=bool)
        keep_mask[del_idx] = False
        fixed_core = base[keep_mask]
        cur = fixed_core.copy()
        # greedy deep-hole refill one at a time, >=50k samples each
        ok = True
        for _ in range(n_del):
            if time.time() > CYCLE_END or time.time() > HARD - 25.0:
                ok = False
                break
            y = _deep_hole(cur, rng, ntry=50000)
            cur = np.vstack([cur, y])
        if not ok:
            break
        X0 = cur.copy()  # (116,6): first 116-n_del are kept, last n_del are new
        n_keep = fixed_core.shape[0]
        free_idx = np.arange(n_keep, 116)
        # stage 1: frozen core hinge 200 steps band 0.01
        X1 = _hinge_frozen(X0, free_idx, iters=200, band=0.01, lr0=0.02)
        # stage 2: all-point annealed softmax 300 steps T 0.02->0.002
        X2 = _softmax_all(X1, steps=300, T0=0.02, T1=0.002, lr0=2.0, lr1=0.25)
        v2 = _maxcos(X2)
        # cheap screen
        if not np.isfinite(v2) or v2 >= 0.598:
            # still allow near-miss diversity? skip LP
            # keep exploring: if topology very different but score bad, ignore
            continue
        # LP polish only promising candidates
        remain = min(CYCLE_END, HARD - 5.0) - time.time()
        if remain < 1.5:
            break
        budget = min(6.0, remain)
        Xp, vp = _lp_polish(X2, time.time() + budget)
        if not np.isfinite(vp):
            continue
        # topology gate: >=5% edge change vs original incumbent
        cand_edges, _ = _edges(Xp, tol=1e-9)
        sym = len(cand_edges.symmetric_difference(inc_edges))
        edge_change = sym / float(n_inc_edges)
        if edge_change < 0.05:
            continue
        if vp < best_val - 1e-12:
            best_val = vp
            best_X = Xp.copy()
            # new incumbent for future surgery
            if len(bases) < 5:
                bases.append(Xp.copy())
            else:
                # replace a random non-fallback base
                ri = int(rng.integers(1, len(bases)))
                bases[ri] = Xp.copy()
        elif vp <= best_val + 5e-6 and len(bases) < 5:
            # tie but new topology: keep as alternate basin seed
            # avoid duplicates: check diff vs existing bases
            dup = False
            for b in bases:
                be, _ = _edges(b, tol=1e-9)
                if len(be.symmetric_difference(cand_edges)) / float(max(len(be), 1)) < 0.02:
                    dup = True
                    break
            if not dup:
                bases.append(Xp.copy())
        # stop early if excellent? continue cycling to use time
        if time.time() > CYCLE_END:
            break

    # final safety: return best (fallback incumbent always returnable)
    # one last short polish if time remains and best improved little
    if time.time() < HARD - 6.0 and best_val < incumbent_val - 1e-12:
        # already polished; just ensure normalized
        pass
    best_X = _norm(best_X)
    # verify no zero rows / NaN; else fallback
    if not np.all(np.isfinite(best_X)) or best_X.shape != (116, 6):
        best_X = _norm(incumbent)
    nn = np.linalg.norm(best_X, axis=1)
    if np.any(nn < 1e-12) or not np.all(np.isfinite(nn)):
        best_X = _norm(incumbent)
    return _norm(best_X)
