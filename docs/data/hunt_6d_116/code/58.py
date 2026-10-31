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

def _contacts(X, tol=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    M = G > (m - tol)
    deg = M.sum(axis=1)
    iu = np.triu_indices(X.shape[0], 1)
    mask = M[iu]
    return m, deg, mask

def _edge_diff(mask_a, mask_b):
    na = int(mask_a.sum())
    if na == 0:
        return 1.0
    ham = int(np.count_nonzero(mask_a != mask_b))
    return ham / float(na)

def _refine_hole(y, core, steps=50, step0=0.08, decay=0.97):
    y = y / (np.linalg.norm(y) + 1e-15)
    step = step0
    for _ in range(steps):
        dd = core @ y
        j = int(np.argmax(dd))
        dj = float(dd[j])
        g = core[j] - dj * y
        nn = float(np.linalg.norm(g))
        if nn < 1e-12:
            break
        g /= nn
        y = y - step * g
        y /= (np.linalg.norm(y) + 1e-15)
        step *= decay
    return y / (np.linalg.norm(y) + 1e-15)

def _hinge_frozen(X, ins_idx, iters=200, band=0.01, lr0=0.02, t_end=None):
    X = _norm(X.copy())
    n = X.shape[0]
    ins = np.asarray(ins_idx, dtype=int)
    lr = lr0
    for it in range(iters):
        if t_end is not None and (it & 31) == 0 and time.time() > t_end:
            break
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(G.max())
        thr = M - band
        Mk = G > thr
        Grad = Mk @ X
        # only inserted rows
        Gi = Grad[ins]
        Xi = X[ins]
        dot = np.sum(Gi * Xi, axis=1, keepdims=True)
        Gt = Gi - dot * Xi
        gn = float(np.max(np.linalg.norm(Gt, axis=1))) if len(ins) > 0 else 0.0
        if gn < 1e-14:
            lr *= 0.995
            continue
        mx = 0.03
        step = lr
        if gn * step > mx:
            step = mx / gn
        X[ins] = X[ins] - step * Gt
        X[ins] = _norm(X[ins])
        lr *= 0.995
    return _norm(X)

def _softmax_all(X, steps=300, T0=0.02, T1=0.002, lr0=2.0, lr1=0.25, t_end=None):
    X = _norm(X.copy())
    for s in range(steps):
        if t_end is not None and (s & 31) == 0 and time.time() > t_end:
            break
        f = s / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** f if T0 > 0 and T1 > 0 else T0
        lr = lr0 * (1 - f) + lr1 * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(G.max())
        E = np.exp((G - M) / max(T, 1e-12))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            break
        Grad = (E @ X) / S
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        gn = float(np.max(np.linalg.norm(Gt, axis=1)))
        if gn > 0:
            mx = 0.05
            eff = lr
            if gn * eff > mx:
                Gt = Gt * (mx / (gn * eff))
        X = X - lr * Gt
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

def solve():
    t0 = time.time()
    HARD = t0 + 170.0
    EXPL = t0 + 140.0
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
    # incumbent snapshot for fallback + topology reference
    inc_x = best_x.copy()
    inc_v = best_v
    _, _, inc_mask = _contacts(inc_x, tol=1e-9)
    inc_mask = inc_mask.copy()
    # quick initial LP touch on incumbent if time allows (kept as fallback)
    try:
        if HARD - time.time() > 30:
            bx, bv = _lp_polish(best_x, min(time.time() + 2.0, HARD - 20))
            # only accept same-topology polish as new incumbent reference if better
            if bv < best_v - 1e-12:
                # check topology: update reference too
                _, _, m2 = _contacts(bx, tol=1e-9)
                d0 = _edge_diff(inc_mask, m2)
                best_x, best_v = bx.copy(), float(bv)
                if d0 < 0.05:
                    inc_x, inc_v, inc_mask = bx.copy(), float(bv), m2.copy()
    except Exception:
        pass
    # ensure best tracks incumbent initially
    if best_v > inc_v:
        best_x, best_v = inc_x.copy(), inc_v
    trial = 0
    del_cycle = [3, 5, 8]
    while time.time() < EXPL:
        if time.time() > HARD - 22:
            break
        trial += 1
        # hubs from current best (starts as incumbent)
        try:
            _, deg, _ = _contacts(best_x, tol=1e-9)
        except Exception:
            deg = np.full(116, 9)
        hubs = np.where(deg >= 14)[0]
        lows = np.where(deg <= 3)[0]
        # if not enough hubs, take top-degree points
        if len(hubs) < 8:
            order = np.argsort(-deg)
            hubs = order[:max(8, len(hubs))]
        k_del = del_cycle[(trial - 1) % len(del_cycle)]
        k_del = min(k_del, len(hubs))
        # choose victims among hubs
        try:
            victims_h = rng.choice(hubs, size=k_del, replace=False)
        except Exception:
            victims_h = hubs[:k_del]
        # optionally 1-2 low-contact points
        n_low = 0
        r = rng.random()
        if len(lows) > 0:
            if r < 0.3:
                n_low = 1
            elif r < 0.5:
                n_low = min(2, len(lows))
        victims_l = np.array([], dtype=int)
        if n_low > 0:
            # exclude already chosen
            pool = np.setdiff1d(lows, victims_h)
            if len(pool) > 0:
                n_low = min(n_low, len(pool))
                victims_l = rng.choice(pool, size=n_low, replace=False)
        victims = np.concatenate([victims_h, victims_l]).astype(int) if len(victims_l) > 0 else victims_h.astype(int)
        k = len(victims)
        if k == 0:
            continue
        victim_set = set(int(v) for v in victims)
        keep_idx = np.array([i for i in range(116) if i not in victim_set], dtype=int)
        kept = best_x[keep_idx].copy()
        # deep-hole insertion greedy one at a time, >=50k samples each
        cur_core = kept.copy()
        new_pts = []
        feasible = True
        for _ in range(k):
            if time.time() > EXPL and trial > 2:
                feasible = False
                break
            try:
                C = rng.standard_normal((50000, 6))
                C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-15)
                dots = C @ cur_core.T
                w = dots.max(axis=1)
                y = C[int(np.argmin(w))].copy()
                y = _refine_hole(y, cur_core, steps=50, step0=0.08, decay=0.97)
            except Exception:
                y = rng.standard_normal(6); y /= (np.linalg.norm(y) + 1e-15)
            new_pts.append(y)
            cur_core = np.vstack([cur_core, y[None, :]])
        if not feasible:
            continue
        # reassemble full code
        X = np.empty((116, 6))
        X[keep_idx] = kept
        for ii, vi in enumerate(victims):
            X[int(vi)] = new_pts[ii]
        X = _norm(X)
        ins_pos = np.asarray([int(v) for v in victims], dtype=int)
        # stage 1: frozen core, 200 hinge steps band 0.01 (only inserted move)
        q_end = min(HARD - 12, EXPL + 5)
        try:
            X = _hinge_frozen(X, ins_pos, iters=200, band=0.01, lr0=0.02, t_end=q_end)
        except Exception:
            pass
        # stage 2: all points 300-step annealed softmax T 0.02->0.002
        try:
            X = _softmax_all(X, steps=300, T0=0.02, T1=0.002, lr0=2.0, lr1=0.25, t_end=q_end)
        except Exception:
            pass
        v_mid = _maxcos(X)
        # cheap screen: LP-polish only if already below 0.598
        if v_mid >= 0.598:
            continue
        if time.time() > HARD - 14:
            continue
        try:
            Xp, v_lp = _lp_polish(X, min(time.time() + 1.2, HARD - 10))
        except Exception:
            continue
        # topology gate: >=5% edge change vs incumbent reference
        try:
            _, _, m_lp = _contacts(Xp, tol=1e-9)
            diff = _edge_diff(inc_mask, m_lp)
        except Exception:
            diff = 1.0
        if diff < 0.05:
            continue
        if v_lp < best_v - 1e-12:
            best_x, best_v = Xp.copy(), float(v_lp)
    # final safety LP on best if time remains
    try:
        remain = HARD - time.time() - 1.0
        if remain > 2.0:
            bx, bv = _lp_polish(best_x, time.time() + min(remain, 12.0))
            if bv < best_v - 1e-12:
                try:
                    _, _, m2 = _contacts(bx, tol=1e-9)
                    d2 = _edge_diff(inc_mask, m2)
                    # allow final polish even if same topology if it improves over incumbent fallback
                    if d2 >= 0.05 or bv < inc_v:
                        best_x, best_v = bx.copy(), float(bv)
                except Exception:
                    if bv < inc_v:
                        best_x, best_v = bx.copy(), float(bv)
    except Exception:
        pass
    # always-returnable fallback: if best is worse than incumbent, return incumbent
    if best_v > inc_v and np.all(np.isfinite(inc_x)):
        return _norm(inc_x)
    return _norm(best_x)
