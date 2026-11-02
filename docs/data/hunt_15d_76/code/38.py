import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _build_c15(seeds):
    # seeds (5,15) unit rows -> X (76,15)
    X = np.empty((76, 15), dtype=np.float64)
    for o in range(5):
        s = seeds[o]
        # X[k,j] = s[(j-k)%15]
        for k in range(15):
            X[o * 15 + k] = np.roll(s, k)
    X[75] = np.ones(15, dtype=np.float64) / np.sqrt(15.0)
    return X

def _build_anti(Y):
    return np.vstack([Y, -Y])

def _seeds_anneal(seeds_init, steps, T0, T1, lr0, lr1, rng=None):
    seeds = _norm(seeds_init.copy())
    best_seeds = seeds.copy()
    Xb = _build_c15(seeds)
    bv = _maxcos(Xb)
    for k in range(steps):
        f = k / max(1, steps - 1)
        T = T0 * (T1 / T0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        X = _build_c15(seeds)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            break
        Gr = (E @ X) / (S * 0.5)  # (76,15)
        # pullback to seeds
        gs = np.zeros_like(seeds)
        for o in range(5):
            acc = np.zeros(15)
            base = o * 15
            for kk in range(15):
                acc += np.roll(Gr[base + kk], -kk)
            gs[o] = acc
        # tangent project
        dot = np.sum(gs * seeds, axis=1, keepdims=True)
        gt = gs - dot * seeds
        # clip
        seeds = seeds - lr * gt
        seeds = _norm(seeds)
        if (k % 200 == 0) or (k == steps - 1):
            v = _maxcos(_build_c15(seeds))
            if v < bv:
                bv = v
                best_seeds = seeds.copy()
    return best_seeds, bv

def _anti_anneal(Y_init, steps, T0, T1, lr0, lr1):
    Y = _norm(Y_init.copy())
    bv = _maxcos(_build_anti(Y))
    best = Y.copy()
    for k in range(steps):
        f = k / max(1, steps - 1)
        T = T0 * (T1 / T0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        X = _build_anti(Y)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            break
        Gr = (E @ X) / (S * 0.5)
        Gt = Gr[:38] - Gr[38:]
        dot = np.sum(Gt * Y, axis=1, keepdims=True)
        Gtt = Gt - dot * Y
        Y = Y - lr * Gtt
        Y = _norm(Y)
        if (k % 200 == 0) or (k == steps - 1):
            v = _maxcos(_build_anti(Y))
            if v < bv:
                bv = v
                best = Y.copy()
    return best, bv

def _free_anneal(X_init, steps, T0, T1, lr0, lr1):
    X = _norm(X_init.copy())
    bv = _maxcos(X)
    best = X.copy()
    n = X.shape[0]
    for k in range(steps):
        f = k / max(1, steps - 1)
        T = T0 * (T1 / T0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            break
        Gr = (E @ X) / (S * 0.5)
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        X = X - lr * Gt
        X = _norm(X)
        if (k % 250 == 0) or (k == steps - 1):
            v = _maxcos(X)
            if v < bv:
                bv = v
                best = X.copy()
    return best, bv

def _lp_polish(x, t_end):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return x
    x = _norm(x.copy())
    n, d = x.shape
    best = _maxcos(x)
    radius = 8e-4
    iu = np.triu_indices(n, 1)
    while time.time() < t_end - 0.6 and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        thr = max(20 * radius, 1e-9)
        sel = c > m - thr
        if int(sel.sum()) > 1400:
            idx = np.argpartition(-c, 1400)[:1400]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _maxcos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    t_end = t0 + 172.0
    rng = np.random.default_rng(0)
    try:
        Xb = np.load("best.npy").astype(float)
    except Exception:
        Xb = None
    try:
        Xr = np.load("record.npy").astype(float)
    except Exception:
        Xr = None
    cands = []
    if Xb is not None and Xb.shape == (76, 15):
        cands.append(_norm(Xb))
    if Xr is not None and Xr.shape == (76, 15):
        cands.append(_norm(Xr))
    if len(cands) == 0:
        R = rng.normal(size=(76, 15))
        return _norm(R)
    best = cands[0].copy()
    bv = _maxcos(best)
    for c in cands[1:]:
        v = _maxcos(c)
        if v < bv:
            best, bv = c.copy(), v
    Xbest = best.copy()
    best_val = bv

    # --- Phase A: C15 symmetric search (few restarts, cheap) ---
    sym_best = None
    sym_val = 1e9
    # 4 restarts
    for r in range(4):
        if time.time() > t_end - 90:
            break
        rs = np.random.default_rng(100 + r)
        s0 = _norm(rs.normal(size=(5, 15)))
        # vary schedule slightly per restart
        if r % 2 == 0:
            s_opt, v = _seeds_anneal(s0, steps=1800, T0=0.02, T1=0.0006, lr0=0.6, lr1=0.03)
        else:
            s_opt, v = _seeds_anneal(s0, steps=1800, T0=0.008, T1=0.0003, lr0=0.4, lr1=0.02)
        if v < sym_val:
            sym_val = v
            sym_best = _build_c15(s_opt)
    # --- Phase B: antipodal 38-pair search ---
    anti_best = None
    anti_val = 1e9
    for r in range(3):
        if time.time() > t_end - 75:
            break
        rs = np.random.default_rng(200 + r)
        y0 = _norm(rs.normal(size=(38, 15)))
        y_opt, v = _anti_anneal(y0, steps=1600, T0=0.015 if r % 2 == 0 else 0.007, T1=0.0004, lr0=0.5, lr1=0.03)
        if v < anti_val:
            anti_val = v
            anti_best = _build_anti(y_opt)

    # choose release starting points: symmetric/antipodal winners + global best
    starts = []
    if sym_best is not None:
        starts.append((sym_best, sym_val))
    if anti_best is not None:
        starts.append((anti_best, anti_val))
    # release best symmetric candidate even if worse (new basin)
    # also include Xbest perturbed slightly as fallback safety is handled at end
    rel_best = None
    rel_val = 1e9
    for (Xs, vs) in starts:
        if time.time() > t_end - 60:
            break
        # short hot release then cool release
        X1, v1 = _free_anneal(Xs, steps=1800, T0=0.012, T1=0.0006, lr0=1.2, lr1=0.08)
        # extra cool polish
        if time.time() < t_end - 55:
            X1, v1 = _free_anneal(X1, steps=1200, T0=0.0012, T1=0.0002, lr0=0.35, lr1=0.03)
        if v1 < rel_val:
            rel_val = v1
            rel_best = X1.copy()

    # also quick release from global best to ensure no regression (short)
    if time.time() < t_end - 45:
        Xg, vg = _free_anneal(Xbest, steps=1200, T0=0.004, T1=0.00025, lr0=0.5, lr1=0.04)
        if vg < rel_val:
            rel_val = vg
            rel_best = Xg.copy()

    overall_X = Xbest.copy()
    overall_v = best_val
    for cand_X, cand_v in ([(rel_best, rel_val)] if rel_best is not None else []):
        if cand_v < overall_v:
            overall_X, overall_v = cand_X.copy(), cand_v

    # final LP polish on the current overall best (time-boxed)
    if time.time() < t_end - 5:
        Xp = _lp_polish(overall_X, t_end)
        vp = _maxcos(Xp)
        if vp < overall_v:
            overall_X, overall_v = Xp, vp
        # safety: if polish hurt or symmetric path failed, keep Xbest
        if overall_v > best_val and best_val < 0.169:
            # still return the better of the two
            overall_X = Xbest if best_val < overall_v else overall_X
    else:
        overall_X = overall_X if overall_v < best_val else Xbest

    # final safety: return best among Xbest and overall_X
    if _maxcos(overall_X) > best_val:
        overall_X = Xbest.copy()
    overall_X = _norm(overall_X)
    return overall_X
