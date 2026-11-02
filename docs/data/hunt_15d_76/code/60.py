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

def _contacts(X, thr=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    C = (G > m - thr)
    # exclude diagonal (already -2)
    deg = C.sum(axis=1).astype(int)
    return deg, m, G

def _refine_hole(v, Y, n_iter=200, T0=0.008, lr0=0.08):
    v = v / (np.linalg.norm(v) + 1e-12)
    for k in range(n_iter):
        f = k / max(1, n_iter - 1)
        Tk = T0 * (0.2 ** f)
        lrk = lr0 * (0.3 ** f)
        s = Y @ v
        m = float(s.max())
        w = np.exp((s - m) / max(Tk, 1e-12))
        ws = float(w.sum())
        if not np.isfinite(ws) or ws <= 0:
            break
        w = w / ws
        g = w @ Y
        g = g - float(np.dot(g, v)) * v
        ng = float(np.linalg.norm(g))
        if ng < 1e-12:
            break
        # limit step
        step = lrk * min(1.0, ng * 5.0)
        ghat = g / (ng + 1e-12)
        v = v - step * ghat
        nv = float(np.linalg.norm(v))
        if not np.isfinite(nv) or nv == 0:
            break
        v = v / nv
    return v

def _surgery(X, hubs, rattlers_move, step_base, rng):
    n, d = X.shape
    Y = X.copy()
    G = Y @ Y.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    # move hubs along negative centroid of tight neighbours
    for hi, h in enumerate(hubs):
        row = G[h].copy()
        row[h] = -2.0
        neigh = np.where(row > m - 1e-7)[0]
        if len(neigh) < 3:
            # fallback: top-5 nearest
            k = min(5, n - 1)
            neigh = np.argpartition(-row, k - 1)[:k]
        Tj = Y[neigh] - (Y[neigh] @ Y[h])[:, None] * Y[h][None, :]
        centroid = Tj.mean(axis=0)
        nrm = float(np.linalg.norm(centroid))
        if nrm < 1e-12 or not np.isfinite(nrm):
            # random tangent
            rv = rng.normal(size=d)
            rv = rv - float(rv @ Y[h]) * Y[h]
            n2 = float(np.linalg.norm(rv))
            if n2 < 1e-12:
                continue
            direction = rv / n2
        else:
            direction = -centroid / nrm
        step = step_base + 0.015 * (hi % 4)  # stays in 0.05-0.15 range
        step = float(np.clip(step, 0.05, 0.15))
        Y[h] = Y[h] + step * direction
        Y[h] = Y[h] / (np.linalg.norm(Y[h]) + 1e-12)
        # update G row for subsequent hubs (cheap full recompute every 2 hubs)
        if hi % 2 == 1:
            G = Y @ Y.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
    # teleport rattlers into largest holes adjacent to hubs
    G = Y @ Y.T
    np.fill_diagonal(G, -2.0)
    for k, r in enumerate(rattlers_move):
        hub = int(hubs[k % len(hubs)]) if len(hubs) > 0 else -1
        mask = np.ones(n, dtype=bool)
        mask[r] = False
        Yo = Y[mask]
        NS = 3000
        C1 = rng.normal(size=(NS // 2, d))
        C1 = _norm(C1)
        # near-hub candidates
        if 0 <= hub < n:
            pert = rng.normal(size=(NS - NS // 2, d)) * 0.6 + Y[hub][None, :]
            pert = _norm(pert)
            C = np.vstack([C1, pert])
        else:
            C = C1
        scores = (C @ Yo.T).max(axis=1)
        bi = int(np.argmin(scores))
        v0 = C[bi]
        v0 = _refine_hole(v0, Yo, n_iter=200, T0=0.008, lr0=0.08)
        Y[r] = v0 / (np.linalg.norm(v0) + 1e-12)
    return _norm(Y)

def _anneal_aug(Y0, steps=2000, T0=0.012, T1=0.0009, lr0=0.02, lr1=0.002,
                lam0=1e-2, mu0=1e-2):
    n, d = Y0.shape
    Y = Y0.copy()
    m1 = np.zeros_like(Y)
    v2 = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1t, b2t = 1.0, 1.0
    a = n / d
    ll = np.log(lr1 / lr0) if lr0 > 0 and lr1 > 0 else 0.0
    lt = np.log(T1 / T0) if T0 > 0 and T1 > 0 else 0.0
    for s in range(steps):
        frac = s / (steps - 1) if steps > 1 else 0.0
        T = T0 * np.exp(lt * frac)
        lr = lr0 * np.exp(ll * frac)
        lam = lam0 * (1.0 - frac)
        mu = mu0 * (1.0 - frac)
        G = Y @ Y.T
        np.fill_diagonal(G, -10.0)
        gm = float(G.max())
        E = np.exp((G - gm) / max(T, 1e-12))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            break
        Gr = 2.0 * (E @ Y) / S
        if lam > 0:
            # normalized load variance penalty
            r = E.sum(axis=1)  # (n,)
            # normalized fractions
            p = r / S
            pb = 1.0 / n
            c = 2.0 * (p - pb)  # (n,)
            A = E @ Y  # (n,d)
            Yc = c[:, None] * Y
            B = E @ Yc
            Gr_load = (c[:, None] * A + B) / (max(T, 1e-12) * max(S, 1e-12))
            Gr = Gr + lam * Gr_load
        if mu > 0:
            M = Y.T @ Y - a * np.eye(d)
            Gr_cov = 4.0 * (Y @ M)
            Gr = Gr + mu * Gr_cov
        Gt = Gr - np.sum(Gr * Y, axis=1, keepdims=True) * Y
        # Adam
        b1t *= b1
        b2t *= b2
        m1 *= b1
        m1 += (1 - b1) * Gt
        v2 *= b2
        v2 += (1 - b2) * (Gt * Gt)
        mh = m1 / (1 - b1t)
        vh = v2 / (1 - b2t)
        Y = Y - lr * mh / (np.sqrt(vh) + eps)
        Y = _norm(Y)
    return Y

def _lp_polish(x, t_end, max_constraints=1000):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = _norm(x)
    n, d = x.shape
    try:
        iu = np.triu_indices(n, 1)
    except Exception:
        return x
    best = _maxcos(x)
    radius = 1e-3
    while time.time() < t_end - 0.6 and radius > 1e-14:
        c = (x @ x.T)[iu]
        mcur = float(c.max())
        thr = max(20 * radius, 1e-9)
        sel = c > mcur - thr
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > max_constraints:
            idx = np.argpartition(-c, max_constraints)[:max_constraints]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
            cnt = max_constraints
        I, J = iu[0][sel], iu[1][sel]
        p, nv = cnt, n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if getattr(res, "status", 1) != 0:
            radius *= 0.5
            continue
        try:
            y = x + np.asarray(res.x[:-1]).reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y = _norm(y)
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _maxcos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    deadline = t0 + 165.0
    best_global = None
    bv_global = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (76, 15):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    v = _maxcos(a)
                    if v < bv_global:
                        bv_global = v
                        best_global = a.copy()
        except Exception:
            pass
    if best_global is None:
        rng0 = np.random.default_rng(0)
        best_global = _norm(rng0.normal(size=(76, 15)))
        bv_global = _maxcos(best_global)
    base = best_global.copy()
    base_val = bv_global

    # identify hubs and rattlers deterministically
    def get_hubs_rattlers(X):
        deg7, m7, G7 = _contacts(X, thr=1e-7)
        order = np.argsort(-deg7, kind="stable")
        hubs = [int(o) for o in order[:6]]
        # rattlers: zero contacts under 1e-7, else smallest degree
        ratt = [int(i) for i in range(76) if deg7[i] == 0]
        if len(ratt) < 6:
            # fill with smallest-degree indices not in hubs
            for o in order[::-1]:
                o = int(o)
                if o not in ratt and o not in hubs:
                    ratt.append(o)
                if len(ratt) >= 6:
                    break
        return hubs, ratt, deg7, m7
    hubs0, ratt0, deg0, m0 = get_hubs_rattlers(base)
    var0 = float(np.var(deg0))

    # deterministic trial configs: (step_base, n_move, hub_subset, seed)
    trials = [
        (0.08, 4, hubs0[:6], 100),
        (0.12, 6, hubs0[:6], 101),
        (0.05, 3, hubs0[:6], 102),
        (0.10, 5, hubs0[:6], 103),
    ]
    # alternative hub choice: use thr 1e-4 degrees for last trial if time
    try:
        deg4, _, _ = _contacts(base, thr=1e-4)
        order4 = np.argsort(-deg4, kind="stable")
        hubs_alt = [int(o) for o in order4[:6]]
    except Exception:
        hubs_alt = hubs0
    trials.append((0.07, 4, hubs_alt, 104))

    best_var = var0
    best_balanced = base.copy()
    cur_best = base.copy()
    cur_best_v = base_val

    for ti, (step_base, nmove, hubs, seed) in enumerate(trials):
        if time.time() > deadline - 30:
            break
        rng = np.random.default_rng(seed)
        # choose rattlers to move: first nmove of ratt0 (deterministic rotation)
        start = (ti * 2) % max(1, len(ratt0))
        rm = []
        for k in range(nmove):
            rm.append(int(ratt0[(start + k) % len(ratt0)]))
        # ensure hubs and rm disjoint
        rm = [r for r in rm if r not in hubs]
        if len(rm) == 0:
            rm = [int(ratt0[0])]
        try:
            Ys = _surgery(cur_best if ti < 2 else base, hubs, rm, step_base, rng)
        except Exception:
            continue
        # anneal with augmented objective, 2000 Adam steps, lambda/mu decay to 0
        try:
            # vary T slightly per trial deterministically
            T0 = 0.012 - 0.002 * (ti % 3)
            Ya = _anneal_aug(Ys, steps=2000, T0=T0, T1=0.0009,
                             lr0=0.02, lr1=0.002, lam0=1e-2, mu0=1e-2)
        except Exception:
            continue
        if not np.all(np.isfinite(Ya)):
            continue
        try:
            v = _maxcos(Ya)
            deg, _, _ = _contacts(Ya, thr=1e-7)
            var = float(np.var(deg))
            deg4b = _contacts(Ya, thr=1e-4)[0]
            var4 = float(np.var(deg4b))
        except Exception:
            continue
        # track balanced improvement
        if var < best_var - 1e-9:
            best_var = var
            best_balanced = Ya.copy()
        # track score improvement (primary)
        if v < cur_best_v - 1e-12 and np.all(np.isfinite(Ya)):
            cur_best = Ya.copy()
            cur_best_v = v
            if v < bv_global:
                best_global = Ya.copy()
                bv_global = v
        # also if variance dropped sharply, use it as next start even if score slightly worse
        # (rebalanced basin): allow small regression
        if var < 0.5 * (var0 + 1e-9) and v < cur_best_v + 0.005:
            # adopt as base for remaining trials
            cur_best = Ya.copy()
            # keep cur_best_v unchanged (don't mask score)
            try:
                hubs_new, ratt_new, _, _ = get_hubs_rattlers(Ya)[:2], None, None, None
            except Exception:
                pass
        if time.time() > deadline - 30:
            break

    # choose polishing start: best score found
    to_polish = best_global.copy() if bv_global <= cur_best_v else cur_best.copy()
    # if we found a much more balanced config with close score, prefer polishing balanced one too
    # polish both briefly if time: first score leader
    remain = deadline - time.time()
    if remain > 4:
        cap = min(15.0, remain - 2.0)
        t_end = time.time() + cap
        try:
            pol = _lp_polish(to_polish, t_end)
            vp = _maxcos(pol)
            if vp < bv_global and np.all(np.isfinite(pol)):
                best_global = pol.copy()
                bv_global = vp
        except Exception:
            pass
    # if time left, also polish balanced candidate
    remain2 = deadline - time.time()
    if remain2 > 6 and best_balanced is not None:
        try:
            vb = _maxcos(best_balanced)
            # only if within 0.01 of best (avoid wasting on bad basin)
            if vb < bv_global + 0.01:
                cap2 = min(8.0, remain2 - 2.0)
                t_end2 = time.time() + cap2
                pol2 = _lp_polish(best_balanced, t_end2)
                vp2 = _maxcos(pol2)
                if vp2 < bv_global and np.all(np.isfinite(pol2)):
                    best_global = pol2.copy()
                    bv_global = vp2
        except Exception:
            pass
    best_global = _norm(best_global)
    if not np.all(np.isfinite(best_global)) or best_global.shape != (76, 15):
        rngf = np.random.default_rng(1)
        best_global = _norm(rngf.normal(size=(76, 15)))
    return best_global
