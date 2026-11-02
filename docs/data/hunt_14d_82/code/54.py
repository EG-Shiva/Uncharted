import time
import itertools
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def build_pool(rng, cap=20000):
    lst = []
    # weight-2 all: 91*4=364
    for i, j in itertools.combinations(range(14), 2):
        for s1 in (-1.0, 1.0):
            for s2 in (-1.0, 1.0):
                v = np.zeros(14)
                v[i] = s1; v[j] = s2
                v /= np.sqrt(2.0)
                lst.append(v)
    # weight-3 all: 364*8=2912
    for i, j, k in itertools.combinations(range(14), 3):
        for mask in range(8):
            v = np.zeros(14)
            v[i] = 1.0 if (mask & 1) else -1.0
            v[j] = 1.0 if (mask & 2) else -1.0
            v[k] = 1.0 if (mask & 4) else -1.0
            v /= np.sqrt(3.0)
            lst.append(v)
    # weight-4 sampled 12000
    for _ in range(12000):
        idx = rng.choice(14, size=4, replace=False)
        s = rng.choice(np.array([-1.0, 1.0]), size=4)
        v = np.zeros(14)
        v[idx] = s
        v /= 2.0
        lst.append(v)
    # weight-5 sampled 2000
    for _ in range(2000):
        idx = rng.choice(14, size=5, replace=False)
        s = rng.choice(np.array([-1.0, 1.0]), size=5)
        v = np.zeros(14)
        v[idx] = s
        v /= np.sqrt(5.0)
        lst.append(v)
    # dense random 2000 to fill cap
    for _ in range(2000):
        v = rng.standard_normal(14)
        n = np.linalg.norm(v)
        if n == 0:
            v[0] = 1.0
            n = 1.0
        v /= n
        lst.append(v)
    P = np.stack(lst).astype(float)
    # cap safety
    if P.shape[0] > cap:
        P = P[:cap]
    return P

def greedy_select(pool, start_idx, n_select=82):
    P, d = pool.shape
    selected = [int(start_idx)]
    # max dot to selected set
    maxd = pool @ pool[start_idx]
    maxd[selected[0]] = 1e9  # exclude self
    for _ in range(1, n_select):
        # choose min maxd among non-selected
        # mask selected to large
        nxt = int(np.argmin(maxd))
        selected.append(nxt)
        # update
        nd = pool @ pool[nxt]
        # don't reselect
        nd[nxt] = 1e9
        # for already selected keep large
        maxd = np.maximum(maxd, nd)
        maxd[nxt] = 1e9
        for s in selected:
            maxd[s] = 1e9
    X = pool[np.array(selected)]
    return _norm(X.copy())

def adam_smoothmax(X_init, steps, lr0, lr1, tau0, tau1, deadline=None):
    X = _norm(X_init.copy())
    n, d = X.shape
    m1 = np.zeros_like(X); m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    import math
    l0, l1 = math.log(lr0), math.log(lr1)
    lt0, lt1 = math.log(tau0), math.log(tau1)
    for s in range(steps):
        if deadline is not None and s % 50 == 0 and time.time() > deadline:
            break
        frac = s / max(steps - 1, 1)
        lr = float(np.exp(l0 + (l1 - l0) * frac))
        tau = float(np.exp(lt0 + (lt1 - lt0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        E = np.exp((G - m) / tau)
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E)) * 0.5
        if not np.isfinite(S) or S <= 0:
            break
        E /= (2.0 * S)
        Gr = E @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        b1p *= b1; b2p *= b2
        m1 = b1 * m1 + (1.0 - b1) * Gt
        m2 = b2 * m2 + (1.0 - b2) * (Gt * Gt)
        mh = m1 / (1.0 - b1p); vh = m2 / (1.0 - b2p)
        X = X - lr * (mh / (np.sqrt(vh) + eps))
        X = _norm(X)
    return X

def lp_polish(X, deadline, radius_init=1.0e-3, max_pairs=1500, max_iter=30):
    X = _norm(X.copy())
    n, d = X.shape
    best = _max_cos(X)
    radius = float(radius_init)
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() < deadline - 0.5 and radius > 1e-14 and it < max_iter:
        it += 1
        G = X @ X.T
        c = G[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 2e-7)
        cnt = int(np.sum(sel))
        if cnt == 0:
            break
        if cnt > max_pairs:
            idx = np.argpartition(-c, max_pairs)[:max_pairs]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = sel & mask
        I = iu[0][sel]; J = iu[1][sel]
        cc = c[sel]
        p = len(I)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            C = np.empty((p, 2 * d + 1), dtype=int)
            cd = np.arange(d)
            C[:, :d] = I[:, None] * d + cd
            C[:, d:2 * d] = J[:, None] * d + cd
            C[:, 2 * d] = n * d
            cols = C.ravel()
            M = np.empty((p, 2 * d + 1))
            M[:, :d] = X[J]
            M[:, d:2 * d] = X[I]
            M[:, 2 * d] = -1.0
            vals = M.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 2.5})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X, best

def orbit_from_seeds(seeds):
    # seeds: (ns, d) -> stack cyclic rolls -> (ns*d, d)
    ns, d = seeds.shape
    outs = []
    for s in range(ns):
        v = seeds[s]
        for k in range(d):
            outs.append(np.roll(v, k))
    return np.stack(outs)

def drop_to_n(X, target=82):
    X = _norm(X.copy())
    while X.shape[0] > target:
        n = X.shape[0]
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        per = G.max(axis=1)
        v = int(np.argmax(per))
        X = np.delete(X, v, axis=0)
    return X

def optimise_orbit(rng, deadline, steps=350):
    d = 14
    ns = 6  # 6*14=84 -> drop 2 to 82
    seeds = _norm(rng.standard_normal((ns, d)))
    m1 = np.zeros_like(seeds); m2 = np.zeros_like(seeds)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    lr0, lr1 = 0.02, 0.002
    tau0, tau1 = 0.004, 0.0005
    import math
    l0, l1 = math.log(lr0), math.log(lr1)
    lt0, lt1 = math.log(tau0), math.log(tau1)
    for s in range(steps):
        if s % 40 == 0 and time.time() > deadline:
            break
        frac = s / max(steps - 1, 1)
        lr = float(np.exp(l0 + (l1 - l0) * frac))
        tau = float(np.exp(lt0 + (lt1 - lt0) * frac))
        # normalize seeds for orbit
        Sn = _norm(seeds)
        X = orbit_from_seeds(Sn)  # 84 x 14
        # gradient of LSE w.r.t X
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        E = np.exp((G - m) / tau)
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E)) * 0.5
        if not np.isfinite(S) or S <= 0:
            break
        E /= (2.0 * S)
        Gr = E @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X  # (84,d)
        # accumulate to seeds via inverse roll
        Gs = np.zeros_like(seeds)
        idx = 0
        for si in range(ns):
            acc = np.zeros(d)
            for k in range(d):
                g = Gt[idx]
                acc += np.roll(g, -k)
                idx += 1
            Gs[si] = acc
        # project to tangent of seeds
        dot2 = np.sum(Gs * Sn, axis=1, keepdims=True)
        Gs = Gs - dot2 * Sn
        b1p *= b1; b2p *= b2
        m1 = b1 * m1 + (1.0 - b1) * Gs
        m2 = b2 * m2 + (1.0 - b2) * (Gs * Gs)
        mh = m1 / (1.0 - b1p); vh = m2 / (1.0 - b2p)
        seeds = Sn - lr * (mh / (np.sqrt(vh) + eps))
    Sn = _norm(seeds)
    X = orbit_from_seeds(Sn)
    X82 = drop_to_n(X, 82)
    return X82

def solve():
    t0 = time.time()
    HARD = t0 + 165.0
    n, d = 82, 14
    rng = np.random.default_rng(7)
    np.random.seed(7)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        Xb = _norm(rng.standard_normal((n, d)))
    else:
        cands.sort(key=lambda c: _max_cos(c))
        Xb = cands[0].copy()
    best_val = _max_cos(Xb)
    X_best = Xb.copy()

    # --- structured pool (cap ~20000) ---
    try:
        pool = build_pool(np.random.default_rng(0))
    except Exception:
        pool = _norm(rng.standard_normal((19000, d)))
    # greedy with 2 starts
    Xg_best = None; vg_best = 1e9
    starts = [0, pool.shape[0] // 2, 123]
    for st in starts:
        if time.time() > HARD - 100:
            break
        try:
            Xg = greedy_select(pool, start_idx=int(st) % pool.shape[0])
            vg = _max_cos(Xg)
            if vg < vg_best:
                vg_best = vg; Xg_best = Xg
        except Exception:
            continue
    Xs_struct = Xg_best
    vs_struct = vg_best
    # short smooth-max Adam on greedy result
    if Xs_struct is not None and time.time() < HARD - 90:
        try:
            Xa = adam_smoothmax(Xs_struct, steps=450, lr0=0.02, lr1=0.002,
                                tau0=0.005, tau1=0.0006,
                                deadline=HARD - 80)
            va = _max_cos(Xa)
            if va < vs_struct:
                Xs_struct, vs_struct = Xa, va
        except Exception:
            pass
    # orbit basin (low-dim seed optimisation)
    if time.time() < HARD - 70:
        try:
            Xo = optimise_orbit(np.random.default_rng(1), deadline=HARD - 60, steps=350)
            # quick Adam release
            Xo2 = adam_smoothmax(Xo, steps=250, lr0=0.015, lr1=0.002,
                                 tau0=0.004, tau1=0.0006,
                                 deadline=HARD - 55)
            vo2 = _max_cos(Xo2)
            vo = _max_cos(Xo)
            if vo2 < vo:
                Xo, vo = Xo2, vo2
            if vo < vs_struct:
                Xs_struct, vs_struct = Xo, vo
        except Exception:
            pass
    # LP polish structured candidate (short)
    if Xs_struct is not None and time.time() < HARD - 40:
        try:
            Xs_p, vs_p = lp_polish(Xs_struct, deadline=min(HARD - 25, time.time() + 14.0),
                                   radius_init=1.0e-3, max_pairs=1500, max_iter=18)
            Xs_struct, vs_struct = Xs_p, vs_p
        except Exception:
            pass
    # light LP polish incumbent for fair compare (short)
    if time.time() < HARD - 20:
        try:
            Xb_p, vb_p = lp_polish(X_best, deadline=min(HARD - 12, time.time() + 8.0),
                                   radius_init=8e-4, max_pairs=1500, max_iter=10)
            if vb_p < best_val:
                X_best, best_val = Xb_p, vb_p
        except Exception:
            pass
    # final choice under 120s comparison spirit
    if Xs_struct is not None:
        Xs_struct = _norm(Xs_struct)
        # guard finite
        if np.all(np.isfinite(Xs_struct)) and vs_struct < best_val:
            X_best, best_val = Xs_struct, vs_struct
    # final tiny polish on winner if time remains
    if time.time() < HARD - 8:
        try:
            Xf, vf = lp_polish(X_best, deadline=HARD - 2.0, radius_init=5e-4,
                               max_pairs=1500, max_iter=8)
            if vf < best_val and np.all(np.isfinite(Xf)):
                X_best, best_val = Xf, vf
        except Exception:
            pass
    X_best = _norm(X_best)
    return X_best
