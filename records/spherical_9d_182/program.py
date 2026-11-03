import time
import itertools
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n < 1e-16] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _max_abs(U):
    G = U @ U.T
    np.fill_diagonal(G, 0.0)
    return float(np.max(np.abs(G)))

def build_pool():
    vecs = []
    vals = (-2, -1, 1, 2)
    for k in (1, 2, 3, 4):
        for pos in itertools.combinations(range(9), k):
            for prod in itertools.product(vals, repeat=k):
                v = np.zeros(9, dtype=np.float64)
                for p, a in zip(pos, prod):
                    v[p] = a
                n = np.linalg.norm(v)
                vecs.append(v / n)
    # (3,1^8)/sqrt17 all sign/permutation patterns
    s17 = np.sqrt(17.0)
    for p3 in range(9):
        for mask in range(512):
            v = np.empty(9, dtype=np.float64)
            for i in range(9):
                s = 1.0 if ((mask >> i) & 1) else -1.0
                if i == p3:
                    v[i] = 3.0 * s
                else:
                    v[i] = s
            vecs.append(v / s17)
    P = np.stack(vecs, axis=0)
    # dedup same direction (keep antipodes distinct)
    seen = set()
    keep = []
    for i in range(P.shape[0]):
        key = tuple(np.round(P[i], 12))
        if key not in seen:
            seen.add(key)
            keep.append(i)
    P = P[np.array(keep)]
    return P

def build_lines_pool(P):
    seen = set()
    keep = []
    for i in range(P.shape[0]):
        v = P[i].copy()
        # canonical sign: first significant coord positive
        for c in v:
            if abs(c) > 1e-12:
                if c < 0:
                    v = -v
                break
        key = tuple(np.round(v, 12))
        if key not in seen:
            seen.add(key)
            keep.append(i)
    return P[np.array(keep)]

def greedy_select(P, nsel, use_abs=False, start=0):
    M = P.shape[0]
    sel = [int(start % M)]
    selset = set(sel)
    s0 = P[sel[0]]
    if use_abs:
        best = np.abs(P @ s0)
    else:
        best = P @ s0
    best = best.copy()
    best[sel[0]] = np.inf
    # exclude exact duplicates of start for abs case handled via lines pool; for safety exclude dot>1-1e-12
    for _ in range(1, nsel):
        # pick min
        idx = int(np.argmin(best))
        sel.append(idx)
        selset.add(idx)
        sn = P[idx]
        if use_abs:
            nd = np.abs(P @ sn)
        else:
            nd = P @ sn
        # update
        np.maximum(best, nd, out=best)
        best[idx] = np.inf
        # also exclude exact duplicates (dot ~1) to avoid picking same direction twice
        if not use_abs:
            dup = np.where(nd > 1 - 1e-12)[0]
            for d in dup:
                if d not in selset:
                    best[d] = np.inf
    return np.array(sel, dtype=int)

def sel_value(S, use_abs=False):
    G = S @ S.T
    np.fill_diagonal(G, -2.0 if not use_abs else 0.0)
    if use_abs:
        return float(np.max(np.abs(G)))
    return float(np.max(G))

def sa_swap(P, sel, steps, use_abs, rng, T0=1e-4, T1=1e-6):
    S = P[sel].copy()
    cur = sel_value(S, use_abs)
    best_sel = sel.copy()
    best = cur
    M = P.shape[0]
    nsel = len(sel)
    selset = set(sel.tolist())
    for it in range(steps):
        frac = it / max(1, steps - 1)
        T = T0 * (T1 / max(T0, 1e-18)) ** frac
        T = max(T, 1e-12)
        i_sel = int(rng.randint(nsel))
        i_pool = int(rng.randint(M))
        if i_pool in selset:
            continue
        old_idx = int(sel[i_sel])
        # quick reject if candidate duplicates existing direction (non-abs)
        # build trial
        trial = S.copy()
        trial[i_sel] = P[i_pool]
        if use_abs:
            G = trial @ trial.T
            np.fill_diagonal(G, 0.0)
            nv = float(np.max(np.abs(G)))
        else:
            G = trial @ trial.T
            np.fill_diagonal(G, -2.0)
            nv = float(np.max(G))
        if nv < cur or rng.rand() < np.exp((cur - nv) / T):
            sel[i_sel] = i_pool
            selset.discard(old_idx)
            selset.add(i_pool)
            S = trial
            cur = nv
            if nv < best:
                best = nv
                best_sel = sel.copy()
    return best_sel, best

def adam_lse(X0, stages, deadline=1e18):
    X = _norm(X0.copy())
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    best = X.copy()
    bestv = _max_cos(X)
    for beta, lr, steps in stages:
        for _ in range(steps):
            if (t & 127) == 0 and time.time() > deadline:
                return best, bestv
            t += 1
            G = X @ X.T
            np.fill_diagonal(G, -10.0)
            mx = float(np.max(G))
            E = np.exp(beta * (G - mx))
            np.fill_diagonal(E, 0.0)
            S = float(np.sum(E))
            if not np.isfinite(S) or S <= 0:
                continue
            Grad = (E @ X) / S
            Gt = Grad - np.sum(Grad * X, axis=1, keepdims=True) * X
            m1 = b1 * m1 + (1.0 - b1) * Gt
            m2 = b2 * m2 + (1.0 - b2) * (Gt * Gt)
            c1 = 1.0 - b1 ** t
            c2 = 1.0 - b2 ** t
            upd = (m1 / c1) / (np.sqrt(m2 / c2) + eps)
            upd = upd - np.sum(upd * X, axis=1, keepdims=True) * X
            nm = np.linalg.norm(upd, axis=1, keepdims=True)
            sc = np.minimum(1.0, 2.0 / (nm + 1e-12))
            X = _norm(X - lr * upd * sc)
            if (t % 200) == 0:
                v = _max_cos(X)
                if v < bestv:
                    bestv = v
                    best = X.copy()
    v = _max_cos(X)
    if v < bestv:
        bestv = v
        best = X.copy()
    return best, bestv

def lp_polish(x, best, deadline):
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    radius = 8e-4
    stall = 0
    while time.time() < deadline and radius > 1e-14:
        Gf = x @ x.T
        cf = Gf[iu]
        m = float(cf.max())
        if m < best:
            best = m
        thresh = max(20.0 * radius, 4e-7)
        sel = cf > (m - thresh)
        p = int(sel.sum())
        if p < 800:
            kk = min(1200, cf.size)
            idx = np.argpartition(-cf, kk - 1)[:kk]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx] = True
            p = kk
        if p > 4000:
            idx = np.where(sel)[0]
            order = np.argsort(cf[idx])[-4000:]
            ns = np.zeros_like(sel, dtype=bool)
            ns[idx[order]] = True
            sel = ns
            p = 4000
        I = iu[0][sel]; J = iu[1][sel]
        cs = cf[sel]
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
            obj = np.zeros(n * d + 1); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            stall += 1
            if stall >= 6:
                rnd = np.random.randn(n, d) * 1e-6
                rnd -= (rnd * x).sum(axis=1, keepdims=True) * x
                x = _norm(x + rnd)
                stall = 0
            continue
        dx = res.x[:-1].reshape(n, d)
        improved = False
        for alpha in (1.0, 0.5, 0.25):
            y = _norm(x + alpha * dx)
            v = _max_cos(y)
            if v < best - 1e-12:
                best = v
                x = y
                radius = min(radius * 1.35, 1.2e-3)
                improved = True
                stall = 0
                break
        if not improved:
            radius *= 0.5
            stall += 1
            if stall >= 7:
                rnd = np.random.randn(n, d) * (radius * 0.5 + 8e-8)
                rnd -= (rnd * x).sum(axis=1, keepdims=True) * x
                yt = _norm(x + rnd)
                vt = _max_cos(yt)
                if vt < best + 3e-7:
                    x = yt
                    if vt < best:
                        best = vt
                stall = 0
    return x, best

def solve():
    np.random.seed(0)
    t0 = time.time()
    t_end = t0 + 168.0
    lp_deadline = t_end - 1.0
    best_x = None
    best_v = 1e99
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (182, 9):
                a = _norm(a)
                v = _max_cos(a)
                if v < best_v:
                    best_v = v
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _norm(np.random.randn(182, 9))
        best_v = _max_cos(best_x)
    incumbent = best_x.copy()
    inc_v = best_v

    # discrete pool
    try:
        P = build_pool()
    except Exception:
        P = None
    cands = []
    # greedy discrete 182
    if P is not None and time.time() < t_end - 60:
        try:
            rng = np.random.RandomState(0)
            sel = greedy_select(P, 182, use_abs=False, start=0)
            sel2, _ = sa_swap(P, sel.copy(), 2500, False, rng, T0=2e-4, T1=2e-6)
            Xd = _norm(P[sel2])
            cands.append(Xd)
            # second start for diversity if time allows
            if time.time() < t_end - 80:
                selb = greedy_select(P, 182, use_abs=False, start=P.shape[0] // 3)
                selb2, _ = sa_swap(P, selb.copy(), 1500, False, rng, T0=2e-4, T1=5e-6)
                cands.append(_norm(P[selb2]))
        except Exception:
            pass
    # antipodal 91 lines variant
    if P is not None and time.time() < t_end - 50:
        try:
            LP_ = build_lines_pool(P)
            rng2 = np.random.RandomState(1)
            lsel = greedy_select(LP_, 91, use_abs=True, start=0)
            lsel2, _ = sa_swap(LP_, lsel.copy(), 2000, True, rng2, T0=1e-3, T1=1e-5)
            U = LP_[lsel2]
            Xa = np.concatenate([U, -U], axis=0)
            Xa = _norm(Xa)
            cands.append(Xa)
        except Exception:
            pass

    # relax every promising discrete selection + incumbent
    stages_fast = [(120.0, 4e-4, 500), (500.0, 2.5e-4, 500), (1500.0, 1.2e-4, 400)]
    stages_inc = [(150.0, 3e-4, 400), (600.0, 2e-4, 400)]
    relaxed_best_x = incumbent.copy()
    relaxed_best_v = inc_v
    relax_deadline = min(t_end - 30, t0 + 150.0)
    for Xd in cands:
        if time.time() > relax_deadline:
            break
        try:
            Xr, vr = adam_lse(Xd, stages_fast, deadline=relax_deadline)
            if vr < relaxed_best_v:
                relaxed_best_v = vr
                relaxed_best_x = Xr.copy()
        except Exception:
            continue
    # incumbent relax
    if time.time() < relax_deadline:
        try:
            Xr, vr = adam_lse(incumbent, stages_inc, deadline=relax_deadline)
            if vr < relaxed_best_v:
                relaxed_best_v = vr
                relaxed_best_x = Xr.copy()
            # also update global best
            if vr < best_v:
                best_v = vr
                best_x = Xr.copy()
        except Exception:
            pass
    if relaxed_best_v < best_v:
        best_v = relaxed_best_v
        best_x = relaxed_best_x.copy()
    else:
        relaxed_best_x = best_x.copy()

    x = relaxed_best_x.copy()
    if time.time() < lp_deadline - 5:
        try:
            x, best_v = lp_polish(x, best_v, lp_deadline)
            best_x = x.copy()
        except Exception:
            pass
    best_x = _norm(best_x)
    # strict feasibility: no zero rows / NaN already handled
    return best_x
