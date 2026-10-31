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

def _rand_orth(rng):
    A = rng.standard_normal((4, 4))
    Q, R = np.linalg.qr(A)
    d = np.sign(np.diag(R))
    d[d == 0] = 1.0
    Q = Q * d
    return Q

def _base32():
    pts = []
    # (±1,±1,0,0)/sqrt2 : 24 points
    sq = 1.0 / np.sqrt(2.0)
    for i in range(4):
        for j in range(i + 1, 4):
            for si in (-1.0, 1.0):
                for sj in (-1.0, 1.0):
                    v = np.zeros(4)
                    v[i] = si * sq
                    v[j] = sj * sq
                    pts.append(v)
    # (±1,0,0,0) : 8 points
    for i in range(4):
        for s in (-1.0, 1.0):
            v = np.zeros(4)
            v[i] = s
            pts.append(v)
    return np.array(pts)  # 32x4

def _base600():
    pts = []
    for i in range(4):
        for s in (-1.0, 1.0):
            v = np.zeros(4)
            v[i] = s
            pts.append(v)
    for bits in range(16):
        v = np.ones(4) * 0.5
        for k in range(4):
            if (bits >> k) & 1:
                v[k] = -0.5
        pts.append(v)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    a = 0.5
    b = phi / 2.0
    c = 1.0 / (2.0 * phi)
    base = np.array([0.0, a, b, c])
    # even permutations
    seen = set()
    for p in itertools.permutations([0, 1, 2, 3]):
        # parity
        inv = 0
        for i in range(4):
            for j in range(i + 1, 4):
                if p[i] > p[j]:
                    inv += 1
        if inv % 2 == 1:
            continue
        vals = base[np.array(p)]
        # vals has zero at position where p==0; signs for non-zero
        nz = [k for k in range(4) if abs(vals[k]) > 1e-12]
        for mask in range(8):
            v = vals.copy()
            for bi, k in enumerate(nz):
                if (mask >> bi) & 1:
                    v[k] = -v[k]
            pts.append(v.copy())
    P = np.array(pts)
    # dedup
    Pr = np.round(P, 8)
    _, idx = np.unique(Pr, axis=0, return_index=True)
    P = P[np.sort(idx)]
    P = _norm(P)
    return P

def _greedy_once(P, t, order):
    m = P.shape[0]
    blocked = np.zeros(m, dtype=bool)
    sel = []
    for v in order:
        if blocked[v]:
            continue
        sel.append(int(v))
        dots = P @ P[v]
        blocked[dots > t] = True
    return sel

def _try_improve_12(P, t, sel, sel_set):
    # one pass of 1-for-2, return new sel if improved else None
    s = len(sel)
    if s == 0:
        return None
    Ps = P[np.array(sel)]
    M = P @ Ps.T
    conf = M > t
    counts = conf.sum(axis=1)
    # iterate j in random-ish order (use current order)
    for j in range(s):
        # candidates conflicting only with j
        mask = (~sel_set) & conf[:, j] & (counts == 1)
        idx = np.where(mask)[0]
        if len(idx) < 2:
            continue
        # find compatible pair
        k = len(idx)
        # cap large k
        if k > 60:
            # subsample first 60 (dots smallest vs others? just take first 60)
            idx = idx[:60]
            k = 60
        Pi = P[idx]
        Gs = Pi @ Pi.T
        # search pair
        found = None
        for a in range(k):
            row = Gs[a, a + 1:]
            b_rel = np.where(row <= t)[0]
            if len(b_rel) > 0:
                found = (int(idx[a]), int(idx[a + 1 + int(b_rel[0])]))
                break
        if found is not None:
            out = sel[j]
            new_sel = sel.copy()
            new_sel[j] = found[0]
            new_sel.append(found[1])
            return new_sel
    return None

def _refill(P, t, keep, rng):
    m = P.shape[0]
    sel_set = np.zeros(m, dtype=bool)
    sel_set[np.array(keep, dtype=int)] = True
    # blocked by keep
    if len(keep) == 0:
        blocked = np.zeros(m, dtype=bool)
    else:
        Mk = P @ P[np.array(keep)].T
        blocked = (Mk > t).any(axis=1)
        # keep nodes themselves are blocked but marked selected
    # order unselected randomly
    order = rng.permutation(m)
    sel = list(keep)
    for v in order:
        v = int(v)
        if sel_set[v]:
            continue
        if blocked[v]:
            continue
        sel.append(v)
        sel_set[v] = True
        dots = P @ P[v]
        blocked[dots > t] = True
    return sel

def _mis_for_t(P, t, budget, rng):
    m = P.shape[0]
    t_end = time.time() + budget
    best = []
    # initial restarts
    while time.time() < t_end:
        order = rng.permutation(m)
        sel = _greedy_once(P, t, order)
        # 1-for-2 loop
        sel_set = np.zeros(m, dtype=bool)
        sel_set[np.array(sel, dtype=int)] = True
        for _ in range(4):
            ns = _try_improve_12(P, t, sel, sel_set)
            if ns is None:
                break
            sel = ns
            sel_set = np.zeros(m, dtype=bool)
            sel_set[np.array(sel, dtype=int)] = True
            if len(sel) >= 168:
                break
        if len(sel) > len(best):
            best = sel
            if len(best) >= 168:
                break
        # kick-refill intensification on best
        if len(best) >= 40 and time.time() < t_end - 0.3:
            # few kicks
            for _ in range(3):
                if time.time() >= t_end:
                    break
                drop = min(3, len(best) // 10 + 1)
                didx = rng.choice(len(best), size=drop, replace=False)
                keep = [best[i] for i in range(len(best)) if i not in set(didx)]
                sel2 = _refill(P, t, keep, rng)
                # improve
                sel_set2 = np.zeros(m, dtype=bool)
                sel_set2[np.array(sel2, dtype=int)] = True
                for _2 in range(3):
                    ns = _try_improve_12(P, t, sel2, sel_set2)
                    if ns is None:
                        break
                    sel2 = ns
                    sel_set2 = np.zeros(m, dtype=bool)
                    sel_set2[np.array(sel2, dtype=int)] = True
                if len(sel2) > len(best):
                    best = sel2
                    if len(best) >= 168:
                        break
            if len(best) >= 168:
                break
        if len(best) >= 168:
            break
    return best

def _complete_voids(Xk, rng, n_target=168, n_sample=8000):
    Xk = Xk.copy()
    while Xk.shape[0] < n_target:
        C = rng.standard_normal((n_sample, 4))
        C = _norm(C)
        D = C @ Xk.T
        md = D.max(axis=1)
        bi = int(np.argmin(md))
        Xk = np.vstack([Xk, C[bi:bi + 1]])
    return Xk

def _lse_refine(X, rng_unused, deadline, steps=700, beta0=40.0, beta1=800.0, lr0=0.02, lr1=0.001, mu=0.9):
    X = _norm(X.copy())
    vel = np.zeros_like(X)
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    for s in range(steps):
        if time.time() > deadline - 0.3:
            break
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = lr0 * (1.0 - f) + lr1 * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cm = float(np.max(G))
        E = np.exp(np.clip(beta * (G - cm), -25.0, 0.0))
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True) + 1e-12
        grad = (E @ X) / rs
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        sc = float(np.mean(np.linalg.norm(gt, axis=1))) + 1e-12
        gt = gt / sc
        vel = mu * vel + (1.0 - mu) * gt
        X = X - lr * vel
        X = _norm(X)
    return X

def _lp_polish(X, deadline):
    X = _norm(X.copy())
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline - 0.5 and radius > 1e-14:
        G = X @ X.T
        c = G[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        csel = c[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    DEADLINE = t0 + 168.0
    SEARCH_END = t0 + 72.0
    rng = np.random.default_rng(7)
    # incumbent
    try:
        xb = np.load("best.npy").astype(float)
        xb = _norm(xb)
    except Exception:
        xb = _norm(rng.standard_normal((168, 4)))
    try:
        xr = np.load("record.npy").astype(float)
        xr = _norm(xr)
        if xr.shape == (168, 4) and _max_cos(xr) < _max_cos(xb):
            xb = xr.copy()
    except Exception:
        pass
    if xb.shape != (168, 4):
        xb = _norm(rng.standard_normal((168, 4)))
    best_val = _max_cos(xb)
    X_best = xb.copy()

    # build algebraic pools
    b32 = _base32()
    b600 = _base600()
    pools = []
    n_rot600 = 18
    for _ in range(n_rot600):
        Q = _rand_orth(rng)
        pools.append(b600 @ Q.T)
    n_rot32 = 25
    for _ in range(n_rot32):
        Q = _rand_orth(rng)
        pools.append(b32 @ Q.T)
    Pstruct = np.vstack(pools)
    Pstruct = _norm(Pstruct)
    Prand = _norm(rng.standard_normal((2000, 4)))
    P = np.vstack([Pstruct, Prand, X_best])
    P = _norm(P)
    # dedup very close duplicates to reduce waste (optional, keep)
    # thresholds binary-searched around 0.850-0.8593
    thresholds = [0.8592, 0.8585, 0.8575, 0.8560, 0.8540, 0.8520, 0.8500]
    total_t = len(thresholds)
    results = {}
    largest = []
    success = {}  # t -> sel
    for ti, t in enumerate(thresholds):
        if time.time() > SEARCH_END - 1.0:
            break
        remain = SEARCH_END - time.time()
        left = total_t - ti
        budget = max(2.0, min(9.0, remain / left))
        sel = _mis_for_t(P, t, budget, rng)
        results[t] = len(sel)
        if len(sel) > len(largest):
            largest = sel
        if len(sel) >= 168:
            success[t] = sel[:168]
            # continue to try lower t
    # choose construction to polish
    cands_to_polish = []
    if len(success) > 0:
        # lowest t first (best certificate)
        for t in sorted(success.keys()):
            Xc = P[np.array(success[t][:168])]
            cands_to_polish.append((t, Xc))
            if len(cands_to_polish) >= 2:
                break
    else:
        if len(largest) > 0:
            Xk = P[np.array(largest)]
            Xk = _complete_voids(Xk, rng, 168, n_sample=8000)
            cands_to_polish.append((1.0, Xk))
        else:
            Xk = _complete_voids(X_best[:10], rng, 168, n_sample=8000)
            cands_to_polish.append((1.0, Xk))
    # polish candidates with LSE + LP
    for _, Xc in cands_to_polish:
        if time.time() > DEADLINE - 12.0:
            break
        Xc = _norm(Xc)
        # LSE
        lse_dead = min(DEADLINE - 10.0, time.time() + 12.0)
        Xc = _lse_refine(Xc, rng, lse_dead, steps=650)
        # LP
        lp_dead = min(DEADLINE - 6.0, time.time() + 12.0)
        Xc = _lp_polish(Xc, lp_dead)
        v = _max_cos(Xc)
        if v < best_val:
            best_val = v
            X_best = Xc.copy()
    # final safety LP squeeze on incumbent best if time remains
    if time.time() < DEADLINE - 6.0:
        Xc = _lp_polish(X_best, DEADLINE - 2.0)
        v = _max_cos(Xc)
        if v < best_val:
            best_val = v
            X_best = Xc.copy()
    X_best = _norm(np.nan_to_num(X_best, nan=0.0, posinf=0.0, neginf=0.0))
    return X_best
