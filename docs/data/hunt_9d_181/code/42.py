import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _lp_polish(X, budget, t_dead=None):
    t0 = time.time()
    x = _norm(X.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    it = 0
    while True:
        if time.time() - t0 >= budget:
            break
        if t_dead is not None and time.time() >= t_dead:
            break
        if radius < 1e-14 or it > 25:
            break
        it += 1
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 1200:
            cs = c[sel]
            idx = np.argpartition(-cs, 1200)[:1200]
            I = I[idx]; J = J[idx]
            cb = cs[idx]
        else:
            cb = c[sel]
            p = len(I)
        nv = n * d + 1
        try:
            cols_mat = np.empty((p, 2 * d + 1), dtype=np.int64)
            cols_mat[:, :d] = I[:, None] * d + ar
            cols_mat[:, d:2 * d] = J[:, None] * d + ar
            cols_mat[:, 2 * d] = n * d
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = cols_mat.ravel()
            vals_mat = np.empty((p, 2 * d + 1))
            vals_mat[:, :d] = x[J]
            vals_mat[:, d:2 * d] = x[I]
            vals_mat[:, 2 * d] = -1.0
            vals = vals_mat.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cb, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            y = _norm(y)
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def _smooth_anneal(X, steps, T0, T1, lr0, lr1, mu=0.55):
    Y = X.copy()
    V = np.zeros_like(Y)
    lT0, lT1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    n = Y.shape[0]
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = Y @ Y.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        W = (G - m) / T
        np.clip(W, -50.0, 0.0, out=W)
        np.exp(W, out=W)
        np.fill_diagonal(W, 0.0)
        rs = np.sum(W, axis=1)
        WY = W @ Y
        r = np.sum(W * G, axis=1)
        inv = np.zeros_like(rs)
        msk = rs > 1e-12
        inv[msk] = 1.0 / rs[msk]
        Dir = WY * inv[:, None] - (r * inv)[:, None] * Y
        Dir[~msk] = 0.0
        V *= mu
        V += Dir
        Y -= lr * V
        Y = _norm(Y)
    return Y

def _build_pool(rng):
    d = 9
    P = []
    # signed axes
    for i in range(d):
        for s in (-1.0, 1.0):
            v = np.zeros(d); v[i] = s
            P.append(v)
    # weight-2: (+-1,+-1)/sqrt2
    import itertools
    for a, b in itertools.combinations(range(d), 2):
        for sa in (-1.0, 1.0):
            for sb in (-1.0, 1.0):
                v = np.zeros(d); v[a] = sa / np.sqrt(2.0); v[b] = sb / np.sqrt(2.0)
                P.append(v)
    # weight-3: all
    for a, b, c in itertools.combinations(range(d), 3):
        for mask in range(8):
            v = np.zeros(d)
            v[a] = 1.0 if (mask & 1) else -1.0
            v[b] = 1.0 if (mask & 2) else -1.0
            v[c] = 1.0 if (mask & 4) else -1.0
            v /= np.sqrt(3.0)
            P.append(v)
    P = np.array(P)  # 834
    # D9 shell: +-1/sqrt9 with even #minus
    D = []
    # enumerate 512, keep 256
    for mask in range(512):
        bits = [(mask >> k) & 1 for k in range(9)]
        if sum(bits) % 2 == 1:
            continue
        v = np.ones(9)
        for k in range(9):
            if bits[k]:
                v[k] = -1.0
        v /= 3.0
        D.append(v)
    D = np.array(D)
    P = np.vstack([P, D])
    # cyclic orbit pool from random reps
    reps = rng.standard_normal((200, 9))
    reps = _norm(reps)
    C = []
    for r in reps:
        for sh in range(9):
            C.append(np.roll(r, sh))
    C = np.array(C)
    P = np.vstack([P, C])
    # dedup roughly: round and unique
    Pr = np.round(P, 8)
    _, idx = np.unique(Pr, axis=0, return_index=True)
    idx = np.sort(idx)
    P = P[idx]
    P = _norm(P)
    return P

def _greedy_select(P, n_sel, rng):
    M = P.shape[0]
    start = int(rng.integers(0, M))
    sel = [start]
    selmask = np.zeros(M, dtype=bool)
    selmask[start] = True
    # scores = max dot to selected
    dots_max = P @ P[start]  # M
    dots_max[start] = np.inf
    for _ in range(1, n_sel):
        # pick argmin of dots_max among unselected
        # mask selected with inf already
        j = int(np.argmin(dots_max))
        sel.append(j)
        selmask[j] = True
        nd = P @ P[j]
        # update max
        dots_max = np.maximum(dots_max, nd)
        dots_max[selmask] = np.inf
    return np.array(sel, dtype=np.int64)

def _swap_sa(P, sel, steps, rng, t_end=None):
    M = P.shape[0]
    n = len(sel)
    X = P[sel].copy()
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    cur = float(np.max(G))
    best_sel = sel.copy()
    best = cur
    sel_set = set(sel.tolist())
    # for fast max-others: recompute per iteration max excluding i
    T0, T1 = 0.004, 0.0002
    for s in range(steps):
        if t_end is not None and s % 200 == 0 and time.time() >= t_end:
            break
        frac = s / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** frac
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, M))
        if j in sel_set:
            continue
        cand = P[j]
        # max dot of cand to others
        d = X @ cand
        d[i] = -2.0
        new_row_max = float(np.max(d))
        # max among unchanged pairs
        # compute max of G excluding row/col i
        # cheap: precompute? do masked max O(n^2) occasionally -> too heavy per step
        # Instead keep full max of others approximated by cur if worst pair doesn't involve i,
        # else recompute. Determine if cur pair involves i:
        # find argmax quickly? We maintain cur; check row i max
        row_i_max = float(np.max(G[i]))
        if row_i_max < cur - 1e-12:
            max_others = cur
        else:
            # need recompute excluding i
            Gm = G.copy()
            Gm[i, :] = -2.0; Gm[:, i] = -2.0
            max_others = float(np.max(Gm))
        new_max = new_row_max if new_row_max > max_others else max_others
        if new_max < cur or rng.random() < np.exp((cur - new_max) / max(T, 1e-9)):
            # accept
            sel_set.remove(int(sel[i])); sel_set.add(j)
            sel[i] = j
            X[i] = cand
            G[i, :] = X @ cand
            G[:, i] = G[i, :]
            G[i, i] = -2.0
            # need fix row i vs self and full symmetry: G[i,:] computed includes self dot=1 -> set -2
            # G[:,i] mirrored; diagonal already -2
            cur = float(np.max(G))
            if cur < best - 1e-12:
                best = cur
                best_sel = sel.copy()
    return best_sel, best

def _expand_orbits(reps, extra):
    nrep = reps.shape[0]
    X = np.empty((nrep * 9 + 1, 9))
    for k in range(nrep):
        r = reps[k]
        for sh in range(9):
            X[k * 9 + sh] = np.roll(r, sh)
    X[-1] = extra
    return _norm(X)

def _sym_anneal(reps0, extra0, steps, T0, T1, lr0, lr1, rng):
    reps = reps0.copy()
    extra = extra0.copy()
    nrep = reps.shape[0]
    Vr = np.zeros_like(reps)
    Ve = np.zeros_like(extra)
    lT0, lT1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    mu = 0.5
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        X = _expand_orbits(reps, extra)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        W = (G - m) / T
        np.clip(W, -50.0, 0.0, out=W)
        np.exp(W, out=W)
        np.fill_diagonal(W, 0.0)
        rs = np.sum(W, axis=1)
        WY = W @ X
        r = np.sum(W * G, axis=1)
        inv = np.zeros_like(rs)
        msk = rs > 1e-12
        inv[msk] = 1.0 / rs[msk]
        Dir = WY * inv[:, None] - (r * inv)[:, None] * X
        # map back to reps
        Gr = np.zeros_like(reps)
        for k in range(nrep):
            acc = np.zeros(9)
            for sh in range(9):
                dd = Dir[k * 9 + sh]
                acc += np.roll(dd, -sh)
            acc /= 9.0
            # tangentialize to rep
            acc -= np.dot(acc, reps[k]) * reps[k]
            Gr[k] = acc
        De = Dir[-1] - np.dot(Dir[-1], extra) * extra
        Vr *= mu; Vr += Gr
        Ve *= mu; Ve += De
        reps -= lr * Vr
        extra -= lr * Ve
        reps = _norm(reps)
        en = np.linalg.norm(extra); 
        if en == 0: extra = rng.standard_normal(9); en = np.linalg.norm(extra)
        extra = extra / en
    return reps, extra

def solve():
    t_start = time.time()
    T_LIM = 170.0
    rng = np.random.default_rng(7)
    n, d = 181, 9
    cands = []
    cvals = []
    # load seeds
    seed_best = None
    seed_val = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < seed_val:
                        seed_val = v
                        seed_best = a.copy()
        except Exception:
            pass
    if seed_best is not None:
        cands.append(seed_best); cvals.append(seed_val)

    # --- structured pool branch ---
    try:
        P = _build_pool(rng)
        sel = _greedy_select(P, n, rng)
        X0 = P[sel]
        v0 = _max_cos(X0)
        # swap SA on true max, time-capped ~25s
        sel2, v2 = _swap_sa(P, sel.copy(), 12000, rng, t_end=t_start + 35.0)
        X1 = P[sel2]
        v1 = _max_cos(X1)
        # continuous anneal of discrete pick
        if time.time() - t_start < 90.0:
            X1a = _smooth_anneal(X1, 700, 0.02, 0.0015, 0.06, 0.006, mu=0.6)
            v1a = _max_cos(X1a)
        else:
            X1a, v1a = X1, v1
        cands.append(X1a); cvals.append(v1a)
        struct_X = X1a; struct_v = v1a
    except Exception:
        struct_X = None; struct_v = np.inf

    # --- cyclic-orbit symmetric branch (20 orbits + 1) ---
    try:
        best_sym_X = None; best_sym_v = np.inf
        # two restarts, cheap
        for restart in range(2):
            if time.time() - t_start > 100.0:
                break
            r0 = _norm(rng.standard_normal((20, 9)))
            e0 = rng.standard_normal(9); e0 /= np.linalg.norm(e0)
            reps, extra = _sym_anneal(r0, e0, 900, 0.025, 0.002, 0.08, 0.008, rng)
            Xs = _expand_orbits(reps, extra)
            # break symmetry: short full anneal
            Xb = _smooth_anneal(Xs, 500, 0.015, 0.0012, 0.05, 0.005, mu=0.55)
            vb = _max_cos(Xb)
            if vb < best_sym_v:
                best_sym_v = vb; best_sym_X = Xb.copy()
        if best_sym_X is not None:
            cands.append(best_sym_X); cvals.append(best_sym_v)
    except Exception:
        pass

    # pick current best among cands
    order = np.argsort(np.array(cvals))
    best = cands[int(order[0])].copy()
    best_val = float(cvals[int(order[0])])

    # joint smooth polish of best structured/sym if time
    try:
        if time.time() - t_start < 120.0:
            # polish top-2 candidates briefly and keep winner
            for idx in list(order[:2]):
                if time.time() - t_start > 125.0:
                    break
                Xc = cands[int(idx)]
                # skip seed (already polished)
                if Xc is best and seed_best is not None and np.allclose(Xc, seed_best):
                    continue
                Xp = _smooth_anneal(Xc, 600, 0.012, 0.0008, 0.04, 0.004, mu=0.6)
                vp = _max_cos(Xp)
                if vp < best_val:
                    best, best_val = Xp.copy(), vp
    except Exception:
        pass

    # final LP polish with remaining time
    try:
        remain = T_LIM - (time.time() - t_start)
        if remain > 4.0:
            budget = min(28.0, remain - 2.0)
            Xr = _lp_polish(best, budget, t_dead=t_start + T_LIM - 1.5)
            vr = _max_cos(Xr)
            if vr <= best_val + 1e-12 and np.all(np.isfinite(Xr)):
                # accept if not worse (LP may tie)
                if vr < best_val:
                    best, best_val = Xr, vr
                else:
                    best = Xr
            else:
                # if worse, keep best
                pass
    except Exception:
        pass

    # safety: if somehow best is worse than seed, return seed
    if seed_best is not None:
        try:
            bv = _max_cos(best)
            if not np.isfinite(bv) or bv > seed_val:
                best = seed_best
        except Exception:
            best = seed_best
    best = _norm(best)
    # tiny shrink guard not needed; ensure finite
    best[~np.isfinite(best)] = 0.0
    # renormalize after fix
    nr = np.linalg.norm(best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    # replace zero rows with random
    for i in range(n):
        if np.linalg.norm(best[i]) < 1e-12 or not np.all(np.isfinite(best[i])):
            v = rng.standard_normal(d); v /= np.linalg.norm(v)
            best[i] = v
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
