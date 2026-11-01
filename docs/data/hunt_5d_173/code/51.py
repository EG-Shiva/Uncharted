import time
import itertools
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

_N = 173
_D = 5

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, seconds):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        C = (X @ X.T)[iu0, iu1]
        m = float(C.max())
        sel = C > m - max(20.0 * radius, 1e-9)
        if int(sel.sum()) > 3500:
            idx = np.where(sel)[0]
            order = np.argsort(C[idx])[-3500:]
            m2 = np.zeros_like(sel)
            m2[idx[order]] = True
            sel = m2
        I = iu0[sel]; J = iu1[sel]
        c = C[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _anneal(X0, iu0, iu1, alphas, iters_list, lrs, deadline=None):
    n, d = X0.shape
    X = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
    m1 = np.zeros_like(X)
    v2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0
    for alpha, iters, lr in zip(alphas, iters_list, lrs):
        for _ in range(int(iters)):
            if deadline is not None and (step % 200 == 0) and time.time() > deadline:
                return X
            step += 1
            G = X @ X.T
            s = G[iu0, iu1]
            mx = float(s.max())
            e = np.exp(alpha * (s - mx))
            se = float(e.sum())
            if se == 0 or not np.isfinite(se):
                continue
            w = e / se
            W = np.zeros((n, n), dtype=np.float64)
            W[iu0, iu1] = w
            W = W + W.T
            grad = W @ X
            dot = (grad * X).sum(axis=1, keepdims=True)
            gt = grad - dot * X
            m1 = b1 * m1 + (1.0 - b1) * gt
            v2 = b2 * v2 + (1.0 - b2) * (gt * gt)
            bc1 = 1.0 - b1 ** step
            bc2 = 1.0 - b2 ** step
            mh = m1 / bc1
            vh = v2 / bc2
            D = mh / (np.sqrt(vh) + eps)
            dpro = (D * X).sum(axis=1, keepdims=True)
            Dt = D - dpro * X
            Xn = X - lr * Dt
            nn = np.linalg.norm(Xn, axis=1, keepdims=True)
            nn[nn == 0] = 1.0
            X = Xn / nn
    return X

def _build_pool(rng):
    pts = []
    # cross (1,0,0,0,0)
    for i in range(5):
        for s in (-1.0, 1.0):
            v = np.zeros(5); v[i] = s
            pts.append(v)
    # (1,1,0,0,0)/sqrt2 perms/signs
    for combo in itertools.combinations(range(5), 2):
        for signs in itertools.product((-1.0, 1.0), repeat=2):
            v = np.zeros(5); v[combo[0]] = signs[0]; v[combo[1]] = signs[1]
            v /= np.linalg.norm(v); pts.append(v.copy())
    # (1,1,1,0,0)/sqrt3
    for combo in itertools.combinations(range(5), 3):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            v = np.zeros(5)
            for k, idx in enumerate(combo):
                v[idx] = signs[k]
            v /= np.linalg.norm(v); pts.append(v.copy())
    # (1,1,1,1,0)/2
    for zero in range(5):
        for bits in range(16):
            v = np.ones(5); v[zero] = 0.0
            for j in range(5):
                if j == zero:
                    continue
                if (bits >> j) & 1:
                    v[j] = -1.0
            v /= np.linalg.norm(v); pts.append(v.copy())
    # hypercube
    for bits in range(32):
        v = np.ones(5)
        for j in range(5):
            if (bits >> j) & 1:
                v[j] = -1.0
        v /= np.linalg.norm(v); pts.append(v.copy())
    # integer lattice shell norm^2<=6 in [-2,2]^5
    for tup in itertools.product((-2, -1, 0, 1, 2), repeat=5):
        if tup[0]==0 and tup[1]==0 and tup[2]==0 and tup[3]==0 and tup[4]==0:
            continue
        s = tup[0]*tup[0]+tup[1]*tup[1]+tup[2]*tup[2]+tup[3]*tup[3]+tup[4]*tup[4]
        if s <= 6:
            v = np.array(tup, dtype=float); v /= np.linalg.norm(v)
            pts.append(v)
    # Fourier-style: (1,cos t, cos2t, sin t, sin2t) normalized variants
    for k in range(600):
        t = 2.0*np.pi*k/600.0 + 0.13
        for ph in (0.0, 0.7):
            v = np.array([1.0, np.cos(t+ph), np.cos(2*t+ph), np.sin(t+ph), np.sin(2*t+ph)], dtype=float)
            v /= np.linalg.norm(v); pts.append(v.copy())
    # random spherical-design points
    R = rng.standard_normal((2500, 5))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    for r in R:
        pts.append(r.copy())
    # cyclic orbit shifts of random bases
    C = rng.standard_normal((400, 5))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    for c in C:
        for sh in range(5):
            pts.append(np.roll(c, sh).copy())
    P = np.stack(pts, axis=0)
    Pr = np.round(P, 8)
    Pu = np.unique(Pr, axis=0).astype(np.float64)
    Pu /= np.linalg.norm(Pu, axis=1, keepdims=True)
    return Pu

def _greedy_select(pool, start_idx, n=_N):
    Np = pool.shape[0]
    sel_mask = np.zeros(Np, dtype=bool)
    selected = [int(start_idx % Np)]
    sel_mask[selected[0]] = True
    best_dot = pool @ pool[selected[0]]
    best_dot[selected[0]] = np.inf
    for _ in range(1, n):
        # argmin over unselected
        idx = int(np.argmin(best_dot))
        # best_dot of selected is inf so argmin avoids them unless all inf
        selected.append(idx)
        sel_mask[idx] = True
        d = pool @ pool[idx]
        # update
        np.maximum(best_dot, d, out=best_dot)
        best_dot[idx] = np.inf
        # keep selected at inf
        # (already)
    return pool[np.array(selected, dtype=int)].copy()

def _signature_penalty(X, thresh=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    contacts = (G > m - thresh).sum(axis=1)
    rattlers = int((contacts == 0).sum())
    return float(contacts.std() + 3.0 * rattlers), int(rattlers), float(contacts.mean())

def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline = t0 + BUDGET
    rng = np.random.default_rng(20261101)
    n, d = _N, _D
    iu0, iu1 = np.triu_indices(n, 1)

    # fallback record
    fallback = None
    fallback_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (n, d):
                an = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(an)
                if v < fallback_val:
                    fallback_val = v
                    fallback = an.copy()
        except Exception:
            pass
    if fallback is None:
        fallback = rng.standard_normal((n, d))
        fallback /= np.linalg.norm(fallback, axis=1, keepdims=True)
        fallback_val = _max_cos(fallback)

    # build pool
    try:
        pool = _build_pool(rng)
    except Exception:
        pool = rng.standard_normal((6000, d))
        pool /= np.linalg.norm(pool, axis=1, keepdims=True)
    Np = pool.shape[0]

    # structurally different starts: spread + lattice-region + random
    n_seeds = 9
    starts = []
    for k in range(n_seeds):
        if k < 5:
            starts.append(int(k * Np / 5))
        else:
            starts.append(int(rng.integers(0, Np)))
    # also force one start near cross polytope (index 0 area) and one near hypercube: use 0 and Np//2 already covered
    X0_list = []
    for s in starts:
        try:
            X0_list.append(_greedy_select(pool, s, n=n))
        except Exception:
            continue
    # add one pure-random greedy from random sub-pool to diversify: random pool only
    try:
        rand_pool = rng.standard_normal((4000, d))
        rand_pool /= np.linalg.norm(rand_pool, axis=1, keepdims=True)
        X0_list.append(_greedy_select(rand_pool, int(rng.integers(0, 4000)), n=n))
    except Exception:
        pass

    # Stage 1: short softmax continuation (soft->hard) for each seed
    short_al = [8.0, 40.0, 120.0, 260.0]
    short_it = [200, 200, 250, 250]
    short_lr = [0.012, 0.008, 0.005, 0.003]
    cands = []
    inner_dl = deadline - 45.0
    for X0 in X0_list:
        if time.time() > inner_dl:
            break
        try:
            y = _anneal(X0, iu0, iu1, short_al, short_it, short_lr, deadline=inner_dl)
            v = _max_cos(y)
            pen, ratt, mean_c = _signature_penalty(y)
            cands.append((v, pen, y))
        except Exception:
            continue
    if not cands:
        return fallback
    # sort by max cos, then balanced participation penalty
    cands.sort(key=lambda t: (t[0], t[1]))
    # Stage 2: harder continuation + LP polish on top few
    best = fallback.copy()
    best_val = fallback_val
    best_pen = 1e9
    # consider top 4 diverse
    top = cands[:4]
    for rank, (v0, pen0, y0) in enumerate(top):
        rem = deadline - time.time()
        if rem < 9.0:
            break
        # harder continuation: effective alpha 60->~400
        try:
            hard_al = [60.0, 150.0, 300.0, 450.0]
            hard_it = [250, 250, 300, 300]
            hard_lr = [0.008, 0.005, 0.003, 0.002]
            y1 = _anneal(y0, iu0, iu1, hard_al, hard_it, hard_lr, deadline=deadline - 22.0)
        except Exception:
            y1 = y0
        v1 = _max_cos(y1)
        rem2 = deadline - time.time()
        if rem2 < 7.0:
            # no time to polish, just compare
            pen1, _, _ = _signature_penalty(y1)
            if (v1 < best_val - 1e-12) or (abs(v1 - best_val) <= 1e-9 and pen1 < best_pen - 1e-9):
                if v1 < best_val + 1e-9:
                    best_val, best, best_pen = (v1 if v1 < best_val else best_val), (y1.copy() if v1 < best_val else best), pen1
            continue
        pt = 6.0 if rem2 > 30 else max(1.5, rem2 - 7.5)
        # give a bit more to first (most promising)
        if rank == 0:
            pt = min(pt + 2.0, rem2 - 6.0)
        pt = min(pt, rem2 - 6.0)
        if pt < 0.8:
            pt = max(0.8, rem2 - 6.0)
        try:
            y2 = _lp_polish(y1, pt)
        except Exception:
            y2 = y1
        v2 = _max_cos(y2)
        pen2, _, _ = _signature_penalty(y2)
        # select by final max cosine first, balanced signature as tie-break
        improved = False
        if v2 < best_val - 1e-12:
            improved = True
        elif abs(v2 - best_val) <= 2e-9 and pen2 < best_pen - 0.05 and v2 < best_val + 5e-9:
            improved = True
        if improved:
            best_val = v2 if v2 < best_val else best_val
            # if tie-break case where v2 slightly worse but much more balanced, still keep max-cos best; don't replace with worse
            if v2 < _max_cos(best) + 1e-12:
                best = y2.copy()
                best_val = v2
                best_pen = pen2
        # also track absolute best max-cos regardless of penalty for final output
        # (best already is absolute best due to first condition)
    # If de novo failed to beat fallback, return fallback (ensures record-level output)
    # else return de novo best
    # final safety polish on returned array if time remains
    rem = deadline - time.time()
    if rem > 1.6:
        try:
            bp = _lp_polish(best, max(0.6, rem - 0.9))
            bv = _max_cos(bp)
            if bv < _max_cos(best):
                best = bp
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
