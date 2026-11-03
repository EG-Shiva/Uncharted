import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _renorm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n < 1e-12] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(x, deadline):
    x = _renorm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    iu0, iu1 = np.triu_indices(n, 1)
    radius = 1e-3
    while time.time() < deadline and radius > 1e-13:
        G = x @ x.T
        cmat = G[iu0, iu1]
        m = float(cmat.max())
        thresh = max(20.0 * radius, 1e-9)
        sel = cmat > m - thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 2500:
            # keep tightest 2500
            idx_all = np.where(sel)[0]
            order = idx_all[np.argsort(cmat[idx_all])]  # ascending, tightest last
            keep = order[-2500:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[keep] = True
            cnt = 2500
        I = iu0[sel].astype(np.int64)
        J = iu1[sel].astype(np.int64)
        cact = cmat[sel]
        p = cnt
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.empty(p * (2 * d + 1), dtype=np.int64)
            vals = np.empty(p * (2 * d + 1), dtype=np.float64)
            # fill row by row via vectorization
            # block for I
            for t in range(p):
                base = t * (2 * d + 1)
                cols[base:base + d] = I[t] * d + np.arange(d)
                vals[base:base + d] = x[J[t]]
                cols[base + d:base + 2 * d] = J[t] * d + np.arange(d)
                vals[base + d:base + 2 * d] = x[I[t]]
                cols[base + 2 * d] = n * d
                vals[base + 2 * d] = -1.0
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            # equality: x_i . dx_i = 0
            erows = np.repeat(np.arange(n), d)
            ecols = np.arange(n * d)
            evals = x.ravel().copy()
            A_eq = coo_matrix((evals, (erows, ecols)), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cact, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0 or res.x is None or not np.all(np.isfinite(res.x)):
            radius *= 0.5
            continue
        dx = res.x[:-1].reshape(n, d)
        # clip
        dx = np.clip(dx, -radius, radius)
        y = _renorm(x + dx)
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best - 1e-15:
            x, best = y, v
            best_x = y.copy()
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return best_x, best

def _fixed_lse_refine_batch(cands, Xf, steps=200, K=600.0, lr0=0.02, lr1=0.002):
    # cands: (C, d), Xf: (m, d)
    C = cands.shape[0]
    out = np.empty_like(cands)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for c in range(C):
        y = cands[c].copy()
        nr = float(np.linalg.norm(y))
        if not np.isfinite(nr) or nr < 1e-12:
            y = Xf[0].copy()
            nr = float(np.linalg.norm(y))
        y = y / max(nr, 1e-12)
        m1 = np.zeros_like(y)
        v1 = np.zeros_like(y)
        ok = True
        for s in range(steps):
            frac = s / max(1, steps - 1)
            lr = lr0 * (lr1 / lr0) ** frac
            dvec = Xf @ y
            mval = float(dvec.max())
            w = np.exp(K * (dvec - mval))
            S = float(w.sum())
            if not np.isfinite(S) or S <= 0:
                break
            grad = (Xf.T @ w) / S
            dot = float(grad @ y)
            Gt = grad - dot * y
            if not np.all(np.isfinite(Gt)):
                break
            m1 = b1 * m1 + (1 - b1) * Gt
            v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
            bc1 = 1.0 - b1 ** (s + 1)
            bc2 = 1.0 - b2 ** (s + 1)
            upd = lr * (m1 / bc1) / (np.sqrt(v1 / bc2) + eps)
            y = y - upd
            nr = float(np.linalg.norm(y))
            if not np.isfinite(nr) or nr < 1e-12:
                ok = False
                break
            y = y / nr
        if not ok or not np.all(np.isfinite(y)):
            y = cands[c] / max(1e-12, float(np.linalg.norm(cands[c])))
        out[c] = y
    return out

def _build_candidates(Xf, old_pos, rng, total=4096):
    m, d = Xf.shape
    cands = []
    # 1) random sphere samples
    n_rand = 2200
    R = rng.standard_normal(size=(n_rand, d))
    R = _renorm(R)
    cands.append(R)
    # 2) antipodes of fixed set
    cands.append(_renorm((-Xf).copy()))
    # 3) normalized differences of former neighbours
    # neighbours: closest (largest dot) to old_pos
    dots = Xf @ old_pos
    n_nbr = min(32, m)
    nbr_idx = np.argpartition(-dots, n_nbr - 1)[:n_nbr]
    diffs = []
    for j in nbr_idx:
        v = old_pos - Xf[j]
        nv = float(np.linalg.norm(v))
        if nv > 1e-12 and np.isfinite(nv):
            diffs.append(v / nv)
        v2 = Xf[j] - old_pos
        nv2 = float(np.linalg.norm(v2))
        # v2 is -v, skip duplicate; instead use pairwise diffs
    # pairwise diffs among neighbours
    for a in range(len(nbr_idx)):
        for b in range(a + 1, len(nbr_idx)):
            v = Xf[nbr_idx[a]] - Xf[nbr_idx[b]]
            nv = float(np.linalg.norm(v))
            if nv > 1e-12:
                diffs.append(v / nv)
                diffs.append(-v / nv)
            if len(diffs) >= 700:
                break
        if len(diffs) >= 700:
            break
    if len(diffs) > 0:
        diffs = np.array(diffs[:700])
        cands.append(diffs)
    # 4) perturbations of old position
    n_pert = total  # will trim later
    sigmas = [0.005, 0.02, 0.06, 0.15, 0.35]
    per_list = []
    per_each = 220  # 5*220=1100
    for sig in sigmas:
        N = rng.standard_normal(size=(per_each, d))
        # tangent projection
        dot = (N * old_pos).sum(axis=1, keepdims=True)
        Nt = N - dot * old_pos
        Y = old_pos[None, :] + sig * Nt
        Y = _renorm(Y)
        per_list.append(Y)
    cands.append(np.vstack(per_list))
    C = np.vstack(cands)
    # adjust to exactly total
    if C.shape[0] >= total:
        # subsample deterministically: keep all antipodes/diffs/pert? just take strided + ensure random part
        # simple: take first total after shuffle with rng
        perm = rng.permutation(C.shape[0])[:total]
        C = C[perm]
    else:
        extra = total - C.shape[0]
        R2 = _renorm(rng.standard_normal(size=(extra, d)))
        C = np.vstack([C, R2])
    return C

def _reinsert_one(X_full, idx, rng):
    n, d = X_full.shape
    mask = np.ones(n, dtype=bool)
    mask[idx] = False
    Xf = X_full[mask]
    old_pos = X_full[idx].copy()
    C = _build_candidates(Xf, old_pos, rng, total=4096)
    # score by true max cosine to fixed set
    D = C @ Xf.T  # (4096, 93)
    scores = D.max(axis=1)
    best16_idx = np.argpartition(scores, 16 - 1)[:16]
    # order them by score for determinism
    best16_idx = best16_idx[np.argsort(scores[best16_idx])]
    C16 = C[best16_idx]
    # refine only those with 200 steps fixed-others LogSumExp Adam
    try:
        R16 = _fixed_lse_refine_batch(C16, Xf, steps=200, K=600.0, lr0=0.02, lr1=0.002)
    except Exception:
        R16 = C16
    D2 = R16 @ Xf.T
    scores2 = D2.max(axis=1)
    b = int(np.argmin(scores2))
    y_best = R16[b] / max(1e-12, float(np.linalg.norm(R16[b])))
    X_prop = X_full.copy()
    X_prop[idx] = y_best
    return X_prop

def solve():
    T0 = time.time()
    BUDGET = 168.0
    END = T0 + BUDGET
    DEADLINE_HARD = T0 + 165.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal(size=(94, 13))
    xb = _renorm(np.asarray(xb, dtype=float))
    if xb.shape != (94, 13):
        # fallback: reshape/pad deterministically
        tmp = rng.standard_normal(size=(94, 13))
        try:
            xr = np.load("record.npy").astype(float)
            xr = _renorm(xr)
            if xr.shape == (94, 13):
                tmp = xr
        except Exception:
            pass
        xb = tmp
    xb = _renorm(xb)
    try:
        xr = np.load("record.npy").astype(float)
        xr = _renorm(xr)
        if xr.shape == (94, 13):
            if _max_cos(xr) < _max_cos(xb):
                xb = xr.copy()
    except Exception:
        pass
    n, d = 94, 13
    best_X = xb.copy()
    best_v = _max_cos(best_X)
    # hub cycling state
    hub_ptr = 0
    attempt = 0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < DEADLINE_HARD - 6.0:
        attempt += 1
        if time.time() > END - 8.0:
            break
        X = best_X.copy()
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        # extract upper triangle for stats
        cfull = G[iu0, iu1]
        m = float(cfull.max())
        # top pair
        am = int(np.argmax(cfull))
        ti, tj = int(iu0[am]), int(iu1[am])
        # near-contact degrees
        # full matrix with diag -2 for per-point stats
        Gfull = X @ X.T
        np.fill_diagonal(Gfull, -2.0)
        slack_full = m - Gfull
        deg1e6 = np.count_nonzero(slack_full < 1e-6, axis=1)
        deg1e7 = np.count_nonzero(slack_full < 1e-7, axis=1)
        # exponential load with scale 1e-7
        # careful overflow: (cij - m)/1e-7 <= 0
        E = np.exp(np.clip((Gfull - m) / 1e-7, -50, 0))
        np.fill_diagonal(E, 0.0)
        esum = E.sum(axis=1)
        in_top = np.zeros(n, dtype=int)
        in_top[ti] = 1
        in_top[tj] = 1
        order = sorted(range(n), key=lambda i: (in_top[i], int(deg1e7[i]), float(esum[i])), reverse=True)
        # occasionally remove two points joined by near-max pair
        do_double = (attempt % 6 == 0)
        if do_double:
            a = int(order[hub_ptr % n])
            # partner: closest to a (max cosine)
            row = Gfull[a].copy()
            row[a] = -2.0
            b = int(np.argmax(row))
            # only treat as double if near-maximal (slack < 5e-7), else single
            if float(m - row[b]) > 5e-7:
                do_double = False
                hubs = [a]
            else:
                hubs = [a, b]
            hub_ptr += 1
        else:
            a = int(order[hub_ptr % n])
            hubs = [a]
            hub_ptr += 1
        cur_full = best_X.copy()
        # sequentially reinsert each hub
        prop = cur_full.copy()
        for h in hubs:
            if time.time() > DEADLINE_HARD - 6.0:
                break
            try:
                prop = _reinsert_one(prop, h, rng)
            except Exception:
                break
        # short 5 s LP polish of full configuration
        polish_deadline = min(time.time() + 5.0, DEADLINE_HARD - 1.0, END - 1.0)
        if polish_deadline <= time.time() + 0.5:
            break
        try:
            Xpol, vpol = _lp_polish(prop, polish_deadline)
        except Exception:
            continue
        vpol = _max_cos(Xpol)
        if vpol < best_v - 1e-15:
            best_X = Xpol.copy()
            best_v = vpol
            hub_ptr = 0  # restart from new hubs
        # else restore incumbent (do nothing) and continue cycling
        if time.time() > END - 8.0:
            break
    best_X = _renorm(best_X)
    # tiny margin shrink not needed; ensure finite
    best_X[~np.isfinite(best_X)] = 0.0
    # renormalize any zeroed rows deterministically
    for i in range(n):
        nr = float(np.linalg.norm(best_X[i]))
        if not np.isfinite(nr) or nr < 1e-12:
            v = rng.standard_normal(size=(d,))
            v /= max(1e-12, float(np.linalg.norm(v)))
            best_X[i] = v
    best_X = _renorm(best_X)
    return best_X
