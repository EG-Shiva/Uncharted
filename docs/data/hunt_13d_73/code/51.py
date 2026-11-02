import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, seconds, t_end):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 5e-4
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end and radius > 1e-14:
        Cfull = X @ X.T
        c = Cfull[iu0, iu1]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 2e-7)
        if np.count_nonzero(sel) > 2000:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:2000]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I, J = iu0[sel], iu1[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = I[:, None] * d + np.arange(d)
        cJ = J[:, None] * d + np.arange(d)
        ct = np.full((p, 1), n * d)
        cols = np.concatenate([cI, cJ, ct], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        nrm = np.linalg.norm(y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        y /= nrm
        vv = _max_cos(y)
        if vv < best:
            X, best = y, vv
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X, best

def _adam_lse(X0, s0, s1, steps, lr0, lr1, t_lim=None, check_every=200):
    n, d = X0.shape
    X = X0.copy()
    m = np.zeros((n, d))
    v = np.zeros((n, d))
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = b1, b2
    iu0, iu1 = np.triu_indices(n, 1)
    l0, l1 = np.log(lr0), np.log(lr1)
    bestX = X.copy()
    bestv = _max_cos(X)
    for k in range(steps):
        if t_lim is not None and (k % check_every == 0) and time.time() > t_lim:
            break
        frac = k / (steps - 1) if steps > 1 else 1.0
        s = s0 + (s1 - s0) * frac
        lr = np.exp(l0 + (l1 - l0) * frac)
        G = X @ X.T
        c = G[iu0, iu1]
        mx = float(c.max())
        if mx < bestv:
            bestv = mx
            bestX = X.copy()
        e = np.exp(s * (c - mx))
        se = float(e.sum())
        if not np.isfinite(se) or se <= 0:
            K = len(c)
            w = np.ones(K) / K
        else:
            w = e / se
        W = np.zeros((n, n))
        W[iu0, iu1] = w
        W[iu1, iu0] = w
        Grad = W @ X
        sprod = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - sprod * X
        m = b1 * m + (1 - b1) * Gt
        v = b2 * v + (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1p)
        vhat = v / (1 - b2p)
        b1p *= b1
        b2p *= b2
        X = X - lr * mhat / (np.sqrt(vhat) + eps)
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X /= nr
    fv = _max_cos(X)
    if fv < bestv:
        bestv = fv
        bestX = X.copy()
    return bestX, bestv

def _build_pool(rng, d=13):
    batches = []
    def make_batch(cnt, kvals):
        k = len(kvals)
        kvals_arr = np.array(kvals, dtype=float)
        R = rng.random((cnt, d))
        idx = np.argsort(R, axis=1)[:, -k:]
        signs = rng.choice(np.array([-1.0, 1.0]), size=(cnt, k))
        vals = np.tile(kvals_arr, (cnt, 1)) * signs
        Pb = np.zeros((cnt, d))
        rows = np.repeat(np.arange(cnt), k)
        cols = idx.ravel()
        Pb[rows, cols] = vals.ravel()
        nr = np.linalg.norm(Pb, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        Pb /= nr
        return Pb
    batches.append(make_batch(22000, [1, 1]))
    batches.append(make_batch(14000, [2, 1]))
    batches.append(make_batch(14000, [3, 1]))
    batches.append(make_batch(20000, [1, 1, 1]))
    batches.append(make_batch(14000, [2, 1, 1]))
    batches.append(make_batch(10000, [2, 2, 1]))
    batches.append(make_batch(8000, [3, 2, 1]))
    batches.append(make_batch(18000, [1, 1, 1, 1]))
    batches.append(make_batch(12000, [2, 1, 1, 1]))
    batches.append(make_batch(14000, [1, 1, 1, 1, 1]))
    batches.append(make_batch(12000, [1, 1, 1, 1, 1, 1]))
    # dense +-1
    cnt = 10000
    dense = rng.choice(np.array([-1.0, 1.0]), size=(cnt, d))
    dense /= np.sqrt(float(d))
    batches.append(dense)
    # random sphere
    cnt2 = 8000
    g = rng.standard_normal(size=(cnt2, d))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    batches.append(g)
    P = np.vstack(batches)
    # deduplicate roughly by rounding
    Pr = np.round(P, decimals=8)
    # use unique rows
    try:
        _, uidx = np.unique(Pr, axis=0, return_index=True)
        P = P[np.sort(uidx)]
    except Exception:
        pass
    return P

def solve():
    t0 = time.time()
    T_END = t0 + 165.0
    LP_END = t0 + 160.0
    SEARCH_END = t0 + 138.0
    n, d = 73, 13
    rng = np.random.default_rng(7)
    # load incumbent as fallback
    Xrec = None
    best_file = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            vv = np.load(fn)
            vv = np.asarray(vv, dtype=np.float64)
            if vv.shape == (n, d):
                vv = vv / np.linalg.norm(vv, axis=1, keepdims=True)
                if np.all(np.isfinite(vv)):
                    cc = _max_cos(vv)
                    if cc < best_file:
                        best_file = cc
                        Xrec = vv.copy()
        except Exception:
            pass
    if Xrec is None:
        Xrec = rng.standard_normal(size=(n, d))
        Xrec /= np.linalg.norm(Xrec, axis=1, keepdims=True)
        best_file = _max_cos(Xrec)
    Xbest = Xrec.copy()
    best = float(best_file)

    # ---- discrete pool ----
    try:
        P = _build_pool(rng, d=d)
    except Exception:
        P = None
    Xdisc = None
    if P is not None:
        try:
            M = P.shape[0]
            # greedy selection minimizing incremental max
            sel = np.zeros(n, dtype=np.int64)
            used = np.zeros(M, dtype=bool)
            cur_max = np.full(M, -2.0)
            # start from random index
            s0idx = int(rng.integers(0, M))
            sel[0] = s0idx
            used[s0idx] = True
            cur_max = P @ P[s0idx]
            cur_max[used] = np.inf
            Slist = [P[s0idx]]
            for k in range(1, n):
                if time.time() > SEARCH_END - 60:
                    # fill remaining randomly among best candidates
                    break
                nxt = int(np.argmin(cur_max))
                # if all inf (should not happen), pick random unused
                if not np.isfinite(cur_max[nxt]):
                    cand = np.where(~used)[0]
                    nxt = int(cand[rng.integers(0, len(cand))])
                sel[k] = nxt
                used[nxt] = True
                # update cur_max with new point
                dots = P @ P[nxt]
                np.maximum(cur_max, dots, out=cur_max)
                cur_max[used] = np.inf
                Slist.append(P[nxt])
            # if early break, fill rest
            filled = len(Slist)
            if filled < n:
                # pick remaining by argmin anyway quickly
                for k in range(filled, n):
                    nxt = int(np.argmin(cur_max))
                    if not np.isfinite(cur_max[nxt]):
                        cand = np.where(~used)[0]
                        nxt = int(cand[0])
                    sel[k] = nxt
                    used[nxt] = True
                    dots = P @ P[nxt]
                    np.maximum(cur_max, dots, out=cur_max)
                    cur_max[used] = np.inf
                    Slist.append(P[nxt])
            S = np.vstack(Slist)  # (73,13), already unit
            # swap repair using cached Gram
            # compute Gps (M x n)
            # do in chunks to avoid huge peak, use float64
            try:
                Gps = P @ S.T  # (M,n)
                Gss = S @ S.T
                # map pool index to selected position for exclusion (avoid re-pick of used? allow but discourage)
                # iterative improvement
                n_swap_iter = 60
                for it in range(n_swap_iter):
                    if time.time() > SEARCH_END - 45:
                        break
                    # current global max and worst pair
                    np.fill_diagonal(Gss, -2.0)
                    amax = float(Gss.max())
                    flat = int(np.argmax(Gss))
                    a = flat // n
                    b = flat % n
                    np.fill_diagonal(Gss, 1.0)
                    # internal max excluding each candidate r
                    # compute quickly: for r in {a,b}, internal = max of Gss over rows/cols != r
                    best_gain = 0.0
                    best_r = -1
                    best_m = -1
                    # precompute row max info of Gps for fast cand max? compute per r on the fly with vectorized ops
                    for r in (int(a), int(b)):
                        # internal excl
                        mask = np.ones(n, dtype=bool)
                        mask[r] = False
                        sub = Gss[np.ix_(mask, mask)]
                        # sub diagonal is 1.0, need ignore diagonal: set diag to -2
                        # instead compute max over off-diagonal: take max of sub with diag masked
                        # quick: fill diag of sub view? copy small (72x72) cheap
                        np.fill_diagonal(sub, -2.0)
                        internal = float(sub.max())
                        # restore (not needed since sub is copy? np.ix_ gives copy)
                        # candidate max excluding r
                        # row-wise max over columns != r
                        # use partition: compute max over all cols except r
                        Gex = np.delete(Gps, r, axis=1)  # copy M x 72 -- heavy (200k*72). Instead compute via max of two parts?
                        # To save memory, compute in chunks
                        # chunk it:
                        cand_max = np.empty(M, dtype=np.float64)
                        cand_max.fill(-2.0)
                        # vectorized without delete: max of (max over :r, max over r+1:)
                        # Gps is (M,n); compute left/right max
                        if r == 0:
                            cand_max = Gps[:, 1:].max(axis=1)
                        elif r == n - 1:
                            cand_max = Gps[:, :-1].max(axis=1)
                        else:
                            lmax = Gps[:, :r].max(axis=1)
                            rmax = Gps[:, r + 1:].max(axis=1)
                            cand_max = np.maximum(lmax, rmax)
                        total = np.maximum(cand_max, internal)
                        # forbid picking already-selected pool rows (except r itself): set total[used_other]=inf?
                        # build mask of pool rows used by other selected points
                        # map selected pool idx -> position; pool rows in sel[mask] are forbidden
                        # quick: set total for those rows to inf except possibly same row as r
                        for jj in range(n):
                            if jj == r:
                                continue
                            total[sel[jj]] = np.inf
                        mstar = int(np.argmin(total))
                        tstar = float(total[mstar])
                        gain = amax - tstar
                        if gain > best_gain:
                            best_gain = gain
                            best_r = r
                            best_m = mstar
                    if best_r >= 0 and best_gain > 1e-12:
                        # perform swap
                        r = best_r
                        mstar = best_m
                        old_pool = int(sel[r])
                        sel[r] = mstar
                        S[r] = P[mstar]
                        # update used flags: old may become unused
                        used[old_pool] = False
                        # check if mstar was already used elsewhere (should not due to inf), but handle
                        used[mstar] = True
                        # update Gps column r and Gss row/col r
                        Gps[:, r] = P @ S[r]
                        Gss[r, :] = S[r] @ S.T
                        Gss[:, r] = Gss[r, :]
                    else:
                        break
            except Exception:
                pass
            Xdisc = S / np.linalg.norm(S, axis=1, keepdims=True)
            dv = _max_cos(Xdisc)
            if dv < best:
                best = dv
                Xbest = Xdisc.copy()
        except Exception:
            Xdisc = None

    # ---- continuous relaxation of discrete code ----
    if Xdisc is not None and time.time() < SEARCH_END:
        try:
            Xa, va = _adam_lse(Xdisc, 30.0, 250.0, 1800, 0.03, 0.004, t_lim=SEARCH_END - 20)
            if va < best:
                best = va
                Xbest = Xa.copy()
            # second anneal hotter
            if time.time() < SEARCH_END - 10:
                Xb, vb = _adam_lse(Xa, 150.0, 600.0, 1800, 0.015, 0.002, t_lim=SEARCH_END)
                if vb < best:
                    best = vb
                    Xbest = Xb.copy()
                Xa = Xb
            # also try shaking the discrete start once more from different anneal if time
            if time.time() < SEARCH_END - 15:
                rng2 = np.random.default_rng(123)
                Xn = Xdisc + rng2.standard_normal(size=(n, d)) * 0.02
                Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
                Xc, vc = _adam_lse(Xn, 30.0, 300.0, 1500, 0.03, 0.003, t_lim=SEARCH_END)
                if vc < best:
                    best = vc
                    Xbest = Xc.copy()
        except Exception:
            pass

    # ---- short polish of incumbent-seeded Adam if discrete path failed to beat file ----
    if time.time() < SEARCH_END - 5:
        try:
            # brief Adam from best to squeeze a bit (cheap)
            Xd, vd = _adam_lse(Xbest, 80.0, 500.0, 1200, 0.015, 0.002, t_lim=SEARCH_END)
            if vd < best:
                best = vd
                Xbest = Xd.copy()
        except Exception:
            pass

    # ---- final LP polish under strict cutoff ----
    try:
        rem = LP_END - time.time()
        if rem > 3:
            secs = min(rem, 18.0)
            Xp, vp = _lp_polish(Xbest, secs, LP_END)
            if vp < best and np.all(np.isfinite(Xp)):
                Xbest, best = Xp, vp
    except Exception:
        pass

    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    # tiny shrink guard against boundary ties due to rounding (does not change score materially)
    return np.asarray(Xbest, dtype=np.float64)
