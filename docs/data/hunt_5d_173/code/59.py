import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _contacts(X, tol=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cnt = (G >= m - tol).sum(axis=1)
    return m, cnt, G


def _greedy_void(F, m, cand):
    # cand (Nc,d) unit, F (Nf,d) unit
    s_max = (cand @ F.T).max(axis=1)
    Nc = cand.shape[0]
    used = np.zeros(Nc, dtype=bool)
    s_cur = s_max.copy()
    chosen = []
    for _ in range(m):
        tmp = s_cur.copy()
        tmp[used] = 2.0
        idx = int(np.argmin(tmp))
        used[idx] = True
        pt = cand[idx].copy()
        chosen.append(pt)
        upd = cand @ pt
        np.maximum(s_cur, upd, out=s_cur)
    return np.stack(chosen, axis=0)


def _refine_block(M0, F, alphas=(50.0, 150.0), maxiter=150, deadline=None):
    m, d = M0.shape
    Nf = F.shape[0]
    # precompute triu indices for self pairs
    iu0, iu1 = np.triu_indices(m, 1)
    nself = len(iu0)
    Y0 = M0.copy().ravel()

    Ycur = Y0.copy()
    for alpha in alphas:
        if deadline is not None and time.time() > deadline:
            break
        def fun_grad(Yf):
            Y = Yf.reshape(m, d)
            nrm = np.linalg.norm(Y, axis=1, keepdims=True) + 1e-300
            M = Y / nrm
            Dc = M @ F.T  # m x Nf
            # self dots
            Ds_full = M @ M.T
            Ds = Ds_full[iu0, iu1]  # (nself,)
            # shift for stability
            mx_c = float(Dc.max()) if Dc.size else -2.0
            mx_s = float(Ds.max()) if Ds.size else -2.0
            shift = mx_c if mx_c > mx_s else mx_s
            Ec = np.exp(alpha * (Dc - shift))
            Es = np.exp(alpha * (Ds - shift))
            S = float(Ec.sum() + Es.sum()) + 1e-300
            f = shift + np.log(S) / alpha
            Wc = Ec / S
            Ws = Es / S
            # grad w.r.t M
            GM = Wc @ F  # m x d
            # self contributions
            if nself > 0:
                # for each pair (i,j): GM[i]+=Ws[k]*M[j], GM[j]+=Ws[k]*M[i]
                # vectorized via accumulation
                for k in range(nself):
                    i = int(iu0[k]); j = int(iu1[k])
                    w = float(Ws[k])
                    GM[i] += w * M[j]
                    GM[j] += w * M[i]
            # chain to Y
            GY = np.zeros_like(Y)
            for i in range(m):
                mi = M[i]
                gi = GM[i]
                dot = float(gi @ mi)
                GY[i] = (gi - dot * mi) / float(nrm[i, 0])
            return float(f), GY.ravel().copy()
        try:
            res = minimize(fun_grad, Ycur, method="L-BFGS-B", jac=True,
                           options={"maxiter": int(maxiter), "ftol": 1e-12, "gtol": 1e-8})
            Ycur = res.x.copy()
        except Exception:
            break
    Y = Ycur.reshape(m, d)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-300
    return Y


def _smooth_anneal(X, seconds, deadline):
    # short softmax repulsion on all points
    n, d = X.shape
    start = time.time()
    v = np.zeros_like(X)
    # schedule
    alphas = [40.0, 100.0, 200.0]
    iters = [120, 120, 120]
    lrs = [0.05, 0.03, 0.015]
    mu = 0.6
    for alpha, itn, lr0 in zip(alphas, iters, lrs):
        lr = float(lr0)
        for s in range(int(itn)):
            if time.time() - start > seconds or time.time() > deadline:
                return X
            G = X @ X.T
            W = np.exp(alpha * (G - 0.75))
            np.fill_diagonal(W, 0.0)
            rs = W.sum(axis=1, keepdims=True) + 1e-300
            A = (W @ X) / rs
            dot = (A * X).sum(axis=1, keepdims=True)
            Ft = A - dot * X
            v = mu * v + (1.0 - mu) * Ft
            X = X - lr * v
            X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-300
            lr *= 0.998
    return X


def _lp_polish(X, seconds, deadline):
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-300)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        try:
            c = (X @ X.T)[iu]
        except Exception:
            break
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = X + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True) + 1e-300
            vv = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if vv < best:
            X, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X


def solve():
    t0 = time.time()
    deadline = t0 + 168.0
    search_deadline = t0 + 115.0
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True) + 1e-300
    except Exception:
        xr = None
    try:
        xb = np.load("best.npy").astype(float)
        xb /= np.linalg.norm(xb, axis=1, keepdims=True) + 1e-300
    except Exception:
        xb = None
    # baseline
    base = None
    base_val = 2.0
    for c in (xb, xr):
        try:
            if c is not None and c.shape == (173, 5):
                c = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-300)
                if np.isfinite(c).all():
                    v = _max_cos(c)
                    if v < base_val:
                        base_val = v
                        base = c.copy()
        except Exception:
            continue
    if base is None:
        rng0 = np.random.RandomState(0)
        tmp = rng0.standard_normal((173, 5))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True) + 1e-300
        base = tmp
        base_val = _max_cos(base)
    # analyse rattler / contacts
    try:
        m0, cnt0, G0 = _contacts(base, tol=1e-9)
    except Exception:
        return base
    n, d = 173, 5
    try:
        r = int(np.argmin(cnt0))
    except Exception:
        r = 0
    try:
        # nearest to rattler (largest cosine)
        row = G0[r].copy()
        row[r] = -2.0
        order_near = list(np.argsort(-row))
        order_near = [int(i) for i in order_near if int(i) != r]
        order_con = list(np.argsort(-cnt0))
        order_con = [int(i) for i in order_con if int(i) != r]
    except Exception:
        order_near = [i for i in range(n) if i != r]
        order_con = [i for i in range(n) if i != r]

    removal_sets = []
    try:
        for mm in (4, 8, 12):
            k = mm - 1
            # A: nearest
            try:
                sA = [r] + order_near[:k]
                removal_sets.append(sA)
            except Exception:
                pass
            # B: highest contact
            try:
                sB = [r] + order_con[:k]
                removal_sets.append(sB)
            except Exception:
                pass
            # C: mix half-half disjoint
            try:
                k1 = (k + 1) // 2
                k2 = k - k1
                part1 = order_near[:k1]
                part2 = []
                for idx in order_con:
                    if idx not in part1 and idx != r:
                        part2.append(idx)
                    if len(part2) >= k2 + 2:
                        break
                part2 = part2[:k2]
                # fill if overlap caused shortage
                if len(part1) + len(part2) < k:
                    for idx in order_near[k1:]:
                        if idx not in part2 and idx not in part1:
                            part1.append(idx)
                        if len(part1) + len(part2) >= k:
                            break
                sC = [r] + list(part1) + list(part2)
                sC = sC[:mm]
                # ensure unique
                if len(set(sC)) == mm:
                    removal_sets.append(sC)
            except Exception:
                pass
    except Exception:
        pass
    # fallback if empty
    if len(removal_sets) == 0:
        removal_sets = [[r] + order_near[:3]]

    best_overall = base.copy()
    best_val = float(base_val)
    # keep pool of near candidates for anneal
    pool = []  # list of (val, X)
    seed_base = 12345
    trial = 0
    for s_idx, rem in enumerate(removal_sets):
        if time.time() > search_deadline:
            break
        # two restarts per set if time allows (different cand sample)
        for restart in range(2):
            if time.time() > search_deadline:
                break
            trial += 1
            try:
                rem_u = sorted(set(int(i) for i in rem))
                mm = len(rem_u)
                if mm < 4 or mm > 13:
                    continue
                mask = np.ones(n, dtype=bool)
                mask[rem_u] = False
                F = best_overall[mask].copy() if False else base[mask].copy()
                # frozen-frozen max (for reference)
                # generate 20000 candidates deterministically
                rs = np.random.RandomState(seed_base + trial * 1000 + restart)
                cand = rs.standard_normal((20000, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True) + 1e-300
                M0 = _greedy_void(F, mm, cand)
                # quick eval before refine
                # refine with time guard
                inner_deadline = min(search_deadline, time.time() + 8.0)
                Mref = _refine_block(M0, F, alphas=(50.0, 150.0), maxiter=120, deadline=inner_deadline)
                Xcand = base.copy()
                Xcand[rem_u] = Mref
                Xcand /= np.linalg.norm(Xcand, axis=1, keepdims=True) + 1e-300
                v = _max_cos(Xcand)
                if np.isfinite(v):
                    pool.append((float(v), Xcand.copy()))
                    # keep pool small sorted
                    pool = sorted(pool, key=lambda t: t[0])[:4]
                if v < best_val - 1e-12:
                    best_val = float(v)
                    best_overall = Xcand.copy()
                if best_val < 0.74:
                    break
            except Exception:
                continue
            # small time check
            if time.time() > search_deadline:
                break
        if best_val < 0.74:
            break

    # decide whether to unfreeze: need frozen optimum below or near 0.75856
    try:
        thresh = 0.75856 + 5e-4  # near
        # best pool value
        pool_best = min([p[0] for p in pool]) if pool else best_val
        do_unfreeze = (pool_best < thresh) or (best_val < base_val)
    except Exception:
        do_unfreeze = False

    # if no candidate beats record, return record per contract (avoid regression)
    # but still allow anneal if near
    if not do_unfreeze:
        # nothing promising: return baseline (which is record or better)
        return base / (np.linalg.norm(base, axis=1, keepdims=True) + 1e-300)

    # unfreeze: short smooth anneal + LP polish on top 1-2 candidates
    try:
        pool_sorted = sorted(pool, key=lambda t: t[0])[:2]
    except Exception:
        pool_sorted = [(best_val, best_overall)]
    final_best = best_overall.copy()
    final_val = best_val
    # ensure baseline included
    try:
        if base_val < final_val:
            final_best = base.copy()
            final_val = base_val
    except Exception:
        pass
    # anneal each candidate briefly then polish best
    anneal_sec = 12.0
    for (pv, px) in pool_sorted:
        try:
            if time.time() > deadline - 30.0:
                break
            remain_anneal = min(anneal_sec, max(2.0, (deadline - 25.0) - time.time()))
            if remain_anneal <= 0:
                break
            Xa = px.copy()
            Xa = _smooth_anneal(Xa, remain_anneal, deadline - 22.0)
            Xa /= np.linalg.norm(Xa, axis=1, keepdims=True) + 1e-300
            va = _max_cos(Xa)
            if va < final_val and np.isfinite(va):
                final_val = va
                final_best = Xa.copy()
        except Exception:
            continue
    # final LP polish on final_best with remaining time
    try:
        remain = (deadline - time.time()) - 2.0
        if remain < 3.0:
            remain = 3.0
        if remain > 32.0:
            remain = 32.0
        if time.time() < deadline - 3.0:
            final_best = _lp_polish(final_best, remain, deadline - 1.0)
            final_best /= np.linalg.norm(final_best, axis=1, keepdims=True) + 1e-300
            fv = _max_cos(final_best)
            if not np.isfinite(fv) or fv > final_val + 1e-9:
                # revert if polish hurt (should not)
                pass
            else:
                final_val = fv
    except Exception:
        pass
    # guard: if no improvement over loaded record, return record
    try:
        if final_val > base_val + 1e-12:
            return base / (np.linalg.norm(base, axis=1, keepdims=True) + 1e-300)
    except Exception:
        pass
    final_best = final_best / (np.linalg.norm(final_best, axis=1, keepdims=True) + 1e-300)
    # final sanity
    try:
        if not np.isfinite(final_best).all():
            return base
        if final_best.shape != (173, 5):
            return base
    except Exception:
        return base
    return final_best
