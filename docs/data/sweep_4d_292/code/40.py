import time
import numpy as np

def _normed(A):
    n = np.sqrt((A*A).sum(axis=1, keepdims=True))
    n = np.maximum(n, 1e-12)
    return A / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contact_hash(X):
    m = _max_cos(X)
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    thr = m - 2e-4
    deg = (G > thr).sum(axis=1)
    deg = tuple(sorted(deg.tolist()))
    return (round(m, 5), deg)

def _pnorm_opt(X0, p, maxiter, m0=None):
    from scipy.optimize import minimize
    N = X0.shape[0]
    if m0 is None:
        m0 = _max_cos(X0)
        if m0 <= 1e-6:
            m0 = 0.9
    inv_m0 = 1.0 / m0
    y0 = X0.ravel().copy()
    pinv = 1.0 / float(p)
    def fg(yy):
        Y = yy.reshape(N, 4)
        nr = np.sqrt((Y*Y).sum(axis=1, keepdims=True))
        nr = np.maximum(nr, 1e-12)
        S = Y / nr
        # chunked Gram for S_sum (block to limit memory)
        BS = 256
        S_sum = 0.0
        # first pass need D for R^p sum; do full for N~300 (small) but chunked
        # compute in blocks
        Rpow = None
        # store D blocks for gradient: just compute full D (300x300 trivial)
        D = S @ S.T
        np.fill_diagonal(D, 0.0)
        # positive part
        Pos = D > 0
        R = np.zeros_like(D)
        R[Pos] = D[Pos] * inv_m0
        # R^p, avoid overflow: clip R to 2.0
        Rp = np.zeros_like(D)
        # power only where Pos
        with np.errstate(over='ignore', invalid='ignore'):
            Rp[Pos] = np.power(R[Pos], p)
        # sum upper triangle
        iu = np.triu_indices(N, 1)
        S_sum = float(Rp[iu].sum())
        if S_sum <= 0 or not np.isfinite(S_sum):
            return 0.0, np.zeros_like(yy)
        f = S_sum ** pinv
        coeff = (S_sum ** (pinv - 1.0)) * inv_m0
        with np.errstate(over='ignore', invalid='ignore'):
            W = np.zeros_like(D)
            W[Pos] = coeff * np.power(R[Pos], p - 1.0)
        # gradient w.r.t S
        GS = W @ S
        dot = (GS * S).sum(axis=1, keepdims=True)
        GY = (GS - dot * S) / nr
        # clip huge grads for stability at large p
        return float(f), GY.ravel()
    try:
        res = minimize(fg, y0, jac=True, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-6, 'maxfun': 100000})
        y = res.x
    except Exception:
        y = y0
    Y = y.reshape(N, 4)
    return _normed(Y)

def _local_relax(X, idx_var, p=64, maxiter=40):
    from scipy.optimize import minimize
    N = X.shape[0]
    idx_var = np.asarray(idx_var, dtype=int)
    K = len(idx_var)
    if K == 0:
        return X
    X = _normed(X)
    m0 = _max_cos(X)
    if m0 <= 1e-6:
        m0 = 0.9
    inv_m0 = 1.0 / m0
    pinv = 1.0 / float(p)
    mask_var = np.zeros(N, dtype=bool)
    mask_var[idx_var] = True
    y0 = X[idx_var].ravel().copy()
    def fg(yy):
        Yv = yy.reshape(K, 4)
        nr = np.sqrt((Yv*Yv).sum(axis=1, keepdims=True))
        nr = np.maximum(nr, 1e-12)
        Sv = Yv / nr
        Full = X.copy()
        Full[idx_var] = Sv
        D = Full @ Full.T
        np.fill_diagonal(D, 0.0)
        Pos = D > 0
        R = np.zeros_like(D)
        R[Pos] = D[Pos] * inv_m0
        iu = np.triu_indices(N, 1)
        with np.errstate(over='ignore', invalid='ignore'):
            Rp = np.zeros_like(D)
            Rp[Pos] = np.power(R[Pos], p)
        S_sum = float(Rp[iu].sum())
        if S_sum <= 0 or not np.isfinite(S_sum):
            return 0.0, np.zeros_like(yy)
        f = S_sum ** pinv
        coeff = (S_sum ** (pinv - 1.0)) * inv_m0
        with np.errstate(over='ignore', invalid='ignore'):
            W = np.zeros_like(D)
            W[Pos] = coeff * np.power(R[Pos], p - 1.0)
        Gfull = W @ Full
        Gv = Gfull[idx_var]
        dot = (Gv * Sv).sum(axis=1, keepdims=True)
        GY = (Gv - dot * Sv) / nr
        return float(f), GY.ravel()
    try:
        res = minimize(fg, y0, jac=True, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-5, 'maxfun': 20000})
        y = res.x
    except Exception:
        y = y0
    Yv = y.reshape(K, 4)
    X2 = X.copy()
    X2[idx_var] = _normed(Yv)
    return _normed(X2)

def _find_hole(X, rng, n_sample=200000, refine=250):
    N = X.shape[0]
    best_s = None
    best_m = 2.0
    BS = 10000
    # chunked sampling
    n_done = 0
    while n_done < n_sample:
        b = min(BS, n_sample - n_done)
        S = rng.standard_normal((b, 4))
        S = _normed(S)
        # chunked max: S @ X.T in one go (b x N small)
        D = S @ X.T
        m = D.max(axis=1)
        i = int(np.argmin(m))
        if float(m[i]) < best_m:
            best_m = float(m[i])
            best_s = S[i].copy()
        n_done += b
    s = best_s.copy()
    # local ascent: minimize max dot
    lr = 0.05
    cur = float((X @ s).max())
    for it in range(refine):
        dots = X @ s
        j = int(np.argmax(dots))
        g = X[j] - dots[j] * s
        gn = float(np.linalg.norm(g))
        if gn < 1e-12:
            # perturb randomly
            g = rng.standard_normal(4)
            g -= (g @ s) * s
            gn = float(np.linalg.norm(g)) + 1e-12
        g /= gn
        # try step
        ns = s - lr * g
        ns /= max(np.linalg.norm(ns), 1e-12)
        nm = float((X @ ns).max())
        if nm < cur:
            s, cur = ns, nm
            lr *= 1.05
        else:
            lr *= 0.5
            if lr < 1e-6:
                break
    return s

def _victim_index(X):
    m = _max_cos(X)
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    # loose-contact degree with delta 0.01
    thr = m - 0.01
    deg = (G > thr).sum(axis=1)
    # dual stress approx: sum of (G - thr)+ over contacts
    stress = np.maximum(G - thr, 0).sum(axis=1)
    # primary degree, secondary stress
    idx = int(np.argmax(deg * 1000.0 + stress))
    # if all zero degree, pick point with max individual max
    if deg.max() == 0:
        idx = int(np.argmax(G.max(axis=1)))
    return idx

def _anneal(X, steps, rng, sigma0=0.01):
    X = _normed(X)
    best = _max_cos(X)
    bestX = X.copy()
    N = X.shape[0]
    T0 = 1e-4
    T1 = 1e-7
    for t in range(steps):
        frac = t / max(1, steps - 1)
        T = T0 * (T1 / T0) ** frac
        sig = sigma0 * (1.0 - 0.7 * frac)
        i = int(rng.integers(0, N))
        g = rng.standard_normal(4) * sig
        g -= (g @ X[i]) * X[i]
        cand = X.copy()
        cand[i] = cand[i] + g
        cand[i] /= max(np.linalg.norm(cand[i]), 1e-12)
        v = _max_cos(cand)
        if v < best or rng.random() < np.exp(-(v - best) / max(T, 1e-12)):
            X = cand
            if v < best:
                best = v
                bestX = X.copy()
    return _normed(bestX)

def _lp_polish(X, deadline):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return X
    X = _normed(X)
    n, d = X.shape
    best = _max_cos(X)
    radius = 8e-4
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        c = (X @ X.T)[iu]
        mval = float(c.max())
        sel = c > mval - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method='highs')
        except Exception:
            break
        if getattr(res, 'status', 1) != 0:
            radius *= 0.5
            continue
        try:
            y = X + res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y = _normed(y)
        vv = _max_cos(y)
        if vv < best:
            X, best, radius = y, vv, min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    TL = 172.0
    deadline = t0 + TL
    final_start = deadline - 30.0
    rng = np.random.default_rng(292004)
    # ---- Phase 1: independent random starts, variable N' ----
    N_list = [288, 290, 292, 294, 296]
    seeds_per = 2
    pool = []  # (maxcos, N', X, hash)
    gen_end = t0 + 78.0
    for Np in N_list:
        for s in range(seeds_per):
            if time.time() > gen_end:
                break
            X0 = _normed(rng.standard_normal((Np, 4)))
            # p-norm continuation 16->64->256
            try:
                X1 = _pnorm_opt(X0, p=16, maxiter=70)
            except Exception:
                X1 = X0
            if time.time() > gen_end:
                # keep X1 anyway
                pass
            try:
                X2 = _pnorm_opt(X1, p=64, maxiter=70)
            except Exception:
                X2 = X1
            try:
                X3 = _pnorm_opt(X2, p=256, maxiter=60)
            except Exception:
                X3 = X2
            v = _max_cos(X3)
            h = _contact_hash(X3)
            pool.append((v, Np, X3, h))
        if time.time() > gen_end:
            break
    # keep best 3-4 distinct per N' (here <=2, dedup)
    kept = []
    for Np in N_list:
        items = [e for e in pool if e[1] == Np]
        items.sort(key=lambda e: e[0])
        uniq = []
        for e in items:
            dup = False
            for u in uniq:
                if abs(e[0] - u[0]) < 2e-4 and e[3][1] == u[3][1]:
                    dup = True
                    break
            if not dup:
                uniq.append(e)
            if len(uniq) >= 4:
                break
        kept.extend(uniq)
    # ---- Phase 2: hole surgery to exactly 292 ----
    adjusted = []
    surg_end = min(final_start - 8.0, t0 + 125.0)
    for (v, Np, X, h) in kept:
        if time.time() > surg_end:
            break
        Xc = X.copy()
        # insert or delete loop
        steps_needed = 292 - Np
        if steps_needed > 0:
            for k in range(steps_needed):
                if time.time() > surg_end:
                    break
                s = _find_hole(Xc, rng, n_sample=120000 if Np < 292 else 60000, refine=150)
                Xc = np.vstack([Xc, s.reshape(1, 4)])
                Xc = _normed(Xc)
                # local relax 20 nearest neighbours + inserted
                dots = (Xc[:-1] @ Xc[-1])
                nn = np.argsort(-dots)[:20]
                idx = np.append(nn, len(Xc) - 1)
                try:
                    Xc = _local_relax(Xc, idx, p=64, maxiter=30)
                except Exception:
                    pass
        elif steps_needed < 0:
            for k in range(-steps_needed):
                if time.time() > surg_end:
                    break
                vic = _victim_index(Xc)
                # neighbours before deletion = closest to victim
                dots = Xc @ Xc[vic]
                dots[vic] = -2.0
                nn = np.argsort(-dots)[:20]
                # delete (nn indices shift after deletion: recompute)
                keep = [i for i in range(Xc.shape[0]) if i != vic]
                # map nn to new indices
                nn2 = []
                for j in nn:
                    if j == vic:
                        continue
                    nj = j - (1 if j > vic else 0)
                    nn2.append(nj)
                Xc = Xc[keep]
                Xc = _normed(Xc)
                try:
                    Xc = _local_relax(Xc, np.array(nn2, dtype=int), p=64, maxiter=30)
                except Exception:
                    pass
        # short anneal + LP polish after reaching 292
        if Xc.shape[0] == 292:
            try:
                Xc = _anneal(Xc, steps=300, rng=rng, sigma0=0.008)
            except Exception:
                pass
            try:
                Xc = _lp_polish(Xc, min(time.time() + 4.0, surg_end + 4.0))
            except Exception:
                pass
            adjusted.append((_max_cos(Xc), Xc))
    if len(adjusted) == 0:
        # fallback: use best kept resized naively if time ran out
        for (v, Np, X, h) in kept:
            Xc = X.copy()
            if Np < 292:
                need = 292 - Np
                extra = _normed(rng.standard_normal((need, 4)))
                Xc = _normed(np.vstack([Xc, extra]))
            elif Np > 292:
                Xc = Xc[:292]
            if Xc.shape[0] == 292:
                adjusted.append((_max_cos(Xc), Xc))
            break
    adjusted.sort(key=lambda e: e[0])
    if len(adjusted) == 0:
        # ultimate fallback random
        Xf = _normed(rng.standard_normal((292, 4)))
        adjusted = [(_max_cos(Xf), Xf)]
    best_adj_val, best_adj = adjusted[0]
    # ---- Phase 3: final 30 s most promising vs incumbent ----
    # only now load incumbent
    inc = None
    inc_val = 2.0
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (292, 4) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if nr.min() > 1e-9:
                    a = a / nr
                    v = _max_cos(a)
                    if v < inc_val:
                        inc_val = v
                        inc = a.copy()
        except Exception:
            pass
    # choose overall best to polish
    if inc is not None and inc_val < best_adj_val:
        cur = inc.copy()
        cur_val = inc_val
    else:
        cur = best_adj.copy()
        cur_val = best_adj_val
    # also consider second best adjusted if very close (topology diversity)
    # spend final time polishing cur (and best_adj if different)
    try:
        cur = _lp_polish(cur, deadline - 0.5)
        cur_val = _max_cos(cur)
    except Exception:
        pass
    # if cur came from incumbent but adjusted is close, try polishing adjusted with remaining time
    if time.time() < deadline - 2.0 and inc is not None and cur is not best_adj:
        # we polished incumbent; check adjusted polished quickly if time left
        try:
            alt = _lp_polish(best_adj, deadline - 0.5)
            alt_val = _max_cos(alt)
            if alt_val < cur_val:
                cur, cur_val = alt, alt_val
        except Exception:
            pass
    cur = _normed(cur)
    return cur
