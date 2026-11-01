import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _per_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1)

def _top_pairs_set(X, K=600):
    n = X.shape[0]
    iu = np.triu_indices(n, 1)
    c = (X @ X.T)[iu]
    if K >= len(c):
        idx = np.argsort(-c)
    else:
        part = np.argpartition(-c, K)[:K]
        idx = part[np.argsort(-c[part])]
    I = iu[0][idx]; J = iu[1][idx]
    s = set(zip(I.tolist(), J.tolist()))
    return s

def _jaccard(s1, s2):
    inter = len(s1 & s2)
    union = len(s1 | s2)
    if union == 0:
        return 1.0
    return inter / union

def _choose_removal(X, k, rng):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    pm = np.max(G, axis=1)
    # contacts with 1e-3 window for spread
    cc = np.sum(G >= m - 1e-3, axis=1)
    r = float(rng.random())
    if r < 0.38:
        # biased to high contacts / high per-max (24-28 analogue)
        order = np.argsort(-(pm + 0.0005 * cc))
        top = order[:max(16, 2 * k)]
        # weighted sample without replacement
        w = np.exp(8.0 * (pm[top] - pm[top].max())) + 0.1
        w = w / w.sum()
        sel = rng.choice(top, size=k, replace=False, p=w)
        return np.sort(sel)
    elif r < 0.58:
        # tightest cliques: collect points from top pairs
        iu = np.triu_indices(n, 1)
        c = (X @ X.T)[iu]
        nt = min(80, len(c))
        part = np.argpartition(-c, nt)[:nt]
        cand_pts = np.unique(np.concatenate([iu[0][part], iu[1][part]]))
        if len(cand_pts) < k:
            cand_pts = np.arange(n)
        # sample preferring higher pm
        w = pm[cand_pts]
        w = np.exp(10.0 * (w - w.max())) + 0.05
        w = w / w.sum()
        sel = rng.choice(cand_pts, size=k, replace=False, p=w)
        return np.sort(sel)
    elif r < 0.72:
        # spatially clustered subset
        seed = int(rng.integers(0, n))
        dots = G[seed].copy()
        dots[seed] = -2
        # nearest = largest cosine
        nn = np.argsort(-dots)[:k - 1]
        sel = np.concatenate([[seed], nn])
        return np.sort(sel.astype(int))
    else:
        sel = rng.choice(n, size=k, replace=False)
        return np.sort(sel)

def _single_fg_factory(Xr, beta):
    def fg(z):
        r = float(np.linalg.norm(z))
        if r < 1e-12:
            r = 1e-12
        y = z / r
        s = Xr @ y
        m = float(np.max(s))
        e = np.exp(beta * (s - m))
        S = float(np.sum(e)) + 1e-300
        F = m + np.log(S) / beta
        w = e / S
        gy = Xr.T @ w
        c = float(gy @ y)
        gz = (gy - c * y) / r
        return float(F), gz
    return fg

def _opt_single(Xr, y0, beta, maxiter=70):
    fg = _single_fg_factory(Xr, beta)
    try:
        res = minimize(fg, y0, jac=True, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-13, 'gtol': 1e-7, 'maxfun': maxiter * 2})
        z = res.x
        nrm = float(np.linalg.norm(z))
        if nrm < 1e-12 or not np.all(np.finite(z)):
            return y0 / (np.linalg.norm(y0) + 1e-300)
        return z / nrm
    except Exception:
        return y0 / (np.linalg.norm(y0) + 1e-300)

def _lse_full_fg(IU, beta):
    def fg(yf):
        Y = yf.reshape(68, 12)
        nr = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        X = Y / nr
        g = (X @ X.T)[IU]
        m = float(np.max(g))
        e = np.exp(beta * (g - m))
        S = float(np.sum(e)) + 1e-300
        F = m + np.log(S) / beta
        w = e / S
        P = np.zeros((68, 68))
        P[IU] = w
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return float(F), GY.ravel()
    return fg

def _run_full_lse(X0, betas, iters):
    y = (X0 / (np.linalg.norm(X0, axis=1, keepdims=True) + 1e-300)).ravel().copy()
    for beta, mi in zip(betas, iters):
        fg = _lse_full_fg(np.triu_indices(68, 1), float(beta))
        try:
            r = minimize(fg, y, jac=True, method='L-BFGS-B',
                         options={'maxiter': mi, 'ftol': 1e-14, 'gtol': 1e-6, 'maxfun': mi * 2, 'maxcor': 10})
            y = r.x.copy()
        except Exception:
            break
    Y = y.reshape(68, 12)
    Y /= (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-300)
    return Y

def _hinge_opt_inserted(Xf, Xi0, target, maxiter=150):
    m = Xf.shape[0]; k = Xi0.shape[0]
    d = Xf.shape[1]
    y0 = (Xi0 / (np.linalg.norm(Xi0, axis=1, keepdims=True) + 1e-300)).ravel().copy()
    def fg(yf):
        Y = yf.reshape(k, d)
        nr = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        Xi = Y / nr
        Gfi = Xf @ Xi.T
        Gii = Xi @ Xi.T
        iu = np.triu_indices(k, 1)
        g2 = Gii[iu] if k > 1 else np.zeros(0)
        v1 = Gfi - target
        p1 = np.where(v1 > 0, v1, 0.0)
        if k > 1:
            v2 = g2 - target
            p2 = np.where(v2 > 0, v2, 0.0)
            F = float(np.sum(p1 * p1) + np.sum(p2 * p2))
            W2 = np.zeros((k, k))
            wv = 2.0 * p2
            W2[iu] = wv
            W2 = W2 + W2.T
            grad_ii = W2 @ Xi
        else:
            F = float(np.sum(p1 * p1))
            grad_ii = np.zeros_like(Xi)
        W1 = 2.0 * p1  # m,k
        grad_fi = (Xf.T @ W1).T  # k,d
        GX = grad_fi + grad_ii
        dot = np.sum(GX * Xi, axis=1, keepdims=True)
        GY = (GX - dot * Xi) / nr
        return float(F), GY.ravel()
    try:
        r = minimize(fg, y0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-8, 'maxfun': maxiter * 2})
        Y = r.x.reshape(k, d)
        Y /= (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-300)
        return Y
    except Exception:
        return Xi0

def _lp_polish(x, deadline, r0=8e-4, cap_p=1800):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() > deadline - 0.4:
            break
        G = x @ x.T
        c = G[iu]
        mxx = float(np.max(c))
        sel = c > mxx - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        csel = c[sel]
        if p > cap_p:
            od = np.argsort(-csel)[:cap_p]
            I, J = I[od], J[od]
            csel = csel[od]
            p = cap_p
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        step = res.x[:-1].reshape(n, d)
        y = x + step
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best - 1e-15:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            ok = False
            for sh in (0.3, 0.1):
                yy = x + step * sh
                yy /= np.linalg.norm(yy, axis=1, keepdims=True)
                vv = _max_cos(yy)
                if vv < best:
                    x, best = yy, vv
                    ok = True
                    break
            if not ok:
                radius *= 0.5
    return x

def solve():
    t0 = time.time()
    T_END = t0 + 170.0
    LP_RESERVE = 14.0
    n, d = 68, 12
    rng = np.random.default_rng(20261031)
    # load incumbent
    bestX = None; best = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.finite(a)):
                    s = _max_cos(a)
                    if s < best:
                        best = s; bestX = a.copy()
        except Exception:
            pass
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        best = _max_cos(bestX)
    base_set = _top_pairs_set(bestX, 600)
    # diverse archive: list of (score, X, pairset)
    archive = [(best, bestX.copy(), base_set)]
    trial = 0
    NCAND = 200000
    while time.time() < T_END - LP_RESERVE:
        trial += 1
        # pick base: mostly best, sometimes diverse
        if len(archive) > 1 and float(rng.random()) < 0.30:
            bi = int(rng.integers(0, len(archive)))
            bscore, bX, _ = archive[bi]
        else:
            bscore, bX = best, bestX
            # archive[0] is best; keep sync
        k = int(rng.integers(2, 9))  # 2..8
        try:
            rem = _choose_removal(bX, k, rng)
        except Exception:
            rem = np.sort(rng.choice(n, size=k, replace=False))
        mask = np.ones(n, dtype=bool); mask[rem] = False
        Xf = bX[mask]  # (68-k,d)
        m0 = Xf.shape[0]
        # massive hole candidates
        try:
            cand = rng.standard_normal((NCAND, d))
        except Exception:
            break
        cand /= (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-300)
        # rank by max cosine to frozen
        try:
            S = cand @ Xf.T
            scores = np.max(S, axis=1)
        except Exception:
            continue
        cur_set = Xf.copy()
        cur_scores = scores.copy()
        inserted = []
        ok_trial = True
        for step in range(k):
            if time.time() > T_END - LP_RESERVE:
                ok_trial = False
                break
            # number to refine: 30 on first step, 12 later (meets >=30 overall)
            nref = 30 if step == 0 else 12
            nref = min(nref, NCAND)
            part = np.argpartition(cur_scores, nref)[:nref]
            # order them
            oo = part[np.argsort(cur_scores[part])]
            beta1 = 800.0
            best_y = None; best_m = 1e9
            for idx in oo:
                y0 = cand[idx]
                # refine vs current set
                y1 = _opt_single(cur_set, y0.copy(), beta1, maxiter=60)
                mm = float(np.max(cur_set @ y1))
                # second beta polish for top few
                if mm < float(np.max(cur_set @ y0)) + 1e-12:
                    y2 = _opt_single(cur_set, y1.copy(), 2500.0, maxiter=40)
                    mm2 = float(np.max(cur_set @ y2))
                    if mm2 < mm:
                        y1, mm = y2, mm2
                if mm < best_m:
                    best_m = mm; best_y = y1
            if best_y is None:
                ok_trial = False
                break
            inserted.append(best_y.copy())
            # extend current set
            cur_set = np.vstack([cur_set, best_y[None, :]])
            # update candidate scores: max(old, cand@new)
            try:
                upd = cand @ best_y
                cur_scores = np.maximum(cur_scores, upd)
            except Exception:
                pass
        if not ok_trial:
            continue
        Xi0 = np.vstack(inserted)  # k,d
        # hinge optimize only inserted points with margins below incumbent
        # targets relative to global best
        try:
            t_aggr = best - 2e-4
            Xi1 = _hinge_opt_inserted(Xf, Xi0, t_aggr, maxiter=120)
            t_fine = best - 5e-5
            Xi2 = _hinge_opt_inserted(Xf, Xi1, t_fine, maxiter=120)
        except Exception:
            Xi2 = Xi0
        Xnew = np.vstack([Xf, Xi2])
        # full log-sum-exp release beta 100 -> 10000
        try:
            Xrel = _run_full_lse(Xnew, [100.0, 500.0, 1500.0, 4000.0, 10000.0], [50, 60, 70, 70, 60])
        except Exception:
            continue
        snew = _max_cos(Xrel)
        # adjacency difference
        try:
            pset = _top_pairs_set(Xrel, 600)
            jac_best = _jaccard(pset, _top_pairs_set(bestX, 600))
        except Exception:
            jac_best = 1.0
            pset = set()
        if snew < best - 1e-12:
            best = snew; bestX = Xrel.copy()
            archive.append((snew, Xrel.copy(), pset))
            # keep archive bounded, keep best + most diverse
            if len(archive) > 7:
                # sort by score, keep best, then diverse among rest
                archive.sort(key=lambda z: z[0])
                kept = [archive[0]]
                for sc, xx, ps in archive[1:]:
                    # keep if diverse vs kept
                    div = all(_jaccard(ps, q[2]) < 0.95 for q in kept)
                    if div or len(kept) < 4:
                        kept.append((sc, xx, ps))
                    if len(kept) >= 7:
                        break
                archive = kept
        elif (jac_best < 0.90) and (snew < best + 2e-3):
            # accept topologically different basin even without immediate gain
            # add if diverse vs all
            diverse = all(_jaccard(pset, q[2]) < 0.92 for _, _, q in [(0, 0, qq) for qq in [a[2] for a in archive]] ) if False else True
            try:
                diverse = all(_jaccard(pset, a[2]) < 0.92 for a in archive)
            except Exception:
                diverse = True
            if diverse:
                archive.append((snew, Xrel.copy(), pset))
                if len(archive) > 7:
                    # drop worst non-best
                    archive.sort(key=lambda z: z[0])
                    # keep best 2 + 5 diverse
                    archive = archive[:7]
        # occasional time check
        if time.time() > T_END - LP_RESERVE:
            break
    # final LP squeeze on best few, reserving final seconds
    try:
        archive.sort(key=lambda z: z[0])
    except Exception:
        pass
    deadline = T_END - 0.5
    # polish top up to 3
    for i in range(min(3, len(archive))):
        if time.time() > deadline - 1.0:
            break
        sc, xx, _ = archive[i]
        # give each a slice
        slice_end = deadline - (2 - i) * 1.0
        if slice_end < time.time() + 0.5:
            slice_end = time.time() + 1.0
        try:
            xp = _lp_polish(xx, min(slice_end, deadline), r0=6e-4)
            sp = _max_cos(xp)
            if sp < best:
                best = sp; bestX = xp.copy()
            archive[i] = (sp, xp.copy(), _top_pairs_set(xp, 600))
        except Exception:
            continue
    # final best sync
    try:
        archive.sort(key=lambda z: z[0])
        if archive[0][0] < best:
            best = archive[0][0]; bestX = archive[0][1].copy()
    except Exception:
        pass
    bestX = bestX / (np.linalg.norm(bestX, axis=1, keepdims=True) + 1e-300)
    return bestX
