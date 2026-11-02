import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def solve():
    t0 = time.time()
    BUDGET = 170.0
    DEADLINE = t0 + BUDGET
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 121, 8
    iu = np.triu_indices(n, 1)
    Iall, Jall = iu[0], iu[1]

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        X0 = rng.standard_normal((n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands = [X0]
    vals = [_max_cos(a) for a in cands]
    Xbest = cands[int(np.argmin(vals))].copy()
    Vbest = float(min(vals))
    # record reference for tabu (use published record if available else best)
    try:
        Rref = np.load("record.npy").astype(float)
        Rref /= np.linalg.norm(Rref, axis=1, keepdims=True)
    except Exception:
        Rref = Xbest.copy()
    TABU_RAD = 0.05
    TABU_W = 20.0

    def procrustes_Q(Y, R):
        M = Y.T @ R
        try:
            U, _, Vt = np.linalg.svd(M, full_matrices=False)
            Q = U @ Vt
            if np.linalg.det(Q) < 0:
                U[:, -1] *= -1
                Q = U @ Vt
            return Q
        except Exception:
            return np.eye(d)

    def hinge_fun_factory(t, use_tabu=True):
        tt = float(t)
        def fun(flat):
            X = flat.reshape(n, d)
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = X / nr
            G = Y @ Y.T
            c = G[iu]
            diff = c - tt
            pos = diff > 0
            f = float((diff[pos] ** 2).sum()) if np.any(pos) else 0.0
            w = np.zeros_like(c)
            w[pos] = 2.0 * diff[pos]
            W = np.zeros((n, n))
            W[Iall, Jall] = w
            W[Jall, Iall] = w
            GY = W @ Y
            dot = (GY * Y).sum(axis=1, keepdims=True)
            GX = (GY - dot * Y) / nr
            if use_tabu:
                try:
                    Q = procrustes_Q(Y, Rref)
                    D = Y @ Q - Rref
                    fn_ = float(np.linalg.norm(D))
                    rmsd = fn_ / np.sqrt(n)
                    if rmsd < TABU_RAD and fn_ > 1e-12:
                        f += TABU_W * (TABU_RAD - rmsd) ** 2
                        Gr = (D @ Q.T) / (np.sqrt(float(n)) * fn_)
                        coeff = -2.0 * TABU_W * (TABU_RAD - rmsd)
                        GY2 = coeff * Gr
                        dot2 = (GY2 * Y).sum(axis=1, keepdims=True)
                        GX += (GY2 - dot2 * Y) / nr
                except Exception:
                    pass
            if not np.isfinite(f):
                return 1e6, np.zeros_like(flat)
            g = GX.ravel()
            g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            return float(f), g
        return fun

    def lbfgs_hinge(Xinit, t, maxiter=300, use_tabu=True):
        fun = hinge_fun_factory(t, use_tabu)
        Xf = Xinit.ravel().copy()
        try:
            res = minimize(fun, Xf, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-20,
                                    "gtol": 1e-12, "maxfun": int(maxiter * 6)})
            Xf = np.asarray(res.x, dtype=float)
        except Exception:
            pass
        X = Xf.reshape(n, d)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        nn = np.linalg.norm(X, axis=1, keepdims=True)
        nn[nn == 0] = 1.0
        X /= nn
        return X

    def find_hole_insert(base, nsample=25000, refine=200):
        # base: (m,d)
        C = rng.standard_normal((nsample, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        D = C @ base.T
        cm = D.max(axis=1)
        k = int(np.argmin(cm))
        bc = C[k].copy()
        cur = float(cm[k])
        for st in range(refine):
            sig = 0.03 * (1.0 - st / float(refine)) + 0.001
            prop = bc + sig * rng.standard_normal(d)
            pn = np.linalg.norm(prop)
            if pn == 0:
                continue
            prop /= pn
            md = float((prop @ base.T).max())
            if md < cur:
                bc = prop
                cur = md
        return bc, cur

    def grow_code(seed121, n_start, nsample=20000):
        # remove worst to n_start, relax, then grow to 121
        G = seed121 @ seed121.T
        np.fill_diagonal(G, -2.0)
        rm = G.max(axis=1)
        order = np.argsort(-rm)
        drop = order[:121 - n_start]
        keep = np.ones(n, dtype=bool)
        keep[drop] = False
        # variable-size hinge optimise for the small code using LSE-ish quick relax
        cur = seed121[keep].copy()  # (n_start,d)
        m = cur.shape[0]
        # quick relax of small code with plain repulsion (few LBFGS steps on max surrogate)
        # simple projected gradient on hinge t=0.46 for this size
        iu_m = np.triu_indices(m, 1)
        def fun_m(flat):
            X = flat.reshape(m, d)
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = X / nr
            Gm = Y @ Y.T
            cc = Gm[iu_m]
            dd = cc - 0.46
            pp = dd > 0
            f = float((dd[pp] ** 2).sum()) if np.any(pp) else 0.0
            w = np.zeros_like(cc); w[pp] = 2.0 * dd[pp]
            Wm = np.zeros((m, m)); Wm[iu_m[0], iu_m[1]] = w; Wm[iu_m[1], iu_m[0]] = w
            GY = Wm @ Y
            dot = (GY * Y).sum(axis=1, keepdims=True)
            GX = (GY - dot * Y) / nr
            return float(f), GX.ravel()
        try:
            Xf = cur.ravel().copy()
            res = minimize(fun_m, Xf, jac=True, method="L-BFGS-B",
                           options={"maxiter": 120, "ftol": 1e-20, "gtol": 1e-10})
            cur = np.asarray(res.x).reshape(m, d)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        except Exception:
            pass
        # grow one by one
        pts = [cur[i] for i in range(m)]
        for _ in range(n - m):
            if time.time() > DEADLINE - 90:
                break
            base = np.stack(pts, axis=0)
            bc, _ = find_hole_insert(base, nsample=nsample, refine=150)
            pts.append(bc)
            # joint hinge relax of grown set (short)
            try:
                mm = len(pts)
                A = np.stack(pts, axis=0)
                iu_mm = np.triu_indices(mm, 1)
                def fun_g(flat):
                    X = flat.reshape(mm, d)
                    nr = np.linalg.norm(X, axis=1, keepdims=True)
                    nr = np.maximum(nr, 1e-12)
                    Y = X / nr
                    Gm = Y @ Y.T
                    cc = Gm[iu_mm]
                    dd = cc - 0.46
                    pp = dd > 0
                    f = float((dd[pp] ** 2).sum()) if np.any(pp) else 0.0
                    w = np.zeros_like(cc); w[pp] = 2.0 * dd[pp]
                    Wm = np.zeros((mm, mm)); Wm[iu_mm[0], iu_mm[1]] = w; Wm[iu_mm[1], iu_mm[0]] = w
                    GY = Wm @ Y
                    dot = (GY * Y).sum(axis=1, keepdims=True)
                    GX = (GY - dot * Y) / nr
                    return float(f), GX.ravel()
                Xf = A.ravel().copy()
                res = minimize(fun_g, Xf, jac=True, method="L-BFGS-B",
                               options={"maxiter": 80, "ftol": 1e-20, "gtol": 1e-10})
                A2 = np.asarray(res.x).reshape(mm, d)
                A2 /= np.linalg.norm(A2, axis=1, keepdims=True)
                pts = [A2[i] for i in range(mm)]
            except Exception:
                pass
        if len(pts) != n:
            # pad/fallback
            while len(pts) < n:
                v = rng.standard_normal(d); v /= np.linalg.norm(v)
                pts.append(v)
        return np.stack(pts, axis=0)

    def lbfgs_anneal(X, betas, maxiters):
        Xf = X.ravel().copy()
        for beta, mi in zip(betas, maxiters):
            if time.time() > DEADLINE - 10.0:
                break
            b = float(beta)
            def fun(flat):
                Xr = flat.reshape(n, d)
                nr = np.linalg.norm(Xr, axis=1, keepdims=True)
                nr = np.maximum(nr, 1e-12)
                Y = Xr / nr
                Gm = Y @ Y.T
                c = Gm[iu]
                mm = float(c.max())
                e = np.exp(b * (c - mm))
                s = float(e.sum())
                if not np.isfinite(s) or s <= 0:
                    return float(mm), np.zeros_like(flat)
                f = mm + np.log(s) / b
                w = e / s
                W = np.zeros((n, n))
                W[Iall, Jall] = w
                W[Jall, Iall] = w
                GY = W @ Y
                dot = (GY * Y).sum(axis=1, keepdims=True)
                GX = (GY - dot * Y) / nr
                return float(f), GX.ravel()
            try:
                res = minimize(fun, Xf, jac=True, method="L-BFGS-B",
                               options={"maxiter": int(mi), "ftol": 1e-20,
                                        "gtol": 1e-12, "maxfun": int(mi * 6)})
                Xf = np.asarray(res.x, dtype=float)
            except Exception:
                break
        Xn = Xf.reshape(n, d)
        Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
        return Xn

    def lp_polish(x, seconds, radius_init=3e-4):
        start = time.time()
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        best = _max_cos(x)
        radius = radius_init
        while time.time() - start < seconds and time.time() < DEADLINE - 1.0:
            if radius < 1e-14:
                break
            G = x @ x.T
            cf = G[iu]
            m = float(cf.max())
            thr = max(20.0 * radius, 2e-9)
            sel = cf > m - thr
            cnt = int(sel.sum())
            if cnt == 0:
                radius *= 0.5
                continue
            if cnt > 2500:
                kth = np.sort(cf)[-2500]
                sel = cf >= kth
                cnt = int(sel.sum())
                if cnt > 3000:
                    idxp = np.argpartition(-cf, 3000)[:3000]
                    mk = np.zeros_like(sel); mk[idxp] = True
                    sel = mk; cnt = 3000
            c = cf[sel]
            I = iu[0][sel]; J = iu[1][sel]
            p = cnt
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            valsA = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            try:
                A_ub = coo_matrix((valsA, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((x.ravel(),
                                   (np.repeat(np.arange(n), d), np.arange(n * d))),
                                  shape=(n, nv)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            obj = np.zeros(nv); obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq,
                              b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                              method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            yn = np.linalg.norm(y, axis=1, keepdims=True)
            yn[yn == 0] = 1.0
            y = y / yn
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius * 1.6, 0.02)
            else:
                radius *= 0.5
        return x, best

    def sorted_gram_vec(X):
        G = X @ X.T
        c = G[iu]
        return np.sort(c)

    # ---- build grown starts (point-count continuation) ----
    grown_inits = []
    for ns_, samp in [(115, 22000), (118, 22000), (120, 18000)]:
        if time.time() > DEADLINE - 120:
            break
        try:
            g = grow_code(Xbest, ns_, nsample=samp)
            g /= np.linalg.norm(g, axis=1, keepdims=True)
            grown_inits.append(g)
        except Exception:
            continue

    # ---- fresh starts ----
    fresh = []
    for s in range(8):
        if time.time() > DEADLINE - 110:
            break
        A = rng.standard_normal((n, d))
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        fresh.append(A)
    # perturbed best (will be repelled by tabu)
    pert = []
    for sig in [0.12, 0.22, 0.32]:
        A = Xbest + sig * rng.standard_normal((n, d))
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        pert.append(A)

    all_inits = grown_inits + fresh + pert
    # threshold continuation targets
    targets = [0.4600, 0.4595, 0.4590, 0.4585]
    survivors = []  # (maxcos, X, t_feas)
    # hinge sweep at first target mainly; then push feasible ones lower
    for Xi in all_inits:
        if time.time() > DEADLINE - 70:
            break
        try:
            # start at t=0.46
            Y = lbfgs_hinge(Xi, targets[0], maxiter=350, use_tabu=True)
            v = _max_cos(Y)
            # try continuation to lower t starting from Y
            best_t = None
            best_Y = Y
            best_v = v
            if v <= targets[0] + 2e-4:
                for tt in targets[1:]:
                    if time.time() > DEADLINE - 70:
                        break
                    Y2 = lbfgs_hinge(best_Y, tt, maxiter=250, use_tabu=True)
                    v2 = _max_cos(Y2)
                    if v2 <= tt + 3e-4:
                        best_Y, best_v, best_t = Y2, v2, tt
                    else:
                        # keep the attempt anyway for ranking if improved
                        if v2 < best_v:
                            best_Y, best_v = Y2, v2
                        break
            survivors.append((best_v, best_Y, best_t))
            if best_v < Vbest - 1e-12:
                Vbest, Xbest = best_v, best_Y.copy()
        except Exception:
            continue

    # distinctness filter via sorted gram vectors
    survivors.sort(key=lambda z: z[0])
    filt = []
    vecs = []
    for v, Y, bt in survivors:
        sv = sorted_gram_vec(Y)
        keep = True
        for sv2 in vecs:
            if np.linalg.norm(sv - sv2) < 5e-4:
                keep = False
                break
        if keep:
            filt.append((v, Y, bt))
            vecs.append(sv)
        if len(filt) >= 4:
            break
    if not filt and survivors:
        filt = survivors[:3]

    # promote best few distinct survivors to full anneal + LP
    sched = ([250.0, 700.0, 1600.0, 3200.0], [120, 130, 160, 220])
    for v, Y, bt in filt:
        if time.time() > DEADLINE - 25:
            break
        try:
            Ya = lbfgs_anneal(Y, sched[0], sched[1])
            va = _max_cos(Ya)
            if va < Vbest:
                Vbest, Xbest = va, Ya.copy()
            Xp, vp = lp_polish(Ya if va < v else Y, 3.0, 3e-4)
            if vp < Vbest:
                Vbest, Xbest = vp, Xp.copy()
        except Exception:
            continue

    # final squeeze on global best
    if time.time() < DEADLINE - 12:
        try:
            Ya = lbfgs_anneal(Xbest, [500.0, 1200.0, 2500.0, 4000.0], [80, 100, 130, 160])
            va = _max_cos(Ya)
            if va < Vbest:
                Vbest, Xbest = va, Ya.copy()
        except Exception:
            pass
    if time.time() < DEADLINE - 4:
        try:
            Xp, vp = lp_polish(Xbest, max(1.0, DEADLINE - time.time() - 1.0), 3e-4)
            if vp < Vbest:
                Vbest, Xbest = vp, Xp.copy()
        except Exception:
            pass

    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    # tiny shrink guard
    return Xbest
