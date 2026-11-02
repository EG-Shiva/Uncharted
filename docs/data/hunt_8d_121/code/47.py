import time
import numpy as np
from scipy.optimize import linprog, minimize, nnls

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def solve():
    t0 = time.time()
    DEADLINE = t0 + 170.0
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 121, 8
    iu = np.triu_indices(n, 1)
    Iall, Jall = iu[0], iu[1]
    M = len(Iall)

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        X0 = rng.standard_normal((n, d))
        X0 = _norm(X0)
        cands = [X0]
    vals = [_max_cos(a) for a in cands]
    Xbest = cands[int(np.argmin(vals))].copy()
    Vbest = float(min(vals))

    # ---------- stress-dual diagnostics ----------
    def stress_analysis(X, top_p=800):
        try:
            G = X @ X.T
            c = G[iu]
            order = np.argsort(-c)
            p = int(min(top_p, M))
            sel = order[:p]
            Is = Iall[sel]; Js = Jall[sel]
            # build A: (n*d) x p
            nd = n * d
            A = np.zeros((nd, p), dtype=float)
            for k in range(p):
                i = int(Is[k]); j = int(Js[k])
                xi = X[i]; xj = X[j]
                cij = float(xi @ xj)
                pixj = xj - cij * xi
                pjxi = xi - cij * xj
                A[i*d:(i+1)*d, k] = pixj
                A[j*d:(j+1)*d, k] = pjxi
            sqrt_mu = 10.0
            B = np.vstack([A, sqrt_mu * np.ones((1, p))])
            cv = np.zeros((nd + 1,))
            cv[-1] = sqrt_mu
            w, _ = nnls(B, cv, maxiter=5 * p if 5*p>5000 else 5000)
            # per-point stats
            s = np.zeros(n)
            deg = np.zeros(n, dtype=int)
            R = np.zeros((n, d))
            mx = float(w.max()) if w.size else 1.0
            thr = mx * 1e-6 if mx > 0 else 1e-12
            for k in range(p):
                i = int(Is[k]); j = int(Js[k])
                wk = float(w[k])
                s[i] += wk; s[j] += wk
                if wk > thr:
                    deg[i] += 1; deg[j] += 1
                xi = X[i]; xj = X[j]
                cij = float(xi @ xj)
                pixj = xj - cij * xi
                pjxi = xi - cij * xj
                R[i] += wk * pixj
                R[j] += wk * pjxi
            res = np.linalg.norm(R, axis=1)
            return w, sel, Is, Js, s, res, deg
        except Exception:
            return None

    sa = stress_analysis(Xbest, top_p=800)
    if sa is None:
        w = np.ones(M); s_pts = np.ones(n); res_pts = np.zeros(n); deg_pts = np.full(n, 12)
        v_pair = np.ones(M)
        Is_c = None
    else:
        w, sel, Is_c, Js_c, s_pts, res_pts, deg_pts = sa
        # inverse-stress pair weights for all pairs
        v_pair = np.ones(M, dtype=float)
        if w.size and float(w.mean()) > 0:
            wm = float(w.mean()) + 1e-18
            gamma = 1.5
            vk = 1.0 / (1.0 + gamma * w / wm)
            # map back: sel indexes into triu order; need position in M
            # sel already are indices into M (order[:p])
            v_pair[sel] = vk
        # also cap extremely small?
        v_pair = np.clip(v_pair, 0.15, 1.0)

    # ---------- helpers ----------
    def find_hole(base, nsample=20000):
        C = rng.standard_normal((nsample, d))
        C = _norm(C)
        D = C @ base.T
        cm = D.max(axis=1)
        k = int(np.argmin(cm))
        return C[k].copy()

    def perturb_stress_guided(X, sigma, s_pts_, res_pts_, deg_pts_):
        Xn = X.copy()
        rmax = float(res_pts_.max()) if float(res_pts_.max()) > 0 else 1.0
        smax = float(s_pts_.max()) if float(s_pts_.max()) > 0 else 1.0
        smean = float(s_pts_.mean()) if s_pts_.size else 1.0
        for i in range(n):
            rn = float(res_pts_[i]) / (rmax + 1e-18)
            bonus = 0.0
            if deg_pts_[i] <= 3:
                bonus += 1.2
            elif deg_pts_[i] <= 6:
                bonus += 0.5
            # overloaded also gets push
            if s_pts_[i] > 2.0 * smean:
                bonus += 0.5
            if s_pts_[i] < 0.2 * smean:
                bonus += 0.8
            sc = sigma * (0.5 + 1.5 * rn + bonus)
            z = rng.standard_normal(d)
            z = z - (z @ Xn[i]) * Xn[i]
            nz = np.linalg.norm(z)
            if nz == 0:
                continue
            z /= nz
            # random magnitude
            Xn[i] = Xn[i] + sc * z
        return _norm(Xn)

    # build starts
    starts = [Xbest.copy()]
    if time.time() < DEADLINE - 100:
        starts.append(perturb_stress_guided(Xbest, 0.012, s_pts, res_pts, deg_pts))
        starts.append(perturb_stress_guided(Xbest, 0.025, s_pts, res_pts, deg_pts))
        # knockout overloaded + hole reinsertion
        try:
            order_s = np.argsort(-s_pts)
            nk = 3
            drop = order_s[:nk]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[drop] = False
            base = Xbest[keep_mask]
            newpts = []
            for _ in range(nk):
                b = np.vstack([base] + newpts) if newpts else base
                h = find_hole(b, nsample=20000)
                # local refine hole
                best_h = h; best_v = float((b @ h).max())
                for _ in range(80):
                    prop = best_h + 0.01 * rng.standard_normal(d)
                    prop /= np.linalg.norm(prop)
                    vv = float((b @ prop).max())
                    if vv < best_v:
                        best_v = vv; best_h = prop
                newpts.append(best_h)
            Xko = np.vstack([base] + newpts)
            # shuffle order to avoid index bias
            perm = rng.permutation(n)
            Xko = Xko[perm]
            # remap v_pair? keep same weights (approx) - recompute uniform for this start
            starts.append(_norm(Xko))
        except Exception:
            pass

    # ---------- weighted LSE + penalties ----------
    def make_fun(beta, lam_iso, lam_bal, v):
        b = float(beta); li = float(lam_iso); lb = float(lam_bal)
        Vsum = float(v.sum())
        tgt = n / d
        def fun(flat):
            X = flat.reshape(n, d)
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = X / nr
            G = Y @ Y.T
            c = G[iu]
            m = float(c.max())
            e = v * np.exp(b * (c - m))
            Z = float(e.sum())
            if not np.finite(Z) or Z <= 0:
                return float(m), np.zeros_like(flat)
            f = m + np.log(Z) / b
            wsoft = e / Z
            W = np.zeros((n, n))
            W[Iall, Jall] = wsoft
            W[Jall, Iall] = wsoft
            GY = W @ Y
            # isotropy
            if li != 0:
                S = Y.T @ Y - tgt * np.eye(d)
                iso = float((S * S).sum())
                f += li * iso
                GY = GY + li * (4.0 * (Y @ S))
            else:
                iso = 0.0
            # balance (variance of row sums)
            if lb != 0:
                Tvec = Y.sum(axis=0)  # d
                dotT = Y @ Tvec  # n
                si = dotT - 1.0
                ms = float(si.mean())
                diff = si - ms  # n
                bal = float((diff * diff).mean())
                f += lb * bal
                U = (diff[:, None] * Y).sum(axis=0)  # d
                # grad_Y_k = (2/n)[ diff_k * Tvec + U ]
                Gbal = (2.0 / n) * (diff[:, None] * Tvec[None, :] + np.tile(U[None, :], (n, 1)))
                GY = GY + lb * Gbal
            dot = (GY * Y).sum(axis=1, keepdims=True)
            GX = (GY - dot * Y) / nr
            g = GX.ravel()
            g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            if not np.finite(f):
                return 1e6, np.zeros_like(flat)
            return float(f), g
        return fun

    def run_stage(Xinit, beta, lam_iso, lam_bal, v, maxiter):
        fun = make_fun(beta, lam_iso, lam_bal, v)
        Xf = Xinit.ravel().copy()
        try:
            res = minimize(fun, Xf, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-20, "gtol": 1e-12, "maxfun": int(maxiter*6)})
            Xf = np.asarray(res.x, dtype=float)
        except Exception:
            pass
        X = Xf.reshape(n, d)
        X = _norm(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0))
        return X

    # phased annealing for each start, keep best by cosine but prefer improved balance
    Xcur_best = Xbest.copy()
    # evaluate balance metric for retention
    def balance_metric(X):
        try:
            G = X @ X.T
            c = G[iu]
            m = float(c.max())
            # residual under uniform on top 800
            p = 800
            order = np.argsort(-c)[:p]
            Is = Iall[order]; Js = Jall[order]
            R = np.zeros((n, d))
            for k in range(p):
                i = int(Is[k]); j = int(Js[k])
                xi = X[i]; xj = X[j]
                cij = float(xi @ xj)
                R[i] += (xj - cij*xi)
                R[j] += (xi - cij*xj)
            rn = float(np.linalg.norm(R) / p)
            # degree variance
            deg = np.zeros(n)
            for k in range(p):
                deg[int(Is[k])] += 1; deg[int(Js[k])] += 1
            bv = float(deg.var())
            return rn, bv
        except Exception:
            return 1e9, 1e9

    r0, b0 = balance_metric(Xbest)

    # schedule: relaxed -> tightening, penalties decay
    betas = [350.0, 900.0, 2200.0, 4500.0]
    lam_isos = [2.0e-4, 1.0e-4, 3.0e-5, 0.0]
    lam_bals = [2.0e-5, 1.0e-5, 3.0e-6, 0.0]
    iters = [120, 120, 160, 200]

    for si, Xs in enumerate(starts):
        if time.time() > DEADLINE - 45:
            break
        X = Xs.copy()
        # for knockout start use uniform weights initially
        vloc = v_pair.copy() if si < 3 else np.ones(M)
        for bi, (beta, li, lb, mi) in enumerate(zip(betas, lam_isos, lam_bals, iters)):
            if time.time() > DEADLINE - 40:
                break
            # gradually remove auxiliary: after stage 1 blend weights to 1
            if bi >= 2:
                # interpolate toward uniform
                alpha = 0.5 if bi == 2 else 0.0
                # vloc blended: v = alpha*v_pair + (1-alpha)*1
                if si < 3:
                    vloc = alpha * v_pair + (1 - alpha) * np.ones(M)
                else:
                    vloc = np.ones(M)
            X = run_stage(X, beta, li, lb, vloc, mi)
            if time.time() > DEADLINE - 40:
                break
        v = _max_cos(X)
        if v < Vbest - 1e-12:
            Vbest = v; Xcur_best = X.copy(); Xbest = X.copy()
        else:
            # retain if stress/imbalance improved even if slightly worse (within 1e-3)
            if v < Vbest + 1e-3:
                rn, bv = balance_metric(X)
                # improvement?
                if rn < r0 * 0.98 or bv < b0 * 0.98:
                    # keep as continuation seed if better balanced: do extra pure LSE push
                    if time.time() < DEADLINE - 50:
                        X2 = run_stage(X, 4500.0, 0.0, 0.0, np.ones(M), 150)
                        v2 = _max_cos(X2)
                        if v2 < Vbest:
                            Vbest = v2; Xcur_best = X2.copy(); Xbest = X2.copy()
                            r0, b0 = balance_metric(Xbest)
                        elif v2 < Vbest + 5e-4 and (rn < r0 or bv < b0):
                            # keep exploring from X2 next? ignore
                            pass

    Xbest = Xcur_best.copy()

    # ---------- LP polish final 20-30s ----------
    def lp_polish(x, seconds, radius_init=3e-4):
        start = time.time()
        x = _norm(x)
        best = _max_cos(x)
        radius = radius_init
        it = 0
        while time.time() - start < seconds and time.time() < DEADLINE - 1.0:
            it += 1
            if it > 150 or radius < 1e-14:
                break
            G = x @ x.T
            cf = G[iu]
            m = float(cf.max())
            thr = max(20.0 * radius, 2e-9)
            selm = cf > m - thr
            cnt = int(selm.sum())
            if cnt == 0:
                radius *= 0.5
                continue
            if cnt > 2500:
                kth = np.sort(cf)[-2500]
                selm = cf >= kth
                cnt = int(selm.sum())
                if cnt > 3000:
                    idxp = np.argpartition(-cf, 3000)[:3000]
                    mk = np.zeros_like(selm); mk[idxp] = True
                    selm = mk; cnt = 3000
            c = cf[selm]
            I = iu[0][selm]; J = iu[1][selm]
            p = cnt
            nv = n * d + 1
            try:
                from scipy.sparse import coo_matrix
                rows = np.repeat(np.arange(p), 2 * d + 1)
                cols = np.concatenate([I[:, None] * d + np.arange(d),
                                       J[:, None] * d + np.arange(d),
                                       np.full((p, 1), n * d)], 1).ravel()
                valsA = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
                A_ub = coo_matrix((valsA, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            obj = np.zeros(nv); obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y = _norm(y)
            vv = _max_cos(y)
            if vv < best:
                x, best = y, vv
                radius = min(radius * 1.6, 0.02)
            else:
                radius *= 0.5
        return x, best

    remain = DEADLINE - time.time()
    sec = min(max(remain - 1.0, 5.0), 30.0)
    if sec > 2:
        try:
            Xbest, Vbest = lp_polish(Xbest, sec)
        except Exception:
            pass
    Xbest = _norm(Xbest)
    return Xbest
