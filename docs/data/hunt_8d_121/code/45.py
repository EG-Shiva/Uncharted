import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def solve():
    t0 = time.time()
    DEADLINE = t0 + 171.0
    np.random.seed(0)
    n, d = 121, 8
    iu = np.triu_indices(n, 1)
    Iall, Jall = iu[0], iu[1]
    Mpair = Iall.size
    c_iso = float(n) / float(d)

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
        rr = np.random.default_rng(0)
        X0 = rr.standard_normal((n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands = [X0]
    vals = [_max_cos(a) for a in cands]
    Xbest = cands[int(np.argmin(vals))].copy()
    Vbest = float(min(vals))

    def row_stats(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        rm = G.max(axis=1)
        m = float(G.max())
        deg = ((G > m - 1e-4).sum(axis=1)).astype(float)
        imb = float(rm.std()) + 0.002 * float(deg.std())
        return m, imb, rm, deg

    # ---- stress estimation on incumbent ----
    def estimate_stress(X, tol=8e-4, cap=2200):
        try:
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            slack = m - G[iu]
            sel = np.where(slack < tol)[0]
            if sel.size < 50:
                sel = np.argsort(slack)[:min(500, Mpair)]
            if sel.size > cap:
                sel = np.argsort(slack)[:cap]
            p = sel.size
            Is = iu[0][sel]; Js = iu[1][sel]
            cs = G[Is, Js] if False else (X[Is] * X[Js]).sum(axis=1)
            # build M (n*d x p)
            Mp = np.zeros((n * d, p), dtype=float)
            for k in range(p):
                i = int(Is[k]); j = int(Js[k])
                c = float(cs[k])
                vi = X[j] - c * X[i]
                vj = X[i] - c * X[j]
                Mp[i * d:(i + 1) * d, k] = vi
                Mp[j * d:(j + 1) * d, k] = vj
            Q = Mp.T @ Mp
            # mirror descent on simplex: min w^T Q w
            w = np.full(p, 1.0 / p)
            # scale step
            diag = np.diag(Q)
            L = float(np.max(diag)) + 1e-12
            eta = 0.6 / max(L, 1e-12)
            eta = min(max(eta, 0.05), 1.0)
            for _ in range(400):
                g = Q @ w
                # centered? mirror descent
                w = w * np.exp(-eta * g)
                s = w.sum()
                if not np.isfinite(s) or s <= 0:
                    w = np.full(p, 1.0 / p)
                    break
                w /= s
                if time.time() > DEADLINE - 5:
                    break
            R = Mp @ w
            Rn = R.reshape(n, d)
            res = np.linalg.norm(Rn, axis=1)
            spts = np.zeros(n)
            np.add.at(spts, Is, w)
            np.add.at(spts, Js, w)
            return sel, Is, Js, w, res, spts, m
        except Exception:
            return None

    est = estimate_stress(Xbest)
    if est is None:
        alpha = np.ones(Mpair)
        res_pts = np.zeros(n)
        s_pts = np.ones(n)
        deg0 = np.full(n, 12.0)
    else:
        sel, Is0, Js0, w0, res_pts, s_pts, m0 = est
        alpha = np.ones(Mpair)
        nw = w0 * float(w0.size)  # mean 1
        a_raw = 1.0 / (nw + 0.5)
        a_raw = a_raw / (a_raw.mean() + 1e-12)
        a_raw = np.clip(a_raw, 0.25, 4.0)
        alpha[sel] = a_raw
        # degree proxy at 1e-4
        _, _, _, deg0 = row_stats(Xbest)
    alpha_sum_mean = float(alpha.mean())

    # per-point diagnostics for targeting
    order_res = np.argsort(-res_pts)
    order_s = np.argsort(s_pts)  # low stress first
    order_deg = np.argsort(-deg0)

    _, imb_best, _, _ = row_stats(Xbest)
    Xbal = Xbest.copy()
    comb_best = Vbest + 0.25 * imb_best
    Vbal = Vbest

    def make_fun(beta, lam_iso, lam_var):
        b = float(beta); li = float(lam_iso); lv = float(lam_var)
        def fun(flat):
            X = flat.reshape(n, d)
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = X / nr
            G = Y @ Y.T
            c = G[iu]
            m = float(c.max())
            z = b * (c - m)
            # clip for exp safety (underflow ok)
            e = alpha * np.exp(z)
            s = float(e.sum())
            if not np.isfinite(s) or s <= 0:
                return float(m), np.zeros_like(flat)
            f = m + np.log(s) / b
            w = e / s
            W = np.zeros((n, n))
            W[Iall, Jall] = w
            W[Jall, Iall] = w
            GY = W @ Y
            # row variance penalty
            fvar = 0.0
            if lv != 0.0:
                Rrow = np.zeros(n)
                np.add.at(Rrow, Iall, e)
                np.add.at(Rrow, Jall, e)
                Rrow = np.maximum(Rrow, 1e-300)
                r = m + np.log(Rrow) / b
                rbar = float(r.mean())
                diff = r - rbar
                fvar = float((diff ** 2).mean())
                f += lv * fvar
                # coeff per pair
                invR = 1.0 / Rrow
                ck = diff[Iall] * invR[Iall] + diff[Jall] * invR[Jall]
                coeff = (2.0 / n) * e * ck
                Wv = np.zeros((n, n))
                Wv[Iall, Jall] = coeff
                Wv[Jall, Iall] = coeff
                GY = GY + lv * (Wv @ Y)
            fiso = 0.0
            if li != 0.0:
                Ft = Y.T @ Y - c_iso * np.eye(d)
                fiso = float((Ft ** 2).mean())
                f += li * fiso
                GY = GY + li * (4.0 / 64.0 * (Y @ Ft))
            dot = (GY * Y).sum(axis=1, keepdims=True)
            GX = (GY - dot * Y) / nr
            GX = np.nan_to_num(GX, nan=0.0, posinf=0.0, neginf=0.0)
            if not np.isfinite(f):
                f = float(m)
            return float(f), GX.ravel()
        return fun

    def lbfgs_stages(Xinit, stages):
        Xf = Xinit.ravel().copy()
        for (beta, li, lv, mi) in stages:
            if time.time() > DEADLINE - 32.0:
                break
            fun = make_fun(beta, li, lv)
            try:
                res = minimize(fun, Xf, jac=True, method="L-BFGS-B",
                               options={"maxiter": int(mi), "ftol": 1e-20, "gtol": 1e-14, "maxfun": int(mi * 6)})
                Xf = np.asarray(res.x, dtype=float)
            except Exception:
                break
        Xn = Xf.reshape(n, d)
        Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)
        nn = np.linalg.norm(Xn, axis=1, keepdims=True)
        nn[nn == 0] = 1.0
        Xn = Xn / nn
        return Xn

    def lp_polish(x, seconds, radius_init=3e-4):
        start = time.time()
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
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
                    mk = np.zeros_like(selm)
                    mk[idxp] = True
                    selm = mk
                    cnt = 3000
            c = cf[selm]
            I = iu[0][selm]; J = iu[1][selm]
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

    # staged schedules: relaxed -> tight
    SCHED_A = [(1500.0, 0.005, 3.0, 60), (4000.0, 0.001, 1.0, 70), (12000.0, 0.0002, 0.2, 70), (30000.0, 0.0, 0.0, 60)]
    SCHED_B = [(2500.0, 0.003, 2.0, 60), (8000.0, 0.0005, 0.5, 70), (25000.0, 0.0, 0.0, 80)]
    SCHED_C = [(1200.0, 0.008, 4.0, 50), (5000.0, 0.001, 0.8, 60), (40000.0, 0.0, 0.0, 70)]

    trial = 0
    # reserve final 26s for LP
    while time.time() < DEADLINE - 26.0 and trial < 60:
        trial += 1
        rng = np.random.default_rng(7000 + trial * 977)
        # choose seed: mostly best, sometimes balanced
        if trial % 4 == 3:
            Xseed = Xbal.copy()
        else:
            Xseed = Xbest.copy()
        # targeted perturbation based on initial diagnostics (fixed) + current degree
        try:
            Gc = Xseed @ Xseed.T
            np.fill_diagonal(Gc, -2.0)
            rm_c = Gc.max(axis=1)
            ord_c = np.argsort(-rm_c)
        except Exception:
            ord_c = order_res
        mode = trial % 5
        Xtry0 = Xseed.copy()
        if mode == 0:
            # shake high-residual + high-degree, reinsert low-stress
            hi = np.unique(np.concatenate([order_res[:10], order_deg[:8], ord_c[:8]]))
            Xtry0[hi] += 0.16 * rng.standard_normal((hi.size, d))
            lo = order_s[:6]
            R = rng.standard_normal((lo.size, d)); R /= np.linalg.norm(R, axis=1, keepdims=True)
            # half random reinsert
            Xtry0[lo[:3]] = R[:3]
            Xtry0[lo[3:]] += 0.10 * rng.standard_normal((lo[3:].size, d))
            Xtry0 += 0.004 * rng.standard_normal((n, d))
            sched = SCHED_A
        elif mode == 1:
            k = 7
            worst = ord_c[:k]
            R = rng.standard_normal((k, d)); R /= np.linalg.norm(R, axis=1, keepdims=True)
            Xtry0[worst] = R
            Xtry0 += 0.003 * rng.standard_normal((n, d))
            sched = SCHED_B
        elif mode == 2:
            u = rng.standard_normal(d); u /= np.linalg.norm(u)
            proj = Xseed @ u
            idx = np.argsort(-proj)[:9]
            R = rng.standard_normal((9, d)); R /= np.linalg.norm(R, axis=1, keepdims=True)
            Xtry0[idx] = R
            Xtry0 += 0.004 * rng.standard_normal((n, d))
            sched = SCHED_C
        elif mode == 3:
            # balanced shake: global + targeted
            Xtry0 = Xseed + 0.02 * rng.standard_normal((n, d))
            hi2 = ord_c[:10]
            Xtry0[hi2] += 0.10 * rng.standard_normal((10, d))
            Xtry0 /= np.linalg.norm(Xtry0, axis=1, keepdims=True)
            sched = SCHED_A
        else:
            sig = 0.22 if (trial % 2 == 0) else 0.09
            Xtry0 = Xseed + sig * rng.standard_normal((n, d))
            sched = SCHED_B
        Xtry0 /= np.linalg.norm(Xtry0, axis=1, keepdims=True)
        try:
            Xtry = lbfgs_stages(Xtry0, sched)
        except Exception:
            continue
        v, imb, _, _ = row_stats(Xtry)
        comb = v + 0.25 * imb
        if v < Vbest - 1e-12:
            # quick small polish to confirm
            Xbest, Vbest = Xtry, v
            if comb < comb_best:
                comb_best, Xbal, Vbal = comb, Xtry.copy(), v
            # refresh diagnostics every improvement (cheap degree only)
            try:
                Gc2 = Xbest @ Xbest.T
                np.fill_diagonal(Gc2, -2.0)
                rm2 = Gc2.max(axis=1)
                ord_c = np.argsort(-rm2)
            except Exception:
                pass
            # re-estimate stress occasionally
            if trial % 6 == 0:
                est2 = estimate_stress(Xbest)
                if est2 is not None:
                    sel2, _, _, w2, res_pts2, s_pts2, _ = est2
                    # keep alpha fixed to original basin per assignment; do not overwrite
                    # only update targeting orders
                    order_res = np.argsort(-res_pts2)
                    order_s = np.argsort(s_pts2)
        elif v < Vbest + 0.004 and comb < comb_best:
            comb_best = comb
            Xbal = Xtry.copy()
            Vbal = v

    # final LP polish on best (and balanced if promising)
    rem = DEADLINE - time.time() - 1.0
    if rem > 3.0:
        # polish best
        Xp, Vp = lp_polish(Xbest, seconds=min(rem * 0.7, 20.0), radius_init=2e-4)
        if Vp < Vbest:
            Xbest, Vbest = Xp, Vp
        rem2 = DEADLINE - time.time() - 1.0
        if rem2 > 3.0 and Vbal < Vbest + 0.001:
            Xq, Vq = lp_polish(Xbal, seconds=min(rem2 * 0.5, 8.0), radius_init=2e-4)
            if Vq < Vbest:
                Xbest, Vbest = Xq, Vq
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    # tiny shrink guard (strict feasibility, no-op for sphere)
    Xbest = np.nan_to_num(Xbest, nan=0.0, posinf=0.0, neginf=0.0)
    nn = np.linalg.norm(Xbest, axis=1, keepdims=True)
    nn[nn == 0] = 1.0
    Xbest = Xbest / nn
    return Xbest
