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
    DEADLINE = t0 + 170.0
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 121, 8
    iu = np.triu_indices(n, 1)
    Iall, Jall = iu[0].astype(int), iu[1].astype(int)
    Pfull = len(Iall)

    def pair_idx(iv, jv):
        # i<j
        return iv * n - iv * (iv + 1) // 2 + (jv - iv - 1)

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

    # ---------- stress diagnostics ----------
    def stress_weights(X, tol=1e-4, cap=2000, iters=300):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # upper triangle slacks
        c = G[iu]
        slack = m - c
        order = np.argsort(slack)
        seln = int((slack < tol).sum())
        seln = max(200, min(seln, cap))
        # take tightest seln
        idx = order[:seln]
        Is = Iall[idx]; Js = Jall[idx]
        cs = c[idx]
        p = seln
        # build A (n*d, p)
        A = np.empty((n * d, p), dtype=float)
        # P_i x_j = x_j - c*x_i
        for k in range(p):
            i = int(Is[k]); j = int(Js[k])
            cc = float(cs[k])
            vi = X[j] - cc * X[i]
            vj = X[i] - cc * X[j]
            col = np.zeros(n * d, dtype=float)
            col[i * d:(i + 1) * d] = vi
            col[j * d:(j + 1) * d] = vj
            A[:, k] = col
        # mirror descent on simplex: min ||A w||^2
        w = np.full(p, 1.0 / p, dtype=float)
        for _ in range(iters):
            Aw = A @ w  # (nd,)
            grad = A.T @ Aw  # (p,)
            rng_ = float(grad.max() - grad.min())
            if not np.isfinite(rng_) or rng_ < 1e-16:
                break
            eta = 0.35 / (rng_ + 1e-16)
            # center for stability
            gc = grad - float(grad.mean())
            w = w * np.exp(-eta * gc)
            s = float(w.sum())
            if not np.isfinite(s) or s <= 0:
                w = np.full(p, 1.0 / p)
                break
            w /= s
        Aw = A @ w
        R = Aw.reshape(n, d)
        res = np.linalg.norm(R, axis=1)
        spt = np.zeros(n)
        np.add.at(spt, Is, w)
        np.add.at(spt, Js, w)
        # inverse-stress pair weights
        pw = p * w
        v = (0.3 + pw) ** (-0.5)
        # clip + normalize mean 1
        v = v / (float(v.mean()) + 1e-16)
        v = np.clip(v, 0.3, 3.0)
        v = v / (float(v.mean()) + 1e-16)
        v_all = np.ones(Pfull, dtype=float)
        full_pos = pair_idx(Is, Js)
        v_all[full_pos] = v
        v_all /= float(v_all.mean())
        # contact counts at 1e-9 for surgery
        G2 = X @ X.T
        np.fill_diagonal(G2, -2.0)
        m2 = float(G2.max())
        tight = (m2 - G2) < 1e-9
        np.fill_diagonal(tight, False)
        cnt = tight.sum(axis=1).astype(int)
        return v_all, w, res, spt, cnt, Is, Js, float(np.linalg.norm(Aw))

    try:
        v_all0, w0, res0, spt0, cnt0, Is0, Js0, anorm0 = stress_weights(Xbest)
    except Exception:
        v_all0 = np.ones(Pfull)
        cnt0 = np.zeros(n, dtype=int)
        res0 = np.zeros(n)
        anorm0 = 0.0

    def find_hole(base, nsample=20000, refine=120):
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
        return bc

    def lbfgs_stage(Xinit, beta, lam_var, lam_iso, v_w, maxiter=150):
        Vsum = float(v_w.sum()) + 1e-16
        Tt = (n / d) * np.eye(d)
        b = float(beta)
        lv = float(lam_var); li = float(lam_iso)
        def fun(flat):
            X = flat.reshape(n, d)
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = X / nr
            G = Y @ Y.T
            c = G[iu]
            mm = float(c.max())
            e = np.exp(b * (c - mm))
            Sw = float((v_w * e).sum())
            if not np.isfinite(Sw) or Sw <= 0:
                return float(mm), np.zeros_like(flat)
            f = mm + (np.log(Sw) - np.log(Vsum)) / b
            q = (v_w * e) / Sw
            W = np.zeros((n, n))
            W[Iall, Jall] = q
            W[Jall, Iall] = q
            GY = W @ Y
            # variance of row sums
            if lv != 0.0:
                r = G.sum(axis=1)
                mu = float(r.mean())
                ev = r - mu
                var = float((ev ** 2).mean())
                f += lv * var
                s_vec = Y.sum(axis=0)
                eY = ev @ Y
                Gv = (ev[:, None] * np.ones((1, d)) * s_vec[None, :])
                Gv = ev[:, None] * s_vec[None, :] + np.ones((n, 1)) * eY[None, :]
                GY = GY + lv * (2.0 / n) * Gv
            if li != 0.0:
                S = Y.T @ Y
                D = S - Tt
                F = float((D ** 2).sum())
                f += li * F
                GY = GY + li * (4.0 * (Y @ D))
            dot = (GY * Y).sum(axis=1, keepdims=True)
            GX = (GY - dot * Y) / nr
            if not np.isfinite(f):
                return 1e6, np.zeros_like(flat)
            g = np.nan_to_num(GX.ravel(), nan=0.0, posinf=0.0, neginf=0.0)
            return float(f), g
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
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def run_schedule(Xs, v_base):
        # staged: (beta, lam_var, lam_iso, alpha_pow, iters)
        sched = [(80, 0.005, 2e-4, 1.0, 130),
                 (150, 0.002, 1e-4, 0.5, 130),
                 (300, 5e-4, 3e-5, 0.25, 140),
                 (600, 0.0, 0.0, 0.0, 160)]
        X = Xs
        for (beta, lv, li, al, mi) in sched:
            if time.time() > DEADLINE - 32:
                break
            if al == 0.0:
                vw = np.ones(Pfull)
            else:
                vw = np.power(np.maximum(v_base, 1e-12), al)
                vw /= float(vw.mean())
            X = lbfgs_stage(X, beta, lv, li, vw, maxiter=mi)
            # keep global best opportunistically
            vv = _max_cos(X)
            global Xbest, Vbest
            if vv < Vbest:
                Vbest = vv
                Xbest = X.copy()
        return X

    # ---------- build replicas ----------
    reps = []
    # rep0: small noise
    try:
        R0 = Xbest + 0.008 * rng.standard_normal((n, d))
        R0 /= np.linalg.norm(R0, axis=1, keepdims=True)
        reps.append(R0)
    except Exception:
        pass
    # rep1: surgery on low-contact points
    try:
        order_low = np.argsort(cnt0)  # ascending
        K = 3
        victims = order_low[:K].tolist()
        base_keep = np.delete(Xbest, victims, axis=0)
        newpts = []
        cur_base = base_keep.copy()
        for _ in range(K):
            h = find_hole(cur_base, nsample=22000, refine=120)
            newpts.append(h)
            cur_base = np.vstack([cur_base, h[None, :]])
        R1 = Xbest.copy()
        for vi, h in zip(victims, newpts):
            R1[vi] = h
        R1 = R1 + 0.012 * rng.standard_normal((n, d))
        R1 /= np.linalg.norm(R1, axis=1, keepdims=True)
        reps.append(R1)
    except Exception:
        pass
    # rep2: larger global hop
    try:
        R2 = Xbest + 0.035 * rng.standard_normal((n, d))
        R2 /= np.linalg.norm(R2, axis=1, keepdims=True)
        reps.append(R2)
    except Exception:
        pass

    # run first two stages for all, then prune
    scored = []
    # quick two-stage filter
    filt_sched = [(80, 0.005, 2e-4, 1.0, 110), (150, 0.002, 1e-4, 0.5, 110)]
    filtered = []
    for R in reps:
        if time.time() > DEADLINE - 45:
            break
        X = R
        for (beta, lv, li, al, mi) in filt_sched:
            if al == 0.0:
                vw = np.ones(Pfull)
            else:
                vw = np.power(np.maximum(v_all0, 1e-12), al)
                vw /= float(vw.mean())
            X = lbfgs_stage(X, beta, lv, li, vw, maxiter=mi)
            if time.time() > DEADLINE - 40:
                break
        vv = _max_cos(X)
        if vv < Vbest:
            Vbest = vv
            Xbest = X.copy()
        filtered.append((vv, X))
    # keep best 2 for full continuation
    filtered.sort(key=lambda t: t[0])
    survivors = [x for _, x in filtered[:2]] if filtered else reps[:1]
    # also add one fresh random if time allows (different basin)
    if time.time() < DEADLINE - 80:
        try:
            Rf = rng.standard_normal((n, d))
            Rf /= np.linalg.norm(Rf, axis=1, keepdims=True)
            # quick relax at relaxed level to avoid wasting time
            Rf = lbfgs_stage(Rf, 80, 0.005, 2e-4, np.ones(Pfull), maxiter=100)
            survivors.append(Rf)
        except Exception:
            pass
    finals = []
    for S in survivors:
        if time.time() > DEADLINE - 35:
            break
        # continue with remaining schedule (300,600)
        for (beta, lv, li, al, mi) in [(300, 5e-4, 3e-5, 0.25, 150), (600, 0.0, 0.0, 0.0, 180)]:
            if time.time() > DEADLINE - 32:
                break
            if al == 0.0:
                vw = np.ones(Pfull)
            else:
                vw = np.power(np.maximum(v_all0, 1e-12), al)
                vw /= float(vw.mean())
            S = lbfgs_stage(S, beta, lv, li, vw, maxiter=mi)
            vv = _max_cos(S)
            if vv < Vbest:
                Vbest = vv
                Xbest = S.copy()
        finals.append(S)
    # include current best
    finals.append(Xbest.copy())

    # ---------- LP polish ----------
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
                    mk = np.zeros_like(sel)
                    mk[idxp] = True
                    sel = mk
                    cnt = 3000
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

    # polish top candidates by cosine
    uniq = []
    for F in finals:
        try:
            uniq.append(( _max_cos(F), F))
        except Exception:
            pass
    uniq.sort(key=lambda t: t[0])
    remain = max(5.0, DEADLINE - time.time() - 1.0)
    # give most time to best
    for k, (vv, F) in enumerate(uniq[:3]):
        if time.time() > DEADLINE - 1.5:
            break
        secs = remain * (0.6 if k == 0 else 0.25 if k == 1 else 0.15)
        secs = max(2.0, min(secs, 22.0))
        Fp, vp = lp_polish(F, seconds=secs)
        if vp < Vbest:
            Vbest = vp
            Xbest = Fp.copy()
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    # tiny shrink guard (strict feasibility not needed but keep finite)
    Xbest = np.nan_to_num(Xbest, nan=0.0, posinf=0.0, neginf=0.0)
    nn = np.linalg.norm(Xbest, axis=1, keepdims=True)
    nn[nn == 0] = 1.0
    Xbest = Xbest / nn
    return Xbest
