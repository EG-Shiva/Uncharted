import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 121, 8
    iu = np.triu_indices(n, 1)
    Iall, Jall = iu[0], iu[1]
    M = len(Iall)
    n_over_d = n / float(d)

    # load incumbent
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
    Xinc = cands[int(np.argmin(vals))].copy()
    Vinc = float(min(vals))
    Xbest = Xinc.copy()
    Vbest = Vinc

    # ---------- stress estimation ----------
    def estimate_stress(X, tol=2e-4, maxiter=120):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        c_all = G[Iall, Jall]
        sel = c_all > m - tol
        # cap to tightest 2500 to bound cost
        if int(sel.sum()) > 2500:
            thr = np.sort(c_all)[-2500]
            sel = c_all >= thr
        I = Iall[sel]; J = Jall[sel]
        c = c_all[sel]
        P = len(c)
        if P == 0:
            return None
        Tij = X[J] - c[:, None] * X[I]
        Tji = X[I] - c[:, None] * X[J]
        # objective over theta with w = P*softmax(theta)
        def fun_grad(th):
            mx = float(th.max()) if P else 0.0
            e = np.exp(th - mx)
            s = float(e.sum())
            w = P * e / s
            R = np.zeros((n, d))
            np.add.at(R, I, w[:, None] * Tij)
            np.add.at(R, J, w[:, None] * Tji)
            Rval = float((R * R).sum())
            # grad wrt w
            # r_I . Tij + r_J . Tji
            gw = 2.0 * ((R[I] * Tij).sum(axis=1) + (R[J] * Tji).sum(axis=1))
            wdotg = float((w * gw).sum())
            meanwg = wdotg / P
            gth = w * (gw - meanwg)
            return Rval, gth
        th0 = np.zeros(P)
        try:
            res = minimize(lambda th: fun_grad(th), th0, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-20, "gtol": 1e-9, "maxfun": int(maxiter*5)})
            th = np.asarray(res.x)
        except Exception:
            th = th0
        mx = float(th.max())
        e = np.exp(th - mx); s = float(e.sum())
        w = P * e / s
        # per point stats
        S = np.zeros(n)
        np.add.at(S, I, w); np.add.at(S, J, w)
        R = np.zeros((n, d))
        np.add.at(R, I, w[:, None] * Tij)
        np.add.at(R, J, w[:, None] * Tji)
        rnorm = np.linalg.norm(R, axis=1)
        return {"I": I, "J": J, "c": c, "w": w, "S": S, "rnorm": rnorm,
                "Rval": float((R*R).sum()), "m": m, "P": P}

    def imbalance_metric(X):
        # contact std at 1e-5 + residual proxy with uniform weights
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        for tol in (1e-5, 1e-4):
            cnt = (G > m - tol).sum(axis=1)
            std5 = float(cnt.std()) if tol == 1e-5 else None
        # uniform residual at 1e-4
        sel = G > m - 1e-4
        R = np.zeros((n, d))
        # sum of tangent vectors
        for i in range(n):
            js = np.where(sel[i])[0]
            js = js[js != i]
            if len(js):
                Pi = X[js] - (X[js] @ X[i])[:, None] * X[i][None, :]
                R[i] = Pi.sum(axis=0) / max(len(js), 1)
        runif = float(np.linalg.norm(R, axis=1).mean())
        return std5 + 5.0 * runif, std5, runif

    # ---------- weighted LSE + var + iso factory ----------
    def make_fun(beta, a_full, lam_var, lam_iso, beta_v):
        beta = float(beta); beta_v = float(beta_v)
        a_full = np.asarray(a_full, dtype=float)
        def fun(flat):
            X = flat.reshape(n, d)
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = X / nr
            G = Y @ Y.T
            c = G[Iall, Jall]
            m = float(c.max())
            # weighted LSE
            ex = beta * (c - m)
            # clip for overflow (ex<=0)
            ex = np.maximum(ex, -60.0)
            ew = a_full * np.exp(ex)
            s = float(ew.sum())
            if not np.isfinite(s) or s <= 0:
                return float(m), np.zeros_like(flat)
            f = m + np.log(s) / beta
            wls = ew / s
            # soft degrees for variance
            exv = np.maximum(beta_v * (c - m), -60.0)
            ev = np.exp(exv)
            deg = np.zeros(n)
            np.add.at(deg, Iall, ev)
            np.add.at(deg, Jall, ev)
            mean_deg = float(deg.mean())
            diff = deg - mean_deg
            var = float((diff * diff).mean())
            # isotropy
            YtY = Y.T @ Y
            Cmat = YtY - n_over_d * np.eye(d)
            iso = float((Cmat * Cmat).sum())
            total = f + lam_var * var + lam_iso * iso
            # grad wrt c
            # dvar/dc
            dvar = 2.0 * beta_v * ev * (deg[Iall] + deg[Jall] - 2.0 * mean_deg) / n
            dLdc = wls + lam_var * dvar
            W = np.zeros((n, n))
            W[Iall, Jall] = dLdc
            W[Jall, Iall] = dLdc
            GY = W @ Y
            # iso grad
            if lam_iso != 0:
                GY = GY + lam_iso * (4.0 * (Y @ Cmat))
            dot = (GY * Y).sum(axis=1, keepdims=True)
            GX = (GY - dot * Y) / nr
            if not np.all(np.isfinite(GX)):
                GX = np.nan_to_num(GX, nan=0.0, posinf=0.0, neginf=0.0)
            return float(total), GX.ravel()
        return fun

    def lbfgs_run(Xinit, beta, a_full, lam_var, lam_iso, beta_v, maxiter):
        fun = make_fun(beta, a_full, lam_var, lam_iso, beta_v)
        Xf = Xinit.ravel().copy()
        try:
            res = minimize(fun, Xf, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-20, "gtol": 1e-12, "maxfun": int(maxiter*6)})
            Xf = np.asarray(res.x, dtype=float)
        except Exception:
            pass
        X = Xf.reshape(n, d)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X

    def pure_lse_fun_factory(beta):
        b = float(beta)
        def fun(flat):
            X = flat.reshape(n, d)
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = X / nr
            G = Y @ Y.T
            c = G[Iall, Jall]
            m = float(c.max())
            e = np.exp(b * np.maximum(c - m, -60.0))
            s = float(e.sum())
            f = m + np.log(s) / b
            w = e / s
            W = np.zeros((n, n))
            W[Iall, Jall] = w; W[Jall, Iall] = w
            GY = W @ Y
            dot = (GY * Y).sum(axis=1, keepdims=True)
            GX = (GY - dot * Y) / nr
            return float(f), GX.ravel()
        return fun

    def pure_anneal(X, betas, iters):
        Xf = X.ravel().copy()
        for beta, mi in zip(betas, iters):
            if time.time() > DEADLINE - 28.0:
                break
            fun = pure_lse_fun_factory(float(beta))
            try:
                res = minimize(fun, Xf, jac=True, method="L-BFGS-B",
                               options={"maxiter": int(mi), "ftol": 1e-20, "gtol": 1e-12, "maxfun": int(mi*6)})
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
        it = 0
        while time.time() - start < seconds and time.time() < DEADLINE - 1.5:
            it += 1
            if it > 80 or radius < 1e-14:
                break
            G = x @ x.T
            cf = G[Iall, Jall]
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
            I = Iall[sel]; J = Jall[sel]
            p = cnt
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None]*d + np.arange(d), J[:, None]*d + np.arange(d), np.full((p,1), n*d)], 1).ravel()
            valsA = np.concatenate([x[J], x[I], -np.ones((p,1))], 1).ravel()
            try:
                A_ub = coo_matrix((valsA, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            obj = np.zeros(nv); obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius*1.6, 0.02)
            else:
                radius *= 0.5
        return x, best

    def find_hole(base, nsample=15000, refine=120):
        C = rng.standard_normal((nsample, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        D = C @ base.T
        cm = D.max(axis=1)
        k = int(np.argmin(cm))
        bc = C[k].copy(); cur = float(cm[k])
        for st in range(refine):
            sig = 0.03 * (1.0 - st / float(refine)) + 0.001
            prop = bc + sig * rng.standard_normal(d)
            prop /= np.linalg.norm(prop)
            md = float((prop @ base.T).max())
            if md < cur:
                bc = prop; cur = md
        return bc

    # ---------- diagnostics on incumbent ----------
    try:
        st0 = estimate_stress(Xinc, tol=2e-4, maxiter=100)
    except Exception:
        st0 = None
    base_imb, base_std, base_res = imbalance_metric(Xinc)
    # per-point badness: low total stress or large residual or few contacts
    Ginc = Xinc @ Xinc.T
    np.fill_diagonal(Ginc, -2.0)
    m_inc = float(Ginc.max())
    cnt_inc = (Ginc > m_inc - 1e-5).sum(axis=1)
    rowmax = Ginc.max(axis=1)
    bad_score = np.zeros(n)
    if st0 is not None:
        S = st0["S"]; Rn = st0["rnorm"]
        Smed = float(np.median(S)) + 1e-12
        bad_score = Rn / (S + 0.3 * Smed) + 0.1 * (Smed / (S + 0.3*Smed))
    # combine with low contacts and high rowmax
    bad_score = bad_score + 0.2 * (cnt_inc.max() - cnt_inc) / max(cnt_inc.max(), 1) + 0.5 * (rowmax - rowmax.min()) / max(rowmax.max()-rowmax.min(), 1e-12)
    order_bad = np.argsort(-bad_score)

    # pair weights inversely related to stress
    def build_weights(st):
        a = np.ones(M)
        if st is None:
            return a
        w = st["w"]; I = st["I"]; J = st["J"]
        med = float(np.median(w)) + 1e-12
        # map pair index: need lookup from (I,J) to flat position
        # build dict via search: use sorting trick
        # M=7260 small: build index matrix
        idxmat = np.full((n, n), -1, dtype=int)
        idxmat[Iall, Jall] = np.arange(M)
        idxmat[Jall, Iall] = np.arange(M)
        kidx = idxmat[I, J]
        inv = med / (w + 0.3 * med)
        inv = np.clip(inv, 0.25, 4.0)
        # power 0.7
        inv = inv ** 0.7
        a[kidx] = inv
        # normalize to mean 1
        a = a / (a.mean())
        return a

    a_inv = build_weights(st0)

    # ---------- generate rebalanced candidates ----------
    pool = [(Xinc.copy(), Vinc, base_imb)]
    # targeted perturbations (larger noise on bad points)
    sigmas = [0.02, 0.035, 0.05]
    for si, sig in enumerate(sigmas):
        if time.time() > DEADLINE - 90:
            break
        Xp = Xinc.copy()
        noise = rng.standard_normal((n, d))
        # scale per point by badness rank
        wscale = np.ones(n) * 0.4
        topk = order_bad[:12]
        wscale[topk] = 1.6
        midk = order_bad[12:40]
        wscale[midk] = 1.0
        Xp = Xp + sig * wscale[:, None] * noise
        Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        # relaxed rebalancing: beta 250, penalties on
        X1 = lbfgs_run(Xp, 250.0, a_inv, 2e-4, 8e-5, 250.0, 150)
        if time.time() > DEADLINE - 85:
            # evaluate and store
            pass
        v1 = _max_cos(X1)
        imb1, _, _ = imbalance_metric(X1)
        pool.append((X1, v1, imb1))
        if v1 < Vbest and imb1 < base_imb + 2.0:
            # keep global best if actually better
            if v1 < Vbest:
                Vbest, Xbest = v1, X1.copy()
        # second stage lowering target: beta 450, reduced penalties
        X2 = lbfgs_run(X1, 450.0, a_inv, 5e-5, 2e-5, 450.0, 120)
        v2 = _max_cos(X2)
        imb2, _, _ = imbalance_metric(X2)
        pool.append((X2, v2, imb2))
        if v2 < Vbest:
            Vbest, Xbest = v2, X2.copy()

    # hole-regrow candidates: drop worst 4 and reinsert
    for rep in range(2):
        if time.time() > DEADLINE - 70:
            break
        Gc = Xinc @ Xinc.T
        np.fill_diagonal(Gc, -2.0)
        rm = Gc.max(axis=1)
        # worst = high rowmax + bad_score
        worst = np.argsort(-(rm + 0.3 * bad_score))[:4]
        keep_mask = np.ones(n, dtype=bool); keep_mask[worst] = False
        base = Xinc[keep_mask]
        newpts = []
        cur_base = base.copy()
        for _ in range(4):
            h = find_hole(cur_base, nsample=12000, refine=80)
            newpts.append(h)
            cur_base = np.vstack([cur_base, h[None, :]])
        Xh = np.vstack([base, np.stack(newpts)])
        # random order to avoid index bias
        perm = rng.permutation(n)
        # need to keep mapping? just permute
        Xh = Xh[perm] if len(Xh) == n else Xh
        Xh /= np.linalg.norm(Xh, axis=1, keepdims=True)
        X1 = lbfgs_run(Xh, 220.0, np.ones(M), 2e-4, 8e-5, 220.0, 150)
        v1 = _max_cos(X1)
        imb1, _, _ = imbalance_metric(X1)
        pool.append((X1, v1, imb1))
        if v1 < Vbest:
            Vbest, Xbest = v1, X1.copy()
        X2 = lbfgs_run(X1, 500.0, np.ones(M), 4e-5, 1.5e-5, 500.0, 120)
        v2 = _max_cos(X2)
        imb2, _, _ = imbalance_metric(X2)
        pool.append((X2, v2, imb2))
        if v2 < Vbest:
            Vbest, Xbest = v2, X2.copy()

    # ---------- select most rebalanced with acceptable cosine ----------
    # retain configs whose imbalance improves even if slightly worse (relaxed 0.465-0.475)
    pool_sorted = sorted(pool, key=lambda t: (t[2], t[1]))
    Xsel = Xinc.copy()
    # prefer improved imbalance and cosine < 0.475 (relaxed) then <0.465
    best_imb = base_imb
    for Xc, vc, ic in pool_sorted:
        if vc < 0.475 and ic < best_imb - 1e-6:
            Xsel = Xc.copy(); best_imb = ic
            break
    else:
        # fallback: best cosine among pool with vc<0.475
        cands475 = [t for t in pool if t[1] < 0.475]
        if cands475:
            Xsel = min(cands475, key=lambda t: t[1])[0].copy()
    # if nothing better, use current best
    if _max_cos(Xsel) > 0.475:
        Xsel = Xbest.copy()

    # ---------- gradual penalty removal through 0.462/0.461/0.460 ----------
    # re-estimate stress on selected for fresh weights
    try:
        st_sel = estimate_stress(Xsel, tol=1.5e-4, maxiter=80)
        a_sel = build_weights(st_sel)
    except Exception:
        a_sel = np.ones(M)
    stages = [
        (350.0, a_sel, 8e-5, 3e-5, 350.0, 130),
        (700.0, np.ones(M), 2e-5, 8e-6, 700.0, 130),
        (1200.0, np.ones(M), 0.0, 3e-6, 1200.0, 130),
    ]
    Xcur = Xsel.copy()
    for beta, a_, lv, li, bv, mi in stages:
        if time.time() > DEADLINE - 45:
            break
        Xcur = lbfgs_run(Xcur, beta, a_, lv, li, bv, mi)
        vc = _max_cos(Xcur)
        if vc < Vbest:
            Vbest, Xbest = vc, Xcur.copy()
    # pure annealing lowering through 0.460
    if time.time() < DEADLINE - 32:
        Xcur = pure_anneal(Xcur, [900.0, 1600.0, 2600.0], [160, 170, 200])
        vc = _max_cos(Xcur)
        if vc < Vbest:
            Vbest, Xbest = vc, Xcur.copy()
        # also anneal incumbent-best in parallel if different
        if time.time() < DEADLINE - 30:
            Xa = pure_anneal(Xbest, [1400.0, 2400.0], [120, 150])
            va = _max_cos(Xa)
            if va < Vbest:
                Vbest, Xbest = va, Xa.copy()
                Xcur = Xa.copy()

    # ---------- final LP polish 20-30s ----------
    remain = DEADLINE - time.time()
    lp_time = min(28.0, max(8.0, remain - 2.0))
    if lp_time > 3:
        # polish top two: Xcur and Xbest
        for Xp in (Xbest.copy(), Xcur.copy()):
            if time.time() > DEADLINE - 2.0:
                break
            try:
                Xl, vl = lp_polish(Xp, seconds=lp_time/2.0)
            except Exception:
                continue
            if vl < Vbest:
                Vbest, Xbest = vl, Xl.copy()

    Xbest /= np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
