import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _balanced_count(x, tol=1e-6, need=5):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = g.max()
    cnt = np.sum(g > m - tol, axis=1)
    return int(np.sum(cnt >= need)), float(m), cnt


def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar,
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                          np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x, best


def solve():
    t0 = time.time()
    BUDGET = 172.0
    EXPLORE = 128.0
    rng = np.random.default_rng(12345)
    n, d = 99, 13

    # load base
    xb = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (99, 13):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    xb = a.copy()
        except Exception:
            pass
    if xb is None:
        xb = rng.standard_normal(size=(n, d))
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
        best_val = _max_cos(xb)

    rec_val = best_val
    try:
        r = np.load("record.npy").astype(float)
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        rec_val = _max_cos(r)
    except Exception:
        pass
    rec_bal, _, _ = _balanced_count(xb, tol=1e-6, need=5)
    # record balanced reference (use record file if available)
    try:
        r2 = np.load("record.npy").astype(float)
        r2 /= np.linalg.norm(r2, axis=1, keepdims=True)
        rec_bal = _balanced_count(r2, tol=1e-6, need=5)[0]
    except Exception:
        pass

    best_x = xb.copy()
    # balanced pool: keep best among balanced even if up to 1e-6 worse
    bal_best_x = xb.copy()
    bal_best_val = best_val
    bal_best_cnt = rec_bal

    iu0 = np.triu_indices(n, 1)
    iu_r = iu0[0]
    iu_c = iu0[1]

    def make_fun(alpha, lam, mu):
        def fun(yflat):
            Y = yflat.reshape(n, d)
            nr = np.linalg.norm(Y, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            X = Y / nr
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            # global LSE
            Gv = G[iu_r, iu_c]
            gmax = Gv.max()
            Ev = np.exp(alpha * (Gv - gmax))
            S = Ev.sum()
            L = gmax + np.log(S) / alpha
            Wv = Ev / S
            # per-point soft max
            growmax = G.max(axis=1)
            # E_row
            Erow = np.exp(alpha * (G - growmax[:, None]))
            # kill diag contribution
            np.fill_diagonal(Erow, 0.0)
            Sumi = Erow.sum(axis=1)
            Sumi = np.maximum(Sumi, 1e-300)
            s = growmax + np.log(Sumi) / alpha
            mean_s = s.mean()
            diff = s - mean_s
            var = np.mean(diff ** 2)
            std = np.sqrt(var + 1e-24)
            # softplus part
            dd = s - L
            ad = alpha * dd
            ad_c = np.clip(ad, -50.0, 50.0)
            sig = 1.0 / (1.0 + np.exp(-ad_c))
            # stable softplus/alpha
            sp = np.where(ad > 20.0, dd, np.log1p(np.exp(ad_c)) / alpha)
            T = sp.mean()
            F = L + lam * std + mu * T
            # gradients coefficients
            if std < 1e-14:
                c_std = np.zeros(n)
            else:
                c_std = lam * diff / (n * std)
            cf = c_std + mu * sig / n
            dF_dL = 1.0 - mu * sig.mean()
            # build Q
            Wsym = np.zeros((n, n))
            Wsym[iu_r, iu_c] = Wv
            Wsym[iu_c, iu_r] = Wv
            U = Erow / Sumi[:, None]
            # Q2 = cf*U symmetrised
            A = cf[:, None] * U
            Q = dF_dL * Wsym + A + A.T
            # grad w.r.t Y
            QX = Q @ X
            b = (Q * G).sum(axis=1)
            GX = (QX - b[:, None] * X) / nr
            return float(F), GX.ravel()
        return fun

    alphas = [40.0, 80.0, 160.0, 320.0, 640.0]
    lams = [1.0, 0.5, 0.25, 0.10, 0.05]

    # build start list
    starts = []
    for sg in (0.01, 0.03, 0.08):
        y = xb + sg * rng.standard_normal(size=(n, d))
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        starts.append(y)
    for k in range(5):
        y = rng.standard_normal(size=(n, d))
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        starts.append(y)

    # homotopy per start
    for si, y0 in enumerate(starts):
        if time.time() - t0 > EXPLORE:
            break
        y = y0.copy()
        # normalise (optimizer works on Y with unit rows initially)
        for ai, (alpha, lam) in enumerate(zip(alphas, lams)):
            if time.time() - t0 > EXPLORE:
                break
            mu = lam  # tie mu to lambda schedule
            fun = make_fun(alpha, lam, mu)
            yflat = y.ravel()
            # iteration budget shrinks with alpha
            if alpha <= 80:
                mxit = 60
            elif alpha <= 160:
                mxit = 70
            else:
                mxit = 80
            try:
                res = minimize(fun, yflat, method="L-BFGS-B", jac=True,
                               options={"maxiter": mxit, "ftol": 1e-12,
                                        "gtol": 1e-7, "maxfun": mxit * 4})
                y = res.x.reshape(n, d)
                nr = np.linalg.norm(y, axis=1, keepdims=True)
                nr[nr == 0] = 1.0
                y = y / nr
            except Exception:
                break
            # early time check
            if time.time() - t0 > EXPLORE:
                break
        # evaluate raw
        try:
            raw_v = _max_cos(y)
        except Exception:
            continue
        # short Direction-1-style polish (1.2s) for topology screening
        remain = EXPLORE - (time.time() - t0)
        pol_s = 1.2 if remain > 4 else 0.0
        if pol_s > 0:
            try:
                yp, vp = _lp_polish(y, pol_s)
            except Exception:
                yp, vp = y, raw_v
        else:
            yp, vp = y, raw_v
        cnt, _, _ = _balanced_count(yp, tol=1e-6, need=5)
        # global best update (always, correctness first)
        if vp < best_val:
            best_val = vp
            best_x = yp.copy()
        # balanced pool update: keep if exceeds record count even if up to 1e-6 worse
        # than current best (target topology)
        if cnt > rec_bal and vp < best_val + 1e-6 + 1e-12:
            # keep the most balanced / best among them
            if cnt > bal_best_cnt or (cnt == bal_best_cnt and vp < bal_best_val):
                bal_best_cnt = cnt
                bal_best_val = vp
                bal_best_x = yp.copy()
        elif cnt > bal_best_cnt and vp < bal_best_val + 1e-6:
            if vp < bal_best_val or cnt > bal_best_cnt:
                bal_best_cnt = cnt
                bal_best_val = vp
                bal_best_x = yp.copy()

    # choose final base: prefer absolute best; also polish balanced best if promising
    # final polish remaining time
    rem = BUDGET - (time.time() - t0) - 1.0
    if rem < 5:
        rem = max(2.0, BUDGET - (time.time() - t0) - 1.0)
    # if balanced pool has high degree and is close, give it a polish chance
    candidates = [(best_x, best_val)]
    if bal_best_cnt > rec_bal and bal_best_val < best_val + 1e-6:
        # avoid duplicate
        if np.linalg.norm(bal_best_x - best_x) > 1e-8:
            candidates.append((bal_best_x, bal_best_val))
    # allocate polish time: most to best
    if len(candidates) == 2 and rem > 10:
        t1 = rem * 0.3
        try:
            yp2, vp2 = _lp_polish(candidates[1][0], t1)
            c2, _, _ = _balanced_count(yp2, tol=1e-6, need=5)
            if vp2 < best_val:
                best_val = vp2
                best_x = yp2.copy()
            # if balanced polish beats, keep it if still balanced
        except Exception:
            pass
        rem = BUDGET - (time.time() - t0) - 1.0
        try:
            best_x, best_val = _lp_polish(best_x, max(2.0, rem))
        except Exception:
            pass
    else:
        try:
            best_x, best_val = _lp_polish(best_x, max(2.0, rem))
        except Exception:
            pass

    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    best_x = np.where(np.isfinite(best_x), best_x, 0.0)
    nr = np.linalg.norm(best_x, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    best_x = best_x / nr
    return best_x
