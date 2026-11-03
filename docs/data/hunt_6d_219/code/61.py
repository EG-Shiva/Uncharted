import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def solve():
    t0 = time.time()
    BUDGET = 168.0
    HOP_END = t0 + 122.0
    n, d = 219, 6
    iu0, iu1 = np.triu_indices(n, 1)
    rng = np.random.RandomState(0)

    def _norm(X):
        nn = np.linalg.norm(X, axis=1, keepdims=True)
        nn = np.maximum(nn, 1e-12)
        return X / nn

    def _max_cos(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return float(np.max(G))

    def _per_max(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return np.max(G, axis=1), G

    x_best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=float)
            if v.shape == (n, d):
                v = _norm(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    x_best = v.copy()
        except Exception:
            pass
    if x_best is None:
        x_best = _norm(rng.randn(n, d))
        best_val = _max_cos(x_best)

    # ---------- egalitarian weighted per-point softmax L-BFGS ----------
    def run_egal(X0, beta, W, maxiter):
        y0 = X0.ravel().copy()
        def fun(y):
            Y = y.reshape(n, d)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Y / rn
            G = X @ X.T
            # exclude diagonal from per-row max
            np.fill_diagonal(G, -1e100)
            m = np.max(G, axis=1)
            A = beta * (G - m[:, None])
            np.maximum(A, -50.0, out=A)
            E = W * np.exp(A)
            # W has 0 diag so diag stays 0
            S = np.sum(E, axis=1) + 1e-300
            loss = float(np.sum(m + np.log(S) / beta))
            P = E / S[:, None]
            Ws = P + P.T
            GX = Ws @ X
            dot = np.sum(GX * X, axis=1, keepdims=True)
            GY = (GX - dot * X) / rn
            return loss, GY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-12, "gtol": 1e-7, "maxfun": 20000})
            Y = res.x.reshape(n, d)
            return _norm(Y)
        except Exception:
            return _norm(X0)

    def run_global(X0, beta, maxiter):
        y0 = X0.ravel().copy()
        def fun(y):
            Y = y.reshape(n, d)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Y / rn
            G = X @ X.T
            g = G[iu0, iu1]
            m = float(np.max(g))
            e = np.exp(np.maximum(beta * (g - m), -50.0))
            S = float(np.sum(e)) + 1e-300
            loss = m + np.log(S) / beta
            w = e / S
            Wm = np.zeros((n, n))
            Wm[iu0, iu1] = w
            Wm = Wm + Wm.T
            GX = Wm @ X
            dot = np.sum(GX * X, axis=1, keepdims=True)
            GY = (GX - dot * X) / rn
            return float(loss), GY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-12, "gtol": 1e-7, "maxfun": 20000})
            Y = res.x.reshape(n, d)
            return _norm(Y)
        except Exception:
            return _norm(X0)

    def lp_polish(x, radius=5e-4, max_steps=5, cap=3500, deadline=None):
        x = _norm(x.copy())
        try:
            best = _max_cos(x)
        except Exception:
            return x
        bx = x.copy()
        r = float(radius)
        for _ in range(int(max_steps)):
            if deadline is not None and time.time() > deadline:
                break
            if r < 1e-14:
                break
            try:
                Gf = x @ x.T
                c = Gf[iu0, iu1]
                m = float(c.max())
                thr = m - max(18.0 * r, 1e-9)
                sel = c > thr
                I = iu0[sel]; J = iu1[sel]; csel = c[sel]
                p = len(I)
                if p == 0:
                    break
                if p > cap:
                    part = np.argpartition(-csel, cap - 1)[:cap]
                    I, J, csel = I[part], J[part], csel[part]
                    p = cap
                rows = np.repeat(np.arange(p), 2 * d + 1)
                cols = np.concatenate([I[:, None] * d + np.arange(d),
                                       J[:, None] * d + np.arange(d),
                                       np.full((p, 1), n * d)], 1).ravel()
                vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
                row_eq = np.repeat(np.arange(n), d)
                col_eq = np.arange(n * d)
                dat_eq = x.ravel()
                A_eq = coo_matrix((dat_eq, (row_eq, col_eq)), shape=(n, n * d + 1)).tocsr()
                obj = np.zeros(n * d + 1); obj[-1] = 1.0
                res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-r, r)] * (n * d) + [(None, None)], method="highs")
                if res.status != 0:
                    r *= 0.5
                    continue
                y = x + res.x[:-1].reshape(n, d)
                y = _norm(y)
                v = _max_cos(y)
                if v < best:
                    x, best = y, v
                    bx = y.copy()
                    r = min(r * 1.4, 3e-3)
                else:
                    r *= 0.5
            except Exception:
                break
        return bx

    def tangent_jitter(X, idx, sigma):
        if len(idx) == 0:
            return X
        X2 = X.copy()
        P = rng.randn(len(idx), d)
        Xi = X2[idx]
        P -= np.sum(P * Xi, axis=1, keepdims=True) * Xi
        nn = np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
        P /= nn
        amp = rng.randn(len(idx), 1) * float(sigma)
        X2[idx] += amp * P
        X2[idx] = _norm(X2[idx])
        return X2

    # initial stress weights
    EPS = 0.15
    def compute_weights(X, beta_s=600.0, slack_add=1e-4, zero_cnt=None):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mg = float(np.max(G))
        slack = mg - G
        A = beta_s * (G - mg)
        np.maximum(A, -50.0, out=A)
        E = np.exp(A)
        np.fill_diagonal(E, 0.0)
        active = slack < slack_add
        np.fill_diagonal(active, False)
        W = np.full((n, n), EPS, dtype=float)
        np.fill_diagonal(W, 0.0)
        row_stress = np.zeros(n)
        if np.any(active):
            vals = E[active]
            mean_e = float(np.mean(vals)) + 1e-300
            dual = np.zeros((n, n))
            dual[active] = vals / mean_e
            dual = np.minimum(dual, 5.0)
            # update zero counters
            if zero_cnt is not None:
                low = active & (dual < 0.02)
                zero_cnt[low] += 1
                zero_cnt[~(low)] = 0 if False else zero_cnt[~(low)] * 0
                # reset non-low active to 0
                reset = active & (~low)
                zero_cnt[reset] = 0
                # persistently zero -> down-weight
                persist = zero_cnt >= 2
                W[active] += dual[active]
                W[persist] = EPS * 0.5
            else:
                W[active] += dual[active]
            # row stress for hub detection
            row_stress = np.sum(dual, axis=1)
        per, _ = _per_max(X)
        return W, row_stress, per, mg

    zero_cnt = np.zeros((n, n), dtype=int)
    x_cur = x_best.copy()
    Wmat, row_stress, per_row, mg0 = compute_weights(x_cur, zero_cnt=zero_cnt)
    bx = x_best.copy()
    best = best_val

    # quick egal warm start at low beta
    try:
        if time.time() < HOP_END:
            xc = run_egal(x_cur, 100.0, Wmat, maxiter=80)
            vc = _max_cos(xc)
            if vc < best:
                best, bx = vc, xc.copy()
                x_cur = xc.copy()
    except Exception:
        pass

    betas = [100.0, 200.0, 400.0, 800.0, 1500.0, 2500.0, 3000.0]
    stage = 0
    lp_every = 2
    while time.time() < HOP_END and (time.time() - t0) < BUDGET - 45:
        for bi, beta in enumerate(betas):
            if time.time() >= HOP_END or (time.time() - t0) >= BUDGET - 45:
                break
            stage += 1
            mit = 150 if beta < 1000 else 180
            # anneal stress extraction beta with stage beta partially
            try:
                x_new = run_egal(x_cur, float(beta), Wmat, maxiter=mit)
            except Exception:
                continue
            try:
                v_new = _max_cos(x_new)
            except Exception:
                continue
            if v_new < best:
                best, bx = v_new, x_new.copy()
            # accept move if not too bad, else drift back toward best
            if v_new < _max_cos(x_cur) + 0.004 or stage % 5 == 0:
                x_cur = x_new
            else:
                # blend toward best to stay near good basin but keep novelty
                try:
                    alpha = 0.15
                    xc2 = (1 - alpha) * x_new + alpha * bx
                    x_cur = _norm(xc2)
                except Exception:
                    x_cur = x_new
            # short LP polish every lp_every stages
            if stage % lp_every == 0:
                try:
                    dl = time.time() + 4.0
                    xp = lp_polish(x_cur, radius=5e-4, max_steps=4, deadline=dl)
                    vp = _max_cos(xp)
                    if vp < best:
                        best, bx = vp, xp.copy()
                    # keep polished as current if improved over current
                    if vp <= v_new:
                        x_cur = xp
                except Exception:
                    pass
            # recompute stresses, update W, drop persistently zero-stress
            try:
                Wmat, row_stress, per_row, mg = compute_weights(x_cur, beta_s=600.0, slack_add=1e-4, zero_cnt=zero_cnt)
            except Exception:
                continue
            # hub / rattler targeted micro-jitter to change topology
            try:
                if stage % 2 == 1:
                    rattler = int(np.argmin(per_row))
                    # hubs: top row_stress
                    hubs = np.argsort(-row_stress)[:4]
                    # jitter rattler a bit more to force contacts
                    x_cur = tangent_jitter(x_cur, np.array([rattler], dtype=int), 0.012)
                    x_cur = tangent_jitter(x_cur, np.array(hubs, dtype=int), 0.008)
                    # tiny global noise
                    Pn = rng.randn(n, d)
                    Pn -= np.sum(Pn * x_cur, axis=1, keepdims=True) * x_cur
                    x_cur = _norm(x_cur + 0.0015 * Pn)
            except Exception:
                pass
            # safety: if drifted far, pull back to best with jitter for new basin
            try:
                vc = _max_cos(x_cur)
                if vc > best + 0.008:
                    pert = rng.randn(n, d)
                    pert -= np.sum(pert * bx, axis=1, keepdims=True) * bx
                    x_cur = _norm(bx + 0.01 * pert)
                    zero_cnt[:] = 0
                    Wmat, row_stress, per_row, mg = compute_weights(x_cur, beta_s=600.0, slack_add=1e-4, zero_cnt=None)
                    zero_cnt[:] = 0
            except Exception:
                pass

    # final sharpening of best with global + egal high beta and LP
    try:
        for hb, it in [(800.0, 80), (1800.0, 80), (3000.0, 60)]:
            if (time.time() - t0) > BUDGET - 28:
                break
            # egal high beta ~ per-point, then global
            xn = run_egal(bx, float(hb), Wmat, maxiter=it)
            vn = _max_cos(xn)
            if vn < best:
                best, bx = vn, xn.copy()
            xg = run_global(bx, float(hb), maxiter=it)
            vg = _max_cos(xg)
            if vg < best:
                best, bx = vg, xg.copy()
    except Exception:
        pass
    try:
        remain = (t0 + BUDGET - 3.0) - time.time()
        if remain > 6:
            dl = time.time() + min(remain - 1.0, 9.0)
            xp = lp_polish(bx, radius=4e-4, max_steps=12, deadline=dl)
            vp = _max_cos(xp)
            if vp < best:
                best, bx = vp, xp.copy()
    except Exception:
        pass

    bx = _norm(bx)
    # fallback
    try:
        if not np.all(np.isfinite(bx)) or _max_cos(bx) > best_val + 1e-12 and best > best_val:
            bx = x_best.copy()
    except Exception:
        bx = x_best.copy()
    # keep absolute best encountered
    try:
        if best < _max_cos(bx) and False:
            pass
    except Exception:
        pass
    # if our search failed to beat incumbent, return incumbent (still valid)
    try:
        vbx = _max_cos(bx)
        if vbx > best_val:
            bx = x_best.copy()
    except Exception:
        bx = x_best.copy()
    bx = _norm(bx)
    return bx
