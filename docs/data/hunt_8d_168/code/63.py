import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _stress_weights(X, I, J, gg, iters=30):
    p = len(I)
    if p == 0:
        return np.zeros(0)
    w = np.ones(p, dtype=np.float64) / p
    Ti = X[J] - gg[:, None] * X[I]
    Tj = X[I] - gg[:, None] * X[J]
    n = X.shape[0]
    for _ in range(iters):
        r = np.zeros_like(X)
        np.add.at(r, I, w[:, None] * Ti)
        np.add.at(r, J, w[:, None] * Tj)
        Ri = r[I]
        Rj = r[J]
        grad = (Ti * Ri).sum(axis=1) + (Tj * Rj).sum(axis=1)
        gmax = grad.max()
        eta = 8.0
        w = w * np.exp(-eta * (grad - gmax))
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            w = np.ones(p, dtype=np.float64) / p
            break
        w /= s
    return w

def _lp_polish(x, deadline, r0=1.2e-3, cap=4000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            c = (x @ x.T)[iu]
        except Exception:
            break
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        if len(I) == 0:
            radius *= 0.5
            continue
        if len(I) > cap:
            idx = np.argsort(c[sel])[-cap:]
            I, J = I[idx], J[idx]
            cc = c[sel][idx]
        else:
            cc = c[sel]
        p = len(I)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            vv = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    BUDGET = 175.0
    deadline = t0 + BUDGET
    n, d = 168, 8
    rng = np.random.RandomState(12345)
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape != (n, d):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            va = _max_cos(a)
            if va < best_val:
                best_val = va
                best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.randn(n, d)
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    incumbent = best.copy()
    inc_val = best_val
    global_best = best.copy()
    global_val = best_val

    def consider(X):
        nonlocal global_best, global_val, incumbent, inc_val
        try:
            v = _max_cos(X)
        except Exception:
            return
        if not np.isfinite(v):
            return
        if v < global_val:
            global_val = v
            global_best = X.copy()
        if v < inc_val:
            inc_val = v
            incumbent = X.copy()

    # ---------- Stage 1: stress/contact-recruiting tangent Adam ~60s ----------
    stage1_end = min(t0 + 62.0, deadline - 90.0)
    if stage1_end < t0 + 5:
        stage1_end = t0 + 5
    try:
        X = incumbent.copy()
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        m_ad = np.zeros_like(X)
        v_ad = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        lr = 0.0035
        w_var = 50.0
        w_slack = 20.0
        slack_tol = 1e-4
        beta0, beta1 = 800.0, 3000.0
        ban = set()
        drop_counts = {}
        deg_factor = np.ones(n)
        step = 0
        # estimate total steps for beta schedule
        max_steps_est = 6000
        iu = np.triu_indices(n, 1)
        while time.time() < stage1_end:
            # allow multiple inner steps per time check
            for _inner in range(100):
                frac = min(1.0, step / max_steps_est)
                beta = beta0 + (beta1 - beta0) * frac
                G = X @ X.T
                np.fill_diagonal(G, -10.0)
                M = float(G.max())
                # per-row max
                jstar = G.argmax(axis=1)
                mi = G[np.arange(n), jstar]
                mean_mi = float(mi.mean())
                # global softmax weights
                E = np.exp(beta * (G - M))
                np.fill_diagonal(E, 0.0)
                # delete banned constraints
                if len(ban) > 0:
                    # ban is set of (i,j) with i<j
                    for (bi, bj) in list(ban):
                        if bi < n and bj < n:
                            E[bi, bj] = 0.0
                            E[bj, bi] = 0.0
                S = float(E.sum() / 2.0) + 1e-30
                Grad_g = (E @ X) / S
                # variance grad via hard argmax
                coeff_var = 2.0 * (mi - mean_mi) / n  # (n,)
                Grad_var = np.zeros_like(X)
                # accumulate: row i gets coeff*X[jstar], row jstar gets coeff*X[i]
                # vectorized via add.at
                np.add.at(Grad_var, np.arange(n), coeff_var[:, None] * X[jstar])
                # for the second term need per-i contribution to jstar row
                # coeff per i contributes to row jstar[i] with vector X[i]
                np.add.at(Grad_var, jstar, (coeff_var[:, None] * X))
                # slack grad
                diff = (M - slack_tol) - mi  # positive if slack
                slack_mask = diff > 0
                Grad_sl = np.zeros_like(X)
                if np.any(slack_mask):
                    coeff_sl = np.zeros(n)
                    coeff_sl[slack_mask] = -2.0 * diff[slack_mask] / n
                    # d/dmi, gradient pushes mi up
                    np.add.at(Grad_sl, np.arange(n)[slack_mask], (coeff_sl[slack_mask, None] * X[jstar[slack_mask]]))
                    # counterpart
                    # need grouped add for jstar of slack only
                    js = jstar[slack_mask]
                    cs = coeff_sl[slack_mask]
                    Xs = X[slack_mask]
                    np.add.at(Grad_sl, js, (cs[:, None] * Xs))
                Grad = Grad_g + w_var * Grad_var + w_slack * Grad_sl
                # reweight inversely to weighted contact degree
                Grad = Grad * deg_factor[:, None]
                # tangent project
                dot = (Grad * X).sum(axis=1, keepdims=True)
                Gt = Grad - dot * X
                # clip huge
                nn = np.sqrt((Gt * Gt).sum(axis=1, keepdims=True)) + 1e-30
                sc = np.minimum(1.0, 3.0 / (nn + 1e-12))
                Gt = Gt * sc
                step += 1
                t = step
                m_ad = b1 * m_ad + (1 - b1) * Gt
                v_ad = b2 * v_ad + (1 - b2) * (Gt * Gt)
                mhat = m_ad / (1 - b1 ** t)
                vhat = v_ad / (1 - b2 ** t)
                DX = -lr * mhat / (np.sqrt(vhat) + eps)
                # tangentize DX
                d2 = (DX * X).sum(axis=1, keepdims=True)
                DX = DX - d2 * X
                # limit step norm
                mxn = float(np.sqrt((DX * DX).sum(axis=1).max())) + 1e-30
                if mxn > 0.01:
                    DX *= (0.01 / mxn)
                X = X + DX
                X /= np.linalg.norm(X, axis=1, keepdims=True)
                # track best occasionally
                if step % 100 == 0:
                    try:
                        vv = _max_cos(X)
                        if vv < global_val:
                            consider(X)
                    except Exception:
                        pass
                # stress update every 500 steps
                if step % 500 == 0:
                    try:
                        G2 = X @ X.T
                        np.fill_diagonal(G2, -10.0)
                        M2 = float(G2.max())
                        sel = G2[iu] > (M2 - 1e-4)
                        I = iu[0][sel]; J = iu[1][sel]
                        gg = G2[I, J]
                        if len(gg) > 0:
                            if len(gg) > 5000:
                                idx = np.argsort(gg)[-5000:]
                                I, J, gg = I[idx], J[idx], gg[idx]
                            w = _stress_weights(X, I, J, gg, iters=30)
                            p = len(w)
                            if p > 0:
                                tiny = w < (0.2 / p)
                                # update persistent counts
                                for k in np.where(tiny)[0]:
                                    a_ = int(I[k]); b_ = int(J[k])
                                    if a_ > b_:
                                        a_, b_ = b_, a_
                                    key = (a_, b_)
                                    drop_counts[key] = drop_counts.get(key, 0) + 1
                                    if drop_counts[key] >= 2:
                                        ban.add(key)
                                for k in np.where(~tiny)[0]:
                                    a_ = int(I[k]); b_ = int(J[k])
                                    if a_ > b_:
                                        a_, b_ = b_, a_
                                    key = (a_, b_)
                                    if key in drop_counts:
                                        drop_counts[key] = 0
                                # weighted degree
                                deg = np.zeros(n)
                                np.add.at(deg, I, w)
                                np.add.at(deg, J, w)
                                mdeg = float(deg.mean()) + 1e-30
                                d0 = 0.5 * mdeg
                                fac = (mdeg + d0) / (deg + d0)
                                fac = np.clip(fac, 0.2, 5.0)
                                # normalize to mean 1
                                fac = fac / (fac.mean() + 1e-30)
                                deg_factor = fac
                    except Exception:
                        pass
                    if time.time() >= stage1_end:
                        break
            if time.time() >= stage1_end:
                break
        consider(X)
    except Exception:
        pass

    # ---------- Stage 2: feasibility homotopy ----------
    try:
        M0 = global_val
        deltas = [1e-5, 3e-5, 1e-4]
        targets = [M0 - dd for dd in deltas]
        for ti, c in enumerate(targets):
            if time.time() > deadline - 55.0:
                break
            # only attempt lower target if previous succeeded (or first)
            # (targets are decreasing, so just attempt in order; if fail, try next smaller? spec says accept lower only when zero -> break on fail)
            try:
                Xh = global_best.copy()
                # tiny jitter to break symmetry
                jit = rng.randn(n, d) * 5e-5
                jit -= (jit * Xh).sum(axis=1, keepdims=True) * Xh
                Xh += jit
                Xh /= np.linalg.norm(Xh, axis=1, keepdims=True)
            except Exception:
                continue
            m_ad = np.zeros_like(Xh)
            v_ad = np.zeros_like(Xh)
            b1, b2, eps = 0.9, 0.999, 1e-8
            lr = 0.004
            t = 0
            per_end = min(deadline - 50.0, time.time() + 18.0)
            # quick check if already feasible
            try:
                if _max_cos(Xh) <= c:
                    consider(Xh)
                    continue
            except Exception:
                pass
            success = False
            while time.time() < per_end:
                for _inner in range(100):
                    t += 1
                    G = Xh @ Xh.T
                    np.fill_diagonal(G, -10.0)
                    V = G - c
                    pos = V > 0
                    if not np.any(pos):
                        success = True
                        break
                    Wv = np.zeros_like(G)
                    Wv[pos] = 2.0 * V[pos]
                    # symmetrize (G symmetric, pos symmetric) already symmetric
                    Grad = (Wv @ Xh)
                    # scale: divide by number of violations to keep lr stable
                    nv = float(pos.sum() / 2.0) + 1e-30
                    Grad = Grad / max(nv, 1.0) * 50.0
                    dot = (Grad * Xh).sum(axis=1, keepdims=True)
                    Gt = Grad - dot * Xh
                    nn = np.sqrt((Gt * Gt).sum(axis=1, keepdims=True)) + 1e-30
                    sc = np.minimum(1.0, 2.0 / (nn + 1e-12))
                    Gt = Gt * sc
                    m_ad = b1 * m_ad + (1 - b1) * Gt
                    v_ad = b2 * v_ad + (1 - b2) * (Gt * Gt)
                    mhat = m_ad / (1 - b1 ** t)
                    vhat = v_ad / (1 - b2 ** t)
                    DX = -lr * mhat / (np.sqrt(vhat) + eps)
                    d2 = (DX * Xh).sum(axis=1, keepdims=True)
                    DX = DX - d2 * Xh
                    mxn = float(np.sqrt((DX * DX).sum(axis=1).max())) + 1e-30
                    if mxn > 0.008:
                        DX *= (0.008 / mxn)
                    Xh = Xh + DX
                    Xh /= np.linalg.norm(Xh, axis=1, keepdims=True)
                    if _inner % 25 == 0:
                        try:
                            if _max_cos(Xh) <= c:
                                success = True
                                break
                        except Exception:
                            pass
                if success:
                    break
                try:
                    if _max_cos(Xh) <= c:
                        success = True
                        break
                except Exception:
                    break
                if time.time() >= per_end:
                    break
            try:
                mv = _max_cos(Xh)
            except Exception:
                continue
            if success or mv <= c + 1e-12:
                consider(Xh)
                # continue to next lower target
            else:
                # keep best encountered even if not feasible? only if improved
                if mv < global_val:
                    consider(Xh)
                break
    except Exception:
        pass

    # ---------- Stage 3: wide LP squeeze ----------
    try:
        lp_deadline = deadline - 2.0
        if lp_deadline > time.time() + 5:
            Xf = _lp_polish(global_best, lp_deadline, r0=1.2e-3, cap=4000)
            consider(Xf)
    except Exception:
        pass

    out = global_best / np.linalg.norm(global_best, axis=1, keepdims=True)
    return out
