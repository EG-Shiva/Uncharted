import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

def _kick(x, rng, frac=0.2, lo=0.03, hi=0.08):
    y = x.copy()
    n, d = y.shape
    k = max(1, int(round(n * frac)))
    idx = rng.choice(n, size=k, replace=False)
    for i in idx:
        xi = y[i].copy()
        r = rng.randn(d)
        r -= float(r @ xi) * xi
        rn = float(np.linalg.norm(r))
        if rn < 1e-12:
            r = rng.randn(d)
            r -= float(r @ xi) * xi
            rn = float(np.linalg.norm(r))
            if rn < 1e-12:
                continue
        r /= rn
        th = float(rng.uniform(lo, hi))
        # random sign via r already random direction
        y[i] = np.cos(th) * xi + np.sin(th) * r
    return _norm(y)

def _lp_polish(x, deadline, r0=0.003, cap=1400):
    x = _norm(x.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x, 1.0
    radius = float(r0)
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            G = x @ x.T
            c = G[iu0, iu1]
            m = float(c.max())
            sel = c > m - max(20.0 * radius, 1e-9)
            if int(sel.sum()) == 0:
                sel[int(np.argmax(c))] = True
            if int(sel.sum()) > cap:
                idx = np.argsort(-c)[:cap]
                s2 = np.zeros_like(sel, dtype=bool)
                s2[idx] = True
                sel = s2
            I = iu0[sel]; J = iu1[sel]
            csel = c[sel]
            p = len(I)
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            tmpc = np.empty((p, 2 * d + 1), dtype=int)
            ar = np.arange(d)
            tmpc[:, :d] = I[:, None] * d + ar
            tmpc[:, d:2 * d] = J[:, None] * d + ar
            tmpc[:, 2 * d] = n * d
            cols = tmpc.ravel()
            cc = np.empty((p, 2 * d + 1), dtype=float)
            cc[:, :d] = x[J]
            cc[:, d:2 * d] = x[I]
            cc[:, 2 * d] = -1.0
            vals = cc.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y = _norm(y)
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        except Exception:
            radius *= 0.5
            continue
    return x, best

def _hinge_lbfgs(x_init, tau, deadline, maxiter=250, Tsoft=0.0025, lam=1e-3):
    x_init = _norm(x_init.copy())
    n, d = x_init.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Np = len(iu0)
    y = x_init.ravel().copy()
    # best tracker via closure
    best_seen = [float('inf'), None]
    def fg(yflat):
        Y = yflat.reshape(n, d)
        rn = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        X = Y / rn
        g = (X @ X.T)[iu0, iu1]
        m = float(g.max())
        viol = g - tau
        pos = viol > 0.0
        hinge = float(np.sum(viol[pos] * viol[pos])) if np.any(pos) else 0.0
        # softmax term
        e = np.exp((g - m) / Tsoft)
        s = float(e.sum())
        if not np.isfinite(s) or s <= 0:
            s = 1e-300
            Lsoft = float(m)
            w = np.ones_like(g) / len(g)
        else:
            Lsoft = float(m + Tsoft * np.log(s / Np))
            w = e / s
        total = hinge + lam * Lsoft
        # gradients wrt X
        wh = np.zeros_like(g)
        if np.any(pos):
            wh[pos] = 2.0 * viol[pos]
        W = np.zeros((n, n))
        # combine: hinge weights + lam*softmax weights on pairs
        # build full weight vector
        wtot = wh + lam * w
        W[iu0, iu1] = wtot
        W = W + W.T
        GX = W @ X
        GY = (GX - (GX * X).sum(axis=1, keepdims=True) * X) / rn
        return float(total), GY.ravel()
    # time-limited: estimate iters from remaining time
    try:
        res = minimize(fg, y, method="L-BFGS-B", jac=True,
                       options={"maxiter": int(maxiter), "maxfun": int(maxiter * 5),
                                "ftol": 1e-14, "gtol": 1e-9})
        y = res.x
    except Exception:
        pass
    Yf = _norm(y.reshape(n, d))
    mc = _max_cos(Yf)
    G = Yf @ Yf.T
    np.fill_diagonal(G, -2.0)
    gg = G[iu0, iu1]
    vv = gg - tau
    vv[vv < 0] = 0.0
    hinge = float(np.sum(vv * vv))
    return Yf, mc, hinge

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    SEARCH_END = t0 + 148.0
    rng = np.random.RandomState(20261102)
    nT = 163
    try:
        bx = np.load("best.npy")
    except Exception:
        try:
            bx = np.load("record.npy")
        except Exception:
            bx = rng.randn(nT, 4)
    bx = np.asarray(bx, dtype=float)
    if bx.shape != (nT, 4) or not np.all(np.isfinite(bx)):
        bx = rng.randn(nT, 4)
    bx = _norm(bx)
    best_x = bx.copy()
    best_val = _max_cos(best_x)
    prev = best_x.copy()
    taus = [0.8559, 0.8555, 0.8550, 0.8545]
    # per-tau restarts budget
    for ti, tau in enumerate(taus):
        if time.time() > SEARCH_END - 6.0:
            break
        tau_best_x = None
        tau_best_mc = float('inf')
        tau_best_hinge = float('inf')
        n_rest = 4 if ti < 2 else 3
        for r in range(n_rest):
            if time.time() > SEARCH_END - 4.0:
                break
            if r == 0:
                base = _kick(prev, rng, frac=0.2, lo=0.03, hi=0.08)
            else:
                base = _kick(best_x, rng, frac=0.2, lo=0.03, hi=0.08)
            # hinge L-BFGS; scale maxiter by remaining time
            remain = SEARCH_END - time.time()
            if remain < 5.0:
                break
            mi = 220
            if remain < 30:
                mi = 120
            x_h, mc_h, hinge_h = _hinge_lbfgs(base, tau, SEARCH_END, maxiter=mi)
            if mc_h < tau_best_mc:
                tau_best_mc = mc_h
                tau_best_x = x_h.copy()
                tau_best_hinge = hinge_h
            # if hinge improved max beyond best, polish immediately
            if mc_h < best_val - 1e-12 or hinge_h == 0.0:
                # polish candidate
                try:
                    x_p, v_p = _lp_polish(x_h, min(SEARCH_END, time.time() + 7.0), r0=0.003)
                except Exception:
                    x_p, v_p = x_h, mc_h
                if v_p < best_val:
                    best_x = x_p.copy()
                    best_val = float(v_p)
                # feasibility reached?
                if v_p <= tau + 1e-9 or mc_h <= tau + 2e-12:
                    prev = best_x.copy()
                    tau_best_x = best_x.copy()
                    tau_best_mc = best_val
                    break
            else:
                # even without beating best, occasionally polish the best-of-tau to try topology repair
                if r == 0 and mc_h < best_val + 0.002:
                    try:
                        x_p, v_p = _lp_polish(x_h, min(SEARCH_END, time.time() + 4.0), r0=0.002)
                    except Exception:
                        x_p, v_p = x_h, mc_h
                    if v_p < best_val:
                        best_x = x_p.copy()
                        best_val = float(v_p)
        # continuation: next tau starts from best hinge result of this tau
        # (previous iterate plus kick), or from overall best if hinge made no progress
        if tau_best_x is not None:
            # prefer overall best if it is much better than hinge result
            if tau_best_mc < best_val + 0.001:
                prev = tau_best_x.copy()
            else:
                prev = best_x.copy()
        # early exit if feasible: continue to next lower tau immediately (already set prev)
        # if best reached below current tau, keep going down
        if best_val <= tau + 1e-9:
            continue
        # if time low break
    # final polish of global best with remaining time
    try:
        if time.time() < DEADLINE - 3.0:
            x_f, v_f = _lp_polish(best_x, DEADLINE - 1.0, r0=0.002)
            if v_f < best_val:
                best_x, best_val = x_f, v_f
    except Exception:
        pass
    best_x = _norm(best_x)
    return best_x
