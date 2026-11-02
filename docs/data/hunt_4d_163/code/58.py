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
    y = _norm(x.copy())
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
            tmpc[:, d:2*d] = J[:, None] * d + ar
            tmpc[:, 2*d] = n * d
            cols = tmpc.ravel()
            cc = np.empty((p, 2 * d + 1), dtype=float)
            cc[:, :d] = x[J]
            cc[:, d:2*d] = x[I]
            cc[:, 2*d] = -1.0
            vals = cc.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y = _norm(y)
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius*1.5, 0.05)
            else:
                radius *= 0.5
        except Exception:
            radius *= 0.5
            continue
    return x, best

def _hinge_lbfgs(x_init, tau, maxiter=250, Tsoft=0.0025, lam=1e-3):
    x_init = _norm(x_init.copy())
    n, d = x_init.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Np = len(iu0)
    y = x_init.ravel().copy()
    def fg(yflat):
        Y = yflat.reshape(n, d)
        rn = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        X = Y / rn
        G = X @ X.T
        g = G[iu0, iu1]
        m = float(g.max())
        viol = g - tau
        pos = viol > 0.0
        if np.any(pos):
            vp = viol[pos]
            hinge = float(np.sum(vp*vp))
        else:
            hinge = 0.0
        # softmax part
        e = np.exp((g - m) / Tsoft)
        s = float(e.sum())
        if not np.isfinite(s) or s <= 0:
            Lsoft = float(m)
            w = np.ones_like(g) / float(len(g))
        else:
            Lsoft = float(m + Tsoft*np.log(s / float(Np)))
            w = e / s
        total = hinge + lam * Lsoft
        if not np.isfinite(total):
            total = 1e6
            return total, np.zeros_like(yflat)
        wh = np.zeros_like(g)
        if np.any(pos):
            wh[pos] = 2.0 * viol[pos]
        wtot = wh + lam * w
        W = np.zeros((n, n))
        W[iu0, iu1] = wtot
        W = W + W.T
        GX = W @ X
        dot = (GX * X).sum(axis=1, keepdims=True)
        GY = (GX - dot * X) / rn
        if not np.all(np.isfinite(GY)):
            GY = np.nan_to_num(GY, nan=0.0, posinf=0.0, neginf=0.0)
        return float(total), GY.ravel()
    try:
        res = minimize(fg, y, method="L-BFGS-B", jac=True,
                       options={"maxiter": int(maxiter), "maxfun": int(maxiter*5),
                                "ftol": 1e-14, "gtol": 1e-9})
        y = res.x
    except Exception:
        pass
    Yf = _norm(y.reshape(n, d))
    mc = _max_cos(Yf)
    G2 = Yf @ Yf.T
    np.fill_diagonal(G2, -2.0)
    gg = G2[iu0, iu1]
    vv = gg - tau
    vv[vv < 0] = 0.0
    hinge = float(np.sum(vv*vv))
    return Yf, hinge, mc

def _smooth_lbfgs(x_init, deadline, taus=(0.003, 0.001), iters=(60, 60)):
    x = _norm(x_init.copy())
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Np = len(iu0)
    y = x.ravel().copy()
    for tau, maxiter in zip(taus, iters):
        if time.time() > deadline - 0.5:
            break
        def fg(yflat, _tau=tau):
            Y = yflat.reshape(n, d)
            rn = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
            X = Y / rn
            g = (X @ X.T)[iu0, iu1]
            m = float(g.max())
            e = np.exp((g - m) / _tau)
            s = float(e.sum())
            if s <= 0 or not np.isfinite(s):
                return float(m), np.zeros_like(yflat)
            L = m + _tau * float(np.log(s / Np))
            w = e / s
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            GY = (GX - (GX*X).sum(axis=1, keepdims=True)*X) / rn
            return float(L), GY.ravel()
        try:
            res = minimize(fg, y, method="L-BFGS-B", jac=True,
                           options={"maxiter": int(maxiter), "maxfun": int(maxiter*4),
                                    "ftol": 1e-14, "gtol": 1e-9})
            y = res.x
            Yt = y.reshape(n, d)
            Yt = _norm(Yt)
            y = Yt.ravel()
        except Exception:
            break
    Xf = _norm(y.reshape(n, d))
    return Xf

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    HINGE_END = t0 + 148.0
    rng = np.random.RandomState(1638559)
    n = 163
    best_x = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (n, 4) or not np.all(np.isfinite(a)):
                continue
            a = _norm(a)
            v = _max_cos(a)
            if v < best_val:
                best_val = v
                best_x = a.copy()
        except Exception:
            continue
    if best_x is None:
        best_x = rng.randn(n, 4)
        best_x = _norm(best_x)
        best_val = _max_cos(best_x)
    global_best_x = best_x.copy()
    global_best_val = float(best_val)
    file_best_x = global_best_x.copy()
    # feasibility continuation
    taus = [0.8559, 0.8555, 0.8550, 0.8545]
    cur_x = global_best_x.copy()
    cur_val = float(global_best_val)
    last_hinge = cur_x.copy()
    for ti, tau in enumerate(taus):
        if time.time() > HINGE_END - 4.0:
            break
        success = False
        # attempts per tau
        natt = 7 if ti == 0 else 6
        for att in range(natt):
            if time.time() > HINGE_END - 3.0:
                break
            if time.time() > DEADLINE - 25:
                break
            if att == 0:
                start = _kick(cur_x, rng, frac=0.2, lo=0.03, hi=0.08)
            elif att % 3 == 1:
                start = _kick(file_best_x, rng, frac=0.2, lo=0.03, hi=0.08)
            elif att % 3 == 2:
                start = _kick(global_best_x, rng, frac=0.2, lo=0.03, hi=0.08)
            else:
                start = _kick(last_hinge, rng, frac=0.2, lo=0.03, hi=0.08)
            # adapt maxiter to remaining time
            try:
                Xh, hinge, mc = _hinge_lbfgs(start, tau, maxiter=250, Tsoft=0.0025, lam=1e-3)
            except Exception:
                continue
            last_hinge = Xh.copy()
            # feasibility check
            if mc <= tau + 2e-9 or hinge <= 1e-14:
                # freeze: LP polish
                dl = min(time.time() + 6.0, DEADLINE - 8.0, HINGE_END + 4.0)
                if dl > time.time() + 0.5:
                    Xp, vp = _lp_polish(Xh, dl, r0=0.003)
                else:
                    Xp, vp = Xh, mc
                # also smooth polish
                if vp <= tau + 1e-7:
                    if vp < global_best_val:
                        global_best_x = Xp.copy()
                        global_best_val = float(vp)
                    cur_x = Xp.copy()
                    cur_val = float(vp)
                    success = True
                    break
                else:
                    if vp < global_best_val:
                        global_best_x = Xp.copy()
                        global_best_val = float(vp)
                    # treat as new cur if improved
                    if vp < cur_val:
                        cur_x = Xp.copy()
                        cur_val = float(vp)
            else:
                # not feasible: polish only if promising
                if mc < global_best_val - 1e-12:
                    dl = min(time.time() + 4.0, DEADLINE - 8.0)
                    if dl > time.time() + 0.5:
                        Xp, vp = _lp_polish(Xh, dl, r0=0.002)
                    else:
                        Xp, vp = Xh, mc
                    if vp < global_best_val:
                        global_best_x = Xp.copy()
                        global_best_val = float(vp)
                        cur_x = Xp.copy()
                        cur_val = float(vp)
                        # if improved a lot, consider tau achieved in max-cos sense?
                        if vp <= tau + 2e-9:
                            success = True
                            break
                    else:
                        # keep hinge result as last for kicks, update cur if hinge mc better than cur?
                        if mc < cur_val:
                            cur_x = Xh.copy()
                            cur_val = float(mc)
                else:
                    if mc < cur_val:
                        cur_x = Xh.copy()
                        cur_val = float(mc)
                    # occasional LP on hinge best to escape
                    if att == natt - 1 and mc < global_best_val + 0.0008:
                        dl = min(time.time() + 3.0, DEADLINE - 8.0)
                        if dl > time.time() + 0.5:
                            try:
                                Xp, vp = _lp_polish(Xh, dl, r0=0.002)
                                if vp < global_best_val:
                                    global_best_x = Xp.copy()
                                    global_best_val = float(vp)
                            except Exception:
                                pass
        if success:
            continue
        # if first tau failed, still step to next tau using current cur (which may have improved)
        # but lower taus are harder, so if cur_val > tau+0.0008, keep trying same difficulty?
        # continue to next tau anyway to follow schedule
    # final intensive polish of global best with remaining time
    try:
        if time.time() < DEADLINE - 6.0:
            Xs = _smooth_lbfgs(global_best_x, DEADLINE - 5.0, taus=(0.002, 0.0007), iters=(70, 70))
            vs = _max_cos(Xs)
            if vs < global_best_val:
                global_best_x = Xs.copy()
                global_best_val = float(vs)
    except Exception:
        pass
    try:
        if time.time() < DEADLINE - 4.0:
            Xp, vp = _lp_polish(global_best_x, DEADLINE - 1.0, r0=0.003)
            if vp < global_best_val:
                global_best_x = Xp.copy()
                global_best_val = float(vp)
    except Exception:
        pass
    global_best_x = _norm(global_best_x)
    # strict finite check + tiny shrink not needed (normalised)
    return global_best_x
