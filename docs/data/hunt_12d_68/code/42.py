import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _per_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _make_lse(n, d, IU, beta):
    def fg(yf):
        Y = yf.reshape(n, d)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = Y / nr
        g = (X @ X.T)[IU]
        m = float(g.max())
        e = np.exp(beta * (g - m))
        S = float(e.sum()) + 1e-300
        F = m + np.log(S) / beta
        w = e / S
        P = np.zeros((n, n))
        P[IU] = w
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return F, GY.ravel()
    return fg

def _make_hinge(n, d, IU, T):
    def fg(yf):
        Y = yf.reshape(n, d)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = Y / nr
        g = (X @ X.T)[IU]
        diff = g - T
        pos = diff > 0
        dp = np.where(pos, diff, 0.0)
        F = float(dp @ dp)
        w = 2.0 * dp
        P = np.zeros((n, n))
        P[IU] = w
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return F, GY.ravel()
    return fg

def _run_lbfgs(x0, fg, maxiter):
    y0 = (x0 / np.linalg.norm(x0, axis=1, keepdims=True)).ravel().copy()
    try:
        r = minimize(fg, y0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-7, 'maxfun': maxiter * 2})
        Y = r.x.reshape(x0.shape)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    except Exception:
        return x0 / np.linalg.norm(x0, axis=1, keepdims=True)

def _best_hole(Xr, rng, ncand=25000):
    d = Xr.shape[1]
    cand = rng.standard_normal((ncand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    s = (cand @ Xr.T).max(axis=1)
    return cand[int(np.argmin(s))]

def _lp_polish(x, deadline, r0=6e-4, cap_p=2000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() > deadline - 0.4:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        csel = c[sel]
        if p > cap_p:
            od = np.argsort(-csel)[:cap_p]
            I, J = I[od], J[od]
            csel = csel[od]
            p = cap_p
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best - 1e-15:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            improved = False
            for shrink in (0.3, 0.1):
                yy = x + res.x[:-1].reshape(n, d) * shrink
                yy /= np.linalg.norm(yy, axis=1, keepdims=True)
                vv = _max_cos(yy)
                if vv < best:
                    x, best = yy, vv
                    improved = True
                    break
            if not improved:
                radius *= 0.5
    return x

def solve():
    t0 = time.time()
    TLIM = 172.0
    deadline_final = t0 + TLIM
    n, d = 68, 12
    IU = np.triu_indices(n, 1)
    x_load = None
    bv = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                s = _max_cos(a)
                if s < bv:
                    bv = s
                    x_load = a.copy()
        except Exception:
            pass
    rng = np.random.default_rng(20261031)
    if x_load is None:
        x_load = rng.standard_normal((n, d))
        x_load /= np.linalg.norm(x_load, axis=1, keepdims=True)
        bv = _max_cos(x_load)
    x_best = x_load.copy()
    v_best = bv
    # quick diverse seeds to get a second basin anchor
    anchors = [x_best.copy()]
    anchor_vals = [v_best]
    for _ in range(3):
        if time.time() - t0 > 12:
            break
        xr = rng.standard_normal((n, d))
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        fg = _make_lse(n, d, IU, 300.0)
        xr = _run_lbfgs(xr, fg, 90)
        fg2 = _make_lse(n, d, IU, 1200.0)
        xr = _run_lbfgs(xr, fg2, 70)
        vv = _max_cos(xr)
        anchors.append(xr.copy()); anchor_vals.append(vv)
        if vv < v_best:
            x_best = xr.copy(); v_best = vv
    # basin-hopping with hinge targeting + topology kicks
    it = 0
    hop_deadline = t0 + 148.0
    while time.time() < hop_deadline:
        it += 1
        # choose source: mostly global best, sometimes 2nd anchor
        if it % 7 == 0 and len(anchors) > 1:
            src = anchors[int(rng.integers(0, len(anchors)))]
        else:
            src = x_best
        x = src.copy()
        mode = it % 4
        if mode == 0:
            # large replacement of k points
            k = int(rng.integers(3, 9))
            idx = rng.choice(n, size=k, replace=False)
            rep = rng.standard_normal((k, d))
            rep /= np.linalg.norm(rep, axis=1, keepdims=True)
            x[idx] = rep
            # small global jitter
            x += 0.015 * rng.standard_normal((n, d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif mode == 1:
            # medium Gaussian kick on subset
            k = int(rng.integers(4, 12))
            idx = rng.choice(n, size=k, replace=False)
            sig = float(rng.uniform(0.25, 0.55))
            x[idx] += sig * rng.standard_normal((k, d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif mode == 2:
            # worst-point hole reinsertion + jitter rest
            pm = _per_max(x)
            k = int(rng.integers(2, 6))
            worst = np.argsort(-pm)[:k]
            mask_keep = np.ones(n, dtype=bool)
            # sequential reinsertion
            for w in worst:
                mask = np.ones(n, dtype=bool); mask[w] = False
                # use current x for rest (with already replaced holes)
                Xr = x[mask]
                h = _best_hole(Xr, rng, ncand=22000)
                x[w] = h
            x += 0.01 * rng.standard_normal((n, d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        else:
            # global shake
            sig = float(rng.uniform(0.08, 0.20))
            x += sig * rng.standard_normal((n, d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        # targeted hinge phase: aim just below best
        lgu = float(rng.uniform(-4.0, -2.4))
        delta = 10.0 ** lgu  # 1e-4 .. 4e-3
        T = v_best - delta
        fg_h = _make_hinge(n, d, IU, float(T))
        xh = _run_lbfgs(x, fg_h, 70)
        vh = _max_cos(xh)
        # minimax re-polish (two sharpness steps)
        fg_a = _make_lse(n, d, IU, 700.0)
        xp = _run_lbfgs(xh, fg_a, 70)
        # adapt second sharpness randomly to diversify
        b2 = 2500.0 if (it % 2 == 0) else 6000.0
        fg_b = _make_lse(n, d, IU, float(b2))
        xp = _run_lbfgs(xp, fg_b, 60)
        vp = _max_cos(xp)
        # also try polishing the pre-hinge point if hinge hurt
        if vh > vp + 1e-6:
            pass
        if vp < v_best - 1e-12:
            x_best = xp.copy(); v_best = vp
            anchors.append(xp.copy()); anchor_vals.append(vp)
            if len(anchors) > 6:
                # keep best 6
                od = np.argsort(np.array(anchor_vals))[:6]
                anchors = [anchors[i] for i in od]
                anchor_vals = [anchor_vals[i] for i in od]
        elif vp < min(anchor_vals) + 0.008 and len(anchors) < 6 and (it % 3 == 0):
            # keep diverse decent anchor
            anchors.append(xp.copy()); anchor_vals.append(vp)
        # early exit if time low
        if time.time() > hop_deadline:
            break
    # final squeeze of best
    remain = deadline_final - time.time()
    if remain > 10:
        try:
            fg = _make_lse(n, d, IU, 3000.0)
            y = _run_lbfgs(x_best, fg, 110)
            if _max_cos(y) < v_best:
                x_best = y; v_best = _max_cos(y)
        except Exception:
            pass
    remain = deadline_final - time.time()
    if remain > 8:
        try:
            fg = _make_lse(n, d, IU, 9000.0)
            y = _run_lbfgs(x_best, fg, 90)
            if _max_cos(y) < v_best:
                x_best = y; v_best = _max_cos(y)
        except Exception:
            pass
    remain = deadline_final - time.time()
    if remain > 3:
        try:
            x_best = _lp_polish(x_best, deadline_final - 0.8, r0=4e-4)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
