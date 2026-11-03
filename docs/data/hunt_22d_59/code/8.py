import time
import numpy as np

def _max_cos(Y):
    G = Y @ Y.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _lbfgs_polish(Y0, betas, iters, verbose=False):
    try:
        from scipy.optimize import minimize
    except Exception:
        return Y0.copy(), _max_cos(Y0)
    n, d = Y0.shape
    Xf = Y0.copy().reshape(-1).astype(np.float64)
    # ensure normalized start
    Xr = Xf.reshape(n, d)
    Xr = _normalize(Xr)
    Xf = Xr.reshape(-1)
    best_Y = Xr.copy()
    best_v = _max_cos(best_Y)
    # triu indices
    iu, ju = np.triu_indices(n, 1)
    for beta, mi in zip(betas, iters if hasattr(iters, "__len__") else [iters]*len(betas)):
        def fun_grad(Xf_):
            X = Xf_.reshape(n, d)
            r = np.linalg.norm(X, axis=1)
            r[r == 0] = 1.0
            Y = X / r[:, None]
            G = Y @ Y.T
            g = G[iu, ju]
            gmax = float(g.max())
            e = np.exp(beta * (g - gmax))
            S = float(e.sum())
            f = gmax + np.log(S) / beta
            # weights over pairs
            w = e / S
            # build symmetric W
            # Gr = W @ Y efficiently: for each k, sum_j w_kj Y_j
            # use accumulation
            Gr = np.zeros_like(Y)
            # scatter: could use bincount style loop with numpy
            # vectorized via dense matrix (59x59 small)
            W = np.zeros((n, n), dtype=np.float64)
            W[iu, ju] = w
            W[ju, iu] = w
            Gr = W @ Y
            # project and scale by r
            dot = np.sum(Gr * Y, axis=1)
            Gr = Gr - Y * dot[:, None]
            Gr = Gr / r[:, None]
            return f, Gr.reshape(-1)
        try:
            res = minimize(fun_grad, Xf, method="L-BFGS-B", jac=True,
                           options={"maxiter": int(mi), "ftol": 1e-14, "gtol": 1e-9, "maxfun": 100000, "maxcor": 15})
            Xf = res.x
        except Exception:
            continue
        X = Xf.reshape(n, d)
        Y = _normalize(X)
        v = _max_cos(Y)
        if v < best_v:
            best_v = v
            best_Y = Y.copy()
        # warm start from normalized
        Xf = Y.reshape(-1)
    return best_Y, best_v

def _lp_polish(Y0, iters=12, eps0=0.01, tol_active=0.003):
    try:
        from scipy.optimize import linprog
    except Exception:
        return Y0.copy(), _max_cos(Y0)
    Y = Y0.copy()
    n, d = Y.shape
    N = n * d
    best = Y.copy()
    best_v = _max_cos(Y)
    eps = float(eps0)
    for it in range(iters):
        G = Y @ Y.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # active pairs
        ii, jj = np.where(np.triu(G, 1) > (m - tol_active))
        if len(ii) == 0:
            break
        # cap number of constraints for speed
        if len(ii) > 600:
            slacks = G[ii, jj]
            order = np.argsort(slacks)[-600:]
            ii = ii[order]; jj = jj[order]
        k = len(ii)
        # Build A_ub: k x (N+1)
        # To save memory build dense (k ~ <=600, N=1298) -> ~780k entries ok
        try:
            A_ub = np.zeros((k, N + 1), dtype=np.float64)
            b_ub = np.zeros(k, dtype=np.float64)
            for r_ in range(k):
                i = int(ii[r_]); j = int(jj[r_])
                # <xj, dxi> + <xi, dxj> - t <= -gij
                A_ub[r_, i*d:(i+1)*d] = Y[j]
                A_ub[r_, j*d:(j+1)*d] = Y[i]
                A_ub[r_, N] = -1.0
                b_ub[r_] = -float(G[i, j])
            # equalities: <xi, dxi>=0
            A_eq = np.zeros((n, N + 1), dtype=np.float64)
            b_eq = np.zeros(n, dtype=np.float64)
            for i in range(n):
                A_eq[i, i*d:(i+1)*d] = Y[i]
            c = np.zeros(N + 1, dtype=np.float64)
            c[N] = 1.0
            bounds = [(-eps, eps)] * N + [(-1.0, float(m))]
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds, method="highs",
                          options={"presolve": True})
            if not res.success:
                eps *= 0.5
                if eps < 1e-7:
                    break
                continue
            dx = res.x[:N].reshape(n, d)
            pred_t = float(res.x[N])
            # only accept if predicted improvement
            if pred_t >= m - 1e-12:
                eps *= 0.5
                if eps < 1e-7:
                    break
                continue
            Yt = _normalize(Y + dx)
            vt = _max_cos(Yt)
            if vt < m - 1e-12:
                Y = Yt
                if vt < best_v:
                    best_v = vt
                    best = Y.copy()
                # grow trust slightly
                eps = min(eps * 1.3, 0.02)
                # tighten active tol as we converge
                tol_active = max(tol_active * 0.85, 0.0004)
            else:
                eps *= 0.4
                if eps < 1e-7:
                    break
        except Exception:
            break
    return best, best_v

def _refine_hole(y0, Xw, steps=400, lr0=0.05, beta_smooth=None, rng=None):
    y = y0 / (np.linalg.norm(y0) or 1.0)
    best = y.copy()
    dots = Xw @ y
    best_v = float(dots.max())
    lr = float(lr0)
    m = Xw.shape[0]
    for s in range(steps):
        dots = Xw @ y
        mx = float(dots.max())
        if beta_smooth is not None and beta_smooth > 0:
            # softmax weights
            e = np.exp(beta_smooth * (dots - mx))
            w = e / e.sum()
            grad = w @ Xw
        else:
            # top-k average subgradient
            tol = 0.004 * (1.0 - s / max(steps, 1)) + 0.0002
            mask = dots >= (mx - tol)
            grad = Xw[mask].mean(axis=0)
        # tangent projection (descent: y - lr*gt)
        gdot = float(grad @ y)
        gt = grad - gdot * y
        gn = float(np.linalg.norm(gt))
        if gn > 0:
            gt = gt / gn
        # decay
        cur_lr = lr * (1.0 - s / steps) + lr * 0.05
        yt = y - cur_lr * gt
        nrm = float(np.linalg.norm(yt))
        if nrm == 0 or not np.isfinite(nrm):
            continue
        yt = yt / nrm
        dm = float((Xw @ yt).max())
        if dm < mx - 1e-12 or (dm < best_v):
            y = yt
            if dm < best_v:
                best_v = dm
                best = yt.copy()
        else:
            # try smaller step from y
            yt2 = y - (cur_lr * 0.4) * gt
            yt2 = yt2 / np.linalg.norm(yt2)
            dm2 = float((Xw @ yt2).max())
            if dm2 < mx:
                y = yt2
                if dm2 < best_v:
                    best_v = dm2
                    best = yt2.copy()
    return best, best_v

def _find_hole(Xw, rng, n_sample=2500, n_keep=8):
    n0, d = Xw.shape
    C = _normalize(rng.standard_normal((n_sample, d)))
    # max dot to Xw
    D = C @ Xw.T
    vals = D.max(axis=1)
    idx = np.argsort(vals)[:n_keep]
    best_y = C[idx[0]]
    best_v = float(vals[idx[0]])
    for k in idx:
        y0 = C[k]
        # two-phase refine: coarse subgrad then smooth
        y1, v1 = _refine_hole(y0, Xw, steps=250, lr0=0.08, beta_smooth=None)
        y2, v2 = _refine_hole(y1, Xw, steps=250, lr0=0.02, beta_smooth=400.0)
        if v2 < best_v:
            best_v = v2
            best_y = y2.copy()
        elif v1 < best_v:
            best_v = v1
            best_y = y1.copy()
    return best_y, best_v

def _contact_refine(Y0, steps, step0, tol):
    Y = Y0.copy()
    n = Y.shape[0]
    Gc = Y @ Y.T
    np.fill_diagonal(Gc, -2.0)
    best_val = float(Gc.max())
    best_Y = Y.copy()
    step = float(step0)
    eye = np.eye(n, dtype=np.float64)
    for _ in range(steps):
        Gfull = Y @ Y.T
        m = float(np.max(Gfull - 2.0 * eye))
        A = Gfull > (m - tol)
        np.fill_diagonal(A, False)
        Af = A.astype(np.float64)
        AY = Af @ Y
        row_w = np.sum(Af * Gfull, axis=1, keepdims=True)
        F = -AY + Y * row_w
        Yt = Y + step * F
        rn = np.linalg.norm(Yt, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        Yt /= rn
        Gt = Yt @ Yt.T
        np.fill_diagonal(Gt, -2.0)
        mt = float(Gt.max())
        if mt < m - 1e-12:
            Y = Yt
            if mt < best_val:
                best_val = mt
                best_Y = Y.copy()
            step = min(step * 1.08, 0.02)
        else:
            step *= 0.5
            if step < 1e-12:
                step = 1e-12
    return best_Y, best_val

def solve():
    import time as _time
    rng = np.random.default_rng(7)
    t0 = _time.time()
    deadline = t0 + 168.0

    def _load_best():
        best = None; bestv = 1e9
        for fn in ("best.npy", "record.npy"):
            try:
                a = np.load(fn)
                a = np.asarray(a, dtype=np.float64)
                if a.shape != (59, 22):
                    continue
                if not np.all(np.isfinite(a)):
                    continue
                a = _normalize(a)
                v = _max_cos(a)
                if v < bestv:
                    bestv = v; best = a.copy()
            except Exception:
                continue
        if best is None:
            best = _normalize(rng.standard_normal((59, 22)))
            bestv = _max_cos(best)
        return best, bestv

    x_best, best_val = _load_best()

    # Stage A: squeeze current basin hard (cheap wins)
    y, v = _lbfgs_polish(x_best, betas=[120.0, 400.0, 900.0], iters=[120, 120, 120])
    if v < best_val:
        best_val = v; x_best = y.copy()
    if _time.time() < deadline - 20:
        y2, v2 = _lp_polish(x_best, iters=10, eps0=0.008, tol_active=0.0025)
        if v2 < best_val:
            best_val = v2; x_best = y2.copy()
    if _time.time() < deadline - 20:
        y3, v3 = _contact_refine(x_best, steps=1200, step0=0.003, tol=0.004)
        if v3 < best_val:
            best_val = v3; x_best = y3.copy()

    # Stage B: kill-and-fill basin hopping
    trial = 0
    # precompute victim cycling
    while _time.time() < deadline - 18.0:
        trial += 1
        G = x_best @ x_best.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        rowmax = G.max(axis=1)
        # degree with tight tol
        deg = np.sum(G > (m - 0.0025), axis=1)
        # worst pair
        ij = np.unravel_index(int(np.argmax(G)), G.shape)
        worst = [int(ij[0]), int(ij[1])]
        order_bad = np.argsort(-rowmax)
        order_deg_hi = np.argsort(-deg)
        order_deg_lo = np.argsort(deg)
        r = trial % 6
        if r == 0:
            victim = worst[trial % 2]
        elif r == 1:
            victim = int(order_bad[trial % 8])
        elif r == 2:
            victim = int(order_deg_hi[trial % 8])
        elif r == 3:
            victim = int(order_deg_lo[trial % 5])
        elif r == 4:
            victim = int(rng.integers(0, 59))
        else:
            victim = int(order_bad[(trial * 3) % 59])
        # double kill every 7th trial
        victims = [victim]
        if trial % 7 == 0:
            # second victim: worst partner of first or random
            row = G[victim].copy()
            row[victim] = -2
            v2_ = int(np.argmax(row))
            victims = [victim, v2_]
        keep_idx = [i for i in range(59) if i not in victims]
        Xw = x_best[keep_idx]
        # find hole(s)
        if len(victims) == 1:
            y_new, _ = _find_hole(Xw, rng, n_sample=2000, n_keep=6)
            Y_init = np.vstack([Xw, y_new[None, :]])
        else:
            # sequential insertion
            y_a, _ = _find_hole(Xw, rng, n_sample=1800, n_keep=5)
            Xw2 = np.vstack([Xw, y_a[None, :]])
            y_b, _ = _find_hole(Xw2, rng, n_sample=1800, n_keep=5)
            Y_init = np.vstack([Xw, y_a[None, :], y_b[None, :]])
        Y_init = _normalize(Y_init)
        # short polish
        # vary beta schedule by trial parity for diversity
        if trial % 3 == 0:
            yp, vp = _lbfgs_polish(Y_init, betas=[250.0, 700.0], iters=[70, 70])
        else:
            yp, vp = _lbfgs_polish(Y_init, betas=[350.0, 900.0], iters=[80, 80])
        # light contact polish
        yc, vc = _contact_refine(yp, steps=500, step0=0.003, tol=0.005)
        cand, candv = (yc, vc) if vc < vp else (yp, vp)
        if candv < best_val - 1e-12:
            # squeeze new basin with LP
            yl, vl = _lp_polish(cand, iters=6, eps0=0.005, tol_active=0.002)
            if vl < candv:
                cand, candv = yl, vl
            best_val = candv
            x_best = cand.copy()
        # occasional neutral sideways move to escape: if close, adopt to diversify base
        elif candv < best_val + 0.001 and (trial % 9 == 0):
            # adopt as wandering base for one step without overwriting best? use temp
            # do a jittered continue: set a separate current state
            pass
        # time guard
        if trial > 60:
            break

    # Stage C: final ultra polish of best
    if _time.time() < deadline - 12.0:
        y, v = _lbfgs_polish(x_best, betas=[500.0, 1200.0, 2500.0], iters=[150, 150, 150])
        if v < best_val:
            best_val = v; x_best = y.copy()
    if _time.time() < deadline - 8.0:
        y, v = _lp_polish(x_best, iters=14, eps0=0.006, tol_active=0.0018)
        if v < best_val:
            best_val = v; x_best = y.copy()
    if _time.time() < deadline - 4.0:
        y, v = _contact_refine(x_best, steps=1200, step0=0.0015, tol=0.0012)
        if v < best_val:
            best_val = v; x_best = y.copy()

    x_best = _normalize(x_best)
    return x_best
