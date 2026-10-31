import numpy as np
import time

def solve():
    import numpy as np
    import time
    from scipy.optimize import minimize, linprog
    N = 26
    t0 = time.time()
    BUDGET = 124.0
    rng = np.random.RandomState(7)

    I, J = np.triu_indices(N, 1)
    M = len(I)
    A0 = np.zeros((M, N), dtype=float)
    A0[np.arange(M), I] = 1.0
    A0[np.arange(M), J] = 1.0
    c_primal = -np.ones(N, dtype=float)
    # dual: min d^T y + w^T z s.t. A0^T y + z >= 1, y,z>=0
    # linprog form: min c_dual^T t, -[A0^T I] t <= -1, t>=0
    A_dual_ub = -np.hstack([A0.T, np.eye(N, dtype=float)])  # (N, M+N)
    b_dual_ub = -np.ones(N, dtype=float)
    bounds_dual = [(0.0, None)] * (M + N)

    def dual_eval(xc, yc):
        # returns S, y_pair(M), z(N); S=0 on failure
        dx = xc[I] - xc[J]
        dy = yc[I] - yc[J]
        dvec = np.sqrt(dx * dx + dy * dy)
        # avoid exact zero (duplicate centres)
        dvec = np.maximum(dvec, 1e-12)
        w = np.minimum(np.minimum(xc, 1.0 - xc), np.minimum(yc, 1.0 - yc))
        # if centre outside, clip -> infeasible value
        if np.any(w < 0):
            return 0.0, np.zeros(M), np.zeros(N)
        w = np.maximum(w, 0.0)
        c_dual = np.concatenate([dvec, w])
        try:
            res = linprog(c_dual, A_ub=A_dual_ub, b_ub=b_dual_ub,
                          bounds=bounds_dual, method='highs')
        except Exception:
            return 0.0, np.zeros(M), np.zeros(N)
        if res.status != 0 or res.x is None:
            return 0.0, np.zeros(M), np.zeros(N)
        t = np.asarray(res.x, dtype=float)
        y = np.maximum(t[:M], 0.0)
        z = np.maximum(t[M:], 0.0)
        S = float(np.dot(dvec, y) + np.dot(w, z))
        return S, y, z

    def primal_eval(xc, yc):
        dx = xc[I] - xc[J]
        dy = yc[I] - yc[J]
        dvec = np.sqrt(dx * dx + dy * dy)
        dvec = np.maximum(dvec, 1e-12)
        w = np.minimum(np.minimum(xc, 1.0 - xc), np.minimum(yc, 1.0 - yc))
        if np.any(w < 0):
            return -1.0, None
        w = np.maximum(w, 0.0)
        bnds = [(0.0, float(wi)) for wi in w]
        try:
            res = linprog(c_primal, A_ub=A0, b_ub=dvec, bounds=bnds, method='highs')
        except Exception:
            return -1.0, None
        if res.status != 0 or res.x is None:
            return -1.0, None
        r = np.maximum(np.asarray(res.x, dtype=float), 0.0)
        return float(np.sum(r)), r

    def centre_fg(v):
        xc = v[:N]; yc = v[N:]
        S, y, z = dual_eval(xc, yc)
        f = -S
        # gradient of S
        gx = np.zeros(N); gy = np.zeros(N)
        # pair part
        dx = xc[I] - xc[J]
        dy = yc[I] - yc[J]
        dvec = np.sqrt(dx * dx + dy * dy)
        nz = dvec > 1e-11
        # only where y significant
        act = (y > 1e-12) & nz
        if np.any(act):
            Ia = I[act]; Ja = J[act]
            ya = y[act]
            da = dvec[act]
            uxa = dx[act] / da
            uya = dy[act] / da
            # accumulate
            np.add.at(gx, Ia, ya * uxa)
            np.add.at(gx, Ja, -ya * uxa)
            np.add.at(gy, Ia, ya * uya)
            np.add.at(gy, Ja, -ya * uya)
        # wall part: derivative of min(x,1-x,y,1-y)
        # choose argmin per i
        for i in range(N):
            zi = z[i]
            if zi < 1e-13:
                continue
            xi = xc[i]; yi = yc[i]
            cands = (xi, 1.0 - xi, yi, 1.0 - yi)
            k = int(np.argmin(cands))
            if k == 0:
                gx[i] += zi
            elif k == 1:
                gx[i] -= zi
            elif k == 2:
                gy[i] += zi
            else:
                gy[i] -= zi
        g = np.concatenate([-gx, -gy])
        return f, g

    def centre_f(v):
        xc = v[:N]; yc = v[N:]
        S, _, _ = dual_eval(xc, yc)
        return -S

    # ---- initial candidates ----
    cands = []
    # 5x5 + extra
    bx = []; by = []
    for ix in range(5):
        for jy in range(5):
            bx.append((ix + 0.5) / 5.0); by.append((jy + 0.5) / 5.0)
    bx.append(0.5); by.append(0.5)
    bx = np.array(bx); by = np.array(by)
    cands.append((bx.copy(), by.copy()))
    # 6x5 subset jitter base (use first 26 of 30 grid)
    gx = []; gy = []
    for ix in range(6):
        for jy in range(5):
            gx.append((ix + 0.5) / 6.0); gy.append((jy + 0.5) / 5.0)
    gx = np.array(gx); gy = np.array(gy)
    cands.append((gx[:N].copy(), gy[:N].copy()))
    # random uniform + jittered grids
    for _ in range(4):
        cands.append((rng.uniform(0.05, 0.95, N), rng.uniform(0.05, 0.95, N)))
    for _ in range(3):
        perm = rng.permutation(len(gx))[:N]
        cands.append((np.clip(gx[perm] + rng.normal(0, 0.02, N), 0.03, 0.97),
                      np.clip(gy[perm] + rng.normal(0, 0.02, N), 0.03, 0.97)))
    # jitter the 5x5 base
    for _ in range(2):
        cands.append((np.clip(bx + rng.normal(0, 0.025, N), 0.03, 0.97),
                      np.clip(by + rng.normal(0, 0.025, N), 0.03, 0.97)))

    bnds_c = [(0.015, 0.985)] * (2 * N)
    best_S = -1.0
    best_xc = None; best_yc = None

    # quick primal screen to order candidates
    scored = []
    for (qx, qy) in cands:
        s, _ = primal_eval(np.clip(qx, 0.01, 0.99), np.clip(qy, 0.01, 0.99))
        scored.append(s)
    order = np.argsort(scored)[::-1]

    n_polish = min(len(cands), 8)
    for oi in range(n_polish):
        if time.time() - t0 > 78.0:
            break
        idx = order[oi]
        qx, qy = cands[idx]
        v0 = np.concatenate([np.clip(qx, 0.015, 0.985), np.clip(qy, 0.015, 0.985)])
        try:
            res = minimize(centre_fg, v0, method='L-BFGS-B', jac=True,
                           bounds=bnds_c,
                           options={'maxiter': 90, 'ftol': 1e-10, 'gtol': 1e-6})
            v = res.x
        except Exception:
            continue
        xc = v[:N]; yc = v[N:]
        s, _ = primal_eval(xc, yc)
        if s > best_S:
            best_S = s
            best_xc = xc.copy(); best_yc = yc.copy()

    if best_xc is None:
        best_xc = np.clip(bx, 0.05, 0.95); best_yc = np.clip(by, 0.05, 0.95)
        best_S, _ = primal_eval(best_xc, best_yc)

    # small perturbations of best + re-polish (rotations/reflections equivalent, use noise)
    for sigma, it in [(0.008, 3), (0.003, 3)]:
        for _ in range(it):
            if time.time() - t0 > 92.0:
                break
            xn = np.clip(best_xc + rng.normal(0, sigma, N), 0.015, 0.985)
            yn = np.clip(best_yc + rng.normal(0, sigma, N), 0.015, 0.985)
            v0 = np.concatenate([xn, yn])
            try:
                res = minimize(centre_fg, v0, method='L-BFGS-B', jac=True,
                               bounds=bnds_c,
                               options={'maxiter': 60, 'ftol': 1e-11, 'gtol': 1e-7})
                v = res.x
            except Exception:
                continue
            xc = v[:N]; yc = v[N:]
            s, _ = primal_eval(xc, yc)
            if s > best_S + 1e-12:
                best_S = s
                best_xc = xc.copy(); best_yc = yc.copy()

    # primal radii for best centres
    s_best, r_best = primal_eval(best_xc, best_yc)
    if r_best is None:
        r_best = np.full(N, 0.05)

    # ---- SLSQP full 78-var polish with analytic Jacobians ----
    # variables v=[x(26),y(26),r(26)]
    def obj_fg(v):
        # maximize sum r -> minimize -sum
        f = -np.sum(v[2 * N:])
        g = np.zeros_like(v)
        g[2 * N:] = -1.0
        return f, g

    def wall_fun(v):
        x = v[:N]; y = v[N:2 * N]; r = v[2 * N:]
        return np.concatenate([x - r, 1.0 - x - r, y - r, 1.0 - y - r])

    def wall_jac(v):
        Jm = np.zeros((4 * N, 3 * N), dtype=float)
        for i in range(N):
            # x-r
            Jm[i, i] = 1.0; Jm[i, 2 * N + i] = -1.0
            # 1-x-r
            Jm[N + i, i] = -1.0; Jm[N + i, 2 * N + i] = -1.0
            # y-r
            Jm[2 * N + i, N + i] = 1.0; Jm[2 * N + i, 2 * N + i] = -1.0
            # 1-y-r
            Jm[3 * N + i, N + i] = -1.0; Jm[3 * N + i, 2 * N + i] = -1.0
        return Jm

    def pair_fun(v):
        x = v[:N]; y = v[N:2 * N]; r = v[2 * N:]
        dx = x[I] - x[J]; dy = y[I] - y[J]
        d = np.sqrt(dx * dx + dy * dy)
        return d - r[I] - r[J]

    def pair_jac(v):
        x = v[:N]; y = v[N:2 * N]
        dx = x[I] - x[J]; dy = y[I] - y[J]
        d = np.sqrt(dx * dx + dy * dy)
        d = np.maximum(d, 1e-14)
        ux = dx / d; uy = dy / d
        Jm = np.zeros((M, 3 * N), dtype=float)
        rows = np.arange(M)
        Jm[rows, I] = ux
        Jm[rows, J] = -ux
        Jm[rows, N + I] = uy
        Jm[rows, N + J] = -uy
        Jm[rows, 2 * N + I] = -1.0
        Jm[rows, 2 * N + J] = -1.0
        return Jm

    # feasible interior start: shrink radii a bit
    r0 = np.maximum(r_best, 1e-7) * 0.9995
    v0 = np.concatenate([best_xc, best_yc, r0])
    bnds78 = [(0.0, 1.0)] * (2 * N) + [(1e-9, 0.35)] * N
    cons = [{'type': 'ineq', 'fun': wall_fun, 'jac': wall_jac},
            {'type': 'ineq', 'fun': pair_fun, 'jac': pair_jac}]
    best_v = v0.copy()
    try:
        res = minimize(obj_fg, v0, method='SLSQP', jac=True,
                       bounds=bnds78, constraints=cons,
                       options={'ftol': 1e-12, 'maxiter': 400, 'disp': False, 'eps': 1e-12})
        if res.success or res.x is not None:
            cand_v = np.asarray(res.x, dtype=float)
            # verify improvement & feasibility roughly
            best_v = cand_v
    except Exception:
        pass
    # second SLSQP pass with tighter start if time
    if time.time() - t0 < BUDGET - 12.0:
        try:
            res2 = minimize(obj_fg, best_v, method='SLSQP', jac=True,
                            bounds=bnds78, constraints=cons,
                            options={'ftol': 1e-13, 'maxiter': 300, 'disp': False, 'eps': 1e-13})
            if res2.success or res2.x is not None:
                best_v = np.asarray(res2.x, dtype=float)
        except Exception:
            pass

    xb = np.clip(best_v[:N], 0.0, 1.0)
    yb = np.clip(best_v[N:2 * N], 0.0, 1.0)
    rb = np.maximum(best_v[2 * N:], 1e-12)

    # final uniform feasibility shrink
    wall = np.minimum(np.minimum(xb, 1.0 - xb), np.minimum(yb, 1.0 - yb))
    Dx = xb[:, None] - xb[None, :]
    Dy = yb[:, None] - yb[None, :]
    D = np.sqrt(Dx * Dx + Dy * Dy)
    with np.errstate(divide='ignore', invalid='ignore'):
        wr = wall / rb
        pr = D / (rb[:, None] + rb[None, :])
    np.fill_diagonal(pr, np.inf)
    a = float(np.min(wr))
    b = float(np.min(pr))
    alpha = min(1.0, a, b)
    if not np.isfinite(alpha) or alpha <= 0:
        # fallback to LP feasible point
        s_best, r_best = primal_eval(best_xc, best_yc)
        xb, yb, rb = best_xc, best_yc, np.maximum(r_best, 1e-9)
        wall = np.minimum(np.minimum(xb, 1.0 - xb), np.minimum(yb, 1.0 - yb))
        Dx = xb[:, None] - xb[None, :]
        Dy = yb[:, None] - yb[None, :]
        D = np.sqrt(Dx * Dx + Dy * Dy)
        with np.errstate(divide='ignore', invalid='ignore'):
            wr = wall / rb
            pr = D / (rb[:, None] + rb[None, :])
        np.fill_diagonal(pr, np.inf)
        a = float(np.min(wr)); b = float(np.min(pr))
        alpha = min(1.0, a, b)
    alpha = alpha * (1.0 - 1e-9) - 1e-12
    if not np.isfinite(alpha) or alpha <= 0:
        alpha = 1.0 - 1e-9
    rb = rb * alpha
    rb = np.maximum(rb, 1e-12)
    out = np.column_stack([xb, yb, rb])
    return out
