import numpy as np
import time

def solve():
    n = 26
    t0 = time.time()
    DEADLINE = 124.0
    rng = np.random.default_rng(7)

    from scipy.optimize import minimize, linprog

    I, J = np.triu_indices(n, 1)
    m = len(I)
    A0 = np.zeros((m, n), dtype=float)
    A0[np.arange(m), I] = 1.0
    A0[np.arange(m), J] = 1.0
    cobj = -np.ones(n, dtype=float)
    EPS = 2e-7

    def lp_eval(xc, yc):
        wall = np.minimum(np.minimum(xc, 1.0 - xc), np.minimum(yc, 1.0 - yc))
        if np.any(wall < 1e-9):
            return -1.0, None
        dx = xc[I] - xc[J]
        dy = yc[I] - yc[J]
        d = np.sqrt(dx * dx + dy * dy) - EPS
        # quick reject: if any d < 1e-9 -> centres coincide
        if np.any(d < 1e-9):
            return -1.0, None
        bnds = [(0.0, float(w - EPS)) for w in wall]
        # all upper bounds must be positive
        for _, ub in bnds:
            if ub <= 0:
                return -1.0, None
        try:
            res = linprog(cobj, A_ub=A0, b_ub=d, bounds=bnds, method='highs')
        except Exception:
            return -1.0, None
        if res.status != 0:
            return -1.0, None
        r = np.asarray(res.x, dtype=float)
        r = np.maximum(r, 0.0)
        s = float(np.sum(r))
        return s, r

    # fallback
    k = int(np.ceil(np.sqrt(n)))
    r0 = 0.5 / k
    fb = []
    for ix in range(k):
        for jy in range(k):
            if len(fb) < n:
                fb.append([(2 * ix + 1) * r0, (2 * jy + 1) * r0])
    fb = np.array(fb, dtype=float)
    best_sum = -1.0
    best_x = fb[:, 0].copy()
    best_y = fb[:, 1].copy()
    best_r = np.full(n, r0 * 0.9)
    s0, r0v = lp_eval(best_x, best_y)
    if s0 > 0:
        best_sum = s0
        best_r = r0v.copy()

    pool = []
    if best_sum > 0:
        pool.append((best_sum, best_x.copy(), best_y.copy(), best_r.copy()))

    # ---------- Phase A: diverse penalized L-BFGS ----------
    def penalty_fun(v, W):
        X = v[0:n]
        Y = v[n:2 * n]
        R = v[2 * n:3 * n]
        gx = np.zeros(n)
        gy = np.zeros(n)
        gr_pen = np.zeros(n)
        pen = 0.0
        v1 = R - X
        m1 = v1 > 0
        if np.any(m1):
            pen += np.sum(v1[m1] ** 2)
            gx[m1] += -2.0 * v1[m1]
            gr_pen[m1] += 2.0 * v1[m1]
        v2 = X + R - 1.0
        m2 = v2 > 0
        if np.any(m2):
            pen += np.sum(v2[m2] ** 2)
            gx[m2] += 2.0 * v2[m2]
            gr_pen[m2] += 2.0 * v2[m2]
        v3 = R - Y
        m3 = v3 > 0
        if np.any(m3):
            pen += np.sum(v3[m3] ** 2)
            gy[m3] += -2.0 * v3[m3]
            gr_pen[m3] += 2.0 * v3[m3]
        v4 = Y + R - 1.0
        m4 = v4 > 0
        if np.any(m4):
            pen += np.sum(v4[m4] ** 2)
            gy[m4] += 2.0 * v4[m4]
            gr_pen[m4] += 2.0 * v4[m4]
        Dx = X[:, None] - X[None, :]
        Dy = Y[:, None] - Y[None, :]
        D = np.sqrt(Dx * Dx + Dy * Dy + 1e-20)
        S = R[:, None] + R[None, :]
        V = S - D
        np.fill_diagonal(V, 0.0)
        np.fill_diagonal(D, 1.0)
        Mask = V > 0
        if np.any(Mask):
            pen += 0.5 * np.sum(V[Mask] ** 2)
            Q = np.zeros((n, n))
            Q[Mask] = -2.0 * V[Mask] / D[Mask]
            gx += np.sum(Q * Dx, axis=1)
            gy += np.sum(Q * Dy, axis=1)
            G = np.zeros((n, n))
            G[Mask] = 2.0 * V[Mask]
            gr_pen += np.sum(G, axis=1)
        loss = -np.sum(R) + W * pen
        g = np.empty_like(v)
        g[0:n] = W * gx
        g[n:2 * n] = W * gy
        g[2 * n:3 * n] = -1.0 + W * gr_pen
        return loss, g

    def make_init(rid):
        if rid == 0:
            xs = []; ys = []
            for ix in range(5):
                for jy in range(5):
                    xs.append((ix + 0.5) / 5.0); ys.append((jy + 0.5) / 5.0)
            xs.append(0.82); ys.append(0.18)
            xc = np.array(xs, float) + rng.normal(0, 0.015, n)
            yc = np.array(ys, float) + rng.normal(0, 0.015, n)
            R0 = 0.09 + 0.02 * rng.random(n)
        elif rid == 1:
            # 6-row staggered template (breaks 5x5)
            counts = [4, 5, 4, 5, 4, 4]  # sum 26
            xs = []; ys = []
            ny = len(counts)
            for r_, cnt in enumerate(counts):
                y = (r_ + 0.5) / ny
                for cc in range(cnt):
                    x = (cc + 0.5 + (0.5 if (r_ % 2 == 1) else 0.0) * 0.5) / (cnt + 0.3)
                    xs.append(x); ys.append(y)
            xc = np.array(xs[:n], float) + rng.normal(0, 0.01, n)
            yc = np.array(ys[:n], float) + rng.normal(0, 0.01, n)
            R0 = 0.08 + 0.03 * rng.random(n)
        elif rid == 2:
            # 4-row template wide circles
            counts = [6, 7, 6, 7]
            xs = []; ys = []
            ny = len(counts)
            for r_, cnt in enumerate(counts):
                y = (r_ + 0.5) / ny
                for cc in range(cnt):
                    x = (cc + 0.5) / (cnt + 0.2) + (0.02 if (r_ % 2) else 0.0)
                    xs.append(x); ys.append(y)
            # take 26
            xs = np.array(xs); ys = np.array(ys)
            perm = rng.permutation(len(xs))[:n]
            xc = xs[perm] + rng.normal(0, 0.012, n)
            yc = ys[perm] + rng.normal(0, 0.012, n)
            R0 = 0.085 + 0.03 * rng.random(n)
        elif rid == 3:
            # hex jitter
            pts = []
            for ix in range(6):
                for jy in range(5):
                    xx = (ix + 0.5) / 6.0 + (0.04 if (jy % 2) else 0.0)
                    yy = (jy + 0.5) / 5.0
                    if xx < 0.98:
                        pts.append((xx, yy))
            pts = np.array(pts)
            perm = rng.permutation(len(pts))[:n]
            xc = pts[perm, 0] + rng.normal(0, 0.01, n)
            yc = pts[perm, 1] + rng.normal(0, 0.01, n)
            R0 = 0.07 + 0.04 * rng.random(n)
        else:
            xc = rng.uniform(0.04, 0.96, n)
            yc = rng.uniform(0.04, 0.96, n)
            R0 = 0.06 + 0.05 * rng.random(n)
        xc = np.clip(xc, 0.03, 0.97)
        yc = np.clip(yc, 0.03, 0.97)
        R0 = np.clip(R0, 0.03, 0.19)
        return xc, yc, R0

    Ws = [2e2, 2e3, 2e4, 2e5]
    bounds78 = [(0.0, 1.0)] * (2 * n) + [(0.02, 0.20)] * n
    T_PEN = 52.0
    restart = 0
    while time.time() - t0 < T_PEN and restart < 40:
        xc0, yc0, R00 = make_init(restart)
        v = np.concatenate([xc0, yc0, R00])
        for W in Ws:
            if time.time() - t0 > T_PEN:
                break
            try:
                res = minimize(penalty_fun, v, args=(W,), jac=True, method='L-BFGS-B',
                               bounds=bounds78, options={'maxiter': 120, 'ftol': 1e-9, 'gtol': 1e-6})
                v = res.x
            except Exception:
                break
        X = np.clip(v[0:n], 0.005, 0.995)
        Y = np.clip(v[n:2 * n], 0.005, 0.995)
        s, r = lp_eval(X, Y)
        if s > best_sum:
            best_sum = s
            best_x = X.copy(); best_y = Y.copy(); best_r = r.copy()
            pool.append((s, X.copy(), Y.copy(), r.copy()))
            # keep pool small sorted
            pool = sorted(pool, key=lambda t: -t[0])[:5]
        elif s > 0 and len(pool) < 5:
            pool.append((s, X.copy(), Y.copy(), r.copy()))
        restart += 1

    # ---------- Phase B: LP-based simulated annealing with topology moves ----------
    # start from best
    cur_x = best_x.copy(); cur_y = best_y.copy()
    cur_s = best_sum
    # temperature schedule
    T0 = 0.004
    T1 = 0.00015
    # estimate total hill iters by time left
    t_hill_end = t0 + DEADLINE - 12.0
    it = 0
    # precompute for speed: use local rng
    while time.time() < t_hill_end:
        frac = (time.time() - t0) / max(1e-9, (t_hill_end - t0))
        frac = min(1.0, max(0.0, frac))
        Temp = T0 * (T1 / T0) ** frac
        sig_g = 0.018 * (1 - frac) + 0.002 * frac
        sig_s = 0.06 * (1 - frac) + 0.008 * frac
        # occasionally restart from best pool member with kick
        if it % 350 == 349:
            # pick from pool or best with big kick to escape
            if len(pool) > 1 and rng.random() < 0.4:
                idx = rng.integers(len(pool))
                cur_x = pool[idx][1].copy() + rng.normal(0, 0.01, n)
                cur_y = pool[idx][2].copy() + rng.normal(0, 0.01, n)
                cur_x = np.clip(cur_x, 0.02, 0.98)
                cur_y = np.clip(cur_y, 0.02, 0.98)
                cur_s, _ = lp_eval(cur_x, cur_y)
                if cur_s < 0:
                    cur_x = best_x.copy(); cur_y = best_y.copy(); cur_s = best_sum
            else:
                # big kick of current best
                if rng.random() < 0.5:
                    cur_x = best_x.copy() + rng.normal(0, 0.03, n)
                    cur_y = best_y.copy() + rng.normal(0, 0.03, n)
                    cur_x = np.clip(cur_x, 0.02, 0.98)
                    cur_y = np.clip(cur_y, 0.02, 0.98)
                    cur_s, _ = lp_eval(cur_x, cur_y)
                    if cur_s < 0:
                        cur_x = best_x.copy(); cur_y = best_y.copy(); cur_s = best_sum
            it += 1
            continue
        u = rng.random()
        nx = cur_x.copy(); ny = cur_y.copy()
        if u < 0.45:
            # global small jitter
            nx = nx + rng.normal(0, sig_g, n)
            ny = ny + rng.normal(0, sig_g, n)
            nx = np.clip(nx, 0.015, 0.985)
            ny = np.clip(ny, 0.015, 0.985)
        elif u < 0.75:
            # single circle medium move
            kk = rng.integers(n)
            nx[kk] += rng.normal(0, sig_s)
            ny[kk] += rng.normal(0, sig_s)
            nx[kk] = np.clip(nx[kk], 0.015, 0.985)
            ny[kk] = np.clip(ny[kk], 0.015, 0.985)
        elif u < 0.90:
            # single teleport to uniform (void seeking)
            kk = rng.integers(n)
            nx[kk] = rng.uniform(0.03, 0.97)
            ny[kk] = rng.uniform(0.03, 0.97)
        else:
            # two-circle joint move / swap neighbourhood: move two at once
            k1 = rng.integers(n); k2 = rng.integers(n)
            nx[k1] += rng.normal(0, sig_s); ny[k1] += rng.normal(0, sig_s)
            nx[k2] += rng.normal(0, sig_s); ny[k2] += rng.normal(0, sig_s)
            nx = np.clip(nx, 0.015, 0.985); ny = np.clip(ny, 0.015, 0.985)
        s_new, r_new = lp_eval(nx, ny)
        if s_new < 0:
            it += 1
            continue
        if s_new >= cur_s or rng.random() < np.exp((s_new - cur_s) / max(Temp, 1e-12)):
            cur_x = nx; cur_y = ny; cur_s = s_new
            if s_new > best_sum:
                best_sum = s_new
                best_x = nx.copy(); best_y = ny.copy(); best_r = r_new.copy()
                pool.append((s_new, nx.copy(), ny.copy(), r_new.copy()))
                pool = sorted(pool, key=lambda t: -t[0])[:5]
        it += 1

    # ---------- Phase C: fine joint polish from best ----------
    try:
        v = np.concatenate([best_x, best_y, np.clip(best_r, 0.02, 0.19)])
        for W in [5e3, 5e4, 5e5]:
            if time.time() - t0 > DEADLINE - 4:
                break
            try:
                res = minimize(penalty_fun, v, args=(W,), jac=True, method='L-BFGS-B',
                               bounds=bounds78, options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
                v = res.x
            except Exception:
                break
        Xp = np.clip(v[0:n], 0.005, 0.995)
        Yp = np.clip(v[n:2 * n], 0.005, 0.995)
        sp, rp = lp_eval(Xp, Yp)
        if sp > best_sum:
            best_sum = sp; best_x = Xp; best_y = Yp; best_r = rp
    except Exception:
        pass

    # final safety: re-solve LP once more and shrink multiplicatively
    try:
        sfin, rfin = lp_eval(best_x, best_y)
        if sfin > 0:
            best_r = rfin
            best_sum = sfin
    except Exception:
        pass
    best_r = np.maximum(np.asarray(best_r, float), 1e-9) * (1.0 - 1e-7)
    # ensure strictly inside and positive
    best_x = np.clip(np.asarray(best_x, float), 1e-6, 1.0 - 1e-6)
    best_y = np.clip(np.asarray(best_y, float), 1e-6, 1.0 - 1e-6)
    # final tiny guard: if any wall violation due to clip, shrink that radius
    for i in range(n):
        w = min(best_x[i], 1.0 - best_x[i], best_y[i], 1.0 - best_y[i])
        if best_r[i] > w:
            best_r[i] = w * (1.0 - 1e-9)
        if best_r[i] <= 0:
            best_r[i] = 1e-6
    # pairwise guard: scale down only violating circles minimally via iterative shrink
    # use uniform factor if needed (rare because EPS margin)
    for _ in range(3):
        Dx = best_x[:, None] - best_x[None, :]
        Dy = best_y[:, None] - best_y[None, :]
        D = np.sqrt(Dx * Dx + Dy * Dy)
        S = best_r[:, None] + best_r[None, :]
        np.fill_diagonal(D, np.inf)
        np.fill_diagonal(S, 0.0)
        viol = S - D
        mv = np.max(viol)
        if mv < 1e-12:
            break
        best_r *= (1.0 - 1e-6)
    out = np.column_stack([best_x, best_y, best_r])
    return out
