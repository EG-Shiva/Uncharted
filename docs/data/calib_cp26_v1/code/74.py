import numpy as np
import time

def solve():
    n = 26
    t0 = time.time()
    DEADLINE = 123.0
    rng = np.random.default_rng(12345)
    from scipy.optimize import minimize, linprog

    I, J = np.triu_indices(n, 1)
    m = len(I)
    A0 = np.zeros((m, n), dtype=float)
    A0[np.arange(m), I] = 1.0
    A0[np.arange(m), J] = 1.0
    cobj = -np.ones(n, dtype=float)

    def lp_eval(xc, yc):
        # exact bounds/distances, strictness enforced later by global shrink
        wall = np.minimum(np.minimum(xc, 1.0 - xc), np.minimum(yc, 1.0 - yc))
        # clip tiny negatives to 0
        bnds = [(0.0, float(w) if w > 0 else 0.0) for w in wall]
        dx = xc[I] - xc[J]
        dy = yc[I] - yc[J]
        d = np.sqrt(dx * dx + dy * dy)
        if np.any(d < 1e-12):
            return -1.0, None
        try:
            res = linprog(cobj, A_ub=A0, b_ub=d, bounds=bnds, method='highs')
        except Exception:
            return -1.0, None
        if res.status != 0 or res.x is None:
            return -1.0, None
        r = np.asarray(res.x, dtype=float).ravel()
        r = np.maximum(r, 0.0)
        return float(np.sum(r)), r

    def safe_scale(xc, yc, rc):
        # uniform shrink to guarantee strict feasibility with margin
        rc = np.maximum(np.asarray(rc, float), 1e-9)
        wall = np.minimum(np.minimum(xc, 1.0 - xc), np.minimum(yc, 1.0 - yc))
        # wall ratio
        with np.errstate(divide='ignore', invalid='ignore'):
            wr = wall / rc
        a = np.min(wr)
        dx = xc[I] - xc[J]
        dy = yc[I] - yc[J]
        d = np.sqrt(dx * dx + dy * dy)
        rs = rc[I] + rc[J]
        with np.errstate(divide='ignore', invalid='ignore'):
            pr = d / rs
        # ignore zero radii pairs
        mask = rs > 1e-12
        b = np.min(pr[mask]) if np.any(mask) else 1.0
        alpha = min(1.0, float(a), float(b))
        alpha = alpha * (1.0 - 3e-7) - 1e-12
        if not np.isfinite(alpha) or alpha <= 0:
            alpha = 0.5
        # also cap so wall margin at least 1e-9
        return rc * alpha, alpha

    # fallback 6x6 grid subset
    k = 6
    r0 = 0.5 / k
    fb = []
    for ix in range(k):
        for jy in range(k):
            if len(fb) < n:
                fb.append([(2 * ix + 1) * r0, (2 * jy + 1) * r0])
    fb = np.array(fb, float)
    best_x = fb[:, 0].copy(); best_y = fb[:, 1].copy()
    s_fb, r_fb = lp_eval(best_x, best_y)
    if s_fb < 0:
        best_r = np.full(n, r0 * 0.9)
        best_sum = float(np.sum(best_r))
        best_r, _ = safe_scale(best_x, best_y, best_r)
        best_sum = float(np.sum(best_r))
    else:
        best_r, _ = safe_scale(best_x, best_y, r_fb)
        best_sum = float(np.sum(best_r))
        # re-evaluate true LP sum for ranking (before shrink) -> use shrunk sum
    pool = [(best_sum, best_x.copy(), best_y.copy(), best_r.copy())]

    # ---------- penalty function ----------
    def penalty_fun(v, W):
        X = v[0:n]; Y = v[n:2 * n]; R = v[2 * n:3 * n]
        gx = np.zeros(n); gy = np.zeros(n); gr_pen = np.zeros(n)
        pen = 0.0
        v1 = R - X
        mm = v1 > 0
        if np.any(mm):
            pen += np.sum(v1[mm] ** 2); gx[mm] += -2.0 * v1[mm]; gr_pen[mm] += 2.0 * v1[mm]
        v2 = X + R - 1.0
        mm = v2 > 0
        if np.any(mm):
            pen += np.sum(v2[mm] ** 2); gx[mm] += 2.0 * v2[mm]; gr_pen[mm] += 2.0 * v2[mm]
        v3 = R - Y
        mm = v3 > 0
        if np.any(mm):
            pen += np.sum(v3[mm] ** 2); gy[mm] += -2.0 * v3[mm]; gr_pen[mm] += 2.0 * v3[mm]
        v4 = Y + R - 1.0
        mm = v4 > 0
        if np.any(mm):
            pen += np.sum(v4[mm] ** 2); gy[mm] += 2.0 * v4[mm]; gr_pen[mm] += 2.0 * v4[mm]
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
            Q = np.zeros((n, n)); Q[Mask] = -2.0 * V[Mask] / D[Mask]
            gx += np.sum(Q * Dx, axis=1); gy += np.sum(Q * Dy, axis=1)
            G = np.zeros((n, n)); G[Mask] = 2.0 * V[Mask]
            gr_pen += np.sum(G, axis=1)
        loss = -np.sum(R) + W * pen
        g = np.empty_like(v)
        g[0:n] = W * gx; g[n:2 * n] = W * gy; g[2 * n:3 * n] = -1.0 + W * gr_pen
        return loss, g

    def make_init(rid):
        # diverse topology-breaking inits
        if rid == 0:
            xs = []; ys = []
            for ix in range(5):
                for jy in range(5):
                    xs.append((ix + 0.5) / 5.0); ys.append((jy + 0.5) / 5.0)
            xs.append(0.85); ys.append(0.15)
            xc = np.array(xs, float) + rng.normal(0, 0.015, n)
            yc = np.array(ys, float) + rng.normal(0, 0.015, n)
            R0 = 0.09 + 0.02 * rng.random(n)
        elif rid % 4 == 1:
            # random row-count template
            nr = int(rng.integers(4, 7))  # 4..6
            # random composition of 26 into nr parts each >=3
            base = [3] * nr
            rem = n - 3 * nr
            for _ in range(rem):
                base[int(rng.integers(nr))] += 1
            rng.shuffle(base)
            xs = []; ys = []
            for rr_, cnt in enumerate(base):
                y = (rr_ + 0.5) / nr
                off = 0.5 * (rr_ % 2) / max(cnt, 1)
                for cc in range(cnt):
                    x = (cc + 0.5) / cnt * 0.94 + 0.03 + off * 0.3
                    xs.append(x); ys.append(y)
            xc = np.array(xs, float) + rng.normal(0, 0.012, n)
            yc = np.array(ys, float) + rng.normal(0, 0.012, n)
            R0 = 0.075 + 0.035 * rng.random(n)
        elif rid % 4 == 2:
            # hex / staggered
            pts = []
            for ix in range(7):
                for jy in range(6):
                    xx = (ix + 0.5) / 7.0 + (0.06 if (jy % 2) else 0.0)
                    yy = (jy + 0.5) / 6.0
                    if xx < 0.985:
                        pts.append((xx, yy))
            pts = np.array(pts)
            perm = rng.permutation(len(pts))[:n]
            xc = pts[perm, 0] + rng.normal(0, 0.012, n)
            yc = pts[perm, 1] + rng.normal(0, 0.012, n)
            R0 = 0.07 + 0.04 * rng.random(n)
        elif rid % 4 == 3:
            # column template (transpose of row) to force different graph
            nc = int(rng.integers(4, 7))
            base = [3] * nc
            rem = n - 3 * nc
            for _ in range(rem):
                base[int(rng.integers(nc))] += 1
            rng.shuffle(base)
            xs = []; ys = []
            for cc_, cnt in enumerate(base):
                x = (cc_ + 0.5) / nc
                for rr_ in range(cnt):
                    y = (rr_ + 0.5) / cnt * 0.94 + 0.03
                    xs.append(x + rng.normal(0, 0.01)); ys.append(y + rng.normal(0, 0.01))
            xc = np.array(xs[:n], float); yc = np.array(ys[:n], float)
            R0 = 0.075 + 0.035 * rng.random(n)
        else:
            xc = rng.uniform(0.04, 0.96, n); yc = rng.uniform(0.04, 0.96, n)
            R0 = 0.06 + 0.05 * rng.random(n)
        xc = np.clip(xc, 0.03, 0.97); yc = np.clip(yc, 0.03, 0.97)
        R0 = np.clip(R0, 0.03, 0.19)
        return xc, yc, R0

    bounds78 = [(0.0, 1.0)] * (2 * n) + [(0.02, 0.20)] * n
    Ws = [5e2, 5e3, 5e4]
    T_PEN = t0 + 38.0
    restart = 0
    try:
        while time.time() < T_PEN and restart < 30:
            xc0, yc0, R00 = make_init(restart)
            v = np.concatenate([xc0, yc0, R00])
            for W in Ws:
                if time.time() > T_PEN:
                    break
                try:
                    res = minimize(penalty_fun, v, args=(W,), jac=True, method='L-BFGS-B',
                                   bounds=bounds78, options={'maxiter': 90, 'ftol': 1e-9, 'gtol': 1e-6})
                    v = res.x
                except Exception:
                    break
            X = np.clip(v[0:n], 0.01, 0.99); Y = np.clip(v[n:2 * n], 0.01, 0.99)
            s, r = lp_eval(X, Y)
            if s > 0:
                rs, _ = safe_scale(X, Y, r)
                ss = float(np.sum(rs))
                if ss > best_sum:
                    best_sum = ss; best_x = X.copy(); best_y = Y.copy(); best_r = rs.copy()
                # pool on raw LP sum (monotonic with shrunk)
                pool.append((ss, X.copy(), Y.copy(), rs.copy()))
                pool = sorted(pool, key=lambda t: -t[0])[:6]
            restart += 1
    except Exception:
        pass

    # ---------- Phase B: LP annealing with void teleport ----------
    try:
        cur_x = best_x.copy(); cur_y = best_y.copy(); cur_s = best_sum
        # get raw radii for void computation (approx from best_r)
        cur_r = best_r.copy()
        T0 = 0.004; T1 = 0.00012
        t_hill_end = t0 + DEADLINE - 14.0
        it = 0
        # pre-sort indices by radius for small-circle bias
        while time.time() < t_hill_end:
            frac = (time.time() - t0) / max(1e-9, (t_hill_end - t0))
            frac = min(1.0, max(0.0, frac))
            Temp = T0 * (T1 / T0) ** frac
            sig_g = 0.016 * (1 - frac) + 0.002 * frac
            sig_s = 0.055 * (1 - frac) + 0.007 * frac
            if it % 400 == 399:
                # kick from pool or best
                if len(pool) > 1 and rng.random() < 0.5:
                    idx = int(rng.integers(len(pool)))
                    cur_x = pool[idx][1].copy() + rng.normal(0, 0.012, n)
                    cur_y = pool[idx][2].copy() + rng.normal(0, 0.012, n)
                else:
                    cur_x = best_x.copy() + rng.normal(0, 0.028, n)
                    cur_y = best_y.copy() + rng.normal(0, 0.028, n)
                cur_x = np.clip(cur_x, 0.015, 0.985); cur_y = np.clip(cur_y, 0.015, 0.985)
                s_new, r_new = lp_eval(cur_x, cur_y)
                if s_new > 0:
                    rs, _ = safe_scale(cur_x, cur_y, r_new)
                    cur_s = float(np.sum(rs)); cur_r = rs.copy()
                    if cur_s > best_sum:
                        best_sum = cur_s; best_x = cur_x.copy(); best_y = cur_y.copy(); best_r = rs.copy()
                        pool.append((cur_s, cur_x.copy(), cur_y.copy(), rs.copy()))
                        pool = sorted(pool, key=lambda t: -t[0])[:6]
                else:
                    cur_x = best_x.copy(); cur_y = best_y.copy(); cur_s = best_sum; cur_r = best_r.copy()
                it += 1
                continue
            u = rng.random()
            nx = cur_x.copy(); ny = cur_y.copy()
            if u < 0.40:
                nx = nx + rng.normal(0, sig_g, n); ny = ny + rng.normal(0, sig_g, n)
                nx = np.clip(nx, 0.015, 0.985); ny = np.clip(ny, 0.015, 0.985)
            elif u < 0.68:
                # single move, bias to small circles
                if rng.random() < 0.6:
                    order = np.argsort(cur_r)
                    kk = order[int(rng.integers(min(n, 10)))]
                else:
                    kk = int(rng.integers(n))
                nx[kk] += rng.normal(0, sig_s); ny[kk] += rng.normal(0, sig_s)
                nx[kk] = np.clip(nx[kk], 0.01, 0.99); ny[kk] = np.clip(ny[kk], 0.01, 0.99)
            elif u < 0.85:
                # void teleport: find largest void by sampling, move a small circle there
                # sample candidates
                ncand = 24
                candx = rng.uniform(0.02, 0.98, ncand); candy = rng.uniform(0.02, 0.98, ncand)
                best_clear = -1e9; bv = (0.5, 0.5)
                for ci in range(ncand):
                    dxv = candx[ci] - cur_x; dyv = candy[ci] - cur_y
                    dv = np.sqrt(dxv * dxv + dyv * dyv) - cur_r
                    wallv = min(candx[ci], 1.0 - candx[ci], candy[ci], 1.0 - candy[ci])
                    clear = min(np.min(dv), wallv)
                    if clear > best_clear:
                        best_clear = clear; bv = (candx[ci], candy[ci])
                order = np.argsort(cur_r)
                kk = order[int(rng.integers(min(n, 8)))]
                nx[kk] = bv[0] + rng.normal(0, 0.005); ny[kk] = bv[1] + rng.normal(0, 0.005)
                nx[kk] = np.clip(nx[kk], 0.01, 0.99); ny[kk] = np.clip(ny[kk], 0.01, 0.99)
            else:
                k1 = int(rng.integers(n)); k2 = int(rng.integers(n))
                nx[k1] += rng.normal(0, sig_s); ny[k1] += rng.normal(0, sig_s)
                nx[k2] += rng.normal(0, sig_s); ny[k2] += rng.normal(0, sig_s)
                nx = np.clip(nx, 0.015, 0.985); ny = np.clip(ny, 0.015, 0.985)
            s_new, r_new = lp_eval(nx, ny)
            if s_new < 0:
                it += 1; continue
            rs, _ = safe_scale(nx, ny, r_new)
            ss = float(np.sum(rs))
            if ss >= cur_s or rng.random() < np.exp((ss - cur_s) / max(Temp, 1e-12)):
                cur_x = nx; cur_y = ny; cur_s = ss; cur_r = rs.copy()
                if ss > best_sum:
                    best_sum = ss; best_x = nx.copy(); best_y = ny.copy(); best_r = rs.copy()
                    pool.append((ss, nx.copy(), ny.copy(), rs.copy()))
                    pool = sorted(pool, key=lambda t: -t[0])[:6]
            it += 1
    except Exception:
        pass

    # ---------- Phase C: polish ----------
    try:
        # joint rejamming from best
        v = np.concatenate([best_x, best_y, np.clip(best_r / max(1e-9, np.min(np.ones(1))), 0.02, 0.19)])
        # best_r already feasible; use as start radii
        v = np.concatenate([best_x, best_y, np.clip(best_r, 0.02, 0.19)])
        for W in [5e3, 5e4, 3e5]:
            if time.time() - t0 > DEADLINE - 5:
                break
            try:
                res = minimize(penalty_fun, v, args=(W,), jac=True, method='L-BFGS-B',
                               bounds=bounds78, options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8})
                v = res.x
            except Exception:
                break
        Xp = np.clip(v[0:n], 0.005, 0.995); Yp = np.clip(v[n:2 * n], 0.005, 0.995)
        sp, rp = lp_eval(Xp, Yp)
        if sp > 0:
            rs, _ = safe_scale(Xp, Yp, rp)
            ss = float(np.sum(rs))
            if ss > best_sum:
                best_sum = ss; best_x = Xp; best_y = Yp; best_r = rs
    except Exception:
        pass

    # greedy coordinate ascent on centres with LP score
    try:
        step = 0.008
        while time.time() - t0 < DEADLINE - 2.0 and step > 0.0004:
            improved = False
            order = list(range(n))
            rng.shuffle(order)
            for kk in order:
                if time.time() - t0 > DEADLINE - 2.0:
                    break
                base_s = best_sum
                best_try = None
                # 8 directions
                for ang in [0, np.pi/2, np.pi, 3*np.pi/2, np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]:
                    tx = best_x.copy(); ty = best_y.copy()
                    tx[kk] += step * np.cos(ang); ty[kk] += step * np.sin(ang)
                    if tx[kk] < 0.01 or tx[kk] > 0.99 or ty[kk] < 0.01 or ty[kk] > 0.99:
                        continue
                    s_new, r_new = lp_eval(tx, ty)
                    if s_new < 0:
                        continue
                    rs, _ = safe_scale(tx, ty, r_new)
                    ss = float(np.sum(rs))
                    if ss > base_s + 1e-10:
                        base_s = ss; best_try = (tx, ty, rs, ss)
                if best_try is not None:
                    best_x, best_y, best_r, best_sum = best_try[0], best_try[1], best_try[2], best_try[3]
                    improved = True
            if not improved:
                step *= 0.5
    except Exception:
        pass

    # final strict enforcement
    try:
        sfin, rfin = lp_eval(best_x, best_y)
        if sfin > 0:
            rs, _ = safe_scale(best_x, best_y, rfin)
            # keep whichever better (best_r already scaled)
            if float(np.sum(rs)) > best_sum - 1e-12:
                best_r = rs; best_sum = float(np.sum(rs))
        # extra shrink
        best_r = np.maximum(np.asarray(best_r, float), 1e-6) * (1.0 - 1e-7)
        # final validation repair if needed
        wall = np.minimum(np.minimum(best_x, 1.0 - best_x), np.minimum(best_y, 1.0 - best_y))
        dx = best_x[I] - best_x[J]; dy = best_y[I] - best_y[J]
        d = np.sqrt(dx * dx + dy * dy)
        rs = best_r[I] + best_r[J]
        need = 1.0
        # wall
        wr = np.min(wall / np.maximum(best_r, 1e-12))
        pr = np.min(d / np.maximum(rs, 1e-12))
        alpha = min(1.0, wr, pr) * (1.0 - 1e-6) - 1e-12
        if alpha < 1.0:
            if alpha <= 0:
                alpha = 0.9
            best_r = best_r * alpha
    except Exception:
        best_r = np.maximum(np.asarray(best_r, float), 1e-6) * 0.999999

    out = np.column_stack([np.clip(best_x, 0.0, 1.0), np.clip(best_y, 0.0, 1.0), np.maximum(best_r, 1e-9)])
    return out
