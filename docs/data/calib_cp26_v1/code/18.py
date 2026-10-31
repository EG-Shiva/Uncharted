import numpy as np
import time

def solve():
    import time as _tm
    import numpy as _np
    from scipy.optimize import linprog
    n = 26
    t0 = _tm.time()
    DEADLINE = 124.0

    iu, ju = _np.triu_indices(n, 1)
    m = len(iu)
    K = m + n
    # primal A (m,n)
    A_primal = _np.zeros((m, n), dtype=float)
    ar = _np.arange(m)
    A_primal[ar, iu] = 1.0
    A_primal[ar, ju] = 1.0
    c_primal = -_np.ones(n, dtype=float)
    # dual A (n,K): M = [A_primal.T , I], A_ub_dual = -M, b_ub=-1
    Mmat = _np.zeros((n, K), dtype=float)
    Mmat[:, :m] = A_primal.T
    Mmat[_np.arange(n), m + _np.arange(n)] = 1.0
    A_dual = -Mmat
    b_dual = -_np.ones(n, dtype=float)
    bounds_dual = [(0.0, None)] * K

    def primal_eval(xc, yc):
        b = _np.minimum(_np.minimum(xc, 1.0 - xc), _np.minimum(yc, 1.0 - yc))
        # need strictly inside
        if _np.min(b) <= 1e-9:
            return None, -1.0
        dx = xc[iu] - xc[ju]
        dy = yc[iu] - yc[ju]
        dvec = _np.sqrt(dx * dx + dy * dy)
        # coincident -> infeasible for positive radii
        if _np.min(dvec) <= 1e-9:
            return None, -1.0
        bounds = [(0.0, float(bi)) for bi in b]
        try:
            res = linprog(c_primal, A_ub=A_primal, b_ub=dvec, bounds=bounds, method='highs')
        except Exception:
            return None, -1.0
        if not res.success:
            return None, -1.0
        r = _np.maximum(_np.array(res.x, dtype=float).flatten(), 0.0)
        s = float(_np.sum(r))
        return r, s

    def dual_val_grad(xc, yc):
        b = _np.minimum(_np.minimum(xc, 1.0 - xc), _np.minimum(yc, 1.0 - yc))
        if _np.min(b) <= 1e-9:
            return -1.0, None, None
        dx = xc[iu] - xc[ju]
        dy = yc[iu] - yc[ju]
        dvec = _np.sqrt(dx * dx + dy * dy)
        if _np.min(dvec) <= 1e-9:
            return -1.0, None, None
        c_dual = _np.concatenate([dvec, b])
        try:
            res = linprog(c_dual, A_ub=A_dual, b_ub=b_dual, bounds=bounds_dual, method='highs')
        except Exception:
            return -1.0, None, None
        if not res.success:
            return -1.0, None, None
        S = float(res.fun)  # min value = max sum
        z = _np.array(res.x, dtype=float).flatten()
        p = z[:m]
        q = z[m:]
        # gradients
        gx = _np.zeros(n, dtype=float)
        gy = _np.zeros(n, dtype=float)
        # pair part; threshold tiny p to save
        mask = p > 1e-12
        if _np.any(mask):
            inv = _np.zeros(m, dtype=float)
            inv[mask] = 1.0 / dvec[mask]
            # avoid huge from near-zero d
            wx = p * inv * dx
            wy = p * inv * dy
            # accumulate: gx[iu]+=wx, gx[ju]-=wx
            _np.add.at(gx, iu[mask], wx[mask])
            _np.add.at(gx, ju[mask], -wx[mask])
            _np.add.at(gy, iu[mask], wy[mask])
            _np.add.at(gy, ju[mask], -wy[mask])
        # wall part
        # determine argmin wall per i
        # walls: x, 1-x, y, 1-y
        for i in range(n):
            qi = q[i]
            if qi <= 1e-12:
                continue
            xi = xc[i]; yi = yc[i]
            w0 = xi; w1 = 1.0 - xi; w2 = yi; w3 = 1.0 - yi
            wm = w0
            idx = 0
            if w1 < wm:
                wm = w1; idx = 1
            if w2 < wm:
                wm = w2; idx = 2
            if w3 < wm:
                wm = w3; idx = 3
            if idx == 0:
                gx[i] += qi
            elif idx == 1:
                gx[i] -= qi
            elif idx == 2:
                gy[i] += qi
            else:
                gy[i] -= qi
        return S, gx, gy

    # global best (centres + radii)
    best_S = -1.0
    best_x = None
    best_y = None
    best_r = None

    def consider(xc, yc, r, s):
        global_vars = 1
        return s

    rng_master = _np.random.RandomState(0)
    init_seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 17]

    # ---- stage 1: dual gradient ascent (Adam) ----
    # time split: use until ~78s
    T1 = 78.0
    restart = 0
    for seed in init_seeds:
        if _tm.time() - t0 > T1:
            break
        rng = _np.random.RandomState(10000 + seed * 977)
        if seed == 0:
            # 5x5 + one extra jittered
            xs = []; ys = []
            for i in range(5):
                for j in range(5):
                    xs.append(0.1 + 0.2 * i); ys.append(0.1 + 0.2 * j)
            xs = _np.array(xs); ys = _np.array(ys)
            # add 26th near centre gap
            xs = _np.append(xs, 0.5); ys = _np.append(ys, 0.62)
            xc = xs + rng.randn(n) * 0.008
            yc = ys + rng.randn(n) * 0.008
        elif seed == 1:
            xs = ( _np.arange(6) + 0.5) / 6.0
            ys = ( _np.arange(5) + 0.5) / 5.0
            xx, yy = _np.meshgrid(xs, ys)
            xy = _np.column_stack([xx.ravel(), yy.ravel()])
            perm = rng.permutation(len(xy))[:n]
            xc = xy[perm, 0] + rng.randn(n) * 0.01
            yc = xy[perm, 1] + rng.randn(n) * 0.01
        else:
            xc = rng.uniform(0.06, 0.94, size=n)
            yc = rng.uniform(0.06, 0.94, size=n)
        xc = _np.clip(xc, 0.03, 0.97)
        yc = _np.clip(yc, 0.03, 0.97)

        # Adam state
        mx = _np.zeros(n); my = _np.zeros(n)
        vx = _np.zeros(n); vy = _np.zeros(n)
        b1, b2, eps = 0.9, 0.999, 1e-8
        lr0 = 0.015
        t_step = 0
        # steps per restart adapt to time
        nsteps = 350
        for it in range(nsteps):
            if _tm.time() - t0 > T1:
                break
            S, gx, gy = dual_val_grad(xc, yc)
            if gx is None:
                # push inside randomly
                xc = xc * 0.9 + 0.05 + rng.randn(n) * 0.01
                yc = yc * 0.9 + 0.05 + rng.randn(n) * 0.01
                xc = _np.clip(xc, 0.03, 0.97); yc = _np.clip(yc, 0.03, 0.97)
                continue
            # track best via primal occasionally (every 25 steps) to avoid extra solves
            if it % 25 == 0:
                r, s = primal_eval(xc, yc)
                if r is not None and s > best_S:
                    best_S = s; best_x = xc.copy(); best_y = yc.copy(); best_r = r.copy()
            # gradient clip
            gn = _np.sqrt(_np.sum(gx*gx) + _np.sum(gy*gy) + 1e-20)
            if gn > 5.0:
                sc = 5.0 / gn; gx *= sc; gy *= sc
            t_step += 1
            lr = lr0 / (1.0 + 0.003 * t_step)
            # small decay + cosine-ish
            mx = b1 * mx + (1 - b1) * gx
            my = b1 * my + (1 - b1) * gy
            vx = b2 * vx + (1 - b2) * gx * gx
            vy = b2 * vy + (1 - b2) * gy * gy
            mxh = mx / (1 - b1 ** t_step)
            myh = my / (1 - b1 ** t_step)
            vxh = vx / (1 - b2 ** t_step)
            vyh = vy / (1 - b2 ** t_step)
            xc = xc + lr * mxh / (_np.sqrt(vxh) + eps)
            yc = yc + lr * myh / (_np.sqrt(vyh) + eps)
            # tiny exploration noise decaying
            if it > 50:
                nl = 0.0006 * (1.0 - it / nsteps)
                if nl > 0:
                    xc += rng.randn(n) * nl
                    yc += rng.randn(n) * nl
            xc = _np.clip(xc, 0.02, 0.98)
            yc = _np.clip(yc, 0.02, 0.98)
        # end restart: final primal
        r, s = primal_eval(xc, yc)
        if r is not None and s > best_S:
            best_S = s; best_x = xc.copy(); best_y = yc.copy(); best_r = r.copy()
        restart += 1

    # ---- stage 2: hill climbing on centres with primal eval ----
    if best_x is None:
        k = int(_np.ceil(_np.sqrt(n)))
        rr = 0.5 / k
        pts = []
        for i in range(k):
            for j in range(k):
                if len(pts) < n:
                    pts.append([(2*i+1)*rr, (2*j+1)*rr, rr*0.999])
        return _np.array(pts, dtype=float)

    rng = _np.random.RandomState(999)
    sigmas = [0.015, 0.008, 0.004, 0.002, 0.001]
    # estimate remaining time
    for sigma in sigmas:
        iters = 900
        for _ in range(iters):
            if _tm.time() - t0 > DEADLINE - 2.0:
                break
            # block move: perturb subset sometimes for finer search
            if rng.rand() < 0.3:
                xn = best_x.copy(); yn = best_y.copy()
                idx = rng.choice(n, size=rng.randint(1, 4), replace=False)
                xn[idx] += sigma * rng.randn(len(idx))
                yn[idx] += sigma * rng.randn(len(idx))
            else:
                xn = best_x + sigma * rng.randn(n)
                yn = best_y + sigma * rng.randn(n)
            xn = _np.clip(xn, 0.02, 0.98)
            yn = _np.clip(yn, 0.02, 0.98)
            r, s = primal_eval(xn, yn)
            if r is not None and s > best_S + 1e-12:
                best_S = s; best_x = xn; best_y = yn; best_r = r
        if _tm.time() - t0 > DEADLINE - 2.0:
            break

    x = best_x; y = best_y; r = best_r.copy()
    # ensure positivity
    r = _np.maximum(r, 1e-7)
    # strict feasibility scaling
    wall = _np.minimum(_np.minimum(x, 1.0 - x), _np.minimum(y, 1.0 - y))
    Dx = x[:, None] - x[None, :]
    Dy = y[:, None] - y[None, :]
    D = _np.sqrt(Dx * Dx + Dy * Dy)
    fw = _np.min(wall / r)
    fp = 1e100
    for i in range(n):
        Di = D[i]
        Ri = r[i]
        for j in range(i + 1, n):
            v = Di[j] / (Ri + r[j])
            if v < fp:
                fp = v
    f = fw if fw < fp else fp
    if not _np.isfinite(f):
        f = fw
    if f < 1.0:
        r = r * f * (1.0 - 1e-9)
    else:
        r = r * (1.0 - 1e-9)
    # re-check tiny margin
    out = _np.column_stack([x, y, r])
    return out
