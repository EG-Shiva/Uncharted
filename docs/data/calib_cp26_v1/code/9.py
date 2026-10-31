import numpy as np
import time

def solve():
    n = 26
    t0 = time.time()
    deadline = t0 + 122.0
    rng = np.random.RandomState(12345)

    # baseline grid
    best_sum = 0.0
    best_arr = None
    k = int(np.ceil(np.sqrt(n)))
    r0 = 0.5 / k * 0.999999
    pts = []
    for i in range(k):
        for j in range(k):
            if len(pts) < n:
                pts.append([(2*i+1)*0.5/k, (2*j+1)*0.5/k, r0])
    pts = np.array(pts, dtype=float)
    best_sum = float(np.sum(pts[:,2]))
    best_arr = pts.copy()

    try:
        from scipy.optimize import minimize
        has_scipy = True
    except Exception:
        return best_arr

    x_bnds = [(0.0, 1.0)]*(2*n) + [(0.01, 0.28)]*n

    def make_fun(W):
        def fun(v):
            xs = v[0:n]
            ys = v[n:2*n]
            rs = v[2*n:3*n]
            sum_r = np.sum(rs)
            vl = rs - xs
            vl = np.where(vl > 0, vl, 0.0)
            vr = xs + rs - 1.0
            vr = np.where(vr > 0, vr, 0.0)
            vb = rs - ys
            vb = np.where(vb > 0, vb, 0.0)
            vt = ys + rs - 1.0
            vt = np.where(vt > 0, vt, 0.0)
            box = np.sum(vl*vl + vr*vr + vb*vb + vt*vt)
            Dx = xs[:, None] - xs[None, :]
            Dy = ys[:, None] - ys[None, :]
            D = np.sqrt(Dx*Dx + Dy*Dy)
            Rsum = rs[:, None] + rs[None, :]
            V = Rsum - D
            np.fill_diagonal(V, 0.0)
            np.fill_diagonal(D, 1.0)
            viol = np.where(V > 0, V, 0.0)
            pair = np.sum(viol*viol) * 0.5
            f = -sum_r + W*(box + pair)
            gx = 2.0*(-vl + vr)
            gy = 2.0*(-vb + vt)
            gr_box = 2.0*(vl + vr + vb + vt)
            Dsafe = np.maximum(D, 1e-8)
            with np.errstate(divide='ignore', invalid='ignore'):
                K = 2.0*viol / Dsafe
            np.fill_diagonal(K, 0.0)
            Ksum = np.sum(K, axis=1)
            gx_pair = np.dot(K, xs) - xs*Ksum
            gy_pair = np.dot(K, ys) - ys*Ksum
            # note: dot(K,xs) = sum_j K_kj * x_j
            gr_pair = 2.0*np.sum(viol, axis=1)
            gx_total = W*(gx + gx_pair)
            gy_total = W*(gy + gy_pair)
            gr_total = -np.ones(n) + W*(gr_box + gr_pair)
            g = np.empty_like(v)
            g[0:n] = gx_total
            g[n:2*n] = gy_total
            g[2*n:3*n] = gr_total
            return f, g
        return fun

    def inflate(xs, ys, rs):
        xs = np.clip(xs, 0.0, 1.0).copy()
        ys = np.clip(ys, 0.0, 1.0).copy()
        rs = np.clip(rs, 0.005, 0.28).copy()
        xs = np.minimum(np.maximum(xs, rs), 1.0 - rs)
        ys = np.minimum(np.maximum(ys, rs), 1.0 - rs)
        # uniform shrink to feasibility (few iterations)
        for _ in range(10):
            Dx = xs[:, None] - xs[None, :]
            Dy = ys[:, None] - ys[None, :]
            D = np.sqrt(Dx*Dx + Dy*Dy)
            np.fill_diagonal(D, 1e9)
            Rsum = rs[:, None] + rs[None, :] + 1e-18
            ratio = D / Rsum
            m = np.min(ratio)
            if m >= 1.0:
                break
            rs = rs * (m * (1.0 - 1e-9))
            # keep inside after shrink (still inside)
        order = np.arange(n)
        for _ in range(60):
            rng.shuffle(order)
            max_inc = 0.0
            for i in order:
                dx = xs[i] - xs
                dy = ys[i] - ys
                dist = np.sqrt(dx*dx + dy*dy)
                dist[i] = np.inf
                lim_other = np.min(dist - rs)
                lim_wall = xs[i]
                if 1.0 - xs[i] < lim_wall:
                    lim_wall = 1.0 - xs[i]
                if ys[i] < lim_wall:
                    lim_wall = ys[i]
                if 1.0 - ys[i] < lim_wall:
                    lim_wall = 1.0 - ys[i]
                rmax = lim_other if lim_other < lim_wall else lim_wall
                rmax = rmax * (1.0 - 1e-9)
                if rmax < 1e-9:
                    rmax = 1e-9
                if rmax > rs[i] + 1e-12:
                    d = rmax - rs[i]
                    if d > max_inc:
                        max_inc = d
                    rs[i] = rmax
            if max_inc < 1e-10:
                break
        rs = rs * (1.0 - 5e-9)
        return xs, ys, rs

    WS = [30.0, 300.0, 3000.0, 30000.0]
    MITS = [100, 150, 200, 300]

    def run_one(v0):
        v = v0.copy()
        for stage in range(len(WS)):
            if time.time() > deadline - 2.0:
                break
            W = WS[stage]
            mi = MITS[stage]
            fun = make_fun(W)
            try:
                res = minimize(fun, v, jac=True, method='L-BFGS-B', bounds=x_bnds,
                               options={'maxiter': mi, 'ftol': 1e-9, 'gtol': 1e-5, 'maxfun': 20000})
                v = res.x
            except Exception:
                break
        return v

    seed_idx = 0
    # precompute a 5x6 grid template
    gx_template = []
    gy_template = []
    for ix in range(5):
        for jy in range(6):
            if len(gx_template) < n:
                gx_template.append((ix+0.5)/5.0)
                gy_template.append((jy+0.5)/6.0)
    gx_template = np.array(gx_template)
    gy_template = np.array(gy_template)

    while True:
        if time.time() > deadline - 8.0:
            break
        # build v0
        if seed_idx % 7 == 0 and seed_idx < 14:
            # jittered grid
            jx = rng.randn(n)*0.02
            jy = rng.randn(n)*0.02
            xs0 = np.clip(gx_template + jx, 0.05, 0.95)
            ys0 = np.clip(gy_template + jy, 0.05, 0.95)
            rs0 = rng.uniform(0.07, 0.11, size=n)
        elif seed_idx % 5 == 1:
            xs0 = rng.rand(n)
            ys0 = rng.rand(n)
            rs0 = rng.uniform(0.08, 0.115, size=n)
        elif seed_idx % 5 == 2:
            xs0 = rng.rand(n)
            ys0 = rng.rand(n)
            rs0 = rng.uniform(0.05, 0.13, size=n)
        elif seed_idx % 5 == 3:
            xs0 = rng.rand(n)
            ys0 = rng.rand(n)
            rs0 = rng.uniform(0.06, 0.12, size=n)
            # force some large circles
            rs0[:4] = rng.uniform(0.11, 0.15, size=4)
        else:
            xs0 = rng.rand(n)
            ys0 = rng.rand(n)
            rs0 = np.full(n, 0.095) + rng.randn(n)*0.015
            rs0 = np.clip(rs0, 0.04, 0.16)
        v0 = np.empty(3*n)
        v0[0:n] = xs0
        v0[n:2*n] = ys0
        v0[2*n:3*n] = rs0
        v_opt = run_one(v0)
        xs = v_opt[0:n]; ys = v_opt[n:2*n]; rs = v_opt[2*n:3*n]
        xs, ys, rs = inflate(xs, ys, rs)
        s = float(np.sum(rs))
        if s > best_sum + 1e-12:
            # verify strictly
            ok = True
            if np.any(rs <= 0): ok = False
            if np.any(xs < rs - 1e-12) or np.any(xs > 1-rs + 1e-12): ok = False
            if np.any(ys < rs - 1e-12) or np.any(ys > 1-rs + 1e-12): ok = False
            if ok:
                Dx = xs[:,None]-xs[None,:]
                Dy = ys[:,None]-ys[None,:]
                D = np.sqrt(Dx*Dx+Dy*Dy)
                for i in range(n):
                    for j in range(i+1, n):
                        if D[i,j] + 1e-12 < rs[i]+rs[j]:
                            ok = False
                            break
                    if not ok: break
            if ok:
                best_sum = s
                best_arr = np.column_stack((xs, ys, rs))
        seed_idx += 1
        if seed_idx > 500:
            break

    # hill climbing on best for remaining time
    if best_arr is not None:
        bxs = best_arr[:,0].copy(); bys = best_arr[:,1].copy(); brs = best_arr[:,2].copy()
        sigma = 0.015
        it = 0
        while time.time() < deadline - 0.5 and it < 8000:
            it += 1
            i = rng.randint(n)
            xs_t = bxs.copy(); ys_t = bys.copy(); rs_t = brs.copy()
            xs_t[i] += rng.randn()*sigma
            ys_t[i] += rng.randn()*sigma
            xs_t[i] = np.clip(xs_t[i], 0.0, 1.0)
            ys_t[i] = np.clip(ys_t[i], 0.0, 1.0)
            # slight radius jitter
            rs_t[i] *= (1.0 + rng.randn()*0.02)
            xs_t, ys_t, rs_t = inflate(xs_t, ys_t, rs_t)
            s = float(np.sum(rs_t))
            if s > best_sum + 1e-10:
                best_sum = s
                bxs, bys, brs = xs_t, ys_t, rs_t
                best_arr = np.column_stack((bxs, bys, brs))
                sigma = min(0.03, sigma*1.05)
            else:
                if it % 200 == 0:
                    sigma *= 0.9
                    if sigma < 0.002:
                        sigma = 0.002
    # final safety: ensure strict feasibility
    if best_arr is not None:
        xs = best_arr[:,0]; ys = best_arr[:,1]; rs = best_arr[:,2].copy()
        rs = rs * (1.0 - 1e-9)
        # re-clip centers if needed
        xs = np.minimum(np.maximum(xs, rs+1e-10), 1.0-rs-1e-10)
        ys = np.minimum(np.maximum(ys, rs+1e-10), 1.0-rs-1e-10)
        best_arr = np.column_stack((xs, ys, rs))
        return best_arr
    return pts
