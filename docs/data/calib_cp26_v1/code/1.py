import numpy as np
import time

def solve():
    n = 26
    t0 = time.time()
    time_limit = 118.0
    rng = np.random.RandomState(0)

    try:
        from scipy.optimize import minimize
    except Exception:
        # fallback grid
        k = int(np.ceil(np.sqrt(n)))
        r = 0.5 / k
        pts = []
        for i in range(k):
            for j in range(k):
                if len(pts) < n:
                    pts.append([(2*i+1)*r, (2*j+1)*r, r*0.999999])
        return np.array(pts)

    # bounds
    bnds = [(0.0, 1.0)]* (2*n) + [(0.008, 0.30)] * n

    def make_fun(W):
        def fun(v):
            xs = v[0:n]
            ys = v[n:2*n]
            rs = v[2*n:3*n]
            sum_r = np.sum(rs)
            # box violations
            vl = rs - xs
            vl = np.where(vl > 0, vl, 0.0)
            vr = xs + rs - 1.0
            vr = np.where(vr > 0, vr, 0.0)
            vb = rs - ys
            vb = np.where(vb > 0, vb, 0.0)
            vt = ys + rs - 1.0
            vt = np.where(vt > 0, vt, 0.0)
            box = np.sum(vl*vl + vr*vr + vb*vb + vt*vt)
            # pairwise
            Dx = xs[:, None] - xs[None, :]
            Dy = ys[:, None] - ys[None, :]
            D = np.sqrt(Dx*Dx + Dy*Dy)
            # avoid zero diag for division
            # set diag to 1
            # use copy to not modify for viol?
            Rsum = rs[:, None] + rs[None, :]
            # raw viol
            V = Rsum - D
            # zero diagonal (D diag 0 gives V=2r>0, must kill)
            np.fill_diagonal(V, 0.0)
            np.fill_diagonal(D, 1.0)
            viol = np.where(V > 0, V, 0.0)
            # pair penalty sum over i<j
            pair = np.sum(viol*viol) * 0.5
            f = -sum_r + W * (box + pair)

            # grads
            # box grad
            gx = np.zeros(n)
            gy = np.zeros(n)
            gr_box = np.zeros(n)
            # vl>0 etc. vl array already relu
            # d box / dx = 2*(-vl + vr)
            gx = 2.0 * (-vl + vr)
            gy = 2.0 * (-vb + vt)
            gr_box = 2.0 * (vl + vr + vb + vt)

            # pair grad
            # K = 2*viol / D
            # D has 1 on diag, viol diag 0 => K diag 0
            with np.errstate(divide='ignore', invalid='ignore'):
                K = 2.0 * viol / D
            # K symmetric
            Ksum = np.sum(K, axis=1)
            gx_pair = np.dot(K, xs) - xs * Ksum
            gy_pair = np.dot(K, ys) - ys * Ksum
            gr_pair = 2.0 * np.sum(viol, axis=1)

            gx_total = W * (gx + gx_pair)
            gy_total = W * (gy + gy_pair)
            gr_total = -np.ones(n) + W * (gr_box + gr_pair)

            g = np.empty_like(v)
            g[0:n] = gx_total
            g[n:2*n] = gy_total
            g[2*n:3*n] = gr_total
            return f, g
        return fun

    def feasible_and_sum(x):
        xs = np.clip(x[0:n], 0.0, 1.0)
        ys = np.clip(x[n:2*n], 0.0, 1.0)
        rs = np.clip(x[2*n:3*n], 0.005, 0.4).copy()
        # clip centers to satisfy border for current rs
        # ensure rs <=0.5
        rs = np.minimum(rs, 0.49)
        xs = np.minimum(np.maximum(xs, rs), 1.0 - rs)
        ys = np.minimum(np.maximum(ys, rs), 1.0 - rs)
        # min pairwise ratio
        Dx = xs[:, None] - xs[None, :]
        Dy = ys[:, None] - ys[None, :]
        D = np.sqrt(Dx*Dx + Dy*Dy)
        Rsum = rs[:, None] + rs[None, :]
        # compute min over i<j
        # fill diag with large
        np.fill_diagonal(D, 1e9)
        # avoid div by zero
        ratio = D / (Rsum + 1e-18)
        # upper triangle min
        # min over all i!=j
        s = np.min(ratio)
        if s > 1.0:
            s = 1.0
        # shrink with safety margin for strict 1e-12
        s = s * (1.0 - 1e-7)
        rs = rs * s
        # after shrink borders still strictly inside (since rs smaller)
        ssum = np.sum(rs)
        arr = np.column_stack((xs, ys, rs))
        return ssum, arr

    def run_opt(v0, W, maxiter):
        fun = make_fun(W)
        try:
            res = minimize(fun, v0, jac=True, method='L-BFGS-B', bounds=bnds,
                           options={'maxiter': maxiter, 'ftol': 1e-9, 'gtol': 1e-6, 'maxfun': 20000})
            return res.x
        except Exception:
            return v0

    # build initial guesses
    inits = []

    # structured 5x5 + 1 gap
    v = np.empty(3*n)
    xs0 = []; ys0 = []; rs0 = []
    for ix in range(5):
        for jy in range(5):
            xs0.append(0.1 + 0.2*ix)
            ys0.append(0.1 + 0.2*jy)
            rs0.append(0.099)
    # 25 circles, need 26: add gap circle at (0.2,0.2) small; shift? we have 25, add one
    xs0 = np.array(xs0); ys0 = np.array(ys0); rs0 = np.array(rs0)
    # replace one? Actually we have 25, need 26 -> add
    xs_all = np.append(xs0, 0.2)
    ys_all = np.append(ys0, 0.2)
    rs_all = np.append(rs0, 0.04)
    # our 5x5 includes (0.1,0.1) etc, gap (0.2,0.2) overlaps heavily; move extra to a gap
    # instead put extra at (0.9,0.1)? Let's perturb: put at 0.5,0.5 gap? keep as is, optimizer will resolve
    # To avoid huge overlap, shrink all a bit
    rs_all = rs_all * 0.9
    v[0:n] = xs_all[:n] if len(xs_all)>=n else np.pad(xs_all,(0,n-len(xs_all)))
    v[n:2*n] = ys_all[:n] if len(ys_all)>=n else np.pad(ys_all,(0,n-len(ys_all)))
    v[2*n:3*n] = rs_all[:n] if len(rs_all)>=n else np.pad(rs_all,(0,n-len(rs_all)),constant_values=0.05)
    inits.append(v.copy())

    # structured 6-col jittered grid (like parent but jittered)
    vv = np.empty(3*n)
    pts = []
    kk = 6
    rr = 0.5/kk*0.95
    cnt = 0
    for ix in range(kk):
        for jy in range(kk):
            if cnt < n:
                vv[cnt] = (2*ix+1)*0.5/kk + rng.randn()*0.005
                vv[n+cnt] = (2*jy+1)*0.5/kk + rng.randn()*0.005
                vv[2*n+cnt] = rr
                cnt += 1
    inits.append(vv.copy())

    # random inits
    n_rand = 48
    for k in range(n_rand):
        vv = np.empty(3*n)
        vv[0:n] = rng.rand(n)
        vv[n:2*n] = rng.rand(n)
        # varied mean radius
        mean_r = rng.uniform(0.06, 0.115)
        spread = rng.uniform(0.01, 0.04)
        rr = np.clip(rng.randn(n)*spread + mean_r, 0.02, 0.20)
        vv[2*n:3*n] = rr
        inits.append(vv)

    best_sum = -1
    best_arr = None

    for idx, v0 in enumerate(inits):
        if time.time() - t0 > time_limit - 8:
            break
        # stage1 low W
        x1 = run_opt(np.clip(v0, [0]*2*n+[0.008]*n, [1]*2*n+[0.30]*n), W=15.0, maxiter=250)
        if time.time() - t0 > time_limit - 8:
            # evaluate anyway
            s, arr = feasible_and_sum(x1)
            if s > best_sum:
                best_sum = s; best_arr = arr
            break
        x2 = run_opt(x1, W=400.0, maxiter=300)
        s, arr = feasible_and_sum(x2)
        if s > best_sum:
            best_sum = s
            best_arr = arr.copy()
        # quick extra polish with very high W from x2
        # only for promising
        if s > 2.3 and (time.time()-t0 < time_limit-15):
            x3 = run_opt(x2, W=2000.0, maxiter=150)
            s3, arr3 = feasible_and_sum(x3)
            if s3 > best_sum:
                best_sum = s3
                best_arr = arr3.copy()

    # final hill climbing from best
    if best_arr is not None:
        # try few perturbations of best
        for rep in range(8):
            if time.time() - t0 > time_limit - 2:
                break
            v0 = np.empty(3*n)
            v0[0:n] = best_arr[:, 0] + rng.randn(n)*0.008
            v0[n:2*n] = best_arr[:, 1] + rng.randn(n)*0.008
            v0[2*n:3*n] = best_arr[:, 2] * (1.0 + rng.randn(n)*0.05)
            v0[0:n] = np.clip(v0[0:n], 0.0, 1.0)
            v0[n:2*n] = np.clip(v0[n:2*n], 0.0, 1.0)
            v0[2*n:3*n] = np.clip(v0[2*n:3*n], 0.008, 0.30)
            x1 = run_opt(v0, W=400.0, maxiter=250)
            x2 = run_opt(x1, W=2000.0, maxiter=120)
            s, arr = feasible_and_sum(x2)
            if s > best_sum:
                best_sum = s
                best_arr = arr.copy()

    if best_arr is None:
        k = int(np.ceil(np.sqrt(n)))
        r = 0.5 / k
        pts = []
        for i in range(k):
            for j in range(k):
                if len(pts) < n:
                    pts.append([(2*i+1)*r, (2*j+1)*r, r*0.999999])
        return np.array(pts)

    # final safety: ensure strict feasibility with extra margin
    # best_arr already shrunk, add tiny extra shrink
    best_arr[:, 2] *= (1.0 - 1e-9)
    # ensure >0
    best_arr[:, 2] = np.maximum(best_arr[:, 2], 1e-6)
    return best_arr
