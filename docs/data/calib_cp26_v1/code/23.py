import numpy as np
import time

def solve():
    N = 26
    t0 = time.time()
    DEADLINE = 126.0  # hard stop margin for 130s
    rng = np.random.RandomState(0)
    np.random.seed(0)

    # ---------- constant LP matrix (pair + wall) ----------
    # pairs
    pair_i = []
    pair_j = []
    for i in range(N):
        for j in range(i + 1, N):
            pair_i.append(i)
            pair_j.append(j)
    pair_i = np.array(pair_i, dtype=int)
    pair_j = np.array(pair_j, dtype=int)
    npair = len(pair_i)
    nrow = npair + N
    A_ub0 = np.zeros((nrow, N), dtype=float)
    # pair rows
    rows = np.arange(npair)
    A_ub0[rows, pair_i] = 1.0
    A_ub0[rows, pair_j] = 1.0
    # wall rows
    for i in range(N):
        A_ub0[npiar := npair + i, i] = 1.0
    c_obj = -np.ones(N, dtype=float)
    bounds_lp = [(0.0, None)] * N
    EPS = 1e-7

    try:
        from scipy.optimize import minimize, linprog
        has_scipy = True
    except Exception:
        has_scipy = False
        minimize = None
        linprog = None

    # ---------- fallback feasible ----------
    k = int(np.ceil(np.sqrt(N)))
    r0f = 0.5 / k * 0.999
    pts = []
    for i in range(k):
        for j in range(k):
            if len(pts) < N:
                pts.append([(2 * i + 1) * 0.5 / k, (2 * j + 1) * 0.5 / k, r0f])
    pts = np.array(pts, dtype=float)
    # shrink to be safe
    pts[:, 2] *= (1.0 - 1e-7)
    best_arr = pts.copy()
    best_sum = float(np.sum(best_arr[:, 2]))
    best_x = best_arr[:, 0].copy()
    best_y = best_arr[:, 1].copy()
    best_r = best_arr[:, 2].copy()

    if not has_scipy:
        return best_arr

    # ---------- LP optimal radii for fixed centres ----------
    # A_ub0 captured
    def radii_lp(x, y):
        # clip centres strictly inside
        # quick wall check
        wall = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        if np.min(wall) <= 1e-9:
            return None, -1.0
        Dx = x[:, None] - x[None, :]
        Dy = y[:, None] - y[None, :]
        D = np.sqrt(Dx * Dx + Dy * Dy)
        # if any coincident -> infeasible
        # build b
        b_ub = np.empty(nrow, dtype=float)
        b_ub[:npair] = D[pair_i, pair_j] - EPS
        b_ub[npair:] = wall - EPS
        if np.min(b_ub) < 0:
            # still try? with zero bounds infeasible if negative
            # allow tiny negative? no, infeasible
            return None, -1.0
        try:
            res = linprog(c_obj, A_ub=A_ub0, b_ub=b_ub, bounds=bounds_lp, method='highs')
        except Exception:
            return None, -1.0
        if not res.success:
            return None, -1.0
        r = np.array(res.x, dtype=float)
        if np.any(~np.isfinite(r)):
            return None, -1.0
        if np.min(r) < 1e-9:
            # degenerate (zero radius) -> reject
            return None, -1.0
        s = float(np.sum(r))
        return r, s

    # try LP polish on fallback grid centres (likely improves)
    try:
        r_try, s_try = radii_lp(best_x, best_y)
        if r_try is not None and s_try > best_sum:
            best_sum = s_try
            best_r = r_try
            best_arr = np.column_stack([best_x, best_y, best_r])
    except Exception:
        pass

    # ---------- joint penalized objective with wall-free parametrization ----------
    # x = u*(1-2r)+r, y = v*(1-2r)+r
    bounds_box = [(0.0, 1.0)] * (2 * N) + [(0.015, 0.25)] * N

    def make_fun(W):
        def fun(zz):
            u = zz[:N]
            vv = zz[N:2 * N]
            rr = zz[2 * N:]
            om2r = 1.0 - 2.0 * rr
            x = u * om2r + rr
            y = vv * om2r + rr
            Dx = x[:, None] - x[None, :]
            Dy = y[:, None] - y[None, :]
            D = np.sqrt(Dx * Dx + Dy * Dy)
            # avoid div by zero on diag
            # set diag to 1 for inversion, 0 for E
            Rsum = rr[:, None] + rr[None, :]
            E = Rsum - D
            # fix diagonal (D diag=0 -> E diag=2r)
            # set diag to 0
            # use flat indexing
            E.flat[::N + 1] = 0.0
            D.flat[::N + 1] = 1.0
            Ep = np.where(E > 0, E, 0.0)
            # penalty 0.5*sum_{i<j} Ep^2
            # triu sum
            # fast: 0.25*sum all (since symmetric)
            Pen = 0.25 * np.sum(Ep * Ep)
            F = -np.sum(rr) + W * Pen
            InvD = 1.0 / D
            InvD.flat[::N + 1] = 0.0
            Wmat = Ep * InvD
            # Gx = -sum_j Wmat*Dx
            Gx = -np.sum(Wmat * Dx, axis=1)
            Gy = -np.sum(Wmat * Dy, axis=1)
            sumEp = np.sum(Ep, axis=1) * 0.5  # row sum counts each pair once? full row sum = sum_j Ep_kj (each pair once per k) -> correct, but our Pen=0.25 sum => dPen/dr direct =0.5*row_sum
            # check: Pen=0.25 sum_{all}Ep^2, dPen/dEp_kj (k!=j) =0.5*Ep_kj, times dEp/dr=1 => 0.5*row_sum. good.
            # dPen/dx: 0.5*Ep * (-Dx/D) summed? For full matrix double? Let's derive: Pen=0.25 sum Ep^2, derivative w.r.t x_k =0.5*sum_j Ep_kj*(-Dx_kj/D) . Our Gx above = -sum Wmat*Dx = -sum Ep*Dx/D. That's 2x too big. So halve.
            Gx *= 0.5
            Gy *= 0.5
            # now dPen/dx=Gx, dPen/dy=Gy, dPen/dr_direct=sumEp
            dFdu = W * Gx * om2r
            dFdv = W * Gy * om2r
            dxdr = 1.0 - 2.0 * u
            dydr = 1.0 - 2.0 * vv
            dFdr = -np.ones(N) + W * (sumEp + Gx * dxdr + Gy * dydr)
            g = np.empty_like(zz)
            g[:N] = dFdu
            g[N:2 * N] = dFdv
            g[2 * N:] = dFdr
            return F, g
        return fun

    WS = [50.0, 500.0, 5000.0, 40000.0]
    MITS = [100, 120, 150, 200]

    def run_joint(z0):
        z = z0.copy()
        for si, W in enumerate(WS):
            if time.time() - t0 > DEADLINE - 12:
                break
            fun = make_fun(W)
            try:
                res = minimize(fun, z, jac=True, method='L-BFGS-B', bounds=bounds_box,
                               options={'maxiter': MITS[si], 'ftol': 1e-9, 'gtol': 1e-5, 'maxfun': 20000})
                z = res.x
            except Exception:
                break
            if time.time() - t0 > DEADLINE - 12:
                break
        return z

    def decode(z):
        u = z[:N]; vv = z[N:2 * N]; rr = z[2 * N:]
        om2r = 1.0 - 2.0 * rr
        x = u * om2r + rr
        y = vv * om2r + rr
        return x, y, rr

    # base grids for init
    # 5x5 grid + extra
    gx5 = []
    gy5 = []
    for ix in range(5):
        for jy in range(5):
            gx5.append((ix + 0.5) / 5.0)
            gy5.append((jy + 0.5) / 5.0)
    gx5 = np.array(gx5); gy5 = np.array(gy5)
    # 6x5 grid (30 pts)
    gx65 = []
    gy65 = []
    for ix in range(6):
        for jy in range(5):
            gx65.append((ix + 0.5) / 6.0)
            gy65.append((jy + 0.5) / 5.0)
    gx65 = np.array(gx65); gy65 = np.array(gy65)

    restart = 0
    # Phase A: joint restarts + LP
    while True:
        if time.time() - t0 > 72.0:
            break
        if restart > 80:
            break
        if time.time() - t0 > DEADLINE - 15:
            break
        # init u0,v0,r0
        mode = restart % 6
        if mode == 0 and restart < 30:
            # jittered 5x5 + 1 extra
            perm = rng.permutation(25)
            # take all 25 + extra random
            ux = gx5 + rng.randn(25) * 0.015
            uy = gy5 + rng.randn(25) * 0.015
            ex = rng.rand() * 0.8 + 0.1
            ey = rng.rand() * 0.8 + 0.1
            u0 = np.append(ux, ex)
            v0 = np.append(uy, ey)
            # shuffle order
            p = rng.permutation(N)
            # we have 26? 25+1=26 good
            u0 = np.clip(u0[p], 0.0, 1.0)
            v0 = np.clip(v0[p], 0.0, 1.0)
            r0 = rng.uniform(0.07, 0.11, size=N)
        elif mode == 1:
            idx = rng.choice(len(gx65), size=N, replace=False)
            u0 = np.clip(gx65[idx] + rng.randn(N) * 0.02, 0.0, 1.0)
            v0 = np.clip(gy65[idx] + rng.randn(N) * 0.02, 0.0, 1.0)
            r0 = rng.uniform(0.06, 0.12, size=N)
        elif mode == 2:
            u0 = rng.rand(N)
            v0 = rng.rand(N)
            r0 = rng.uniform(0.06, 0.12, size=N)
        elif mode == 3:
            u0 = rng.rand(N)
            v0 = rng.rand(N)
            r0 = np.full(N, 0.095) + rng.randn(N) * 0.015
            r0 = np.clip(r0, 0.03, 0.16)
        elif mode == 4:
            # current best perturbed (exploit)
            # convert best_x to u
            # u = (x - r)/(1-2r) using best_r clipped
            rc = np.clip(best_r, 0.015, 0.24)
            denom = 1.0 - 2.0 * rc
            denom = np.where(np.abs(denom) < 0.2, 0.5 * np.sign(denom + 1e-9), denom)
            ub = (best_x - rc) / denom
            vb = (best_y - rc) / denom
            u0 = np.clip(ub + rng.randn(N) * 0.05, 0.0, 1.0)
            v0 = np.clip(vb + rng.randn(N) * 0.05, 0.0, 1.0)
            r0 = np.clip(best_r + rng.randn(N) * 0.008, 0.015, 0.25)
        else:
            u0 = rng.rand(N)
            v0 = rng.rand(N)
            r0 = rng.uniform(0.04, 0.13, size=N)
        z0 = np.concatenate([u0, v0, r0])
        zopt = run_joint(z0)
        xh, yh, rh = decode(zopt)
        # LP polish
        try:
            r_opt, s_opt = radii_lp(np.clip(xh, 1e-6, 1 - 1e-6), np.clip(yh, 1e-6, 1 - 1e-6))
        except Exception:
            r_opt, s_opt = None, -1
        if r_opt is not None and s_opt > best_sum + 1e-12:
            best_sum = s_opt
            best_x = np.clip(xh, 1e-6, 1 - 1e-6).copy()
            best_y = np.clip(yh, 1e-6, 1 - 1e-6).copy()
            best_r = r_opt.copy()
            best_arr = np.column_stack([best_x, best_y, best_r])
        restart += 1

    # ---------- Phase B: LP-guided hill climbing on centres ----------
    # elite: keep best
    sig = 0.025
    it = 0
    # precompute for speed: reuse radii_lp
    # adaptive loop
    noimp = 0
    while time.time() - t0 < DEADLINE - 2.5:
        it += 1
        if it > 20000:
            break
        # choose 1 or 2 indices
        if rng.rand() < 0.75:
            idx = np.array([rng.randint(N)])
        else:
            idx = rng.choice(N, size=2, replace=False)
        # sigma schedule with occasional large jumps
        if rng.rand() < 0.05:
            s = sig * 4.0
        else:
            s = sig
        xt = best_x.copy()
        yt = best_y.copy()
        xt[idx] += rng.randn(len(idx)) * s
        yt[idx] += rng.randn(len(idx)) * s
        # keep inside
        xt = np.clip(xt, 0.003, 0.997)
        yt = np.clip(yt, 0.003, 0.997)
        # quick separation check: if any pair < 0.02 -> skip (would need tiny radii)
        # cheap check only for moved indices vs all
        ok_sep = True
        for ii in idx:
            dx = xt[ii] - xt
            dy = yt[ii] - yt
            dx[ii] = 1.0
            dy[ii] = 1.0
            if np.min(dx * dx + dy * dy) < (0.015 ** 2):
                ok_sep = False
                break
        if not ok_sep:
            noimp += 1
            continue
        try:
            r_opt, s_opt = radii_lp(xt, yt)
        except Exception:
            noimp += 1
            continue
        if r_opt is not None and s_opt > best_sum + 1e-10:
            best_sum = s_opt
            best_x, best_y, best_r = xt, yt, r_opt
            best_arr = np.column_stack([best_x, best_y, best_r])
            noimp = 0
            # slightly increase sig on success
            sig = min(0.04, sig * 1.02)
        else:
            noimp += 1
        if noimp % 250 == 0:
            sig *= 0.85
            if sig < 0.002:
                sig = 0.002
                # random restart kick from best with larger noise to escape
                if noimp > 1500:
                    # perturb all slightly and re-evaluate via joint? just shake best a bit
                    xt2 = np.clip(best_x + rng.randn(N) * 0.01, 0.003, 0.997)
                    yt2 = np.clip(best_y + rng.randn(N) * 0.01, 0.003, 0.997)
                    try:
                        r2, s2 = radii_lp(xt2, yt2)
                        if r2 is not None and s2 > best_sum - 0.02:
                            # allow sideways move to diversify if close
                            if rng.rand() < 0.3:
                                best_x, best_y = xt2, yt2
                                # keep best_r/sum unchanged unless improved
                                if s2 > best_sum:
                                    best_sum = s2
                                    best_r = r2
                                    best_arr = np.column_stack([best_x, best_y, best_r])
                    except Exception:
                        pass
                    noimp = 0
                    sig = 0.015
        # periodic time check already in while

    # ---------- final safety shrink ----------
    xs = best_arr[:, 0].copy()
    ys = best_arr[:, 1].copy()
    rs = best_arr[:, 2].copy()
    # ensure centres strictly inside after shrink: shrink radii a bit
    rs = rs * (1.0 - 1e-9) - 1e-12
    rs = np.maximum(rs, 1e-9)
    # re-clip centres if needed (should already satisfy with EPS margin)
    # if violation, shrink more
    for _ in range(5):
        wall_ok = np.all(xs >= rs - 1e-12) and np.all(xs <= 1 - rs + 1e-12) and np.all(ys >= rs - 1e-12) and np.all(ys <= 1 - rs + 1e-12)
        Dx = xs[:, None] - xs[None, :]
        Dy = ys[:, None] - ys[None, :]
        D = np.sqrt(Dx * Dx + Dy * Dy)
        np.fill_diagonal(D, 1e9)
        Rsum = rs[:, None] + rs[None, :]
        m = np.min(D - Rsum)
        if wall_ok and m >= 1e-12:
            break
        rs *= (1.0 - 1e-7)
        # also pull centres inside
        xs = np.minimum(np.maximum(xs, rs + 1e-10), 1.0 - rs - 1e-10)
        ys = np.minimum(np.maximum(ys, rs + 1e-10), 1.0 - rs - 1e-10)
    best_arr = np.column_stack([xs, ys, rs])
    return best_arr
