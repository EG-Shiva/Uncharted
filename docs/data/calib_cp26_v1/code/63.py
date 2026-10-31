import numpy as np
import time

def solve():
    import time as _tm
    import numpy as _np
    n = 26
    t0 = _tm.time()
    DEADLINE = 119.0
    T_PEN_END = 34.0  # penalty phase budget

    iu, ju = _np.triu_indices(n, 1)
    m = len(iu)
    A0 = _np.zeros((m, n), dtype=float)
    ar = _np.arange(m)
    A0[ar, iu] = 1.0
    A0[ar, ju] = 1.0
    c_obj = -_np.ones(n, dtype=float)

    try:
        from scipy.optimize import minimize, linprog
        has_opt = True
    except Exception:
        has_opt = False

    def eval_lp(xc, yc):
        # fast LP: max sum r s.t. r_i+r_j<=d_ij, 0<=r_i<=b_i
        try:
            b = _np.minimum(_np.minimum(xc, 1.0 - xc), _np.minimum(yc, 1.0 - yc))
            if _np.any(b < 0):
                return -1.0, None
            b = _np.maximum(b, 0.0)
            dx = xc[iu] - xc[ju]
            dy = yc[iu] - yc[ju]
            dvec = _np.sqrt(dx * dx + dy * dy)
            bounds = [(0.0, float(bi)) for bi in b]
            res = linprog(c_obj, A_ub=A0, b_ub=dvec, bounds=bounds, method='highs')
            if not res.success or res.x is None:
                return -1.0, None
            r = _np.asarray(res.x, dtype=float).flatten()
            r = _np.maximum(r, 0.0)
            s = float(_np.sum(r))
            if not _np.isfinite(s) or s <= 0:
                return -1.0, None
            return s, r
        except Exception:
            return -1.0, None

    # fallback grid
    k = int(_np.ceil(_np.sqrt(n)))
    rr = 0.5 / k
    fb = []
    for ix in range(k):
        for jy in range(k):
            if len(fb) < n:
                fb.append([(2 * ix + 1) * rr, (2 * jy + 1) * rr])
    fb = _np.array(fb, dtype=float)
    fxc = fb[:, 0].copy(); fyc = fb[:, 1].copy()

    best_sum = -1.0
    best_xc = fxc.copy(); best_yc = fyc.copy(); best_r = None
    s0, r0 = eval_lp(fxc, fyc)
    if s0 > 0:
        best_sum = s0; best_r = r0.copy()

    elite = []  # list of (sum, xc, yc, r)
    def add_elite(s, xc, yc, r):
        nonlocal elite
        if s is None or s <= 0 or r is None:
            return
        # distinctness check
        for (se, xe, ye, re) in elite:
            if abs(se - s) < 1e-9:
                if _np.max(_np.abs(xe - xc)) < 1e-4 and _np.max(_np.abs(ye - yc)) < 1e-4:
                    return
        elite.append((s, xc.copy(), yc.copy(), r.copy()))
        elite.sort(key=lambda t: -t[0])
        elite[:] = elite[:10]

    if best_sum > 0:
        add_elite(best_sum, best_xc, best_yc, best_r)

    rng = _np.random.RandomState(0)

    # ---------- Phase A: short joint penalized L-BFGS-B to get incumbent ----------
    if has_opt:
        bounds78 = [(0.0, 1.0)] * (2 * n) + [(0.005, 0.20)] * n
        Ws = [50.0, 500.0, 5000.0, 50000.0]
        maxiters = [60, 60, 80, 100]
        def make_init(rid):
            if rid == 0:
                xs = []; ys = []
                for ix in range(5):
                    for jy in range(5):
                        xs.append((ix + 0.5) / 5.0); ys.append((jy + 0.5) / 5.0)
                xs.append(0.5); ys.append(0.5)
                xc = _np.array(xs, float) + rng.normal(0, 0.02, n)
                yc = _np.array(ys, float) + rng.normal(0, 0.02, n)
                xc = _np.clip(xc, 0.02, 0.98); yc = _np.clip(yc, 0.02, 0.98)
                R0 = 0.09 + 0.02 * rng.rand(n)
                return xc, yc, R0
            elif rid == 1:
                xc = rng.uniform(0.05, 0.95, n)
                yc = rng.uniform(0.05, 0.95, n)
                R0 = 0.08 + 0.04 * rng.rand(n)
                return xc, yc, R0
            else:
                xc = rng.uniform(0.03, 0.97, n)
                yc = rng.uniform(0.03, 0.97, n)
                R0 = 0.06 + 0.05 * rng.rand(n)
                return xc, yc, R0
        restart = 0
        while _tm.time() - t0 < T_PEN_END and restart < 40:
            try:
                xc0, yc0, R00 = make_init(restart)
                v = _np.concatenate([xc0, yc0, R00])
                v = _np.clip(v, 0.0, 1.0)
                v[2 * n:] = _np.clip(v[2 * n:], 0.005, 0.20)
                aborted = False
                for wi, W in enumerate(Ws):
                    if _tm.time() - t0 >= T_PEN_END:
                        aborted = True
                        break
                    def fun(vv, W=W):
                        X = vv[0:n]; Y = vv[n:2 * n]; R = vv[2 * n:3 * n]
                        pen = 0.0
                        gx = _np.zeros(n); gy = _np.zeros(n); gr_pen = _np.zeros(n)
                        v1 = R - X
                        m1 = v1 > 0
                        if _np.any(m1):
                            pen += _np.sum(v1[m1] * v1[m1])
                            gx[m1] += -2.0 * v1[m1]
                            gr_pen[m1] += 2.0 * v1[m1]
                        v2 = X + R - 1.0
                        m2 = v2 > 0
                        if _np.any(m2):
                            pen += _np.sum(v2[m2] * v2[m2])
                            gx[m2] += 2.0 * v2[m2]
                            gr_pen[m2] += 2.0 * v2[m2]
                        v3 = R - Y
                        m3 = v3 > 0
                        if _np.any(m3):
                            pen += _np.sum(v3[m3] * v3[m3])
                            gy[m3] += -2.0 * v3[m3]
                            gr_pen[m3] += 2.0 * v3[m3]
                        v4 = Y + R - 1.0
                        m4 = v4 > 0
                        if _np.any(m4):
                            pen += _np.sum(v4[m4] * v4[m4])
                            gy[m4] += 2.0 * v4[m4]
                            gr_pen[m4] += 2.0 * v4[m4]
                        Dx = X[:, None] - X[None, :]
                        Dy = Y[:, None] - Y[None, :]
                        D = _np.sqrt(Dx * Dx + Dy * Dy + 1e-16)
                        S = R[:, None] + R[None, :]
                        V = S - D
                        _np.fill_diagonal(V, 0.0)
                        Vp = _np.maximum(V, 0.0)
                        _np.fill_diagonal(Vp, 0.0)
                        pen += 0.25 * _np.sum(Vp * Vp)
                        _np.fill_diagonal(D, 1.0)
                        Q = Vp / D
                        gx += -_np.sum(Q * Dx, axis=1)
                        gy += -_np.sum(Q * Dy, axis=1)
                        gr_pen += _np.sum(Vp, axis=1)
                        loss = -_np.sum(R) + W * pen
                        g = _np.empty_like(vv)
                        g[0:n] = W * gx
                        g[n:2 * n] = W * gy
                        g[2 * n:3 * n] = -1.0 + W * gr_pen
                        return loss, g
                    try:
                        res = minimize(fun, v, method='L-BFGS-B', jac=True, bounds=bounds78,
                                       options={'maxiter': maxiters[wi], 'ftol': 1e-9, 'gtol': 1e-5})
                        v = res.x
                    except Exception:
                        aborted = True
                        break
                if not aborted:
                    X = _np.clip(v[0:n], 0.0, 1.0); Y = _np.clip(v[n:2 * n], 0.0, 1.0)
                    s_lp, r_lp = eval_lp(_np.clip(X, 0.002, 0.998), _np.clip(Y, 0.002, 0.998))
                    if s_lp is not None and s_lp > 0:
                        Xc = _np.clip(X, 0.002, 0.998); Yc = _np.clip(Y, 0.002, 0.998)
                        # re-eval to match clipped
                        if s_lp > best_sum:
                            best_sum = s_lp; best_xc = Xc.copy(); best_yc = Yc.copy(); best_r = r_lp.copy()
                        add_elite(s_lp, Xc, Yc, r_lp)
            except Exception:
                pass
            restart += 1

    # ensure best from elite
    if len(elite) > 0:
        elite.sort(key=lambda t: -t[0])
        if elite[0][0] > best_sum:
            best_sum, best_xc, best_yc, best_r = elite[0][0], elite[0][1].copy(), elite[0][2].copy(), elite[0][3].copy()

    # ---------- Phase B: centre-only basin-hopping with topology moves ----------
    rng2 = _np.random.RandomState(1)
    # 41x41 grid coordinates
    G1 = _np.linspace(0.005, 0.995, 41)
    GXm, GYm = _np.meshgrid(G1, G1)
    GX = GXm.flatten(); GY = GYm.flatten()
    NG = GX.shape[0]
    GWALL = _np.minimum(_np.minimum(GX, 1.0 - GX), _np.minimum(GY, 1.0 - GY))

    cur_xc = best_xc.copy(); cur_yc = best_yc.copy()
    cur_sum = best_sum; cur_r = best_r.copy() if best_r is not None else None

    t_basin = _tm.time()
    # temperature schedule 0.002 -> 0.00005
    T_START = 0.002; T_END = 0.00005

    it = 0
    noimp = 0

    def temp_now():
        now = _tm.time()
        total = max(1.0, (DEADLINE - 4.0) - t_basin)
        prog = (now - t_basin) / total
        prog = min(1.0, max(0.0, prog))
        return T_START * ((T_END / T_START) ** prog)

    # helper: greedy reinsert of indices on 41x41 grid (budgeted)
    def ruin_reinsert_3(cxc, cyc, cs, cr):
        # find 3 smallest by radius
        try:
            order = _np.argsort(cr)
            rem = order[:3].tolist()
        except Exception:
            rem = [int(rng2.randint(n)) for _ in range(3)]
        txc = cxc.copy(); tyc = cyc.copy()
        # LP of current as baseline
        best_s_local, best_r_local = cs, cr
        for kk_idx, kk in enumerate(rem):
            if _tm.time() - t0 > DEADLINE - 6.0:
                break
            # vectorized heuristic to order candidates
            others = [i for i in range(n) if i != kk]
            ox = txc[others]; oy = tyc[others]
            # distances grid to others
            # chunked to avoid huge memory: 1681x25 fine
            dx = GX[:, None] - ox[None, :]
            dy = GY[:, None] - oy[None, :]
            dmin = _np.sqrt(dx * dx + dy * dy).min(axis=1)
            heur = _np.minimum(GWALL, dmin * 0.5)
            # order descending, take top budget
            budget = 220
            ord_idx = _np.argsort(-heur)
            # always include current position implicitly via best tracking
            cand_best_s = -1.0; cand_best_x = txc[kk]; cand_best_y = tyc[kk]; cand_best_r = None
            # evaluate in heuristic order
            cnt = 0
            for gi in ord_idx:
                if cnt >= budget:
                    break
                # quick prune: if wall tiny and heur tiny skip tail?
                # still evaluate top budget only
                gx = float(GX[gi]); gy = float(GY[gi])
                # skip if essentially duplicate of another evaluated? no
                txc[kk] = gx; tyc[kk] = gy
                # fast reject if wall<1e-4 (infeasible)
                if gx < 0.001 or gx > 0.999 or gy < 0.001 or gy > 0.999:
                    continue
                s, r = eval_lp(txc, tyc)
                cnt += 1
                if _tm.time() - t0 > DEADLINE - 6.0:
                    break
                if s > cand_best_s:
                    cand_best_s = s; cand_best_x = gx; cand_best_y = gy; cand_best_r = r
                    # early exit if excellent? no
            if cand_best_s > 0:
                txc[kk] = cand_best_x; tyc[kk] = cand_best_y
                best_s_local, best_r_local = cand_best_s, cand_best_r
            else:
                # revert this kk
                txc[kk] = cxc[kk]; tyc[kk] = cyc[kk]
        # final eval to sync
        fs, fr = eval_lp(txc, tyc)
        if fs < 0:
            return cs, cxc.copy(), cyc.copy(), cr
        return fs, txc, tyc, fr

    def largest_void_point(cxc, cyc, cr, exclude=-1):
        # dense grid search for largest empty region ignoring exclude
        try:
            idx = [i for i in range(n) if i != exclude]
            ox = cxc[idx]; oy = cyc[idx]
            # use current radii to estimate clearance: min(wall, min(dist - r_i))
            # if dist-r negative, clearance negative
            if cr is not None:
                orad = cr[idx]
            else:
                orad = _np.zeros(len(idx))
            dx = GX[:, None] - ox[None, :]
            dy = GY[:, None] - oy[None, :]
            dist = _np.sqrt(dx * dx + dy * dy)
            clear = dist - orad[None, :]
            cmin = clear.min(axis=1)
            score = _np.minimum(GWALL, cmin)
            bi = int(_np.argmax(score))
            return float(GX[bi]), float(GY[bi]), float(score[bi])
        except Exception:
            return 0.5, 0.5, 0.0

    # basin loop
    while _tm.time() - t0 < DEADLINE - 4.0:
        it += 1
        T = temp_now()
        u = rng2.rand()
        # time guard for expensive moves
        time_left = (DEADLINE - 4.0) - (_tm.time() - t0)
        # choose move
        try:
            if u < 0.50:
                # single-circle translation
                ch = rng2.rand()
                if ch < 0.3:
                    sigma = 0.03
                elif ch < 0.7:
                    sigma = 0.01
                else:
                    sigma = 0.003
                i = int(rng2.randint(n))
                txc = cur_xc.copy(); tyc = cur_yc.copy()
                txc[i] = float(_np.clip(cur_xc[i] + rng2.normal(0, sigma), 0.002, 0.998))
                tyc[i] = float(_np.clip(cur_yc[i] + rng2.normal(0, sigma), 0.002, 0.998))
                s, r = eval_lp(txc, tyc)
                if s < 0:
                    noimp += 1
                    continue
                delta = s - cur_sum
                if delta >= 0 or rng2.rand() < _np.exp(delta / max(T, 1e-9)):
                    cur_xc, cur_yc, cur_sum, cur_r = txc, tyc, s, r
                    add_elite(s, txc, tyc, r)
                    if s > best_sum:
                        best_sum = s; best_xc = txc.copy(); best_yc = tyc.copy(); best_r = r.copy()
                        noimp = 0
                    else:
                        noimp += 1 if delta < 0 else 0
                else:
                    noimp += 1
            elif u < 0.64:
                # pair translation
                ch = rng2.rand()
                sigma = 0.01 if ch < 0.5 else (0.03 if ch < 0.75 else 0.003)
                i = int(rng2.randint(n)); j = int(rng2.randint(n))
                while j == i:
                    j = int(rng2.randint(n))
                txc = cur_xc.copy(); tyc = cur_yc.copy()
                txc[i] = float(_np.clip(cur_xc[i] + rng2.normal(0, sigma), 0.002, 0.998))
                tyc[i] = float(_np.clip(cur_yc[i] + rng2.normal(0, sigma), 0.002, 0.998))
                txc[j] = float(_np.clip(cur_xc[j] + rng2.normal(0, sigma), 0.002, 0.998))
                tyc[j] = float(_np.clip(cur_yc[j] + rng2.normal(0, sigma), 0.002, 0.998))
                s, r = eval_lp(txc, tyc)
                if s < 0:
                    noimp += 1
                    continue
                delta = s - cur_sum
                if delta >= 0 or rng2.rand() < _np.exp(delta / max(T, 1e-9)):
                    cur_xc, cur_yc, cur_sum, cur_r = txc, tyc, s, r
                    add_elite(s, txc, tyc, r)
                    if s > best_sum:
                        best_sum = s; best_xc = txc.copy(); best_yc = tyc.copy(); best_r = r.copy()
                        noimp = 0
                    else:
                        noimp += 1 if delta < 0 else 0
                else:
                    noimp += 1
            elif u < 0.74:
                # swap large-small positions
                if cur_r is None:
                    continue
                order = _np.argsort(cur_r)
                small_pool = order[:8].tolist()
                large_pool = order[-8:].tolist()
                si = small_pool[int(rng2.randint(len(small_pool)))]
                li = large_pool[int(rng2.randint(len(large_pool)))]
                txc = cur_xc.copy(); tyc = cur_yc.copy()
                txc[si], txc[li] = float(cur_xc[li]), float(cur_xc[si])
                tyc[si], tyc[li] = float(cur_yc[li]), float(cur_yc[si])
                # small jitter after swap to break exact symmetry
                txc[si] = float(_np.clip(txc[si] + rng2.normal(0, 0.003), 0.002, 0.998))
                tyc[si] = float(_np.clip(tyc[si] + rng2.normal(0, 0.003), 0.002, 0.998))
                txc[li] = float(_np.clip(txc[li] + rng2.normal(0, 0.003), 0.002, 0.998))
                tyc[li] = float(_np.clip(tyc[li] + rng2.normal(0, 0.003), 0.002, 0.998))
                s, r = eval_lp(txc, tyc)
                if s < 0:
                    noimp += 1
                    continue
                delta = s - cur_sum
                if delta >= 0 or rng2.rand() < _np.exp(delta / max(T, 1e-9)):
                    cur_xc, cur_yc, cur_sum, cur_r = txc, tyc, s, r
                    add_elite(s, txc, tyc, r)
                    if s > best_sum:
                        best_sum = s; best_xc = txc.copy(); best_yc = tyc.copy(); best_r = r.copy()
                        noimp = 0
                    else:
                        noimp += 1 if delta < 0 else 0
                else:
                    noimp += 1
            elif u < 0.82:
                # ruin-recreate 3 smallest via 41x41 greedy (expensive)
                if time_left < 12.0:
                    # fallback to single move if little time
                    continue
                if cur_r is None:
                    continue
                fs, fxc2, fyc2, fr = ruin_reinsert_3(cur_xc, cur_yc, cur_sum, cur_r)
                if fs < 0:
                    noimp += 1
                    continue
                delta = fs - cur_sum
                if delta >= 0 or rng2.rand() < _np.exp(delta / max(T, 1e-9)):
                    cur_xc, cur_yc, cur_sum, cur_r = fxc2, fyc2, fs, fr
                    add_elite(fs, fxc2, fyc2, fr)
                    if fs > best_sum:
                        best_sum = fs; best_xc = fxc2.copy(); best_yc = fyc2.copy(); best_r = fr.copy()
                        noimp = 0
                    else:
                        noimp += 1 if delta < 0 else 0
                else:
                    noimp += 1
            elif u < 0.90:
                # relocate one large circle into largest empty region
                if cur_r is None:
                    continue
                order = _np.argsort(cur_r)
                li = order[-6:][int(rng2.randint(6))]
                vx, vy, _ = largest_void_point(cur_xc, cur_yc, cur_r, exclude=li)
                txc = cur_xc.copy(); tyc = cur_yc.copy()
                # small jitter around void centre
                txc[li] = float(_np.clip(vx + rng2.normal(0, 0.002), 0.002, 0.998))
                tyc[li] = float(_np.clip(vy + rng2.normal(0, 0.002), 0.002, 0.998))
                s, r = eval_lp(txc, tyc)
                if s < 0:
                    noimp += 1
                    continue
                delta = s - cur_sum
                if delta >= 0 or rng2.rand() < _np.exp(delta / max(T, 1e-9)):
                    cur_xc, cur_yc, cur_sum, cur_r = txc, tyc, s, r
                    add_elite(s, txc, tyc, r)
                    if s > best_sum:
                        best_sum = s; best_xc = txc.copy(); best_yc = tyc.copy(); best_r = r.copy()
                        noimp = 0
                    else:
                        noimp += 1 if delta < 0 else 0
                else:
                    noimp += 1
            else:
                # reflect/rotate whole set by multiples of 90 deg
                ksym = int(rng2.randint(8))
                txc = cur_xc.copy(); tyc = cur_yc.copy()
                x = cur_xc; y = cur_yc
                if ksym == 0:
                    txc, tyc = 1.0 - x, y
                elif ksym == 1:
                    txc, tyc = x, 1.0 - y
                elif ksym == 2:
                    txc, tyc = 1.0 - x, 1.0 - y
                elif ksym == 3:
                    txc, tyc = y, x
                elif ksym == 4:
                    txc, tyc = 1.0 - y, x
                elif ksym == 5:
                    txc, tyc = y, 1.0 - x
                elif ksym == 6:
                    txc, tyc = 1.0 - y, 1.0 - x
                else:
                    # random small global jitter instead
                    txc = _np.clip(x + rng2.normal(0, 0.004, n), 0.002, 0.998)
                    tyc = _np.clip(y + rng2.normal(0, 0.004, n), 0.002, 0.998)
                s, r = eval_lp(txc, tyc)
                if s < 0:
                    noimp += 1
                    continue
                # symmetry preserves score; accept to diversify
                delta = s - cur_sum
                if delta >= -1e-9 or rng2.rand() < 0.3:
                    cur_xc, cur_yc, cur_sum, cur_r = txc, tyc, s, r
                    add_elite(s, txc, tyc, r)
                    if s > best_sum + 1e-12:
                        best_sum = s; best_xc = txc.copy(); best_yc = tyc.copy(); best_r = r.copy()
                        noimp = 0
            # elite restart if stuck
            if noimp > 600:
                if len(elite) > 1 and rng2.rand() < 0.6:
                    pi = int(rng2.randint(len(elite)))
                    cur_xc = elite[pi][1].copy() + rng2.normal(0, 0.008, n)
                    cur_yc = elite[pi][2].copy() + rng2.normal(0, 0.008, n)
                    cur_xc = _np.clip(cur_xc, 0.002, 0.998)
                    cur_yc = _np.clip(cur_yc, 0.002, 0.998)
                    s, r = eval_lp(cur_xc, cur_yc)
                    if s > 0:
                        cur_sum, cur_r = s, r
                        if s > best_sum:
                            best_sum = s; best_xc = cur_xc.copy(); best_yc = cur_yc.copy(); best_r = r.copy()
                        add_elite(s, cur_xc, cur_yc, r)
                else:
                    # return to best with kick
                    cur_xc = best_xc.copy() + rng2.normal(0, 0.005, n)
                    cur_yc = best_yc.copy() + rng2.normal(0, 0.005, n)
                    cur_xc = _np.clip(cur_xc, 0.002, 0.998)
                    cur_yc = _np.clip(cur_yc, 0.002, 0.998)
                    s, r = eval_lp(cur_xc, cur_yc)
                    if s > 0:
                        cur_sum, cur_r = s, r
                noimp = 0
        except Exception:
            # never crash basin loop
            try:
                noimp += 1
                if _tm.time() - t0 > DEADLINE - 4.0:
                    break
            except Exception:
                break

    # ---------- final: pick best elite, shrink for strict feasibility ----------
    try:
        if len(elite) > 0:
            elite.sort(key=lambda t: -t[0])
            best_sum, best_xc, best_yc, best_r = elite[0][0], elite[0][1].copy(), elite[0][2].copy(), elite[0][3].copy()
        # re-evaluate to be safe
        sfin, rfin = eval_lp(best_xc, best_yc)
        if sfin > 0:
            best_sum, best_r = sfin, rfin
        else:
            # keep old
            pass
        X = _np.clip(best_xc, 0.0, 1.0); Y = _np.clip(best_yc, 0.0, 1.0)
        R = _np.maximum(_np.asarray(best_r, dtype=float).flatten(), 1e-12)
        # compute max feasible scale
        wall = _np.minimum(_np.minimum(X, 1.0 - X), _np.minimum(Y, 1.0 - Y))
        # avoid division by zero
        with _np.errstate(divide='ignore', invalid='ignore'):
            ratio_w = wall / R
        a_w = float(_np.min(ratio_w))
        Dx = X[:, None] - X[None, :]
        Dy = Y[:, None] - Y[None, :]
        D = _np.sqrt(Dx * Dx + Dy * Dy)
        S = R[:, None] + R[None, :]
        with _np.errstate(divide='ignore', invalid='ignore'):
            ratio_p = D / S
        _np.fill_diagonal(ratio_p, _np.inf)
        a_p = float(_np.min(ratio_p))
        alpha = min(1.0, a_w, a_p)
        # strict shrink
        alpha = alpha * (1.0 - 1e-7) - 1e-12
        if not _np.isfinite(alpha) or alpha <= 0:
            alpha = 0.99
        R = R * alpha
        # final safety: ensure inside and non-overlap with margin
        # clip tiny
        R = _np.maximum(R, 1e-9)
        out = _np.column_stack([X, Y, R])
        return out
    except Exception:
        # ultimate fallback: feasible grid
        try:
            k2 = int(_np.ceil(_np.sqrt(n)))
            rr2 = 0.5 / k2
            pts = []
            for ix in range(k2):
                for jy in range(k2):
                    if len(pts) < n:
                        pts.append([(2 * ix + 1) * rr2, (2 * jy + 1) * rr2, rr2 * 0.999 * (1 - 1e-7)])
            return _np.array(pts, dtype=float)
        except Exception:
            return _np.array([_np.array([0.1, 0.1, 0.05]) for _ in range(26)], dtype=float)
