import numpy as np
import time

def solve():
    n = 26
    t0 = time.time()
    DEADLINE = 126.0
    rng = np.random.default_rng(0)

    from scipy.optimize import linprog

    I, J = np.triu_indices(n, 1)
    m = len(I)
    A0 = np.zeros((m, n), dtype=float)
    A0[np.arange(m), I] = 1.0
    A0[np.arange(m), J] = 1.0
    c = -np.ones(n, dtype=float)

    def lp_eval(xc, yc, best_s=-1.0):
        # wall
        w = np.minimum(np.minimum(xc, 1.0 - xc), np.minimum(yc, 1.0 - yc))
        if np.any(w < 0):
            return -1.0, None
        # prune by wall sum
        sw = float(np.sum(w))
        if sw <= best_s:
            return -1.0, None
        dx = xc[I] - xc[J]
        dy = yc[I] - yc[J]
        d = np.sqrt(dx * dx + dy * dy)
        # prune: sum r <= 0.5*sum(d_ij / (n-1))? classic bound: (n-1)*sum r <= sum d
        # sum r <= sum(d)/ (n-1) ??? since each r appears in (n-1) pairs: sum_{i<j}(r_i+r_j)=(n-1)sum r <= sum d
        sd = float(np.sum(d))
        if sd / float(n - 1) <= best_s:
            return -1.0, None
        bnds = [(0.0, float(wi)) for wi in w]
        try:
            res = linprog(c, A_ub=A0, b_ub=d, bounds=bnds, method='highs')
        except Exception:
            return -1.0, None
        if res.status != 0:
            return -1.0, None
        r = np.asarray(res.x, dtype=float)
        r = np.maximum(r, 0.0)
        s = float(np.sum(r))
        if s <= best_s:
            return s, None
        return s, r

    # fallback grid
    k = int(np.ceil(np.sqrt(n)))
    rr0 = 0.5 / k
    fb_x = np.zeros(n); fb_y = np.zeros(n)
    idx = 0
    for ix in range(k):
        for jy in range(k):
            if idx < n:
                fb_x[idx] = (2*ix+1)*rr0
                fb_y[idx] = (2*jy+1)*rr0
                idx += 1

    best_s, best_r = lp_eval(fb_x, fb_y, -1.0)
    if best_r is None:
        best_s = -1.0
        best_xc = fb_x.copy(); best_yc = fb_y.copy(); best_r = np.full(n, 0.05)
    else:
        best_xc = fb_x.copy(); best_yc = fb_y.copy(); best_r = best_r.copy()

    # diverse initial pool
    def random_row_init():
        nrows = int(rng.integers(5, 7))  # 5 or 6
        base = n // nrows
        rem = n - base * nrows
        counts = [base]*nrows
        for kk in range(rem):
            counts[int(rng.integers(0, nrows))] += 1
        # fix sum in case of rounding (should already sum)
        # adjust if needed
        while sum(counts) < n:
            counts[int(rng.integers(0, nrows))] += 1
        while sum(counts) > n:
            ii = int(rng.integers(0, nrows))
            if counts[ii] > 3:
                counts[ii] -= 1
        xc = np.zeros(n); yc = np.zeros(n)
        p = 0
        for r_ in range(nrows):
            y0 = (r_+0.5)/nrows
            cnt = counts[r_]
            for l in range(cnt):
                x0 = (l+0.5)/cnt
                xc[p] = x0 + rng.normal(0, 0.015)
                yc[p] = y0 + rng.normal(0, 0.015)
                p += 1
        xc = np.clip(xc, 0.02, 0.98); yc = np.clip(yc, 0.02, 0.98)
        # random transpose half the time for column stratification
        if rng.random() < 0.5:
            xc, yc = yc.copy(), xc.copy()
        return xc, yc

    n_init = 220
    for trial in range(n_init):
        if time.time() - t0 > 18.0:
            break
        rtype = trial % 5
        if rtype == 0:
            xc = rng.uniform(0.03, 0.97, n)
            yc = rng.uniform(0.03, 0.97, n)
        elif rtype == 1:
            xc = fb_x + rng.normal(0, 0.02, n)
            yc = fb_y + rng.normal(0, 0.02, n)
            xc = np.clip(xc, 0.02, 0.98); yc = np.clip(yc, 0.02, 0.98)
        elif rtype == 2:
            # hex jitter
            xc = np.zeros(n); yc = np.zeros(n)
            pts = []
            for ix in range(6):
                for jy in range(5):
                    xx = (ix+0.5)/6.0 + (0.04 if (jy % 2 == 1) else 0.0)
                    if xx > 0.98: xx -= 0.08
                    yy = (jy+0.5)/5.0
                    pts.append((xx, yy))
            pts = np.array(pts)
            perm = rng.permutation(len(pts))[:n]
            for kk in range(n):
                if kk < len(perm):
                    xc[kk], yc[kk] = pts[perm[kk]]
                else:
                    xc[kk] = rng.uniform(0.05, 0.95); yc[kk] = rng.uniform(0.05, 0.95)
            xc = xc + rng.normal(0, 0.012, n); yc = yc + rng.normal(0, 0.012, n)
            xc = np.clip(xc, 0.02, 0.98); yc = np.clip(yc, 0.02, 0.98)
        else:
            xc, yc = random_row_init()
        s, r_ = lp_eval(xc, yc, best_s)
        if r_ is not None and s > best_s:
            best_s = s
            best_xc = xc.copy(); best_yc = yc.copy(); best_r = r_.copy()

    # hill climbing with ruin-and-recreate
    sigma = 0.022
    stagnant = 0
    it = 0
    # preallocate candidates for void search
    while time.time() - t0 < DEADLINE - 3.0:
        it += 1
        # periodic void relocation of worst circle
        if it % 220 == 0:
            try:
                iw = int(np.argmin(best_r))
                # find void: sample random points, score by clearance
                ncand = 260
                candx = rng.uniform(0.0, 1.0, ncand)
                candy = rng.uniform(0.0, 1.0, ncand)
                # distances to other centres
                ox = best_xc[np.arange(n) != iw]
                oy = best_yc[np.arange(n) != iw]
                orad = best_r[np.arange(n) != iw]
                # compute clearance
                best_c = -1.0; bx = 0.5; by = 0.5
                for cc in range(ncand):
                    dxv = ox - candx[cc]; dyv = oy - candy[cc]
                    dd = np.sqrt(dxv*dxv + dyv*dyv) - orad
                    wallc = min(candx[cc], 1.0-candx[cc], candy[cc], 1.0-candy[cc])
                    clr = min(float(np.min(dd)), wallc)
                    if clr > best_c:
                        best_c = clr; bx = candx[cc]; by = candy[cc]
                # propose move
                px = best_xc.copy(); py = best_yc.copy()
                px[iw] = np.clip(bx, 0.01, 0.99); py[iw] = np.clip(by, 0.01, 0.99)
                s2, r2 = lp_eval(px, py, best_s)
                if r2 is not None and s2 > best_s:
                    best_s = s2; best_xc = px; best_yc = py; best_r = r2
                    stagnant = 0
                    continue
            except Exception:
                pass

        # occasional global jitter
        if stagnant > 900 and (it % 50 == 0):
            # big kick: jitter all
            px = best_xc + rng.normal(0, 0.015, n)
            py = best_yc + rng.normal(0, 0.015, n)
            px = np.clip(px, 0.01, 0.99); py = np.clip(py, 0.01, 0.99)
            s2, r2 = lp_eval(px, py, best_s)
            if r2 is not None and s2 > best_s:
                best_s = s2; best_xc = px; best_yc = py; best_r = r2
                stagnant = 0
                sigma = max(0.008, sigma*0.95)
            else:
                # relocate 2 smallest to random voids
                try:
                    order = np.argsort(best_r)
                    px2 = best_xc.copy(); py2 = best_yc.copy()
                    for qq in range(2):
                        iw2 = int(order[qq])
                        px2[iw2] = rng.uniform(0.05, 0.95); py2[iw2] = rng.uniform(0.05, 0.95)
                    s3, r3 = lp_eval(px2, py2, best_s - 0.05)
                    # accept even if slightly worse to diversify? only if not too bad
                    if r3 is not None and s3 > best_s - 0.03:
                        # hill climb from there a bit is handled by continuing; accept as new base if improves else keep but perturb
                        if s3 > best_s:
                            best_s = s3; best_xc = px2; best_yc = py2; best_r = r3
                        else:
                            # random restart base: keep best but reduce sigma?
                            pass
                        stagnant = 0
                        sigma = 0.018
                except Exception:
                    pass
            continue

        # single-circle Gaussian move
        i = int(rng.integers(0, n))
        # bias to small circles more often
        if rng.random() < 0.30:
            i = int(np.argmin(best_r + 0.01*rng.random(n)))
        px = best_xc.copy(); py = best_yc.copy()
        px[i] += rng.normal(0, sigma)
        py[i] += rng.normal(0, sigma)
        # clip
        if px[i] < 0.005: px[i] = 0.005
        if px[i] > 0.995: px[i] = 0.995
        if py[i] < 0.005: py[i] = 0.005
        if py[i] > 0.995: py[i] = 0.995
        s2, r2 = lp_eval(px, py, best_s)
        if r2 is not None and s2 > best_s + 1e-12:
            best_s = s2; best_xc = px; best_yc = py; best_r = r2
            stagnant = 0
            # slowly grow sigma on success
            # sigma = min(0.025, sigma*1.001)
        else:
            stagnant += 1
            if stagnant % 400 == 0:
                sigma *= 0.88
                if sigma < 0.0012:
                    sigma = 0.0012
                    # force void attempt next loop by resetting counter
                    stagnant = 950

    # final safety shrink
    Rf = np.maximum(np.asarray(best_r, dtype=float), 1e-12) * (1.0 - 1e-7)
    # extra absolute guard
    Rf = np.maximum(Rf - 1e-12, 1e-9)
    Xf = np.clip(np.asarray(best_xc, dtype=float), 0.0, 1.0)
    Yf = np.clip(np.asarray(best_yc, dtype=float), 0.0, 1.0)
    # uniform rescale if any violation due to solver tolerance
    wall = np.minimum(np.minimum(Xf, 1.0-Xf), np.minimum(Yf, 1.0-Yf))
    Dx = Xf[:, None]-Xf[None, :]
    Dy = Yf[:, None]-Yf[None, :]
    D = np.sqrt(Dx*Dx+Dy*Dy)
    S = Rf[:, None]+Rf[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        rw = wall / Rf
        rp = D / S
    np.fill_diagonal(rp, np.inf)
    alpha = min(1.0, float(np.min(rw)), float(np.min(rp)))
    alpha = alpha*(1.0-1e-9)-1e-12
    if alpha < 1.0 and alpha > 0 and np.isfinite(alpha):
        Rf = Rf*alpha
    arr = np.column_stack([Xf, Yf, Rf])
    return arr
