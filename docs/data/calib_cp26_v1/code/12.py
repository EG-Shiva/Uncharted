import numpy as np
import time

def solve():
    n = 26
    t0 = time.time()
    HARD = 124.0
    SEARCH = 112.0
    rng = np.random.RandomState(0)
    try:
        from scipy.optimize import linprog
        has_lp = True
    except Exception:
        has_lp = False

    if not has_lp:
        k = int(np.ceil(np.sqrt(n)))
        r = 0.5 / k
        pts = []
        for i in range(k):
            for j in range(k):
                if len(pts) < n:
                    pts.append([(2*i+1)*r, (2*j+1)*r, r*0.999999])
        return np.array(pts)

    I, J = np.triu_indices(n, 1)
    m = len(I)
    A0 = np.zeros((m, n), dtype=float)
    A0[np.arange(m), I] = 1.0
    A0[np.arange(m), J] = 1.0
    c = -np.ones(n, dtype=float)

    def lp_eval(xc, yc):
        b = np.minimum(np.minimum(xc, 1.0 - xc), np.minimum(yc, 1.0 - yc))
        if np.any(b <= 1e-12):
            return -1.0, None
        d = np.sqrt((xc[I] - xc[J]) ** 2 + (yc[I] - yc[J]) ** 2)
        # avoid exact zero causing issues: keep as is, linprog handles
        bnds = [(0.0, float(bi)) for bi in b]
        try:
            res = linprog(c, A_ub=A0, b_ub=d, bounds=bnds, method='highs')
        except Exception:
            return -1.0, None
        if res.status != 0:
            return -1.0, None
        r = np.asarray(res.x, dtype=float)
        r = np.maximum(r, 0.0)
        s = float(np.sum(r))
        return s, r

    def make_init(rid):
        if rid == 0:
            # 5x5 grid + extra
            xs = []; ys = []
            for ix in range(5):
                for jy in range(5):
                    xs.append((ix + 0.5) / 5.0)
                    ys.append((jy + 0.5) / 5.0)
            xs.append(0.5); ys.append(0.5)
            xc = np.array(xs, dtype=float) + rng.normal(0, 0.015, n)
            yc = np.array(ys, dtype=float) + rng.normal(0, 0.015, n)
            xc = np.clip(xc, 0.02, 0.98)
            yc = np.clip(yc, 0.02, 0.98)
            return xc, yc
        elif rid == 1:
            # hex-ish
            xc = rng.uniform(0.05, 0.95, n)
            yc = rng.uniform(0.05, 0.95, n)
            return xc, yc
        else:
            xc = rng.uniform(0.03, 0.97, n)
            yc = rng.uniform(0.03, 0.97, n)
            return xc, yc

    best_sum = -1.0
    best_xc = None; best_yc = None; best_r = None

    # initial eval to have something
    xc0, yc0 = make_init(0)
    s0, r0 = lp_eval(xc0, yc0)
    if s0 > 0:
        best_sum, best_xc, best_yc, best_r = s0, xc0.copy(), yc0.copy(), r0.copy()

    rid = 0
    # outer restarts
    while time.time() - t0 < SEARCH:
        if rid == 0 and best_xc is not None and rid == 0:
            # reuse first init already evaluated as start for hill
            cur_xc, cur_yc = xc0.copy(), yc0.copy()
            cur_sum, cur_r = s0, r0.copy() if r0 is not None else (s0, None)
            if cur_sum < 0:
                cur_xc, cur_yc = make_init(99)
                cur_sum, cur_r = lp_eval(cur_xc, cur_yc)
        else:
            cur_xc, cur_yc = make_init(rid)
            cur_sum, cur_r = lp_eval(cur_xc, cur_yc)
            if cur_sum < 0:
                rid += 1
                continue
        if cur_sum > best_sum:
            best_sum = cur_sum; best_xc = cur_xc.copy(); best_yc = cur_yc.copy(); best_r = cur_r.copy()

        # hill steps for this restart
        max_steps = 3500
        for step in range(max_steps):
            if time.time() - t0 >= SEARCH:
                break
            # annealed sigma
            sigma = 0.05 * np.exp(-step / 900.0) + 0.0015
            if rng.rand() < 0.88:
                i = rng.randint(n)
                dx = rng.normal(0, sigma)
                dy = rng.normal(0, sigma)
                txc = cur_xc.copy(); tyc = cur_yc.copy()
                nx = cur_xc[i] + dx; ny = cur_yc[i] + dy
                if nx < 0.005: nx = 0.005
                if nx > 0.995: nx = 0.995
                if ny < 0.005: ny = 0.005
                if ny > 0.995: ny = 0.995
                txc[i] = nx; tyc[i] = ny
            else:
                # global small shake
                txc = cur_xc + rng.normal(0, sigma * 0.4, n)
                tyc = cur_yc + rng.normal(0, sigma * 0.4, n)
                txc = np.clip(txc, 0.005, 0.995)
                tyc = np.clip(tyc, 0.005, 0.995)
            s, r = lp_eval(txc, tyc)
            if s > cur_sum + 1e-12:
                cur_xc, cur_yc, cur_sum, cur_r = txc, tyc, s, r
                if s > best_sum:
                    best_sum = s; best_xc = txc.copy(); best_yc = tyc.copy(); best_r = r.copy()
        rid += 1
        if rid > 60:
            break

    # final fine polish on best with tiny sigma
    if best_xc is not None:
        cur_xc, cur_yc, cur_sum, cur_r = best_xc.copy(), best_yc.copy(), best_sum, best_r.copy()
        step = 0
        while time.time() - t0 < HARD:
            sigma = 0.004 if (step % 500) < 400 else 0.015
            i = rng.randint(n)
            txc = cur_xc.copy(); tyc = cur_yc.copy()
            txc[i] = np.clip(cur_xc[i] + rng.normal(0, sigma), 0.005, 0.995)
            tyc[i] = np.clip(cur_yc[i] + rng.normal(0, sigma), 0.005, 0.995)
            s, r = lp_eval(txc, tyc)
            if s > cur_sum + 1e-12:
                cur_xc, cur_yc, cur_sum, cur_r = txc, tyc, s, r
                if s > best_sum:
                    best_sum = s; best_xc = txc.copy(); best_yc = tyc.copy(); best_r = r.copy()
            step += 1
            if step > 20000:
                break

    if best_xc is None:
        k = int(np.ceil(np.sqrt(n)))
        rr = 0.5 / k
        pts = []
        for i in range(k):
            for j in range(k):
                if len(pts) < n:
                    pts.append([(2*i+1)*rr, (2*j+1)*rr, rr*0.999999])
        return np.array(pts)

    xc, yc, r = best_xc.copy(), best_yc.copy(), best_r.copy()
    # exact feasibility scaling
    b = np.minimum(np.minimum(xc, 1.0 - xc), np.minimum(yc, 1.0 - yc))
    d = np.sqrt((xc[I] - xc[J]) ** 2 + (yc[I] - yc[J]) ** 2)
    Rsum = r[I] + r[J]
    mask = Rsum > 1e-14
    s_pair = 1.0
    if np.any(mask):
        ratios = d[mask] / (Rsum[mask] + 1e-30)
        s_pair = float(np.min(ratios))
    # box ratios
    s_box = float(np.min(b / (r + 1e-30)))
    s = min(s_pair, s_box, 1.0)
    # if solver slightly infeasible s<1 corrects it, else tiny shrink for strictness
    scale = s * (1.0 - 1e-8)
    if scale > 1.0:
        scale = 1.0 - 1e-8
    if scale <= 0:
        scale = 0.99
    r = r * scale
    r = np.maximum(r, 1e-9)
    arr = np.column_stack((xc, yc, r))
    return arr
