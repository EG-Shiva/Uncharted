import numpy as np
from scipy.optimize import linprog

def solve():
    n = 26
    rng = np.random.default_rng(0)

    def polish_lp(xc, yc):
        # clip centres strictly inside
        xc = np.clip(np.asarray(xc, dtype=float), 0.002, 0.998)
        yc = np.clip(np.asarray(yc, dtype=float), 0.002, 0.998)
        b = np.minimum(np.minimum(xc, 1.0 - xc), np.minimum(yc, 1.0 - yc)) - 1e-9
        b = np.maximum(b, 1e-9)
        # build pairwise constraints ri+rj <= dij - margin
        m = n * (n - 1) // 2
        A_ub = np.zeros((m, n), dtype=float)
        b_ub = np.zeros(m, dtype=float)
        k = 0
        for i in range(n):
            xi = xc[i]; yi = yc[i]
            for j in range(i + 1, n):
                A_ub[k, i] = 1.0
                A_ub[k, j] = 1.0
                dx = xi - xc[j]
                dy = yi - yc[j]
                d = float(np.sqrt(dx * dx + dy * dy))
                bb = d - 2e-9
                if bb < 0:
                    bb = 0.0
                b_ub[k] = bb
                k += 1
        c = -np.ones(n, dtype=float)
        bounds = [(0.0, float(bi)) for bi in b]
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        except Exception:
            return None, -1.0
        if res.status != 0:
            return None, -1.0
        r = np.asarray(res.x, dtype=float)
        # ensure strictly positive
        r = np.maximum(r, 1e-9)
        s = float(np.sum(r))
        return r, s

    def adam_run(x0, y0, r0, steps=3000, lr=0.02):
        x = x0.astype(float).copy()
        y = y0.astype(float).copy()
        r = r0.astype(float).copy()
        mx = np.zeros(n); my = np.zeros(n); mr = np.zeros(n)
        vx = np.zeros(n); vy = np.zeros(n); vr = np.zeros(n)
        b1 = 0.9; b2 = 0.999; eps = 1e-8
        W0 = 10.0; W1 = 5000.0
        for t in range(1, steps + 1):
            frac = t / steps
            W = W0 * ((W1 / W0) ** frac)
            # distances
            Dx = x[:, None] - x[None, :]
            Dy = y[:, None] - y[None, :]
            D2 = Dx * Dx + Dy * Dy + 1e-16
            D = np.sqrt(D2)
            S = r[:, None] + r[None, :] - D
            V = np.maximum(S, 0.0)
            np.fill_diagonal(V, 0.0)
            np.fill_diagonal(D, 1.0)  # avoid div by zero, diag unused
            # overlap grads
            # Gx = -2W sum_j V*Dx/D
            Qx = V * Dx / D
            Qy = V * Dy / D
            Gx_o = -2.0 * W * np.sum(Qx, axis=1)
            Gy_o = -2.0 * W * np.sum(Qy, axis=1)
            Gr_o = 2.0 * W * np.sum(V, axis=1)
            # wall
            low_x = np.maximum(r - x, 0.0)
            high_x = np.maximum(x + r - 1.0, 0.0)
            low_y = np.maximum(r - y, 0.0)
            high_y = np.maximum(y + r - 1.0, 0.0)
            Gx_w = 2.0 * W * (-low_x + high_x)
            Gy_w = 2.0 * W * (-low_y + high_y)
            Gr_w = 2.0 * W * (low_x + high_x + low_y + high_y)
            Gx = Gx_o + Gx_w
            Gy = Gy_o + Gy_w
            Gr = -np.ones(n) + Gr_o + Gr_w
            # adam updates
            mx = b1 * mx + (1 - b1) * Gx
            my = b1 * my + (1 - b1) * Gy
            mr = b1 * mr + (1 - b1) * Gr
            vx = b2 * vx + (1 - b2) * (Gx * Gx)
            vy = b2 * vy + (1 - b2) * (Gy * Gy)
            vr = b2 * vr + (1 - b2) * (Gr * Gr)
            c1 = 1.0 - b1 ** t
            c2 = 1.0 - b2 ** t
            mxh = mx / c1; vxh = vx / c2
            myh = my / c1; vyh = vy / c2
            mrh = mr / c1; vrh = vr / c2
            x -= lr * mxh / (np.sqrt(vxh) + eps)
            y -= lr * myh / (np.sqrt(vyh) + eps)
            # smaller lr for radii to grow stably
            r -= (lr * 0.5) * mrh / (np.sqrt(vrh) + eps)
            # project to box
            np.clip(x, 0.0, 1.0, out=x)
            np.clip(y, 0.0, 1.0, out=y)
            np.clip(r, 0.005, 0.30, out=r)
        return x, y, r

    best_sum = -1.0
    best_x = None; best_y = None; best_r = None

    Nrestart = 10
    Steps = 3000
    for rs in range(Nrestart):
        if rs == 0:
            # jittered 5x5 grid + one extra
            xs = []; ys = []
            for ii in range(5):
                for jj in range(5):
                    xs.append((ii + 0.5) / 5.0)
                    ys.append((jj + 0.5) / 5.0)
            xs.append(0.5); ys.append(0.5)
            x0 = np.array(xs) + rng.normal(0, 0.01, n)
            y0 = np.array(ys) + rng.normal(0, 0.01, n)
            x0 = np.clip(x0, 0.05, 0.95)
            y0 = np.clip(y0, 0.05, 0.95)
            r0 = np.full(n, 0.09) + rng.normal(0, 0.005, n)
            r0 = np.clip(r0, 0.05, 0.12)
        elif rs == 1:
            # hex-like init
            x0 = rng.uniform(0.05, 0.95, n)
            y0 = rng.uniform(0.05, 0.95, n)
            r0 = np.full(n, 0.10)
        else:
            x0 = rng.uniform(0.05, 0.95, n)
            y0 = rng.uniform(0.05, 0.95, n)
            r0 = rng.uniform(0.07, 0.11, n)
        # vary lr slightly per restart for diversity
        lr = 0.018 + 0.004 * ((rs % 3) - 1)
        x1, y1, r1 = adam_run(x0, y0, r0, steps=Steps, lr=lr)
        r_opt, s_opt = polish_lp(x1, y1)
        if r_opt is None:
            continue
        # require all radii reasonably positive, else discard
        if np.min(r_opt) < 5e-5:
            # still allow but penalize; keep if sum is best
            pass
        if s_opt > best_sum:
            best_sum = s_opt
            best_x = np.clip(x1, 0.002, 0.998).copy()
            best_y = np.clip(y1, 0.002, 0.998).copy()
            best_r = r_opt.copy()

    # fallback if nothing found
    if best_x is None:
        k = int(np.ceil(np.sqrt(n)))
        rr = 0.5 / k * 0.999
        pts = []
        for i in range(k):
            for j in range(k):
                if len(pts) < n:
                    pts.append([(2 * i + 1) * rr, (2 * j + 1) * rr, rr * 0.999])
        return np.array(pts)

    # stochastic hill climbing on centres using LP value
    # 200 evaluations
    for it in range(200):
        sigma = 0.012 * (1.0 - it / 200.0) + 0.001
        xt = best_x + rng.normal(0, sigma, n)
        yt = best_y + rng.normal(0, sigma, n)
        xt = np.clip(xt, 0.002, 0.998)
        yt = np.clip(yt, 0.002, 0.998)
        r_try, s_try = polish_lp(xt, yt)
        if r_try is None:
            continue
        if s_try > best_sum + 1e-12:
            best_sum = s_try
            best_x = xt
            best_y = yt
            best_r = r_try

    # final strict feasibility shrink
    # recompute exact margins and shrink uniformly by tiny factor
    best_r = np.maximum(best_r, 1e-7)
    # ensure wall
    bwall = np.minimum(np.minimum(best_x, 1.0 - best_x), np.minimum(best_y, 1.0 - best_y))
    # if any violation due to numerics, scale that circle
    for i in range(n):
        if best_r[i] >= bwall[i]:
            best_r[i] = bwall[i] - 1e-9
            if best_r[i] < 1e-7:
                best_r[i] = 1e-7
    # pairwise: compute max ratio and scale if needed (individual greedy + global safety)
    # global safety factor to guarantee 1e-12 tolerance
    best_r = best_r * (1.0 - 1e-7)
    # final verification loop: shrink violators individually
    for _ in range(5):
        viol = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = best_x[i] - best_x[j]
                dy = best_y[i] - best_y[j]
                d = float(np.sqrt(dx * dx + dy * dy))
                if best_r[i] + best_r[j] > d - 1e-12:
                    # shrink both proportionally
                    need = (best_r[i] + best_r[j]) / max(d - 2e-12, 1e-12)
                    if need > 1.0:
                        best_r[i] /= need
                        best_r[j] /= need
                        # tiny extra
                        best_r[i] *= (1.0 - 1e-12)
                        best_r[j] *= (1.0 - 1e-12)
                        viol = True
        if not viol:
            break

    out = np.column_stack([best_x, best_y, best_r])
    return out
