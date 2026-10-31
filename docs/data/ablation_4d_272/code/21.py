import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds):
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    rep_n = np.repeat(np.arange(n), d)
    col_n = np.arange(n * d)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        sel = c_all > m - max(20.0 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 6000:
            idx = np.where(sel)[0]
            part = np.argpartition(-c_all[idx], 6000 - 1)[:6000]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[part]] = True
            sel = keep
        I = iu0[0][sel]
        J = iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (rep_n, col_n)), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def smooth_descent(x0, beta0, beta1, steps, s0, s1, mu=0.85):
    n, d = x0.shape
    x = x0.copy()
    vel = np.zeros_like(x)
    lb0 = np.log(beta0)
    lb1 = np.log(beta1)
    ls0 = np.log(s0)
    ls1 = np.log(s1)
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * frac))
        smax = float(np.exp(ls0 + (ls1 - ls0) * frac))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # weights
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            break
        GX = (E @ x) / S
        dot = np.sum(GX * x, axis=1, keepdims=True)
        gt = GX - dot * x
        # normalize grad to avoid blowup
        gn = np.linalg.norm(gt, axis=1).max()
        if gn > 1e-12:
            gt = gt / gn
        vel = mu * vel + (1.0 - mu) * gt
        vn = np.linalg.norm(vel, axis=1).max()
        if vn > 1e-14:
            x = x - (smax / vn) * vel
            x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x


def find_hole(X, rng, n_samples=8000, refine=80):
    n, d = X.shape
    C = rng.randn(n_samples, d)
    C = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
    dots = C @ X.T
    md = dots.max(axis=1)
    bi = int(np.argmin(md))
    c = C[bi].copy()
    vel = np.zeros(d)
    step = 0.05
    for it in range(refine):
        frac = it / max(refine - 1, 1)
        beta = 60.0 * (400.0 / 60.0) ** frac
        s = 0.03 * (0.005 / 0.03) ** frac
        dv = X @ c  # (n,)
        m = float(dv.max())
        w = np.exp(beta * (dv - m))
        sw = float(w.sum())
        if sw <= 0 or not np.isfinite(sw):
            break
        g = (w @ X) / sw  # gradient of L w.r.t c
        gt = g - np.dot(g, c) * c
        nn = float(np.linalg.norm(gt))
        if nn < 1e-14:
            break
        gt = gt / nn
        vel = 0.8 * vel + 0.2 * gt
        vn = float(np.linalg.norm(vel))
        if vn < 1e-14:
            break
        c = c - (s / vn) * vel
        c = c / max(np.linalg.norm(c), 1e-12)
    return c


def solve():
    np.random.seed(0)
    rng = np.random.RandomState(0)
    t0 = time.time()
    HARD = 172.0
    LOOP_END = 148.0
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.randn(272, 4)
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = xb
    xb = np.asarray(xb, dtype=np.float64)
    xr = np.asarray(xr, dtype=np.float64)
    xb = xb / np.maximum(np.linalg.norm(xb, axis=1, keepdims=True), 1e-12)
    xr = xr / np.maximum(np.linalg.norm(xr, axis=1, keepdims=True), 1e-12)
    vb = max_cos(xb)
    vr = max_cos(xr)
    if vr < vb:
        best = xr.copy()
        best_val = vr
    else:
        best = xb.copy()
        best_val = vb
    n, d = best.shape

    trial = 0
    # small catalogue of annealings
    anneals = [
        (60, 250, 350, 0.04, 0.004),
        (100, 400, 400, 0.03, 0.003),
        (40, 150, 250, 0.06, 0.006),
        (150, 500, 300, 0.02, 0.002),
        (80, 300, 500, 0.05, 0.003),
    ]

    while time.time() - t0 < LOOP_END:
        trial += 1
        if time.time() - t0 > LOOP_END - 2.0:
            break
        mode = rng.rand()
        x_try = best.copy()
        if mode < 0.30:
            # isotropic gaussian shake, varied scale
            sigma = float(rng.choice([0.003, 0.006, 0.012, 0.025, 0.05, 0.10]))
            x_try = best + sigma * rng.randn(n, d)
            x_try = x_try / np.maximum(np.linalg.norm(x_try, axis=1, keepdims=True), 1e-12)
        elif mode < 0.50:
            # coherent cluster shift: pick cap and move together
            ci = int(rng.randint(n))
            center = best[ci]
            dots_c = best @ center
            # random cap threshold
            thr = float(rng.choice([0.90, 0.93, 0.96, 0.98]))
            mask = dots_c > thr
            if int(mask.sum()) < 2:
                mask = np.zeros(n, dtype=bool)
                mask[ci] = True
            sigma_c = float(rng.choice([0.01, 0.02, 0.04, 0.08]))
            sigma_i = sigma_c * 0.4
            shift = rng.randn(d)
            shift = shift / max(np.linalg.norm(shift), 1e-12)
            # project shift to tangent at center to stay on sphere region
            shift = shift - np.dot(shift, center) * center
            x_try[mask] += sigma_c * shift + sigma_i * rng.randn(int(mask.sum()), d)
            # also tiny global jitter
            x_try += 0.002 * rng.randn(n, d)
            x_try = x_try / np.maximum(np.linalg.norm(x_try, axis=1, keepdims=True), 1e-12)
        elif mode < 0.70:
            # hole surgery: replace worst / rattler points by hole centres
            try:
                G0 = best @ best.T
                np.fill_diagonal(G0, -2.0)
                rowmax = G0.max(axis=1)
                # worst indices
                order = np.argsort(-rowmax)
                k = int(rng.randint(1, 4))
                # mix worst and rattlers
                ratt = np.where(rowmax < best_val - 0.01)[0]
                cand_idx = list(order[:8])
                if len(ratt) > 0:
                    cand_idx += list(rng.choice(ratt, min(len(ratt), 4), replace=False))
                idx = np.array(rng.choice(cand_idx, k, replace=False))
                x_try = best + 0.005 * rng.randn(n, d)
                for j in idx:
                    h = find_hole(best, rng, n_samples=6000, refine=60)
                    # small jitter around hole
                    h = h + 0.01 * rng.randn(d)
                    h = h / max(np.linalg.norm(h), 1e-12)
                    x_try[j] = h
                x_try = x_try / np.maximum(np.linalg.norm(x_try, axis=1, keepdims=True), 1e-12)
            except Exception:
                x_try = best + 0.01 * rng.randn(n, d)
                x_try = x_try / np.maximum(np.linalg.norm(x_try, axis=1, keepdims=True), 1e-12)
        elif mode < 0.85:
            # worst-pair split: push the two points of worst pairs apart
            try:
                G0 = best @ best.T
                np.fill_diagonal(G0, -2.0)
                # find top few pairs
                iu = np.triu_indices(n, 1)
                vals = G0[iu]
                top = np.argpartition(-vals, min(6, len(vals) - 1))[:6]
                pairs = [(int(iu[0][t]), int(iu[1][t])) for t in top]
                x_try = best + 0.004 * rng.randn(n, d)
                for (a, b) in pairs[:3]:
                    # move a and b apart along their difference tangent
                    diff = x_try[a] - x_try[b]
                    diff = diff - 0.5 * (np.dot(diff, x_try[a]) * x_try[a] + np.dot(diff, x_try[b]) * x_try[b])
                    nn = np.linalg.norm(diff)
                    if nn < 1e-12:
                        continue
                    diff /= nn
                    mag = float(rng.uniform(0.005, 0.03))
                    x_try[a] += mag * diff
                    x_try[b] -= mag * diff
                x_try = x_try / np.maximum(np.linalg.norm(x_try, axis=1, keepdims=True), 1e-12)
            except Exception:
                x_try = best + 0.01 * rng.randn(n, d)
                x_try = x_try / np.maximum(np.linalg.norm(x_try, axis=1, keepdims=True), 1e-12)
        else:
            # random-subspace remix: rotate a random subset by small random rotation
            m = int(rng.randint(8, 40))
            idx = rng.choice(n, m, replace=False)
            Q, _ = np.linalg.qr(rng.randn(d, d))
            if float(np.linalg.det(Q)) < 0:
                Q[:, 0] = -Q[:, 0]
            # interpolate between I and Q
            alpha = float(rng.uniform(0.05, 0.4))
            # slerp-ish: apply partial rotation via linear blend + re-orthonormalize effect on points
            # build R = I + alpha*(Q-I)
            R = np.eye(d) + alpha * (Q - np.eye(d))
            x_try[idx] = x_try[idx] @ R.T
            x_try += 0.003 * rng.randn(n, d)
            x_try = x_try / np.maximum(np.linalg.norm(x_try, axis=1, keepdims=True), 1e-12)

        # annealed manifold descent
        ai = int(rng.randint(len(anneals)))
        b0, b1, steps, s0, s1 = anneals[ai]
        # randomize steps a bit
        steps = int(steps * float(rng.uniform(0.8, 1.2)))
        steps = max(150, min(600, steps))
        try:
            x_opt = smooth_descent(x_try, b0, b1, steps, s0, s1)
        except Exception:
            continue
        v0 = max_cos(x_opt)
        # quick filter
        if not np.isfinite(v0):
            continue
        if v0 > best_val + 0.012:
            continue
        # promising -> short LP lock (only if close)
        if v0 < best_val + 0.002:
            rem = LOOP_END - (time.time() - t0)
            if rem < 0.3:
                # no time for LP, just compare
                if v0 < best_val - 1e-12:
                    best = x_opt
                    best_val = v0
                continue
            lp_t = 0.9 if v0 < best_val else 0.45
            lp_t = min(lp_t, max(0.2, rem - 0.15))
            try:
                x_pol = lp_polish(x_opt, lp_t)
            except Exception:
                continue
            v = max_cos(x_pol)
            if v < best_val - 1e-12 and np.isfinite(v):
                best = x_pol
                best_val = v
        else:
            if v0 < best_val - 1e-12:
                best = x_opt
                best_val = v0
        if time.time() - t0 >= HARD - 6:
            break

    remain = HARD - (time.time() - t0)
    if remain > 2.5:
        try:
            best = lp_polish(best, min(remain - 1.0, 22.0))
        except Exception:
            pass
    best = best / np.maximum(np.linalg.norm(best, axis=1, keepdims=True), 1e-12)
    bad = ~np.all(np.isfinite(best), axis=1)
    if np.any(bad):
        rnd = rng.randn(int(np.sum(bad)), d)
        rnd = rnd / np.maximum(np.linalg.norm(rnd, axis=1, keepdims=True), 1e-12)
        best[bad] = rnd
    norms = np.linalg.norm(best, axis=1, keepdims=True)
    bad2 = (norms[:, 0] < 1e-9)
    if np.any(bad2):
        rnd = rng.randn(int(np.sum(bad2)), d)
        rnd = rnd / np.maximum(np.linalg.norm(rnd, axis=1, keepdims=True), 1e-12)
        best[bad2] = rnd
    return best
