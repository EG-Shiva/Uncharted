import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds, r0=2e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = float(r0)
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel_idx = np.where(c_all > thr)[0]
        if sel_idx.size == 0:
            break
        if sel_idx.size > 3500:
            part = np.argpartition(c_all, -3500)[-3500:]
            sel_idx = part
        I = iu[0][sel_idx]
        J = iu[1][sel_idx]
        c_sel = c_all[sel_idx]
        p = sel_idx.size
        nv = n * d + 1
        cols_b = np.empty((p, 2 * d + 1), dtype=np.int64)
        cols_b[:, :d] = I[:, None] * d + ar[None, :]
        cols_b[:, d:2 * d] = J[:, None] * d + ar[None, :]
        cols_b[:, 2 * d] = n * d
        vals_b = np.empty((p, 2 * d + 1), dtype=np.float64)
        vals_b[:, :d] = x[J]
        vals_b[:, d:2 * d] = x[I]
        vals_b[:, 2 * d] = -1.0
        rows = np.repeat(np.arange(p), 2 * d + 1)
        try:
            A_ub = coo_matrix((vals_b.ravel(), (rows, cols_b.ravel())), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
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
        if v < best - 1e-15:
            x = y
            best = v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
            if radius < 1e-12:
                break
    return x


def smooth_gd(x, steps, a0, a1, s0, s1, mu=0.8, deadline=1e30):
    n, d = x.shape
    vel = np.zeros_like(x)
    for it in range(steps):
        if it % 50 == 0 and time.time() > deadline:
            break
        frac = it / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac if a0 != a1 else a0
        smax = s0 * (s1 / s0) ** frac if s0 != s1 else s0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # softmax weights
        W = np.exp(np.clip(alpha * (G - m), -50.0, 0.0))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        vel = mu * vel + gt
        nrm = float(np.linalg.norm(vel, axis=1).max())
        if nrm > 1e-14:
            x = x - (smax / nrm) * vel
            x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x


def refine_hole(Y, c0, beta=60.0, steps=220, s0=0.08):
    c = c0 / max(np.linalg.norm(c0), 1e-12)
    vel = np.zeros(4)
    s = float(s0)
    for _ in range(steps):
        dots = Y @ c
        m = float(dots.max())
        w = np.exp(np.clip(beta * (dots - m), -50.0, 0.0))
        sw = float(w.sum())
        if sw <= 0 or not np.isfinite(sw):
            break
        g = (w @ Y) / sw
        gt = g - float(g @ c) * c
        nn = float(np.linalg.norm(gt))
        if nn < 1e-14:
            break
        gt = gt / nn
        vel = 0.5 * vel + gt
        vn = float(np.linalg.norm(vel))
        if vn < 1e-14:
            break
        c = c - (s / vn) * vel
        n = float(np.linalg.norm(c))
        if n < 1e-12 or not np.isfinite(n):
            break
        c = c / n
        s *= 0.997
    return c / max(np.linalg.norm(c), 1e-12)


def largest_holes(Y, rng, n_samples=50000, n_cand=6, beta=60.0):
    d = Y.shape[1]
    best_vals = np.full(n_cand, 2.0)
    best_vecs = rng.standard_normal((n_cand, d))
    best_vecs /= np.linalg.norm(best_vecs, axis=1, keepdims=True)
    bs = 5000
    done = 0
    while done < n_samples:
        cur = min(bs, n_samples - done)
        C = rng.standard_normal((cur, d))
        C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
        S = C @ Y.T
        sm = S.max(axis=1)
        # merge into candidate pool: keep smallest sm
        # simple: take best few of this batch and try to insert
        k = min(4, cur)
        idx = np.argpartition(sm, k - 1)[:k]
        for j in idx:
            v = float(sm[j])
            worst = int(np.argmax(best_vals))
            if v < best_vals[worst]:
                best_vals[worst] = v
                best_vecs[worst] = C[j]
        done += cur
    # refine each candidate with LSE minimisation
    refined = []
    for i in range(n_cand):
        c = refine_hole(Y, best_vecs[i], beta=beta, steps=200, s0=0.06)
        # score = max dot
        md = float((Y @ c).max())
        refined.append((md, c))
    refined.sort(key=lambda t: t[0])
    return refined[0][1], refined[0][0]


def contact_degree(X, m, gap=0.002):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    deg = np.count_nonzero(G > (m - gap), axis=1)
    rmax = G.max(axis=1)
    return deg, rmax, G


def get_delete_idx(X, kind, k, rng):
    n = X.shape[0]
    k = int(min(max(k, 1), n - 10))
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    if kind == "degree":
        deg, rmax, _ = contact_degree(X, m, gap=0.002)
        # stress proxy: degree primary, rmax secondary
        order = np.lexsort((rmax, deg))
        return order[-k:]
    elif kind == "cluster":
        seed = int(rng.integers(0, n))
        dots = X @ X[seed]
        order = np.argsort(dots)
        return order[-k:]
    else:  # random
        return rng.choice(n, k, replace=False)


def solve():
    np.random.seed(0)
    t0 = time.time()
    HARD = 172.0
    LOOP_END = 148.0
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = np.load("record.npy")
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
    n_target = 272
    # ensure shape
    if best.shape[0] != n_target:
        # fallback: tile/random
        rng0 = np.random.default_rng(0)
        tmp = rng0.standard_normal((n_target, 4))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        best = tmp
        best_val = max_cos(best)
    n, d = best.shape

    rng = np.random.default_rng(0)
    # quick polish baseline
    if time.time() - t0 < LOOP_END:
        try:
            rem = LOOP_END - (time.time() - t0)
            bp = lp_polish(best, min(2.0, max(0.3, rem * 0.1)))
            vp = max_cos(bp)
            if vp < best_val:
                best, best_val = bp, vp
        except Exception:
            pass

    strategies = [
        ("degree", 32), ("cluster", 32), ("random", 32),
        ("degree", 24), ("cluster", 24), ("random", 28),
        ("degree", 40), ("cluster", 40), ("random", 36),
    ]
    si = 0
    trial = 0
    while time.time() - t0 < LOOP_END:
        trial += 1
        kind, k = strategies[si % len(strategies)]
        si += 1
        # vary k slightly
        kj = int(k + int(rng.integers(-3, 4)))
        kj = int(np.clip(kj, 24, 40))
        try:
            del_idx = get_delete_idx(best, kind, kj, rng)
        except Exception:
            continue
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[del_idx] = False
        core = best[keep_mask].copy()
        nc = core.shape[0]
        # anneal + LP-polish remaining code
        try:
            # small jitter to leave basin
            core = core + 0.004 * rng.standard_normal(core.shape)
            core /= np.maximum(np.linalg.norm(core, axis=1, keepdims=True), 1e-12)
            dl = time.time() + 4.0
            # anneal core
            core = smooth_gd(core, 350, 25.0, 95.0, 0.02, 0.002, mu=0.8, deadline=dl)
            if time.time() - t0 > LOOP_END - 8:
                continue
            core = lp_polish(core, 0.9, r0=2e-3)
            vc = max_cos(core)
            # require substantially below record to have headroom
            # with fewer points should be easy; else jitter more and continue
            if vc > 0.895:
                # still try regrow but with extra anneal
                core = smooth_gd(core, 200, 30.0, 100.0, 0.015, 0.002, mu=0.82, deadline=time.time() + 2.0)
                vc = max_cos(core)
        except Exception:
            continue
        # regrow one at a time
        try:
            Y = core.copy()
            n_cur = Y.shape[0]
            n_need = n_target - n_cur
            # freeze schedule: full re-anneal every B insertions
            B = int(rng.integers(4, 9))  # 4-8
            since_anneal = 0
            ok = True
            for t in range(n_need):
                if time.time() - t0 > LOOP_END - 6:
                    ok = False
                    break
                hole, _ = largest_holes(Y, rng, n_samples=50000, n_cand=6, beta=60.0)
                # freezing older points: hole already optimised with Y frozen
                Y = np.vstack([Y, hole[None, :]])
                # brief local re-optimisation of just-inserted point (frozen others)
                # do one more short refine alternating with tiny full nudge
                Y[-1] = refine_hole(Y[:-1], Y[-1], beta=60.0, steps=80, s0=0.03)
                since_anneal += 1
                if since_anneal >= B and (t + 1) < n_need:
                    # fully re-anneal current set
                    Y = smooth_gd(Y, 160, 25.0, 90.0, 0.012, 0.002, mu=0.8, deadline=time.time() + 3.0)
                    since_anneal = 0
                if time.time() - t0 > LOOP_END - 4:
                    # fill remaining quickly with holes without full anneal
                    pass
            if not ok:
                # incomplete due to time: skip
                continue
            # final full anneal of completed 272
            Y = smooth_gd(Y, 250, 25.0, 110.0, 0.015, 0.0015, mu=0.82, deadline=time.time() + 4.0)
            if time.time() - t0 > LOOP_END:
                # still evaluate without LP
                vY = max_cos(Y)
                if vY < best_val - 1e-12:
                    best, best_val = Y, vY
                continue
            Yp = lp_polish(Y, 1.4, r0=2e-3)
            vYp = max_cos(Yp)
            if vYp < best_val - 1e-12 and np.all(np.isfinite(Yp)):
                best, best_val = Yp, vYp
            else:
                # also try unpolished
                vY = max_cos(Y)
                if vY < best_val - 1e-12 and np.all(np.isfinite(Y)):
                    # polish next round; keep
                    best, best_val = Y, vY
        except Exception:
            continue
        if time.time() - t0 >= LOOP_END:
            break

    remain = HARD - (time.time() - t0)
    if remain > 2.5:
        try:
            best = lp_polish(best, min(remain - 1.0, 15.0), r0=1.5e-3)
        except Exception:
            pass
    best = best / np.maximum(np.linalg.norm(best, axis=1, keepdims=True), 1e-12)
    bad = ~np.all(np.isfinite(best), axis=1)
    if np.any(bad):
        r = np.random.randn(int(np.sum(bad)), 4)
        r /= np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-12)
        best[bad] = r
    return best
