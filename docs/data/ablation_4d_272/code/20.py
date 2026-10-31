import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds, cap=6000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = c_all.max()
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], cap - 1)[:cap]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[order]] = True
            sel = keep
        c = c_all[sel]
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return x


def _softmax_gd(x, steps, a0, a1, s0, s1, mu=0.8):
    n, d = x.shape
    vel = np.zeros_like(x)
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac if a0 != a1 else a0
        smax = s0 * (s1 / s0) ** frac if s0 != s1 else s0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        vel = mu * vel + gt
        nrm = np.linalg.norm(vel, axis=1).max()
        if nrm > 1e-14:
            x = x - (smax / nrm) * vel
            x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


_IU0 = None
_IU1 = None

def _topk_gd(x, steps, K0, K1, s0, s1, mu=0.8):
    global _IU0, _IU1
    n, d = x.shape
    if _IU0 is None:
        _IU0, _IU1 = np.triu_indices(n, 1)
    IU0, IU1 = _IU0, _IU1
    vel = np.zeros_like(x)
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        K = int(round(K0 * (K1 / K0) ** frac)) if K0 != K1 else K0
        K = max(10, min(K, len(IU0)))
        smax = s0 * (s1 / s0) ** frac if s0 != s1 else s0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        cg = G[IU0, IU1]
        # threshold for top K
        part = np.argpartition(-cg, K - 1)[:K]
        W = np.zeros_like(G)
        W[IU0[part], IU1[part]] = 1.0
        W[IU1[part], IU0[part]] = 1.0
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        vel = mu * vel + gt
        nrm = np.linalg.norm(vel, axis=1).max()
        if nrm > 1e-14:
            x = x - (smax / nrm) * vel
            x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _coord_refine(x, sweeps=2, steps_per=60, s=0.01):
    # coordinate descent on worst points: minimize softmax over its row
    n, d = x.shape
    for _ in range(sweeps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        rowmax = G.max(axis=1)
        order = np.argsort(-rowmax)[:24]
        for p in order:
            dots = x @ x[p]
            dots[p] = -2.0
            # focus on near-max neighbours of p
            m = dots.max()
            # weight neighbours
            w = np.exp(60.0 * (dots - m))
            w[p] = 0.0
            g = (w[:, None] * x).sum(axis=0)
            # few tangential steps
            c = x[p].copy()
            v = np.zeros(4)
            ss = s
            for _ in range(steps_per):
                gt = g - np.dot(g, c) * c
                # recompute g occasionally? keep fixed for speed, refresh halfway
                nn = np.linalg.norm(gt)
                if nn < 1e-14:
                    break
                gt = gt / nn
                v = 0.6 * v + gt
                vn = np.linalg.norm(v)
                if vn < 1e-14:
                    break
                c = c - (ss / vn) * v
                c /= np.linalg.norm(c)
                ss *= 0.99
            x[p] = c
        # refresh G next sweep
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _optimize_single(Y, c0, steps=160, s=0.03):
    c = c0 / np.linalg.norm(c0)
    vel = np.zeros(4)
    for _ in range(steps):
        dots = Y @ c
        m = dots.max()
        sel = dots > m - 1e-3
        if np.count_nonzero(sel) == 0:
            g = Y[np.argmax(dots)]
        else:
            g = Y[sel].sum(axis=0)
        gt = g - np.dot(g, c) * c
        nn = np.linalg.norm(gt)
        if nn < 1e-14:
            break
        gt = gt / nn
        vel = 0.5 * vel + gt
        vn = np.linalg.norm(vel)
        if vn < 1e-14:
            break
        c = c - (s / vn) * vel
        c /= np.linalg.norm(c)
        s *= 0.995
    return c / np.linalg.norm(c)


def _hole_reinsert(x, rng, n_samples=10000):
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    k = int(np.argmax(G))
    i, j = divmod(k, n)
    Gi = G[i].copy(); Gj = G[j].copy()
    Gi[j] = -2; Gj[i] = -2
    move = i if Gi.max() > Gj.max() else j
    if rng.random() < 0.25:
        move = int(rng.integers(n))
        # bias to bad points: pick among worst 30
        if rng.random() < 0.7:
            rowmax = G.max(axis=1)
            worst = np.argpartition(-rowmax, 30)[:30]
            move = int(worst[rng.integers(len(worst))])
    Y = np.delete(x, move, axis=0)
    C = rng.standard_normal((n_samples, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    best_c = None
    best_s = 2.0
    bs = 2500
    for a in range(0, n_samples, bs):
        Cb = C[a:a + bs]
        S = Cb @ Y.T
        sm = S.max(axis=1)
        kk = int(np.argmin(sm))
        if sm[kk] < best_s:
            best_s = float(sm[kk])
            best_c = Cb[kk].copy()
    best_c = _optimize_single(Y, best_c, steps=180, s=0.03)
    y = x.copy()
    y[move] = best_c / np.linalg.norm(best_c)
    return y


def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(20261031)
    n, d = 272, 4

    try:
        bx = np.load("best.npy")
    except Exception:
        bx = None
    try:
        rx = np.load("record.npy")
    except Exception:
        rx = None
    cands0 = []
    if bx is not None:
        bx = np.asarray(bx, dtype=np.float64)
        if bx.shape == (n, d):
            bx = bx / np.linalg.norm(bx, axis=1, keepdims=True)
            cands0.append(bx)
    if rx is not None:
        rx = np.asarray(rx, dtype=np.float64)
        if rx.shape == (n, d):
            rx = rx / np.linalg.norm(rx, axis=1, keepdims=True)
            cands0.append(rx)
    if len(cands0) == 0:
        bx = rng.standard_normal((n, d))
        bx /= np.linalg.norm(bx, axis=1, keepdims=True)
        cands0.append(bx)
    # pick best of loaded
    best_x = cands0[0].copy()
    best_v = _max_cos(best_x)
    for c in cands0[1:]:
        v = _max_cos(c)
        if v < best_v:
            best_x, best_v = c.copy(), v

    archive = [(best_v, best_x.copy())]

    def push_archive(v, xx):
        nonlocal archive
        # keep top 3 distinct
        for vv, _ in archive:
            if abs(vv - v) < 1e-9:
                return
        archive.append((v, xx.copy()))
        archive.sort(key=lambda t: t[0])
        archive = archive[:3]

    it = 0
    sigmas = [0.004, 0.010, 0.022, 0.045, 0.085]
    while time.time() - t0 < BUDGET - 13.0:
        it += 1
        remain = BUDGET - 13.0 - (time.time() - t0)
        if remain < 3:
            break
        # choose base
        if len(archive) > 1 and rng.random() < 0.22:
            base = archive[1][1]
        elif len(archive) > 2 and rng.random() < 0.08:
            base = archive[2][1]
        else:
            base = best_x
        r = rng.random()
        # rowmax for rattler boost (cheap)
        try:
            Gtmp = base @ base.T
            np.fill_diagonal(Gtmp, -2.0)
            rowmax = Gtmp.max(axis=1)
            mcur = float(Gtmp.max())
            is_rat = rowmax < (mcur - 5e-4)
        except Exception:
            is_rat = np.zeros(n, dtype=bool)
        xp = None
        if r < 0.42:
            sigma = sigmas[(it - 1) % len(sigmas)] * (0.7 + 0.6 * rng.random())
            noise = rng.standard_normal((n, d)) * sigma
            if np.any(is_rat):
                noise[is_rat] *= 3.0
            xp = base + noise
            if rng.random() < 0.5:
                iu = np.triu_indices(n, 1)
                cg = (base @ base.T)[iu]
                top = np.argpartition(-cg, 19)[:20]
                ch = int(top[rng.integers(len(top))])
                i = int(iu[0][ch]); j = int(iu[1][ch])
                push = 0.5 * sigma + 0.005
                xp[i] += push * (base[i] - base[j])
                xp[j] += push * (base[j] - base[i])
            xp /= np.linalg.norm(xp, axis=1, keepdims=True)
            a0 = 22.0 + 10.0 * rng.random()
            a1 = 140.0 + 100.0 * rng.random()
            s0 = 0.007 + 0.007 * rng.random()
            s1 = 0.0006 + 0.001 * rng.random()
            steps = 500 + int(rng.integers(0, 450))
            if remain < 30:
                steps = min(steps, 400)
            xp = _softmax_gd(xp, steps, a0, a1, s0, s1, mu=0.8)
        elif r < 0.62:
            sigma = sigmas[rng.integers(len(sigmas))] * (0.7 + 0.6 * rng.random())
            noise = rng.standard_normal((n, d)) * sigma
            if np.any(is_rat):
                noise[is_rat] *= 3.0
            xp = base + noise
            xp /= np.linalg.norm(xp, axis=1, keepdims=True)
            K0 = int(800 + rng.integers(0, 800))
            K1 = int(80 + rng.integers(0, 200))
            if K1 > K0:
                K0, K1 = K1, K0
            s0 = 0.006 + 0.006 * rng.random()
            s1 = 0.0006 + 0.001 * rng.random()
            steps = 450 + int(rng.integers(0, 400))
            if remain < 30:
                steps = min(steps, 350)
            xp = _topk_gd(xp, steps, K0, K1, s0, s1, mu=0.8)
        elif r < 0.80:
            ns = 8000 + int(rng.integers(0, 6000))
            xp = _hole_reinsert(base, rng, n_samples=ns)
            # short gd to settle
            a0 = 25.0 + 10.0 * rng.random()
            a1 = 160.0 + 80.0 * rng.random()
            xp = _softmax_gd(xp, 350, a0, a1, 0.006, 0.0008, mu=0.8)
        else:
            # subset reshuffle
            m = int(rng.integers(4, 26))
            xp = base.copy()
            idx = rng.choice(n, size=m, replace=False)
            # bias to worst points
            if rng.random() < 0.7:
                G2 = base @ base.T
                np.fill_diagonal(G2, -2.0)
                rm = G2.max(axis=1)
                worst = np.argpartition(-rm, 40)[:40]
                idx = rng.choice(worst, size=min(m, len(worst)), replace=False)
            rnd = rng.standard_normal((len(idx), d))
            rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
            xp[idx] = rnd
            a0 = 20.0 + 10.0 * rng.random()
            a1 = 150.0 + 100.0 * rng.random()
            steps = 550 + int(rng.integers(0, 400))
            if remain < 30:
                steps = min(steps, 400)
            if rng.random() < 0.5:
                xp = _softmax_gd(xp, steps, a0, a1, 0.008, 0.0008, mu=0.8)
            else:
                xp = _topk_gd(xp, steps, 1200, 150, 0.007, 0.0008, mu=0.8)
        v = _max_cos(xp)
        # quick coordinate settle if promising
        if v < best_v + 0.004:
            xp = _coord_refine(xp, sweeps=1, steps_per=40, s=0.008)
            v = _max_cos(xp)
        # decide LP
        if v < best_v + 0.0015:
            rem2 = BUDGET - 13.0 - (time.time() - t0)
            if rem2 < 2.5:
                # no time for LP, just record
                if v < best_v - 1e-12:
                    best_x, best_v = xp.copy(), v
                    push_archive(v, xp)
                continue
            lp_time = min(4.5, max(2.0, rem2 / 7.0))
            # smaller cap when far, larger when close
            cap = 6000 if v < best_v + 0.0005 else 4000
            yp = _lp_polish(xp, lp_time, cap=cap)
            vp = _max_cos(yp)
            if vp < best_v - 1e-12:
                best_x, best_v = yp, vp
                push_archive(vp, yp)
            elif v < best_v - 5e-5:
                # GD alone good but LP failed to beat: keep GD point for diversity
                push_archive(v, xp)
                # also try to keep best GD if it beats best (should not happen here)
                if v < best_v - 1e-12:
                    best_x, best_v = xp, v
            else:
                if vp < archive[-1][0] - 1e-7 and rng.random() < 0.3:
                    push_archive(vp, yp)
        else:
            # far from best, occasionally keep second basin
            if len(archive) < 3 and v < best_v + 0.008 and rng.random() < 0.05:
                push_archive(v, xp)

    remain = BUDGET - (time.time() - t0)
    if remain > 3:
        best_x = _lp_polish(best_x, min(remain - 1.0, 10.0), cap=12000)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
