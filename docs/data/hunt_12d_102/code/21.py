import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def adam_refine(x, steps, beta0, beta1, lr):
    n, d = x.shape
    m1 = np.zeros_like(x)
    m2 = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best_x = x.copy()
    best_v = max_cos(x)
    for t in range(1, steps + 1):
        beta = beta0 + (beta1 - beta0) * (t / steps)
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        mval = np.max(G)
        W = np.exp(beta * (G - mval))
        np.fill_diagonal(W, 0.0)
        s = np.sum(W)
        if s <= 0 or not np.isfinite(s):
            break
        W /= s
        grad = (W @ x) * 0.5
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        m1 = b1 * m1 + (1.0 - b1) * gt
        m2 = b2 * m2 + (1.0 - b2) * (gt * gt)
        m1h = m1 / (1.0 - b1 ** t)
        m2h = m2 / (1.0 - b2 ** t)
        step = lr * m1h / (np.sqrt(m2h) + eps)
        x = x - step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[~np.isfinite(nr)] = 1.0
        nr[nr < 1e-12] = 1e-12
        x = x / nr
        if t % 60 == 0 or t == steps:
            v = max_cos(x)
            if v < best_v:
                best_v = v
                best_x = x.copy()
    return best_x


def lp_polish(x, seconds, r0=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    best_x = x.copy()
    radius = float(r0)
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        try:
            c = (x @ x.T)[iu]
        except Exception:
            break
        m = float(np.max(c))
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 2200:
            k = 2200
            idx = np.argpartition(c, len(c) - k)[len(c) - k:]
            sel = np.zeros_like(sel)
            sel[idx] = True
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(nr)):
            radius *= 0.5
            continue
        nr[nr < 1e-12] = 1e-12
        y = y / nr
        v = max_cos(y)
        if v < best:
            x = y
            best = v
            best_x = y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x


def best_hole_insert(x_best, worst_idx, rng, n_cand=12000):
    n, d = x_best.shape
    mask = np.ones(n, dtype=bool)
    mask[worst_idx] = False
    Xrem = x_best[mask]
    cand = rng.randn(n_cand, d)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    # max dot to remaining set
    sc = (cand @ Xrem.T).max(axis=1)
    order = np.argsort(sc)
    # take a few best holes, pick one quasi-randomly among top to diversify
    top = order[:32]
    pick = top[rng.randint(len(top))]
    q = cand[pick].copy()
    # refine hole against fixed set with soft-max repulsion
    qq = q.copy()
    for _ in range(120):
        dots = Xrem @ qq
        mm = float(np.max(dots))
        w = np.exp((dots - mm) / 0.008)
        ws = float(np.sum(w))
        if ws == 0 or not np.isfinite(ws):
            break
        w /= ws
        g = w @ Xrem
        # project to tangent and step to minimize max dot
        gt = g - float(np.dot(g, qq)) * qq
        qq = qq - 0.05 * gt
        nq = float(np.linalg.norm(qq))
        if not np.isfinite(nq) or nq < 1e-12:
            break
        qq /= nq
    x_new = x_best.copy()
    # replace each worst index with perturbed variants of refined hole
    for k, wi in enumerate(worst_idx):
        if k == 0:
            x_new[wi] = qq
        else:
            v = qq + 0.02 * rng.randn(d)
            v /= np.linalg.norm(v)
            x_new[wi] = v
    return x_new / np.linalg.norm(x_new, axis=1, keepdims=True)


def solve():
    np.random.seed(0)
    t0 = time.time()
    rng = np.random.RandomState(20261102)
    n, d = 102, 12
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = rng.randn(n, d)
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (n, d):
        if xb.size == n * d:
            xb = xb.reshape(n, d)
        else:
            xb = rng.randn(n, d)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = max_cos(xb)
    try:
        xr = np.load("record.npy")
        xr = np.asarray(xr, dtype=float)
        if xr.shape == (n, d):
            xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
            vr = max_cos(xr)
            if vr < best_val:
                xb = xr
                best_val = vr
    except Exception:
        pass
    x_best = xb.copy()
    loop_end = t0 + 152.0
    hard_end = t0 + 172.0
    trial = 0
    stall = 0
    sigmas_small = [0.008, 0.015, 0.025, 0.04]
    while time.time() < loop_end and time.time() < hard_end - 5.0:
        trial += 1
        now = time.time()
        if now >= loop_end:
            break
        # contact overload estimate
        try:
            Gtmp = x_best @ x_best.T
            np.fill_diagonal(Gtmp, -2.0)
            mtmp = float(np.max(Gtmp))
            cnt = np.sum(Gtmp > mtmp - 0.004, axis=1)
        except Exception:
            cnt = np.zeros(n)
            mtmp = best_val
        mode = trial % 7
        # stall-adaptive boost
        boost = 1.0
        if stall > 12:
            boost = 2.0
        if stall > 25:
            boost = 3.0
        x_try = None
        if mode == 2 or mode == 5:
            # hole re-insertion of 1-3 worst hubs
            nk = 1 + (trial % 3)  # 1..3
            if stall > 15:
                nk = min(4, nk + 1)
            worst = np.argsort(-cnt)[:nk]
            # occasionally include a random peripheral point
            if trial % 14 == 0:
                worst = np.append(worst[: max(1, nk - 1)], rng.randint(n))
                worst = np.unique(worst)
            try:
                x_try = best_hole_insert(x_best, worst, rng)
                # small global jitter after insertion
                x_try = x_try + (0.004 * boost) * rng.randn(n, d)
            except Exception:
                x_try = x_best + 0.02 * rng.randn(n, d)
        elif mode == 4:
            # large basin jump
            sigma = 0.11 * boost
            x_try = x_best + sigma * rng.randn(n, d)
            # re-randomize single worst point sometimes
            if trial % 2 == 0:
                worst = int(np.argmax(cnt))
                x_try[worst] = rng.randn(d)
        elif mode == 6:
            # medium shake + push top pairs apart explicitly
            sigma = 0.035 * boost
            x_try = x_best + sigma * rng.randn(n, d)
            try:
                Gu = x_best @ x_best.T
                np.fill_diagonal(Gu, -2.0)
                # find top 8 pairs
                iu = np.triu_indices(n, 1)
                c = Gu[iu]
                topk = np.argpartition(-c, 8)[:8]
                for kk in topk:
                    i, j = int(iu[0][kk]), int(iu[1][kk])
                    # push i away from j tangentially
                    v = x_best[j] - float(np.dot(x_best[j], x_best[i])) * x_best[i]
                    x_try[i] += 0.02 * boost * v
                    w = x_best[i] - float(np.dot(x_best[i], x_best[j])) * x_best[j]
                    x_try[j] += 0.02 * boost * w
            except Exception:
                pass
        else:
            # hub-targeted gaussian shake
            sigma = sigmas_small[(trial - 1) % len(sigmas_small)] * boost
            x_try = x_best + sigma * rng.randn(n, d)
            k = 6 + (trial % 5)
            idx = np.argsort(-cnt)[:k]
            x_try[idx] += (sigma * 2.0) * rng.randn(k, d)
            if stall > 20 and trial % 3 == 0:
                worst = int(np.argmax(cnt))
                x_try[worst] = rng.randn(d)
        x_try = x_try / np.linalg.norm(x_try, axis=1, keepdims=True)
        steps = 350
        if mode == 4:
            steps = 450
        beta0 = 25 + (trial % 4) * 10
        beta1 = 150 + (trial % 3) * 40
        lr = 0.006 + 0.004 * ((trial * 7) % 5) / 4.0
        x_ref = adam_refine(x_try, steps, beta0, beta1, lr)
        vr = max_cos(x_ref)
        remain = min(loop_end, hard_end - 4.0) - time.time()
        if remain < 0.35:
            if vr < best_val - 1e-12:
                x_best = x_ref
                best_val = vr
                stall = 0
            else:
                stall += 1
            continue
        secs = min(1.4, remain)
        if mode == 4:
            secs = min(1.0, remain)
        r0 = 3e-3
        try:
            x_pol = lp_polish(x_ref, seconds=secs, r0=r0)
        except Exception:
            x_pol = x_ref
        v = max_cos(x_pol)
        if v < best_val - 1e-12:
            x_best = x_pol.copy()
            best_val = v
            stall = 0
        elif vr < best_val - 1e-12 and vr <= v:
            x_best = x_ref.copy()
            best_val = vr
            stall = 0
        else:
            stall += 1
        if time.time() > hard_end - 6.0:
            break
    secs_final = (hard_end - 2.0) - time.time()
    if secs_final > 0.5:
        secs_final = min(secs_final, 18.0)
        try:
            x_best = lp_polish(x_best, seconds=secs_final, r0=1e-3)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
