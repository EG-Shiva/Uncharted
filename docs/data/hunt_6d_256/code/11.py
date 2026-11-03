import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _anneal_track(x0, steps, a0, a1, lr):
    x = x0.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best_val = _max_cos(x)
    best_x = x.copy()
    la0, la1 = np.log(a0), np.log(a1)
    b1p, b2p = 1.0, 1.0
    for t in range(1, steps + 1):
        alpha = np.exp(la0 + (la1 - la0) * t / steps)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        W = np.exp((G - gmax) * alpha)
        np.fill_diagonal(W, 0.0)
        S = W.sum() * 0.5
        if S <= 0 or not np.isfinite(S):
            continue
        grad = (W @ x) / S
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1 - b1) * gt
        v *= b2
        v += (1 - b2) * (gt * gt)
        mhat = m / (1 - b1p)
        vhat = v / (1 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if (t % 200 == 0) or (t == steps):
            G2 = x @ x.T
            np.fill_diagonal(G2, -2.0)
            val = float(G2.max())
            if val < best_val:
                best_val = val
                best_x = x.copy()
    return best_x, best_val

def _lp_polish_deadline(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        m = cfull.max()
        sel = cfull > m - max(20 * radius, 1e-9)
        # cap size for speed
        idx = np.where(sel)[0]
        if idx.size == 0:
            break
        if idx.size > 7000:
            # keep largest
            order = np.argsort(-cfull[idx])[:7000]
            idx = idx[order]
            sel = np.zeros_like(sel)
            sel[idx] = True
        c = cfull[sel]
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _micro_polish(x, deadline, rng):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    step = 5e-4
    noimp = 0
    it = 0
    while time.time() < deadline:
        it += 1
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        flat = np.argmax(G)
        ai, aj = divmod(flat, n)
        # other max excluding ai and aj separately: compute quickly
        # mask row/col ai
        # other_max_ai = max over pairs not involving ai
        # we compute on fly per trial using precomputed overall top values
        # simple: compute other_max_ai by max of G with row ai/col ai = -2
        # do it with vector ops (n=256 cheap)
        improved = False
        # try both endpoints
        for target in (ai, aj):
            # other max not involving target
            # copy row/col masking via temporary view
            # G max excluding target:
            # take max of G after setting row target and col target to -2
            # avoid full copy: compute max of 4 blocks via slicing? just copy small 256x256
            # cheap enough once per target
            Gm = G.copy()
            Gm[target, :] = -2.0
            Gm[:, target] = -2.0
            other_max = float(Gm.max())
            # if other_max already >= best, cannot improve by moving target alone unless new_local < best but best is still other_max -> no global gain
            # still try if other_max < best (unique max involves target)
            # if other_max >= best - 1e-15, moving target alone cannot beat best; skip to save time unless both ai,aj share max
            # but second max may be other_max, so need new_local < best to improve
            # attempt proposals
            # base direction away from partner
            partner = aj if target == ai else ai
            base = x[partner] - float(np.dot(x[partner], x[target])) * x[target]
            nb = float(np.linalg.norm(base))
            if nb > 1e-12:
                base = base / nb
            else:
                base = np.zeros(d)
            # try several proposals
            for _ in range(25):
                if time.time() >= deadline:
                    break
                rnd = rng.normal(size=d)
                rnd = rnd - float(np.dot(rnd, x[target])) * x[target]
                nr = float(np.linalg.norm(rnd))
                if nr < 1e-12:
                    continue
                rnd /= nr
                direction = -base * 0.85 + rnd * 0.65
                direction = direction - float(np.dot(direction, x[target])) * x[target]
                nd = float(np.linalg.norm(direction))
                if nd < 1e-12:
                    continue
                direction /= nd
                xn = x[target] + step * direction
                xn /= np.linalg.norm(xn)
                # local max of xn vs others
                dots = x @ xn
                dots[target] = -2.0
                new_local = float(dots.max())
                cand = new_local if new_local > other_max else other_max
                if cand < best - 1e-14:
                    x[target] = xn
                    # update G row/col for next target (recompute next outer loop)
                    best = cand
                    improved = True
                    break
            if improved:
                break
        if improved:
            noimp = 0
            # keep step
        else:
            noimp += 1
            if noimp % 8 == 0:
                step *= 0.5
                if step < 1e-7:
                    step = 1e-7
                    # random kick to escape: perturb a random among top-10 worst points slightly
                    # do tiny random move that does not worsen too much
                    noimp = 0
                    # pick random point among worst 10
                    worst = G.max(axis=1)
                    order = np.argsort(-worst)[:10]
                    pick = order[int(rng.integers(10))]
                    pert = rng.normal(size=d)
                    pert = pert - float(np.dot(pert, x[pick])) * x[pick]
                    pert /= (np.linalg.norm(pert) + 1e-12)
                    xn = x[pick] + (step * 2.0) * pert
                    xn /= np.linalg.norm(xn)
                    # accept if not much worse (< best+1e-7) to diversify? keep greedy: only accept if not worse
                    dots = x @ xn
                    dots[pick] = -2.0
                    new_local = float(dots.max())
                    Gm2 = G.copy()
                    Gm2[pick, :] = -2.0
                    Gm2[:, pick] = -2.0
                    other_max2 = float(Gm2.max())
                    cand2 = new_local if new_local > other_max2 else other_max2
                    if cand2 < best + 5e-09:
                        x[pick] = xn
                        if cand2 < best:
                            best = cand2
        if step < 5e-08 and noimp > 30:
            break
    return x

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    ANNEAL_END = t0 + 118.0
    LP_END = t0 + 150.0
    MICRO_END = t0 + 172.0
    rng = _np.random.default_rng(20261103)
    x_best = None
    for fn in ("best.npy", "record.npy"):
        try:
            x_best = _np.load(fn)
            break
        except Exception:
            continue
    if x_best is None:
        x_best = rng.normal(size=(256, 6))
    x_best = _np.asarray(x_best, dtype=float)
    if x_best.shape != (256, 6):
        tmp = rng.normal(size=(256, 6))
        try:
            r = min(x_best.shape[0], 256)
            cc = min(x_best.shape[1], 6)
            tmp[:r, :cc] = x_best[:r, :cc]
        except Exception:
            pass
        x_best = tmp
    x_best = x_best / _np.linalg.norm(x_best, axis=1, keepdims=True)
    best = _max_cos(x_best)
    n, d = 256, 6
    cycle = 0
    while _time.time() < ANNEAL_END:
        cycle += 1
        rem = ANNEAL_END - _time.time()
        if rem < 3:
            break
        x = x_best.copy()
        G = x @ x.T
        _np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        worst = G.max(axis=1)
        order = _np.argsort(-worst)
        mode = cycle % 6
        if mode == 1:
            # void reinsert 2 worst points
            idx = order[:2]
            for r in idx:
                Z = rng.normal(size=(20000, 6))
                Z /= _np.linalg.norm(Z, axis=1, keepdims=True)
                S = Z @ x.T
                S[:, int(r)] = -2.0
                m2 = S.max(axis=1)
                pick = int(_np.argmin(m2))
                x[int(r)] = Z[pick]
            over = order[:8]
            x[over] += 0.05 * rng.normal(size=(8, 6))
            x /= _np.linalg.norm(x, axis=1, keepdims=True)
            steps, a0, a1, lr = 700, 50.0, 1500.0, 0.010
        elif mode == 2:
            c0 = int(rng.integers(n))
            dots = x @ x[c0]
            nearest = _np.argsort(-dots)[:12]
            x[nearest] += 0.35 * rng.normal(size=(12, 6))
            x /= _np.linalg.norm(x, axis=1, keepdims=True)
            steps, a0, a1, lr = 800, 30.0, 1000.0, 0.012
        elif mode == 3:
            x += 0.03 * rng.normal(size=x.shape)
            x /= _np.linalg.norm(x, axis=1, keepdims=True)
            steps, a0, a1, lr = 600, 80.0, 2000.0, 0.008
        elif mode == 4:
            x += 0.12 * rng.normal(size=x.shape)
            x /= _np.linalg.norm(x, axis=1, keepdims=True)
            steps, a0, a1, lr = 800, 40.0, 1200.0, 0.010
        elif mode == 5:
            pick = rng.choice(n, size=16, replace=False)
            x[pick] += 0.30 * rng.normal(size=(16, 6))
            x /= _np.linalg.norm(x, axis=1, keepdims=True)
            steps, a0, a1, lr = 700, 40.0, 1400.0, 0.011
        else:
            pick = rng.choice(n, size=32, replace=False)
            x[pick] += 0.50 * rng.normal(size=(32, 6))
            x /= _np.linalg.norm(x, axis=1, keepdims=True)
            steps, a0, a1, lr = 900, 25.0, 900.0, 0.013
        if rem < 30:
            steps = min(steps, 400)
        xb, vb = _anneal_track(x, steps, a0, a1, lr)
        if vb < best - 1e-12 and _np.all(_np.isfinite(xb)):
            x_best = xb
            best = vb
    if _time.time() < LP_END:
        try:
            x_best = _lp_polish_deadline(x_best, LP_END)
        except Exception:
            pass
    if _time.time() < MICRO_END:
        try:
            x_best = _micro_polish(x_best, MICRO_END, rng)
        except Exception:
            pass
    x_best = x_best / _np.linalg.norm(x_best, axis=1, keepdims=True)
    x_best = _np.where(_np.isfinite(x_best), x_best, 0.0)
    nr = _np.linalg.norm(x_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    x_best = x_best / nr
    return x_best
