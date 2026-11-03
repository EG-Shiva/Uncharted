import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _contacts(x, thresh=1e-9):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = np.max(g)
    cnt = np.sum(g > m - thresh, axis=1)
    per_max = np.max(g, axis=1)
    return m, cnt, per_max, g

def _refine(y, steps, T0, T1, lr0, lr1, t0=None, t_budget=None):
    n, d = y.shape
    # log interp for T
    lt0 = np.log(T0); lt1 = np.log(T1)
    for s in range(steps):
        if t0 is not None and (s & 127) == 0:
            if time.time() - t0 > t_budget:
                break
        frac = s / max(steps - 1, 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        lr = lr0 + (lr1 - lr0) * frac
        G = y @ y.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        # softmax weights
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        row = E.sum(axis=1, keepdims=True) + 1e-30
        Avg = (E @ y) / row
        dot = np.sum(Avg * y, axis=1, keepdims=True)
        Gt = Avg - dot * y
        y = y - lr * Gt
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
    return y

def _hard_polish(y, steps, lr, t0=None, t_budget=None):
    n, d = y.shape
    for s in range(steps):
        if t0 is not None and (s & 127) == 0:
            if time.time() - t0 > t_budget:
                break
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        # only pairs very close to max push
        thr = m - 1e-7 if s < steps // 2 else m - 1e-9
        # build force from contacts
        Mask = G > thr
        # weighted by (G - thr + eps) to focus hardest
        W = np.where(Mask, np.exp((G - m) / 0.0008), 0.0)
        np.fill_diagonal(W, 0.0)
        row = W.sum(axis=1, keepdims=True) + 1e-30
        Avg = (W @ y) / row
        dot = np.sum(Avg * y, axis=1, keepdims=True)
        Gt = Avg - dot * y
        # only move points that have contacts
        has = (row[:, 0] > 1e-18)
        step = np.zeros_like(y)
        step[has] = Gt[has]
        y = y - lr * step
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        # decay lr slightly
        lr *= 0.9995
    return y

def _deepest_hole(Xkeep, rng, n_samples=6000):
    d = Xkeep.shape[1]
    C = rng.standard_normal(size=(n_samples, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    S = C @ Xkeep.T
    mx = S.max(axis=1)
    bi = int(np.argmin(mx))
    v = C[bi].copy()
    # local refinement on softmax max
    for _ in range(150):
        dots = Xkeep @ v
        m = float(dots.max())
        T = 0.004
        w = np.exp((dots - m) / T)
        w /= (w.sum() + 1e-30)
        grad = w @ Xkeep
        g = grad - float(grad @ v) * v
        v = v - 0.12 * g
        nrm = float(np.linalg.norm(v))
        if nrm == 0:
            v = C[bi]
        else:
            v = v / nrm
    return v

def solve():
    t0 = time.time()
    BUDGET = 170.0  # must finish <180
    rng = np.random.default_rng(0)
    n, d = 123, 13
    # load start
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal(size=(n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)

    # quick light touch on start is NOT needed (already polished); keep as is
    cur_best = best.copy()
    cur_val = best_val

    it = 0
    # main basin-hopping loop
    while True:
        if time.time() - t0 > BUDGET - 8.0:
            break
        it += 1
        y = cur_best.copy()
        mode = it % 4
        try:
            m, cnt, per_max, G = _contacts(y, thresh=1e-9)
        except Exception:
            m, cnt = cur_val, np.zeros(n)
            per_max = np.zeros(n)
        if mode == 0:
            # deepest-hole reinsertion of 1-2 loosest points
            order = np.argsort(cnt)  # ascending
            k = 1 if (it % 8 != 0) else 2
            victims = order[:k].copy()
            # keep set
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[victims] = False
            Xkeep = y[keep_mask]
            for vi in victims:
                # recompute hole w.r.t current keep (including already replaced)
                v = _deepest_hole(Xkeep, rng, n_samples=5000)
                y[vi] = v
                Xkeep = np.vstack([Xkeep, v[None, :]])
            # small noise to neighbours of victims to break symmetry
            y += rng.standard_normal(size=(n, d)) * 0.008
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps = 1100
            T0, T1 = 0.015, 0.0012
            lr0, lr1 = 0.14, 0.02
        elif mode == 1:
            # worst-hub kick: worst point + its top neighbours
            worst = int(np.argmax(per_max))
            # neighbours sorted by dot with worst
            dots = G[worst].copy()
            nbrs = np.argsort(-dots)[:8]
            y[worst] += rng.standard_normal(size=(d,)) * 0.35
            for nb in nbrs[1:5]:
                y[nb] += rng.standard_normal(size=(d,)) * 0.18
            # tiny global jitter
            y += rng.standard_normal(size=(n, d)) * 0.004
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps = 1100
            T0, T1 = 0.02, 0.001
            lr0, lr1 = 0.16, 0.02
        elif mode == 2:
            # medium global shake
            sig = 0.025 if (it % 2 == 0) else 0.045
            y += rng.standard_normal(size=(n, d)) * sig
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps = 900
            T0, T1 = 0.012, 0.001
            lr0, lr1 = 0.12, 0.02
        else:
            # targeted shake of low-contact + high-contact mix
            order_lo = np.argsort(cnt)
            order_hi = np.argsort(-cnt)
            sel = np.concatenate([order_lo[:3], order_hi[:3]])
            for idx in sel:
                y[idx] += rng.standard_normal(size=(d,)) * 0.12
            y += rng.standard_normal(size=(n, d)) * 0.006
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps = 1000
            T0, T1 = 0.015, 0.0008
            lr0, lr1 = 0.14, 0.015

        # adapt steps to remaining time
        remain = BUDGET - 8.0 - (time.time() - t0)
        if remain < 12:
            steps = min(steps, 400)
        # refine
        y = _refine(y, steps, T0, T1, lr0, lr1, t0=t0, t_budget=BUDGET - 6.0)
        v = _max_cos(y)
        if v < cur_val - 1e-12:
            cur_val = v
            cur_best = y.copy()
            # extra hard tighten immediately on improvement if time allows
            if time.time() - t0 < BUDGET - 10.0:
                y2 = _hard_polish(y.copy(), 400, 0.03, t0=t0, t_budget=BUDGET - 6.0)
                v2 = _max_cos(y2)
                if v2 < cur_val:
                    cur_val = v2
                    cur_best = y2.copy()
        # early exit if time low
        if time.time() - t0 > BUDGET - 8.0:
            break

    # final tightening of the best with remaining time (max ~6s)
    try:
        if time.time() - t0 < BUDGET - 1.0:
            y = cur_best.copy()
            # short softmax cool-down then hard polish
            left = BUDGET - 2.0 - (time.time() - t0)
            if left > 1.0:
                s1 = int(min(800, left * 250))
                y = _refine(y, s1, 0.006, 0.0006, 0.08, 0.01, t0=t0, t_budget=BUDGET - 1.5)
                v = _max_cos(y)
                if v < cur_val:
                    cur_val = v
                    cur_best = y.copy()
                else:
                    y = cur_best.copy()
            if time.time() - t0 < BUDGET - 1.0:
                y2 = _hard_polish(y, 600, 0.02, t0=t0, t_budget=BUDGET - 0.5)
                v2 = _max_cos(y2)
                if v2 < cur_val:
                    cur_val = v2
                    cur_best = y2.copy()
    except Exception:
        pass

    cur_best = cur_best / np.linalg.norm(cur_best, axis=1, keepdims=True)
    # tiny shrink guard against rounding pushing max up: not needed, just return
    return cur_best
