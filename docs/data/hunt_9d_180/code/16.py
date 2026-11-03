import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def contacts_per_point(x, thresh=1e-9):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    C = (G > m - thresh)
    # count per row (each contact counted once per row, diag is False)
    return m, np.sum(C, axis=1).astype(int), G

def lp_polish(x, seconds, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    t_start = time.time()
    if deadline is None:
        deadline = t_start + seconds
    iu = np.triu_indices(n, 1)
    while time.time() - t_start < seconds and time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(np.max(c))
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        csub = c[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def smooth_relax(x, steps, T0, T1, lr0, mu=0.8):
    n, d = x.shape
    x = x.copy()
    vel = np.zeros_like(x)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(np.max(G))
        E = np.exp((G - gmax) / T)
        np.fill_diagonal(E, 0.0)
        row = np.sum(E, axis=1, keepdims=True) + 1e-12
        grad = E @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gtan = grad - dot * x
        step = gtan / row
        nm = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        step = step * np.minimum(1.0, 0.5 / nm)
        vel = mu * vel + step
        x = x - lr * vel
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def find_void(others, rng, n_try=4000, n_refine=300):
    d = others.shape[1]
    cand = rng.randn(n_try, d)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    m = np.max(dots, axis=1)
    bi = int(np.argmin(m))
    y = cand[bi].copy()
    cur = float(m[bi])
    for k in range(n_refine):
        frac = k / max(1, n_refine - 1)
        sig = 0.08 * (0.001 / 0.08) ** frac
        prop = y + sig * rng.randn(d)
        prop /= np.linalg.norm(prop)
        pm = float(np.max(prop @ others.T))
        if pm < cur:
            y = prop
            cur = pm
    return y, cur

def solve():
    t0 = time.time()
    deadline = t0 + 176.0
    rng = np.random.RandomState(7)
    # load starts
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            c = np.asarray(np.load(fn), dtype=float)
            if c.shape == (180, 9):
                c = c / np.linalg.norm(c, axis=1, keepdims=True)
                cands.append((max_cos(c), c))
        except Exception:
            pass
    if len(cands) == 0:
        x_best = rng.randn(180, 9)
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
        b_best = max_cos(x_best)
    else:
        cands.sort(key=lambda t: t[0])
        b_best, x_best = cands[0][0], cands[0][1].copy()
    n, d = 180, 9
    x_cur = x_best.copy()
    cur_val = b_best
    # quick initial contact info
    try:
        _, cnt0, _ = contacts_per_point(x_best)
    except Exception:
        cnt0 = np.ones(n, dtype=int)
    # exploration budget: leave ~45s for final polish
    explore_end = min(deadline - 50.0, t0 + 120.0)
    it = 0
    lp_budget_total = 12.0
    lp_used = 0.0
    while time.time() < explore_end:
        it += 1
        # contact weights from current best or current
        try:
            _, cnt, _ = contacts_per_point(x_cur)
        except Exception:
            cnt = np.ones(n, dtype=int)
        # choose operation
        r = rng.rand()
        y = x_cur.copy()
        if r < 0.35:
            # contact-weighted Gaussian kick on subset
            k = int(rng.choice([5, 8, 12, 20]))
            w = (cnt.astype(float) + 1.0)
            w = w / w.sum()
            idx = rng.choice(n, size=min(k, n), replace=False, p=w)
            sigma = float(rng.choice([0.02, 0.05, 0.10, 0.18]))
            y[idx] += sigma * rng.randn(len(idx), d)
            y[idx] /= np.linalg.norm(y[idx], axis=1, keepdims=True)
            steps = 350
            T0, T1, lr0 = 0.015, 0.0015, 0.06
        elif r < 0.65:
            # void re-insertion of worst points
            k = int(rng.choice([1, 2, 3]))
            worst = np.argsort(-cnt)[:k]
            keep = np.ones(n, dtype=bool); keep[worst] = False
            others = y[keep]
            for wi in worst:
                nv_, _ = find_void(others, rng, n_try=3000, n_refine=200)
                y[wi] = nv_
                others = np.vstack([others, nv_[None, :]])
            steps = 300
            T0, T1, lr0 = 0.010, 0.0010, 0.05
        else:
            # global small shake + anneal (topology change)
            sigma = 0.015 + 0.03 * rng.rand()
            y += sigma * rng.randn(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            # also shake rattlers harder if any
            zero_idx = np.where(cnt <= 1)[0]
            if len(zero_idx) > 0:
                y[zero_idx] += 0.2 * rng.randn(len(zero_idx), d)
                y[zero_idx] /= np.linalg.norm(y[zero_idx], axis=1, keepdims=True)
            steps = 250
            T0, T1, lr0 = 0.02, 0.002, 0.07
        # time check before relax
        if time.time() > explore_end:
            break
        y = smooth_relax(y, steps, T0, T1, lr0)
        v = max_cos(y)
        # acceptance: improve current or small uphill to wander
        if v < cur_val or (rng.rand() < 0.08 and v < cur_val + 0.005):
            x_cur = y
            cur_val = v
        if v < b_best - 1e-12:
            # squeeze with short LP if budget remains
            if lp_used < lp_budget_total and time.time() < explore_end - 3:
                tlp = time.time()
                allow = min(3.0, explore_end - 3 - time.time(), deadline - 50 - time.time())
                if allow > 0.8:
                    y2 = lp_polish(y, allow)
                    v2 = max_cos(y2)
                    lp_used += time.time() - tlp
                    if v2 < v:
                        y, v = y2, v2
            if v < b_best:
                b_best = v
                x_best = y.copy()
                x_cur = y.copy()
                cur_val = v
        # occasional restart from best if wandering too high
        if cur_val > b_best + 0.01:
            x_cur = x_best.copy()
            cur_val = b_best
        # safety: stop if time low
        if time.time() > explore_end:
            break
    # final polish of best
    remaining = deadline - time.time()
    if remaining > 5:
        # spend most on LP polish
        lp_time = min(remaining - 2.0, 42.0)
        if lp_time > 1:
            try:
                x_best = lp_polish(x_best, lp_time, deadline=deadline - 1.0)
                b_best = max_cos(x_best)
            except Exception:
                pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
