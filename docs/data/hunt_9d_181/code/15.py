import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, deadline, init_radius=0.006, max_pairs=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = init_radius
    iu0 = np.triu_indices(n, 1)
    # cache iu arrays
    iu_r = iu0[0]
    iu_c = iu0[1]
    while True:
        now = time.time()
        if now + 0.35 >= deadline:
            break
        if radius < 1e-13:
            break
        G = x @ x.T
        c_all = G[iu_r, iu_c]
        m = c_all.max()
        thr = m - max(20.0 * radius, 1e-9)
        sel_idx = np.where(c_all > thr)[0]
        if len(sel_idx) > max_pairs:
            part = np.argpartition(c_all, -max_pairs)[-max_pairs:]
            sel_idx = part
        p = len(sel_idx)
        if p == 0:
            radius *= 0.5
            continue
        I = iu_r[sel_idx]
        J = iu_c[sel_idx]
        csel = c_all[sel_idx]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.4, 0.03)
        else:
            radius *= 0.5
    return x


def exp_smooth(x, steps, alpha, lr, max_step=0.02):
    for _ in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        step = lr * gt
        nm = np.linalg.norm(step, axis=1).max()
        if nm > max_step and nm > 0:
            step *= (max_step / nm)
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def pinch(x, steps, width, lr, max_step=0.02):
    for _ in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        W = G - (m - width)
        W[W < 0] = 0.0
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        step = lr * gt
        nm = np.linalg.norm(step, axis=1).max()
        if nm > max_step and nm > 0:
            step *= (max_step / nm)
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def deepest_hole(Xkeep, rng, n_cand=5000):
    d = Xkeep.shape[1]
    cand = rng.standard_normal(size=(n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ Xkeep.T
    worst = dots.max(axis=1)
    return cand[np.argmin(worst)]


def solve():
    t0 = time.time()
    LIMIT = 171.0
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = np.load("record.npy").astype(float)
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        vr = max_cos(xr)
    except Exception:
        vr = 1e9
        xr = None
    xb = np.asarray(xb, dtype=float)
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    vb = max_cos(xb)
    if vr < vb:
        xb, vb = xr, vr
    n, d = xb.shape
    x_best = xb.copy()
    best_val = vb

    sigmas = [0.02, 0.05, 0.09, 0.14, 0.03, 0.07, 0.18, 0.04, 0.11, 0.06]
    it = 0
    # reserve time for final polish
    while time.time() - t0 < LIMIT - 26.0:
        it += 1
        if time.time() - t0 > LIMIT - 27.0:
            break
        sigma = sigmas[(it - 1) % len(sigmas)]
        y = x_best.copy()
        # contact stats of best for targeted moves
        G0 = x_best @ x_best.T
        np.fill_diagonal(G0, -2.0)
        m0 = G0.max()
        cnt0 = (G0 > m0 - 0.01).sum(axis=1)
        cnt_tight = (G0 > m0 - 1e-7).sum(axis=1)
        choice = rng.random()
        if choice < 0.32:
            # global shake
            y = y + sigma * rng.standard_normal(size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif choice < 0.58:
            # local shake biased to most overloaded hub
            hub = int(np.argmax(cnt0 + rng.random(n) * 0.5))
            dots = y[hub] @ y.T
            nn = np.argsort(-dots)[:24]
            noise = np.zeros_like(y)
            noise[nn] = (sigma * 2.2) * rng.standard_normal(size=(len(nn), d))
            mask = np.ones(n, dtype=bool)
            mask[nn] = False
            noise[mask] = (sigma * 0.25) * rng.standard_normal(size=(int(mask.sum()), d))
            y = y + noise
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif choice < 0.82:
            # surgery: remove 1-3 most overloaded points to deepest holes
            k = int(rng.integers(1, 4))
            # order by tight contacts primarily
            order = np.argsort(-(cnt_tight * 100 + cnt0 + rng.random(n) * 0.3))
            victims = order[:k]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[victims] = False
            Xkeep = y[keep_mask]
            for v in victims:
                h = deepest_hole(Xkeep, rng, n_cand=5000)
                h = h + 0.006 * rng.standard_normal(size=d)
                h /= np.linalg.norm(h)
                y[v] = h
                Xkeep = np.vstack([Xkeep, h.reshape(1, -1)])
            # jitter neighbours of holes slightly to break symmetry
            y = y + 0.004 * rng.standard_normal(size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        else:
            # rattler / loose reinsertion + small global jitter
            loose_thr = m0 - 0.01
            cnt = cnt0
            # victims: lowest contact (rattlers first)
            k = int(rng.integers(2, 5))
            victims = np.argsort(cnt + rng.random(n) * 0.4)[:k]
            # force include a zero-contact if exists
            zeros = np.where(cnt_tight == 0)[0]
            if len(zeros) > 0:
                victims[0] = int(zeros[int(rng.integers(0, len(zeros)))])
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[victims] = False
            Xkeep = y[keep_mask]
            for v in victims:
                h = deepest_hole(Xkeep, rng, n_cand=5000)
                h = h + 0.008 * rng.standard_normal(size=d)
                h /= np.linalg.norm(h)
                y[v] = h
                Xkeep = np.vstack([Xkeep, h.reshape(1, -1)])
            y = y + (sigma * 0.3) * rng.standard_normal(size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)

        # dual-phase annealing, alternate schedules
        if it % 3 == 0:
            y = exp_smooth(y, steps=300, alpha=30.0, lr=0.004)
            y = pinch(y, steps=220, width=0.03, lr=0.28)
        elif it % 3 == 1:
            y = exp_smooth(y, steps=220, alpha=16.0, lr=0.006)
            y = pinch(y, steps=280, width=0.045, lr=0.24)
        else:
            y = exp_smooth(y, steps=350, alpha=55.0, lr=0.003)
            y = pinch(y, steps=180, width=0.02, lr=0.32)

        remain = LIMIT - 26.0 - (time.time() - t0)
        if remain < 1.5:
            break
        budget_deadline = time.time() + min(2.6, remain)
        y = lp_polish(y, deadline=budget_deadline, init_radius=0.007)
        v = max_cos(y)
        if v < best_val - 1e-12:
            x_best = y
            best_val = v
        # opportunistic rattler relocation on best every 6 iters
        if it % 6 == 0 and time.time() - t0 < LIMIT - 30.0:
            G = x_best @ x_best.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            cnt = (G > m - 1e-9).sum(axis=1)
            ratt = np.where(cnt == 0)[0]
            if len(ratt) > 0:
                yt = x_best.copy()
                keep_m = np.ones(n, dtype=bool)
                keep_m[ratt] = False
                Xk = yt[keep_m]
                for vv in ratt:
                    h = deepest_hole(Xk, rng, n_cand=6000)
                    trial = float((Xk @ h).max())
                    if trial < m - 1e-9:
                        yt[vv] = h
                        Xk = np.vstack([Xk, h.reshape(1, -1)])
                vt = max_cos(yt)
                if vt <= best_val + 1e-12:
                    dl = time.time() + min(2.0, LIMIT - 26.0 - (time.time() - t0))
                    if dl > time.time() + 0.6:
                        yt = lp_polish(yt, deadline=dl, init_radius=0.003)
                        vt2 = max_cos(yt)
                        if vt2 < best_val - 1e-12:
                            x_best, best_val = yt, vt2

    rem = LIMIT - (time.time() - t0) - 0.8
    if rem > 2.0:
        x_best = lp_polish(x_best, deadline=time.time() + rem, init_radius=0.004)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
