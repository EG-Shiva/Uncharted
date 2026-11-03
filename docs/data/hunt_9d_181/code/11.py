import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds, init_radius=0.008):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = init_radius
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        now = time.time()
        if now - start >= seconds or radius < 1e-14:
            break
        if seconds - (now - start) < 0.12:
            break
        G = x @ x.T
        c_all = G[iu]
        m = c_all.max()
        thr = m - max(20.0 * radius, 1e-9)
        sel_idx = np.where(c_all > thr)[0]
        if len(sel_idx) > 2500:
            part = np.argpartition(c_all, -2500)[-2500:]
            sel_idx = part
        I = iu[0][sel_idx]
        J = iu[1][sel_idx]
        csel = c_all[sel_idx]
        p = len(sel_idx)
        if p == 0:
            radius *= 0.5
            continue
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
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _exp_steps(x, steps, alpha, lr, max_step=0.02):
    n, d = x.shape
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


def _pinch_steps(x, steps, width, lr, max_step=0.02):
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


def _deepest_hole(Xkeep, rng, n_cand=8000):
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
        vr = _max_cos(xr)
    except Exception:
        vr = 1e9
        xr = None
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    vb = _max_cos(xb)
    if vr < vb:
        xb, vb = xr, vr
    n, d = xb.shape
    x_best = xb.copy()
    best_val = vb

    sigmas = [0.02, 0.05, 0.10, 0.16, 0.03, 0.08, 0.22, 0.04, 0.12, 0.06]
    it = 0
    # exploration, reserve ~22s for final polish
    while time.time() - t0 < LIMIT - 24.0:
        it += 1
        sigma = sigmas[(it - 1) % len(sigmas)]
        # randomize sigma a bit
        sigma = sigma * float(rng.uniform(0.7, 1.3))
        y = x_best.copy()
        choice = rng.random()
        if choice < 0.32:
            # global shake
            y = y + sigma * rng.standard_normal(size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif choice < 0.58:
            # contact-biased kick: overloads get hard kick
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            cnt = (G > m - 0.01).sum(axis=1).astype(float)
            k = int(rng.integers(3, 13))
            probs = cnt + 2.0
            probs /= probs.sum()
            idx = rng.choice(n, size=k, replace=False, p=probs)
            noise = rng.standard_normal(size=y.shape) * (sigma * 0.15)
            noise[idx] += rng.standard_normal(size=(k, d)) * (sigma * 1.6)
            y = y + noise
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif choice < 0.78:
            # cluster shake around random center
            center = int(rng.integers(0, n))
            dots = y[center] @ y.T
            nn = np.argsort(-dots)[:int(rng.integers(12, 30))]
            noise = np.zeros_like(y)
            noise[nn] = (sigma * 2.2) * rng.standard_normal(size=(len(nn), d))
            mask = np.ones(n, dtype=bool)
            mask[nn] = False
            noise[mask] = (sigma * 0.15) * rng.standard_normal(size=(int(mask.sum()), d))
            y = y + noise
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        else:
            # void reinsertion of loose + one jammed victim
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            cnt = (G > m - 0.01).sum(axis=1)
            k = int(rng.integers(2, 5))
            victims = np.argsort(cnt + rng.random(n) * 0.5)[:k]
            if rng.random() < 0.55:
                cnt2 = (G > m - 1e-7).sum(axis=1)
                victims[0] = int(np.argmax(cnt2 + rng.random(n) * 0.3))
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[victims] = False
            Xkeep = y[keep_mask]
            for v in victims:
                h = _deepest_hole(Xkeep, rng, n_cand=6000)
                h = h + 0.008 * rng.standard_normal(size=d)
                h /= np.linalg.norm(h)
                y[v] = h
                Xkeep = np.vstack([Xkeep, h.reshape(1, -1)])
            # small global jitter after reinsert
            y = y + (sigma * 0.25) * rng.standard_normal(size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)

        # annealed smoothing with randomized temps
        a1 = float(rng.choice([16.0, 22.0, 30.0, 40.0]))
        lr1 = 0.12 / a1 * float(rng.uniform(0.8, 1.2))
        y = _exp_steps(y, steps=280, alpha=a1, lr=lr1)
        if rng.random() < 0.5:
            a2 = float(rng.choice([60.0, 90.0, 130.0, 180.0]))
            lr2 = 0.12 / a2 * float(rng.uniform(0.8, 1.2))
            y = _exp_steps(y, steps=320, alpha=a2, lr=lr2)
        else:
            w = float(rng.choice([0.02, 0.035, 0.05]))
            y = _pinch_steps(y, steps=220, width=w, lr=float(rng.uniform(0.2, 0.35)))
            a2 = float(rng.choice([80.0, 140.0]))
            lr2 = 0.12 / a2
            y = _exp_steps(y, steps=180, alpha=a2, lr=lr2)
        # final pinch to tighten
        y = _pinch_steps(y, steps=150, width=0.025, lr=0.28)

        v = _max_cos(y)
        # LP polish only if promising (saves huge time)
        if v < best_val + 0.0015:
            remain = LIMIT - 24.0 - (time.time() - t0)
            if remain > 2.0:
                budget = min(2.2, remain)
                y = _lp_polish(y, seconds=budget, init_radius=0.008)
                v = _max_cos(y)
        if v < best_val - 1e-12:
            x_best = y
            best_val = v

        # opportunistic rattler-to-hole move on incumbent
        if it % 7 == 0 and time.time() - t0 < LIMIT - 28.0:
            G = x_best @ x_best.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            cnt = (G > m - 1e-9).sum(axis=1)
            ratt = np.where(cnt == 0)[0]
            loose = np.where(cnt <= 3)[0]
            cand_idx = ratt if len(ratt) > 0 else loose[:3] if len(loose) > 0 else np.array([], dtype=int)
            if len(cand_idx) > 0:
                yt = x_best.copy()
                keep_m = np.ones(n, dtype=bool)
                keep_m[cand_idx] = False
                Xk = yt[keep_m]
                moved = False
                for vv in cand_idx:
                    h = _deepest_hole(Xk, rng, n_cand=8000)
                    trial = float((Xk @ h).max())
                    if trial < m - 5e-7:
                        yt[vv] = h
                        Xk = np.vstack([Xk, h.reshape(1, -1)])
                        moved = True
                if moved:
                    vt = _max_cos(yt)
                    if vt <= best_val + 1e-9:
                        rem2 = LIMIT - 24.0 - (time.time() - t0)
                        if rem2 > 2.0:
                            yt = _lp_polish(yt, seconds=min(2.0, rem2), init_radius=0.003)
                            vt = _max_cos(yt)
                        if vt < best_val - 1e-12:
                            x_best, best_val = yt, vt
        if time.time() - t0 >= LIMIT - 24.0:
            break

    rem = LIMIT - (time.time() - t0) - 1.0
    if rem > 2.0:
        x_best = _lp_polish(x_best, seconds=rem, init_radius=0.004)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
