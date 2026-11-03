import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds, init_radius=0.008):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
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
        if len(sel_idx) > 3500:
            part = np.argpartition(c_all, -3500)[-3500:]
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
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def smooth_anneal(x, stages, mu=0.82, max_step=0.025):
    n, d = x.shape
    v = np.zeros_like(x)
    for alpha, steps, lr, ms in stages:
        for _ in range(steps):
            G = x @ x.T
            np.fill_diagonal(G, -10.0)
            m = G.max()
            # soft-max weights
            W = np.exp(alpha * (G - m))
            np.fill_diagonal(W, 0.0)
            grad = W @ x
            dot = np.sum(grad * x, axis=1, keepdims=True)
            gt = grad - dot * x
            # normalize to keep lr comparable across alpha
            s = np.abs(gt).mean() + 1e-12
            # momentum
            v = mu * v + gt / s
            step = lr * v
            nm = np.linalg.norm(step, axis=1).max()
            lim = ms if ms is not None else max_step
            if nm > lim and nm > 0:
                step *= (lim / nm)
            x = x - step
            x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def pinch_mom(x, steps, width, lr, mu=0.8, max_step=0.02):
    n, d = x.shape
    v = np.zeros_like(x)
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
        s = np.abs(gt).mean() + 1e-12
        v = mu * v + gt / s
        step = lr * v
        nm = np.linalg.norm(step, axis=1).max()
        if nm > max_step and nm > 0:
            step *= (max_step / nm)
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def k_deepest_holes(Xkeep, rng, k, n_cand=9000):
    d = Xkeep.shape[1]
    cand = rng.standard_normal(size=(n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ Xkeep.T
    worst = dots.max(axis=1)
    order = np.argsort(worst)
    holes = []
    for idx in order:
        h = cand[idx]
        ok = True
        for q in holes:
            if float(h @ q) > 0.75:
                ok = False
                break
        if ok:
            holes.append(h)
            if len(holes) >= k:
                break
    # fallback fill
    j = 0
    while len(holes) < k:
        holes.append(cand[order[j]])
        j += 1
    return holes


def contacts_count(x, tol):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    cnt = (G > m - tol).sum(axis=1)
    return cnt, m, G


def solve():
    t0 = time.time()
    LIMIT = 172.0
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
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    vb = max_cos(xb)
    if vr < vb:
        xb, vb = xr.copy(), vr

    n, d = xb.shape
    x_best = xb.copy()
    best_val = vb
    elite = [x_best.copy()]
    elite_vals = [best_val]
    if xr is not None and abs(vr - vb) > 1e-12:
        elite.append(xr.copy())
        elite_vals.append(vr)

    sigmas = [0.02, 0.05, 0.09, 0.14, 0.03, 0.20, 0.07, 0.11, 0.04, 0.16]
    it = 0
    while time.time() - t0 < LIMIT - 32.0:
        it += 1
        sigma = sigmas[(it - 1) % len(sigmas)]
        # pick base: mostly best, sometimes second elite for crossover
        if len(elite) > 1 and rng.random() < 0.18:
            # crossover: mix best with another elite
            other = elite[int(rng.integers(1, len(elite)))]
            y = x_best.copy()
            # random subset from other (align by greedy matching? use random)
            kswap = int(rng.integers(20, 60))
            idx = rng.choice(n, size=kswap, replace=False)
            y[idx] = other[idx]
            # resolve duplicates by hole filling for worst offenders
            G0 = y @ y.T
            np.fill_diagonal(G0, -2.0)
            # jitter swapped part
            y[idx] += 0.02 * rng.standard_normal(size=(kswap, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        else:
            y = x_best.copy()

        op = rng.random()
        if op < 0.22:
            # global shake
            y = y + sigma * rng.standard_normal(size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif op < 0.45:
            # jammed-core dilation
            cnt, m, G = contacts_count(y, 1e-7)
            center = int(np.argmax(cnt + rng.random(n) * 0.5))
            dots = y[center] @ y.T
            K = int(rng.integers(10, 22))
            nn = np.argsort(-dots)[:K]
            centroid = y[nn].mean(axis=0)
            centroid /= (np.linalg.norm(centroid) + 1e-12)
            dil = float(rng.uniform(0.05, 0.22))
            y[nn] = y[nn] + dil * (y[nn] - centroid) + (sigma * 1.2) * rng.standard_normal(size=(K, d))
            # tiny background noise
            mask = np.ones(n, dtype=bool); mask[nn] = False
            y[mask] += (sigma * 0.15) * rng.standard_normal(size=(int(mask.sum()), d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif op < 0.68:
            # high-contact removal + distinct hole reinsertion
            cnt, m, G = contacts_count(y, 0.01)
            k = int(rng.integers(2, 5))
            # victims: 1 worst + (k-1) lowest-contact (rattler-like) or random
            cntT, mT, _ = contacts_count(y, 1e-7)
            worst = int(np.argmax(cntT))
            low = np.argsort(cnt)[:max(k - 1, 0)]
            victims = [worst] + [int(v) for v in low if v != worst]
            victims = victims[:k]
            # add random victim occasionally
            if rng.random() < 0.4:
                victims[-1] = int(rng.integers(0, n))
                victims = list(dict.fromkeys(victims))
            keep_mask = np.ones(n, dtype=bool); keep_mask[victims] = False
            Xkeep = y[keep_mask]
            holes = k_deepest_holes(Xkeep, rng, len(victims), n_cand=9000)
            Xacc = Xkeep
            for vv, h in zip(victims, holes):
                h = h + 0.008 * rng.standard_normal(size=d)
                h /= np.linalg.norm(h)
                y[vv] = h
                Xacc = np.vstack([Xacc, h.reshape(1, -1)])
        elif op < 0.85:
            # worst-pair separation kick + neighbourhood jitter
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            ab = np.argmax(G)
            a, b = divmod(int(ab), n)
            kick = float(rng.uniform(0.05, 0.15))
            y[a] = y[a] - kick * y[b] + sigma * 0.5 * rng.standard_normal(size=d)
            y[b] = y[b] - kick * y[a] + sigma * 0.5 * rng.standard_normal(size=d)
            # jitter their neighbours
            da = y[a] @ y.T; nb = np.argsort(-da)[1:9]
            y[nb] += (sigma * 0.4) * rng.standard_normal(size=(len(nb), d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        else:
            # local cluster shake around overloaded hub
            cnt, m, G = contacts_count(y, 1e-7)
            center = int(np.argmax(cnt))
            dots = y[center] @ y.T
            nn = np.argsort(-dots)[:20]
            noise = np.zeros_like(y)
            noise[nn] = (sigma * 2.2) * rng.standard_normal(size=(len(nn), d))
            mask = np.ones(n, dtype=bool); mask[nn] = False
            noise[mask] = (sigma * 0.2) * rng.standard_normal(size=(int(mask.sum()), d))
            y = y + noise
            y /= np.linalg.norm(y, axis=1, keepdims=True)

        # annealed smoothing with varying schedule
        if it % 3 == 0:
            stages = [(12, 120, 0.05, 0.030), (28, 140, 0.03, 0.020), (55, 120, 0.015, 0.012)]
        elif it % 3 == 1:
            stages = [(18, 150, 0.05, 0.030), (40, 150, 0.025, 0.018)]
        else:
            stages = [(10, 100, 0.06, 0.035), (22, 120, 0.035, 0.022), (45, 130, 0.018, 0.014)]
        y = smooth_anneal(y, stages)
        if it % 2 == 0:
            y = pinch_mom(y, steps=180, width=0.025, lr=0.03, max_step=0.018)
        else:
            y = pinch_mom(y, steps=220, width=0.04, lr=0.035, max_step=0.018)

        remain = LIMIT - 32.0 - (time.time() - t0)
        if remain < 2.0:
            break
        budget = min(4.0 if it % 2 == 0 else 2.5, remain)
        y = lp_polish(y, seconds=budget, init_radius=0.008)
        v = max_cos(y)
        if v < best_val - 1e-12:
            x_best = y
            best_val = v
            elite.insert(0, y.copy()); elite_vals.insert(0, v)
            elite = elite[:4]; elite_vals = elite_vals[:4]
        elif v < min(elite_vals) + 0.002 and len(elite) < 4 and rng.random() < 0.2:
            elite.append(y.copy()); elite_vals.append(v)
        # opportunistic rattler-to-hole move on current best every few iters
        if it % 6 == 0 and time.time() - t0 < LIMIT - 36.0:
            cnt0, m0, _ = contacts_count(x_best, 1e-9)
            ratt = np.where(cnt0 == 0)[0]
            if len(ratt) > 0:
                yt = x_best.copy()
                keep_m = np.ones(n, dtype=bool); keep_m[ratt] = False
                Xk = yt[keep_m]
                holes = k_deepest_holes(Xk, rng, len(ratt), n_cand=9000)
                for vv, h in zip(ratt, holes):
                    trial = float((Xk @ h).max())
                    if trial < m0 - 1e-9:
                        yt[vv] = h
                        Xk = np.vstack([Xk, h.reshape(1, -1)])
                vt = max_cos(yt)
                if vt <= best_val:
                    rb = min(3.0, LIMIT - 32.0 - (time.time() - t0))
                    if rb > 0.8:
                        yt = lp_polish(yt, seconds=rb, init_radius=0.003)
                        vt2 = max_cos(yt)
                        if vt2 < best_val - 1e-12:
                            x_best, best_val = yt, vt2
                            elite.insert(0, yt.copy()); elite_vals.insert(0, vt2)
                            elite = elite[:4]; elite_vals = elite_vals[:4]

    rem = LIMIT - (time.time() - t0) - 1.0
    if rem > 2.0:
        x_best = lp_polish(x_best, seconds=rem, init_radius=0.004)

    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
