import time
import numpy as np


def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def _row_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1), float(np.max(G))


def _random_sphere(n, d, rng):
    x = rng.standard_normal(size=(n, d))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _smooth_opt(x, steps, tau0, tau1, lr0, lr1, mom=0.9):
    n, d = x.shape
    vel = np.zeros_like(x)
    lt0, lt1 = np.log(tau0), np.log(tau1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(1, steps - 1)
        tau = float(np.exp(lt0 + (lt1 - lt0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        # per-row softmax weights
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        row = np.sum(W, axis=1, keepdims=True)
        row[row < 1e-12] = 1e-12
        grad = (W @ x) / row
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        gn = np.linalg.norm(grad, axis=1, keepdims=True)
        # clip huge grads
        big = gn > 2.0
        if np.any(big):
            grad = grad * np.where(big, 2.0 / np.maximum(gn, 1e-12), 1.0)
        vel = mom * vel + (1.0 - mom) * grad
        bc = 1.0 - mom ** (t + 1)
        step = vel / bc
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
    return x


def _lp_polish(x, deadline):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thresh)[0]
        if idx.size == 0:
            break
        if idx.size > 3000:
            order = np.argpartition(c[idx], -3000)[-3000:]
            idx = idx[order]
        I = iu[0][idx]
        J = iu[1][idx]
        csel = c[idx]
        p = idx.size
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = I[:, None] * d + np.arange(d)
        cJ = J[:, None] * d + np.arange(d)
        cols = np.concatenate([cI, cJ, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x = y
            best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best


def solve():
    import time as _time
    t0 = _time.time()
    HARD_END = t0 + 175.0
    EXPLORE_END = t0 + 122.0
    rng = np.random.default_rng(0)
    n, d = 153, 4
    best_x = None
    best_v = 1e9
    for fname in ("best.npy", "record.npy"):
        try:
            a = np.load(fname)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best_v:
                    best_v = v
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _random_sphere(n, d, rng)
        best_v = _max_cos(best_x)
    cur_x = best_x.copy()
    cur_v = best_v

    sigmas_all = [0.008, 0.02, 0.05, 0.10, 0.20, 0.35]
    it = 0
    # temperature for uphill acceptance
    while _time.time() < EXPLORE_END:
        it += 1
        mode = it % 8
        sigma = sigmas_all[(it // 2) % len(sigmas_all)]
        base = cur_x if (it % 3 != 0) else best_x
        y = base.copy()
        if mode == 0 or mode == 1:
            # global gaussian kick
            noise = rng.standard_normal(size=(n, d)) * sigma
            # tangent-ish: remove radial part
            dot = np.sum(noise * y, axis=1, keepdims=True)
            noise = noise - dot * y
            y = y + noise
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif mode == 2:
            # subset kick
            k = 12 + (it % 10)
            idx = rng.choice(n, size=k, replace=False)
            noise = rng.standard_normal(size=(k, d)) * (sigma * 2.0 + 0.05)
            dot = np.sum(noise * y[idx], axis=1, keepdims=True)
            noise = noise - dot * y[idx]
            y[idx] = y[idx] + noise
            y[idx] /= np.linalg.norm(y[idx], axis=1, keepdims=True)
        elif mode == 3:
            # replace worst point by best hole
            rm, m = _row_max(y)
            worst = int(np.argmax(rm))
            C = 2000
            cand = rng.standard_normal(size=(C, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            dots = cand @ y.T
            # don't consider self (worst point will be replaced) -> set col to -inf
            dots[:, worst] = -2.0
            worst_dot = np.max(dots, axis=1)
            pick = int(np.argmin(worst_dot))
            y[worst] = cand[pick]
        elif mode == 4:
            # rattler shake: move free points to holes
            rm, m = _row_max(y)
            ratt = np.where(rm < m - 0.004)[0]
            if ratt.size == 0:
                noise = rng.standard_normal(size=(n, d)) * sigma
                dot = np.sum(noise * y, axis=1, keepdims=True)
                noise = noise - dot * y
                y = y + noise
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            else:
                # replace up to 4 rattlers greedily
                nr = min(int(ratt.size), 4)
                sel = rng.choice(ratt, size=nr, replace=False)
                for s in sel:
                    C = 800
                    cand = rng.standard_normal(size=(C, d))
                    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                    dots = cand @ y.T
                    dots[:, s] = -2.0
                    wd = np.max(dots, axis=1)
                    pick = int(np.argmin(wd))
                    y[s] = cand[pick]
        elif mode == 5:
            # separate a max pair: push one endpoint away
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            flat = int(np.argmax(G))
            ii, jj = flat // n, flat % n
            # move ii away from jj tangentially
            direction = y[ii] - (float(y[ii] @ y[jj])) * y[jj]
            nd = np.linalg.norm(direction)
            if nd < 1e-12:
                direction = rng.standard_normal(size=(d,))
                direction -= float(direction @ y[ii]) * y[ii]
                nd = np.linalg.norm(direction)
            direction /= max(nd, 1e-12)
            step = 0.05 + rng.random() * 0.25
            y[ii] = y[ii] + step * direction
            y[ii] /= np.linalg.norm(y[ii])
            # small noise elsewhere
            noise = rng.standard_normal(size=(n, d)) * (0.01)
            y = y + noise
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif mode == 6:
            # two-point hole insertion
            rm, m = _row_max(y)
            order = np.argsort(rm)[-2:]
            for s in order:
                C = 1200
                cand = rng.standard_normal(size=(C, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                dots = cand @ y.T
                dots[:, int(s)] = -2.0
                wd = np.max(dots, axis=1)
                pick = int(np.argmin(wd))
                y[int(s)] = cand[pick]
        else:
            # random rotation-ish large kick from best
            y = best_x.copy()
            noise = rng.standard_normal(size=(n, d)) * (0.15 + 0.1 * (it % 3))
            dot = np.sum(noise * y, axis=1, keepdims=True)
            noise = noise - dot * y
            y = y + noise
            y /= np.linalg.norm(y, axis=1, keepdims=True)

        # choose smooth params cyclically
        cyc = it % 4
        if cyc == 0:
            steps, tau0, tau1, lr0, lr1, mom = 450, 0.05, 0.004, 0.08, 0.006, 0.9
        elif cyc == 1:
            steps, tau0, tau1, lr0, lr1, mom = 600, 0.03, 0.0025, 0.06, 0.004, 0.88
        elif cyc == 2:
            steps, tau0, tau1, lr0, lr1, mom = 350, 0.08, 0.008, 0.10, 0.008, 0.9
        else:
            steps, tau0, tau1, lr0, lr1, mom = 550, 0.04, 0.0018, 0.05, 0.003, 0.85

        # shorten if time low
        if _time.time() > EXPLORE_END - 8.0:
            steps = min(steps, 250)

        y = _smooth_opt(y, steps, tau0, tau1, lr0, lr1, mom)
        v = _max_cos(y)
        if v < best_v - 1e-12:
            best_v = v
            best_x = y.copy()
            cur_x = y.copy()
            cur_v = v
        elif v < cur_v:
            cur_x = y.copy()
            cur_v = v
        else:
            # uphill acceptance with decaying temperature
            frac = max(0.0, (_time.time() - t0) / max(1.0, EXPLORE_END - t0))
            T = 0.002 * (1.0 - frac) + 0.0001
            if rng.random() < np.exp((cur_v - v) / T) * 0.15:
                cur_x = y.copy()
                cur_v = v
            # reset to best periodically if stuck
            if it % 25 == 0:
                cur_x = best_x.copy()
                cur_v = best_v
        if _time.time() >= EXPLORE_END:
            break

    # fine smooth before LP
    try:
        if _time.time() < HARD_END - 30.0:
            best_x = _smooth_opt(best_x, 700, 0.012, 0.0015, 0.025, 0.002, 0.9)
            best_v = _max_cos(best_x)
    except Exception:
        pass
    # final LP polish with remaining time
    try:
        remain = HARD_END - _time.time() - 2.0
        if remain > 5.0:
            best_x, best_v = _lp_polish(best_x, HARD_END - 2.0)
    except Exception:
        pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
