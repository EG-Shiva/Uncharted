import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _norm(x):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _polish(X, steps, lr0, lr1, tau0, tau1, mom):
    n, d = X.shape
    V = np.zeros_like(X)
    # geometric lr decay, linear tau decay
    for k in range(steps):
        f = k / max(1, steps - 1)
        if lr0 > 0 and lr1 > 0:
            lr = lr0 * ((lr1 / lr0) ** f)
        else:
            lr = lr0
        tau = tau0 * (1.0 - f) + tau1 * f
        tau = max(tau, 1e-4)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        # softmax weights, worst=1
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        S = W @ X
        dot = np.sum(S * X, axis=1, keepdims=True)
        Gr = S - dot * X
        denom = np.sum(W, axis=1, keepdims=True) * 0.5 + 1.0
        Gr = Gr / denom
        V = mom * V + (1.0 - mom) * Gr
        X = X - lr * V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def _lp_polish(x, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline:
        if radius < 1e-14:
            break
        c_all = (x @ x.T)[iu]
        m = float(c_all.max())
        sel = c_all > m - max(20.0 * radius, 1e-9)
        cnt = int(np.sum(sel))
        if cnt == 0:
            break
        # cap constraints for speed
        if cnt > 6000:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:6000]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I = iu[0][sel]
        J = iu[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 131, 9
    best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = _norm(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val

    BUDGET = 130.0
    T0a, T1a = 1.0e-4, 2.0e-7

    it = 0
    iu_tri = np.triu_indices(n, 1)
    while time.time() - t0 < BUDGET:
        it += 1
        elapsed = time.time() - t0
        frac_t = elapsed / BUDGET
        Tan = T0a * ((T1a / T0a) ** frac_t)
        # occasionally restart wandering point from best
        if cur_val > best_val + 3e-4 and rng.random() < 0.3:
            cur = best.copy()
            cur_val = best_val

        # analyse current jam
        Gc = cur @ cur.T
        np.fill_diagonal(Gc, -2.0)
        row_max = np.max(Gc, axis=1)
        mcur = float(np.max(row_max))
        order = np.argsort(row_max)  # ascending: rattlers first
        # hub weights favour large row_max
        hw = np.exp((row_max - mcur) / 0.003)
        hw = hw / np.sum(hw)
        # worst pairs
        flat = Gc[iu_tri]
        # top 30 worst
        if len(flat) > 30:
            part = np.argpartition(-flat, 30)[:30]
        else:
            part = np.arange(len(flat))
        # random choice among them
        pick = rng.choice(part)

        Y = cur.copy()
        cycle = int(rng.integers(0, 8))
        if cycle == 0:
            Y = Y + 0.008 * rng.standard_normal(Y.shape)
        elif cycle == 1:
            Y = Y + 0.022 * rng.standard_normal(Y.shape)
        elif cycle == 2:
            k = 1
            idx = rng.choice(n, k, replace=False, p=hw)
            Y[idx] += 0.15 * rng.standard_normal((k, d))
        elif cycle == 3:
            k = int(rng.integers(2, 4))
            idx = rng.choice(n, k, replace=False, p=hw)
            Y[idx] += 0.20 * rng.standard_normal((k, d))
        elif cycle == 4:
            # rattler to worst-midpoint (1 point)
            wi = int(iu_tri[0][pick]); wj = int(iu_tri[1][pick])
            r = int(order[int(rng.integers(0, min(12, n)))])
            mid = cur[wi] + cur[wj]
            nm = np.linalg.norm(mid)
            if nm < 1e-12:
                mid = rng.standard_normal(d)
                nm = np.linalg.norm(mid)
            mid = mid / nm
            sig = float(rng.uniform(0.03, 0.10))
            mid = mid + sig * rng.standard_normal(d)
            mid = mid / np.linalg.norm(mid)
            Y[r] = mid
            Y += 0.005 * rng.standard_normal(Y.shape)
        elif cycle == 5:
            # two rattlers to two worst midpoints
            pick2 = rng.choice(part, size=2, replace=False)
            for t in range(2):
                wi = int(iu_tri[0][pick2[t]]); wj = int(iu_tri[1][pick2[t]])
                r = int(order[int(rng.integers(0, min(12, n)))])
                mid = cur[wi] + cur[wj]
                nm = np.linalg.norm(mid)
                if nm < 1e-12:
                    mid = rng.standard_normal(d)
                    nm = np.linalg.norm(mid)
                mid = mid / nm
                sig = float(rng.uniform(0.03, 0.12))
                mid = mid + sig * rng.standard_normal(d)
                mid = mid / np.linalg.norm(mid)
                Y[r] = mid
            Y += 0.005 * rng.standard_normal(Y.shape)
        elif cycle == 6:
            # random-subset rigid rotation: preserves intra-subset dots
            k = int(rng.integers(8, 16))
            idx = rng.choice(n, k, replace=False)
            A = rng.standard_normal((d, 2))
            Q, _ = np.linalg.qr(A)
            u = Q[:, 0]; v2 = Q[:, 1]
            th = float(rng.uniform(0.08, 0.25)) * (1.0 if rng.random() < 0.5 else -1.0)
            co, si = np.cos(th), np.sin(th)
            for ii in idx:
                x = Y[ii]
                a = float(np.dot(x, u)); b = float(np.dot(x, v2))
                rest = x - a * u - b * v2
                a2 = a * co - b * si
                b2 = a * si + b * co
                xn = rest + a2 * u + b2 * v2
                Y[ii] = xn / np.linalg.norm(xn)
            Y += 0.004 * rng.standard_normal(Y.shape)
        else:
            Y = Y + 0.012 * rng.standard_normal(Y.shape)
            idx = rng.choice(n, 1, replace=False, p=hw)
            Y[idx] += 0.12 * rng.standard_normal((1, d))
        Y = _norm(Y)

        # randomized polish hyperparams
        lr0 = float(rng.uniform(0.04, 0.10))
        lr1 = float(rng.uniform(0.001, 0.004))
        tau0 = float(rng.uniform(0.004, 0.012))
        tau1 = float(rng.uniform(0.001, 0.003))
        mom = float(rng.uniform(0.70, 0.90))
        steps = int(rng.integers(900, 1700))
        if cycle in (2, 3, 5):
            steps = int(steps * 1.25)
        Y = _polish(Y, steps, lr0, lr1, tau0, tau1, mom)
        # short fine tail
        Y = _polish(Y, 250, 0.004, 0.0008, 0.002, 0.001, 0.85)
        v = _max_cos(Y)
        if v < best_val - 1e-12:
            best = Y.copy()
            best_val = v
        # annealing acceptance on wandering state
        if v < cur_val:
            cur = Y
            cur_val = v
        else:
            if Tan > 0 and rng.random() < np.exp((cur_val - v) / Tan):
                cur = Y
                cur_val = v
        if time.time() - t0 > BUDGET + 2:
            break

    # final long polish + LP squeeze
    try:
        best = _polish(best, 2500, 0.05, 0.0006, 0.006, 0.0012, 0.85)
    except Exception:
        pass
    remain = 178.0 - (time.time() - t0)
    if remain > 6:
        try:
            best = _lp_polish(best, time.time() + min(remain - 1.0, 32.0))
        except Exception:
            pass
    best = _norm(best)
    return best
