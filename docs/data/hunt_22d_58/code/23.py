import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _quench_momentum(X, steps, b0, b1, lr0, lr1, mu=0.82, clip=0.05, seed_noise=0.0, rng=None):
    n, d = X.shape
    V = np.zeros_like(X)
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # global soft-max weights
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        sw = float(W.sum())
        if not np.isfinite(sw) or sw <= 0:
            break
        W /= sw
        Gr = W @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        V = mu * V + Gt
        # bias-ish scaling: keep stable
        step = lr * V
        # retangent (V already ~tangent but correct drift)
        sdot = np.sum(step * X, axis=1, keepdims=True)
        step = step - sdot * X
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        over = sn > clip
        step = np.where(over, step * (clip / sn), step)
        X = X - step
        if seed_noise > 0 and rng is not None:
            X = X + seed_noise * rng.standard_normal(X.shape) * (1.0 - f)
            # retangent noise
            nd = np.sum((X) * 0, axis=1)  # noop
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X


def _greedy_push(X, steps, e0, e1):
    n = X.shape[0]
    for s in range(steps):
        f = s / max(1, steps - 1)
        eps = e0 * (e1 / e0) ** f if e0 != e1 else e0
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        k = int(np.argmax(G))
        i = k // n
        j = k % n
        g = float(G[i, j])
        xi = X[i].copy()
        xj = X[j].copy()
        ti = xj - g * xi
        tj = xi - g * xj
        ni = float(np.linalg.norm(ti)) + 1e-12
        nj = float(np.linalg.norm(tj)) + 1e-12
        X[i] = xi - eps * ti / ni
        X[j] = xj - eps * tj / nj
        X[i] /= np.linalg.norm(X[i]) + 1e-12
        X[j] /= np.linalg.norm(X[j]) + 1e-12
    return X


def _hole_best(X, victim, rng, n_cand=4000):
    n, d = X.shape
    mask = np.ones(n, dtype=bool)
    mask[victim] = False
    Y = X[mask]
    C = rng.standard_normal((n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
    dots = C @ Y.T
    worst = dots.max(axis=1)
    bi = int(np.argmin(worst))
    X2 = X.copy()
    X2[victim] = C[bi]
    return X2


def _degrees(X, m, gap=0.004):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    cnt = (G > m - gap).sum(axis=1)
    return cnt, G


def _lp_polish(X, seconds, deadline):
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    n, d = X.shape
    best = _max_cos(X)
    radius = 8e-4
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    itlp = 0
    while True:
        if time.time() - start >= seconds:
            break
        if time.time() >= deadline:
            break
        if radius < 1e-13:
            break
        itlp += 1
        if itlp > 40:
            break
        G = X @ X.T
        c = G[iu0]
        m = float(c.max())
        window = max(20.0 * radius, 4e-5)
        order = np.argsort(-c)
        sel_list = []
        for idx in order:
            if c[idx] > m - window and len(sel_list) < 600:
                sel_list.append(int(idx))
            elif c[idx] <= m - window:
                break
            if len(sel_list) >= 600:
                break
        if len(sel_list) == 0:
            break
        sel = np.array(sel_list)
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(sel)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-12)
        v = _max_cos(y)
        if v < best - 1e-15:
            X = y
            best = v
            radius = min(radius * 1.5, 0.015)
        else:
            radius *= 0.5
    return X


def solve():
    t0 = time.time()
    TIME_LIMIT = 165.0
    deadline = t0 + TIME_LIMIT
    rng = np.random.default_rng(20261102)
    n, d = 58, 22
    best = None
    best_val = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            v = np.load(f).astype(float)
            if v.shape != (58, 22):
                continue
            v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
            c = _max_cos(v)
            if c < best_val:
                best_val = c
                best = v.copy()
        except Exception:
            continue
    if best is None:
        z = rng.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
        best = z
        best_val = _max_cos(best)

    # light initial squeeze
    try:
        best = _lp_polish(best, 2.0, deadline - 5.0)
        best_val = _max_cos(best)
    except Exception:
        pass

    hop_end = min(t0 + 128.0, deadline - 28.0)
    it = 0
    while time.time() < hop_end:
        if time.time() > deadline - 28.0:
            break
        it += 1
        if it > 120:
            break
        mode = it % 6
        y = best.copy()
        try:
            m0 = best_val
            cnt, G0 = _degrees(best, m0, gap=0.004)
            order_hi = np.argsort(-cnt)
            order_lo = np.argsort(cnt)
            if mode == 0:
                sig = 0.0035
                y = y + sig * rng.standard_normal(y.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True) + 1e-12
                y = _quench_momentum(y, 750, 70, 550, 0.005, 0.0006, mu=0.82)
                y = _greedy_push(y, 40, 0.004, 0.0005)
                y = _quench_momentum(y, 250, 800, 2200, 0.001, 0.00025, mu=0.8)
            elif mode == 1:
                sig = 0.007
                y = y + sig * rng.standard_normal(y.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True) + 1e-12
                y = _quench_momentum(y, 850, 60, 650, 0.005, 0.0005, mu=0.84)
                y = _greedy_push(y, 60, 0.005, 0.0005)
                y = _quench_momentum(y, 250, 900, 2500, 0.0009, 0.00022, mu=0.8)
            elif mode == 2:
                # rewire top hub into largest hole
                victim = int(order_hi[0])
                if it % 2 == 1 and len(order_hi) > 2:
                    victim = int(order_hi[int(rng.integers(0, 3))])
                y = _hole_best(best, victim, rng, n_cand=4000)
                # small global jitter
                y = y + 0.002 * rng.standard_normal(y.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True) + 1e-12
                y = _quench_momentum(y, 950, 90, 750, 0.004, 0.00045, mu=0.83)
                y = _quench_momentum(y, 250, 1000, 2800, 0.0008, 0.0002, mu=0.8)
            elif mode == 3:
                # worst-pair victim
                np.fill_diagonal(G0, -2.0)
                k = int(np.argmax(G0))
                i0 = k // n
                j0 = k % n
                victim = i0 if rng.random() < 0.5 else j0
                y = _hole_best(best, victim, rng, n_cand=4000)
                other = j0 if victim == i0 else i0
                y[other] = y[other] + 0.008 * rng.standard_normal(d)
                y /= np.linalg.norm(y, axis=1, keepdims=True) + 1e-12
                y = _quench_momentum(y, 900, 70, 700, 0.0045, 0.00045, mu=0.83)
                y = _greedy_push(y, 50, 0.004, 0.0004)
                y = _quench_momentum(y, 250, 1000, 2600, 0.0008, 0.0002, mu=0.8)
            elif mode == 4:
                # shake light + hub neighbourhood
                k = 4
                idx = np.concatenate([order_lo[:2], order_hi[:2]])
                idx = np.unique(idx)[:k]
                y[idx] = y[idx] + 0.025 * rng.standard_normal((len(idx), d))
                y /= np.linalg.norm(y, axis=1, keepdims=True) + 1e-12
                y = _quench_momentum(y, 850, 55, 600, 0.005, 0.0005, mu=0.84)
                y = _quench_momentum(y, 250, 900, 2400, 0.0009, 0.00022, mu=0.8)
            else:
                # double random reinit
                kk = 2
                idx = rng.choice(n, size=kk, replace=False)
                r = rng.standard_normal((kk, d))
                r /= np.linalg.norm(r, axis=1, keepdims=True) + 1e-12
                y[idx] = r
                y = y + 0.001 * rng.standard_normal(y.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True) + 1e-12
                y = _quench_momentum(y, 1000, 80, 700, 0.004, 0.0004, mu=0.83)
                y = _quench_momentum(y, 300, 1000, 2800, 0.0008, 0.0002, mu=0.8)
        except Exception:
            continue
        try:
            v = _max_cos(y)
        except Exception:
            continue
        if v < best_val - 1e-12:
            # short LP squeeze only for clear wins to save time
            try:
                if v < best_val - 5e-7:
                    y2 = _lp_polish(y, 0.9, deadline - 5.0)
                    v2 = _max_cos(y2)
                    if v2 < v:
                        y, v = y2, v2
            except Exception:
                pass
            best_val = v
            best = y.copy()

    # final sharpen: high-beta quench then LP
    remain = deadline - time.time() - 1.0
    if remain > 5:
        try:
            best = _quench_momentum(best, 400, 1500, 3200, 0.0009, 0.00018, mu=0.78)
            best_val = _max_cos(best)
        except Exception:
            pass
    remain = deadline - time.time() - 0.8
    if remain > 3:
        try:
            lp_sec = min(remain - 0.5, 16.0)
            polished = _lp_polish(best, lp_sec, deadline - 0.5)
            vp = _max_cos(polished)
            if vp < best_val:
                best = polished
                best_val = vp
        except Exception:
            pass

    best = best / (np.linalg.norm(best, axis=1, keepdims=True) + 1e-12)
    best = np.nan_to_num(best, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    best = best / nr
    return best
