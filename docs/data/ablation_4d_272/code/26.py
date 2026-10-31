import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _per_point_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)


def _lp_polish(x, seconds, r0=2e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.05:
            break
        gfull = x @ x.T
        c = gfull[iu0]
        m = float(c.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        if not np.any(sel):
            break
        I = iu0[0][sel]
        J = iu0[1][sel]
        csel = c[sel]
        if len(csel) > 4000:
            idx = np.argpartition(csel, -4000)[-4000:]
            I = I[idx]; J = J[idx]; csel = csel[idx]
        p = len(I)
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
        A_eq = coo_matrix((x.ravel(),
                           (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, n * d + 1)).tocsr()
        obj = np.zeros(n * d + 1)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel,
                          A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _smooth_anneal(X, iters=1200, k0=10.0, k1=250.0,
                   lr0=0.02, lr1=0.002, beta=0.85):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    vel = np.zeros_like(X)
    lk0 = np.log(k0); lk1 = np.log(k1)
    for it in range(iters):
        frac = it / max(iters - 1, 1)
        k = float(np.exp(lk0 + (lk1 - lk0) * frac))
        lr = float(lr0 + (lr1 - lr0) * frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(k * (G - m))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum())
        if not np.isfinite(S) or S <= 0:
            break
        Grad = (W @ X) * (2.0 / S)
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        nr = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        scale = np.minimum(1.0, 0.5 / nr)
        if float(nr.max()) > 0.5:
            Gt = Gt * scale
        vel = beta * vel - lr * Gt
        X = X + vel
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return X


def _find_holes(X, nh, ns=30000, sep=0.90):
    Z = np.random.randn(ns, 4)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    D = Z @ X.T
    m = D.max(axis=1)
    order = np.argsort(m)
    holes = []
    for idx in order:
        z = Z[idx]
        ok = True
        for h in holes:
            if float(h @ z) > sep:
                ok = False
                break
        if ok:
            holes.append(z)
            if len(holes) >= nh:
                break
    if len(holes) < nh:
        for idx in order:
            holes.append(Z[idx])
            if len(holes) >= nh:
                break
    return np.array(holes), [float((np.array(h) @ X.T).max()) for h in holes]


def solve():
    t0 = time.time()
    TIME_BUDGET = 172.0
    np.random.seed(7)

    X_best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (272, 4):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    X_best = a.copy()
        except Exception:
            pass
    if X_best is None:
        X_best = np.random.randn(272, 4)
        X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
        best_val = _max_cos(X_best)

    X_cur = X_best.copy()
    cur_val = best_val

    trial = 0
    no_accept = 0
    # explore until reserve for final polish
    while True:
        elapsed = time.time() - t0
        if elapsed > TIME_BUDGET - 48.0:
            break
        if trial > 300:
            break
        # temperature schedule: high early, low late
        frac_e = elapsed / max(TIME_BUDGET - 48.0, 1.0)
        Temp = 0.004 * (1.0 - frac_e) + 0.0004

        mode = trial % 6
        Y = None
        try:
            if mode == 0:
                # large global kick from walker
                sig = 0.22 if (trial % 12 == 0) else 0.12
                Y = X_cur + sig * np.random.randn(*X_cur.shape)
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                # balanced then sharp
                Y = _smooth_anneal(Y, iters=900, k0=8.0, k1=180.0,
                                   lr0=0.020, lr1=0.002, beta=0.85)
                Y = _smooth_anneal(Y, iters=600, k0=30.0, k1=900.0,
                                   lr0=0.010, lr1=0.001, beta=0.88)
            elif mode == 1:
                # very large kick
                sig = 0.35
                Y = X_cur + sig * np.random.randn(*X_cur.shape)
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                Y = _smooth_anneal(Y, iters=1100, k0=10.0, k1=250.0,
                                   lr0=0.022, lr1=0.002, beta=0.82)
            elif mode == 2:
                # multi-point hole surgery: move worst k to holes
                k = 6 + (trial % 5) * 2  # 6..14
                ppm = _per_point_max(X_cur)
                worst = np.argsort(ppm)[-k:]
                holes, _ = _find_holes(X_cur, k, ns=25000, sep=0.88)
                Y = X_cur.copy()
                for ii, wi in enumerate(worst):
                    h = holes[ii] + 1e-3 * np.random.randn(4)
                    h /= np.linalg.norm(h)
                    Y[int(wi)] = h
                Y = Y + 0.004 * np.random.randn(*Y.shape)
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                Y = _smooth_anneal(Y, iters=1000, k0=12.0, k1=600.0,
                                   lr0=0.015, lr1=0.0012, beta=0.85)
            elif mode == 3:
                # random block replacement
                k = 10 + (trial % 4) * 5  # 10..25
                idx = np.random.choice(272, size=k, replace=False)
                # bias half to worst
                ppm = _per_point_max(X_cur)
                worst = np.argsort(ppm)[-k // 2:]
                idx[:k // 2] = worst
                Y = X_cur.copy()
                R = np.random.randn(k, 4)
                R /= np.linalg.norm(R, axis=1, keepdims=True)
                Y[idx] = R
                Y = Y + 0.003 * np.random.randn(*Y.shape)
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                Y = _smooth_anneal(Y, iters=1200, k0=8.0, k1=300.0,
                                   lr0=0.020, lr1=0.0015, beta=0.85)
            elif mode == 4:
                # cluster breakup: pick center, scramble neighbourhood
                c = int(np.random.randint(0, 272))
                dots = X_cur @ X_cur[c]
                neigh = np.where(dots > 0.82)[0]
                if len(neigh) < 4:
                    neigh = np.argsort(dots)[-8:]
                # keep center, scramble others with large noise + holes
                holes, _ = _find_holes(X_cur, min(len(neigh), 10), ns=20000, sep=0.85)
                Y = X_cur.copy()
                cnt = 0
                for wi in neigh:
                    if int(wi) == c:
                        continue
                    if cnt < len(holes) and (np.random.rand() < 0.6):
                        h = holes[cnt] + 1e-3 * np.random.randn(4)
                        h /= np.linalg.norm(h)
                        Y[int(wi)] = h
                    else:
                        Y[int(wi)] = Y[int(wi)] + 0.5 * np.random.randn(4)
                        Y[int(wi)] /= np.linalg.norm(Y[int(wi)])
                    cnt += 1
                Y = Y + 0.003 * np.random.randn(*Y.shape)
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                Y = _smooth_anneal(Y, iters=1000, k0=10.0, k1=500.0,
                                   lr0=0.016, lr1=0.0012, beta=0.85)
            else:
                # single worst to largest hole + medium global jitter
                ppm = _per_point_max(X_cur)
                order = np.argsort(ppm)[-12:]
                wi = int(order[np.random.randint(0, len(order))])
                holes, _ = _find_holes(X_cur, 3, ns=25000, sep=0.90)
                Y = X_cur.copy()
                Y[wi] = holes[0] / np.linalg.norm(holes[0])
                Y = Y + 0.06 * np.random.randn(*Y.shape)
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                Y = _smooth_anneal(Y, iters=900, k0=15.0, k1=800.0,
                                   lr0=0.012, lr1=0.001, beta=0.87)
        except Exception:
            Y = None
        trial += 1
        if Y is None:
            continue
        try:
            v0 = _max_cos(Y)
        except Exception:
            continue
        # gate: if far worse than walker, skip expensive LP but may still move walker slightly?
        if v0 > cur_val + 0.025:
            # too bad, don't waste LP; occasionally accept as diversification
            continue
        remain_explore = TIME_BUDGET - 48.0 - (time.time() - t0)
        if remain_explore < 2.0:
            if v0 < best_val - 1e-12:
                X_best = Y; best_val = v0
            if v0 < cur_val - 1e-12:
                X_cur = Y; cur_val = v0
            continue
        lp_time = min(3.0, max(1.2, remain_explore / 8.0))
        # for very promising, give a bit more
        if v0 < best_val + 0.001:
            lp_time = min(4.0, lp_time + 1.0)
        try:
            Y2 = _lp_polish(Y, seconds=lp_time, r0=2e-3)
            v2 = _max_cos(Y2)
        except Exception:
            continue
        # update best
        if v2 < best_val - 1e-12:
            X_best = Y2.copy()
            best_val = v2
            # walker follows best
            X_cur = Y2.copy()
            cur_val = v2
            no_accept = 0
            continue
        # walker acceptance (Metropolis) to change basin
        if v2 < cur_val - 1e-12:
            X_cur = Y2.copy(); cur_val = v2; no_accept = 0
        else:
            delta = v2 - cur_val
            prob = float(np.exp(-delta / max(Temp, 1e-9))) if Temp > 0 else 0.0
            if np.random.rand() < prob:
                X_cur = Y2.copy(); cur_val = v2; no_accept = 0
            else:
                no_accept += 1
        # if stuck long, restart walker from best with big shake
        if no_accept >= 12:
            try:
                Yb = X_best + 0.30 * np.random.randn(*X_best.shape)
                Yb /= np.linalg.norm(Yb, axis=1, keepdims=True)
                Yb = _smooth_anneal(Yb, iters=700, k0=8.0, k1=200.0,
                                   lr0=0.020, lr1=0.002, beta=0.85)
                X_cur = Yb; cur_val = _max_cos(Yb)
            except Exception:
                X_cur = X_best.copy(); cur_val = best_val
            no_accept = 0

    # final thorough polish of best with remaining time
    try:
        remain = TIME_BUDGET - (time.time() - t0) - 1.5
        if remain > 4.0:
            if remain > 28.0:
                X_best = _lp_polish(X_best, seconds=remain * 0.45, r0=2e-3)
                remain = TIME_BUDGET - (time.time() - t0) - 1.5
                if remain > 4.0:
                    X_best = _lp_polish(X_best, seconds=remain, r0=7e-4)
            else:
                X_best = _lp_polish(X_best, seconds=remain, r0=8e-4)
            best_val = _max_cos(X_best)
    except Exception:
        pass
    X_best = X_best / np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best
