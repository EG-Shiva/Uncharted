import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _smooth_descent(y, steps, T0, T1, lr0, lr1):
    n, d = y.shape
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = T0 + (T1 - T0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = y @ y.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        row = E.sum(axis=1, keepdims=True) + 1e-20
        Avg = (E @ y) / row
        dot = (Avg * y).sum(axis=1, keepdims=True)
        Gt = Avg - dot * y
        y = y - lr * Gt
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
    return y


def _find_hole(Ym, rng, n_sample=12000, refine=120):
    n, d = Ym.shape
    R = rng.normal(size=(n_sample, d))
    R /= np.linalg.norm(R, axis=1, keepdims=True) + 1e-20
    scores = (R @ Ym.T).max(axis=1)
    h = R[int(np.argmin(scores))].copy()
    Th = 0.015
    lr_h = 0.25
    for _ in range(refine):
        dots = Ym @ h
        m0 = float(dots.max())
        W = np.exp((dots - m0) / Th)
        W /= W.sum() + 1e-20
        g = W @ Ym
        tg = g - float(g @ h) * h
        h = h - lr_h * tg
        nh = float(np.linalg.norm(h))
        if nh == 0:
            h = R[int(rng.integers(n_sample))].copy()
        else:
            h = h / nh
    return h


def _lp_polish(x, seconds, p_cap=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        window = max(20.0 * radius, 1e-9)
        if np.count_nonzero(c_all > m - window) > p_cap:
            idx = np.argpartition(-c_all, p_cap - 1)[:p_cap]
            I = iu0[idx]
            J = iu1[idx]
            csel = c_all[idx]
            p = p_cap
        else:
            sel = c_all > m - window
            I = iu0[sel]
            J = iu1[sel]
            csel = c_all[sel]
            p = int(I.shape[0])
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            erows = np.repeat(np.arange(n), d)
            ecols = np.arange(n * d)
            A_eq = coo_matrix((x.ravel(), (erows, ecols)), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True) + 1e-20
        v = _max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return best_x


def solve():
    t0 = time.time()
    rng = np.random.default_rng(0)
    n, d = 123, 13
    X0 = None
    best_val0 = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best_val0:
                    best_val0 = v
                    X0 = a.copy()
        except Exception:
            pass
    if X0 is None:
        X0 = rng.normal(size=(n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        best_val0 = _max_cos(X0)
    best = X0.copy()
    best_val = float(best_val0)
    cur = X0.copy()
    cur_val = float(best_val0)

    loop_end = t0 + 132.0

    it = 0
    while time.time() < loop_end:
        it += 1
        Gc = cur @ cur.T
        np.fill_diagonal(Gc, -2.0)
        mc = float(Gc.max())
        # contacts with small tolerance for selection
        Cc = (Gc > mc - 5e-4).sum(axis=1)
        order_low = np.argsort(Cc)
        order_high = np.argsort(-Cc)
        y = cur.copy()
        mode = it % 4
        if mode == 0:
            k = 2 if (it % 8 == 0) else 1
            idx = order_low[:k]
            for ii in idx:
                mask = np.ones(n, dtype=bool)
                mask[int(ii)] = False
                Ym = y[mask]
                h = _find_hole(Ym, rng)
                y[int(ii)] = h
            # tiny global jitter to break symmetry
            y = y + 0.002 * rng.normal(size=(n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_descent(y, 500, 0.010, 0.0015, 0.03, 0.008)
        elif mode == 1:
            hub = int(order_high[0])
            gh = y @ y[hub]
            neigh = np.argsort(-gh)
            targets = [hub]
            for cand in neigh:
                if int(cand) != hub:
                    targets.append(int(cand))
                if len(targets) >= 9:
                    break
            # add 2 random others
            extra = rng.choice(n, 2, replace=False)
            for e in extra:
                targets.append(int(e))
            targets = list(set(targets))
            sigma = 0.030 if (it % 12 != 1) else 0.055
            noise = rng.normal(size=(len(targets), d))
            y[targets] = y[targets] + sigma * noise
            y[targets] /= np.linalg.norm(y[targets], axis=1, keepdims=True)
            y = _smooth_descent(y, 350, 0.008, 0.0012, 0.025, 0.008)
        elif mode == 2:
            k = 10
            idx = rng.choice(n, k, replace=False)
            sigma = 0.025
            y[idx] = y[idx] + sigma * rng.normal(size=(k, d))
            y[idx] /= np.linalg.norm(y[idx], axis=1, keepdims=True)
            y = _smooth_descent(y, 350, 0.009, 0.0012, 0.025, 0.008)
        else:
            sigma = 0.008
            y = y + sigma * rng.normal(size=(n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_descent(y, 300, 0.007, 0.0010, 0.020, 0.007)
        v = _max_cos(y)
        if v < best_val:
            best_val = v
            best = y.copy()
        if v < cur_val:
            cur_val = v
            cur = y.copy()
        elif v < cur_val + 0.003 and rng.random() < 0.15:
            cur_val = v
            cur = y.copy()
        # keep exploring from best occasionally
        if it % 25 == 0 and rng.random() < 0.5:
            cur = best.copy()
            cur_val = best_val
        if time.time() >= loop_end:
            break

    remain = 172.0 - (time.time() - t0)
    if remain < 5.0:
        return best / np.linalg.norm(best, axis=1, keepdims=True)
    lp_time = float(min(max(remain, 5.0), 30.0))
    try:
        polished = _lp_polish(best, lp_time)
        vp = _max_cos(polished)
        if vp < best_val:
            best = polished
            best_val = vp
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
