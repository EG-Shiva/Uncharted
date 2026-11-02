import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _worst_per_point(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)


def _contact_counts(x, m, tol=1e-9):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return np.sum(g > m - tol, axis=1)


def _greedy_reinsert(y, to_replace, rng, n_cand=8000):
    n, d = y.shape
    for k in to_replace:
        mask = np.ones(n, dtype=bool)
        mask[k] = False
        # fixed set includes already reinserted points
        Xf = y[mask]
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ Xf.T
        worst = dots.max(axis=1)
        bi = int(np.argmin(worst))
        y[k] = cand[bi]
    # one local hill climb round
    for k in to_replace:
        others = np.delete(y, k, axis=0)
        cur = y[k].copy()
        cur_w = float((others @ cur).max())
        trial = cur[None, :] + rng.standard_normal(size=(300, d)) * 0.03
        trial /= np.linalg.norm(trial, axis=1, keepdims=True)
        dw = (trial @ others.T).max(axis=1)
        bi = int(np.argmin(dw))
        if dw[bi] < cur_w:
            y[k] = trial[bi]
    return y


def _smooth(x, steps=80, lr=0.3, margin=0.015):
    n, d = x.shape
    for _ in range(steps):
        g = x @ x.T
        np.fill_diagonal(g, -2.0)
        m = float(g.max())
        thresh = m - margin
        C = g - thresh
        C[C < 0] = 0.0
        np.fill_diagonal(C, 0.0)
        Gr = C @ x
        dot = np.sum(Gr * x, axis=1, keepdims=True)
        Gt = Gr - dot * x
        mx = float(np.linalg.norm(Gt, axis=1).max()) if n > 0 else 0.0
        if mx > 0.5 and mx > 0:
            Gt *= (0.5 / mx)
        x = x - lr * Gt
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        lr *= 0.99
    return x


def _lp_polish(x, deadline, init_radius=6e-3, max_pairs=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(init_radius)
    iuI, iuJ = np.triu_indices(n, 1)
    while True:
        now = time.time()
        if now >= deadline - 0.25:
            break
        if radius < 1e-13:
            break
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > max_pairs:
            idx = np.where(sel)[0]
            cv = c[idx]
            part = np.argpartition(cv, -max_pairs)[-max_pairs:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
            cnt = max_pairs
        I = iuI[sel]
        J = iuJ[sel]
        csel = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        cr = cols.reshape(p, 2 * d + 1)
        vr = vals.reshape(p, 2 * d + 1)
        ar = np.arange(d)
        cr[:, :d] = I[:, None] * d + ar[None, :]
        cr[:, d:2 * d] = J[:, None] * d + ar[None, :]
        cr[:, 2 * d] = n * d
        vr[:, :d] = x[J]
        vr[:, d:2 * d] = x[I]
        vr[:, 2 * d] = -1.0
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        tl = deadline - time.time() - 0.08
        if tl < 0.15:
            break
        tl = min(tl, 1.5)
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": tl})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            x = y
            best = v
            radius = min(radius * 1.4, 0.02)
        else:
            radius *= 0.5
    return x, best


def solve():
    t0 = time.time()
    HOP_END = t0 + 148.0
    FINAL_END = t0 + 174.0
    rng = np.random.default_rng(0)
    x0 = None
    b0 = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < b0:
                    b0 = v
                    x0 = a.copy()
        except Exception:
            pass
    if x0 is None:
        a = rng.standard_normal(size=(323, 7))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        x0 = a
        b0 = _max_cos(a)
    x_best = x0.copy()
    best = float(b0)
    n, d = x_best.shape
    it = 0
    no_improve = 0
    while True:
        now = time.time()
        if now >= HOP_END - 1.0:
            break
        remain = HOP_END - now
        if remain < 1.2:
            break
        s = it % 7
        x = x_best.copy()
        do_smooth = 0
        if s == 0:
            sig = 0.018 * (1.0 + 0.05 * min(no_improve, 10))
            x = x + rng.standard_normal(size=x.shape) * sig
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            budget = 3.0
            init_r = 0.006
        elif s == 1:
            w = _worst_per_point(x)
            k = 35
            idx = np.argpartition(w, -k)[-k:]
            x[idx] += rng.standard_normal(size=(k, d)) * 0.07
            x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
            do_smooth = 40
            budget = 3.0
            init_r = 0.008
        elif s == 2:
            cc = _contact_counts(x, best, tol=1e-9)
            ratt = np.where(cc == 0)[0].tolist()
            order = np.argsort(-cc)
            torep = [int(v) for v in ratt]
            for o in order:
                if len(torep) >= 6:
                    break
                if int(o) not in torep:
                    torep.append(int(o))
            torep = torep[:6]
            if len(torep) == 0:
                torep = [int(v) for v in order[:6]]
            x = _greedy_reinsert(x, torep, rng, n_cand=8000)
            x += rng.standard_normal(size=x.shape) * 0.003
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            do_smooth = 80
            budget = 3.5
            init_r = 0.008
        elif s == 3:
            k = 25
            idx = rng.choice(n, size=k, replace=False)
            x[idx] += rng.standard_normal(size=(k, d)) * 0.06
            x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
            budget = 3.0
            init_r = 0.007
        elif s == 4:
            x = x + rng.standard_normal(size=x.shape) * 0.032
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            do_smooth = 40
            budget = 3.0
            init_r = 0.007
        elif s == 5:
            w = _worst_per_point(x)
            k = 12
            idx = np.argpartition(w, -k)[-k:].tolist()
            torep = [int(v) for v in idx]
            x = _greedy_reinsert(x, torep, rng, n_cand=6000)
            x += rng.standard_normal(size=x.shape) * 0.004
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            do_smooth = 80
            budget = 3.5
            init_r = 0.008
        else:
            w = _worst_per_point(x)
            k = 20
            idx = np.argpartition(w, -k)[-k:]
            x[idx] += rng.standard_normal(size=(k, d)) * 0.10
            x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
            do_smooth = 40
            budget = 3.0
            init_r = 0.009
        budget = min(budget, remain)
        if do_smooth:
            try:
                x = _smooth(x, steps=do_smooth, lr=0.3, margin=0.015)
            except Exception:
                pass
        deadline = min(time.time() + budget, HOP_END)
        try:
            y, v = _lp_polish(x, deadline, init_radius=init_r)
        except Exception:
            it += 1
            no_improve += 1
            continue
        if v < best - 1e-12:
            best = float(v)
            x_best = y
            no_improve = 0
        else:
            no_improve += 1
        it += 1
    remain = FINAL_END - time.time()
    if remain > 2.0:
        try:
            x_best, best = _lp_polish(x_best, FINAL_END, init_radius=1.5e-3, max_pairs=2500)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
