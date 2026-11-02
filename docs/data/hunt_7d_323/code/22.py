import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _worst_per_point(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)

def _contact_counts(x, m, tol=2e-7):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return np.sum(g > m - tol, axis=1)

def _greedy_reinsert(y, to_replace, rng, n_cand=16000):
    n, d = y.shape
    fixed_mask = np.ones(n, dtype=bool)
    fixed_mask[np.array(to_replace, dtype=int)] = False
    X_fixed = y[fixed_mask]
    acc = []  # newly placed
    for k in to_replace:
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        if acc:
            Xa = np.vstack([X_fixed] + acc)
        else:
            Xa = X_fixed
        dots = cand @ Xa.T
        worst = dots.max(axis=1)
        bi = int(np.argmin(worst))
        bp = cand[bi]
        # local refine around best: 2000 pts small cone
        ref = rng.standard_normal(size=(2000, d)) * 0.06 + bp[None, :]
        ref /= np.linalg.norm(ref, axis=1, keepdims=True)
        dw = (ref @ Xa.T).max(axis=1)
        bj = int(np.argmin(dw))
        if dw[bj] < worst[bi]:
            bp = ref[bj]
        acc.append(bp[None, :])
        y[k] = bp
    return y

def _anneal(x_init, steps, T0, T1, lr0, lr1, mu, rng, deadline=None):
    v = x_init.copy()
    n, d = v.shape
    vel = np.zeros_like(v)
    best = _max_cos(v)
    best_v = v.copy()
    for s in range(steps):
        if deadline is not None and (s % 200 == 0) and time.time() > deadline:
            break
        frac = s / max(1, steps)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = v @ v.T
        np.fill_diagonal(G, -1.0)
        m = float(G.max())
        thr = -7.0 * T
        if thr < -0.08:
            thr = -0.08
        # D = G - m
        # mask
        # use in-place to save alloc
        D = G - m
        mask = D > thr
        # weights
        W = np.zeros_like(G)
        # exp only on mask
        # D[mask]/T in [-7,0]
        W[mask] = np.exp(D[mask] / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ v
        dot = np.sum(grad * v, axis=1, keepdims=True)
        gt = grad - dot * v
        nrm = np.linalg.norm(gt, axis=1, keepdims=True) + 1e-12
        gt_n = gt / nrm
        vel = mu * vel + (1.0 - mu) * gt_n
        v = v - lr * vel
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        if s % 80 == 79 or s == steps - 1:
            cur = _max_cos(v)
            if cur < best:
                best = cur
                best_v = v.copy()
    return best_v, best

def _lp_polish(x, seconds, init_radius=2e-3, cap=2500):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return x
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(init_radius)
    start = time.time()
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start >= seconds:
            break
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            cidx = c[idx]
            part = np.argpartition(cidx, -cap)[-cap:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
            cnt = cap
        I = iuI[sel]
        J = iuJ[sel]
        csel = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        Rm = cols.reshape(p, 2 * d + 1)
        Vm = vals.reshape(p, 2 * d + 1)
        ar = np.arange(d)
        Rm[:, :d] = I[:, None] * d + ar
        Rm[:, d:2 * d] = J[:, None] * d + ar
        Rm[:, 2 * d] = n * d
        Vm[:, :d] = x[J]
        Vm[:, d:2 * d] = x[I]
        Vm[:, 2 * d] = -1.0
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    DEADLINE = t0 + 175.0
    HOP_END = t0 + 145.0
    rng = np.random.default_rng(12345)
    x0 = None
    b0 = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
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
    # precompute for speed
    while time.time() < HOP_END:
        remain = HOP_END - (time.time() - t0) - 0.0
        if remain < 0.6:
            break
        s = it % 7
        x = x_best.copy()
        if s == 0:
            sig = 0.018 * (1.0 + 0.05 * min(no_improve, 10))
            x = x + rng.standard_normal(size=x.shape) * sig
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, T0, T1, lr0, lr1, mu = 1200, 0.012, 0.0012, 0.008, 0.0006, 0.7
        elif s == 1:
            w = _worst_per_point(x)
            k = 10
            idx = np.argpartition(w, -k)[-k:]
            x[idx] += rng.standard_normal(size=(k, d)) * 0.09
            x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
            steps, T0, T1, lr0, lr1, mu = 1400, 0.010, 0.0010, 0.007, 0.0005, 0.75
        elif s == 2:
            m = best
            cc = _contact_counts(x, m, tol=2e-7)
            ratt = np.where(cc == 0)[0].tolist()
            order = np.argsort(-cc)
            torep = []
            for r in ratt:
                torep.append(int(r))
            for o in order:
                if len(torep) >= 7:
                    break
                if int(o) not in torep:
                    torep.append(int(o))
            torep = torep[:7]
            # also add one random worst
            w = _worst_per_point(x)
            iw = int(np.argmax(w))
            if iw not in torep:
                torep[-1] = iw
            x = _greedy_reinsert(x, torep, rng, n_cand=16000)
            x += rng.standard_normal(size=x.shape) * 0.002
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, T0, T1, lr0, lr1, mu = 1000, 0.008, 0.0008, 0.006, 0.0004, 0.7
        elif s == 3:
            # cluster surgery: random center + nearest neighbors
            ci = int(rng.integers(0, n))
            dots = x @ x[ci]
            k = 11
            idx = np.argpartition(dots, -k)[-k:].tolist()
            x = _greedy_reinsert(x, [int(v) for v in idx], rng, n_cand=14000)
            x += rng.standard_normal(size=x.shape) * 0.002
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, T0, T1, lr0, lr1, mu = 1100, 0.010, 0.0010, 0.007, 0.0005, 0.7
        elif s == 4:
            k = 26
            idx = rng.choice(n, size=k, replace=False)
            x[idx] += rng.standard_normal(size=(k, d)) * 0.055
            x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
            steps, T0, T1, lr0, lr1, mu = 1300, 0.012, 0.0012, 0.007, 0.0005, 0.7
        elif s == 5:
            x = x + rng.standard_normal(size=x.shape) * 0.035
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, T0, T1, lr0, lr1, mu = 1500, 0.014, 0.0010, 0.008, 0.0005, 0.75
        else:
            w = _worst_per_point(x)
            k = 6
            idx = np.argpartition(w, -k)[-k:].tolist()
            x = _greedy_reinsert(x, [int(v) for v in idx], rng, n_cand=12000)
            x += rng.standard_normal(size=x.shape) * 0.004
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, T0, T1, lr0, lr1, mu = 1200, 0.009, 0.0009, 0.006, 0.0004, 0.7
        try:
            y, vy = _anneal(x, steps, T0, T1, lr0, lr1, mu, rng, deadline=HOP_END)
        except Exception:
            it += 1
            no_improve += 1
            continue
        # vy is best inside anneal; y is best config
        if vy < best - 1e-12:
            best = float(vy)
            x_best = y.copy()
            no_improve = 0
        else:
            no_improve += 1
        it += 1
        if time.time() > HOP_END:
            break
    # final LP squeeze
    remain = DEADLINE - time.time() - 1.0
    if remain > 3:
        try:
            x_best = _lp_polish(x_best, seconds=min(remain, 27.0), init_radius=2e-3, cap=2500)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # safety: replace non-finite
    if not np.all(np.isfinite(x_best)):
        x_best = x0.copy()
    return x_best
