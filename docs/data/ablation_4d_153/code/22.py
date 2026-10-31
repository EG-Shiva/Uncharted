import time
import numpy as np


def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def _row_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1)


def _random_sphere(n, d, rng):
    x = rng.standard_normal(size=(n, d))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _power_opt(x, steps, p0, p1, lr0, lr1, mom=0.86):
    n, d = x.shape
    vel = np.zeros_like(x)
    lp0, lp1 = np.log(p0), np.log(p1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(1, steps - 1)
        P = float(np.exp(lp0 + (lp1 - lp0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        Q = (1.0 + G) * 0.5
        np.clip(Q, 0.0, 1.0, out=Q)
        with np.errstate(under='ignore', over='ignore'):
            W = np.power(Q, P - 1.0)
        np.fill_diagonal(W, 0.0)
        row = np.sum(W, axis=1, keepdims=True)
        row = np.maximum(row, 1e-18)
        grad = (W @ x) / row
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        gn = np.linalg.norm(grad, axis=1, keepdims=True)
        bad = gn[:, 0] > 2.0
        if np.any(bad):
            grad[bad] *= (2.0 / gn[bad]).reshape(-1, 1)
        vel = mom * vel + (1.0 - mom) * grad
        bc = 1.0 - mom ** (t + 1)
        step = vel / max(bc, 1e-12)
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
    return x


def _softmax_opt(x, steps, tau0, tau1, lr0, lr1, mom=0.9):
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
        with np.errstate(over='ignore', under='ignore'):
            W = np.exp((G - m) / max(tau, 1e-12))
        np.fill_diagonal(W, 0.0)
        # symmetrize slightly to keep dynamics stable
        # per-row normalize
        row = np.sum(W, axis=1, keepdims=True)
        row = np.maximum(row, 1e-18)
        grad = (W @ x) / row
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        gn = np.linalg.norm(grad, axis=1, keepdims=True)
        bad = gn[:, 0] > 2.0
        if np.any(bad):
            grad[bad] *= (2.0 / gn[bad]).reshape(-1, 1)
        vel = mom * vel + (1.0 - mom) * grad
        bc = 1.0 - mom ** (t + 1)
        step = vel / max(bc, 1e-12)
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
    return x


def _refine_single(x, idx, rng, S=2500, steps=350, tau0=0.02, lr0=0.15):
    n, d = x.shape
    others = np.delete(x, idx, axis=0)
    # current value
    cur_v = x[idx].copy()
    cur_val = float(np.max(others @ cur_v))
    # random sampling for good start
    cand = rng.standard_normal(size=(S, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    vals = np.max(cand @ others.T, axis=1)
    b = int(np.argmin(vals))
    if vals[b] < cur_val:
        cur_v = cand[b]
        cur_val = float(vals[b])
    # local tangential softmax descent on this single vector
    v = cur_v.copy()
    best_v = cur_v.copy()
    best_val = cur_val
    vel = np.zeros(d)
    mom = 0.7
    for t in range(steps):
        f = t / max(1, steps - 1)
        tau = tau0 * (1.0 - 0.9 * f) + 0.002 * f
        lr = lr0 * (1.0 - f) + 0.002 * f
        dots = others @ v
        m = float(np.max(dots))
        with np.errstate(over='ignore'):
            w = np.exp((dots - m) / max(tau, 1e-12))
        s = np.sum(w)
        if s <= 0 or not np.isfinite(s):
            break
        w /= s
        g = w @ others  # weighted neighbor mean
        gt = g - np.dot(g, v) * v
        gn = float(np.linalg.norm(gt))
        if gn > 1.0:
            gt *= (1.0 / gn)
        vel = mom * vel + (1.0 - mom) * gt
        v = v - lr * vel / (1.0 - mom ** (t + 1))
        nv = float(np.linalg.norm(v))
        if nv == 0 or not np.isfinite(nv):
            v = best_v.copy()
            vel *= 0.5
            continue
        v /= nv
        val = float(np.max(others @ v))
        if val < best_val:
            best_val = val
            best_v = v.copy()
    return best_v, best_val


def _sweep_worst(x, rng, topk=10, S=2500):
    x = x.copy()
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    cur_m = float(np.max(G))
    row = np.max(G, axis=1)
    order = np.argsort(-row)[:topk]
    for idx in order:
        idx = int(idx)
        old_row = float(np.max(x[idx] @ np.delete(x, idx, axis=0).T))
        nv, nval = _refine_single(x, idx, rng, S=S, steps=300)
        if nval < old_row - 1e-13 and nval <= cur_m + 1e-12:
            x[idx] = nv
            # update cur_m lazily
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            cur_m = float(np.max(G))
            row = np.max(G, axis=1)
    return x, cur_m


def _kill_reinsert(x, rng, k, S=2500):
    x = x.copy()
    n = x.shape[0]
    row = _row_max(x)
    worst = np.argsort(-row)[:max(k, 1)]
    # mix worst with random to diversify
    kill = set(int(v) for v in worst[:k])
    while len(kill) < k:
        kill.add(int(rng.integers(0, n)))
    kill = list(kill)
    # temporarily remove? sequential greedy reinsertion
    for idx in kill:
        nv, nval = _refine_single(x, idx, rng, S=S, steps=350)
        x[idx] = nv
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _rattler_shuffle(x, rng, margin=7e-4, S=5000):
    x = x.copy()
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    row = np.max(G, axis=1)
    ratt = np.where(row < m - margin)[0]
    if len(ratt) == 0:
        ratt = np.where(row < m - 2e-4)[0]
        if len(ratt) == 0:
            return x, m
    for idx in ratt:
        idx = int(idx)
        others = np.delete(x, idx, axis=0)
        cand = rng.standard_normal(size=(S, x.shape[1]))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        vals = np.max(cand @ others.T, axis=1)
        feas = np.where(vals < m - 5e-5)[0]
        if len(feas) == 0:
            continue
        if rng.random() < 0.5:
            pick = int(feas[int(np.argmin(vals[feas]))])
        else:
            pick = int(rng.choice(feas))
        # local tune the picked feasible point a bit
        x[idx] = cand[pick]
        nv, nval = _refine_single(x, idx, rng, S=800, steps=200)
        if nval < m:
            x[idx] = nv
        else:
            x[idx] = cand[pick]
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x, _max_cos(x)


def _lp_polish(x, deadline, cap=2600):
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
        if idx.size > cap:
            order = np.argpartition(c[idx], -cap)[-cap:]
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
        if time.time() >= deadline:
            break
    return x, best


def solve():
    t0 = time.time()
    HARD_END = t0 + 175.0
    rng = np.random.default_rng(20261031)
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

    # quick initial LP to reach true local optimum
    try:
        if time.time() < HARD_END - 60:
            best_x, best_v = _lp_polish(best_x, min(time.time() + 4.0, HARD_END - 50.0))
            cur_x, cur_v = best_x.copy(), best_v
    except Exception:
        pass

    sigmas = [0.004, 0.010, 0.022, 0.05, 0.10, 0.20, 0.35]
    it = 0
    while time.time() < HARD_END - 22.0:
        it += 1
        mode = it % 8
        sigma = sigmas[rng.integers(0, len(sigmas))]
        base = best_x if (it % 3 != 0) else cur_x
        # build start point with topology-changing move
        if mode == 0:
            tmp, _ = _rattler_shuffle(base, rng, margin=7e-4, S=4000)
            x0 = tmp + 0.006 * rng.standard_normal(size=(n, d))
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            use_soft = False
            steps, p0, p1, lr0, lr1 = 900, 20.0, 240.0, 0.07, 0.003
        elif mode == 1:
            k = int(rng.integers(2, 7))
            tmp = _kill_reinsert(base, rng, k=k, S=3000)
            x0 = tmp + sigma * 0.5 * rng.standard_normal(size=(n, d))
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            use_soft = (it % 2 == 0)
            steps, p0, p1, lr0, lr1 = 1100, 25.0, 300.0, 0.08, 0.003
        elif mode == 2:
            x0 = base + sigma * rng.standard_normal(size=(n, d))
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            if rng.random() < 0.4:
                kk = int(rng.integers(1, 4))
                idx = rng.choice(n, size=kk, replace=False)
                rp = rng.standard_normal(size=(kk, d))
                rp /= np.linalg.norm(rp, axis=1, keepdims=True)
                x0[idx] = rp
            use_soft = True
            steps, p0, p1, lr0, lr1 = 1000, 18.0, 220.0, 0.07, 0.003
        elif mode == 3:
            tmp = _kill_reinsert(base, rng, k=int(rng.integers(4, 11)), S=2000)
            x0 = tmp
            use_soft = False
            steps, p0, p1, lr0, lr1 = 1200, 22.0, 280.0, 0.09, 0.003
        elif mode == 4:
            tmp, _ = _rattler_shuffle(best_x, rng, margin=5e-4, S=5000)
            x0 = tmp + 0.012 * rng.standard_normal(size=(n, d))
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            use_soft = (it % 2 == 1)
            steps, p0, p1, lr0, lr1 = 1000, 30.0, 320.0, 0.08, 0.0025
        elif mode == 5:
            x0 = base + 0.08 * rng.standard_normal(size=(n, d))
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            use_soft = False
            steps, p0, p1, lr0, lr1 = 1100, 25.0, 300.0, 0.08, 0.003
        elif mode == 6:
            # heavy kick + fresh points
            x0 = base + sigma * rng.standard_normal(size=(n, d))
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            k = int(rng.integers(2, 6))
            idx = rng.choice(n, size=k, replace=False)
            rp = rng.standard_normal(size=(k, d))
            rp /= np.linalg.norm(rp, axis=1, keepdims=True)
            x0[idx] = rp
            use_soft = True
            steps, p0, p1, lr0, lr1 = 1200, 22.0, 260.0, 0.09, 0.003
        else:
            x0 = base + 0.02 * rng.standard_normal(size=(n, d))
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            use_soft = False
            steps, p0, p1, lr0, lr1 = 900, 20.0, 240.0, 0.06, 0.0025

        if use_soft:
            tau0 = float(rng.uniform(0.015, 0.06))
            tau1 = float(rng.uniform(0.0015, 0.008))
            try:
                x1 = _softmax_opt(x0, steps, tau0, tau1, lr0, lr1, mom=0.88)
            except Exception:
                x1 = x0
        else:
            try:
                x1 = _power_opt(x0, steps, p0, p1, lr0, lr1, mom=0.86)
            except Exception:
                x1 = x0
        v1 = _max_cos(x1)
        # monotone exact hole deepening (never hurts much)
        try:
            x1h, v1h = _sweep_worst(x1, rng, topk=8, S=2500)
            if v1h <= v1 + 1e-15:
                x1, v1 = x1h, v1h
        except Exception:
            pass
        # short LP if promising
        if v1 < best_v + 0.004 and time.time() < HARD_END - 24.0:
            dl = min(time.time() + 0.8, HARD_END - 22.0)
            if dl > time.time() + 0.12:
                try:
                    x1p, v1p = _lp_polish(x1, dl)
                    if v1p < v1:
                        x1, v1 = x1p, v1p
                except Exception:
                    pass
        if v1 < best_v:
            best_x = x1.copy()
            best_v = v1
        # annealing acceptance for cur
        frac = 1.0 - (time.time() - t0) / 155.0
        frac = max(0.0, min(1.0, frac))
        tol = 0.003 * frac + 0.0003
        if v1 < cur_v + tol:
            cur_x = x1.copy()
            cur_v = v1
            if cur_v > best_v + 0.008:
                cur_x = best_x.copy()
                cur_v = best_v
        else:
            if rng.random() < 0.10:
                cur_x = best_x.copy()
                cur_v = best_v
        if time.time() >= HARD_END - 22.0:
            break

    # final thorough polish
    try:
        best_x, best_v = _lp_polish(best_x, HARD_END - 8.0)
    except Exception:
        pass
    for _ in range(5):
        if time.time() > HARD_END - 7.0:
            break
        try:
            xh, vh = _sweep_worst(best_x, rng, topk=20, S=6000)
        except Exception:
            break
        if vh < best_v - 1e-13:
            best_x, best_v = xh, vh
            try:
                best_x, best_v = _lp_polish(best_x, min(time.time() + 3.0, HARD_END - 1.5))
            except Exception:
                pass
        else:
            break
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
