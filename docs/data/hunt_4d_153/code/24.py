import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _contact_counts(x, m=None, thresh=1e-7):
    n = x.shape[0]
    iu0 = np.triu_indices(n, 1)
    g = (x @ x.T)[iu0]
    if m is None:
        m = float(np.max(g))
    tight = g > m - thresh
    cnt = np.zeros(n, dtype=int)
    np.add.at(cnt, iu0[0][tight], 1)
    np.add.at(cnt, iu0[1][tight], 1)
    return m, cnt, iu0, g

def _lp_polish(x, budget):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < budget and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(np.max(c_all))
        sel = c_all > m - max(20.0 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        csel = c_all[sel]
        p = len(I)
        if p == 0:
            break
        if p > 3000:
            # keep tightest 3000
            idx = np.argpartition(-csel, 3000)[:3000]
            I = I[idx]; J = J[idx]; csel = csel[idx]
            p = 3000
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= budget:
            break
    return x, best

def _smooth(x0, steps, beta0, beta1, lr0, lr1, Iu, Ju, freeze=None):
    n, d = x0.shape
    x = x0.copy()
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    if steps <= 1:
        lrb = 0.0; betb = 0.0
    else:
        lrb = np.log(lr1 / lr0) / (steps - 1)
        betb = np.log(beta1 / beta0) / (steps - 1)
    fr = None
    if freeze is not None:
        fr = np.zeros(n, dtype=bool); fr[freeze] = True
    for s in range(steps):
        beta = beta0 * np.exp(betb * s)
        lr = lr0 * np.exp(lrb * s)
        G = x @ x.T
        g = G[Iu, Ju]
        gmax = np.max(g)
        e = np.exp(beta * (g - gmax))
        Z = np.sum(e)
        if not np.isfinite(Z) or Z <= 0:
            break
        w = e / Z
        W = np.zeros((n, n))
        W[Iu, Ju] = w
        W = W + W.T
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        if fr is not None:
            gt[fr] = 0.0
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        c1 = 1 - b1 ** (s + 1); c2 = 1 - b2 ** (s + 1)
        step = (m1 / c1) / (np.sqrt(v1 / c2) + eps)
        # clip step to avoid blowup
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(nr, 1e-12, out=nr)
        x /= nr
    return x

def _hole(rng, x_keep, n_samp=16000):
    c = rng.normal(size=(n_samp, 4))
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    s = c @ x_keep.T
    m = np.max(s, axis=1)
    i = int(np.argmin(m))
    return c[i]

def solve():
    t0 = time.time()
    LIMIT = 172.0
    rng = np.random.default_rng(12345)
    n, d = 153, 4
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            c = np.asarray(np.load(fn), dtype=float).reshape(n, d)
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            if np.all(np.isfinite(c)):
                cands.append((c, _max_cos(c)))
        except Exception:
            pass
    if len(cands) == 0:
        c = rng.normal(size=(n, d)); c /= np.linalg.norm(c, axis=1, keepdims=True)
        cands.append((c, _max_cos(c)))
    cands.sort(key=lambda t: t[1])
    x_best, best_val = cands[0][0].copy(), float(cands[0][1])
    if len(cands) > 1:
        x_second = cands[1][0].copy(); v_second = float(cands[1][1])
    else:
        x_second = x_best.copy(); v_second = best_val
    Iu, Ju = np.triu_indices(n, 1)
    # quick snap of incumbent
    if time.time() - t0 < LIMIT - 20:
        try:
            x_best, best_val = _lp_polish(x_best, 2.5)
        except Exception:
            pass
    x_cur = x_best.copy(); v_cur = float(best_val)
    x_cur2 = x_second.copy(); v_cur2 = float(v_second)

    it = 0
    while True:
        now = time.time() - t0
        remain = LIMIT - now
        if remain < 9:
            break
        it += 1
        # temperature for Metropolis on walker
        frac = now / LIMIT
        temp = 0.0025 * (1.0 - frac) + 0.0002
        # choose base: mostly best walker, sometimes second walker
        if rng.random() < 0.15:
            base = x_cur2; bv = v_cur2
            is_second = True
        else:
            base = x_cur; bv = v_cur
            is_second = False
        try:
            _, cnt, _, _ = _contact_counts(base, thresh=1e-7)
        except Exception:
            cnt = np.full(n, 5, dtype=int)
        ratt = np.where(cnt <= 1)[0]
        low = np.where(cnt <= 3)[0]
        hubs = np.where(cnt >= 7)[0]
        mode = it % 8
        y = base.copy()
        if mode == 0:
            y += rng.normal(0, 0.012, size=y.shape)
        elif mode == 1:
            y += rng.normal(0, 0.04, size=y.shape)
            k = min(2, len(low))
            if k > 0 and rng.random() < 0.8:
                victims = rng.choice(low, size=k, replace=False)
                keep = np.ones(n, dtype=bool); keep[victims] = False
                xk = y[keep]
                for vv in victims:
                    y[vv] = _hole(rng, xk, n_samp=14000)
                    y[vv] += rng.normal(0, 0.008, size=4)
                    xk = np.vstack([xk, y[vv] / np.linalg.norm(y[vv])])
        elif mode == 2:
            # coherent cap shift
            u = rng.normal(size=4); u /= np.linalg.norm(u)
            proj = y @ u
            # cap size ~ 8-25 points
            kcap = int(rng.integers(8, 26))
            thr = np.sort(proj)[n - kcap]
            cap = np.where(proj >= thr)[0]
            # random tangent direction
            t = rng.normal(size=4); t -= (t @ u) * u
            tn = np.linalg.norm(t)
            if tn < 1e-12:
                t = np.zeros(4); t[0] = 1.0
            else:
                t /= tn
            amp = float(rng.uniform(0.05, 0.22))
            y[cap] += amp * t + rng.normal(0, 0.015, size=(len(cap), d))
            # one hole reinsert outside cap
            if len(ratt) > 0 and rng.random() < 0.6:
                vv = int(rng.choice(ratt))
                mask = np.ones(n, dtype=bool); mask[vv] = False
                y[vv] = _hole(rng, y[mask], n_samp=14000)
        elif mode == 3:
            y += rng.normal(0, 0.02, size=y.shape)
            victims = rng.choice(n, size=3, replace=False)
            mask = np.ones(n, dtype=bool); mask[victims] = False
            xk = y[mask]
            for vv in victims:
                y[vv] = _hole(rng, xk, n_samp=14000)
                xk = np.vstack([xk, y[vv] / np.linalg.norm(y[vv])])
        elif mode == 4:
            y += rng.normal(0, 0.10, size=y.shape)
            if len(hubs) > 0:
                y[hubs] += rng.normal(0, 0.04, size=(len(hubs), d))
        elif mode == 5:
            # hub-to-hole surgery
            y += rng.normal(0, 0.015, size=y.shape)
            nh = min(2, len(hubs)) if len(hubs) > 0 else 1
            if len(hubs) > 0:
                sel = hubs[rng.choice(len(hubs), size=nh, replace=False)]
            else:
                sel = rng.choice(n, size=1, replace=False)
            for vv in sel:
                mask = np.ones(n, dtype=bool); mask[vv] = False
                y[vv] = _hole(rng, y[mask], n_samp=16000)
                y[vv] += rng.normal(0, 0.005, size=4)
            if len(ratt) > 0:
                for vv in ratt[:1]:
                    mask = np.ones(n, dtype=bool); mask[vv] = False
                    y[vv] = _hole(rng, y[mask], n_samp=12000)
        elif mode == 6:
            y += rng.normal(0, 0.20, size=y.shape)
        else:
            y += rng.normal(0, 0.05, size=y.shape)
            vv = int(rng.integers(n))
            if rng.random() < 0.5:
                y[vv] = rng.normal(size=4)
            else:
                mask = np.ones(n, dtype=bool); mask[vv] = False
                y[vv] = _hole(rng, y[mask], n_samp=14000)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        if mode == 6:
            beta0, beta1 = 50.0, 700.0
            lr0, lr1 = 0.03, 0.002
            steps = 450
        elif mode in (2, 5):
            beta0, beta1 = 100.0, 1800.0
            lr0, lr1 = 0.022, 0.0015
            steps = 500
        else:
            beta0, beta1 = 80.0, 1500.0
            lr0, lr1 = 0.025, 0.002
            steps = 500
        try:
            y2 = _smooth(y, steps, beta0, beta1, lr0, lr1, Iu, Ju)
        except Exception:
            continue
        v0 = _max_cos(y2)
        # pre-filter: don't waste LP on terrible draws (relative to walker)
        if v0 > bv + 0.015:
            continue
        try:
            y2b = _smooth(y2, 180, 900.0, 3200.0, 0.007, 0.0007, Iu, Ju)
            v0b = _max_cos(y2b)
            if v0b < v0:
                y2, v0 = y2b, v0b
        except Exception:
            pass
        if v0 > bv + 0.008:
            # allow occasional sideways without LP? just continue
            continue
        lp_budget = 2.0 if v0 < bv + 0.002 else 1.0
        if remain < 16:
            lp_budget = min(lp_budget, 0.9)
        try:
            y3, v1 = _lp_polish(y2, lp_budget)
        except Exception:
            continue
        # global best update
        if v1 < best_val - 1e-12:
            x_best, best_val = y3, float(v1)
            x_cur, v_cur = y3.copy(), float(v1)
            try:
                if time.time() - t0 < LIMIT - 12:
                    x_best, best_val = _lp_polish(x_best, 1.2)
                    x_cur, v_cur = x_best.copy(), float(best_val)
            except Exception:
                pass
            continue
        # walker Metropolis update
        if is_second:
            if v1 < v_cur2 or rng.random() < np.exp((v_cur2 - v1) / temp):
                x_cur2, v_cur2 = y3, float(v1)
                # promote if second beats first walker
                if v_cur2 < v_cur:
                    x_cur, v_cur = x_cur2.copy(), float(v_cur2)
        else:
            if v1 < v_cur or rng.random() < np.exp((v_cur - v1) / temp):
                x_cur, v_cur = y3, float(v1)
        # occasional restart of second walker from best with big kick
        if it % 25 == 0 and v_cur2 > best_val + 0.004:
            try:
                tmp = x_best + rng.normal(0, 0.08, size=x_best.shape)
                tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
                tmp = _smooth(tmp, 300, 80.0, 1200.0, 0.02, 0.002, Iu, Ju)
                vtmp = _max_cos(tmp)
                if vtmp < v_cur2:
                    x_cur2, v_cur2 = tmp, float(vtmp)
            except Exception:
                pass
    try:
        rem = LIMIT - (time.time() - t0)
        if rem > 2.5:
            x_best, best_val = _lp_polish(x_best, min(rem - 1.0, 6.0))
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
