import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def smooth_anneal(v, alpha0, alpha1, steps, lr, beta=0.9, check_every=200):
    n, d = v.shape
    cur = v.copy()
    best_s = max_cos(cur)
    best = cur.copy()
    mom = np.zeros_like(cur)
    for s in range(steps):
        frac = s / max(steps, 1)
        alpha = alpha0 * (alpha1 / max(alpha0, 1e-12)) ** frac if alpha0 > 0 else alpha1
        G = cur @ cur.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ cur
        proj = np.sum(Grad * cur, axis=1, keepdims=True)
        Gt = Grad - proj * cur
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        Gt = Gt / rs
        mom = beta * mom + (1.0 - beta) * Gt
        bc = 1.0 - beta ** (s + 1)
        step = mom / bc
        cur_lr = lr * (1.0 - 0.4 * frac)
        cur = cur - cur_lr * step
        cur /= np.linalg.norm(cur, axis=1, keepdims=True) + 1e-24
        if (s + 1) % check_every == 0 or s == steps - 1:
            sc = max_cos(cur)
            if sc < best_s:
                best_s = sc
                best = cur.copy()
    return best, best_s


def refine_void(y0, others, steps=140):
    y = y0.copy()
    y /= np.linalg.norm(y) + 1e-24
    for t in range(steps):
        frac = t / max(steps, 1)
        alpha = 40.0 + 90.0 * frac
        dvec = others @ y
        m0 = dvec.max()
        w = np.exp(alpha * (dvec - m0))
        ws = w.sum() + 1e-12
        g = ((w[:, None] * others).sum(axis=0)) / ws
        gt = g - np.dot(g, y) * y
        lr = 0.20 * (1.0 - 0.6 * frac)
        y = y - lr * gt
        n = np.linalg.norm(y) + 1e-24
        y /= n
    return y


def find_void_replacement(others, rng, n_cand=4000):
    d = others.shape[1]
    cand = rng.standard_normal(size=(n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True) + 1e-24
    dots = cand @ others.T
    md = dots.max(axis=1)
    order = np.argsort(md)[:4]
    best_y = cand[order[0]].copy()
    best_m = float(md[order[0]])
    for b in order:
        y0 = cand[b].copy()
        y = refine_void(y0, others, steps=140)
        m = float(np.max(others @ y))
        if m < best_m:
            best_m = m
            best_y = y.copy()
    return best_y


def lp_polish(x, seconds, t0, deadline):
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-24)
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() - t0 > seconds:
            break
        cfull = (x @ x.T)[iu]
        m = float(cfull.max())
        thr = m - max(20 * radius, 1e-9)
        sel = cfull > thr
        I0, J0 = iu[0][sel], iu[1][sel]
        p0 = len(I0)
        if p0 == 0:
            break
        if p0 > 2500:
            # keep tightest 2500
            idx = np.argsort(cfull[sel])[-2500:]
            I0, J0 = I0[idx], J0[idx]
        I, J = I0, J0
        p = len(I)
        csel = cfull[sel]
        if p0 > 2500:
            csel = csel[idx]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
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
        y /= np.linalg.norm(y, axis=1, keepdims=True) + 1e-24
        v = max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if best < max_cos(x) - 1e-12:
            x = best_x.copy()
    return best_x


def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline = t0 + BUDGET
    EXPLORE_UNTIL = t0 + 144.0
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal(size=(99, 13))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (99, 13):
        tmp = rng.standard_normal(size=(99, 13))
        r = min(xb.shape[0], 99)
        c = min(xb.shape[1], 13)
        tmp[:r, :c] = xb[:r, :c]
        xb = tmp
    xb = xb / (np.linalg.norm(xb, axis=1, keepdims=True) + 1e-24)
    n, d = 99, 13
    best_val = max_cos(xb)
    best_x = xb.copy()
    cur_x = xb.copy()
    cur_val = best_val
    pop = [(best_val, best_x.copy())]

    it = 0
    sigmas = [0.02, 0.05, 0.10, 0.18, 0.30, 0.45]
    while time.time() < EXPLORE_UNTIL and it < 400:
        it += 1
        # temperature for sideways walk
        Temp = 0.0025 * (1.0 - 0.5 * (time.time() - t0) / max(EXPLORE_UNTIL - t0, 1e-9))
        Temp = max(Temp, 0.0008)
        base = cur_x
        mode = rng.random()
        sigma = sigmas[int(rng.integers(0, len(sigmas)))]
        # worst-point info from base
        G0 = base @ base.T
        np.fill_diagonal(G0, -2.0)
        rowmax = G0.max(axis=1)
        order_worst = np.argsort(rowmax)
        if mode < 0.30:
            noise = rng.standard_normal(size=(n, d)) * sigma
            x0 = base + noise
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True) + 1e-24
        elif mode < 0.60:
            # cluster surgery around hub
            hub = int(order_worst[-1])
            ksize = int(rng.integers(6, 13))
            nn = np.argsort(-G0[hub])[:ksize]
            x0 = base.copy()
            csig = sigma * 1.6 + 0.03
            x0[nn] += rng.standard_normal(size=(len(nn), d)) * csig
            # also shake hub a bit more
            x0[hub] += rng.standard_normal(size=(d,)) * (csig * 0.7)
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True) + 1e-24
        elif mode < 0.88:
            # void reinsertion of worst 1-3 points with optimized holes
            kk = int(rng.integers(1, 4))
            idx = order_worst[-kk:]
            x0 = base.copy()
            # sequentially replace to avoid duplicate voids
            for k in idx:
                others = np.delete(x0, k, axis=0)
                ynew = find_void_replacement(others, rng, n_cand=3500)
                # small jitter to break symmetry
                ynew = ynew + rng.standard_normal(size=(d,)) * 0.005
                ynew /= np.linalg.norm(ynew) + 1e-24
                x0[k] = ynew
        else:
            # large global jump
            noise = rng.standard_normal(size=(n, d)) * (0.5 + 0.2 * rng.random())
            x0 = base + noise
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True) + 1e-24

        # anneal params
        a0 = float(rng.choice([12.0, 18.0, 25.0]))
        a1 = float(rng.choice([90.0, 120.0, 160.0, 200.0]))
        steps = int(rng.integers(1100, 2300))
        lr = float(rng.uniform(0.02, 0.065))
        x1, s1 = smooth_anneal(x0, a0, a1, steps, lr, beta=0.9, check_every=250)
        # second quick focus at high alpha
        if rng.random() < 0.6 and time.time() < EXPLORE_UNTIL - 2:
            x1b, s1b = smooth_anneal(x1, a1 * 0.8, a1 * 1.5 + 30.0, 500, lr * 0.5, beta=0.9, check_every=250)
            # keep better of the two
            v1 = max_cos(x1)
            v1b = max_cos(x1b)
            if v1b < v1:
                x1 = x1b
                s1 = v1b
            else:
                s1 = v1
        else:
            s1 = max_cos(x1)

        if s1 < best_val - 1e-12:
            best_val = float(s1)
            best_x = x1.copy()
            pop.append((best_val, best_x.copy()))
            pop = sorted(pop, key=lambda z: z[0])[:5]
        # basin-hopping acceptance for current walk
        if s1 < cur_val or rng.random() < np.exp((cur_val - s1) / Temp):
            cur_x = x1.copy()
            cur_val = float(s1)
        # occasional restart to best if wandering too far
        if cur_val > best_val + 0.015 and rng.random() < 0.5:
            # jump back to best or 2nd best
            if len(pop) > 1 and rng.random() < 0.3:
                cur_x = pop[1][1].copy()
                cur_val = float(pop[1][0])
            else:
                cur_x = best_x.copy()
                cur_val = float(best_val)
        # keep pop fresh
        if len(pop) < 5 and s1 < (pop[-1][0] + 0.002):
            # avoid duplicates
            if all(max_cos((x1 + q) / 2) > 1e-6 or abs(s1 - v) > 1e-7 for v, q in pop):
                pop.append((float(s1), x1.copy()))
                pop = sorted(pop, key=lambda z: z[0])[:5]

    # final polish
    remain_deadline = deadline - 1.0
    if time.time() < remain_deadline:
        # polish best with most time
        time_left = remain_deadline - time.time()
        # reserve a bit for runner-up if ample
        if len(pop) > 1 and time_left > 30:
            best_x = lp_polish(best_x, time_left * 0.75, t0, remain_deadline)
            bv = max_cos(best_x)
            # brief polish of second candidate
            if time.time() < remain_deadline - 6:
                qx = pop[1][1].copy()
                qx = lp_polish(qx, (remain_deadline - time.time()) * 0.8, t0, remain_deadline)
                qv = max_cos(qx)
                if qv < bv:
                    best_x = qx.copy()
                    bv = qv
        else:
            best_x = lp_polish(best_x, time_left, t0, remain_deadline)
    best_x = best_x / (np.linalg.norm(best_x, axis=1, keepdims=True) + 1e-24)
    return best_x
