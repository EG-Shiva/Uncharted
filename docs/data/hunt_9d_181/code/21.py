import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def smooth_exp(x, steps, a0, a1, s0, s1, mu=0.9):
    m = np.zeros_like(x)
    for k in range(steps):
        frac = k / max(1, steps - 1)
        alpha = a0 + (a1 - a0) * frac
        step = s0 + (s1 - s0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        W = np.exp(alpha * (G - gmax))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        m = mu * m + (1.0 - mu) * gt
        rms = np.sqrt(np.mean(m * m)) + 1e-12
        x = x - (step / rms) * m
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def find_void(x, exclude_idx, rng, n_cand=3000):
    n, d = x.shape
    mask = np.ones(n, dtype=bool)
    mask[exclude_idx] = False
    Xo = x[mask]
    cand = rng.standard_normal((n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ Xo.T
    vals = dots.max(axis=1)
    return cand[int(np.argmin(vals))]


def lp_polish(x, seconds, init_radius=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    xb = x.copy()
    radius = init_radius
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start + 0.25 > seconds and radius < 6e-4:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu]
        mcur = c_all.max()
        thresh = max(20 * radius, 1e-9)
        sel = c_all > mcur - thresh
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 3000:
            kth = np.partition(c_all, -3000)[-3000]
            sel = c_all >= kth
            cnt = int(sel.sum())
            if cnt > 3500:
                idx = np.where(sel)[0]
                order = np.argsort(-c_all[idx])[:3500]
                mask = np.zeros_like(c_all, dtype=bool)
                mask[idx[order]] = True
                sel = mask
        I = iu[0][sel]
        J = iu[1][sel]
        csel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            best = v
            xb = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            x = xb.copy() if best < max_cos(x) else x
            radius *= 0.5
            # keep x at xb for stability
            x = xb.copy()
    return xb


def solve():
    t0 = time.time()
    BUDGET = 168.0
    EXPL = 122.0
    rng = np.random.default_rng(0)
    cand = []
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (181, 9):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cand.append(a)
        except Exception:
            pass
    if not cand:
        tmp = rng.normal(size=(181, 9))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        cand.append(tmp)
    scores = [max_cos(a) for a in cand]
    bi = int(np.argmin(scores))
    xb0 = cand[bi].copy()
    vb0 = scores[bi]
    n, d = xb0.shape

    x_best = xb0.copy()
    v_best = vb0
    x_cur = xb0.copy()
    v_cur = vb0
    elites = [(v_best, x_best.copy())]

    def add_elite(v, x):
        # keep top 5 distinct
        for vv, _ in elites:
            if abs(v - vv) < 1e-12:
                return
        elites.append((v, x.copy()))
        elites.sort(key=lambda t: t[0])
        del elites[5:]

    it = 0
    T0, T1 = 3e-04, 1e-05
    while time.time() - t0 < EXPL:
        it += 1
        elapsed = time.time() - t0
        frac = elapsed / EXPL
        T = T0 * (1 - frac) + T1 * frac
        base = x_best if rng.random() < 0.25 else x_cur
        y = base.copy()
        # contact stats of base
        G0 = base @ base.T
        np.fill_diagonal(G0, -2.0)
        vmax0 = G0.max()
        cnt = np.sum(G0 > vmax0 - 0.01, axis=1).astype(float)

        mode = rng.random()
        if mode < 0.45:
            k = int(rng.integers(3, 12))
            probs = cnt + 2.0
            probs /= probs.sum()
            idx = rng.choice(n, size=k, replace=False, p=probs)
            sig_soft = float(rng.uniform(0.002, 0.007))
            sig_hard = float(rng.uniform(0.03, 0.13))
            noise = rng.standard_normal((n, d)) * sig_soft
            noise[idx] += rng.standard_normal((k, d)) * sig_hard
            y = y + noise
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            if rng.random() < 0.55:
                j = int(np.argmin(cnt + rng.random(n) * 1.0))
                try:
                    y[j] = find_void(y, j, rng, n_cand=2500)
                except Exception:
                    pass
        elif mode < 0.70:
            sig = float(rng.uniform(0.003, 0.008))
            y = y + rng.standard_normal((n, d)) * sig
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            ntel = int(rng.integers(1, 4))
            for _ in range(ntel):
                # recompute-ish low point
                G1 = y @ y.T
                np.fill_diagonal(G1, -2.0)
                vm = G1.max()
                c1 = np.sum(G1 > vm - 0.01, axis=1)
                j = int(np.argmin(c1 + rng.random(n) * 1.0))
                try:
                    y[j] = find_void(y, j, rng, n_cand=2500)
                except Exception:
                    pass
        elif mode < 0.85:
            # cluster shake around most jammed point
            j0 = int(np.argmax(cnt + rng.random(n) * 0.5))
            dists = G0[j0]
            order = np.argsort(-dists)[:int(rng.integers(6, 18))]
            sig = float(rng.uniform(0.02, 0.09))
            y[order] += rng.standard_normal((len(order), d)) * sig
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        else:
            sig = float(rng.uniform(0.008, 0.025))
            y = y + rng.standard_normal((n, d)) * sig
            y /= np.linalg.norm(y, axis=1, keepdims=True)

        # anneal params
        a0 = float(rng.uniform(35, 80))
        a1 = float(rng.choice([150.0, 220.0, 300.0, 400.0]))
        s0 = float(rng.uniform(0.005, 0.012))
        s1 = float(rng.uniform(0.0015, 0.004))
        mu = float(rng.uniform(0.85, 0.95))
        # two stages
        try:
            y = smooth_exp(y, steps=380, a0=a0, a1=(a0 + a1) / 2, s0=s0, s1=(s0 + s1) / 2, mu=mu)
            y = smooth_exp(y, steps=620, a0=(a0 + a1) / 2, a1=a1, s0=(s0 + s1) / 2, s1=s1, mu=mu)
        except Exception:
            continue
        v = max_cos(y)
        add_elite(v, y)
        if v < v_best:
            v_best = v
            x_best = y.copy()
            x_cur = y.copy()
            v_cur = v
        elif v < v_cur or rng.random() < np.exp((v_cur - v) / max(T, 1e-12)):
            x_cur = y.copy()
            v_cur = v
        # occasional restart from best if stuck
        if it % 60 == 0 and v_cur > v_best + 0.002:
            x_cur = x_best.copy()
            v_cur = v_best
        if time.time() - t0 > EXPL:
            break

    elites.sort(key=lambda t: t[0])
    # final LP polish: polish elites in order while time remains
    best_x = elites[0][1].copy()
    best_v = max_cos(best_x)
    # also consider xb0 if better
    if vb0 < best_v:
        best_v = vb0
        best_x = xb0.copy()
    # build polish queue: best elites (up to 4) + xb0
    queue = []
    seen = set()
    for vv, xx in elites[:4]:
        queue.append(xx.copy())
    queue.append(xb0.copy())
    # deduplicate roughly by score
    # polish best first with most time
    rem = BUDGET - (time.time() - t0)
    # allocate: first candidate gets up to 20s, rest share
    time_per = [18.0, 8.0, 6.0, 5.0, 5.0]
    for qi, qx in enumerate(queue):
        if time.time() - t0 > BUDGET - 2.0:
            break
        tp = time_per[qi] if qi < len(time_per) else 5.0
        tp = min(tp, BUDGET - (time.time() - t0) - 1.0)
        if tp < 2.0:
            # still try quick polish if candidate looks promising
            if max_cos(qx) > best_v + 0.001:
                continue
            tp = max(2.0, tp)
            if tp <= 0.5:
                break
        try:
            px = lp_polish(qx, seconds=tp, init_radius=2e-3 if qi == 0 else 1e-3)
            pv = max_cos(px)
            if pv < best_v:
                best_v = pv
                best_x = px.copy()
        except Exception:
            continue
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
