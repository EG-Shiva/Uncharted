import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def relax_anneal(x, steps, T0, T1, lr0, mu=0.7):
    n, d = x.shape
    x = x.copy()
    vel = np.zeros_like(x)
    best_x = x.copy()
    # initial value
    G0 = x @ x.T
    np.fill_diagonal(G0, -2.0)
    best_v = float(np.max(G0))
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 * (T1 / T0) ** frac if T0 > 0 else T1
        lr = lr0 * (1.0 - 0.4 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(np.max(G))
        E = np.exp((G - gmax) / T)
        np.fill_diagonal(E, 0.0)
        row = np.sum(E, axis=1) + 1e-12
        grad = E @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gtan = grad - dot * x
        step = gtan / row[:, None]
        nm = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        step = step * np.minimum(1.0, 0.5 / nm)
        vel = mu * vel + step
        x = x - lr * vel
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if (s & 63) == 63 or s == steps - 1:
            if gmax < best_v:
                best_v = gmax
                best_x = x.copy()
    # final check
    Gf = x @ x.T
    np.fill_diagonal(Gf, -2.0)
    vf = float(np.max(Gf))
    if vf < best_v:
        return x, vf
    return best_x, best_v


def find_void(others, rng, n_try=4000, n_refine=200):
    d = others.shape[1]
    cand = rng.randn(n_try, d)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    m = np.max(dots, axis=1)
    bi = int(np.argmin(m))
    y = cand[bi].copy()
    cur = float(m[bi])
    for k in range(n_refine):
        frac = k / max(1, n_refine - 1)
        sig = 0.05 * (0.005 / 0.05) ** frac
        prop = y + sig * rng.randn(d)
        nrm = np.linalg.norm(prop)
        if nrm < 1e-12:
            continue
        prop /= nrm
        pm = float(np.max(prop @ others.T))
        if pm < cur:
            y = prop
            cur = pm
    return y, cur


def lp_polish(x, seconds, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    t_start = time.time()
    if deadline is None:
        deadline = t_start + seconds
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() - t_start < seconds and time.time() < deadline and radius > 1e-14 and it < 60:
        it += 1
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(np.max(c))
        sel = c > m - max(20 * radius, 1e-9)
        I = iu[0][sel]
        J = iu[1][sel]
        csub = c[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
            x, best, radius = y, v, min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    deadline = t0 + 178.0
    rng = np.random.RandomState(0)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            c = np.asarray(np.load(fn), dtype=float)
            if c.shape == (180, 9):
                c = c / np.linalg.norm(c, axis=1, keepdims=True)
                cands.append((max_cos(c), c))
        except Exception:
            pass
    if len(cands) == 0:
        x0 = rng.randn(180, 9)
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        b_best = max_cos(x0)
        x_best = x0
    else:
        cands.sort(key=lambda t: t[0])
        b_best, x_best = cands[0][0], cands[0][1].copy()
    n, d = 180, 9
    x_cur = x_best.copy()
    cur_val = b_best
    explore_end = t0 + 125.0
    # if very little time, shorten
    if deadline - t0 < 150:
        explore_end = t0 + max(10.0, (deadline - t0) - 45.0)
    it = 0
    noimp = 0
    while time.time() < explore_end and time.time() < deadline - 50:
        it += 1
        progress = (time.time() - t0) / max(1e-9, (explore_end - t0))
        # per-point worst value for bias
        Gc = x_cur @ x_cur.T
        np.fill_diagonal(Gc, -2.0)
        pmax = np.max(Gc, axis=1)
        order = np.argsort(pmax)
        worst30 = order[-30:]
        rattler_like = order[:10]
        do_reinsert = (it % 5 == 0) or (rng.rand() < 0.28)
        x_new = x_cur.copy()
        if do_reinsert:
            n_re = 1 if rng.rand() < 0.7 else 2
            # pick worst points
            idxs = rng.choice(worst30, size=n_re, replace=False)
            # also occasionally replace a random low-contact point
            if rng.rand() < 0.15:
                idxs[0] = rattler_like[rng.randint(len(rattler_like))]
            for idx in idxs:
                mask = np.ones(n, dtype=bool)
                mask[idx] = False
                others = x_new[mask]
                y, _ = find_void(others, rng, n_try=4000, n_refine=180)
                x_new[idx] = y
        else:
            k = int(rng.randint(4, 12))
            idxs = rng.choice(worst30, size=min(k, 30), replace=False)
            # sometimes add a random point to move rattlers
            if rng.rand() < 0.3:
                extra = int(rng.randint(n))
                idxs = np.unique(np.append(idxs, extra))
                k = len(idxs)
            sigma = (0.030 - 0.018 * progress) * (0.7 + 0.6 * rng.rand())
            sigma = max(0.006, sigma)
            for idx in idxs:
                x_new[idx] = x_new[idx] + sigma * rng.randn(d)
                x_new[idx] /= np.linalg.norm(x_new[idx])
        # anneal params
        T0 = 0.012 + 0.018 * rng.rand()
        T1 = T0 * (0.08 + 0.08 * rng.rand())
        steps = 180 + int(rng.randint(160))
        lr0 = 0.20 + 0.18 * rng.rand()
        x_rel, v_rel = relax_anneal(x_new, steps, T0, T1, lr0, mu=0.7)
        tol = 0.0025 * (1.0 - progress)
        if v_rel < cur_val + tol:
            x_cur = x_rel
            cur_val = v_rel
            if v_rel < cur_val - 1e-12:
                pass
            noimp = 0 if v_rel < cur_val + 1e-12 else noimp + 1
            # best update
            if v_rel < b_best:
                b_best = v_rel
                x_best = x_rel.copy()
                noimp = 0
            elif v_rel <= cur_val:
                noimp = 0
            else:
                noimp += 1
        else:
            noimp += 1
        if noimp > 25:
            x_cur = x_best.copy()
            cur_val = b_best
            noimp = 0
            # small kick to avoid exact repeat
            j = int(rng.randint(n))
            x_cur[j] = x_cur[j] + 0.005 * rng.randn(d)
            x_cur[j] /= np.linalg.norm(x_cur[j])
            cur_val = max_cos(x_cur)
    # final polish of best
    remaining = deadline - time.time() - 2.0
    if remaining > 4.0:
        pol = min(38.0, remaining)
        try:
            x_best = lp_polish(x_best, pol, deadline=deadline - 1.0)
        except Exception:
            pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
