import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def _greedy_descent(x, n_iters, init_step=3e-4):
    x = _normalize(x.copy())
    best = _max_cos(x)
    step = init_step
    n = x.shape[0]
    for _ in range(n_iters):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        k = int(np.argmax(G))
        i = k // n
        j = k % n
        dot = float(G[i, j])
        vi = dot * x[i] - x[j]
        vj = dot * x[j] - x[i]
        ni = float(np.linalg.norm(vi))
        nj = float(np.linalg.norm(vj))
        if ni < 1e-14 or nj < 1e-14:
            step *= 0.5
            if step < 1e-13:
                step = init_step * 0.5
            continue
        vi /= ni
        vj /= nj
        # symmetric trial
        y = x.copy()
        y[i] = x[i] + step * vi
        y[j] = x[j] + step * vj
        y[i] /= np.linalg.norm(y[i])
        y[j] /= np.linalg.norm(y[j])
        v = _max_cos(y)
        if v < best:
            x = y
            best = v
            step = min(step * 1.25, 5e-3)
            continue
        # single-sided trials
        y2 = x.copy()
        y2[i] = x[i] + step * vi
        y2[i] /= np.linalg.norm(y2[i])
        v2 = _max_cos(y2)
        y3 = x.copy()
        y3[j] = x[j] + step * vj
        y3[j] /= np.linalg.norm(y3[j])
        v3 = _max_cos(y3)
        if v2 < best and v2 <= v3:
            x = y2
            best = v2
            step = min(step * 1.15, 5e-3)
        elif v3 < best:
            x = y3
            best = v3
            step = min(step * 1.15, 5e-3)
        else:
            step *= 0.5
            if step < 1e-13:
                step = init_step * 0.3
                # give up this round
                pass
    return x, best


def _softmax_adam(x_init, betas, lrs, steps_list):
    x = _normalize(x_init.copy())
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    t = 0
    b1p = 1.0
    b2p = 1.0
    for beta, lr, steps in zip(betas, lrs, steps_list):
        for _ in range(steps):
            t += 1
            b1p *= b1
            b2p *= b2
            G = x @ x.T
            np.fill_diagonal(G, -10.0)
            Mx = float(np.max(G))
            # stable exp
            E = np.exp(beta * (G - Mx))
            np.fill_diagonal(E, 0.0)
            S = float(np.sum(E))
            if not np.isfinite(S) or S <= 0:
                continue
            Grad = (E @ x) / S
            d = np.sum(Grad * x, axis=1, keepdims=True)
            Gt = Grad - d * x
            m = b1 * m + (1.0 - b1) * Gt
            v = b2 * v + (1.0 - b2) * (Gt * Gt)
            mhat = m / (1.0 - b1p)
            vhat = v / (1.0 - b2p)
            upd = mhat / (np.sqrt(vhat) + eps)
            dn = np.sum(upd * x, axis=1, keepdims=True)
            upd_t = upd - dn * x
            x = x - lr * upd_t
            x = _normalize(x)
    return x


def _lp_polish(x, seconds, t_deadline):
    x = _normalize(x.copy())
    try:
        best = _max_cos(x)
    except Exception:
        return x
    n, d = x.shape
    radius = 8e-4
    iu0 = np.triu_indices(n, 1)
    while time.time() < t_deadline and (time.time() - (t_deadline - seconds)) < seconds and radius > 1e-14:
        if time.time() >= t_deadline:
            break
        try:
            Gfull = x @ x.T
            c_all = Gfull[iu0]
            m = float(np.max(c_all))
            thr = m - max(20.0 * radius, 1e-9)
            sel = c_all > thr
            # cap size for speed
            p = int(np.sum(sel))
            if p < 2:
                sel = np.zeros_like(sel, dtype=bool)
                # take top 500
                kk = min(500, c_all.size)
                idx = np.argpartition(-c_all, kk - 1)[:kk]
                sel[idx] = True
                p = int(np.sum(sel))
            if p > 3000:
                # keep largest 3000
                c_sel = c_all[sel]
                # indices within selected
                order = np.argpartition(-c_sel, 3000 - 1)[:3000]
                # map back
                full_idx = np.where(sel)[0][order]
                new_sel = np.zeros_like(sel, dtype=bool)
                new_sel[full_idx] = True
                sel = new_sel
                p = 3000
            I = iu0[0][sel]
            J = iu0[1][sel]
            c_sel_vals = c_all[sel]
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate(
                [I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1
            ).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix(
                (np.tile(x.ravel(), 1), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, n * d + 1),
            ).tocsr()
            obj = np.zeros(n * d + 1)
            obj[-1] = 1.0
            res = linprog(
                obj,
                A_ub=A_ub,
                b_ub=-c_sel_vals,
                A_eq=A_eq,
                b_eq=np.zeros(n),
                bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                method="highs",
            )
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y = _normalize(y)
            vv = _max_cos(y)
            if vv < best:
                x = y
                best = vv
                radius = min(radius * 1.5, 0.02)
            else:
                radius *= 0.5
        except Exception:
            try:
                radius *= 0.5
            except Exception:
                break
            continue
    return x


def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.default_rng(0)
    np.random.seed(0)

    # load starting point
    x_best = None
    best_val = 1e9
    for fname in ("best.npy", "record.npy"):
        try:
            y = np.load(fname)
            y = np.asarray(y, dtype=float)
            if y.shape != (182, 9):
                continue
            y = _normalize(y)
            if np.any(~np.isfinite(y)):
                continue
            val = _max_cos(y)
            if val < best_val:
                best_val = val
                x_best = y.copy()
        except Exception:
            continue
    if x_best is None:
        x_best = _normalize(rng.standard_normal(size=(182, 9)))
        best_val = _max_cos(x_best)

    # quick greedy shave
    try:
        x_best, best_val = _greedy_descent(x_best, n_iters=250, init_step=3e-4)
    except Exception:
        pass

    # exploratory annealing with different perturbations
    sigmas = [0.0, 0.006, 0.012, 0.025, 0.008, 0.04, 0.015, 0.005, 0.02, 0.03]
    trial = 0
    # schedules: early broad, late focused
    while time.time() < deadline - 38.0:
        sigma = sigmas[trial % len(sigmas)]
        trial += 1
        if time.time() >= deadline - 38.0:
            break
        try:
            if sigma == 0.0:
                x0 = x_best.copy()
            else:
                pert = rng.standard_normal(size=x_best.shape)
                x0 = x_best + sigma * pert
                x0 = _normalize(x0)
            # vary beta slightly per trial for diversity
            if trial % 3 == 0:
                betas = [60, 140, 280, 550]
                lrs = [0.006, 0.0035, 0.002, 0.001]
                stepss = [500, 500, 600, 700]
            elif trial % 3 == 1:
                betas = [40, 100, 220, 450]
                lrs = [0.007, 0.004, 0.0025, 0.0012]
                stepss = [600, 600, 600, 600]
            else:
                betas = [80, 180, 350, 600]
                lrs = [0.004, 0.0025, 0.0015, 0.0008]
                stepss = [400, 500, 600, 700]
            x1 = _softmax_adam(x0, betas, lrs, stepss)
            if time.time() >= deadline - 36.0:
                # still evaluate
                v_raw = _max_cos(x1)
                if v_raw < best_val:
                    x_best = x1
                    best_val = v_raw
                break
            x1g, v1 = _greedy_descent(x1, n_iters=250, init_step=2e-4)
            if v1 < best_val - 1e-12:
                x_best = x1g
                best_val = v1
            else:
                # check raw too (greedy may have failed)
                v_raw = _max_cos(x1)
                if v_raw < best_val - 1e-12:
                    x_best = _normalize(x1)
                    best_val = v_raw
        except Exception:
            continue
        # safety: if very little time, stop
        if trial > 30:
            break

    # final jam: LP + greedy
    remaining = deadline - time.time() - 2.0
    if remaining > 6.0:
        lp_time = min(remaining * 0.75, 70.0)
        t_dl = time.time() + lp_time
        try:
            x_lp = _lp_polish(x_best, seconds=lp_time, t_deadline=t_dl)
            v_lp = _max_cos(x_lp)
            if v_lp < best_val:
                x_best = x_lp
                best_val = v_lp
        except Exception:
            pass
    # last tiny greedy to squeeze
    if time.time() < deadline - 1.0:
        try:
            x_best, best_val = _greedy_descent(x_best, n_iters=200, init_step=1e-4)
        except Exception:
            pass

    x_best = _normalize(x_best)
    return x_best
