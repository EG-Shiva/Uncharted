import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish_bd(x, seconds, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    t_start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while True:
        if time.time() - t_start > seconds:
            break
        if deadline is not None and time.time() > deadline - 0.8:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c_all = G[iu]
        m = c_all.max()
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 2600:
            idx = np.where(sel)[0]
            # keep tightest 2600
            part = np.argpartition(c_all[idx], -2600)[-2600:]
            sel[:] = False
            sel[idx[part]] = True
            cnt = 2600
        I = iu[0][sel]
        J = iu[1][sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols_b = np.empty((p, 2 * d + 1), dtype=np.int64)
        cols_b[:, :d] = I[:, None] * d + ar[None, :]
        cols_b[:, d:2 * d] = J[:, None] * d + ar[None, :]
        cols_b[:, 2 * d] = n * d
        vals_b = np.empty((p, 2 * d + 1), dtype=np.float64)
        vals_b[:, :d] = x[J]
        vals_b[:, d:2 * d] = x[I]
        vals_b[:, 2 * d] = -1.0
        try:
            A_ub = coo_matrix((vals_b.ravel(), (rows, cols_b.ravel())), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        b_ub = -c_all[sel]
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nrm = np.linalg.norm(y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        y /= nrm
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def anneal_run(v, steps, T0, T1, lr0, lr1, beta=0.85):
    mom = np.zeros_like(v)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 * (1.0 - frac) + T1 * frac
        lr = lr0 * (1.0 - frac) + lr1 * frac
        G = v @ v.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        E = (G - m) / T
        np.clip(E, -50.0, 20.0, out=E)
        np.exp(E, out=E)
        np.fill_diagonal(E, 0.0)
        grad = E @ v
        dot = (grad * v).sum(axis=1, keepdims=True)
        grad -= dot * v
        rms = np.sqrt(np.mean(grad * grad)) + 1e-30
        grad /= rms
        mom = beta * mom + (1.0 - beta) * grad
        bc = 1.0 - beta ** (s + 1)
        step = mom * (lr / bc)
        rn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        cap = lr * 3.0 + 1e-12
        scale = np.minimum(1.0, cap / rn)
        step *= scale
        v = v - step
        v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-30)
    return v


def largest_voids(X_fixed, n_need, rng, n_cand=4000):
    d = X_fixed.shape[1]
    cand = rng.standard_normal(size=(n_cand, d))
    cand /= (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-30)
    dots = cand @ X_fixed.T
    scores = dots.max(axis=1)
    used = np.zeros(n_cand, dtype=bool)
    rem = scores.copy()
    out = []
    for _ in range(n_need):
        tmp = rem.copy()
        tmp[used] = np.inf
        i = int(np.argmin(tmp))
        out.append(cand[i])
        used[i] = True
        dd = cand @ cand[i]
        rem = np.maximum(rem, dd)
    return np.array(out)


def solve():
    t0 = time.time()
    BUDGET = 168.0
    deadline = t0 + BUDGET
    rng = np.random.RandomState(0)

    try:
        xb = np.load("best.npy")
    except Exception:
        xb = np.load("record.npy")
    xb = np.asarray(xb, dtype=np.float64)
    # ensure shape (180,9)
    if xb.shape != (180, 9):
        # try to reshape / truncate deterministically
        flat = xb.ravel()
        need = 180 * 9
        if flat.size < need:
            rep = np.tile(flat, int(np.ceil(need / flat.size)))[:need]
            xb = rep.reshape(180, 9)
        else:
            xb = flat[:need].reshape(180, 9)
    xb = xb / (np.linalg.norm(xb, axis=1, keepdims=True) + 1e-30)
    best = max_cos(xb)
    try:
        xr = np.load("record.npy")
        xr = np.asarray(xr, dtype=np.float64)
        if xr.shape == (180, 9):
            xr = xr / (np.linalg.norm(xr, axis=1, keepdims=True) + 1e-30)
            vr = max_cos(xr)
            if vr < best:
                best = vr
                xb = xr
    except Exception:
        pass

    x_best = xb.copy()

    trial = 0
    # main basin-hopping loop
    while True:
        now = time.time()
        time_left = deadline - now
        if time_left < 7.0:
            break
        if trial > 60:
            break
        trial += 1

        # contact counts of current best
        G = x_best @ x_best.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        # contacts within 1e-7 of max
        C = (G > m - 1e-7).sum(axis=1)
        order = np.argsort(C)  # low -> high

        n_reinsert = 2 + (trial % 3)  # 2,3,4 cycle
        # victims: lowest contacts
        victims = order[:n_reinsert].copy()
        # fixed set
        mask = np.ones(len(x_best), dtype=bool)
        mask[victims] = False
        X_fixed = x_best[mask]

        # voids
        try:
            new_pos = largest_voids(X_fixed, n_reinsert, rng, n_cand=4000)
        except Exception:
            new_pos = None

        x_try = x_best.copy()
        if new_pos is not None:
            x_try[victims] = new_pos

        # kick overloaded + random
        sigma_kick = 0.08 + 0.04 * (trial % 4)  # 0.08..0.20
        n_kick = 12
        kicked = order[-n_kick:].copy()
        # add some random indices deterministically
        rnd_idx = rng.choice(len(x_try), size=8, replace=False)
        kicked = np.unique(np.concatenate([kicked, rnd_idx]))
        noise = rng.standard_normal(size=(len(kicked), x_try.shape[1])) * sigma_kick
        x_try[kicked] += noise
        # tiny global jitter
        x_try += rng.standard_normal(size=x_try.shape) * 0.008
        x_try /= (np.linalg.norm(x_try, axis=1, keepdims=True) + 1e-30)

        # anneal params vary per trial
        T0 = 0.025 + 0.005 * (trial % 3)
        T1 = 0.004
        lr0 = 0.018 + 0.004 * (trial % 2)
        lr1 = 0.003
        steps = 500 if time_left > 30 else 300

        try:
            x_ann = anneal_run(x_try, steps, T0, T1, lr0, lr1)
        except Exception:
            continue

        # quick check: if annealed max is terrible, skip LP? still polish if promising
        # allow LP only if within reasonable margin
        try:
            v_ann = max_cos(x_ann)
        except Exception:
            continue
        if v_ann > best + 0.03 and (trial % 5 != 0):
            continue

        lp_time = min(6.0, max(1.5, time_left - 3.0))
        # shorter LP early, longer if close
        if v_ann < best + 0.005:
            lp_time = min(7.0, max(2.5, time_left - 2.0))
        try:
            x_pol = lp_polish_bd(x_ann, seconds=lp_time, deadline=deadline)
        except Exception:
            continue
        try:
            v = max_cos(x_pol)
        except Exception:
            continue
        if v < best - 1e-12 and np.all(np.isfinite(x_pol)):
            best = v
            x_best = x_pol / (np.linalg.norm(x_pol, axis=1, keepdims=True) + 1e-30)

        # safety: check deadline
        if time.time() > deadline - 1.0:
            break

    # final tiny polish if time remains
    if deadline - time.time() > 3.0:
        try:
            x_best = lp_polish_bd(x_best, seconds=min(5.0, deadline - time.time() - 0.5), deadline=deadline)
        except Exception:
            pass
    x_best = x_best / (np.linalg.norm(x_best, axis=1, keepdims=True) + 1e-30)
    # guard NaN/zero
    x_best[~np.isfinite(x_best)] = 0.0
    # replace any zero rows deterministically
    nrm = np.linalg.norm(x_best, axis=1)
    for i in np.where(nrm < 1e-12)[0]:
        v = rng.standard_normal(size=x_best.shape[1])
        v /= (np.linalg.norm(v) + 1e-30)
        x_best[i] = v
    x_best = x_best / (np.linalg.norm(x_best, axis=1, keepdims=True) + 1e-30)
    return x_best
