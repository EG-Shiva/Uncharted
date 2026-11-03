import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def count_viol(X, t):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return int(np.count_nonzero(G > t) // 2)


def build_active(X, t, window=0.02):
    G = X @ X.T
    n = X.shape[0]
    iu0 = np.triu_indices(n, 1)
    c = G[iu0]
    sel = c > (t - window)
    return iu0[0][sel].copy(), iu0[1][sel].copy()


def hinge_optimize(X_init, t, deadline, rng, lr=0.004):
    X = X_init.copy()
    n, d = X.shape
    m = np.zeros_like(X)
    vv = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    adam_t = 0
    cur_best_max = max_cos(X)
    best_X = X.copy()
    best_max = cur_best_max
    # violation count best
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best_viol = int(np.count_nonzero(G0 > t) // 2)
    last_improve = time.time()
    perturbed = False

    I, J = build_active(X, t)
    if len(I) == 0:
        return best_X, best_max, (best_max < t)
    steps = 0
    Gt_last = None

    while time.time() < deadline:
        # inner 50 steps with fixed active list
        for _ in range(50):
            if time.time() >= deadline:
                break
            if len(I) == 0:
                break
            # need to handle indices still valid (same n)
            c = np.sum(X[I] * X[J], axis=1)
            diff = c - t
            w = 2.0 * diff + 4e-4 * (diff ** 3)
            G = np.zeros_like(X)
            # analytic gradient scatter
            np.add.at(G, I, w[:, None] * X[J])
            np.add.at(G, J, w[:, None] * X[I])
            dot = np.sum(G * X, axis=1, keepdims=True)
            Gt = G - dot * X
            Gt_last = Gt
            adam_t += 1
            m = b1 * m + (1 - b1) * Gt
            vv = b2 * vv + (1 - b2) * (Gt * Gt)
            mhat = m / (1 - b1 ** adam_t)
            vhat = vv / (1 - b2 ** adam_t)
            D = mhat / (np.sqrt(vhat) + eps)
            # per-row step clipping at 0.01 : lr*|D_row| <= 0.01
            nrm = np.linalg.norm(D, axis=1)
            # avoid div by zero
            over = nrm * lr > 0.01
            if np.any(over):
                scale = np.ones(n)
                scale[over] = 0.01 / (lr * nrm[over] + 1e-16)
                D = D * scale[:, None]
            X = X - lr * D
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr[nr < 1e-12] = 1.0
            X /= nr
            steps += 1
        if time.time() >= deadline:
            break
        cur_max = max_cos(X)
        if cur_max < best_max:
            best_max = cur_max
            best_X = X.copy()
        # violation count
        Gf = X @ X.T
        np.fill_diagonal(Gf, -2.0)
        cur_viol = int(np.count_nonzero(Gf > t) // 2)
        if cur_viol < best_viol:
            best_viol = cur_viol
            last_improve = time.time()
        if cur_max < t:
            # success
            return X.copy(), cur_max, True
        # stall check: 5s without reducing violation count
        if (time.time() - last_improve > 5.0) and (not perturbed):
            # perturb only points participating in violated pairs
            iu = np.triu_indices(n, 1)
            # reuse Gf
            sel_v = Gf[iu] > t
            if np.any(sel_v):
                Iv = iu[0][sel_v]
                Jv = iu[1][sel_v]
                involved = np.unique(np.concatenate([Iv, Jv]))
                if Gt_last is None:
                    Gt_last = np.zeros_like(X)
                for idx in involved:
                    g = Gt_last[idx]
                    gn = np.linalg.norm(g)
                    if gn > 1e-16:
                        ghat = g / gn
                    else:
                        ghat = np.zeros(d)
                    r = rng.standard_normal(d)
                    r = r - np.dot(r, X[idx]) * X[idx]
                    rn = np.linalg.norm(r)
                    if rn < 1e-12:
                        r = rng.standard_normal(d)
                        r = r - np.dot(r, X[idx]) * X[idx]
                        rn = np.linalg.norm(r) + 1e-16
                    rhat = r / (rn + 1e-16)
                    disp = -0.02 * ghat + 0.02 * rhat
                    X[idx] = X[idx] + disp
                X /= np.linalg.norm(X, axis=1, keepdims=True)
                # reset momentum for involved
                m[involved] = 0.0
                vv[involved] = 0.0
                last_improve = time.time()
            perturbed = True
        # rebuild active list
        I, J = build_active(X, t)
        if len(I) == 0:
            return X.copy(), cur_max, True
        # if active huge, keep as is (hinge handles); but cap rebuild cost fine
        if time.time() >= deadline:
            break
    # return best found for this target
    return best_X, best_max, (best_max < t)


def lp_polish_deadline(x, deadline, r_init=0.004, cap=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = r_init
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        mval = float(c_all.max())
        thresh = mval - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > cap:
            idx_all = np.where(sel)[0]
            # keep largest cap
            part = np.argpartition(-c_all[idx_all], cap)[:cap]
            keep = np.zeros_like(sel)
            keep[idx_all[part]] = True
            sel = keep
        I = iu0[0][sel]
        J = iu0[1][sel]
        c_sel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y = y / yn
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x


def solve():
    t_start = time.time()
    HARD_STOP = t_start + 160.0
    rng = np.random.default_rng(0)
    n, d = 176, 9
    # load start
    X0 = None
    best_load_val = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(nr > 1e-12):
                    a = a / nr
                    v = max_cos(a)
                    if v < best_load_val:
                        best_load_val = v
                        X0 = a.copy()
        except Exception:
            continue
    if X0 is None:
        X0 = rng.standard_normal((n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
    cur = X0.copy()
    cur_val = max_cos(cur)
    overall_best = cur.copy()
    overall_val = cur_val

    targets = [0.4610, 0.4605, 0.4600, 0.4590]
    for t in targets:
        if time.time() > HARD_STOP - 25.0:
            break
        # skip if already below target? still try next lower
        if overall_val < t:
            cur = overall_best.copy()
        else:
            # start from overall best (least violating)
            cur = overall_best.copy()
        deadline = min(time.time() + 20.0, HARD_STOP - 22.0)
        if deadline - time.time() < 2.0:
            break
        try:
            out_X, out_val, success = hinge_optimize(cur, t, deadline, rng)
        except Exception:
            continue
        if out_val < overall_val:
            overall_val = out_val
            overall_best = out_X.copy()
        if success:
            cur = out_X.copy()
            # continue to next lower target
            continue
        else:
            # keep least-violating as start for next (still try lower? spec says sequence)
            # continue anyway from best
            continue

    # final LP polish of best feasible/least-violating, 20s but respect hard stop
    lp_deadline = min(time.time() + 20.0, HARD_STOP)
    if time.time() < lp_deadline - 1.0:
        try:
            polished = lp_polish_deadline(overall_best, lp_deadline)
            pv = max_cos(polished)
            if pv < overall_val:
                overall_best = polished
                overall_val = pv
        except Exception:
            pass
    overall_best = overall_best / np.linalg.norm(overall_best, axis=1, keepdims=True)
    return overall_best
