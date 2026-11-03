import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def smooth_quench(x, steps=500, a0=40.0, a1=350.0, lr=0.003):
    x = x.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    vv = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(steps):
        frac = t / max(steps - 1, 1)
        alpha = a0 * ((a1 / a0) ** frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        E = np.exp(alpha * (G - M))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if S <= 0 or not np.isfinite(S):
            break
        Grad = (E @ x) / S
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        m = b1 * m + (1 - b1) * Gt
        vv = b2 * vv + (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1 ** (t + 1))
        vhat = vv / (1 - b2 ** (t + 1))
        step = mhat / (np.sqrt(vhat) + eps)
        np.clip(step, -3.0, 3.0, out=step)
        lrt = lr * (1.0 - 0.5 * frac)
        x -= lrt * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
        if not np.all(np.isfinite(x)):
            break
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def lp_polish_deadline(x, deadline, radius_init=3e-3, cap=2000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = radius_init
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        c_all = Gfull[iu0]
        mval = float(c_all.max())
        thresh = mval - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx_all = np.where(sel)[0]
            part = np.argpartition(-c_all[idx_all], cap)[:cap]
            keep = np.zeros_like(sel)
            keep[idx_all[part]] = True
            sel = keep
            cnt = cap
        I = iu0[0][sel]
        J = iu0[1][sel]
        c = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        if time.time() >= deadline:
            break
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y /= nr
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
        if time.time() >= deadline:
            break
    return x


def knockout_reinsert(cand, k, rng, stress, ntrials=512):
    n, d = cand.shape
    # pick k worst by stress (per-max + contact load)
    idx = np.argpartition(-stress, k)[:k]
    idx = idx[np.argsort(-stress[idx])]
    mask = np.ones(n, dtype=bool)
    mask[idx] = False
    cur = cand[mask].copy()
    for j in idx:
        trials = rng.standard_normal((ntrials, d))
        trials /= np.linalg.norm(trials, axis=1, keepdims=True)
        mm = (trials @ cur.T).max(axis=1)
        cand[j] = trials[np.argmin(mm)]
        cur = np.vstack([cur, cand[j][None, :]])
    return cand


def solve():
    t0 = time.time()
    BUDGET = 165.0
    deadline_main = t0 + BUDGET - 10.0
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.standard_normal((176, 9))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (176, 9):
        xb = rng.standard_normal((176, 9))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = max_cos(xb)
    best_x = xb.copy()
    try:
        xr = np.load("record.npy")
        xr = np.asarray(xr, dtype=float)
        if xr.shape == (176, 9):
            xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
            vr = max_cos(xr)
            if vr < best_val:
                best_val, best_x = vr, xr.copy()
    except Exception:
        pass

    it = 0
    # pre-polish incumbent briefly to ensure jammed baseline
    if time.time() < deadline_main:
        best_x = lp_polish_deadline(best_x, min(deadline_main, time.time() + 2.0),
                                    radius_init=1e-3, cap=2000)
        best_val = max_cos(best_x)

    while time.time() < deadline_main and it < 1000:
        it += 1
        if time.time() >= deadline_main:
            break
        cand = best_x.copy()
        mode = it % 5
        G = cand @ cand.T
        np.fill_diagonal(G, -2.0)
        per_max = G.max(axis=1)
        # contact load: count pairs within 2e-3 of max (cheap jam proxy)
        m0 = float(per_max.max())
        contact_cnt = (G > m0 - 0.002).sum(axis=1)
        stress = per_max + 0.0008 * contact_cnt.astype(float)
        # small random tie-break
        stress = stress + 1e-6 * rng.standard_normal(176)

        if mode == 0:
            cand = cand + 0.025 * rng.standard_normal(cand.shape)
        elif mode == 1:
            k = 2 + (it % 3)  # 2..4
            cand = knockout_reinsert(cand, k, rng, stress, ntrials=512)
            cand = cand + 0.012 * rng.standard_normal(cand.shape)
        elif mode == 2:
            k = 1 + (it % 2)
            cand = knockout_reinsert(cand, k, rng, stress, ntrials=1024)
            cand = cand + 0.03 * rng.standard_normal(cand.shape)
        elif mode == 3:
            sel = rng.choice(176, size=20, replace=False)
            cand[sel] += 0.12 * rng.standard_normal((20, 9))
            cand += 0.008 * rng.standard_normal(cand.shape)
        else:
            # shake hubs harder: top stressed points
            k = 8
            idx = np.argpartition(-stress, k)[:k]
            cand[idx] += 0.10 * rng.standard_normal((k, 9))
            cand += 0.015 * rng.standard_normal(cand.shape)

        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        # fast quench for screening
        a0 = 40.0 if (it % 2 == 0) else 60.0
        lr = 0.003 if (it % 3 != 0) else 0.002
        steps = 450 if (it % 4 != 0) else 650
        cand_s = smooth_quench(cand, steps=steps, a0=a0, a1=380.0, lr=lr)
        v = max_cos(cand_s)
        # only spend LP if close to best
        if v < best_val + 0.0025:
            remain = deadline_main - time.time()
            if remain < 0.6:
                # no time for LP, just check
                if v < best_val - 1e-12:
                    best_val, best_x = v, cand_s.copy()
                continue
            lp_secs = 0.9 if v > best_val - 0.0005 else 1.6
            lp_secs = min(lp_secs, remain - 0.1)
            if lp_secs > 0.3:
                cand2 = lp_polish_deadline(cand_s, time.time() + lp_secs,
                                           radius_init=4e-3, cap=2000)
                v2 = max_cos(cand2)
                if v2 < best_val - 1e-12:
                    best_val = v2
                    best_x = cand2.copy()
                    continue
            if v < best_val - 1e-12:
                best_val, best_x = v, cand_s.copy()
        # occasional intensify around incumbent
        if it % 25 == 0:
            if time.time() + 1.5 >= deadline_main:
                continue
            tmp = best_x + 0.006 * rng.standard_normal(best_x.shape)
            tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
            tmp = smooth_quench(tmp, steps=400, a0=90.0, a1=500.0, lr=0.0015)
            vt = max_cos(tmp)
            if vt < best_val + 0.001:
                tmp2 = lp_polish_deadline(tmp, time.time() + 1.0,
                                          radius_init=2e-3, cap=2000)
                vtt = max_cos(tmp2)
                if vtt < best_val - 1e-12:
                    best_val, best_x = vtt, tmp2.copy()

    # final squeeze with remaining time (max ~9s)
    rem = t0 + BUDGET - time.time()
    rem = min(max(rem - 0.4, 0.0), 9.0)
    if rem > 0.8:
        best_x = lp_polish_deadline(best_x, time.time() + rem,
                                    radius_init=8e-4, cap=2500)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
